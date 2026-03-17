import os
import time
from crewai import Crew, Process
from app.config import validate_config
from app.agent_guardrails import normalize_agent_memo, validate_agent_memo
from app.agents import (
    create_market_researcher,
    create_sentiment_analyst,
    create_financial_analyst,
    create_memo_writer,
)
from app.tasks import (
    create_research_task,
    create_sentiment_task,
    create_financial_analysis_task,
    create_memo_task,
)


def build_investment_crew(ticker: str) -> Crew:
    """
    Build and return a CrewAI crew for investment research.

    The crew runs 4 agents sequentially:
    1. Market Researcher -> gathers news
    2. Sentiment Analyst -> runs FinBERT on news
    3. Financial Analyst -> pulls financial data
    4. Memo Writer -> synthesizes everything into an investment memo

    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL')

    Returns:
        Configured Crew ready to kickoff
    """
    validate_config()

    # Ensure output directory exists
    os.makedirs("output", exist_ok=True)

    # Create agents
    researcher = create_market_researcher()
    sentiment_analyst = create_sentiment_analyst()
    financial_analyst = create_financial_analyst()
    memo_writer = create_memo_writer()

    # Create tasks
    research_task = create_research_task(researcher, ticker)
    sentiment_task = create_sentiment_task(sentiment_analyst, ticker)
    financial_task = create_financial_analysis_task(financial_analyst, ticker)
    memo_task = create_memo_task(memo_writer, ticker)

    # Wire up context: each task gets output from previous tasks
    sentiment_task.context = [research_task]
    financial_task.context = [research_task, sentiment_task]
    memo_task.context = [research_task, sentiment_task, financial_task]

    def _step_callback(step_output):
        """Sleep between agent steps to avoid Groq's 12k TPM rolling window limit."""
        time.sleep(20)

    # Build crew
    crew = Crew(
        agents=[researcher, sentiment_analyst, financial_analyst, memo_writer],
        tasks=[research_task, sentiment_task, financial_task, memo_task],
        process=Process.sequential,
        verbose=True,
        step_callback=_step_callback,
    )

    return crew


def run_analysis(ticker: str) -> dict:
    """
    Run the full investment analysis pipeline for a ticker.

    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL')

    Returns:
        Dictionary with 'memo' (the investment memo text) and 'metadata'
    """
    ticker = ticker.upper().strip()

    crew = build_investment_crew(ticker)
    result = crew.kickoff()
    memo = result.raw if hasattr(result, "raw") else str(result)
    memo = normalize_agent_memo(memo)
    is_valid, validated_or_error = validate_agent_memo(memo)
    if not is_valid:
        raise ValueError(f"Agentic deep dive failed validation: {validated_or_error}")

    return {
        "ticker": ticker,
        "memo": validated_or_error,
        "token_usage": result.token_usage if hasattr(result, "token_usage") else {},
    }


# CLI entry point
if __name__ == "__main__":
    import sys

    ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    print(f"\n{'='*60}")
    print(f"  Deep Research Investment Agent - Analyzing {ticker}")
    print(f"{'='*60}\n")

    result = run_analysis(ticker)

    print(f"\n{'='*60}")
    print(f"  Analysis Complete!")
    print(f"{'='*60}\n")
    print(result["memo"])
