import litellm
from crewai import Agent, LLM
from app.config import GROQ_MODEL
from app.tools.search_tools import search_financial_news, search_company_overview
from app.tools.finance_tools import get_stock_data
from app.tools.sentiment_tools import analyze_sentiment

# Automatically retry on rate limit errors (429) with exponential backoff
litellm.num_retries = 5
litellm.retry_on_status_codes = [429]


def get_llm():
    return LLM(model=f"groq/{GROQ_MODEL}", temperature=0.3, max_tokens=800)


def create_market_researcher() -> Agent:
    return Agent(
        role="Senior Market Research Analyst",
        goal=(
            "Find the most recent and relevant news articles, earnings reports, "
            "and analyst opinions about the given stock ticker. Provide comprehensive "
            "coverage of recent developments."
        ),
        backstory=(
            "You are a veteran Wall Street research analyst with 15+ years of experience "
            "tracking equities across all sectors. You have a keen eye for finding the most "
            "impactful news that moves stock prices and a talent for separating signal from noise."
        ),
        tools=[search_financial_news, search_company_overview],
        llm=get_llm(),
        verbose=True,
        allow_delegation=False,
        max_rpm=3,
    )


def create_sentiment_analyst() -> Agent:
    return Agent(
        role="NLP & Sentiment Analysis Specialist",
        goal=(
            "Analyze the sentiment of financial news articles using the FinBERT model. "
            "Provide detailed per-article sentiment scores and an aggregate sentiment signal."
        ),
        backstory=(
            "You are a quantitative analyst specializing in Natural Language Processing "
            "applied to financial markets. You use state-of-the-art NLP models like FinBERT "
            "to extract sentiment signals from news and earnings transcripts. Your sentiment "
            "analysis has been used by hedge funds to generate alpha."
        ),
        tools=[analyze_sentiment],
        llm=get_llm(),
        verbose=True,
        allow_delegation=False,
        max_rpm=3,
    )


def create_financial_analyst() -> Agent:
    return Agent(
        role="Quantitative Financial Analyst",
        goal=(
            "Retrieve and analyze key financial metrics, valuation ratios, price trends, "
            "and analyst targets for the given stock ticker."
        ),
        backstory=(
            "You are a CFA-certified financial analyst with deep expertise in equity valuation, "
            "fundamental analysis, and quantitative finance. You can quickly assess a company's "
            "financial health by examining key ratios and metrics."
        ),
        tools=[get_stock_data],
        llm=get_llm(),
        verbose=True,
        allow_delegation=False,
        max_rpm=3,
    )


def create_memo_writer() -> Agent:
    return Agent(
        role="Senior Equity Research Editor",
        goal=(
            "Synthesize all research findings into a professional, structured research deep dive "
            "that adds context, risks, and scenario framing without overriding the deterministic verdict."
        ),
        backstory=(
            "You are a senior equity research editor. You turn analyst notes, news, and market data "
            "into concise research narratives that surface what matters, what is uncertain, and what "
            "an investor should verify next. You do not invent unsupported facts or issue authoritative trades."
        ),
        tools=[],
        llm=get_llm(),
        verbose=True,
        allow_delegation=False,
        max_rpm=3,
    )
