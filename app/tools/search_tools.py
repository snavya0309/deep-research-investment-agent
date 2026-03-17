import os
import json
from crewai.tools import tool
from tavily import TavilyClient

CREW_NEWS_RESULTS = 2
CREW_EARNINGS_RESULTS = 2
CREW_ARTICLE_CONTENT_LIMIT = 120


def _company_terms(company_name: str) -> list[str]:
    stop_words = {"inc", "inc.", "corp", "corp.", "corporation", "company", "co", "co.", "ltd", "ltd.", "plc", "holdings"}
    return [part.lower() for part in company_name.replace("&", " ").split() if part and part.lower() not in stop_words and len(part) > 2]


def _is_relevant_result(result: dict, ticker: str, company_name: str) -> bool:
    haystack = " ".join([
        result.get("title", ""),
        result.get("content", ""),
        result.get("url", ""),
    ]).lower()
    if ticker.lower() in haystack:
        return True
    return any(term in haystack for term in _company_terms(company_name))


@tool("Search Financial News")
def search_financial_news(ticker: str, company_name: str = "") -> str:
    """
    Search for recent financial news, earnings reports, and analyst opinions
    about a stock ticker. Returns a JSON string of news articles with titles,
    sources, and content summaries.

    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL')
        company_name: Optional company name for better search results
    """
    client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

    search_query = f'"{ticker}" "{company_name}" stock news analysis earnings analyst'.strip()

    # Search for recent news
    news_response = client.search(
        query=search_query,
        search_depth="basic",
        max_results=CREW_NEWS_RESULTS,
        include_answer=True,
        topic="news",
    )

    # Search for earnings and analyst reports
    earnings_response = client.search(
        query=f"{ticker} earnings report analyst rating {company_name}".strip(),
        search_depth="basic",
        max_results=CREW_EARNINGS_RESULTS,
        topic="news",
    )

    articles = []
    seen_urls = set()

    for result in news_response.get("results", []) + earnings_response.get("results", []):
        url = result.get("url", "")
        if url in seen_urls:
            continue
        if not _is_relevant_result(result, ticker, company_name):
            continue
        seen_urls.add(url)

        # Truncate content to avoid exceeding LLM token limits
        content = result.get("content", "N/A")
        if len(content) > CREW_ARTICLE_CONTENT_LIMIT:
            content = content[:CREW_ARTICLE_CONTENT_LIMIT] + "..."

        articles.append({
            "title": result.get("title", "N/A"),
            "url": url,
            "content": content,
            "score": result.get("score", 0),
        })

    output = {
        "ticker": ticker,
        "total_articles": len(articles),
        "ai_summary": news_response.get("answer", ""),
        "articles": articles,
    }

    return json.dumps(output, indent=2)


@tool("Search Company Overview")
def search_company_overview(ticker: str) -> str:
    """
    Search for general company information, business model, and industry context.

    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL')
    """
    client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

    response = client.search(
        query=f"{ticker} company overview business model industry",
        search_depth="basic",
        max_results=3,
        include_answer=True,
    )

    return json.dumps({
        "ticker": ticker,
        "overview": response.get("answer", ""),
        "sources": [
            {"title": r.get("title", ""), "url": r.get("url", "")}
            for r in response.get("results", [])
        ],
    }, indent=2)
