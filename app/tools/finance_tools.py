import json
from datetime import datetime, timedelta
from crewai.tools import tool
import yfinance as yf


def _safe_get(info: dict, key: str, default="N/A"):
    """Safely get a value from yfinance info dict."""
    val = info.get(key, default)
    return val if val is not None else default


@tool("Get Stock Financial Data")
def get_stock_data(ticker: str) -> str:
    """
    Fetch comprehensive financial data for a stock ticker including price,
    fundamentals, key ratios, and 30-day price history.

    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL')
    """
    stock = yf.Ticker(ticker)

    try:
        info = stock.info
    except Exception as e:
        return json.dumps({"error": f"Failed to fetch data for {ticker}: {str(e)}"})

    # Key metrics
    financials = {
        "ticker": ticker,
        "company_name": _safe_get(info, "longName"),
        "sector": _safe_get(info, "sector"),
        "industry": _safe_get(info, "industry"),
        "description": _safe_get(info, "longBusinessSummary", ""),

        # Price data
        "current_price": _safe_get(info, "currentPrice"),
        "previous_close": _safe_get(info, "previousClose"),
        "fifty_two_week_high": _safe_get(info, "fiftyTwoWeekHigh"),
        "fifty_two_week_low": _safe_get(info, "fiftyTwoWeekLow"),
        "fifty_day_average": _safe_get(info, "fiftyDayAverage"),
        "two_hundred_day_average": _safe_get(info, "twoHundredDayAverage"),

        # Valuation
        "market_cap": _safe_get(info, "marketCap"),
        "pe_ratio": _safe_get(info, "trailingPE"),
        "forward_pe": _safe_get(info, "forwardPE"),
        "peg_ratio": _safe_get(info, "pegRatio"),
        "price_to_book": _safe_get(info, "priceToBook"),

        # Financials
        "revenue": _safe_get(info, "totalRevenue"),
        "revenue_growth": _safe_get(info, "revenueGrowth"),
        "earnings_per_share": _safe_get(info, "trailingEps"),
        "forward_eps": _safe_get(info, "forwardEps"),
        "profit_margin": _safe_get(info, "profitMargins"),
        "operating_margin": _safe_get(info, "operatingMargins"),
        "return_on_equity": _safe_get(info, "returnOnEquity"),

        # Dividends
        "dividend_yield": _safe_get(info, "dividendYield"),
        "dividend_rate": _safe_get(info, "dividendRate"),

        # Analyst targets
        "target_high_price": _safe_get(info, "targetHighPrice"),
        "target_low_price": _safe_get(info, "targetLowPrice"),
        "target_mean_price": _safe_get(info, "targetMeanPrice"),
        "recommendation_key": _safe_get(info, "recommendationKey"),
        "number_of_analyst_opinions": _safe_get(info, "numberOfAnalystOpinions"),
    }

    # 30-day price history
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        hist = stock.history(start=start_date.strftime("%Y-%m-%d"), end=end_date.strftime("%Y-%m-%d"))

        price_history = []
        for date, row in hist.iterrows():
            price_history.append({
                "date": date.strftime("%Y-%m-%d"),
                "close": round(row["Close"], 2),
                "volume": int(row["Volume"]),
            })

        financials["price_history_30d"] = price_history
        financials["price_change_30d"] = round(
            ((price_history[-1]["close"] - price_history[0]["close"]) / price_history[0]["close"]) * 100, 2
        ) if len(price_history) >= 2 else "N/A"
    except Exception:
        financials["price_history_30d"] = []
        financials["price_change_30d"] = "N/A"

    return json.dumps(financials, indent=2, default=str)
