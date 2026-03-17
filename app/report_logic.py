import re


DEFAULT_THEME = {
    "bg": "#0f1117",
    "panel": "#161b22",
    "panel_alt": "#1d2430",
    "border": "#2d3648",
    "text": "#e6edf3",
    "muted": "#9aa4b2",
    "accent": "#7aa2f7",
    "bull": "#35c47c",
    "bear": "#ff6b6b",
    "neutral": "#f2c14e",
    "bull_soft": "#10281c",
    "bear_soft": "#311619",
    "neutral_soft": "#32290f",
}


def get_sentiment_display(signal, theme=None):
    theme = theme or DEFAULT_THEME
    return {
        "BULLISH": {
            "label": "Positive Outlook",
            "color": theme["bull"],
            "bg": theme["bull_soft"],
            "explanation": (
                "Recent news about this stock is mostly positive. "
                "Investors and analysts appear optimistic about its future."
            ),
        },
        "BEARISH": {
            "label": "Cautious Outlook",
            "color": theme["bear"],
            "bg": theme["bear_soft"],
            "explanation": (
                "Recent news raises concerns about this stock. "
                "Investors and analysts appear cautious or worried."
            ),
        },
    }.get(signal, {
        "label": "Mixed Signals",
        "color": theme["neutral"],
        "bg": theme["neutral_soft"],
        "explanation": (
            "News about this stock is a mix of good and bad. "
            "There is no clear consensus among investors right now."
        ),
    })


def get_recommendation_display(rec_key):
    return {
        "strongBuy": ("Strong Buy", "Experts strongly recommend buying this stock"),
        "buy": ("Buy", "Experts lean toward buying this stock"),
        "hold": ("Hold", "Experts suggest holding rather than buying or selling aggressively"),
        "sell": ("Sell", "Experts suggest considering selling"),
        "strongSell": ("Strong Sell", "Experts strongly advise against holding this stock"),
    }.get(rec_key, ("No Rating", "No analyst consensus available"))


def get_sentiment_sample_note(total_articles):
    if not isinstance(total_articles, int) or total_articles <= 0:
        return "Sentiment signal unavailable."
    if total_articles < 5:
        return f"Sentiment based on a small sample of {total_articles} recent articles."
    if total_articles < 10:
        return f"Sentiment based on {total_articles} recent articles."
    return f"Sentiment based on a broader sample of {total_articles} recent articles."


def get_company_terms(company_name):
    stop_words = {"inc", "inc.", "corp", "corp.", "corporation", "company", "co", "co.", "ltd", "ltd.", "plc", "holdings"}
    return [part.lower() for part in re.split(r"[^A-Za-z0-9]+", company_name or "") if part and part.lower() not in stop_words and len(part) > 2]


def is_relevant_article(result, ticker, company_name):
    haystack = " ".join([
        result.get("title", ""),
        result.get("content", ""),
        result.get("url", ""),
    ]).lower()
    ticker_token = (ticker or "").lower()
    if ticker_token and ticker_token in haystack:
        return True

    company_terms = get_company_terms(company_name)
    if not company_terms:
        return False
    return sum(1 for term in company_terms if term in haystack) >= 1


def get_recommendation_color(label, theme=None):
    theme = theme or DEFAULT_THEME
    return {
        "BUY": theme["bull"],
        "HOLD": theme["neutral"],
        "SELL": theme["bear"],
    }.get(label, theme["muted"])


def format_currency(value):
    if isinstance(value, (int, float)):
        return f"${value:,.2f}"
    return "N/A"


def format_market_cap(value):
    if isinstance(value, (int, float)):
        if value >= 1e12:
            return f"${value/1e12:.2f}T"
        return f"${value/1e9:.1f}B"
    return "N/A"


def get_confidence_level(financial_data, sentiment_data, recommendation):
    score = 0
    if financial_data and isinstance(financial_data.get("current_price"), (int, float)):
        score += 1
    if financial_data and isinstance(financial_data.get("target_mean_price"), (int, float)):
        score += 1
    if financial_data and financial_data.get("recommendation_key") not in ("", "N/A", None):
        score += 1
    if financial_data and isinstance(financial_data.get("number_of_analyst_opinions"), (int, float)) and financial_data.get("number_of_analyst_opinions", 0) >= 10:
        score += 1
    if sentiment_data and sentiment_data.get("total_analyzed", 0) >= 6:
        score += 1

    signal = recommendation.get("signal")
    label = recommendation.get("label")
    aligned = (
        (label == "BUY" and signal == "BULLISH") or
        (label == "HOLD" and signal == "NEUTRAL") or
        (label == "SELL" and signal == "BEARISH")
    )
    if aligned:
        score += 1

    if signal == "NEUTRAL":
        score -= 1
    if recommendation.get("conflict_note"):
        score -= 1
    change_30d = financial_data.get("price_change_30d") if financial_data else None
    if isinstance(change_30d, (int, float)):
        if label == "BUY" and change_30d < 0:
            score -= 1
        elif label == "SELL" and change_30d > 0:
            score -= 1

    if score >= 5:
        return "Higher", "Market data is complete and the main signals are mostly aligned."
    if score >= 3:
        return "Moderate", "Useful directional signal, but there is meaningful uncertainty."
    return "Lower", "The app has limited evidence or conflicting inputs for this call."


def build_evidence_cards(financial_data, sentiment_data, recommendation):
    confidence_label, confidence_text = get_confidence_level(financial_data, sentiment_data, recommendation)
    article_count = sentiment_data.get("total_analyzed", 0) if sentiment_data else 0
    return [
        {
            "label": "Why this verdict",
            "value": recommendation["label"],
            "text": recommendation["summary"],
        },
        {
            "label": "Trust level",
            "value": confidence_label,
            "text": confidence_text,
        },
        {
            "label": "Evidence used",
            "value": f"{article_count} articles",
            "text": "The call combines analyst consensus, target price, recent news tone, and 30-day price trend.",
        },
    ]


def build_recommendation(financial_data, sentiment_data, theme=None):
    rec_key = financial_data.get("recommendation_key", "") if financial_data else ""
    cp = financial_data.get("current_price") if financial_data else None
    tm = financial_data.get("target_mean_price") if financial_data else None
    change_30d = financial_data.get("price_change_30d") if financial_data else None
    signal = sentiment_data.get("overall_signal", "NEUTRAL") if sentiment_data else "NEUTRAL"

    analyst_score = {
        "strongBuy": 1.0,
        "buy": 0.6,
        "hold": 0.0,
        "sell": -0.6,
        "strongSell": -1.0,
    }.get(rec_key, 0.0)
    sentiment_score = {"BULLISH": 0.35, "NEUTRAL": 0.0, "BEARISH": -0.35}.get(signal, 0.0)

    upside = None
    upside_score = 0.0
    if isinstance(cp, (int, float)) and isinstance(tm, (int, float)) and cp:
        upside = ((tm - cp) / cp) * 100
        if upside >= 12:
            upside_score = 0.8
        elif upside >= 4:
            upside_score = 0.3
        elif upside <= -12:
            upside_score = -0.8
        elif upside <= -4:
            upside_score = -0.3

    trend_score = 0.0
    if isinstance(change_30d, (int, float)):
        if change_30d >= 6:
            trend_score = 0.2
        elif change_30d <= -6:
            trend_score = -0.2

    combined_score = analyst_score * 0.5 + upside_score * 0.3 + sentiment_score * 0.15 + trend_score * 0.05
    if combined_score >= 0.3:
        label = "BUY"
    elif combined_score <= -0.3:
        label = "SELL"
    else:
        label = "HOLD"

    analyst_label, analyst_text = get_recommendation_display(rec_key)
    total_articles = sentiment_data.get("total_analyzed", 0) if sentiment_data else 0
    reasons = []
    if analyst_label != "No Rating":
        reasons.append(f"Wall Street consensus is {analyst_label.lower()}.")
    if upside is not None:
        direction = "upside" if upside >= 0 else "downside"
        reasons.append(f"Analysts' average target implies {abs(upside):.1f}% {direction} from the current price.")
    if signal == "BULLISH":
        reasons.append("Recent coverage has a positive tone.")
    elif signal == "BEARISH":
        reasons.append("Recent coverage leans negative.")
    if isinstance(change_30d, (int, float)):
        reasons.append(f"The stock is {change_30d:+.1f}% over the last 30 days.")

    conflict_note = None
    if label == "BUY" and signal == "BEARISH":
        conflict_note = "The recommendation is still positive because analyst targets outweigh the current negative news tone."
    elif label == "SELL" and signal == "BULLISH":
        conflict_note = "The recommendation is still negative because market and valuation data outweigh the current positive news tone."

    summary = " ".join(reasons[:3]) if reasons else "Recommendation based on the available market data."
    if conflict_note:
        summary = f"{summary} {conflict_note}"

    return {
        "label": label,
        "color": get_recommendation_color(label, theme=theme),
        "summary": summary,
        "reasons": reasons,
        "analyst_consensus": analyst_label,
        "analyst_consensus_help": analyst_text,
        "price_target": tm,
        "current_price": cp,
        "upside": upside,
        "signal": signal,
        "score": combined_score,
        "conflict_note": conflict_note,
        "sentiment_sample_note": get_sentiment_sample_note(total_articles),
    }


def build_full_report(ticker, financial_data, sentiment_data, recommendation):
    company_name = financial_data.get("company_name", ticker) if financial_data else ticker
    sector = financial_data.get("sector", "N/A") if financial_data else "N/A"
    price = financial_data.get("current_price") if financial_data else None
    pe = financial_data.get("pe_ratio") if financial_data else None
    market_cap = financial_data.get("market_cap") if financial_data else None
    target_mean = financial_data.get("target_mean_price") if financial_data else None
    sentiment_label = get_sentiment_display(recommendation["signal"])["label"]
    sentiment_score = sentiment_data.get("average_sentiment_score") if sentiment_data else None
    market_cap_display = format_market_cap(market_cap)
    score_text = f"{sentiment_score * 100:.1f}/100" if isinstance(sentiment_score, (int, float)) else "N/A"
    recommendation_lines = "\n".join(f"- {reason}" for reason in recommendation["reasons"]) or "- No detailed rationale available."
    sampled_articles = sentiment_data.get("sampled_articles", [])[:5] if sentiment_data else []
    news_lines = "\n".join(
        f"- {article.get('title', 'Untitled')} ({article.get('source', 'Unknown source')})"
        for article in sampled_articles
    ) or "- No relevant recent articles were captured for this run."
    confidence_label, confidence_text = get_confidence_level(financial_data, sentiment_data, recommendation)

    return f"""# Research Summary: {ticker}

## Executive Summary
{company_name} currently screens as **{recommendation['label']}**. {recommendation['summary']}

## Company Overview & Key Metrics
- Company: {company_name}
- Sector: {sector}
- Current Price: {format_currency(price)}
- Analyst Target: {format_currency(target_mean)}
- Analyst Consensus: {recommendation['analyst_consensus']}
- Market Cap: {market_cap_display}
- P/E Ratio: {f"{pe:.1f}" if isinstance(pe, (int, float)) else "N/A"}

## Recent News Reviewed
{news_lines}

## Sentiment Analysis
- Market Tone: {sentiment_label}
- FinBERT Score: {score_text}
- Interpretation: {get_sentiment_display(recommendation['signal'])['explanation']}

## Current View
**{recommendation['label']}**

{recommendation_lines}

## Confidence & Limitations
- Confidence Level: {confidence_label}
- Assessment: {confidence_text}
- Coverage Note: {recommendation['sentiment_sample_note']}

This report is generated directly from market data and sentiment inputs to support research workflows. It is not investment advice and should be verified against primary sources before acting on it.
"""
