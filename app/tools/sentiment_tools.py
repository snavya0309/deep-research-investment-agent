import json
import re
from crewai.tools import tool
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# Lazy-load the model to avoid loading on import
_sentiment_pipeline = None


def _get_sentiment_pipeline():
    """Load FinBERT model lazily (first call downloads ~400MB model)."""
    global _sentiment_pipeline
    if _sentiment_pipeline is None:
        model_name = "ProsusAI/finbert"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        _sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model=model,
            tokenizer=tokenizer,
            truncation=True,
            max_length=512,
        )
    return _sentiment_pipeline


def _parse_text_items(texts: str):
    try:
        items = json.loads(texts)
    except json.JSONDecodeError:
        items = None

    if isinstance(items, dict):
        return [items]
    if isinstance(items, list):
        return items

    blocks = [block.strip() for block in re.split(r"\n\s*---\s*\n", texts) if block.strip()]
    parsed_items = []
    for block in blocks:
        title_match = re.search(r"^Title:\s*(.+)$", block, flags=re.MULTILINE)
        content_match = re.search(r"^Content:\s*([\s\S]+)$", block, flags=re.MULTILINE)
        if title_match and content_match:
            parsed_items.append({
                "title": title_match.group(1).strip(),
                "content": content_match.group(1).strip(),
            })

    if parsed_items:
        return parsed_items

    return [{"title": "Input Text", "content": texts}]


@tool("Analyze Financial Sentiment")
def analyze_sentiment(texts: str) -> str:
    """
    Analyze the sentiment of financial text using FinBERT.
    Accepts a JSON string containing a list of text items to analyze.
    Each item should have 'title' and 'content' fields.
    Returns sentiment scores (positive, negative, neutral) for each text
    and an aggregate summary.

    Args:
        texts: Either a JSON string with a list of objects containing 'title'
               and 'content' keys, or a plain-text block formatted as:
               Title: ...
               Content: ...
               ---
               Title: ...
               Content: ...
    """
    pipe = _get_sentiment_pipeline()
    items = _parse_text_items(texts)

    results = []
    sentiment_counts = {"positive": 0, "negative": 0, "neutral": 0}
    total_score = 0.0

    for item in items:
        # Use title + content for sentiment, truncate to avoid token limits
        text = f"{item.get('title', '')}. {item.get('content', '')}"[:1000]

        if not text.strip():
            continue

        try:
            prediction = pipe(text)[0]
            label = prediction["label"].lower()
            score = prediction["score"]

            # Convert to a -1 to +1 scale for aggregate scoring
            if label == "positive":
                numeric_score = score
            elif label == "negative":
                numeric_score = -score
            else:
                numeric_score = 0.0

            sentiment_counts[label] += 1
            total_score += numeric_score

            results.append({
                "title": item.get("title", "N/A"),
                "sentiment": label,
                "confidence": round(score, 4),
                "numeric_score": round(numeric_score, 4),
            })
        except Exception as e:
            results.append({
                "title": item.get("title", "N/A"),
                "sentiment": "error",
                "confidence": 0,
                "error": str(e),
            })

    total_articles = len(results)
    avg_score = round(total_score / total_articles, 4) if total_articles > 0 else 0

    # Determine overall sentiment
    if avg_score > 0.15:
        overall = "BULLISH"
    elif avg_score < -0.15:
        overall = "BEARISH"
    else:
        overall = "NEUTRAL"

    output = {
        "total_analyzed": total_articles,
        "sentiment_breakdown": sentiment_counts,
        "average_sentiment_score": avg_score,
        "overall_signal": overall,
        "per_article_sentiment": results,
    }

    return json.dumps(output, indent=2)
