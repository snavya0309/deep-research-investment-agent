import unittest

from app.agent_guardrails import validate_agent_memo
from app.report_logic import (
    build_evidence_cards,
    build_full_report,
    build_recommendation,
    get_confidence_level,
    get_recommendation_display,
    get_sentiment_display,
    is_relevant_article,
)
from app.tools.search_tools import _is_relevant_result
from app.tools.sentiment_tools import _parse_text_items


def sample_financial_data(**overrides):
    data = {
        "company_name": "Apple Inc.",
        "sector": "Technology",
        "current_price": 250.12,
        "target_mean_price": 295.44,
        "recommendation_key": "buy",
        "price_change_30d": -4.44,
        "number_of_analyst_opinions": 35,
        "market_cap": 3.68e12,
        "pe_ratio": 31.7,
    }
    data.update(overrides)
    return data


def sample_sentiment_data(**overrides):
    data = {
        "overall_signal": "NEUTRAL",
        "average_sentiment_score": -0.054,
        "total_analyzed": 12,
        "sentiment_breakdown": {"positive": 4, "neutral": 4, "negative": 4},
        "sampled_articles": [
            {"title": "Apple launches new product line", "source": "Reuters"},
            {"title": "Analysts revisit Apple targets", "source": "Bloomberg"},
        ],
    }
    data.update(overrides)
    return data


class ReportLogicTests(unittest.TestCase):
    def test_relevant_article_requires_ticker_or_company_match(self):
        relevant = {
            "title": "Apple earnings beat expectations",
            "content": "AAPL posted stronger services growth this quarter.",
            "url": "https://example.com/apple-earnings",
        }
        irrelevant = {
            "title": "Dragon Mining profit soars",
            "content": "The miner reported strong quarterly results.",
            "url": "https://example.com/dragon-mining",
        }

        self.assertTrue(is_relevant_article(relevant, "AAPL", "Apple Inc."))
        self.assertFalse(is_relevant_article(irrelevant, "AAPL", "Apple Inc."))

    def test_buy_recommendation_can_stand_with_neutral_sentiment_when_targets_are_strong(self):
        recommendation = build_recommendation(sample_financial_data(), sample_sentiment_data())

        self.assertEqual(recommendation["label"], "BUY")
        self.assertEqual(recommendation["analyst_consensus"], "Buy")
        self.assertGreater(recommendation["upside"], 0)
        self.assertIn("Wall Street consensus is buy.", recommendation["summary"])

    def test_conflicting_buy_and_bearish_sentiment_sets_conflict_note_and_moderate_confidence(self):
        financial_data = sample_financial_data()
        sentiment_data = sample_sentiment_data(
            overall_signal="BEARISH",
            average_sentiment_score=-0.35,
            sentiment_breakdown={"positive": 1, "neutral": 2, "negative": 9},
        )

        recommendation = build_recommendation(financial_data, sentiment_data)
        confidence_label, _ = get_confidence_level(financial_data, sentiment_data, recommendation)

        self.assertEqual(recommendation["label"], "BUY")
        self.assertIsNotNone(recommendation["conflict_note"])
        self.assertEqual(confidence_label, "Moderate")

    def test_full_report_is_deterministic_and_includes_recent_news_and_limitations(self):
        recommendation = build_recommendation(sample_financial_data(), sample_sentiment_data())
        report = build_full_report("AAPL", sample_financial_data(), sample_sentiment_data(), recommendation)

        self.assertIn("# Research Summary: AAPL", report)
        self.assertIn("## Recent News Reviewed", report)
        self.assertIn("Apple launches new product line (Reuters)", report)
        self.assertIn("## Confidence & Limitations", report)
        self.assertIn("It is not investment advice", report)

    def test_evidence_cards_report_article_count_and_confidence(self):
        recommendation = build_recommendation(sample_financial_data(), sample_sentiment_data())
        cards = build_evidence_cards(sample_financial_data(), sample_sentiment_data(), recommendation)

        self.assertEqual(len(cards), 3)
        self.assertEqual(cards[1]["label"], "Trust level")
        self.assertEqual(cards[2]["value"], "12 articles")

    def test_display_helpers_return_expected_labels(self):
        bullish = get_sentiment_display("BULLISH")
        buy_label, _ = get_recommendation_display("buy")

        self.assertEqual(bullish["label"], "Positive Outlook")
        self.assertEqual(buy_label, "Buy")

    def test_search_tool_relevance_matches_company_or_ticker(self):
        result = {
            "title": "Apple supplier update",
            "content": "AAPL supply chain commentary points to stronger iPhone demand.",
            "url": "https://example.com/apple-supplier",
        }
        self.assertTrue(_is_relevant_result(result, "AAPL", "Apple Inc."))

    def test_search_tool_relevance_rejects_unrelated_article(self):
        result = {
            "title": "Dragon Mining profit soars",
            "content": "The mining company posted strong quarterly results.",
            "url": "https://example.com/dragon-mining",
        }
        self.assertFalse(_is_relevant_result(result, "AAPL", "Apple Inc."))

    def test_sentiment_parser_accepts_plain_text_article_blocks(self):
        payload = (
            "Title: Apple beats earnings expectations\n"
            "Content: AAPL reported stronger than expected services growth.\n"
            "---\n"
            "Title: Analysts lift Apple target\n"
            "Content: Brokers raised price targets after the earnings release."
        )
        parsed = _parse_text_items(payload)

        self.assertEqual(len(parsed), 2)
        self.assertEqual(parsed[0]["title"], "Apple beats earnings expectations")
        self.assertIn("services growth", parsed[0]["content"])

    def test_agent_memo_validation_accepts_expected_structure(self):
        memo = """# Agentic Deep Dive: AAPL

## Executive Summary
Apple remains well covered, but the current setup still requires verification against primary sources.

## Key Developments
- Services growth remains in focus.

## Risks & Watchpoints
- Consumer demand could soften.

## Bull / Base / Bear
- Bull: Margins and services stay resilient.
- Base: Growth remains steady.
- Bear: Demand weakens and valuation compresses.

## Open Questions
- How durable is device demand?

## Source Caveats
- Article coverage is sampled and may miss important developments.

This narrative is supplementary research synthesis. Verify all key inputs against primary sources before acting.
"""
        is_valid, _ = validate_agent_memo(memo)
        self.assertTrue(is_valid)

    def test_agent_memo_validation_rejects_missing_sections(self):
        is_valid, message = validate_agent_memo("# Agentic Deep Dive: AAPL\n\n## Executive Summary\nShort note.")
        self.assertFalse(is_valid)
        self.assertIn("missing required sections", message)


if __name__ == "__main__":
    unittest.main()
