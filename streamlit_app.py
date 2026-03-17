import streamlit as st
import json
import time
import os
import html
import plotly.graph_objects as go
from app.crew import run_analysis
from app.agent_guardrails import normalize_agent_memo
from app.tools.finance_tools import get_stock_data
from app.tools.sentiment_tools import analyze_sentiment
from app.report_logic import (
    build_evidence_cards,
    build_full_report,
    build_recommendation,
    format_currency,
    format_market_cap,
    get_confidence_level,
    get_company_terms,
    get_recommendation_display,
    get_sentiment_display,
    is_relevant_article,
)


# --- Page Config ---
st.set_page_config(
    page_title="Stock Research Assistant",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

THEME = {
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

# --- Custom CSS ---
st.markdown("""
<style>
    .stApp {
        background: radial-gradient(circle at top left, #182133 0%, #0f1117 42%);
    }
    .main-header {
        font-size: 2.5rem; font-weight: 700;
        background: linear-gradient(90deg, #f3f6fb, #a9c1ff);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .sub-header { font-size: 1.05rem; color: #9aa4b2; margin-top: 0; }
    .signal-card {
        border-radius: 14px;
        padding: 1rem 1.25rem;
        margin-bottom: 1rem;
        font-size: 1rem;
        border: 1px solid #2d3648;
    }
    .signal-title {
        display: block;
        font-size: 1.05rem;
        font-weight: 700;
        margin-bottom: 0.35rem;
    }
    .signal-text {
        color: #d6deea;
        line-height: 1.45;
    }
    .recommendation-card {
        border-radius: 16px;
        padding: 1.2rem 1.3rem;
        border: 1px solid #2d3648;
        background: rgba(22, 27, 34, 0.9);
        margin-bottom: 1rem;
    }
    .recommendation-label {
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #9aa4b2;
    }
    .recommendation-value {
        font-size: 1.8rem;
        font-weight: 700;
        margin: 0.2rem 0 0.5rem;
    }
    .recommendation-meta {
        color: #d6deea;
        line-height: 1.5;
    }
    .section-intro {
        color: #9aa4b2;
        margin: -0.2rem 0 1rem;
        line-height: 1.45;
    }
    .evidence-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 0.9rem;
        margin-bottom: 1rem;
    }
    .evidence-card {
        border-radius: 14px;
        padding: 1rem 1.1rem;
        border: 1px solid #2d3648;
        background: rgba(22, 27, 34, 0.86);
    }
    .evidence-label {
        color: #9aa4b2;
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.35rem;
    }
    .evidence-value {
        color: #e6edf3;
        font-size: 1.15rem;
        font-weight: 700;
        margin-bottom: 0.35rem;
    }
    .evidence-text {
        color: #d6deea;
        line-height: 1.45;
        font-size: 0.95rem;
    }
</style>
""", unsafe_allow_html=True)


# ── Helper functions ──────────────────────────────────────────────────────────


def render_evidence_grid(cards):
    html = ["<div class='evidence-grid'>"]
    for card in cards:
        html.append(
            "<div class='evidence-card'>"
            f"<div class='evidence-label'>{html_escape(card['label'])}</div>"
            f"<div class='evidence-value'>{html_escape(card['value'])}</div>"
            f"<div class='evidence-text'>{html_escape(card['text'])}</div>"
            "</div>"
        )
    html.append("</div>")
    st.markdown("".join(html), unsafe_allow_html=True)


def html_escape(value):
    return html.escape(str(value), quote=True)


def render_sentiment_sources(sentiment_data):
    articles = sentiment_data.get("sampled_articles", []) if sentiment_data else []
    if not articles:
        st.info("No article sources were captured for this run.")
        return

    for index, article in enumerate(articles, start=1):
        title = article.get("title", "Untitled article")
        source = article.get("source", "Unknown source")
        url = article.get("url", "")
        preview = article.get("content_preview", "")
        sentiment = article.get("sentiment", "unknown").title()
        confidence = article.get("confidence")
        confidence_text = f"{confidence * 100:.1f}%" if isinstance(confidence, (int, float)) else "N/A"

        st.markdown(f"**{index}. {title}**")
        st.caption(f"{source} | Sentiment: {sentiment} | Confidence: {confidence_text}")
        if preview:
            st.write(preview)
        if url:
            st.markdown(f"[Open source article]({url})")
        st.divider()


def create_sentiment_gauge(score, signal):
    display = get_sentiment_display(signal)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(score * 100, 1),
        number={"suffix": " / 100", "font": {"size": 32}},
        title={"text": display["label"], "font": {"size": 17}},
        gauge={
            "axis": {
                "range": [-100, 100],
                "tickvals": [-100, -50, 0, 50, 100],
                "ticktext": ["Very Negative", "Negative", "Neutral", "Positive", "Very Positive"],
            },
            "bar": {"color": display["color"], "thickness": 0.3},
            "bgcolor": THEME["panel"],
            "steps": [
                {"range": [-100, -15], "color": "#3a1b20"},
                {"range": [ -15,  15], "color": "#3a3017"},
                {"range": [  15, 100], "color": "#173323"},
            ],
            "threshold": {
                "line": {"color": display["color"], "width": 5},
                "thickness": 0.8,
                "value": score * 100,
            },
        },
    ))
    fig.update_layout(
        height=270,
        margin=dict(l=30, r=30, t=60, b=10),
        paper_bgcolor=THEME["panel"],
        font=dict(color=THEME["text"]),
    )
    return fig


def create_sentiment_donut(breakdown):
    fig = go.Figure(go.Pie(
        labels=["Positive", "Neutral", "Negative"],
        values=[
            breakdown.get("positive", 0),
            breakdown.get("neutral",  0),
            breakdown.get("negative", 0),
        ],
        hole=0.55,
        marker=dict(colors=[THEME["bull"], THEME["neutral"], THEME["bear"]]),
        textinfo="percent",
        textfont=dict(size=13, color=THEME["text"]),
    ))
    fig.update_layout(
        title="Article tone breakdown",
        height=270, margin=dict(l=10, r=10, t=50, b=10),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, x=0),
        paper_bgcolor=THEME["panel"],
        font=dict(color=THEME["text"]),
    )
    return fig


def create_price_target_chart(cp, tl, tm, th, wl, wh):
    x_min = min(cp, tl, tm, th, wl, wh)
    x_max = max(cp, tl, tm, th, wl, wh)
    padding = max((x_max - x_min) * 0.08, 5)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[wl, tl, cp, tm, th, wh],
        y=[0, 0, 0, 0, 0, 0],
        mode="markers",
        marker=dict(size=1, color="rgba(0,0,0,0)"),
        hoverinfo="skip",
        showlegend=False,
    ))
    fig.add_shape(type="rect", x0=wl,  x1=wh,  y0=-0.45, y1=0.45,
                  fillcolor="rgba(122, 162, 247, 0.12)", line=dict(color=THEME["accent"], width=1))
    fig.add_shape(type="rect", x0=tl,  x1=th,  y0=-0.35, y1=0.35,
                  fillcolor="rgba(53, 196, 124, 0.16)", line=dict(color=THEME["bull"], width=1))
    fig.add_shape(type="line", x0=tm,  x1=tm,  y0=-0.45, y1=0.45,
                  line=dict(color=THEME["bull"], width=2, dash="dash"))
    fig.add_shape(type="line", x0=cp,  x1=cp,  y0=-0.55, y1=0.55,
                  line=dict(color=THEME["accent"], width=3))
    for x, text, color, ypos in [
        (cp, f"<b>Now<br>${cp:.0f}</b>", THEME["accent"], 0.70),
        (tm, f"Analyst target<br>${tm:.0f}", THEME["bull"], -0.85),
        (wl, f"52W Low<br>${wl:.0f}", THEME["muted"], 0.70),
        (wh, f"52W High<br>${wh:.0f}", THEME["muted"], 0.70),
    ]:
        fig.add_annotation(x=x, y=ypos, text=text, showarrow=False,
                           font=dict(color=color, size=11))
    fig.update_layout(
        title="Where the current price sits inside the analyst range",
        xaxis_title="Price (USD)",
        xaxis=dict(range=[x_min - padding, x_max + padding], gridcolor="rgba(154,164,178,0.15)"),
        yaxis=dict(visible=False, range=[-1.1, 1.1]),
        height=230, margin=dict(l=40, r=40, t=60, b=60),
        paper_bgcolor=THEME["panel"], plot_bgcolor=THEME["panel"],
        font=dict(color=THEME["text"]),
    )
    return fig


def hex_to_rgba(hex_color, alpha):
    hex_color = hex_color.lstrip("#")
    if len(hex_color) != 6:
        raise ValueError(f"Expected a 6-digit hex color, got: {hex_color}")
    red = int(hex_color[0:2], 16)
    green = int(hex_color[2:4], 16)
    blue = int(hex_color[4:6], 16)
    return f"rgba({red}, {green}, {blue}, {alpha})"


def run_quick_sentiment(ticker, company_name="", max_results=12):
    """Fetch news via Tavily + score with local FinBERT (no LLM tokens)."""
    from tavily import TavilyClient
    client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    resp = client.search(
        query=f'"{ticker}" "{company_name}" stock news earnings analyst'.strip(),
        search_depth="basic", max_results=max_results, topic="news",
    )
    source_articles = []
    articles = []
    for result in resp.get("results", []):
        if not result.get("title"):
            continue
        if not is_relevant_article(result, ticker, company_name):
            continue
        content_preview = result.get("content", "")[:300]
        source_name = result.get("source")
        if not source_name and result.get("url"):
            source_name = result.get("url", "").split("/")[2]
        articles.append({"title": result.get("title", ""), "content": content_preview})
        source_articles.append({
            "title": result.get("title", ""),
            "url": result.get("url", ""),
            "source": source_name or "Unknown source",
            "content_preview": content_preview,
        })
    if not articles:
        return None
    analysis = json.loads(analyze_sentiment.run(json.dumps(articles)))
    per_article = analysis.get("per_article_sentiment", [])
    for article, sentiment_row in zip(source_articles, per_article):
        article["sentiment"] = sentiment_row.get("sentiment", "unknown")
        article["confidence"] = sentiment_row.get("confidence")
    analysis["sampled_articles"] = source_articles
    return analysis
# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<p class="main-header">Stock Research Assistant</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">Research support for public equities, combining market data, analyst targets, and news sentiment in a clear decision workflow.</p>',
    unsafe_allow_html=True,
)
st.divider()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Research Setup")
    ticker = st.text_input(
        "Ticker",
        value="AAPL", max_chars=10, placeholder="e.g., AAPL, TSLA, MSFT",
        help="Enter the listed ticker symbol for the company you want to review.",
    ).upper().strip()
    include_agentic_deep_dive = st.checkbox(
        "Include agentic deep dive",
        value=False,
        help="Runs the CrewAI multi-agent workflow for a slower but richer narrative synthesis.",
    )
    analyze_button = st.button("Generate Research View", type="primary", width="stretch")
    st.divider()
    st.markdown("### Workflow")
    st.markdown("""
    1. **News collection** gathers recent coverage
    2. **Sentiment scoring** evaluates tone article by article
    3. **Market data review** pulls pricing and analyst targets
    4. **Research summary** combines the signals into a recommendation
    5. **Agentic deep dive** optionally generates a longer research narrative
    """)
    st.divider()
    st.caption("For research support only. Not investment advice. Verify key inputs before making decisions.")


# ── Main ──────────────────────────────────────────────────────────────────────
if analyze_button and ticker:
    status  = st.empty()
    progress = st.progress(0)
    analysis_started = time.time()

    # Step 1 — financial data
    status.info("Fetching live market data...")
    progress.progress(10)
    try:
        financial_data = json.loads(get_stock_data.run(ticker))
    except Exception as e:
        financial_data = {}
        st.warning(f"Could not fetch market data: {e}")

    # Step 2 — quick FinBERT sentiment (no LLM tokens)
    status.info("Scoring news sentiment...")
    progress.progress(20)
    try:
        company_name = financial_data.get("company_name", "") if financial_data else ""
        sentiment_data = run_quick_sentiment(ticker, company_name)
    except Exception as e:
        sentiment_data = None
        st.warning(f"Sentiment preview unavailable: {e}")

    # Quick metric strip
    if financial_data and "error" not in financial_data:
        price      = financial_data.get("current_price")
        prev_close = financial_data.get("previous_close")
        mc         = financial_data.get("market_cap")
        pe         = financial_data.get("pe_ratio")
        change_30d = financial_data.get("price_change_30d")

        delta_str = None
        if price and prev_close:
            pct = (price - prev_close) / prev_close * 100
            delta_str = f"{pct:+.2f}% vs yesterday"

        mc_display = "N/A"
        if isinstance(mc, (int, float)):
            mc_display = f"${mc/1e12:.2f}T" if mc >= 1e12 else f"${mc/1e9:.1f}B"

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Current Price",   f"${price}" if price else "N/A", delta_str)
        c2.metric("Company Value",   mc_display,
                  help="Total value of all shares — also called market capitalisation")
        c3.metric("P/E Ratio",       f"{pe:.1f}" if isinstance(pe, float) else "N/A",
                  help="Price-to-Earnings: how much investors pay per $1 of profit. Higher = more expensive.")
        c4.metric("30-Day Change",   f"{change_30d}%" if change_30d and change_30d != "N/A" else "N/A")

    # Step 3 — compile report
    status.info("Compiling the research view...")
    progress.progress(60)
    recommendation = build_recommendation(financial_data, sentiment_data)
    structured_report = build_full_report(ticker, financial_data, sentiment_data, recommendation)
    agent_result = None
    normalized_agent_memo = ""

    if include_agentic_deep_dive:
        status.info("Running agentic deep dive...")
        progress.progress(80)
        try:
            agent_result = run_analysis(ticker)
            normalized_agent_memo = normalize_agent_memo(agent_result.get("memo", ""))
        except Exception as e:
            st.warning(f"Agentic deep dive unavailable: {e}")

    elapsed = round(time.time() - analysis_started, 1)
    progress.progress(100)
    status.success(f"Analysis complete in {elapsed}s.")

    st.divider()
    tab_labels = ["Visual Summary", "Full Report"]
    if include_agentic_deep_dive:
        tab_labels.append("Agentic Deep Dive")
    tab_labels.append("Sources and Evidence")
    tab_objects = st.tabs(tab_labels)
    tab_visual = tab_objects[0]
    tab_memo = tab_objects[1]
    if include_agentic_deep_dive:
        tab_agent = tab_objects[2]
        tab_data = tab_objects[3]
    else:
        tab_agent = None
        tab_data = tab_objects[2]

    # ── Visual Summary tab ────────────────────────────────────────────
    with tab_visual:
        st.subheader("Why this verdict")
        st.markdown(
            "<p class='section-intro'>Users should be able to see the recommendation, the main evidence behind it, and the confidence limits without reading the full report.</p>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<div class='recommendation-card'>"
            f"<div class='recommendation-label'>Overall view</div>"
            f"<div class='recommendation-value' style='color:{html_escape(recommendation['color'])}'>{html_escape(recommendation['label'])}</div>"
            f"<div class='recommendation-meta'>{html_escape(recommendation['summary'])}</div>"
            f"<div class='recommendation-meta' style='margin-top:0.6rem;color:{THEME['muted']}'>{html_escape(recommendation['sentiment_sample_note'])}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
        render_evidence_grid(build_evidence_cards(financial_data, sentiment_data, recommendation))
        with st.expander("How to decide whether this analysis is trustworthy"):
            st.markdown(
                "- Check whether analyst targets, analyst consensus, and news tone broadly agree.\n"
                "- Treat the verdict as lower confidence when the app explicitly says signals conflict.\n"
                "- Use the Sources and Evidence tab to verify that the source inputs look plausible.\n"
                "- Treat this output as research support, not as a standalone trading instruction."
            )

        # Sentiment section
        st.subheader("How is the market feeling about this stock?")
        if sentiment_data:
            signal    = sentiment_data.get("overall_signal", "NEUTRAL")
            score     = sentiment_data.get("average_sentiment_score", 0)
            breakdown = sentiment_data.get("sentiment_breakdown", {})
            display   = get_sentiment_display(signal)

            st.markdown(
                f"<div class='signal-card' style='background:{display['bg']};"
                f"border-left:5px solid {display['color']}'>"
                f"<span class='signal-title' style='color:{html_escape(display['color'])}'>{html_escape(display['label'])}</span>"
                f"<span class='signal-text'>{html_escape(display['explanation'])}</span>"
                f"<span class='signal-text' style='display:block;margin-top:0.45rem;color:{THEME['muted']}'>{html_escape(recommendation['sentiment_sample_note'])}</span></div>",
                unsafe_allow_html=True,
            )
            if recommendation["conflict_note"]:
                st.caption(recommendation["conflict_note"])
            col_g, col_d = st.columns(2)
            with col_g:
                st.plotly_chart(create_sentiment_gauge(score, signal), width="stretch")
            with col_d:
                if any(breakdown.values()):
                    st.plotly_chart(create_sentiment_donut(breakdown), width="stretch")
        else:
            st.info("Sentiment data not available.")

        st.divider()

        # 30-day price chart
        st.subheader("How has the stock price moved over the last 30 days?")
        price_history = financial_data.get("price_history_30d", []) if financial_data else []
        if price_history:
            dates  = [p["date"]  for p in price_history]
            prices = [p["close"] for p in price_history]
            trend_color = "#28a745" if prices[-1] >= prices[0] else "#dc3545"
            fig_price = go.Figure()
            fig_price.add_trace(go.Scatter(
                x=dates, y=prices, mode="lines",
                fill="tozeroy",
                line=dict(color=trend_color, width=2.5),
                fillcolor=hex_to_rgba(trend_color, 0.13),
            ))
            fig_price.update_layout(
                xaxis_title="Date", yaxis_title="Price (USD)",
                height=320,
                margin=dict(l=40, r=40, t=20, b=40),
                paper_bgcolor=THEME["panel"],
                plot_bgcolor=THEME["panel"],
                font=dict(color=THEME["text"]),
                xaxis=dict(gridcolor="rgba(154,164,178,0.15)"),
                yaxis=dict(gridcolor="rgba(154,164,178,0.15)"),
            )
            st.plotly_chart(fig_price, width="stretch")

        st.divider()

        # Analyst target chart
        st.subheader("Why analysts may disagree with the headlines")
        st.markdown(
            "<p class='section-intro'>This section shows whether Wall Street targets still point higher even when recent news flow is weak.</p>",
            unsafe_allow_html=True,
        )
        if financial_data and "error" not in financial_data:
            cp  = financial_data.get("current_price")
            tl  = financial_data.get("target_low_price")
            tm  = financial_data.get("target_mean_price")
            th  = financial_data.get("target_high_price")
            wl  = financial_data.get("fifty_two_week_low")
            wh  = financial_data.get("fifty_two_week_high")
            rec = financial_data.get("recommendation_key", "")

            if all(isinstance(v, (int, float)) for v in [cp, tl, tm, th, wl, wh]):
                st.plotly_chart(
                    create_price_target_chart(cp, tl, tm, th, wl, wh),
                    width="stretch",
                )
                upside = ((tm - cp) / cp * 100) if cp else 0
                rec_label, rec_text = get_recommendation_display(rec)
                ca, cb, cc = st.columns(3)
                ca.metric("Analyst Consensus", rec_label, help=rec_text)
                cb.metric(
                    "Potential Upside to Target", f"{upside:+.1f}%",
                    help="How much the stock could gain if it reaches the analysts' average price target",
                )
                cc.metric(
                    "Target Price", format_currency(tm),
                    help="Average analyst target from the market data provider",
                )
            else:
                st.info("Price target data not available for this ticker.")

    # ── Full Report tab ───────────────────────────────────────────────
    with tab_memo:
        st.markdown(structured_report)

    # ── Agentic Deep Dive tab ────────────────────────────────────────
    if tab_agent is not None:
        with tab_agent:
            st.subheader("Agentic Deep Dive")
            st.markdown(
                "<p class='section-intro'>This narrative is generated by the CrewAI agent workflow. It is supplementary research synthesis and does not override the deterministic verdict shown in the summary.</p>",
                unsafe_allow_html=True,
            )
            st.info(
                "Use this tab for richer context, risk framing, and narrative synthesis. Use the deterministic summary and sources tabs for the authoritative numbers and recommendation."
            )
            if normalized_agent_memo:
                st.markdown(normalized_agent_memo)
            else:
                st.warning("No agentic narrative was returned for this run.")
            if agent_result and agent_result.get("token_usage"):
                with st.expander("Agent run metadata"):
                    st.markdown(f"**Ticker:** {agent_result.get('ticker', ticker)}")
                    st.json(agent_result["token_usage"])

    # ── Sources and Evidence tab ─────────────────────────────────────
    with tab_data:
        st.subheader("What the app used")
        st.markdown(
            "<p class='section-intro'>This tab is the audit trail for the recommendation: source articles, market inputs, and model run details.</p>",
            unsafe_allow_html=True,
        )
        left_col, right_col = st.columns([1.2, 0.8])
        with left_col:
            st.markdown("#### News sources used for sentiment")
            render_sentiment_sources(sentiment_data)
        with right_col:
            st.markdown("#### Market data inputs")
            if financial_data and "error" not in financial_data:
                rows = {
                    "Provider": "Yahoo Finance via yfinance",
                    "Company": financial_data.get("company_name", "N/A"),
                    "Current Price": format_currency(financial_data.get("current_price")),
                    "52-Week High": format_currency(financial_data.get("fifty_two_week_high")),
                    "52-Week Low": format_currency(financial_data.get("fifty_two_week_low")),
                    "P/E Ratio": financial_data.get("pe_ratio", "N/A"),
                    "Market Cap": format_market_cap(financial_data.get("market_cap")),
                    "Target Mean": format_currency(financial_data.get("target_mean_price")),
                    "Analyst Recommendation": financial_data.get("recommendation_key", "N/A"),
                    "Analyst Opinion Count": financial_data.get("number_of_analyst_opinions", "N/A"),
                }
                for key, value in rows.items():
                    st.markdown(f"**{key}:** {value}")
            else:
                st.info("Market data was unavailable for this run.")

            st.markdown("#### Run details")
            st.markdown(f"**Ticker:** {ticker}")
            st.markdown(f"**Execution Time:** {elapsed}s")
            st.markdown("**Sentiment Model:** FinBERT")
            st.markdown("**Recommendation Logic:** Weighted blend of analyst consensus, target upside, sentiment, and 30-day trend")
            st.markdown("**Report Generation:** Deterministic summary built from the fetched market data and filtered sentiment inputs")
            st.markdown(f"**Agentic Deep Dive:** {'Enabled' if include_agentic_deep_dive else 'Disabled'}")

elif not ticker:
    st.info("Enter a listed ticker in the sidebar and generate a research view to begin.")
else:
    st.markdown("### Start with a sample ticker")
    cols = st.columns(5)
    for i, t in enumerate(["AAPL", "TSLA", "MSFT", "NVDA", "GOOGL"]):
        cols[i].code(t)
    st.info("Enter a ticker in the sidebar and click **Generate Research View** to load the analysis.")
