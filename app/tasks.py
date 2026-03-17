from crewai import Task, Agent


def create_research_task(agent: Agent, ticker: str) -> Task:
    return Task(
        description=(
            f"Research the stock ticker '{ticker}' thoroughly.\n\n"
            f"1. Use the 'Search Financial News' tool to find recent news articles about {ticker}.\n"
            f"2. Use the 'Search Company Overview' tool to get company background.\n"
            f"3. Compile all findings into a structured report.\n\n"
            f"Your output MUST include:\n"
            f"- A list of 3-5 recent news articles with their titles and content summaries\n"
            f"- Key recent developments (earnings, product launches, regulatory news)\n"
            f"- Current analyst sentiment and notable analyst opinions\n"
            f"- Any major risks or concerns mentioned in recent coverage"
        ),
        expected_output=(
            "A comprehensive JSON-formatted research report containing:\n"
            "1. 'company_name': The full company name\n"
            "2. 'articles': List of articles, each with 'title' and 'content'\n"
            "3. 'key_developments': List of bullet points on recent developments\n"
            "4. 'analyst_opinions': Summary of analyst views"
        ),
        agent=agent,
    )


def create_sentiment_task(agent: Agent, ticker: str) -> Task:
    return Task(
        description=(
            f"Analyze the sentiment of the news articles gathered about {ticker}.\n\n"
            f"1. Take the articles from the research report (previous task output).\n"
            f"2. Extract the articles list and format them as simple plain text blocks with 'Title:' and 'Content:' fields.\n"
            f"3. Pass that plain-text article block to the 'Analyze Financial Sentiment' tool.\n"
            f"4. Interpret the results and provide your expert analysis.\n\n"
            f"IMPORTANT: Do not send escaped JSON inside the function call.\n"
            f"Format the tool input like this:\n"
            f"Title: ...\n"
            f"Content: ...\n"
            f"---\n"
            f"Title: ...\n"
            f"Content: ...\n"
        ),
        expected_output=(
            "A detailed sentiment analysis report including:\n"
            "1. Per-article sentiment breakdown (positive/negative/neutral with confidence scores)\n"
            "2. Aggregate sentiment metrics (counts, average score)\n"
            "3. Overall sentiment signal (BULLISH/BEARISH/NEUTRAL)\n"
            "4. A plain-English interpretation (avoid jargon) explaining what the sentiment "
            "means for everyday investors — e.g. instead of 'BULLISH', say "
            "'investors are optimistic' or 'the outlook is positive'."
        ),
        agent=agent,
    )


def create_financial_analysis_task(agent: Agent, ticker: str) -> Task:
    return Task(
        description=(
            f"Retrieve and analyze the financial data for {ticker}.\n\n"
            f"1. Use the 'Get Stock Financial Data' tool with the ticker '{ticker}'.\n"
            f"2. Analyze the key metrics returned.\n"
            f"3. Identify strengths and weaknesses in the financials.\n"
            f"4. Compare current price to analyst targets.\n"
            f"5. Assess the 30-day price trend."
        ),
        expected_output=(
            "A structured financial analysis including:\n"
            "1. Key metrics summary (price, market cap, P/E, revenue growth, etc.)\n"
            "2. Valuation assessment (overvalued/fairly valued/undervalued based on ratios)\n"
            "3. Price trend analysis (30-day momentum)\n"
            "4. Comparison to analyst price targets\n"
            "5. Financial health assessment"
        ),
        agent=agent,
    )


def create_memo_task(agent: Agent, ticker: str) -> Task:
    return Task(
        description=(
            f"Write a professional research deep dive for {ticker} by synthesizing all previous research.\n\n"
            f"You have access to:\n"
            f"1. Market research with recent news and developments\n"
            f"2. Sentiment analysis with FinBERT scores\n"
            f"3. Financial data analysis with key metrics\n\n"
            f"Write a grounded research note following this EXACT structure:\n\n"
            f"# Agentic Deep Dive: {ticker}\n\n"
            f"## Executive Summary\n"
            f"(2-3 sentence overview of the main takeaway, grounded in the provided evidence only)\n\n"
            f"## Key Developments\n"
            f"(Summarize the most important recent developments and why they matter)\n\n"
            f"## Risks & Watchpoints\n"
            f"(List 3-5 key risks, uncertainties, or signals to monitor)\n\n"
            f"## Bull / Base / Bear\n"
            f"(Present scenario framing in plain English; do not assign probabilities unless given)\n\n"
            f"## Open Questions\n"
            f"(List factual gaps or items a user should verify before acting)\n\n"
            f"## Source Caveats\n"
            f"(Explicitly state limitations in the article sample, sentiment model, or market-data coverage)\n\n"
            f"IMPORTANT:\n"
            f"- Write in clear, professional language for a general audience. Avoid or explain all jargon.\n"
            f"- Keep the tone concise, professional, and suitable for a production research product.\n"
            f"- Do not output a BUY, HOLD, or SELL call.\n"
            f"- Do not invent price targets, analyst counts, or company facts that were not provided by the tools.\n"
            f"- If evidence is weak or conflicting, say that explicitly.\n"
            f"- End with this exact line: 'This narrative is supplementary research synthesis. Verify all key inputs against primary sources before acting.'"
        ),
        expected_output=(
            "A complete, professionally formatted research note in Markdown format "
            "following the exact structure specified. It must stay grounded in the provided "
            "data and must not include an independent recommendation."
        ),
        agent=agent,
        output_file="output/memo.md",
    )
