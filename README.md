# Deep Research Investment Agent

Stock research app for public equities with:

- a deterministic research view built from live market data, filtered news sentiment, and transparent recommendation logic
- an optional agentic deep dive for longer-form research synthesis

This is a research-support tool, not investment advice.

[Live Demo](https://deep-research-investment-agent-6bk9bh5mm2uedzw8juy75k.streamlit.app/)

## Features

- Live market data with analyst targets and 30-day price history
- Local FinBERT sentiment scoring on recent news
- Deterministic recommendation with visible evidence and confidence limits
- `Sources and Evidence` tab for auditability
- Optional CrewAI deep dive for risks, developments, and bull/base/bear framing
- Streamlit UI plus FastAPI endpoint

## Stack

- Streamlit
- FastAPI
- CrewAI
- FinBERT
- Tavily
- yfinance
- Plotly

## Quick Start

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Set the keys you need in `.env`:

```bash
GROQ_API_KEY=...
TAVILY_API_KEY=...
HF_TOKEN=...
```

Run the app:

```bash
streamlit run streamlit_app.py
```

Optional API:

```bash
python -m api.main
```

Optional CLI deep dive:

```bash
python -m app.crew AAPL
```

## How It Works

1. Fetch market data from `yfinance`
2. Search recent news with Tavily
3. Score article sentiment with FinBERT
4. Build a deterministic research summary
5. Optionally run a CrewAI deep dive for richer narrative context

## Project Structure

```text
app/
  agents.py
  crew.py
  report_logic.py
  tasks.py
  tools/
api/
streamlit_app.py
tests/
```

## Tests

```bash
./venv/bin/python -m unittest tests.test_report_logic
```

## Disclaimer

This project supports research workflows. Verify important inputs against primary sources before making financial decisions.
