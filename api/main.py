import time
from typing import Any

from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.crew import run_analysis
from app.config import ALLOWED_ORIGINS, API_AUTH_TOKEN

app = FastAPI(
    title="Deep Research Investment Agent API",
    description="AI-powered stock analysis using multi-agent orchestration, FinBERT sentiment analysis, and live market data.",
    version="1.0.0",
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["POST", "GET"],
    allow_headers=["Authorization", "Content-Type", "X-API-Key"],
)


# --- Schemas ---

class AnalyzeRequest(BaseModel):
    ticker: str = Field(
        ...,
        description="Stock ticker symbol (e.g., 'AAPL', 'TSLA', 'MSFT')",
        min_length=1,
        max_length=10,
        pattern=r"^[A-Za-z.\-]+$",
        examples=["AAPL"],
    )


class AnalyzeResponse(BaseModel):
    ticker: str
    memo: str
    execution_time_seconds: float
    token_usage: dict[str, Any] = Field(default_factory=dict)


def require_api_key(x_api_key: str | None) -> None:
    if not API_AUTH_TOKEN:
        return
    if x_api_key != API_AUTH_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")


# --- Endpoints ---

@app.get("/api/health")
def health_check():
    return {"status": "healthy", "service": "deep-research-agent"}


@app.post("/api/analyze", response_model=AnalyzeResponse)
def analyze_stock(request: AnalyzeRequest, x_api_key: str | None = Header(default=None, alias="X-API-Key")):
    """
    Run the full investment analysis pipeline for a stock ticker.

    This kicks off a multi-agent crew that:
    1. Researches recent news via Tavily
    2. Analyzes sentiment with FinBERT
    3. Pulls financial data via yfinance
    4. Generates a structured investment memo
    """
    start_time = time.time()
    require_api_key(x_api_key)

    try:
        result = run_analysis(request.ticker)
    except EnvironmentError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed for {request.ticker}: {str(e)}"
        )

    execution_time = round(time.time() - start_time, 2)

    return AnalyzeResponse(
        ticker=result["ticker"],
        memo=result["memo"],
        execution_time_seconds=execution_time,
        token_usage=result.get("token_usage", {}),
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
