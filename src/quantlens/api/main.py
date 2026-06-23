"""FastAPI application exposing the analysis endpoints."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from quantlens import __version__, guardrails, llm, rag
from quantlens.config import settings
from quantlens.data.market import fetch_close
from quantlens.explain import rule_based
from quantlens.quant import signals

app = FastAPI(title=settings.app_name, version=__version__)


class HealthResponse(BaseModel):
    status: str
    version: str


class AnalyzeResponse(BaseModel):
    ticker: str
    last_price: float
    rsi: float
    momentum_20d: float
    annualized_volatility: float
    explanation: str


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness probe."""
    return HealthResponse(status="ok", version=__version__)


@app.get("/analyze", response_model=AnalyzeResponse)
def analyze(ticker: str) -> AnalyzeResponse:
    """Compute quant signals for a B3 ticker and explain them.

    The explanation is produced by the configured LLM (local Ollama model by
    default); if the model is unavailable it falls back to a deterministic
    rule-based summary, so the endpoint never fails on the explanation step.
    """
    try:
        close = fetch_close(ticker)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    symbol = ticker.upper()
    rsi_value = float(signals.rsi(close).iloc[-1])
    mom = signals.momentum(close)
    vol = signals.annualized_volatility(close)
    last = float(close.iloc[-1])

    context = "\n\n".join(rag.retrieve(f"RSI momentum volatility {symbol}", k=2))
    generated = llm.explain(symbol, rsi_value, mom, vol, context=context)
    explanation = (
        generated
        if generated and guardrails.validate(generated).ok
        else rule_based(symbol, rsi_value, mom, vol)
    )

    return AnalyzeResponse(
        ticker=symbol,
        last_price=round(last, 2),
        rsi=round(rsi_value, 1),
        momentum_20d=round(mom, 4),
        annualized_volatility=round(vol, 4),
        explanation=explanation,
    )
