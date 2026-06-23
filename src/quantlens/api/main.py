"""FastAPI application exposing the analysis endpoints."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from quantlens import __version__, llm
from quantlens.config import settings
from quantlens.data.market import fetch_close
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

    explanation = llm.explain(symbol, rsi_value, mom, vol) or _rule_based(
        symbol, rsi_value, mom, vol
    )

    return AnalyzeResponse(
        ticker=symbol,
        last_price=round(last, 2),
        rsi=round(rsi_value, 1),
        momentum_20d=round(mom, 4),
        annualized_volatility=round(vol, 4),
        explanation=explanation,
    )


def _rule_based(ticker: str, rsi_value: float, mom: float, vol: float) -> str:
    """Deterministic fallback explanation (no LLM required)."""
    trend = "an uptrend" if mom > 0 else "a downtrend"
    if rsi_value > 70:
        rsi_note = "overbought (RSI > 70)"
    elif rsi_value < 30:
        rsi_note = "oversold (RSI < 30)"
    else:
        rsi_note = "neutral"
    return (
        f"{ticker} is in {trend} over the last 20 sessions "
        f"(momentum {mom:+.1%}); RSI {rsi_value:.0f} ({rsi_note}); "
        f"annualized volatility {vol:.1%}."
    )
