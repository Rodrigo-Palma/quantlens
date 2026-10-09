"""FastAPI application exposing the analysis endpoints."""

from __future__ import annotations

import math
from typing import Literal

from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel

from quantlens import __version__, guardrails, llm, rag
from quantlens.config import settings
from quantlens.data.market import fetch_close
from quantlens.explain import rule_based
from quantlens.observability import RequestTrace, configure_logging
from quantlens.quant import regime, signals

# RSI(14) needs 15 closes and 20-session momentum needs 21; below that a signal
# is NaN, which is not valid JSON. Reject the request instead of serving NaN.
MIN_OBSERVATIONS = 21

ExplanationSource = Literal["llm", "fallback"]

configure_logging()
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
    explanation_source: ExplanationSource
    guardrail_violations: list[str]


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness probe."""
    return HealthResponse(status="ok", version=__version__)


@app.get("/analyze", response_model=AnalyzeResponse)
def analyze(ticker: str, response: Response) -> AnalyzeResponse:
    """Compute quant signals for a B3 ticker and explain them.

    The explanation comes from the configured LLM. If the model is unavailable or
    its output trips the guardrail, a deterministic rule-based summary is served
    instead, and ``explanation_source`` says which one the caller got.
    """
    trace = RequestTrace(endpoint="/analyze", ticker=ticker.upper())
    response.headers["X-Request-ID"] = trace.request_id
    try:
        return _analyze(trace.ticker, trace)
    except HTTPException as exc:
        trace.status = exc.status_code
        exc.headers = {**(exc.headers or {}), "X-Request-ID": trace.request_id}
        raise
    finally:
        trace.emit()


def _analyze(symbol: str, trace: RequestTrace) -> AnalyzeResponse:
    with trace.stage("fetch"):
        try:
            close = fetch_close(symbol)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
    if len(close) < MIN_OBSERVATIONS:
        raise HTTPException(
            status_code=422,
            detail=f"need at least {MIN_OBSERVATIONS} observations, got {len(close)}",
        )

    with trace.stage("signals"):
        rsi_value = float(signals.rsi(close).iloc[-1])
        mom = signals.momentum(close)
        vol = signals.annualized_volatility(close)
        last = float(close.iloc[-1])
    if not all(math.isfinite(x) for x in (rsi_value, mom, vol, last)):
        raise HTTPException(status_code=422, detail="signals are undefined for these observations")

    current = regime.classify(rsi_value, mom, vol)
    with trace.stage("retrieve"):
        context = "\n\n".join(rag.retrieve_for_regime(current))
    with trace.stage("llm"):
        generated = llm.explain(symbol, rsi_value, mom, vol, context=context)
    with trace.stage("guardrail"):
        verdict = guardrails.validate(generated) if generated else None

    source, explanation, violations = _choose(symbol, rsi_value, mom, vol, generated, verdict)
    trace.attributes = {
        "explanation_source": source,
        "fallback_reason": _fallback_reason(generated, verdict),
        "guardrail_violations": violations,
        "regime": [current.rsi, current.trend, current.volatility],
        "llm_provider": settings.llm_provider,
        "llm_model": settings.llm_model,
    }
    return AnalyzeResponse(
        ticker=symbol,
        last_price=round(last, 2),
        rsi=round(rsi_value, 1),
        momentum_20d=round(mom, 4),
        annualized_volatility=round(vol, 4),
        explanation=explanation,
        explanation_source=source,
        guardrail_violations=violations,
    )


def _choose(
    symbol: str,
    rsi_value: float,
    mom: float,
    vol: float,
    generated: str | None,
    verdict: guardrails.GuardrailResult | None,
) -> tuple[ExplanationSource, str, list[str]]:
    if generated and verdict is not None and verdict.ok:
        return "llm", generated, []
    violations = list(verdict.violations) if verdict is not None else []
    return "fallback", rule_based(symbol, rsi_value, mom, vol), violations


def _fallback_reason(
    generated: str | None, verdict: guardrails.GuardrailResult | None
) -> str | None:
    if not generated:
        return "llm_unavailable"
    if verdict is not None and not verdict.ok:
        return "guardrail_violation"
    return None
