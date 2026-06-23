"""Deterministic, rule-based explanation of quant signals (no LLM required).

Used both as the safe fallback in the API and as the ground truth in the
offline evaluation harness.
"""

from __future__ import annotations


def rule_based(ticker: str, rsi: float, mom: float, vol: float) -> str:
    """Build a descriptive explanation from the computed signals."""
    trend = "an uptrend" if mom > 0 else "a downtrend"
    if rsi > 70:
        rsi_note = "overbought (RSI > 70)"
    elif rsi < 30:
        rsi_note = "oversold (RSI < 30)"
    else:
        rsi_note = "neutral"
    return (
        f"{ticker} is in {trend} over the last 20 sessions "
        f"(momentum {mom:+.1%}); RSI {rsi:.0f} ({rsi_note}); "
        f"annualized volatility {vol:.1%}."
    )
