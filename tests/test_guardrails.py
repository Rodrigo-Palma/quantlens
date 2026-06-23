"""Tests for output guardrails."""

from __future__ import annotations

from quantlens import guardrails


def test_clean_explanation_passes() -> None:
    text = "PETR4 is in an uptrend; RSI 72 (overbought); annualized volatility 35%."
    assert guardrails.validate(text).ok


def test_advice_is_flagged() -> None:
    result = guardrails.validate("You should buy PETR4 now, it is guaranteed.")
    assert not result.ok
    assert result.violations
