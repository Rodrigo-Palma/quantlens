"""The deterministic fallback explainer (was the v0.5 'eval', which only checked itself)."""

from __future__ import annotations

import pytest

from quantlens.explain import rule_based


@pytest.mark.parametrize(
    ("rsi", "mom", "label", "trend"),
    [
        (75.0, 0.08, "overbought", "uptrend"),
        (25.0, -0.06, "oversold", "downtrend"),
        (50.0, 0.01, "neutral", "uptrend"),
    ],
)
def test_rule_based_states_regime_and_trend(rsi: float, mom: float, label: str, trend: str) -> None:
    text = rule_based("PETR4", rsi, mom, 0.3)
    assert label in text
    assert trend in text
    assert "volatility 30.0%" in text
