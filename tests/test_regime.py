"""Signal to regime mapping (boundaries follow the glossary)."""

from __future__ import annotations

import pytest

from quantlens.quant import regime


@pytest.mark.parametrize(
    ("rsi", "expected"),
    [(29.9, "oversold"), (30.0, "neutral"), (70.0, "neutral"), (70.1, "overbought")],
)
def test_rsi_regime_boundaries(rsi: float, expected: str) -> None:
    assert regime.rsi_regime(rsi) == expected


@pytest.mark.parametrize(
    ("mom", "expected"), [(0.01, "uptrend"), (0.0, "downtrend"), (-0.2, "downtrend")]
)
def test_trend(mom: float, expected: str) -> None:
    assert regime.trend(mom) == expected


@pytest.mark.parametrize(
    ("vol", "expected"),
    [(0.19, "low"), (0.20, "moderate"), (0.40, "moderate"), (0.41, "high")],
)
def test_volatility_regime_boundaries(vol: float, expected: str) -> None:
    assert regime.volatility_regime(vol) == expected
