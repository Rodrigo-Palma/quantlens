"""Unit tests for the quant signals (deterministic, no network)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantlens.quant import signals


def test_rsi_all_gains_is_100() -> None:
    close = pd.Series(np.arange(1, 30, dtype=float))
    assert signals.rsi(close, period=14).iloc[-1] == 100.0


def test_rsi_within_bounds() -> None:
    rng = np.random.default_rng(42)
    close = pd.Series(100 + rng.standard_normal(200).cumsum())
    rsi = signals.rsi(close).dropna()
    assert rsi.between(0.0, 100.0).all()


def test_momentum_positive_in_uptrend() -> None:
    close = pd.Series(np.linspace(10.0, 20.0, 50))
    assert signals.momentum(close, window=20) > 0


def test_annualized_volatility_non_negative() -> None:
    rng = np.random.default_rng(0)
    close = pd.Series(100 + rng.standard_normal(120).cumsum())
    assert signals.annualized_volatility(close) >= 0
