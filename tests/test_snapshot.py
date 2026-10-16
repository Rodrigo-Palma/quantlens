"""Signal snapshot shared by the API and the Streamlit demo."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from quantlens.quant import snapshot


def _series(n: int, seed: int = 7) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(100.0 * np.exp(np.cumsum(rng.normal(0.0005, 0.02, size=n))), name="close")


def test_compute_returns_finite_signals_and_last_close() -> None:
    close = _series(120)
    snap = snapshot.compute(close)
    assert all(math.isfinite(x) for x in (snap.rsi, snap.momentum, snap.volatility, snap.last))
    assert snap.last == pytest.approx(float(close.iloc[-1]))


@pytest.mark.parametrize("points", [1, 15, snapshot.MIN_OBSERVATIONS - 1])
def test_short_series_raises_instead_of_nan(points: int) -> None:
    with pytest.raises(snapshot.SignalsUnavailableError, match="observations"):
        snapshot.compute(_series(points))


def test_minimum_length_is_accepted() -> None:
    snapshot.compute(_series(snapshot.MIN_OBSERVATIONS))


def test_flat_series_raises_undefined() -> None:
    flat = pd.Series([10.0] * 60, name="close")
    with pytest.raises(snapshot.SignalsUnavailableError, match="undefined"):
        snapshot.compute(flat)


def test_signals_unavailable_is_a_value_error() -> None:
    assert issubclass(snapshot.SignalsUnavailableError, ValueError)
