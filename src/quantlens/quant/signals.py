"""Quantitative signals over equity price series.

All functions are pure and deterministic (no I/O), so they are fully
unit-testable without network access.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index using Wilder's smoothing.

    Returns a Series in the 0-100 range; an uninterrupted run of gains
    converges to 100.
    """
    if period < 1:
        raise ValueError("period must be >= 1")
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return (100.0 - (100.0 / (1.0 + rs))).rename("rsi")


def simple_returns(close: pd.Series) -> pd.Series:
    """Period-over-period simple returns."""
    return close.pct_change().rename("returns")


def annualized_volatility(close: pd.Series, periods_per_year: int = 252) -> float:
    """Annualized standard deviation of simple returns."""
    rets = simple_returns(close).dropna()
    if rets.empty:
        return float("nan")
    return float(rets.std(ddof=1) * np.sqrt(periods_per_year))


def momentum(close: pd.Series, window: int = 20) -> float:
    """Total return over the trailing ``window`` observations."""
    if len(close) <= window:
        return float("nan")
    return float(close.iloc[-1] / close.iloc[-window - 1] - 1.0)
