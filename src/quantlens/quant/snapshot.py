"""The signals of one close series, or a clear reason they cannot be computed.

Shared by the API and the Streamlit demo, so both reject a series the signals
are undefined for instead of reporting NaN.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

from quantlens.quant import signals

# RSI(14) needs 15 closes and 20-session momentum needs 21; below that a signal
# is NaN, which is not valid JSON and not a reading anyone can use.
MIN_OBSERVATIONS = 21


class SignalsUnavailableError(ValueError):
    """The series is too short, or the signals are not finite numbers for it."""


@dataclass(frozen=True)
class Snapshot:
    rsi: float
    momentum: float
    volatility: float
    last: float


def require_observations(close: pd.Series) -> None:
    """Raise ``SignalsUnavailableError`` when ``close`` is shorter than the signals need."""
    if len(close) < MIN_OBSERVATIONS:
        raise SignalsUnavailableError(
            f"need at least {MIN_OBSERVATIONS} observations, got {len(close)}"
        )


def compute(close: pd.Series) -> Snapshot:
    """RSI(14), 20-session momentum, annualized volatility and the last close."""
    require_observations(close)
    snap = Snapshot(
        rsi=float(signals.rsi(close).iloc[-1]),
        momentum=signals.momentum(close),
        volatility=signals.annualized_volatility(close),
        last=float(close.iloc[-1]),
    )
    if not all(math.isfinite(x) for x in (snap.rsi, snap.momentum, snap.volatility, snap.last)):
        raise SignalsUnavailableError("signals are undefined for these observations")
    return snap
