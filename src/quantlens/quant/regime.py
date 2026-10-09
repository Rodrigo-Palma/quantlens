"""Map numeric signals to the discrete regimes used for retrieval and evaluation.

Thresholds follow the conventions written in the knowledge base: RSI 30/70, the
sign of 20-session momentum, and 20%/40% annualized volatility.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

RSI_OVERSOLD = 30.0
RSI_OVERBOUGHT = 70.0
VOL_LOW = 0.20
VOL_HIGH = 0.40

RsiRegime = Literal["oversold", "neutral", "overbought"]
Trend = Literal["uptrend", "downtrend"]
VolRegime = Literal["low", "moderate", "high"]


@dataclass(frozen=True)
class Regime:
    rsi: RsiRegime
    trend: Trend
    volatility: VolRegime

    def queries(self) -> tuple[str, str, str]:
        """One retrieval query per signal, in the knowledge base's own terms."""
        rsi_terms = {
            "oversold": "oversold RSI below 30",
            "neutral": "neutral RSI between 30 and 70",
            "overbought": "overbought RSI above 70",
        }[self.rsi]
        trend_terms = {
            "uptrend": "uptrend positive momentum",
            "downtrend": "downtrend negative momentum",
        }[self.trend]
        return rsi_terms, trend_terms, f"{self.volatility} volatility"

    def query(self) -> str:
        """All three signal queries joined into one."""
        return " ".join(self.queries())


def rsi_regime(rsi: float) -> RsiRegime:
    if rsi > RSI_OVERBOUGHT:
        return "overbought"
    if rsi < RSI_OVERSOLD:
        return "oversold"
    return "neutral"


def trend(mom: float) -> Trend:
    return "uptrend" if mom > 0 else "downtrend"


def volatility_regime(vol: float) -> VolRegime:
    if vol > VOL_HIGH:
        return "high"
    if vol < VOL_LOW:
        return "low"
    return "moderate"


def classify(rsi: float, mom: float, vol: float) -> Regime:
    return Regime(rsi=rsi_regime(rsi), trend=trend(mom), volatility=volatility_regime(vol))
