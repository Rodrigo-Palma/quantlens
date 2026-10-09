"""The faithfulness case grid and the verdict type shared by both checkers.

A seeded grid: 3 RSI regimes x 2 trend directions x 3 volatility regimes, 10
cases per cell (n = 180). Values are drawn away from the 30/70 and 20%/40%
boundaries, so a label is never ambiguous; boundary behavior is covered by unit
tests, not here. Some cells pair signals that rarely co-occur in real prices
(RSI 85 with negative momentum): they are kept on purpose, because they test
whether the model reads the numbers or pattern-matches a story.

The low and high volatility cells (120 cases) date from v0.6. The moderate
cells (60 cases) were added in v0.7 from a separate seeded stream, so the v0.6
cases keep their exact values and their recorded generations stay valid.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from quantlens import llm
from quantlens.quant.regime import classify
from quantlens.rag import retrieve_for_regime

SEED = 2026
SEED_V07 = SEED + 1
CASES_PER_CELL = 10
CHECKS = ("numbers", "rsi_label", "vol_label", "trend", "guardrail")

_TICKERS = ("PETR4", "VALE3", "ITUB4", "BBAS3", "ABEV3", "BBDC4", "B3SA3", "WEGE3")
_RSI_BANDS = {"oversold": (12.0, 28.0), "neutral": (35.0, 65.0), "overbought": (72.0, 88.0)}
_MOM_BANDS = {"up": (0.02, 0.15), "down": (-0.15, -0.02)}
_VOL_BANDS = {"low": (0.10, 0.18), "high": (0.42, 0.70)}
_VOL_BANDS_V07 = {"moderate": (0.24, 0.36)}


@dataclass(frozen=True)
class Case:
    case_id: str
    ticker: str
    rsi: float
    mom: float
    vol: float

    def prompt(self, with_context: bool = True) -> str:
        """The exact prompt the API would send (``with_context=False``: no retrieval)."""
        regime = classify(self.rsi, self.mom, self.vol)
        context = "\n\n".join(retrieve_for_regime(regime)) if with_context else None
        return llm.build_prompt(self.ticker, self.rsi, self.mom, self.vol, context)


@dataclass(frozen=True)
class Verdict:
    case_id: str
    checks: dict[str, bool]
    notes: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return all(self.checks.values())


def _grid(
    rng: random.Random, vol_bands: dict[str, tuple[float, float]], per_cell: int
) -> list[Case]:
    cases = []
    for rsi_name, (rsi_lo, rsi_hi) in _RSI_BANDS.items():
        for mom_name, (mom_lo, mom_hi) in _MOM_BANDS.items():
            for vol_name, (vol_lo, vol_hi) in vol_bands.items():
                for i in range(per_cell):
                    cases.append(
                        Case(
                            case_id=f"{rsi_name}-{mom_name}-{vol_name}-{i:02d}",
                            ticker=rng.choice(_TICKERS),
                            rsi=round(rng.uniform(rsi_lo, rsi_hi), 1),
                            mom=round(rng.uniform(mom_lo, mom_hi), 4),
                            vol=round(rng.uniform(vol_lo, vol_hi), 4),
                        )
                    )
    return cases


def build_cases(per_cell: int = CASES_PER_CELL) -> tuple[Case, ...]:
    """The v0.6 low/high grid (120 cases) followed by the v0.7 moderate cells (60)."""
    v06 = _grid(random.Random(SEED), _VOL_BANDS, per_cell)
    v07 = _grid(random.Random(SEED_V07), _VOL_BANDS_V07, per_cell)
    return tuple(v06 + v07)
