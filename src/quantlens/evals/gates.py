"""Regression gates: each measured metric must stay within its recorded bound.

Bounds live in ``data/gates.json``. A ``min`` bound is the 95% Wilson lower limit
recorded when the metric was last accepted; a ``max`` bound is the upper limit.
The data and the scorers are deterministic, so a crossing means the code changed
the result, not that a sample was unlucky.
"""

from __future__ import annotations

import json
from importlib import resources


def load_bounds() -> dict[str, dict[str, float]]:
    path = resources.files("quantlens.evals").joinpath("data/gates.json")
    bounds: dict[str, dict[str, float]] = json.loads(path.read_text(encoding="utf-8"))
    return bounds


def check(
    measured: dict[str, float], bounds: dict[str, dict[str, float]] | None = None
) -> list[str]:
    """Return one message per violated or missing gate (empty when all pass)."""
    bounds = load_bounds() if bounds is None else bounds
    failures = []
    for name, bound in bounds.items():
        if name not in measured:
            failures.append(f"{name}: not measured")
            continue
        value = measured[name]
        if "min" in bound and value < bound["min"]:
            failures.append(f"{name}: {value:.4f} < min {bound['min']:.4f}")
        if "max" in bound and value > bound["max"]:
            failures.append(f"{name}: {value:.4f} > max {bound['max']:.4f}")
    return failures
