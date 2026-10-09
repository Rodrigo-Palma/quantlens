"""Regression gates: a ratchet on exact counts.

The eval data, the recorded generations and the scorers are deterministic, so a
metric does not move between runs unless the code moved it. A sampling interval
is the wrong tolerance for that: a Wilson lower bound would let a model lose 3 of
120 cases with CI still green. Each gate in ``data/gates.json`` therefore records
the accepted count itself:

* ``min``: a count that must not drop (cases passed, advice blocked, hits).
* ``max``: a count that must not grow (clean sentences blocked, stale prompts).
* ``eq``: a denominator that must not change without a deliberate update.

Any worsening fails. An improvement passes but is reported, and
``python -m quantlens.evals --update-gates`` rewrites the file with the measured
counts, to be committed on its own so the change of baseline is explicit. The
Wilson intervals stay in the report, where they describe how far a result might
generalize, not whether the code regressed.
"""

from __future__ import annotations

import json
from importlib import resources
from pathlib import Path

Bounds = dict[str, dict[str, int]]
_DIRECTIONS = ("min", "max", "eq")


def source_path() -> Path:
    return Path(__file__).parent / "data" / "gates.json"


def load_bounds() -> Bounds:
    path = resources.files("quantlens.evals").joinpath("data/gates.json")
    bounds: Bounds = json.loads(path.read_text(encoding="utf-8"))
    return bounds


def check(measured: dict[str, int], bounds: Bounds | None = None) -> list[str]:
    """Return one message per violated or missing gate (empty when all pass)."""
    bounds = load_bounds() if bounds is None else bounds
    failures = []
    for name, bound in bounds.items():
        if name not in measured:
            failures.append(f"{name}: not measured")
            continue
        value = measured[name]
        if "min" in bound and value < bound["min"]:
            failures.append(f"{name}: {value} < recorded {bound['min']}")
        if "max" in bound and value > bound["max"]:
            failures.append(f"{name}: {value} > recorded {bound['max']}")
        if "eq" in bound and value != bound["eq"]:
            failures.append(f"{name}: {value} != recorded {bound['eq']}")
    return failures


def improvements(measured: dict[str, int], bounds: Bounds | None = None) -> list[str]:
    """Gates the measurement beats: the ratchet can be tightened with --update-gates."""
    bounds = load_bounds() if bounds is None else bounds
    out = []
    for name, bound in bounds.items():
        value = measured.get(name)
        if value is None:
            continue
        if ("min" in bound and value > bound["min"]) or ("max" in bound and value < bound["max"]):
            recorded = bound.get("min", bound.get("max"))
            out.append(f"{name}: {value} (recorded {recorded})")
    return out


def updated(measured: dict[str, int], bounds: Bounds) -> Bounds:
    """``bounds`` with every recorded count replaced by the measured one."""
    out: Bounds = {}
    for name, bound in bounds.items():
        if name not in measured:
            raise KeyError(f"{name}: not measured")
        out[name] = {d: measured[name] for d in _DIRECTIONS if d in bound}
    return out


def write(bounds: Bounds, path: Path | None = None) -> Path:
    target = path or source_path()
    lines = [f"  {json.dumps(name)}: {json.dumps(bound)}" for name, bound in bounds.items()]
    target.write_text("{\n" + ",\n".join(lines) + "\n}\n", encoding="utf-8")
    return target
