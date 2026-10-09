"""Faithfulness eval: is a generated explanation true to the signals it was given?

Cases come from ``quantlens.evals.cases`` (a seeded 3 x 2 x 3 grid, n = 180) and
each output is scored by the five code checks in ``quantlens.evals.checks``. How
often those checks catch a known error is measured separately, by mutating
correct outputs (``quantlens.evals.checker_audit``).
"""

from __future__ import annotations

from dataclasses import dataclass

from quantlens.evals import cassette
from quantlens.evals.cases import CHECKS, Case, Verdict, build_cases
from quantlens.evals.checks import score
from quantlens.evals.stats import Rate, mcnemar_exact, mde_two_proportions, wilson
from quantlens.explain import rule_based

__all__ = ["CHECKS", "Case", "SystemScore", "Verdict", "build_cases", "run", "score"]


def rule_based_texts(cases: tuple[Case, ...]) -> dict[str, str]:
    """The deterministic baseline's output for every case."""
    return {c.case_id: rule_based(c.ticker, c.rsi, c.mom, c.vol) for c in cases}


DEFAULT_MODEL = "qwen3:32b"
BASELINE = "rule_based (baseline)"
# Pass rate at which the minimum detectable effect is reported. At a 100% rate
# the normal approximation degenerates, so a fixed, stated reference is used.
MDE_REFERENCE_RATE = 0.95


@dataclass(frozen=True)
class SystemScore:
    system: str
    verdicts: tuple[Verdict, ...]
    stale_prompts: int

    @property
    def n(self) -> int:
        return len(self.verdicts)

    def rate(self) -> Rate:
        return wilson(sum(v.passed for v in self.verdicts), self.n)

    def check_rate(self, check: str) -> Rate:
        return wilson(sum(v.checks[check] for v in self.verdicts), self.n)


def run() -> list[SystemScore]:
    """Score the deterministic baseline and every recorded model on the same cases."""
    cases = build_cases()
    by_id = {c.case_id: c for c in cases}
    baseline = rule_based_texts(cases)
    scores = [SystemScore(BASELINE, tuple(score(baseline[c.case_id], c) for c in cases), 0)]
    for name, records in cassette.load_all().items():
        with_context = records[0].variant == cassette.WITH_RETRIEVAL
        current = {r.case_id: r for r in records if r.case_id in by_id}
        verdicts = tuple(
            score(current[c.case_id].text if c.case_id in current else None, c) for c in cases
        )
        stale = sum(
            1
            for c in cases
            if c.case_id not in current
            or current[c.case_id].prompt_sha256 != cassette.prompt_hash(c.prompt(with_context))
        )
        scores.append(SystemScore(name, verdicts, stale))
    return scores


def _paired(a: SystemScore, b: SystemScore) -> tuple[int, int, float]:
    only_a = sum(x.passed and not y.passed for x, y in zip(a.verdicts, b.verdicts, strict=True))
    only_b = sum(y.passed and not x.passed for x, y in zip(a.verdicts, b.verdicts, strict=True))
    return only_a, only_b, mcnemar_exact(only_a, only_b)


def report(scores: list[SystemScore]) -> list[str]:
    n = scores[0].n if scores else 0
    lines = [f"Faithfulness: {n} cases, pass = all {len(CHECKS)} checks (95% Wilson)"]
    lines.append(f"  {'system':24} {'pass':>33}   " + "  ".join(f"{c:>9}" for c in CHECKS))
    for s in scores:
        per_check = "  ".join(f"{s.check_rate(c).value:9.1%}" for c in CHECKS)
        stale = f"  [{s.stale_prompts} stale prompts]" if s.stale_prompts else ""
        lines.append(f"  {s.system:24} {s.rate()!s:>33}   {per_check}{stale}")
    models = {s.system: s for s in scores if s.system != BASELINE}
    if DEFAULT_MODEL in models:
        default = models[DEFAULT_MODEL]
        mde = mde_two_proportions(MDE_REFERENCE_RATE, default.n)
        lines.append(
            f"  MDE at n={default.n}, two-sided alpha 0.05, power 0.80, around a "
            f"{MDE_REFERENCE_RATE:.0%} pass rate: {mde * 100:.1f} points"
        )
        others = [scores[0], *[s for s in models.values() if s is not default]]
        lines += [_paired_line(default, other) for other in others]
    for name, ablated in models.items():
        base = models.get(name.removesuffix(f" ({cassette.WITHOUT_RETRIEVAL})"))
        is_ablation = name.endswith(f"({cassette.WITHOUT_RETRIEVAL})")
        if is_ablation and base is not None and base.system != DEFAULT_MODEL:
            lines.append(_paired_line(base, ablated))
    return lines


def _paired_line(a: SystemScore, b: SystemScore) -> str:
    only_a, only_b, p = _paired(a, b)
    return (
        f"  paired {a.system} vs {b.system}: only the first passes {only_a}, "
        f"only the second passes {only_b}, exact McNemar p = {p:.3g}"
    )


def failures(scores: list[SystemScore], system: str) -> list[tuple[str, tuple[str, ...]]]:
    """(case_id, notes) for every failed case of ``system``; used for manual audit."""
    target = next(s for s in scores if s.system == system)
    return [(v.case_id, v.notes) for v in target.verdicts if not v.passed]


def measured(scores: list[SystemScore]) -> dict[str, int]:
    out: dict[str, int] = {}
    for s in scores:
        key = "baseline" if s.system == BASELINE else s.system
        out[f"faithfulness.{key}.passed"] = s.rate().successes
        out[f"faithfulness.{key}.n"] = s.n
        out[f"faithfulness.{key}.stale_prompts"] = s.stale_prompts
    return out
