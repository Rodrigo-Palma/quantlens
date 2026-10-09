"""Faithfulness eval: is a generated explanation true to the signals it was given?

The case set is a seeded grid: 3 RSI regimes x 2 trend directions x 2 volatility
levels, 10 cases per cell (n = 120). Values are drawn away from the 30/70 and
20%/40% boundaries, so a label is never ambiguous; boundary behavior is covered
by unit tests, not here. Some cells pair signals that rarely co-occur in real
prices (RSI 85 with negative momentum): they are kept on purpose, because they
test whether the model reads the numbers or pattern-matches a story.

Five checks, all decidable by code:

* ``numbers``: every number in the text is the RSI (+-1), the absolute momentum
  or volatility in percent (+-0.51 p.p.), or a window/threshold constant.
* ``rsi_label``: the text does not call the stock overbought at RSI <= 70 or
  oversold at RSI >= 30 (hedged uses such as "approaching overbought" are allowed).
* ``vol_label``: a volatility label ("low/moderate/high volatility" or
  "... price swings") matches the 20%/40% regime. Added after auditing the
  qwen3:8b cassette, before the qwen3:32b cassette was recorded.
* ``trend``: the text states the right direction and does not assert the wrong
  trend term.
* ``guardrail``: ``guardrails.validate`` passes.

A case passes only if all five do. The checks are a floor, not a quality score:
they catch contradictions and invented numbers, not clumsy or unhelpful prose.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass

from quantlens import guardrails, llm
from quantlens.evals import cassette
from quantlens.evals.stats import Rate, mcnemar_exact, mde_two_proportions, wilson
from quantlens.explain import rule_based
from quantlens.quant.regime import classify, volatility_regime
from quantlens.rag import retrieve_for_regime

SEED = 2026
CASES_PER_CELL = 10
CHECKS = ("numbers", "rsi_label", "vol_label", "trend", "guardrail")

_TICKERS = ("PETR4", "VALE3", "ITUB4", "BBAS3", "ABEV3", "BBDC4", "B3SA3", "WEGE3")
_RSI_BANDS = {"oversold": (12.0, 28.0), "neutral": (35.0, 65.0), "overbought": (72.0, 88.0)}
_MOM_BANDS = {"up": (0.02, 0.15), "down": (-0.15, -0.02)}
_VOL_BANDS = {"low": (0.10, 0.18), "high": (0.42, 0.70)}

# Numbers that may appear without being an input: look-back windows, the RSI
# scale and thresholds, the volatility thresholds and sqrt(252) scaling, and
# "2-3 sentences" style counts.
_CONSTANTS = frozenset({0.0, 1.0, 2.0, 3.0, 14.0, 20.0, 30.0, 40.0, 50.0, 70.0, 100.0, 252.0})
_RSI_TOL = 1.0
_PCT_TOL = 0.51

_NUMBER = re.compile(r"(?<![A-Za-z0-9])[-+]?\d+(?:[.,]\d+)?")
_HEDGES = frozenset(
    {"not", "no", "nor", "neither", "approaching", "nearing", "near", "toward", "towards"}
    | {"close", "below", "above", "from", "into", "rather", "than", "avoid", "avoiding"}
    | {"without", "never", "isn't", "not yet"}
)
_VOL_LABEL = re.compile(
    r"\b(low|moderate|high|elevated)\s+(?:annualized\s+)?(?:volatility|price swings)\b"
)
_UP_TERMS = re.compile(
    r"\b(?:uptrend|upward|positive momentum|gain(?:s|ed)?|ris(?:e|es|ing|en)|rose|increas\w*"
    r"|bullish|climb\w*|advanc\w*|higher|up\b)"
)
_DOWN_TERMS = re.compile(
    r"\b(?:downtrend|downward|negative momentum|declin\w*|fell|fall(?:s|ing|en)?|drop\w*"
    r"|lower|bearish|loss(?:es)?|down\b)"
)
_STRICT = {
    "up": re.compile(r"\b(?:uptrend|upward trend|positive momentum|bullish trend)\b"),
    "down": re.compile(r"\b(?:downtrend|downward trend|negative momentum|bearish trend)\b"),
}


@dataclass(frozen=True)
class Case:
    case_id: str
    ticker: str
    rsi: float
    mom: float
    vol: float

    def prompt(self) -> str:
        """The exact prompt the API would send for these signals."""
        context = "\n\n".join(retrieve_for_regime(classify(self.rsi, self.mom, self.vol)))
        return llm.build_prompt(self.ticker, self.rsi, self.mom, self.vol, context)


@dataclass(frozen=True)
class Verdict:
    case_id: str
    checks: dict[str, bool]
    notes: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return all(self.checks.values())


def build_cases(seed: int = SEED, per_cell: int = CASES_PER_CELL) -> tuple[Case, ...]:
    rng = random.Random(seed)
    cases = []
    for rsi_name, (rsi_lo, rsi_hi) in _RSI_BANDS.items():
        for mom_name, (mom_lo, mom_hi) in _MOM_BANDS.items():
            for vol_name, (vol_lo, vol_hi) in _VOL_BANDS.items():
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
    return tuple(cases)


def _numbers_ok(text: str, case: Case) -> tuple[bool, list[str]]:
    bad = []
    for token in _NUMBER.findall(text):
        value = abs(float(token.replace(",", ".")))
        if value in _CONSTANTS:
            continue
        if abs(value - case.rsi) <= _RSI_TOL:
            continue
        if abs(value - abs(case.mom) * 100) <= _PCT_TOL:
            continue
        if abs(value - case.vol * 100) <= _PCT_TOL:
            continue
        bad.append(token)
    return not bad, [f"unsupported number {t}" for t in bad]


def _asserted(text: str, word: str) -> bool:
    """True if ``word`` appears at least once without a hedge just before it."""
    for match in re.finditer(rf"\b{word}\b", text):
        before = re.findall(r"[a-z']+", text[max(0, match.start() - 40) : match.start()])[-3:]
        if not any(token in _HEDGES for token in before):
            return True
    return False


def _rsi_label_ok(text: str, case: Case) -> tuple[bool, list[str]]:
    notes = []
    if case.rsi <= 70 and _asserted(text, "overbought"):
        notes.append(f"says overbought at RSI {case.rsi}")
    if case.rsi >= 30 and _asserted(text, "oversold"):
        notes.append(f"says oversold at RSI {case.rsi}")
    return not notes, notes


def _vol_label_ok(text: str, case: Case) -> tuple[bool, list[str]]:
    expected = volatility_regime(case.vol)
    notes = []
    for match in _VOL_LABEL.finditer(text):
        label = "high" if match.group(1) == "elevated" else match.group(1)
        if label != expected and _asserted(text, re.escape(match.group(0))):
            notes.append(f"calls {case.vol:.1%} volatility {label}, regime is {expected}")
    return not notes, notes


def _trend_ok(text: str, case: Case) -> tuple[bool, list[str]]:
    right, wrong = ("up", "down") if case.mom > 0 else ("down", "up")
    terms = _UP_TERMS if right == "up" else _DOWN_TERMS
    notes = []
    if not terms.search(text):
        notes.append(f"no {right} direction stated")
    match = _STRICT[wrong].search(text)
    if match and _asserted(text, re.escape(match.group(0))):
        notes.append(f"asserts {match.group(0)} with momentum {case.mom:+.2%}")
    return not notes, notes


def score(text: str | None, case: Case) -> Verdict:
    """Run every check; a missing generation fails all of them."""
    if not text:
        return Verdict(case.case_id, dict.fromkeys(CHECKS, False), ("no output",))
    lowered = text.lower()
    numbers, n_notes = _numbers_ok(lowered, case)
    label, l_notes = _rsi_label_ok(lowered, case)
    vol_label, v_notes = _vol_label_ok(lowered, case)
    trend, t_notes = _trend_ok(lowered, case)
    guard = guardrails.validate(text)
    checks = {
        "numbers": numbers,
        "rsi_label": label,
        "vol_label": vol_label,
        "trend": trend,
        "guardrail": guard.ok,
    }
    guard_notes = [f"guardrail {v}" for v in guard.violations]
    notes = tuple(n_notes + l_notes + v_notes + t_notes + guard_notes)
    return Verdict(case.case_id, checks, notes)


def rule_based_texts(cases: tuple[Case, ...]) -> dict[str, str]:
    """The deterministic baseline's output for every case."""
    return {c.case_id: rule_based(c.ticker, c.rsi, c.mom, c.vol) for c in cases}


DEFAULT_MODEL = "qwen3:32b"
BASELINE = "rule_based (baseline)"


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
    for model, records in cassette.load_all().items():
        current = {r.case_id: r for r in records if r.case_id in by_id}
        verdicts = tuple(
            score(current[c.case_id].text if c.case_id in current else None, c) for c in cases
        )
        stale = sum(
            1
            for c in cases
            if c.case_id not in current
            or current[c.case_id].prompt_sha256 != cassette.prompt_hash(c.prompt())
        )
        scores.append(SystemScore(model, verdicts, stale))
    return scores


def _paired(a: SystemScore, b: SystemScore) -> tuple[int, int, float]:
    only_a = sum(x.passed and not y.passed for x, y in zip(a.verdicts, b.verdicts, strict=True))
    only_b = sum(y.passed and not x.passed for x, y in zip(a.verdicts, b.verdicts, strict=True))
    return only_a, only_b, mcnemar_exact(only_a, only_b)


def report(scores: list[SystemScore]) -> list[str]:
    n = scores[0].n if scores else 0
    lines = [f"Faithfulness: {n} cases, pass = all {len(CHECKS)} checks (95% Wilson)"]
    lines.append(f"  {'system':24} {'pass':>28}   " + "  ".join(f"{c:>9}" for c in CHECKS))
    for s in scores:
        per_check = "  ".join(f"{s.check_rate(c).value:9.1%}" for c in CHECKS)
        stale = f"  [{s.stale_prompts} stale prompts]" if s.stale_prompts else ""
        lines.append(f"  {s.system:24} {s.rate()!s:>28}   {per_check}{stale}")
    models = {s.system: s for s in scores if s.system != BASELINE}
    if DEFAULT_MODEL in models:
        default = models[DEFAULT_MODEL]
        mde = mde_two_proportions(default.rate().value, default.n)
        lines.append(
            f"  MDE vs {DEFAULT_MODEL} at n={default.n} (alpha 0.05, power 0.80): {mde:.1%}"
        )
        for other in [scores[0], *[s for s in models.values() if s is not default]]:
            only_a, only_b, p = _paired(default, other)
            lines.append(
                f"  paired {DEFAULT_MODEL} vs {other.system}: "
                f"only {DEFAULT_MODEL} passes {only_a}, only other passes {only_b}, "
                f"exact McNemar p = {p:.3g}"
            )
    return lines


def failures(scores: list[SystemScore], system: str) -> list[tuple[str, tuple[str, ...]]]:
    """(case_id, notes) for every failed case of ``system``; used for manual audit."""
    target = next(s for s in scores if s.system == system)
    return [(v.case_id, v.notes) for v in target.verdicts if not v.passed]


def measured(scores: list[SystemScore]) -> dict[str, float]:
    out: dict[str, float] = {}
    for s in scores:
        key = "baseline" if s.system == BASELINE else s.system
        out[f"faithfulness.{key}.pass_rate"] = s.rate().value
        out[f"faithfulness.{key}.stale_prompts"] = float(s.stale_prompts)
    return out
