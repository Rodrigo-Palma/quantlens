"""Checker audit: how often do the faithfulness checks catch a known error?

A pass rate of 100% means little if the checker cannot see the errors it claims
to rule out. This module takes every recorded LLM output that the current checker
passes, injects one known error at a time, and counts how many mutated texts the
checker fails. The same mutated texts are scored by the v0.6 checker
(``checks_v06``) as the baseline.

Mutations (each applied only where the source text has the thing to mutate):

* ``sign_flip``: the momentum figure is written with the opposite sign.
* ``number_swap``: the momentum and volatility figures trade places.
* ``invented_value``: the volatility figure is moved by 9 points.
* ``invented_constant``: the volatility figure becomes 40% (30% in the high
  regime), a number the v0.6 checker allowed anywhere.
* ``direction_flip``: every trend term and direction verb is inverted
  ("uptrend" -> "downtrend", "rose" -> "fell"), numbers untouched.
* ``wrong_verb``: the sentence holding the momentum figure is restated with the
  opposite verb and no trend term ("ABEV3 rose 13.05% over the last 20
  sessions." when it fell), the error the v0.6 lexicon could not see.
* ``rsi_label_swap``: "oversold" <-> "overbought"; "neutral" -> "overbought".
* ``vol_label_swap``: the volatility label is replaced by a wrong one.
* ``comparison_flip``: "above" <-> "below" (and "acima" <-> "abaixo") where the
  word compares against a threshold ("17.06%, below 20%" -> "above 20%");
  comparisons with no threshold number ("above its average") are left alone.

Every mutated text is wrong by construction, so the rate is a recall. Its limit:
the mutations were designed together with the v0.7 checker, so their recall is
in-sample. The out-of-sample number is ``heldout``: errors planted by another
model (``scripts/generate_checker_errors.py``) in correct outputs, each read by
hand to confirm it is wrong, scored once by both checkers.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from importlib import resources

from quantlens.evals import cassette, checks, checks_v06
from quantlens.evals.cases import Case, Verdict, build_cases
from quantlens.evals.stats import Rate, wilson
from quantlens.quant.regime import rsi_regime, volatility_regime

MUTATIONS = (
    "sign_flip",
    "number_swap",
    "invented_value",
    "invented_constant",
    "direction_flip",
    "wrong_verb",
    "rsi_label_swap",
    "vol_label_swap",
    "comparison_flip",
)
INVENTED_SHIFT = 9.0
CHECKERS: dict[str, Callable[[str | None, Case], Verdict]] = {
    "v0.6 checker": checks_v06.score,
    "v0.7 checker": checks.score,
}

_PERCENT = re.compile(r"(?<![A-Za-z0-9.])([-+]?)(\d+(?:\.\d+)?)(?=\s*%)")
_DIRECTION_SWAPS = {
    "uptrend": "downtrend",
    "downtrend": "uptrend",
    "upward": "downward",
    "downward": "upward",
    "positive": "negative",
    "negative": "positive",
    "bullish": "bearish",
    "bearish": "bullish",
    "rose": "fell",
    "fell": "rose",
    "risen": "fallen",
    "fallen": "risen",
    "rising": "falling",
    "falling": "rising",
    "gained": "lost",
    "lost": "gained",
    "gain": "loss",
    "climbed": "dropped",
    "dropped": "climbed",
    "declined": "rose",
    "decline": "gain",
    "increased": "decreased",
    "decreased": "increased",
    "higher": "lower",
    "lower": "higher",
}
_DIRECTION = re.compile(r"\b(" + "|".join(_DIRECTION_SWAPS) + r")\b", re.IGNORECASE)
_RSI_LABEL = re.compile(r"\b(oversold|overbought|neutral)\b", re.IGNORECASE)
_VOL_WORD = re.compile(
    r"\b(low|moderate|high|elevated)(\s+(?:annualized\s+)?"
    r"(?:volatility|price swings|(?:price\s+)?fluctuations))\b",
    re.IGNORECASE,
)

_COMPARISON_SWAPS = {"above": "below", "below": "above", "acima": "abaixo", "abaixo": "acima"}
# The comparison word, up to 4 filler words ("the oversold level of", "de"), then a
# threshold that is not a window ("below its 20-day mean" is not a threshold).
_THRESHOLD_COMPARISON = re.compile(
    r"\b(above|below|acima|abaixo)\b"
    r"(?=(?:\s+(?:the|an?|of|de|do|da|level|threshold|mark|line|rsi|overbought|oversold"
    r"|neutral|high|low|moderate|volatility))*\s+(?:20|30|40|50|70)(?![\d.]\d)"
    r"(?!\s*-?\s*(?:day|session|trading|period)))",
    re.IGNORECASE,
)


def _figure(text: str, case: Case, target: float) -> re.Match[str] | None:
    """The single percent figure equal to ``target`` (None if absent or ambiguous)."""
    if abs(abs(case.mom) * 100 - case.vol * 100) <= 2 * checks.PCT_TOL:
        return None
    hits = [m for m in _PERCENT.finditer(text) if abs(float(m.group(2)) - target) <= checks.PCT_TOL]
    return hits[0] if hits else None


def _replace(text: str, match: re.Match[str], group: int, new: str) -> str:
    return text[: match.start(group)] + new + text[match.end(group) :]


def _sign_flip(text: str, case: Case) -> str | None:
    match = _figure(text, case, abs(case.mom) * 100)
    if match is None:
        return None
    sign = "+" if case.mom < 0 else "-"
    return text[: match.start()] + sign + match.group(2) + text[match.end() :]


def _number_swap(text: str, case: Case) -> str | None:
    mom = _figure(text, case, abs(case.mom) * 100)
    vol = _figure(text, case, case.vol * 100)
    if mom is None or vol is None:
        return None
    first, second = sorted((mom, vol), key=lambda m: m.start())
    swapped = _replace(text, second, 2, first.group(2))
    return _replace(swapped, first, 2, second.group(2))


def _invented(text: str, case: Case, value: str) -> str | None:
    vol = _figure(text, case, case.vol * 100)
    return None if vol is None else _replace(text, vol, 2, value)


def _invented_value(text: str, case: Case) -> str | None:
    vol = _figure(text, case, case.vol * 100)
    if vol is None:
        return None
    decimals = len(vol.group(2).partition(".")[2])
    return _invented(text, case, f"{case.vol * 100 + INVENTED_SHIFT:.{decimals}f}")


def _invented_constant(text: str, case: Case) -> str | None:
    return _invented(text, case, "30" if volatility_regime(case.vol) == "high" else "40")


def _swap_word(match: re.Match[str], table: dict[str, str]) -> str:
    word = match.group(0)
    new = table[word.lower()]
    return new.capitalize() if word[0].isupper() else new


def _direction_flip(text: str, case: Case) -> str | None:
    flipped = _DIRECTION.sub(lambda m: _swap_word(m, _DIRECTION_SWAPS), text)
    return flipped if flipped != text else None


def _wrong_verb(text: str, case: Case) -> str | None:
    match = _figure(text, case, abs(case.mom) * 100)
    if match is None:
        return None
    ends = [m.end() for m in re.finditer(r"[.!?](?=\s|$)", text)]
    start = max((e for e in ends if e <= match.start()), default=0)
    end = min((e for e in ends if e > match.start()), default=len(text))
    verb = "rose" if case.mom < 0 else "fell"
    sentence = f"{case.ticker} {verb} {match.group(2)}% over the last 20 sessions."
    return f"{text[:start]} {sentence}{text[end:]}".strip()


def _rsi_label_swap(text: str, case: Case) -> str | None:
    regime = rsi_regime(case.rsi)
    table = {"oversold": "overbought", "overbought": "oversold"}
    if regime == "neutral":
        table = {"neutral": "overbought"}
    swapped = _RSI_LABEL.sub(
        lambda m: _swap_word(m, table) if m.group(0).lower() in table else m.group(0), text
    )
    return swapped if swapped != text else None


def _vol_label_swap(text: str, case: Case) -> str | None:
    wrong = "high" if volatility_regime(case.vol) != "high" else "low"
    swapped = _VOL_WORD.sub(lambda m: wrong + m.group(2), text)
    return swapped if swapped != text else None


def _comparison_flip(text: str, case: Case) -> str | None:
    flipped = _THRESHOLD_COMPARISON.sub(lambda m: _swap_word(m, _COMPARISON_SWAPS), text)
    return flipped if flipped != text else None


_MUTATORS: dict[str, Callable[[str, Case], str | None]] = {
    "sign_flip": _sign_flip,
    "number_swap": _number_swap,
    "invented_value": _invented_value,
    "invented_constant": _invented_constant,
    "direction_flip": _direction_flip,
    "wrong_verb": _wrong_verb,
    "rsi_label_swap": _rsi_label_swap,
    "vol_label_swap": _vol_label_swap,
    "comparison_flip": _comparison_flip,
}


def mutate(kind: str, text: str, case: Case) -> str | None:
    """The mutated text, or ``None`` when ``text`` has nothing to mutate."""
    return _MUTATORS[kind](text, case)


@dataclass(frozen=True)
class AuditScore:
    mutation: str
    checker: str
    recall: Rate


# The mutation pool is the rag and no-rag cassettes the v0.7 checker was built
# on. Later baselines (the ``thresholds`` prompt) are scored, not mutated, so
# adding one does not move the audit counts.
AUDIT_VARIANTS = (cassette.WITH_RETRIEVAL, cassette.WITHOUT_RETRIEVAL)


def sources() -> list[tuple[str, Case]]:
    """Distinct recorded LLM outputs (audit variants only) that the checker passes."""
    by_id = {c.case_id: c for c in build_cases()}
    seen: set[str] = set()
    out = []
    for records in cassette.load_all().values():
        for record in records:
            case = by_id.get(record.case_id)
            if record.variant not in AUDIT_VARIANTS:
                continue
            if case is None or not record.text or record.text in seen:
                continue
            if checks.score(record.text, case).passed:
                seen.add(record.text)
                out.append((record.text, case))
    return out


# Labeled by what each batch holds, not by what was asked: batch ``numeric`` asked
# for a wrong figure, sign or direction and got 32 trend flips, so it measures no
# number or sign error out of sample.
HELDOUT_KINDS = {"any": "planted, any error", "numeric": "planted, trend direction"}


def load_heldout() -> dict[str, list[tuple[str, Case]]]:
    """Reviewed errors planted by another model, by batch: (wrong text, its case)."""
    path = resources.files("quantlens.evals").joinpath("data/checker_errors.jsonl")
    by_id = {c.case_id: c for c in build_cases()}
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    out: dict[str, list[tuple[str, Case]]] = {kind: [] for kind in HELDOUT_KINDS}
    for row in rows:
        out[row["kind"]].append((row["text"], by_id[row["case_id"]]))
    return out


def run() -> list[AuditScore]:
    pool = sources()
    scores = []
    for kind, planted in load_heldout().items():
        if not planted:
            continue
        for name, score in CHECKERS.items():
            caught = sum(not score(text, case).passed for text, case in planted)
            scores.append(AuditScore(HELDOUT_KINDS[kind], name, wilson(caught, len(planted))))
    for kind in MUTATIONS:
        mutated = [(m, case) for text, case in pool if (m := mutate(kind, text, case))]
        if not mutated:
            continue
        for name, score in CHECKERS.items():
            caught = sum(not score(text, case).passed for text, case in mutated)
            scores.append(AuditScore(kind, name, wilson(caught, len(mutated))))
    return scores


def report(scores: list[AuditScore]) -> list[str]:
    lines = [
        f"Checker audit: recall on correct LLM outputs with one injected error "
        f"({len(sources())} source texts, 95% Wilson)"
    ]
    names = list(CHECKERS)
    lines.append(f"  {'mutation':26} " + " ".join(f"{n:>34}" for n in names))
    for kind in (*MUTATIONS, *HELDOUT_KINDS.values()):
        row = {s.checker: s.recall for s in scores if s.mutation == kind}
        if row:
            lines.append(f"  {kind:26} " + " ".join(f"{row[n]!s:>34}" for n in names))
    return lines


def measured(scores: list[AuditScore]) -> dict[str, int]:
    out: dict[str, int] = {}
    for s in scores:
        if s.checker == "v0.7 checker":
            labels = {label: f"heldout_{kind}" for kind, label in HELDOUT_KINDS.items()}
            key = labels.get(s.mutation, s.mutation)
            out[f"checker_audit.{key}.caught"] = s.recall.successes
            out[f"checker_audit.{key}.n"] = s.recall.n
    return out
