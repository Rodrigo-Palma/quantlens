"""The v0.6 faithfulness checker, kept verbatim as the baseline of the checker audit.

It is not used to score anything. ``quantlens.evals.checker_audit`` runs it on the
same mutated outputs as the current checker, so the gain of the v0.7 rewrite is a
measured number. Its blind spots: numbers are matched by absolute value against
any signal (no sign, no attribution), the 20/30/40/70 constants pass anywhere,
and any loose direction word ("losses", "up") satisfies the trend check.
"""

from __future__ import annotations

import re

from quantlens import guardrails
from quantlens.evals.cases import CHECKS, Case, Verdict
from quantlens.quant.regime import volatility_regime

# Numbers that may appear without being an input: look-back windows, the RSI
# scale and thresholds, the volatility thresholds and sqrt(252) scaling, and
# "2-3 sentences" style counts.
_CONSTANTS = frozenset({0.0, 1.0, 2.0, 3.0, 14.0, 20.0, 30.0, 40.0, 50.0, 70.0, 100.0, 252.0})
_RSI_TOL = 1.0
_HEDGE_WINDOW = 5
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
        window = text[max(0, match.start() - 60) : match.start()]
        before = re.findall(r"[a-z']+", window)[-_HEDGE_WINDOW:]
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
