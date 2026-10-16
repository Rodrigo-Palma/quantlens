"""Code checks that decide whether an explanation is faithful to its signals (v0.7).

Five checks, all decidable by code:

* ``numbers``: every number is bound to the signal it describes. The signal is
  the nearest signal word in the same sentence ("momentum" / "rose" for the
  20-session return, "volatility" / "swings" for volatility, "RSI"). A number
  passes if it is the value of that signal (RSI within 1, percentages within
  0.51 points), carries the right sign when it is written with one, or is a
  constant in a form that cannot be read as a value: a threshold after a
  comparison ("below 20%", "RSI > 70", "between 30 and 70"), a range
  ("0 to 100") or a window ("RSI(14)", "20 sessions"). A comparison against a
  threshold must also be true of the signal it is about: "RSI 75, below 70",
  "volatility of 17%, above 20%" and, at RSI 70, "below 70" all fail; "not
  above 70" holds at 70. Ranges ("within 30-70") are not judged.
* ``rsi_label``: no unhedged "overbought" at RSI <= 70, "oversold" at RSI >= 30,
  or "neutral" outside 30-70.
* ``vol_label``: a "low/moderate/high/elevated volatility | price swings |
  price fluctuations" label matches the 20%/40% regime.
* ``trend``: the text states the right direction, and asserts no wrong one:
  neither a trend term ("uptrend", "positive momentum") nor a direction verb
  ("rose", "rallied", "fell") in a sentence about the 20-session return.
* ``guardrail``: ``guardrails.validate`` passes.

The checks are a floor: they catch contradictions and invented numbers, not
clumsy or unhelpful prose. Their recall on known errors is measured by
``quantlens.evals.checker_audit``; the v0.6 checker they replace is kept in
``checks_v06`` as its baseline.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from quantlens import guardrails
from quantlens.evals.cases import CHECKS, Case, Verdict
from quantlens.quant.regime import rsi_regime, volatility_regime

RSI_TOL = 1.0
PCT_TOL = 0.51
_HEDGE_WINDOW = 5

_NUMBER = re.compile(r"(?<![A-Za-z0-9.])([-+]?)(\d+(?:\.\d+)?)(\s*(?:%|percent\b))?")
_SENTENCE_END = re.compile(r"[.!?](?=\s|$)")
_SIGNAL_WORDS = {
    "rsi": re.compile(r"\brsi\b|relative strength"),
    "momentum": re.compile(
        r"\b(?:momentum|returns?|performance|uptrend|downtrend|trend|rall(?:y|ied)|rose|risen"
        r"|climbed|gain|gained|advanced|increased|fell|fallen|decline|declined|dropped|lost"
        r"|decreased)\b"
    ),
    "volatility": re.compile(r"\bvolatil\w*|\bswings?\b|\bfluctuations?\b|standard deviation"),
}
_THRESHOLDS = frozenset({0.0, 20.0, 30.0, 40.0, 50.0, 70.0, 100.0})
_WINDOWS = frozenset({14.0, 20.0, 252.0})
_COMPARISON = re.compile(
    r"(?:below|under|above|over|than|exceeds?|exceeding|threshold(?: of)?|around|near"
    r"|<|>|<=|>=)\s*(?:the\s+)?(?:rsi\s+)?(?:level\s+of\s+)?$"
)
# "-", en dash and em dash all join a range ("30-70", "30\u201370").
_DASH = "(?:-|\u2013|\u2014)"
_RANGE = re.compile(
    rf"\b(?:between|from)\s+(\d+)\s*%?\s*(?:and|to|{_DASH})\s*(\d+)\s*%?"
    rf"|\b(\d+)\s*%?\s*(?:{_DASH}|to)\s*(\d+)\b"
)
# A comparison word, up to 4 filler words, then the threshold (``$`` = the number).
_FILLER = (
    r"the|an?|its|of|de|do|da|level|threshold|mark|line|rsi|overbought|oversold|neutral"
    r"|high|low|moderate|volatility|regime|band|boundary"
)
_CLAIM = re.compile(
    rf"(?P<neg>\b(?:not|no longer|never|n[aã]o(?: est[aá])?)\s+)?"
    r"\b(?P<op>below|under|beneath|above|over|exceed(?:s|ed|ing)?|surpass(?:es|ed|ing)?"
    r"|(?:less|lower|more|greater|higher)\s+than|abaixo|acima)\b"
    rf"(?:\s+(?:{_FILLER}))*\s*$"
)
_BELOW = frozenset({"below", "under", "beneath", "less than", "lower than", "abaixo"})
_COMPARABLE = frozenset({"rsi", "volatility"})
_CLAUSE_BREAK = re.compile(r"[,;:]")
_WINDOW_WORD = r"^\s*-?\s*(?:day|session|trading|period|month)"
_WINDOW_AFTER = re.compile(rf"{_WINDOW_WORD}|^\)")

_HEDGES = frozenset(
    {"not", "no", "nor", "neither", "approaching", "nearing", "near", "toward", "towards"}
    | {"close", "below", "above", "from", "into", "rather", "than", "avoid", "avoiding"}
    | {"without", "never", "isn't", "not yet", "between"}
)
_VOL_LABEL = re.compile(
    r"\b(low|moderate|high|elevated)\s+(?:annualized\s+)?"
    r"(?:volatility|price swings|(?:price\s+)?fluctuations)\b"
)
_TREND_TERMS = {
    "up": re.compile(
        r"\b(?:uptrend|upward(?: trend| momentum| price movement)?|positive (?:momentum|trend)"
        r"|bullish trend)\b"
    ),
    "down": re.compile(
        r"\b(?:downtrend|downward(?: trend| momentum| price movement)?"
        r"|negative (?:momentum|trend)|bearish trend)\b"
    ),
}
_DIRECTION_VERBS = {
    "up": re.compile(
        r"\b(?:rose|risen|rises|climbed|climbs|rall(?:y|ied|ies)|gained|advanced|increased"
        r"|appreciated|(?:is|was|went|moved|trended) up|higher than)\b"
    ),
    "down": re.compile(
        r"\b(?:fell|fallen|falls(?!\s+(?:into|within|in|under|below|above|between))|declined|dropped|lost|slid|slipped|decreased|depreciated"
        r"|(?:is|was|went|moved|trended) down|lower than)\b"
    ),
}

# "both upward and downward movement" names two directions at once: no claim.
_TWO_SIDED = re.compile(r"\b(?:up|down)(?:ward)?\s+(?:and|or)\s+(?:up|down)(?:ward)?\b")


@dataclass(frozen=True)
class _Number:
    sign: str
    value: float
    is_percent: bool
    start: int
    end: int


def _numbers(text: str) -> list[_Number]:
    return [
        _Number(m.group(1), float(m.group(2)), bool(m.group(3)), m.start(), m.end(2))
        for m in _NUMBER.finditer(text)
    ]


def _sentence_spans(text: str) -> list[tuple[int, int]]:
    """Sentence boundaries; a period inside "13.05" is not followed by a space."""
    spans, start = [], 0
    for match in _SENTENCE_END.finditer(text):
        spans.append((start, match.end()))
        start = match.end()
    if text[start:].strip():
        spans.append((start, len(text)))
    return spans


def _context(text: str, num: _Number, span: tuple[int, int]) -> str | None:
    """The signal whose word is nearest to ``num`` in its sentence (ties go left).

    A word after the number only counts within the same clause ("17.06% volatility"),
    not across a comma ("-7.75%, and the RSI").
    """
    best: tuple[int, int, str] | None = None
    for signal, pattern in _SIGNAL_WORDS.items():
        for match in pattern.finditer(text, span[0], span[1]):
            if match.end() <= num.start:
                key = (num.start - match.end(), 0, signal)
            elif match.start() >= num.end and not _CLAUSE_BREAK.search(
                text, num.end, match.start()
            ):
                key = (match.start() - num.end, 1, signal)
            else:
                continue
            best = key if best is None or key < best else best
    return best[2] if best else None


def _range_positions(text: str) -> set[int]:
    """Start offsets of numbers inside a range such as "between 30 and 70"."""
    positions: set[int] = set()
    for match in _RANGE.finditer(text):
        groups = [i for i in range(1, 5) if match.group(i)]
        values = [float(match.group(i)) for i in groups]
        if all(v in _THRESHOLDS for v in values):
            positions.update(match.start(i) for i in groups)
    return positions


def _claim(text: str, num: _Number) -> tuple[str, bool] | None:
    """("below" | "above", negated) if a threshold ``num`` closes a comparison."""
    if num.sign or num.value not in _THRESHOLDS or re.search(_WINDOW_WORD, text[num.end :]):
        return None
    match = _CLAIM.search(text[max(0, num.start - 60) : num.start])
    if match is None:
        return None
    op = re.sub(r"\s+", " ", match.group("op"))
    return ("below" if op in _BELOW else "above"), bool(match.group("neg"))


def _comparison_note(text: str, num: _Number, case: Case, span: tuple[int, int]) -> str | None:
    """A note if ``num`` is a threshold the text compares its signal against wrongly."""
    claim = _claim(text, num)
    signal = _context(text, num, span)
    if claim is None or signal not in _COMPARABLE:
        return None
    direction, negated = claim
    actual = case.rsi if signal == "rsi" else round(case.vol * 100, 2)
    holds = actual < num.value if direction == "below" else actual > num.value
    if holds != negated:
        return None
    said = f"{'not ' if negated else ''}{direction} {num.value:g}"
    name = "RSI" if signal == "rsi" else "volatility"
    return f"says {name} {actual:g} is {said} (comparison)"


def _is_constant(text: str, num: _Number, ranges: set[int]) -> bool:
    if num.sign == "-":
        return False
    if num.start in ranges or _claim(text, num):
        return True
    if num.value in _THRESHOLDS and _COMPARISON.search(text[max(0, num.start - 40) : num.start]):
        return True
    after = text[num.end : num.end + 12]
    window_form = _WINDOW_AFTER.search(after) or text[num.start - 1 : num.start] == "("
    return num.value in _WINDOWS and not num.is_percent and bool(window_form)


def _roles(num: _Number, case: Case) -> set[str]:
    roles = set()
    if not num.is_percent and abs(num.value - case.rsi) <= RSI_TOL:
        roles.add("rsi")
    if abs(num.value - abs(case.mom) * 100) <= PCT_TOL:
        roles.add("momentum")
    if abs(num.value - case.vol * 100) <= PCT_TOL:
        roles.add("volatility")
    return roles


def _sign_ok(num: _Number, role: str, case: Case) -> bool:
    if not num.sign:
        return True
    if role != "momentum":
        return num.sign == "+"
    return (num.sign == "-") == (case.mom < 0)


def _role(text: str, num: _Number, case: Case, span: tuple[int, int]) -> str | None:
    """The signal ``num`` is the value of, read in context; ``None`` if it is none."""
    roles = _roles(num, case)
    context = _context(text, num, span)
    if context in roles:
        return context
    if roles and context is None:
        return sorted(roles)[0]
    return None


def _number_note(text: str, num: _Number, case: Case, span: tuple[int, int]) -> str | None:
    token = f"{num.sign}{num.value:g}{'%' if num.is_percent else ''}"
    role = _role(text, num, case, span)
    if role is not None:
        if _sign_ok(num, role, case):
            return None
        return f"{token} has the wrong sign for {role} {case.mom:+.2%}"
    if _is_constant(text, num, _range_positions(text)):
        return None
    roles = _roles(num, case)
    if roles:
        context = _context(text, num, span)
        return f"{token} is the {'/'.join(sorted(roles))} value, written as {context}"
    return f"unsupported number {token}"


def numbers_ok(text: str, case: Case) -> tuple[bool, list[str]]:
    located = _located(text)
    notes = [note for num, span in located if (note := _number_note(text, num, case, span))]
    notes += [note for num, span in located if (note := _comparison_note(text, num, case, span))]
    return not notes, notes


def asserted(text: str, pattern: str) -> bool:
    """True if ``pattern`` appears at least once without a hedge just before it.

    The hedge must be in the same clause: in "neither overbought nor oversold,
    remaining in overbought territory" the "nor" does not reach past the comma.
    """
    for match in re.finditer(rf"\b{pattern}\b", text):
        window = text[max(0, match.start() - 60) : match.start()]
        window = re.split(r"[,;.:](?!\d)", window)[-1]
        before = re.findall(r"[a-z']+", window)[-_HEDGE_WINDOW:]
        if not any(token in _HEDGES for token in before):
            return True
    return False


def rsi_label_ok(text: str, case: Case) -> tuple[bool, list[str]]:
    regime = rsi_regime(case.rsi)
    notes = [
        f"says {label} at RSI {case.rsi}"
        for label in ("overbought", "oversold", "neutral")
        if label != regime and asserted(text, label)
    ]
    return not notes, notes


def vol_label_ok(text: str, case: Case) -> tuple[bool, list[str]]:
    expected = volatility_regime(case.vol)
    notes = []
    for match in _VOL_LABEL.finditer(text):
        label = "high" if match.group(1) == "elevated" else match.group(1)
        if label != expected and asserted(text, re.escape(match.group(0))):
            notes.append(f"calls {case.vol:.1%} volatility {label}, regime is {expected}")
    return not notes, notes


def _located(text: str) -> list[tuple[_Number, tuple[int, int]]]:
    """Every number with offsets into ``text`` and the span of its sentence."""
    out = []
    for span in _sentence_spans(text):
        for num in _numbers(text[span[0] : span[1]]):
            shifted = _Number(
                num.sign, num.value, num.is_percent, num.start + span[0], num.end + span[0]
            )
            out.append((shifted, span))
    return out


def _momentum_sentences(text: str, case: Case) -> list[str]:
    """Sentences about the 20-session return: they hold its value or say "momentum"."""
    holders = {span for num, span in _located(text) if _role(text, num, case, span) == "momentum"}
    out = []
    for start, end in _sentence_spans(text):
        sentence = text[start:end]
        holds_value = (start, end) in holders
        if holds_value or ("momentum" in sentence and "rsi" not in sentence):
            out.append(sentence)
    return out


def _direction_claims(text: str, case: Case, direction: str) -> list[str]:
    one_sided = _TWO_SIDED.sub("two-sided", text)
    claims = [m.group(0) for m in _TREND_TERMS[direction].finditer(one_sided)]
    for sentence in _momentum_sentences(text, case):
        sentence = _TWO_SIDED.sub("two-sided", sentence)
        claims += [m.group(0) for m in _DIRECTION_VERBS[direction].finditer(sentence)]
    return [c for c in claims if asserted(one_sided, re.escape(c))]


def _signed_momentum(text: str, case: Case) -> set[str]:
    located = _located(text)
    return {n.sign for n, span in located if n.sign and _role(text, n, case, span) == "momentum"}


def trend_ok(text: str, case: Case) -> tuple[bool, list[str]]:
    right, wrong = ("up", "down") if case.mom > 0 else ("down", "up")
    notes = []
    right_sign = "+" if right == "up" else "-"
    if not _direction_claims(text, case, right) and right_sign not in _signed_momentum(text, case):
        notes.append(f"no {right} direction stated")
    notes += [
        f"asserts {claim!r} with momentum {case.mom:+.2%}"
        for claim in dict.fromkeys(_direction_claims(text, case, wrong))
    ]
    return not notes, notes


def score(text: str | None, case: Case) -> Verdict:
    """Run every check; a missing generation fails all of them."""
    if not text:
        return Verdict(case.case_id, dict.fromkeys(CHECKS, False), ("no output",))
    lowered = text.lower()
    results = {
        "numbers": numbers_ok(lowered, case),
        "rsi_label": rsi_label_ok(lowered, case),
        "vol_label": vol_label_ok(lowered, case),
        "trend": trend_ok(lowered, case),
    }
    guard = guardrails.validate(text)
    checks = {name: ok for name, (ok, _) in results.items()} | {"guardrail": guard.ok}
    notes = [note for _, notes in results.values() for note in notes]
    notes += [f"guardrail {v}" for v in guard.violations]
    return Verdict(case.case_id, checks, tuple(notes))
