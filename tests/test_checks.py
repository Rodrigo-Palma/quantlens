"""The v0.7 faithfulness checker: the errors the v0.6 checker missed, and fair text."""

from __future__ import annotations

import pytest

from quantlens.evals import checks, checks_v06
from quantlens.evals.cases import Case

# oversold-down-low-00 from the grid: ABEV3, RSI 12.1, momentum -13.05%, vol 17.06%.
_ABEV3 = Case("oversold-down-low-00", "ABEV3", rsi=12.1, mom=-0.1305, vol=0.1706)
_CORRECT = (
    "ABEV3 is in a downtrend, with momentum of -13.05% over the last 20 sessions. "
    "The RSI(14) of 12 is oversold (below 30). Volatility is low at 17.06%, below 20%."
)

# The three errors from the v0.6 review; each passed every v0.6 check.
_REVIEW_ERRORS = {
    "sign inverted": (
        "ABEV3 rose 13.05% over 20 sessions, a strong rally. The RSI(14) of 12 is "
        "oversold, a balance of gains and losses. Volatility is low at 17.06%."
    ),
    "momentum and volatility swapped": (
        "ABEV3 climbed 17.06% over 20 sessions, with gains and losses. The RSI(14) is "
        "12, oversold, and it has low volatility of 13.05%."
    ),
    "invented constants": (
        "ABEV3 is in a downtrend with momentum of 30% and volatility of 40%. "
        "The RSI(14) of 12 is oversold."
    ),
}


def test_correct_text_passes() -> None:
    verdict = checks.score(_CORRECT, _ABEV3)
    assert verdict.passed, verdict.notes


@pytest.mark.parametrize("kind", sorted(_REVIEW_ERRORS))
def test_review_errors_fail_now_and_passed_before(kind: str) -> None:
    text = _REVIEW_ERRORS[kind]
    assert checks_v06.score(text, _ABEV3).passed
    assert not checks.score(text, _ABEV3).passed


def test_wrong_sign_is_named() -> None:
    text = _CORRECT.replace("-13.05%", "+13.05%")
    assert any("wrong sign" in note for note in checks.score(text, _ABEV3).notes)


def test_swapped_numbers_are_misattributed() -> None:
    text = "ABEV3 is in a downtrend, momentum of -17.06%. Annualized volatility is 13.05%."
    notes = checks.score(text, _ABEV3).notes
    assert any("written as volatility" in note for note in notes)


def test_threshold_constants_need_a_comparison() -> None:
    assert checks.numbers_ok("volatility of 17.06%, below the 20% threshold", _ABEV3)[0]
    assert checks.numbers_ok("rsi(14) of 12, in the 30\u201370 range", _ABEV3)[0]
    assert checks.numbers_ok("rsi between 30 and 70 is neutral; 0 to 100", _ABEV3)[0]
    assert not checks.numbers_ok("volatility of 20%", _ABEV3)[0]
    assert not checks.numbers_ok("an rsi of 70", _ABEV3)[0]


def test_window_constants_need_a_window_form() -> None:
    assert checks.numbers_ok("over the last 20 sessions, rsi(14), a 14-day window", _ABEV3)[0]
    assert not checks.numbers_ok("momentum of 14", _ABEV3)[0]


def test_direction_verb_in_momentum_sentence_is_a_wrong_claim() -> None:
    ok, notes = checks.trend_ok("abev3 rose 13.05% over the last 20 sessions.", _ABEV3)
    assert not ok
    assert "asserts 'rose'" in notes[1]


def test_direction_verb_outside_momentum_sentences_is_ignored() -> None:
    text = (
        "abev3 is in a downtrend, momentum -13.05%. the rsi of 12 says it rose "
        "less than it fell in recent days."
    )
    assert checks.trend_ok(text, _ABEV3)[0]


def test_falls_into_a_category_is_not_a_direction() -> None:
    case = Case("t", "ITUB4", rsi=40.0, mom=0.1298, vol=0.1312)
    text = "itub4 is in an uptrend. the annualized volatility of 13.12% falls into the low band."
    assert checks.trend_ok(text, case)[0]


def test_negative_volatility_is_a_wrong_sign() -> None:
    assert not checks.numbers_ok("volatility of -17.06%", _ABEV3)[0]


def test_neutral_label_outside_30_70_fails() -> None:
    ok, notes = checks.rsi_label_ok("the rsi of 12 is neutral.", _ABEV3)
    assert not ok
    assert notes == ["says neutral at RSI 12.1"]


def test_hedge_does_not_cross_a_comma() -> None:
    case = Case("t", "VALE3", rsi=47.0, mom=-0.06, vol=0.11)
    text = "the rsi of 47 is neither overbought nor oversold, remaining in overbought territory."
    assert not checks.rsi_label_ok(text, case)[0]


def test_price_fluctuations_label_is_checked() -> None:
    assert not checks.vol_label_ok("17.06% reflects moderate price fluctuations", _ABEV3)[0]
    assert checks.vol_label_ok("17.06% reflects low price fluctuations", _ABEV3)[0]


def test_missing_output_fails_everything() -> None:
    verdict = checks.score(None, _ABEV3)
    assert not any(verdict.checks.values())


def test_between_the_labels_is_a_hedge() -> None:
    case = Case("t", "ITUB4", rsi=49.9, mom=-0.0996, vol=0.3523)
    text = "the rsi(14) is at 50, a neutral position between overbought and oversold conditions."
    assert checks.rsi_label_ok(text, case)[0]


# RSI 75 for "RSI 75, below 70"; RSI exactly at the 70 threshold for the boundary.
_HOT = Case("t", "PETR4", rsi=75.0, mom=0.05, vol=0.1706)
_AT_70 = Case("t", "PETR4", rsi=70.0, mom=0.05, vol=0.1706)


@pytest.mark.parametrize(
    ("text", "case"),
    [
        ("the rsi(14) of 12 is oversold (below 30).", _ABEV3),
        ("volatility of 17.06%, below the 20% threshold.", _ABEV3),
        ("rsi of 75, above the overbought level of 70.", _HOT),
        ("rsi of 75 exceeds 70.", _HOT),
        ("rsi of 75, more than 70.", _HOT),
        ("rsi of 12, under 30.", _ABEV3),
        ("rsi of 12, not above 30.", _ABEV3),
        ("o rsi de 12 está abaixo de 30.", _ABEV3),
        ("volatility over the last 20 sessions is 17.06%.", _ABEV3),
        ("rsi of 70, at the 70 line.", _AT_70),
        ("rsi of 70, not above 70.", _AT_70),
        ("rsi of 70, not below 70.", _AT_70),
    ],
)
def test_true_threshold_comparison_passes(text: str, case: Case) -> None:
    ok, notes = checks.numbers_ok(text, case)
    assert ok, notes


@pytest.mark.parametrize(
    ("text", "case"),
    [
        ("rsi of 75, below 70.", _HOT),
        ("rsi of 12, above 30.", _ABEV3),
        ("the rsi(14) of 12 is oversold (above 30).", _ABEV3),
        ("volatility of 17.06%, above the 20% threshold.", _ABEV3),
        ("rsi of 12 exceeds 30.", _ABEV3),
        ("rsi of 75, less than 70.", _HOT),
        ("rsi of 12, not below 30.", _ABEV3),
        ("o rsi de 12 está acima de 30.", _ABEV3),
        ("rsi of 70, below 70.", _AT_70),
        ("rsi of 70, above 70.", _AT_70),
    ],
)
def test_inverted_threshold_comparison_fails(text: str, case: Case) -> None:
    ok, notes = checks.numbers_ok(text, case)
    assert not ok
    assert any("comparison" in note for note in notes), notes


def test_inverted_comparison_fails_the_whole_verdict() -> None:
    text = _CORRECT.replace("below 30", "above 30")
    verdict = checks.score(text, _ABEV3)
    assert not verdict.checks["numbers"]
    assert "says RSI 12.1 is above 30 (comparison)" in verdict.notes


@pytest.mark.parametrize(
    "phrase",
    ["both upward and downward movement", "downward or upward swings", "moves up and down"],
)
def test_two_sided_movement_is_not_a_direction_claim(phrase: str) -> None:
    """Seen in qwen3:8b (thresholds): "potential for both upward and downward movement"."""
    text = (
        "ABEV3 is in a downtrend, with momentum of -13.05% over 20 sessions. The RSI(14) "
        f"of 12 is oversold. Low volatility of 17.06% still allows {phrase}."
    )
    assert checks.score(text, _ABEV3).checks["trend"]


def test_one_sided_upward_is_still_a_wrong_claim() -> None:
    text = (
        "ABEV3 is in a downtrend, with momentum of -13.05%. The RSI(14) of 12 is "
        "oversold. Low volatility of 17.06% and upward momentum."
    )
    assert not checks.score(text, _ABEV3).checks["trend"]
