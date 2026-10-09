"""The checker audit: mutations are real errors and the report is honest about n."""

from __future__ import annotations

import pytest

from quantlens.evals import checker_audit, checks
from quantlens.evals.cases import Case

_CASE = Case("t", "ABEV3", rsi=12.1, mom=-0.1305, vol=0.1706)
_TEXT = (
    "ABEV3 is in a downtrend, with negative momentum of -13.05% over the last 20 "
    "sessions. The RSI(14) of 12 is oversold. It has low volatility of 17.06%."
)


def test_source_text_passes() -> None:
    assert checks.score(_TEXT, _CASE).passed


@pytest.mark.parametrize("kind", checker_audit.MUTATIONS)
def test_every_mutation_applies_and_is_caught(kind: str) -> None:
    mutated = checker_audit.mutate(kind, _TEXT, _CASE)
    assert mutated is not None
    assert mutated != _TEXT
    assert not checks.score(mutated, _CASE).passed


def test_mutations_change_what_they_say() -> None:
    assert "+13.05%" in str(checker_audit.mutate("sign_flip", _TEXT, _CASE))
    swapped = str(checker_audit.mutate("number_swap", _TEXT, _CASE))
    assert "-17.06%" in swapped and "13.05%." in swapped
    assert "26.06%" in str(checker_audit.mutate("invented_value", _TEXT, _CASE))
    assert "40%" in str(checker_audit.mutate("invented_constant", _TEXT, _CASE))
    assert "uptrend" in str(checker_audit.mutate("direction_flip", _TEXT, _CASE))
    assert "ABEV3 rose 13.05%" in str(checker_audit.mutate("wrong_verb", _TEXT, _CASE))
    assert "is overbought" in str(checker_audit.mutate("rsi_label_swap", _TEXT, _CASE))
    assert "high volatility" in str(checker_audit.mutate("vol_label_swap", _TEXT, _CASE))


def test_mutation_is_skipped_when_nothing_to_mutate() -> None:
    plain = "ABEV3 is in a downtrend. The RSI(14) of 12 is oversold."
    assert checker_audit.mutate("number_swap", plain, _CASE) is None
    assert checker_audit.mutate("vol_label_swap", plain, _CASE) is None


def test_ambiguous_figures_are_not_mutated() -> None:
    close = Case("t", "ABEV3", rsi=12.1, mom=-0.1305, vol=0.1310)
    assert checker_audit.mutate("sign_flip", "momentum -13.05%, vol 13.10%", close) is None


def test_neutral_rsi_label_becomes_overbought() -> None:
    case = Case("t", "ABEV3", rsi=50.0, mom=0.05, vol=0.30)
    mutated = checker_audit.mutate("rsi_label_swap", "The RSI of 50 is neutral.", case)
    assert mutated == "The RSI of 50 is overbought."


def test_run_reports_both_checkers_on_recorded_outputs() -> None:
    scores = checker_audit.run()
    assert {s.checker for s in scores} == set(checker_audit.CHECKERS)
    lines = checker_audit.report(scores)
    assert "v0.6 checker" in lines[1]
    measured = checker_audit.measured(scores)
    assert measured["checker_audit.sign_flip.n"] > 0
