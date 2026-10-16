"""The checker audit: mutations are real errors and the report is honest about n."""

from __future__ import annotations

import pytest

from quantlens.evals import checker_audit, checks
from quantlens.evals.cases import Case

_CASE = Case("t", "ABEV3", rsi=12.1, mom=-0.1305, vol=0.1706)
_TEXT = (
    "ABEV3 is in a downtrend, with negative momentum of -13.05% over the last 20 "
    "sessions. The RSI(14) of 12 is oversold. It has low volatility of 17.06%, "
    "below 20%."
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
    assert "-17.06%" in swapped and "of 13.05%," in swapped
    assert "26.06%" in str(checker_audit.mutate("invented_value", _TEXT, _CASE))
    assert "40%" in str(checker_audit.mutate("invented_constant", _TEXT, _CASE))
    assert "uptrend" in str(checker_audit.mutate("direction_flip", _TEXT, _CASE))
    assert "ABEV3 rose 13.05%" in str(checker_audit.mutate("wrong_verb", _TEXT, _CASE))
    assert "is overbought" in str(checker_audit.mutate("rsi_label_swap", _TEXT, _CASE))
    assert "high volatility" in str(checker_audit.mutate("vol_label_swap", _TEXT, _CASE))
    assert "17.06%, above 20%" in str(checker_audit.mutate("comparison_flip", _TEXT, _CASE))


@pytest.mark.parametrize(
    ("text", "flipped"),
    [
        ("The RSI of 12 is below 30.", "The RSI of 12 is above 30."),
        ("RSI 12, Below the oversold level of 30.", "RSI 12, Above the oversold level of 30."),
        ("O RSI de 12 est\u00e1 abaixo de 30.", "O RSI de 12 est\u00e1 acima de 30."),
        ("Volatility is 50.78%, above 50%.", "Volatility is 50.78%, below 50%."),
    ],
)
def test_comparison_flip_inverts_threshold_comparisons(text: str, flipped: str) -> None:
    assert checker_audit.mutate("comparison_flip", text, _CASE) == flipped


@pytest.mark.parametrize(
    "text",
    [
        "Prices sit above their recent average.",
        "Volatility below its 20-day mean is 17.06%.",
        "The RSI of 12 is within the 30-70 range.",
    ],
)
def test_comparison_flip_skips_comparisons_without_a_threshold(text: str) -> None:
    assert checker_audit.mutate("comparison_flip", text, _CASE) is None


def test_mutation_is_skipped_when_nothing_to_mutate() -> None:
    plain = "ABEV3 is in a downtrend. The RSI(14) of 12 is oversold."
    assert checker_audit.mutate("number_swap", plain, _CASE) is None
    assert checker_audit.mutate("vol_label_swap", plain, _CASE) is None
    assert checker_audit.mutate("comparison_flip", plain, _CASE) is None


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


def test_mutation_pool_ignores_thresholds_cassettes(monkeypatch: pytest.MonkeyPatch) -> None:
    """The audit pool stays the four rag/no-rag cassettes the checker was built on."""
    from quantlens.evals import cassette
    from quantlens.evals.cases import build_cases

    case = build_cases()[0]
    text = (
        f"{case.ticker} is in an uptrend, with momentum of {case.mom:.2%}. "
        f"Its RSI of {case.rsi:.0f} is oversold, and volatility of {case.vol:.2%} is low."
    )
    assert checks.score(text, case).passed

    def _rec(variant: str, body: str) -> cassette.Record:
        return cassette.Record(case.case_id, "m:1", body, 1.0, "h", "d", "0", "hw", "x", variant)

    monkeypatch.setattr(
        cassette,
        "load_all",
        lambda: {
            "m:1": [_rec(cassette.WITH_RETRIEVAL, text)],
            "m:1 (thresholds)": [_rec(cassette.THRESHOLDS, text + " Calm.")],
        },
    )
    assert checker_audit.sources() == [(text, case)]


def test_numeric_heldout_batch_is_labeled_by_what_it_contains() -> None:
    """Asked for a number, sign or direction error, the planting model only flipped
    the trend; the report must not claim out-of-sample number or sign coverage."""
    batch = checker_audit.load_heldout()["numeric"]
    failed = {
        name
        for text, case in batch
        for name, ok in checks.score(text, case).checks.items()
        if not ok
    }
    assert failed == {"trend"}
    assert checker_audit.HELDOUT_KINDS["numeric"] == "planted, trend direction"
