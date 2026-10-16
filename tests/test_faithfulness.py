"""The faithfulness checker: it must catch contradictions and spare correct text."""

from __future__ import annotations

from pathlib import Path

import pytest

from quantlens import llm
from quantlens.evals import cassette, faithfulness
from quantlens.evals.faithfulness import Case, score
from quantlens.evals.stats import mcnemar_exact

_UP_OVERBOUGHT = Case("t", "PETR4", rsi=75.0, mom=0.08, vol=0.35)
_DOWN_NEUTRAL = Case("t", "VALE3", rsi=50.0, mom=-0.06, vol=0.45)


def test_case_grid_is_seeded_balanced_and_unambiguous() -> None:
    cases = faithfulness.build_cases()
    assert len(cases) == 180
    assert cases == faithfulness.build_cases()
    assert len({c.case_id for c in cases}) == 180
    assert all(not 28 < c.rsi < 35 and not 65 < c.rsi < 72 for c in cases)
    assert all(abs(c.mom) >= 0.02 for c in cases)
    assert all(not 18 < c.vol * 100 < 24 and not 36 < c.vol * 100 < 42 for c in cases)
    moderate = [c for c in cases if "-moderate-" in c.case_id]
    assert len(moderate) == 60
    assert all(0.20 < c.vol < 0.40 for c in moderate)


def test_v06_cases_keep_their_values() -> None:
    """The moderate cells come from their own stream: the first 120 cases are unchanged."""
    first = faithfulness.build_cases()[0]
    assert (first.case_id, first.ticker, first.rsi) == ("oversold-up-low-00", "VALE3", 17.1)


def test_rule_based_baseline_passes_every_check() -> None:
    scores = faithfulness.run()
    baseline = next(s for s in scores if s.system == faithfulness.BASELINE)
    assert baseline.rate().successes == baseline.n


def test_faithful_text_passes() -> None:
    text = (
        "PETR4 is in an uptrend, up 8% over 20 sessions. RSI(14) is 75, above 70, "
        "so it is overbought. Annualized volatility is 35.0%."
    )
    verdict = score(text, _UP_OVERBOUGHT)
    assert verdict.passed, verdict.notes


def test_invented_number_fails_numbers() -> None:
    verdict = score("PETR4 is in an uptrend; RSI 75; volatility 42%.", _UP_OVERBOUGHT)
    assert not verdict.checks["numbers"]
    assert "unsupported number 42%" in verdict.notes


def test_wrong_rsi_label_fails() -> None:
    verdict = score("VALE3 is oversold and in a downtrend.", _DOWN_NEUTRAL)
    assert not verdict.checks["rsi_label"]


def test_hedged_label_is_not_a_contradiction() -> None:
    text = "VALE3 is in a downtrend; RSI 50 is neither overbought nor oversold."
    assert score(text, _DOWN_NEUTRAL).checks["rsi_label"]


def test_wrong_trend_fails() -> None:
    verdict = score("VALE3 shows positive momentum and is in an uptrend.", _DOWN_NEUTRAL)
    assert not verdict.checks["trend"]
    assert any("asserts" in note for note in verdict.notes)


def test_missing_direction_fails_trend() -> None:
    assert not score("VALE3 has an RSI of 50.", _DOWN_NEUTRAL).checks["trend"]


def test_advice_fails_guardrail() -> None:
    verdict = score("PETR4 is in an uptrend; you should buy it.", _UP_OVERBOUGHT)
    assert not verdict.checks["guardrail"]


def test_missing_output_fails_everything() -> None:
    verdict = score(None, _UP_OVERBOUGHT)
    assert not any(verdict.checks.values())
    assert verdict.notes == ("no output",)


def test_mcnemar_exact_values() -> None:
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(10, 0) == pytest.approx(2 / 1024)
    assert mcnemar_exact(5, 5) == 1.0
    with pytest.raises(ValueError):
        mcnemar_exact(-1, 2)


def test_cassette_round_trip(tmp_path: Path) -> None:
    record = cassette.Record(
        case_id="c1",
        model="m:1",
        text="hello",
        latency_ms=1.5,
        prompt_sha256=cassette.prompt_hash("p"),
        model_digest="d",
        ollama_version="0",
        hardware="hw",
        recorded_at="2026-10-09T00:00:00+00:00",
    )
    path = cassette.write("m:1", [record], tmp_path)
    assert path.name == "m-1.jsonl"
    assert path.read_text(encoding="utf-8").count("\n") == 1


def test_load_all_without_cassette_dir_is_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cassette, "CASSETTE_DIR", "data/does-not-exist")
    assert cassette.load_all() == {}


def test_wrong_volatility_label_fails() -> None:
    case = Case("t", "PETR4", rsi=50.0, mom=0.05, vol=0.17)
    verdict = score(
        "PETR4 is in an uptrend; volatility of 17% reflects moderate price swings.", case
    )
    assert not verdict.checks["vol_label"]
    assert score("PETR4 is in an uptrend with low volatility.", case).checks["vol_label"]


def test_elevated_counts_as_high() -> None:
    case = Case("t", "PETR4", rsi=50.0, mom=0.05, vol=0.55)
    assert score("PETR4 is in an uptrend with elevated volatility.", case).checks["vol_label"]


def test_coordinated_negation_is_a_hedge() -> None:
    text = "VALE3 is in a downtrend, without signs of overbought or oversold conditions."
    assert score(text, _DOWN_NEUTRAL).checks["rsi_label"]


def test_case_prompt_variants_differ_only_in_the_reference_block() -> None:
    case = faithfulness.build_cases()[0]
    bare = case.prompt(cassette.WITHOUT_RETRIEVAL)
    thresholds = case.prompt(cassette.THRESHOLDS)
    rag = case.prompt(cassette.WITH_RETRIEVAL)
    assert case.prompt() == rag
    assert thresholds == bare + thresholds.removeprefix(bare)
    assert thresholds.endswith(llm.THRESHOLDS_REFERENCE + "\n")
    assert "## " not in thresholds
    assert "## " in rag
    with pytest.raises(ValueError, match="unknown prompt variant"):
        case.prompt("few-shot")


def test_slug_and_system_name_of_the_thresholds_variant() -> None:
    assert cassette.slug("qwen3:32b", cassette.THRESHOLDS) == "qwen3-32b.thresholds"
    record = cassette.Record("c", "m:1", "t", 1.0, "h", "d", "0", "hw", "x", cassette.THRESHOLDS)
    assert record.system == "m:1 (thresholds)"


def _system(name: str, passed: list[bool]) -> faithfulness.SystemScore:
    checks = dict.fromkeys(faithfulness.CHECKS, True)
    verdicts = tuple(
        faithfulness.Verdict(f"c{i}", {**checks, "numbers": ok}, ()) for i, ok in enumerate(passed)
    )
    return faithfulness.SystemScore(name, verdicts, 0)


def test_report_pairs_thresholds_with_rag_and_no_rag_for_every_model() -> None:
    scores = [
        _system(faithfulness.BASELINE, [True] * 4),
        *[
            _system(f"{model}{suffix}", flags)
            for model in ("qwen3:32b", "qwen3:8b")
            for suffix, flags in (
                ("", [True, True, True, True]),
                (" (no-rag)", [False, False, True, True]),
                (" (thresholds)", [True, False, True, True]),
            )
        ],
    ]
    lines = "\n".join(faithfulness.report(scores))
    for model in ("qwen3:32b", "qwen3:8b"):
        assert (
            f"paired {model} (thresholds) vs {model}: only the first passes 0, "
            "only the second passes 1, exact McNemar p = 1"
        ) in lines
        assert (
            f"paired {model} (thresholds) vs {model} (no-rag): only the first passes 1, "
            "only the second passes 0, exact McNemar p = 1"
        ) in lines
