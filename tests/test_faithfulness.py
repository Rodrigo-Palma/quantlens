"""The faithfulness checker: it must catch contradictions and spare correct text."""

from __future__ import annotations

from pathlib import Path

import pytest

from quantlens.evals import cassette, faithfulness
from quantlens.evals.faithfulness import Case, score
from quantlens.evals.stats import mcnemar_exact

_UP_OVERBOUGHT = Case("t", "PETR4", rsi=75.0, mom=0.08, vol=0.35)
_DOWN_NEUTRAL = Case("t", "VALE3", rsi=50.0, mom=-0.06, vol=0.45)


def test_case_grid_is_seeded_balanced_and_unambiguous() -> None:
    cases = faithfulness.build_cases()
    assert len(cases) == 120
    assert cases == faithfulness.build_cases()
    assert len({c.case_id for c in cases}) == 120
    assert all(not 28 < c.rsi < 35 and not 65 < c.rsi < 72 for c in cases)
    assert all(abs(c.mom) >= 0.02 for c in cases)


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
    assert "unsupported number 42" in verdict.notes


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
