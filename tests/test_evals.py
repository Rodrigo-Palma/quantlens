"""The eval runner, its gates and the guardrail scorer."""

from __future__ import annotations

import pytest

from quantlens.evals import __main__ as runner
from quantlens.evals import gates, guardrail


def test_runner_passes_all_gates(capsys: pytest.CaptureFixture[str]) -> None:
    assert runner.main() == 0
    out = capsys.readouterr().out
    assert "All eval gates passed." in out
    assert "v0.5 deny-list" in out


def test_runner_reports_gate_failure(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(runner, "check", lambda measured: ["x: 0.1 < min 0.9"])
    assert runner.main() == 1
    assert "GATE FAILED" in capsys.readouterr().out


def test_gates_flag_min_max_and_missing() -> None:
    bounds = {"a": {"min": 0.5}, "b": {"max": 0.1}, "c": {"min": 0.0}}
    failures = gates.check({"a": 0.4, "b": 0.2}, bounds)
    assert failures == ["a: 0.4000 < min 0.5000", "b: 0.2000 > max 0.1000", "c: not measured"]
    assert gates.check({"a": 0.5, "b": 0.1, "c": 0.0}, bounds) == []


def test_recorded_gates_are_all_measured() -> None:
    from quantlens.evals import retrieval

    measured = {**retrieval.measured(), **guardrail.measured(guardrail.run())}
    assert set(gates.load_bounds()) <= set(measured)


def test_v05_baseline_is_the_old_substring_list() -> None:
    assert guardrail.v05_blocks("You should note this")
    assert not guardrail.v05_blocks("Compre agora")


def test_score_lists_misses_and_false_positives() -> None:
    examples = (
        guardrail.Example("buy now", "advice", "en"),
        guardrail.Example("hold tight", "advice", "en"),
        guardrail.Example("you should note", "clean", "en"),
    )
    result = guardrail.score("v0.5", guardrail.v05_blocks, "toy", examples)
    assert result.missed == ("hold tight",)
    assert result.wrongly_blocked == ("you should note",)
    assert result.recall.successes == 1
