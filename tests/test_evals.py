"""The eval runner, its gates and the guardrail scorer."""

from __future__ import annotations

import pytest

from quantlens.evals import __main__ as runner
from quantlens.evals import gates, guardrail


def test_runner_passes_all_gates(capsys: pytest.CaptureFixture[str]) -> None:
    assert runner.main([]) == 0
    out = capsys.readouterr().out
    assert "All eval gates passed." in out
    assert "v0.5 deny-list" in out


def test_runner_reports_gate_failure(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(runner.gates, "check", lambda measured: ["x: 1 < recorded 2"])
    assert runner.main([]) == 1
    assert "GATE FAILED" in capsys.readouterr().out


def test_gates_fail_on_any_worsening_of_a_count() -> None:
    bounds = {"a": {"min": 87}, "b": {"max": 2}, "n": {"eq": 106}, "c": {"min": 0}}
    failures = gates.check({"a": 86, "b": 3, "n": 105}, bounds)
    assert failures == [
        "a: 86 < recorded 87",
        "b: 3 > recorded 2",
        "n: 105 != recorded 106",
        "c: not measured",
    ]
    assert gates.check({"a": 87, "b": 2, "n": 106, "c": 0}, bounds) == []


def test_improvement_passes_and_is_reported() -> None:
    bounds = {"a": {"min": 87}, "b": {"max": 2}, "n": {"eq": 106}}
    measured = {"a": 90, "b": 1, "n": 106}
    assert gates.check(measured, bounds) == []
    assert gates.improvements(measured, bounds) == ["a: 90 (recorded 87)", "b: 1 (recorded 2)"]


def test_update_rewrites_counts_and_keeps_directions(tmp_path: object) -> None:
    from pathlib import Path

    bounds = {"a": {"min": 87}, "n": {"eq": 106}}
    new = gates.updated({"a": 90, "n": 110, "extra": 1}, bounds)
    assert new == {"a": {"min": 90}, "n": {"eq": 110}}
    path = gates.write(new, Path(str(tmp_path)) / "gates.json")
    assert path.read_text(encoding="utf-8").startswith('{\n  "a": {"min": 90}')
    with pytest.raises(KeyError):
        gates.updated({}, bounds)


def test_runner_update_gates_writes_the_file(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    written = {}
    monkeypatch.setattr(runner.gates, "write", lambda bounds: written.update(bounds) or "x")
    assert runner.main(["--update-gates"]) == 0
    assert set(written) == set(gates.load_bounds())
    assert "Gates rewritten" in capsys.readouterr().out


def test_recorded_gates_are_all_measured() -> None:
    _, measured = runner.measure()
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
