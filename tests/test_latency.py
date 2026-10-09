"""LLM latency measurement with the model call mocked."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from quantlens.evals import latency


def test_measure_warms_up_then_times_n_prompts(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(latency, "_ollama_meta", lambda model: ("0.1", "digest"))
    result = latency.measure("m:1", 5, generate=lambda prompt, model: calls.append(model) or "x")
    assert len(calls) == 6
    assert result["n"] == 5
    assert result["model_digest"] == "digest"
    assert float(str(result["p50_s"])) <= float(str(result["p95_s"])) <= float(str(result["max_s"]))


def test_nearest_rank() -> None:
    assert latency.nearest_rank([3.0, 1.0, 2.0, 4.0], 0.95) == 4.0
    assert latency.nearest_rank([1.0], 0.5) == 1.0


def test_load_missing_and_present(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    target = tmp_path / "latency.json"
    monkeypatch.setattr(latency, "LATENCY_FILE", target)
    assert latency.load() == {}
    target.write_text(json.dumps({"m:1": {"n": 3}}), encoding="utf-8")
    assert latency.load() == {"m:1": {"n": 3}}
