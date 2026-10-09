"""The cassette recorder, with Ollama mocked."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
import pytest

from quantlens.evals import cassette, record
from quantlens.evals.faithfulness import build_cases


def _fake_get(url: str, **kwargs: Any) -> httpx.Response:
    request = httpx.Request("GET", url)
    if url.endswith("/api/version"):
        return httpx.Response(200, json={"version": "9.9.9"}, request=request)
    models = [{"name": "m:1", "digest": "abc"}]
    return httpx.Response(200, json={"models": models}, request=request)


def test_record_writes_one_line_per_case_with_provenance(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(record.httpx, "get", _fake_get)
    monkeypatch.setattr(record.llm, "generate", lambda prompt, model: f"text for {model}")
    records = record.record("m:1")
    assert len(records) == len(build_cases())
    first = records[0]
    assert (first.model, first.model_digest, first.ollama_version) == ("m:1", "abc", "9.9.9")
    assert first.prompt_sha256 == cassette.prompt_hash(build_cases()[0].prompt())
    path = cassette.write("m:1", records, tmp_path)
    assert len(path.read_text(encoding="utf-8").splitlines()) == len(records)


def test_unknown_model_digest(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(record.httpx, "get", _fake_get)
    assert record._ollama_meta("other:1") == ("9.9.9", "unknown")


def test_main_refuses_non_ollama_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(record.settings, "llm_provider", "openai")
    monkeypatch.setattr("sys.argv", ["record"])
    with pytest.raises(SystemExit, match="LLM_PROVIDER=ollama"):
        record.main()


def test_main_records_each_model(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(record.settings, "llm_provider", "ollama")
    monkeypatch.setattr("sys.argv", ["record", "--model", "m:1"])
    monkeypatch.setattr(record, "record", lambda model: [])
    written: list[str] = []
    monkeypatch.setattr(record.cassette, "write", lambda model, recs: written.append(model))
    record.main()
    assert written == ["m:1"]


def test_hardware_is_a_non_empty_label() -> None:
    assert record.hardware()
