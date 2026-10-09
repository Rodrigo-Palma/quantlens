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
    monkeypatch.setattr(record, "record", lambda model, with_context=True, keep=None: [])
    written: list[str] = []
    monkeypatch.setattr(
        record.cassette, "write", lambda model, recs, variant: written.append(model)
    )
    record.main()
    assert written == ["m:1"]


def test_hardware_is_a_non_empty_label() -> None:
    assert record.hardware()


def _record(case_id: str, digest: str = "abc", text: str | None = "old") -> cassette.Record:
    case = next(c for c in build_cases() if c.case_id == case_id)
    return cassette.Record(
        case_id=case_id,
        model="m:1",
        text=text,
        latency_ms=1.0,
        prompt_sha256=cassette.prompt_hash(case.prompt()),
        model_digest=digest,
        ollama_version="9.9.9",
        hardware="hw",
        recorded_at="2026-10-09T00:00:00+00:00",
    )


def test_missing_only_keeps_valid_records_and_fills_the_rest(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    first, second = build_cases()[0].case_id, build_cases()[1].case_id
    stale = cassette.Record(**{**_record(second).__dict__, "prompt_sha256": "stale"})
    path = cassette.write("m:1", [_record(first), stale], tmp_path)
    keep = record.reusable(path, with_context=True, digest="abc")
    assert set(keep) == {first}
    monkeypatch.setattr(record.httpx, "get", _fake_get)
    monkeypatch.setattr(record.llm, "generate", lambda prompt, model: "new")
    records = record.record("m:1", keep=keep)
    assert len(records) == len(build_cases())
    assert records[0].text == "old"
    assert all(r.text == "new" for r in records[1:])


def test_missing_only_refuses_another_model_digest(tmp_path: Path) -> None:
    path = cassette.write("m:1", [_record(build_cases()[0].case_id, digest="old")], tmp_path)
    with pytest.raises(SystemExit, match="recorded with"):
        record.reusable(path, with_context=True, digest="abc")


def test_missing_only_without_a_cassette_records_everything(tmp_path: Path) -> None:
    assert record.reusable(tmp_path / "none.jsonl", with_context=True, digest="abc") == {}


def test_main_missing_only_reads_the_existing_cassette(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(record.settings, "llm_provider", "ollama")
    monkeypatch.setattr("sys.argv", ["record", "--model", "m:1", "--missing-only"])
    monkeypatch.setattr(record.httpx, "get", _fake_get)
    seen: dict[str, object] = {}
    monkeypatch.setattr(record, "reusable", lambda path, ctx, digest: {"x": digest})
    monkeypatch.setattr(
        record, "record", lambda model, with_context=True, keep=None: seen.update(keep=keep) or []
    )
    monkeypatch.setattr(record.cassette, "write", lambda model, recs, variant: None)
    record.main()
    assert seen["keep"] == {"x": "abc"}
