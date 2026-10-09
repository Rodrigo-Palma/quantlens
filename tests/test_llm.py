"""LLM provider layer with the HTTP transport mocked (no network)."""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from quantlens import llm
from quantlens.config import settings


class _Recorder:
    def __init__(self, payload: Any, status: int = 200) -> None:
        self.payload = payload
        self.status = status
        self.calls: list[dict[str, Any]] = []

    def __call__(self, url: str, **kwargs: Any) -> httpx.Response:
        self.calls.append({"url": url, **kwargs})
        request = httpx.Request("POST", url)
        return httpx.Response(self.status, json=self.payload, request=request)


@pytest.fixture
def provider(monkeypatch: pytest.MonkeyPatch) -> Any:
    def _use(name: str, payload: Any, status: int = 200) -> _Recorder:
        recorder = _Recorder(payload, status)
        monkeypatch.setattr(settings, "llm_provider", name)
        monkeypatch.setattr(llm.httpx, "post", recorder)
        return recorder

    return _use


def test_ollama_strips_reasoning_and_sends_decoding_options(provider: Any) -> None:
    recorder = provider("ollama", {"response": "<think>x</think> Neutral RSI. "})
    assert llm.explain("PETR4", 50.0, 0.01, 0.2) == "Neutral RSI."
    body = recorder.calls[0]["json"]
    assert recorder.calls[0]["url"].endswith("/api/generate")
    assert body["think"] is False
    assert body["options"] == {"temperature": 0.0, "seed": 0}


def test_context_is_appended_to_prompt(provider: Any) -> None:
    recorder = provider("ollama", {"response": "ok"})
    llm.explain("PETR4", 50.0, 0.01, 0.2, context="## RSI\nbounded 0-100")
    assert "bounded 0-100" in recorder.calls[0]["json"]["prompt"]


def test_openai_compatible_provider_sends_bearer_key(
    provider: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "llm_api_key", "test-key")
    monkeypatch.setattr(settings, "llm_base_url", "https://api.example.com/")
    recorder = provider("openai", {"choices": [{"message": {"content": "Uptrend, RSI 60."}}]})
    assert llm.explain("VALE3", 60.0, 0.05, 0.3) == "Uptrend, RSI 60."
    call = recorder.calls[0]
    assert call["url"] == "https://api.example.com/v1/chat/completions"
    assert call["headers"] == {"Authorization": "Bearer test-key"}
    assert call["json"]["messages"][0]["role"] == "user"


def test_openai_provider_without_key_sends_no_auth_header(
    provider: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "llm_api_key", None)
    recorder = provider("openai", {"choices": [{"message": {"content": "ok"}}]})
    llm.explain("VALE3", 60.0, 0.05, 0.3)
    assert recorder.calls[0]["headers"] == {}


@pytest.mark.parametrize(
    ("name", "payload", "status"),
    [
        ("ollama", {"unexpected": "shape"}, 200),
        ("ollama", {"response": "ok"}, 500),
        ("openai", {"choices": []}, 200),
        ("ollama", {"response": "<think>only reasoning</think>   "}, 200),
        ("anthropic", {"response": "ok"}, 200),
    ],
    ids=["malformed", "http-500", "no-choices", "empty-after-think", "unknown-provider"],
)
def test_failures_return_none(provider: Any, name: str, payload: Any, status: int) -> None:
    provider(name, payload, status)
    assert llm.explain("PETR4", 50.0, 0.0, 0.2) is None


def test_model_override_is_sent(provider: Any) -> None:
    recorder = provider("ollama", {"response": "ok"})
    llm.generate("prompt", model="qwen3:8b")
    assert recorder.calls[0]["json"]["model"] == "qwen3:8b"
