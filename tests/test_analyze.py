"""/analyze end to end with the network mocked: market data and the LLM HTTP call."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Iterator

import httpx
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from quantlens import llm
from quantlens.api import main
from quantlens.config import settings

client = TestClient(main.app)

_STAGES = {"fetch", "signals", "retrieve", "llm", "guardrail"}


def _series(n: int = 120, seed: int = 7) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(100.0 * np.exp(np.cumsum(rng.normal(0.0005, 0.02, size=n))), name="close")


@pytest.fixture
def market(monkeypatch: pytest.MonkeyPatch) -> Callable[[pd.Series], None]:
    def _install(close: pd.Series) -> None:
        monkeypatch.setattr(main, "fetch_close", lambda ticker: close)

    _install(_series())
    return _install


@pytest.fixture
def ollama(
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[Callable[[httpx.Response | Exception], None]]:
    monkeypatch.setattr(settings, "llm_provider", "ollama")
    state: dict[str, httpx.Response | Exception] = {}

    def _post(url: str, **kwargs: object) -> httpx.Response:
        outcome = state["outcome"]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(llm.httpx, "post", _post)

    def _set(outcome: httpx.Response | Exception) -> None:
        state["outcome"] = outcome

    yield _set


def _reply(text: str) -> httpx.Response:
    request = httpx.Request("POST", "http://ollama/api/generate")
    return httpx.Response(200, json={"response": text}, request=request)


def test_llm_ok_is_served_as_llm(market: object, ollama: Callable[..., None]) -> None:
    ollama(_reply("<think>scratch</think>PETR4 shows a neutral RSI and moderate volatility."))
    body = client.get("/analyze", params={"ticker": "petr4"}).json()
    assert body["explanation_source"] == "llm"
    assert body["explanation"] == "PETR4 shows a neutral RSI and moderate volatility."
    assert body["guardrail_violations"] == []
    assert body["ticker"] == "PETR4"


def test_llm_advice_falls_back_and_lists_violation(
    market: object, ollama: Callable[..., None]
) -> None:
    ollama(_reply("The RSI is neutral, so you should buy PETR4 today."))
    body = client.get("/analyze", params={"ticker": "PETR4"}).json()
    assert body["explanation_source"] == "fallback"
    assert body["guardrail_violations"]
    assert "you should" not in body["explanation"].lower()
    assert body["explanation"].startswith("PETR4 is in")


def test_ollama_down_falls_back(market: object, ollama: Callable[..., None]) -> None:
    ollama(httpx.ConnectError("connection refused"))
    response = client.get("/analyze", params={"ticker": "PETR4"})
    assert response.status_code == 200
    body = response.json()
    assert body["explanation_source"] == "fallback"
    assert body["guardrail_violations"] == []


@pytest.mark.parametrize("points", [10, 15, 20])
def test_short_series_is_422_not_nan(
    market: Callable[[pd.Series], None], ollama: Callable[..., None], points: int
) -> None:
    market(_series(n=points))
    response = client.get("/analyze", params={"ticker": "PETR4"})
    assert response.status_code == 422
    assert "observations" in response.json()["detail"]


def test_unknown_ticker_is_404(monkeypatch: pytest.MonkeyPatch) -> None:
    def _missing(ticker: str) -> pd.Series:
        raise ValueError(f"no data for {ticker}.SA")

    monkeypatch.setattr(main, "fetch_close", _missing)
    response = client.get("/analyze", params={"ticker": "NOPE3"})
    assert response.status_code == 404


def test_request_emits_one_json_log_line_with_stage_latencies(
    market: object, ollama: Callable[..., None], caplog: pytest.LogCaptureFixture
) -> None:
    ollama(httpx.ConnectError("down"))
    with caplog.at_level(logging.INFO, logger="quantlens.request"):
        response = client.get("/analyze", params={"ticker": "PETR4"})
    records = [r for r in caplog.records if r.name == "quantlens.request"]
    assert len(records) == 1
    event = json.loads(records[0].getMessage())
    assert event["request_id"] == response.headers["x-request-id"]
    assert set(event["latency_ms"]) == _STAGES
    assert event["explanation_source"] == "fallback"
    assert event["fallback_reason"] == "llm_unavailable"
    assert event["status"] == 200


def test_guardrail_fallback_reason_is_logged(
    market: object, ollama: Callable[..., None], caplog: pytest.LogCaptureFixture
) -> None:
    ollama(_reply("This trade is guaranteed."))
    with caplog.at_level(logging.INFO, logger="quantlens.request"):
        client.get("/analyze", params={"ticker": "PETR4"})
    event = json.loads(caplog.records[-1].getMessage())
    assert event["fallback_reason"] == "guardrail_violation"
    assert event["guardrail_violations"]


def test_error_path_is_logged_with_status(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def _missing(ticker: str) -> pd.Series:
        raise ValueError("no data")

    monkeypatch.setattr(main, "fetch_close", _missing)
    with caplog.at_level(logging.INFO, logger="quantlens.request"):
        client.get("/analyze", params={"ticker": "NOPE3"})
    event = json.loads(caplog.records[-1].getMessage())
    assert event["status"] == 404
    assert "fetch" in event["latency_ms"]


def test_flat_series_has_undefined_rsi_and_is_422(
    market: Callable[[pd.Series], None], ollama: Callable[..., None]
) -> None:
    market(pd.Series([10.0] * 60, name="close"))
    response = client.get("/analyze", params={"ticker": "PETR4"})
    assert response.status_code == 422
    assert "undefined" in response.json()["detail"]
