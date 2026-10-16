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
from quantlens.ratelimit import RateLimiter

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
    assert "last_price" not in body
    assert body["last_adjusted_close"] > 0


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


@pytest.mark.parametrize("bad", ["", "PETR", "PETR123", "PETR4.SA", "../etc", "A" * 200, "1234"])
def test_rejects_malformed_ticker_with_422(monkeypatch: pytest.MonkeyPatch, bad: str) -> None:
    def _never(ticker: str) -> pd.Series:
        raise AssertionError("fetch must not run for a malformed ticker")

    monkeypatch.setattr(main, "fetch_close", _never)
    response = client.get("/analyze", params={"ticker": bad})
    assert response.status_code == 422


@pytest.mark.parametrize("good", ["PETR4", "taee11", "PETR4F"])
def test_accepts_b3_ticker_forms(market: object, ollama: Callable[..., None], good: str) -> None:
    ollama(_reply("Neutral RSI, an uptrend and moderate volatility."))
    assert client.get("/analyze", params={"ticker": good}).status_code == 200


@pytest.mark.parametrize(("requested", "fetched"), [("PETR4F", "PETR4"), ("taee11f", "TAEE11")])
def test_fractional_ticker_is_analyzed_with_the_standard_lot_series(
    monkeypatch: pytest.MonkeyPatch, ollama: Callable[..., None], requested: str, fetched: str
) -> None:
    # Yahoo has no usable history for the fractional symbol (PETR4F.SA returns a
    # single row); the fractional market trades the same share as the lot.
    seen: list[str] = []

    def _fetch(ticker: str) -> pd.Series:
        seen.append(ticker)
        return _series()

    monkeypatch.setattr(main, "fetch_close", _fetch)
    ollama(_reply("Neutral RSI, an uptrend and moderate volatility."))
    response = client.get("/analyze", params={"ticker": requested})
    assert response.status_code == 200
    assert seen == [fetched]
    assert response.json()["ticker"] == requested.upper()


def test_repeated_request_reuses_series_and_explanation(
    monkeypatch: pytest.MonkeyPatch, ollama: Callable[..., None]
) -> None:
    calls = {"fetch": 0, "llm": 0}
    close = _series()

    def _fetch(ticker: str) -> pd.Series:
        calls["fetch"] += 1
        return close

    def _explain(*args: object, **kwargs: object) -> str:
        calls["llm"] += 1
        return "Neutral RSI, an uptrend and moderate volatility."

    monkeypatch.setattr(main, "fetch_close", _fetch)
    monkeypatch.setattr(main.llm, "explain", _explain)
    first = client.get("/analyze", params={"ticker": "PETR4"}).json()
    second = client.get("/analyze", params={"ticker": "petr4"}).json()
    assert first == second
    assert calls == {"fetch": 1, "llm": 1}


def test_unavailable_llm_is_not_cached(monkeypatch: pytest.MonkeyPatch, market: object) -> None:
    replies = iter([None, "Neutral RSI, an uptrend and moderate volatility."])
    monkeypatch.setattr(main.llm, "explain", lambda *a, **k: next(replies))
    first = client.get("/analyze", params={"ticker": "PETR4"}).json()
    second = client.get("/analyze", params={"ticker": "PETR4"}).json()
    assert first["explanation_source"] == "fallback"
    assert second["explanation_source"] == "llm"


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def limiter(monkeypatch: pytest.MonkeyPatch) -> Callable[[int], _Clock]:
    """Swap in a limiter with a small burst and a frozen clock."""

    def _install(burst: int) -> _Clock:
        clock = _Clock()
        limiter = RateLimiter(per_min=60, burst=burst, max_clients=100, clock=clock)
        monkeypatch.setattr(main, "_limiter", limiter)
        return clock

    return _install


def test_requests_past_the_burst_get_429_with_retry_after(
    market: object,
    ollama: Callable[..., None],
    limiter: Callable[[int], _Clock],
    caplog: pytest.LogCaptureFixture,
) -> None:
    ollama(httpx.ConnectError("down"))
    limiter(2)
    statuses = [client.get("/analyze", params={"ticker": "PETR4"}).status_code for _ in range(2)]
    with caplog.at_level(logging.INFO, logger="quantlens.request"):
        refused = client.get("/analyze", params={"ticker": "PETR4"})
    assert statuses == [200, 200]
    assert refused.status_code == 429
    assert refused.headers["retry-after"] == "1"
    event = json.loads(caplog.records[-1].getMessage())
    assert event["status"] == 429
    assert event["request_id"] == refused.headers["x-request-id"]
    assert event["latency_ms"] == {}


def test_retry_after_is_a_whole_number_of_seconds(
    market: object, ollama: Callable[..., None], limiter: Callable[[int], _Clock]
) -> None:
    ollama(httpx.ConnectError("down"))
    clock = limiter(1)
    client.get("/analyze", params={"ticker": "PETR4"})
    clock.now = 0.25
    assert client.get("/analyze", params={"ticker": "PETR4"}).headers["retry-after"] == "1"


def test_bucket_refills_over_time(
    market: object, ollama: Callable[..., None], limiter: Callable[[int], _Clock]
) -> None:
    ollama(httpx.ConnectError("down"))
    clock = limiter(1)
    assert client.get("/analyze", params={"ticker": "PETR4"}).status_code == 200
    assert client.get("/analyze", params={"ticker": "PETR4"}).status_code == 429
    clock.now = 1.0
    assert client.get("/analyze", params={"ticker": "PETR4"}).status_code == 200


def test_unknown_ticker_consumes_the_bucket(
    monkeypatch: pytest.MonkeyPatch, limiter: Callable[[int], _Clock]
) -> None:
    calls = {"fetch": 0}

    def _missing(ticker: str) -> pd.Series:
        calls["fetch"] += 1
        raise ValueError("no data")

    monkeypatch.setattr(main, "fetch_close", _missing)
    limiter(1)
    assert client.get("/analyze", params={"ticker": "NOPE3"}).status_code == 404
    assert client.get("/analyze", params={"ticker": "NOPE4"}).status_code == 429
    assert calls == {"fetch": 1}


def test_health_is_exempt_from_the_rate_limit(
    market: object, ollama: Callable[..., None], limiter: Callable[[int], _Clock]
) -> None:
    ollama(httpx.ConnectError("down"))
    limiter(1)
    client.get("/analyze", params={"ticker": "PETR4"})
    assert client.get("/analyze", params={"ticker": "PETR4"}).status_code == 429
    assert all(client.get("/health").status_code == 200 for _ in range(5))


def test_forwarded_for_shares_one_bucket_without_trusted_proxies(
    market: object, ollama: Callable[..., None], limiter: Callable[[int], _Clock]
) -> None:
    ollama(httpx.ConnectError("down"))
    limiter(1)
    proxied = TestClient(main.app, client=("10.0.0.1", 50000))
    first = proxied.get("/analyze", params={"ticker": "PETR4"}, headers=_xff("203.0.113.1"))
    second = proxied.get("/analyze", params={"ticker": "PETR4"}, headers=_xff("203.0.113.2"))
    assert (first.status_code, second.status_code) == (200, 429)


def test_forwarded_for_is_honoured_from_a_trusted_proxy(
    monkeypatch: pytest.MonkeyPatch,
    market: object,
    ollama: Callable[..., None],
    limiter: Callable[[int], _Clock],
) -> None:
    ollama(httpx.ConnectError("down"))
    monkeypatch.setattr(settings, "trusted_proxies", "10.0.0.1")
    limiter(1)
    proxied = TestClient(main.app, client=("10.0.0.1", 50000))
    first = proxied.get("/analyze", params={"ticker": "PETR4"}, headers=_xff("203.0.113.1"))
    second = proxied.get("/analyze", params={"ticker": "PETR4"}, headers=_xff("203.0.113.2"))
    again = proxied.get("/analyze", params={"ticker": "PETR4"}, headers=_xff("203.0.113.1"))
    assert (first.status_code, second.status_code, again.status_code) == (200, 200, 429)


def _xff(address: str) -> dict[str, str]:
    return {"X-Forwarded-For": address}


@pytest.fixture
def saturated_llm() -> Iterator[None]:
    """Hold every LLM slot, as if the model were busy with other requests."""
    held = 0
    while main._llm_slots.acquire(blocking=False):
        held += 1
    assert held == settings.llm_max_concurrency
    try:
        yield
    finally:
        for _ in range(held):
            main._llm_slots.release()


def test_saturated_llm_falls_back_without_calling_it(
    monkeypatch: pytest.MonkeyPatch,
    market: object,
    saturated_llm: None,
    caplog: pytest.LogCaptureFixture,
) -> None:
    def _never(*args: object, **kwargs: object) -> str:
        raise AssertionError("the LLM must not be called while saturated")

    monkeypatch.setattr(main.llm, "explain", _never)
    with caplog.at_level(logging.INFO, logger="quantlens.request"):
        response = client.get("/analyze", params={"ticker": "PETR4"})
    assert response.status_code == 200
    assert response.json()["explanation_source"] == "fallback"
    event = json.loads(caplog.records[-1].getMessage())
    assert event["fallback_reason"] == "llm_busy"


def test_cached_explanation_is_served_while_llm_is_saturated(
    monkeypatch: pytest.MonkeyPatch, market: object
) -> None:
    replies = iter(["Neutral RSI, an uptrend and moderate volatility."])
    monkeypatch.setattr(main.llm, "explain", lambda *a, **k: next(replies))
    client.get("/analyze", params={"ticker": "PETR4"})
    held = [main._llm_slots.acquire(blocking=False) for _ in range(settings.llm_max_concurrency)]
    try:
        body = client.get("/analyze", params={"ticker": "PETR4"}).json()
    finally:
        for _ in range(sum(held)):
            main._llm_slots.release()
    assert all(held)
    assert body["explanation_source"] == "llm"


def test_llm_slot_is_released_when_the_call_raises(
    monkeypatch: pytest.MonkeyPatch, market: object
) -> None:
    def _boom(*args: object, **kwargs: object) -> str:
        raise RuntimeError("provider bug")

    monkeypatch.setattr(main.llm, "explain", _boom)
    with pytest.raises(RuntimeError):
        client.get("/analyze", params={"ticker": "PETR4"})
    held = [main._llm_slots.acquire(blocking=False) for _ in range(settings.llm_max_concurrency)]
    for _ in range(sum(held)):
        main._llm_slots.release()
    assert all(held)
