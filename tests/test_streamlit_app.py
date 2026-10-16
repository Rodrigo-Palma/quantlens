"""The Streamlit demo runs the API's guards, exercised with a fake ``streamlit``."""

from __future__ import annotations

import runpy
import sys
import types
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from quantlens.data import market

APP = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"


class _Column:
    def __init__(self, calls: dict[str, list[Any]]) -> None:
        self._calls = calls

    def metric(self, label: str, value: str) -> None:
        self._calls["metric"].append((label, value))


def _fake_streamlit(ticker: str) -> tuple[types.ModuleType, dict[str, list[Any]]]:
    calls: dict[str, list[Any]] = {"error": [], "info": [], "metric": []}
    st = types.ModuleType("streamlit")

    def _record(name: str) -> Callable[..., None]:
        return lambda *args, **kwargs: calls.setdefault(name, []).append(args)

    st.set_page_config = _record("set_page_config")  # type: ignore[attr-defined]
    st.title = _record("title")  # type: ignore[attr-defined]
    st.caption = _record("caption")  # type: ignore[attr-defined]
    st.line_chart = _record("line_chart")  # type: ignore[attr-defined]
    st.error = lambda msg: calls["error"].append(msg)  # type: ignore[attr-defined]
    st.info = lambda msg: calls["info"].append(msg)  # type: ignore[attr-defined]
    st.text_input = lambda *a, **k: ticker  # type: ignore[attr-defined]
    st.button = lambda *a, **k: True  # type: ignore[attr-defined]
    st.columns = lambda n: [_Column(calls) for _ in range(n)]  # type: ignore[attr-defined]
    return st, calls


def _run(monkeypatch: pytest.MonkeyPatch, ticker: str, close: pd.Series) -> tuple[dict, list[str]]:
    st, calls = _fake_streamlit(ticker)
    fetched: list[str] = []

    def _fetch(symbol: str, **kwargs: Any) -> pd.Series:
        fetched.append(symbol)
        return close

    monkeypatch.setitem(sys.modules, "streamlit", st)
    monkeypatch.setattr(market, "fetch_close", _fetch)
    runpy.run_path(str(APP), run_name="__main__")
    return calls, fetched


def _series(n: int) -> pd.Series:
    rng = np.random.default_rng(7)
    return pd.Series(100.0 * np.exp(np.cumsum(rng.normal(0.0005, 0.02, size=n))), name="close")


def test_fractional_ticker_fetches_standard_lot(monkeypatch: pytest.MonkeyPatch) -> None:
    calls, fetched = _run(monkeypatch, "petr4f", _series(120))
    assert fetched == ["PETR4"]
    assert calls["error"] == []
    assert len(calls["info"]) == 1
    assert "nan" not in calls["info"][0].lower()


@pytest.mark.parametrize("close", [_series(1), pd.Series([10.0] * 60, name="close")])
def test_unusable_series_shows_error_not_nan(
    monkeypatch: pytest.MonkeyPatch, close: pd.Series
) -> None:
    calls, _ = _run(monkeypatch, "PETR4", close)
    assert len(calls["error"]) == 1
    assert calls["info"] == []
    assert calls["metric"] == []
