"""Market data adapter with yfinance mocked (no network)."""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from quantlens.data import market


def _patch_download(monkeypatch: pytest.MonkeyPatch, frame: pd.DataFrame | None) -> list[str]:
    seen: list[str] = []

    def _download(symbol: str, **kwargs: Any) -> pd.DataFrame | None:
        seen.append(symbol)
        return frame

    monkeypatch.setattr(market.yf, "download", _download)
    return seen


def test_appends_sa_suffix_and_returns_close(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _patch_download(monkeypatch, pd.DataFrame({"Close": [10.0, 11.0]}))
    close = market.fetch_close("PETR4")
    assert seen == ["PETR4.SA"]
    assert close.name == "close"
    assert close.tolist() == [10.0, 11.0]


def test_keeps_existing_suffix(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _patch_download(monkeypatch, pd.DataFrame({"Close": [1.0]}))
    market.fetch_close("VALE3.SA")
    assert seen == ["VALE3.SA"]


def test_multiindex_columns_are_flattened(monkeypatch: pytest.MonkeyPatch) -> None:
    columns = pd.MultiIndex.from_tuples([("Close", "PETR4.SA")])
    _patch_download(monkeypatch, pd.DataFrame([[5.0], [6.0]], columns=columns))
    assert market.fetch_close("PETR4").tolist() == [5.0, 6.0]


@pytest.mark.parametrize("frame", [None, pd.DataFrame()], ids=["none", "empty"])
def test_no_data_raises_value_error(
    monkeypatch: pytest.MonkeyPatch, frame: pd.DataFrame | None
) -> None:
    _patch_download(monkeypatch, frame)
    with pytest.raises(ValueError, match="no data"):
        market.fetch_close("NOPE3")
