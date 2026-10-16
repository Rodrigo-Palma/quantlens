"""Public market data access via yfinance. Network I/O lives only here."""

from __future__ import annotations

import pandas as pd
import yfinance as yf


def market_symbol(ticker: str) -> str:
    """The symbol whose price history stands for ``ticker`` (PETR4F -> PETR4).

    A fractional-market ticker is the same share as the standard lot, and Yahoo
    has no usable history for the F symbol (one row for PETR4F.SA).
    """
    return ticker[:-1] if ticker.endswith("F") else ticker


def fetch_close(ticker: str, period: str = "6mo", interval: str = "1d") -> pd.Series:
    """Fetch the adjusted-close series for a B3 ticker (e.g. ``PETR4``).

    Prices are adjusted for dividends, JCP and splits (``auto_adjust=True``), so
    returns are comparable across distribution dates; past values are not the
    prices that traded. A ``.SA`` suffix is appended automatically when missing.
    """
    symbol = ticker if ticker.endswith(".SA") else f"{ticker}.SA"
    frame = yf.download(symbol, period=period, interval=interval, progress=False, auto_adjust=True)
    if frame is None or frame.empty:
        raise ValueError(f"no data for {symbol}")
    close = frame["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    return close.rename("close")
