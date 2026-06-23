"""Public market data access via yfinance. Network I/O lives only here."""

from __future__ import annotations

import pandas as pd
import yfinance as yf


def fetch_close(ticker: str, period: str = "6mo", interval: str = "1d") -> pd.Series:
    """Fetch the adjusted-close series for a B3 ticker (e.g. ``PETR4``).

    A ``.SA`` suffix is appended automatically when missing.
    """
    symbol = ticker if ticker.endswith(".SA") else f"{ticker}.SA"
    frame = yf.download(symbol, period=period, interval=interval, progress=False, auto_adjust=True)
    if frame is None or frame.empty:
        raise ValueError(f"no data for {symbol}")
    close = frame["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    return close.rename("close")
