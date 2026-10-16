"""Streamlit demo for QuantLens.

Optional extra::

    uv sync --extra demo
    uv run streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import streamlit as st

from quantlens.data.market import fetch_close, market_symbol
from quantlens.explain import rule_based
from quantlens.quant import snapshot

st.set_page_config(page_title="QuantLens")
st.title("QuantLens: B3 quant analyst")
st.caption("Educational engineering demo. Not investment advice.")

ticker = st.text_input("B3 ticker", value="PETR4").strip().upper()

if st.button("Analyze") and ticker:
    try:
        # Same guards as the API: a fractional ticker uses the standard-lot series,
        # and a series the signals are undefined for is an error, not NaN.
        close = fetch_close(market_symbol(ticker))
        snap = snapshot.compute(close)
    except ValueError as exc:
        st.error(str(exc))
    else:
        col1, col2, col3 = st.columns(3)
        col1.metric("RSI(14)", f"{snap.rsi:.0f}")
        col2.metric("Momentum 20d", f"{snap.momentum:+.1%}")
        col3.metric("Ann. volatility", f"{snap.volatility:.1%}")

        st.caption("Adjusted close (dividends, JCP and splits).")
        st.line_chart(close)
        st.info(rule_based(ticker, snap.rsi, snap.momentum, snap.volatility))
