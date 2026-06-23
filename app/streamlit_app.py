"""Streamlit demo for QuantLens.

Optional extra::

    uv sync --extra demo
    uv run streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import streamlit as st

from quantlens.data.market import fetch_close
from quantlens.explain import rule_based
from quantlens.quant import signals

st.set_page_config(page_title="QuantLens", page_icon="📈")
st.title("📈 QuantLens — B3 quant analyst")
st.caption("Educational engineering demo. Not investment advice.")

ticker = st.text_input("B3 ticker", value="PETR4")

if st.button("Analyze") and ticker:
    try:
        close = fetch_close(ticker)
    except ValueError as exc:
        st.error(str(exc))
    else:
        rsi = float(signals.rsi(close).iloc[-1])
        mom = signals.momentum(close)
        vol = signals.annualized_volatility(close)

        col1, col2, col3 = st.columns(3)
        col1.metric("RSI(14)", f"{rsi:.0f}")
        col2.metric("Momentum 20d", f"{mom:+.1%}")
        col3.metric("Ann. volatility", f"{vol:.1%}")

        st.line_chart(close)
        st.info(rule_based(ticker.upper(), rsi, mom, vol))
