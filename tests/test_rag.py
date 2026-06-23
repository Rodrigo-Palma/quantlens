"""Tests for the BM25 knowledge retriever (offline, deterministic)."""

from __future__ import annotations

from quantlens import rag


def test_retrieve_returns_relevant_chunk() -> None:
    results = rag.retrieve("what does an overbought RSI mean", k=1)
    assert results
    assert "RSI" in results[0]


def test_retrieve_respects_k() -> None:
    assert len(rag.retrieve("volatility risk", k=3)) <= 3
