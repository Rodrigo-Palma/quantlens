"""Tests for the BM25 knowledge retriever (offline, deterministic)."""

from __future__ import annotations

import itertools

import pytest

from quantlens import rag
from quantlens.evals import retrieval
from quantlens.quant.regime import Regime, classify


def _titles(texts: list[str]) -> list[str]:
    return [text.splitlines()[0].removeprefix("## ") for text in texts]


def test_retrieve_returns_relevant_chunk() -> None:
    results = rag.retrieve("what does an overbought RSI mean", k=1)
    assert _titles(results) == ["Overbought RSI (above 70)"]


def test_retrieve_respects_k() -> None:
    assert len(rag.retrieve("volatility risk", k=3)) == 3


def test_word_tokenizer_ignores_punctuation() -> None:
    top = rag.rank("Wilder smoothing.", "word")[0]
    assert top.title == "RSI (Relative Strength Index)"


def test_chunks_have_unique_titles() -> None:
    titles = [chunk.title for chunk in rag.load_chunks()]
    assert len(titles) == len(set(titles))


@pytest.mark.parametrize(
    ("rsi", "expected"), [(20.0, "Oversold"), (50.0, "Neutral"), (80.0, "Overbought")]
)
def test_rsi_regime_drives_the_retrieved_definition(rsi: float, expected: str) -> None:
    titles = _titles(rag.retrieve_for_regime(classify(rsi, 0.05, 0.30)))
    assert titles[0].startswith(expected)


def test_different_rsi_regimes_retrieve_different_context() -> None:
    contexts = {tuple(rag.retrieve_for_regime(classify(rsi, 0.05, 0.3))) for rsi in (20, 50, 80)}
    assert len(contexts) == 3


def test_every_regime_retrieves_its_three_sections() -> None:
    for regime, relevant in retrieval.regime_cases():
        assert set(_titles(rag.retrieve_for_regime(regime))) == relevant, regime


def test_regime_cases_cover_the_full_grid() -> None:
    cases = retrieval.regime_cases()
    seen = {(r.rsi, r.trend, r.volatility) for r, _ in cases}
    grid = itertools.product(
        ("oversold", "neutral", "overbought"), ("uptrend", "downtrend"), ("low", "moderate", "high")
    )
    assert seen == set(grid)


def test_joint_query_is_the_three_signal_queries() -> None:
    regime = Regime(rsi="neutral", trend="uptrend", volatility="low")
    assert regime.query() == " ".join(regime.queries())


def test_bm25_beats_fixed_and_random_on_free_text() -> None:
    free, _ = retrieval.run()
    by_name = {score.system: score for score in free}
    bm25 = by_name["bm25 (word tokens)"]
    fixed = by_name["fixed query (v0.5)"]
    h1, _, mrr, _ = retrieval.random_expectation(len(rag.load_chunks()))
    assert bm25.hit_at_1.low > fixed.hit_at_1.high
    assert bm25.hit_at_1.low > h1
    assert bm25.mrr > mrr


def test_report_lists_every_system() -> None:
    text = "\n".join(retrieval.report())
    for name in ("bm25 per-signal (API)", "fixed query (v0.5)", "random (expected)"):
        assert name in text
