"""Retrieval eval: does the right glossary section come back?

Two query sets:

* ``free_text``: 32 hand-labeled questions, one relevant section each, mixing
  lexical overlap with paraphrase. Metrics: hit@1, hit@2 and MRR.
* ``regime``: every signal regime the API can see (3 RSI x 2 trend x
  3 volatility = 18, the full population, not a sample), with 3 relevant
  sections each. Metric: recall@3, for the per-signal retrieval the API uses
  and for a single joint query.

Systems: BM25 with the word tokenizer (current), BM25 with the v0.5 whitespace
tokenizer, the v0.5 fixed query (the same chunks for every request) and the
analytic expectation of a uniformly random ranking.
"""

from __future__ import annotations

import itertools
import json
from collections.abc import Callable
from dataclasses import dataclass
from importlib import resources

from quantlens import rag
from quantlens.evals.stats import Rate, wilson
from quantlens.quant.regime import Regime

LEGACY_FIXED_QUERY = "RSI momentum volatility PETR4"

_RSI_SECTION = {
    "oversold": "Oversold RSI (below 30)",
    "neutral": "Neutral RSI (between 30 and 70)",
    "overbought": "Overbought RSI (above 70)",
}
_TREND_SECTION = {
    "uptrend": "Uptrend (positive momentum)",
    "downtrend": "Downtrend (negative momentum)",
}
_VOL_SECTION = {
    "low": "Low volatility (below 20% a year)",
    "moderate": "Moderate volatility (20% to 40% a year)",
    "high": "High volatility (above 40% a year)",
}

Ranker = Callable[[str], list[str]]


@dataclass(frozen=True)
class LabeledQuery:
    query: str
    expected: str


@dataclass(frozen=True)
class FreeTextScore:
    system: str
    hit_at_1: Rate
    hit_at_2: Rate
    mrr: float


@dataclass(frozen=True)
class RegimeScore:
    system: str
    recall_at_3: Rate


def load_queries() -> tuple[LabeledQuery, ...]:
    path = resources.files("quantlens.evals").joinpath("data/retrieval_queries.jsonl")
    lines = path.read_text(encoding="utf-8").splitlines()
    return tuple(LabeledQuery(**json.loads(line)) for line in lines if line.strip())


def regime_cases() -> tuple[tuple[Regime, frozenset[str]], ...]:
    cases = []
    for rsi, trend, vol in itertools.product(_RSI_SECTION, _TREND_SECTION, _VOL_SECTION):
        regime = Regime(rsi=rsi, trend=trend, volatility=vol)  # type: ignore[arg-type]
        relevant = frozenset({_RSI_SECTION[rsi], _TREND_SECTION[trend], _VOL_SECTION[vol]})
        cases.append((regime, relevant))
    return tuple(cases)


def _bm25(tokenizer: rag.Tokenizer) -> Ranker:
    return lambda query: [chunk.title for chunk in rag.rank(query, tokenizer)]


def _fixed() -> Ranker:
    frozen = [chunk.title for chunk in rag.rank(LEGACY_FIXED_QUERY, "whitespace")]
    return lambda query: frozen


def rankers() -> dict[str, Ranker]:
    return {
        "bm25 (word tokens)": _bm25("word"),
        "bm25 (whitespace, v0.5)": _bm25("whitespace"),
        "fixed query (v0.5)": _fixed(),
    }


def score_free_text(name: str, ranker: Ranker, queries: tuple[LabeledQuery, ...]) -> FreeTextScore:
    positions = [ranker(q.query).index(q.expected) + 1 for q in queries]
    n = len(positions)
    return FreeTextScore(
        system=name,
        hit_at_1=wilson(sum(p <= 1 for p in positions), n),
        hit_at_2=wilson(sum(p <= 2 for p in positions), n),
        mrr=sum(1 / p for p in positions) / n,
    )


def score_regimes(name: str, ranker: Ranker) -> RegimeScore:
    found = total = 0
    for regime, relevant in regime_cases():
        top = set(ranker(regime.query())[:3])
        found += len(top & relevant)
        total += len(relevant)
    return RegimeScore(system=name, recall_at_3=wilson(found, total))


def score_per_signal() -> RegimeScore:
    """The API path: ``rag.retrieve_for_regime`` (one top-1 query per signal)."""
    found = total = 0
    for regime, relevant in regime_cases():
        titles = {_title(text) for text in rag.retrieve_for_regime(regime)}
        found += len(titles & relevant)
        total += len(relevant)
    return RegimeScore(system="bm25 per-signal (API)", recall_at_3=wilson(found, total))


def _title(text: str) -> str:
    return text.splitlines()[0].removeprefix("## ").strip()


def random_expectation(n_chunks: int) -> tuple[float, float, float, float]:
    """(hit@1, hit@2, MRR, recall@3) of a uniformly random ranking."""
    mrr = sum(1 / i for i in range(1, n_chunks + 1)) / n_chunks
    return 1 / n_chunks, 2 / n_chunks, mrr, 3 / n_chunks


def run() -> tuple[list[FreeTextScore], list[RegimeScore]]:
    queries = load_queries()
    free = [score_free_text(name, r, queries) for name, r in rankers().items()]
    regimes = [score_per_signal()]
    regimes += [score_regimes(f"{name}, joint", r) for name, r in rankers().items()]
    return free, regimes


def report() -> list[str]:
    free, regimes = run()
    n_chunks = len(rag.load_chunks())
    h1, h2, mrr, r3 = random_expectation(n_chunks)
    lines = [f"Retrieval: {len(load_queries())} free-text queries, {n_chunks} chunks"]
    lines.append(f"  {'system':26} {'hit@1':>26} {'hit@2':>26} {'MRR':>6}")
    for s in free:
        lines.append(f"  {s.system:26} {s.hit_at_1!s:>26} {s.hit_at_2!s:>26} {s.mrr:6.3f}")
    lines.append(f"  {'random (expected)':26} {h1:>26.1%} {h2:>26.1%} {mrr:6.3f}")
    lines.append(f"Retrieval: all {len(regime_cases())} signal regimes, recall@3 (exhaustive)")
    for r in regimes:
        rate = r.recall_at_3
        lines.append(f"  {r.system:34} {rate.successes}/{rate.n} = {rate.value:.1%}")
    lines.append(f"  {'random (expected)':34} {r3:.1%}")
    return lines


def measured() -> dict[str, float]:
    """Gate inputs: BM25 quality on free text and per-signal recall on regimes."""
    free, regimes = run()
    bm25 = next(s for s in free if s.system == "bm25 (word tokens)")
    per_signal = next(r for r in regimes if r.system == "bm25 per-signal (API)")
    return {
        "retrieval.free_text.bm25.hit_at_1": bm25.hit_at_1.value,
        "retrieval.regime.per_signal.recall_at_3": per_signal.recall_at_3.value,
    }
