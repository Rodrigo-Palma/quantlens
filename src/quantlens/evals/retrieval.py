"""Retrieval eval: does the right glossary section come back?

Query sets:

* ``free_text`` (dev): 32 hand-labeled questions, one relevant section each,
  mixing lexical overlap with paraphrase. Written in the same commit as the
  glossary rewrite and the word tokenizer, so it is in-sample.
* ``heldout``: questions written by a different author (another LLM that saw
  only the section titles), labels reviewed by hand, measured once against the
  frozen glossary and tokenizer. The out-of-sample number.

  Metrics on both: hit@1, hit@2 and MRR, and an exact McNemar test on hit@1 of
  the word tokenizer against the v0.5 whitespace tokenizer.
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
from quantlens.evals.stats import Rate, mcnemar_exact, wilson
from quantlens.quant.regime import Regime

LEGACY_FIXED_QUERY = "RSI momentum volatility PETR4"
QUERY_SETS = {"free_text": "retrieval_queries", "heldout": "retrieval_heldout"}

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
    top1: tuple[bool, ...]


@dataclass(frozen=True)
class RegimeScore:
    system: str
    recall_at_3: Rate


def load_queries(name: str = "retrieval_queries") -> tuple[LabeledQuery, ...]:
    path = resources.files("quantlens.evals").joinpath(f"data/{name}.jsonl")
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
        top1=tuple(p == 1 for p in positions),
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


Scores = dict[str, list[FreeTextScore]]
WORD, WHITESPACE = "bm25 (word tokens)", "bm25 (whitespace, v0.5)"


def run() -> tuple[Scores, list[RegimeScore]]:
    ranks = rankers()
    free = {
        label: [score_free_text(name, r, load_queries(file)) for name, r in ranks.items()]
        for label, file in QUERY_SETS.items()
    }
    regimes = [score_per_signal()]
    regimes += [score_regimes(f"{name}, joint", r) for name, r in ranks.items()]
    return free, regimes


def tokenizer_test(scores: list[FreeTextScore]) -> tuple[int, int, float]:
    """(only word hits, only whitespace hits, exact McNemar p) on hit@1."""
    word = next(s for s in scores if s.system == WORD).top1
    space = next(s for s in scores if s.system == WHITESPACE).top1
    only_word = sum(a and not b for a, b in zip(word, space, strict=True))
    only_space = sum(b and not a for a, b in zip(word, space, strict=True))
    return only_word, only_space, mcnemar_exact(only_word, only_space)


def report() -> list[str]:
    free, regimes = run()
    n_chunks = len(rag.load_chunks())
    h1, h2, mrr, r3 = random_expectation(n_chunks)
    lines = []
    for label, scores in free.items():
        title = "in-sample dev set" if label == "free_text" else "held-out, other author"
        lines.append(f"Retrieval: {scores[0].hit_at_1.n} {title} queries, {n_chunks} chunks")
        lines.append(f"  {'system':26} {'hit@1':>26} {'hit@2':>26} {'MRR':>6}")
        for s in scores:
            lines.append(f"  {s.system:26} {s.hit_at_1!s:>26} {s.hit_at_2!s:>26} {s.mrr:6.3f}")
        lines.append(f"  {'random (expected)':26} {h1:>26.1%} {h2:>26.1%} {mrr:6.3f}")
        only_word, only_space, p = tokenizer_test(scores)
        lines.append(
            f"  paired hit@1, word vs whitespace tokens: only word {only_word}, "
            f"only whitespace {only_space}, exact McNemar p = {p:.3g}"
        )
    lines.append(f"Retrieval: all {len(regime_cases())} signal regimes, recall@3 (exhaustive)")
    for r in regimes:
        rate = r.recall_at_3
        lines.append(f"  {r.system:34} {rate.successes}/{rate.n} = {rate.value:.1%}")
    lines.append(f"  {'random (expected)':34} {r3:.1%}")
    return lines


def measured() -> dict[str, int]:
    """Gate inputs: BM25 hit@1 counts on each query set and per-signal recall."""
    free, regimes = run()
    out: dict[str, int] = {}
    for label, scores in free.items():
        bm25 = next(s for s in scores if s.system == WORD)
        out[f"retrieval.{label}.bm25.hits_at_1"] = bm25.hit_at_1.successes
        out[f"retrieval.{label}.n"] = bm25.hit_at_1.n
    per_signal = next(r for r in regimes if r.system == "bm25 per-signal (API)")
    out["retrieval.regime.per_signal.found_at_3"] = per_signal.recall_at_3.successes
    out["retrieval.regime.n"] = per_signal.recall_at_3.n
    return out
