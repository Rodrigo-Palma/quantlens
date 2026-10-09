"""Lexical retrieval over the local knowledge base (BM25, fully offline).

The glossary is split into one chunk per ``##`` section. The API builds its query
from the signal regime (see ``quantlens.quant.regime``), so different regimes
ground the LLM with different definitions; the retrieval eval measures how often
the right section comes back.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from typing import Literal

from rank_bm25 import BM25Okapi

from quantlens.quant.regime import Regime

Tokenizer = Literal["word", "whitespace"]

_WORD_RE = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True)
class Chunk:
    title: str
    text: str


def _word_tokens(text: str) -> list[str]:
    """Lowercase alphanumeric tokens: ``"(RSI)"`` and ``"rsi"`` match."""
    return _WORD_RE.findall(text.lower())


def _whitespace_tokens(text: str) -> list[str]:
    """The v0.5 tokenizer, kept so the eval can measure what the change bought."""
    return text.lower().split()


_TOKENIZERS: dict[Tokenizer, Callable[[str], list[str]]] = {
    "word": _word_tokens,
    "whitespace": _whitespace_tokens,
}


def load_chunks() -> tuple[Chunk, ...]:
    """Split the glossary into chunks, one per ``##`` heading section."""
    text = (
        resources.files("quantlens.knowledge").joinpath("glossary.md").read_text(encoding="utf-8")
    )
    sections = re.split(r"(?m)^## ", text)[1:]
    chunks = []
    for section in sections:
        title, _, _body = section.partition("\n")
        chunks.append(Chunk(title=title.strip(), text=f"## {section}".strip()))
    return tuple(chunks)


@lru_cache(maxsize=len(_TOKENIZERS))
def _index(tokenizer: Tokenizer) -> tuple[BM25Okapi, tuple[Chunk, ...]]:
    chunks = load_chunks()
    tokenize = _TOKENIZERS[tokenizer]
    return BM25Okapi([tokenize(chunk.text) for chunk in chunks]), chunks


def rank(query: str, tokenizer: Tokenizer = "word") -> list[Chunk]:
    """All chunks ordered by BM25 score for ``query`` (stable on ties)."""
    bm25, chunks = _index(tokenizer)
    scores = bm25.get_scores(_TOKENIZERS[tokenizer](query))
    order = sorted(range(len(chunks)), key=lambda i: (-scores[i], i))
    return [chunks[i] for i in order]


def retrieve(query: str, k: int = 3) -> list[str]:
    """Return the text of the top-``k`` chunks most relevant to ``query``."""
    return [chunk.text for chunk in rank(query)[:k]]


def retrieve_for_regime(regime: Regime) -> list[str]:
    """Top-1 chunk per signal (RSI, trend, volatility), de-duplicated in order.

    One query per signal keeps a section that mentions every RSI threshold (the
    neutral one) from crowding out the volatility section in a joint query.
    """
    texts: list[str] = []
    for query in regime.queries():
        top = rank(query)[0].text
        if top not in texts:
            texts.append(top)
    return texts
