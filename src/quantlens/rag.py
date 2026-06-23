"""Lightweight retrieval over the local knowledge base (BM25, fully offline).

This is a dependency-light RAG baseline. The public ``retrieve`` signature is
stable, so swapping in embeddings + a vector DB (pgvector/Qdrant) later is an
internal change only.
"""

from __future__ import annotations

from functools import lru_cache
from importlib import resources

from rank_bm25 import BM25Okapi


def _load_chunks() -> list[str]:
    """Split the glossary into chunks, one per ``##`` heading section."""
    text = (
        resources.files("quantlens.knowledge").joinpath("glossary.md").read_text(encoding="utf-8")
    )
    chunks: list[str] = []
    current: list[str] = []
    for line in text.splitlines():
        if line.startswith("## ") and current:
            chunks.append("\n".join(current).strip())
            current = [line]
        else:
            current.append(line)
    if current:
        chunks.append("\n".join(current).strip())
    return [chunk for chunk in chunks if chunk]


@lru_cache(maxsize=1)
def _index() -> tuple[BM25Okapi, list[str]]:
    chunks = _load_chunks()
    tokenized = [chunk.lower().split() for chunk in chunks]
    return BM25Okapi(tokenized), chunks


def retrieve(query: str, k: int = 2) -> list[str]:
    """Return the top-``k`` knowledge snippets most relevant to ``query``."""
    bm25, chunks = _index()
    scores = bm25.get_scores(query.lower().split())
    ranked = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)
    return [chunks[i] for i in ranked[:k]]
