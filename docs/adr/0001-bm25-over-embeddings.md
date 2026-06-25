# ADR 0001 — BM25 lexical retrieval for the RAG baseline

**Status:** Accepted · **Date:** 2026-06

## Context

The explanation step needs grounding context (definitions of RSI, momentum,
volatility) so the LLM explains terms accurately instead of hallucinating them.
The knowledge base is small (a curated glossary, tens of chunks) and the project
is local-first with a hard "runs offline, no API key, no service to stand up"
constraint.

## Decision

Use **BM25** (`rank-bm25`) over chunks split by `##` heading, with a stable
`retrieve(query, k)` interface. The index is built once and cached
(`lru_cache`).

## Alternatives considered

- **Sentence-embeddings + vector DB (pgvector / Qdrant).** Better semantic
  recall on paraphrased queries, but adds a model download, a service or a
  native dependency, and cold-start cost — for a glossary of tens of chunks the
  recall gain does not pay for the operational weight, and it breaks the
  "offline in CI" guarantee.
- **Naive substring / keyword match.** Zero dependencies, but no term
  weighting; common words dominate ranking.

## Consequences

- ✅ Fully offline, deterministic, microsecond-latency retrieval (see the
  benchmark table in the README — BM25 p50 ≈ 13 µs).
- ✅ The `retrieve` signature is stable, so swapping in embeddings later is an
  internal change with no caller impact.
- ⚠️ Lexical matching misses semantic paraphrase. **Accepted** at this corpus
  size; revisit ADR if the knowledge base grows past a few hundred chunks or
  starts serving free-text user questions.
