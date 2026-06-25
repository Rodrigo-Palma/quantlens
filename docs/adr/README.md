# Architecture Decision Records

Short, dated records of the non-obvious engineering decisions in QuantLens and
the trade-offs behind them. Each ADR states the context, the decision, the
alternatives that were rejected, and the consequences (including the downside we
accepted).

| ADR | Decision | Status |
|---|---|---|
| [0001](0001-bm25-over-embeddings.md) | BM25 lexical retrieval for the RAG baseline (not embeddings + vector DB) | Accepted |
| [0002](0002-local-first-llm.md) | Local-first LLM via Ollama behind a provider-swappable interface | Accepted |
| [0003](0003-rule-based-guardrails-and-fallback.md) | Rule-based output guardrails + deterministic fallback (not an LLM judge) | Accepted |
| [0004](0004-offline-deterministic-evals.md) | Offline, deterministic eval harness gated in CI | Accepted |

These are deliberately small. They exist to show *why*, not to be exhaustive.
