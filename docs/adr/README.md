# Architecture Decision Records

Short, dated records of the non-obvious engineering decisions in QuantLens and
the trade-offs behind them. Each ADR states the context, the decision, the
alternatives that were rejected, and the consequences (including the downside we
accepted).

| ADR | Decision | Status |
|---|---|---|
| [0001](0001-bm25-over-embeddings.md) | BM25 retrieval queried by signal regime (not embeddings + vector DB), with measured recall | Accepted (revised 2026-10) |
| [0002](0002-local-first-llm.md) | Local-first LLM via Ollama behind a provider setting; greedy, seeded decoding | Accepted (revised 2026-10) |
| [0003](0003-rule-based-guardrails-and-fallback.md) | Rule-based EN + PT-BR guardrails and deterministic fallback (not an LLM judge) | Accepted (revised 2026-10) |
| [0004](0004-offline-deterministic-evals.md) | Offline, deterministic eval harness gated in CI | Superseded by 0005 |
| [0005](0005-faithfulness-eval-with-cassettes.md) | Faithfulness eval of real LLM output, scored offline from recorded cassettes | Accepted |

These are deliberately small. They exist to show *why*, not to be exhaustive.
