# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/) and the project adheres to
[Semantic Versioning](https://semver.org/).

## [0.6.0] - 2026-10-09

### Added
- Faithfulness eval of real LLM output: 120 seeded cases, five code checks
  (numbers, RSI label, volatility label, trend, guardrail), outputs recorded
  from Ollama into
  versioned cassettes and scored offline in CI, with Wilson intervals, the
  minimum detectable effect and an exact McNemar test against the rule-based
  baseline (`make evals`, `make evals-record`, ADR 0005). Each model is also
  recorded without the retrieved context (`--no-context`).
- Measured on the 120 cases (95% Wilson): qwen3:32b 120/120 [96.9%, 100%],
  qwen3:8b 115/120 = 95.8% [90.6%, 98.2%], rule-based baseline 120/120.
  Without the retrieved context qwen3:32b drops to 101/120 = 84.2% (exact
  McNemar p = 3.8e-6).
- LLM latency on a warm model, idle server, Apple M3 Max, n = 30
  (`make bench-llm`): qwen3:32b p50 7.8 s, p95 8.9 s;
  qwen3:8b p50 2.4 s, p95 2.7 s.
- Terminal demo regenerated from `docs/demo.tape`.
- Guardrail eval on 685 labeled EN + PT-BR sentences in three sets (dev, round
  1, round 2 out-of-sample), with the v0.5 deny-list as baseline.
- Retrieval eval: 32 labeled free-text queries and all 18 signal regimes,
  against the v0.5 fixed query and a random ranking.
- `/analyze` returns `explanation_source` (`llm` or `fallback`) and
  `guardrail_violations`, sets `X-Request-ID`, and logs one JSON line per
  request with per-stage latency and the fallback reason.
- `openai` provider for any OpenAI-compatible `/v1/chat/completions` endpoint.
- Greedy, seeded decoding with reasoning traces off (`LLM_TEMPERATURE`,
  `LLM_SEED`, `LLM_THINK`).
- CI matrix on Python 3.12 and 3.13, `uv sync --locked`, coverage floor of 90%,
  and a Docker job that waits for the container healthcheck.

### Changed
- **Breaking:** `last_price` is now `last_adjusted_close`. The value was always
  a close adjusted for dividends, JCP and splits.
- Retrieval queries the glossary once per signal regime instead of a constant
  query, so different regimes ground the prompt with different definitions.
  Measured: hit@1 on free-text questions 26/32 (v0.5 fixed query: 2/32, the
  same as random).
- Guardrail rewritten as word-bounded EN + PT-BR rules with negation handling.
  Measured on the out-of-sample round 2 set: recall 87/106 = 82.1%
  [73.7%, 88.2%] (v0.5: 15/106 = 14.2% [8.8%, 22.0%]); false positives 2/120 =
  1.7% [0.5%, 5.9%] (v0.5: 6/120 = 5.0%). Portuguese recall is 43/60; plural
  imperatives are a known gap.
- Series too short for the signals return 422 instead of NaN in the JSON.
- The Docker image installs from the lockfile with a pinned uv, runs as a
  non-root user and has a `/health` healthcheck.
- `scripts/benchmark.py` times only the offline stages and reports the recorded
  LLM latency. Quality moved to `python -m quantlens.evals`.

### Removed
- The v0.5 eval (`quantlens.evals` module), which checked the rule-based
  explainer against its own template. It is now a unit test.

## [0.5.0] - 2026-06-25

### Added
- Reproducible offline benchmark (`scripts/benchmark.py`, `make bench`): per-stage
  latency (p50/p95/p99) plus guardrail-recall and eval quality, run as a CI gate.
- Architecture Decision Records in `docs/adr/` (BM25 vs embeddings, local-first
  LLM, rule-based guardrails vs LLM judge, offline deterministic evals).
- README "Results / Benchmarks" (measured numbers) and "Limitations & next steps".

### Changed
- CI now runs the benchmark as a regression gate on guardrail/eval quality.

## [0.4.0] - 2026-06-23

### Added
- LoRA fine-tuning script (`scripts/finetune_lora.py`) with synthetic data.
- Streamlit demo (`app/streamlit_app.py`).
- `docker-compose.yml` for containerized serving.
- Optional extras: `finetune`, `demo`.

## [0.3.0] - 2026-06-23

### Added
- Output guardrails rejecting investment-advice / guarantee phrasing.
- Offline eval harness (faithfulness + guardrails), gated in CI.
- Shared `quantlens.explain` rule-based explainer (API + evals).

## [0.2.0] - 2026-06-23

### Added
- Local knowledge base + BM25 retriever (RAG) grounding the explanation.
- `/analyze` augments the LLM prompt with retrieved context.

## [0.1.0] - 2026-06-23

### Added
- Quant engine: RSI, simple returns, annualized volatility, momentum.
- FastAPI service with `/health` and `/analyze` endpoints.
- LLM explanation layer: local model via Ollama by default (the hosted-API
  provider promised here only shipped in 0.6.0, as the `openai` provider).
- Deterministic rule-based fallback when no LLM is available.
- Test suite, ruff, mypy, and GitHub Actions CI.
