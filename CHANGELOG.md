# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/) and the project adheres to
[Semantic Versioning](https://semver.org/).

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
- LLM explanation layer — local model via Ollama by default, swappable to a
  paid API (Anthropic/OpenAI) through configuration.
- Deterministic rule-based fallback when no LLM is available.
- Test suite, ruff, mypy, and GitHub Actions CI.
