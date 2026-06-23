# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/) and the project adheres to
[Semantic Versioning](https://semver.org/).

## [0.1.0] - 2026-06-23

### Added
- Quant engine: RSI, simple returns, annualized volatility, momentum.
- FastAPI service with `/health` and `/analyze` endpoints.
- LLM explanation layer — local model via Ollama by default, swappable to a
  paid API (Anthropic/OpenAI) through configuration.
- Deterministic rule-based fallback when no LLM is available.
- Test suite, ruff, mypy, and GitHub Actions CI.
