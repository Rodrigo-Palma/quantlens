# QuantLens — AI Quant Analyst for the Brazilian Stock Market (B3)

[![CI](https://github.com/Rodrigo-Palma/quantlens/actions/workflows/ci.yml/badge.svg)](https://github.com/Rodrigo-Palma/quantlens/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

> Open-source AI agent that pulls B3 market data, computes **quantitative signals**
> (momentum, volatility, RSI) and **explains the thesis in plain language** — grounded
> by retrieval (RAG) and protected by output guardrails and an offline eval suite.
> Local-first: runs on a local LLM (Ollama) with no API key, swappable to a paid API.

<!-- ![demo](docs/demo.gif) -->

## Why

Most stock screeners output numbers. **QuantLens turns quant signals into a reasoned,
evaluated explanation** an investor can actually read — combining classic quantitative
finance with a modern LLM stack. It is built as an end-to-end engineering showcase:
`data → quant → RAG → LLM → guardrails/evals → API → demo`.

## Features

- 📈 **Quant engine** — RSI, momentum, annualized volatility (pure, unit-tested)
- 📚 **RAG** — BM25 retrieval over a local knowledge base, grounding the explanation
- 🤖 **LLM explanation** — local model via Ollama by default; swappable to a paid API
- 🛡️ **Guardrails** — rejects investment-advice / guarantee phrasing
- 🧪 **Offline evals** — faithfulness + guardrail checks gated in CI (no network/LLM)
- 🚀 **FastAPI** service + Streamlit demo, containerized
- 🔧 **LoRA fine-tuning** script to specialize the explainer

## Architecture

```
[yfinance]                              public market data
     │
     ▼
[Quant Engine]  ── RSI, momentum, annualized volatility
     │  structured signals
     ▼
[RAG]  ── BM25 over local knowledge base (glossary)  →  grounding context
     │
     ▼
[LLM]  ── local Ollama model (swappable to a paid API)
     │  natural-language explanation
     ▼
[Guardrails + Evals]  ── no-advice checks / faithfulness  (evals run in CI)
     │
     ▼
[FastAPI]  ──>  [Streamlit demo]
```

## Quickstart

```bash
git clone https://github.com/Rodrigo-Palma/quantlens.git
cd quantlens
uv sync --extra dev
cp .env.example .env            # defaults to a local Ollama model (no key)
uv run uvicorn quantlens.api.main:app --reload
curl "http://localhost:8000/analyze?ticker=PETR4"
```

Interactive docs at `http://localhost:8000/docs`. The explanation uses your local
Ollama model (`qwen3:32b` by default); if Ollama is not running it falls back to a
deterministic, guardrail-safe summary.

## Development

```bash
make install   # uv sync --extra dev
make lint      # ruff check
make type      # mypy
make test      # pytest + coverage
uv run python -m quantlens.evals   # offline eval suite
```

## Demo

```bash
uv sync --extra demo
uv run streamlit run app/streamlit_app.py
```

## Fine-tuning (LoRA)

Specialize the explainer on synthetic signal→explanation pairs:

```bash
uv sync --extra finetune
uv run python scripts/finetune_lora.py --max-samples 64 --epochs 1
```

Defaults are CPU-friendly for a smoke run; use a GPU and a larger base model for
real training.

## Deploy

```bash
docker compose up --build      # serves the API on :8000
```

## Roadmap

- [x] **v0.1** — MVP: `ticker → quant signals → explanation`
- [x] **v0.2** — RAG grounding over a local knowledge base
- [x] **v0.3** — output guardrails + offline eval suite gated in CI
- [x] **v0.4** — LoRA fine-tuning script + containerized serving + demo

## Disclaimer

Educational / engineering project. **Not investment advice.** Uses public data only.

## Author

**Rodrigo Stachlewski Palma** — Data & AI Engineer (Azure Data Scientist & Fabric certified).
[LinkedIn](https://linkedin.com/in/rodrigospalma/) · [GitHub](https://github.com/Rodrigo-Palma)
