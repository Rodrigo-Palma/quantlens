# PregãoAI — AI Quant Analyst for the Brazilian Stock Market (B3)

[![CI](https://github.com/Rodrigo-Palma/quantlens/actions/workflows/ci.yml/badge.svg)](https://github.com/Rodrigo-Palma/quantlens/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

> Open-source AI agent that pulls B3 market data, computes **quantitative signals**
> (momentum, volatility, RSI, options Greeks) and **explains the thesis in plain language** —
> with RAG over fundamentals, a fine-tuned explainer, and automatically evaluated outputs.

<!-- ![demo](docs/demo.gif) -->

## Why

Most stock screeners output numbers. **PregãoAI turns quant signals into a reasoned,
evaluated explanation** an investor can actually read — combining classic quantitative
finance with a modern LLM agent. It is built as an end-to-end engineering showcase:
`data → quant → RAG → agent → evals → API → demo`.

## Features

- 📈 **Quant engine** — momentum, volatility, RSI (Black–Scholes Greeks for options on the roadmap)
- 🤖 **LLM agent** with tools (quote, signal, fundamentals lookup) — *roadmap v0.2*
- 📚 **RAG** over fundamentals / glossary / filings — *roadmap v0.2*
- 🧪 **Output evals** (faithfulness, hallucination) gated in CI — *roadmap v0.3*
- 🚀 **FastAPI** service + live demo, containerized — *roadmap v0.4*

## Architecture

```
[yfinance / brapi]                      public market data
        │
        ▼
[Quant Engine]  ── momentum, volatility, RSI, Black–Scholes (options)
        │  structured signals (JSON)
        ▼
[RAG]  ── fundamentals / glossary / news  →  vector DB (pgvector / Qdrant)
        │
        ▼
[LLM Agent]  ── tools: get_quote, compute_signal, search_fundamentals
        │  natural-language explanation
        ▼
[Guardrails + Evals]  ── faithfulness / hallucination / format  (runs in CI)
        │
        ▼
[FastAPI]  ──>  [Demo UI]  +  GIF
```

## Quickstart

```bash
git clone https://github.com/Rodrigo-Palma/quantlens.git
cd quantlens
uv sync --extra dev
cp .env.example .env            # add your LLM API key (optional in v0.1)
uv run uvicorn quantlens.api.main:app --reload
# then:
curl "http://localhost:8000/analyze?ticker=PETR4"
```

Interactive docs at `http://localhost:8000/docs`.

## Development

```bash
make install   # uv sync --extra dev
make lint      # ruff check
make type      # mypy
make test      # pytest + coverage
make run       # uvicorn (reload)
```

## Roadmap

- [x] **v0.1** — MVP: `ticker → quant signals → explanation`
- [ ] **v0.2** — RAG over fundamentals + LLM agent with tools
- [ ] **v0.3** — output evals + guardrails gated in CI
- [ ] **v0.4** — LoRA fine-tuning + cloud serving (Docker)

## Disclaimer

Educational / engineering project. **Not investment advice.** Uses public data only.

## Author

**Rodrigo Stachlewski Palma** — Data & AI Engineer (Azure Data Scientist & Fabric certified).
[LinkedIn](https://linkedin.com/in/rodrigospalma/) · [GitHub](https://github.com/Rodrigo-Palma)
