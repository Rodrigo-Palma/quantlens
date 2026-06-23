# Contributing

Thanks for your interest in QuantLens.

## Setup

```bash
uv sync --extra dev
uv run pre-commit install
```

## Workflow

1. Create a branch from `main`.
2. Make your change with tests.
3. Run `make all` (lint + types + tests) — it must pass.
4. Open a PR; CI must be green.

## Conventions

- Code, comments and docs in English.
- Type hints required (`mypy`), formatting via `ruff format`.
- Keep network I/O isolated in `quantlens.data`; keep `quantlens.quant` pure.
