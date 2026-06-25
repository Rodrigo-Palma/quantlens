.PHONY: install lint fmt type test bench evals run all

install:
	uv sync --extra dev

lint:
	uv run ruff check .

fmt:
	uv run ruff format .

type:
	uv run mypy

test:
	uv run pytest

evals:
	uv run python -m quantlens.evals

bench:
	uv run python scripts/benchmark.py

run:
	uv run uvicorn quantlens.api.main:app --reload

all: lint type test evals
