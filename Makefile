.PHONY: install lint fmt fmt-check type test bench evals run docker all

install:
	uv sync --locked --extra dev

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

fmt-check:
	uv run ruff format --check .

docker:
	docker build -t quantlens:dev .

run:
	uv run uvicorn quantlens.api.main:app --reload

all: lint fmt-check type test evals bench
