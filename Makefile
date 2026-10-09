.PHONY: install lint fmt fmt-check type test bench evals evals-record bench-llm run docker all

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

# Re-record the LLM cassettes (needs a local Ollama with the models pulled).
evals-record:
	LLM_PROVIDER=ollama uv run python -m quantlens.evals.record --model qwen3:32b --model qwen3:8b

bench:
	uv run python scripts/benchmark.py

# Re-measure LLM latency on a warm model (run with no other load on Ollama).
bench-llm:
	LLM_PROVIDER=ollama uv run python -m quantlens.evals.latency --model qwen3:32b --model qwen3:8b --n 30

fmt-check:
	uv run ruff format --check .

docker:
	docker build -t quantlens:dev .

run:
	uv run uvicorn quantlens.api.main:app --reload

all: lint fmt-check type test evals bench
