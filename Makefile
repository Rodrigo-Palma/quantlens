.PHONY: install lint fmt type test run all

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

run:
	uv run uvicorn quantlens.api.main:app --reload

all: lint type test
