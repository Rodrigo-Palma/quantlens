FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# uv for fast, reproducible installs
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

COPY pyproject.toml README.md ./
COPY src ./src
RUN uv sync --no-dev

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "quantlens.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
