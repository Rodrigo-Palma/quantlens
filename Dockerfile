FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app

# uv pinned by version so the image build is reproducible.
COPY --from=ghcr.io/astral-sh/uv:0.12.24 /uv /usr/local/bin/uv

# Dependencies first (cached layer), resolved strictly from the committed lockfile.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-dev --no-install-project

COPY src ./src
RUN uv sync --locked --no-dev

RUN useradd --create-home --uid 10001 app && chown -R app /app
USER app

ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD ["python", "-c", "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2).status == 200 else 1)"]

CMD ["uvicorn", "quantlens.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
