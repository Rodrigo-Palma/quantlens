"""Application configuration loaded from the environment / .env file."""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings. Secrets come from the environment, never the repo."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "QuantLens"

    # LLM provider: "ollama" (local, default, no key) or "openai" (any
    # OpenAI-compatible /v1/chat/completions endpoint; key via LLM_API_KEY).
    llm_provider: str = "ollama"
    llm_base_url: str = "http://localhost:11434"
    llm_model: str = "qwen3:32b"
    llm_api_key: str | None = None
    llm_timeout: float = 60.0
    # Decoding: greedy and seeded so explanations (and eval cassettes) are
    # reproducible; reasoning traces off because the task is a 3-sentence summary.
    llm_temperature: float = 0.0
    llm_seed: int = 0
    llm_think: bool = False

    # Seconds a fetched series and a generated explanation are reused (0 disables).
    cache_ttl_s: float = 900.0
    cache_max_entries: int = 256

    # /analyze rate limit per client address, in this process only: a token
    # bucket of RATE_LIMIT_BURST refilled at RATE_LIMIT_PER_MIN, over at most
    # RATE_LIMIT_MAX_CLIENTS tracked addresses (least recently seen evicted).
    rate_limit_per_min: float = Field(default=30.0, gt=0)
    rate_limit_burst: int = Field(default=10, gt=0)
    rate_limit_max_clients: int = Field(default=10_000, gt=0)
    # Comma-separated IPs or CIDRs whose X-Forwarded-For is believed. Empty
    # means the header is ignored and the direct peer is the client.
    trusted_proxies: str = ""
    # LLM calls in flight at once; a request past it gets the rule-based
    # explanation instead of waiting.
    llm_max_concurrency: int = Field(default=2, gt=0)


settings = Settings()
