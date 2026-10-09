"""Application configuration loaded from the environment / .env file."""

from __future__ import annotations

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


settings = Settings()
