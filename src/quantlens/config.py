"""Application configuration loaded from the environment / .env file."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings. Secrets come from the environment, never the repo."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "QuantLens"

    # LLM provider. Default is a local model served by Ollama (offline, no key).
    # Switch to a paid API (e.g. "anthropic", "openai") later without code changes.
    llm_provider: str = "ollama"
    llm_base_url: str = "http://localhost:11434"
    llm_model: str = "qwen3:32b"
    llm_api_key: str | None = None
    llm_timeout: float = 60.0


settings = Settings()
