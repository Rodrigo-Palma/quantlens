"""LLM explanation layer.

Defaults to a local model served by Ollama (offline, no API key). The provider
is selected via settings, so swapping to a paid API (Anthropic/OpenAI) later is
a configuration change, not a code change.
"""

from __future__ import annotations

import re

import httpx

from quantlens.config import settings

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)

_PROMPT = (
    "You are a quantitative equity analyst. Given the signals below for a "
    "Brazilian (B3) stock, write a concise 2-3 sentence explanation for a retail "
    "investor. Be factual: mention the trend, the RSI condition and the "
    "volatility. Do NOT give buy/sell advice.\n\n"
    "Ticker: {ticker}\n"
    "Momentum (20 sessions): {mom:.2%}\n"
    "RSI(14): {rsi:.0f}\n"
    "Annualized volatility: {vol:.2%}\n"
)


def explain(ticker: str, rsi: float, mom: float, vol: float) -> str | None:
    """Return an LLM-generated explanation, or ``None`` if unavailable.

    Callers should fall back to a deterministic explanation when this returns
    ``None`` (e.g. the local model is not running).
    """
    prompt = _PROMPT.format(ticker=ticker, rsi=rsi, mom=mom, vol=vol)
    try:
        if settings.llm_provider == "ollama":
            return _ollama_generate(prompt)
        # Paid API providers (anthropic/openai) plug in here on the roadmap.
        return None
    except (httpx.HTTPError, KeyError, ValueError):
        return None


def _ollama_generate(prompt: str) -> str:
    response = httpx.post(
        f"{settings.llm_base_url}/api/generate",
        json={"model": settings.llm_model, "prompt": prompt, "stream": False},
        timeout=settings.llm_timeout,
    )
    response.raise_for_status()
    raw = str(response.json()["response"])
    return _THINK_RE.sub("", raw).strip()
