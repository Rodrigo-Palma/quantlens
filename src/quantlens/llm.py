"""LLM explanation layer.

Two providers sit behind ``explain``, selected by ``settings.llm_provider``:

* ``ollama``: a local model through Ollama's native ``/api/generate`` (the default,
  offline, no API key).
* ``openai``: any OpenAI-compatible ``/v1/chat/completions`` endpoint (OpenAI,
  vLLM, LM Studio, or Ollama's own ``/v1``), with the key read from
  ``LLM_API_KEY``.

Any other provider value, a transport error or a malformed reply returns ``None``
and the caller serves the deterministic explanation instead.
"""

from __future__ import annotations

import re

import httpx

from quantlens.config import settings

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)

PROMPT_TEMPLATE = (
    "You are a quantitative equity analyst. Given the signals below for a "
    "Brazilian (B3) stock, write a concise 2-3 sentence explanation for a retail "
    "investor. Be factual: mention the trend, the RSI condition and the "
    "volatility. Do NOT give buy/sell advice.\n\n"
    "Ticker: {ticker}\n"
    "Momentum (20 sessions): {mom:.2%}\n"
    "RSI(14): {rsi:.0f}\n"
    "Annualized volatility: {vol:.2%}\n"
)


def build_prompt(ticker: str, rsi: float, mom: float, vol: float, context: str | None) -> str:
    """The exact prompt sent to the model (shared by the API and the eval recorder)."""
    prompt = PROMPT_TEMPLATE.format(ticker=ticker, rsi=rsi, mom=mom, vol=vol)
    if context:
        prompt += (
            "\nReference material (use only to define terms accurately, "
            f"do not quote verbatim):\n{context}\n"
        )
    return prompt


def explain(
    ticker: str,
    rsi: float,
    mom: float,
    vol: float,
    context: str | None = None,
) -> str | None:
    """Return an LLM-generated explanation, or ``None`` if unavailable.

    ``context`` is optional retrieved reference material (RAG) used to ground
    term definitions.
    """
    return generate(build_prompt(ticker, rsi, mom, vol, context))


def generate(prompt: str, model: str | None = None) -> str | None:
    """Send ``prompt`` to the configured provider; ``None`` on any failure."""
    chosen = model or settings.llm_model
    try:
        if settings.llm_provider == "ollama":
            return _clean(_ollama_generate(prompt, chosen))
        if settings.llm_provider == "openai":
            return _clean(_openai_chat(prompt, chosen))
        return None
    except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError):
        return None


def _clean(raw: str) -> str | None:
    text = _THINK_RE.sub("", raw).strip()
    return text or None


def _ollama_generate(prompt: str, model: str) -> str:
    response = httpx.post(
        f"{settings.llm_base_url}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "think": settings.llm_think,
            "options": {"temperature": settings.llm_temperature, "seed": settings.llm_seed},
        },
        timeout=settings.llm_timeout,
    )
    response.raise_for_status()
    return str(response.json()["response"])


def _openai_chat(prompt: str, model: str) -> str:
    headers = {"Authorization": f"Bearer {settings.llm_api_key}"} if settings.llm_api_key else {}
    response = httpx.post(
        f"{settings.llm_base_url.rstrip('/')}/v1/chat/completions",
        headers=headers,
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": settings.llm_temperature,
            "seed": settings.llm_seed,
        },
        timeout=settings.llm_timeout,
    )
    response.raise_for_status()
    return str(response.json()["choices"][0]["message"]["content"])
