# ADR 0002 — Local-first LLM via Ollama, behind a swappable provider

**Status:** Accepted · **Date:** 2026-06

## Context

The system needs an LLM to turn structured signals into a readable explanation.
Two pressures pull in opposite directions: a paid API (Anthropic/OpenAI) gives
the best quality with zero ops, but it costs money per request, requires a key,
and makes the repo impossible to run or test for free; a local model is free and
private but lower quality and hardware-bound.

## Decision

Default to a **local model served by Ollama** (`qwen3:32b`), selected via
`settings.llm_provider`. The `llm.explain(...)` function returns `None` when the
model is unavailable, and callers fall back to the deterministic explainer. The
provider is a **configuration value, not a code path the caller knows about** —
switching to a paid API is a `.env` change plus one branch in `llm.py`.

## Alternatives considered

- **Paid API as default.** Best quality, but adds cost and a key requirement to
  a portfolio/educational project, and blocks free CI.
- **Hard dependency on a local model (no fallback).** Then the API endpoint
  fails whenever Ollama is down — unacceptable for a service.

## Consequences

- ✅ Anyone can clone and run the project end to end with no API key and no cost.
- ✅ The endpoint never fails on the explanation step — it degrades to a
  deterministic, guardrail-safe summary.
- ✅ Production deployments can opt into a paid API for quality without touching
  call sites.
- ⚠️ Local-model output quality is below frontier APIs, and large local models
  need real hardware. **Accepted**: the deterministic fallback bounds the
  worst case, and the provider switch buys quality when it matters.
