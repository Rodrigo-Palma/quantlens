# ADR 0002: Local-first LLM via Ollama, behind a provider setting

**Status:** Accepted (revised 2026-10 for v0.6) · **Date:** 2026-06

## Context

The system needs an LLM to turn structured signals into a readable explanation.
Two pressures pull in opposite directions: a hosted API gives the best quality
with zero ops, but it costs money per request, requires a key, and makes the
repo impossible to run or test for free; a local model is free and private but
lower quality and hardware-bound.

## Decision

Default to a **local model served by Ollama** (`qwen3:32b`), selected via
`settings.llm_provider`. Two providers exist and are tested with the HTTP
transport mocked:

- `ollama`: native `/api/generate`.
- `openai`: any OpenAI-compatible `/v1/chat/completions` endpoint, key from
  `LLM_API_KEY`. Verified live against Ollama's own `/v1` endpoint; it has not
  been run against a hosted vendor from this repo.

`llm.explain(...)` returns `None` when the model is unavailable or the reply is
malformed, and the API falls back to the deterministic explainer and says so in
`explanation_source`.

Decoding is greedy and seeded (`temperature=0`, `seed=0`) and qwen3 reasoning
traces are off (`think=false`). The task is a 3-sentence summary of numbers the
prompt already contains; reasoning tokens add latency without a measured gain,
and deterministic decoding is what lets the eval cassettes
([ADR 0005](0005-faithfulness-eval-with-cassettes.md)) stand for production
behavior.

## Alternatives considered

- **Hosted API as default.** Best quality, but adds cost and a key requirement
  and blocks free CI.
- **A vendor SDK per provider (Anthropic, OpenAI).** More features (tool use,
  streaming), but two more dependencies for a single text completion. The
  OpenAI-compatible wire format already covers OpenAI, vLLM, LM Studio and
  Ollama.
- **Hard dependency on a local model (no fallback).** Then the endpoint fails
  whenever Ollama is down.

## Consequences

- Anyone can clone and run the project end to end with no API key and no cost.
- The endpoint never fails on the explanation step; it degrades to a
  deterministic, guardrail-safe summary, and the response says which one the
  caller got.
- Local-model latency is seconds, not milliseconds (see the README table), and
  large local models need real hardware. **Accepted**: the fallback bounds the
  worst case.
- An Anthropic-native provider does not exist. Adding one is a new function in
  `llm.py`, not a configuration change.
