# ADR 0003 — Rule-based output guardrails + deterministic fallback

**Status:** Accepted · **Date:** 2026-06

## Context

This tool describes equity signals; it must **never emit investment advice or
guarantees** ("you should buy", "risk-free", "guaranteed"). LLM output is
non-deterministic, so even a well-prompted model can drift into advice phrasing.
A guardrail is required on the output, and it has to be enforceable in CI with no
network.

## Decision

Validate every generated explanation against a **deterministic deny-list of
advice/guarantee phrases** (`guardrails.validate`). If the LLM output trips a
rule, the API discards it and serves the **rule-based deterministic explainer**
instead. The same validator is asserted in the offline eval suite.

## Alternatives considered

- **LLM-as-judge guardrail.** Catches paraphrased advice the deny-list misses,
  but it is non-deterministic, adds latency and cost, can itself hallucinate a
  verdict, and cannot run offline in CI. Wrong tool for a hard safety invariant.
- **No guardrail, prompt-only.** Relies entirely on the model behaving — not
  acceptable for a safety property.

## Consequences

- ✅ The safety invariant is deterministic, microsecond-cheap (p50 ≈ 1 µs) and
  testable offline. Measured guardrail recall on the probe set: 4/4 advice
  phrases blocked, 0 false positives (see README benchmark).
- ✅ A guardrail trip degrades gracefully to a safe explanation rather than
  failing the request.
- ⚠️ A deny-list is lexical: novel phrasings of advice can slip through. The
  blast radius is bounded because the *input* prompt also forbids advice, so the
  guardrail is a second line, not the only one. An LLM-judge can be added as an
  **advisory** layer later without removing the deterministic gate.
