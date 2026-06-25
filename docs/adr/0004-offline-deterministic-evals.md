# ADR 0004 — Offline, deterministic eval harness gated in CI

**Status:** Accepted · **Date:** 2026-06

## Context

"Has evals" is cheap to claim and hard to trust. To be worth anything, the eval
suite has to (a) run on every push, (b) fail the build on a regression, and
(c) not depend on a network call or a non-deterministic LLM — otherwise CI is
flaky and the signal is noise.

## Decision

Evaluate the **deterministic rule-based explainer** (the same code used as the
production fallback) over a small labeled `EVAL_SET`, checking two properties:
**faithfulness** (the text reflects the input signals — mentions RSI, the right
trend, volatility) and **guardrail compliance** (no advice phrasing). The runner
exits non-zero below a 100% pass threshold and is wired into CI.

## Alternatives considered

- **LLM-graded evals (ragas / LLM-as-judge) against the live model.** Higher
  fidelity to real output quality, but non-deterministic and network-bound —
  flaky in CI and unable to gate a build reliably. Belongs in a separate,
  nightly/manual job, not the blocking PR check.
- **No automated evals.** The README would claim "evals" with nothing enforcing
  them — exactly the gap this ADR exists to close.

## Consequences

- ✅ A regression in the explanation logic fails CI deterministically and for
  free (current: 4/4 cases pass).
- ✅ Because the fallback explainer is what gets evaluated, the property tested
  is the property shipped in the worst case.
- ⚠️ This does **not** measure the *LLM's* output quality — only the
  deterministic floor. Measuring the model itself (faithfulness/grounding
  scoring of generated text) is the next eval layer and is intentionally kept
  out of the blocking CI gate. Tracked in the README "next steps".
