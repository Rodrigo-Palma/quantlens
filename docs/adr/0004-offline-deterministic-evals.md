# ADR 0004: Offline, deterministic eval harness gated in CI

**Status:** Superseded by [ADR 0005](0005-faithfulness-eval-with-cassettes.md) (2026-10) · **Date:** 2026-06

## Context

"Has evals" is cheap to claim and hard to trust. To be worth anything, the eval
suite has to (a) run on every push, (b) fail the build on a regression, and
(c) not depend on a network call or a non-deterministic LLM.

## Decision (v0.3 to v0.5)

Evaluate the deterministic rule-based explainer over a 4-case `EVAL_SET`,
checking that the text mentions RSI, volatility and the right trend word, and
that the guardrail passes. Exit non-zero below 100%.

## Why it was superseded

The checks were written against the output format of the same rule-based
explainer they evaluated, so 4/4 was guaranteed by construction: the eval
could only fail if someone edited the template. With n = 4 the 95% Wilson
interval of 4/4 is [51%, 100%], and it never looked at LLM output, which is the
part of the system that can actually be wrong.

What survives in v0.6:

- The requirement: offline, deterministic, blocking in CI.
- The check itself, demoted to a unit test of the explainer
  (`tests/test_explain.py`).
- The rule-based explainer, now the **baseline** that the LLM is compared
  against in ADR 0005, on the same cases and the same checks.
