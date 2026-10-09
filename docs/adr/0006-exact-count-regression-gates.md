# ADR 0006: Regression gates on exact counts, not on interval bounds

**Status:** Accepted · **Date:** 2026-10

## Context

Up to v0.6 each gate in `data/gates.json` was the 95% Wilson lower bound of the
accepted measurement. Every input to `make evals` is fixed: the case grid is
seeded, the LLM outputs are recorded cassettes, the guardrail and retrieval sets
are files in the repo and every scorer is code. The same commit always produces
the same counts, so the only thing that can move a metric is a code or data
change. Under the interval bounds that change could be a real regression and CI
stayed green:

| Metric | Accepted | Still passed the v0.6 gate |
|---|---|---|
| qwen3:32b faithfulness | 120/120 | 117/120 |
| guardrail recall, round 2 | 87/106 | 79/106 |
| retrieval hit@1, dev set | 26/32 | 21/32 |

A sampling interval answers "how far might this generalize". It is the wrong
tolerance for "did this commit change the result", which has no sampling noise.

## Decision

- Gates record the accepted **count**: `min` for counts that must not drop
  (cases passed, advice blocked, hits, mutations caught), `max` for counts that
  must not grow (clean text blocked, stale prompts), and `eq` for every
  denominator, so a set cannot shrink to make a rate look better.
- Any worsening fails CI. An improvement passes and is printed as "better than
  recorded".
- `python -m quantlens.evals --update-gates` rewrites `gates.json` from the
  current run. That file is committed on its own, so every change of baseline
  is a visible, reviewable diff.
- Wilson intervals stay in the report and the README, where they describe the
  uncertainty of the generalization.

## Alternatives considered

- **Keep interval bounds.** Rejected for the reason above.
- **Fixed tolerance (for example 1 case).** Still lets a real regression
  through and has no principled value for the tolerance.
- **Exact equality on every count.** Would also fail on improvements, which
  pushes people to update gates without looking. The ratchet fails only on the
  direction that matters and reports the other.

## Consequences

- A deliberate trade (for example a guardrail rule that blocks more advice but
  one more clean sentence) needs a gate update in the same pull request, which
  is the point: the trade is written down.
- Re-recording a cassette changes the counts; the gate update rides along with
  the new cassette.
