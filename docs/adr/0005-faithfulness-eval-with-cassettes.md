# ADR 0005: Faithfulness eval of real LLM output, scored offline from cassettes

**Status:** Accepted (revised for v0.7) · **Date:** 2026-10

## Context

Up to v0.5 the only eval checked the rule-based explainer against its own
template ([ADR 0004](0004-offline-deterministic-evals.md)). The LLM output,
the part of the system that can say "oversold" at RSI 75 or invent a
volatility figure, was never scored. Two constraints shape any fix: CI has no
GPU and no Ollama, and a number without an interval or a baseline does not
show that anything changed.

## Decision

**Cases.** A seeded grid (`cases.build_cases`): 3 RSI regimes x 2 trend
directions x 3 volatility regimes, 10 cases per cell, n = 180. v0.6 had only
the low and high volatility cells (n = 120, seed 2026) and so never scored the
moderate regime (20% to 40%), the most common one for B3 large caps. v0.7 adds
the 60 moderate cases (24% to 36%) from a separate stream (seed 2027), so the
original 120 keep their exact values and recordings. Values are drawn away from
the 30/70 and 20%/40% thresholds so the right label is never ambiguous. Some cells pair signals that rarely co-occur (RSI 85 with
negative momentum) on purpose: they show whether the model reads the numbers.
Each case is sent with the exact prompt the API builds, including the
regime-retrieved context.

**Checks, all decidable by code** (`checks.py`, v0.7).

| Check | Passes when |
|---|---|
| `numbers` | every number is the value of the signal named next to it in the same sentence (RSI within 1, percentages within 0.51 points) with the right sign when one is written, or a constant in a form that cannot be read as a value: a threshold after a comparison ("below 20%", "RSI > 70"), a range ("30 to 70") or a window ("RSI(14)", "20 sessions") |
| `rsi_label` | no unhedged "overbought" at RSI at or below 70, "oversold" at or above 30, or "neutral" outside 30 to 70; the hedge must be in the same clause |
| `vol_label` | a "low/moderate/high/elevated volatility", "... price swings" or "... price fluctuations" label matches the 20%/40% regime |
| `trend` | the text states the right direction and asserts no wrong one: neither a trend term ("uptrend", "positive momentum") nor a direction verb ("rose", "rallied", "fell") in a sentence about the momentum figure |
| `guardrail` | `guardrails.validate` passes |

**Why v0.7 rewrote the checker.** A review showed the v0.6 checks passing
text that was plainly wrong: "ABEV3 rose 13.05%... a strong rally" when it
fell 13.05%; momentum and volatility figures swapped; "momentum of 30% and
volatility of 40%" for a stock at -13% and 17%. `numbers` compared absolute
values against any signal, the 20/30/40/70 constants passed anywhere, and any
loose direction word ("losses", "up") satisfied `trend`. The v0.6 checker is
kept verbatim in `checks_v06.py` as the baseline of the audit below.

**The checker is measured** (`checker_audit.py`). Two sources of known errors:

- Mutations of every correct recorded output, one error at a time: sign flip,
  swapped figures, invented value (+9 points), invented constant (30% or
  40%), inverted direction terms, a wrong direction verb with no trend term,
  swapped RSI label, swapped volatility label. Designed together with the v0.7
  checker, so in-sample.
- Errors planted by `gemma4:31b-it-qat` in 96 correct outputs of qwen3:32b and
  qwen3:8b (`scripts/generate_checker_errors.py`): it saw the true signals and
  the conventions and never saw the checker; in 64 it chose the error, in 32 it
  was asked for a figure, sign or direction error. Each rewrite was read by
  hand before scoring and measured once. This is the out-of-sample recall.

A case passes only if all five pass. The checks are a floor: they catch
contradictions and invented numbers, not unhelpful or clumsy prose.

**Systems on the same cases.** The rule-based explainer (the production
fallback) as the baseline, `qwen3:32b` (production default) and `qwen3:8b`
(a smaller local model), each also recorded **without the retrieved context**
(`--no-context`) to measure what retrieval contributes. Decoding is greedy and
seeded, reasoning off ([ADR 0002](0002-local-first-llm.md)).

**Cassettes instead of live calls in CI.** `make evals-record` sends every case
to a local Ollama server once and writes one JSONL per model with the text, the
call latency, the prompt hash, the model digest, the Ollama version and the
hardware. CI scores the cassettes offline. If the prompt or the retrieval
changes, the prompt hashes stop matching, the report flags the cassette as
stale and the gate fails until it is re-recorded.

**Statistics.** Pass rates with 95% Wilson intervals; the minimum detectable
effect at this n (two-sided 5%, power 80%); systems compared with an exact
McNemar test on the paired cases, since every system sees the same 120.

**Latency is measured separately.** Cassette latencies were taken while other
processes shared the GPU (qwen3:8b looked slower than qwen3:32b), so they are
provenance only. `python -m quantlens.evals.latency` times a warm model on
production prompts on an idle server.

**Gate.** CI fails if any recorded count gets worse: cases passed per system,
mutations and planted errors caught, or stale prompts
([ADR 0006](0006-exact-count-regression-gates.md)).

## Measured (`make evals`)

Recorded 2026-10-09 on an Apple M3 Max, Ollama 0.35.1, greedy decoding (the
moderate cells the same day, same model digests). 95% Wilson intervals.

| System | Pass | numbers | rsi_label | vol_label | trend | guardrail |
|---|---|---:|---:|---:|---:|---:|
| rule-based (baseline) | 180/180 = 100% [97.9%, 100%] | 100% | 100% | 100% | 100% | 100% |
| qwen3:32b | 180/180 = 100% [97.9%, 100%] | 100% | 100% | 100% | 100% | 100% |
| qwen3:32b, no retrieval | 84/180 = 46.7% [39.5%, 53.9%] | 100% | 83.9% | 58.3% | 100% | 100% |
| qwen3:8b | 175/180 = 97.2% [93.7%, 98.8%] | 100% | 100% | 97.2% | 100% | 100% |
| qwen3:8b, no retrieval | 103/180 = 57.2% [49.9%, 64.2%] | 100% | 88.3% | 65.6% | 100% | 100% |

Paired comparisons (exact McNemar on the discordant cases):

| A vs B | only A passes | only B passes | p |
|---|---:|---:|---:|
| qwen3:32b vs qwen3:32b, no retrieval | 96 | 0 | 2.5e-29 |
| qwen3:32b vs qwen3:8b | 5 | 0 | 0.0625 |
| qwen3:8b vs qwen3:8b, no retrieval | 72 | 0 | 4.2e-22 |
| qwen3:32b vs rule-based | 0 | 0 | 1 |

Minimum detectable difference at n = 180 around a 95% pass rate (two-sided
5%, power 80%): 6.4 points.

Checker recall (share of known-wrong texts it fails):

| Error source | n | v0.6 checker | v0.7 checker |
|---|---:|---:|---:|
| sign flip of the momentum figure | 483 | 0% | 100% |
| momentum and volatility figures swapped | 482 | 0% | 100% |
| invented value (volatility + 9 points) | 507 | 98.4% | 100% |
| invented constant (30% or 40%) | 507 | 0% | 100% |
| direction terms inverted | 542 | 100% | 100% |
| wrong direction verb, no trend term | 483 | 34.0% | 100% |
| RSI label swapped | 537 | 98.0% | 100% |
| volatility label swapped | 367 | 87.7% | 99.2% |
| planted by gemma4, error of its choice | 64 | 24/64 = 37.5% | 59/64 = 92.2% [83.0%, 96.6%] |
| planted by gemma4, figure, sign or direction | 32 | 32/32 = 100% | 32/32 = 100% [89.3%, 100%] |

The mutations run on the 542 distinct correct outputs across the four
cassettes; n differs per row because a mutation applies only where the text has
the thing to mutate. Both planted batches came out narrow: left to choose,
gemma4 changed an RSI or volatility label every time, and asked for a figure,
sign or direction error it swapped "uptrend" and "downtrend" every time. So
the out-of-sample recall covers labels and trend terms; the number and sign
checks are measured only by the in-sample mutations. The 5 planted errors the
v0.7 checker misses are all volatility labels in wordings it does not read
("reflects low risk", "suggests high price movements", "is considered
moderate", "falls into the moderate volatility category").

What changed from v0.6, on the same 120 v0.6 cases: no system's pass count
moved for the retrieval runs (qwen3:32b 120/120, qwen3:8b 115/120). The
no-retrieval runs dropped from 101 to 64 (qwen3:32b) and from 107 to 51
(qwen3:8b), every new failure a volatility label written as "moderate price
fluctuations" at a volatility below 20%, which v0.6 did not read. The stricter
number, sign and direction checks found no new error in any of the 720 outputs.

Reading:

- Retrieval matters for both models. Without the glossary in context they
  apply their own bands: RSI 35 to 43 called oversold, 60 to 64 overbought, and
  volatility under 20% called moderate.
- For qwen3:8b the 5 remaining failures with retrieval are all the same
  phrase, "volatility of 16% to 18% reflects moderate price swings".
- The LLM ties the template baseline: these checks cannot show that it is more
  faithful than a template, only that it is not less faithful.
- Audit trail of checker changes made after reading outputs, each applied to
  every system: v0.6 widened the hedge window from 3 to 5 words; v0.7 accepted
  en-dash ranges, stopped a word after a comma from claiming a figure ("-7.75%,
  and the RSI"), ignored "falls into" as a direction verb, accepted "positive
  trend" / "negative trend", and treated "between overbought and oversold" as a
  hedge. All five had failed correct text.

## Alternatives considered

- **Live LLM calls in CI.** Needs a GPU runner or a paid key, is slow, and is
  non-deterministic across runs and hardware. A cassette pins exactly what was
  scored.
- **LLM-as-judge for faithfulness.** Grades prose quality the code checks
  cannot, but adds a second model whose own error rate would need measuring
  first. Numbers, labels and direction are checkable exactly, so code checks
  come first.
- **Sampling with temperature and reporting a mean over seeds.** Closer to
  how some deployments run, but the API decodes greedily, so the eval does too.

## Consequences

- The LLM's faithfulness is a number with an interval, compared with a
  baseline, reproducible offline in a few seconds.
- The cassette is a snapshot: a new model, prompt or Ollama version needs a
  re-record, and the gate makes that explicit instead of silently scoring stale
  text.
- The baseline passes every check by construction (its text is built from the
  same numbers). It is the ceiling for these checks, not a competitor on
  readability, which this eval does not measure.
- The v0.7 checker was tuned on the same recorded outputs it scores, and its
  mutations were designed with it. Only the planted-error recall is
  out-of-sample, and it covers the error types one other model chose.
