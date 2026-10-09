# ADR 0005: Faithfulness eval of real LLM output, scored offline from cassettes

**Status:** Accepted · **Date:** 2026-10

## Context

Up to v0.5 the only eval checked the rule-based explainer against its own
template ([ADR 0004](0004-offline-deterministic-evals.md)). The LLM output,
the part of the system that can say "oversold" at RSI 75 or invent a
volatility figure, was never scored. Two constraints shape any fix: CI has no
GPU and no Ollama, and a number without an interval or a baseline does not
show that anything changed.

## Decision

**Cases.** A seeded grid (`faithfulness.build_cases`, seed 2026): 3 RSI regimes
x 2 trend directions x 2 volatility levels, 10 cases per cell, n = 120. Values
are drawn away from the 30/70 and 20%/40% thresholds so the right label is
never ambiguous. Some cells pair signals that rarely co-occur (RSI 85 with
negative momentum) on purpose: they show whether the model reads the numbers.
Each case is sent with the exact prompt the API builds, including the
regime-retrieved context.

**Checks, all decidable by code.** Four were fixed before any output was read;
the fifth (`vol_label`) was added after reading the qwen3:8b cassette and before
the qwen3:32b cassette existed.

| Check | Passes when |
|---|---|
| `numbers` | every number in the text is the RSI (within 1), the absolute momentum or the volatility in percent (within 0.51 points), or a window/threshold constant |
| `rsi_label` | the text does not call the stock overbought at RSI at or below 70, or oversold at or above 30 (hedged uses such as "approaching overbought" are allowed) |
| `vol_label` | a "low/moderate/high volatility" or "... price swings" label matches the 20%/40% regime |
| `trend` | the text states the right direction and does not assert the opposite trend term |
| `guardrail` | `guardrails.validate` passes |

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

**Gate.** CI fails if a model's pass rate drops below the Wilson lower bound
recorded in `data/gates.json`, if the baseline drops below its own, or if any
cassette is stale.

## Measured (`make evals`)

Recorded 2026-10-09 on an Apple M3 Max, Ollama 0.35.1, greedy decoding.
95% Wilson intervals.

| System | Pass | numbers | rsi_label | vol_label | trend | guardrail |
|---|---:|---:|---:|---:|---:|---:|
| rule-based (baseline) | 120/120 = 100% [96.9%, 100%] | 100% | 100% | 100% | 100% | 100% |
| qwen3:32b | 120/120 = 100% [96.9%, 100%] | 100% | 100% | 100% | 100% | 100% |
| qwen3:32b, no retrieval | 101/120 = 84.2% [76.6%, 89.6%] | 100% | 85.8% | 98.3% | 100% | 100% |
| qwen3:8b | 115/120 = 95.8% [90.6%, 98.2%] | 100% | 100% | 95.8% | 100% | 100% |
| qwen3:8b, no retrieval | 107/120 = 89.2% [82.3%, 93.6%] | 100% | 89.2% | 99.2% | 100% | 100% |

Paired comparisons (exact McNemar on the discordant cases):

| A vs B | only A passes | only B passes | p |
|---|---:|---:|---:|
| qwen3:32b vs qwen3:32b, no retrieval | 19 | 0 | 3.8e-6 |
| qwen3:32b vs qwen3:8b | 5 | 0 | 0.0625 |
| qwen3:8b vs qwen3:8b, no retrieval | 13 | 5 | 0.096 |
| qwen3:32b vs rule-based | 0 | 0 | 1 |

Minimum detectable difference at n = 120 around a 95% pass rate (two-sided
5%, power 80%): 7.9 points.

Reading:

- Retrieval matters for the default model: without the glossary in context,
  qwen3:32b uses its own RSI bands (36 to 40 called oversold, 60 to 64
  overbought). With it, no label errors. This measures adherence to the 30/70
  convention the system defines; a prompt that states the thresholds directly
  was not tested and might do as well.
- For qwen3:8b retrieval removes all 13 RSI label errors but 5 volatility label
  errors appear ("volatility of 17.26% reflects moderate price swings"), so the
  overall difference is not significant (p = 0.096).
- No model invented a number, got the trend direction wrong, or tripped the
  guardrail in 480 outputs.
- The LLM ties the template baseline: these checks cannot show that it is more
  faithful than a template, only that it is not less faithful.
- Audit trail: one qwen3:32b failure was a checker false positive (negation
  four words before "oversold"); the hedge window went from 3 to 5 words for
  every system. Both checker changes are separate commits.

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
  baseline, reproducible offline in about a second.
- The cassette is a snapshot: a new model, prompt or Ollama version needs a
  re-record, and the gate makes that explicit instead of silently scoring stale
  text.
- The baseline passes every check by construction (its text is built from the
  same numbers). It is the ceiling for these four checks, not a competitor on
  readability, which this eval does not measure.
