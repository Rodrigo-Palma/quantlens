# ADR 0003: Rule-based output guardrails and deterministic fallback

**Status:** Accepted (revised 2026-10 for v0.6) · **Date:** 2026-06

## Context

This tool describes equity signals; it must **never emit investment advice or
guarantees**. LLM output is non-deterministic, so even a well-prompted model can
drift into advice phrasing. The product is for B3 and the default model
(`qwen3`) can answer in Portuguese. A guardrail is required on the output, and
it has to be enforceable in CI with no network.

The v0.5 guardrail was an English substring list. Its measured quality was
"4/4 recall, 0/3 false positives" on 7 probes: the 95% Wilson interval of 4/4
is [51%, 100%], so the number said nothing.

## Decision

Validate every generated explanation against **word-bounded regex rules** over
lowercased, accent-stripped text, in English and Brazilian Portuguese
(`guardrails.validate`). Rules are grouped as directives, imperatives,
recommendations and guarantees. Guarantee rules are skipped when a negation
appears in the 4 words before the match in the same clause ("returns are not
guaranteed"); directive rules never are ("you should not buy" is advice). If
the LLM output trips any rule, the API serves the rule-based explainer and
returns the violations in `guardrail_violations`.

## Measured (`make evals`)

Three labeled sets, built and used in order so the last one is out-of-sample
(provenance in `src/quantlens/evals/data/README.md`). 95% Wilson intervals.

| Set | Role for v0.6 | v0.5 recall | v0.6 recall | v0.5 FP | v0.6 FP |
|---|---|---:|---:|---:|---:|
| dev (hand-written, 110 + 110) | designed on | 15/110 = 13.6% | 110/110 | 7/110 = 6.4% | 0/110 |
| round 1 (gemma4, 120 + 119) | revised on | 27/120 = 22.5% | 119/120 | 6/119 = 5.0% | 0/119 |
| **round 2 (gemma4, 106 + 120)** | **test** | **15/106 = 14.2% [8.8%, 22.0%]** | **87/106 = 82.1% [73.7%, 88.2%]** | 6/120 = 5.0% | **2/120 = 1.7% [0.5%, 5.9%]** |

Round 2 by language: English recall 44/46 = 95.7% [85.5%, 98.8%], Portuguese
43/60 = 71.7% [59.2%, 81.5%]. v0.5 caught 0 Portuguese sentences in every set.

What this says:

- On data the patterns never saw, recall rose from 14% to 82% and the false
  positive rate did not rise (the intervals for recall do not overlap).
- In-sample numbers overstate the filter. The first frozen patterns scored
  110/110 on the dev set and 67/120 = 55.8% on round 1, the first set they had
  not seen: a 44 point drop. Only round 2 is quoted as the result.
- Portuguese is the weak side. 15 of the 19 round-2 misses use the plural
  imperative ("saiam", "vendam", "mantenham", "comprem"), a conjugation absent
  from both dev sets. It is a known gap; fixing it requires a round 3 to
  measure, not a change fitted on round 2.
- The two false positives negate with words outside the negation list ("far
  from being risk-free", "não indica que ... seja isento de risco").

## Alternatives considered

- **LLM-as-judge guardrail.** Catches paraphrased advice the rules miss, but it
  is non-deterministic, adds latency and cost, can itself hallucinate a verdict,
  and cannot run offline in CI. Wrong tool for a hard safety invariant; a
  reasonable advisory layer on top.
- **A small trained classifier.** Likely better recall on paraphrase. Needs a
  few thousand labeled sentences to beat these rules with confidence; the eval
  sets here (685 sentences) would be its test data, not its training data.
- **No guardrail, prompt-only.** Relies entirely on the model behaving.

## Consequences

- The safety check is deterministic, offline and testable. It costs about
  60 µs per explanation (was 1 µs for the substring list), negligible next to a
  multi-second LLM call.
- A guardrail trip degrades to a safe explanation rather than failing the
  request, and the violation is visible in the response and the request log.
- Roughly 1 in 5 advice sentences of an unseen style still passes (round 2).
  The prompt also forbids advice, so the guardrail is a second line, not the
  only one. In the 480 recorded LLM explanations
  ([ADR 0005](0005-faithfulness-eval-with-cassettes.md)) the guardrail fired 0
  times: with this prompt, neither model produced advice the rules recognize.
- CI fails if recall on any set drops below its recorded Wilson lower bound or
  the false-positive rate rises above its upper bound.
