# ADR 0001: BM25 lexical retrieval, queried by signal regime

**Status:** Accepted (revised 2026-10 for v0.6) · **Date:** 2026-06

## Context

The explanation step needs grounding context (what an RSI above 70 means, what
a 45% volatility implies) so the LLM explains terms accurately instead of
inventing them. The knowledge base is a curated glossary of 16 sections and the
project is local-first: it runs offline, with no API key and no service to
stand up.

Up to v0.5 the API queried the index with the constant
`"RSI momentum volatility {ticker}"`. The ticker never matched the glossary, so
every request got the same two chunks: retrieval that could not change the
prompt. That was a claim the code did not deliver.

## Decision

- Use **BM25** (`rank-bm25`) over chunks split by `##` heading, tokenized into
  lowercase alphanumeric words (v0.5 split on whitespace, so `"(RSI)"` and
  `"rsi"` never matched).
- The glossary has one section per regime: overbought, neutral and oversold RSI,
  uptrend and downtrend, low, moderate and high volatility.
- The API maps the signals to a regime (`quant/regime.py`, thresholds 30/70 and
  20%/40%) and sends **one query per signal**, keeping the top section of each.

## Measured (`make evals`)

Held-out free-text set (added in v0.7): 60 questions from `gemma4:31b-it-qat`,
which saw only the section titles, labels reviewed by hand, measured once with
the glossary and tokenizer frozen. 95% Wilson intervals.

| System | hit@1 | hit@2 | MRR |
|---|---:|---:|---:|
| BM25, word tokens (v0.6) | 34/60 = 56.7% [44.1%, 68.4%] | 65.0% | 0.675 |
| BM25, whitespace tokens (v0.5) | 28/60 = 46.7% [34.6%, 59.1%] | 56.7% | 0.602 |
| Fixed query (v0.5 API) | 3/60 = 5.0% [1.7%, 13.7%] | 8.3% | 0.189 |
| Random ranking (expected) | 6.2% | 12.5% | 0.211 |

Dev free-text set: 32 hand-labeled questions written in the same commit as the
glossary rewrite and the word tokenizer, so in-sample.

| System | hit@1 | hit@2 | MRR |
|---|---:|---:|---:|
| BM25, word tokens (v0.6) | 26/32 = 81.2% [64.7%, 91.1%] | 81.2% | 0.858 |
| BM25, whitespace tokens (v0.5) | 21/32 = 65.6% [48.3%, 79.6%] | 75.0% | 0.757 |
| Fixed query (v0.5 API) | 2/32 = 6.2% [1.7%, 20.1%] | 12.5% | 0.219 |
| Random ranking (expected) | 6.2% | 12.5% | 0.211 |

Regime set: all 18 regimes the API can produce, 3 relevant sections each
(exhaustive, so no interval). recall@3:

| System | recall@3 |
|---|---:|
| BM25, one query per signal (v0.6 API) | 54/54 = 100% |
| BM25, one joint query, word tokens | 46/54 = 85.2% |
| BM25, one joint query, whitespace tokens | 48/54 = 88.9% |
| Fixed query (v0.5 API) | 6/54 = 11.1% |
| Random ranking (expected) | 18.8% |

What these numbers say and do not say:

- BM25 clearly beats the v0.5 fixed query and random (the intervals do not
  overlap). The v0.5 API did no better than random at picking a definition.
- The in-sample number overstated quality: 81.2% hit@1 on the dev set, 56.7%
  on questions from another author.
- Word tokens vs whitespace tokens, exact McNemar on the paired hit@1: 5 vs 0
  discordant on the dev set (p = 0.06), 7 vs 1 on the held-out set (p = 0.07).
  Consistent in direction, not a demonstrated difference on either.
- In a joint query the neutral-RSI section (which mentions 30, 70, overbought
  and oversold) crowds out the volatility section 8 times in 18. Per-signal
  queries fix that by construction.
- The regime 100% is close to a lookup: the queries are written in the
  glossary's own words, by the same author. It proves the wiring, not
  retrieval quality. The held-out free-text set is the quality number.
- The misses are paraphrases with no shared words ("stretched after a big
  rally", "too cheap based on its relative strength") and numeric questions
  ("volatility of 55% a year"). Lexical retrieval cannot answer those.

## Alternatives considered

- **Sentence embeddings + vector DB (pgvector / Qdrant).** Would likely recover
  the paraphrase misses, but adds a model download, a service or a native
  dependency, and cold-start cost. The API never sends free text, only regime
  queries, where BM25 already returns the right sections.
- **Direct lookup by regime, no retrieval.** Equivalent for the API path today
  and simpler. Retrieval stays because adding a glossary section needs no code
  change, and because the free-text number above is what a question-answering
  endpoint would start from. If neither materializes, a lookup table is the
  honest replacement.

## Consequences

- Different regimes ground the prompt with different definitions (tested).
- Fully offline, deterministic, microsecond-latency retrieval.
- Revisit if the knowledge base grows past a few hundred sections or the API
  starts accepting free-text questions: that is where the paraphrase misses
  would matter.
