# Eval data

All files are small, synthetic and contain no personal data.

## `retrieval_queries.jsonl`

32 questions, each labeled with the one glossary section that answers it.
Written by hand to mix lexical overlap ("RSI around 50") with paraphrase ("the
stock looks stretched after a big rally") and numeric questions ("volatility of
55% a year").

## Guardrail sets

Each line is `{"text", "label": "advice" | "clean", "lang": "en" | "pt"}`.
"advice" means the sentence tells the reader to act (buy, sell, hold, enter,
exit...) or promises an outcome; "clean" sentences only describe prices or
signals, and many are deliberate near misses ("selling pressure", "not a buy
recommendation", "the risk-free rate").

| File | Author | Size | Role |
|---|---|---|---|
| `guardrail_dev.jsonl` | hand-written | 110 advice, 110 clean | designed the v0.6 patterns (in-sample) |
| `guardrail_round1.jsonl` | `gemma4:31b-it-qat`, round 1 | 120 advice, 119 clean | measured once, then used to revise the patterns (in-sample) |
| `guardrail_round2.jsonl` | `gemma4:31b-it-qat`, round 2 | 106 advice, 120 clean | measured once against the frozen patterns (out-of-sample) |

Generation: `scripts/generate_guardrail_heldout.py --round N`. The generator
was asked for advice or descriptive sentences in four styles per round (round 1:
group chat, sell-side note, hyped post, cautious newsletter; round 2: WhatsApp
group, influencer script, brokerage summary, forum reply), 15 sentences per
call, temperature 0.9, fixed seeds. It never saw the patterns.

Label review: every generated sentence was read and its label checked before
the guardrail was run on the set. Round 1: one clean sentence excluded as
ambiguous ("A recomendação técnica baseada exclusivamente no gráfico aponta para
a manutenção da tendência de baixa."). Round 2: no changes. Exact duplicates of
an earlier set are dropped.

Order of events, visible in the git history: dev set and v0.6 patterns
committed, round 1 generated and measured, patterns revised and committed,
round 2 generated and measured. Round 2 has not been used to change a pattern.

## `cassettes/`

Recorded LLM outputs for the faithfulness eval, one JSONL per model. See
`quantlens/evals/cassette.py` and ADR 0005.

## `gates.json`

Regression bounds checked by `python -m quantlens.evals`. `min` is the 95%
Wilson lower limit of the accepted measurement, `max` the upper limit.
