"""Ask a different model to plant one factual error in correct explanations.

The mutations in ``checker_audit`` are errors I designed together with the v0.7
checker, so their recall is in-sample. This script gets errors from another
author: a local LLM that sees a correct recorded explanation and the true
signals, never the checker, and rewrites the text with exactly one error of its
choosing. Every output was read by hand before the checkers were run on it; a
rewrite whose "error" is not actually wrong (or that changed nothing) is
dropped, and the drops are listed in ``data/README.md``.

Two batches, because left to choose the generator planted only label errors:

* ``--kind any`` (64 rewrites, seed 900): the error type is up to the model.
* ``--kind numeric`` (32 rewrites, seed 1900): a wrong figure, a wrong sign or
  a wrong trend direction, the error types the v0.7 rewrite targeted.

Run (needs Ollama):
    uv run python scripts/generate_checker_errors.py --kind any
    uv run python scripts/generate_checker_errors.py --kind numeric
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import httpx

from quantlens.evals import cassette, checks
from quantlens.evals.cases import build_cases

_DATA = Path(__file__).parents[1] / "src/quantlens/evals/data"
_KINDS = {
    "any": (900, 64, "Choose the kind of error yourself."),
    "numeric": (
        1900,
        32,
        "The error must be one of: a wrong momentum or volatility figure, the momentum "
        "written with the wrong sign, or the trend direction stated the wrong way "
        "(for example rising when it fell). Do not change the RSI or volatility labels.",
    ),
}
_SYSTEMS = ("qwen3:32b", "qwen3:8b")
_SCHEMA = {
    "type": "object",
    "properties": {"text": {"type": "string"}, "error": {"type": "string"}},
    "required": ["text", "error"],
}


def _ask(base: str, model: str, prompt: str, seed: int) -> dict[str, str]:
    response = httpx.post(
        f"{base}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "think": False,
            "format": _SCHEMA,
            "options": {"temperature": 0.9, "seed": seed},
        },
        timeout=600,
    )
    response.raise_for_status()
    answer = json.loads(response.json()["response"])
    return {"text": str(answer["text"]).strip(), "error": str(answer["error"]).strip()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="gemma4:31b-it-qat")
    parser.add_argument("--base-url", default="http://localhost:11434")
    parser.add_argument("--kind", choices=sorted(_KINDS), default="any")
    args = parser.parse_args()
    seed, n, instruction = _KINDS[args.kind]
    by_id = {c.case_id: c for c in build_cases()}
    recorded = cassette.load_all()
    pool = [
        (system, record.text, by_id[record.case_id])
        for system in _SYSTEMS
        for record in recorded[system]
        if record.text and checks.score(record.text, by_id[record.case_id]).passed
    ]
    rows = []
    for i, (system, text, case) in enumerate(random.Random(seed).sample(pool, n)):
        prompt = (
            "These are the true signals of a Brazilian stock:\n"
            f"Ticker: {case.ticker}\nMomentum (20 sessions): {case.mom:.2%}\n"
            f"RSI(14): {case.rsi:.0f}\nAnnualized volatility: {case.vol:.2%}\n"
            "Conventions: RSI above 70 is overbought, below 30 oversold, otherwise "
            "neutral; volatility below 20% is low, 20% to 40% moderate, above 40% high.\n\n"
            f"This explanation is accurate:\n{text}\n\n"
            "Rewrite it so that it contains exactly ONE factual error that contradicts "
            f"the signals above. {instruction} Keep everything else "
            "as close to the original as possible and keep it natural. Answer as JSON "
            '{"text": "<rewritten explanation>", "error": "<what you changed>"}.'
        )
        answer = _ask(args.base_url, args.model, prompt, seed=seed + i)
        row = {"case_id": case.case_id, "source": system, "kind": args.kind, "original": text}
        rows.append({**row, **answer})
        print(f"{i + 1}/{n} {case.case_id}: {answer['error']}", flush=True)
    out = _DATA / f"checker_errors.{args.kind}.raw.jsonl"
    out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", "utf-8")
    print(f"wrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    main()
