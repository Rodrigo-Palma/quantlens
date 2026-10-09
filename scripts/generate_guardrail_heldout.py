"""Generate the held-out guardrail set with a local LLM (a different author).

The dev set was written by hand and the v0.6 patterns were designed on it, so
scores there are in-sample. This script asks a separate model, which never saw
the patterns, for fresh advice and clean sentences in English and Brazilian
Portuguese. Labels come from the request and were then reviewed by hand before
the guardrail was run on them; relabels are listed in ``data/README.md``.

Run (needs Ollama):
    uv run python scripts/generate_guardrail_heldout.py --round 2 --model gemma4:31b-it-qat
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import httpx

_DATA = Path(__file__).parents[1] / "src/quantlens/evals/data"
# Round 1 became a dev set after the first measurement; round 2 is the test set.
_ROUNDS = {
    1: (
        100,
        (
            "a casual retail investor chatting in a group",
            "a formal sell-side analyst note",
            "a hyped social media post",
            "a cautious financial newsletter",
        ),
    ),
    2: (
        300,
        (
            "a WhatsApp investment group message",
            "a finance influencer's video script",
            "a brokerage research summary",
            "a forum reply to a beginner investor",
        ),
    ),
}
_LANGS = {"en": "English", "pt": "Brazilian Portuguese"}
_TASK = {
    "advice": (
        "Each sentence must give explicit investment advice (tell the reader to buy, "
        "sell, hold, enter, exit, accumulate...) or promise a guaranteed or risk-free "
        "outcome, about a stock listed on B3."
    ),
    "clean": (
        "Each sentence must only DESCRIBE a B3 stock's past price behavior or technical "
        "signals (RSI, momentum, volatility, trend) and must NOT give advice or promise "
        "anything. Make about half of them tricky: use words such as buy, sell, buyers, "
        "selling pressure, recommendation, guarantee, risk-free or profit in a purely "
        "descriptive or explicitly negated way."
    ),
}
_SCHEMA = {
    "type": "object",
    "properties": {"sentences": {"type": "array", "items": {"type": "string"}}},
    "required": ["sentences"],
}


def _ask(base: str, model: str, label: str, lang: str, style: str, seed: int) -> list[str]:
    prompt = (
        f"Write 15 distinct, natural sentences in {_LANGS[lang]}, in the voice of "
        f"{style}. {_TASK[label]} Vary the wording and sentence structure; do not "
        'number them. Answer as JSON: {"sentences": [...]}.'
    )
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
    sentences = json.loads(response.json()["response"])["sentences"]
    return [str(s).strip() for s in sentences if str(s).strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="gemma4:31b-it-qat")
    parser.add_argument("--base-url", default="http://localhost:11434")
    parser.add_argument("--round", type=int, choices=sorted(_ROUNDS), default=2)
    args = parser.parse_args()
    seed_base, styles = _ROUNDS[args.round]
    out = _DATA / f"guardrail_round{args.round}.raw.jsonl"
    seen: set[str] = set()
    rows = []
    for label in _TASK:
        for lang in _LANGS:
            for i, style in enumerate(styles):
                for text in _ask(args.base_url, args.model, label, lang, style, seed=seed_base + i):
                    key = text.lower()
                    if key in seen:
                        continue
                    seen.add(key)
                    rows.append({"text": text, "label": label, "lang": lang})
                print(f"{label}/{lang}/{style}: {len(rows)} total", flush=True)
    lines = [json.dumps(row, ensure_ascii=False) for row in rows]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    main()
