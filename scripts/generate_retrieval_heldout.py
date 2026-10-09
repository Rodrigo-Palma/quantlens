"""Generate the held-out retrieval questions with a local LLM (a different author).

The 32 dev questions, the glossary rewrite and the word tokenizer were made
together, so hit@1 on them is in-sample. This script asks a separate model for
fresh questions. It sees only the section titles, never the glossary text, the
dev questions or the tokenizer. Labels come from the request and were reviewed
by hand before retrieval was run on them; drops and relabels are listed in
``data/README.md``.

Run (needs Ollama):
    uv run python scripts/generate_retrieval_heldout.py --model gemma4:31b-it-qat
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import httpx

from quantlens.rag import load_chunks

_DATA = Path(__file__).parents[1] / "src/quantlens/evals/data"
_SEED = 700
_PER_SECTION = 4
_SCHEMA = {
    "type": "object",
    "properties": {"questions": {"type": "array", "items": {"type": "string"}}},
    "required": ["questions"],
}


def _ask(base: str, model: str, title: str, others: list[str], seed: int) -> list[str]:
    prompt = (
        "A retail investor is reading a short glossary about Brazilian (B3) stocks. "
        f"Write {_PER_SECTION} distinct questions or search phrases they might type "
        f'that are answered by the glossary entry titled "{title}" and by none of '
        f"these other entries: {'; '.join(others)}. Mix styles: a casual question, a "
        "paraphrase that avoids the words in the title, a question with a concrete "
        "number, and a short search phrase. English only. Answer as JSON: "
        '{"questions": [...]}.'
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
    questions = json.loads(response.json()["response"])["questions"]
    return [str(q).strip() for q in questions if str(q).strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="gemma4:31b-it-qat")
    parser.add_argument("--base-url", default="http://localhost:11434")
    args = parser.parse_args()
    titles = [chunk.title for chunk in load_chunks()]
    rows = []
    for i, title in enumerate(titles):
        others = [t for t in titles if t != title]
        for question in _ask(args.base_url, args.model, title, others, seed=_SEED + i):
            rows.append({"query": question, "expected": title})
        print(f"{title}: {len(rows)} total", flush=True)
    out = _DATA / "retrieval_heldout.raw.jsonl"
    out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", "utf-8")
    print(f"wrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    main()
