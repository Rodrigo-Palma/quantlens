"""Measure LLM call latency on an idle local Ollama server.

Cassette latencies are provenance, not a benchmark: they were taken while other
processes could share the GPU. This module times a warm model on production
prompts (with retrieval), one call at a time, and writes ``data/latency.json``,
which ``scripts/benchmark.py`` reports.

Usage: ``python -m quantlens.evals.latency --model qwen3:32b --n 30``
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import statistics
import time
from collections.abc import Callable
from pathlib import Path

from quantlens import llm
from quantlens.config import settings
from quantlens.evals.faithfulness import build_cases
from quantlens.evals.record import _ollama_meta, hardware

LATENCY_FILE = Path(__file__).parent / "data" / "latency.json"


def nearest_rank(samples: list[float], q: float) -> float:
    ordered = sorted(samples)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def measure(
    model: str, n: int, generate: Callable[[str, str], str | None] | None = None
) -> dict[str, object]:
    """Warm the model with one call, then time ``n`` production prompts."""
    call = generate or (lambda prompt, name: llm.generate(prompt, model=name))
    cases = build_cases()
    step = max(1, len(cases) // n)
    prompts = [case.prompt() for case in cases[::step][:n]]
    call(prompts[0], model)
    seconds = []
    for prompt in prompts:
        start = time.perf_counter()
        call(prompt, model)
        seconds.append(time.perf_counter() - start)
    version, digest = _ollama_meta(model)
    return {
        "n": len(seconds),
        "p50_s": round(statistics.median(seconds), 2),
        "p95_s": round(nearest_rank(seconds, 0.95), 2),
        "max_s": round(max(seconds), 2),
        "hardware": hardware(),
        "model_digest": digest,
        "ollama_version": version,
        "measured_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
    }


def load() -> dict[str, dict[str, object]]:
    if not LATENCY_FILE.exists():
        return {}
    data: dict[str, dict[str, object]] = json.loads(LATENCY_FILE.read_text(encoding="utf-8"))
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", required=True)
    parser.add_argument("--n", type=int, default=30)
    args = parser.parse_args()
    if settings.llm_provider != "ollama":
        raise SystemExit("latency measurement requires LLM_PROVIDER=ollama")
    results = load()
    for model in args.model:
        results[model] = measure(model, args.n)
        print(model, results[model], flush=True)
    LATENCY_FILE.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":  # pragma: no cover
    main()
