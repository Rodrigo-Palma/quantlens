"""Re-record the LLM cassettes against a local Ollama server.

Usage: ``python -m quantlens.evals.record --model qwen3:32b --model qwen3:8b``
(``make evals-record``). Calls run one at a time so each latency is the cost of a
single request on an otherwise idle server.

``--missing-only`` keeps every recorded case whose prompt still matches and only
records the rest (new cases or stale prompts). It refuses to mix generations from
a different model digest into one cassette.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import platform
import subprocess
import time
from pathlib import Path

import httpx

from quantlens import llm
from quantlens.config import settings
from quantlens.evals import cassette
from quantlens.evals.faithfulness import build_cases

DEFAULT_MODELS = ("qwen3:32b", "qwen3:8b")


def hardware() -> str:
    if platform.system() == "Darwin":
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.stdout.strip():
            return result.stdout.strip()
    return f"{platform.machine()} {platform.system()}"


def _ollama_meta(model: str) -> tuple[str, str]:
    base = settings.llm_base_url
    version = str(httpx.get(f"{base}/api/version", timeout=10).json()["version"])
    tags = httpx.get(f"{base}/api/tags", timeout=10).json()["models"]
    digest = next((str(t["digest"]) for t in tags if t["name"] == model), "unknown")
    return version, digest


def reusable(path: Path, with_context: bool, digest: str) -> dict[str, cassette.Record]:
    """Recorded cases in ``path`` whose prompt still matches, keyed by case id."""
    if not path.is_file():
        return {}
    lines = path.read_text(encoding="utf-8").splitlines()
    existing = [cassette.Record(**json.loads(line)) for line in lines if line.strip()]
    digests = {r.model_digest for r in existing}
    if digests - {digest}:
        raise SystemExit(f"{path.name}: recorded with {sorted(digests)}, server has {digest}")
    prompts = {c.case_id: cassette.prompt_hash(c.prompt(with_context)) for c in build_cases()}
    return {
        r.case_id: r
        for r in existing
        if r.text is not None and prompts.get(r.case_id) == r.prompt_sha256
    }


def record(
    model: str, with_context: bool = True, keep: dict[str, cassette.Record] | None = None
) -> list[cassette.Record]:
    version, digest = _ollama_meta(model)
    machine = hardware()
    variant = cassette.WITH_RETRIEVAL if with_context else cassette.WITHOUT_RETRIEVAL
    keep = keep or {}
    records = []
    for case in build_cases():
        if case.case_id in keep:
            records.append(keep[case.case_id])
            continue
        prompt = case.prompt(with_context)
        start = time.perf_counter()
        text = llm.generate(prompt, model=model)
        elapsed_ms = (time.perf_counter() - start) * 1000
        records.append(
            cassette.Record(
                case_id=case.case_id,
                model=model,
                text=text,
                latency_ms=round(elapsed_ms, 1),
                prompt_sha256=cassette.prompt_hash(prompt),
                model_digest=digest,
                ollama_version=version,
                hardware=machine,
                recorded_at=dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
                variant=variant,
            )
        )
        print(f"{model} {case.case_id} {elapsed_ms / 1000:.1f}s", flush=True)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", help="Ollama model (repeatable)")
    parser.add_argument(
        "--no-context", action="store_true", help="ablation: send the prompt without retrieval"
    )
    parser.add_argument(
        "--missing-only", action="store_true", help="only record cases not already recorded"
    )
    args = parser.parse_args()
    if settings.llm_provider != "ollama":
        raise SystemExit("recording requires LLM_PROVIDER=ollama")
    variant = cassette.WITHOUT_RETRIEVAL if args.no_context else cassette.WITH_RETRIEVAL
    for model in args.model or DEFAULT_MODELS:
        keep = {}
        if args.missing_only:
            path = cassette.source_dir() / f"{cassette.slug(model, variant)}.jsonl"
            keep = reusable(path, not args.no_context, _ollama_meta(model)[1])
        records = record(model, with_context=not args.no_context, keep=keep)
        path = cassette.write(model, records, variant=variant)
        print(f"wrote {path}")


if __name__ == "__main__":  # pragma: no cover
    main()
