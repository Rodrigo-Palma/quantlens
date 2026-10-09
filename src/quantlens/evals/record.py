"""Re-record the LLM cassettes against a local Ollama server.

Usage: ``python -m quantlens.evals.record --model qwen3:32b --model qwen3:8b``
(``make evals-record``). Calls run one at a time so each latency is the cost of a
single request on an otherwise idle server.
"""

from __future__ import annotations

import argparse
import datetime as dt
import platform
import subprocess
import time

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


def record(model: str) -> list[cassette.Record]:
    version, digest = _ollama_meta(model)
    machine = hardware()
    records = []
    for case in build_cases():
        prompt = case.prompt()
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
            )
        )
        print(f"{model} {case.case_id} {elapsed_ms / 1000:.1f}s", flush=True)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", help="Ollama model (repeatable)")
    args = parser.parse_args()
    if settings.llm_provider != "ollama":
        raise SystemExit("recording requires LLM_PROVIDER=ollama")
    for model in args.model or DEFAULT_MODELS:
        path = cassette.write(model, record(model))
        print(f"wrote {path}")


if __name__ == "__main__":  # pragma: no cover
    main()
