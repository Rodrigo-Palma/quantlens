"""Recorded LLM outputs ("cassettes") so CI scores real generations offline.

One JSONL file per model in ``data/cassettes``. Each line is one eval case: the
generated text (``null`` when the call failed), the wall-clock latency of the
call, and enough provenance (model digest, Ollama version, hardware, prompt hash)
to tell whether a re-recording is comparable.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from importlib import resources
from pathlib import Path

CASSETTE_DIR = "data/cassettes"


@dataclass(frozen=True)
class Record:
    case_id: str
    model: str
    text: str | None
    latency_ms: float
    prompt_sha256: str
    model_digest: str
    ollama_version: str
    hardware: str
    recorded_at: str


def prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def slug(model: str) -> str:
    return model.replace(":", "-").replace("/", "-")


def source_dir() -> Path:
    """Cassette directory in the source tree (where the recorder writes)."""
    return Path(__file__).parent / CASSETTE_DIR


def write(model: str, records: list[Record], directory: Path | None = None) -> Path:
    target = (directory or source_dir()) / f"{slug(model)}.jsonl"
    target.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(asdict(record), ensure_ascii=False, sort_keys=True) for record in records]
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


def load_all() -> dict[str, list[Record]]:
    """Every recorded model, keyed by model name, records in file order."""
    root = resources.files("quantlens.evals").joinpath(CASSETTE_DIR)
    out: dict[str, list[Record]] = {}
    if not root.is_dir():
        return out
    for entry in sorted(root.iterdir(), key=lambda e: e.name):
        if not entry.name.endswith(".jsonl"):
            continue
        lines = entry.read_text(encoding="utf-8").splitlines()
        records = [Record(**json.loads(line)) for line in lines if line.strip()]
        if records:
            out[records[0].model] = records
    return out
