"""Adjudication of the two held-out sets, kept apart from the frozen labels.

``checker_errors.jsonl`` and ``retrieval_heldout.jsonl`` were labeled, measured
once and frozen. Relabeling them after seeing the scores would leak the result
into the labels, so the review lives in ``data/adjudications.jsonl``: one row
per held-out row, with a verdict.

* ``ok``: the frozen label stands.
* ``ambiguous``: retrieval only; another glossary section answers the question
  at least as well as the labeled one (``alt_expected`` lists it). For the
  checker set: the text may not contradict the signals.
* ``mislabeled``: the frozen label is wrong. Retrieval: ``alt_expected`` holds
  the right section. Checker: the planted text is not an error.

Each row carries the SHA-256 (first 12 hex digits) of the raw line it judged,
so a reorder or edit of a frozen file breaks the binding loudly.

The tolerant metric is secondary and never gated. Retrieval: a hit counts if
rank 1 is the label or, for an ``ambiguous`` row, one of its alternatives; a
``mislabeled`` row is scored against its alternatives only. Checker: recall
over the rows adjudicated ``ok`` (the ones that are certainly errors).

Review protocol: single reviewer, the author of the system; not independent.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from importlib import resources
from typing import Any

from quantlens.evals import checks, retrieval
from quantlens.evals.cases import build_cases
from quantlens.evals.stats import Rate, wilson

DATASETS = ("checker_errors", "retrieval_heldout")
VERDICTS = ("ok", "ambiguous", "mislabeled")
FIELDS = ("id", "dataset", "verdict", "alt_expected", "note", "row_sha")
SHA_DIGITS = 12
PROTOCOL = "single reviewer, the author; not independent"


@dataclass(frozen=True)
class Adjudication:
    id: str
    dataset: str
    verdict: str
    alt_expected: tuple[str, ...] | None
    note: str
    row_sha: str = ""


def row_id(dataset: str, line_number: int) -> str:
    """``dataset:NNN``, the 1-based line number in the frozen file."""
    return f"{dataset}:{line_number:03d}"


def sha(line: str) -> str:
    return hashlib.sha256(line.encode("utf-8")).hexdigest()[:SHA_DIGITS]


def _read(name: str) -> list[str]:
    path = resources.files("quantlens.evals").joinpath(f"data/{name}.jsonl")
    return [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def source_rows(dataset: str) -> list[str]:
    """The raw lines of a frozen held-out file, in order."""
    if dataset not in DATASETS:
        raise ValueError(f"unknown dataset {dataset!r}")
    return _read(dataset)


def parse(raw: dict[str, Any]) -> Adjudication:
    missing = [f for f in FIELDS if f not in raw]
    if missing:
        raise ValueError(f"adjudication {raw.get('id')!r}: missing {missing}")
    if raw["verdict"] not in VERDICTS:
        raise ValueError(f"adjudication {raw['id']!r}: verdict {raw['verdict']!r}")
    if raw["id"].partition(":")[0] != raw["dataset"] or raw["dataset"] not in DATASETS:
        raise ValueError(f"adjudication {raw['id']!r}: dataset {raw['dataset']!r}")
    alt = raw["alt_expected"]
    return Adjudication(
        id=raw["id"],
        dataset=raw["dataset"],
        verdict=raw["verdict"],
        alt_expected=None if alt is None else tuple(alt),
        note=raw["note"],
        row_sha=raw["row_sha"],
    )


def load() -> tuple[Adjudication, ...]:
    return tuple(parse(json.loads(line)) for line in _read("adjudications"))


def _by_id(dataset: str) -> dict[str, Adjudication]:
    return {a.id: a for a in load() if a.dataset == dataset}


def tolerant_hits(
    top1: list[str], expected: list[str], adjudications: dict[str, Adjudication]
) -> Rate:
    """Retrieval hit@1 accepting the adjudicated alternatives (row order = file order)."""
    hits = 0
    for n, (got, label) in enumerate(zip(top1, expected, strict=True), start=1):
        a = adjudications[row_id("retrieval_heldout", n)]
        alt = set(a.alt_expected or ())
        accepted = alt if a.verdict == "mislabeled" else {label} | alt
        hits += got in accepted
    return wilson(hits, len(top1))


def tolerant_recall(caught: list[bool], adjudications: dict[str, Adjudication]) -> Rate:
    """Checker recall over the rows adjudicated ``ok`` (certainly wrong texts)."""
    kept = [
        hit
        for n, hit in enumerate(caught, start=1)
        if adjudications[row_id("checker_errors", n)].verdict == "ok"
    ]
    return wilson(sum(kept), len(kept))


def retrieval_rows() -> tuple[list[str], list[str]]:
    """(BM25 rank-1 title, frozen label) for each held-out question, in file order."""
    ranker = retrieval.rankers()[retrieval.WORD]
    queries = retrieval.load_queries("retrieval_heldout")
    return [ranker(q.query)[0] for q in queries], [q.expected for q in queries]


def checker_rows() -> list[bool]:
    """Whether the v0.7 checker fails each planted text, in file order."""
    by_id = {c.case_id: c for c in build_cases()}
    rows = [json.loads(line) for line in source_rows("checker_errors")]
    return [not checks.score(r["text"], by_id[r["case_id"]]).passed for r in rows]


def _counts(adjudications: dict[str, Adjudication]) -> str:
    verdicts = [a.verdict for a in adjudications.values()]
    return f"{verdicts.count('ambiguous')} ambiguous, {verdicts.count('mislabeled')} mislabeled"


def report() -> list[str]:
    top1, expected = retrieval_rows()
    ret_adj = _by_id("retrieval_heldout")
    strict_ret = wilson(sum(t == e for t, e in zip(top1, expected, strict=True)), len(top1))
    caught = checker_rows()
    chk_adj = _by_id("checker_errors")
    strict_chk = wilson(sum(caught), len(caught))
    return [
        f"Adjudicated held-out sets ({PROTOCOL}); gates use the strict counts only",
        f"  retrieval_heldout, BM25 hit@1 ({_counts(ret_adj)})",
        f"    strict {strict_ret}",
        f"    tolerant (secondary) {tolerant_hits(top1, expected, ret_adj)}",
        f"  checker_errors, v0.7 checker recall ({_counts(chk_adj)})",
        f"    strict {strict_chk}",
        f"    tolerant (secondary) {tolerant_recall(caught, chk_adj)}",
    ]
