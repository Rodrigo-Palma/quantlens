"""Adjudication of the two held-out sets: schema, coverage and the secondary metric."""

from __future__ import annotations

import pytest

from quantlens import rag
from quantlens.evals import adjudication
from quantlens.evals.adjudication import Adjudication


def _adj(row_id: str, verdict: str, alt: tuple[str, ...] | None = ()) -> Adjudication:
    dataset = row_id.partition(":")[0]
    return Adjudication(id=row_id, dataset=dataset, verdict=verdict, alt_expected=alt, note="x")


def test_every_held_out_row_is_adjudicated_exactly_once() -> None:
    adjudications = adjudication.load()
    ids = [a.id for a in adjudications]
    assert len(ids) == len(set(ids)) == 156
    expected = {
        adjudication.row_id(dataset, n)
        for dataset in adjudication.DATASETS
        for n in range(1, len(adjudication.source_rows(dataset)) + 1)
    }
    assert set(ids) == expected


def test_each_adjudication_is_bound_to_the_frozen_row_it_judged() -> None:
    for a in adjudication.load():
        dataset, _, index = a.id.partition(":")
        assert a.dataset == dataset
        line = adjudication.source_rows(dataset)[int(index) - 1]
        assert a.row_sha == adjudication.sha(line), a.id


def test_verdicts_and_alternatives_follow_the_schema() -> None:
    titles = {chunk.title for chunk in rag.load_chunks()}
    for a in adjudication.load():
        assert a.verdict in adjudication.VERDICTS
        assert a.note.strip(), a.id
        if a.dataset == "checker_errors":
            assert a.alt_expected is None, a.id
            continue
        assert a.alt_expected is not None
        assert set(a.alt_expected) <= titles, a.id
        assert bool(a.alt_expected) == (a.verdict != "ok"), a.id


def test_validate_rejects_bad_rows() -> None:
    with pytest.raises(ValueError, match="verdict"):
        adjudication.parse(
            {
                "id": "checker_errors:001",
                "dataset": "checker_errors",
                "verdict": "maybe",
                "alt_expected": None,
                "note": "x",
                "row_sha": "0",
            }
        )
    with pytest.raises(ValueError, match="dataset"):
        adjudication.parse(
            {
                "id": "checker_errors:001",
                "dataset": "retrieval_heldout",
                "verdict": "ok",
                "alt_expected": [],
                "note": "x",
                "row_sha": "0",
            }
        )
    with pytest.raises(ValueError, match="missing"):
        adjudication.parse({"id": "checker_errors:001"})


def test_tolerant_retrieval_accepts_alternatives_only_where_adjudicated() -> None:
    top1 = ["A", "B", "C", "D"]
    expected = ["A", "X", "X", "X"]
    adjudications = {
        "retrieval_heldout:001": _adj("retrieval_heldout:001", "ok"),
        "retrieval_heldout:002": _adj("retrieval_heldout:002", "ambiguous", ("B",)),
        "retrieval_heldout:003": _adj("retrieval_heldout:003", "mislabeled", ("C",)),
        "retrieval_heldout:004": _adj("retrieval_heldout:004", "ok"),
    }
    rate = adjudication.tolerant_hits(top1, expected, adjudications)
    assert (rate.successes, rate.n) == (3, 4)


def test_mislabeled_drops_the_original_label() -> None:
    adjudications = {"retrieval_heldout:001": _adj("retrieval_heldout:001", "mislabeled", ("C",))}
    rate = adjudication.tolerant_hits(["A"], ["A"], adjudications)
    assert (rate.successes, rate.n) == (0, 1)


def test_tolerant_checker_recall_keeps_only_rows_adjudicated_wrong() -> None:
    caught = [True, False, False, True]
    adjudications = {
        "checker_errors:001": _adj("checker_errors:001", "ok", None),
        "checker_errors:002": _adj("checker_errors:002", "mislabeled", None),
        "checker_errors:003": _adj("checker_errors:003", "ambiguous", None),
        "checker_errors:004": _adj("checker_errors:004", "ok", None),
    }
    rate = adjudication.tolerant_recall(caught, adjudications)
    assert (rate.successes, rate.n) == (2, 2)


def test_report_prints_strict_and_secondary_lines() -> None:
    lines = adjudication.report()
    text = "\n".join(lines)
    assert "strict 34/60" in text
    assert "strict 91/96" in text
    assert text.count("secondary") >= 2
    assert "single reviewer" in text
