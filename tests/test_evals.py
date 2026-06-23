"""The offline eval set must pass on the deterministic explainer."""

from __future__ import annotations

from quantlens import evals


def test_eval_set_all_pass() -> None:
    results = evals.run()
    failed = [r.ticker for r in results if not r.passed]
    assert not failed, f"failing cases: {failed}"
