"""Wilson interval and minimum detectable effect."""

from __future__ import annotations

import pytest

from quantlens.evals.stats import mde_two_proportions, wilson


def test_wilson_matches_reference_values() -> None:
    rate = wilson(4, 4)
    assert rate.low == pytest.approx(0.5101, abs=1e-4)
    assert rate.high == 1.0
    assert wilson(5, 5).low == pytest.approx(0.5655, abs=1e-4)


def test_wilson_edges_are_exact() -> None:
    assert wilson(0, 5).low == 0.0
    assert wilson(0, 5).high == pytest.approx(0.4345, abs=1e-4)
    assert wilson(7, 7).high == 1.0


def test_wilson_interval_contains_the_estimate() -> None:
    rate = wilson(37, 120)
    assert rate.low < rate.value < rate.high
    assert "37/120" in str(rate)


@pytest.mark.parametrize(("k", "n"), [(1, 0), (-1, 5), (6, 5)])
def test_wilson_rejects_invalid_counts(k: int, n: int) -> None:
    with pytest.raises(ValueError):
        wilson(k, n)


def test_rate_of_empty_sample_is_zero() -> None:
    from quantlens.evals.stats import Rate

    assert Rate(0, 0, 0.0, 0.0).value == 0.0


def test_mde_shrinks_with_n_and_is_finite_at_the_edges() -> None:
    assert mde_two_proportions(0.5, 400) < mde_two_proportions(0.5, 100)
    assert mde_two_proportions(0.5, 120) == pytest.approx(0.1809, abs=1e-3)
    assert 0 < mde_two_proportions(1.0, 120) < 0.1


def test_mde_rejects_empty_arm() -> None:
    with pytest.raises(ValueError):
        mde_two_proportions(0.5, 0)
