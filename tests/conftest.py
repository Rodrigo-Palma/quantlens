"""Shared fixtures."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from quantlens.api import main


@pytest.fixture(autouse=True)
def _empty_api_caches() -> Iterator[None]:
    """Each test sees a cold cache and a full rate-limit bucket; no state leaks across tests."""
    _reset()
    yield
    _reset()


def _reset() -> None:
    main._closes.clear()
    main._explanations.clear()
    main._limiter.clear()
