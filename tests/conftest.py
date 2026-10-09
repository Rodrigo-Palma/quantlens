"""Shared fixtures."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from quantlens.api import main


@pytest.fixture(autouse=True)
def _empty_api_caches() -> Iterator[None]:
    """Each test sees a cold cache; a cached series must not leak across tests."""
    main._closes.clear()
    main._explanations.clear()
    yield
    main._closes.clear()
    main._explanations.clear()
