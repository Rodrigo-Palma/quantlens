"""The TTL cache used on the request path."""

from __future__ import annotations

import pytest

from quantlens.cache import TTLCache


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def test_value_expires_after_ttl() -> None:
    clock = _Clock()
    cache: TTLCache[str] = TTLCache(10.0, 4, clock)
    cache.put("k", "v")
    clock.now = 9.9
    assert cache.get("k") == "v"
    clock.now = 10.0
    assert cache.get("k") is None
    assert len(cache) == 0


def test_least_recently_used_entry_is_evicted() -> None:
    cache: TTLCache[int] = TTLCache(60.0, 2)
    cache.put("a", 1)
    cache.put("b", 2)
    assert cache.get("a") == 1
    cache.put("c", 3)
    assert cache.get("b") is None
    assert (cache.get("a"), cache.get("c")) == (1, 3)


def test_zero_ttl_disables_caching() -> None:
    cache: TTLCache[int] = TTLCache(0.0, 2)
    cache.put("a", 1)
    assert cache.get("a") is None


def test_invalid_configuration_is_rejected() -> None:
    with pytest.raises(ValueError):
        TTLCache(-1.0, 2)
    with pytest.raises(ValueError):
        TTLCache(1.0, 0)
