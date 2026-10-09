"""A small thread-safe TTL cache for the request path.

Each ``/analyze`` call costs a yfinance download and, with the default model,
about 8 s of GPU. Daily closes and greedy decoding make both repeatable, so the
API keeps recent results for ``settings.cache_ttl_s`` seconds. Failures (a
fallback, a missing series) are never cached.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Hashable


class TTLCache[V]:
    def __init__(
        self, ttl_s: float, max_entries: int, clock: Callable[[], float] = time.monotonic
    ) -> None:
        if ttl_s < 0 or max_entries < 1:
            raise ValueError("ttl_s must be >= 0 and max_entries >= 1")
        self._ttl = ttl_s
        self._max = max_entries
        self._clock = clock
        self._items: OrderedDict[Hashable, tuple[float, V]] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: Hashable) -> V | None:
        with self._lock:
            item = self._items.get(key)
            if item is None:
                return None
            expires, value = item
            if self._clock() >= expires:
                del self._items[key]
                return None
            self._items.move_to_end(key)
            return value

    def put(self, key: Hashable, value: V) -> None:
        if self._ttl == 0:
            return
        with self._lock:
            self._items[key] = (self._clock() + self._ttl, value)
            self._items.move_to_end(key)
            while len(self._items) > self._max:
                self._items.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._items.clear()

    def __len__(self) -> int:
        return len(self._items)
