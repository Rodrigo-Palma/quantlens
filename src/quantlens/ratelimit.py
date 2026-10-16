"""Per-client token bucket for the request path, and the client address behind a proxy.

``/analyze`` is a sync endpoint run in Starlette's threadpool, so the limiter
guards its state with a ``threading.Lock``. Every new client address creates a
bucket; the map is an LRU capped at ``max_clients`` so a stream of distinct
addresses cannot grow memory without bound. An evicted client comes back with a
full bucket, which is the price of the cap.

The bucket key is an IPv4 address, or the /64 network of an IPv6 address. A
single subscriber is usually handed a whole /64 (ISPs, VPS providers) and can
rotate through it at will, so keying by the full IPv6 address would give every
request a fresh bucket. IPv4-mapped IPv6 addresses are keyed as the IPv4 address.

The limit lives in one process: it is not shared between replicas or workers.
"""

from __future__ import annotations

import ipaddress
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from functools import lru_cache

type _Network = ipaddress.IPv4Network | ipaddress.IPv6Network

IPV6_PREFIX = 64


class RateLimiter:
    """Token bucket per key: ``burst`` requests at once, refilled at ``per_min`` a minute."""

    def __init__(
        self,
        per_min: float,
        burst: int,
        max_clients: int,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if per_min <= 0 or burst < 1 or max_clients < 1:
            raise ValueError("per_min, burst and max_clients must be positive")
        self._rate = per_min / 60.0
        self._burst = float(burst)
        self._max = max_clients
        self._clock = clock
        # key -> (tokens left, time they were counted)
        self._buckets: OrderedDict[str, tuple[float, float]] = OrderedDict()
        self._lock = threading.Lock()

    def check(self, key: str) -> float | None:
        """Take one token for ``key``: ``None`` if allowed, else seconds until one is free."""
        with self._lock:
            now = self._clock()
            tokens, counted_at = self._buckets.get(key, (self._burst, now))
            tokens = min(self._burst, tokens + (now - counted_at) * self._rate)
            allowed = tokens >= 1.0
            self._buckets[key] = (tokens - 1.0 if allowed else tokens, now)
            self._buckets.move_to_end(key)
            while len(self._buckets) > self._max:
                self._buckets.popitem(last=False)
            return None if allowed else (1.0 - tokens) / self._rate

    def clear(self) -> None:
        with self._lock:
            self._buckets.clear()

    def __len__(self) -> int:
        return len(self._buckets)


def client_address(peer: str, forwarded_for: str | None, trusted_proxies: str) -> str:
    """The address a request is limited by.

    ``X-Forwarded-For`` is read only when the direct peer is a trusted proxy.
    Hops are walked right to left, skipping trusted proxies, and the first other
    hop is the client: everything left of it was written by the client and can
    be spoofed. Without trusted proxies the peer is used, so behind a proxy every
    client shares the proxy's bucket. The result is the bucket key from
    :func:`_bucket_key`.
    """
    networks = parse_networks(trusted_proxies)
    if not networks or not _is_trusted(peer, networks):
        return _bucket_key(peer)
    hops = [hop.strip() for hop in (forwarded_for or "").split(",") if hop.strip()]
    for hop in reversed(hops):
        if not _is_trusted(hop, networks):
            return _bucket_key(hop)
    return _bucket_key(peer)


def _bucket_key(address: str) -> str:
    """IPv4 as is, IPv6 as its /64 network; anything unparsable is kept verbatim."""
    try:
        ip = ipaddress.ip_address(address)
    except ValueError:
        return address
    if isinstance(ip, ipaddress.IPv6Address):
        if ip.ipv4_mapped is not None:
            return str(ip.ipv4_mapped)
        return str(ipaddress.ip_network(f"{ip}/{IPV6_PREFIX}", strict=False))
    return str(ip)


def _is_trusted(address: str, networks: tuple[_Network, ...]) -> bool:
    try:
        ip = ipaddress.ip_address(address)
    except ValueError:
        return False
    return any(ip in network for network in networks)


@lru_cache(maxsize=8)
def parse_networks(trusted_proxies: str) -> tuple[_Network, ...]:
    entries = [entry.strip() for entry in trusted_proxies.split(",") if entry.strip()]
    try:
        return tuple(ipaddress.ip_network(entry, strict=False) for entry in entries)
    except ValueError as exc:
        raise ValueError(f"TRUSTED_PROXIES must be IPs or CIDRs: {exc}") from exc
