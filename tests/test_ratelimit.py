"""The per-client token bucket and the client address behind a proxy."""

from __future__ import annotations

import threading

import pytest
from pydantic import ValidationError

from quantlens.config import Settings
from quantlens.ratelimit import RateLimiter, client_address


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def test_burst_is_allowed_then_refused_with_retry_after() -> None:
    limiter = RateLimiter(per_min=60, burst=3, max_clients=10, clock=_Clock())
    assert [limiter.check("a") for _ in range(3)] == [None, None, None]
    assert limiter.check("a") == pytest.approx(1.0)


def test_bucket_refills_at_the_per_minute_rate() -> None:
    clock = _Clock()
    limiter = RateLimiter(per_min=30, burst=1, max_clients=10, clock=clock)
    assert limiter.check("a") is None
    clock.now = 1.0
    assert limiter.check("a") == pytest.approx(1.0)
    clock.now = 2.0
    assert limiter.check("a") is None


def test_refill_never_exceeds_the_burst() -> None:
    clock = _Clock()
    limiter = RateLimiter(per_min=60, burst=2, max_clients=10, clock=clock)
    limiter.check("a")
    clock.now = 3600.0
    assert [limiter.check("a") for _ in range(3)] == [None, None, pytest.approx(1.0)]


def test_clients_have_separate_buckets() -> None:
    limiter = RateLimiter(per_min=60, burst=1, max_clients=10, clock=_Clock())
    assert limiter.check("a") is None
    assert limiter.check("b") is None
    assert limiter.check("a") is not None


def test_client_map_never_exceeds_max_clients() -> None:
    limiter = RateLimiter(per_min=60, burst=1, max_clients=5, clock=_Clock())
    for i in range(1000):
        limiter.check(f"10.0.{i // 256}.{i % 256}")
        assert len(limiter) <= 5
    assert len(limiter) == 5


def test_least_recently_seen_client_is_evicted_first() -> None:
    limiter = RateLimiter(per_min=60, burst=1, max_clients=2, clock=_Clock())
    limiter.check("a")
    limiter.check("b")
    limiter.check("a")  # refused, but a is now the most recent
    limiter.check("c")  # evicts b
    assert limiter.check("a") is not None
    assert limiter.check("b") is None


def test_exactly_burst_requests_pass_under_contention() -> None:
    burst, threads = 7, 64
    limiter = RateLimiter(per_min=60, burst=burst, max_clients=10, clock=_Clock())
    start = threading.Barrier(threads)
    results: list[float | None] = []
    results_lock = threading.Lock()

    def _hit() -> None:
        start.wait()
        outcome = limiter.check("same-ip")
        with results_lock:
            results.append(outcome)

    workers = [threading.Thread(target=_hit) for _ in range(threads)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()
    assert len(results) == threads
    assert sum(r is None for r in results) == burst


def test_clear_forgets_every_client() -> None:
    limiter = RateLimiter(per_min=60, burst=1, max_clients=10, clock=_Clock())
    limiter.check("a")
    limiter.clear()
    assert len(limiter) == 0
    assert limiter.check("a") is None


@pytest.mark.parametrize(
    ("per_min", "burst", "max_clients"), [(0, 1, 1), (1, 0, 1), (1, 1, 0), (-1, 1, 1)]
)
def test_rejects_non_positive_settings(per_min: float, burst: int, max_clients: int) -> None:
    with pytest.raises(ValueError, match="positive"):
        RateLimiter(per_min=per_min, burst=burst, max_clients=max_clients)


def test_forwarded_for_is_ignored_without_trusted_proxies() -> None:
    assert client_address("10.0.0.1", "203.0.113.9", "") == "10.0.0.1"


def test_forwarded_for_is_ignored_from_an_untrusted_peer() -> None:
    assert client_address("10.0.0.1", "203.0.113.9", "10.0.0.2") == "10.0.0.1"


def test_forwarded_for_is_used_from_a_trusted_proxy() -> None:
    assert client_address("10.0.0.1", "203.0.113.9", "10.0.0.1") == "203.0.113.9"


def test_trusted_proxies_accept_networks() -> None:
    assert client_address("10.1.2.3", "203.0.113.9", "10.0.0.0/8") == "203.0.113.9"


def test_spoofed_left_entries_are_skipped() -> None:
    # The client controls everything left of the hop our proxy appended.
    header = "1.1.1.1, 203.0.113.9"
    assert client_address("10.0.0.1", header, "10.0.0.1") == "203.0.113.9"


def test_chained_trusted_proxies_are_skipped() -> None:
    header = "203.0.113.9, 10.0.0.5"
    assert client_address("10.0.0.1", header, "10.0.0.0/8") == "203.0.113.9"


@pytest.mark.parametrize("header", [None, "", " , ", "10.0.0.5"])
def test_trusted_peer_without_a_client_hop_falls_back_to_the_peer(header: str | None) -> None:
    assert client_address("10.0.0.1", header, "10.0.0.0/8") == "10.0.0.1"


def test_unparsable_peer_is_never_trusted() -> None:
    assert client_address("testclient", "203.0.113.9", "10.0.0.0/8") == "testclient"


def test_unparsable_forwarded_hop_is_used_as_is() -> None:
    assert client_address("10.0.0.1", "unknown", "10.0.0.1") == "unknown"


def test_malformed_trusted_proxies_fail_clearly() -> None:
    with pytest.raises(ValueError, match="TRUSTED_PROXIES"):
        client_address("10.0.0.1", "203.0.113.9", "not-an-ip")


@pytest.mark.parametrize("value", ["not-an-ip", "10.0.0.0/33", "10.0.0.1, 10.0.0"])
def test_settings_reject_malformed_trusted_proxies(value: str) -> None:
    with pytest.raises(ValidationError, match="TRUSTED_PROXIES"):
        Settings(trusted_proxies=value)


def test_settings_accept_trusted_proxies() -> None:
    assert Settings(trusted_proxies="10.0.0.1, 192.168.0.0/16, ::1").trusted_proxies


@pytest.mark.parametrize(
    "field",
    ["rate_limit_per_min", "rate_limit_burst", "rate_limit_max_clients", "llm_max_concurrency"],
)
def test_settings_reject_non_positive_limits(field: str) -> None:
    with pytest.raises(ValidationError):
        Settings(**{field: 0})
