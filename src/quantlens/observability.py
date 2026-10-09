"""Structured request logging: one JSON line per request with per-stage latency."""

from __future__ import annotations

import json
import logging
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field

REQUEST_LOGGER = "quantlens.request"

_log = logging.getLogger(REQUEST_LOGGER)


def configure_logging(level: int = logging.INFO) -> None:
    """Attach a bare ``%(message)s`` handler once, so each event is one JSON line."""
    logger = logging.getLogger("quantlens")
    logger.setLevel(level)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)


@dataclass
class RequestTrace:
    """Accumulates what happened during one request and emits it as JSON."""

    endpoint: str
    ticker: str
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    status: int = 200
    latency_ms: dict[str, float] = field(default_factory=dict)
    attributes: dict[str, object] = field(default_factory=dict)

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            self.latency_ms[name] = round((time.perf_counter() - start) * 1000, 3)

    def emit(self) -> None:
        event = {
            "event": "request",
            "request_id": self.request_id,
            "endpoint": self.endpoint,
            "ticker": self.ticker,
            "status": self.status,
            "latency_ms": self.latency_ms,
            "total_ms": round(sum(self.latency_ms.values()), 3),
            **self.attributes,
        }
        _log.info(json.dumps(event, sort_keys=True))
