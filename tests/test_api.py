"""API smoke tests (no network)."""

from __future__ import annotations

from fastapi.testclient import TestClient

from quantlens.api.main import app

client = TestClient(app)


def test_health_ok() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "version" in body
