"""End-to-end route tests using a stub detector — avoids loading real models."""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from ._helpers import auth_headers, build_test_app


@pytest.fixture
def client():
    app = build_test_app()
    transport = ASGITransport(app=app)
    return AsyncClient(transport=transport, base_url="http://test")


@pytest.mark.asyncio
async def test_redact_default(client: AsyncClient):
    async with client as c:
        r = await c.post(
            "/v1/redact",
            json={"text": "Joe at joe@example.com lives in Elgin, TX."},
            headers=auth_headers(),
        )
    assert r.status_code == 200
    body = r.json()
    assert body["detector"] == "fake"
    assert body["redacted_text"] == "Joe at [EMAIL] lives in [ADDRESS]."
    assert body["summary"] == {"span_count": 2, "by_label": {"EMAIL": 1, "ADDRESS": 1}}


@pytest.mark.asyncio
async def test_redact_filter_categories(client: AsyncClient):
    async with client as c:
        r = await c.post(
            "/v1/redact",
            json={
                "text": "Joe at joe@example.com lives in Elgin, TX.",
                "categories": ["EMAIL"],
            },
            headers=auth_headers(),
        )
    assert r.status_code == 200
    body = r.json()
    assert body["redacted_text"] == "Joe at [EMAIL] lives in Elgin, TX."
    assert [s["label"] for s in body["detected_spans"]] == ["EMAIL"]


@pytest.mark.asyncio
async def test_redact_invalid_category(client: AsyncClient):
    async with client as c:
        r = await c.post(
            "/v1/redact",
            json={"text": "x", "categories": ["NOT_REAL"]},
            headers=auth_headers(),
        )
    assert r.status_code == 400


@pytest.mark.asyncio
async def test_unknown_detector(client: AsyncClient):
    async with client as c:
        r = await c.post(
            "/v1/redact",
            json={"text": "x", "detector": "ghost"},
            headers=auth_headers(),
        )
    assert r.status_code == 400


@pytest.mark.asyncio
async def test_opf_native_rejected_for_non_opf(client: AsyncClient):
    async with client as c:
        r = await c.post(
            "/v1/redact",
            json={"text": "x", "placeholder_format": "opf_native"},
            headers=auth_headers(),
        )
    assert r.status_code == 400


@pytest.mark.asyncio
async def test_detect_no_redacted_text(client: AsyncClient):
    async with client as c:
        r = await c.post(
            "/v1/detect",
            json={"text": "joe@example.com"},
            headers=auth_headers(),
        )
    body = r.json()
    assert "redacted_text" not in body
    assert body["summary"]["span_count"] == 1


@pytest.mark.asyncio
async def test_list_detectors(client: AsyncClient):
    async with client as c:
        r = await c.get("/v1/detectors", headers=auth_headers())
    body = r.json()
    assert body["default"] == "fake"
    assert body["detectors"][0]["name"] == "fake"
    assert body["detectors"][0]["loaded"] is True


@pytest.mark.asyncio
async def test_health_unauth(client: AsyncClient):
    """Health probe must work without API key (Cloud Run liveness)."""
    async with client as c:
        r = await c.get("/v1/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


@pytest.mark.asyncio
async def test_request_id_header(client: AsyncClient):
    async with client as c:
        r = await c.get("/v1/health")
    assert "x-request-id" in r.headers


@pytest.mark.asyncio
async def test_request_id_echoed():
    app = build_test_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.get("/v1/health", headers={"x-request-id": "fixed-id-123"})
    assert r.headers["x-request-id"] == "fixed-id-123"
