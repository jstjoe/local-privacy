"""End-to-end route tests using a stub detector — avoids loading real models.

The tests inject a deterministic FakeDetector into the registry and exercise
the /v1 surface plus the legacy compat shim's plumbing (without OPF model load).
"""

from __future__ import annotations

import asyncio

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from opf_api.registry import DetectorEntry
from opf_api.routes import router


class FakeDetector:
    name = "fake"

    def detect(self, text: str, **_):
        spans = []
        # Match `joe@example.com` for EMAIL.
        idx = text.find("joe@example.com")
        if idx != -1:
            spans.append(
                {
                    "label": "EMAIL",
                    "raw_label": "EMAIL_ADDRESS",
                    "start": idx,
                    "end": idx + len("joe@example.com"),
                    "text": "joe@example.com",
                }
            )
        # Match `Elgin, TX` for ADDRESS.
        idx = text.find("Elgin, TX")
        if idx != -1:
            spans.append(
                {
                    "label": "ADDRESS",
                    "raw_label": "LOCATION_ADDRESS",
                    "start": idx,
                    "end": idx + len("Elgin, TX"),
                    "text": "Elgin, TX",
                }
            )
        return {"spans": spans, "latency_ms": 0.1, "error": None}


def _build_app() -> FastAPI:
    app = FastAPI()
    app.include_router(router, prefix="/v1")
    fake_entry = DetectorEntry(
        name="fake", factory=lambda: FakeDetector()  # noqa: PIE807
    )
    fake_entry.instance = FakeDetector()
    app.state.registry = {"fake": fake_entry}
    app.state.default_detector = "fake"
    app.state.schema_version = 1
    return app


@pytest.fixture
def client():
    app = _build_app()
    transport = ASGITransport(app=app)
    return AsyncClient(transport=transport, base_url="http://test")


@pytest.mark.asyncio
async def test_redact_default(client: AsyncClient):
    async with client as c:
        r = await c.post("/v1/redact", json={"text": "Joe at joe@example.com lives in Elgin, TX."})
    assert r.status_code == 200
    body = r.json()
    assert body["detector"] == "fake"
    assert body["redacted_text"] == "Joe at [EMAIL] lives in [ADDRESS]."
    assert body["summary"] == {"span_count": 2, "by_label": {"EMAIL": 1, "ADDRESS": 1}}
    assert {s["label"] for s in body["detected_spans"]} == {"EMAIL", "ADDRESS"}


@pytest.mark.asyncio
async def test_redact_filter_categories(client: AsyncClient):
    async with client as c:
        r = await c.post(
            "/v1/redact",
            json={
                "text": "Joe at joe@example.com lives in Elgin, TX.",
                "categories": ["EMAIL"],
            },
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
        )
    assert r.status_code == 400
    assert "NOT_REAL" in r.json()["detail"]


@pytest.mark.asyncio
async def test_unknown_detector(client: AsyncClient):
    async with client as c:
        r = await c.post("/v1/redact", json={"text": "x", "detector": "ghost"})
    assert r.status_code == 400


@pytest.mark.asyncio
async def test_opf_native_rejected_for_non_opf(client: AsyncClient):
    async with client as c:
        r = await c.post(
            "/v1/redact",
            json={"text": "x", "placeholder_format": "opf_native"},
        )
    assert r.status_code == 400


@pytest.mark.asyncio
async def test_detect_no_redacted_text(client: AsyncClient):
    async with client as c:
        r = await c.post("/v1/detect", json={"text": "joe@example.com"})
    body = r.json()
    assert "redacted_text" not in body
    assert body["summary"]["span_count"] == 1


@pytest.mark.asyncio
async def test_list_detectors(client: AsyncClient):
    async with client as c:
        r = await c.get("/v1/detectors")
    body = r.json()
    assert body["default"] == "fake"
    assert body["detectors"][0]["name"] == "fake"
    assert body["detectors"][0]["loaded"] is True


@pytest.mark.asyncio
async def test_health(client: AsyncClient):
    async with client as c:
        r = await c.get("/v1/health")
    body = r.json()
    assert body["status"] == "ok"
    assert body["default_detector"] == "fake"
    assert "fake" in body["loaded_detectors"]
