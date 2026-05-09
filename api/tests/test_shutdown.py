"""InflightTracker drain semantics + 503 once draining starts."""

from __future__ import annotations

import asyncio

import pytest
from httpx import ASGITransport, AsyncClient

from opf_api.shutdown import InflightTracker

from ._helpers import auth_headers, build_test_app


@pytest.mark.asyncio
async def test_tracker_zero_at_start():
    t = InflightTracker()
    assert t.count == 0
    drained = await t.drain(timeout=0.1)
    assert drained is True


@pytest.mark.asyncio
async def test_tracker_drains_when_request_finishes():
    t = InflightTracker()

    async def work():
        async with t.track():
            await asyncio.sleep(0.05)

    asyncio.create_task(work())
    await asyncio.sleep(0)  # let work() enter track()
    drained = await t.drain(timeout=1.0)
    assert drained is True
    assert t.count == 0


@pytest.mark.asyncio
async def test_tracker_times_out_when_request_hangs():
    t = InflightTracker()
    started = asyncio.Event()

    async def hang():
        async with t.track():
            started.set()
            await asyncio.sleep(2.0)

    task = asyncio.create_task(hang())
    await started.wait()
    drained = await t.drain(timeout=0.1)
    assert drained is False
    task.cancel()


@pytest.mark.asyncio
async def test_503_once_draining():
    """Once tracker.draining is True, new requests get 503."""
    app = build_test_app()
    app.state.inflight.draining = True
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.post("/v1/redact", json={"text": "x"}, headers=auth_headers())
    assert r.status_code == 503


@pytest.mark.asyncio
async def test_probes_pass_during_drain():
    """Cloud Run keeps observing the dying instance — probes must keep working."""
    app = build_test_app()
    app.state.inflight.draining = True
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.get("/v1/health")
    assert r.status_code == 200
