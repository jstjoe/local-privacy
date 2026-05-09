"""API-key auth — missing/wrong key returns 401; probes are skip-listed."""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from ._helpers import build_test_app


@pytest.mark.asyncio
async def test_missing_key_returns_401():
    app = build_test_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.post("/v1/redact", json={"text": "x"})
    assert r.status_code == 401


@pytest.mark.asyncio
async def test_wrong_key_returns_401():
    app = build_test_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.post(
            "/v1/redact",
            json={"text": "x"},
            headers={"x-api-key": "nope", "content-type": "application/json"},
        )
    assert r.status_code == 401


@pytest.mark.asyncio
async def test_probes_skip_auth():
    """Cloud Run probes hit /v1/health, /v1/ready, /metrics with no key."""
    app = build_test_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        for path in ("/v1/health", "/v1/ready", "/metrics"):
            r = await c.get(path)
            assert r.status_code != 401, f"{path} should not require a key"


@pytest.mark.asyncio
async def test_auth_disabled():
    app = build_test_app(api_keys="", auth_disabled=True)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.post(
            "/v1/redact",
            json={"text": "x"},
            headers={"content-type": "application/json"},
        )
    # 200 (or 422 — but never 401) confirms auth is bypassed.
    assert r.status_code != 401


@pytest.mark.asyncio
async def test_no_keys_no_disable_raises():
    """Refuse to start with no auth configured at all."""
    import os

    os.environ.pop("API_KEYS", None)
    os.environ["AUTH_DISABLED"] = "0"
    from opf_api.main import create_app

    with pytest.raises(RuntimeError, match="No API_KEYS configured"):
        create_app(with_lifespan=False)
