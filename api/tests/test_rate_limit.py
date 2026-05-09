"""Per-API-key rate limiting via slowapi.

The default limiter uses an in-memory backend (one counter per process).
Tests share that counter across pytest runs in the same process — reset
the singleton per test so they don't pollute each other.
"""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from ._helpers import auth_headers, build_test_app


def _reset_limiter():
    """slowapi keeps storage on the Limiter instance; clearing keeps the
    same Limiter (the routes are decorated with it) but drops all counters."""
    from opf_api.rate_limit import limiter

    if hasattr(limiter, "_storage") and hasattr(limiter._storage, "storage"):
        limiter._storage.storage.clear()
    if hasattr(limiter, "reset"):
        try:
            limiter.reset()
        except Exception:  # noqa: BLE001
            pass


@pytest.mark.asyncio
async def test_redact_under_limit_passes():
    _reset_limiter()
    app = build_test_app(
        extra_env={"RATE_LIMIT_REDACT": "100/minute"}
    )
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        for _ in range(3):
            r = await c.post("/v1/redact", json={"text": "x"}, headers=auth_headers())
            assert r.status_code == 200


@pytest.mark.asyncio
async def test_redact_over_limit_returns_429():
    """slowapi route rules are evaluated at decorator time, so we can't
    simply bump RATE_LIMIT_REDACT here — it'd already be cached. Instead,
    drive enough traffic past the **default** rule (60/min) to trip it.
    Skipped unless the default is conservative enough that 70 quick requests
    actually trip it."""
    _reset_limiter()
    app = build_test_app()
    hit_429 = False
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        for _ in range(70):
            r = await c.post("/v1/redact", json={"text": "x"}, headers=auth_headers())
            if r.status_code == 429:
                hit_429 = True
                break
    assert hit_429, "expected 429 within 70 requests under default 30/min limit"
