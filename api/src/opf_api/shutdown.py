"""In-flight request tracker for graceful drain on SIGTERM.

Cloud Run sends SIGTERM with a configurable grace period (default 10s,
extend up to 600s with `--timeout`). Lifespan calls `drain()` after the
yield; the tracker waits for in-flight requests to finish, bounded by
`SHUTDOWN_DRAIN_SECONDS` (default 20s).

Once draining starts, new requests get 503. Probes (/v1/health, /v1/ready)
bypass the gate so Cloud Run can keep observing the dying instance.
"""

from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse


DEFAULT_DRAIN_SECONDS = 20.0
PROBE_PATHS = {"/v1/health", "/v1/ready", "/metrics"}


def drain_seconds() -> float:
    return float(os.environ.get("SHUTDOWN_DRAIN_SECONDS", DEFAULT_DRAIN_SECONDS))


class InflightTracker:
    def __init__(self) -> None:
        self._count = 0
        self._zero = asyncio.Event()
        self._zero.set()
        self.draining = False

    @property
    def count(self) -> int:
        return self._count

    @asynccontextmanager
    async def track(self):
        self._count += 1
        self._zero.clear()
        try:
            yield
        finally:
            self._count -= 1
            if self._count == 0:
                self._zero.set()

    async def drain(self, timeout: float) -> bool:
        """Set the draining flag, then wait for in-flight count to hit zero
        (or `timeout` seconds elapse). Returns True if drained cleanly."""
        self.draining = True
        if self._count == 0:
            return True
        try:
            await asyncio.wait_for(self._zero.wait(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            return False


class InflightMiddleware(BaseHTTPMiddleware):
    """Track every request through the tracker; reject with 503 once drain
    has started. Probe paths bypass both the tracker and the gate."""

    def __init__(self, app, *, tracker: InflightTracker) -> None:
        super().__init__(app)
        self.tracker = tracker

    async def dispatch(self, request: Request, call_next):
        if request.url.path in PROBE_PATHS:
            return await call_next(request)
        if self.tracker.draining:
            return JSONResponse(
                {"detail": "server shutting down"}, status_code=503
            )
        async with self.tracker.track():
            return await call_next(request)
