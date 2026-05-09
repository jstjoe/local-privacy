"""API-key auth middleware.

Loads accepted keys from `API_KEYS` (comma-separated). On Cloud Run that env
is sourced from Secret Manager via `--set-secrets API_KEYS=privacy-api-keys:latest`,
so rotation = bump the secret version, redeploy revision.

Set `AUTH_DISABLED=1` for local dev. Probes (/v1/health, /v1/ready) and the
Prometheus scrape (/metrics) bypass auth so Cloud Run + scrapers can hit them
without a key.
"""

from __future__ import annotations

import os

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse


PROBE_PATHS = {"/v1/health", "/v1/ready", "/metrics"}


def load_api_keys() -> set[str]:
    raw = os.environ.get("API_KEYS", "")
    return {k.strip() for k in raw.split(",") if k.strip()}


def _key_prefix(key: str) -> str:
    """First 6 chars — safe to log; never log the full key."""
    return key[:6]


class APIKeyMiddleware(BaseHTTPMiddleware):
    def __init__(
        self,
        app,
        *,
        valid_keys: set[str],
        skip_paths: set[str] = PROBE_PATHS,
        disabled: bool = False,
    ) -> None:
        super().__init__(app)
        self.valid_keys = valid_keys
        self.skip_paths = skip_paths
        self.disabled = disabled

    async def dispatch(self, request: Request, call_next):
        if self.disabled or request.url.path in self.skip_paths:
            return await call_next(request)
        key = request.headers.get("x-api-key")
        if not key or key not in self.valid_keys:
            return JSONResponse({"detail": "unauthorized"}, status_code=401)
        request.state.api_key = key
        request.state.api_key_prefix = _key_prefix(key)
        return await call_next(request)
