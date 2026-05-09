"""Per-API-key rate limiting via slowapi.

In-memory storage = per-instance counters. With `--max-instances=N`, true
ceiling is `N * limit`. Document this in deploy/README.md. For tighter
enforcement, future plan: Redis (Memorystore) backing via slowapi's
`storage_uri`.
"""

from __future__ import annotations

import os

from slowapi import Limiter
from starlette.requests import Request


def _key_func(request: Request) -> str:
    """Bucket on API key when present, fall back to client IP for unauthed
    paths (probes are skip-listed in routes — so this only matters when
    AUTH_DISABLED=1)."""
    key = getattr(request.state, "api_key", None)
    if key:
        return f"key:{key}"
    client = request.client
    return f"ip:{client.host if client else 'unknown'}"


def default_limit() -> str:
    return os.environ.get("RATE_LIMIT_DEFAULT", "60/minute")


def redact_limit() -> str:
    return os.environ.get("RATE_LIMIT_REDACT", "30/minute")


def detect_limit() -> str:
    return os.environ.get("RATE_LIMIT_DETECT", "60/minute")


def build_limiter() -> Limiter:
    return Limiter(key_func=_key_func, default_limits=[default_limit()])


# Module-level singleton so route decorators in routes.py and the lookup in
# main.py see the same instance. `default_limits` is fixed at construction;
# per-route rules are evaluated lazily by slowapi from the decorator string.
limiter = build_limiter()
