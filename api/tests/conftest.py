"""Set API_KEYS before opf_api.main imports — module-level `app = create_app()`
runs at import time and would otherwise refuse to start."""

from __future__ import annotations

import os

os.environ.setdefault("API_KEYS", "test-key")
os.environ.setdefault("AUTH_DISABLED", "0")


import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_rate_limiter():
    """slowapi keeps an in-memory counter on the module-level Limiter
    singleton — clear between tests so they don't poison each other."""
    from opf_api.rate_limit import limiter

    storage = getattr(limiter, "_storage", None)
    inner = getattr(storage, "storage", None)
    if isinstance(inner, dict):
        inner.clear()
    yield
