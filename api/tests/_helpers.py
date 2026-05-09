"""Shared test helpers: build a hardened app with a stub detector registry."""

from __future__ import annotations

import os
from typing import Any

from fastapi import FastAPI

from opf_api.main import SCHEMA_VERSION, create_app
from opf_api.registry import DetectorEntry


class FakeDetector:
    name = "fake"

    def detect(self, text: str, **_: Any) -> dict:
        spans: list[dict] = []
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


class SlowDetector:
    """Sleeps `delay_s` per call — used to test request timeout."""

    name = "slow"

    def __init__(self, delay_s: float) -> None:
        self.delay_s = delay_s

    def detect(self, text: str, **_: Any) -> dict:
        import time

        time.sleep(self.delay_s)
        return {"spans": [], "latency_ms": self.delay_s * 1000, "error": None}


def build_test_app(
    *,
    api_keys: str = "test-key",
    auth_disabled: bool = False,
    extra_env: dict[str, str] | None = None,
    detectors: dict[str, Any] | None = None,
    default: str = "fake",
) -> FastAPI:
    """Construct a hardened app for tests. Uses `with_lifespan=False` to
    skip OPF model load; injects the stub registry afterward."""
    os.environ["API_KEYS"] = api_keys
    os.environ["AUTH_DISABLED"] = "1" if auth_disabled else "0"
    for k, v in (extra_env or {}).items():
        os.environ[k] = v

    app = create_app(with_lifespan=False)
    instances = detectors or {"fake": FakeDetector()}
    registry: dict[str, DetectorEntry] = {}
    for name, inst in instances.items():
        entry = DetectorEntry(name=name, factory=lambda i=inst: i)
        entry.instance = inst
        registry[name] = entry
    app.state.registry = registry
    app.state.default_detector = default
    app.state.eager_load = list(registry)
    app.state.schema_version = SCHEMA_VERSION
    return app


def auth_headers(key: str = "test-key") -> dict[str, str]:
    return {"x-api-key": key, "content-type": "application/json"}
