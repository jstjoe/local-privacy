"""Structured JSON logs + request-ID middleware + Prometheus metrics.

Cloud Logging auto-parses one-JSON-per-line stdout into structured payloads,
so we configure structlog to emit JSON via the stdlib logger. `request_id`
is bound per-request via contextvars, so every log line on the request's
event loop task carries it without explicit threading.

Custom Prometheus metrics live in `METRICS` (counters + a histogram). The
built-in HTTP metrics (request count, duration) come from
prometheus-fastapi-instrumentator and are exposed at /metrics alongside ours.
"""

from __future__ import annotations

import logging
import sys
import time
import uuid

import structlog
from prometheus_client import Counter, Histogram
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request


def configure_logging(level: str = "INFO") -> None:
    """Wire structlog to emit JSON via the stdlib logger.

    Idempotent — safe to call from lifespan + tests.
    """
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=level,
        force=True,
    )
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level.upper(), logging.INFO)
        ),
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


logger = structlog.get_logger("opf_api")


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Generate per-request UUID, bind it to structlog contextvars, and emit
    one access-log line per response with status + latency."""

    async def dispatch(self, request: Request, call_next):
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex
        request.state.request_id = rid
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(
            request_id=rid,
            path=request.url.path,
            method=request.method,
        )
        t0 = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            logger.exception("request_failed")
            raise
        latency_ms = (time.perf_counter() - t0) * 1000
        response.headers["x-request-id"] = rid
        api_prefix = getattr(request.state, "api_key_prefix", None)
        logger.info(
            "request",
            status=response.status_code,
            latency_ms=round(latency_ms, 2),
            api_key_prefix=api_prefix,
        )
        return response


# Custom metrics. Built-in HTTP request count + duration come from
# prometheus-fastapi-instrumentator (registered in main.py).
METRIC_DETECTOR_CALLS = Counter(
    "pii_detector_calls_total",
    "PII detector invocations.",
    ["detector", "status"],
)
METRIC_DETECTOR_LATENCY = Histogram(
    "pii_detector_latency_seconds",
    "Per-detector inference latency.",
    ["detector"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0),
)
METRIC_SPANS_DETECTED = Counter(
    "pii_spans_detected_total",
    "Spans returned by detectors, by canonical label.",
    ["detector", "label"],
)


def record_detection(
    detector: str, status: str, latency_ms: float, labels: list[str]
) -> None:
    METRIC_DETECTOR_CALLS.labels(detector=detector, status=status).inc()
    METRIC_DETECTOR_LATENCY.labels(detector=detector).observe(latency_ms / 1000.0)
    for lbl in labels:
        METRIC_SPANS_DETECTED.labels(detector=detector, label=lbl).inc()
