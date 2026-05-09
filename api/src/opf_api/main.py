from __future__ import annotations

import os
from contextlib import asynccontextmanager

# Triton has no stable Apple Silicon support — set before any opf imports.
os.environ.setdefault("OPF_MOE_TRITON", "0")

from fastapi import FastAPI  # noqa: E402
from prometheus_fastapi_instrumentator import Instrumentator  # noqa: E402
from slowapi import _rate_limit_exceeded_handler  # noqa: E402
from slowapi.errors import RateLimitExceeded  # noqa: E402

from .auth import APIKeyMiddleware, load_api_keys  # noqa: E402
from .limits import MaxBodySizeMiddleware, max_body_bytes  # noqa: E402
from .observability import (  # noqa: E402
    RequestContextMiddleware,
    configure_logging,
    logger,
)
from .rate_limit import limiter  # noqa: E402
from .registry import build_default_registry  # noqa: E402
from .routes import router  # noqa: E402
from .routes_legacy import legacy_router  # noqa: E402
from .shutdown import (  # noqa: E402
    InflightMiddleware,
    InflightTracker,
    drain_seconds,
)


SCHEMA_VERSION = 1


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging(os.environ.get("LOG_LEVEL", "INFO"))

    registry = build_default_registry()
    default = os.environ.get("DEFAULT_DETECTOR", "opf")
    if default not in registry:
        raise RuntimeError(
            f"DEFAULT_DETECTOR={default!r} not in registry. "
            f"Available: {sorted(registry)}"
        )
    app.state.registry = registry
    app.state.default_detector = default
    app.state.schema_version = SCHEMA_VERSION

    eager = os.environ.get("EAGER_LOAD", default)
    eager_names = [n.strip() for n in eager.split(",") if n.strip()]
    app.state.eager_load = eager_names
    for name in eager_names:
        if name not in registry:
            logger.warning("eager_load_unknown", detector=name)
            continue
        await registry[name].get()
    logger.info("startup_complete", default=default, eager_loaded=eager_names)
    try:
        yield
    finally:
        timeout = drain_seconds()
        logger.info("draining", timeout_s=timeout, inflight=app.state.inflight.count)
        drained = await app.state.inflight.drain(timeout)
        logger.info("drained", clean=drained, inflight=app.state.inflight.count)


def create_app(*, with_lifespan: bool = True) -> FastAPI:
    """Build the app and wire middlewares. Tests pass `with_lifespan=False`
    to skip the OPF model load and inject a stub registry directly into
    `app.state` after construction."""
    app = FastAPI(
        title="Privacy-detection API",
        version="0.3.0",
        lifespan=lifespan if with_lifespan else None,
        description=(
            "Unified PII detection across OPF, GLiNER, Presidio, and Skyflow. "
            "Pick a backend with the `detector` field; canonical labels apply uniformly."
        ),
    )

    tracker = InflightTracker()
    app.state.inflight = tracker

    valid_keys = load_api_keys()
    auth_disabled = os.environ.get("AUTH_DISABLED") == "1"
    if not valid_keys and not auth_disabled:
        # Fail loud — silently allowing all traffic is the worst outcome.
        raise RuntimeError(
            "No API_KEYS configured. Set API_KEYS=key1,key2,... or AUTH_DISABLED=1 for dev."
        )

    # Middleware order matters — outer wraps inner. Starlette executes them
    # in reverse-add order on the request, then forward on the response.
    # Outermost first (top of stack = runs on request first):
    #   1. RequestContextMiddleware (binds request_id before anything else)
    #   2. InflightMiddleware       (track + 503 on drain)
    #   3. MaxBodySizeMiddleware    (cheap reject before parsing)
    #   4. APIKeyMiddleware         (auth gate)
    # Add in reverse since add_middleware prepends.
    app.add_middleware(
        APIKeyMiddleware, valid_keys=valid_keys, disabled=auth_disabled
    )
    app.add_middleware(MaxBodySizeMiddleware, max_bytes=max_body_bytes())
    app.add_middleware(InflightMiddleware, tracker=tracker)
    app.add_middleware(RequestContextMiddleware)

    # slowapi uses app.state.limiter for rule lookup at request time.
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    # Prometheus /metrics — auth-skipped (skip_paths in auth middleware).
    Instrumentator().instrument(app).expose(app, include_in_schema=False)

    app.include_router(router, prefix="/v1")
    app.include_router(legacy_router)
    return app


app = create_app()
