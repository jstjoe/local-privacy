from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

# Triton has no stable Apple Silicon support — set before any opf imports.
os.environ.setdefault("OPF_MOE_TRITON", "0")

from fastapi import FastAPI  # noqa: E402

from .registry import build_default_registry  # noqa: E402
from .routes import router  # noqa: E402
from .routes_legacy import legacy_router  # noqa: E402
from .vault_tokens import TokenVaultClient  # noqa: E402


logger = logging.getLogger("opf_api")
logging.basicConfig(level=logging.INFO)


SCHEMA_VERSION = 1


@asynccontextmanager
async def lifespan(app: FastAPI):
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
    app.state.token_vault_client = TokenVaultClient.from_env()
    if app.state.token_vault_client is None:
        logger.info(
            "token vault not configured; /v1/tokenize 'vault_token' mode will 400"
        )
    else:
        logger.info("token vault configured for /v1/tokenize 'vault_token' mode")

    eager = os.environ.get("EAGER_LOAD", default)
    eager_names = [n.strip() for n in eager.split(",") if n.strip()]
    for name in eager_names:
        if name not in registry:
            logger.warning("EAGER_LOAD: unknown detector %r, skipping", name)
            continue
        await registry[name].get()
    logger.info("default=%s eager-loaded=%s", default, eager_names)
    yield


app = FastAPI(
    title="Privacy-detection API",
    version="0.2.0",
    lifespan=lifespan,
    description=(
        "Unified PII detection across OPF, GLiNER, Presidio, and Skyflow. "
        "Pick a backend with the `detector` field; canonical labels apply uniformly."
    ),
)
app.include_router(router, prefix="/v1")
app.include_router(legacy_router)
