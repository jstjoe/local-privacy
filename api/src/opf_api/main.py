from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

# Triton has no stable Apple Silicon support — set before any opf imports.
os.environ.setdefault("OPF_MOE_TRITON", "0")

from fastapi import FastAPI  # noqa: E402
from scalar_fastapi import get_scalar_api_reference  # noqa: E402

from .registry import build_default_registry  # noqa: E402
from .routes import router  # noqa: E402
from .vault_tokens import TokenVaultClient  # noqa: E402


logger = logging.getLogger("opf_api")
logging.basicConfig(level=logging.INFO)


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
    app.state.token_vault_client = TokenVaultClient.from_env()
    if app.state.token_vault_client is None:
        logger.info(
            "token vault not configured; /api/replace 'label_token' mode will 400"
        )
    else:
        logger.info("token vault configured for /api/replace 'label_token' mode")

    eager = os.environ.get("EAGER_LOAD", default)
    eager_names = [n.strip() for n in eager.split(",") if n.strip()]
    for name in eager_names:
        if name not in registry:
            logger.warning("EAGER_LOAD: unknown detector %r, skipping", name)
            continue
        await registry[name].get()
    logger.info("default=%s eager-loaded=%s", default, eager_names)
    try:
        yield
    finally:
        client = getattr(app.state, "token_vault_client", None)
        if client is not None:
            client.close()


API_DESCRIPTION = """\
Unified PII detection across **OPF**, **GLiNER**, **Presidio**, and **Skyflow**.
Pick a backend with the `detector` field; canonical labels apply uniformly across all of them.

## Endpoints

- `POST /api/find` — find sensitive data; return spans only, no text rewriting.
- `POST /api/replace` — find spans and replace each one under one of four modes.
- `GET /api/detectors` — list registered detectors and their category coverage.
- `GET /api/health` — liveness probe; does not exercise detector backends.

## Versioning

The API is pre-`1.0`. `info.version` is the only version surface today —
every breaking change to request or response shape lands as a minor bump
(`0.x.0 -> 0.(x+1).0`); non-breaking additions land as patch bumps.

URL paths are unversioned (`/api/...`). When the API stabilizes, the
intent is to move to **header-based date-string versioning** in the style
of Stripe — clients will pin to a release date via `API-Version:
2026-05-13`. Until that header lands, treat `info.version` as the contract.

## Canonical labels

Every detector's raw output maps into a 15-label taxonomy:
`PERSON`, `EMAIL`, `PHONE`, `ADDRESS`, `URL`, `DATE`, `ACCOUNT`, `SECRET`, `USERNAME`,
`DEMOGRAPHIC`, `ORGANIZATION`, `OCCUPATION`, `MONEY`, `VEHICLE`, `PHYSICAL`.
Detectors vary in coverage — `GET /api/detectors` reports each detector's category list.
"""

OPENAPI_TAGS = [
    {
        "name": "Find",
        "description": "Find-only endpoint. Returns spans without rewriting the input.",
    },
    {
        "name": "Replace",
        "description": (
            "Find plus rewriting under one of four modes: "
            "`redact`, `label`, `label_number`, `label_token`."
        ),
    },
    {
        "name": "Meta",
        "description": "Registry and liveness endpoints.",
    },
]

app = FastAPI(
    title="Privacy-detection API",
    version="0.5.0",
    summary="Unified PII detection across OPF, GLiNER, Presidio, and Skyflow.",
    description=API_DESCRIPTION,
    lifespan=lifespan,
    openapi_tags=OPENAPI_TAGS,
    contact={
        "name": "local-privacy",
        "url": "https://github.com/jstjoe/local-privacy",
    },
    servers=[
        {"url": "http://localhost:8000", "description": "Local dev"},
    ],
)
app.include_router(router, prefix="/api")


@app.get("/scalar", include_in_schema=False)
async def scalar_html():
    return get_scalar_api_reference(
        openapi_url=app.openapi_url,
        title=app.title,
    )
