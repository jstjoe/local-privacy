# Privacy-detection API — overview

The `opf-api` service exposes every benchmark detector in this repo behind one HTTP contract. Choose a backend with the `detector` field on each request; canonical labels apply uniformly across all of them.

## What it does

- `POST /api/find` — find sensitive data; return spans without rewriting the input text.
- `POST /api/replace` — find spans and replace each one under one of four modes.
- `GET /api/detectors` — list registered detectors and their canonical category coverage.
- `GET /api/health` — liveness probe.

## Reference UIs

The same OpenAPI spec powers three in-app reference UIs:

- `/scalar` — Scalar, modern three-pane reference with curl/JS/Python samples.
- `/docs` — Swagger UI with try-it-out.
- `/redoc` — ReDoc, classic read-only three-pane.

For the hosted public reference, see the GitHub Pages site rooted at this `docs/` directory.

## Running locally

```sh
DEFAULT_DETECTOR=opf EAGER_LOAD=opf uvicorn opf_api.main:app --reload
```

Then visit one of `/scalar`, `/docs`, `/redoc`. The OpenAPI spec itself lives at `/openapi.json` on the running server, and a frozen copy is checked in at [`api/openapi.json`](../api/openapi.json) (regenerated via `uv run --package opf-api opf-api-export-openapi --out docs/api`).

## Versioning

The API is pre-`1.0`. `info.version` (in this OpenAPI spec) is the only version surface today — currently `0.5.0`. Each breaking change to request or response shape lands as a minor bump (`0.x.0 -> 0.(x+1).0`); non-breaking additions land as patch bumps.

URL paths are unversioned (`/api/...`). When the API stabilizes, the intent is to move to **header-based date-string versioning** in the style of Stripe — clients will pin to a release date via `API-Version: 2026-05-13`. Until that header lands, treat `info.version` as the contract.
