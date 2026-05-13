# Privacy-detection API — overview

The `opf-api` service exposes every benchmark detector in this repo behind one HTTP contract. Choose a backend with the `detector` field on each request; canonical labels apply uniformly across all of them.

## What it does

- `POST /v1/detect` — return PII spans without rewriting the input text.
- `POST /v1/sanitize` — detect and rewrite spans under one of four modes.
- `GET /v1/detectors` — list registered detectors and their canonical category coverage.
- `GET /v1/health` — liveness probe.

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

Two version numbers appear in this API. They are independent.

- `info.version` (in this OpenAPI spec) tracks the contract. Currently `0.2.0`.
- `schema_version` (in every response payload) tracks request/response payload shape. Currently `2`. Clients should pin on this.

A future field rename or semantic shift bumps `schema_version`. A new endpoint, new field, or doc-only change does not.
