# Plan 07 — Production hardening for Cloud Run

## Why

Plan 06 shipped the unified API; this one makes it deployable. Target: **Google Cloud Run** with the OPF model running on CPU (current CPU latency — 515 ms p50 / 922 ms p99 — is acceptable per stakeholder). Skyflow continues to be a proxy. We use a stable Skyflow API key, so token rotation is out of scope here.

Goal: turn the current single-tenant FastAPI app into something a small team can rely on — with auth, request-shape guardrails, structured logs, basic metrics, real readiness, basic abuse protection, an automated test+build pipeline, and clean shutdown.

## Scope

In:
1. **Auth** — API-key middleware (X-API-Key header), keys from Secret Manager
2. **Input limits** — max text length + per-request timeout + max body size
3. **Observability** — JSON logs (Cloud Logging picks up automatically) + per-request ID + Prometheus metrics endpoint
4. **Readiness probe** — split `/v1/health` (liveness) from `/v1/ready` (eager-load detectors actually loaded; Skyflow auth check)
5. **Rate limiting** — slowapi, per-API-key buckets, in-memory (per-instance), document the limitation
6. **CI** — GitHub Actions: ruff, pytest, Docker build smoke test, Artifact Registry push on tag
7. **Graceful shutdown** — drain in-flight requests on SIGTERM, bounded by a deadline

Out:
- Skyflow bearer-token rotation (using stable API key)
- Auth beyond static API keys (no IAM, no Cloud Endpoints) — defer
- GPU deployment for OPF — defer (CPU is acceptable)
- Distributed rate limiting (would need Redis) — document the per-instance caveat
- Custom domain / TLS — Cloud Run handles this natively
- Multi-region — single region for v1

## Layout

```
api/
├── src/opf_api/
│   ├── auth.py          # NEW — APIKeyMiddleware
│   ├── observability.py # NEW — JSON logger, request-id middleware, metrics
│   ├── limits.py        # NEW — body size + timeout middleware/dependency
│   ├── rate_limit.py    # NEW — slowapi setup (per-key buckets)
│   ├── shutdown.py      # NEW — InflightTracker + graceful drain
│   ├── main.py          # MODIFY — wire middlewares + lifespan drain
│   ├── routes.py        # MODIFY — split /health vs /ready, attach rate limits
│   └── schemas.py       # MODIFY — text max_length on RedactRequest
├── tests/
│   ├── test_routes.py        # existing
│   ├── test_auth.py          # NEW
│   ├── test_limits.py        # NEW
│   └── test_rate_limit.py    # NEW
├── Dockerfile           # MODIFY — listen on $PORT, healthcheck
├── pyproject.toml       # MODIFY — slowapi, prometheus-fastapi-instrumentator, structlog
└── deploy/
    ├── cloudrun.yaml    # NEW — service spec template
    └── README.md        # NEW — gcloud deploy commands

.github/
└── workflows/
    ├── ci.yml           # NEW — lint + test on PR
    └── release.yml      # NEW — build + push image on tag
```

## 1. Auth — API-key middleware

`api/src/opf_api/auth.py`:

```python
class APIKeyMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, *, valid_keys: set[str], skip_paths: set[str]):
        super().__init__(app)
        self.valid_keys = valid_keys
        self.skip_paths = skip_paths   # /v1/health, /v1/ready, /metrics

    async def dispatch(self, request, call_next):
        if request.url.path in self.skip_paths:
            return await call_next(request)
        key = request.headers.get("x-api-key")
        if not key or key not in self.valid_keys:
            return JSONResponse({"detail": "unauthorized"}, status_code=401)
        request.state.api_key = key  # for rate limiting + logs
        return await call_next(request)
```

Configuration:

| env | meaning |
| --- | --- |
| `API_KEYS` | comma-list of accepted keys (loaded from Secret Manager) |
| `AUTH_DISABLED` | dev-only escape hatch |

For Cloud Run we mount the secret as `API_KEYS` env via `--set-secrets API_KEYS=privacy-api-keys:latest`. Rotation = bump the secret version, redeploy revision.

## 2. Input limits

Three layers:

**a. Pydantic on the schema** — declarative, returns 422:

```python
class RedactRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=100_000)  # 100 KB cap
    # ... existing fields
```

**b. Body size middleware** — kills oversized requests before Pydantic parses them:

```python
class MaxBodySizeMiddleware(BaseHTTPMiddleware):
    """Reject Content-Length > MAX_BODY_BYTES with 413."""
```

Default 200 KB (text + JSON overhead). Configurable via `MAX_BODY_BYTES` env.

**c. Per-request timeout** — wraps the detector call:

```python
async with asyncio.timeout(REQUEST_TIMEOUT_SECONDS):
    result = await asyncio.to_thread(detector.detect, text)
```

Returns 504 on timeout. Default 30s. Configurable via `REQUEST_TIMEOUT_SECONDS`.

## 3. Observability — logs + metrics

**Structured JSON logs** via `structlog` configured to emit one JSON line per record. Cloud Logging auto-parses these into structured payloads. Fields: `timestamp`, `severity`, `request_id`, `api_key_prefix` (first 6 chars), `detector`, `latency_ms`, `span_count`, `path`, `status`.

**Request-ID middleware** — generates a UUID per request, attaches to `request.state.request_id`, sets `X-Request-ID` response header, includes in every log line (via `structlog.contextvars`).

**Prometheus metrics** via `prometheus-fastapi-instrumentator`:

- `http_requests_total{path,method,status}` (built-in)
- `http_request_duration_seconds_bucket{path,method}` (built-in)
- `pii_detector_calls_total{detector,status}` (custom)
- `pii_detector_latency_seconds_bucket{detector}` (custom)
- `pii_spans_detected_total{detector,label}` (custom)

Exposed at `/metrics` (auth-skipped). Cloud Run can scrape via [Managed Service for Prometheus sidecar](https://cloud.google.com/stackdriver/docs/managed-prometheus/setup-managed#self-deployed-cloudrun); easier alternative for v1: enable `--add-metrics-collector` (auto-collected request metrics) and skip Prometheus for now. Pick one when implementing.

## 4. Readiness probe split

```
GET /v1/health   -> liveness    (always 200 if process is up)
GET /v1/ready    -> readiness   (200 only when:
                                   - all EAGER_LOAD detectors loaded
                                   - if Skyflow registered: bearer credential present)
```

Cloud Run startup probe + liveness probe wired to these:

```yaml
startupProbe:
  httpGet: { path: /v1/ready, port: 8080 }
  failureThreshold: 30   # 30 * 10s = 5min for OPF to load
  periodSeconds: 10
livenessProbe:
  httpGet: { path: /v1/health, port: 8080 }
  periodSeconds: 30
```

OPF load takes ~30-60s on a Cloud Run cold-start; the startup probe gives it 5 minutes before declaring failure.

## 5. Rate limiting

`slowapi` keyed on the API key (or IP if anonymous endpoint):

```python
limiter = Limiter(key_func=lambda r: r.state.api_key, default_limits=["60/minute"])
```

Per-route overrides:

```python
@router.post("/redact")
@limiter.limit("30/minute")
async def redact(request: Request, body: RedactRequest):
    ...
```

Configurable via env (`RATE_LIMIT_REDACT`, `RATE_LIMIT_DETECT`). Returns 429 with `Retry-After` header.

**Caveat**: in-memory storage means each Cloud Run instance has its own counter. With `--max-instances=N`, true throughput cap is `N * limit`. Document this. For tighter limits, future plan: add Redis (Memorystore) backing.

## 6. Graceful shutdown

`shutdown.py`:

```python
class InflightTracker:
    def __init__(self) -> None:
        self._count = 0
        self._zero = asyncio.Event()
        self._zero.set()

    @asynccontextmanager
    async def track(self):
        self._count += 1
        self._zero.clear()
        try:
            yield
        finally:
            self._count -= 1
            if self._count == 0:
                self._zero.set()

    async def drain(self, timeout: float):
        try:
            await asyncio.wait_for(self._zero.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            pass
```

Wire via middleware. In `lifespan`:

```python
yield   # ... server runs ...
logger.info("draining in-flight requests (timeout=%ss)", drain_timeout)
await app.state.inflight.drain(drain_timeout)
```

Cloud Run sends SIGTERM with a 10s default grace period, configurable up to 600s with `--timeout`. Use `SHUTDOWN_DRAIN_SECONDS=20`, set Cloud Run grace period to match.

## 7. CI

`.github/workflows/ci.yml` (runs on every PR + push to main):

```yaml
jobs:
  lint:
    steps:
      - uses: astral-sh/setup-uv@v3
      - run: uv run ruff check .
      - run: uv run ruff format --check .
  test:
    strategy:
      matrix: { python-version: ["3.10", "3.11", "3.12"] }
    steps:
      - uses: astral-sh/setup-uv@v3
      - run: uv sync
      - run: uv run pytest api/tests/ -v
  docker-build:
    steps:
      - uses: docker/setup-buildx-action@v3
      - run: docker build -f api/Dockerfile --build-arg INCLUDE_OPF=0 --build-arg INCLUDE_GLINER=0 --build-arg INCLUDE_PRESIDIO=0 -t privacy-api:smoke .
      # Smoke (proxy-only image is small + builds in <2min). Real images built in release.yml.
```

`.github/workflows/release.yml` (runs on git tag `v*`):

```yaml
- auth to GCP via workload-identity-federation
- docker buildx build --platform linux/amd64 \
    --build-arg INCLUDE_OPF=1 \
    -t $REGION-docker.pkg.dev/$PROJECT/privacy/api:$TAG \
    --push .
```

Requires GCP Workload Identity setup (one-time): GCP service account for GitHub Actions, OIDC trust, IAM `artifactregistry.writer`. Document in `deploy/README.md`.

## Cloud Run deployment

Single service, "full" image (OPF + GLiNER + Presidio + Skyflow). Two-service split (slim + full) is the optimization plan if cost is an issue.

```sh
gcloud run deploy privacy-api \
  --image=$REGION-docker.pkg.dev/$PROJECT/privacy/api:latest \
  --region=us-central1 \
  --memory=4Gi \
  --cpu=2 \
  --concurrency=4 \
  --min-instances=1 \
  --max-instances=10 \
  --timeout=60 \
  --no-cpu-throttling \
  --set-env-vars=DEFAULT_DETECTOR=opf,EAGER_LOAD=opf,REQUEST_TIMEOUT_SECONDS=30,SHUTDOWN_DRAIN_SECONDS=20 \
  --set-secrets=API_KEYS=privacy-api-keys:latest,SKYFLOW_BEARER_TOKEN=skyflow-token:latest,SKYFLOW_VAULT_URL=skyflow-vault-url:latest,SKYFLOW_VAULT_ID=skyflow-vault-id:latest
```

Notes:
- `--memory=4Gi` — OPF needs ~3 GB (model + inference state); 4 GiB leaves headroom. Bump to 8 GiB if loading multiple detectors eagerly.
- `--cpu=2` + `--no-cpu-throttling` — required for OPF latency. Throttling halves CPU when idle which kills OPF p99.
- `--concurrency=4` — model has a `call_lock`; >1 is fine because of FastAPI's async dispatch but most calls serialize. 4 is a safe default; tune by measuring queue depth.
- `--min-instances=1` — keeps OPF warm. Cold start = checkpoint reload (~30s). Costs ~$30-50/mo for one always-on 4 GiB / 2 vCPU instance.
- `--max-instances=10` — caps spend. Adjust per traffic.
- `--timeout=60` — Cloud Run request timeout. Should exceed app `REQUEST_TIMEOUT_SECONDS` so the app handles timeouts (returns 504) before Cloud Run kills the request.

## Files

To create:

- [api/src/opf_api/auth.py](../api/src/opf_api/auth.py)
- [api/src/opf_api/observability.py](../api/src/opf_api/observability.py)
- [api/src/opf_api/limits.py](../api/src/opf_api/limits.py)
- [api/src/opf_api/rate_limit.py](../api/src/opf_api/rate_limit.py)
- [api/src/opf_api/shutdown.py](../api/src/opf_api/shutdown.py)
- [api/tests/test_auth.py](../api/tests/test_auth.py)
- [api/tests/test_limits.py](../api/tests/test_limits.py)
- [api/tests/test_rate_limit.py](../api/tests/test_rate_limit.py)
- [api/deploy/cloudrun.yaml](../api/deploy/cloudrun.yaml)
- [api/deploy/README.md](../api/deploy/README.md)
- [.github/workflows/ci.yml](../.github/workflows/ci.yml)
- [.github/workflows/release.yml](../.github/workflows/release.yml)

To modify:

- [api/src/opf_api/main.py](../api/src/opf_api/main.py) — wire middlewares, attach inflight tracker, drain on shutdown
- [api/src/opf_api/routes.py](../api/src/opf_api/routes.py) — split /health vs /ready, attach rate limits, wrap detect in timeout
- [api/src/opf_api/schemas.py](../api/src/opf_api/schemas.py) — add `max_length=100_000` to text
- [api/Dockerfile](../api/Dockerfile) — read `$PORT` env (Cloud Run sets it)
- [api/pyproject.toml](../api/pyproject.toml) — add `slowapi`, `prometheus-fastapi-instrumentator`, `structlog`
- [README.md](../README.md) — link to deploy/README.md

## Verification

1. `uv sync && uv run pytest api/tests/ -v` — all green
2. `API_KEYS=test-key DEFAULT_DETECTOR=opf EAGER_LOAD=opf uv run uvicorn opf_api.main:app --port 8765`
3. `curl localhost:8765/v1/redact -d '{"text":"x"}'` → 401 (no key)
4. `curl -H 'x-api-key: test-key' -H 'content-type: application/json' localhost:8765/v1/redact -d '{"text":"joe@example.com"}'` → 200, response carries `X-Request-ID` header
5. Send 31 requests in 60s → 30 OK + 1 with 429
6. `curl localhost:8765/v1/ready` → 200 only when OPF is loaded
7. `curl localhost:8765/metrics` → Prometheus exposition with custom counters
8. `kill -TERM <uvicorn_pid>` while a request is mid-flight → request finishes; subsequent /v1/redact rejected with 503
9. `docker build -f api/Dockerfile --build-arg INCLUDE_OPF=1 -t privacy-api:test .` succeeds
10. CI run on a PR shows green lint + test + build jobs

## Effort

- Step 1 auth: ~2 hours (middleware + tests)
- Step 2 input limits: ~1 hour (schema + middleware + timeout)
- Step 3 observability: ~3 hours (structlog wiring + metrics + request-id)
- Step 4 readiness split: ~1 hour
- Step 5 rate limit: ~2 hours
- Step 6 CI: ~3 hours (lint + test workflow + workload-identity setup is the slow part)
- Step 7 graceful shutdown: ~2 hours
- Cloud Run deploy + secrets: ~2 hours
- **Total: ~2 days** of focused work; could be split across two PRs (everything but CI/deploy in PR1, CI/deploy in PR2).

## Out of scope (deferred)

- Skyflow token rotation (using stable API key)
- IAM-based service-to-service auth (Cloud Endpoints / API Gateway)
- Distributed rate limiting (Redis-backed)
- Multi-region deployment
- GPU deployment for OPF
- Per-tenant config (one set of detectors per service)
- Reversible tokenization
- Custom domains (Cloud Run auto-issues `*.run.app`; map a domain when needed)
