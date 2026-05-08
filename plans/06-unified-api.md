# Plan 06 — Unified privacy-detection API

## Why

The current `api/` server wraps a single detector (OPF) with an OPF-specific request contract — `categories` accepts raw OPF labels (`private_email`, `private_person`) and there's no way to swap backends. As we add more detectors (GLiNER, Presidio, Skyflow, OpenMed, etc.) the right abstraction is one API service that exposes a single contract and routes per request to whichever backend the deployer configures.

Goal: **same client code works regardless of which detector is doing the work**. Switch from OPF to Skyflow by changing one query parameter or env var.

## Design

### One contract, canonical labels

Request:

```http
POST /v1/redact
content-type: application/json

{
  "text": "Joe at joe@example.com lives in Elgin, TX.",
  "detector": "skyflow_minimal",
  "categories": ["EMAIL", "PHONE", "DATE"],
  "decode_mode": "viterbi"
}
```

- `text` — required
- `detector` — optional; falls back to `DEFAULT_DETECTOR` env var (e.g. `"opf"`)
- `categories` — optional canonical-label allow-list (`PERSON`, `EMAIL`, `PHONE`, `ADDRESS`, `URL`, `DATE`, `ACCOUNT`, `SECRET`, `USERNAME`, `DEMOGRAPHIC`). When omitted, all categories the detector supports are returned.
- `decode_mode` — optional, OPF-only; ignored by other detectors

Response:

```json
{
  "schema_version": 1,
  "detector": "skyflow_minimal",
  "text": "Joe at joe@example.com lives in Elgin, TX.",
  "detected_spans": [
    {"label": "EMAIL", "raw_label": "EMAIL_ADDRESS", "start": 7, "end": 22, "text": "joe@example.com", "placeholder": "[EMAIL]"}
  ],
  "redacted_text": "Joe at [EMAIL] lives in Elgin, TX.",
  "summary": {"span_count": 1, "by_label": {"EMAIL": 1}},
  "warning": null
}
```

`label` is always canonical. `raw_label` preserves the detector-native label so downstream auditors can trace where each detection came from.

### Backends, all behind the same `Detector` protocol

Reuse [eval/src/opf_eval/detectors/base.py](../eval/src/opf_eval/detectors/base.py) — it already canonicalizes labels and returns the shape we want.

| backend | mode | notes |
| --- | --- | --- |
| `opf` | in-process | loads checkpoint at boot or first call |
| `gliner` | in-process | downloads HF model on first use |
| `presidio` | in-process | needs spaCy model(s) installed |
| `skyflow` / `skyflow_minimal` / `skyflow_full` | **proxy** | API forwards request to Skyflow Detect, translates response |

Skyflow is the only detector where the API is just a proxy — credentials read from env at boot, requests issued via `httpx` per call.

### Detector registry

`api/src/opf_api/registry.py`:

```python
@dataclass
class DetectorEntry:
    name: str
    factory: Callable[[], Detector]   # lazy
    loaded: bool = False
    instance: Detector | None = None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
```

A central registry maps detector names to factories. Lazy-load on first request — avoids paying boot-time for detectors nobody asks for.

```python
REGISTRY: dict[str, Callable[[], Detector]] = {
    "opf": lambda: OPFDetector(device=os.environ.get("OPF_DEVICE", "cpu")),
    "skyflow": lambda: SkyflowDetector(),
    "skyflow_minimal": lambda: SkyflowDetector(entity_types=list(SKYFLOW_MINIMAL_ENTITY_TYPES)),
    "presidio": lambda: PresidioDetector(),
    "gliner": lambda: GLiNERDetector(),
    # add new detectors here as plan 05 ships
}
```

## Files

- **Modify** [api/src/opf_api/main.py](../api/src/opf_api/main.py) — replace single-OPF lifespan with detector-registry init
- **Modify** [api/src/opf_api/routes.py](../api/src/opf_api/routes.py) — accept `detector` field, dispatch via registry, canonical-label filtering
- **Modify** [api/src/opf_api/schemas.py](../api/src/opf_api/schemas.py) — add `detector` and canonical-label types
- **Add** [api/src/opf_api/registry.py](../api/src/opf_api/registry.py) — detector factory + lazy-load logic
- **Modify** [api/pyproject.toml](../api/pyproject.toml) — add deps from `eval/` (httpx, presidio-analyzer, gliner, etc.) or restructure to import the eval package directly
- **Modify** [api/Dockerfile](../api/Dockerfile) — bake spaCy models, optional eager-load env vars, document size implications

## Routes

### `POST /v1/detect`

Returns detected spans, no redaction. Same body shape as `/v1/redact` minus the `placeholder` and `redacted_text` fields in the response.

### `POST /v1/redact`

Detect + redact + return both. Default placeholder format `[CATEGORY]` is detector-agnostic; OPF's native placeholders (`<PRIVATE_EMAIL>`) used only when caller passes `placeholder_format=opf_native` (optional).

### `GET /v1/detectors`

Lists available detectors and their canonical-category coverage:

```json
{
  "default": "opf",
  "detectors": [
    {"name": "opf", "categories": ["PERSON","EMAIL","PHONE","ADDRESS","URL","DATE","ACCOUNT","SECRET"], "loaded": true},
    {"name": "gliner", "categories": ["PERSON","EMAIL","PHONE","ADDRESS","URL","DATE","ACCOUNT","SECRET","USERNAME"], "loaded": false},
    {"name": "skyflow_minimal", "categories": [...], "loaded": false, "proxy": true}
  ]
}
```

Coverage data lives in `taxonomy.CANONICAL_MAP` already.

### `GET /v1/health`

```json
{
  "status": "ok",
  "default_detector": "opf",
  "loaded_detectors": ["opf"],
  "schema_version": 1
}
```

## Concurrency model

- **In-process detectors** (OPF, GLiNER, Presidio): single instance per process, wrapped in an `asyncio.Lock`. Same pattern as today.
- **Skyflow proxy**: stateless from the server's perspective. Reuse a single `httpx.AsyncClient` with a connection pool sized via `SKYFLOW_MAX_CONNECTIONS` env var.
- For multi-CPU containers wanting parallelism, spawn multiple worker processes (`--workers N` to uvicorn). Each gets its own detector instance.

## Configuration via env

| env var | meaning | default |
| --- | --- | --- |
| `DEFAULT_DETECTOR` | name returned when client omits `detector` | `opf` |
| `EAGER_LOAD` | comma-list of detectors to load at boot | `${DEFAULT_DETECTOR}` |
| `OPF_DEVICE` | OPF inference device | `cpu` |
| `OPF_DECODE_MODE` | OPF decoder mode | `viterbi` |
| `SKYFLOW_VAULT_URL` / `SKYFLOW_VAULT_ID` / `SKYFLOW_BEARER_TOKEN` | Skyflow auth | required if any `skyflow*` detector is hit |
| `SKYFLOW_MAX_CONNECTIONS` | httpx pool size | `10` |
| `OPF_MOE_TRITON` | Triton kernel toggle | `0` |

## Container profiles

Big-container approach: single image with all four local backends + spaCy + checkpoints baked in. ~6-8 GB. Useful for "one detector to rule them all" deployments.

Slim profiles via build args:

```dockerfile
ARG INCLUDE_OPF=1
ARG INCLUDE_GLINER=0
ARG INCLUDE_PRESIDIO=0
RUN if [ "$INCLUDE_OPF" = "1" ]; then python -c "from opf._common.checkpoint_download import ensure_default_checkpoint; ensure_default_checkpoint()"; fi
RUN if [ "$INCLUDE_GLINER" = "1" ]; then python -c "from gliner import GLiNER; GLiNER.from_pretrained('urchade/gliner_multi_pii-v1')"; fi
```

So `docker build --build-arg INCLUDE_PRESIDIO=1 --build-arg INCLUDE_OPF=0 ...` produces a Presidio-only image. Skyflow is always available since it's just an HTTP client.

## Risks / open questions

- **Cold-start latency on lazy load.** First request to an un-loaded detector might wait 5-30s for OPF to load. Document and recommend `EAGER_LOAD` for production. Consider returning a 503 + Retry-After header during load, with the caller polling `/v1/health` until ready.
- **Memory pressure when multiple detectors are loaded simultaneously.** OPF (2.8 GB) + GLiNER (~500 MB) + Presidio + spaCy (~1 GB) plus model overhead can exceed 8 GB. Document expected RAM. The `EAGER_LOAD` env var lets deployers control this.
- **Concurrent inference on the same in-process detector.** The asyncio.Lock serializes calls. For high-throughput single-detector deploys, run multiple uvicorn workers — but each worker holds a full copy of the model. Tradeoff between latency and memory.
- **Skyflow auth rotation.** Bearer tokens expire (~24h typical). Either (a) require restart, (b) re-read env on each request (slow), or (c) add a webhook for token refresh. Start with (a), document.
- **Versioning.** Move from current unversioned `/redact` to `/v1/redact` since the contract changes. Keep the old routes as deprecated aliases for one release.
- **Decoder-specific options.** `decode_mode` only applies to OPF; ignored by others. Document. Skyflow has its own knobs (`token_type`, `transformations`) that we'd need a different field for if ever exposed.
- **`presidio_multilang` adds spacy_model dependencies** — image grows by ~3 GB if all 6 spaCy models are baked in. Consider only-English by default in the container.

## Implementation order

1. **Extract the eval `Detector` protocol** as the shared contract — confirm it's importable from `api/` (may need to convert to a separate `opf-detectors` shared package, or just import directly from `opf_eval` with a workspace dep)
2. **Build the registry** with lazy load + lock
3. **Refactor routes** to accept `detector` + canonical `categories`, dispatch via registry
4. **Add `/v1/detectors` and `/v1/health`** endpoints
5. **Update Dockerfile** with build args for slim profiles
6. **Backwards-compat layer** — keep old `/redact` and `/detect` routing to the default detector + old (lowercase OPF) categories filter, with a deprecation warning header
7. **Tests** — at minimum, an integration test per detector using `httpx.AsyncClient` against `app` directly

## Effort

- Steps 1-2 (extract + registry): ~3 hours
- Step 3 (route refactor + canonical filter): ~3 hours
- Step 4 (info routes): ~1 hour
- Step 5 (Dockerfile): ~2 hours
- Step 6 (back-compat): ~1 hour
- Step 7 (tests): ~2 hours
- **Total: ~1-1.5 days**

## Out of scope

- AuthN/AuthZ (handle in reverse proxy / API gateway)
- Rate limiting (same)
- Batch endpoint (`POST /v1/redact/batch`) — separate plan if needed
- Streaming responses for long inputs
- WebSocket interface
- Reversible tokenization / re-identification (Skyflow has it, OPF doesn't, GLiNER doesn't — would need its own design)
- Multi-tenant config (one set of detectors per deployer; no per-tenant auth)
- Model fine-tuning via the API (separate concern; train offline, deploy)
