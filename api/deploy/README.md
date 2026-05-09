# Cloud Run deployment

Single-region deploy of the Privacy-detection API. Cloud Run executes the OPF model on CPU; current latency (~515 ms p50 / 922 ms p99) is acceptable. Skyflow remains an upstream proxy.

## One-time setup

### 1. GCP project + Artifact Registry repo

```sh
export PROJECT_ID=your-gcp-project
export REGION=us-central1

gcloud artifacts repositories create privacy \
  --repository-format=docker \
  --location=$REGION \
  --description="Privacy-detection API images"
```

### 2. Secrets in Secret Manager

```sh
# API keys (comma-separated). Rotation = bump version, redeploy revision.
echo -n "key1,key2,key3" | gcloud secrets create privacy-api-keys --data-file=-

# Skyflow credentials (only needed if the skyflow detector is in the image).
echo -n "<bearer>"      | gcloud secrets create skyflow-token --data-file=-
echo -n "<vault url>"   | gcloud secrets create skyflow-vault-url --data-file=-
echo -n "<vault id>"    | gcloud secrets create skyflow-vault-id --data-file=-
```

Grant the service's runtime SA `roles/secretmanager.secretAccessor`.

### 3. Workload Identity Federation for GitHub Actions

```sh
gcloud iam workload-identity-pools create github \
  --location=global --display-name="GitHub Actions"

gcloud iam workload-identity-pools providers create-oidc github-provider \
  --location=global \
  --workload-identity-pool=github \
  --display-name="GitHub OIDC" \
  --attribute-mapping="google.subject=assertion.sub,attribute.repository=assertion.repository" \
  --issuer-uri="https://token.actions.githubusercontent.com"

# Service account that pushes to Artifact Registry
gcloud iam service-accounts create gh-deployer

gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:gh-deployer@$PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/artifactregistry.writer"

# Bind the GitHub repo to the SA
PROJECT_NUMBER=$(gcloud projects describe $PROJECT_ID --format="value(projectNumber)")
gcloud iam service-accounts add-iam-policy-binding \
  gh-deployer@$PROJECT_ID.iam.gserviceaccount.com \
  --role="roles/iam.workloadIdentityUser" \
  --member="principalSet://iam.googleapis.com/projects/$PROJECT_NUMBER/locations/global/workloadIdentityPools/github/attribute.repository/<owner>/<repo>"
```

GitHub Actions secrets to set in the repo:
- `GCP_PROJECT_ID`
- `GCP_WIF_PROVIDER` — `projects/$PROJECT_NUMBER/locations/global/workloadIdentityPools/github/providers/github-provider`
- `GCP_SERVICE_ACCOUNT` — `gh-deployer@$PROJECT_ID.iam.gserviceaccount.com`

## Build + push (per release)

Tag a release; `release.yml` builds and pushes the full image.

```sh
git tag v0.3.0 && git push --tags
```

Or trigger manually from the Actions UI (`workflow_dispatch`).

## Deploy

Substitute placeholders in `cloudrun.yaml` and apply:

```sh
sed -e "s|PROJECT_ID|$PROJECT_ID|g" \
    -e "s|REGION|$REGION|g" \
    -e "s|IMAGE_TAG|v0.3.0|g" \
    api/deploy/cloudrun.yaml > /tmp/cloudrun.rendered.yaml

gcloud run services replace /tmp/cloudrun.rendered.yaml --region=$REGION
```

Or use the imperative form (equivalent settings):

```sh
gcloud run deploy privacy-api \
  --image=$REGION-docker.pkg.dev/$PROJECT_ID/privacy/api:v0.3.0 \
  --region=$REGION \
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

## Knobs explained

| flag | why |
|---|---|
| `--memory=4Gi` | OPF needs ~3 GB resident; 4 GiB leaves headroom. Bump to 8 GiB if eager-loading multiple detectors. |
| `--cpu=2 --no-cpu-throttling` | Throttling halves CPU when idle, which destroys OPF p99. Required for steady latency. |
| `--concurrency=4` | The detector instance has a `call_lock`; >1 still helps because async overlap covers JSON parsing/serialization. Tune by watching queue depth. |
| `--min-instances=1` | Keeps OPF warm. Cold start = checkpoint reload (~30s). ~$30-50/mo for one always-on 4 GiB / 2 vCPU instance. |
| `--max-instances=10` | Caps spend. |
| `--timeout=60` | Cloud Run request timeout. Must exceed app `REQUEST_TIMEOUT_SECONDS` so the app returns 504 before Cloud Run kills the request. |

## Rate-limit caveat

slowapi uses in-memory storage, so each Cloud Run instance has its own counter. With `--max-instances=10` and `RATE_LIMIT_REDACT=30/minute`, the true ceiling is `10 * 30 = 300/minute` per key. For tighter enforcement, swap to a Redis (Memorystore) backend — out of scope for this iteration.

## Observability

- **Logs**: structlog emits one JSON line per record to stdout. Cloud Logging auto-parses. Filter on `jsonPayload.request_id` to trace a single request, `jsonPayload.detector` to slice by backend.
- **Metrics**: `/metrics` exposes Prometheus format. Custom counters: `pii_detector_calls_total`, `pii_detector_latency_seconds`, `pii_spans_detected_total`. To scrape from Cloud Run, deploy [Managed Service for Prometheus](https://cloud.google.com/stackdriver/docs/managed-prometheus) sidecar — or skip and rely on Cloud Run's built-in request count + latency until needed.

## Rollback

Cloud Run keeps every revision. Roll back with:

```sh
gcloud run services update-traffic privacy-api \
  --to-revisions=privacy-api-00007-abc=100 \
  --region=$REGION
```

(get revision names from `gcloud run revisions list --service=privacy-api`)
