# Detectors

Detectors are registered backends, lazy-loaded on first use unless eager-loaded at startup.

| Name | Backend | Notes |
|---|---|---|
| `opf` | OPF local model, ~2.8 GB | Trained categories only. |
| `skyflow` | HTTP proxy to Skyflow Detect API | Requires `SKYFLOW_VAULT_*` env. `proxy: true` in `/v1/detectors`. |
| `presidio` | Microsoft Presidio, English spaCy model | Registered only if `presidio-analyzer` is installed. |
| `presidio_multilang` | Presidio with all 6 spaCy languages | Each `<lang>_core_news_lg` model must be installed. |
| `gliner` | `urchade/gliner_multi_pii-v1`, multilingual | Registered only if `gliner` is installed. |
| `gliner_nvidia` | `nvidia/gliner-PII`, 570M params | Same vocabulary as `gliner`; GPU recommended. |
| `gliner_gretel_small` | `gretelai/gretel-gliner-bi-small-v1.0` | Uses Gretel's label space. |
| `gliner_gretel_large` | `gretelai/gretel-gliner-bi-large-v1.0` | Uses Gretel's label space. |
| `ai4privacy_modernbert` | ModernBERT-based multilingual anonymiser (~150M params) | Requires `transformers`. 8 languages (en, fr, de, es, it, nl, hi, te). |

Hit `GET /v1/detectors` to see what's currently registered in your deployment.

## Eager-loading

By default, only the detector named in `DEFAULT_DETECTOR` is loaded at startup. Other detectors initialise on first request. Set `EAGER_LOAD` (comma-separated names) to warm additional detectors at startup — useful when you want a predictable first-request latency.

## Proxy detectors

`proxy: true` means the detector calls an external service rather than running locally. The `skyflow` detector is the only one shipped today. Proxy detectors require additional env (see the [auth and env guide](auth-and-env.md)) and surface upstream errors as `502` responses.
