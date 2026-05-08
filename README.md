# local-privacy

Benchmark harness comparing local and hosted PII detectors on PII-Masking-300k.

Detectors covered:

- **OPF** — OpenAI Privacy Filter, open-weight, local
- **GLiNER** — small open-weight zero-shot NER, local
- **Microsoft Presidio** — regex + spaCy NER, local
- **Skyflow Detect API** — hosted

Plus a minimal FastAPI server wrapping OPF for a privacy-preprocessing layer.

## Layout

```text
local-privacy/
├── eval/         # benchmark harness — fixtures, detectors, runner, metrics, report
├── api/          # FastAPI server wrapping OPF
├── notebooks/    # Colab notebook for hosted runs
├── plans/        # specs for additional experiments (see plans/README.md)
└── privacy-filter/   # OPF source — cloned separately, gitignored
```

## Setup

```sh
# 1. Clone OPF source as a sibling to the eval/api packages
git clone https://github.com/openai/privacy-filter

# 2. Install everything as a uv workspace
uv sync
source .venv/bin/activate

# 3. Download Presidio's English spaCy model (required for the presidio detector)
python -m spacy download en_core_web_lg
```

For multilingual Presidio, also:

```sh
for lang in nl fr de it es; do python -m spacy download ${lang}_core_news_lg; done
```

## Quick smoke test

```sh
# 1. Materialize a 100-example sample from PII-Masking-300k
python -m opf_eval.fixtures --out eval/data/sample_100.jsonl --n 100

# 2. Run local-only detectors (no API creds needed)
python -m opf_eval.runner \
    --fixtures eval/data/sample_100.jsonl \
    --detectors opf,gliner,presidio \
    --out eval/results/runs/smoke/

# 3. View the report
python -m opf_eval.report --run eval/results/runs/smoke/ --fixtures eval/data/sample_100.jsonl
cat eval/results/runs/smoke/report.md
```

## Adding Skyflow

Set credentials via env vars (use a `.env` file for convenience — already gitignored):

```sh
export SKYFLOW_VAULT_URL="https://<your-vault>.vault.skyflowapis.com"
export SKYFLOW_VAULT_ID="<vault-uuid>"
export SKYFLOW_BEARER_TOKEN="<short-lived-bearer>"
```

Then run with one of the Skyflow detector names:

```sh
python -m opf_eval.runner \
    --fixtures eval/data/sample_100.jsonl \
    --detectors skyflow_minimal \
    --reuse-from eval/results/runs/smoke/ \
    --out eval/results/runs/with_skyflow/
```

`--reuse-from` copies existing detector outputs from a previous run so you don't re-run OPF/GLiNER/Presidio.

## Detector names

| name | what it is |
| --- | --- |
| `opf` | OpenAI Privacy Filter (default Viterbi decoder) |
| `opf_calibrated` | OPF with a custom Viterbi calibration JSON (`--opf-calibration-path`) |
| `gliner` | GLiNER multilingual PII model |
| `presidio` | Presidio English-only |
| `presidio_multilang` | Presidio with all 6 spaCy models |
| `skyflow` | Skyflow Detect API constrained to OPF's 8 categories |
| `skyflow_full` | Skyflow Detect API unconstrained (~70 entity types) |
| `skyflow_minimal` | Skyflow with the empirically-tuned entity allowlist |
| `skyflow_constrained` | Alias for `skyflow` |

## Fixtures and reports

- `python -m opf_eval.fixtures --n N --out path` — materialize N examples, deterministic seed
- `python -m opf_eval.runner --detectors X,Y --fixtures path --out dir` — run detectors against fixtures
- `python -m opf_eval.report --run dir --fixtures path` — emit `report.md`
- `python -m opf_eval.report ... --canonical-labels DATE` — restrict scoring to a single category

## Notebooks

[notebooks/pii_detector_comparison.ipynb](notebooks/pii_detector_comparison.ipynb) is a Colab-ready version of the same harness. Edit the `HARNESS_REPO` URL in the setup cell to your fork before sharing.

## Plans for further experiments

[plans/](plans/) — see plans/README.md for an index. Currently filed:

- 01: Microsoft Presidio baseline (shipped)
- 02: Fine-tune OPF on PII-Masking-300k
- 03: GLiNER baseline (shipped)
- 04: LLM-as-detector via LM Studio

## API server

The `api/` directory is a minimal FastAPI server wrapping OPF as a privacy-preprocessing layer. Run with:

```sh
uvicorn opf_api.main:app --reload
```

Routes: `POST /redact`, `POST /detect`, `GET /health`. See [api/Dockerfile](api/Dockerfile) for containerization.
