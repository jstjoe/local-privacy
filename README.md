# local-privacy

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jstjoe/local-privacy/blob/main/notebooks/pii_detector_comparison.ipynb)

Benchmark harness comparing local and hosted PII detectors on PII-Masking-300k.

See [**RESULTS.md**](RESULTS.md) for headline numbers (overall + per-category + per-language F1 and latency at n=1000).

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

## Category coverage

The harness projects each detector's native entity vocabulary into a 10-label canonical taxonomy ([eval/src/opf_eval/taxonomy.py](eval/src/opf_eval/taxonomy.py)) so detectors can be compared on equal footing. The canonical labels are: PERSON, EMAIL, PHONE, ADDRESS, URL, DATE, ACCOUNT, SECRET, USERNAME, DEMOGRAPHIC.

| canonical | OPF | Skyflow Detect | Presidio | GLiNER |
| --- | --- | --- | --- | --- |
| PERSON | `private_person` | `NAME`, `NAME_GIVEN`, `NAME_FAMILY`, `NAME_MEDICAL_PROFESSIONAL` | `PERSON` | `person` |
| EMAIL | `private_email` | `EMAIL_ADDRESS` | `EMAIL_ADDRESS` | `email` |
| PHONE | `private_phone` | `PHONE_NUMBER` | `PHONE_NUMBER` | `phone number` |
| ADDRESS | `private_address` | `LOCATION`, `LOCATION_ADDRESS`, `LOCATION_ADDRESS_STREET`, `LOCATION_CITY`, `LOCATION_STATE`, `LOCATION_ZIP`, `LOCATION_COUNTRY`, `LOCATION_COORDINATE` | `LOCATION` | `address`, `location`, `city`, `country`, `postal code` |
| URL | `private_url` | `URL`, `IP_ADDRESS` | `URL`, `IP_ADDRESS` | `url`, `ip address` |
| DATE | `private_date` | `DATE`, `DATE_INTERVAL`, `DOB`, `TIME`, `DAY`, `MONTH`, `YEAR` | `DATE_TIME` | `date`, `date of birth`, `time` |
| ACCOUNT | `account_number` | `ACCOUNT_NUMBER`, `BANK_ACCOUNT`, `CREDIT_CARD`, `ROUTING_NUMBER`, `NUMERICAL_PII`, `SSN`, `PASSPORT_NUMBER`, `DRIVER_LICENSE`, `HEALTHCARE_NUMBER` | `CREDIT_CARD`, `IBAN_CODE`, `US_SSN`, `US_PASSPORT`, `US_DRIVER_LICENSE`, `US_BANK_NUMBER`, `US_ITIN`, `UK_NHS`, `UK_NINO`, `ES_NIE`, `ES_NIF`, `IT_DRIVER_LICENSE`, `IT_FISCAL_CODE`, `IT_IDENTITY_CARD`, `IT_PASSPORT`, `IT_VAT_CODE`, `AU_ABN`, `AU_ACN`, `AU_MEDICARE`, `AU_TFN`, `IN_AADHAAR`, `IN_PAN`, `IN_VEHICLE_REGISTRATION`, `MEDICAL_LICENSE`, `CRYPTO` | `social security number`, `passport`, `passport number`, `driver license`, `driver's license`, `credit card`, `credit card number`, `account number`, `bank account`, `national id`, `id number`, `tax id` |
| SECRET | `secret` | `PASSWORD` | — | `password` |
| USERNAME | — | `USERNAME` | — | `username` |
| DEMOGRAPHIC | — | `GENDER`, `AGE`, `GENDER_SEXUALITY`, `MARITAL_STATUS` | `NRP` | — |

Notes:

- **OPF** has 8 native categories — fixed at training time, not configurable. No native USERNAME or DEMOGRAPHIC support.
- **Skyflow Detect** exposes ~70 entity types in total. The list above is what the harness asks for via the `entity_types` request parameter. The `skyflow_minimal` detector config (recommended) sends a tuned subset that drops bare `name` / `location` / `location_address` (low gold-hit rate per `eval/scripts/analyze_skyflow_hitrate.py`).
- **Presidio** entries are the default English recognizers. Multilingual Presidio adds language-specific spaCy NER, not new entity types. No native SECRET or USERNAME recognizer.
- **GLiNER** is zero-shot and accepts any natural-language prompt. The vocabulary above is what the harness sends to the `urchade/gliner_multi_pii-v1` checkpoint; tuning the prompts is one of the easier ways to move GLiNER's per-category numbers.
- **Categories not in this table** (OCCUPATION, ORGANIZATION, MEDICAL_PROCESS, etc. that Skyflow detects) are out of scope for this benchmark — PII-Masking-300k doesn't label them.

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
