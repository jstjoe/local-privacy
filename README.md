# local-privacy

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jstjoe/local-privacy/blob/main/notebooks/pii_detector_comparison.ipynb)

Benchmark harness comparing local and hosted PII detectors against any of five ai4privacy datasets (PII-Masking-200k/300k/400k, OpenPII nano/mini).

See [**RESULTS.md**](RESULTS.md) for headline numbers (overall + per-category + per-language F1 and latency at n=1000 on PII-Masking-300k).

Detectors covered:

- **OPF** — OpenAI Privacy Filter, open-weight, local
- **GLiNER** — small open-weight zero-shot NER, local
- **Microsoft Presidio** — regex + spaCy NER, local
- **Skyflow Detect API** — hosted

Pick `(detector, dataset, sample size)` on the CLI; detectors that take a per-call label set (Skyflow, GLiNER) auto-configure to the chosen dataset's vocabulary.

Plus a unified FastAPI service ([api/](api/)) exposing all detectors behind one contract.

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
# 1. Materialize a 100-example sample from a dataset of your choice.
#    Default is pii_masking_300k for back-compat. Other names below.
python -m opf_eval.fixtures --dataset openpii_nano --out eval/data/openpii_nano_100.jsonl --n 100

# 2. Run local-only detectors (no API creds needed)
python -m opf_eval.runner \
    --dataset openpii_nano \
    --fixtures eval/data/openpii_nano_100.jsonl \
    --detectors opf,gliner,presidio \
    --out eval/results/runs/smoke/

# 3. View the report (two scoring views — "fair" per detector +
#    "raw" against the full dataset vocabulary)
python -m opf_eval.report --run eval/results/runs/smoke/ --fixtures eval/data/openpii_nano_100.jsonl
cat eval/results/runs/smoke/report.md
```

## Datasets

Five ai4privacy variants pre-registered. Three distinct annotation vocabularies (verified via record dump in [eval/src/opf_eval/datasets/](eval/src/opf_eval/datasets/)):

| `--dataset` | size | vocab | notes |
| --- | --- | --- | --- |
| `openpii_nano` | 1k | OpenPII | smoke / CI |
| `openpii_mini` | 10k | OpenPII | mid-size |
| `pii_masking_200k` | 200k | own (`FIRSTNAME`/`LASTNAME`/`PHONENUMBER`/...) | older |
| `pii_masking_300k` | 300k | numbered names (`GIVENNAME1`/`LASTNAME1`/...) | current default |
| `pii_masking_400k` | 400k | OpenPII | newest legacy variant |

The runner writes the dataset name into the manifest; the report uses it to drive both scoring views.

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
| `gliner` | GLiNER multilingual PII model — prompts auto-restricted to dataset's canonical labels |
| `presidio` | Presidio English-only |
| `presidio_multilang` | Presidio with all 6 spaCy models |
| `skyflow` | Skyflow Detect API; `entity_types` auto-derived from dataset's canonical labels |
| `skyflow_full` | Skyflow Detect API unconstrained (~70 entity types) |
| `skyflow_minimal` | Skyflow with the hand-tuned PII-Masking-300k allowlist (manual override; not dataset-aware) |
| `skyflow_constrained` | Alias for `skyflow` |

## Canonical entity types

The harness projects every detector's native entity vocabulary and every dataset's gold-label vocabulary into a 10-label canonical taxonomy ([eval/src/opf_eval/taxonomy.py](eval/src/opf_eval/taxonomy.py)) so apples-to-apples comparison is possible. The 10 canonical labels are:

| canonical | meaning |
| --- | --- |
| `PERSON` | Names (full, given, family, titles) |
| `EMAIL` | Email addresses |
| `PHONE` | Phone numbers, IMEI |
| `ADDRESS` | Street, city, state/region, postcode, country, coordinates, building |
| `URL` | URLs and IP addresses |
| `DATE` | Dates, times, DOB |
| `ACCOUNT` | Account / credit card / SSN / passport / driver-licence / bank / crypto identifiers |
| `SECRET` | Passwords |
| `USERNAME` | Logins / handles |
| `DEMOGRAPHIC` | Gender / sex / age / nationality (Presidio's `NRP`) |

### Quick reference: who supports what

A check means at least one raw label maps to that canonical category in [taxonomy.py](eval/src/opf_eval/taxonomy.py). Dashes mean the source has no annotations / no recognizer for that canonical type.

| canonical | pii_masking_200k | pii_masking_300k | openpii (400k, nano, mini) | OPF | Skyflow | Presidio | GLiNER |
| --- | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| PERSON | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| EMAIL | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| PHONE | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| ADDRESS | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| URL | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| DATE | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| ACCOUNT | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| SECRET | ✓ | ✓ | ✓ | ✓ | ✓ | — | ✓ |
| USERNAME | ✓ | ✓ | ✓ | — | ✓ | — | ✓ |
| DEMOGRAPHIC | ✓ | ✓ (`SEX` only) | ✓ | — | ✓ | ✓ (`NRP`) | — |

This drives the **fair scoring view** in the report: each detector's headline F1 is computed against `dataset_canonicals ∩ detector_supported_canonicals` so detectors aren't punished for labels they don't claim.

### Dataset → canonical mapping (raw labels)

| canonical | pii_masking_200k | pii_masking_300k | openpii (400k, nano, mini) |
| --- | --- | --- | --- |
| PERSON | `FIRSTNAME`, `MIDDLENAME`, `LASTNAME`, `PREFIX`, `SUFFIX` | `GIVENNAME1`, `GIVENNAME2`, `LASTNAME1`, `LASTNAME2`, `LASTNAME3`, `TITLE` | `GIVENNAME`, `SURNAME`, `TITLE` |
| EMAIL | `EMAIL` | `EMAIL` | `EMAIL` |
| PHONE | `PHONENUMBER`, `PHONEIMEI` | `TEL` | `TELEPHONENUM` |
| ADDRESS | `STREET`, `CITY`, `COUNTY`, `STATE`, `ZIPCODE`, `BUILDINGNUMBER`, `SECONDARYADDRESS`, `NEARBYGPSCOORDINATE` | `STREET`, `CITY`, `STATE`, `COUNTRY`, `POSTCODE`, `BUILDING`, `SECADDRESS`, `GEOCOORD` | `STREET`, `CITY`, `STATE`, `ZIPCODE`, `BUILDINGNUM`, `SECONDARYADDRESS` |
| URL | `URL`, `IP`, `IPV4`, `IPV6` | `IP` | `URL`, `IP`, `IPV4`, `IPV6` |
| DATE | `DATE`, `TIME`, `DOB` | `DATE`, `TIME`, `BOD` | `DATE`, `TIME`, `DATEOFBIRTH` |
| ACCOUNT | `ACCOUNTNUMBER`, `ACCOUNTNAME`, `CREDITCARDNUMBER`, `CREDITCARDISSUER`, `CREDITCARDCVV`, `BITCOINADDRESS`, `ETHEREUMADDRESS`, `LITECOINADDRESS`, `IBAN`, `BIC`, `PIN` | `SOCIALNUMBER`, `IDCARD`, `PASSPORT`, `DRIVERLICENSE` | `ACCOUNTNUM`, `CREDITCARDNUMBER`, `IDCARDNUM`, `SOCIALNUM`, `PASSPORTNUM`, `DRIVERLICENSENUM`, `TAXNUM` |
| SECRET | `PASSWORD` | `PASS` | `PASSWORD` |
| USERNAME | `USERNAME` | `USERNAME` | `USERNAME` |
| DEMOGRAPHIC | `GENDER`, `SEX`, `AGE` | `SEX` | `GENDER`, `SEX`, `AGE` |

Each dataset has additional raw labels not yet mapped to a canonical (e.g. pii_masking_200k's `JOBTITLE`, `JOBAREA`, `COMPANYNAME`, `USERAGENT`). Those are silently dropped at fixture-write time. Add a column to `CANONICAL_MAP` if you want them scored.

### Detector → canonical mapping (raw labels)

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
- **Skyflow Detect** exposes 69 entity types in total (plus an `all` meta-value). The 38 mapped above are what the harness asks for via the `entity_types` request parameter. The `skyflow` detector now auto-derives this set from the chosen dataset's canonical labels; `skyflow_minimal` keeps a hand-tuned PII-Masking-300k preset (drops bare `name` / `location` / `location_address`, low gold-hit rate per `eval/scripts/analyze_skyflow_hitrate.py`). The other 31 entity types are listed below.
- **Presidio** entries are the default English recognizers. Multilingual Presidio adds language-specific spaCy NER, not new entity types. No native SECRET or USERNAME recognizer.
- **GLiNER** is zero-shot and accepts any natural-language prompt. The vocabulary above is what the harness sends to the `urchade/gliner_multi_pii-v1` checkpoint; with `--dataset NAME` set, the prompt list is auto-restricted to canonicals the dataset annotates. Tuning the prompts is one of the easier ways to move GLiNER's per-category numbers.

### Additional Skyflow categories (out of scope for this benchmark)

PII-Masking-300k doesn't label these, so the harness doesn't grade them — but Skyflow detects them and they're worth knowing about when sizing the platform for use cases beyond what this benchmark covers.

- **Health / medical:** `BLOOD_TYPE`, `CONDITION`, `DOSE`, `DRUG`, `EFFECT`, `INJURY`, `MEDICAL_CODE`, `MEDICAL_PROCESS`, `ORGANIZATION_MEDICAL_FACILITY`
- **Personal characteristics:** `ORIGIN`, `PHYSICAL_ATTRIBUTE`, `POLITICAL_AFFILIATION`, `RELIGION`, `SEXUALITY`, `ZODIAC_SIGN`
- **Financial / market:** `CORPORATE_ACTION`, `CREDIT_CARD_EXPIRATION`, `CVV`, `FINANCIAL_METRIC`, `MONEY`, `STATISTICS`, `TREND`
- **Organization / work:** `EVENT`, `OCCUPATION`, `ORGANIZATION`, `ORGANIZATION_ID`, `PROJECT`
- **Other:** `DURATION`, `FILENAME`, `LANGUAGE`, `PRODUCT`, `VEHICLE_ID`

Source: `DeidentifyStringRequest.entity_types` enum in [detect.openapi.json](detect.openapi.json).

### How `skyflow_minimal` was tuned

Skyflow's default behavior is to return whichever of its ~70 entity types it's confident about. Two issues with that for benchmarking:

1. **Out-of-scope categories drag precision down.** PII-Masking-300k doesn't label OCCUPATION, ORGANIZATION, MEDICAL_PROCESS, etc., so every correct Skyflow detection of those counts as a false positive. The `skyflow` detector restricts the request to OPF's 8 categories to avoid this.
2. **Bare general types are noisy fallbacks.** Skyflow returns broader labels like `NAME` (not `NAME_GIVEN`), `LOCATION` (not `LOCATION_CITY`), `LOCATION_ADDRESS` (not `LOCATION_ADDRESS_STREET`) when uncertain. These have much lower gold-hit rates than the granular subtypes.

The methodology, in [eval/scripts/analyze_skyflow_hitrate.py](eval/scripts/analyze_skyflow_hitrate.py): for each raw entity type Skyflow returned in a 1k benchmark run, count how often it overlapped a gold span (≥0.5 IoU, same canonical). Sort by hit rate. Drop entity types below 50%.

Findings on the 1k sample:

| dropped raw label | gold hit rate | reason |
| --- | --- | --- |
| `NAME` | 38% | redundant with `NAME_GIVEN` (83%) + `NAME_FAMILY` (85%) |
| `LOCATION` | 46% | redundant with `LOCATION_CITY` (79%) and friends |
| `LOCATION_ADDRESS` | 20% | redundant with `LOCATION_ADDRESS_STREET` (89%) |
| `NAME_MEDICAL_PROFESSIONAL` | 42% | low-confidence subtype |
| `CREDIT_CARD` | 43% | sparse, low precision in our data |
| `MONTH` / `YEAR` / `DAY` | 0–50% | rarely hit gold (gold uses full dates) |

The resulting allowlist is `SKYFLOW_MINIMAL_ENTITY_TYPES` in [eval/src/opf_eval/taxonomy.py](eval/src/opf_eval/taxonomy.py) — 24 entity types instead of the 33 in the default constrained config.

Impact at n=1000 (full numbers in [RESULTS.md](RESULTS.md)):

- **Overall F1: 0.819 → 0.835** (+1.6)
- **ADDRESS F1: 0.870 → 0.926** (+5.6) — the headline improvement
- **PERSON F1: 0.701 → 0.748** (+4.7)
- All other categories within ±1 F1
- **Latency p99: 207 ms → 146 ms** (smaller response payload)

Same approach would work for any hosted PII detector with a configurable entity allowlist: collect 1k of detection output against your gold, drop the raw entity types under ~50% hit rate.

## Fixtures and reports

- `python -m opf_eval.fixtures --dataset NAME --n N --out path` — materialize N examples (deterministic seed)
- `python -m opf_eval.runner --dataset NAME --detectors X,Y --fixtures path --out dir` — run detectors; manifest carries dataset name + vocab
- `python -m opf_eval.report --run dir --fixtures path` — emit `report.md` with both fair (per-detector scope) + raw (full dataset vocab) views
- `python -m opf_eval.report ... --canonical-labels DATE` — override both views to a single explicit label set (one-category drilldowns)

### Two scoring views

The report emits two SemEval sections per run:

- **Fair view** — each detector scored against `dataset_canonicals ∩ detector_supported_canonicals`. Each row's `n labels` column shows the per-detector scope. Apples-to-apples within each detector's claimed coverage; doesn't punish broader vocabularies or flatter narrower ones.
- **Raw view** — every detector scored against the dataset's full annotated set. Labels a detector doesn't support take zero recall; reflects out-of-the-box coverage.

The greedy per-category breakdown stays under the raw view (single per-label table; `—` where a detector doesn't claim the label).

## Notebooks

[notebooks/pii_detector_comparison.ipynb](notebooks/pii_detector_comparison.ipynb) is a Colab-ready version of the same harness. Edit the `HARNESS_REPO` URL in the setup cell to your fork before sharing.

## Plans for further experiments

[plans/](plans/) — see [plans/README.md](plans/README.md) for the full index. Highlights:

- 01: Microsoft Presidio baseline (shipped)
- 03: GLiNER baseline (shipped)
- 06: Unified privacy-detection API (shipped)
- 08: SemEval scoring via nervaluate (shipped)
- 09: Multi-dataset fixtures + per-detector scoring (this PR)
- 02, 04, 05: model & training experiments (not yet shipped)
- 07: Cloud Run hardening (planned)

## API server

The `api/` directory is a unified FastAPI service exposing all benchmark detectors behind one contract. Pick the backend with the `detector` field; canonical labels apply uniformly. Run with:

```sh
DEFAULT_DETECTOR=opf EAGER_LOAD=opf uvicorn opf_api.main:app --reload
```

Routes: `POST /v1/redact`, `POST /v1/detect`, `GET /v1/detectors`, `GET /v1/health`. Legacy `/redact`/`/detect`/`/health` remain (OPF-only, deprecated). See [api/Dockerfile](api/Dockerfile) for slim/full container profiles via build args.
