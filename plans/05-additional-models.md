# Plan 05 — Add four more PII-focused open-weight models

## Why

The current open-weight tier has just two distinct architectures (OPF and GLiNER) with one checkpoint each. Adding more PII-trained variants tells us how much of OPF/GLiNER's quality is architectural vs. training-data, and gives users more options when they want to deploy locally without committing to any single vendor's weights.

Models to add:

| name | repo | model card highlights |
| --- | --- | --- |
| Gretel small | `gretelai/gretel-gliner-bi-small-v1.0` | bi-encoder GLiNER, 41 PII labels, F1 0.94 on Gretel's own data, Apache 2.0 |
| Gretel large | `gretelai/gretel-gliner-bi-large-v1.0` | larger bi-encoder GLiNER, same 41 labels, F1 0.95, Apache 2.0 |
| Nvidia PII | `nvidia/gliner-PII` | built on `urchade/gliner_large-v2.1` (570M params), 55+ labels, NVIDIA Open Model License |
| OpenMed PF multilingual | `OpenMed/privacy-filter-multilingual` | OPF-architecture fine-tune with **54 categories** across **16 languages**, 217-class BIOES head, Apache 2.0 |

## Scope

- Three new GLiNER detector names (small Gretel/large Gretel/Nvidia) — reuse `GLiNERDetector` with model-specific args
- One new OPF-architecture detector — but **not** a drop-in for our existing `OPFDetector` class because OpenMed redefined the label space (54 vs OPF's 8 categories) and ships in HF Transformers format, not OPF's native checkpoint format
- Per-variant prompt/label vocabularies in `taxonomy.py`
- New `OpenMedDetector` class

## Files

- **Modify** [eval/src/opf_eval/detectors/gliner.py](eval/src/opf_eval/detectors/gliner.py) — accept optional `prompts` list (override the global `gliner_prompts()` default)
- **Modify** [eval/src/opf_eval/runner.py](eval/src/opf_eval/runner.py) — register four new detector names
- **Modify** [eval/src/opf_eval/taxonomy.py](eval/src/opf_eval/taxonomy.py) — add `gliner_gretel` (41 labels) and `openmed` (54 labels) columns; add helper to fetch Gretel's prompt list
- **Add** [eval/src/opf_eval/detectors/openmed.py](eval/src/opf_eval/detectors/openmed.py) — new detector class wrapping the `openmed` Python library or transformers' `AutoModelForTokenClassification` directly
- **Modify** [eval/pyproject.toml](eval/pyproject.toml) — add `openmed[hf]>=...` (verify version)

## Per-model implementation notes

### Gretel small / large — `gretelai/gretel-gliner-bi-{small,large}-v1.0`

- **Architecture:** bi-encoder GLiNER (different from `urchade/gliner_multi_pii-v1`'s uni-encoder; bi-encoder is faster at inference because the label embeddings can be pre-computed)
- **41 labels** trained on, snake_case, more granular than our defaults:

  ```text
  medical_record_number, date_of_birth, ssn, date, first_name, email, last_name,
  customer_id, employee_id, name, street_address, phone_number, ipv4,
  credit_card_number, license_plate, address, user_name, device_identifier,
  bank_routing_number, date_time, company_name, unique_identifier,
  biometric_identifier, account_number, city, certificate_license_number, time,
  postcode, vehicle_identifier, coordinate, country, api_key, ipv6, password,
  health_plan_beneficiary_number, national_id, tax_id, url, state, swift_bic,
  cvv, pin
  ```

- **Recommended threshold:** 0.7 (per model card; higher than our default 0.5)
- **English-only training data** (`gretelai/gretel-pii-masking-en-v1`). Likely tanks on Dutch/French/etc. fixtures — confirm in per-language slicing.
- **Use Gretel's own labels as prompts**, not our generic ones. Map via taxonomy:

  | canonical | Gretel labels |
  | --- | --- |
  | PERSON | `first_name`, `last_name`, `name`, `user_name` |
  | EMAIL | `email` |
  | PHONE | `phone_number` |
  | ADDRESS | `address`, `street_address`, `city`, `state`, `country`, `postcode`, `coordinate` |
  | URL | `url`, `ipv4`, `ipv6` |
  | DATE | `date`, `date_of_birth`, `date_time`, `time` |
  | ACCOUNT | `ssn`, `credit_card_number`, `bank_routing_number`, `account_number`, `national_id`, `tax_id`, `swift_bic`, `cvv`, `pin`, `medical_record_number`, `health_plan_beneficiary_number`, `unique_identifier`, `customer_id`, `employee_id`, `device_identifier`, `biometric_identifier`, `certificate_license_number`, `license_plate`, `vehicle_identifier` |
  | SECRET | `password`, `api_key` |
  | USERNAME | `user_name` |
  | DEMOGRAPHIC | (none) |

  Note `user_name` is mapped to both PERSON and USERNAME — pick one canonical at scoring time. Probably USERNAME for fairness.

### Nvidia PII — `nvidia/gliner-PII`

- **Built on `urchade/gliner_large-v2.1`** — same architecture as our default GLiNER, but a different (larger, 570M-parameter) base. Latency will be higher; memory usage too.
- **55+ labels** but the model card doesn't enumerate them. Their own example uses our same vocabulary (`email`, `phone_number`, `user_name`). **Start with our existing `gliner_prompts()` defaults** — only customize if F1 looks low.
- **Recommended threshold:** 0.3 (lower than 0.5; more aggressive)
- **License:** NVIDIA Open Model License (not Apache). Permits commercial + non-commercial use but with attribution and termination clauses different from Apache 2.0. **Flag in any writeup.**
- **Reported metrics:** F1 0.70 on Argilla PII, 0.64 on AI4Privacy. PII-Masking-300k is closer to AI4Privacy in style — could expect F1 in the ~0.65-0.75 range on our benchmark.

### OpenMed PF multilingual — `OpenMed/privacy-filter-multilingual`

This is the most interesting and the most work. Critical facts from the card:

- **OPF-architecture** (1.4B params, 50M active, top-4-of-128 MoE), but **fine-tuned with `opf train`** so the underlying compute is identical. Inference latency should be similar to vanilla OPF.
- **54 PII categories** — far more granular than OPF's 8. Spans medical (MEDICAL_RECORD_NUMBER), demographic (GENDER, AGE, EYECOLOR, HEIGHT), digital (USERAGENT, IPADDRESS, MACADDRESS), crypto (BITCOINADDRESS, ETHEREUMADDRESS, LITECOINADDRESS), vehicle (VIN, VRM), and more.
- **217-class BIOES head** (54 categories × {B,I,E,S} + O background) — different shape from OPF's 33-class head.
- **Multilingual:** trained on 16 languages (Arabic, Bengali, Chinese, Dutch, English, French, German, Hindi, Italian, Japanese, Korean, Portuguese, Spanish, Telugu, Turkish, Vietnamese). Should crush vanilla OPF on non-English.
- **Not OPF checkpoint format** — packaged as a standard HuggingFace Transformers token-classification model. Cannot use `OPFDetector(model="OpenMed/...")` as-is.

**Two ways to load:**

1. **Recommended (per model card):** install `openmed[hf]`, use `extract_pii()`:

   ```python
   from openmed import extract_pii
   result = extract_pii(text, model_name="OpenMed/privacy-filter-multilingual")
   # returns spans with label, text, start, end, confidence
   ```

   Includes built-in BIOES Viterbi decoding and span refinement. Adds a dependency on the `openmed` library; check its license + maturity.

2. **Direct via Transformers:**

   ```python
   from transformers import pipeline
   pipe = pipeline("token-classification",
                   model="OpenMed/privacy-filter-multilingual",
                   trust_remote_code=True,
                   aggregation_strategy="first")
   results = pipe(text)
   ```

   Requires `trust_remote_code=True`. We'd handle BIOES aggregation ourselves (or rely on `aggregation_strategy="first"`, which is approximate).

I'd start with option 1 — let the OpenMed library do the BIOES decoding correctly. If `openmed[hf]` adds too much surface area, fall back to option 2.

**Label mapping (54 → 10 canonicals):** straightforward but tedious. Most labels map cleanly:

  | canonical | OpenMed labels |
  | --- | --- |
  | PERSON | `FIRSTNAME`, `MIDDLENAME`, `LASTNAME`, `PREFIX` |
  | EMAIL | `EMAIL` |
  | PHONE | `PHONE` |
  | ADDRESS | `STREET`, `BUILDINGNUMBER`, `SECONDARYADDRESS`, `CITY`, `COUNTY`, `STATE`, `ZIPCODE`, `GPSCOORDINATES`, `ORDINALDIRECTION` |
  | URL | `URL`, `IPADDRESS`, `MACADDRESS` |
  | DATE | `DATE`, `DATEOFBIRTH`, `TIME` |
  | ACCOUNT | `SSN`, `ACCOUNTNAME`, `BANKACCOUNT`, `IBAN`, `BIC`, `CREDITCARD`, `CREDITCARDISSUER`, `CVV`, `PIN`, `MASKEDNUMBER`, `BITCOINADDRESS`, `ETHEREUMADDRESS`, `LITECOINADDRESS`, `VIN`, `VRM`, `IMEI` |
  | SECRET | `PASSWORD` |
  | USERNAME | `USERNAME`, `USERAGENT` |
  | DEMOGRAPHIC | `AGE`, `GENDER`, `SEX`, `EYECOLOR`, `HEIGHT`, `OCCUPATION`, `JOBTITLE`, `JOBDEPARTMENT`, `ORGANIZATION` |

  Categories left out of mapping (treat as out-of-scope, won't show up in restricted view): `AMOUNT`, `CURRENCY`, `CURRENCYCODE`, `CURRENCYNAME`, `CURRENCYSYMBOL`. PII-Masking-300k doesn't label monetary amounts.

## Risks / open questions

- **Per-variant prompt sensitivity (GLiNER).** Gretel's snake_case labels probably need to be sent verbatim to get the model's full quality. Confirm with smoke-test runs.
- **Threshold tuning.** Gretel recommends 0.7, Nvidia recommends 0.3, our default GLiNER uses 0.5. Wrong threshold could underperform either model by 5+ F1.
- **License variation.** NVIDIA Open Model License is not Apache 2.0. Document in any writeup that compares licenses.
- **`openmed[hf]` library footprint and stability.** Brand new. Verify it doesn't pull in heavy unwanted deps (vllm, accelerate, etc.) and that its API is stable.
- **OpenMed BIOES vs OPF Viterbi.** Different decoding paths could produce different greedy-span behavior than vanilla OPF. Worth checking on the same DATE examples that OPF struggles with.
- **Disk pressure.** Gretel small ~200 MB, Gretel large ~500 MB, Nvidia ~2 GB (570M params), OpenMed ~3 GB. Total ~6 GB additional download for someone running all four.
- **Memory at inference.** Gretel large + Nvidia + OpenMed simultaneously may exceed Colab free tier RAM. Run them serially.

## Verification

```sh
# 1. Smoke each variant individually on the existing 100 fixtures
for variant in gliner_gretel_small gliner_gretel_large gliner_nvidia openmed_pf_multi; do
  uv run python -m opf_eval.runner \
      --fixtures eval/data/sample_100.jsonl \
      --detectors $variant \
      --reuse-from eval/results/runs/run_100/ \
      --out eval/results/runs/run_100_$variant/
  uv run python -m opf_eval.report --run eval/results/runs/run_100_$variant/ --fixtures eval/data/sample_100.jsonl
done

# 2. Full 1k bench reusing existing detector outputs
uv run python -m opf_eval.runner \
    --fixtures eval/data/sample_1k.jsonl \
    --detectors gliner_gretel_small,gliner_gretel_large,gliner_nvidia,openmed_pf_multi \
    --reuse-from eval/results/runs/run_1k_with_gliner/ \
    --out eval/results/runs/run_1k_with_variants/

uv run python -m opf_eval.report \
    --run eval/results/runs/run_1k_with_variants/ \
    --fixtures eval/data/sample_1k.jsonl
```

What I'd watch for:

- **Gretel small vs large:** scaling within the same recipe. If small captures most of large's quality, recommend small.
- **Gretel English vs other languages:** confirm the English-only training tanks recall on Dutch/German/etc. (expect 30+ F1 drop).
- **Nvidia vs urchade/gliner_multi_pii-v1:** does the larger base model justify ~3× the latency?
- **OpenMed vs vanilla OPF on per-language:** OpenMed should crush vanilla OPF on non-English. If it doesn't, the multilingual fine-tune story is overstated.
- **OpenMed vs Skyflow_minimal:** the headline open-weight question. OpenMed has 54 categories and 16 languages — could be the open-weight equivalent of Skyflow_minimal in reach. If F1 is competitive, this is the strongest local-deployment story we've found.
- **DATE recall on OpenMed:** does the 217-class BIOES head avoid OPF's greedy-span pathology? If yes, that's the missing piece OPF was lacking.

## Effort

- **Gretel small + large:** ~1 hour each (label list known, just runner registration + taxonomy column + threshold). Total ~2 hours.
- **Nvidia:** ~30 min (single name registration + threshold override).
- **OpenMed:** ~3-4 hours including (a) library install + smoke test, (b) `OpenMedDetector` class, (c) 54-label taxonomy column, (d) handling the new `openmed` dep + BIOES output shape.
- **Total:** ~5-6 hours.

## Out of scope

- Fine-tuning these variants ourselves (covered in [plan 02](02-finetune-opf.md))
- Quantized variants (e.g. OpenMed has MLX-8bit variants for Apple Silicon — could be added later)
- Per-prompt grid search per model — start with documented defaults
- Comparing GLiNER bi-encoder vs uni-encoder architectures formally — we just pick what each variant ships
