# Plan 05 — Add five more PII-focused open-weight models

## Why

The current open-weight tier has just two distinct architectures (OPF and GLiNER) with one checkpoint each. Adding more PII-trained variants tells us how much of OPF/GLiNER's quality is architectural vs. training-data, and gives users more options when they want to deploy locally without committing to any single vendor's weights.

Models to add:

| name | repo | base / params | license | model card highlights |
| --- | --- | --- | --- | --- |
| Gretel small | `gretelai/gretel-gliner-bi-small-v1.0` | bi-encoder GLiNER | Apache 2.0 | 41 PII labels, F1 0.94 on Gretel's own data, English-only training |
| Gretel large | `gretelai/gretel-gliner-bi-large-v1.0` | bi-encoder GLiNER | Apache 2.0 | larger bi-encoder, same 41 labels, F1 0.95 |
| Nvidia PII | `nvidia/gliner-PII` | `urchade/gliner_large-v2.1` (570M) | NVIDIA Open Model License | 55+ labels, F1 0.70 on Argilla PII / 0.64 on AI4Privacy |
| ai4privacy ModernBERT | `ai4privacy/llama-ai4privacy-multilingual-categorical-anonymiser-openpii` | ModernBERT-base | MIT | 8 languages (fr/en/de/te/hi/it/es/nl), F1 0.915 on OpenPII 500k test, OpenPII vocab |
| OpenMed PF multilingual | `OpenMed/privacy-filter-multilingual` | OPF-architecture (1.4B params, 50M active) | Apache 2.0 | **54 categories** across **16 languages**, 217-class BIOES head |

(The `llama-` prefix on the ai4privacy model name is misleading — the actual base is `answerdotai/ModernBERT-base`, per its config + model card.)

## Scope

- Three new GLiNER detector names (Gretel small / large / Nvidia) — reuse `GLiNERDetector` with model-specific args
- One transformers-pipeline detector for the ai4privacy ModernBERT model — new `Ai4PrivacyDetector` class
- One new OPF-architecture detector for OpenMed — new `OpenMedDetector` class (different decoder + label space than vanilla OPF)
- Per-variant label/prompt vocabularies in `taxonomy.py` (extends the existing 15-canonical mapping; most of the new categories already covered by plan 09's expansion)

## Files

- **Modify** [eval/src/opf_eval/runner.py](../eval/src/opf_eval/runner.py) — register five new detector names; thread `dataset_canonicals_set` for prompt restriction where applicable
- **Modify** [eval/src/opf_eval/taxonomy.py](../eval/src/opf_eval/taxonomy.py) — add `gretel_gliner` and `openmed` columns; the ai4privacy model uses OpenPII vocab so its column is already populated
- **Add** [eval/src/opf_eval/detectors/ai4privacy.py](../eval/src/opf_eval/detectors/ai4privacy.py) — new detector class wrapping `transformers.pipeline("token-classification", ...)`
- **Add** [eval/src/opf_eval/detectors/openmed.py](../eval/src/opf_eval/detectors/openmed.py) — new detector class wrapping `openmed[hf]`
- **Modify** [eval/pyproject.toml](../eval/pyproject.toml) — add `transformers>=4.40` (likely already pulled in transitively) and `openmed[hf]>=…`

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
  | PERSON | `first_name`, `last_name`, `name` |
  | EMAIL | `email` |
  | PHONE | `phone_number` |
  | ADDRESS | `address`, `street_address`, `city`, `state`, `country`, `postcode`, `coordinate` |
  | URL | `url`, `ipv4`, `ipv6` |
  | DATE | `date`, `date_of_birth`, `date_time`, `time` |
  | ACCOUNT | `ssn`, `credit_card_number`, `bank_routing_number`, `account_number`, `national_id`, `tax_id`, `swift_bic`, `cvv`, `pin`, `medical_record_number`, `health_plan_beneficiary_number`, `unique_identifier`, `customer_id`, `employee_id`, `device_identifier`, `biometric_identifier`, `certificate_license_number` |
  | SECRET | `password`, `api_key` |
  | USERNAME | `user_name` |
  | ORGANIZATION | `company_name` |
  | VEHICLE | `license_plate`, `vehicle_identifier` |

  No DEMOGRAPHIC, OCCUPATION, MONEY, or PHYSICAL coverage. (`name` could ambiguously go to PERSON or ORGANIZATION; sticking with PERSON.)

### Nvidia PII — `nvidia/gliner-PII`

- **Built on `urchade/gliner_large-v2.1`** — same architecture as our default GLiNER, but a different (larger, 570M-parameter) base. Latency will be higher; memory usage too.
- **55+ labels** but the model card doesn't enumerate them. Their own example uses our same vocabulary (`email`, `phone_number`, `user_name`). **Start with our existing `gliner_prompts(dataset_canonicals)` defaults** — only customize if F1 looks low.
- **Recommended threshold:** 0.3 (lower than 0.5; more aggressive)
- **License:** NVIDIA Open Model License (not Apache). Permits commercial + non-commercial use but with attribution and termination clauses different from Apache 2.0. **Flag in any writeup.**
- **Reported metrics:** F1 0.70 on Argilla PII, 0.64 on AI4Privacy. Could expect F1 in the ~0.65-0.75 range on PII-Masking-300k.

### ai4privacy ModernBERT — `ai4privacy/llama-ai4privacy-multilingual-categorical-anonymiser-openpii`

- **Architecture:** `answerdotai/ModernBERT-base` token-classification head. ~150M params, light enough to run on CPU at reasonable throughput.
- **Multilingual:** 8 languages — fr, en, de, te, hi, it, es, nl. Strong overlap with PII-Masking-300k's 6 languages and 200k's 4.
- **Trained on:** [ai4privacy/open-pii-masking-500k-ai4privacy](https://huggingface.co/datasets/ai4privacy/open-pii-masking-500k-ai4privacy) — same OpenPII vocabulary used by our `pii_masking_400k`/`openpii_nano`/`openpii_mini` datasets. **Labels are already mapped** in `taxonomy.py` via the `openpii` column added in plan 09.
- **Self-reported metrics on OpenPII 500k test:** F1 0.915, P 0.876, R 0.958. High recall, slightly weaker precision (lots of FPs on numeric IDs like `PASSPORTNUM`, `DRIVERLICENSENUM`).
- **License:** MIT.
- **Loading:** standard transformers pipeline:

  ```python
  from transformers import pipeline
  pipe = pipeline(
      "token-classification",
      model="ai4privacy/llama-ai4privacy-multilingual-categorical-anonymiser-openpii",
      aggregation_strategy="first",  # or "simple"; first is more conservative
  )
  spans = pipe(text)
  # each span: {"entity_group": "GIVENNAME", "score": 0.99, "start": 7, "end": 12, "word": "Joe"}
  ```

- **Label mapping:** OpenPII vocab. Pass each span's `entity_group` through `taxonomy.dataset_to_canonical("openpii", entity_group)` — already does the right thing.
- **Detector class shape:** thin wrapper around the pipeline; the only model-specific logic is BIO-aggregation choice and the canonical-label mapping. Could be 30 lines.
- **Why this model is interesting:** smallest of the five (~150M params), lowest latency expected, native to the OpenPII vocabulary which means **no label mismatch noise** when scoring against `pii_masking_400k`/`openpii_nano`/`openpii_mini` — the cleanest possible same-vocab benchmark.

### OpenMed PF multilingual — `OpenMed/privacy-filter-multilingual`

This is the most interesting and the most work.

- **OPF-architecture** (1.4B params, 50M active, top-4-of-128 MoE), but **fine-tuned with `opf train`** so the underlying compute is identical. Inference latency should be similar to vanilla OPF.
- **54 PII categories** — far more granular than OPF's 8. Spans medical (MEDICAL_RECORD_NUMBER), demographic (GENDER, AGE, EYECOLOR, HEIGHT), digital (USERAGENT, IPADDRESS, MACADDRESS), financial (AMOUNT, CURRENCY, CURRENCYCODE), crypto (BITCOINADDRESS, ETHEREUMADDRESS, LITECOINADDRESS), vehicle (VIN, VRM, IMEI), employment (OCCUPATION, JOBTITLE, JOBDEPARTMENT, ORGANIZATION) and more.
- **217-class BIOES head** (54 categories × {B,I,E,S} + O background) — different shape from OPF's 33-class head.
- **Multilingual:** trained on 16 languages. Should crush vanilla OPF on non-English.
- **Not OPF checkpoint format** — packaged as a standard HuggingFace Transformers token-classification model. Cannot use `OPFDetector(model="OpenMed/...")` as-is.

**Two ways to load:**

1. **Recommended (per model card):** install `openmed[hf]`, use `extract_pii()`. Includes built-in BIOES Viterbi decoding and span refinement. Adds a dependency on the `openmed` library; check its license + maturity.

   ```python
   from openmed import extract_pii
   result = extract_pii(text, model_name="OpenMed/privacy-filter-multilingual")
   ```

2. **Direct via Transformers** with `aggregation_strategy="first"` — approximate but no extra dep.

I'd start with option 1 — let the OpenMed library do the BIOES decoding correctly. If `openmed[hf]` adds too much surface area, fall back to option 2.

**Label mapping (54 → 15 canonicals):** straightforward. Most of the categories that previously had nowhere to go are now covered by plan 09's MONEY / VEHICLE / PHYSICAL / ORGANIZATION / OCCUPATION:

  | canonical | OpenMed labels |
  | --- | --- |
  | PERSON | `FIRSTNAME`, `MIDDLENAME`, `LASTNAME`, `PREFIX` |
  | EMAIL | `EMAIL` |
  | PHONE | `PHONE`, `IMEI` |
  | ADDRESS | `STREET`, `BUILDINGNUMBER`, `SECONDARYADDRESS`, `CITY`, `COUNTY`, `STATE`, `ZIPCODE`, `GPSCOORDINATES`, `ORDINALDIRECTION` |
  | URL | `URL`, `IPADDRESS`, `MACADDRESS` |
  | DATE | `DATE`, `DATEOFBIRTH`, `TIME` |
  | ACCOUNT | `SSN`, `ACCOUNTNAME`, `BANKACCOUNT`, `IBAN`, `BIC`, `CREDITCARD`, `CREDITCARDISSUER`, `CVV`, `PIN`, `MASKEDNUMBER`, `BITCOINADDRESS`, `ETHEREUMADDRESS`, `LITECOINADDRESS` |
  | SECRET | `PASSWORD` |
  | USERNAME | `USERNAME`, `USERAGENT` |
  | DEMOGRAPHIC | `AGE`, `GENDER`, `SEX` |
  | ORGANIZATION | `ORGANIZATION` |
  | OCCUPATION | `OCCUPATION`, `JOBTITLE`, `JOBDEPARTMENT` |
  | MONEY | `AMOUNT`, `CURRENCY`, `CURRENCYCODE`, `CURRENCYNAME`, `CURRENCYSYMBOL` |
  | VEHICLE | `VIN`, `VRM` |
  | PHYSICAL | `EYECOLOR`, `HEIGHT` |

  Now covers every OpenMed category — no longer has out-of-scope labels.

## Ordering / suggested implementation sequence

Easy → hard:

1. **ai4privacy ModernBERT** (~1 hour). Lowest-friction: HF pipeline + already-mapped labels. Best return per hour for testing the multi-dataset benchmark machinery against a fresh model.
2. **Gretel small + large** (~2 hours total). Just label list + threshold + name registration. Good test of GLiNER's `prompts=` kwarg with a much larger custom prompt set.
3. **Nvidia** (~30 min). Single name registration + threshold override. Reuses default `gliner_prompts()`.
4. **OpenMed** (~3-4 hours). Most work: `openmed[hf]` install, OpenMedDetector class, 54-label taxonomy column, BIOES decoder verification.

Total: ~6-7 hours.

## Risks / open questions

- **Per-variant prompt sensitivity (GLiNER).** Gretel's snake_case labels probably need to be sent verbatim to get the model's full quality. Confirm with smoke-test runs.
- **Threshold tuning.** Gretel recommends 0.7, Nvidia recommends 0.3, our default GLiNER uses 0.5. Wrong threshold could underperform either model by 5+ F1.
- **License variation.** NVIDIA Open Model License is not Apache. Document in any writeup that compares licenses.
- **`openmed[hf]` library footprint and stability.** Brand new. Verify it doesn't pull in heavy unwanted deps (vllm, accelerate, etc.) and that its API is stable.
- **OpenMed BIOES vs OPF Viterbi.** Different decoding paths could produce different greedy-span behavior than vanilla OPF. Worth checking on the same DATE examples that OPF struggles with.
- **ai4privacy aggregation strategy.** `aggregation_strategy="first"` can split contiguous-but-different-token-id entities; `"simple"` over-merges. Smoke-test both on a known fixture to pick.
- **Disk pressure.** ai4privacy ModernBERT ~150 MB, Gretel small ~200 MB, Gretel large ~500 MB, Nvidia ~2 GB, OpenMed ~3 GB. Total ~6 GB additional download for someone running all five.
- **Memory at inference.** Gretel large + Nvidia + OpenMed + ai4privacy simultaneously may exceed Colab free tier RAM. Run them serially.

## Verification

```sh
# 1. Smoke each variant individually on the existing 100 fixtures of pii_masking_300k
for variant in ai4privacy_modernbert gliner_gretel_small gliner_gretel_large gliner_nvidia openmed_pf_multi; do
  uv run python -m opf_eval.runner \
      --dataset pii_masking_300k \
      --fixtures eval/data/sample_100.jsonl \
      --detectors $variant \
      --reuse-from eval/results/runs/run_100/ \
      --out eval/results/runs/run_100_$variant/
  uv run python -m opf_eval.report \
      --run eval/results/runs/run_100_$variant/ \
      --fixtures eval/data/sample_100.jsonl
done

# 2. Full 1k bench on 300k, reusing existing detector outputs
uv run python -m opf_eval.runner \
    --dataset pii_masking_300k \
    --fixtures eval/data/sample_1k.jsonl \
    --detectors ai4privacy_modernbert,gliner_gretel_small,gliner_gretel_large,gliner_nvidia,openmed_pf_multi \
    --reuse-from eval/results/runs/run_1k_prod_v2/ \
    --out eval/results/runs/run_1k_with_variants/

uv run python -m opf_eval.report \
    --run eval/results/runs/run_1k_with_variants/ \
    --fixtures eval/data/sample_1k.jsonl

# 3. Same on 200k — the harder vocabulary test
uv run python -m opf_eval.runner \
    --dataset pii_masking_200k \
    --fixtures eval/data/pii_masking_200k_1k.jsonl \
    --detectors ai4privacy_modernbert,gliner_gretel_small,gliner_gretel_large,gliner_nvidia,openmed_pf_multi \
    --reuse-from eval/results/runs/run_pii_masking_200k_1k/ \
    --out eval/results/runs/run_200k_1k_with_variants/

# 4. ai4privacy + an OpenPII dataset = same-vocab cleanest possible test
uv run python -m opf_eval.fixtures --dataset openpii_nano --out eval/data/openpii_nano_1k.jsonl --n 1000
uv run python -m opf_eval.runner \
    --dataset openpii_nano \
    --fixtures eval/data/openpii_nano_1k.jsonl \
    --detectors ai4privacy_modernbert,opf,gliner,skyflow \
    --skyflow-min-interval-ms 100 \
    --out eval/results/runs/run_openpii_nano_1k/
uv run python -m opf_eval.report \
    --run eval/results/runs/run_openpii_nano_1k/ \
    --fixtures eval/data/openpii_nano_1k.jsonl
```

What I'd watch for:

- **Gretel small vs large:** scaling within the same recipe. If small captures most of large's quality, recommend small.
- **Gretel English vs other languages:** confirm the English-only training tanks recall on Dutch/German/etc. (expect 30+ F1 drop).
- **Nvidia vs urchade/gliner_multi_pii-v1:** does the larger base model justify ~3× the latency?
- **ai4privacy vs vanilla OPF on multilingual:** OpenPII-trained ModernBERT should beat vanilla OPF on non-English. The "fair view" on `openpii_nano` is the cleanest test (no vocabulary mismatch — both detectors trained on related-style data).
- **OpenMed vs vanilla OPF on per-language:** OpenMed should crush vanilla OPF on non-English. If it doesn't, the multilingual fine-tune story is overstated.
- **OpenMed vs Skyflow** (the hosted leader): the headline open-weight question. OpenMed has 54 categories and 16 languages — could be the open-weight equivalent of Skyflow in reach. If F1 is competitive, this is the strongest local-deployment story we've found.
- **DATE recall on OpenMed:** does the 217-class BIOES head avoid OPF's greedy-span pathology? If yes, that's the missing piece OPF was lacking.
- **ai4privacy on the new canonicals (200k):** does it fire on `MONEY`/`OCCUPATION`/etc. that the OpenPII training set doesn't include? Probably not — that's a coverage gap to surface in raw view.

## Effort

- **ai4privacy ModernBERT:** ~1 hour (HF pipeline + 30-line wrapper; labels already mapped)
- **Gretel small + large:** ~1 hour each (label list known, just runner registration + taxonomy column + threshold)
- **Nvidia:** ~30 min (single name registration + threshold override)
- **OpenMed:** ~3-4 hours including (a) library install + smoke test, (b) `OpenMedDetector` class, (c) 54-label taxonomy column, (d) handling the new `openmed` dep + BIOES output shape
- **Total:** ~6-7 hours

## Out of scope

- Fine-tuning these variants ourselves (covered in [plan 02](02-finetune-opf.md))
- Quantized variants (e.g. OpenMed has MLX-8bit variants for Apple Silicon — could be added later)
- Per-prompt grid search per model — start with documented defaults
- Comparing GLiNER bi-encoder vs uni-encoder architectures formally — we just pick what each variant ships
- Re-training the ai4privacy ModernBERT on PII-Masking-300k specifically (closest to plan 02's fine-tune-OPF idea, applied to a different base model)
