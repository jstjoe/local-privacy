# Plan 12 — Roadmap: public benchmarks, custom data, newer models, notebook series

## Why

The harness began as "OPF vs Skyflow on one ai4privacy dataset". It's becoming a
general PII-detection benchmark and a content series. That needs:

1. **Public benchmarks** beyond ai4privacy, so results mean something outside
   one synthetic distribution.
2. **Custom datasets** anyone can plug in without editing core modules.
3. A **notebook series** that teaches the concepts module by module.
4. **Current models**: several strong open-weight PII detectors shipped in 2026.

## Status

| step | what | status |
| --- | --- | --- |
| 1 | Foundations: detector registry, two-level taxonomy, custom datasets, fixture validation, run-dir guards, shared scoring module | **done** |
| 2 | Split the workshop notebooks into the 00–06 series ([notebooks/README.md](../notebooks/README.md)) | **done** |
| 3 | Public benchmark loaders (below) | next |
| 4 | New detectors (below) | next |
| 5 | Cross-benchmark report: detectors × benchmarks in one table | after 3 + 4 |

### What step 1 changed (for anyone extending the harness)

- **Labels.** There are 15 coarse categories (unchanged) plus 24 fine sub-types:
  `GOV_ID`, `MEDICAL_ID`, `EMPLOYEE_ID`, `BANK_ACCOUNT`, `GIVEN_NAME`,
  `IP_ADDRESS`, and so on. Each source has its own vocabulary in
  [`taxonomy/vocabs.py`](../eval/src/opf_eval/taxonomy/vocabs.py); `taxonomy.register_vocab()` adds one at runtime.
  Reports take `--level fine`. The coarse view is pinned to the pre-refactor
  mapping by [`tests/data/taxonomy_v1_snapshot.json`](../eval/tests/data/taxonomy_v1_snapshot.json).
- **Detectors.** Each is a `DetectorSpec` in [`detectors/registry.py`](../eval/src/opf_eval/detectors/registry.py), with
  lazy imports. `register_gliner_model()` and `register_hf_token_classifier()`
  each add a new checkpoint in one call. The API builds from the same registry.
- **Datasets.** `register_dataset()` takes a local file or an HF id, a schema
  (`spans`, `privacy_mask` or `bio`), and an optional `label_map`. Fixtures
  get a `.meta.json` sidecar, which the runner and report read.

## Step 3 — public benchmarks

Research notes are from Sept 2026. The sandbox these were gathered in couldn't
reach huggingface.co or arxiv.org, so HF-card details (license, size) marked
**unverified** need a check before relying on them.

| priority | benchmark | source | license | format → loader | notes |
| --- | --- | --- | --- | --- | --- |
| 1 | **Tonic PrivacyBench** | HF `TonicAI/Privacy-Bench`; metrics code [TonicAI/privacy_bench_metrics](https://github.com/TonicAI/privacy_bench_metrics) | unverified ("to be confirmed" in repo) | human-annotated split: row text + char spans → `spans` | Synthetic workplace exports (Slack, email, documents). Use the **human-annotated** files (6 of 21 sets, 5,198 messages, 8,805 spans, 5 labels: given/family name, email, username, org) for P/R/F1. The generated gold only covers seed characters, so it's recall-only. Format changed 2026-09-22 ("v2 / HF-native"), so pin `hf_revision`. Page gold uses 10 labels, including `EMPLOYEE_ID` and `ACCOUNT_NUMBER`. |
| 2 | **NVIDIA Nemotron-PII** | HF `nvidia/Nemotron-PII` | CC BY 4.0 (unverified) | char spans → `spans` + new vocab | 100k English records across 50+ industries and 55+ PII/PHI types. Fine labels (`ssn`, `medical_record_number`, `employee_id`, …) map directly onto the new sub-types. It's nvidia/gliner-PII's training data, so flag in-distribution results. |
| 3 | **TAB** (Text Anonymization Benchmark) | [NorskRegnesentral/text-anonymization-benchmark](https://github.com/NorskRegnesentral/text-anonymization-benchmark) | MIT | standoff char offsets, several annotators → custom loader | Real ECHR court text rather than synthetic. Labels PERSON / CODE / LOC / ORG / DEM / DATETIME / QUANTITY / MISC. Pick one annotator per doc or take the majority. |
| 4 | **PII-TRACE** | HF `perplexity-ai/PII-TRACE` | MIT (unverified) | multi-turn dialogues → flatten turns, shift offsets | 13 languages. It's the eval set behind Perplexity's PII-Tracer model. |
| 5 | Gretel finance multilingual / en-v1 | HF `gretelai/synthetic_pii_finance_multilingual`, `gretelai/gretel-pii-masking-en-v1` | Apache-2.0 (unverified) | spans (verify) | Financial identifiers (IBAN, SWIFT, API keys); 7 languages |
| 6 | ai4privacy openpii-1m / open-pii-masking-500k | HF `ai4privacy/pii-masking-openpii-1m`, `…/open-pii-masking-500k-ai4privacy` | CC BY 4.0; **500k also carries the Llama Community License** | `privacy_mask` (existing loader) | Successors to the sets we already run. |
| later | SPY (medical/legal) | [LogicZMaksimka/SPY_Dataset](https://github.com/LogicZMaksimka/SPY_Dataset) | **no license file** | tokens + trailing whitespace → `bio` | The `bio` loader already handles its format. The licence blocks publishing results. |
| later | REDACT (ServiceNow, 25 langs), PIIMB, RedactionBench, DialogPII | arXiv 2606.19881, HF `piimb/pii-masking-benchmark`, arXiv 2606.18782, Zenodo 20863452 | unverified | varies | Check release status first. RedactionBench is untyped ("must redact"), so it would be scored with Exact/Partial only. |

### micro1

micro1 hasn't published a PII benchmark dataset. Their [Sept 24, 2026 write-up](https://www.micro1.ai/research/pii-transformation-for-enterprise-datasets)
introduces **flow-transform 1.0** and an **Enterprise De-Identification
Bench**: synthetic tabular data for one company, with hostile formatting,
decoys, and a five-pillar "Transformation Quality Index". That bench isn't
released; they say they hope to release the generator and scoring code.

Their public detection checks are reproducible with this harness and make a
good external reference protocol:

- PrivacyBench human-annotated gold, P/R/F1 under overlap and exact matching. Their reported overlap / exact F1:

  | system | overlap F1 | exact F1 |
  | --- | --- | --- |
  | flow-transform | 96.0 | 93.4 |
  | Tonic Textual | 94.5 | 92.3 |
  | Presidio | 88.1 | 86.2 |
  | GLiNER2 | 86.9 | 84.4 |

  Presidio and GLiNER2 are the ones we can reproduce ourselves, so they double as a cross-check on our loader and scoring.
- 500 docs sampled from the Nemotron-PII **test** split, strict exact-span P/R/F1.
- 500 English records from the open-pii-masking-500k **validation** split, strict exact-span.

Two ideas worth borrowing later: **decoys** (SSN-shaped part numbers,
IP-shaped version strings) to measure false-positive robustness, and
**hostile formatting** (spaced digits, homoglyphs, zero-width characters) as a
stress-test dataset. Both could be synthetic fixtures generated by us.

### Loader work per benchmark

1. Add a vocabulary to [`taxonomy/vocabs.py`](../eval/src/opf_eval/taxonomy/vocabs.py) (or register one in the loader module).
2. Register the dataset in [`datasets/__init__.py`](../eval/src/opf_eval/datasets/__init__.py) with `hf_revision` pinned. Add a custom loader only when the schema isn't `spans` / `privacy_mask` / `bio`.
3. Record license, languages and the URL on the `DatasetConfig` (shown in `datasets.describe()`).
4. Add a tiny checked-in sample to `eval/tests/data/` and a loader test.
5. Add a row to the README datasets table.

## Step 4 — detectors to add

Ranked by value for effort. The numbers are the publishers' claims, or
[PII Arena](https://github.com/Quakeer444/pii-arena) (an independent
41-dataset leaderboard, frozen 2026-09-09), which is itself a useful
reference harness.

| priority | model | license | how | notes |
| --- | --- | --- | --- | --- |
| 1 | `fastino/gliner2-privacy-filter-PII-multi` (GLiNER2-PII) | Apache-2.0 | new `gliner2` adapter (`pip install gliner2`) | #1 on PII Arena, best on SPY. 7 languages, 512-token window, so split long inputs. Use the returned offsets, not `text.find()`. |
| 2 | `perplexity-ai/pplx-pii-masking`, `perplexity-ai/PII-Tracer` | MIT | custom adapter (`trust_remote_code`, transformers ≥ 5.2) | Bidirectional Qwen3 0.6B with a label set close to OPF's plus `other_pii`. **Split inputs at 4096 tokens**: the tokenizer wrongly advertises 131k. |
| 3 | `knowledgator/gliner-pii-{edge,small,base,large}-v1.0` | Apache-2.0 | `register_gliner_model(...)` — no new code | Large leads PIIMB exact F1. Edge ships a UINT8 ONNX build, a good CPU/edge data point. |
| 4 | `LiquidAI/LFM2.5-Encoder-350M-PII-Detector` | LFM Open License (**revenue threshold**) | `register_hf_token_classifier` (verify) | 8k context, fast on CPU, 16 languages. Benchmark it, but flag the license. |
| 5 | `OpenMed/privacy-filter-nemotron` (v1) | Apache-2.0 | OPF adapter with a new label map | OPF architecture with 55 fine labels: isolates "same model, finer taxonomy". Avoid v2 until its license is clarified. The existing [`openmed.py`](../eval/src/opf_eval/detectors/openmed.py) notes loader issues with the `openai_privacy_filter` model type. |
| 6 | `tabularisai/eu-pii-safeguard` | Apache-2.0 | `register_hf_token_classifier` | XLM-R large, 26 EU languages, EU identifiers (IBAN, national IDs). |
| alt | `DataikuNLP/kiji-pii-model-onnx` | Apache-2.0 | HF / ONNX | Ultra-light DistilBERT INT8 CPU baseline |

Skip: Piiranha (CC BY-NC-ND), small-LLM fine-tunes (they trail encoders on
RedactionBench and give unreliable offsets), and Granite Guardian / Llama
Guard (classifiers, no spans).

Integration notes:

- Every model above misreports `tokenizer.model_max_length`. Hard-code each
  one's window in its registration, and add shared chunking to the base
  detector.
- Put the published license on every `DetectorSpec`. The notebooks show it.

## Step 5 — cross-benchmark report

`opf_eval.suite`: run a list of (dataset, n, seed) samples × detectors into
one run dir per sample, then emit a single table of detectors × benchmarks
(fair Strict F1 at coarse and fine levels, plus Type F1) with a coverage
footnote per cell. This is likely the headline asset for the content series.
It also feeds `design/` (the benchmark canvas).
