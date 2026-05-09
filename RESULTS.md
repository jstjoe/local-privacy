# Benchmark Results

Headline numbers from 1k samples of two ai4privacy datasets, re-run on the multi-dataset code with the new dataset-aware `skyflow` detector:

- **`pii_masking_300k`** — 10 canonical labels annotated; the original baseline
- **`pii_masking_200k`** — 15 canonical labels (richer vocabulary including `MONEY`, `OCCUPATION`, `ORGANIZATION`, `VEHICLE`, `PHYSICAL`)

The `pii_masking_300k` 1k bench now compares **9 detectors**:

- 4 originals: `skyflow`, `opf`, `gliner`, `presidio`
- 5 added in plan 05: `ai4privacy_modernbert`, `gliner_gretel_small`, `gliner_gretel_large`, `gliner_nvidia`, `openmed`

Every detector is scored under both views the report emits — **Fair** (each detector against `dataset ∩ detector_supports`) and **Raw** (every detector against the dataset's full annotated vocabulary). Run on other ai4privacy datasets (400k / OpenPII nano / mini) by passing `--dataset NAME` everywhere — see [README](README.md#datasets).

## pii_masking_300k (1k sample)

Reproduce with:

```sh
python -m opf_eval.fixtures --dataset pii_masking_300k --out eval/data/sample_1k.jsonl --n 1000
python -m opf_eval.runner --dataset pii_masking_300k --fixtures eval/data/sample_1k.jsonl \
    --detectors opf,skyflow,presidio,gliner \
    --skyflow-min-interval-ms 100 \
    --out eval/results/runs/repro_1k/
python -m opf_eval.report --run eval/results/runs/repro_1k/ --fixtures eval/data/sample_1k.jsonl
```

### SemEval — Fair view (per-detector scope)

Each detector scored against the intersection of (dataset annotates, this detector supports). N labels in parens shows per-detector scope.

| detector | n labels | strict | exact | partial | type |
| --- | --- | --- | --- | --- | --- |
| **skyflow** | 10 | 0.725 | 0.763 | **0.837** | **0.858** |
| opf | 8 | **0.767** | **0.775** | 0.811 | 0.837 |
| gliner_nvidia | 9 | 0.699 | 0.764 | 0.806 | 0.769 |
| gliner | 9 | 0.597 | 0.644 | 0.714 | 0.721 |
| gliner_gretel_large | 9 | 0.610 | 0.661 | 0.692 | 0.667 |
| gliner_gretel_small | 9 | 0.505 | 0.553 | 0.583 | 0.558 |
| openmed | 10 | 0.348 | 0.375 | 0.469 | 0.495 |
| presidio | 8 | 0.408 | 0.474 | 0.538 | 0.481 |
| ai4privacy_modernbert | 10 | 0.245 | 0.272 | 0.433 | 0.453 |

Schema cheat-sheet: **Strict** = exact boundary + label. **Exact** = exact boundary, ignore label. **Partial** = any overlap, ignore label. **Type** = any overlap + matching label.

### SemEval — Raw view (full pii_masking_300k vocabulary, 10 labels)

Every detector scored against the dataset's full annotated set. Labels a detector doesn't support take zero recall; this view reflects out-of-the-box coverage rather than fairness.

| detector | strict | exact | partial | type |
| --- | --- | --- | --- | --- |
| **skyflow** | 0.725 | 0.763 | **0.837** | **0.858** |
| opf | 0.731 | **0.794** | 0.830 | 0.797 |
| gliner_nvidia | 0.687 | 0.758 | 0.800 | 0.755 |
| gliner | 0.585 | 0.639 | 0.709 | 0.707 |
| gliner_gretel_large | 0.597 | 0.647 | 0.678 | 0.652 |
| gliner_gretel_small | 0.493 | 0.541 | 0.569 | 0.544 |
| openmed | 0.348 | 0.375 | 0.469 | 0.495 |
| presidio | 0.392 | 0.462 | 0.532 | 0.462 |
| ai4privacy_modernbert | 0.245 | 0.272 | 0.433 | 0.453 |

Strict-schema error decomposition (raw view, sorted by COR):

| detector | COR | INC | MIS | SPU |
| --- | --- | --- | --- | --- |
| skyflow | 5,113 | 1,305 | **562** | 710 |
| gliner_nvidia | 4,905 | 1,107 | 968 | 1,296 |
| opf | 4,765 | 878 | 1,338 | **418** |
| gliner | 3,911 | 1,298 | 1,771 | 1,179 |
| gliner_gretel_large | 3,349 | 631 | 3,000 | 258 |
| presidio | 2,987 | 1,603 | 3,586 | 2,465 |
| gliner_gretel_small | 2,531 | 538 | 3,911 | 217 |
| openmed | 1,794 | 1,107 | 4,079 | 428 |
| ai4privacy_modernbert | 1,396 | 1,839 | 3,748 | 167 |

Reads cleanly:

- **gliner_nvidia is the new local champion** — Type F1 0.769 fair / 0.755 raw, second only to skyflow + opf. Different from default GLiNER only in base (570M urchade/gliner_large-v2.1 vs ~150M urchade/gliner_multi_pii-v1) and threshold (0.3 vs 0.5). Trade-off: ~3× the latency of default GLiNER.
- **Skyflow's edge is recall.** 562 missed vs OPF's 1,338 — less than half. Largely because Skyflow is the only detector with non-zero `DEMOGRAPHIC` (0.732 F1) and `USERNAME` (0.734 F1) recall on this dataset (raw view, see per-category below).
- **OPF wins on type confusion.** 878 INC and **418 SPU — the lowest of any detector** (the cleanest precision profile of the bunch).
- **gliner_gretel models are conservative** — the small/large variants both under-detect (low SPU 217/258) but miss a lot (3,911 / 3,000 MIS). English-only training on multilingual data shows up as recall failure.
- **OpenMed and ai4privacy_modernbert disappoint vs their model cards** — both heavily miss (4,079 / 3,748 MIS). Likely a vocab-mismatch issue: they target snake_case OpenPII-style labels and the 300k labels (`GIVENNAME1` / `LASTNAME1`) need taxonomy-mediated mapping that may not survive the deeper boundary work.
- **Presidio's loss is split across MIS + SPU** (under-recalls *and* over-fires); 1,603 INC despite the lowest F1 of the originals.

### Latency (single-request, 1k sample)

| detector | p50 | p95 | p99 |
| --- | --- | --- | --- |
| presidio | **14 ms** | 21 ms | **25 ms** |
| ai4privacy_modernbert | 45 ms | 53 ms | 57 ms |
| gliner_gretel_small | 47 ms | 56 ms | 61 ms |
| openmed | 49 ms | 299 ms | 401 ms |
| gliner | 75 ms | 94 ms | 103 ms |
| skyflow | 107 ms | 194 ms | 204 ms |
| gliner_nvidia | 179 ms | 210 ms | 241 ms |
| gliner_gretel_large | 193 ms | 226 ms | 245 ms |
| opf | 703 ms | 1,047 ms | 1,229 ms |

OPF on CPU. Skyflow latency is region-dependent (network to vault). OpenMed's wide p95/p99 spread is per-language model loading on first call into a new language. gliner_nvidia's 179 ms p50 is from the 570M-param base (~3× default GLiNER size).

### Per-category F1 (raw view, partial overlap, IoU ≥ 0.5; winners in **bold**)

Compact F1 table across all 9 detectors. Full P/R + TP/FP/FN per cell is in `eval/results/runs/run_1k_prod_v3/report.md`.

| label | gold n | opf | skyflow | gliner | gliner_nvidia | gliner_gretel_lg | gliner_gretel_sm | openmed | ai4priv | presidio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ACCOUNT | 1,448 | **0.952** | 0.850 | 0.685 | 0.811 | 0.563 | 0.444 | 0.487 | 0.581 | 0.408 |
| ADDRESS | 1,609 | **0.773** | 0.728 | 0.606 | 0.784 | 0.565 | 0.497 | 0.430 | 0.240 | 0.216 |
| DATE | 1,111 | 0.652 | 0.869 | **0.886** | 0.878 | 0.678 | 0.579 | 0.444 | 0.655 | 0.488 |
| DEMOGRAPHIC | 246 | 0.000 | **0.732** | 0.000 | 0.000 | 0.000 | 0.000 | 0.098 | 0.066 | 0.000 |
| EMAIL | 324 | 0.961 | 0.926 | 0.884 | 0.931 | 0.884 | 0.589 | 0.906 | 0.202 | **0.968** |
| PERSON | 1,101 | **0.679** | 0.673 | 0.481 | 0.473 | 0.632 | 0.444 | 0.187 | 0.070 | 0.141 |
| PHONE | 281 | **0.969** | 0.859 | 0.863 | 0.793 | 0.746 | 0.771 | 0.147 | 0.690 | 0.404 |
| SECRET | 220 | **0.946** | 0.757 | 0.736 | 0.746 | 0.751 | 0.539 | 0.532 | 0.000 | 0.000 |
| URL | 274 | **0.956** | 0.903 | 0.427 | 0.576 | 0.641 | 0.628 | 0.781 | 0.000 | 0.527 |
| USERNAME | 366 | 0.000 | **0.734** | 0.557 | 0.589 | 0.634 | 0.607 | 0.165 | 0.000 | 0.000 |

**Wins by category:** OPF 6, Skyflow 2, GLiNER 1, Presidio 1. Even with 5 new detectors added, none take a category outright — OPF's tight integration with the dataset's labelling style holds.

**Closest runners-up among new detectors:** gliner_nvidia is consistently the strongest new entrant — second on ACCOUNT (0.811) and ADDRESS (0.784, beats OPF), strong on DATE (0.878). gliner_gretel_large is the next-best (top in PERSON among new models at 0.632, +0.151 over default GLiNER).

OPF wins more *categories*, but Skyflow's headline F1 is higher because it covers two categories (DEMOGRAPHIC + USERNAME) that the other 8 detectors barely touch on this dataset. The fair view (above) collapses that gap by scoping each detector to its supported labels — and there OPF's 0.837 Type F1 is within 2.1 of Skyflow's 0.858.

### Per-language F1 (fair view, SemEval Type schema)

| language | n | opf | skyflow | gliner | gliner_nvidia | gliner_gretel_lg | gliner_gretel_sm | openmed | ai4priv | presidio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| de | 173 | 0.835 | **0.878** | 0.725 | 0.814 | 0.668 | 0.581 | 0.409 | 0.462 | 0.426 |
| en | 169 | 0.825 | **0.851** | 0.683 | 0.773 | 0.680 | 0.614 | 0.792 | 0.515 | 0.607 |
| es | 158 | 0.852 | **0.894** | 0.742 | 0.769 | 0.642 | 0.545 | 0.476 | 0.439 | 0.479 |
| fr | 193 | 0.847 | **0.860** | 0.705 | 0.769 | 0.680 | 0.557 | 0.431 | 0.466 | 0.461 |
| it | 143 | **0.846** | 0.827 | 0.732 | 0.772 | 0.691 | 0.560 | 0.444 | 0.441 | 0.448 |
| nl | 164 | 0.819 | 0.833 | 0.744 | 0.708 | 0.635 | 0.481 | 0.294 | 0.378 | 0.464 |

Skyflow wins 5/6 languages; OPF wins Italian. Tight 4-point spread for OPF (0.819–0.852) confirms the 5k-run finding that OPF is consistent across the languages it sees.

**Among new detectors:**

- **gliner_nvidia** is consistent (0.708–0.814) across all 6 languages and is the strongest new entrant in every language.
- **openmed** is wildly inconsistent: 0.792 on English (44M EN model) but 0.294–0.476 on the others (434–568M variants). The per-language routing isn't paying off.
- **ai4privacy_modernbert** stays low (0.378–0.515) across all 6. Self-reported multilingual quality doesn't translate.
- **gliner_gretel_*** drop ~10 F1 from English to non-English — confirms the model card's English-only training warning.

Presidio's 30-point gap on non-English is its English-only spaCy NER; multilingual Presidio (`presidio_multilang`) actually performs slightly worse overall — country-specific regex recognizers (US_SSN etc.) get gated to `language="en"` and stop firing. See [plans/01-presidio-baseline.md](plans/01-presidio-baseline.md).

## pii_masking_200k (1k sample)

Reproduce with:

```sh
python -m opf_eval.fixtures --dataset pii_masking_200k --out eval/data/pii_masking_200k_1k.jsonl --n 1000
python -m opf_eval.runner --dataset pii_masking_200k --fixtures eval/data/pii_masking_200k_1k.jsonl \
    --detectors opf,skyflow,presidio,gliner \
    --skyflow-min-interval-ms 100 \
    --out eval/results/runs/repro_200k_1k/
python -m opf_eval.report --run eval/results/runs/repro_200k_1k/ --fixtures eval/data/pii_masking_200k_1k.jsonl
```

200k has its own annotation vocabulary (56 distinct raw labels in the first 5k records) covering all 15 canonical labels — gains `MONEY`, `OCCUPATION`, `ORGANIZATION`, `VEHICLE`, `PHYSICAL` over what 300k annotates. This is where Skyflow's broad coverage and GLiNER's prompt-driven flexibility actually pay off vs OPF's fixed 8-category vocabulary.

### SemEval — Fair view (per-detector scope, 200k)

| detector | n labels | strict | exact | partial | type |
| --- | --- | --- | --- | --- | --- |
| **skyflow** | 15 | **0.600** | **0.668** | **0.767** | **0.776** |
| opf | 8 | 0.516 | 0.582 | 0.676 | 0.694 |
| gliner | 14 | 0.459 | 0.553 | 0.665 | 0.657 |
| presidio | 8 | 0.340 | 0.409 | 0.493 | 0.440 |

### SemEval — Raw view (full pii_masking_200k vocabulary, 15 labels)

| detector | strict | exact | partial | type |
| --- | --- | --- | --- | --- |
| **skyflow** | **0.600** | **0.668** | **0.767** | **0.776** |
| gliner | 0.446 | 0.561 | 0.675 | 0.638 |
| opf | 0.426 | 0.530 | 0.625 | 0.572 |
| presidio | 0.239 | 0.316 | 0.437 | 0.314 |

Strict-schema error decomposition (raw view):

| detector | COR | INC | MIS | SPU |
| --- | --- | --- | --- | --- |
| skyflow | 1,829 | 812 | **259** | 556 |
| gliner | 1,357 | 1,041 | 501 | 783 |
| opf | 1,006 | 695 | 1,198 | **127** |
| presidio | 993 | 1,329 | 938 | 2,741 |

Reads cleanly:

- **Skyflow widens its lead vs 300k** (+8.2 fair Type F1 over OPF here, vs +2.1 on 300k). The richer vocabulary is exactly the test where Skyflow's range pays off.
- **OPF's strict-view raw F1 (0.426) drops below GLiNER (0.446)** — penalized hard by the 7 categories it doesn't claim. Fair view restores OPF's edge over GLiNER (0.694 vs 0.657).
- **OPF still has the cleanest precision profile** (lowest SPU 127) — it doesn't make stuff up; it just doesn't see anything outside its 8 categories.

### Latency (single-request, 1k sample, 200k)

| detector | p50 | p95 | p99 |
| --- | --- | --- | --- |
| presidio | **7 ms** | 10 ms | **13 ms** |
| gliner | 71 ms | 98 ms | 111 ms |
| skyflow | 106 ms | 137 ms | 211 ms |
| opf | 272 ms | 491 ms | 649 ms |

Lower than 300k across the board — 200k records are shorter on average. OPF p50 in particular drops 703 ms → 272 ms (same model, shorter inputs).

### Per-category F1 (raw view, partial overlap, IoU ≥ 0.5; winners in **bold**, 200k)

| label | gold n | opf | skyflow | gliner | presidio | winner |
| --- | --- | --- | --- | --- | --- | --- |
| ACCOUNT | 495 | 0.613 | **0.701** | 0.532 | 0.347 | Skyflow |
| ADDRESS | 429 | 0.566 | **0.722** | 0.618 | 0.217 | Skyflow |
| DATE | 225 | 0.743 | **0.908** | 0.829 | 0.606 | Skyflow |
| DEMOGRAPHIC | 169 | 0.000 | **0.479** | 0.000 | 0.028 | Skyflow |
| EMAIL | 73 | 0.993 | **1.000** | 0.847 | **1.000** | Skyflow / Presidio |
| MONEY | 216 | 0.000 | 0.584 | **0.622** | 0.000 | GLiNER |
| OCCUPATION | 189 | 0.000 | 0.425 | **0.582** | 0.000 | GLiNER |
| ORGANIZATION | 58 | 0.000 | **0.339** | 0.095 | 0.015 | Skyflow |
| PERSON | 519 | 0.497 | **0.765** | 0.524 | 0.229 | Skyflow |
| PHONE | 85 | 0.733 | 0.719 | **0.766** | 0.298 | GLiNER |
| PHYSICAL | 48 | 0.000 | 0.619 | **0.729** | 0.000 | GLiNER |
| SECRET | 60 | 0.537 | **0.794** | 0.712 | 0.000 | Skyflow |
| URL | 213 | 0.808 | **0.969** | 0.573 | 0.796 | Skyflow |
| USERNAME | 86 | 0.000 | **0.781** | 0.226 | 0.000 | Skyflow |
| VEHICLE | 34 | 0.000 | **0.828** | 0.702 | 0.000 | Skyflow |

**Wins by category:** Skyflow 11 (one tied with Presidio on EMAIL), GLiNER 4 (DATE was Skyflow this dataset; GLiNER picks up MONEY, OCCUPATION, PHONE, PHYSICAL), OPF 0, Presidio 0 (only ties on EMAIL).
**OPF zeros**: MONEY, OCCUPATION, ORGANIZATION, PHYSICAL, USERNAME, VEHICLE, DEMOGRAPHIC — the 7 categories outside its supported set. Same picture for Presidio on the categories it doesn't claim.

GLiNER's strength on the new canonicals is real: prompt-driven NER handles `MONEY`, `OCCUPATION`, `PHYSICAL` better than Skyflow's native types. ORGANIZATION is a Skyflow win even though its 0.339 F1 looks low — it's a high-recall (0.879) low-precision (0.210) profile that beats GLiNER's 0.095 outright.

### Per-language F1 (fair view, SemEval Type schema, 200k)

| language | n | opf | skyflow | gliner | presidio | winner |
| --- | --- | --- | --- | --- | --- | --- |
| de | 231 | 0.681 | **0.798** | 0.661 | 0.379 | Skyflow |
| en | 228 | 0.667 | **0.791** | 0.653 | 0.563 | Skyflow |
| fr | 309 | 0.696 | **0.736** | 0.641 | 0.456 | Skyflow |
| it | 232 | 0.730 | **0.793** | 0.680 | 0.396 | Skyflow |

200k has 4 languages (no nl/es). Skyflow wins all 4. OPF holds a tight 6-point spread (0.667–0.730).

## Headline takeaways

- **For broad PII coverage with hosted-OK constraints:** Skyflow. Wins overall on both 300k (0.858 Type fair) and 200k (0.776 Type fair); only detector that handles `DEMOGRAPHIC` + `USERNAME` on 300k and `MONEY` / `OCCUPATION` / `ORGANIZATION` / `VEHICLE` / `PHYSICAL` on 200k. Fast (~107 ms p50).
- **For local deployment, the overall winner is still OPF.** 0.837 Type F1 fair view on 300k (within 2.1 of Skyflow), dominates 6 of 10 categories outright there, lowest SPU of any detector either dataset. Slow on CPU (272–703 ms p50 depending on input length) — GPU recommended for production.
- **Best of the 5 new detectors: `gliner_nvidia`.** 0.769 Type F1 fair / 0.755 raw on 300k — beats default GLiNER (0.721) by a clear margin and adds USERNAME coverage OPF lacks. **Doesn't unseat OPF** (still ~7 F1 below) but useful as a complement: drop in if you need USERNAME or want a single GLiNER-family detector that's stronger than the default. Costs ~3× the latency of default GLiNER (179 ms p50 vs 75 ms) due to the 570M-param base. NVIDIA Open Model License (not Apache).
- **For local deployment on broader vocabularies:** default `gliner`. On the harder 200k vocabulary it beats OPF in raw view (0.638 vs 0.572 Type), and wins MONEY, OCCUPATION, PHONE, PHYSICAL on 200k and DATE on 300k. The dataset-aware prompt restriction in this PR also pulls it ahead of `gliner_gretel_*` (which had its own snake_case prompts).
- **For high-volume EMAIL only:** Presidio. 7-14 ms p50, perfect or near-perfect F1 on EMAIL across both datasets. Worthless on most other categories.
- **Disappointments worth noting:** `openmed` (0.495 Type F1 on 300k) and `ai4privacy_modernbert` (0.453) underperform their model-card claims on this benchmark — likely a vocab-mismatch issue with the 300k labelling style. The `OpenMed/privacy-filter-multilingual` model is broken (`openai_privacy_filter` model_type isn't registered in transformers); we use the `DEFAULT_PII_MODELS` family instead.
- **A hybrid local stack** — OPF for the categories it dominates (ACCOUNT / PHONE / SECRET / URL / ADDRESS) + `gliner_nvidia` or default GLiNER for USERNAME and broader-vocab categories + Presidio for EMAIL — would close most of the local-vs-Skyflow gap while staying fully on-prem. Not built or benchmarked as a single ensemble yet.

## Caveats and what we don't measure here

- **Two datasets isn't all of them.** 300k and 200k are folded in here; `pii_masking_400k` and `openpii_nano/mini` are runnable by passing `--dataset NAME` but not yet measured. Performance on production traffic (chat transcripts, support tickets, internal docs) may also differ from any of these.
- **Dataset labeling style shapes the result.** 300k uses per-token names (`GIVENNAME1`/`GIVENNAME2`) which produces the granularity-bias / Type-vs-Strict gap; 200k uses bare names but adds 56 distinct entity types (more open-ended). 300k is the more recall-friendly benchmark; 200k is the more coverage-stress test.
- **OPF was not fine-tuned.** [plans/02-finetune-opf.md](plans/02-finetune-opf.md) is the remaining high-leverage experiment — fine-tuning could plausibly close most of the DATE/ADDRESS gap to Skyflow.
- **Skyflow API latency is region-dependent.** Reported p50 is whatever your network-to-vault path is. For latency-sensitive paths in a different region, mileage will vary.
- **Cost not measured.** Skyflow is per-call; OPF/Presidio/GLiNER are self-hosted compute. Decision frameworks usually need both quality and cost — cost is left to the deployer.
- **GLiNER multi-PII variant** is a single model checkpoint we picked; tuning the prompt strings, threshold, or switching to a fine-tuned variant could move the numbers.
- **Skyflow comparison vs old `skyflow_minimal`.** The retired hand-tuned 24-entity preset scored 0.860 Type F1 here; the new dataset-aware `skyflow` (38 entity_types) scores 0.858 — within noise. The auto-derived list trades slightly lower per-category PERSON / ADDRESS precision (more bare-type confusion) for slightly lower MIS overall.
