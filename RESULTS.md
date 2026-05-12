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
| **skyflow** | 10 | **0.778** | **0.812** | **0.857** | **0.852** |
| opf | 8 | 0.767 | 0.775 | 0.811 | 0.837 |
| gliner_nvidia | 9 | 0.698 | 0.740 | 0.776 | 0.760 |
| gliner | 9 | 0.577 | 0.608 | 0.671 | 0.692 |
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
| **skyflow** | **0.778** | 0.812 | **0.857** | **0.852** |
| opf | 0.731 | 0.794 | 0.830 | 0.797 |
| gliner_nvidia | 0.684 | 0.729 | 0.765 | 0.746 |
| gliner | 0.565 | 0.601 | 0.664 | 0.677 |
| gliner_gretel_large | 0.597 | 0.647 | 0.678 | 0.652 |
| gliner_gretel_small | 0.493 | 0.541 | 0.569 | 0.544 |
| openmed | 0.348 | 0.375 | 0.469 | 0.495 |
| presidio | 0.392 | 0.462 | 0.532 | 0.462 |
| ai4privacy_modernbert | 0.245 | 0.272 | 0.433 | 0.453 |

Strict-schema error decomposition (raw view, sorted by COR):

| detector | COR | INC | MIS | SPU |
| --- | --- | --- | --- | --- |
| skyflow | 5,489 | 872 | **619** | 776 |
| gliner_nvidia | 2,575 | 437 | 1,087 | 414 |
| opf | 4,765 | 878 | 1,338 | **418** |
| gliner | 3,453 | 995 | 2,532 | 801 |
| gliner_gretel_large | 3,349 | 631 | 3,000 | 258 |
| presidio | 2,987 | 1,603 | 3,586 | 2,465 |
| gliner_gretel_small | 2,531 | 538 | 3,911 | 217 |
| openmed | 1,794 | 1,107 | 4,079 | 428 |
| ai4privacy_modernbert | 1,396 | 1,839 | 3,748 | 167 |

Reads cleanly:

- **Skyflow now wins all four schemas** — including strict (0.778, was 0.725 with the old request set that bundled `NAME` + `LOCATION` + `LOCATION_ADDRESS` parents). Switching the canonical mapping to request only the **components** Skyflow's vocabulary actually has (`NAME_GIVEN`/`NAME_FAMILY`, `LOCATION_CITY`/`STATE`/`ZIP`/etc., dropping `DAY`/`MONTH`/`YEAR` sub-units below `DATE`) lifts Strict F1 +5.3, Exact +4.9, Partial +2.0. Type F1 down 0.6 (the parent labels were padding it via boundary-tolerant overlap). Per-category PERSON +6.2 and ADDRESS +11.9 — exactly where the hierarchy mismatch was hurting.
- **Skyflow's edge is recall.** 619 missed vs OPF's 1,338 — less than half. Largely because Skyflow is the only detector with non-zero `DEMOGRAPHIC` (0.732 F1) and `USERNAME` (0.734 F1) recall on this dataset (raw view, see per-category below).
- **OPF wins on type confusion + precision.** 878 INC and **418 SPU — the lowest of any detector** (cleanest precision profile of the bunch). With the new Skyflow request set OPF's INC count is now identical to Skyflow's (878 vs 872) — the boundary fix removed Skyflow's previous 433 INC overhang.
- **gliner_nvidia is the strongest of the 5 new detectors** — Type F1 0.760 fair / 0.746 raw, +7 over default GLiNER. Different from default GLiNER only in base (570M urchade/gliner_large-v2.1 vs ~150M urchade/gliner_multi_pii-v1); both share the standardized 0.7 threshold. At that threshold the 570M base barely loses recall (COR 4905 → 2575) but slashes false positives (SPU 1296 → 414) — high-confidence predictions survive the higher cutoff. Trade-off: ~3× the latency of default GLiNER.
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
| ACCOUNT | 1,448 | **0.952** | 0.849 | 0.685 | 0.700 | 0.563 | 0.444 | 0.487 | 0.581 | 0.408 |
| ADDRESS | 1,609 | 0.773 | **0.847** | 0.606 | 0.836 | 0.565 | 0.497 | 0.430 | 0.240 | 0.216 |
| DATE | 1,111 | 0.652 | 0.870 | **0.886** | 0.904 | 0.678 | 0.579 | 0.444 | 0.655 | 0.488 |
| DEMOGRAPHIC | 246 | 0.000 | **0.732** | 0.000 | 0.000 | 0.000 | 0.000 | 0.098 | 0.066 | 0.000 |
| EMAIL | 324 | 0.961 | 0.924 | 0.884 | 0.947 | 0.884 | 0.589 | 0.906 | 0.202 | **0.968** |
| PERSON | 1,101 | 0.679 | **0.735** | 0.481 | 0.535 | 0.632 | 0.444 | 0.187 | 0.070 | 0.141 |
| PHONE | 281 | **0.969** | 0.860 | 0.863 | 0.829 | 0.746 | 0.771 | 0.147 | 0.690 | 0.404 |
| SECRET | 220 | **0.946** | 0.757 | 0.736 | 0.797 | 0.751 | 0.539 | 0.532 | 0.000 | 0.000 |
| URL | 274 | **0.956** | 0.901 | 0.427 | 0.610 | 0.641 | 0.628 | 0.781 | 0.000 | 0.527 |
| USERNAME | 366 | 0.000 | **0.734** | 0.557 | 0.704 | 0.634 | 0.607 | 0.165 | 0.000 | 0.000 |

**Wins by category:** OPF 4 (ACCOUNT, PHONE, SECRET, URL), Skyflow 4 (ADDRESS, DEMOGRAPHIC, PERSON, USERNAME), GLiNER 1 (DATE), Presidio 1 (EMAIL). The hierarchy fix flipped ADDRESS and PERSON from OPF wins to Skyflow wins (+11.9 and +6.2 F1 respectively).

**Closest runners-up among new detectors:** gliner_nvidia is consistently the strongest new entrant — second on ADDRESS (0.836) and strong on DATE (0.904), with notable USERNAME coverage (0.704) the other GLiNER variants lack. gliner_gretel_large is the next-best (top in PERSON among new models at 0.632, +0.151 over default GLiNER).

OPF and Skyflow are now even on per-category wins (4 each); Skyflow's headline F1 lead comes from owning DEMOGRAPHIC + USERNAME outright (the other 8 detectors barely touch them on 300k). The fair view above collapses that gap by scoping each detector to its supported labels — and there OPF's 0.837 Type F1 is within 1.5 of Skyflow's 0.852.

### Per-language F1 (fair view, SemEval Type schema)

| language | n | opf | skyflow | gliner | gliner_nvidia | gliner_gretel_lg | gliner_gretel_sm | openmed | ai4priv | presidio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| de | 173 | 0.835 | **0.869** | 0.725 | 0.798 | 0.668 | 0.581 | 0.409 | 0.462 | 0.426 |
| en | 169 | 0.825 | **0.845** | 0.683 | 0.783 | 0.680 | 0.614 | 0.792 | 0.515 | 0.607 |
| es | 158 | 0.852 | **0.893** | 0.742 | 0.754 | 0.642 | 0.545 | 0.476 | 0.439 | 0.479 |
| fr | 193 | 0.847 | **0.855** | 0.705 | 0.765 | 0.680 | 0.557 | 0.431 | 0.466 | 0.461 |
| it | 143 | **0.846** | 0.819 | 0.732 | 0.759 | 0.691 | 0.560 | 0.444 | 0.441 | 0.448 |
| nl | 164 | 0.819 | **0.829** | 0.744 | 0.696 | 0.635 | 0.481 | 0.294 | 0.378 | 0.464 |

Skyflow wins 5/6 languages; OPF wins Italian. Tight 4-point spread for OPF (0.819–0.852) confirms the 5k-run finding that OPF is consistent across the languages it sees.

**Among new detectors:**

- **gliner_nvidia** is consistent (0.696–0.798) across all 6 languages and is the strongest new entrant in every language.
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
| **skyflow** | 15 | **0.637** | **0.700** | **0.778** | **0.763** |
| opf | 8 | 0.516 | 0.582 | 0.675 | 0.693 |
| gliner | 14 | 0.471 | 0.552 | 0.665 | 0.672 |
| presidio | 8 | 0.335 | 0.403 | 0.489 | 0.439 |

### SemEval — Raw view (full pii_masking_200k vocabulary, 15 labels)

| detector | strict | exact | partial | type |
| --- | --- | --- | --- | --- |
| **skyflow** | **0.637** | **0.700** | **0.778** | **0.763** |
| gliner | 0.456 | 0.558 | 0.671 | 0.651 |
| opf | 0.426 | 0.530 | 0.625 | 0.572 |
| presidio | 0.297 | 0.362 | 0.460 | 0.388 |

Strict-schema error decomposition (raw view):

| detector | COR | INC | MIS | SPU |
| --- | --- | --- | --- | --- |
| skyflow | 1,926 | 665 | **309** | 559 |
| gliner | 1,269 | 916 | 714 | 482 |
| opf | 1,006 | 696 | 1,197 | **127** |
| presidio | 917 | 809 | 1,465 | 1,267 |

Reads cleanly:

- **Skyflow keeps its lead** (+7.0 fair Type F1 over OPF here; 200k's richer vocabulary still favors Skyflow's broader claims). The hierarchy fix shifted Strict +3.7 / Exact +3.2 / Partial +1.1 / Type −1.3 — same pattern as 300k (boundary-aware schemas gain, Type schema loses a sliver of its previous parent-label slack).
- **OPF's strict-view raw F1 (0.426) drops below GLiNER (0.456)** — penalized hard by the 7 categories it doesn't claim. Fair view restores OPF's edge over GLiNER (0.693 vs 0.672 Type).
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
| ACCOUNT | 495 | 0.596 | **0.704** | 0.442 | 0.525 | Skyflow |
| ADDRESS | 429 | 0.663 | **0.805** | 0.782 | 0.035 | Skyflow |
| DATE | 225 | 0.782 | **0.917** | 0.886 | 0.780 | Skyflow |
| DEMOGRAPHIC | 169 | 0.000 | **0.694** | 0.000 | 0.000 | Skyflow |
| EMAIL | 73 | **1.000** | **1.000** | 0.862 | 0.830 | OPF / Skyflow |
| MONEY | 216 | 0.000 | 0.777 | **0.798** | 0.000 | GLiNER |
| OCCUPATION | 189 | 0.000 | 0.444 | **0.556** | 0.000 | GLiNER |
| ORGANIZATION | 58 | 0.000 | **0.477** | 0.162 | 0.000 | Skyflow |
| PERSON | 519 | 0.781 | **0.873** | 0.849 | 0.400 | Skyflow |
| PHONE | 85 | 0.787 | 0.682 | **0.808** | 0.242 | GLiNER |
| PHYSICAL | 48 | 0.000 | 0.824 | **0.843** | 0.000 | GLiNER |
| SECRET | 60 | 0.739 | **0.883** | 0.726 | 0.000 | Skyflow |
| URL | 213 | 0.810 | **0.939** | 0.652 | 0.684 | Skyflow |
| USERNAME | 86 | 0.000 | **0.721** | 0.140 | 0.000 | Skyflow |
| VEHICLE | 34 | 0.000 | **0.727** | 0.655 | 0.000 | Skyflow |

**Wins by category:** Skyflow 11 (one tied with OPF on EMAIL — both perfect), GLiNER 4 (MONEY, OCCUPATION, PHONE, PHYSICAL), OPF 0 outright, Presidio 0.
**OPF zeros**: MONEY, OCCUPATION, ORGANIZATION, PHYSICAL, USERNAME, VEHICLE, DEMOGRAPHIC — the 7 categories outside its supported set. Same picture for Presidio on the categories it doesn't claim.

GLiNER's strength on the new canonicals is real: prompt-driven NER handles `MONEY`, `OCCUPATION`, `PHYSICAL` better than Skyflow's native types. ORGANIZATION is a Skyflow win at 0.477 F1 — a high-recall (0.897) low-precision (0.325) profile that beats GLiNER's 0.162 outright.

### Per-language F1 (fair view, SemEval Type schema, 200k)

| language | n | opf | skyflow | gliner | presidio | winner |
| --- | --- | --- | --- | --- | --- | --- |
| de | 231 | 0.680 | **0.793** | 0.679 | 0.384 | Skyflow |
| en | 228 | 0.666 | **0.777** | 0.669 | 0.541 | Skyflow |
| fr | 309 | 0.696 | **0.732** | 0.650 | 0.455 | Skyflow |
| it | 232 | 0.730 | **0.762** | 0.698 | 0.398 | Skyflow |

200k has 4 languages (no nl/es). Skyflow wins all 4. OPF holds a tight 6-point spread (0.666–0.730).

## Headline takeaways

- **For broad PII coverage with hosted-OK constraints:** Skyflow. Wins overall on both 300k (0.852 Type / 0.778 Strict fair) and 200k (0.763 Type / 0.637 Strict fair); the only detector that handles `DEMOGRAPHIC` + `USERNAME` on 300k and `MONEY` / `OCCUPATION` / `ORGANIZATION` / `VEHICLE` / `PHYSICAL` on 200k. Fast (~107 ms p50). The hierarchy-aware Skyflow request set (drop generic `NAME` / `LOCATION` parents in favor of components) lifted Strict +5.3 / Exact +4.9 on 300k and made Skyflow win all four schemas, even-up with OPF on per-category wins (4 each).
- **For local deployment, the overall winner is still OPF.** 0.837 Type F1 fair view on 300k (within 1.5 of Skyflow), wins ACCOUNT/PHONE/SECRET/URL outright, lowest SPU of any detector either dataset (cleanest precision profile). Slow on CPU (272–703 ms p50 depending on input length) — GPU recommended for production.
- **Best of the 5 new detectors: `gliner_nvidia`.** 0.760 Type F1 fair / 0.746 raw on 300k — beats default GLiNER (0.692) by a clear margin and adds USERNAME coverage OPF lacks. **Doesn't unseat OPF** (still ~8 F1 below) but useful as a complement: drop in if you need USERNAME or want a single GLiNER-family detector that's stronger than the default. Costs ~3× the latency of default GLiNER (179 ms p50 vs 75 ms) due to the 570M-param base. NVIDIA Open Model License (not Apache).
- **For local deployment on broader vocabularies:** default `gliner`. On the harder 200k vocabulary it beats OPF in raw view (0.651 vs 0.572 Type), and wins MONEY, OCCUPATION, PHONE, PHYSICAL on 200k and DATE on 300k. The dataset-aware prompt restriction in this PR also pulls it ahead of `gliner_gretel_*` (which had its own snake_case prompts).
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
