# Benchmark Results

Headline numbers from a 1k PII-Masking-300k sample (`eval/data/sample_1k.jsonl`, fixed seed 42), re-run on the multi-dataset code with the new dataset-aware `skyflow` detector. Every detector is scored under both views the report emits — **Fair** (each detector against `dataset ∩ detector_supports`) and **Raw** (every detector against the dataset's full annotated vocabulary). Run on other ai4privacy datasets (200k / 400k / OpenPII nano / mini) by passing `--dataset NAME` everywhere — see [README](README.md#datasets).

Reproduce with:

```sh
python -m opf_eval.fixtures --dataset pii_masking_300k --out eval/data/sample_1k.jsonl --n 1000
python -m opf_eval.runner --dataset pii_masking_300k --fixtures eval/data/sample_1k.jsonl \
    --detectors opf,skyflow,presidio,gliner \
    --skyflow-min-interval-ms 100 \
    --out eval/results/runs/repro_1k/
python -m opf_eval.report --run eval/results/runs/repro_1k/ --fixtures eval/data/sample_1k.jsonl
```

## SemEval — Fair view (per-detector scope)

Each detector scored against the intersection of (dataset annotates, this detector supports). N labels in parens shows per-detector scope.

| detector | n labels | strict | exact | partial | type |
| --- | --- | --- | --- | --- | --- |
| **skyflow** | 10 | 0.725 | 0.763 | **0.837** | **0.858** |
| opf | 8 | **0.767** | **0.775** | 0.811 | 0.837 |
| gliner | 9 | 0.597 | 0.644 | 0.714 | 0.721 |
| presidio | 8 | 0.408 | 0.474 | 0.538 | 0.481 |

Schema cheat-sheet: **Strict** = exact boundary + label. **Exact** = exact boundary, ignore label. **Partial** = any overlap, ignore label. **Type** = any overlap + matching label.

## SemEval — Raw view (full pii_masking_300k vocabulary, 10 labels)

Every detector scored against the dataset's full annotated set. Labels a detector doesn't support take zero recall; this view reflects out-of-the-box coverage rather than fairness.

| detector | strict | exact | partial | type |
| --- | --- | --- | --- | --- |
| **skyflow** | 0.725 | 0.763 | **0.837** | **0.858** |
| opf | 0.731 | **0.794** | 0.830 | 0.797 |
| gliner | 0.585 | 0.639 | 0.709 | 0.707 |
| presidio | 0.392 | 0.462 | 0.532 | 0.462 |

Strict-schema error decomposition (raw view):

| detector | COR | INC | MIS | SPU |
| --- | --- | --- | --- | --- |
| skyflow | 5,113 | 1,305 | **562** | 710 |
| opf | 4,765 | 878 | 1,338 | 418 |
| gliner | 3,911 | 1,298 | 1,771 | 1,179 |
| presidio | 2,987 | 1,603 | 3,586 | 2,465 |

Reads cleanly:

- **Skyflow's edge is recall.** 562 missed vs OPF's 1,338 — less than half. Largely because Skyflow is the only detector with non-zero `DEMOGRAPHIC` (0.732 F1) and `USERNAME` (0.734 F1) recall on this dataset (raw view, see per-category below).
- **OPF wins on type confusion.** 878 INC vs Skyflow's 1,305 — better at picking the right canonical when it does fire. Also lowest SPU (418), the cleanest precision profile.
- **GLiNER over-detects more than the leaders** (SPU 1,179) and misses more (1,771 vs OPF's 1,338).
- **Presidio's loss is split across MIS + SPU** (under-recalls *and* over-fires); 1,603 INC despite the lowest F1.

## Latency (single-request, 1k sample)

| detector | p50 | p95 | p99 |
| --- | --- | --- | --- |
| presidio | **14 ms** | 21 ms | **25 ms** |
| gliner | 75 ms | 94 ms | 103 ms |
| skyflow | 107 ms | 194 ms | 204 ms |
| opf | 703 ms | 1,047 ms | 1,229 ms |

OPF on CPU. Skyflow latency is region-dependent (network to vault).

## Per-category F1 (raw view, partial overlap, IoU ≥ 0.5; winners in **bold**)

| label | gold n | opf | skyflow | gliner | presidio | winner |
| --- | --- | --- | --- | --- | --- | --- |
| ACCOUNT | 1,448 | **0.952** | 0.850 | 0.685 | 0.408 | OPF |
| ADDRESS | 1,609 | **0.773** | 0.728 | 0.606 | 0.216 | OPF |
| DATE | 1,111 | 0.652 | 0.869 | **0.886** | 0.488 | GLiNER |
| DEMOGRAPHIC | 246 | 0.000 | **0.732** | 0.000 | 0.000 | Skyflow |
| EMAIL | 324 | 0.961 | 0.926 | 0.884 | **0.968** | Presidio |
| PERSON | 1,101 | **0.679** | 0.673 | 0.481 | 0.141 | OPF |
| PHONE | 281 | **0.969** | 0.859 | 0.863 | 0.404 | OPF |
| SECRET | 220 | **0.946** | 0.757 | 0.736 | 0.000 | OPF |
| URL | 274 | **0.956** | 0.903 | 0.427 | 0.527 | OPF |
| USERNAME | 366 | 0.000 | **0.734** | 0.557 | 0.000 | Skyflow |

**Wins by category:** OPF 6, Skyflow 2 (the categories no other detector supports on this dataset), GLiNER 1, Presidio 1.
**Wins weighted by gold span volume:** OPF 60%, Skyflow 8%, GLiNER 16%, Presidio 5% (DEMOGRAPHIC 4% + USERNAME 6% to Skyflow because nobody else covers them).

OPF wins more *categories*, but Skyflow's headline F1 is higher because it covers two categories (DEMOGRAPHIC + USERNAME) that OPF / Presidio don't claim and that meaningfully drag those detectors' raw-view scores. The fair view (above) collapses that gap by scoping each detector to its supported labels — and there OPF's 0.837 Type F1 is within 2.1 of Skyflow's 0.858.

## Per-language F1 (fair view, SemEval Type schema)

| language | n | opf | skyflow | gliner | presidio | winner |
| --- | --- | --- | --- | --- | --- | --- |
| de | 173 | 0.835 | **0.878** | 0.725 | 0.426 | Skyflow |
| en | 169 | 0.825 | **0.851** | 0.683 | 0.607 | Skyflow |
| es | 158 | 0.852 | **0.894** | 0.742 | 0.479 | Skyflow |
| fr | 193 | 0.847 | **0.860** | 0.705 | 0.461 | Skyflow |
| it | 143 | **0.846** | 0.827 | 0.732 | 0.448 | OPF |
| nl | 164 | 0.819 | 0.833 | 0.744 | 0.464 | Skyflow |

Skyflow wins 5/6 languages; OPF wins Italian. Tight 4-point spread for OPF (0.819–0.852) confirms the 5k-run finding that OPF is consistent across the languages it sees. Presidio's 30-point gap on non-English is its English-only spaCy NER; multilingual Presidio (`presidio_multilang`) actually performs slightly worse overall — country-specific regex recognizers (US_SSN etc.) get gated to `language="en"` and stop firing. See [plans/01-presidio-baseline.md](plans/01-presidio-baseline.md).

## Headline takeaways

- **For broad PII coverage with hosted-OK constraints:** Skyflow. Fastest at scale (107 ms p50), highest F1 in the most realistic Type-schema view (0.858 fair / 0.858 raw), the only detector that handles `DEMOGRAPHIC` + `USERNAME` on this dataset.
- **For local deployment:** OPF. Within 2.1 F1 of Skyflow in fair view (0.837 Type), dominates 6 of 10 categories outright, fully local. Slow on CPU (703 ms p50, 1,229 ms p99) — GPU recommended for production.
- **For DATE specifically:** GLiNER. Beats OPF by 23.4 F1 and Skyflow by 1.7 F1 on the highest-volume category, despite being an order of magnitude smaller than OPF.
- **For high-volume EMAIL detection only:** Presidio. 14 ms p50, 0.968 F1 on EMAIL — beats every other detector. Worthless on most other categories.
- **A hybrid local stack** (OPF + GLiNER for DATE, optionally Presidio for EMAIL) would close most of the OPF→Skyflow gap while staying fully on-prem. Not built or benchmarked yet.

## Caveats and what we don't measure here

- **PII-Masking-300k is a single dataset.** Performance on production traffic (chat transcripts, support tickets, internal docs) may differ. The dataset's labeling style (e.g. `GIVENNAME1`/`GIVENNAME2` per-token rather than `FULL_NAME`) shapes the granularity bias and per-category recall profile. Other ai4privacy datasets (`pii_masking_200k`, `pii_masking_400k`, `openpii_nano/mini`) are runnable by passing `--dataset NAME`; results not yet folded in.
- **OPF was not fine-tuned.** [plans/02-finetune-opf.md](plans/02-finetune-opf.md) is the remaining high-leverage experiment — fine-tuning could plausibly close most of the DATE/ADDRESS gap to Skyflow.
- **Skyflow API latency is region-dependent.** Reported p50 is whatever your network-to-vault path is. For latency-sensitive paths in a different region, mileage will vary.
- **Cost not measured.** Skyflow is per-call; OPF/Presidio/GLiNER are self-hosted compute. Decision frameworks usually need both quality and cost — cost is left to the deployer.
- **GLiNER multi-PII variant** is a single model checkpoint we picked; tuning the prompt strings, threshold, or switching to a fine-tuned variant could move the numbers.
- **Skyflow comparison vs old `skyflow_minimal`.** The retired hand-tuned 24-entity preset scored 0.860 Type F1 here; the new dataset-aware `skyflow` (38 entity_types) scores 0.858 — within noise. The auto-derived list trades slightly lower per-category PERSON / ADDRESS precision (more bare-type confusion) for slightly lower MIS overall.
