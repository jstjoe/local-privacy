# Benchmark Results

Headline numbers from the 1k PII-Masking-300k sample (`eval/data/sample_1k.jsonl`, fixed seed 42), restricted to OPF's 8 supported categories. The 5k run confirmed these numbers for OPF / GLiNER / Presidio within ±1 F1; Skyflow_minimal at 5k is pending.

Reproduce with:

```sh
python -m opf_eval.fixtures --out eval/data/sample_1k.jsonl --n 1000
python -m opf_eval.runner --fixtures eval/data/sample_1k.jsonl \
    --detectors opf,skyflow_minimal,presidio,gliner \
    --out eval/results/runs/repro_1k/
python -m opf_eval.report --run eval/results/runs/repro_1k/ --fixtures eval/data/sample_1k.jsonl
```

## Headline (restricted partial scoring, IoU ≥ 0.5)

| detector | precision | recall | F1 | latency p50 | latency p99 |
| --- | --- | --- | --- | --- | --- |
| **skyflow_minimal** | 0.821 | **0.850** | **0.835** | 105 ms | 318 ms |
| **opf** | **0.844** | 0.784 | 0.813 | 515 ms | 922 ms |
| **gliner** | 0.705 | 0.638 | 0.670 | 72 ms | 102 ms |
| **presidio** | 0.351 | 0.380 | 0.365 | **11 ms** | **20 ms** |

Granularity-neutral view (same scoring after merging adjacent same-canonical spans on both predictions and gold):

| detector | F1 |
| --- | --- |
| skyflow_minimal | **0.851** |
| opf | 0.825 |
| gliner | 0.696 |
| presidio | 0.385 |

## Per-category F1 (winners in **bold**)

| label | gold n | opf | skyflow_minimal | gliner | presidio | winner |
| --- | --- | --- | --- | --- | --- | --- |
| ACCOUNT | 1,448 | **0.952** | 0.843 | 0.685 | 0.408 | OPF |
| ADDRESS | 1,609 | 0.773 | **0.847** | 0.606 | 0.242 | Skyflow |
| DATE | 1,111 | 0.652 | 0.869 | **0.886** | 0.488 | GLiNER |
| EMAIL | 324 | 0.961 | 0.909 | 0.884 | **0.968** | Presidio |
| PERSON | 1,098 | 0.678 | **0.734** | 0.479 | 0.141 | Skyflow |
| PHONE | 281 | **0.969** | 0.860 | 0.863 | 0.404 | OPF |
| SECRET | 220 | **0.946** | 0.758 | 0.736 | 0.000 | OPF |
| URL | 274 | **0.956** | 0.898 | 0.427 | 0.527 | OPF |

**Wins by category:** OPF 4, Skyflow_minimal 2, GLiNER 1, Presidio 1.
**Wins weighted by gold span volume:** OPF 32%, Skyflow 51%, GLiNER 17%, Presidio 5%.

OPF wins more *categories*, but Skyflow_minimal wins the high-volume ones (ADDRESS + PERSON = 2,707 spans). DATE alone (1,111 spans, GLiNER's only win) is bigger than EMAIL + PHONE + SECRET + URL combined.

## Per-language F1 (restricted, partial)

| language | n | opf | skyflow_minimal | gliner | presidio | winner |
| --- | --- | --- | --- | --- | --- | --- |
| Dutch | 164 | 0.789 | **0.817** | 0.681 | 0.316 | Skyflow |
| English | 169 | 0.791 | **0.837** | 0.629 | 0.432 | Skyflow |
| French | 193 | 0.829 | **0.846** | 0.661 | 0.378 | Skyflow |
| German | 173 | 0.814 | **0.842** | 0.693 | 0.324 | Skyflow |
| Italian | 143 | **0.810** | 0.787 | 0.645 | 0.366 | OPF |
| Spanish | 158 | 0.844 | **0.875** | 0.707 | 0.383 | Skyflow |

Skyflow_minimal wins 5 of 6 languages. The 5k confirmation at language level: OPF wins **all 6** languages when Skyflow isn't in the comparison, with a tight 4-point spread (0.786–0.830). The Italian-OPF result at 1k was real.

Presidio's 30-point gap on non-English languages is its English-only spaCy NER. Multilingual Presidio (`presidio_multilang`) actually performs slightly worse overall — country-specific regex recognizers (US_SSN etc.) get gated to `language="en"` and stop firing. See [plans/01-presidio-baseline.md](plans/01-presidio-baseline.md).

## Headline takeaways

- **For broad PII coverage with hosted-OK constraints:** Skyflow_minimal. Fastest at scale (105 ms p50), highest F1 in granularity-neutral terms (0.851), best on the high-volume PERSON and ADDRESS categories.
- **For local deployment:** OPF. Within 2.6 F1 of Skyflow_minimal in the most realistic view, dominates 4 of 8 categories, fully local. Slow on CPU (515 ms p50, 922 ms p99) — GPU recommended for production.
- **For DATE specifically:** GLiNER. Beats OPF by 22.7 F1 and Skyflow by 1.7 F1, despite being an order of magnitude smaller than OPF. The greedy-span pathology in OPF that we couldn't fix with Viterbi calibration just doesn't exist in GLiNER.
- **For high-volume EMAIL detection only:** Presidio. 11 ms p50, 0.968 F1 on EMAIL — beats every other detector. Worthless on most other categories.
- **A hybrid local stack** (OPF + GLiNER for DATE, optionally Presidio for EMAIL) would close most of the OPF→Skyflow gap while staying fully on-prem. Not built or benchmarked yet.

## Caveats and what we don't measure here

- **PII-Masking-300k is a single dataset.** Performance on production traffic (chat transcripts, support tickets, internal docs) may differ. The dataset's labeling style (e.g. `GIVENNAME1`/`GIVENNAME2` per-token rather than `FULL_NAME`) shapes the granularity bias and per-category recall profile.
- **OPF was not fine-tuned.** [plans/02-finetune-opf.md](plans/02-finetune-opf.md) is the remaining high-leverage experiment — fine-tuning could plausibly close most of the DATE/ADDRESS gap to Skyflow.
- **Skyflow API latency is region-dependent.** Reported p50 is whatever your network-to-vault path is. For latency-sensitive paths in a different region, mileage will vary.
- **Cost not measured.** Skyflow is per-call; OPF/Presidio/GLiNER are self-hosted compute. Decision frameworks usually need both quality and cost — cost is left to the deployer.
- **GLiNER multi-PII variant** is a single model checkpoint we picked; tuning the prompt strings, threshold, or switching to a fine-tuned variant could move the numbers.
