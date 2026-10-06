# Plans

Side-quest specs for extending the OPF vs Skyflow benchmark. Each plan is brief
(roughly one Pomodoro to read), self-contained, and implementable independently.

| # | plan | effort | unblocks |
| --- | --- | --- | --- |
| 01 | [Presidio baseline](01-presidio-baseline.md) | ~2h | "is the ML detector worth it vs free regex+NER?" |
| 02 | [Fine-tune OPF](02-finetune-opf.md) | ~1 day (cloud GPU) | "can OPF beat Skyflow if trained on this data?" |
| 03 | [GLiNER baseline](03-gliner-baseline.md) | ~3-4h | "is OPF the right open-weight choice or would GLiNER do as well?" |
| 04 | [LLM-as-detector via LM Studio](04-llm-detector-lmstudio.md) | ~3-4h | "what's the quality ceiling of a generative model on this task?" |
| 05 | [Additional PII-focused models](05-additional-models.md) | ~6-7h | "does any other open-weight PII model (Gretel ×2, Nvidia, ai4privacy ModernBERT, OpenMed) beat what we have?" |
| 06 | [Unified privacy-detection API](06-unified-api.md) | ~1–1.5 days | "one client contract, switch backend with a flag" |
| 07 | [Cloud Run hardening](07-cloud-run-hardening.md) | ~2 days | "auth + limits + observability + CI to make the API deployable to GCP" |
| 08 | [SemEval scoring via nervaluate](08-nervaluate-scoring.md) | ~3-4h | "where exactly is each detector losing points — missed, mis-typed, spurious?" |
| 09 | [Multi-dataset fixtures + per-detector scoring](09-multi-dataset-fixtures.md) | ~1-1.5 days | "general benchmark — pick (detector, dataset, size); detectors auto-configure per dataset; score against own claims + dataset as a whole" |
| 10 | [Redaction + tokenization demo](10-redaction-tokenization-demo.md) | shipped | "what does sanitized text look like under each mode?" |
| 11 | [Use: search over sanitized text](11-use-section.md) | shipped | "which sanitization modes keep search working?" |
| 12 | [Roadmap: benchmarks, datasets, models, notebooks](12-roadmap.md) | multi-PR | "public benchmarks (PrivacyBench, Nemotron-PII, TAB…), newer open-weight detectors, cross-benchmark report" |
| 13 | [Evaluate on your own data without gold labels](13-unlabeled-data-notebook.md) | ~4.5–5.5 days, 4 PRs | "how good is each detector on my files, with LLM silver labels or LLM review or Clef-flash review?" |

## Suggested order

1. **#01 Presidio** first — cheapest, possibly reframes the whole comparison
2. **#02 Fine-tune OPF** — most informative, biggest "if it works, change the recommendation" upside
3. **#03 GLiNER** if interested in a third open-weight peer
4. **#04 LLM detector** for research signal, not deployment

## Conventions

- New detectors register in `opf_eval.detectors.registry` (see [`builtin.py`](../eval/src/opf_eval/detectors/builtin.py)); `register_gliner_model` / `register_hf_token_classifier` cover the common shapes
- New label vocabularies (detector or dataset) go in [`taxonomy/vocabs.py`](../eval/src/opf_eval/taxonomy/vocabs.py), or are registered at runtime with `taxonomy.register_vocab`
- New datasets register with `opf_eval.datasets.register_dataset` (local file or HF id; `spans`, `privacy_mask` or `bio` schema)
- Reports auto-pick up new detectors from the manifest

Plans 01–09 predate the detector registry and two-level taxonomy; their "add a column to `CANONICAL_MAP`" / "`runner._build_detector()`" steps map onto the conventions above.
