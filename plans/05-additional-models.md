# Plan 05 — Add four more PII-focused open-weight models

## Why

The current open-weight tier has just two distinct architectures (OPF and GLiNER) with one checkpoint each. Adding more PII-trained variants tells us how much of OPF/GLiNER's quality is architectural vs. training-data, and gives users more options when they want to deploy locally without committing to any single vendor's weights.

Models to add:

| name | repo | what it is |
| --- | --- | --- |
| Gretel small | `gretelai/gretel-gliner-bi-small-v1.0` | GLiNER variant fine-tuned by Gretel for PII |
| Gretel large | `gretelai/gretel-gliner-bi-large-v1.0` | larger Gretel variant |
| Nvidia PII | `nvidia/gliner-PII` | Nvidia's PII-tuned GLiNER |
| OpenMed PF multilingual | `OpenMed/privacy-filter-multilingual` | OPF variant fine-tuned for multilingual PII (presumably medical-tuned given the org) |

## Scope

- Three new GLiNER detector names + one new OPF detector name
- Reuse existing `GLiNERDetector` and `OPFDetector` classes — these already accept `model_name`/`model` constructor args
- Optional per-variant prompt overrides if recommended labels differ from our current set
- No changes to taxonomy, fixtures, or report logic

## Files

- **Modify** `eval/src/opf_eval/runner.py` — register four new names in `_build_detector`:
  - `gliner_gretel_small`
  - `gliner_gretel_large`
  - `gliner_nvidia`
  - `opf_openmed_multilingual`
- **Modify** `eval/src/opf_eval/detectors/gliner.py` — possibly accept per-variant prompt overrides if model cards recommend specific label phrasings
- **Modify** `eval/src/opf_eval/detectors/opf.py` — confirm `OPFDetector(model="OpenMed/...")` triggers HF Hub download (OPF uses `huggingface_hub` already, but its `ensure_default_checkpoint` flow may only handle the default repo)
- **Optionally modify** `eval/src/opf_eval/taxonomy.py` — if a variant has a meaningfully different label vocabulary, add a per-variant column

## Implementation notes

### GLiNER variants (Gretel small, Gretel large, Nvidia PII)

Should be ~5 lines each:

```python
if name == "gliner_gretel_small":
    return GLiNERDetector(model_name="gretelai/gretel-gliner-bi-small-v1.0")
if name == "gliner_gretel_large":
    return GLiNERDetector(model_name="gretelai/gretel-gliner-bi-large-v1.0")
if name == "gliner_nvidia":
    return GLiNERDetector(model_name="nvidia/gliner-PII")
```

**Open question per model:** check each model card for the exact label strings the model was fine-tuned on. The current prompts in `taxonomy.gliner_prompts()` were chosen for `urchade/gliner_multi_pii-v1`. Variants tuned on a different label set (e.g. Gretel may use `"first name"` and `"last name"` instead of `"person"`; Nvidia may use specific PII-X labels) will benefit from variant-specific prompts. Quickest path: try our default prompts first, then check if F1 looks low and revisit.

If variant-specific prompts are needed, extend `GLiNERDetector` to accept an optional `prompts: list[str]` arg, and add per-detector prompt lists in `taxonomy.py` (e.g. `GLINER_GRETEL_PROMPTS`).

### OpenMed OPF variant

Two unknowns to resolve before this is a one-liner:

1. **Checkpoint format compatibility.** OPF's `OPF(model=...)` accepts a local directory path containing `config.json`, `model.safetensors`, optional `viterbi_calibration.json`. Verify the OpenMed repo follows this format. If yes, point and shoot. If the repo is just safetensors + a HF-style `config.json`, may need conversion or a custom loader.
2. **HF Hub download path.** OPF uses `huggingface_hub` for the default checkpoint, but `resolve_checkpoint_path` may only call `ensure_default_checkpoint()` for the canonical repo. Likely needs:

   ```python
   from huggingface_hub import snapshot_download
   ckpt_dir = snapshot_download("OpenMed/privacy-filter-multilingual")
   detector = OPFDetector(model=ckpt_dir)
   ```

   Wrap that in the runner registration.

3. **Label space.** OPF was trained on its 8 categories (`private_person`, `private_email`, etc.). If OpenMed fine-tuned with the same label space, no taxonomy work needed. If they redefined the label space (e.g. for medical PII categories), the variant needs its own taxonomy column.

## Risks / open questions

- **Model card discrepancies.** Each variant likely documents specific label strings or invocation conventions. Skipping the model card and assuming our defaults work risks bad numbers that don't represent the model's true capability.
- **License compatibility.** Verify each model's license permits benchmark redistribution (Apache 2.0 is fine; some research models have non-commercial clauses).
- **Disk pressure.** Each GLiNER variant is ~200–500 MB, OpenMed OPF likely ~2–3 GB. First-run downloads add up. Document expected disk usage in the model registration comments.
- **Latency comparison.** GLiNER large will be slower than small. Worth noting in the report — especially for the recommendation matrix.

## Verification

```sh
# 1. Smoke test each variant individually on 100 examples
for variant in gliner_gretel_small gliner_gretel_large gliner_nvidia opf_openmed_multilingual; do
  uv run python -m opf_eval.runner \
      --fixtures eval/data/sample_100.jsonl \
      --detectors $variant \
      --reuse-from eval/results/runs/run_100/ \
      --out eval/results/runs/run_100_$variant/
done

# 2. Full 1k bench against the existing competitors (reuse-from is your friend)
uv run python -m opf_eval.runner \
    --fixtures eval/data/sample_1k.jsonl \
    --detectors gliner_gretel_small,gliner_gretel_large,gliner_nvidia,opf_openmed_multilingual \
    --reuse-from eval/results/runs/run_1k_with_gliner/ \
    --out eval/results/runs/run_1k_with_variants/

uv run python -m opf_eval.report \
    --run eval/results/runs/run_1k_with_variants/ \
    --fixtures eval/data/sample_1k.jsonl
```

What I'd watch for in the report:

- **Gretel small vs Gretel large** — model-size scaling within the same training recipe. Tells us whether bigger GLiNER is worth the latency/memory cost.
- **Gretel vs Nvidia vs urchade** — how much PII-focused training data shifts the per-category profile. Possible that one variant dominates DATE while another wins ADDRESS.
- **OpenMed multilingual OPF vs vanilla OPF** — does the multilingual fine-tune retain English quality while improving non-English? Per-language slicing is the right diagnostic.
- **Latency** — Gretel large and OpenMed will be the slowest. Confirm they're still useful at p99.

## Effort

- **GLiNER variants:** ~30 min each if our default prompts work, ~2 hours each if per-variant prompts need tuning. Total ~2-6 hours.
- **OpenMed OPF variant:** ~3-4 hours including the HF Hub download wrinkle and any taxonomy changes. Could be longer if checkpoint format requires conversion.
- **Total:** half a day to a full day of work + 1-2 benchmark runs.

## Out of scope

- Fine-tuning these variants ourselves (covered separately in [plan 02](02-finetune-opf.md))
- Custom-prompt grid search per GLiNER variant (could merit a separate plan if we want to extract every last F1 point)
- Comparing different GLiNER architectures (uni-encoder vs bi-encoder) — we just take what each variant ships
- Quantized variants of any of these (could be relevant for latency-bound deployments but separate concern)
