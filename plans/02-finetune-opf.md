# Plan 02 — Fine-tune OPF on PII-Masking-300k

## Why

The 1k benchmark showed OPF loses 21.7 F1 on DATE and 7.4 F1 on ADDRESS to Skyflow_minimal. Both gaps trace to OPF's pre-training labeling `"2040-06-02 00:00:00"` as one DATE span where PII-Masking-300k labels it as two. Fine-tuning teaches OPF to split spans the way the dataset wants. If it works, OPF probably leapfrogs Skyflow on this dataset — which would be the central case for adopting OPF over hosted PII detection.

This is the single most informative experiment we haven't run.

## Scope

- Convert PII-Masking-300k subset to OPF training format
- Train/val/test split with no leakage
- Run `opf train` with default 8-category label space (no custom ontology)
- Add `opf_finetuned` detector that loads the new checkpoint
- Compare against vanilla OPF + skyflow_minimal on a held-out test set

## Files

- **Add** `eval/src/opf_eval/finetune/prepare_data.py` — converts PII-Masking-300k to OPF training JSONL
- **Add** `eval/src/opf_eval/finetune/run_train.sh` — wraps `opf train` with our hyperparams
- **Add** `eval/src/opf_eval/finetune/README.md` — repro instructions
- **Modify** `eval/src/opf_eval/detectors/opf.py` — already accepts `model=` path, just confirm the runner can pass it
- **Modify** `eval/src/opf_eval/runner.py` — register `opf_finetuned`, take `--opf-checkpoint` arg
- **Add** `eval/data/finetuning/{train,val,test}.jsonl` — gitignored

## Data preparation

OPF training format (per [privacy-filter/FINETUNING.md](../privacy-filter/FINETUNING.md) and demo at [examples/data/finetuning_secret_demo/train.jsonl](../privacy-filter/examples/data/finetuning_secret_demo/train.jsonl)):

```json
{"text": "...", "label": [{"category": "private_email", "start": 12, "end": 27}, ...]}
```

Mapping from PII-Masking-300k canonical → OPF category (already exists in `taxonomy.py`):
- PERSON → `private_person`
- EMAIL → `private_email`
- PHONE → `private_phone`
- ADDRESS → `private_address`
- URL → `private_url`
- DATE → `private_date`
- ACCOUNT → `account_number`
- SECRET → `secret`
- USERNAME, DEMOGRAPHIC → drop (no native OPF category)

## Splits — critical to avoid leakage

- **train**: 20k examples sampled from PII-Masking-300k `train` split, fixed seed
- **val** (training-time validation): 1k from `train` split, disjoint from train
- **test** (post-training eval): the existing `sample_1k.jsonl` we already use **regenerated from PII-Masking-300k `validation` split** (currently it's sampled from `train`)

The test set must use PII-Masking-300k's `validation` split that the fine-tuned checkpoint has never seen. Update `fixtures.py` to accept `--split validation`. Re-materialize a `sample_1k_holdout.jsonl` for the post-training comparison.

## Training

```sh
opf train eval/data/finetuning/train.jsonl \
  --validation-dataset eval/data/finetuning/val.jsonl \
  --output-dir eval/checkpoints/opf_pii300k_v1/
```

Hyperparams to start (mirror demo, lower epoch count for larger dataset):
- `--epochs 3` (demo uses 40 for tiny toy data; 3-5 should be enough at 20k)
- `--batch-size 8` (GPU-permitting)
- `--learning-rate 2e-5` (lower than demo's 2e-4 since dataset is larger)

## Hardware

- **CPU on Mac**: 24h+. Don't.
- **A100 on Lambda/Modal/Vast**: ~1-2h, ~$3-5 total. Right answer.
- **Apple Silicon ≥32GB unified memory**: 6-12h, free. Doable if patient.

## Verification

```sh
# 1. Re-materialize the held-out test set from validation split
uv run python -m opf_eval.fixtures --out eval/data/sample_1k_holdout.jsonl --n 1000 --split validation

# 2. Bench the new checkpoint against the existing baseline
uv run python -m opf_eval.runner \
    --fixtures eval/data/sample_1k_holdout.jsonl \
    --detectors opf,opf_finetuned,skyflow_minimal \
    --opf-checkpoint eval/checkpoints/opf_pii300k_v1/ \
    --out eval/results/runs/run_1k_finetuned/

uv run python -m opf_eval.report --run eval/results/runs/run_1k_finetuned/ --fixtures eval/data/sample_1k_holdout.jsonl
```

Watch for:
- DATE F1: did the gap to Skyflow close? Hypothesis: 0.652 → 0.85+
- ADDRESS F1: same hypothesis, smaller magnitude
- Other categories: hopefully stable, possible regression on EMAIL/PHONE/URL/SECRET if fine-tuning over-corrects

## Risks / open questions

- **Checkpoint size + load time.** Fine-tuned checkpoint will be ~2.8GB. `OPFDetector(model=path)` should already handle this (the API accepts arbitrary model paths). Verify.
- **Triton fallback.** `OPF_MOE_TRITON=0` still required on Apple Silicon for inference; check if training requires the same.
- **Catastrophic forgetting.** Fine-tuning on the dataset's specific patterns could degrade categories that were fine before. Hold our breath on EMAIL/PHONE/URL.
- **Dataset license.** PII-Masking-300k's license terms — confirm fine-tuning + redistributing the resulting checkpoint is OK if we plan to share it.
- **GPU access.** Need a plan: pick a cloud provider, set up an account, confirm pricing.

## Effort

~1 day end-to-end if cloud GPU is sorted ahead of time. Most of the work is data conversion + clean splits — `opf train` itself is one command.

## Out of scope

- Custom label space (sticking with the default 8 categories)
- Multi-stage training (e.g. continued pre-training)
- Quantization / distillation of the result
- Cross-language transfer experiments
