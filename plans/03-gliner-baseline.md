# Plan 03 — Add GLiNER as a second open-weight detector

## Why

GLiNER is the closest peer to OPF: small (~200MB), open-weight (Apache 2.0), runs on CPU at modest speed, fine-tunable. Same deployment story as OPF. Adding it lets us answer "is OPF the right open-weight choice, or would GLiNER do as well or better" with the same harness.

You have prior experience with GLiNER (per the original plan doc). That makes this faster than starting from scratch on a new model.

## Scope

- New `GLiNERDetector` integrated into the existing harness
- Pick GLiNER variant (multi-lang vs English-only depending on dataset target)
- Map GLiNER's prompted entity types to canonical labels

## Files

- **Add** `eval/src/opf_eval/detectors/gliner.py` — `GLiNERDetector` class
- **Modify** `eval/src/opf_eval/detectors/__init__.py` — export
- **Modify** `eval/src/opf_eval/runner.py` — register `gliner` name
- **Modify** `eval/src/opf_eval/taxonomy.py` — add `gliner` column. GLiNER labels are user-defined prompts, so the canonical map decides what we ask for.
- **Modify** `eval/pyproject.toml` — add `gliner`

## Implementation notes

- Use `gliner-multi-v2.1` or `gliner-multi-pii-v1` to handle multilingual fixtures
- Pass our 8 OPF canonicals as English-language entity prompts: e.g. `["person name", "email address", "phone number", "physical address", "url", "date", "account number", "password"]`
- GLiNER returns `[{"start": int, "end": int, "label": "person name", "score": float}]` — convert to our Span dict
- Set a confidence threshold (default 0.5, tune empirically)

## Risks / open questions

- **Prompt sensitivity.** GLiNER's quality depends heavily on entity prompt phrasing. `"date"` may underperform `"date or time"` etc. Worth a small grid search.
- **Multi-lang model size.** Multi-lang variants are larger and slower. Verify CPU latency vs OPF.
- **Entity vocabulary.** GLiNER can request multiple synonyms per entity. May want to request `["person", "name", "first name", "last name"]` and merge canonicalize after.

## Verification

```sh
uv sync
uv run python -m opf_eval.runner \
    --fixtures eval/data/sample_100.jsonl \
    --detectors opf,gliner,skyflow_minimal \
    --out eval/results/runs/run_100_gliner/
uv run python -m opf_eval.report --run eval/results/runs/run_100_gliner/ --fixtures eval/data/sample_100.jsonl
```

What I'd watch:
- Per-category profile vs OPF — does GLiNER win the categories OPF struggles with (DATE, ADDRESS)?
- Latency — GLiNER on CPU is typically 100-300ms per example, faster than OPF
- Per-language F1 — multi-lang models can be uneven

## Effort

~3-4 hours including prompt experimentation.

## Out of scope

- Fine-tuning GLiNER (separate plan if it scores well baseline)
- Building custom GLiNER prompts beyond the obvious 8 categories
- Comparing GLiNER variants exhaustively (pick one good one)
