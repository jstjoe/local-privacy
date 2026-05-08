# Plan 08 — SemEval-standard scoring via `nervaluate`

## Why

Today's metrics (`eval/src/opf_eval/metrics.py`) collapse all wrong predictions into FP and all missed gold into FN — a single greedy 1:1 match per (text, canonical) pair. That tells us a detector's F1, but not *why* it's losing points: is it missing entities, mis-classifying types, or returning spurious extras? When [Presidio scores 0.365 F1](../RESULTS.md), we have no diagnostic answer.

[`nervaluate`](https://github.com/MantisAI/nervaluate) implements the [SemEval 2013 Task 9.1](https://davidsbatista.net/assets/documents/others/semeval_2013-task-9_1-evaluation-metrics.pdf) NER evaluation scheme. It decomposes errors into 5 categories (COR, INC, PAR, MIS, SPU) across 4 schemas (Strict / Exact / Partial / Type), giving a principled error-attribution view. The "Type" schema also gives a cleaner answer to OPF's greedy-span problem (`"2040-06-02 00:00:00"` as one DATE) than our current "merged-adjacent" hack.

Goal: **replace the merged-adjacent granularity-neutral view with SemEval scoring**. The Type schema is the principled equivalent — same intent (don't over-penalize boundary granularity), but a standard rather than our hack. Keep the existing greedy-1:1 "restricted partial" headline (it's what stakeholders already read), keep latency + per-language + disagreement buckets (nervaluate doesn't address those). Drop the merged-adjacent table and replace with a SemEval section.

## Scope

In:

- New metrics module that runs nervaluate per detector/fixture set
- New report section: 4-schema headline + per-error-type breakdown per detector
- Reuses existing canonical-label mapping and restricted-to-OPF-8 filter
- **Remove** the merged-adjacent granularity-neutral view from `report.py` and `RESULTS.md` (Type schema replaces it)

Out:

- Removing the existing greedy-1:1 "restricted partial" headline (keep — stakeholders read it)
- Latency, per-language, and disagreement buckets (nervaluate doesn't address these)
- Token-level / BIO-formatted scoring (we use char offsets, prodi.gy span loader fits us)
- Tweaking nervaluate's matching algorithm (out-of-the-box)

## Files

- **Add** [eval/src/opf_eval/nervaluate_metrics.py](../eval/src/opf_eval/nervaluate_metrics.py) — wraps nervaluate, returns a structured per-detector result
- **Modify** [eval/src/opf_eval/report.py](../eval/src/opf_eval/report.py) — add "SemEval scoring" section after the existing headline tables
- **Modify** [eval/pyproject.toml](../eval/pyproject.toml) — add `nervaluate>=0.2`
- **Modify** [RESULTS.md](../RESULTS.md) — add the new headline table once a 1k run produces it
- **Modify** [README.md](../README.md) — link to the SemEval section in `report.md`

## Implementation notes

### Input format

nervaluate accepts prodi.gy-style spans directly:

```python
true = [
    [{"label": "EMAIL", "start": 7, "end": 22}, {"label": "ADDRESS", "start": 32, "end": 41}],
    # ... one inner list per fixture
]
pred = [...]   # same shape, parallel index
```

Our existing pipeline already produces `{label, start, end, ...}` dicts in canonical form for both gold (via `taxonomy.pii300k_to_canonical`) and predictions (via each detector's canonical mapper). The conversion is a flat re-shape:

```python
def to_nervaluate(rows: list[dict], pred_rows_by_id: dict[str, list[Span]]):
    true_per_doc = [
        [{"label": s["label"], "start": s["start"], "end": s["end"]} for s in r["gold_spans"]]
        for r in rows
    ]
    pred_per_doc = [
        [{"label": s["label"], "start": s["start"], "end": s["end"]} for s in pred_rows_by_id[r["id"]]]
        for r in rows
    ]
    return true_per_doc, pred_per_doc
```

### Filter to canonical labels

The `tags=` argument scopes evaluation to a list of labels — anything outside is treated as non-entity. Use the same restricted set as today's "restricted partial" view:

```python
from opf_eval.taxonomy import OPF_CANONICAL_LABELS  # 8 labels OPF natively supports

evaluator = Evaluator(
    true=true_per_doc,
    pred=pred_per_doc,
    tags=list(OPF_CANONICAL_LABELS),
    loader="default",   # accepts the {label,start,end} dict shape directly
)
results, results_per_tag, *_ = evaluator.evaluate()
```

### What we add to the report

A third headline table (after the current "Restricted partial" and "Granularity-neutral" tables):

```markdown
## SemEval (nervaluate)

### Headline (per schema)

| detector       | strict F1 | exact F1 | partial F1 | type F1 |
| -------------- | --------- | -------- | ---------- | ------- |
| skyflow_minimal|   0.74    |   0.78   |    0.85    |  0.88   |
| opf            |   0.71    |   0.76   |    0.81    |  0.89   |
| gliner         |   0.55    |   0.61   |    0.69    |  0.74   |
| presidio       |   0.31    |   0.34   |    0.38    |  0.43   |

### Error decomposition (Strict schema)

| detector       | COR  | INC  | PAR  | MIS  | SPU  |
| -------------- | ---- | ---- | ---- | ---- | ---- |
| skyflow_minimal| 5,012| 387  | 0    | 743  | 412  |
| opf            | 4,834| 612  | 0    | 901  | 318  |
...
```

The COR/INC/PAR/MIS/SPU table is the part that's net-new diagnostic value — it's the thing we cannot answer today.

### Type schema replaces merged-adjacent

The Type schema (any overlap + matching type = COR) is the principled equivalent of our merged-adjacent hack — both reward "got the type right, boundary differs from gold". This PR removes `_merge_adjacent` from `report.py` and the corresponding "granularity-neutral" table from `RESULTS.md`. The SemEval Type column subsumes them.

Confirmed empirically on the 1k sample (after implementation): OPF merged-adjacent F1 0.825 ↔ Type F1 0.837 (close, both granularity-tolerant). Presidio shows a larger gap (Type is more generous about wrong-boundary matches like `example.com` inside `joe@example.com`) — that's a feature, not a bug.

### Per-detector module signature

```python
# nervaluate_metrics.py

@dataclass(frozen=True)
class SemEvalResult:
    detector: str
    by_schema: dict[str, dict[str, float]]   # {"strict": {"f1": ..., "p": ..., "r": ...}, ...}
    error_counts: dict[str, dict[str, int]]  # {"strict": {"COR": ..., "INC": ..., ...}, ...}
    by_label: dict[str, dict[str, dict[str, float]]]  # tag -> schema -> {p, r, f1}

def score(detector: str, fixtures: list[dict], pred_rows: list[dict], tags: list[str]) -> SemEvalResult:
    ...
```

`pred_rows` is the same `raw_<detector>.jsonl` we already load in `report.py`.

## Risks / open questions

- **Performance**. nervaluate's matching does more work per instance than our greedy 1:1 (it tracks all four schemas + decomposes errors). For 5k fixtures × 6 detectors that's 30k Evaluator calls. Likely fine (each instance is short), but benchmark first; if slow, we can pre-aggregate or call nervaluate once per (detector, schema) over the full list.
- **Char-offset assumption**. nervaluate's `default` loader expects `{label, start, end}` with the offsets matching the same coordinate system on both sides. Our gold and predictions are both char-offset, so this should "just work" — verify in step 1.
- **Multi-label gold spans.** PII-Masking-300k can have overlapping `GIVENNAME1`/`GIVENNAME2` annotations that map to the same canonical (PERSON). After taxonomy collapse we may have duplicate spans at the same offset. Decide: dedupe, or leave (nervaluate may handle either).
- **Label scoping vs total counts.** nervaluate's `tags=` filter changes what counts as an entity. The current report's "restricted" view applies the same filter logic, so totals should align — but worth a sanity check on day 1.
- **API stability.** nervaluate is at 0.x; the result tuple shape has changed historically. Pin a minor version (`nervaluate>=0.2,<0.3`) and re-run on upgrades.
- **Numbers may shock readers.** Strict-schema F1 is *much* lower than partial-IoU F1 (no credit for boundary mismatches). When publishing, lead with Type or Partial schema in the headline so the OPF/Skyflow comparison is recognizable next to the current numbers; show Strict in the breakdown.

## Verification

1. `uv add nervaluate && uv sync`
2. Sanity check: run `nervaluate_metrics.score(...)` on a 100-fixture run; confirm `COR + MIS == gold_count` (sanity invariant) and `COR + SPU == pred_count`.
3. Compare nervaluate's "Type" F1 to our merged-adjacent F1 for the same detector — should be within ±2 points if both are doing the same thing conceptually.
4. Add the new section to a 1k report; eyeball that `presidio` shows high MIS for non-English categories and high SPU for English (matching what we already know).
5. Repeat on the existing 5k run.
6. Drop into [RESULTS.md](../RESULTS.md) once it stabilizes.

## Effort

- Add dependency + write `nervaluate_metrics.py`: ~1 hour
- Wire the new section in `report.py`: ~1 hour
- Sanity checks + regenerate reports: ~1 hour
- README + RESULTS update: ~30 min
- **Total: ~3-4 hours**

## Out of scope

- Removing the current greedy-1:1 metrics
- Removing the merged-adjacent view (defer until Type schema confirms equivalence)
- Token-level / BIO scoring
- Tweaking nervaluate's internal matching algorithm
- Per-language SemEval breakdown (easy follow-up but not in v1)
- Generating nervaluate's full text report — we want the structured numbers, not its built-in markdown
