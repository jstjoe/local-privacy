# Plan 09 — Multi-dataset benchmark support

## Why

Today everything is wired to a single dataset (`ai4privacy/pii-masking-300k`) and the report's restricted-scoring view assumes OPF's label set is the comparison axis (the project's original framing was "OPF vs Skyflow"). As this turns into a general PII-detector benchmark, both assumptions need to come out:

- One dataset is a known caveat in [RESULTS.md](../RESULTS.md). Multiple ai4privacy variants give us volume, vocabulary, and composition diversity.
- Restricting every detector's headline to OPF's 8 categories punishes detectors with broader vocabularies and flatters detectors with narrower ones. Each detector should be scored on what *it* claims to detect, intersected with what *the dataset* annotates.

Datasets to support:

| dataset | size | vocab | use |
| --- | --- | --- | --- |
| [openpii-masking-nano-1k](https://huggingface.co/datasets/ai4privacy/openpii-masking-nano-1k) | 1k | OpenPII | smoke tests + CI; pinned 1k subset |
| [openpii-masking-mini-10k](https://huggingface.co/datasets/ai4privacy/openpii-masking-mini-10k) | 10k | OpenPII | mid-size benchmark, faster iteration |
| [pii-masking-200k](https://huggingface.co/datasets/ai4privacy/pii-masking-200k) | 200k | legacy PII-Masking | older baseline |
| [pii-masking-300k](https://huggingface.co/datasets/ai4privacy/pii-masking-300k) | 300k | legacy PII-Masking | current default |
| [pii-masking-400k](https://huggingface.co/datasets/ai4privacy/pii-masking-400k) | 400k | legacy PII-Masking | newest legacy variant |

Goal: pick the dataset on the command line, materialize fixtures with that dataset's vocabulary, and emit a report with **two scoring views per detector**:

1. **Fair score** — each detector restricted to (dataset annotates ∩ this detector's supported labels). Apples-to-apples within each detector's claimed coverage.
2. **Raw dataset score** — each detector evaluated against the dataset's full annotated label set; labels the detector doesn't support count as misses. Reflects real-world coverage gaps.

## Scope

In:

- Dataset registry mapping a short name → loader config (HF id, label-vocab key, default split)
- `--dataset` flag on the fixtures CLI; the registry resolves everything else
- Per-dataset canonical-label mapping (extend `taxonomy.CANONICAL_MAP` with one column per dataset vocab)
- Embed dataset metadata in fixtures + manifest so the report knows which annotation set was used
- **Two new scoring views per detector** (fair + raw); replaces the current single OPF-restricted headline
- `--canonical-labels` override remains for one-category drilldowns

Out:

- Adding non-ai4privacy datasets (deferred — would need new schema parsers)
- Synthetic / hand-curated fixtures (deferred)
- Per-dataset detector tuning (e.g. a different Presidio config per dataset language mix)
- Cross-dataset comparison tables in one report (would need a separate harness layer)

## Layout

```
eval/src/opf_eval/
├── fixtures.py              # MODIFY — accept --dataset, dispatch to loader
├── datasets/                # NEW package
│   ├── __init__.py          # NEW — DATASETS registry
│   ├── base.py              # NEW — DatasetConfig dataclass
│   ├── pii_masking.py       # NEW — loader for pii-masking-200k/300k/400k
│   └── openpii.py           # NEW — loader for openpii-masking-nano/mini/...
├── taxonomy.py              # MODIFY — add 'openpii' column; rename current
│                            #   'pii300k' to 'pii_masking' (covers 200k/300k/400k)
├── runner.py                # MODIFY — manifest carries dataset name
└── report.py                # MODIFY — restricted-label default reads from manifest
```

## Dataset registry

```python
# eval/src/opf_eval/datasets/base.py

@dataclass(frozen=True)
class DatasetConfig:
    name: str               # short name used on the CLI ("pii_masking_300k")
    hf_id: str              # "ai4privacy/pii-masking-300k"
    default_split: str
    vocab_key: str          # which CANONICAL_MAP column to use ("pii_masking", "openpii")
    loader: Callable[[dict, str], list[dict]]   # record -> canonical spans
```

```python
# eval/src/opf_eval/datasets/__init__.py

from .pii_masking import load_pii_masking
from .openpii import load_openpii

DATASETS: dict[str, DatasetConfig] = {
    "openpii_nano": DatasetConfig("openpii_nano", "ai4privacy/openpii-masking-nano-1k",
                                   "train", "openpii", load_openpii),
    "openpii_mini": DatasetConfig("openpii_mini", "ai4privacy/openpii-masking-mini-10k",
                                   "train", "openpii", load_openpii),
    "pii_masking_200k": DatasetConfig("pii_masking_200k", "ai4privacy/pii-masking-200k",
                                       "train", "pii_masking", load_pii_masking),
    "pii_masking_300k": DatasetConfig("pii_masking_300k", "ai4privacy/pii-masking-300k",
                                       "train", "pii_masking", load_pii_masking),
    "pii_masking_400k": DatasetConfig("pii_masking_400k", "ai4privacy/pii-masking-400k",
                                       "train", "pii_masking", load_pii_masking),
}
DEFAULT_DATASET = "pii_masking_300k"   # backwards-compatible
```

The two loader functions encapsulate the dataset-specific schema parsing — `load_pii_masking` is essentially today's `_record_spans` in `fixtures.py`; `load_openpii` is new and validated in step 1.

## Taxonomy extensions

The current `CANONICAL_MAP` has columns `opf`, `skyflow`, `pii300k`, `presidio`, `gliner`. Two changes:

1. **Rename `pii300k` → `pii_masking`** since the same vocabulary applies to 200k / 300k / 400k. Keep `pii300k_to_canonical()` as a deprecation-only alias for one release (called from no places after this PR; only here for any downstream consumers).
2. **Add `openpii`** column — populated from the OpenPII vocabulary. The OpenPII schema reportedly extends the legacy vocab with a few new types (e.g. `IDENTIFIER`, normalized tense for `DATEOFBIRTH` vs `DATE`); needs verification against an actual record before the column is filled.

Per-dataset reverse maps follow the existing pattern:

```python
_PII_MASKING_TO_CANONICAL = _build_reverse("pii_masking")
_OPENPII_TO_CANONICAL = _build_reverse("openpii")

def to_canonical(vocab_key: str, raw_label: str) -> str | None:
    if vocab_key == "pii_masking":
        return _PII_MASKING_TO_CANONICAL.get(raw_label)
    if vocab_key == "openpii":
        return _OPENPII_TO_CANONICAL.get(raw_label)
    raise ValueError(f"unknown dataset vocab: {vocab_key}")
```

Each loader calls `to_canonical(config.vocab_key, raw_label)` instead of the hard-coded `pii300k_to_canonical`.

## Two scoring views per detector

Today `report.py` restricts every detector's headline to `OPF_CANONICAL_LABELS`. That answers exactly one question — "how does each detector compare on OPF's eight categories?" — and is the wrong answer for any of these:

- "What's GLiNER's quality on the entities GLiNER actually claims to detect?" (today: depressed by labels GLiNER doesn't support)
- "If I deploy Presidio against this dataset as-is, what fraction of the PII does it find?" (today: not reported at all)
- "Does dataset X annotate more entity types than dataset Y?" (today: hidden by the OPF filter)

Two views per detector replace the single OPF-restricted view:

### View 1: Fair score (per detector)

For each detector, score against `dataset_canonicals(vocab) ∩ detector_supported_canonicals(detector)`. This is the apples-to-apples question: how good is this detector on the entities it claims to detect *and* this dataset annotates?

```python
def detector_supported_canonicals(detector: str) -> set[str]:
    """Canonical labels this detector can produce (variants like
    skyflow_full / presidio_multilang collapse to their parent vocab)."""
    source = _DETECTOR_VOCAB_KEY.get(detector, detector)   # see below
    return {
        c for c, by_src in CANONICAL_MAP.items()
        if by_src.get(source)
    }

def dataset_canonicals(vocab_key: str) -> set[str]:
    return {
        c for c, by_src in CANONICAL_MAP.items()
        if by_src.get(vocab_key)
    }

def fair_labels(detector: str, vocab_key: str) -> set[str]:
    return detector_supported_canonicals(detector) & dataset_canonicals(vocab_key)
```

`_DETECTOR_VOCAB_KEY` collapses variants:

```python
_DETECTOR_VOCAB_KEY = {
    "skyflow_full": "skyflow",
    "presidio_multilang": "presidio",
    # opf, gliner, presidio, skyflow, opf_calibrated all use their own name
}
```

The fair view's headline columns therefore differ across detectors — each row says "scored on N labels: …" so the reader knows what's being compared.

### View 2: Raw dataset score (per detector, full vocabulary)

For each detector, score against `dataset_canonicals(vocab)` — the full annotated set, no per-detector trimming. Labels a detector doesn't support count as misses (zero recall on those labels), which is the honest answer to "what fraction of this dataset's PII does this detector find out-of-the-box?"

The two views together let a reader answer both "is this detector well-tuned to its own claims?" (fair) and "does this detector cover what I actually need?" (raw).

### Manifest + override

The runner writes `dataset` into the manifest; `report.py` reads it and computes both views automatically. `--canonical-labels` still works as a per-category drilldown override (forces both views to the same explicit set).

### Per-category section is unchanged

The per-category breakdown already shows P/R/F1 per label per detector — no change needed. Detectors that don't support a label show `—` in their column, which already conveys "doesn't claim to detect this".

## Dataset-aware detector configuration

Skyflow and GLiNER both take a per-call label set. Previously those sets were hard-coded: `SKYFLOW_MINIMAL_ENTITY_TYPES` was tuned on PII-Masking-300k specifically; `gliner_prompts()` always returned the full union. With dataset as a first-class input, both auto-derive from the dataset's vocabulary so that picking `(detector, dataset, size)` runs the right configuration without manual flags. The retired `skyflow_minimal` and `skyflow_constrained` aliases are removed in this PR — the new default `skyflow` subsumes both.

Per-detector behavior:

| detector | configuration source after this PR |
| --- | --- |
| `skyflow` | `canonical_to_skyflow_request_types(dataset_canonicals)` — replaces "request everything" + the hand-tuned 24-entity preset |
| `skyflow_full` | unchanged — explicit unconstrained call returning all ~70 entity types |
| `gliner` | prompts restricted to `gliner_prompts(dataset_canonicals)` |
| `opf` / `presidio` | unchanged — fixed output vocab; the "request these only" filter is applied at scoring time, not call time |

Runner threads `dataset_canonicals(vocab_key)` into `_build_detector`:

```python
def _build_detector(name: str, *, dataset_canonicals_set: set[str]) -> Detector:
    if name == "skyflow":
        entity_types = canonical_to_skyflow_request_types(sorted(dataset_canonicals_set))
        return SkyflowDetector(entity_types=entity_types)
    if name == "skyflow_full":
        return SkyflowDetector(entity_types=None)
    if name == "gliner":
        return GLiNERDetector(prompts=gliner_prompts(dataset_canonicals_set))
    if name == "opf":
        return OPFDetector(...)   # vocab is fixed
    ...
```

`gliner_prompts_for` is a small extension to today's `gliner_prompts()`:

```python
def gliner_prompts_for(canonicals: Iterable[str] | None = None) -> list[str]:
    """Return GLiNER prompts for the given canonical labels (or all if None)."""
    targets = set(canonicals) if canonicals is not None else set(CANONICAL_MAP)
    out, seen = [], set()
    for canonical, by_source in CANONICAL_MAP.items():
        if canonical not in targets:
            continue
        for p in by_source.get("gliner", ()):
            if p not in seen:
                seen.add(p)
                out.append(p)
    return out
```

`GLiNERDetector` gains a `prompts: list[str] | None = None` constructor arg; defaults to `gliner_prompts_for(None)` for backwards compat.

### Effect on the "raw" scoring view

The raw view still scores each detector against the dataset's full canonical set — but with detectors only *asked for* the dataset's labels, we stop penalizing them for spurious detections of labels the dataset doesn't annotate. That's the correct behavior:

- Coverage gaps remain visible (a detector that doesn't support `USERNAME` still takes zero recall on it in the raw view).
- "FPs that aren't actually FPs in the real world" disappear, which is what the dataset's narrower lens would have created anyway.

### Optional override

If a user wants to opt out of dataset-aware config (e.g. to reproduce older runs):

```sh
python -m opf_eval.runner ... --detector-entities skyflow=ALL
python -m opf_eval.runner ... --detector-entities skyflow=email_address,phone_number
```

`ALL` = today's request-everything behavior; comma-list = explicit Skyflow request types or GLiNER prompts. Out of scope for v1; document the intended syntax for future plans.

## Implementation notes

### Step 1 — Schema verification

Before writing the OpenPII loader, **fetch one record** from `openpii-masking-nano-1k` and dump its column names + types. The loader is built from that. `pii-masking-200k`/`400k` are likely identical to `300k`'s schema (same maintainers, same lineage) but verify by fetching one record from each. Spend ≤30 minutes here, no more.

### Step 2 — Manifest changes

```json
{
  "started_at": "...",
  "fixtures": "eval/data/openpii_mini.jsonl",
  "dataset": "openpii_mini",
  "n_examples": 10000,
  "detectors": [...]
}
```

`report.py` reads `dataset`; if missing (older runs), falls back to the current OPF-canonical default.

### Step 3 — Fixtures CLI

```sh
python -m opf_eval.fixtures --dataset openpii_nano --out eval/data/openpii_nano.jsonl --n 1000
```

Default `--dataset` stays `pii_masking_300k` for backwards compatibility. `--n` larger than the dataset size = take everything.

### Step 4 — Backwards-compatibility

Existing fixtures (no `dataset` field, no manifest entry) keep working. The fixtures schema is unchanged: still `{id, text, gold_spans, language}` — the only difference is which vocabulary `gold_spans[*].label` came from, which is constant within a fixtures file by construction.

## Risks / open questions

- **`skyflow` semantic change is breaking.** Pre-PR `skyflow` requested OPF's 8 categories; post-PR it requests the dataset's full canonical set. The previously-shipped `skyflow_minimal` and `skyflow_constrained` aliases are removed. Mitigation: document in CHANGELOG; for the historical request-everything behavior pass `skyflow_full`.
- **OpenPII schema unknowns**. The model cards I've seen don't enumerate every entity type. Step 1 of the implementation (record dump) is non-negotiable.
- **OpenPII vocab might collapse to legacy after canonicalization**. If 95% of OpenPII labels round-trip to the same canonical labels as `pii-masking-300k`, the practical difference is just dataset size + composition. Note this in the verification step; it doesn't kill the plan, but may shrink the "added value" story.
- **Per-dataset language coverage**. PII-Masking-300k's six languages may not be the same set in OpenPII. The per-language report section already only emits rows for languages present in the fixture set, so this is benign — just don't promise coverage we don't have.
- **HF download size**. `pii-masking-400k` is the largest; full materialization with `streaming=False` will hit `~/.cache/huggingface/`. Add a note in the README about cache location and cleanup. Optional follow-up: switch to `streaming=True` for big sets.
- **Field naming consistency**. The fixtures file embeds the dataset name in the manifest, not in each record. If someone manually concatenates fixtures from multiple datasets, the manifest lies. Fine for v1; the runner only operates on one fixtures file at a time.

## Verification

1. Dump one record per new dataset; confirm columns match the legacy schema or document the diff.
2. `python -m opf_eval.fixtures --dataset openpii_nano --out eval/data/openpii_nano.jsonl --n 1000` → produces 1000 lines, all with non-empty `gold_spans`, no `null` languages where the source provides them.
3. Re-materialize today's `sample_1k.jsonl` via `--dataset pii_masking_300k --n 1000 --seed 42` and `diff` against the existing file — should be byte-identical.
4. Run a full benchmark on `openpii_nano`: `python -m opf_eval.runner --fixtures eval/data/openpii_nano.jsonl --detectors opf,gliner,presidio --out eval/results/runs/openpii_nano_v1/` and report — confirm:
   - The "fair" view's per-row label set differs by detector (each scoped to that detector's supported labels).
   - The "raw" view's column header lists the dataset's full annotated set.
   - On a label a detector doesn't support: fair-view F1 ignores it; raw-view F1 takes a recall hit.
5. Sanity check: a detector whose supported set fully covers the dataset's vocabulary should score the same in both views.
6. Verify the SemEval section (plan 08) still scores cleanly — same detector outputs, different fixtures, both views.
7. **Dataset-aware config sanity**: on `openpii_nano`, log the actual `entity_types` Skyflow was called with and the actual prompts GLiNER was called with. Confirm both lists derive from the dataset's annotated canonicals — not the union of all canonicals.
8. Re-run `skyflow` on `pii_masking_300k` and compare to the previously-published `skyflow_minimal` numbers in RESULTS.md; expect fair-view F1 within ±2 (the auto-derived list is broader than the hand-tuned one but should track close).

## Effort

- Step 1 schema dump + decisions: ~30 min
- Datasets package + registry + per-dataset loaders: ~2 hours
- Taxonomy extensions: ~1 hour (mostly mapping work)
- Fixtures + runner + manifest wiring: ~1 hour
- Dataset-aware detector config (skyflow + gliner builders, `gliner_prompts_for`): ~1 hour
- Report refactor for fair + raw views (replaces the single-view headline): ~2 hours
- Verification runs (nano + a re-mat of existing 1k + skyflow re-run on 300k): ~1.5 hours
- README + RESULTS update: ~1 hour (the framing shift from "OPF vs Skyflow" to "general benchmark" needs language tweaks too)
- **Total: ~1-1.5 days**

## Out of scope

- Adding non-ai4privacy datasets (e.g. `Mendeley`, `gretel-ai/synth-financial-pii`)
- Hand-curated / synthetic fixtures
- Cross-dataset overlay reports ("OPF on dataset A vs dataset B in one table") — separate plan if useful
- Re-tuning Skyflow's `entity_types` per dataset (current minimal set was tuned on 300k; may be sub-optimal on others)
- Per-dataset cost/budget estimates for Skyflow runs
- Streaming-mode dataset loading (acceptable to require local cache for v1)
