"""Score a run directory: load predictions + gold, pick label sets, call nervaluate.

The one place that decides *which* labels a detector is scored on and *how*
span labels are projected before comparison. `report`, `ensemble` and the
notebooks all go through here.

Views:
    fair  each detector scored on (dataset annotates ∩ detector supports)
    raw   every detector scored on everything the dataset annotates

Levels:
    coarse  the 15 top-level categories (historical default)
    fine    finest granularity both sides share, per category — see
            `taxonomy.shared_projector`
"""

from __future__ import annotations

import json
import random
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path

from . import taxonomy
from .io import index_by_id, iter_jsonl, read_jsonl
from .metrics import latency_summary
from .nervaluate_metrics import SCHEMAS, SemEvalResult
from .nervaluate_metrics import score as semeval_score

VIEWS = ("fair", "raw")
SCHEMA_TITLES = {"strict": "Strict", "exact": "Exact", "partial": "Partial", "ent_type": "Type"}


@dataclass
class Run:
    """A runner output directory loaded into memory."""

    dir: Path
    manifest: dict
    fixtures: list[dict]
    predictions: dict[str, dict[str, dict]] = field(default_factory=dict)  # det -> id -> row

    @classmethod
    def load(
        cls,
        run_dir: str | Path,
        fixtures: str | Path | None = None,
        *,
        detectors: Iterable[str] | None = None,
    ) -> "Run":
        run_dir = Path(run_dir)
        manifest = json.loads((run_dir / "manifest.json").read_text())
        fx_path = Path(fixtures) if fixtures else Path(manifest["fixtures"])
        run = cls(run_dir, manifest, read_jsonl(fx_path))
        for det in detectors if detectors is not None else manifest["detectors"]:
            path = run_dir / f"raw_{det}.jsonl"
            if path.exists():
                run.predictions[det] = index_by_id(iter_jsonl(path))
        return run

    @property
    def detectors(self) -> list[str]:
        return [d for d in self.manifest["detectors"] if d in self.predictions]

    @property
    def ids(self) -> list[str]:
        return [r["id"] for r in self.fixtures]

    @property
    def dataset(self) -> str | None:
        return self.manifest.get("dataset")

    def dataset_labels(self, level: str = "coarse") -> set[str]:
        """Labels the fixtures' dataset annotates, at `level`."""
        if self.manifest.get("labels") is not None:
            fine = set(self.manifest["labels"])
        else:  # manifests written before labels were recorded
            vocab = self.manifest.get("vocab_key")
            if not vocab:
                from .datasets import DEFAULT_DATASET, get

                vocab = get(self.manifest.get("dataset") or DEFAULT_DATASET).vocab_key
            fine = taxonomy.vocab_labels(vocab, "fine")
        return {taxonomy.project(lbl, level) for lbl in fine}

    def reload(self, detector: str) -> None:
        """(Re)read one detector's raw file — e.g. after an ensemble wrote it."""
        path = self.dir / f"raw_{detector}.jsonl"
        self.predictions[detector] = index_by_id(iter_jsonl(path))
        if detector not in self.manifest["detectors"]:
            self.manifest["detectors"] = sorted({*self.manifest["detectors"], detector})


# ------------------------------------------------------------ label sets


@dataclass(frozen=True)
class Scope:
    """What one detector is scored on: allowed labels + label projection."""

    labels: frozenset[str]
    project: Callable[[str], str]


def scope_for(
    run: Run,
    detector: str,
    *,
    level: str = "coarse",
    view: str = "fair",
    labels: Iterable[str] | None = None,
) -> Scope:
    """Label scope for `detector` under (`level`, `view`).

    `labels`, when given, overrides both views with an explicit set (e.g. a
    one-category drilldown); labels are projected to `level` first.
    """
    taxonomy.check_level(level)
    if view not in VIEWS:
        raise ValueError(f"unknown view {view!r}; expected one of {VIEWS}")
    if level == "coarse":
        proj: Callable[[str], str] = taxonomy.parent
        ds = run.dataset_labels("coarse")
        det = taxonomy.detector_supported_canonicals(detector, "coarse")
    else:
        ds_fine = run.dataset_labels("fine")
        det_fine = taxonomy.detector_supported_canonicals(detector, "fine")
        proj = taxonomy.shared_projector(det_fine, ds_fine)
        ds = {proj(x) for x in ds_fine}
        det = {proj(x) for x in det_fine}
    if labels is not None:
        allow = {proj(x) for x in labels}
    else:
        allow = (ds & det) if view == "fair" else ds
        if level == "fine":
            # In categories scored at fine granularity, keep the coarse
            # parent in scope too: a detector saying ACCOUNT where the gold
            # says GOV_ID found the right span with a less specific label —
            # that should score as INC, not vanish into a miss.
            split = {taxonomy.parent(x) for x in allow if taxonomy.parent(x) != x}
            allow |= (split & det) if view == "fair" else split
    return Scope(frozenset(allow), proj)


def _span_label(span: dict, level: str) -> str:
    if level == "coarse":
        return taxonomy.parent(span["label"])
    return span.get("fine_label") or span["label"]


def _project(spans: list[dict], scope: Scope, level: str) -> list[dict]:
    out = []
    for s in spans:
        lbl = scope.project(_span_label(s, level))
        if lbl in scope.labels:
            out.append({"label": lbl, "start": int(s["start"]), "end": int(s["end"])})
    return out


@dataclass
class Pairs:
    pairs: list[tuple[list[dict], list[dict]]]
    latencies: list[float]
    errors: int
    missing: int


def build_pairs(
    run: Run,
    detector: str,
    scope: Scope,
    *,
    level: str = "coarse",
    ids: Iterable[str] | None = None,
    language: str | None = None,
) -> Pairs:
    """(pred, gold) span lists per fixture, restricted and projected by `scope`.

    Records the detector errored on are skipped and counted; fixtures with no
    prediction row at all are counted as `missing`.
    """
    id_filter = set(ids) if ids is not None else None
    preds = run.predictions.get(detector, {})
    out = Pairs([], [], 0, 0)
    for ex in run.fixtures:
        if id_filter is not None and ex["id"] not in id_filter:
            continue
        if language and ex.get("language") != language:
            continue
        rec = preds.get(ex["id"])
        if rec is None:
            out.missing += 1
            continue
        if rec.get("error"):
            out.errors += 1
            continue
        out.pairs.append((
            _project(rec.get("spans") or [], scope, level),
            _project(ex.get("gold_spans") or [], scope, level),
        ))
        out.latencies.append(float(rec.get("latency_ms") or 0.0))
    return out


# --------------------------------------------------------------- scoring


def score_detector(
    run: Run,
    detector: str,
    *,
    level: str = "coarse",
    view: str = "fair",
    labels: Iterable[str] | None = None,
    ids: Iterable[str] | None = None,
    language: str | None = None,
) -> SemEvalResult | None:
    """SemEval scores for one detector, or None if its scope is empty."""
    scope = scope_for(run, detector, level=level, view=view, labels=labels)
    if not scope.labels:
        return None
    p = build_pairs(run, detector, scope, level=level, ids=ids, language=language)
    return semeval_score(detector=detector, pairs=p.pairs, tags=sorted(scope.labels))


def score_run(
    run: Run,
    *,
    level: str = "coarse",
    view: str = "fair",
    labels: Iterable[str] | None = None,
    ids: Iterable[str] | None = None,
    detectors: Iterable[str] | None = None,
) -> dict[str, SemEvalResult | None]:
    dets = list(detectors) if detectors is not None else run.detectors
    labels = list(labels) if labels is not None else None
    return {
        d: score_detector(run, d, level=level, view=view, labels=labels, ids=ids)
        for d in dets
    }


def headline(results: dict[str, SemEvalResult | None]) -> list[dict]:
    """One row per detector: F1 under each schema plus Strict P/R."""
    rows = []
    for det, r in results.items():
        if r is None:
            rows.append({"detector": det, **{s: None for s in SCHEMAS}})
            continue
        rows.append({
            "detector": det,
            **{s: float(r.by_schema[s]["f1"]) for s in SCHEMAS},
            "strict_precision": float(r.by_schema["strict"]["precision"]),
            "strict_recall": float(r.by_schema["strict"]["recall"]),
        })
    return rows


def per_label_f1(
    result: SemEvalResult | None, labels: Iterable[str], *, schema: str = "ent_type"
) -> dict[str, float]:
    if result is None:
        return {lbl: 0.0 for lbl in labels}
    return {
        lbl: float(result.by_label.get(lbl, {}).get(schema, {}).get("f1", 0.0))
        for lbl in labels
    }


def coverage(run: Run) -> list[dict]:
    """Per detector: fixtures scored, errored, and missing a prediction row."""
    rows = []
    ids = run.ids
    for det in run.detectors:
        preds = run.predictions[det]
        errors = sum(1 for i in ids if i in preds and preds[i].get("error"))
        missing = sum(1 for i in ids if i not in preds)
        rows.append({
            "detector": det,
            "n": len(ids),
            "scored": len(ids) - errors - missing,
            "errors": errors,
            "missing": missing,
        })
    return rows


def latencies(run: Run, detector: str) -> dict[str, float]:
    """p50/p95/p99 per-call latency (ms) over non-errored records."""
    vals = [
        float(r.get("latency_ms") or 0.0)
        for r in run.predictions.get(detector, {}).values()
        if not r.get("error")
    ]
    return latency_summary(vals)


def holdout_split(ids: Iterable[str], frac: float, seed: int = 42) -> tuple[set[str], set[str]]:
    """Deterministic (fit, score) split of fixture ids — `frac` goes to fit."""
    shuffled = list(ids)
    random.Random(seed).shuffle(shuffled)
    k = int(len(shuffled) * frac)
    return set(shuffled[:k]), set(shuffled[k:])
