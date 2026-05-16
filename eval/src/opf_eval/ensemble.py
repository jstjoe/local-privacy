"""Post-hoc composite-detector ensembles.

Reads raw_<detector>.jsonl files emitted by `runner.run`, builds a per-label
recipe (which detector wins on which canonical label, by Type-schema F1
against the gold spans in the fixture file), and emits a new
`raw_<ensemble_name>.jsonl` synthesised from the recipe. The same report
pipeline that scores individual detectors picks this up automatically.

Strategies
----------
`category_best`
    For each canonical label, pick the detector with the highest per-label
    Type-schema F1 on this dataset, and use only its predictions for that
    label. Per-record span lists are the union over labels, deduped by
    position (earliest start wins; ties broken by longer span).

Caveat: the recipe is built against the same gold spans the ensemble is
later scored on, so reported numbers are an upper bound rather than a
generalisation estimate. For honest cross-validation, split fixtures into
recipe-fit and recipe-eval halves before calling `build_recipe_*`.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from pathlib import Path

from .datasets import DEFAULT_DATASET, get as get_dataset_config, names as dataset_names
from .nervaluate_metrics import score as semeval_score
from .taxonomy import dataset_canonicals


def _read_jsonl(path: Path) -> Iterable[dict]:
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _register_in_manifest(out_dir: Path, name: str) -> None:
    """Add the ensemble name to manifest.json's `detectors` list so the
    report cell (which reads detectors from the manifest, not via glob)
    picks the new raw file up. No-op if no manifest exists or the entry
    is already present.
    """
    manifest_path = out_dir / "manifest.json"
    if not manifest_path.exists():
        return
    try:
        manifest = json.loads(manifest_path.read_text())
    except json.JSONDecodeError:
        return
    detectors = list(manifest.get("detectors") or [])
    if name in detectors:
        return
    detectors.append(name)
    manifest["detectors"] = sorted(set(detectors))
    manifest_path.write_text(json.dumps(manifest, indent=2))


def _detectors_in(out_dir: Path) -> list[str]:
    """All non-ensemble detectors with a raw file in out_dir, sorted."""
    out = []
    for p in sorted(out_dir.glob("raw_*.jsonl")):
        name = p.stem.removeprefix("raw_")
        if name.startswith("ensemble_"):
            continue
        out.append(name)
    return out


def build_recipe_category_best(
    out_dir: Path,
    fixtures_path: Path,
    labels: list[str],
    *,
    excluded_detectors: set[str] | None = None,
    fit_ids: set[str] | None = None,
) -> tuple[dict[str, str], dict[str, dict[str, float]]]:
    """Pick the detector with the highest Type-schema F1 per canonical label.

    Returns (recipe, per_label_f1) where:
      recipe[label] = detector_name chosen for that label (omitted if no
        detector registered a positive F1 for that label)
      per_label_f1[label][detector] = F1 (Type schema) for inspection / logging

    `fit_ids`, when given, restricts the F1 calculation pool to that subset
    of fixture ids — useful when running a holdout split (fit the recipe on
    one half, score it on the other) to get a generalisation estimate
    instead of an upper bound.
    """
    excluded = set(excluded_detectors or ())
    fixture_index = {r["id"]: r for r in _read_jsonl(fixtures_path)}
    if fit_ids is not None:
        fixture_index = {i: r for i, r in fixture_index.items() if i in fit_ids}
    detectors = [d for d in _detectors_in(out_dir) if d not in excluded]
    label_set = set(labels)

    per_label_f1: dict[str, dict[str, float]] = {lbl: {} for lbl in labels}
    for det in detectors:
        path = out_dir / f"raw_{det}.jsonl"
        pairs = []
        for r in _read_jsonl(path):
            if r.get("error"):
                continue
            fx = fixture_index.get(r["id"])
            if fx is None:
                continue
            gold = [s for s in fx["gold_spans"] if s["label"] in label_set]
            pred = [s for s in (r.get("spans") or []) if s["label"] in label_set]
            pairs.append((pred, gold))
        if not pairs:
            continue
        sem = semeval_score(detector=det, pairs=pairs, tags=labels)
        for lbl in labels:
            f1 = sem.by_label.get(lbl, {}).get("ent_type", {}).get("f1", 0.0)
            per_label_f1[lbl][det] = float(f1)

    recipe: dict[str, str] = {}
    for lbl, f1s in per_label_f1.items():
        if not f1s:
            continue
        best = max(f1s.items(), key=lambda kv: (kv[1], kv[0]))
        # Skip labels where no detector scores above zero — emitting an empty
        # entry would just add noise to the ensemble's output.
        if best[1] > 0.0:
            recipe[lbl] = best[0]
    return recipe, per_label_f1


def apply_recipe(
    recipe: dict[str, str],
    out_dir: Path,
    fixtures_path: Path,
    *,
    ensemble_name: str = "ensemble_category_best",
) -> Path:
    """Synthesise raw_<ensemble_name>.jsonl from the recipe.

    For each fixture id, gather every span from the recipe's chosen detector
    whose label matches that detector's assigned label. Union over labels,
    sort by start (longer first on ties), drop strict overlaps so the output
    matches the single-detector contract (non-overlapping spans).
    """
    fixture_records = list(_read_jsonl(fixtures_path))
    needed = set(recipe.values())
    by_detector: dict[str, dict[str, dict]] = {}
    for det in needed:
        path = out_dir / f"raw_{det}.jsonl"
        if not path.exists():
            raise FileNotFoundError(
                f"recipe references detector {det!r} but {path} is missing"
            )
        by_detector[det] = {r["id"]: r for r in _read_jsonl(path)}

    out_path = out_dir / f"raw_{ensemble_name}.jsonl"
    _register_in_manifest(out_dir, ensemble_name)
    with out_path.open("w") as f:
        for fx in fixture_records:
            fid = fx["id"]
            picked: list[dict] = []
            latency_ms = 0.0
            errors: list[str] = []
            for lbl, det in recipe.items():
                rec = by_detector[det].get(fid)
                if rec is None:
                    continue
                if rec.get("error"):
                    errors.append(f"{det}:{lbl}: {rec['error']}")
                for s in rec.get("spans") or []:
                    if s["label"] == lbl:
                        picked.append(s)
                latency_ms += float(rec.get("latency_ms") or 0.0)

            # Dedupe overlapping spans. Sort by start ascending, longer first
            # on ties so the more informative span wins. Drop any later span
            # whose start lies before the previous accepted span's end.
            picked.sort(key=lambda s: (int(s["start"]), -(int(s["end"]) - int(s["start"]))))
            deduped: list[dict] = []
            prev_end = -1
            for s in picked:
                if int(s["start"]) >= prev_end:
                    deduped.append(s)
                    prev_end = int(s["end"])

            f.write(
                json.dumps(
                    {
                        "id": fid,
                        "detector": ensemble_name,
                        "spans": deduped,
                        "latency_ms": latency_ms,
                        "error": "; ".join(errors) if errors else None,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    return out_path


def run_category_best(
    out_dir: Path,
    fixtures_path: Path,
    *,
    dataset: str = DEFAULT_DATASET,
    excluded_detectors: set[str] | None = None,
    ensemble_name: str = "ensemble_category_best",
    fit_ids: set[str] | None = None,
) -> tuple[Path, dict[str, str], dict[str, dict[str, float]]]:
    """Build recipe + apply. Returns (out_path, recipe, per_label_f1).

    `fit_ids` is forwarded to `build_recipe_category_best` so callers can
    fit the recipe on one half of their fixtures and score on the other —
    pass the fit-half ids here, then run `report.build_report` against a
    fixtures file containing only the score-half ids.
    """
    cfg = get_dataset_config(dataset)
    labels = sorted(dataset_canonicals(cfg.vocab_key))
    recipe, per_label_f1 = build_recipe_category_best(
        out_dir,
        fixtures_path,
        labels,
        excluded_detectors=excluded_detectors,
        fit_ids=fit_ids,
    )
    out_path = apply_recipe(recipe, out_dir, fixtures_path, ensemble_name=ensemble_name)
    return out_path, recipe, per_label_f1


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", required=True, type=Path,
                    help="run directory containing raw_<detector>.jsonl files")
    ap.add_argument("--fixtures", required=True, type=Path,
                    help="JSONL with gold spans (the same file fed to runner.run)")
    ap.add_argument("--dataset", default=DEFAULT_DATASET, choices=dataset_names(),
                    help=f"dataset name for label vocab (default: {DEFAULT_DATASET})")
    ap.add_argument("--strategy", default="category_best",
                    choices=["category_best"],
                    help="ensemble strategy")
    ap.add_argument("--exclude", action="append", default=[],
                    help="detector names to exclude from the recipe; repeatable")
    ap.add_argument("--name", default=None,
                    help="output ensemble name; defaults to ensemble_<strategy>")
    args = ap.parse_args()

    name = args.name or f"ensemble_{args.strategy}"
    if args.strategy == "category_best":
        out_path, recipe, per_label_f1 = run_category_best(
            args.out_dir,
            args.fixtures,
            dataset=args.dataset,
            excluded_detectors=set(args.exclude),
            ensemble_name=name,
        )
    else:  # pragma: no cover — argparse choices guards this
        raise ValueError(f"unknown strategy: {args.strategy}")

    print(f"recipe ({args.strategy}):")
    for lbl in sorted(recipe):
        choices = sorted(per_label_f1[lbl].items(), key=lambda kv: -kv[1])
        head = ", ".join(f"{d}={f:.2f}" for d, f in choices[:3])
        print(f"  {lbl:<14} -> {recipe[lbl]:<24}  (candidates: {head})")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
