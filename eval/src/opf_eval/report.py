"""Read a runner output dir + the source fixtures, emit report.md.

Two scoring views per detector:

- **Fair**  — each detector scored against `dataset ∩ detector_supports`.
  Apples-to-apples within each detector's claimed coverage.
- **Raw**   — each detector scored against the full set the dataset annotates.
  Labels a detector doesn't support count as misses; reflects real-world
  out-of-the-box coverage.

Manifest carries `dataset` (registry name); the report defaults the per-
detector label sets from there. `--canonical-labels` overrides both views
to a single explicit set (use for one-category drilldowns).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .datasets import DEFAULT_DATASET, get as get_dataset_config
from .metrics import PRF, score
from .nervaluate_metrics import score as semeval_score
from .taxonomy import (
    CANONICAL_LABELS,
    dataset_canonicals,
    fair_labels,
)


def _load_jsonl(path: Path) -> list[dict]:
    out = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _index(records: list[dict], key: str = "id") -> dict[str, dict]:
    return {r[key]: r for r in records}


def _fmt_prf(p: PRF) -> str:
    return f"{p.precision:.3f} / {p.recall:.3f} / {p.f1:.3f}  (tp={p.tp} fp={p.fp} fn={p.fn})"


def _filter_spans(spans: list[dict], allow: set[str] | None) -> list[dict]:
    if allow is None:
        return spans
    return [s for s in spans if s["label"] in allow]


def _build_pairs(
    detector: str,
    fixture_records: list[dict],
    detector_records: dict[str, dict[str, dict]],
    *,
    allow: set[str] | None,
    language_filter: str | None = None,
) -> tuple[list[tuple[list[dict], list[dict]]], list[float], int]:
    pairs: list[tuple[list[dict], list[dict]]] = []
    latencies: list[float] = []
    errors = 0
    for ex in fixture_records:
        if language_filter and ex.get("language") != language_filter:
            continue
        rec = detector_records[detector].get(ex["id"])
        if rec is None:
            continue
        if rec.get("error"):
            errors += 1
            continue
        pred = _filter_spans(rec["spans"], allow)
        gold = _filter_spans(ex["gold_spans"], allow)
        pairs.append((pred, gold))
        latencies.append(rec["latency_ms"])
    return pairs, latencies, errors


def _greedy_per_label(
    detectors: list[str],
    fixture_records: list[dict],
    detector_records: dict[str, dict[str, dict]],
    *,
    allow: set[str],
) -> dict[str, object]:
    out: dict[str, object] = {}
    for det in detectors:
        pairs, _, _ = _build_pairs(
            det, fixture_records, detector_records, allow=allow,
        )
        out[det] = score(pairs)
    return out


def _per_label_section(
    title: str,
    detectors: list[str],
    score_by_det: dict[str, object],
) -> list[str]:
    lines: list[str] = [f"### {title}", ""]
    all_labels: set[str] = set()
    for det in detectors:
        all_labels.update(score_by_det[det].per_label_partial.keys())  # type: ignore[union-attr]
    lines.append("| label | " + " | ".join(detectors) + " |")
    lines.append("|---|" + "|".join("---" for _ in detectors) + "|")
    for lbl in sorted(all_labels):
        row = [lbl]
        for det in detectors:
            p = score_by_det[det].per_label_partial.get(lbl)  # type: ignore[union-attr]
            row.append(_fmt_prf(p) if p else "—")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    return lines


def _semeval_view(
    detectors: list[str],
    fixture_records: list[dict],
    detector_records: dict[str, dict[str, dict]],
    *,
    label_set_per_det: dict[str, set[str]],
    title: str,
    description: str,
) -> list[str]:
    """Render a SemEval section. `label_set_per_det` maps each detector to
    the labels it's scored against — different per detector for the fair
    view, identical for the raw view."""
    results = {}
    for det in detectors:
        labels = label_set_per_det[det]
        if not labels:
            results[det] = None
            continue
        pairs, _, _ = _build_pairs(
            det, fixture_records, detector_records, allow=labels,
        )
        results[det] = semeval_score(detector=det, pairs=pairs, tags=sorted(labels))

    lines: list[str] = [
        f"## {title}",
        "",
        description,
        "",
        "### Headline F1 by schema",
        "",
        "| detector | n labels | strict | exact | partial | type |",
        "|---|---|---|---|---|---|",
    ]
    for det in detectors:
        r = results[det]
        n = len(label_set_per_det[det])
        if r is None:
            lines.append(f"| {det} | 0 | — | — | — | — |")
            continue
        lines.append(
            "| {det} | {n} | {strict:.3f} | {exact:.3f} | {partial:.3f} | {ent:.3f} |".format(
                det=det, n=n,
                strict=r.by_schema["strict"]["f1"],
                exact=r.by_schema["exact"]["f1"],
                partial=r.by_schema["partial"]["f1"],
                ent=r.by_schema["ent_type"]["f1"],
            )
        )
    lines.append("")

    lines.extend([
        "### Error decomposition (Strict schema)",
        "",
        "| detector | COR | INC | PAR | MIS | SPU |",
        "|---|---|---|---|---|---|",
    ])
    for det in detectors:
        r = results[det]
        if r is None:
            lines.append(f"| {det} | — | — | — | — | — |")
            continue
        m = r.by_schema["strict"]
        lines.append(
            "| {det} | {c} | {i} | {p} | {ms} | {sp} |".format(
                det=det,
                c=m["correct"], i=m["incorrect"], p=m["partial"],
                ms=m["missed"], sp=m["spurious"],
            )
        )
    lines.append("")

    # Show which labels each detector was scored against (small per-row table).
    lines.extend(["### Label scopes", ""])
    for det in detectors:
        labels = label_set_per_det[det]
        lines.append(
            f"- **{det}** ({len(labels)}): {', '.join(sorted(labels)) if labels else '—'}"
        )
    lines.append("")
    return lines


def _per_language_semeval_section(
    detectors: list[str],
    fixture_records: list[dict],
    detector_records: dict[str, dict[str, dict]],
    *,
    label_set_per_det: dict[str, set[str]],
    languages: list[str],
) -> list[str]:
    """Per-language SemEval Type-schema F1 (fair view: per-detector scope)."""
    lines: list[str] = [
        "## Per-language (fair view, SemEval Type F1)",
        "",
        "Each detector is scored against its own (dataset ∩ detector-supported)"
        " label set per language. Type schema = any overlap + matching label.",
        "",
        "| language | n | " + " | ".join(detectors) + " |",
        "|---|---|" + "|".join("---" for _ in detectors) + "|",
    ]
    for lang in languages:
        n = sum(1 for ex in fixture_records if ex.get("language") == lang)
        row = [lang, str(n)]
        for det in detectors:
            labels = label_set_per_det[det]
            if not labels:
                row.append("—")
                continue
            pairs, _, _ = _build_pairs(
                det, fixture_records, detector_records,
                allow=labels, language_filter=lang,
            )
            r = semeval_score(detector=det, pairs=pairs, tags=sorted(labels))
            row.append(f"{r.by_schema['ent_type']['f1']:.3f}")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    return lines


def build_report(
    run_dir: Path,
    fixtures: Path,
    *,
    canonical_labels: tuple[str, ...] | None = None,
    max_disagreements: int = 50,
) -> str:
    """canonical_labels: when given, force both fair and raw views to this
    explicit label set (degenerates the two views into one). When None, the
    label sets come from the manifest's `dataset` field — fair = per-detector
    intersection, raw = full dataset vocabulary.
    """
    manifest = json.loads((run_dir / "manifest.json").read_text())
    detectors: list[str] = manifest["detectors"]
    fixture_records = _load_jsonl(fixtures)

    detector_records: dict[str, dict[str, dict]] = {}
    for det in detectors:
        path = run_dir / f"raw_{det}.jsonl"
        detector_records[det] = _index(_load_jsonl(path))

    dataset_name = manifest.get("dataset") or DEFAULT_DATASET
    vocab_key = manifest.get("vocab_key")
    if not vocab_key:
        # Older manifests: derive from the registry.
        vocab_key = get_dataset_config(dataset_name).vocab_key

    if canonical_labels:
        # Override mode: both views use the same explicit set.
        forced = set(canonical_labels)
        fair_set = {det: forced for det in detectors}
        raw_set = forced
    else:
        ds_canon = dataset_canonicals(vocab_key)
        fair_set = {det: fair_labels(det, vocab_key) for det in detectors}
        raw_set = ds_canon

    lines: list[str] = [
        f"# PII detector benchmark — {manifest['started_at']}",
        "",
        f"- dataset: `{dataset_name}` (vocab `{vocab_key}`)",
        f"- fixtures: `{manifest['fixtures']}` ({manifest['n_examples']} examples)",
        f"- detectors: {', '.join(detectors)}",
        "",
    ]

    # SemEval — Fair view (per-detector scope)
    lines.extend(_semeval_view(
        detectors, fixture_records, detector_records,
        label_set_per_det=fair_set,
        title="SemEval — Fair view (per-detector scope)",
        description=(
            "Each detector scored against the intersection of (dataset annotates,"
            " this detector supports). Apples-to-apples within each detector's"
            " claimed coverage. Schema definitions: **Strict** = exact boundary"
            " + label. **Exact** = boundary, ignore label. **Partial** = any"
            " overlap, ignore label. **Type** = any overlap + matching label."
        ),
    ))

    # SemEval — Raw view (full dataset vocabulary)
    lines.extend(_semeval_view(
        detectors, fixture_records, detector_records,
        label_set_per_det={det: raw_set for det in detectors},
        title="SemEval — Raw dataset view (full vocabulary)",
        description=(
            f"Every detector scored against the dataset's full annotated set"
            f" ({len(raw_set)} canonical labels). Labels a detector doesn't"
            f" support take zero recall here, so this view reflects out-of-the-"
            f"box coverage rather than fairness."
        ),
    ))

    # Per-category greedy partial — uses raw view's label set (uniform across detectors)
    score_raw = _greedy_per_label(
        detectors, fixture_records, detector_records, allow=raw_set,
    )
    lines.extend(_per_label_section(
        "Per-category breakdown — raw view (partial overlap, IoU >= 0.5)",
        detectors,
        score_raw,
    ))

    # Per-language fair view
    languages = sorted({
        ex.get("language") for ex in fixture_records if ex.get("language")
    })
    if languages:
        lines.extend(_per_language_semeval_section(
            detectors, fixture_records, detector_records,
            label_set_per_det=fair_set,
            languages=languages,
        ))

    # Disagreement appendix (uniform — uses dataset's full vocab)
    if "opf" in detectors and "skyflow" in detectors:
        lines.append("## Disagreements — all categories (capped at {} per bucket)".format(max_disagreements))
        lines.append("")
        opf_only, sky_only = [], []
        for ex in fixture_records:
            opf = detector_records["opf"].get(ex["id"], {})
            sky = detector_records["skyflow"].get(ex["id"], {})
            opf_spans = {(s["label"], s["start"], s["end"]) for s in opf.get("spans") or []}
            sky_spans = {(s["label"], s["start"], s["end"]) for s in sky.get("spans") or []}
            if opf_spans - sky_spans:
                opf_only.append((ex, opf_spans - sky_spans))
            if sky_spans - opf_spans:
                sky_only.append((ex, sky_spans - opf_spans))
        for title, bucket in (("OPF caught, Skyflow missed", opf_only), ("Skyflow caught, OPF missed", sky_only)):
            lines.append(f"### {title} ({len(bucket)} total)")
            lines.append("")
            for ex, diff in bucket[:max_disagreements]:
                lines.append(f"- `{ex['id']}` — {sorted(diff)}")
                lines.append(f"  > {ex['text'][:160]!r}")
            lines.append("")

    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--fixtures", required=True, type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument(
        "--canonical-labels",
        default="",
        help=(
            "Optional comma-separated canonical label override. When set, both"
            " the fair and raw views are forced to this explicit set (one-"
            " category drilldowns, e.g. 'DATE'). Default empty = per-detector"
            " fair view + dataset-wide raw view derived from manifest."
            f" Available: {', '.join(CANONICAL_LABELS)}."
        ),
    )
    args = ap.parse_args()
    canonicals = (
        tuple(x.strip() for x in args.canonical_labels.split(",") if x.strip())
        if args.canonical_labels
        else None
    )
    md = build_report(args.run, args.fixtures, canonical_labels=canonicals)
    out = args.out or (args.run / "report.md")
    out.write_text(md)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
