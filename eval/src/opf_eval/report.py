"""Read a runner output dir + the source fixtures, emit report.md.

Two scoring views per detector:

- **Fair**  — each detector scored against `dataset ∩ detector_supports`.
  Apples-to-apples within each detector's claimed coverage.
- **Raw**   — each detector scored against the full set the dataset annotates.
  Labels a detector doesn't support count as misses; reflects real-world
  out-of-the-box coverage.

`level="coarse"` (default) scores the 15 top-level categories;
`level="fine"` keeps sub-types (GOV_ID vs BANK_ACCOUNT, GIVEN_NAME vs
FAMILY_NAME, ...) wherever both the detector and the dataset distinguish them.

The manifest carries the dataset and its annotated labels; the report derives
label sets from there. `--canonical-labels` overrides both views to a single
explicit set (use for one-category drilldowns).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .nervaluate_metrics import SemEvalResult
from .scoring import Run, build_pairs, coverage, scope_for, score_detector
from .taxonomy import ALL_LABELS, LEVELS, check_level


def _fmt_per_label(metrics: dict[str, float | int]) -> str:
    """Render one (P / R / F1, COR/INC/MIS/SPU) cell from a nervaluate per-tag
    schema dict."""
    return (
        f"{metrics['precision']:.3f} / {metrics['recall']:.3f} / {metrics['f1']:.3f}"
        f"  (cor={metrics['correct']} inc={metrics['incorrect']}"
        f" mis={metrics['missed']} spu={metrics['spurious']})"
    )


def _per_label_section(
    title: str,
    detectors: list[str],
    semeval_by_det: dict[str, SemEvalResult | None],
    labels: list[str],
    *,
    schema: str = "ent_type",
) -> list[str]:
    """Per-label P/R/F1 + COR/INC/MIS/SPU from nervaluate's per-tag results.
    Default schema = `ent_type` (Type — any overlap + matching label),
    consistent with the per-language headline."""
    lines: list[str] = [f"### {title}", ""]
    lines.append("| label | " + " | ".join(detectors) + " |")
    lines.append("|---|" + "|".join("---" for _ in detectors) + "|")
    for lbl in sorted(labels):
        row = [lbl]
        for det in detectors:
            r = semeval_by_det.get(det)
            if r is None or lbl not in r.by_label:
                row.append("—")
                continue
            row.append(_fmt_per_label(r.by_label[lbl][schema]))
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    return lines


def _semeval_view(
    run: Run,
    detectors: list[str],
    *,
    level: str,
    view: str,
    labels: tuple[str, ...] | None,
    title: str,
    description: str,
) -> tuple[list[str], dict[str, SemEvalResult | None], dict[str, set[str]]]:
    """Render a SemEval section. Returns (lines, per-detector results, per-
    detector label scopes) so callers can reuse them for follow-on tables."""
    results: dict[str, SemEvalResult | None] = {}
    scopes: dict[str, set[str]] = {}
    for det in detectors:
        scope = scope_for(run, det, level=level, view=view, labels=labels)
        scopes[det] = set(scope.labels)
        results[det] = score_detector(run, det, level=level, view=view, labels=labels)

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
        n = len(scopes[det])
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
        labels_ = scopes[det]
        lines.append(
            f"- **{det}** ({len(labels_)}): {', '.join(sorted(labels_)) if labels_ else '—'}"
        )
    lines.append("")
    return lines, results, scopes


def _per_language_semeval_section(
    run: Run,
    detectors: list[str],
    *,
    level: str,
    labels: tuple[str, ...] | None,
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
        n = sum(1 for ex in run.fixtures if ex.get("language") == lang)
        row = [lang, str(n)]
        for det in detectors:
            r = score_detector(run, det, level=level, view="fair", labels=labels, language=lang)
            row.append("—" if r is None else f"{r.by_schema['ent_type']['f1']:.3f}")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    return lines


def _coverage_section(run: Run) -> list[str]:
    rows = coverage(run)
    lines = [
        "## Coverage",
        "",
        "Fixtures each detector was scored on. Errored records (the detector"
        " raised or the API call failed) and fixtures with no prediction row"
        " are excluded from every score above.",
        "",
        "| detector | scored | errors | missing |",
        "|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['detector']} | {r['scored']}/{r['n']} | {r['errors']} | {r['missing']} |"
        )
    lines.append("")
    return lines


def build_report(
    run_dir: Path,
    fixtures: Path | None = None,
    *,
    canonical_labels: tuple[str, ...] | None = None,
    level: str = "coarse",
) -> str:
    """Render the Markdown report for a run directory.

    fixtures: defaults to the path recorded in the manifest.
    canonical_labels: when given, force both fair and raw views to this
        explicit label set (degenerates the two views into one).
    level: `coarse` (default) or `fine`.
    """
    check_level(level)
    run = Run.load(run_dir, fixtures)
    detectors = run.detectors
    manifest = run.manifest
    ds_labels = run.dataset_labels(level)
    labels = tuple(canonical_labels) if canonical_labels else None

    dataset = manifest.get("dataset") or "(unregistered)"
    vocab = manifest.get("vocab_key")
    lines: list[str] = [
        f"# PII detector benchmark — {manifest['started_at']}",
        "",
        f"- dataset: `{dataset}`" + (f" (vocab `{vocab}`)" if vocab else ""),
        f"- fixtures: `{manifest['fixtures']}` ({manifest['n_examples']} examples)",
        f"- detectors: {', '.join(detectors)}",
        f"- label level: `{level}`",
        "",
    ]

    fair_lines, _, _ = _semeval_view(
        run, detectors, level=level, view="fair", labels=labels,
        title="SemEval — Fair view (per-detector scope)",
        description=(
            "Each detector scored against the intersection of (dataset annotates,"
            " this detector supports). Apples-to-apples within each detector's"
            " claimed coverage. Schema definitions: **Strict** = exact boundary"
            " + label. **Exact** = boundary, ignore label. **Partial** = any"
            " overlap, ignore label. **Type** = any overlap + matching label."
        ),
    )
    lines.extend(fair_lines)

    # Raw view (full dataset vocabulary). Its per-tag results feed the
    # per-category breakdown below — the same scoring run.
    raw_lines, raw_results, raw_scopes = _semeval_view(
        run, detectors, level=level, view="raw", labels=labels,
        title="SemEval — Raw dataset view (full vocabulary)",
        description=(
            f"Every detector scored against the dataset's full annotated set"
            f" ({len(ds_labels) if labels is None else len(labels)} canonical labels)."
            f" Labels a detector doesn't support take zero recall here, so this"
            f" view reflects out-of-the-box coverage rather than fairness."
        ),
    )
    lines.extend(raw_lines)

    # At fine level each detector's raw scope can differ (categories it can't
    # split stay coarse), so the breakdown rows are the union.
    breakdown_labels = sorted(set().union(*raw_scopes.values())) if raw_scopes else []
    lines.extend(_per_label_section(
        "Per-category breakdown — raw view (SemEval Type schema)",
        detectors,
        raw_results,
        breakdown_labels,
    ))

    languages = sorted({ex.get("language") for ex in run.fixtures if ex.get("language")})
    if languages:
        lines.extend(_per_language_semeval_section(
            run, detectors, level=level, labels=labels, languages=languages,
        ))

    lines.extend(_coverage_section(run))
    return "\n".join(lines)


# Kept for callers that imported the old private helper.
_build_pairs = build_pairs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--fixtures", type=Path, default=None,
                    help="fixtures file (default: the one recorded in the manifest)")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--level", choices=LEVELS, default="coarse",
                    help="label granularity: coarse (15 categories) or fine (sub-types)")
    ap.add_argument(
        "--canonical-labels",
        default="",
        help=(
            "Optional comma-separated canonical label override. When set, both"
            " the fair and raw views are forced to this explicit set (one-"
            " category drilldowns, e.g. 'DATE'). Default empty = per-detector"
            " fair view + dataset-wide raw view derived from manifest."
            f" Available: {', '.join(ALL_LABELS)}."
        ),
    )
    args = ap.parse_args()
    canonicals = (
        tuple(x.strip() for x in args.canonical_labels.split(",") if x.strip())
        if args.canonical_labels
        else None
    )
    md = build_report(args.run, args.fixtures, canonical_labels=canonicals, level=args.level)
    out = args.out or (args.run / "report.md")
    out.write_text(md)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
