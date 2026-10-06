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

The fixtures' meta sidecar says where the gold came from (plan 13):

1. With `"gold": "none"` (your own text or files from `fixtures.from_texts`
   or `from_documents`) there is nothing to score against. The report says
   so and points to silver labels and reviews instead of printing precision
   and recall. Coverage and span counts are still shown.
2. With `"gold": "silver"` (from `silver.generate`) the report is titled
   "Silver-label report" and names under the title the LLMs that made the
   labels because the scores then measure agreement with those LLMs and not
   ground truth. Pass the silver file as `fixtures` (`--fixtures`) because
   the run's manifest records the unlabeled file the detectors ran on.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from pathlib import Path

from .io import GOLD_NONE, GOLD_SILVER, read_meta
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
    semeval_by_det: dict[str, "SemEvalResult | None"],
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
) -> tuple[list[str], dict[str, "SemEvalResult | None"], dict[str, set[str]]]:
    """Render a SemEval section. Returns (lines, per-detector results, per-
    detector label scopes) so callers can reuse them for follow-on tables."""
    results: dict[str, "SemEvalResult | None"] = {}
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
        lines.append(f"| {r['detector']} | {r['scored']}/{r['n']} | {r['errors']} | {r['missing']} |")
    lines.append("")
    return lines


def _gold_kind(meta: dict | None) -> str | None:
    """The meta's gold source: "none", "silver", or None for real gold."""
    return (meta or {}).get("gold")


def silver_callout(meta: dict) -> str:
    """The note under a silver-label report's title naming the labelers."""
    src = meta.get("gold_source") or {}
    labelers = src.get("labelers") or src.get("models") or ["unrecorded LLMs"]
    who = ", ".join(str(x) for x in labelers)
    if len(labelers) > 1:
        how = f", merged by {src.get('merge') or 'an unrecorded rule'}"
    else:
        how = ""
    return (
        f"> Gold spans here are LLM silver labels from {who}{how}."
        " Scores measure agreement with those labels, not ground truth."
    )


def _span_counts_section(run: Run) -> list[str]:
    lines = [
        "## Detected spans",
        "",
        "| detector | spans | records with spans |",
        "|---|---|---|",
    ]
    for det in run.detectors:
        rows = [r for r in run.predictions[det].values() if not r.get("error")]
        n_spans = sum(len(r.get("spans") or []) for r in rows)
        with_spans = sum(1 for r in rows if r.get("spans"))
        lines.append(f"| {det} | {n_spans} | {with_spans} |")
    lines.append("")
    return lines


def unlabeled_note(run: Run) -> str:
    """The short report for fixtures without gold: no scores, just pointers."""
    m = run.manifest
    lines = [
        f"# PII detector run — {m.get('started_at', '')}",
        "",
        f"- fixtures: `{m.get('fixtures')}` ({m.get('n_examples', len(run.fixtures))} examples)",
        f"- detectors: {', '.join(run.detectors) or '—'}",
        "",
        (
            "> **No gold labels, so no precision or recall.** These fixtures were written from"
            " your own text or files and have no gold spans to score against. Estimate the"
            " scores another way (notebook 07):"
        ),
        ">",
        (
            "> 1. **Silver labels**: `silver.generate(fixtures, labelers, out)` has LLMs annotate"
            " the records; then rebuild this report with `fixtures=out` (CLI: `--fixtures`)."
        ),
        (
            "> 2. **LLM review**: `opf_eval.review.llm.review_run(run_dir, reviewer)` judges each"
            " detected span and looks for missed PII."
        ),
        (
            "> 3. **Classifier review**: `opf_eval.review.classifier.review_run(run_dir, backend)`"
            " asks a local decision model about each span and segment."
        ),
        "",
    ]
    lines.extend(_span_counts_section(run))
    lines.extend(_coverage_section(run))
    return "\n".join(lines)


def silver_failed_note(run: Run, fx_path: Path, meta: dict) -> str:
    """The short report for silver fixtures where no record got labels.

    Every labeler failed on every record (a bad key, an outage, a model
    that refuses), so there is nothing to score. Printing F1 0.000 would
    read like a real result.
    """
    m = run.manifest
    n_errors = meta.get("n_errors") or {}
    files = meta.get("labeler_files") or {}
    lines = [
        f"# Silver-label report — {m.get('started_at', '')}",
        "",
        silver_callout(meta),
        "",
        f"- fixtures: `{fx_path}` ({len(meta.get('error_ids') or [])} records)",
        f"- detectors: {', '.join(run.detectors) or '—'}",
        "",
        (
            "> **No silver labels, so no scores.** Every labeler failed on every record, so"
            " no record has silver gold spans to score against. Each labeler's file holds the"
            " error for each record. Fix the cause and run `silver.generate` again."
        ),
        "",
    ]
    if n_errors:
        lines.extend(["| labeler | errors | file |", "|---|---|---|"])
        for name, n in n_errors.items():
            lines.append(f"| {name} | {n} | `{files.get(name, '—')}` |")
        lines.append("")
    lines.extend(_span_counts_section(run))
    return "\n".join(lines)


def _detector_selection(detectors: Iterable[str] | None) -> list[str] | None:
    """The requested detector names in order without duplicates, or None for all.

    A bare string counts as one name. An empty selection raises `ValueError`
    because a report with no detectors has nothing to show.
    """
    if detectors is None:
        return None
    names = [detectors] if isinstance(detectors, str) else list(detectors)
    names = list(dict.fromkeys(n.strip() for n in names if n.strip()))
    if not names:
        raise ValueError("detectors is empty; pass None to report every detector")
    return names


def build_report(
    run_dir: Path,
    fixtures: Path | None = None,
    *,
    canonical_labels: tuple[str, ...] | None = None,
    level: str = "coarse",
    detectors: Iterable[str] | None = None,
) -> str:
    """Render the Markdown report for a run directory.

    fixtures: defaults to the path recorded in the manifest.
    canonical_labels: when given, force both fair and raw views to this
        explicit label set (degenerates the two views into one).
    level: `coarse` (default) or `fine`.
    detectors: when given, limit every section to these detectors. The
        report keeps the run's detector order. A name the run has no
        predictions for raises `ValueError`.

    Fixtures without gold (meta `"gold": "none"`) get a short note instead
    of scores (no exception). Silver fixtures (`"gold": "silver"`) get a
    "Silver-label report" title, a callout naming the labelers, and are
    scored on the labels the labelers were asked for. Records that every
    labeler failed on (meta `error_ids`) are left out of silver scores; when
    that leaves no record, a short note replaces the scores.
    """
    check_level(level)
    run_dir = Path(run_dir)
    fx_path = Path(fixtures) if fixtures else Path(
        json.loads((run_dir / "manifest.json").read_text())["fixtures"]
    )
    fx_meta = read_meta(fx_path)
    gold = _gold_kind(fx_meta)
    wanted = _detector_selection(detectors)
    run = Run.load(run_dir, fx_path, detectors=wanted)
    if wanted is not None:
        unknown = [d for d in wanted if d not in run.detectors]
        if unknown:
            available = Run.load(run_dir, fx_path).detectors
            raise ValueError(
                f"no predictions in {run_dir} for detector(s) {', '.join(unknown)};"
                f" the run has {', '.join(available) or 'none'}"
            )
    if gold == GOLD_NONE:
        return unlabeled_note(run)
    if gold == GOLD_SILVER and fx_meta.get("labels"):
        # The silver labels annotate exactly what the labelers were asked
        # for, which may be narrower than the targets the detectors ran on.
        run.manifest = {**run.manifest, "labels": list(fx_meta["labels"])}
    # Records every labeler failed on have empty silver gold. Scoring them
    # would charge detectors a false positive for every span they found.
    unlabelled_ids = set(fx_meta.get("error_ids") or []) if gold == GOLD_SILVER else set()
    if unlabelled_ids:
        run.fixtures = [r for r in run.fixtures if r["id"] not in unlabelled_ids]
        if not run.fixtures:
            return silver_failed_note(run, fx_path, fx_meta)
    detectors = run.detectors
    manifest = run.manifest
    ds_labels = run.dataset_labels(level)
    labels = tuple(canonical_labels) if canonical_labels else None

    dataset = manifest.get("dataset") or "(unregistered)"
    vocab = manifest.get("vocab_key")
    title = "Silver-label report" if gold == GOLD_SILVER else "PII detector benchmark"
    lines: list[str] = [f"# {title} — {manifest['started_at']}", ""]
    if gold == GOLD_SILVER:
        lines.extend([silver_callout(fx_meta), ""])
    lines.extend([
        f"- dataset: `{dataset}`" + (f" (vocab `{vocab}`)" if vocab else ""),
        (
            f"- fixtures: `{fx_path if gold == GOLD_SILVER else manifest['fixtures']}`"
            f" ({manifest['n_examples']} examples)"
        ),
        f"- detectors: {', '.join(detectors)}",
        f"- label level: `{level}`",
        *(
            [f"- left out: {len(unlabelled_ids)} records that no labeler could label"]
            if unlabelled_ids else []
        ),
        "",
    ])

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
    ap.add_argument(
        "--detectors",
        default="",
        help=(
            "Optional comma-separated detector names. When set, the report"
            " covers only these detectors. Default empty = every detector in the run."
        ),
    )
    args = ap.parse_args()
    detectors = [x.strip() for x in args.detectors.split(",") if x.strip()] or None
    canonicals = (
        tuple(x.strip() for x in args.canonical_labels.split(",") if x.strip())
        if args.canonical_labels
        else None
    )
    md = build_report(
        args.run, args.fixtures,
        canonical_labels=canonicals, level=args.level, detectors=detectors,
    )
    out = args.out or (args.run / "report.md")
    out.write_text(md)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
