"""Read a runner output dir + the source fixtures, emit report.md.

Restricted scoring: predictions and gold are both filtered to the canonical
labels OPF natively supports (default; configurable via --canonical-labels).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .metrics import PRF, latency_summary, score
from .nervaluate_metrics import SemEvalResult
from .nervaluate_metrics import score as semeval_score
from .taxonomy import OPF_CANONICAL_LABELS


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


def _compute_scores(
    detectors: list[str],
    fixture_records: list[dict],
    detector_records: dict[str, dict[str, dict]],
    *,
    allow: set[str] | None,
    language_filter: str | None = None,
) -> tuple[dict[str, object], dict[str, dict[str, float]], dict[str, int]]:
    """Returns (greedy_score_by_det, latency_by_det, error_count_by_det)."""
    score_by_det: dict[str, object] = {}
    latency_by_det: dict[str, dict[str, float]] = {}
    errors_by_det: dict[str, int] = {}
    for det in detectors:
        pairs, latencies, errors = _build_pairs(
            det, fixture_records, detector_records,
            allow=allow, language_filter=language_filter,
        )
        score_by_det[det] = score(pairs)
        latency_by_det[det] = latency_summary(latencies)
        errors_by_det[det] = errors
    return score_by_det, latency_by_det, errors_by_det


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


def _semeval_section(
    detectors: list[str],
    fixture_records: list[dict],
    detector_records: dict[str, dict[str, dict]],
    *,
    allow: set[str],
) -> list[str]:
    """SemEval 2013 9.1 scoring via nervaluate.

    Schema F1 across all four schemas, plus the per-detector COR/INC/PAR/MIS/SPU
    breakdown under the Strict schema (the most diagnostic).
    """
    tags = sorted(allow)
    results: dict[str, SemEvalResult] = {}
    for det in detectors:
        pairs, _, _ = _build_pairs(det, fixture_records, detector_records, allow=allow)
        results[det] = semeval_score(detector=det, pairs=pairs, tags=tags)

    lines: list[str] = [
        "## SemEval (nervaluate) — restricted to OPF categories",
        "",
        "Schema definitions: **Strict** = exact boundary + label. **Exact** = exact"
        " boundary, ignore label. **Partial** = any overlap, ignore label. **Type**"
        " = any overlap + matching label. Type and Partial give partial credit for"
        " boundary mismatches (e.g. OPF labelling `2040-06-02 00:00:00` as one"
        " DATE when gold splits it).",
        "",
        "### Headline F1 by schema",
        "",
        "| detector | strict | exact | partial | type |",
        "|---|---|---|---|---|",
    ]
    for det in detectors:
        r = results[det]
        lines.append(
            "| {det} | {strict:.3f} | {exact:.3f} | {partial:.3f} | {ent:.3f} |".format(
                det=det,
                strict=r.by_schema["strict"]["f1"],
                exact=r.by_schema["exact"]["f1"],
                partial=r.by_schema["partial"]["f1"],
                ent=r.by_schema["ent_type"]["f1"],
            )
        )
    lines.append("")

    # COR/INC/PAR/MIS/SPU under the Strict schema — the diagnostic table.
    lines.extend([
        "### Error decomposition (Strict schema)",
        "",
        "| detector | COR | INC | PAR | MIS | SPU |",
        "|---|---|---|---|---|---|",
    ])
    for det in detectors:
        m = results[det].by_schema["strict"]
        lines.append(
            "| {det} | {c} | {i} | {p} | {ms} | {sp} |".format(
                det=det,
                c=m["correct"], i=m["incorrect"], p=m["partial"],
                ms=m["missed"], sp=m["spurious"],
            )
        )
    lines.append("")
    return lines


def _per_language_semeval_section(
    detectors: list[str],
    fixture_records: list[dict],
    detector_records: dict[str, dict[str, dict]],
    *,
    allow: set[str],
    languages: list[str],
) -> list[str]:
    """Per-language SemEval Type-schema F1.

    Type schema (any overlap + matching label) is the most readable headline
    number per language — boundary tolerance keeps language-level results
    comparable across detectors with different granularity habits.
    """
    tags = sorted(allow)
    lines: list[str] = [
        "## Per-language (restricted to OPF categories, SemEval Type F1)",
        "",
        "Type schema = any overlap + matching label. See the SemEval headline"
        " section above for the full four-schema breakdown over the whole"
        " sample.",
        "",
        "| language | n | " + " | ".join(detectors) + " |",
        "|---|---|" + "|".join("---" for _ in detectors) + "|",
    ]
    for lang in languages:
        n = sum(1 for ex in fixture_records if ex.get("language") == lang)
        row = [lang, str(n)]
        for det in detectors:
            pairs, _, _ = _build_pairs(
                det, fixture_records, detector_records,
                allow=allow, language_filter=lang,
            )
            r = semeval_score(detector=det, pairs=pairs, tags=tags)
            row.append(f"{r.by_schema['ent_type']['f1']:.3f}")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    return lines


def build_report(
    run_dir: Path,
    fixtures: Path,
    *,
    canonical_labels: tuple[str, ...] | None = OPF_CANONICAL_LABELS,
    max_disagreements: int = 50,
) -> str:
    """canonical_labels: scope the SemEval, per-category, and per-language
    sections to these labels (defaults to OPF's 8). Pass an empty tuple to
    disable the scoped sections entirely.
    """
    manifest = json.loads((run_dir / "manifest.json").read_text())
    detectors: list[str] = manifest["detectors"]
    fixture_records = _load_jsonl(fixtures)

    detector_records: dict[str, dict[str, dict]] = {}
    for det in detectors:
        path = run_dir / f"raw_{det}.jsonl"
        detector_records[det] = _index(_load_jsonl(path))

    lines: list[str] = [
        f"# OPF vs Skyflow benchmark — {manifest['started_at']}",
        "",
        f"- fixtures: `{manifest['fixtures']}` ({manifest['n_examples']} examples)",
        f"- detectors: {', '.join(detectors)}",
        "",
    ]

    # Restricted to canonical_labels (default = OPF's 8). SemEval headline
    # leads; greedy per-category breakdown follows for the per-label P/R/F1
    # detail SemEval doesn't replicate as cleanly.
    allow = set(canonical_labels) if canonical_labels else None
    if allow:
        lines.extend(_semeval_section(
            detectors, fixture_records, detector_records, allow=allow,
        ))
        score_restricted, _, _ = _compute_scores(
            detectors, fixture_records, detector_records, allow=allow,
        )
        lines.extend(_per_label_section(
            "Per-category, restricted (partial overlap, IoU >= 0.5)",
            detectors,
            score_restricted,
        ))

    # Per-language slicing (only if the fixtures carry a language field)
    languages = sorted({
        ex.get("language") for ex in fixture_records if ex.get("language")
    })
    if languages and allow:
        lines.extend(_per_language_semeval_section(
            detectors, fixture_records, detector_records, allow=allow,
            languages=languages,
        ))

    # Disagreement appendix (still uses all categories — that's the interesting signal)
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
        default=",".join(OPF_CANONICAL_LABELS),
        help=(
            "Comma-separated canonical labels scoping the SemEval, per-category,"
            " and per-language sections. Default = OPF's 8 categories. Single"
            " labels (e.g. 'DATE') give a tight signal for one-category"
            " experiments. Empty string disables the scoped sections entirely."
            f" Available: {', '.join(OPF_CANONICAL_LABELS)}."
        ),
    )
    args = ap.parse_args()
    canonicals = (
        tuple(x.strip() for x in args.canonical_labels.split(",") if x.strip())
        if args.canonical_labels
        else ()
    )
    md = build_report(args.run, args.fixtures, canonical_labels=canonicals)
    out = args.out or (args.run / "report.md")
    out.write_text(md)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
