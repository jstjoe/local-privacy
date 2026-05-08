"""Read a runner output dir + the source fixtures, emit report.md.

Restricted scoring: predictions and gold are both filtered to the canonical
labels OPF natively supports (default; configurable via --canonical-labels).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .metrics import PRF, latency_summary, score
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


def _fmt_prf_compact(p: PRF) -> str:
    return f"{p.precision:.3f} / {p.recall:.3f} / {p.f1:.3f}"


def _filter_spans(spans: list[dict], allow: set[str] | None) -> list[dict]:
    if allow is None:
        return spans
    return [s for s in spans if s["label"] in allow]


def _merge_adjacent(spans: list[dict], gap_tolerance: int = 2) -> list[dict]:
    """Coalesce adjacent same-canonical-label spans into one.

    Two spans are merged when:
      - same canonical label
      - the second starts within `gap_tolerance` chars of where the first ended
        (allows for a comma, space, or "and" between them)
    Removes granularity bias before scoring.
    """
    if not spans:
        return spans
    sorted_spans = sorted(spans, key=lambda s: (s["start"], s["end"]))
    merged: list[dict] = [dict(sorted_spans[0])]
    for s in sorted_spans[1:]:
        prev = merged[-1]
        if s["label"] == prev["label"] and s["start"] <= prev["end"] + gap_tolerance:
            prev["end"] = max(prev["end"], s["end"])
            prev["text"] = (prev.get("text") or "") + (s.get("text") or "")
        else:
            merged.append(dict(s))
    return merged


def _compute_scores(
    detectors: list[str],
    fixture_records: list[dict],
    detector_records: dict[str, dict[str, dict]],
    *,
    allow: set[str] | None,
    merge_adjacent: bool = False,
    language_filter: str | None = None,
) -> tuple[dict[str, object], dict[str, dict[str, float]], dict[str, int]]:
    """Returns (score_by_det, latency_by_det, error_count_by_det)."""
    score_by_det: dict[str, object] = {}
    latency_by_det: dict[str, dict[str, float]] = {}
    errors_by_det: dict[str, int] = {}
    for det in detectors:
        pairs: list[tuple[list[dict], list[dict]]] = []
        latencies: list[float] = []
        errors = 0
        for ex in fixture_records:
            if language_filter and ex.get("language") != language_filter:
                continue
            rec = detector_records[det].get(ex["id"])
            if rec is None:
                continue
            if rec.get("error"):
                errors += 1
                continue
            pred = _filter_spans(rec["spans"], allow)
            gold = _filter_spans(ex["gold_spans"], allow)
            if merge_adjacent:
                pred = _merge_adjacent(pred)
                gold = _merge_adjacent(gold)
            pairs.append((pred, gold))
            latencies.append(rec["latency_ms"])
        score_by_det[det] = score(pairs)
        latency_by_det[det] = latency_summary(latencies)
        errors_by_det[det] = errors
    return score_by_det, latency_by_det, errors_by_det


def _scoring_section(
    title: str,
    detectors: list[str],
    fixture_records: list[dict],
    detector_records: dict[str, dict[str, dict]],
    *,
    allow: set[str] | None,
    merge_adjacent: bool = False,
    language_filter: str | None = None,
) -> tuple[list[str], dict[str, object]]:
    score_by_det, _, _ = _compute_scores(
        detectors,
        fixture_records,
        detector_records,
        allow=allow,
        merge_adjacent=merge_adjacent,
        language_filter=language_filter,
    )
    lines: list[str] = [f"## {title}", ""]
    lines.append("| detector | exact P/R/F1 | partial P/R/F1 |")
    lines.append("|---|---|---|")
    for det in detectors:
        rep = score_by_det[det]
        lines.append(
            f"| {det} | {_fmt_prf_compact(rep.overall_exact)} | {_fmt_prf_compact(rep.overall_partial)} |"  # type: ignore[union-attr]
        )
    lines.append("")
    return lines, score_by_det


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


def build_report(
    run_dir: Path,
    fixtures: Path,
    *,
    canonical_labels: tuple[str, ...] | None = OPF_CANONICAL_LABELS,
    max_disagreements: int = 50,
) -> str:
    """canonical_labels: when not None, also emit a second 'restricted' headline
    + per-category section scored against just these labels (defaults to OPF's 8).
    Pass an empty tuple to disable the restricted view.
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

    # Restricted to canonical_labels (default = OPF's 8)
    allow = set(canonical_labels) if canonical_labels else None
    if allow:
        h_restricted, score_restricted = _scoring_section(
            f"Headline — restricted to {{{', '.join(sorted(allow))}}}",
            detectors,
            fixture_records,
            detector_records,
            allow=allow,
        )
        lines.extend(h_restricted)
        lines.extend(_per_label_section(
            "Per-category, restricted (partial overlap, IoU >= 0.5)",
            detectors,
            score_restricted,
        ))

        h_merged, _ = _scoring_section(
            "Headline — restricted + merged-adjacent (granularity-neutral)",
            detectors,
            fixture_records,
            detector_records,
            allow=allow,
            merge_adjacent=True,
        )
        lines.extend(h_merged)

    # Per-language slicing (only if the fixtures carry a language field)
    languages = sorted({
        ex.get("language") for ex in fixture_records if ex.get("language")
    })
    if languages and allow:
        lines.append("## Per-language (restricted to OPF categories, partial F1)")
        lines.append("")
        lines.append("| language | n | " + " | ".join(detectors) + " |")
        lines.append("|---|---|" + "|".join("---" for _ in detectors) + "|")
        for lang in languages:
            n = sum(1 for ex in fixture_records if ex.get("language") == lang)
            by_det, _, _ = _compute_scores(
                detectors,
                fixture_records,
                detector_records,
                allow=allow,
                language_filter=lang,
            )
            row = [lang, str(n)]
            for det in detectors:
                p = by_det[det].overall_partial  # type: ignore[union-attr]
                row.append(f"{p.f1:.3f}")
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

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
            "Comma-separated canonical labels for the restricted, merged-adjacent, "
            "and per-language sections. Default = OPF's 8 categories. Single labels "
            "(e.g. 'DATE') give a tight signal for one-category experiments. "
            "Empty string disables the restricted view entirely. "
            f"Available: {', '.join(OPF_CANONICAL_LABELS)}."
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
