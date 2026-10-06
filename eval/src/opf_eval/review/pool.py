"""Pool every detector's spans so each unique span is reviewed once.

Reviews (LLM or classifier) judge spans, not detectors. Two detectors that
find the same span at the same place with the same coarse label share one
judgement, and the verdict is credited to both. That halves the cost of a
review when detectors agree and keeps their estimates consistent with each
other, which is the pooling method used in information-retrieval evaluation.

    from opf_eval import scoring
    from opf_eval.review.pool import pooled_spans

    run = scoring.Run.load(run_dir)
    pooled = pooled_spans(run)
    pooled["text-0001"]
    # [{"start": 8, "end": 28, "label": "EMAIL", "fine_label": "EMAIL",
    #   "text": "jane.doe@example.com", "detectors": ["gliner", "presidio"],
    #   "canonical": True}, ...]

The helpers below the main function are shared by `review.llm` and
`review.classifier`: which labels a review targets, which records a detector
produced usable output for, and a progress bar that degrades to nothing.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import TypeVar

from .. import taxonomy
from ..io import iter_jsonl, read_meta
from ..scoring import Run

T = TypeVar("T")


def pooled_spans(
    run: Run,
    detectors: Iterable[str] | None = None,
    *,
    labels: Iterable[str] | None = None,
) -> dict[str, list[dict]]:
    """Unique spans per record across `detectors` (default: every detector in the run).

    A span is identified by `(start, end, coarse label)`. Each pooled span is
    `{"start", "end", "label", "fine_label", "text", "detectors", "canonical"}`:

    1. `label` is the coarse canonical label and `fine_label` is the most
       common fine label the detectors gave it (ties go to the first detector).
    2. `text` is cut from the fixture text so it is exact even when a
       detector reported a normalised span text.
    3. `detectors` lists every detector that produced the span in sorted order.
    4. `canonical` is False for labels outside the taxonomy (a detector's raw
       label that no vocabulary maps). Such spans are kept so they can be
       shown, but reviews neither judge nor score them.

    Rows a detector errored on are ignored. When `labels` is given, canonical
    spans whose coarse label is outside the labels' coarse categories are
    dropped because a review only judges the categories it targets.
    Every fixture id gets an entry, so a record with no spans maps to `[]`.
    Spans are sorted by start, then end, then label.
    """
    dets = _detectors(run, detectors)
    scope = coarse_scope(labels) if labels is not None else None
    out: dict[str, list[dict]] = {}
    for ex in run.fixtures:
        text = ex.get("text") or ""
        found: dict[tuple[int, int, str], dict] = {}
        fine_votes: dict[tuple[int, int, str], Counter] = {}
        for det in dets:
            row = run.predictions.get(det, {}).get(ex["id"])
            if row is None or row.get("error"):
                continue
            seen_here: set[tuple[int, int, str]] = set()
            for span in row.get("spans") or []:
                start, end = int(span["start"]), int(span["end"])
                if not 0 <= start < end <= len(text):
                    continue  # empty or out-of-range spans cannot be reviewed
                coarse = taxonomy.parent(span["label"])
                canonical = taxonomy.is_known(coarse)
                if canonical and scope is not None and coarse not in scope:
                    continue
                key = (start, end, coarse)
                fine = span.get("fine_label") or span["label"]
                if not taxonomy.is_known(fine) or taxonomy.parent(fine) != coarse:
                    fine = coarse
                entry = found.get(key)
                if entry is None:
                    entry = found[key] = {
                        "start": start,
                        "end": end,
                        "label": coarse,
                        "fine_label": fine,
                        "text": text[start:end],
                        "detectors": [],
                        "canonical": canonical,
                    }
                    fine_votes[key] = Counter()
                if key in seen_here:
                    continue  # a detector repeating a span still counts once
                seen_here.add(key)
                entry["detectors"].append(det)
                fine_votes[key][fine] += 1
        for key, entry in found.items():
            votes = fine_votes[key]
            best = max(votes.values())
            # Counter keeps insertion order, so ties go to the first detector.
            entry["fine_label"] = next(lbl for lbl, n in votes.items() if n == best)
            entry["detectors"] = sorted(entry["detectors"])
        out[ex["id"]] = sorted(found.values(), key=lambda s: (s["start"], s["end"], s["label"]))
    return out


# ------------------------------------------------------------ shared helpers


def _detectors(run: Run, detectors: Iterable[str] | None) -> list[str]:
    if detectors is None:
        return list(run.detectors)
    dets = list(detectors)
    unknown = [d for d in dets if d not in run.predictions]
    if unknown:
        raise ValueError(
            f"no raw results for detector(s) {unknown} in {run.dir}; "
            f"available: {run.detectors}"
        )
    return dets


def coarse_scope(labels: Iterable[str]) -> set[str]:
    """Coarse categories covered by `labels` (fine labels count for their parent)."""
    return {taxonomy.parent(lbl) for lbl in labels if taxonomy.is_known(lbl)}


def target_labels(run: Run, labels: Iterable[str] | None = None, fixtures: str | Path | None = None) -> list[str]:
    """The labels a review targets.

    Order: explicit `labels`; the fixtures sidecar's `labels` (what
    `from_texts` / `from_documents` / silver labels recorded); the run
    manifest's `labels`; every canonical label. Unknown labels are an error
    because they could never be judged or scored.
    """
    if labels is not None:
        out = sorted(set(labels))
    else:
        fx = fixtures or run.manifest.get("fixtures")
        meta = read_meta(fx) if fx else None
        if meta and meta.get("labels"):
            out = sorted(set(meta["labels"]))
        elif run.manifest.get("labels"):
            out = sorted(set(run.manifest["labels"]))
        else:
            out = sorted(taxonomy.ALL_LABELS)
    unknown = [lbl for lbl in out if not taxonomy.is_known(lbl)]
    if unknown:
        raise ValueError(f"unknown label(s) {unknown}; use canonical labels such as {', '.join(taxonomy.COARSE_LABELS[:5])}")
    if not out:
        raise ValueError("labels must not be empty")
    return out


def coarse_labels(labels: Iterable[str]) -> list[str]:
    """Coarse categories of `labels` in taxonomy order (the order reports use)."""
    scope = coarse_scope(labels)
    return [lbl for lbl in taxonomy.COARSE_LABELS if lbl in scope]


def ok_ids(run_dir: str | Path, detector: str) -> set[str]:
    """Record ids `detector` produced a non-error row for, read from its raw file."""
    path = Path(run_dir) / f"raw_{detector}.jsonl"
    if not path.exists():
        return set()
    return {r["id"] for r in iter_jsonl(path) if not r.get("error")}


def load_run(run_dir: str | Path, fixtures: str | Path | None = None, detectors: Iterable[str] | None = None) -> Run:
    """`scoring.Run.load` with a clear error for a missing run directory."""
    run_dir = Path(run_dir)
    if not (run_dir / "manifest.json").exists():
        raise FileNotFoundError(f"{run_dir} has no manifest.json; run the detectors first (nb.ensure_run)")
    run = Run.load(run_dir, fixtures, detectors=detectors)
    if detectors is not None:
        _detectors(run, detectors)  # raise for a detector without results
    return run


def resolve_file(run_dir: str | Path, review_file: str | Path) -> Path:
    """A review file given by path, or by name inside `run_dir`."""
    p = Path(review_file)
    if p.exists():
        return p
    q = Path(run_dir) / p.name
    if q.exists():
        return q
    raise FileNotFoundError(f"review file {review_file} not found (also looked in {run_dir})")


def review_meta_path(review_file: str | Path) -> Path:
    """`review_x.jsonl` -> `review_x.meta.json`."""
    p = Path(review_file)
    return p.with_name(p.stem + ".meta.json")


_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


def safe_name(value: str) -> str:
    """A file-name-safe version of a provider or model id (`openai/gpt-oss` -> `openai-gpt-oss`)."""
    return _UNSAFE.sub("-", value).strip("-") or "x"


def context(text: str, start: int, end: int, width: int = 60) -> str:
    """`text` around [start, end) with the span in ⟦ ⟧ and ellipses where it was cut."""
    lo, hi = max(0, start - width), min(len(text), end + width)
    return (
        ("…" if lo else "")
        + text[lo:start]
        + "⟦" + text[start:end] + "⟧"
        + text[end:hi]
        + ("…" if hi < len(text) else "")
    )


def progress(items: Sequence[T] | Iterable[T], *, total: int | None = None, desc: str = "", enabled: bool = True) -> Iterator[T]:
    """Wrap `items` in a tqdm bar when tqdm is installed and `enabled`; else pass through."""
    if not enabled:
        yield from items
        return
    try:
        from tqdm.auto import tqdm
    except ImportError:  # tqdm is optional; progress is a nicety
        yield from items
        return
    yield from tqdm(items, total=total, desc=desc, leave=False)


def ratio(num: float, den: float) -> float | None:
    """num / den, or None when there is nothing to divide by."""
    return num / den if den else None


def f1(p: float | None, r: float | None) -> float | None:
    if p is None or r is None:
        return None
    return 2 * p * r / (p + r) if p + r else 0.0
