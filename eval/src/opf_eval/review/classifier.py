"""Classifier review: typed yes/no and choice questions about spans and text (plan 13, Part 6).

A decision model such as Cloudflare's Clef-flash answers typed questions
with a probability per option instead of free text. This module asks two
kinds of question and turns the answers into per-detector estimates.

1. **Per span** (once per unique pooled span, see `review.pool`): "is the
   text inside ⟦ ⟧ personal data?", "is <label> the right type for it?" and
   "which type fits best?". A span counts as correct at threshold `t` when
   both yes/no probabilities reach `t`. That gives a precision estimate.
2. **Per segment** (per detector): each sentence is shown with that
   detector's spans replaced by `[LABEL]` placeholders, and the model is
   asked whether personal data is still left. The share of segments where
   it says yes is the detector's **residual-PII rate**. A segment is one
   sentence of one line (see `split_segments`), so the rate is per sentence.
   Classifiers return no spans, so this is a recall signal per segment and
   not per span.

Requests use the Jev / SystemOne request format, so any backend that speaks
it fits behind `DecisionBackend`:

    {"model": "clef-flash",
     "state": {...},
     "questions": {"is_pii": {"type": "noul", "instructions": "..."},
                   "label":  {"type": "choice", "instructions": "...", "criteria": {...}}}}

and each answer is `{"answers": {"is_pii": {"type": "noul", "noul": 0.93},
"label": {"type": "choice", "choice": "EMAIL", "confidence": 0.88,
"probabilities": {...}}}}`.

    from opf_eval.review import classifier, clef

    backend = clef.ClefBackend()                       # or classifier.StubBackend()
    path = classifier.review_run(run_dir, backend)
    classifier.summarize(run_dir, path)                # precision + residual rate
    classifier.threshold_sweep(run_dir, path, gold_fixtures)   # calibration on real gold

Questions use the coarse categories of the target labels: 15 options at
most keep each request short, and the review scores at the coarse level.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol, runtime_checkable

from .. import taxonomy
from ..io import iter_jsonl, read_jsonl, write_jsonl
from .pool import (
    coarse_labels,
    coarse_scope,
    load_run,
    pooled_spans,
    progress,
    ratio,
    resolve_file,
    review_meta_path,
    safe_name,
    target_labels,
)

DEFAULT_MODEL = "clef-flash"
NOT_PII = "NOT_PII"
NONE = "NONE"
OPEN, CLOSE = "⟦", "⟧"
DEFAULT_THRESHOLDS = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8)

IS_PII_Q = f"Is the text inside {OPEN} {CLOSE} personal or sensitive data?"
LABEL_CHOICE_Q = "Which type fits best?"
RESIDUAL_Q = (
    "Does this text still contain personal or sensitive data that is not already "
    "replaced by a [PLACEHOLDER]?"
)
RESIDUAL_TYPE_Q = "Which type of personal or sensitive data is still in the text?"


@runtime_checkable
class DecisionBackend(Protocol):
    """Answers SystemOne-format requests, one response per request, in order."""

    name: str
    remote: bool

    def answer(self, requests: list[dict]) -> list[dict]: ...


# ------------------------------------------------------------ request builders


def _criteria(labels: Iterable[str], extra: tuple[str, str]) -> dict[str, str]:
    out = {lbl: taxonomy.LABELS[lbl].description for lbl in coarse_labels(labels)}
    out[extra[0]] = extra[1]
    return out


def span_request(
    text: str,
    span: Mapping,
    labels: Iterable[str],
    *,
    window: int = 300,
    model: str = DEFAULT_MODEL,
) -> dict:
    """SystemOne request with three questions about one span.

    The state holds `window` characters either side of the span with the
    span wrapped in ⟦ ⟧ (ellipses mark a cut), the span text and the proposed
    coarse label. Questions: `is_pii` (noul), `label_ok` (noul, names the
    label and its description) and `label` (choice over the coarse target
    labels plus NOT_PII).
    """
    start, end = int(span["start"]), int(span["end"])
    label = taxonomy.parent(span["label"])
    lo, hi = max(0, start - window), min(len(text), end + window)
    ctx = (
        ("…" if lo else "")
        + text[lo:start] + OPEN + text[start:end] + CLOSE + text[end:hi]
        + ("…" if hi < len(text) else "")
    )
    info = taxonomy.LABELS.get(label)
    description = info.description if info else label
    return {
        "model": model,
        "state": {"context": ctx, "span": text[start:end], "proposed_label": label},
        "questions": {
            "is_pii": {"type": "noul", "instructions": IS_PII_Q},
            "label_ok": {
                "type": "noul",
                "instructions": f"Is {label} ({description}) the right type for it?",
            },
            "label": {
                "type": "choice",
                "instructions": LABEL_CHOICE_Q,
                "criteria": _criteria(labels, (NOT_PII, "Not personal data")),
            },
        },
    }


_SENTENCE_END = re.compile(r"[.!?;](?=\s)")
# Where `split_segments` ends a sentence: terminal punctuation, optional
# closing quotes or brackets, then whitespace. A semicolon is not a sentence
# end here because it often separates fields of one record ("Jane; 555-1234").
_SPLIT_END = re.compile(r"[.!?]+[\"')\]’”]*(?=\s)")
# Words that end in a full stop without ending the sentence ("Dr. Jane Doe").
_ABBREVIATIONS = frozenset({
    "mr", "mrs", "ms", "mx", "dr", "prof", "rev", "hon", "st", "jr", "sr", "mt", "ft",
    "no", "nr", "vs", "etc", "inc", "ltd", "co", "corp", "ave", "rd", "blvd", "apt",
    "dept", "approx", "fig", "e.g", "i.e", "a.m", "p.m", "u.s",
})


def _inside(pos: int, spans: Sequence[tuple[int, int]]) -> bool:
    return any(s < pos < e for s, e in spans)


def _merge(spans: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    """Sorted, disjoint intervals: overlapping spans become their union.

    Spans that only touch (one ends where the next starts) stay apart
    because a cut between them is outside both.
    """
    out: list[list[int]] = []
    for s, e in sorted(spans):
        if out and s < out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [(s, e) for s, e in out]


def _is_abbreviation(text: str, stop: int) -> bool:
    """True when the full stop at `stop` ends an abbreviation or an initial ("J.")."""
    if text[stop] != ".":
        return False
    i = stop
    while i > 0 and (text[i - 1].isalnum() or text[i - 1] == "."):
        i -= 1
    word = text[i:stop].lower()
    return word in _ABBREVIATIONS or (len(word) == 1 and word.isalpha())


def _sentences(
    text: str, a: int, b: int, avoid: Sequence[tuple[int, int]]
) -> list[tuple[int, int]]:
    """Split `text[a:b]` after every sentence end outside `avoid` that is not an abbreviation."""
    out, start = [], a
    for m in _SPLIT_END.finditer(text, a, b):
        cut = m.end()
        if cut >= b or _inside(cut, avoid) or _is_abbreviation(text, m.start()):
            continue
        out.append((start, cut))
        start = cut
    out.append((start, b))
    return out


def _trim(text: str, a: int, b: int) -> tuple[int, int]:
    while a < b and text[a].isspace():
        a += 1
    while b > a and text[b - 1].isspace():
        b -= 1
    return a, b


def _cut(text: str, pos: int, end: int, max_chars: int, avoid: Sequence[tuple[int, int]]) -> int:
    """Where to end a piece that starts at `pos` and is longer than `max_chars`.

    Prefer the last sentence end, then the last whitespace, within the limit
    and outside every span. A span longer than the limit is never cut.
    `avoid` must be sorted and disjoint (see `_merge`): then moving a hard cut
    to the edge of the span it falls in can never land inside another span.
    """
    limit = pos + max_chars
    best = None
    # One character past the limit, so the lookahead can see the space after
    # a sentence that ends exactly at the limit.
    for m in _SENTENCE_END.finditer(text, pos, min(limit + 1, end)):
        if m.end() <= limit and not _inside(m.end(), avoid):
            best = m.end()
    if best is None:
        for i in range(limit, pos, -1):
            if text[i - 1].isspace() and not _inside(i, avoid):
                best = i
                break
    if best is None or best <= pos:
        best = limit
        for s, e in avoid:
            if s < best < e:
                best = s if s > pos else e
        best = min(max(best, pos + 1), end)
    return best


def split_segments(
    text: str,
    *,
    max_chars: int = 600,
    avoid: Iterable[Mapping] | Iterable[tuple[int, int]] = (),
) -> list[tuple[int, int]]:
    """Split `text` into sentences of at most `max_chars` characters.

    Every line is split after each sentence end (`.`, `!` or `?` followed by
    whitespace). A full stop after a common abbreviation such as "Dr." or
    after a single-letter initial does not end a sentence. A sentence longer
    than `max_chars` is cut at its last clause end or space within the limit,
    or hard at the limit when it has none.

    Segments are trimmed of surrounding whitespace, and segments without a
    letter or digit are left out. No cut falls inside a span in `avoid`
    (dicts with start/end, or pairs): a line break or sentence end inside a
    span joins the two pieces, and a span longer than `max_chars` stays
    whole. Overlapping spans are treated as one span.
    """
    if max_chars < 1:
        raise ValueError("max_chars must be positive")
    spans = _merge(
        (int(s["start"]), int(s["end"])) if isinstance(s, Mapping) else (int(s[0]), int(s[1]))
        for s in avoid
    )
    breaks = [m.start() for m in re.finditer("\n", text) if not _inside(m.start(), spans)]
    blocks, prev = [], 0
    for b in breaks:
        blocks.append((prev, b))
        prev = b + 1
    blocks.append((prev, len(text)))

    out: list[tuple[int, int]] = []
    for block_a, block_b in blocks:
        for a, b in _sentences(text, *_trim(text, block_a, block_b), spans):
            a, b = _trim(text, a, b)
            while b - a > max_chars:
                c = _cut(text, a, b, max_chars, spans)
                out.append(_trim(text, a, c))
                a, _ = _trim(text, c, b)
            if a < b:
                out.append((a, b))
    return [(a, b) for a, b in out if a < b and any(ch.isalnum() for ch in text[a:b])]


def mask(text: str, start: int, end: int, spans: Iterable[Mapping]) -> str:
    """`text[start:end]` with every span replaced by `[COARSE_LABEL]`.

    Spans are clipped to the segment. Where spans overlap, the first one
    (by start) wins and the rest of the overlap is swallowed.
    """
    pieces: list[str] = []
    cursor = start
    for s in sorted(spans, key=lambda s: (int(s["start"]), -int(s["end"]))):
        s_start, s_end = max(int(s["start"]), start), min(int(s["end"]), end)
        if s_end <= s_start or s_end <= cursor:
            continue
        if s_start >= cursor:
            pieces.append(text[cursor:s_start])
            pieces.append(f"[{taxonomy.parent(s['label'])}]")
        cursor = s_end
    pieces.append(text[cursor:end])
    return "".join(pieces)


def segment_requests(
    text: str,
    spans: Iterable[Mapping],
    labels: Iterable[str],
    *,
    max_chars: int = 600,
    segments: Sequence[tuple[int, int]] | None = None,
    model: str = DEFAULT_MODEL,
) -> list[tuple[int, int, dict]]:
    """`(start, end, request)` per segment, with `spans` replaced by placeholders.

    Questions: `residual` (noul: is personal data still present outside the
    placeholders?) and `residual_type` (choice over the coarse target labels
    plus NONE). `segments` fixes the boundaries; `review_run` passes the same
    boundaries for every detector so their residual rates share a
    denominator. Without it the text is split with `split_segments`, avoiding
    `spans`.
    """
    spans = list(spans)
    bounds = (
        segments if segments is not None
        else split_segments(text, max_chars=max_chars, avoid=spans)
    )
    criteria = _criteria(labels, (NONE, "No personal data is left"))
    out = []
    for a, b in bounds:
        out.append((a, b, {
            "model": model,
            "state": {"text": mask(text, a, b, spans)},
            "questions": {
                "residual": {"type": "noul", "instructions": RESIDUAL_Q},
                "residual_type": {
                    "type": "choice",
                    "instructions": RESIDUAL_TYPE_Q,
                    "criteria": dict(criteria),
                },
            },
        }))
    return out


_PLACEHOLDER = re.compile(r"\[[A-Z_]+\]")


def _nothing_left(request: dict) -> bool:
    """True when masking left no letter or digit, so there is nothing to ask about."""
    rest = _PLACEHOLDER.sub("", request["state"]["text"])
    return not any(ch.isalnum() for ch in rest)


# -------------------------------------------------------------- answers


def noul_answer(p: float) -> dict:
    """A SystemOne yes/no answer with probability `p` of yes."""
    return {"type": "noul", "noul": round(float(p), 4)}


def choice_answer(probabilities: Mapping[str, float]) -> dict:
    """A SystemOne choice answer: the most likely option and its probability."""
    best = max(probabilities, key=probabilities.__getitem__)
    return {
        "type": "choice",
        "choice": best,
        "confidence": round(float(probabilities[best]), 4),
        "probabilities": {k: round(float(v), 4) for k, v in probabilities.items()},
    }


def _default_answers(request: dict) -> dict:
    """Constant answers: 0.5 for yes/no and a uniform distribution over choice options.

    These answers carry no information, but they are not neutral in
    `summarize`. A span or segment counts as yes when its probability is at
    least the threshold, so at the default threshold of 0.5 every question
    counts as yes. A detector then shows precision 1.0 and a residual-PII rate
    of 1.0 at the same time. Any constant answer gives a summary like this, so
    pass a responder when the numbers should mean something.
    """
    answers = {}
    for qid, q in request["questions"].items():
        if q["type"] == "noul":
            answers[qid] = noul_answer(0.5)
        else:
            options = (
                list(q["criteria"]) if isinstance(q["criteria"], Mapping)
                else [str(i) for i in range(len(q["criteria"]))]
            )
            answers[qid] = choice_answer({o: 1 / len(options) for o in options})
    return answers


class StubBackend:
    """A `DecisionBackend` driven by a Python callable, for tests and the notebook's dry run.

    `responder(request)` returns either a full response (`{"answers": {...}}`)
    or just the answers mapping. Without a responder every yes/no question
    gets 0.5 and every choice a uniform distribution. Those default answers
    count as yes at the default threshold of 0.5, so `summarize` then reports
    precision 1.0 and a residual-PII rate of 1.0 for every detector. They only
    show that the plumbing works. Requests are recorded in `.requests` for
    assertions.

        backend = StubBackend(lambda req: {
            "is_pii": noul_answer(0.9), "label_ok": noul_answer(0.8),
            "label": choice_answer({"EMAIL": 0.9, "NOT_PII": 0.1}),
        } if "questions" in req and "is_pii" in req["questions"] else {
            "residual": noul_answer(0.1), "residual_type": choice_answer({"NONE": 1.0}),
        })
    """

    def __init__(
        self,
        responder: Callable[[dict], dict] | None = None,
        *,
        name: str = "stub",
        remote: bool = False,
    ):
        self.responder = responder or _default_answers
        self.name = name
        self.remote = remote
        self.requests: list[dict] = []
        self.batches: list[int] = []

    def __repr__(self) -> str:
        return f"StubBackend(name={self.name!r})"

    def answer(self, requests: list[dict]) -> list[dict]:
        self.batches.append(len(requests))
        out = []
        for req in requests:
            self.requests.append(req)
            res = self.responder(req)
            if "answers" not in res:
                res = {"model": req.get("model", self.name), "answers": res}
            out.append(res)
        return out

    def close(self) -> None:
        """Nothing to free; present so callers can treat every backend alike."""


def _p_noul(response: dict, qid: str) -> float:
    a = response["answers"][qid]
    return float(a["noul"] if isinstance(a, Mapping) else a)


def _choice(response: dict, qid: str) -> tuple[str, float]:
    a = response["answers"][qid]
    return str(a["choice"]), float(a.get("confidence", 0.0))


def _answer_all(
    backend: DecisionBackend,
    requests: list[dict],
    *,
    batch_size: int,
    desc: str,
    show: bool,
) -> list[dict | Exception]:
    """One response (or the exception that stopped it) per request, in order.

    A failing batch is retried one request at a time so a single bad request
    (say, one that does not fit the model's context) costs only its own row.
    """
    batch_size = max(1, int(batch_size))
    out: list[dict | Exception] = []
    starts = range(0, len(requests), batch_size)
    for i in progress(starts, total=len(starts), desc=desc, enabled=show):
        chunk = requests[i:i + batch_size]
        try:
            res = backend.answer(chunk)
            if len(res) != len(chunk):
                raise RuntimeError(
                    f"{backend.name} returned {len(res)} answers for {len(chunk)} requests"
                )
            out.extend(res)
        except Exception as batch_exc:  # noqa: BLE001 — isolate the failing request
            if len(chunk) == 1:
                out.append(batch_exc)
                continue
            for req in chunk:
                try:
                    single = backend.answer([req])
                    if len(single) != 1:
                        raise RuntimeError(
                            f"{backend.name} returned {len(single)} answers for 1 request"
                        )
                    out.append(single[0])
                except Exception as exc:  # noqa: BLE001
                    out.append(exc)
    # A bug here must never truncate the review silently.
    if len(out) != len(requests):  # pragma: no cover
        raise RuntimeError(f"{len(out)} results for {len(requests)} requests")
    return out


def _error(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"


# A response that lacks a question or holds a value of the wrong type. Each
# one becomes an error row for its own request instead of stopping the run.
_PARSE_ERRORS = (KeyError, TypeError, ValueError, AttributeError, IndexError)


def _span_answers(res: dict) -> dict:
    best, p_best = _choice(res, "label")
    return {
        "p_pii": _p_noul(res, "is_pii"),
        "p_label_ok": _p_noul(res, "label_ok"),
        "best_label": best,
        "p_best": p_best,
    }


def _segment_answers(res: dict) -> dict:
    kind, p_kind = _choice(res, "residual_type")
    return {
        "p_residual": _p_noul(res, "residual"),
        "residual_type": kind,
        "p_residual_type": p_kind,
    }


# ------------------------------------------------------------------- review


def review_file_name(backend: DecisionBackend) -> str:
    return f"review_{safe_name(backend.name)}.jsonl"


def review_run(
    run_dir: str | Path,
    backend: DecisionBackend,
    *,
    fixtures: str | Path | None = None,
    detectors: Iterable[str] | None = None,
    labels: Iterable[str] | None = None,
    threshold: float = 0.5,
    batch_size: int = 8,
    progress: bool = True,
    window: int = 300,
    max_chars: int = 600,
) -> Path:
    """Run the span pass and the residual pass and write `run_dir/review_<backend>.jsonl`.

    Rows:

        {"id", "kind": "span", "start", "end", "label", "fine_label", "text", "detectors",
         "p_pii", "p_label_ok", "best_label", "p_best"}
        {"id", "kind": "segment", "detector", "start", "end", "p_residual",
         "residual_type", "p_residual_type"}
        {"id", "kind": "error", "pass": "span" | "segment", "start", "end", "error", ...}

    The span pass asks once per unique pooled span. The residual pass masks
    each detector's own spans (every label it emitted because the question
    asks about any personal data) in the same segments for every detector.
    A segment with nothing left after masking is not sent and is written
    with p_residual 0.0. Records a detector errored on get no segments for
    that detector. `threshold` is stored in the meta file as the default for
    `summarize`; the rows keep raw probabilities.

    A request the backend raised on, or answered without one of the
    questions, becomes an error row for that request alone.
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be at least 1, got {batch_size}")
    # Read the argument once: a generator would be empty by the second use.
    detectors = list(detectors) if detectors is not None else None
    run = load_run(run_dir, fixtures, detectors)
    run_dir = Path(run_dir)
    dets = list(detectors) if detectors is not None else run.detectors
    labels = target_labels(run, labels, fixtures)
    pooled = pooled_spans(run, dets, labels=labels)
    model = getattr(backend, "model_name", DEFAULT_MODEL)

    # ---- span pass
    span_items = [
        (ex, span)
        for ex in run.fixtures
        for span in pooled[ex["id"]]
        if span["canonical"]
    ]
    span_reqs = [
        span_request(ex["text"], s, labels, window=window, model=model) for ex, s in span_items
    ]
    span_res = _answer_all(
        backend, span_reqs, batch_size=batch_size, desc=f"{backend.name} spans", show=progress
    )
    rows: list[dict] = []
    # _answer_all returns exactly one result per request, so the lengths match.
    for (ex, span), res in zip(span_items, span_res, strict=True):
        base = {"id": ex["id"], "start": span["start"], "end": span["end"]}
        if not isinstance(res, Exception):
            try:
                answers = _span_answers(res)
            except _PARSE_ERRORS as exc:
                res = exc
        if isinstance(res, Exception):
            rows.append({
                **base,
                "kind": "error",
                "pass": "span",
                "detectors": span["detectors"],
                "error": _error(res),
            })
            continue
        rows.append({
            **base,
            "kind": "span",
            "label": span["label"],
            "fine_label": span["fine_label"],
            "text": span["text"],
            "detectors": span["detectors"],
            **answers,
        })

    # ---- residual pass
    seg_items: list[tuple[str, str, int, int, dict]] = []
    trivial: list[tuple[str, str, int, int]] = []
    for ex in run.fixtures:
        text = ex.get("text") or ""
        if not text.strip():
            continue
        det_spans: dict[str, list[dict]] = {}
        for det in dets:
            row = run.predictions.get(det, {}).get(ex["id"])
            if row is None or row.get("error"):
                continue
            det_spans[det] = [
                s for s in row.get("spans") or []
                if 0 <= int(s["start"]) < int(s["end"]) <= len(text)
            ]
        if not det_spans:
            continue
        everything = [s for spans in det_spans.values() for s in spans]
        bounds = split_segments(text, max_chars=max_chars, avoid=everything)
        for det, spans in det_spans.items():
            for a, b, req in segment_requests(text, spans, labels, segments=bounds, model=model):
                if _nothing_left(req):
                    trivial.append((ex["id"], det, a, b))
                else:
                    seg_items.append((ex["id"], det, a, b, req))
    seg_res = _answer_all(
        backend,
        [it[4] for it in seg_items],
        batch_size=batch_size,
        desc=f"{backend.name} segments",
        show=progress,
    )
    seg_rows: list[dict] = []
    for (rid, det, a, b, _), res in zip(seg_items, seg_res, strict=True):
        base = {"id": rid, "detector": det, "start": a, "end": b}
        if not isinstance(res, Exception):
            try:
                answers = _segment_answers(res)
            except _PARSE_ERRORS as exc:
                res = exc
        if isinstance(res, Exception):
            seg_rows.append({**base, "kind": "error", "pass": "segment", "error": _error(res)})
            continue
        seg_rows.append({**base, "kind": "segment", **answers})
    for rid, det, a, b in trivial:
        seg_rows.append({
            "id": rid, "detector": det, "start": a, "end": b, "kind": "segment",
            "p_residual": 0.0, "residual_type": NONE, "p_residual_type": 1.0, "trivial": True,
        })
    order = {ex["id"]: i for i, ex in enumerate(run.fixtures)}
    det_order = {d: i for i, d in enumerate(dets)}
    seg_rows.sort(key=lambda r: (order[r["id"]], det_order[r["detector"]], r["start"]))
    rows.extend(seg_rows)

    out = run_dir / review_file_name(backend)
    write_jsonl(out, rows)
    meta = {
        "backend": backend.name,
        "remote": bool(getattr(backend, "remote", False)),
        "model": model,
        "threshold": threshold,
        "labels": labels,
        "detectors": dets,
        "fixtures": str(fixtures or run.manifest.get("fixtures")),
        "window": window,
        "max_chars": max_chars,
        "n_span_requests": len(span_reqs),
        "n_segment_requests": len(seg_items),
        "n_trivial_segments": len(trivial),
        "n_errors": sum(1 for r in rows if r["kind"] == "error"),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    review_meta_path(out).write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    return out


# ----------------------------------------------------------------- summary


def _load(run_dir: str | Path, review_file: str | Path) -> tuple[list[dict], dict]:
    path = resolve_file(run_dir, review_file)
    meta_file = review_meta_path(path)
    meta = json.loads(meta_file.read_text()) if meta_file.exists() else {}
    return read_jsonl(path), meta


def _detectors_of(rows: list[dict], meta: dict) -> list[str]:
    if meta.get("detectors"):
        return list(meta["detectors"])
    found = {d for r in rows for d in r.get("detectors") or []} | {
        r["detector"] for r in rows if r.get("detector")
    }
    return sorted(found)


def summarize(
    run_dir: str | Path, review_file: str | Path, *, threshold: float | None = None
) -> list[dict]:
    """Per-detector estimates at `threshold` (default: the one stored at review time).

    Rows: `{"detector", "n_spans", "precision", "precision_any_label",
    "mean_p_pii", "n_segments", "residual_rate"}` where

    1. `precision` is the share of the detector's spans with p_pii >= t and
       p_label_ok >= t.
    2. `precision_any_label` is the share with p_pii >= t (the span is PII of
       some type).
    3. `residual_rate` is the share of the detector's segments with
       p_residual >= t.

    Errored requests are left out. A value is None when its count is 0.
    """
    rows, meta = _load(run_dir, review_file)
    t = float(threshold if threshold is not None else meta.get("threshold", 0.5))
    out = []
    for det in _detectors_of(rows, meta):
        spans = [r for r in rows if r["kind"] == "span" and det in r["detectors"]]
        segs = [r for r in rows if r["kind"] == "segment" and r["detector"] == det]
        n = len(spans)
        out.append({
            "detector": det,
            "n_spans": n,
            "precision": ratio(
                sum(1 for r in spans if r["p_pii"] >= t and r["p_label_ok"] >= t), n
            ),
            "precision_any_label": ratio(sum(1 for r in spans if r["p_pii"] >= t), n),
            "mean_p_pii": ratio(sum(r["p_pii"] for r in spans), n),
            "n_segments": len(segs),
            "residual_rate": ratio(sum(1 for r in segs if r["p_residual"] >= t), len(segs)),
        })
    return out


def threshold_sweep(
    run_dir: str | Path,
    review_file: str | Path,
    gold_fixtures: str | Path,
    thresholds: Iterable[float] = DEFAULT_THRESHOLDS,
) -> list[dict]:
    """Classifier estimates against real gold, per threshold and detector.

    For calibration on a dataset with gold spans (a `pii_masking_200k`
    sample). Rows: `{"threshold", "detector", "n_spans",
    "estimated_precision", "true_precision", "precision_error",
    "n_segments", "estimated_residual_rate", "true_residual_rate",
    "residual_error"}`.

    1. True precision is the share of the detector's reviewed spans that
       match a gold span exactly (same boundaries and same coarse label).
    2. True residual rate is the share of the detector's segments holding a
       gold span (in the review's label scope) that none of the detector's
       own spans overlaps.

    The `*_error` columns are estimate minus truth, so a positive value
    means the classifier is too optimistic about precision or too
    pessimistic about leftovers.
    """
    rows, meta = _load(run_dir, review_file)
    scope = coarse_scope(meta.get("labels") or taxonomy.ALL_LABELS)
    gold: dict[str, list[tuple[int, int, str]]] = {}
    for rec in iter_jsonl(gold_fixtures):
        gold[rec["id"]] = [
            (int(g["start"]), int(g["end"]), taxonomy.parent(g["label"]))
            for g in rec.get("gold_spans") or []
            if taxonomy.parent(g["label"]) in scope
        ]
    gold_sets = {rid: set(spans) for rid, spans in gold.items()}
    dets = _detectors_of(rows, meta)
    raw: dict[str, dict[str, list[tuple[int, int]]]] = {}
    for det in dets:
        path = Path(run_dir) / f"raw_{det}.jsonl"
        raw[det] = {
            r["id"]: [(int(s["start"]), int(s["end"])) for s in r.get("spans") or []]
            for r in (iter_jsonl(path) if path.exists() else [])
            if not r.get("error")
        }

    truth: dict[str, dict] = {}
    for det in dets:
        spans = [r for r in rows if r["kind"] == "span" and det in r["detectors"]]
        segs = [r for r in rows if r["kind"] == "segment" and r["detector"] == det]
        exact = sum(
            1 for r in spans
            if (r["start"], r["end"], taxonomy.parent(r["label"])) in gold_sets.get(r["id"], set())
        )
        leftover = 0
        for r in segs:
            own = raw[det].get(r["id"], [])
            for gs, ge, _ in gold.get(r["id"], []):
                inside = r["start"] < ge and gs < r["end"]
                if inside and not any(ps < ge and gs < pe for ps, pe in own):
                    leftover += 1
                    break
        truth[det] = {
            "spans": spans,
            "segs": segs,
            "true_precision": ratio(exact, len(spans)),
            "true_residual_rate": ratio(leftover, len(segs)),
        }

    out = []
    for t in thresholds:
        for det in dets:
            info = truth[det]
            spans, segs = info["spans"], info["segs"]
            est_p = ratio(
                sum(1 for r in spans if r["p_pii"] >= t and r["p_label_ok"] >= t), len(spans)
            )
            est_r = ratio(sum(1 for r in segs if r["p_residual"] >= t), len(segs))
            out.append({
                "threshold": t,
                "detector": det,
                "n_spans": len(spans),
                "estimated_precision": est_p,
                "true_precision": info["true_precision"],
                "precision_error": _diff(est_p, info["true_precision"]),
                "n_segments": len(segs),
                "estimated_residual_rate": est_r,
                "true_residual_rate": info["true_residual_rate"],
                "residual_error": _diff(est_r, info["true_residual_rate"]),
            })
    return out


def _diff(a: float | None, b: float | None) -> float | None:
    return None if a is None or b is None else a - b
