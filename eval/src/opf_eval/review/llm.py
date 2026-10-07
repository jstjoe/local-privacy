"""LLM review: estimate precision and recall without gold labels (plan 13, Part 5).

An LLM reviewer sees each record once, together with every span any detector
found in it (pooled and deduplicated, see `review.pool`). It returns

1. a verdict for every candidate span: `correct`, `wrong_label` (it is
   personal data but the category is wrong), `boundary` (right category but
   the span is too long or too short) or `not_pii`, and
2. the personal data in the text that no candidate covers ("missed").

Verdicts are credited to every detector that produced the span. From them:

1. **precision** of a detector = its spans judged `correct` / its judged
   spans. The lenient view also counts `boundary`.
2. **recall** of a detector = the true entities it found / every true entity
   the review knows about. True entities are the pooled spans the reviewer
   accepted as personal data plus the misses. Spans judged `correct` that
   overlap are one entity. A `boundary` or `wrong_label` span joins the one
   entity it overlaps most, so an over-long span that swallows two entities
   never merges them (see `_entities`). When no detector found the second
   entity on its own, the reviewer lists it as a miss and it stays a miss
   (see `_covered`).
   This is pooled recall: PII that every detector *and* the reviewer
   overlooked is invisible, so the estimate is an **upper bound**.

    from opf_eval import llm
    from opf_eval.review import llm as review_llm

    reviewer = llm.make_client("anthropic", cache_dir=ws.root / "llm_cache")
    path = review_llm.review_run(run_dir, reviewer)
    review_llm.summarize(run_dir, path)          # one row per detector
    review_llm.disagreements(run_dir, path)      # spans not judged correct + misses

Output is `run_dir/review_llm_<provider>_<model>.jsonl` with one row per judged
span, per located miss, per dropped miss and per failed record, and a
`.meta.json` sidecar naming the reviewer and prompt version. Rows of a record
that a server-side fallback model answered carry `fallback_model`, and the
meta counts those records in `n_fallback_records`. A record whose answer
stops at max_tokens is reviewed again in halves of its candidates (see
`_judge`), and the meta counts those records in `n_split_records`. Wrap the
client in `llm.CachedClient` (or pass `cache_dir` to `make_client`) so a rerun
costs nothing and returns the same numbers.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from .. import taxonomy
from ..io import iter_jsonl, read_jsonl, write_jsonl
from ..llm import describe
from ..llm.base import LLMError, pop_fallbacks
from ..silver import locate
from .pool import (
    context,
    f1,
    load_run,
    ok_ids,
    pooled_spans,
    progress,
    ratio,
    resolve_file,
    review_meta_path,
    safe_name,
    target_labels,
)

# v2: PII inside a candidate judged not_pii must be listed under "missed".
# v3: a second entity that an over-long "boundary" candidate runs into is a
# miss too, and the version is part of the prompt so a bump changes the cache key.
PROMPT_VERSION = "review-v3"
VERDICTS = ("correct", "wrong_label", "boundary", "not_pii")
# Written for a candidate the reviewer did not return a verdict for, even after
# the client's schema retry. Such spans are left out of every number.
UNJUDGED = "unjudged"
# Verdicts that say the span is personal data, so a true entity exists there.
_PII_VERDICTS = frozenset({"correct", "wrong_label", "boundary"})
# Context shown around each candidate so repeated values point at one occurrence.
_CANDIDATE_CONTEXT = 30
# Output budget of one review call. Reasoning models think before they answer
# and the thinking counts against it, so the client default of 8,000 cuts off
# records with many candidates. The anthropic SDK refuses a non-streaming call
# whose max_tokens implies more than 10 minutes (above about 21,333 tokens), so
# 16,000 stays clear of that guard. It matches `silver.MAX_TOKENS`.
MAX_TOKENS = 16000

_SYSTEM = """\
You review the output of personal-data (PII) detectors.

You get a TEXT and a numbered list of CANDIDATES. Each candidate is one span that a \
detector marked in the text, given as its exact characters, its label, its character \
offsets and a little surrounding context with the span in ⟦ ⟧. A candidate refers to \
that one occurrence only.

Task 1: give exactly one verdict for every candidate index:
- "correct": the span is personal or sensitive data, its boundaries are right, and \
the label names the right category.
- "wrong_label": the span is personal or sensitive data, but the label names the wrong \
category. Put the right label in "label".
- "boundary": the span is personal or sensitive data of the labelled category, but it \
is too long or too short (for example it cuts a name in half or swallows extra words).
- "not_pii": the span is not personal or sensitive data of any listed category.
Judge the label at the level of its category. A general label such as PERSON is right \
for a first name. Set "label" to null unless the verdict is "wrong_label". Keep "note" \
to a few words, or leave it empty.

Task 2: list in "missed" every piece of personal data of the listed categories that no \
candidate covers, even partly. Also list personal data that lies inside a candidate you \
judged "not_pii", for example a name inside a longer span that is not personal data as \
a whole. Also list a second piece of personal data that a "boundary" candidate runs into \
when that candidate is too long because it swallows part of another piece of data. List \
every occurrence. Copy "value" character for character from the text. Set "context" to \
the value with up to about six words around it, also copied exactly, so the right \
occurrence can be found. Otherwise do not list anything covered by a candidate you judged \
"correct", "wrong_label" or "boundary". Return an empty list when nothing was missed.

LABELS:
{labels}

(prompt {version})
"""


def judge_schema(n_spans: int, labels: Sequence[str]) -> dict:
    """JSON schema of one review answer for a record with `n_spans` candidates.

    `index` is 1-based to match the numbered list in the prompt. The array
    must hold exactly `n_spans` verdicts. Anthropic's structured outputs drop
    those count limits, but `llm.schema.validate` enforces them and the client
    retries once with the error. Every object lists all its properties in
    `required` with `additionalProperties: false`, so OpenAI strict mode accepts
    the schema.
    """
    labels = list(labels)
    if not labels:
        raise ValueError("labels must not be empty")
    label_enum = {"type": "string", "enum": labels}
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["verdicts", "missed"],
        "properties": {
            "verdicts": {
                "type": "array",
                "minItems": n_spans,
                "maxItems": n_spans,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["index", "verdict", "label", "note"],
                    "properties": {
                        "index": {"type": "integer", "minimum": 1, "maximum": max(n_spans, 1)},
                        "verdict": {"type": "string", "enum": list(VERDICTS)},
                        "label": {"anyOf": [label_enum, {"type": "null"}]},
                        "note": {"type": "string"},
                    },
                },
            },
            "missed": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["value", "label", "context"],
                    "properties": {
                        "value": {"type": "string"},
                        "label": label_enum,
                        "context": {"type": "string"},
                    },
                },
            },
        },
    }


def _label_lines(labels: Sequence[str]) -> str:
    return "\n".join(f"- {lbl}: {taxonomy.LABELS[lbl].description}" for lbl in labels)


def _candidate_label(span: dict) -> str:
    fine = span.get("fine_label")
    if fine and fine != span["label"]:
        return f"{span['label']} ({fine})"
    return span["label"]


def build_prompt(text: str, pooled: Sequence[dict], labels: Sequence[str]) -> tuple[str, str]:
    """(system, user) for one record.

    The system turn defines the verdicts and lists every target label with its
    description. The user turn shows the text and then the candidates, one
    per line: `[i] LABEL "span text" at start-end: …context ⟦span⟧ context…`.
    """
    system = _SYSTEM.format(labels=_label_lines(labels), version=PROMPT_VERSION)
    lines = [
        "TEXT:",
        "<<<",
        text,
        ">>>",
        "",
        f"CANDIDATES ({len(pooled)}):",
    ]
    if not pooled:
        lines.append("(none: no detector marked anything in this text)")
    for i, span in enumerate(pooled, start=1):
        snippet = context(text, span["start"], span["end"], _CANDIDATE_CONTEXT).replace("\n", " ")
        lines.append(
            f"[{i}] {_candidate_label(span)} {json.dumps(span['text'], ensure_ascii=False)} "
            f"at {span['start']}-{span['end']}: {snippet}"
        )
    return system, "\n".join(lines)


# ------------------------------------------------------------------- review


def review_file_name(reviewer) -> str:
    """`review_llm_<provider>_<model>.jsonl` with unsafe characters replaced."""
    return f"review_llm_{safe_name(reviewer.provider)}_{safe_name(reviewer.model)}.jsonl"


def _review_record(
    reviewer, ex: dict, candidates: list[dict], labels: list[str]
) -> tuple[list[dict], bool]:
    """Rows for one record and whether its candidates had to be split.

    Errors from the client become one error row.
    """
    rid = ex["id"]
    text = ex.get("text") or ""
    pop_fallbacks(reviewer)  # this record's calls run on this thread; start clean
    splits = [0]
    try:
        verdicts, found, dropped = _judge(reviewer, text, candidates, labels, splits)
    except Exception as exc:  # noqa: BLE001 — any failure becomes an error row, never a guess
        error = {"id": rid, "kind": "error", "error": f"{type(exc).__name__}: {exc}"}
        return [error], splits[0] > 0
    rows = _answer_rows(rid, text, candidates, verdicts, found, dropped)
    fallbacks = pop_fallbacks(reviewer)
    if fallbacks:
        for row in rows:
            row["fallback_model"] = fallbacks[-1]
    return rows, splits[0] > 0


def _judge(
    reviewer, text: str, candidates: list[dict], labels: list[str], splits: list[int]
) -> tuple[dict[int, dict], list[dict], list[dict]]:
    """Ask the reviewer about `candidates` in `text`.

    Returns the verdicts keyed by the 1-based index into `candidates`, the
    located misses and the dropped misses. When the answer stops at
    max_tokens the candidates are cut in half and each half is reviewed
    against the same whole text, down to a single candidate. The halves'
    verdicts are mapped back to their indexes in `candidates` and their misses
    are joined. Each half sees only its own candidates and so lists the PII
    that the other half covers as missed. `_answer_rows` drops those misses
    because it checks them against every candidate's verdict. Each split adds
    one to `splits[0]`. Any other error propagates, and so does a truncated
    answer for a single candidate or for a record without candidates.
    """
    system, user = build_prompt(text, candidates, labels)
    try:
        answer = reviewer.complete_json(
            system=system,
            user=user,
            schema=judge_schema(len(candidates), labels),
            name="review",
            max_tokens=MAX_TOKENS,
        )
    except LLMError as exc:
        if not exc.truncated or len(candidates) <= 1:
            raise
        splits[0] += 1
        mid = len(candidates) // 2
        verdicts, found, dropped = _judge(reviewer, text, candidates[:mid], labels, splits)
        right, found_right, dropped_right = _judge(
            reviewer, text, candidates[mid:], labels, splits
        )
        verdicts.update({mid + i: v for i, v in right.items()})
        return verdicts, found + found_right, dropped + dropped_right
    verdicts: dict[int, dict] = {}
    for v in answer.get("verdicts") or []:
        i = int(v["index"])
        if 1 <= i <= len(candidates):
            verdicts.setdefault(i, v)  # a repeated index keeps its first verdict
    found, dropped = locate(text, list(answer.get("missed") or []))
    return verdicts, found, dropped


def _answer_rows(
    rid: str,
    text: str,
    candidates: list[dict],
    verdicts: Mapping[int, dict],
    found: list[dict],
    dropped: list[dict],
) -> list[dict]:
    """Span rows for every candidate, then the located misses and the dropped ones.

    `verdicts` is keyed by the 1-based candidate index. `found` and `dropped`
    may come from several calls about the same text. A miss is kept once per
    `(start, end)` and a dropped value once per value, label and reason. A
    miss that an accepted candidate stands for is dropped (see `_covered`).
    A miss with the exact boundaries of a candidate judged `not_pii` is
    dropped with the reason `judged_not_pii` because the call that saw that
    candidate rejected it. A half of a split record lists the other half's
    candidates as missed because it cannot see them, and without this rule a
    candidate the other half rejected would be both a `not_pii` span and a miss.
    """
    rows: list[dict] = []
    for i, span in enumerate(candidates, start=1):
        v = verdicts.get(i)
        verdict = v["verdict"] if v else UNJUDGED
        corrected = v.get("label") if v and verdict == "wrong_label" else None
        rows.append({
            "id": rid,
            "kind": "span",
            "start": span["start"],
            "end": span["end"],
            "label": span["label"],
            "fine_label": span["fine_label"],
            "text": span["text"],
            "detectors": span["detectors"],
            "verdict": verdict,
            "corrected_label": corrected,
            "note": (v.get("note") or "") if v else "",
        })

    unique: dict[tuple[int, int], dict] = {}
    for span in found:
        unique.setdefault((span["start"], span["end"]), span)
    rejected = {(r["start"], r["end"]) for r in rows if r["verdict"] == "not_pii"}
    dropped = list(dropped)
    for key, span in sorted(unique.items()):
        if key in rejected and not _covered(key, rows):
            dropped.append({
                "value": span["text"],
                "label": span.get("fine_label") or span["label"],
                "reason": "judged_not_pii",
            })
            continue
        if _covered(key, rows):
            # The value sits under an accepted candidate, so it was not missed
            # by all detectors. That candidate already carries the verdict.
            dropped.append({
                "value": span["text"],
                "label": span.get("fine_label") or span["label"],
                "reason": "covered_by_candidate",
            })
            continue
        rows.append({
            "id": rid,
            "kind": "missed",
            "start": span["start"],
            "end": span["end"],
            "label": taxonomy.parent(span["label"]),
            "fine_label": span.get("fine_label") or span["label"],
            "text": text[span["start"]:span["end"]],
        })
    seen: set[tuple] = set()
    for item in dropped:
        row = {
            "id": rid,
            "kind": "dropped",
            "value": item.get("value"),
            "label": item.get("label"),
            "reason": item.get("reason", "not_found"),
        }
        key = (row["value"], row["label"], row["reason"])
        if key not in seen:
            seen.add(key)
            rows.append(row)
    return rows


def _covered(miss: tuple[int, int], rows: list[dict]) -> bool:
    """True when a candidate the reviewer accepted already stands for `miss`.

    Only a candidate judged personal data covers a miss. PII under a not_pii
    or unjudged candidate was found by no detector, so it stays a miss and
    counts in every detector's recall denominator. An accepted candidate
    covers a miss when it holds all of it, or when its boundaries were
    judged right (correct or wrong_label) and it overlaps the miss. A
    `boundary` candidate that only partly overlaps the miss covers it only
    when it overlaps no candidate with right boundaries. Otherwise it is an
    over-long span that belongs to that other candidate's entity and ran into
    this second entity, which no detector found on its own.
    """
    def overlaps(a: tuple[int, int], b: tuple[int, int]) -> bool:
        return a[0] < b[1] and b[0] < a[1]

    accepted = [r for r in rows if r.get("verdict") in _PII_VERDICTS]
    exact = [(r["start"], r["end"]) for r in accepted if r["verdict"] != "boundary"]
    for r in accepted:
        span = (r["start"], r["end"])
        if span[0] <= miss[0] and miss[1] <= span[1]:
            return True
        if not overlaps(span, miss):
            continue
        if r["verdict"] != "boundary" or not any(overlaps(span, e) for e in exact):
            return True
    return False


def review_run(
    run_dir: str | Path,
    reviewer,
    *,
    fixtures: str | Path | None = None,
    detectors: Iterable[str] | None = None,
    labels: Iterable[str] | None = None,
    max_workers: int = 4,
    progress: bool = True,
) -> Path:
    """Review every record of a run with `reviewer` (an `llm.LLMClient`) and write the rows.

    fixtures: the fixtures file the run was made from (default: the run
        manifest's). Pass it when the file moved since the run.
    detectors: which detectors to pool (default: every detector in the run).
    labels: target labels (default: the fixtures sidecar's, else the
        manifest's). Candidates outside these categories are not reviewed.
    max_workers: concurrent reviewer calls.

    One call per record, including records with no candidates because the
    reviewer can still find misses there. A record whose answer stops at
    max_tokens is reviewed again in halves of its candidates, which adds
    calls (see `_judge`). Blank records are skipped. Returns the path of
    `review_llm_<provider>_<model>.jsonl` in `run_dir`.
    """
    # Read the argument once: a generator would be empty by the second use.
    detectors = list(detectors) if detectors is not None else None
    run = load_run(run_dir, fixtures, detectors)
    run_dir = Path(run_dir)
    dets = list(detectors) if detectors is not None else run.detectors
    labels = target_labels(run, labels, fixtures)
    pooled = pooled_spans(run, dets, labels=labels)
    work = [
        (ex, [s for s in pooled[ex["id"]] if s["canonical"]])
        for ex in run.fixtures
        if (ex.get("text") or "").strip()
    ]

    def one(item: tuple[dict, list[dict]]) -> tuple[list[dict], bool]:
        ex, candidates = item
        return _review_record(reviewer, ex, candidates, labels)

    with ThreadPoolExecutor(max_workers=max(1, max_workers)) as pool:
        # map() keeps submission order, so the file follows the fixtures.
        results = list(_progress(
            pool.map(one, work),
            total=len(work),
            desc=f"review {reviewer.model}",
            enabled=progress,
        ))

    out = run_dir / review_file_name(reviewer)
    rows = [r for rs, _ in results for r in rs]
    write_jsonl(out, rows)
    kinds = [r["kind"] for r in rows]
    meta = {
        "reviewer": describe(reviewer),
        "provider": reviewer.provider,
        "model": reviewer.model,
        "remote": bool(getattr(reviewer, "remote", True)),
        "prompt_version": PROMPT_VERSION,
        "labels": labels,
        "detectors": dets,
        "fixtures": str(fixtures or run.manifest.get("fixtures")),
        "n_records": len(work),
        "reviewed_ids": [ex["id"] for ex, _ in work],
        "n_errors": kinds.count("error"),
        "n_spans": kinds.count("span"),
        "n_unjudged": sum(1 for r in rows if r.get("verdict") == UNJUDGED),
        "n_missed": kinds.count("missed"),
        "n_dropped": kinds.count("dropped"),
        "n_fallback_records": len({r["id"] for r in rows if r.get("fallback_model")}),
        "n_split_records": sum(1 for _, split in results if split),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    review_meta_path(out).write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    return out


def _progress(items, *, total: int, desc: str, enabled: bool):
    return progress(items, total=total, desc=desc, enabled=enabled)


def dry_run_responder(system: str, user: str, schema: dict) -> dict:
    """A `StubClient` responder that accepts every candidate and finds no misses.

    For running the notebook end to end without keys:
    `llm.make_client("stub", responder=review.llm.dry_run_responder)`. The
    numbers it produces mean nothing (every precision is 1.0).
    """
    n = schema["properties"]["verdicts"].get("minItems", 0)
    return {
        "verdicts": [
            {"index": i, "verdict": "correct", "label": None, "note": "dry run"}
            for i in range(1, n + 1)
        ],
        "missed": [],
    }


def estimate(
    run_dir: str | Path,
    *,
    fixtures: str | Path | None = None,
    detectors: Iterable[str] | None = None,
    labels: Iterable[str] | None = None,
) -> dict:
    """What `review_run` would send, for the notebook's cost note before a hosted call.

    Returns `{"n_calls", "n_candidates", "n_chars"}`: one call per non-blank
    record, the candidates it would judge and the characters of record text
    (the prompt adds roughly 2,000 characters of instructions per call).
    `n_calls` is a lower bound because a record whose answer stops at
    max_tokens is reviewed again in halves of its candidates and every split
    adds calls.
    """
    detectors = list(detectors) if detectors is not None else None
    run = load_run(run_dir, fixtures, detectors)
    labels = target_labels(run, labels, fixtures)
    pooled = pooled_spans(run, detectors, labels=labels)
    recs = [ex for ex in run.fixtures if (ex.get("text") or "").strip()]
    return {
        "n_calls": len(recs),
        "n_candidates": sum(sum(1 for s in pooled[ex["id"]] if s["canonical"]) for ex in recs),
        "n_chars": sum(len(ex["text"]) for ex in recs),
    }


# ----------------------------------------------------------------- summary


def _read_review(run_dir: str | Path, review_file: str | Path) -> tuple[Path, list[dict], dict]:
    path = resolve_file(run_dir, review_file)
    meta_file = review_meta_path(path)
    meta = json.loads(meta_file.read_text()) if meta_file.exists() else {}
    return path, read_jsonl(path), meta


def _overlap(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))


def _entities(spans: list[dict]) -> list[list[dict]]:
    """Group a record's PII spans (verdict correct, wrong_label or boundary) into entities.

    Spans judged `correct` anchor the entities: overlapping correct spans are
    one entity. Every other span joins the entity whose anchor it overlaps
    most and never more than one, so a `boundary` span that runs over two
    names cannot merge them into one entity and earn a detector credit for
    both. A span that overlaps no anchor starts its own entity, which later
    non-correct spans can join the same way.
    """
    order = sorted(spans, key=lambda s: (s["start"], s["end"]))
    groups: list[list[dict]] = []
    anchors: list[tuple[int, int]] = []
    for s in (s for s in order if s["verdict"] == "correct"):
        if groups and s["start"] < anchors[-1][1]:
            groups[-1].append(s)
            anchors[-1] = (anchors[-1][0], max(anchors[-1][1], s["end"]))
        else:
            groups.append([s])
            anchors.append((s["start"], s["end"]))
    for s in (s for s in order if s["verdict"] != "correct"):
        span = (s["start"], s["end"])
        best = max(range(len(groups)), key=lambda i: _overlap(span, anchors[i]), default=None)
        if best is not None and _overlap(span, anchors[best]) > 0:
            groups[best].append(s)
        else:
            groups.append([s])
            anchors.append(span)
    return groups


def summarize(
    run_dir: str | Path,
    review_file: str | Path,
    *,
    lenient: bool = False,
) -> list[dict]:
    """Estimated precision and recall per detector from a review file.

    Each row is `{"detector", "n_records", "n_spans", "correct",
    "wrong_label", "boundary", "not_pii", "precision", "recall", "f1",
    "missed_by_all"}`.

    1. `n_spans` counts the detector's judged spans. `precision` is correct /
       n_spans, or (correct + boundary) / n_spans when `lenient`.
    2. `recall` is pooled recall and an upper bound. Its denominator is every
       true entity the review knows of: the pooled spans judged correct,
       wrong_label or boundary plus the reviewer's misses. Overlapping
       correct spans form one entity and any other accepted span joins the
       single entity it overlaps most (see `_entities`). A detector finds an
       entity when one of its spans in it was judged correct (or boundary
       when `lenient`).
    3. Records the reviewer failed on are excluded from every number and so
       are records the detector itself errored on (for that detector only).
    4. `missed_by_all` counts the reviewer's misses in the detector's records.
    """
    _, rows, meta = _read_review(run_dir, review_file)
    failed = {r["id"] for r in rows if r["kind"] == "error"}
    by_id: dict[str, dict[str, list[dict]]] = {}
    for r in rows:
        if r["id"] in failed or r["kind"] not in ("span", "missed"):
            continue
        by_id.setdefault(r["id"], {"span": [], "missed": []})[r["kind"]].append(r)
    # A clean record with no candidates and no misses writes no rows, so the
    # meta's id list is what says it was reviewed.
    reviewed = set(meta.get("reviewed_ids") or ()) | {r["id"] for r in rows}
    hit_verdicts = {"correct", "boundary"} if lenient else {"correct"}
    detectors = meta.get("detectors") or sorted({d for r in rows for d in r.get("detectors") or []})

    out = []
    for det in detectors:
        valid = (reviewed - failed) & ok_ids(run_dir, det)
        counts = dict.fromkeys(VERDICTS, 0)
        n_entities = n_found = n_missed = 0
        for rid in valid:
            rec = by_id.get(rid, {"span": [], "missed": []})
            for s in rec["span"]:
                if det in s["detectors"] and s["verdict"] in counts:
                    counts[s["verdict"]] += 1
            pii = [s for s in rec["span"] if s["verdict"] in _PII_VERDICTS]
            for entity in _entities(pii):
                n_entities += 1
                if any(det in s["detectors"] and s["verdict"] in hit_verdicts for s in entity):
                    n_found += 1
            n_missed += len(rec["missed"])
        judged = sum(counts.values())
        hits = counts["correct"] + (counts["boundary"] if lenient else 0)
        precision = ratio(hits, judged)
        recall = ratio(n_found, n_entities + n_missed)
        out.append({
            "detector": det,
            "n_records": len(valid),
            "n_spans": judged,
            **counts,
            "precision": precision,
            "recall": recall,
            "f1": f1(precision, recall),
            "missed_by_all": n_missed,
        })
    return out


def disagreements(
    run_dir: str | Path,
    review_file: str | Path,
    *,
    limit: int | None = 50,
    fixtures: str | Path | None = None,
    width: int = 60,
) -> list[dict]:
    """Spans the reviewer did not accept as correct, then the misses, with context.

    Rows are `{"id", "kind", "verdict", "label", "corrected_label", "text",
    "detectors", "note", "start", "end", "context"}`; misses have verdict
    "missed" and no detectors. When the record came from a document, the row
    also has `doc_id`, `doc_start` and `doc_end` (offsets in the whole
    document) so `fixtures.map_spans_to_documents`-style tables can show the
    page or cell. `context` shows ±`width` characters with the span in ⟦ ⟧.
    """
    _, rows, meta = _read_review(run_dir, review_file)
    fx = fixtures or meta.get("fixtures")
    if not fx or fx == "None":
        fx = json.loads((Path(run_dir) / "manifest.json").read_text())["fixtures"]
    records = {r["id"]: r for r in iter_jsonl(fx)}
    spans = [r for r in rows if r["kind"] == "span" and r["verdict"] not in ("correct", UNJUDGED)]
    misses = [r for r in rows if r["kind"] == "missed"]
    out: list[dict] = []
    for r in spans + misses:
        rec = records.get(r["id"]) or {}
        text = rec.get("text") or ""
        row = {
            "id": r["id"],
            "kind": r["kind"],
            "verdict": r.get("verdict", "missed"),
            "label": r.get("fine_label") or r["label"],
            "corrected_label": r.get("corrected_label"),
            "text": r["text"],
            "detectors": r.get("detectors", []),
            "note": r.get("note", ""),
            "start": r["start"],
            "end": r["end"],
            "context": context(text, r["start"], r["end"], width) if text else "",
        }
        if rec.get("doc_id") is not None:
            offset = int(rec.get("offset") or 0)
            row.update(
                doc_id=rec["doc_id"], doc_start=r["start"] + offset, doc_end=r["end"] + offset
            )
        out.append(row)
        if limit is not None and len(out) >= limit:
            break
    return out


def verdict_counts(review_file: str | Path) -> Mapping[str, int]:
    """How often each verdict (and "missed", "dropped", "error") occurs in a review file."""
    counts: dict[str, int] = {}
    for r in iter_jsonl(review_file):
        key = r.get("verdict") if r["kind"] == "span" else r["kind"]
        counts[key] = counts.get(key, 0) + 1
    return counts
