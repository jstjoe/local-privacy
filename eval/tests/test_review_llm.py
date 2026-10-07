"""Pooled spans and LLM review (plan 13, Part 5) with a stub reviewer.

The run directories are written by hand (manifest + raw files + fixtures)
so every number in the summary can be worked out on paper. One test goes
through `fixtures.from_texts` and `runner.run` with a registered regex
detector to show the review fits the real pipeline. No network.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from opf_eval import fixtures as fx
from opf_eval import scoring
from opf_eval.detectors import make_span, registry
from opf_eval.io import meta_path, read_jsonl, write_jsonl
from opf_eval.llm import CachedClient, LLMError, StubClient
from opf_eval.llm.schema import for_anthropic, strict_problems, validate
from opf_eval.review import llm as rl
from opf_eval.review.pool import pooled_spans
from opf_eval.runner import run as run_detectors
from opf_eval.taxonomy import register_vocab

LABELS = ["EMAIL", "GOV_ID", "PERSON", "PHONE"]


def _span(text: str, value: str, label: str, fine: str | None = None, nth: int = 0) -> dict:
    start = [m.start() for m in re.finditer(re.escape(value), text)][nth]
    return {
        "label": label,
        "fine_label": fine or label,
        "raw_label": (fine or label).lower(),
        "start": start,
        "end": start + len(value),
        "text": value,
    }


def make_run(
    root: Path,
    records: dict[str, str | dict],
    preds: dict[str, dict[str, list[dict] | str]],
    *,
    labels: list[str] = LABELS,
    gold: str = "none",
) -> tuple[Path, Path]:
    """A run dir + fixtures. `preds[det][id]` is a span list, or a string for an error row."""
    fixtures = root / "fixtures.jsonl"
    rows = []
    for rid, rec in records.items():
        rec = {"text": rec} if isinstance(rec, str) else rec
        rows.append({"id": rid, "language": None, "gold_spans": [], **rec})
    write_jsonl(fixtures, rows)
    meta_path(fixtures).write_text(json.dumps({"gold": gold, "labels": labels}))
    run_dir = root / "run"
    run_dir.mkdir()
    (run_dir / "manifest.json").write_text(json.dumps({
        "fixtures": str(fixtures), "dataset": None, "vocab_key": None,
        "labels": labels, "detectors": sorted(preds),
    }))
    for det, by_id in preds.items():
        out = []
        for rid, spans in by_id.items():
            if isinstance(spans, str):
                out.append(
                    {"id": rid, "detector": det, "spans": [], "latency_ms": 1.0, "error": spans}
                )
            else:
                out.append(
                    {"id": rid, "detector": det, "spans": spans, "latency_ms": 1.0, "error": None}
                )
        write_jsonl(run_dir / f"raw_{det}.jsonl", out)
    return run_dir, fixtures


def candidates(user: str) -> list[tuple[int, str, str]]:
    """(index, label, span text) for each candidate line of a review prompt."""
    out = []
    pattern = (
        r'^\[(\d+)\] ([A-Z_]+)(?: \([A-Z_]+\))? '
        r'("(?:[^"\\]|\\.)*") at '
    )
    for m in re.finditer(pattern, user, re.MULTILINE):
        out.append((int(m.group(1)), m.group(2), json.loads(m.group(3))))
    return out


def record_text(user: str) -> str:
    return user.split("<<<\n", 1)[1].split("\n>>>", 1)[0]


def responder_from(
    verdicts: dict[str, tuple[str, str | None]],
    missed: dict[str, list[dict]] | None = None,
    fail_on: str | None = None,
):
    """A reviewer that judges candidates by their text and reports fixed misses per text."""
    missed = missed or {}

    def respond(system: str, user: str, schema: dict) -> dict:
        text = record_text(user)
        if fail_on and fail_on in text:
            raise LLMError("refused: test")
        out = []
        for i, _, span_text in candidates(user):
            verdict, label = verdicts.get(span_text, ("not_pii", None))
            out.append({"index": i, "verdict": verdict, "label": label, "note": ""})
        found = [m for key, items in missed.items() if key in text for m in items]
        return {"verdicts": out, "missed": found}

    return respond


# ---------------------------------------------------------------- pooling

T1 = "Contact Jane Doe at jane@x.com or 555-1234. SSN 123-45-6789."
T2 = "Call Bob tomorrow."
T3 = "Write to sue@y.org now."


def standard_run(tmp_path: Path) -> tuple[Path, Path]:
    preds = {
        "a": {
            "r1": [_span(T1, "Jane Doe", "PERSON"), _span(T1, "jane@x.com", "EMAIL")],
            "r2": [_span(T2, "Bob", "PERSON", "GIVEN_NAME")],
            "r3": [_span(T3, "sue@y.org", "EMAIL")],
        },
        "b": {
            "r1": [
                _span(T1, "Jane Doe", "PERSON"),
                _span(T1, "Contact", "PERSON"),
                _span(T1, "555-1234", "ACCOUNT"),
                _span(T1, "jane@x", "EMAIL"),
            ],
            "r2": "timeout",
            "r3": [_span(T3, "sue@y.org", "EMAIL")],
        },
    }
    return make_run(tmp_path, {"r1": T1, "r2": T2, "r3": T3}, preds)


STANDARD_VERDICTS = {
    "Jane Doe": ("correct", None),
    "jane@x.com": ("correct", None),
    "Bob": ("correct", None),
    "sue@y.org": ("correct", None),
    "Contact": ("not_pii", None),
    "555-1234": ("wrong_label", "PHONE"),
    "jane@x": ("boundary", None),
}
STANDARD_MISSED = {
    "SSN 123": [{"value": "123-45-6789", "label": "GOV_ID", "context": "SSN 123-45-6789."}]
}


def test_pooled_spans_dedups_and_credits_every_detector(tmp_path):
    run_dir, _ = standard_run(tmp_path)
    pooled = pooled_spans(scoring.Run.load(run_dir))
    r1 = pooled["r1"]
    assert [(s["text"], s["label"]) for s in r1] == [
        ("Contact", "PERSON"), ("Jane Doe", "PERSON"), ("jane@x", "EMAIL"),
        ("jane@x.com", "EMAIL"), ("555-1234", "ACCOUNT"),
    ]
    jane = r1[1]
    assert jane["detectors"] == ["a", "b"]
    assert jane["canonical"] is True
    assert set(jane) == {"start", "end", "label", "fine_label", "text", "detectors", "canonical"}
    # b errored on r2, so only a's span is pooled there.
    assert [(s["text"], s["detectors"], s["fine_label"]) for s in pooled["r2"]] == [
        ("Bob", ["a"], "GIVEN_NAME")
    ]
    assert pooled["r3"][0]["detectors"] == ["a", "b"]


def test_pooled_spans_flags_unknown_labels_and_applies_scope(tmp_path):
    text = "Mia lives in Oslo, mia@z.io"
    preds = {
        "a": {"r1": [
            _span(text, "Mia", "PERSON", "GIVEN_NAME"),
            {**_span(text, "Oslo", "NRP"), "fine_label": "NRP"},
            _span(text, "mia@z.io", "EMAIL"),
            {"label": "EMAIL", "start": 5, "end": 5, "text": ""},      # empty: ignored
            {"label": "EMAIL", "start": 20, "end": 999, "text": "x"},  # out of range: ignored
            _span(text, "Mia", "PERSON", "GIVEN_NAME"),                # repeated by one detector
        ]},
        "b": {"r1": [_span(text, "Mia", "PERSON", "FAMILY_NAME")], "r2": []},
        "c": {"r1": [_span(text, "Mia", "PERSON", "FAMILY_NAME")]},
    }
    run_dir, _ = make_run(tmp_path, {"r1": text, "r2": "nothing here"}, preds)
    run = scoring.Run.load(run_dir)
    pooled = pooled_spans(run)
    by_text = {s["text"]: s for s in pooled["r1"]}
    assert by_text["Oslo"]["canonical"] is False
    assert by_text["Mia"]["detectors"] == ["a", "b", "c"]
    # Two votes for FAMILY_NAME beat one (the repeated span counts once).
    assert by_text["Mia"]["fine_label"] == "FAMILY_NAME"
    assert pooled["r2"] == []
    scoped = pooled_spans(run, labels=["PERSON", "GIVEN_NAME"])
    # Unknown labels survive the scope.
    assert sorted(s["text"] for s in scoped["r1"]) == ["Mia", "Oslo"]
    only_a = pooled_spans(run, ["a"])
    assert {s["text"]: s["detectors"] for s in only_a["r1"]}["Mia"] == ["a"]
    with pytest.raises(ValueError, match="no raw results"):
        pooled_spans(run, ["nope"])


def test_fine_label_tie_goes_to_first_detector(tmp_path):
    text = "Ann Lee"
    preds = {
        "a": {"r1": [_span(text, "Ann", "PERSON", "GIVEN_NAME")]},
        "b": {"r1": [_span(text, "Ann", "PERSON", "FAMILY_NAME")]},
    }
    run_dir, _ = make_run(tmp_path, {"r1": text}, preds)
    assert pooled_spans(scoring.Run.load(run_dir))["r1"][0]["fine_label"] == "GIVEN_NAME"


# ----------------------------------------------------------------- schema


def test_judge_schema_is_strict_and_validates_answers():
    schema = rl.judge_schema(2, LABELS)
    assert strict_problems(schema) == []
    # Claude drops the count limits; the client-side validator still enforces them.
    assert "maxItems" not in for_anthropic(schema)["properties"]["verdicts"]
    good = {
        "verdicts": [
            {"index": 1, "verdict": "correct", "label": None, "note": ""},
            {"index": 2, "verdict": "wrong_label", "label": "PHONE", "note": "a phone"},
        ],
        "missed": [{"value": "x", "label": "EMAIL", "context": "x"}],
    }
    assert validate(good, schema) == []
    too_few = {**good, "verdicts": good["verdicts"][:1]}
    assert any("at least 2" in e for e in validate(too_few, schema))
    bad_index = {**good, "verdicts": [good["verdicts"][0], {**good["verdicts"][1], "index": 3}]}
    assert validate(bad_index, schema)
    bad_label = {**good, "missed": [{"value": "x", "label": "NOPE", "context": ""}]}
    assert validate(bad_label, schema)
    assert validate({"verdicts": [], "missed": []}, rl.judge_schema(0, LABELS)) == []
    with pytest.raises(ValueError):
        rl.judge_schema(1, [])


def test_build_prompt_lists_labels_and_numbered_candidates():
    pooled = [
        {**_span(T1, "Jane Doe", "PERSON"), "detectors": ["a"]},
        {**_span(T1, "555-1234", "ACCOUNT", "GOV_ID"), "detectors": ["b"]},
    ]
    system, user = rl.build_prompt(T1, pooled, LABELS)
    assert "- GOV_ID: Government ID" in system and "not_pii" in system
    assert record_text(user) == T1
    assert candidates(user) == [(1, "PERSON", "Jane Doe"), (2, "ACCOUNT", "555-1234")]
    assert "ACCOUNT (GOV_ID)" in user
    assert "⟦Jane Doe⟧" in user and "at 8-16" in user
    _, empty = rl.build_prompt("hello", [], LABELS)
    assert "CANDIDATES (0)" in empty and "(none" in empty


def test_dry_run_responder_matches_schema():
    for n in (0, 3):
        schema = rl.judge_schema(n, LABELS)
        answer = rl.dry_run_responder("s", "u", schema)
        assert validate(answer, schema) == []
        assert [v["index"] for v in answer["verdicts"]] == list(range(1, n + 1))


# ----------------------------------------------------------------- review


def test_review_run_writes_rows_and_meta(tmp_path):
    run_dir, _ = standard_run(tmp_path)
    reviewer = StubClient(responder_from(STANDARD_VERDICTS, STANDARD_MISSED), model="org/model:v1")
    path = rl.review_run(run_dir, reviewer, progress=False, max_workers=3)
    assert path == run_dir / "review_llm_stub_org-model-v1.jsonl"
    rows = read_jsonl(path)
    assert [r["id"] for r in rows] == sorted(r["id"] for r in rows)  # fixture order kept
    spans = [r for r in rows if r["kind"] == "span"]
    assert len(spans) == 7  # 5 in r1, 1 in r2, 1 in r3
    wrong = next(r for r in spans if r["text"] == "555-1234")
    assert wrong["verdict"] == "wrong_label" and wrong["corrected_label"] == "PHONE"
    assert wrong["detectors"] == ["b"]
    assert all(r["corrected_label"] is None for r in spans if r["verdict"] != "wrong_label")
    missed = [r for r in rows if r["kind"] == "missed"]
    assert missed == [{
        "id": "r1", "kind": "missed", "start": T1.index("123-45"), "end": T1.index("123-45") + 11,
        "label": "ACCOUNT", "fine_label": "GOV_ID", "text": "123-45-6789",
    }]
    meta = json.loads(path.with_name(path.stem + ".meta.json").read_text())
    assert meta["prompt_version"] == rl.PROMPT_VERSION
    assert meta["labels"] == LABELS and meta["detectors"] == ["a", "b"]
    assert meta["n_records"] == 3 and meta["n_errors"] == 0 and meta["n_missed"] == 1
    assert meta["reviewed_ids"] == ["r1", "r2", "r3"]
    assert "local stub" in meta["reviewer"] and meta["remote"] is False
    assert len(reviewer.calls) == 3


def test_reviewer_errors_become_error_rows(tmp_path):
    run_dir, _ = standard_run(tmp_path)
    reviewer = StubClient(responder_from(STANDARD_VERDICTS, STANDARD_MISSED, fail_on="sue@"))
    path = rl.review_run(run_dir, reviewer, progress=False)
    errors = [r for r in read_jsonl(path) if r["kind"] == "error"]
    assert errors == [{"id": "r3", "kind": "error", "error": "LLMError: refused: test"}]
    assert not [r for r in read_jsonl(path) if r["id"] == "r3" and r["kind"] != "error"]


class _Budgeted(StubClient):
    """A stub that records `max_tokens` and stops early above `max_candidates` candidates.

    `fail` names an `LLMError` to raise for every prompt instead of answering.
    """

    def __init__(self, responder, *, max_candidates: int | None = None, fail=None):
        super().__init__(responder)
        self.max_candidates = max_candidates
        self.fail = fail
        self.budgets: list[int] = []
        self.sizes: list[int] = []

    def complete_json(self, *, system, user, schema, name="result", max_tokens=8000):
        self.budgets.append(max_tokens)
        self.sizes.append(len(candidates(user)))
        if self.fail is not None:
            raise self.fail
        if self.max_candidates is not None and len(candidates(user)) > self.max_candidates:
            raise LLMError(f"hit max_tokens={max_tokens}", truncated=True)
        return super().complete_json(system=system, user=user, schema=schema, name=name)


def own_misses_responder(system: str, user: str, schema: dict) -> dict:
    """Judges like STANDARD_VERDICTS and lists the PII that its own candidates do not cover.

    A real reviewer that sees only some of the candidates lists the rest as
    missed, so each half of a split record reports the other half's PII.
    """
    text = record_text(user)
    answer = responder_from(STANDARD_VERDICTS, STANDARD_MISSED)(system, user, schema)
    own = {span_text for _, _, span_text in candidates(user)}
    for value, label in (("Jane Doe", "PERSON"), ("jane@x.com", "EMAIL")):
        if value in text and value not in own:
            answer["missed"].append({"value": value, "label": label, "context": value})
    answer["missed"].append({"value": "not in text", "label": "PERSON", "context": ""})
    return answer


def test_review_calls_ask_for_a_larger_output_budget(tmp_path):
    run_dir, _ = standard_run(tmp_path)
    reviewer = _Budgeted(rl.dry_run_responder)
    rl.review_run(run_dir, reviewer, progress=False)
    assert reviewer.budgets == [rl.MAX_TOKENS] * 3
    # Above the client default of 8,000 and below the anthropic SDK's
    # non-streaming guard, which refuses more than 10 minutes at 128k tokens/hour.
    assert 8000 < rl.MAX_TOKENS <= 10 * 60 * 128_000 // (60 * 60)


def test_a_truncated_record_is_reviewed_again_in_halves_of_its_candidates(tmp_path):
    run_dir, _ = standard_run(tmp_path)
    whole = rl.review_run(run_dir, StubClient(own_misses_responder), progress=False)
    rows_whole = read_jsonl(whole)
    summary_whole = rl.summarize(run_dir, whole)
    assert json.loads(rl.review_meta_path(whole).read_text())["n_split_records"] == 0

    # r1 has five candidates: 5 -> 2 + 3 -> 2 + (1 + 2).
    reviewer = _Budgeted(own_misses_responder, max_candidates=2)
    path = rl.review_run(run_dir, reviewer, progress=False, max_workers=1)
    rows = read_jsonl(path)
    assert not [r for r in rows if r["kind"] == "error"]
    assert reviewer.sizes == [5, 2, 3, 1, 2, 1, 1]

    def kind(rs, k):
        return [r for r in rs if r["kind"] == k]

    # Every candidate keeps its own verdict and the halves' misses are joined.
    assert kind(rows, "span") == kind(rows_whole, "span")
    assert kind(rows, "missed") == kind(rows_whole, "missed")
    assert [r["text"] for r in kind(rows, "missed")] == ["123-45-6789"]
    # A half that lists the other half's PII as missed is dropped once per value.
    dropped = sorted((r["value"], r["reason"]) for r in kind(rows, "dropped") if r["id"] == "r1")
    assert dropped == [
        ("Jane Doe", "covered_by_candidate"),
        ("jane@x.com", "covered_by_candidate"),
        ("not in text", "not_found"),
    ]
    assert rl.summarize(run_dir, path) == summary_whole
    meta = json.loads(rl.review_meta_path(path).read_text())
    assert meta["n_split_records"] == 1 and meta["n_errors"] == 0


def test_only_truncation_splits_and_one_candidate_is_not_split(tmp_path):
    preds = {"a": {"r1": [_span(T1, "Jane Doe", "PERSON"), _span(T1, "jane@x.com", "EMAIL")],
                   "r2": [_span(T2, "Bob", "PERSON")],
                   "r3": []}}
    run_dir, _ = make_run(tmp_path, {"r1": T1, "r2": T2, "r3": T3}, preds)

    refused = _Budgeted(rl.dry_run_responder, fail=LLMError("refused: test"))
    path = rl.review_run(run_dir, refused, progress=False, max_workers=1)
    assert refused.sizes == [2, 1, 0]  # one call per record, no split
    assert [r["kind"] for r in read_jsonl(path)] == ["error"] * 3

    cut = _Budgeted(rl.dry_run_responder, fail=LLMError("hit max_tokens", truncated=True))
    path = rl.review_run(run_dir, cut, progress=False, max_workers=1)
    # r1 splits once and its first single candidate stops early again, which
    # fails the record. r2 has one candidate and r3 has none (it only asks for
    # misses), so neither can be split.
    assert cut.sizes == [2, 1, 1, 0]
    assert [r["kind"] for r in read_jsonl(path)] == ["error"] * 3
    assert json.loads(rl.review_meta_path(path).read_text())["n_split_records"] == 1


def test_an_out_of_range_index_in_a_half_does_not_judge_the_other_half():
    text = "Ann, Bob, Cy and Di."
    spans = [
        {**_span(text, name, "PERSON"), "detectors": ["a"]} for name in ("Ann", "Bob", "Cy", "Di")
    ]

    class Loose:
        """Answers without schema checks, as a provider that drops index limits might."""

        provider, model = "loose", "m"

        def complete_json(self, *, system, user, schema, name="result", max_tokens=8000):
            got = [t for _, _, t in candidates(user)]
            if len(got) > 2:
                raise LLMError("hit max_tokens", truncated=True)
            if got == ["Ann", "Bob"]:
                # Index 3 does not exist in this half.
                return {"verdicts": [
                    {"index": i, "verdict": "not_pii", "label": None, "note": ""} for i in (1, 2, 3)
                ], "missed": []}
            return {"verdicts": [{"index": 2, "verdict": "correct", "label": None, "note": ""}],
                    "missed": []}

    rows, split = rl._review_record(Loose(), {"id": "r", "text": text}, spans, ["PERSON"])
    assert split is True
    assert [(r["text"], r["verdict"]) for r in rows] == [
        ("Ann", "not_pii"), ("Bob", "not_pii"), ("Cy", rl.UNJUDGED), ("Di", "correct"),
    ]


def test_a_half_does_not_list_a_candidate_the_other_half_rejected_as_missed():
    text = "Ann met Acme Corp."
    spans = [
        {**_span(text, "Ann", "PERSON"), "detectors": ["a"]},
        {**_span(text, "Acme Corp", "PERSON"), "detectors": ["a"]},
    ]

    class Halves:
        """Stops early on both candidates and lists the candidate it cannot see as missed."""

        provider, model = "halves", "m"

        def complete_json(self, *, system, user, schema, name="result", max_tokens=8000):
            got = [t for _, _, t in candidates(user)]
            if len(got) > 1:
                raise LLMError("hit max_tokens", truncated=True)
            verdict = "correct" if got == ["Ann"] else "not_pii"
            other = "Acme Corp" if got == ["Ann"] else "Ann"
            return {
                "verdicts": [{"index": 1, "verdict": verdict, "label": None, "note": ""}],
                "missed": [{"value": other, "label": "PERSON", "context": text}],
            }

    rows, split = rl._review_record(Halves(), {"id": "r", "text": text}, spans, ["PERSON"])
    assert split is True
    assert [(r["text"], r["verdict"]) for r in rows if r["kind"] == "span"] == [
        ("Ann", "correct"), ("Acme Corp", "not_pii"),
    ]
    # The half that judged Acme Corp rejected it, so the other half's miss is dropped.
    assert not [r for r in rows if r["kind"] == "missed"]
    assert sorted((r["value"], r["reason"]) for r in rows if r["kind"] == "dropped") == [
        ("Acme Corp", "judged_not_pii"), ("Ann", "covered_by_candidate"),
    ]


def test_review_locates_misses_with_silver_locate():
    from opf_eval import silver

    assert rl.locate is silver.locate


def test_missing_and_repeated_indexes_are_unjudged(tmp_path):
    run_dir, _ = standard_run(tmp_path)

    def respond(system, user, schema):
        n = len(candidates(user))
        # Every verdict points at candidate 1, so the others get none.
        return {
            "verdicts": [{"index": 1, "verdict": "correct", "label": None, "note": ""}] * n,
            "missed": [],
        }

    path = rl.review_run(run_dir, StubClient(respond), progress=False, max_workers=1)
    r1 = [r for r in read_jsonl(path) if r["id"] == "r1" and r["kind"] == "span"]
    assert [r["verdict"] for r in r1] == ["correct"] + [rl.UNJUDGED] * 4
    summary = {r["detector"]: r for r in rl.summarize(run_dir, path)}
    # Unjudged spans are left out: b's only judged span in r1 is "Contact" (index 1).
    assert summary["b"]["n_spans"] == 2  # Contact in r1 + sue@y.org in r3


def test_missed_values_are_located_or_dropped(tmp_path):
    text = "Ann paid with 4111 1111 1111 1111 and ann@q.com. Ann again."
    preds = {"a": {"r1": [_span(text, "ann@q.com", "EMAIL")]}}
    run_dir, _ = make_run(tmp_path, {"r1": text}, preds, labels=["CREDIT_CARD", "EMAIL", "PERSON"])
    missed = [
        # The second occurrence is found through the context.
        {"value": "Ann", "label": "PERSON", "context": "Ann again."},
        {"value": "4111 1111 1111 1111", "label": "CREDIT_CARD", "context": "paid with 4111"},
        {"value": "ann@q.com", "label": "EMAIL", "context": "and ann@q.com"},  # already a candidate
        {"value": "not in text", "label": "PERSON", "context": ""},
    ]
    reviewer = StubClient(responder_from({"ann@q.com": ("correct", None)}, {"Ann paid": missed}))
    path = rl.review_run(run_dir, reviewer, progress=False)
    rows = read_jsonl(path)
    found = sorted((r["text"], r["start"]) for r in rows if r["kind"] == "missed")
    assert ("4111 1111 1111 1111", 14) in found
    assert ("Ann", text.rindex("Ann")) in found
    dropped = {(r["value"], r["reason"]) for r in rows if r["kind"] == "dropped"}
    assert ("not in text", "not_found") in dropped
    assert ("ann@q.com", "covered_by_candidate") in dropped
    meta = json.loads((run_dir / (path.stem + ".meta.json")).read_text())
    assert meta["n_dropped"] == len(dropped)


def test_blank_records_are_not_sent(tmp_path):
    preds = {"a": {"r1": [], "r2": []}}
    run_dir, _ = make_run(tmp_path, {"r1": "Hi Tom.", "r2": "   \n"}, preds)
    reviewer = StubClient(rl.dry_run_responder)
    rl.review_run(run_dir, reviewer, progress=False)
    assert len(reviewer.calls) == 1


def test_cached_reviewer_reruns_without_calls(tmp_path):
    run_dir, _ = standard_run(tmp_path)
    inner = StubClient(responder_from(STANDARD_VERDICTS, STANDARD_MISSED))
    first = rl.review_run(run_dir, CachedClient(inner, tmp_path / "cache"), progress=False)
    rows_first = read_jsonl(first)
    cached = CachedClient(inner, tmp_path / "cache")
    second = rl.review_run(run_dir, cached, progress=False)
    assert cached.hits == 3 and cached.misses == 0 and len(inner.calls) == 3
    assert read_jsonl(second) == rows_first


# ---------------------------------------------------------------- summary


def test_summarize_maths(tmp_path):
    run_dir, _ = standard_run(tmp_path)
    reviewer = StubClient(responder_from(STANDARD_VERDICTS, STANDARD_MISSED, fail_on="Bob"))
    path = rl.review_run(run_dir, reviewer, progress=False)
    s = {r["detector"]: r for r in rl.summarize(run_dir, path)}
    # r2 failed review, so both detectors are scored on r1 and r3 only.
    # Entities in r1: Jane Doe, the e-mail (jane@x.com + jane@x overlap), 555-1234
    # (wrong_label is still PII) and the missed SSN = 4; r3: sue@y.org = 1.
    a, b = s["a"], s["b"]
    assert (a["n_records"], a["n_spans"], a["correct"]) == (2, 3, 3)
    assert a["precision"] == 1.0
    assert a["recall"] == pytest.approx(3 / 5)
    assert a["f1"] == pytest.approx(2 * 1.0 * 0.6 / 1.6)
    assert a["missed_by_all"] == 1
    counts = (b["n_spans"], b["correct"], b["wrong_label"], b["boundary"], b["not_pii"])
    assert counts == (5, 2, 1, 1, 1)
    assert b["precision"] == pytest.approx(2 / 5)
    assert b["recall"] == pytest.approx(2 / 5)  # Jane Doe + sue@y.org
    lenient = {r["detector"]: r for r in rl.summarize(run_dir, path, lenient=True)}
    assert lenient["b"]["precision"] == pytest.approx(3 / 5)
    # The e-mail entity counts too because of the boundary span.
    assert lenient["b"]["recall"] == pytest.approx(3 / 5)
    assert lenient["a"]["precision"] == 1.0


def test_a_span_bridging_two_entities_does_not_merge_them(tmp_path):
    # "long" swallows the name and the e-mail in one boundary span. Without
    # care the two true entities chain into one through it, and a detector
    # that found only the e-mail would get credit for the name as well.
    text = "Reach Jane Doe jane@x.com today."
    preds = {
        "long": {"r1": [_span(text, "Jane Doe jane@x.com", "PERSON")]},
        "both": {"r1": [_span(text, "Jane Doe", "PERSON"), _span(text, "jane@x.com", "EMAIL")]},
        "email_only": {"r1": [_span(text, "jane@x.com", "EMAIL")]},
    }
    run_dir, _ = make_run(tmp_path, {"r1": text}, preds)
    verdicts = {
        "Jane Doe jane@x.com": ("boundary", None),
        "Jane Doe": ("correct", None),
        "jane@x.com": ("correct", None),
    }
    path = rl.review_run(run_dir, StubClient(responder_from(verdicts)), progress=False)
    s = {r["detector"]: r for r in rl.summarize(run_dir, path)}
    assert s["email_only"]["recall"] == pytest.approx(1 / 2)
    assert s["both"]["recall"] == 1.0
    assert s["long"]["recall"] == 0.0
    lenient = {r["detector"]: r for r in rl.summarize(run_dir, path, lenient=True)}
    # The boundary span earns at most one entity: the one it overlaps most.
    assert lenient["long"]["recall"] == pytest.approx(1 / 2)
    assert lenient["email_only"]["recall"] == pytest.approx(1 / 2)


def test_a_miss_inside_an_over_long_boundary_span_of_another_entity_stays_missed(tmp_path):
    # B's boundary span belongs to Jane Doe (A's correct span) and runs into
    # John Roe, which no detector found on its own. John Roe must stay a miss.
    text = "Jane Doe met John Roe today"
    preds = {
        "A": {"r1": [_span(text, "Jane Doe", "PERSON")]},
        "B": {"r1": [_span(text, "Jane Doe met John", "PERSON")]},
    }
    run_dir, _ = make_run(tmp_path, {"r1": text}, preds)
    verdicts = {"Jane Doe": ("correct", None), "Jane Doe met John": ("boundary", None)}
    missed = {
        "John Roe": [{"value": "John Roe", "label": "PERSON", "context": "met John Roe today"}]
    }
    path = rl.review_run(run_dir, StubClient(responder_from(verdicts, missed)), progress=False)
    rows = read_jsonl(path)
    assert [(r["kind"], r.get("text")) for r in rows if r["kind"] != "span"] == [
        ("missed", "John Roe")
    ]
    strict = {r["detector"]: r for r in rl.summarize(run_dir, path)}
    assert strict["A"]["recall"] == pytest.approx(1 / 2)
    assert strict["A"]["missed_by_all"] == 1
    lenient = {r["detector"]: r for r in rl.summarize(run_dir, path, lenient=True)}
    assert lenient["B"]["recall"] == pytest.approx(1 / 2)
    # The prompt asks the reviewer for this kind of miss.
    system, _ = rl.build_prompt(text, [], LABELS)
    assert 'a "boundary" candidate runs into' in system.replace("\n", " ")


@pytest.mark.parametrize("candidates,miss,covered", [
    ([(0, 17, "boundary")], (13, 21), True),  # the boundary span is this entity
    ([(0, 8, "correct"), (0, 17, "boundary")], (13, 21), False),  # it belongs to Jane Doe
    ([(0, 8, "wrong_label"), (0, 17, "boundary")], (13, 21), False),
    ([(0, 8, "correct"), (0, 21, "boundary")], (13, 21), True),  # holds all of the miss
    ([(10, 18, "correct")], (13, 21), True),
    ([(10, 18, "not_pii")], (13, 21), False),
])
def test_covered_rules(candidates, miss, covered):
    rows = [{"start": a, "end": b, "verdict": v} for a, b, v in candidates]
    assert rl._covered(miss, rows) is covered


def test_review_rows_name_a_fallback_model(tmp_path):
    from opf_eval.llm.base import FallbackLog

    run_dir, _ = standard_run(tmp_path)

    class Rescued(StubClient):
        def __init__(self):
            super().__init__(responder_from(STANDARD_VERDICTS, STANDARD_MISSED))
            self._log = FallbackLog()

        def complete_json(self, **kw):
            answer = super().complete_json(**kw)
            self._log.note("claude-sonnet-4-5" if "Bob" in kw["user"] else None)
            return answer

        def pop_fallbacks(self):
            return self._log.pop()

    path = rl.review_run(run_dir, Rescued(), progress=False)
    rows = read_jsonl(path)
    assert {r["id"] for r in rows if r.get("fallback_model") == "claude-sonnet-4-5"} == {"r2"}
    assert json.loads(rl.review_meta_path(path).read_text())["n_fallback_records"] == 1


def test_docs_and_plan_name_the_current_prompt_version():
    root = Path(__file__).resolve().parents[2]
    for doc in ("eval/docs/your-own-data.md", "plans/13-unlabeled-data-notebook.md"):
        named = re.findall(r"prompt(?: version)? is `(review-v\d+)`", (root / doc).read_text())
        assert named == [rl.PROMPT_VERSION], doc


def test_review_prompt_embeds_its_version(monkeypatch):
    from opf_eval.llm.cache import cache_key

    def key():
        system, user = rl.build_prompt(T1, [], LABELS)
        return cache_key(provider="p", model="m", system=system, user=user,
                         schema=rl.judge_schema(1, LABELS), name="review")

    before = key()
    assert rl.PROMPT_VERSION in rl.build_prompt(T1, [], LABELS)[0]
    monkeypatch.setattr(rl, "PROMPT_VERSION", "review-test")
    assert key() != before


def test_entities_attach_each_other_span_to_one_anchor():
    def sp(a, b, verdict):
        return {"start": a, "end": b, "verdict": verdict}
    groups = rl._entities([
        sp(0, 8, "correct"), sp(9, 19, "correct"), sp(0, 19, "boundary"),
        sp(30, 35, "wrong_label"), sp(32, 40, "boundary"), sp(5, 7, "correct"),
    ])
    assert [[(s["start"], s["end"]) for s in g] for g in groups] == [
        [(0, 8), (5, 7)], [(9, 19), (0, 19)], [(30, 35), (32, 40)],
    ]


def test_pii_inside_a_not_pii_candidate_stays_missed(tmp_path):
    text = "Invoice for Jane Doe due today."
    preds = {
        "wide": {"r1": [_span(text, "Invoice for Jane Doe", "PERSON")]},
        "none": {"r1": []},
    }
    run_dir, _ = make_run(tmp_path, {"r1": text}, preds)
    missed = {"Invoice": [{"value": "Jane Doe", "label": "PERSON", "context": "for Jane Doe due"}]}
    reviewer = StubClient(responder_from({"Invoice for Jane Doe": ("not_pii", None)}, missed))
    path = rl.review_run(run_dir, reviewer, progress=False)
    rows = read_jsonl(path)
    assert [(r["kind"], r.get("text")) for r in rows if r["kind"] != "span"] == [
        ("missed", "Jane Doe")
    ]
    s = {r["detector"]: r for r in rl.summarize(run_dir, path)}
    assert s["none"]["recall"] == 0.0 and s["none"]["missed_by_all"] == 1
    assert s["wide"]["recall"] == 0.0 and s["wide"]["precision"] == 0.0
    # The prompt asks for exactly these misses.
    system, _ = rl.build_prompt(text, [], LABELS)
    assert 'inside a candidate you judged "not_pii"' in system.replace("\n", " ")


def test_detectors_may_be_a_generator(tmp_path):
    run_dir, _ = standard_run(tmp_path)
    from_generator = rl.estimate(run_dir, detectors=(d for d in ["a"]))
    assert from_generator == rl.estimate(run_dir, detectors=["a"])
    assert rl.estimate(run_dir, detectors=["a"])["n_candidates"] == 4
    path = rl.review_run(
        run_dir, StubClient(rl.dry_run_responder), detectors=(d for d in ["a"]), progress=False
    )
    meta = json.loads((run_dir / (path.stem + ".meta.json")).read_text())
    assert meta["detectors"] == ["a"] and meta["n_spans"] == 4


def test_summarize_excludes_records_a_detector_errored_on(tmp_path):
    run_dir, _ = standard_run(tmp_path)
    reviewer = StubClient(responder_from(STANDARD_VERDICTS, STANDARD_MISSED))
    path = rl.review_run(run_dir, reviewer, progress=False)
    # A bare name resolves inside run_dir.
    s = {r["detector"]: r for r in rl.summarize(run_dir, path.name)}
    assert s["a"]["n_records"] == 3
    assert s["b"]["n_records"] == 2  # b errored on r2
    # a: r1 4 entities + r2 Bob + r3 sue = 6; found Jane Doe, jane@x.com, Bob, sue = 4.
    assert s["a"]["recall"] == pytest.approx(4 / 6)


def test_summarize_with_no_spans_gives_none(tmp_path):
    run_dir, _ = make_run(tmp_path, {"r1": "plain words"}, {"a": {"r1": []}})
    path = rl.review_run(run_dir, StubClient(rl.dry_run_responder), progress=False)
    (row,) = rl.summarize(run_dir, path)
    assert row["n_records"] == 1 and row["n_spans"] == 0
    assert row["precision"] is None and row["recall"] is None and row["f1"] is None


def test_disagreements_have_context_and_document_offsets(tmp_path):
    records = {
        "d1#0": {"text": T1, "doc_id": "abc:file.txt", "offset": 1000},
        "r3": T3,
    }
    preds = {"b": {
        "d1#0": [_span(T1, "Contact", "PERSON"), _span(T1, "Jane Doe", "PERSON")], "r3": []
    }}
    run_dir, _ = make_run(tmp_path, records, preds)
    reviewer = StubClient(responder_from(STANDARD_VERDICTS, STANDARD_MISSED))
    path = rl.review_run(run_dir, reviewer, progress=False)
    rows = rl.disagreements(run_dir, path)
    assert [(r["kind"], r["verdict"], r["text"]) for r in rows] == [
        ("span", "not_pii", "Contact"),
        ("missed", "missed", "123-45-6789"),
    ]
    assert rows[0]["context"].startswith("⟦Contact⟧ Jane Doe")
    assert rows[0]["doc_id"] == "abc:file.txt" and rows[0]["doc_start"] == 1000
    assert rows[1]["doc_end"] == 1000 + T1.index("123-45") + 11
    assert rl.disagreements(run_dir, path, limit=1) == rows[:1]
    counts = rl.verdict_counts(path)
    assert counts == {"not_pii": 1, "correct": 1, "missed": 1}


def test_estimate_counts_calls_and_candidates(tmp_path):
    run_dir, _ = standard_run(tmp_path)
    est = rl.estimate(run_dir)
    assert est == {"n_calls": 3, "n_candidates": 7, "n_chars": len(T1) + len(T2) + len(T3)}


def test_labels_default_to_fixtures_meta_and_scope_candidates(tmp_path):
    preds = {"a": {"r1": [_span(T1, "Jane Doe", "PERSON"), _span(T1, "jane@x.com", "EMAIL")]}}
    run_dir, _ = make_run(tmp_path, {"r1": T1}, preds, labels=["EMAIL"])
    reviewer = StubClient(rl.dry_run_responder)
    path = rl.review_run(run_dir, reviewer, progress=False)
    system, user = reviewer.calls[0]
    assert "- EMAIL:" in system and "- PERSON:" not in system
    assert [c[2] for c in candidates(user)] == ["jane@x.com"]
    assert json.loads(path.with_name(path.stem + ".meta.json").read_text())["labels"] == ["EMAIL"]
    with pytest.raises(ValueError, match="unknown label"):
        rl.review_run(run_dir, reviewer, labels=["BOGUS"], progress=False)


def test_labels_fall_back_to_manifest_then_taxonomy(tmp_path):
    from opf_eval import taxonomy
    from opf_eval.review.pool import target_labels

    run_dir, fixtures = make_run(tmp_path, {"r1": T1}, {"a": {"r1": []}}, labels=["EMAIL"])
    meta_path(fixtures).unlink()
    run = scoring.Run.load(run_dir)
    assert target_labels(run) == ["EMAIL"]  # from the manifest
    run.manifest.pop("labels")
    run.manifest.pop("fixtures")
    assert target_labels(run) == sorted(taxonomy.ALL_LABELS)
    assert target_labels(run, ["PHONE", "EMAIL", "PHONE"]) == ["EMAIL", "PHONE"]


def test_missing_run_dir_is_a_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="manifest.json"):
        rl.review_run(tmp_path / "nope", StubClient(rl.dry_run_responder), progress=False)


# -------------------------------------------- through the real pipeline

register_vocab(
    "review_test_vocab", {"mail": "EMAIL", "phone": "PHONE"}, kind="detector", overwrite=True
)
_MAIL = re.compile(r"[\w.]+@[\w.]+\w")
_PHONE = re.compile(r"\d{3}-\d{4}")


class _RegexDetector:
    name = "review_regex"

    def detect(self, text: str, **_: object) -> dict:
        spans = [
            make_span("review_test_vocab", "mail", m.start(), m.end(), m.group())
            for m in _MAIL.finditer(text)
        ]
        spans += [
            make_span("review_test_vocab", "phone", m.start(), m.end(), m.group())
            for m in _PHONE.finditer(text)
        ]
        return {"spans": spans, "latency_ms": 0.1, "error": None}


@registry.detector("review_regex", vocab="review_test_vocab", overwrite=True)
def _build_regex(ctx):
    return _RegexDetector()


def test_review_on_unlabeled_fixtures_from_texts(tmp_path):
    fixtures = fx.from_texts(
        {"t1": "Mail ann@q.com or call 555-1234.", "t2": "Nothing to see. Bob Ray signed."},
        tmp_path / "custom.jsonl",
        labels=["EMAIL", "PERSON", "PHONE"],
    )
    run_dir = tmp_path / "run"
    run_detectors(fixtures, ["review_regex"], run_dir)
    reviewer = StubClient(responder_from(
        {"ann@q.com": ("correct", None), "555-1234": ("correct", None)},
        {"Bob Ray": [{"value": "Bob Ray", "label": "PERSON", "context": "Bob Ray signed"}]},
    ))
    path = rl.review_run(run_dir, reviewer, progress=False)
    (row,) = rl.summarize(run_dir, path)
    assert row["detector"] == "review_regex"
    assert row["precision"] == 1.0
    assert row["recall"] == pytest.approx(2 / 3)
    assert row["missed_by_all"] == 1
