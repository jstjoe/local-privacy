"""Classifier review (plan 13, Part 6) with `StubBackend`: request builders,
segmenting and masking, the review file, summary maths and the threshold
sweep against gold. No model is loaded."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from opf_eval.io import meta_path, read_jsonl, write_jsonl
from opf_eval.review import DecisionBackend, StubBackend
from opf_eval.review import classifier as rc
from opf_eval.review.classifier import choice_answer, noul_answer

LABELS = ["EMAIL", "GOV_ID", "PERSON", "PHONE"]


def _span(text: str, value: str, label: str, fine: str | None = None) -> dict:
    start = text.index(value)
    return {"label": label, "fine_label": fine or label, "raw_label": label.lower(),
            "start": start, "end": start + len(value), "text": value}


def make_run(root: Path, records: dict[str, str], preds: dict, *, labels=LABELS, gold=None) -> tuple[Path, Path]:
    fixtures = root / "fixtures.jsonl"
    write_jsonl(fixtures, [
        {"id": rid, "text": text, "language": None, "gold_spans": (gold or {}).get(rid, [])}
        for rid, text in records.items()
    ])
    meta_path(fixtures).write_text(json.dumps({"gold": "none" if gold is None else None, "labels": labels}))
    run_dir = root / "run"
    run_dir.mkdir()
    (run_dir / "manifest.json").write_text(json.dumps({
        "fixtures": str(fixtures), "labels": labels, "detectors": sorted(preds),
    }))
    for det, by_id in preds.items():
        write_jsonl(run_dir / f"raw_{det}.jsonl", [
            {"id": rid, "detector": det, "spans": [] if isinstance(s, str) else s,
             "latency_ms": 1.0, "error": s if isinstance(s, str) else None}
            for rid, s in by_id.items()
        ])
    return run_dir, fixtures


# --------------------------------------------------------------- requests

TEXT = "Dear Jane Doe,\nplease call 555-1234 or mail jane@x.com.\nThanks."


def test_span_request_shape():
    span = _span(TEXT, "555-1234", "PHONE")
    req = rc.span_request(TEXT, span, LABELS)
    assert req["model"] == rc.DEFAULT_MODEL
    assert req["state"]["span"] == "555-1234" and req["state"]["proposed_label"] == "PHONE"
    assert "call ⟦555-1234⟧ or" in req["state"]["context"]
    q = req["questions"]
    assert q["is_pii"] == {"type": "noul", "instructions": rc.IS_PII_Q}
    assert q["label_ok"]["type"] == "noul"
    assert q["label_ok"]["instructions"] == "Is PHONE (Phone and fax numbers, IMEI) the right type for it?"
    # Coarse categories of the target labels, in taxonomy order, plus NOT_PII.
    assert list(q["label"]["criteria"]) == ["PERSON", "EMAIL", "PHONE", "ACCOUNT", "NOT_PII"]
    assert q["label"]["type"] == "choice"


def test_span_request_window_marks_cuts():
    text = "x" * 50 + "ann@q.com" + "y" * 50
    req = rc.span_request(text, _span(text, "ann@q.com", "EMAIL"), LABELS, window=10)
    assert req["state"]["context"] == "…" + "x" * 10 + "⟦ann@q.com⟧" + "y" * 10 + "…"
    whole = rc.span_request(text, _span(text, "ann@q.com", "EMAIL"), LABELS, window=500)
    assert "…" not in whole["state"]["context"]


def test_split_segments_lines_sentences_and_trimming():
    text = "  First line.  \n\n---\nSecond. Third sentence here. Fourth!\n"
    segs = rc.split_segments(text, max_chars=20)
    pieces = [text[a:b] for a, b in segs]
    assert pieces == ["First line.", "Second.", "Third sentence here.", "Fourth!"]
    assert all(b - a <= 20 for a, b in segs)
    assert rc.split_segments("", max_chars=10) == []
    with pytest.raises(ValueError):
        rc.split_segments("x", max_chars=0)


def test_split_segments_never_cuts_inside_a_span():
    text = "Ship to 12 Long Street\nSpringfield now please. Thank you very much."
    addr = {"start": text.index("12"), "end": text.index("Springfield") + len("Springfield"), "label": "ADDRESS"}
    segs = rc.split_segments(text, max_chars=200, avoid=[addr])
    # The newline sits inside the address, so the two lines stay one segment.
    # The sentence end after "please." still splits.
    assert [text[a:b] for a, b in segs] == [
        "Ship to 12 Long Street\nSpringfield now please.", "Thank you very much.",
    ]
    long_span = {"start": 0, "end": 40, "label": "PERSON"}
    text2 = "a" * 40 + " tail words here"
    segs2 = rc.split_segments(text2, max_chars=10, avoid=[long_span])
    assert segs2[0] == (0, 40)  # a span longer than max_chars stays whole
    # Plain text with no spaces gets a hard cut at the limit.
    assert rc.split_segments("abcdefghij" * 3, max_chars=10) == [(0, 10), (10, 20), (20, 30)]
    # Tuples work as well as span dicts.
    assert rc.split_segments(text, max_chars=200, avoid=[(addr["start"], addr["end"])]) == segs


def test_split_segments_splits_every_sentence_not_just_long_lines():
    text = "My name is Jane Doe. I live at 4 Elm St. Call me on 555-1234! Ok?"
    pieces = [text[a:b] for a, b in rc.split_segments(text)]
    # One segment per sentence, so the residual rate is per sentence. "St."
    # is an abbreviation and does not end a sentence.
    assert pieces == ["My name is Jane Doe.", "I live at 4 Elm St. Call me on 555-1234!", "Ok?"]
    # Initials and titles stay with the name; a semicolon does not split.
    text2 = "Dr. J. Smith called; he left 555-0000. Done."
    assert [text2[a:b] for a, b in rc.split_segments(text2)] == ["Dr. J. Smith called; he left 555-0000.", "Done."]
    # A sentence end inside a span does not split it.
    text3 = "Ask for Acme Inc. Ltd. Sales now. Bye."
    span = (text3.index("Acme"), text3.index(" now"))
    assert [text3[a:b] for a, b in rc.split_segments(text3, avoid=[span])] == ["Ask for Acme Inc. Ltd. Sales now.", "Bye."]


def test_split_segments_never_cuts_inside_overlapping_spans():
    # Two detectors' spans overlap across the limit. The hard cut must move
    # out of their union, not just out of the last span it checked.
    assert rc.split_segments("x" * 650, max_chars=600, avoid=[(580, 598), (590, 620)]) == [(0, 580), (580, 650)]
    segs = rc.split_segments("x" * 40, max_chars=8, avoid=[(5, 15), (10, 20)])
    assert all(not (s < a < e) and not (s < b < e) for a, b in segs for s, e in [(5, 20)])
    import random
    rnd = random.Random(7)
    for _ in range(2000):
        n = rnd.randint(1, 120)
        text = "".join(rnd.choice("ab .\n") for _ in range(n))
        spans = []
        for _ in range(rnd.randint(0, 6)):
            s = rnd.randint(0, n - 1)
            spans.append((s, rnd.randint(s + 1, n)))
        segs = rc.split_segments(text, max_chars=rnd.randint(1, 30), avoid=spans)
        for s, e in spans:
            # Trimming may drop whitespace at a span's edge, but no span is
            # ever shared by two segments.
            assert sum(1 for a, b in segs if a < e and s < b) <= 1, (text, spans, segs)


def test_mask_replaces_clips_and_merges():
    spans = [_span(TEXT, "Jane Doe", "PERSON", "GIVEN_NAME"), _span(TEXT, "jane@x.com", "EMAIL")]
    assert rc.mask(TEXT, 0, len(TEXT), spans) == "Dear [PERSON],\nplease call 555-1234 or mail [EMAIL].\nThanks."
    # Clipped to the segment: half the name sits inside it.
    a = TEXT.index("Doe")
    assert rc.mask(TEXT, a, a + 4, spans) == "[PERSON],"
    overlap = [{"start": 5, "end": 13, "label": "PERSON"}, {"start": 10, "end": 15, "label": "PHONE"}]
    # The overlap swallows ",\n" (positions 13-14) up to the end of the second span.
    assert rc.mask(TEXT, 0, 20, overlap) == "Dear [PERSON]pleas"


def test_segment_requests_shape_and_fixed_bounds():
    spans = [_span(TEXT, "Jane Doe", "PERSON")]
    reqs = rc.segment_requests(TEXT, spans, LABELS)
    assert [r[2]["state"]["text"] for r in reqs] == [
        "Dear [PERSON],", "please call 555-1234 or mail jane@x.com.", "Thanks.",
    ]
    q = reqs[0][2]["questions"]
    assert q["residual"] == {"type": "noul", "instructions": rc.RESIDUAL_Q}
    assert list(q["residual_type"]["criteria"])[-1] == "NONE"
    fixed = rc.segment_requests(TEXT, [], LABELS, segments=[(0, 4)])
    assert [(a, b, r["state"]["text"]) for a, b, r in fixed] == [(0, 4, "Dear")]


def test_stub_backend_defaults_and_helpers():
    backend = StubBackend()
    assert isinstance(backend, DecisionBackend)
    req = rc.span_request(TEXT, _span(TEXT, "Jane Doe", "PERSON"), LABELS)
    (res,) = backend.answer([req])
    assert res["answers"]["is_pii"] == {"type": "noul", "noul": 0.5}
    probs = res["answers"]["label"]["probabilities"]
    assert set(probs) == set(req["questions"]["label"]["criteria"])
    assert sum(probs.values()) == pytest.approx(1.0, abs=1e-3)
    assert backend.requests == [req] and backend.batches == [1]
    assert choice_answer({"A": 0.2, "B": 0.8}) == {
        "type": "choice", "choice": "B", "confidence": 0.8, "probabilities": {"A": 0.2, "B": 0.8},
    }
    assert noul_answer(0.123456) == {"type": "noul", "noul": 0.1235}
    full = StubBackend(lambda r: {"model": "m", "answers": {"x": noul_answer(1)}}).answer([req])
    assert full == [{"model": "m", "answers": {"x": {"type": "noul", "noul": 1.0}}}]


# ----------------------------------------------------------------- review

T1 = "Contact Jane Doe at jane@x.com.\nCall 555-1234 today."
T2 = "Bob wrote. SSN 123-45-6789 is his."
# What the stub model believes, keyed by span text: (p_pii, p_label_ok).
BELIEFS = {"Jane Doe": (0.9, 0.9), "jane@x.com": (0.95, 0.8), "Contact": (0.2, 0.3),
           "555-1234": (0.8, 0.35), "Bob": (0.7, 0.6)}


def oracle(req: dict) -> dict:
    """Span questions answer from BELIEFS; a segment is residual when a digit or '@' survives masking."""
    if "is_pii" in req["questions"]:
        p_pii, p_ok = BELIEFS[req["state"]["span"]]
        return {
            "is_pii": noul_answer(p_pii), "label_ok": noul_answer(p_ok),
            "label": choice_answer({req["state"]["proposed_label"]: p_pii, "NOT_PII": 1 - p_pii}),
        }
    text = re.sub(r"\[[A-Z_]+\]", "", req["state"]["text"])
    leftover = bool(re.search(r"\d|@", text))
    return {
        "residual": noul_answer(0.9 if leftover else 0.1),
        "residual_type": choice_answer({"PHONE": 0.9, "NONE": 0.1} if leftover else {"NONE": 0.9, "PHONE": 0.1}),
    }


def standard_run(tmp_path: Path, gold=None):
    preds = {
        "a": {"r1": [_span(T1, "Jane Doe", "PERSON"), _span(T1, "jane@x.com", "EMAIL"), _span(T1, "555-1234", "PHONE")],
              "r2": [_span(T2, "Bob", "PERSON")]},
        "b": {"r1": [_span(T1, "Jane Doe", "PERSON"), _span(T1, "Contact", "PERSON")],
              "r2": "timeout"},
    }
    return make_run(tmp_path, {"r1": T1, "r2": T2}, preds, gold=gold)


def test_review_run_rows_and_meta(tmp_path):
    run_dir, _ = standard_run(tmp_path)
    backend = StubBackend(oracle, name="oracle")
    path = rc.review_run(run_dir, backend, batch_size=2, progress=False)
    assert path == run_dir / "review_oracle.jsonl"
    rows = read_jsonl(path)
    spans = [r for r in rows if r["kind"] == "span"]
    # Jane Doe is asked about once although both detectors found it.
    assert sorted(r["text"] for r in spans) == ["555-1234", "Bob", "Contact", "Jane Doe", "jane@x.com"]
    jane = next(r for r in spans if r["text"] == "Jane Doe")
    assert jane["detectors"] == ["a", "b"]
    assert (jane["p_pii"], jane["p_label_ok"], jane["best_label"], jane["p_best"]) == (0.9, 0.9, "PERSON", 0.9)
    segs = [r for r in rows if r["kind"] == "segment"]
    # r1 has two lines (both detectors), r2 two sentences (a only: b errored there).
    assert [(r["id"], r["detector"]) for r in segs] == [
        ("r1", "a"), ("r1", "a"), ("r1", "b"), ("r1", "b"), ("r2", "a"), ("r2", "a"),
    ]
    a_r1 = [r for r in segs if r["detector"] == "a" and r["id"] == "r1"]
    assert [r["p_residual"] for r in a_r1] == [0.1, 0.1]
    b_r1 = [r for r in segs if r["detector"] == "b" and r["id"] == "r1"]
    assert [r["p_residual"] for r in b_r1] == [0.9, 0.9]  # b left the e-mail and the phone
    assert b_r1[0]["residual_type"] == "PHONE"
    # Segment boundaries are shared across detectors.
    assert [(r["start"], r["end"]) for r in a_r1] == [(r["start"], r["end"]) for r in b_r1]
    meta = json.loads(path.with_name("review_oracle.meta.json").read_text())
    assert meta["backend"] == "oracle" and meta["threshold"] == 0.5 and meta["labels"] == LABELS
    assert meta["n_span_requests"] == 5 and meta["n_segment_requests"] == 6 and meta["n_errors"] == 0
    assert max(backend.batches) <= 2


def test_fully_masked_segments_are_not_sent(tmp_path):
    text = "jane@x.com\nCall me."
    preds = {"a": {"r1": [_span(text, "jane@x.com", "EMAIL")]}}
    run_dir, _ = make_run(tmp_path, {"r1": text}, preds)
    backend = StubBackend(oracle)
    path = rc.review_run(run_dir, backend, progress=False)
    segs = [r for r in read_jsonl(path) if r["kind"] == "segment"]
    assert [(r["p_residual"], r.get("trivial", False)) for r in segs] == [(0.0, True), (0.1, False)]
    sent = [r for r in backend.requests if "residual" in r["questions"]]
    assert [r["state"]["text"] for r in sent] == ["Call me."]


def test_failed_requests_are_isolated(tmp_path):
    run_dir, _ = standard_run(tmp_path)

    def flaky(req):
        if req["state"].get("span") == "Contact":
            raise RuntimeError("context too long")
        return oracle(req)

    backend = StubBackend(flaky)
    path = rc.review_run(run_dir, backend, batch_size=4, progress=False)
    rows = read_jsonl(path)
    errors = [r for r in rows if r["kind"] == "error"]
    assert len(errors) == 1
    assert errors[0]["pass"] == "span" and errors[0]["error"] == "RuntimeError: context too long"
    assert errors[0]["detectors"] == ["b"]
    assert len([r for r in rows if r["kind"] == "span"]) == 4  # the rest of the batch survived


def test_wrong_answer_count_is_an_error(tmp_path):
    run_dir, _ = standard_run(tmp_path)

    class Short(StubBackend):
        def answer(self, requests):
            return super().answer(requests)[:-1] if len(requests) > 1 else []

    path = rc.review_run(run_dir, Short(oracle), progress=False)
    rows = read_jsonl(path)
    assert rows and all(r["kind"] == "error" for r in rows)


def test_batch_size_below_one_is_an_error(tmp_path):
    run_dir, _ = standard_run(tmp_path)
    for bad in (0, -3):
        with pytest.raises(ValueError, match="batch_size"):
            rc.review_run(run_dir, StubBackend(oracle), batch_size=bad, progress=False)


def test_malformed_answers_become_error_rows(tmp_path):
    run_dir, _ = standard_run(tmp_path)

    def partial(req):
        res = oracle(req)
        if req["state"].get("span") == "Contact":
            del res["label_ok"]                      # a span answer missing a question
        if "residual" in req["questions"] and "SSN" in req["state"]["text"]:
            res["residual"] = {"type": "noul", "noul": None}   # a value of the wrong type
        return res

    path = rc.review_run(run_dir, StubBackend(partial), batch_size=8, progress=False)
    rows = read_jsonl(path)
    errors = [r for r in rows if r["kind"] == "error"]
    assert sorted((r["pass"], r["error"].split(":")[0]) for r in errors) == [("segment", "TypeError"), ("span", "KeyError")]
    # Every other request still has its row.
    assert len([r for r in rows if r["kind"] == "span"]) == 4
    assert len([r for r in rows if r["kind"] == "segment"]) == 5
    meta = json.loads(path.with_name(path.stem + ".meta.json").read_text())
    assert meta["n_errors"] == 2


def test_detectors_may_be_a_generator(tmp_path):
    run_dir, _ = standard_run(tmp_path)
    path = rc.review_run(run_dir, StubBackend(oracle), detectors=(d for d in ["a"]), progress=False)
    meta = json.loads(path.with_name(path.stem + ".meta.json").read_text())
    assert meta["detectors"] == ["a"] and meta["n_span_requests"] == 4


def test_summarize_maths_and_threshold_override(tmp_path):
    run_dir, _ = standard_run(tmp_path)
    path = rc.review_run(run_dir, StubBackend(oracle), progress=False)
    s = {r["detector"]: r for r in rc.summarize(run_dir, path)}
    # a: Jane Doe (0.9, 0.9) ok, jane@x.com (0.95, 0.8) ok, 555-1234 (0.8, 0.35) label not ok, Bob (0.7, 0.6) ok.
    assert s["a"]["n_spans"] == 4
    assert s["a"]["precision"] == pytest.approx(3 / 4)
    assert s["a"]["precision_any_label"] == 1.0
    assert s["a"]["mean_p_pii"] == pytest.approx((0.9 + 0.95 + 0.8 + 0.7) / 4)
    # a's r1 lines are clean after masking; in r2's second sentence the SSN survives.
    assert s["a"]["n_segments"] == 4 and s["a"]["residual_rate"] == pytest.approx(1 / 4)
    # b: Jane Doe ok, Contact (0.2, 0.3) not PII.
    assert s["b"]["precision"] == 0.5 and s["b"]["precision_any_label"] == 0.5
    assert s["b"]["n_segments"] == 2 and s["b"]["residual_rate"] == 1.0
    strict = {r["detector"]: r for r in rc.summarize(run_dir, path.name, threshold=0.85)}
    assert strict["a"]["precision"] == pytest.approx(1 / 4)        # only Jane Doe clears 0.85 twice
    assert strict["a"]["precision_any_label"] == pytest.approx(2 / 4)
    assert strict["a"]["residual_rate"] == pytest.approx(1 / 4) and strict["b"]["residual_rate"] == 1.0


def test_summarize_none_without_spans(tmp_path):
    run_dir, _ = make_run(tmp_path, {"r1": "plain"}, {"a": {"r1": []}})
    path = rc.review_run(run_dir, StubBackend(), progress=False)
    (row,) = rc.summarize(run_dir, path)
    assert row["n_spans"] == 0 and row["precision"] is None and row["mean_p_pii"] is None
    # The default stub answers 0.5, which reaches the 0.5 threshold.
    assert row["n_segments"] == 1 and row["residual_rate"] == 1.0
    assert rc.summarize(run_dir, path, threshold=0.6)[0]["residual_rate"] == 0.0


def test_default_stub_answers_count_as_yes_at_the_default_threshold(tmp_path):
    # The StubBackend docstring warns about this: the constant 0.5 answers
    # reach the default threshold, so a dry run without a responder reports a
    # perfect detector that still leaves PII in every segment.
    run_dir, _ = standard_run(tmp_path)
    path = rc.review_run(run_dir, StubBackend(), progress=False)
    row = {r["detector"]: r for r in rc.summarize(run_dir, path)}["a"]
    assert row["precision"] == 1.0 and row["residual_rate"] == 1.0
    above = {r["detector"]: r for r in rc.summarize(run_dir, path, threshold=0.51)}["a"]
    assert above["precision"] == 0.0 and above["residual_rate"] == 0.0


def test_threshold_sweep_against_gold(tmp_path):
    gold = {
        "r1": [_span(T1, "Jane Doe", "PERSON"), _span(T1, "jane@x.com", "EMAIL"), _span(T1, "555-1234", "PHONE")],
        "r2": [_span(T2, "Bob", "PERSON"), _span(T2, "123-45-6789", "ACCOUNT", "GOV_ID")],
    }
    run_dir, fixtures = standard_run(tmp_path, gold=gold)
    path = rc.review_run(run_dir, StubBackend(oracle), progress=False)
    rows = rc.threshold_sweep(run_dir, path, fixtures, thresholds=(0.5, 0.85))
    by = {(r["threshold"], r["detector"]): r for r in rows}
    a, b = by[(0.5, "a")], by[(0.5, "b")]
    assert a["true_precision"] == 1.0 and a["estimated_precision"] == pytest.approx(0.75)
    assert a["precision_error"] == pytest.approx(-0.25)
    assert b["true_precision"] == 0.5 and b["estimated_precision"] == 0.5 and b["precision_error"] == 0.0
    # a missed the SSN in r2, so 1 of its 4 segments truly has PII left.
    assert a["true_residual_rate"] == pytest.approx(1 / 4)
    assert b["true_residual_rate"] == 1.0  # both of b's r1 lines keep the e-mail or the phone
    assert by[(0.85, "a")]["estimated_precision"] == pytest.approx(0.25)
    assert by[(0.85, "a")]["true_precision"] == 1.0
    assert [r["threshold"] for r in rows] == [0.5, 0.5, 0.85, 0.85]
