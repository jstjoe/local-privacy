"""Silver labels (plan 13, Part 4): prompt and schema, locating values,
merging labelers, agreement, `generate` end to end, the report's handling of
unlabeled and silver fixtures, and `calibrate` on a tiny local dataset.

Every LLM here is a `StubClient`; no test calls a hosted API.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import unicodedata
from itertools import pairwise

import pytest

from opf_eval import datasets, fixtures, report, runner, silver
from opf_eval.detectors import make_span, registry
from opf_eval.fixtures import (
    documents_path,
    from_documents,
    from_texts,
    validate_fixtures,
)
from opf_eval.io import read_jsonl, read_meta
from opf_eval.llm import CachedClient, LLMError, StubClient, describe
from opf_eval.llm.schema import strict_problems, validate
from opf_eval.taxonomy import LABELS, register_vocab


def _text_of(user: str) -> str:
    """The record text inside the user message."""
    m = re.search(r"<text>\n(.*)\n</text>", user, re.DOTALL)
    assert m, user
    return m.group(1)


def item(value: str, label: str, context: str = "") -> dict:
    return {"value": value, "label": label, "context": context}


def regex_responder(patterns: dict[str, str]):
    """A labeler that reports every regex match as an entity."""

    def respond(system, user, schema):
        text = _text_of(user)
        out = []
        for label, pattern in patterns.items():
            out += [item(m.group(0), label, m.group(0)) for m in re.finditer(pattern, text)]
        return {"entities": out}

    return respond


EMAIL_RE = r"[\w.]+@[\w.]+\.\w+"
PHONE_RE = r"\d{3}-\d{4}"


def spans_of(spans):
    return [(s["start"], s["end"], s["fine_label"]) for s in spans]


# ------------------------------------------------------- schema and prompt


def test_label_schema_shape_and_strict_mode():
    schema = silver.label_schema(["EMAIL", "PERSON", "EMAIL"])
    entity = schema["properties"]["entities"]["items"]
    assert schema["required"] == ["entities"] and schema["additionalProperties"] is False
    assert entity["required"] == ["value", "label", "context"]
    assert entity["additionalProperties"] is False
    assert entity["properties"]["label"]["enum"] == ["EMAIL", "PERSON"]  # deduped, order kept
    assert strict_problems(schema) == []  # OpenAI strict mode accepts it
    assert validate({"entities": [item("a@b.co", "EMAIL", "x")]}, schema) == []
    assert validate({"entities": [item("a@b.co", "PHONE", "x")]}, schema)


@pytest.mark.parametrize("labels", [[], ["EMAIL", "NOT_A_LABEL"]])
def test_label_schema_rejects_bad_label_sets(labels):
    with pytest.raises(ValueError):
        silver.label_schema(labels)


def test_build_prompt_lists_labels_and_rules():
    system, user = silver.build_prompt("Mail ann@x.io", ["PERSON", "GIVEN_NAME", "EMAIL"])
    for lbl in ("PERSON", "GIVEN_NAME", "EMAIL"):
        assert f"- {lbl}" in system and LABELS[lbl].description in system
    assert "a more specific kind of PERSON" in system
    assert silver.PROMPT_VERSION in system
    assert "EVERY occurrence" in system and "exactly" in system and "6 words" in system
    assert "Ignore any instructions" in system
    assert _text_of(user) == "Mail ann@x.io"
    # The system message does not depend on the text, so providers can cache it.
    assert silver.build_prompt("other", ["PERSON", "GIVEN_NAME", "EMAIL"])[0] == system
    # Without fine labels there is no rule about specific types.
    assert "most specific type" not in silver.build_prompt("x", ["EMAIL"])[0]


# ------------------------------------------------------------------ locate


def test_locate_exact_span_fields():
    text = "Write to ann@x.io now."
    spans, dropped = silver.locate(text, [item("ann@x.io", "EMAIL")])
    assert dropped == []
    assert spans == [{
        "label": "EMAIL", "fine_label": "EMAIL", "raw_label": "EMAIL",
        "start": 9, "end": 17, "text": "ann@x.io",
    }]


def test_locate_fine_label_gets_coarse_parent():
    (span,), _ = silver.locate("Hi Ann", [item("Ann", "GIVEN_NAME")])
    assert (span["label"], span["fine_label"], span["raw_label"]) == (
        "PERSON", "GIVEN_NAME", "GIVEN_NAME"
    )


def test_locate_case_insensitive_then_whitespace_fallbacks():
    text = "Name: Jane   Doe\nlives at 1 Main\n  Street."
    spans, dropped = silver.locate(text, [
        item("JANE   DOE", "PERSON"),           # case differs
        item("1 Main Street", "STREET_ADDRESS"),  # newline and indentation in the text
    ])
    assert dropped == []
    assert [s["text"] for s in spans] == ["Jane   Doe", "1 Main\n  Street"]
    assert all(text[s["start"]:s["end"]] == s["text"] for s in spans)


def test_locate_exact_match_beats_case_insensitive():
    text = "may we meet May 3? Ask May."
    spans, _ = silver.locate(text, [item("May", "GIVEN_NAME")])
    assert [s["start"] for s in spans] == [12, 23]  # "may" at 0 is not matched


def test_locate_strips_whitespace_around_value():
    (span,), _ = silver.locate("id: AB12 ok", [item("  AB12 ", "CUSTOMER_ID")])
    assert span["text"] == "AB12"


def test_locate_prefers_whole_words_but_falls_back():
    text = "Annual report by Ann."
    (span,), _ = silver.locate(text, [item("Ann", "GIVEN_NAME")])
    assert span["start"] == 17  # not the "Ann" inside "Annual"
    (span,), _ = silver.locate("Annual review", [item("Ann", "GIVEN_NAME")])
    assert span["start"] == 0  # only an embedded match exists: use it


@pytest.mark.parametrize(("text", "value", "context", "want"), [
    # The exact match sits inside a longer word, the stand-alone one differs
    # in case: the stand-alone one wins, before or after the embedded one.
    ("MARIA called Marianne", "Maria", "MARIA called", (0, 5)),
    ("Annabel met ANN at noon", "Ann", "met ANN at", (12, 15)),
    # Same with the whitespace tier: "Ann  Lee" is stand-alone, "AnnLee" is not.
    ("xAnn Leex and Ann  Lee", "Ann Lee", "", (14, 22)),
])
def test_locate_stand_alone_match_in_a_later_tier_beats_an_embedded_one(text, value, context, want):
    (span,), dropped = silver.locate(text, [item(value, "PERSON", context)])
    assert (span["start"], span["end"]) == want and dropped == []


def test_locate_matches_across_unicode_normalisation_forms():
    nfd = unicodedata.normalize("NFD", "José García lives in Málaga")
    spans, dropped = silver.locate(
        nfd, [item("José García", "PERSON"), item("málaga", "CITY", "in Málaga")]
    )
    assert dropped == []
    assert [s["text"] for s in spans] == [nfd[:13], nfd[-7:]]  # whole clusters, marks included
    assert [unicodedata.normalize("NFC", s["text"]) for s in spans] == ["José García", "Málaga"]
    # An NFD value in NFC text is found too.
    (span,), _ = silver.locate("Hi José!", [item(unicodedata.normalize("NFD", "José"), "PERSON")])
    assert (span["start"], span["end"], span["text"]) == (3, 7, "José")


@pytest.mark.parametrize("value,label,prefix", [
    ("김민수", "PERSON", "고객: "),  # Hangul jamo have combining class 0 but compose
    ("கொழும்பு", "ADDRESS", "Address: "),  # so do Tamil vowel signs
])
def test_locate_finds_nfc_values_in_nfd_text_whose_marks_have_class_zero(value, label, prefix):
    nfd = unicodedata.normalize("NFD", value)
    text = prefix + nfd + " 님"
    (span,), dropped = silver.locate(text, [item(value, label)])
    assert dropped == []
    assert (span["start"], span["end"], span["text"]) == (len(prefix), len(prefix) + len(nfd), nfd)


def test_nfc_view_is_nfc_and_maps_back_to_whole_clusters():
    import random

    rng = random.Random(7)
    pool = list(unicodedata.normalize("NFD", "김민수 கொழும்பு José Ångström ﬁ Ω ḳ́ e\u0301\u0323 각"))
    for _ in range(500):
        text = "".join(rng.choice(pool) for _ in range(rng.randint(0, 20)))
        nfc, starts, ends = silver._nfc_view(text)
        assert nfc == unicodedata.normalize("NFC", text), repr(text)
        assert len(starts) == len(ends) == len(nfc)
        assert all(a < b for a, b in zip(starts, ends, strict=True))


def test_locate_values_next_to_cjk_text_are_not_embedded():
    text = "联系张伟电话。张伟是经理"
    spans, _ = silver.locate(text, [item("张伟", "PERSON")])
    assert [s["start"] for s in spans] == [2, 7]


def test_locate_context_picks_the_occurrence():
    text = "From Ann to Bob. Later Ann called Bob again."
    spans, _ = silver.locate(text, [item("Bob", "PERSON", "Ann called Bob again")])
    assert [s["start"] for s in spans] == [34]


def test_locate_all_mode_labels_every_occurrence_without_context():
    text = "Ann, Ann and Ann"
    spans, dropped = silver.locate(text, [item("Ann", "PERSON")])
    assert [s["start"] for s in spans] == [0, 5, 13] and dropped == []
    # A context that matches every occurrence pins nothing.
    spans, _ = silver.locate(text, [item("Ann", "PERSON", "Ann")])
    assert len(spans) == 3


def test_locate_all_mode_keeps_only_pinned_occurrences_when_every_item_is_pinned():
    text = "Ann met Bob. Bob met Ann. Ann left."
    spans, _ = silver.locate(text, [
        item("Ann", "PERSON", "Ann met Bob"),
        item("Ann", "PERSON", "Ann left"),
    ])
    assert [s["start"] for s in spans] == [0, 26]
    # One unpinned item in the group means every occurrence is labelled.
    spans, _ = silver.locate(text, [item("Ann", "PERSON", "Ann met Bob"), item("Ann", "PERSON")])
    assert [s["start"] for s in spans] == [0, 21, 26]


def test_locate_listed_mode_takes_first_occurrences_and_drops_extras():
    text = "Ann, Ann and Ann"
    spans, dropped = silver.locate(text, [item("Ann", "PERSON")] * 2, occurrences="listed")
    assert [s["start"] for s in spans] == [0, 5] and dropped == []
    spans, dropped = silver.locate(text, [item("Ann", "PERSON")] * 4, occurrences="listed")
    assert len(spans) == 3
    assert [d["reason"] for d in dropped] == ["extra_occurrence"]


def test_locate_listed_mode_pinned_items_choose_first():
    text = "Ann, Ann and Ann"
    spans, _ = silver.locate(
        text, [item("Ann", "PERSON"), item("Ann", "PERSON", "and Ann")], occurrences="listed",
    )
    assert [s["start"] for s in spans] == [0, 13]


def test_locate_drops_with_reasons():
    text = "Mail ann@x.io"
    spans, dropped = silver.locate(text, [
        item("bob@y.io", "EMAIL"),
        item("   ", "EMAIL"),
        item("ann@x.io", "NOT_A_LABEL"),
        item("ann@x.io", "PHONE"),
        {"value": None, "label": "EMAIL"},
    ], labels=["EMAIL"])
    assert spans == []
    assert [d["reason"] for d in dropped] == [
        "not_found", "empty", "unknown_label", "unknown_label", "empty",
    ]
    assert dropped[0] == {
        "value": "bob@y.io", "label": "EMAIL", "context": "", "reason": "not_found"
    }


def test_locate_overlaps_keep_the_longer_span():
    text = "Dr Jane Doe called"
    spans, dropped = silver.locate(text, [item("Jane", "GIVEN_NAME"), item("Jane Doe", "PERSON")])
    assert spans_of(spans) == [(3, 11, "PERSON")]
    assert dropped == [{
        "value": "Jane", "label": "GIVEN_NAME", "context": "", "reason": "overlap",
        "start": 3, "end": 7,
    }]


def test_locate_equal_length_overlap_keeps_the_first():
    spans, dropped = silver.locate("Paris", [item("Paris", "CITY"), item("Paris", "GIVEN_NAME")])
    assert spans_of(spans) == [(0, 5, "CITY")]
    assert [(d["label"], d["reason"]) for d in dropped] == [("GIVEN_NAME", "overlap")]
    # Same span twice with the same label is not a conflict.
    spans, dropped = silver.locate("Paris", [item("Paris", "CITY"), item("paris", "CITY")])
    assert len(spans) == 1 and dropped == []


@pytest.mark.parametrize("occurrences", ["all", "listed"])
def test_locate_context_pin_beats_an_unpinned_item_on_the_same_span(occurrences):
    text = "Jordan went to Jordan."
    items = [item("Jordan", "PERSON"), item("Jordan", "COUNTRY", "to Jordan.")]
    spans, dropped = silver.locate(text, items, occurrences=occurrences)
    # The COUNTRY item's context pins the second "Jordan"; the PERSON item
    # names no occurrence and so must not override that pin.
    assert (15, 21, "COUNTRY") in spans_of(spans)
    assert (15, 21, "PERSON") not in spans_of(spans)
    assert all(d["reason"] == "overlap" and d["label"] == "PERSON" for d in dropped)


def test_locate_output_is_sorted_and_disjoint():
    text = "b@y.io then a@x.io then 555-0101"
    spans, _ = silver.locate(
        text, [item("555-0101", "PHONE"), item("a@x.io", "EMAIL"), item("b@y.io", "EMAIL")]
    )
    assert [s["start"] for s in spans] == sorted(s["start"] for s in spans)
    assert all(a["end"] <= b["start"] for a, b in pairwise(spans))


def test_locate_rejects_unknown_mode():
    with pytest.raises(ValueError):
        silver.locate("x", [], occurrences="some")


# ------------------------------------------------------------ label_record


def test_label_record_calls_the_labeler_once():
    client = StubClient(regex_responder({"EMAIL": EMAIL_RE}))
    spans, dropped = silver.label_record(client, "mail ann@x.io", ["EMAIL", "PHONE"])
    assert spans_of(spans) == [(5, 13, "EMAIL")] and dropped == []
    ((system, user),) = client.calls
    assert "- PHONE" in system and _text_of(user) == "mail ann@x.io"


def test_label_record_blank_text_makes_no_call():
    client = StubClient(regex_responder({"EMAIL": EMAIL_RE}))
    assert silver.label_record(client, "  \n ", ["EMAIL"]) == ([], [])
    assert client.calls == []


def test_label_record_splits_long_text_and_shifts_offsets():
    lines = [f"line {i} reach me at user{i}@x.io or 555-{i:04d}" for i in range(40)]
    text = "\r\n".join(lines)  # CRLF must not shift offsets
    client = StubClient(regex_responder({"EMAIL": EMAIL_RE, "PHONE": PHONE_RE}))
    spans, _ = silver.label_record(client, text, ["EMAIL", "PHONE"], max_chars=300)
    assert len(client.calls) > 1
    assert all(len(_text_of(u)) <= 300 for _, u in client.calls)
    assert len(spans) == 80
    assert all(text[s["start"]:s["end"]] == s["text"] for s in spans)
    assert {s["text"] for s in spans if s["label"] == "EMAIL"} == {
        f"user{i}@x.io" for i in range(40)
    }


def test_label_record_propagates_llm_errors():
    def refuse(system, user, schema):
        raise LLMError("refused: cyber")

    with pytest.raises(LLMError):
        silver.label_record(StubClient(refuse), "ann@x.io", ["EMAIL"])


def test_label_record_halves_a_piece_whose_answer_hits_max_tokens():
    lines = [f"Jane Doe{i},jane{i}@x.io,555-{i:04d}" for i in range(60)]
    text = "\n".join(lines)
    regex = regex_responder({"EMAIL": EMAIL_RE, "PHONE": PHONE_RE})

    def short_answers(system, user, schema):
        # Like a real model: a long enough answer runs out of output tokens.
        if len(_text_of(user)) > 400:
            raise LLMError("stub hit max_tokens=16000 before finishing its answer", truncated=True)
        return regex(system, user, schema)

    client = StubClient(short_answers)
    spans, dropped = silver.label_record(client, text, ["EMAIL", "PHONE"], min_split_chars=100)
    assert dropped == []
    assert len(spans) == 120
    assert all(text[s["start"]:s["end"]] == s["text"] for s in spans)
    assert [s["start"] for s in spans] == sorted(s["start"] for s in spans)
    assert {s["text"] for s in spans if s["label"] == "EMAIL"} == {
        f"jane{i}@x.io" for i in range(60)
    }
    texts = [_text_of(u) for _, u in client.calls]
    assert len(texts[0]) == len(text)  # first the whole record, then halves
    # 1859 chars -> two halves of ~930 -> four of ~465 -> eight of ~232.
    assert len(texts) == 1 + 2 + 4 + 8
    answered = [t for t in texts if len(t) <= 400]
    assert len(answered) == 8 and "".join(answered) == text  # in order, no gaps or overlaps
    whole = set(lines)
    assert all(set(t.strip("\n").split("\n")) <= whole for t in texts)  # cuts fall between lines


def test_label_record_gives_up_splitting_at_min_split_chars():
    def always_long(system, user, schema):
        raise LLMError("stub hit max_tokens=16000 before finishing its answer", truncated=True)

    client = StubClient(always_long)
    with pytest.raises(LLMError, match="max_tokens"):
        silver.label_record(client, "a@x.io\n" * 100, ["EMAIL"], min_split_chars=200)
    # 700 chars -> 350 + 350 -> 175 * 4: the first quarter is too small to
    # cut again, so the error propagates after 1 + 2 + 2 calls (depth first).
    assert [len(_text_of(u)) for _, u in client.calls] == [700, 350, 175]


def test_label_record_splits_on_the_truncated_flag_and_not_the_message():
    # The split-and-retry reads LLMError.truncated. A message that happens to
    # mention max_tokens is not a truncation, and the wording of a real
    # truncation can change without stopping the split.
    def mentions_max_tokens(system, user, schema):
        raise LLMError("transport failed while max_tokens was 16000")

    client = StubClient(mentions_max_tokens)
    with pytest.raises(LLMError):
        silver.label_record(client, "a@x.io\n" * 300, ["EMAIL"], min_split_chars=100)
    assert len(client.calls) == 1

    regex = regex_responder({"EMAIL": EMAIL_RE})

    def reworded(system, user, schema):
        if len(_text_of(user)) > 400:
            raise LLMError("the answer was cut short", truncated=True)
        return regex(system, user, schema)

    client = StubClient(reworded)
    spans, _ = silver.label_record(client, "a@x.io\n" * 100, ["EMAIL"], min_split_chars=100)
    assert len(spans) == 100 and len(client.calls) > 1


def test_label_record_does_not_split_on_other_errors():
    def refuse(system, user, schema):
        raise LLMError("refused: cyber")

    client = StubClient(refuse)
    with pytest.raises(LLMError, match="refused"):
        silver.label_record(client, "a@x.io\n" * 300, ["EMAIL"], min_split_chars=100)
    assert len(client.calls) == 1


# ------------------------------------------------------------------- merge


def S(start, end, label="EMAIL", fine=None):
    return {
        "label": label,
        "fine_label": fine or label,
        "raw_label": fine or label,
        "start": start,
        "end": end,
    }


def test_merge_modes():
    a = [S(0, 5), S(10, 15, "PHONE"), S(20, 25)]
    b = [S(0, 5), S(10, 15, "PHONE")]
    c = [S(0, 5), S(30, 35)]
    lists = [a, b, c]

    def key(spans):
        return [(s["start"], s["end"], s["votes"]) for s in spans]

    assert key(silver.merge(lists, "union")) == [(0, 5, 3), (10, 15, 2), (20, 25, 1), (30, 35, 1)]
    assert key(silver.merge(lists, "majority")) == [(0, 5, 3), (10, 15, 2)]
    assert key(silver.merge(lists, "intersection")) == [(0, 5, 3)]


def test_merge_majority_needs_more_than_half():
    lists = [[S(0, 5)], [S(0, 5)], [], []]
    assert silver.merge(lists, "majority") == []  # 2 of 4 is not a majority
    assert len(silver.merge(lists[:3], "majority")) == 1


def test_merge_single_labeler_keeps_everything():
    spans = [S(0, 5), S(8, 9, "PHONE")]
    assert spans_of(silver.merge([spans], "intersection")) == spans_of(spans)


def test_merge_agreement_needs_same_coarse_label_and_iou():
    # Same coarse label (PERSON) with different fine labels still agree.
    merged = silver.merge([[S(0, 8, "PERSON", "GIVEN_NAME")], [S(0, 8, "PERSON")]], "intersection")
    assert len(merged) == 1
    # Different coarse labels do not.
    assert silver.merge([[S(0, 8, "PERSON")], [S(0, 8, "ORGANIZATION")]], "intersection") == []
    # IoU 4/8 = 0.5 agrees; 3/8 does not.
    assert len(silver.merge([[S(0, 8)], [S(0, 4)]], "intersection")) == 1
    assert silver.merge([[S(0, 8)], [S(0, 3)]], "intersection") == []


def test_merge_counts_only_labelers_that_agree_with_the_kept_span():
    # A agrees with B and B agrees with C, but A and C do not agree (IoU 0.25).
    a, b, c = [S(0, 10, "PERSON")], [S(3, 13, "PERSON")], [S(6, 16, "PERSON")]
    (span,) = silver.merge([a, b, c], "intersection")
    # B's span is the one all agree with.
    assert (span["start"], span["end"], span["votes"]) == (3, 13, 3)
    assert [(s["start"], s["votes"]) for s in silver.merge([a, c], "union")] == [(0, 1)]
    assert silver.merge([a, c], "intersection") == []
    # "Dr. Jane Doe", "Jane Doe" and "Jane": only "Jane Doe" has every labeler's agreement.
    lists = [[S(0, 12, "PERSON")], [S(4, 12, "PERSON")], [S(4, 8, "PERSON")]]
    (span,) = silver.merge(lists, "intersection")
    assert (span["start"], span["end"], span["votes"]) == (4, 12, 3)
    # Chained spans never earn a unanimous vote that no single span has.
    lists = [[S(0, 10, "PERSON")], [S(4, 14, "PERSON")], [S(9, 19, "PERSON")]]
    assert all(
        not (s["start"] == 0 and s["votes"] == 3) for s in silver.merge(lists, "union")
    )


def test_merge_picks_most_common_boundary_then_longest():
    merged = silver.merge([[S(0, 8)], [S(0, 8)], [S(0, 6)]], "union")
    assert (merged[0]["start"], merged[0]["end"]) == (0, 8)
    merged = silver.merge([[S(0, 6)], [S(0, 8)]], "union")  # tie -> longest
    assert (merged[0]["start"], merged[0]["end"]) == (0, 8)


def test_merge_picks_most_common_fine_label():
    lists = [
        [S(0, 8, "PERSON", "PERSON")],
        [S(0, 8, "PERSON", "GIVEN_NAME")],
        [S(0, 8, "PERSON", "GIVEN_NAME")],
    ]
    (span,) = silver.merge(lists, "union")
    assert (span["label"], span["fine_label"], span["raw_label"]) == (
        "PERSON", "GIVEN_NAME", "GIVEN_NAME"
    )
    (span,) = silver.merge(lists[:2], "union")  # tie -> first labeler's
    assert span["fine_label"] == "PERSON"


def test_merge_keeps_text_and_resolves_leftover_overlaps():
    a = [{**S(0, 5, "ADDRESS", "CITY"), "text": "Paris"}]
    b = [{**S(0, 12, "PERSON"), "text": "Paris Hilton"}]
    c = [{**S(0, 5, "ADDRESS", "CITY"), "text": "Paris"}]
    merged = silver.merge([a, b, c], "union")
    # Both entities pass "union" but overlap: more votes wins.
    assert [(s["text"], s["votes"]) for s in merged] == [("Paris", 2)]


def test_merge_validates_input():
    assert silver.merge([], "union") == []
    with pytest.raises(ValueError):
        silver.merge([[S(0, 1)]], "vote")


# --------------------------------------------------------------- agreement


def test_agreement_maths():
    per = {
        "a": {"r1": [S(0, 5), S(10, 15, "PHONE")], "r2": [S(0, 3)]},
        "b": {"r1": [S(0, 5), S(10, 14, "PHONE")], "r2": [S(0, 3), S(5, 9)], "r3": [S(0, 1)]},
        "c": {"r1": [], "r2": []},
    }
    rows = {(r["a"], r["b"]): r for r in silver.agreement(per)}
    ab = rows[("a", "b")]
    # Compared on r1 and r2 only (r3 is missing from a). Matches: (0,5) and (0,3).
    assert (ab["n_a"], ab["n_b"], ab["matched"], ab["n_records"]) == (3, 4, 2, 2)
    assert ab["precision"] == pytest.approx(2 / 4)
    assert ab["recall"] == pytest.approx(2 / 3)
    assert ab["f1"] == pytest.approx(2 * 2 / 7)
    ac = rows[("a", "c")]
    assert (ac["f1"], ac["precision"], ac["recall"]) == (0.0, 0.0, 0.0)
    assert set(rows) == {("a", "b"), ("a", "c"), ("b", "c")}


def test_agreement_uses_coarse_labels_and_counts_duplicates():
    per = {
        "a": {"r": [S(0, 4, "PERSON", "GIVEN_NAME")]},
        "b": {"r": [S(0, 4, "PERSON", "PERSON"), S(0, 4, "PERSON", "PERSON")]},
    }
    (row,) = silver.agreement(per)
    assert (row["matched"], row["n_a"], row["n_b"]) == (1, 1, 2)


def test_agreement_two_empty_labelers_agree():
    (row,) = silver.agreement({"a": {"r": []}, "b": {"r": []}})
    assert row["f1"] == row["precision"] == row["recall"] == 1.0
    assert silver.agreement({"a": {"r": []}}) == []


def test_agreement_is_none_when_no_record_was_compared():
    # Both labelers failed everywhere, so `generate` passes them no answered records.
    (row,) = silver.agreement({"a": {}, "b": {}})
    assert row["n_records"] == 0
    assert row["f1"] is None and row["precision"] is None and row["recall"] is None
    # Answers for disjoint records are not a comparison either.
    (row,) = silver.agreement({"a": {"r1": [S(0, 5)]}, "b": {"r2": [S(0, 5)]}})
    assert row["n_records"] == 0 and row["f1"] is None


# ---------------------------------------------------------------- generate


TEXTS = {
    "t1": "Write to ann@x.io or call 555-0101.",
    "t2": "Bob's mail is bob@y.io and his phone 555-0199. Also ann@x.io.",
    "t3": "Nothing to see here.",
}


@pytest.fixture
def unlabeled(tmp_path):
    return from_texts(TEXTS, tmp_path / "mine.jsonl", labels=["EMAIL", "PHONE", "PERSON"])


def test_generate_two_labelers_end_to_end(unlabeled, tmp_path, capsys):
    full = StubClient(regex_responder({"EMAIL": EMAIL_RE, "PHONE": PHONE_RE}), model="full")
    mails = StubClient(regex_responder({"EMAIL": EMAIL_RE}), model="mails")
    out = silver.generate(
        unlabeled, [full, mails], tmp_path / "mine.silver.jsonl", merge_how="union"
    )
    assert out == tmp_path / "mine.silver.jsonl"

    rows = {r["id"]: r for r in read_jsonl(out)}
    assert list(rows) == ["t1", "t2", "t3"]
    assert [s["text"] for s in rows["t2"]["gold_spans"]] == ["bob@y.io", "555-0199", "ann@x.io"]
    assert [s["votes"] for s in rows["t2"]["gold_spans"]] == [2, 1, 2]
    assert rows["t3"]["gold_spans"] == []
    rep = validate_fixtures(out)
    assert rep.ok and not rep.unlabeled and rep.n_spans == 5 and not rep.warnings

    meta = read_meta(out)
    assert meta["gold"] == "silver" and not fixtures.is_unlabeled(out) and silver.is_silver(out)
    src = meta["gold_source"]
    assert src["kind"] == "llm" and src["merge"] == "union" and src["occurrences"] == "all"
    assert src["prompt_version"] == silver.PROMPT_VERSION
    assert src["labelers"] == [describe(full), describe(mails)]
    assert src["models"] == ["full", "mails"] and src["names"] == ["stub_full", "stub_mails"]
    assert meta["labels"] == ["EMAIL", "PERSON", "PHONE"]  # from the unlabeled meta
    assert meta["unlabeled_fixtures"] == str(unlabeled)
    assert meta["source"] == "texts" and meta["dataset"] == "custom:mine"  # copied from input meta
    assert meta["n_errors"] == {"stub_full": 0, "stub_mails": 0}
    assert meta["error_ids"] == [] and meta["n_spans"] == 5 and meta["n_written"] == 3
    (agree,) = meta["agreement"]
    assert (agree["a"], agree["b"], agree["n_a"], agree["n_b"], agree["matched"]) == (
        "stub_full", "stub_mails", 5, 3, 3,
    )

    raw = read_jsonl(tmp_path / "mine.silver.labeler_stub_full.jsonl")
    assert [r["id"] for r in raw] == ["t1", "t2", "t3"]
    assert set(raw[0]) == {"id", "spans", "dropped", "error"} and raw[0]["error"] is None
    assert meta["labeler_files"] == {
        "stub_full": "mine.silver.labeler_stub_full.jsonl",
        "stub_mails": "mine.silver.labeler_stub_mails.jsonl",
    }
    # The unlabeled input is untouched.
    assert all(r["gold_spans"] == [] for r in read_jsonl(unlabeled))
    out_text = capsys.readouterr().out
    assert "[silver] 6/6 records labelled" in out_text and "span F1" in out_text


def test_generate_majority_with_two_labelers_needs_both(unlabeled, tmp_path):
    full = StubClient(regex_responder({"EMAIL": EMAIL_RE, "PHONE": PHONE_RE}), model="full")
    mails = StubClient(regex_responder({"EMAIL": EMAIL_RE}), model="mails")
    out = silver.generate(unlabeled, [full, mails], tmp_path / "s.jsonl", progress=False)
    labels = {s["label"] for r in read_jsonl(out) for s in r["gold_spans"]}
    assert labels == {"EMAIL"}


def test_generate_records_errors_and_merges_the_rest(unlabeled, tmp_path):
    def flaky(system, user, schema):
        if "bob@" in user:
            raise LLMError("refused: privacy")
        return regex_responder({"EMAIL": EMAIL_RE})(system, user, schema)

    def broken(system, user, schema):
        raise LLMError("the API is down")

    good = StubClient(regex_responder({"EMAIL": EMAIL_RE, "PHONE": PHONE_RE}), model="good")
    out = silver.generate(
        unlabeled, [StubClient(flaky, model="flaky"), good], tmp_path / "s.jsonl",
        merge_how="intersection", progress=False,
    )
    meta = read_meta(out)
    assert meta["n_errors"] == {"stub_flaky": 1, "stub_good": 0}
    assert meta["error_ids"] == []
    t2 = next(r for r in read_jsonl(out) if r["id"] == "t2")
    # Only `good` answered for t2, so "intersection" of one labeler keeps its spans.
    assert [s["text"] for s in t2["gold_spans"]] == ["bob@y.io", "555-0199", "ann@x.io"]
    raw = {r["id"]: r for r in read_jsonl(tmp_path / "s.labeler_stub_flaky.jsonl")}
    assert raw["t2"]["error"] == "refused: privacy" and raw["t2"]["spans"] == []

    out = silver.generate(unlabeled, [StubClient(broken)], tmp_path / "s2.jsonl", progress=False)
    meta = read_meta(out)
    assert meta["error_ids"] == ["t1", "t2", "t3"] and meta["n_errors"] == {"stub_stub": 3}
    assert all(r["gold_spans"] == [] for r in read_jsonl(out))


def test_generate_other_exceptions_stop_the_run(unlabeled, tmp_path):
    def bug(system, user, schema):
        raise KeyError("bug")

    with pytest.raises(KeyError):
        silver.generate(unlabeled, [StubClient(bug)], tmp_path / "s.jsonl", progress=False)


def test_generate_survives_lone_surrogates_in_text_and_answers(tmp_path):
    texts = {"a": "Ann \ud83d wrote to Bob", "b": "Bob Smith lives here"}
    fx = from_texts(texts, tmp_path / "surr.jsonl", labels=["PERSON"])

    def half_emoji(system, user, schema):
        # Copies half of an emoji pair, as a model can.
        return {"entities": [item("Ann \ud83d", "PERSON"), item("Bob", "PERSON")]}

    out = silver.generate(fx, [StubClient(half_emoji)], tmp_path / "s.jsonl", progress=False)
    rows = {r["id"]: r for r in read_jsonl(out)}
    assert rows["a"]["text"] == texts["a"]  # written with escapes and read back unchanged
    assert [sp["text"] for sp in rows["b"]["gold_spans"]] == ["Bob"]
    assert read_meta(out)["n_errors"] == {"stub_stub": 0}


def test_generate_records_which_records_a_fallback_model_answered(unlabeled, tmp_path, capsys):
    from opf_eval.llm.base import FallbackLog

    class Rescued(StubClient):
        def __init__(self):
            super().__init__(regex_responder({"EMAIL": EMAIL_RE}), model="claude-opus-5-5")
            self._log = FallbackLog()

        def complete_json(self, *, system, user, schema, name="result", max_tokens=8000):
            answer = super().complete_json(system=system, user=user, schema=schema)
            self._log.note("claude-sonnet-4-5" if "bob@" in user else None)
            return answer

        def pop_fallbacks(self):
            return self._log.pop()

    out = silver.generate(unlabeled, [Rescued()], tmp_path / "s.jsonl", max_workers=3)
    raw = {r["id"]: r for r in read_jsonl(tmp_path / "s.labeler_stub_claude-opus-5-5.jsonl")}
    assert raw["t2"]["fallback_models"] == ["claude-sonnet-4-5"]
    assert "fallback_models" not in raw["t1"] and "fallback_models" not in raw["t3"]
    assert read_meta(out)["n_fallback"] == {"stub_claude-opus-5-5": 1}
    assert "1 records were answered by a server-side fallback model" in capsys.readouterr().out


def test_generate_dropped_values_are_logged(unlabeled, tmp_path):
    def invents(system, user, schema):
        return {"entities": [item("nobody@nowhere.io", "EMAIL")]}

    out = silver.generate(unlabeled, [StubClient(invents)], tmp_path / "s.jsonl", progress=False)
    meta = read_meta(out)
    assert meta["n_dropped"] == {"stub_stub": 3} and meta["n_spans"] == 0
    raw = read_jsonl(tmp_path / "s.labeler_stub_stub.jsonl")
    assert raw[0]["dropped"][0]["reason"] == "not_found"


def test_generate_labels_and_level(unlabeled, tmp_path):
    seen = []

    def spy(system, user, schema):
        seen.append(schema["properties"]["entities"]["items"]["properties"]["label"]["enum"])
        return {"entities": []}

    silver.generate(
        unlabeled, [StubClient(spy)], tmp_path / "a.jsonl", labels=["EMAIL"], progress=False
    )
    assert seen[-1] == ["EMAIL"]
    assert read_meta(tmp_path / "a.jsonl")["labels"] == ["EMAIL"]
    silver.generate(
        unlabeled, [StubClient(spy)], tmp_path / "b.jsonl",
        labels=["GIVEN_NAME", "FAMILY_NAME", "EMAIL"], level="coarse", progress=False,
    )
    assert seen[-1] == ["PERSON", "EMAIL"]


def test_generate_without_labels_in_meta_uses_every_label(tmp_path):
    fx = tmp_path / "plain.jsonl"
    fx.write_text(json.dumps({"id": "1", "text": "hi", "gold_spans": []}) + "\n")
    out = silver.generate(fx, [StubClient()], tmp_path / "s.jsonl", progress=False)
    assert len(read_meta(out)["labels"]) == len(LABELS)


def test_generate_keeps_document_fields_and_segment_map(tmp_path):
    from opf_eval.documents import from_text

    doc = from_text("Contact\nann@x.io\nthanks", id="doc1")
    unl = from_documents([doc], tmp_path / "docs.jsonl", labels=["EMAIL"])
    out = silver.generate(unl, [StubClient(regex_responder({"EMAIL": EMAIL_RE}))],
                          tmp_path / "docs.silver.jsonl", progress=False)
    (rec,) = read_jsonl(out)
    assert rec["doc_id"] == "doc1" and rec["offset"] == 0 and rec["id"] == "doc1#0"
    assert documents_path(out).read_text() == documents_path(unl).read_text()
    assert fixtures.load_documents(out)["doc1"].locate(8, 16) == [{"line": 2}]


def test_generate_rejects_bad_arguments(unlabeled, tmp_path):
    with pytest.raises(ValueError, match="at least one"):
        silver.generate(unlabeled, [], tmp_path / "s.jsonl")
    with pytest.raises(ValueError, match="merge"):
        silver.generate(unlabeled, [StubClient()], tmp_path / "s.jsonl", merge_how="vote")
    with pytest.raises(ValueError, match="occurrences"):
        silver.generate(unlabeled, [StubClient()], tmp_path / "s.jsonl", occurrences="some")
    with pytest.raises(ValueError, match="differ"):
        silver.generate(unlabeled, [StubClient()], unlabeled)


def test_generate_warns_when_input_has_gold(tmp_path):
    fx = tmp_path / "gold.jsonl"
    fx.write_text(json.dumps({"id": "1", "text": "ann@x.io", "gold_spans": [S(0, 8)]}) + "\n")
    with pytest.warns(UserWarning, match="already has gold"):
        silver.generate(fx, [StubClient()], tmp_path / "s.jsonl", labels=["EMAIL"], progress=False)


def test_generate_same_model_twice_gets_distinct_files(unlabeled, tmp_path):
    a, b = StubClient(), StubClient()
    out = silver.generate(unlabeled, [a, b], tmp_path / "s.jsonl", progress=False)
    assert read_meta(out)["gold_source"]["names"] == ["stub_stub", "stub_stub_2"]
    assert (tmp_path / "s.labeler_stub_stub_2.jsonl").exists()


def test_generate_through_the_cache_is_free_the_second_time(unlabeled, tmp_path):
    inner = StubClient(regex_responder({"EMAIL": EMAIL_RE}))
    cached = CachedClient(inner, tmp_path / "cache")
    first = silver.generate(unlabeled, [cached], tmp_path / "a.jsonl", progress=False)
    n_calls = len(inner.calls)
    second = silver.generate(unlabeled, [cached], tmp_path / "b.jsonl", progress=False)
    assert len(inner.calls) == n_calls == 3 and cached.hits == 3

    def strip(p):
        return [r["gold_spans"] for r in read_jsonl(p)]

    assert strip(first) == strip(second)


def test_generate_is_deterministic_across_worker_counts(unlabeled, tmp_path):
    client = StubClient(regex_responder({"EMAIL": EMAIL_RE, "PHONE": PHONE_RE}))
    a = silver.generate(unlabeled, [client], tmp_path / "a.jsonl", max_workers=1, progress=False)
    b = silver.generate(unlabeled, [client], tmp_path / "b.jsonl", max_workers=8, progress=False)
    assert read_jsonl(a) == read_jsonl(b)


def test_plan_calls(unlabeled, tmp_path):
    plan = silver.plan_calls(unlabeled, 2)
    assert plan == {
        "n_records": 3, "n_chars": sum(map(len, TEXTS.values())), "n_calls": 6, "n_labelers": 2
    }
    long = from_texts(["x" * 50 + "\n" + "y" * 50, "   "], tmp_path / "long.jsonl")
    assert silver.plan_calls(long, [StubClient()], max_chars=60)["n_calls"] == 2


def test_importing_silver_does_not_import_llm_sdks():
    code = (
        "import sys, opf_eval.silver\n"
        "bad = [m for m in ('anthropic', 'openai', 'google.auth', 'pypdf', 'docx', 'openpyxl') "
        "if m in sys.modules]\n"
        "assert not bad, bad\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


# ------------------------------------------------------------------ report


class _MailDetector:
    """Finds email addresses, and calls every "555-..." number a PHONE."""

    name = "silver_mail"

    def detect(self, text, **_):
        spans = [
            make_span("silvervocab", "mail", m.start(), m.end(), m.group(0))
            for m in re.finditer(EMAIL_RE, text)
        ]
        return {"spans": spans, "latency_ms": 0.0, "error": None}


register_vocab("silvervocab", {"mail": "EMAIL", "tel": "PHONE"}, kind="detector", overwrite=True)
registry.register_detector(
    registry.DetectorSpec(
        name="silver_mail", vocab="silvervocab", factory=lambda ctx: _MailDetector()
    ),
    overwrite=True,
)


@pytest.fixture
def run_on_unlabeled(unlabeled, tmp_path):
    run_dir = tmp_path / "run"
    runner.run(unlabeled, ["silver_mail"], run_dir)
    return run_dir


def test_report_refuses_scores_without_gold(run_on_unlabeled):
    md = report.build_report(run_on_unlabeled)
    assert "No gold labels, so no precision or recall" in md
    assert "silver.generate" in md and "review_run" in md
    assert "SemEval" not in md and "precision |" not in md
    assert "| silver_mail | 3 | 2 |" in md  # spans found, records with spans
    assert "| silver_mail | 3/3 | 0 | 0 |" in md  # coverage


def test_report_on_silver_labels(run_on_unlabeled, unlabeled, tmp_path):
    full = StubClient(regex_responder({"EMAIL": EMAIL_RE, "PHONE": PHONE_RE}), model="full")
    mails = StubClient(regex_responder({"EMAIL": EMAIL_RE}), model="mails")
    out = silver.generate(unlabeled, [full, mails], tmp_path / "mine.silver.jsonl",
                          merge_how="union", progress=False)
    md = report.build_report(run_on_unlabeled, fixtures=out)
    lines = md.splitlines()
    assert lines[0].startswith("# Silver-label report — ")
    assert lines[2] == (
        f"> Gold spans here are LLM silver labels from {describe(full)}, {describe(mails)},"
        " merged by union. Scores measure agreement with those labels, not ground truth."
    )
    assert f"- fixtures: `{out}`" in md
    # 5 silver spans (3 emails, 2 phones); the detector finds the 3 emails:
    # strict F1 = 2*3/(3+5) in the fair view (EMAIL, PHONE) and the raw view
    # (EMAIL, PERSON, PHONE).
    assert f"| silver_mail | 2 | {2 * 3 / 8:.3f} |" in md
    assert f"| silver_mail | 3 | {2 * 3 / 8:.3f} |" in md
    assert "| silver_mail | 3 | 0 | 0 | 2 | 0 |" in md  # COR INC PAR MIS SPU


def test_report_silver_single_labeler_and_error_records(run_on_unlabeled, unlabeled, tmp_path):
    def no_t1(system, user, schema):
        if "ann@x.io or call" in user:
            raise LLMError("refused")
        return regex_responder({"EMAIL": EMAIL_RE})(system, user, schema)

    client = StubClient(no_t1, model="one")
    out = silver.generate(unlabeled, [client], tmp_path / "s.jsonl", progress=False)
    assert read_meta(out)["error_ids"] == ["t1"]
    md = report.build_report(run_on_unlabeled, fixtures=out)
    assert f"from {describe(client)}. Scores measure agreement" in md
    assert "merged by" not in md
    assert "- left out: 1 records that no labeler could label" in md
    assert "| silver_mail | 2/2 | 0 | 0 |" in md  # t1 is not scored
    assert "| silver_mail | 2 | 1.000 | 1.000 | 1.000 | 1.000 |" in md


def test_report_silver_with_every_record_failed_prints_no_scores(
    run_on_unlabeled, unlabeled, tmp_path
):
    def down(system, user, schema):
        raise LLMError("down")

    client = StubClient(down, model="down")
    out = silver.generate(unlabeled, [client], tmp_path / "s.jsonl", progress=False)
    assert sorted(read_meta(out)["error_ids"]) == ["t1", "t2", "t3"]
    md = report.build_report(run_on_unlabeled, fixtures=out)
    assert md.splitlines()[0].startswith("# Silver-label report — ")
    assert "No silver labels, so no scores" in md
    name = silver.labeler_name(client)
    assert f"| {name} | 3 | `s.labeler_{name}.jsonl` |" in md
    assert "SemEval" not in md and "0.000" not in md
    assert "| silver_mail | 3 | 2 |" in md  # detected spans are still listed


def test_report_gold_title_unchanged(tmp_path):
    fx = tmp_path / "fx.jsonl"
    fx.write_text(json.dumps({"id": "1", "text": "ann@x.io", "gold_spans": [S(0, 8)]}) + "\n")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "raw_silver_mail.jsonl").write_text(json.dumps({
        "id": "1", "detector": "silver_mail", "latency_ms": 1, "error": None, "spans": [S(0, 8)]
    }) + "\n")
    (run_dir / "manifest.json").write_text(json.dumps({
        "started_at": "t", "fixtures": str(fx), "n_examples": 1, "detectors": ["silver_mail"],
        "labels": ["EMAIL"],
    }))
    md = report.build_report(run_dir)
    assert md.startswith("# PII detector benchmark — t") and "silver labels" not in md


def test_report_cli_with_silver_fixtures(
    run_on_unlabeled, unlabeled, tmp_path, monkeypatch, capsys
):
    out = silver.generate(unlabeled, [StubClient(regex_responder({"EMAIL": EMAIL_RE}))],
                          tmp_path / "s.jsonl", progress=False)
    monkeypatch.setattr(
        sys, "argv", ["report", "--run", str(run_on_unlabeled), "--fixtures", str(out)]
    )
    report.main()
    assert (run_on_unlabeled / "report.md").read_text().startswith("# Silver-label report")
    monkeypatch.setattr(
        sys, "argv", ["report", "--run", str(run_on_unlabeled), "--out", str(tmp_path / "u.md")]
    )
    report.main()
    assert "No gold labels" in (tmp_path / "u.md").read_text()


# --------------------------------------------------------------- calibrate


CALIB_ROWS = [
    {"id": "a", "text": "Mail alice@example.com or call 555-0101.", "spans": [
        {"start": 5, "end": 22, "label": "EMAIL"},
        {"start": 31, "end": 39, "label": "PHONE"},
    ]},
    {"id": "b", "text": "Jane wrote from jane@example.org today.", "spans": [
        {"start": 0, "end": 4, "label": "GIVEN_NAME"},
        {"start": 16, "end": 32, "label": "EMAIL"},
    ]},
    {"id": "c", "text": "nothing here", "spans": []},
]


@pytest.fixture
def calib_dataset(tmp_path):
    path = tmp_path / "calib.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in CALIB_ROWS))
    datasets.register_dataset("t_silver_calib", path=path, overwrite=True)
    return "t_silver_calib"


def gold_responder(skip_labels=()):
    """A labeler that answers with the dataset's own gold values."""
    by_text = {r["text"]: r["spans"] for r in CALIB_ROWS}

    def respond(system, user, schema):
        text = _text_of(user)
        return {"entities": [
            item(text[s["start"]:s["end"]], s["label"], text[max(0, s["start"] - 5):s["end"] + 5])
            for s in by_text[text] if s["label"] not in skip_labels
        ]}

    return respond


def test_calibrate_perfect_labeler(calib_dataset, tmp_path):
    client = StubClient(gold_responder(), model="oracle")
    res = silver.calibrate(client, dataset=calib_dataset, n=3, workdir=tmp_path / "w")
    for schema in ("strict", "exact", "partial", "ent_type"):
        assert res[schema]["f1"] == 1.0
    assert res["strict"]["correct"] == 4 and res["strict"]["missed"] == 0
    assert res["n"] == 3 and res["n_errors"] == 0
    assert res["labels"] == ["EMAIL", "PERSON", "PHONE"] and res["level"] == "coarse"
    assert res["labeler"] == describe(client) and res["dataset"] == calib_dataset
    assert set(res["by_label"]) == {"EMAIL", "PERSON", "PHONE"}
    # The labeler was asked for the dataset's annotated (fine) labels.
    assert read_meta(res["silver_fixtures"])["labels"] == ["EMAIL", "GIVEN_NAME", "PHONE"]
    assert all(r["gold_spans"] for r in read_jsonl(res["gold_fixtures"])[:2])  # gold untouched
    assert fixtures.is_unlabeled(tmp_path / "w" / f"calib_{calib_dataset}_n3_s42.unlabeled.jsonl")


def test_calibrate_fine_level_and_misses(calib_dataset, tmp_path):
    client = StubClient(gold_responder(skip_labels={"PHONE"}), model="nophone")
    res = silver.calibrate(client, dataset=calib_dataset, n=3, workdir=tmp_path, level="fine")
    assert res["labels"] == ["EMAIL", "GIVEN_NAME", "PERSON", "PHONE"]
    assert res["strict"]["precision"] == 1.0
    assert res["strict"]["recall"] == pytest.approx(3 / 4)
    assert res["by_label"]["PHONE"]["recall"] == 0.0


def test_calibrate_leaves_out_errored_records(calib_dataset, tmp_path):
    def flaky(system, user, schema):
        if "Jane" in _text_of(user):
            raise LLMError("refused")
        return gold_responder()(system, user, schema)

    res = silver.calibrate(StubClient(flaky), dataset=calib_dataset, n=3, workdir=tmp_path)
    assert res["n"] == 2 and res["n_errors"] == 1 and res["strict"]["f1"] == 1.0
