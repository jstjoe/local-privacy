"""Custom datasets: registration, file readers, schema loaders, fixtures."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from opf_eval import datasets, fixtures
from opf_eval.datasets.loaders import bio_spans, privacy_mask_loader
from opf_eval.io import read_jsonl, read_meta


def _jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


TEXT = "Mail alice@example.com or call 555-0101."
ROWS = [
    {"id": "a", "text": TEXT, "spans": [
        {"start": 5, "end": 22, "label": "EMAIL"},
        {"start": 31, "end": 39, "label": "PHONE"},
    ]},
    {"id": "b", "text": "SSN 123-45-6789", "spans": [{"start": 4, "end": 15, "label": "GOV_ID"}]},
    {"id": "c", "text": "nothing here", "spans": []},
]


@pytest.fixture
def jsonl_dataset(tmp_path):
    path = _jsonl(tmp_path / "mine.jsonl", ROWS)
    datasets.register_dataset("t_mine", path=path, overwrite=True)
    return path


def test_register_local_canonical_dataset(jsonl_dataset, tmp_path):
    out = tmp_path / "fx.jsonl"
    n = fixtures.materialize(out, None, dataset="t_mine")
    assert n == 3
    recs = read_jsonl(out)
    assert [r["id"] for r in recs] == ["a", "b", "c"]  # n=None keeps source order
    gov = recs[1]["gold_spans"][0]
    assert gov["label"] == "ACCOUNT" and gov["fine_label"] == "GOV_ID"
    meta = read_meta(out)
    # Identity vocab: annotated labels = what the gold actually contains.
    assert meta["labels"] == ["EMAIL", "GOV_ID", "PHONE"]
    assert meta["dataset"] == "t_mine"


def test_label_map_and_unmapped_labels_dropped(tmp_path):
    path = _jsonl(tmp_path / "raw.jsonl", [
        {"id": "1", "body": "Dr. Ada, MRN 998877", "pii": [
            {"begin": 4, "end": 7, "kind": "patient"},
            {"begin": 13, "end": 19, "kind": "mrn"},
            {"begin": 0, "end": 3, "kind": "honorific"},  # not in label_map -> dropped
        ]},
    ])
    datasets.register_dataset(
        "t_claims", path=path, overwrite=True,
        label_map={"patient": "PERSON", "mrn": "MEDICAL_ID"},
        text_field="body", spans_field="pii", label_key="kind", start_key="begin",
    )
    out = tmp_path / "fx.jsonl"
    fixtures.materialize(out, None, dataset="t_claims")
    (rec,) = read_jsonl(out)
    assert [(s["fine_label"], s["raw_label"]) for s in rec["gold_spans"]] == [
        ("PERSON", "patient"), ("MEDICAL_ID", "mrn"),
    ]
    assert set(read_meta(out)["labels"]) == {"PERSON", "MEDICAL_ID"}


def test_csv_row_per_span(tmp_path):
    path = tmp_path / "d.csv"
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "text", "label", "start", "end"])
        w.writeheader()
        w.writerow({"id": "1", "text": TEXT, "label": "EMAIL", "start": 5, "end": 22})
        w.writerow({"id": "1", "text": TEXT, "label": "PHONE", "start": 31, "end": 39})
        w.writerow({"id": "2", "text": "clean", "label": "", "start": "", "end": ""})
    datasets.register_dataset("t_csv", path=path, overwrite=True)
    out = tmp_path / "fx.jsonl"
    assert fixtures.materialize(out, None, dataset="t_csv") == 2
    recs = read_jsonl(out)
    assert [len(r["gold_spans"]) for r in recs] == [2, 0]


def test_csv_spans_column(tmp_path):
    path = tmp_path / "d.csv"
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["text", "spans"])
        w.writeheader()
        w.writerow({"text": TEXT, "spans": json.dumps(ROWS[0]["spans"])})
    datasets.register_dataset("t_csv2", path=path, overwrite=True)
    out = tmp_path / "fx.jsonl"
    fixtures.materialize(out, None, dataset="t_csv2")
    assert len(read_jsonl(out)[0]["gold_spans"]) == 2


def test_bio_spans_offsets():
    tokens = ["Call", "Ana", "Silva", "at", "555", "-", "0101", "."]
    tags = ["O", "B-NAME", "I-NAME", "O", "B-TEL", "I-TEL", "I-TEL", "O"]
    ws = [True, True, True, True, False, False, False, False]
    text, spans = bio_spans(tokens, tags, whitespace=ws)
    assert text == "Call Ana Silva at 555-0101."
    assert [(lbl, text[s:e]) for lbl, s, e in spans] == [("NAME", "Ana Silva"), ("TEL", "555-0101")]


def test_bio_spans_bioes_and_lenient_i():
    _, spans = bio_spans(["a", "b", "c", "d"], ["S-X", "I-Y", "E-Y", "I-X"])
    assert [lbl for lbl, _, _ in spans] == ["X", "Y", "X"]


def test_bio_loader_with_int_tags(tmp_path):
    path = _jsonl(tmp_path / "bio.jsonl", [{"tokens": ["hi", "bob"], "ner_tags": [0, 1]}])
    datasets.register_dataset(
        "t_bio", path=path, schema="bio", overwrite=True,
        label_map={"PER": "PERSON"}, tag_names=["O", "B-PER"],
    )
    out = tmp_path / "fx.jsonl"
    fixtures.materialize(out, None, dataset="t_bio")
    (rec,) = read_jsonl(out)
    assert rec["text"] == "hi bob"
    assert rec["gold_spans"][0]["start"] == 3


def test_privacy_mask_loader_shape():
    """The ai4privacy loader keeps its historical record shape."""
    loader = privacy_mask_loader("pii300k")
    (rec,) = loader([{
        "id": 7, "source_text": "Hi Ana", "language": "English",
        "privacy_mask": [{"label": "GIVENNAME1", "start": 3, "end": 6, "value": "Ana"}],
    }])
    assert rec == {
        "id": "7", "text": "Hi Ana", "language": "en",
        "gold_spans": [{"label": "PERSON", "fine_label": "GIVEN_NAME",
                        "raw_label": "GIVENNAME1", "start": 3, "end": 6}],
    }


def test_sampling_is_seeded(jsonl_dataset, tmp_path):
    a, b, c = tmp_path / "a.jsonl", tmp_path / "b.jsonl", tmp_path / "c.jsonl"
    fixtures.materialize(a, 2, dataset="t_mine", seed=1)
    fixtures.materialize(b, 2, dataset="t_mine", seed=1)
    assert a.read_text() == b.read_text()
    fixtures.materialize(c, 3, dataset="t_mine", seed=2)
    assert len(read_jsonl(c)) == 3


def test_ensure_fixtures_reuses_only_matching_sample(jsonl_dataset, tmp_path):
    out = tmp_path / "fx.jsonl"
    _, reused = fixtures.ensure_fixtures(out, 2, dataset="t_mine", seed=1)
    assert not reused
    _, reused = fixtures.ensure_fixtures(out, 2, dataset="t_mine", seed=1)
    assert reused
    # A different seed must not silently reuse the old sample.
    _, reused = fixtures.ensure_fixtures(out, 2, dataset="t_mine", seed=2)
    assert not reused
    # Hand-edited file -> hash mismatch -> regenerated.
    out.write_text(out.read_text() + "\n")
    _, reused = fixtures.ensure_fixtures(out, 2, dataset="t_mine", seed=2)
    assert not reused


def test_out_of_bounds_gold_spans_dropped(tmp_path):
    path = _jsonl(tmp_path / "bad.jsonl", [
        {"id": "1", "text": "short", "spans": [{"start": 0, "end": 99, "label": "EMAIL"}]},
    ])
    datasets.register_dataset("t_oob", path=path, overwrite=True)
    out = tmp_path / "fx.jsonl"
    fixtures.materialize(out, None, dataset="t_oob")
    assert read_jsonl(out)[0]["gold_spans"] == []
    assert read_meta(out)["dropped_spans"] == {"out_of_bounds": 1}


def test_validate_fixtures(tmp_path):
    path = _jsonl(tmp_path / "fx.jsonl", [
        {"id": "1", "text": "abc", "gold_spans": [{"label": "EMAIL", "start": 0, "end": 2}]},
        {"id": "1", "text": "dup id", "gold_spans": []},
        {"id": "2", "text": "abc", "gold_spans": [{"label": "EMAIL", "start": 2, "end": 9}]},
        {"id": "3", "text": "abcdef", "gold_spans": [
            {"label": "WHATEVER", "start": 0, "end": 3},
            {"label": "EMAIL", "start": 1, "end": 4},
        ]},
    ])
    rep = fixtures.validate_fixtures(path)
    assert not rep.ok
    assert any("duplicate id" in e for e in rep.errors)
    assert any("outside text" in e for e in rep.errors)
    assert any("not canonical" in w for w in rep.warnings)
    assert any("overlaps" in w for w in rep.warnings)


def test_explicit_labels_override(tmp_path):
    path = _jsonl(tmp_path / "d.jsonl", ROWS)
    cfg = datasets.register_dataset(
        "t_labels", path=path, overwrite=True, labels=["EMAIL", "PHONE", "GOV_ID", "PERSON"]
    )
    assert datasets.annotated_labels(cfg, level="coarse") == {"EMAIL", "PHONE", "ACCOUNT", "PERSON"}
    with pytest.raises(ValueError, match="unknown canonical"):
        datasets.register_dataset("t_badlabels", path=path, labels=["NOPE"], overwrite=True)


def test_builtin_datasets_still_registered():
    assert {"pii_masking_300k", "pii_masking_200k", "pii_masking_400k",
            "openpii_nano", "openpii_mini"} <= set(datasets.names())
    assert datasets.get("pii_masking_300k").vocab_key == "pii300k"
