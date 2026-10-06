"""Unlabeled fixtures (`fixtures.from_texts` / `from_documents`): records with
empty gold, the `"gold": "none"` sidecar, the runner on top of them, and
mapping detector spans back to whole documents."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from opf_eval import fixtures, runner, taxonomy
from opf_eval.detectors import make_span, registry
from opf_eval.documents import Document, Segment, chunk, from_text, parse, parse_many
from opf_eval.io import meta_path, read_jsonl, read_meta, write_jsonl
from opf_eval.taxonomy import register_vocab

EMAIL = "jane.doe@example.com"
EMAIL_RE = re.compile(r"[\w.]+@[\w.]+\.\w+")

# A dependency-free detector that tags e-mail addresses, registered once for
# the runner tests. It records the label set the runner built it with.
register_vocab("unl_vocab", {"mail": "EMAIL"}, kind="detector", overwrite=True)
BUILT_WITH: list[frozenset[str] | None] = []


class EmailDetector:
    name = "unl_email"

    def detect(self, text: str, **_):
        spans = [make_span("unl_vocab", "mail", m.start(), m.end(), m.group()) for m in EMAIL_RE.finditer(text)]
        return {"spans": spans, "latency_ms": 0.1, "error": None}


@registry.detector("unl_email", vocab="unl_vocab", overwrite=True)
def _unl_email(ctx):
    BUILT_WITH.append(ctx.dataset_labels)
    return EmailDetector()


# ---------------------------------------------------------------- helpers


def _docs() -> list[Document]:
    """Two pasted documents; the second is long enough to need chunks."""
    short = from_text(f"Ticket 1\nCustomer {EMAIL} wrote in.", id="doc-a")
    lines = [f"line {i:02d} filler text" for i in range(30)]
    lines[25] = "contact bob@example.org today"
    long = from_text("\n".join(lines), id="doc-b", source="tickets/long.txt")
    return [short, long]


# ----------------------------------------------------------------- labels


def test_default_target_labels():
    fine = fixtures.default_target_labels()
    assert fine == sorted(taxonomy.ALL_LABELS)
    assert fixtures.default_target_labels("coarse") == sorted(taxonomy.COARSE_LABELS)
    assert set(fixtures.default_target_labels("coarse")) < set(fine)
    with pytest.raises(ValueError):
        fixtures.default_target_labels("medium")


# ------------------------------------------------------------- from_texts


def test_from_texts_sequence(tmp_path):
    out = fixtures.from_texts(["Call Jane at 555-0100", "   ", f"Mail {EMAIL}"], tmp_path / "pasted.jsonl",
                              language="en")
    assert out == tmp_path / "pasted.jsonl"
    rows = read_jsonl(out)
    # The blank text is left out but keeps its number.
    assert [r["id"] for r in rows] == ["text-0001", "text-0003"]
    assert rows[0] == {"id": "text-0001", "text": "Call Jane at 555-0100", "language": "en", "gold_spans": []}
    assert "doc_id" not in rows[0]

    meta = read_meta(out)
    assert meta["gold"] == fixtures.UNLABELED == "none"
    assert meta["dataset"] == "custom:pasted"
    assert meta["source"] == "texts"
    assert meta["vocab_key"] is None
    assert meta["labels"] == fixtures.default_target_labels()
    assert meta["n_written"] == 2
    assert meta["n_blank_skipped"] == 1
    assert meta["documents"] == {}
    assert meta["version"] == fixtures.META_VERSION
    assert meta["sha256"] and meta["created_at"]
    assert fixtures.is_unlabeled(out)


def test_from_texts_mapping_and_labels(tmp_path):
    out = fixtures.from_texts({"t1": "a", "ticket-9": "b"}, tmp_path / "m.jsonl", labels=["EMAIL", "PHONE", "EMAIL"])
    assert [r["id"] for r in read_jsonl(out)] == ["t1", "ticket-9"]
    assert read_meta(out)["labels"] == ["EMAIL", "PHONE"]
    with pytest.raises(ValueError, match="unknown label"):
        fixtures.from_texts(["x"], tmp_path / "bad.jsonl", labels=["EMAIL", "SHOE_SIZE"])
    with pytest.raises(ValueError, match="empty"):
        fixtures.from_texts(["x"], tmp_path / "bad.jsonl", labels=[])
    with pytest.raises(TypeError):
        fixtures.from_texts([b"bytes"], tmp_path / "bad.jsonl")


def test_validate_unlabeled_fixtures(tmp_path):
    out = fixtures.from_texts(["hello"], tmp_path / "u.jsonl")
    rep = fixtures.validate_fixtures(out)
    assert rep.ok and rep.unlabeled and rep.n_spans == 0
    assert "unlabeled (no gold spans)" in rep.summary()

    # A materialized-style file without the sidecar is not "unlabeled".
    plain = tmp_path / "plain.jsonl"
    write_jsonl(plain, [{"id": "x", "text": "hi", "gold_spans": []}])
    rep = fixtures.validate_fixtures(plain)
    assert rep.ok and not rep.unlabeled
    assert "0 gold spans" in rep.summary()
    assert not fixtures.is_unlabeled(plain)

    # Gold spans in a file marked unlabeled are suspicious: warn.
    rows = read_jsonl(out)
    rows[0]["gold_spans"] = [{"label": "PERSON", "start": 0, "end": 5}]
    write_jsonl(out, rows)
    rep = fixtures.validate_fixtures(out)
    assert rep.ok and any("unlabeled" in w for w in rep.warnings)


def test_validate_checks_chunk_offsets(tmp_path):
    p = tmp_path / "bad.jsonl"
    write_jsonl(p, [
        {"id": "a#0", "text": "x", "gold_spans": [], "doc_id": "a", "offset": -1},
        {"id": "a#1", "text": "y", "gold_spans": [], "doc_id": "a", "offset": "3"},
        {"id": "a#2", "text": "z", "gold_spans": [], "doc_id": "a", "offset": 2},
    ])
    rep = fixtures.validate_fixtures(p)
    assert len(rep.errors) == 2
    assert all("offset" in e for e in rep.errors)


# --------------------------------------------------------- from_documents


def test_from_documents_records_and_meta(tmp_path):
    docs = _docs()
    out = fixtures.from_documents(docs, tmp_path / "files.jsonl", max_chars=200, language="en")
    rows = read_jsonl(out)
    by_doc: dict[str, list[dict]] = {}
    for r in rows:
        by_doc.setdefault(r["doc_id"], []).append(r)
        assert r["gold_spans"] == [] and r["language"] == "en"
    assert [r["id"] for r in by_doc["doc-a"]] == ["doc-a#0"]
    assert len(by_doc["doc-b"]) == len(chunk(docs[1], max_chars=200)) > 1
    for doc in docs:
        expected = chunk(doc, max_chars=200)
        got = by_doc[doc.id]
        assert [r["id"] for r in got] == [f"{doc.id}#{k}" for k in range(len(expected))]
        assert [r["offset"] for r in got] == [a for a, _ in expected]
        for r, (a, b) in zip(got, expected):
            assert r["text"] == doc.text[a:b]
        assert "".join(r["text"] for r in got) == doc.text

    meta = read_meta(out)
    assert meta["source"] == "files" and meta["gold"] == "none"
    assert meta["dataset"] == "custom:files"
    assert meta["max_chars"] == 200
    assert meta["n_written"] == len(rows)
    assert meta["documents"]["doc-b"] == {
        "source": "tickets/long.txt", "kind": "text", "n_chunks": len(by_doc["doc-b"]), "warnings": [],
    }
    assert fixtures.validate_fixtures(out).ok
    assert fixtures.documents_path(out).exists()


def test_from_documents_drops_whitespace_only_chunks(tmp_path):
    # 18 + 2 trailing newlines with max_chars=19: the "\n\n" tail cannot join
    # the last chunk, so chunk() leaves it on its own and from_documents drops it.
    doc = from_text("x" * 12 + "\n\n" + "y" * 18 + "\n\n", id="d")
    ranges = chunk(doc, max_chars=19)
    assert doc.text[ranges[-1][0]:ranges[-1][1]] == "\n\n"
    out = fixtures.from_documents([doc], tmp_path / "w.jsonl", max_chars=19)
    rows = read_jsonl(out)
    assert [r["id"] for r in rows] == [f"d#{k}" for k in range(len(ranges) - 1)]
    assert all(r["text"].strip() for r in rows)
    assert [r["offset"] for r in rows] == [a for a, _ in ranges[:-1]]
    assert read_meta(out)["documents"]["d"]["n_chunks"] == len(rows)
    assert fixtures.validate_fixtures(out).ok


def test_from_documents_edge_cases(tmp_path):
    empty = Document("scan:empty.pdf", "empty.pdf", "pdf", "", [], ["page 1 has no text layer (needs OCR)"])
    out = fixtures.from_documents([empty, from_text("x", id="d")], tmp_path / "e.jsonl")
    assert [r["id"] for r in read_jsonl(out)] == ["d#0"]
    assert read_meta(out)["documents"]["scan:empty.pdf"] == {
        "source": "empty.pdf", "kind": "pdf", "n_chunks": 0, "warnings": ["page 1 has no text layer (needs OCR)"],
    }
    with pytest.raises(ValueError, match="duplicate document id"):
        fixtures.from_documents([from_text("x", id="d"), from_text("y", id="d")], tmp_path / "dup.jsonl")
    # Writing texts over a documents fixture removes the stale segment map.
    fixtures.from_texts(["t"], out)
    assert not fixtures.documents_path(out).exists()


def test_read_document_map_and_load_documents(tmp_path):
    docs = _docs()
    out = fixtures.from_documents(docs, tmp_path / "f.jsonl", max_chars=200)
    dmap = fixtures.read_document_map(out)
    assert dmap["doc-a#0"] == {"doc_id": "doc-a", "source": "<pasted>", "kind": "text", "offset": 0}
    assert all(v["doc_id"] == "doc-b" for k, v in dmap.items() if k.startswith("doc-b#"))

    loaded = fixtures.load_documents(out)
    assert list(loaded) == ["doc-a", "doc-b"]
    for doc in docs:
        got = loaded[doc.id]
        assert (got.text, got.segments, got.kind, got.source) == (doc.text, doc.segments, doc.kind, doc.source)

    # Hand-edited fixtures no longer line up with the sidecar.
    rows = read_jsonl(out)
    rows[-1]["text"] += "extra"
    write_jsonl(out, rows)
    with pytest.raises(ValueError, match="sidecar says"):
        fixtures.load_documents(out)
    fixtures.documents_path(out).unlink()
    with pytest.raises(FileNotFoundError):
        fixtures.load_documents(out)

    texts = fixtures.from_texts(["a"], tmp_path / "t.jsonl")
    assert fixtures.read_document_map(texts) == {}


def _raw(run_dir: Path, detector: str, rows: list[dict]) -> None:
    write_jsonl(run_dir / f"raw_{detector}.jsonl", rows)


def test_map_spans_to_documents_shifts_offsets(tmp_path):
    docs = _docs()
    out = fixtures.from_documents(docs, tmp_path / "f.jsonl", max_chars=200)
    records = read_jsonl(out)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    raw = []
    for rec in records:
        r = EmailDetector().detect(rec["text"])
        raw.append({"id": rec["id"], "detector": "unl_email", **r})
    _raw(run_dir, "unl_email", raw)

    rows = fixtures.map_spans_to_documents(out, run_dir, "unl_email", docs={d.id: d for d in docs})
    assert [(r["doc_id"], r["text"]) for r in rows] == [("doc-a", EMAIL), ("doc-b", "bob@example.org")]
    for r in rows:
        doc = next(d for d in docs if d.id == r["doc_id"])
        assert doc.text[r["doc_start"]:r["doc_end"]] == r["text"]
        assert r["label"] == r["fine_label"] == "EMAIL"
        assert r["detector"] == "unl_email"
    assert rows[0]["where"] == [{"line": 2}]
    assert rows[1]["where"] == [{"line": 26}]
    assert rows[1]["source"] == "tickets/long.txt"
    # The second document's span sits in a later chunk, so its offset was shifted.
    assert rows[1]["chunk_id"] != "doc-b#0"
    assert rows[1]["doc_start"] > 200

    # Without `docs` the segment sidecar fills `where`; a list of docs works too.
    assert fixtures.map_spans_to_documents(out, run_dir, "unl_email") == rows
    assert fixtures.map_spans_to_documents(out, run_dir, "unl_email", docs=docs) == rows
    fixtures.documents_path(out).unlink()
    bare = fixtures.map_spans_to_documents(out, run_dir, "unl_email")
    assert [r["where"] for r in bare] == [[], []]
    assert [r["doc_start"] for r in bare] == [r["doc_start"] for r in rows]


@pytest.mark.parametrize("files,max_chars", [
    # A whitespace-only file gives one blank chunk and no records.
    ({"notes.txt": f"Jane Doe, {EMAIL}\n", "blank.md": "\n"}, 4000),
    # The trailing newline does not fit in the last chunk.
    ({"a.txt": "A" * 39 + f" {EMAIL}" + "\n"}, 60),
    # A run of blank lines longer than a chunk sits between two lines.
    ({"b.txt": "x\n" + "\n" * 50 + f"{EMAIL}\n"}, 20),
])
def test_whitespace_only_chunks_still_rebuild_documents(tmp_path, files, max_chars):
    folder = tmp_path / "docs"
    folder.mkdir()
    for name, text in files.items():
        (folder / name).write_text(text)
    docs, _ = parse_many(folder)
    out = fixtures.from_documents(docs, tmp_path / "f.jsonl", max_chars=max_chars)
    records = read_jsonl(out)
    assert all(r["text"].strip() for r in records)
    loaded = fixtures.load_documents(out)
    for doc in docs:
        assert loaded[doc.id].text == doc.text
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _raw(run_dir, "unl_email", [{"id": r["id"], "detector": "unl_email", **EmailDetector().detect(r["text"])}
                                for r in records])
    rows = fixtures.map_spans_to_documents(out, run_dir, "unl_email")
    assert [r["text"] for r in rows] == [EMAIL]
    assert rows[0]["where"]


def test_map_spans_warns_when_the_sidecar_no_longer_fits(tmp_path):
    out = fixtures.from_documents([from_text(f"mail {EMAIL}", id="d")], tmp_path / "f.jsonl")
    rows = read_jsonl(out)
    rows[0]["text"] += " extra"
    write_jsonl(out, rows)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _raw(run_dir, "unl_email", [{"id": "d#0", "detector": "unl_email", **EmailDetector().detect(rows[0]["text"])}])
    with pytest.warns(UserWarning, match="sidecar says"):
        got = fixtures.map_spans_to_documents(out, run_dir, "unl_email")
    assert [(r["text"], r["where"]) for r in got] == [(EMAIL, [])]


def test_map_spans_skips_non_document_records(tmp_path):
    out = fixtures.from_texts([f"mail {EMAIL}"], tmp_path / "t.jsonl")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _raw(run_dir, "d", [{"id": "text-0001", "detector": "d", "spans": [
        {"label": "EMAIL", "raw_label": "x", "start": 5, "end": 25, "text": EMAIL}], "latency_ms": 1, "error": None}])
    assert fixtures.map_spans_to_documents(out, run_dir, "d") == []


# ----------------------------------------------------------------- runner


def test_resolve_labels_reads_target_labels(tmp_path):
    out = fixtures.from_texts(["x"], tmp_path / "u.jsonl", labels=["EMAIL", "PERSON"])
    name, vocab, labels = runner.resolve_labels(out, read_jsonl(out), None)
    assert (name, vocab, labels) == ("custom:u", None, {"EMAIL", "PERSON"})


def test_runner_on_unlabeled_document_fixtures(tmp_path):
    root = tmp_path / "inbox"
    root.mkdir()
    (root / "people.csv").write_text(f"name,email\nJane Doe,{EMAIL}\n", encoding="utf-8")
    (root / "note.md").write_text("# Note\n\nPing bob@example.org\n", encoding="utf-8")
    docs, skipped = parse_many(root)
    assert skipped == []
    fx = fixtures.from_documents(docs, tmp_path / "data" / "inbox.jsonl")

    BUILT_WITH.clear()
    manifest = runner.run(fx, ["unl_email"], tmp_path / "runs" / "inbox")
    assert manifest["labels"] == fixtures.default_target_labels()
    assert manifest["dataset"] == "custom:inbox"
    assert manifest["vocab_key"] is None
    assert manifest["n_examples"] == len(docs)
    assert BUILT_WITH == [frozenset(fixtures.default_target_labels())]

    rows = fixtures.map_spans_to_documents(fx, tmp_path / "runs" / "inbox", "unl_email")
    where = {r["text"]: r["where"] for r in rows}
    assert where == {
        "bob@example.org": [{"line": 3}],
        EMAIL: [{"row": 2, "column": "email"}],
    }


def test_runner_respects_custom_target_labels(tmp_path):
    fx = fixtures.from_texts([f"mail {EMAIL}"], tmp_path / "u.jsonl", labels=["EMAIL"])
    BUILT_WITH.clear()
    manifest = runner.run(fx, ["unl_email"], tmp_path / "run")
    assert manifest["labels"] == ["EMAIL"]
    assert BUILT_WITH == [frozenset({"EMAIL"})]
    raw = read_jsonl(tmp_path / "run" / "raw_unl_email.jsonl")
    assert raw[0]["spans"][0]["text"] == EMAIL
    assert json.loads((tmp_path / "run" / "manifest.json").read_text())["labels"] == ["EMAIL"]


def test_xlsx_end_to_end_locates_cell(tmp_path):
    openpyxl = pytest.importorskip("openpyxl")
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "People"
    ws.append(["Name", "Email"])
    ws.append(["Jane Doe", EMAIL])
    path = tmp_path / "people.xlsx"
    wb.save(path)

    doc = parse(path)
    fx = fixtures.from_documents([doc], tmp_path / "x.jsonl", max_chars=20)
    assert len(read_jsonl(fx)) > 1  # forced into several chunks
    runner.run(fx, ["unl_email"], tmp_path / "run")
    rows = fixtures.map_spans_to_documents(fx, tmp_path / "run", "unl_email")
    assert [(r["text"], r["where"]) for r in rows] == [(EMAIL, [{"sheet": "People", "cell": "B2"}])]
    assert meta_path(fx).exists()


def test_segments_survive_json_round_trip(tmp_path):
    doc = Document("d", "s", "xlsx", "a\tb", [Segment(0, 1, {"sheet": "S", "cell": "A1"}),
                                              Segment(2, 3, {"sheet": "S", "cell": "B1"})])
    fx = fixtures.from_documents([doc], tmp_path / "s.jsonl")
    assert fixtures.load_documents(fx)["d"].segments == doc.segments


def test_gold_meta_values_are_shared():
    # fixtures, silver, report and nb all spell the sidecar's "gold" values
    # through opf_eval.io so a writer and a reader cannot drift apart.
    from opf_eval import io, silver
    from opf_eval import fixtures as fx

    assert fx.UNLABELED == io.GOLD_NONE == "none"
    assert silver.SILVER == io.GOLD_SILVER == "silver"
