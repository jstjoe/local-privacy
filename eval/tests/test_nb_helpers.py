"""Notebook helper modules: session, tables, secrets, demos, error analysis."""

from __future__ import annotations

import json
import re

import pytest

from opf_eval import nb, scoring
from opf_eval.demo import sanitize, search


@pytest.fixture
def ws_root(tmp_path, monkeypatch):
    monkeypatch.setenv("PII_BENCH_HOME", str(tmp_path / "ws"))
    return tmp_path / "ws"


def test_session_roundtrip_and_derived_paths(ws_root):
    s = nb.session(dataset="openpii_nano", n=50, seed=7, detectors=["presidio"])
    assert s.fixtures_path == ws_root / "data" / "openpii_nano_50_s7.jsonl"
    assert s.run_dir == ws_root / "results" / "runs" / "openpii_nano_50_s7"
    loaded = nb.load_session(quiet=True)
    assert loaded == s
    # Updating one field keeps the others.
    s2 = nb.session(n=200)
    assert (s2.dataset, s2.n, s2.seed) == ("openpii_nano", 200, 7)
    assert s2.run_dir.name == "openpii_nano_200_s7"


def test_load_session_defaults_without_file(ws_root):
    s = nb.load_session(quiet=True)
    assert s.dataset == nb.DEFAULT_DATASET
    assert s.detectors == list(nb.DEFAULT_DETECTORS)


def test_md_table():
    md = nb.md_table([{"a": 1.23456, "b": None, "c": "x|y"}], headers={"a": "A"})
    assert md.splitlines() == ["| A | b | c |", "|---|---|---|", "| 1.235 | — | x\\|y |"]
    assert nb.md_table([]) == "_(no rows)_"


def test_load_secrets_from_dotenv(tmp_path, monkeypatch):
    monkeypatch.delenv("NB_TEST_SECRET", raising=False)
    env = tmp_path / ".env"
    env.write_text("# comment\nexport NB_TEST_SECRET='s3cret'\n")
    found = nb.load_secrets("NB_TEST_SECRET", "NB_TEST_MISSING", dotenv=env)
    assert found == {"NB_TEST_SECRET": True, "NB_TEST_MISSING": False}
    import os

    assert os.environ["NB_TEST_SECRET"] == "s3cret"


_EMAIL = re.compile(r"[\w.]+@[\w.]+\w")


class _EmailDetector:
    def detect(self, text, **_):
        return {
            "spans": [
                {"label": "EMAIL", "raw_label": "e", "start": m.start(), "end": m.end(), "text": m.group()}
                for m in _EMAIL.finditer(text)
            ],
            "latency_ms": 0.0,
            "error": None,
        }


def test_search_corpus_shape():
    corpus, queries = search.build_corpus()
    assert len(corpus) == 130
    assert [sorted(q.relevant) for q in queries] == [[0, 1, 2, 3, 4], [5, 6, 7]]
    assert search.tokenize("see [EMAIL_u8UBDWQ] and [EMAIL].") == ["see", "[email_u8ubdwq]", "and", "email"]


def test_search_demo_end_to_end():
    pytest.importorskip("bm25s")
    corpus, queries = search.build_corpus()
    det = _EmailDetector()
    modes = ["redact", "label", "label_number"]
    by_doc = search.sanitize_all(det, corpus, modes)
    by_q = search.sanitize_all(det, [q.text for q in queries], modes)
    rows = search.score(queries, by_q, search.retrieve(by_doc, by_q, k=5))
    v = {r["mode"]: r for r in search.verdict(rows)}
    assert v["plain"]["avg_recall"] == 1.0
    # Privacy-preserving modes that can't tell values apart can't find alice.
    assert v["label"]["avg_recall"] < 0.5


def test_sanitize_side_by_side_markdown():
    fx = {"id": "x", "text": "mail bob@example.com"}
    spans = _EmailDetector().detect(fx["text"])["spans"]
    md = sanitize.side_by_side(fx, {"regex": spans}, modes=["redact", "label"])
    assert "| regex | mail ******** | mail [EMAIL] |" in md


def test_error_examples(tmp_path):
    fx = tmp_path / "fx.jsonl"
    fx.write_text(json.dumps({
        "id": "1", "text": "ann@x.io and bob@y.io call 555-0101",
        "gold_spans": [
            {"label": "EMAIL", "start": 0, "end": 8},
            {"label": "EMAIL", "start": 13, "end": 21},
            {"label": "PHONE", "start": 27, "end": 35},
        ],
    }) + "\n")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "raw_det.jsonl").write_text(json.dumps({
        "id": "1", "detector": "det", "latency_ms": 1, "error": None,
        "spans": [
            {"label": "EMAIL", "start": 0, "end": 8},
            {"label": "PHONE", "start": 13, "end": 21},   # mislabeled
            {"label": "EMAIL", "start": 22, "end": 26},   # spurious
        ],
    }) + "\n")
    (run_dir / "manifest.json").write_text(json.dumps({
        "started_at": "t", "fixtures": str(fx), "n_examples": 1, "detectors": ["det"],
        "labels": ["EMAIL", "PHONE"],
    }))
    run = scoring.Run.load(run_dir)
    kw = dict(view="raw")
    assert [e["span"] for e in scoring.errors(run, "det", kind="missed", **kw)] == ["555-0101"]
    assert [e["span"] for e in scoring.errors(run, "det", kind="spurious", **kw)] == ["call"]
    (mis,) = scoring.errors(run, "det", kind="mislabeled", **kw)
    assert mis["span"] == "bob@y.io" and mis["predicted"] == "PHONE: bob@y.io"
