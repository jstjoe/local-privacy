"""Shipped baselines: export a run, seed a later run from it, Drive setup."""

from __future__ import annotations

import json

import pytest

from opf_eval import baseline, datasets, nb
from opf_eval.io import meta_path, read_jsonl, write_jsonl


@pytest.fixture
def ws_root(tmp_path, monkeypatch):
    monkeypatch.setenv("PII_BENCH_HOME", str(tmp_path / "ws"))
    return tmp_path / "ws"


@pytest.fixture
def base_root(tmp_path, monkeypatch):
    root = tmp_path / "baselines"
    monkeypatch.setattr(baseline, "BASELINES_DIR", root)
    return root


@pytest.fixture
def emails(tmp_path):
    path = tmp_path / "emails.jsonl"
    rows = [
        {"id": f"r{i}", "text": f"write to user{i}@example.com",
         "spans": [{"start": 9, "end": 9 + len(f"user{i}@example.com"), "label": "EMAIL"}]}
        for i in range(10)
    ]
    write_jsonl(path, rows)
    datasets.register_dataset("t_emails", path=path, overwrite=True)
    return path


def _fake_raw(fixtures, run_dir, det):
    run_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        {"id": r["id"], "detector": det, "latency_ms": 1.0, "error": None,
         "spans": [{**s, "text": r["text"][s["start"]:s["end"]]} for s in r["gold_spans"]]}
        for r in read_jsonl(fixtures)
    ]
    write_jsonl(run_dir / f"raw_{det}.jsonl", rows)
    (run_dir / "manifest.json").write_text(json.dumps({"detectors": [det], "detector_options": {}}))


def test_export_then_seed_a_smaller_sample(ws_root, base_root, emails):
    big = nb.session(dataset="t_emails", n=8, seed=1, detectors=["regex"])
    _fake_raw(nb.ensure_fixtures(big), big.run_dir, "regex")
    out = nb.export_baseline(big)
    info = json.loads((out / "baseline.json").read_text())
    key = (info["dataset"], info["seed"], info["n"], info["detectors"])
    assert key == ("t_emails", 1, 8, ["regex"])
    assert "text" not in json.dumps(info).replace("text_sha1", "")  # no fixture text stored

    # Same seed, fewer records: a prefix of the exported sample.
    small = nb.session(n=4)
    run_dir = nb.ensure_run(small)
    assert sorted(r["id"] for r in read_jsonl(run_dir / "raw_regex.jsonl")) == sorted(
        r["id"] for r in read_jsonl(small.fixtures_path)
    )
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["detectors"] == ["regex"] and manifest["n_examples"] == 4


def test_seed_skips_on_mismatch(ws_root, base_root, emails, tmp_path):
    s = nb.session(dataset="t_emails", n=8, seed=1, detectors=["regex"])
    fx = nb.ensure_fixtures(s)
    _fake_raw(fx, s.run_dir, "regex")
    baseline.export(fx, s.run_dir)

    # Different detector options -> not used.
    options = {"regex": {"t": 1}}
    assert baseline.seed_run_dir(fx, tmp_path / "a", ["regex"], detector_options=options) == []
    # Unknown detector -> not used.
    assert baseline.seed_run_dir(fx, tmp_path / "b", ["other"]) == []
    # A record's text changed (e.g. a new dataset revision) -> not used.
    rows = read_jsonl(fx)
    rows[0]["text"] += "!"
    changed = tmp_path / "changed.jsonl"
    write_jsonl(changed, rows)
    meta_path(changed).write_text(meta_path(fx).read_text())
    assert baseline.seed_run_dir(changed, tmp_path / "c", ["regex"]) == []
    # Matching -> used.
    assert baseline.seed_run_dir(fx, tmp_path / "d", ["regex"]) == ["regex"]


def test_ensure_run_without_baseline_flag_does_not_seed(ws_root, base_root, emails, monkeypatch):
    s = nb.session(dataset="t_emails", n=4, seed=1, detectors=["regex"])
    _fake_raw(nb.ensure_fixtures(s), s.run_dir, "regex")
    baseline.export(s.fixtures_path, s.run_dir)
    s2 = nb.session(run_name="fresh")
    called = {}
    monkeypatch.setattr(
        "opf_eval.runner.run", lambda fx, todo, *a, **k: called.setdefault("todo", todo)
    )
    nb.ensure_run(s2, baseline=False)
    assert called["todo"] == ["regex"]


def test_setup_drive_is_ignored_outside_colab(ws_root, capsys):
    ws = nb.setup(drive=True, quiet=False)
    assert ws.root == ws_root
    assert "only applies on Colab" in capsys.readouterr().out
