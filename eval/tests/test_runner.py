"""Unit tests for `opf_eval.runner` cleanup helper + end-to-end smoke."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from opf_eval.runner import _free_detector, run


# ----------------------------- _free_detector ---------------------------------


def test_free_detector_no_close_no_attrs():
    """Bare object: helper must not raise."""
    class Bare:
        pass

    _free_detector(Bare())


def test_free_detector_calls_close_and_nulls_heavy_attrs():
    """A detector with close() and heavy attrs: close fires, attrs go to None."""
    class FakeDet:
        def __init__(self):
            self._opf = object()
            self._model = object()
            self._pipe = object()
            self._loaders = {"en": object()}
            self.closed = False

        def close(self):
            self.closed = True

    d = FakeDet()
    _free_detector(d)
    assert d.closed, "close() was not called"
    # Heavy attrs are nulled by the fallback path even when close() doesn't
    # touch them — defense-in-depth for detectors that forget to.
    assert d._opf is None
    assert d._model is None
    assert d._pipe is None


def test_free_detector_swallows_close_exception():
    """A failing close() must not propagate — cleanup is best-effort."""
    class BadClose:
        def close(self):
            raise RuntimeError("boom")

    _free_detector(BadClose())  # no raise


def test_free_detector_handles_missing_torch(monkeypatch):
    """If torch isn't importable, the CUDA branch is skipped without error."""
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *a, **kw):
        if name == "torch":
            raise ImportError("torch not installed")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    _free_detector(object())  # no raise


# ----------------------------- runner.run smoke -------------------------------


def test_runner_run_writes_raw_file_and_invokes_cleanup(tmp_path: Path):
    """End-to-end: runner.run iterates detectors, writes raw JSONL, and the
    finally-block cleanup fires (verified by patching _free_detector)."""
    fx = tmp_path / "fx.jsonl"
    fx.write_text(
        json.dumps(
            {
                "id": "x1",
                "text": "Contact alice@example.com for info.",
                "language": "en",
                "gold_spans": [{"label": "EMAIL", "start": 8, "end": 25}],
            }
        )
        + "\n"
    )
    out = tmp_path / "out"

    pytest.importorskip("presidio_analyzer")
    pytest.importorskip("spacy")

    run(fixtures=fx, detector_names=["presidio"], out_dir=out, device="cpu")

    raw = out / "raw_presidio.jsonl"
    assert raw.exists()
    rows = [json.loads(line) for line in raw.read_text().splitlines() if line.strip()]
    assert len(rows) == 1
    assert rows[0]["id"] == "x1"
    assert rows[0]["detector"] == "presidio"
    assert any(s["label"] == "EMAIL" for s in rows[0]["spans"])


def test_runner_run_cleans_up_on_detector_exception(tmp_path: Path, monkeypatch):
    """If detector building succeeds but detect() raises, _free_detector
    must still run via the try/finally."""
    fx = tmp_path / "fx.jsonl"
    fx.write_text(json.dumps({"id": "x1", "text": "hi", "language": "en"}) + "\n")
    out = tmp_path / "out"

    freed: list[object] = []

    class ExplodingDetector:
        name = "exploder"

        def __init__(self):
            self._model = object()

        def detect(self, text, **_ctx):
            raise RuntimeError("detect failure")

    from opf_eval import runner as runner_mod

    monkeypatch.setattr(runner_mod, "_build_detector", lambda name, **_kw: ExplodingDetector())
    real_free = runner_mod._free_detector

    def tracking_free(det):
        freed.append(det)
        real_free(det)

    monkeypatch.setattr(runner_mod, "_free_detector", tracking_free)

    with pytest.raises(RuntimeError, match="detect failure"):
        run(fixtures=fx, detector_names=["exploder"], out_dir=out, device="cpu")

    assert len(freed) == 1, "cleanup did not run after detect() raised"
    assert freed[0]._model is None, "_free_detector did not null heavy attr on exception path"
