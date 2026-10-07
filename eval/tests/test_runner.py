"""Unit tests for `opf_eval.runner`: the cleanup helpers, the raw-file writer and an
end-to-end smoke test."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from opf_eval import runner as runner_mod
from opf_eval.runner import _clear_run_dir, _free_detector, _run_one, run

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
    spacy = pytest.importorskip("spacy")
    # Without the model Presidio downloads about 400 MB, and tests never download models.
    if not spacy.util.is_package("en_core_web_lg"):
        pytest.skip(
            "spaCy model en_core_web_lg is not installed "
            "(python -m spacy download en_core_web_lg)"
        )

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


# ----------------------------- _clear_run_dir ---------------------------------


def _fill_run_dir(out: Path) -> tuple[list[str], list[str]]:
    """Write a run dir's results plus some unrelated files and return both name lists."""
    out.mkdir(parents=True, exist_ok=True)
    results = [
        "raw_presidio.jsonl", "manifest.json", "report.md",
        "review_llm_anthropic_claude-x.jsonl", "review_llm_anthropic_claude-x.meta.json",
        "review_clef.jsonl", "review_clef.meta.json",
        # A sidecar whose review file is already gone, and a temp file from a killed run.
        "review_old.meta.json", ".raw_gliner.jsonl.123.tmp",
    ]
    unrelated = ["notes.md", "reviewer_notes.txt", "my_review.jsonl", "export.csv"]
    for name in results + unrelated:
        (out / name).write_text("{}\n")
    return results, unrelated


def test_clear_run_dir_removes_reviews_and_keeps_other_files(tmp_path: Path):
    out = tmp_path / "out"
    results, unrelated = _fill_run_dir(out)
    _clear_run_dir(out)
    assert [n for n in results if (out / n).exists()] == []
    assert sorted(p.name for p in out.iterdir()) == sorted(unrelated)


class _Echo:
    name = "echo"

    def __init__(self, fail_on: str | None = None):
        self.fail_on = fail_on

    def detect(self, text, **_ctx):
        if text == self.fail_on:
            raise RuntimeError("detect failure")
        return {"spans": [], "latency_ms": 0.0, "error": None}


def _fixtures(path: Path, texts: list[str]) -> Path:
    path.write_text("".join(
        json.dumps({"id": f"x{i}", "text": t, "gold_spans": []}) + "\n" for i, t in enumerate(texts)
    ))
    return path


def test_run_fresh_clears_review_files(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(runner_mod, "_build_detector", lambda name, **_kw: _Echo())
    out = tmp_path / "out"
    results, unrelated = _fill_run_dir(out)
    fx = _fixtures(tmp_path / "fx.jsonl", ["a", "b"])
    run(fixtures=fx, detector_names=["echo"], out_dir=out, device="cpu", fresh=True)
    left = sorted(p.name for p in out.iterdir())
    assert left == sorted([*unrelated, "manifest.json", "raw_echo.jsonl"])


# ----------------------------- _run_one ---------------------------------------


@pytest.mark.parametrize("workers", [1, 3])
def test_run_one_leaves_no_partial_raw_file(tmp_path: Path, workers: int):
    examples = [{"id": f"x{i}", "text": t} for i, t in enumerate(["a", "b", "boom", "c"])]
    out_path = tmp_path / "raw_echo.jsonl"
    with pytest.raises(RuntimeError, match="detect failure"):
        _run_one("echo", _Echo(fail_on="boom"), examples, out_path,
                 remote=True, workers=workers, min_interval_ms=0.0)
    # Neither the rows written before the failure nor the temp file survive.
    assert list(tmp_path.iterdir()) == []


def test_run_one_failure_keeps_the_previous_raw_file(tmp_path: Path):
    examples = [{"id": f"x{i}", "text": t} for i, t in enumerate(["a", "boom"])]
    out_path = tmp_path / "raw_echo.jsonl"
    out_path.write_text('{"id": "x0"}\n{"id": "x1"}\n')
    with pytest.raises(RuntimeError):
        _run_one("echo", _Echo(fail_on="boom"), examples, out_path,
                 remote=False, workers=1, min_interval_ms=0.0)
    assert out_path.read_text() == '{"id": "x0"}\n{"id": "x1"}\n'
    assert sorted(p.name for p in tmp_path.iterdir()) == ["raw_echo.jsonl"]


def test_run_one_writes_every_row_on_success(tmp_path: Path):
    examples = [{"id": f"x{i}", "text": t} for i, t in enumerate(["a", "b", "c"])]
    out_path = tmp_path / "raw_echo.jsonl"
    _run_one("echo", _Echo(), examples, out_path, remote=False, workers=1, min_interval_ms=0.0)
    rows = [json.loads(line) for line in out_path.read_text().splitlines()]
    assert [r["id"] for r in rows] == ["x0", "x1", "x2"]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["raw_echo.jsonl"]
