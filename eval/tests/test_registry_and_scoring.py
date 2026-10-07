"""Detector registry, runner run-dir guards, and fine-level scoring — all
end-to-end with a stub detector, so no model downloads are needed."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from opf_eval import runner, scoring
from opf_eval.detectors import make_span, registry
from opf_eval.report import build_report
from opf_eval.taxonomy import register_vocab


class KeywordDetector:
    """Tags fixed substrings — deterministic, dependency-free."""

    name = "kw"

    def __init__(self, table: dict[str, str]):
        self.table = table

    def detect(self, text: str, **_):
        spans = []
        for needle, raw in self.table.items():
            i = text.find(needle)
            if i != -1:
                spans.append(make_span("kwvocab", raw, i, i + len(needle), needle))
        return {"spans": spans, "latency_ms": 1.0, "error": None}


register_vocab(
    "kwvocab",
    {"mail": "EMAIL", "ssn": "GOV_ID", "acct": "ACCOUNT", "who": "PERSON"},
    kind="detector",
    overwrite=True,
)
registry.register_detector(
    registry.DetectorSpec(
        name="kw_fine",
        vocab="kwvocab",
        factory=lambda ctx: KeywordDetector(
            {"alice@example.com": "mail", "123-45-6789": "ssn", "Alice": "who"}
        ),
    ),
    overwrite=True,
)


@registry.detector("kw_coarse", vocab="kwvocab", overwrite=True)
def _kw_coarse(ctx):
    # Emits ACCOUNT (coarse) for the SSN — right region, less specific label.
    return KeywordDetector({"alice@example.com": "mail", "123-45-6789": "acct"})


def _fixtures(tmp_path: Path) -> Path:
    text = "Alice: alice@example.com, SSN 123-45-6789"
    rec = {
        "id": "r1", "text": text, "language": "en",
        "gold_spans": [
            {"label": "PERSON", "fine_label": "GIVEN_NAME", "raw_label": "x", "start": 0, "end": 5},
            {"label": "EMAIL", "fine_label": "EMAIL", "raw_label": "x", "start": 7, "end": 24},
            {"label": "ACCOUNT", "fine_label": "GOV_ID", "raw_label": "x", "start": 30, "end": 41},
        ],
    }
    fx = tmp_path / "fx.jsonl"
    fx.write_text(json.dumps(rec) + "\n")
    return fx


# ---------------------------------------------------------------- registry


def test_builtin_detectors_registered_without_heavy_imports():
    for name in ("opf", "gliner", "gliner_nvidia", "presidio", "skyflow", "openmed",
                 "ai4privacy_modernbert", "gliner_gretel_small", "opf_calibrated", "skyflow_full"):
        assert name in registry.names()


def test_importing_registry_loads_no_model_libraries():
    # Fresh interpreter: other tests may legitimately import these.
    code = (
        "import sys; from opf_eval.detectors import registry; registry.describe(); "
        "import opf_eval.runner, opf_eval.report, opf_eval.transforms; "
        "heavy = [m for m in ('torch', 'transformers', 'gliner', 'presidio_analyzer', 'opf') "
        "if m in sys.modules]; assert not heavy, heavy"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_unknown_detector_raises():
    with pytest.raises(ValueError, match="unknown detector"):
        registry.build("no_such_detector")


def test_missing_dependency_is_reported():
    registry.register_detector(
        registry.DetectorSpec(
            name="needs_missing", vocab="opf", factory=lambda ctx: None,  # type: ignore[arg-type]
            requires=("definitely_not_a_module_xyz",),
        ),
        overwrite=True,
    )
    assert not registry.get("needs_missing").available()
    with pytest.raises(ImportError, match="definitely_not_a_module_xyz"):
        registry.build("needs_missing")


def test_build_sets_registered_name():
    det = registry.build("kw_coarse")
    assert det.name == "kw_coarse"


# ------------------------------------------------------------------ runner


def test_runner_end_to_end_with_custom_detector(tmp_path):
    fx = _fixtures(tmp_path)
    out = tmp_path / "run"
    manifest = runner.run(fx, ["kw_fine", "kw_coarse"], out)
    assert manifest["labels"] == ["EMAIL", "GIVEN_NAME", "GOV_ID"]  # observed gold labels
    assert manifest["dataset"] is None
    row = json.loads((out / "raw_kw_fine.jsonl").read_text())
    assert {s["fine_label"] for s in row["spans"]} == {"EMAIL", "GOV_ID", "PERSON"}


def test_runner_refuses_mismatched_fixtures(tmp_path):
    fx = _fixtures(tmp_path)
    out = tmp_path / "run"
    runner.run(fx, ["kw_fine"], out)
    other = tmp_path / "other.jsonl"
    other.write_text(fx.read_text().replace("r1", "r2"))
    with pytest.raises(RuntimeError, match="different fixtures"):
        runner.run(other, ["kw_fine"], out)
    # fresh=True clears the stale results and proceeds.
    runner.run(other, ["kw_coarse"], out, fresh=True)
    assert not (out / "raw_kw_fine.jsonl").exists()
    assert json.loads((out / "manifest.json").read_text())["detectors"] == ["kw_coarse"]


def test_runner_adds_detectors_to_same_fixtures(tmp_path):
    fx = _fixtures(tmp_path)
    out = tmp_path / "run"
    runner.run(fx, ["kw_fine"], out)
    runner.run(fx, ["kw_coarse"], out)
    assert json.loads((out / "manifest.json").read_text())["detectors"] == ["kw_coarse", "kw_fine"]


def test_runner_passes_detector_options(tmp_path, monkeypatch):
    seen = {}

    @registry.detector("kw_opts", vocab="kwvocab", overwrite=True)
    def _f(ctx):
        seen.update(ctx.options)
        return KeywordDetector({})

    runner.run(_fixtures(tmp_path), ["kw_opts"], tmp_path / "run",
               detector_options={"kw_opts": {"threshold": 0.3}})
    assert seen == {"threshold": 0.3}


# ----------------------------------------------------------------- scoring


@pytest.fixture
def scored_run(tmp_path):
    fx = _fixtures(tmp_path)
    out = tmp_path / "run"
    runner.run(fx, ["kw_fine", "kw_coarse"], out)
    return scoring.Run.load(out)


def test_coarse_level_both_detectors_perfect(scored_run):
    res = scoring.score_run(scored_run, level="coarse", view="raw")
    # Coarse: GIVEN_NAME->PERSON, GOV_ID->ACCOUNT. kw_coarse misses PERSON.
    assert res["kw_fine"].by_schema["strict"]["f1"] == 1.0
    assert res["kw_coarse"].by_schema["strict"]["recall"] == pytest.approx(2 / 3)


def test_fine_level_distinguishes_granularity(scored_run):
    fine = scoring.score_run(scored_run, level="fine", view="fair")
    # kw_fine emits GOV_ID and matches the fine gold exactly. Its PERSON
    # category rolls up (detector has no fine PERSON labels) so the coarse
    # PERSON prediction still counts.
    assert fine["kw_fine"].by_schema["strict"]["f1"] == 1.0
    scope = scoring.scope_for(scored_run, "kw_fine", level="fine", view="fair")
    # ACCOUNT stays in scope as the parent of the fine-scored GOV_ID.
    assert {"GOV_ID", "ACCOUNT", "EMAIL", "PERSON"} == set(scope.labels)
    # kw_coarse said ACCOUNT where the gold says GOV_ID: right span, wrong
    # (less specific) label under the strict schema.
    m = fine["kw_coarse"].by_schema["strict"]
    assert m["incorrect"] == 1 and m["correct"] == 1


def test_coverage_and_holdout(scored_run):
    assert scoring.coverage(scored_run)[0]["scored"] == 1
    fit, score = scoring.holdout_split([f"id{i}" for i in range(10)], 0.5, seed=3)
    assert len(fit) == 5 and not fit & score
    assert scoring.holdout_split([f"id{i}" for i in range(10)], 0.5, seed=3) == (fit, score)


def test_report_fine_level_renders(scored_run):
    md = build_report(scored_run.dir, level="fine")
    assert "label level: `fine`" in md
    assert "GOV_ID" in md
    assert "## Coverage" in md


def test_report_limits_to_selected_detectors(scored_run):
    md = build_report(scored_run.dir, detectors=["kw_coarse"])
    assert "- detectors: kw_coarse\n" in md
    assert "| kw_coarse |" in md
    assert "kw_fine" not in md
    # Duplicates collapse and the run's detector order wins over the request's.
    both = build_report(scored_run.dir, detectors=("kw_coarse", "kw_fine", "kw_coarse"))
    assert both == build_report(scored_run.dir)


def test_report_rejects_unknown_or_empty_detector_selection(scored_run):
    with pytest.raises(ValueError, match="nope; the run has kw_coarse, kw_fine"):
        build_report(scored_run.dir, detectors=["kw_fine", "nope"])
    with pytest.raises(ValueError, match="empty"):
        build_report(scored_run.dir, detectors=[])


def test_report_cli_detectors_flag(scored_run, monkeypatch):
    from opf_eval import report

    out = scored_run.dir / "only_fine.md"
    monkeypatch.setattr(sys, "argv", [
        "report", "--run", str(scored_run.dir), "--out", str(out), "--detectors", " kw_fine ,",
    ])
    report.main()
    md = out.read_text()
    assert "- detectors: kw_fine\n" in md
    assert "kw_coarse" not in md
