"""Unit tests for `opf_eval.ensemble`.

Builds a tiny synthetic run directory + fixture file, exercises the
category-best recipe builder and the apply step, and verifies the output
JSONL matches the contract that `report.build_report` expects.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from opf_eval.ensemble import (
    _detectors_in,
    _register_in_manifest,
    apply_recipe,
    build_recipe_category_best,
    run_category_best,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def _span(label: str, start: int, end: int, text: str = "") -> dict:
    return {"label": label, "raw_label": label, "start": start, "end": end, "text": text}


@pytest.fixture
def run_dir(tmp_path: Path) -> tuple[Path, Path]:
    """A 3-record fixture set with two detectors that disagree by label.

    detA wins on EMAIL (recalls both gold emails, misses the phone).
    detB wins on PHONE (recalls the phone, hallucinates an extra EMAIL).
    """
    out_dir = tmp_path / "out"
    fx_path = tmp_path / "fx.jsonl"
    _write_jsonl(
        fx_path,
        [
            {"id": "r1", "text": "mail alice@example.com", "language": "en",
             "gold_spans": [_span("EMAIL", 5, 22, "alice@example.com")]},
            {"id": "r2", "text": "call +1-555-0101", "language": "en",
             "gold_spans": [_span("PHONE", 5, 16, "+1-555-0101")]},
            {"id": "r3", "text": "bob@example.com and +1-555-0202", "language": "en",
             "gold_spans": [
                 _span("EMAIL", 0, 15, "bob@example.com"),
                 _span("PHONE", 20, 31, "+1-555-0202"),
             ]},
        ],
    )

    # detA: nails emails, blind to phones.
    _write_jsonl(
        out_dir / "raw_detA.jsonl",
        [
            {"id": "r1", "detector": "detA",
             "spans": [_span("EMAIL", 5, 22)], "latency_ms": 1.0, "error": None},
            {"id": "r2", "detector": "detA",
             "spans": [], "latency_ms": 1.0, "error": None},
            {"id": "r3", "detector": "detA",
             "spans": [_span("EMAIL", 0, 15)], "latency_ms": 1.0, "error": None},
        ],
    )
    # detB: nails phones, fabricates an extra EMAIL on r2 (false positive).
    _write_jsonl(
        out_dir / "raw_detB.jsonl",
        [
            {"id": "r1", "detector": "detB",
             "spans": [], "latency_ms": 2.0, "error": None},
            {"id": "r2", "detector": "detB",
             "spans": [_span("PHONE", 5, 16), _span("EMAIL", 0, 4)],  # FP email
             "latency_ms": 2.0, "error": None},
            {"id": "r3", "detector": "detB",
             "spans": [_span("PHONE", 20, 31)], "latency_ms": 2.0, "error": None},
        ],
    )
    return out_dir, fx_path


def test_detectors_in_skips_ensemble_files(tmp_path: Path):
    """_detectors_in should ignore raw_ensemble_*.jsonl so re-runs don't
    treat a previous ensemble as just another detector."""
    out = tmp_path / "out"
    out.mkdir()
    (out / "raw_alpha.jsonl").touch()
    (out / "raw_beta.jsonl").touch()
    (out / "raw_ensemble_category_best.jsonl").touch()
    assert _detectors_in(out) == ["alpha", "beta"]


def test_build_recipe_category_best_picks_per_label_winner(run_dir):
    out_dir, fx_path = run_dir
    labels = ["EMAIL", "PHONE", "PERSON"]
    recipe, per_label_f1 = build_recipe_category_best(out_dir, fx_path, labels)
    assert recipe == {"EMAIL": "detA", "PHONE": "detB"}
    # PERSON has no signal anywhere — must be omitted, not silently mapped.
    assert "PERSON" not in recipe
    # per_label_f1 is exposed for inspection.
    assert per_label_f1["EMAIL"]["detA"] > per_label_f1["EMAIL"]["detB"]
    assert per_label_f1["PHONE"]["detB"] > per_label_f1["PHONE"]["detA"]


def test_apply_recipe_writes_union_of_winning_spans(run_dir):
    out_dir, fx_path = run_dir
    recipe = {"EMAIL": "detA", "PHONE": "detB"}
    out_path = apply_recipe(recipe, out_dir, fx_path, ensemble_name="ens_test")
    assert out_path.name == "raw_ens_test.jsonl"
    rows = [json.loads(l) for l in out_path.read_text().splitlines() if l.strip()]
    by_id = {r["id"]: r for r in rows}

    # r1: one EMAIL from detA.
    assert [(s["label"], s["start"], s["end"]) for s in by_id["r1"]["spans"]] == [
        ("EMAIL", 5, 22),
    ]
    # r2: one PHONE from detB. detB's stray EMAIL is dropped because the
    # recipe maps EMAIL to detA, not detB.
    assert [(s["label"], s["start"], s["end"]) for s in by_id["r2"]["spans"]] == [
        ("PHONE", 5, 16),
    ]
    # r3: union of detA's EMAIL and detB's PHONE, sorted by start.
    assert [(s["label"], s["start"], s["end"]) for s in by_id["r3"]["spans"]] == [
        ("EMAIL", 0, 15),
        ("PHONE", 20, 31),
    ]
    # Latency is the sum across constituent detectors used for that record.
    assert by_id["r1"]["latency_ms"] == pytest.approx(1.0 + 2.0)
    # detector field matches the requested ensemble name so the report
    # discovery code (which keys on file stem) doesn't see a mismatch.
    assert by_id["r1"]["detector"] == "ens_test"


def test_apply_recipe_dedupes_overlapping_spans(tmp_path: Path):
    """If two recipe-chosen detectors emit overlapping spans for different
    labels on the same record, the earlier-and-longer one wins; the
    overlap-loser is dropped so the output is non-overlapping."""
    out_dir = tmp_path / "out"
    fx_path = tmp_path / "fx.jsonl"
    _write_jsonl(
        fx_path,
        [{"id": "r1", "text": "alice@example.com", "language": "en",
          "gold_spans": [_span("EMAIL", 0, 17)]}],
    )
    _write_jsonl(
        out_dir / "raw_detA.jsonl",
        [{"id": "r1", "detector": "detA",
          "spans": [_span("EMAIL", 0, 17)], "latency_ms": 1.0, "error": None}],
    )
    _write_jsonl(
        out_dir / "raw_detB.jsonl",
        # detB's USERNAME starts mid-email — overlaps with detA's EMAIL span.
        [{"id": "r1", "detector": "detB",
          "spans": [_span("USERNAME", 5, 12)], "latency_ms": 1.0, "error": None}],
    )
    recipe = {"EMAIL": "detA", "USERNAME": "detB"}
    out_path = apply_recipe(recipe, out_dir, fx_path, ensemble_name="ens_overlap")
    rows = [json.loads(l) for l in out_path.read_text().splitlines() if l.strip()]
    spans = rows[0]["spans"]
    # The EMAIL (starts at 0, longer) wins; USERNAME (starts at 5) is dropped.
    assert [(s["label"], s["start"], s["end"]) for s in spans] == [("EMAIL", 0, 17)]


def test_apply_recipe_raises_if_recipe_references_missing_detector(tmp_path: Path):
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    fx_path = tmp_path / "fx.jsonl"
    fx_path.write_text(json.dumps({"id": "r1", "text": "x", "gold_spans": []}) + "\n")
    with pytest.raises(FileNotFoundError, match="raw_ghost.jsonl"):
        apply_recipe({"EMAIL": "ghost"}, out_dir, fx_path)


def test_run_category_best_end_to_end(run_dir):
    """The convenience wrapper builds the recipe + applies it in one call."""
    out_dir, fx_path = run_dir
    out_path, recipe, per_label_f1 = run_category_best(
        out_dir, fx_path, dataset="pii_masking_200k"
    )
    assert out_path == out_dir / "raw_ensemble_category_best.jsonl"
    assert out_path.exists()
    assert recipe.get("EMAIL") == "detA"
    assert recipe.get("PHONE") == "detB"
    # Per-label F1 must include every label in the dataset's canonical set,
    # even if no detector scored above zero on it (e.g. PERSON, which is in
    # the pii_masking_200k vocab but neither detA nor detB emit).
    assert "PERSON" in per_label_f1


def test_apply_recipe_registers_ensemble_in_manifest(run_dir):
    """`report.build_report` reads the detector list from manifest.json,
    not from a glob of raw_*.jsonl. The ensemble has to add itself to the
    manifest or its raw file is silently ignored at report time —
    visualisations (which glob) showed it but the report did not."""
    out_dir, fx_path = run_dir
    # Seed a runner-style manifest with just the original detectors.
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps({
        "started_at": "2026-05-15T00:00:00Z",
        "fixtures": str(fx_path),
        "dataset": "pii_masking_200k",
        "vocab_key": "pii200k",
        "n_examples": 3,
        "detectors": ["detA", "detB"],
    }))

    apply_recipe({"EMAIL": "detA", "PHONE": "detB"}, out_dir, fx_path,
                 ensemble_name="ensemble_category_best")

    manifest = json.loads(manifest_path.read_text())
    assert "ensemble_category_best" in manifest["detectors"]
    # Original detectors must still be present.
    assert "detA" in manifest["detectors"]
    assert "detB" in manifest["detectors"]


def test_apply_recipe_no_manifest_is_noop(tmp_path):
    """If the run dir has no manifest.json yet (e.g. running ensemble
    standalone via the CLI on a directory not produced by `runner.run`),
    the registration step must not crash."""
    out_dir = tmp_path / "out"
    fx_path = tmp_path / "fx.jsonl"
    fx_path.write_text(json.dumps({"id": "r1", "text": "x", "language": "en",
                                    "gold_spans": []}) + "\n")
    (out_dir).mkdir()
    (out_dir / "raw_detA.jsonl").write_text(json.dumps({
        "id": "r1", "detector": "detA",
        "spans": [{"label": "EMAIL", "raw_label": "EMAIL", "start": 0, "end": 1, "text": "x"}],
        "latency_ms": 1.0, "error": None,
    }) + "\n")
    # No manifest.json exists — apply_recipe must still succeed.
    apply_recipe({"EMAIL": "detA"}, out_dir, fx_path)
    assert not (out_dir / "manifest.json").exists()


def test_ensemble_reports_full_canonical_support_for_fair_view():
    """The fair-view headline reads each detector's `detector_supported_
    canonicals` and intersects with the dataset. Ensembles need to report
    the full canonical set so the fair view doesn't render an empty row
    for them (visible bug: section 7's headline showed `n labels=0` and
    no scores for the ensemble before this fix)."""
    from opf_eval.taxonomy import (
        CANONICAL_LABELS,
        detector_supported_canonicals,
        fair_labels,
    )

    full = set(CANONICAL_LABELS)
    assert detector_supported_canonicals("ensemble_category_best") == full
    # Any ensemble_* name follows the same rule, including custom names
    # passed via apply_recipe(ensemble_name=...).
    assert detector_supported_canonicals("ensemble_custom_strategy") == full
    # Fair-view intersection with a real dataset must be non-empty.
    assert fair_labels("ensemble_category_best", "pii200k")


def test_register_in_manifest_idempotent(tmp_path):
    """Re-running the ensemble must not duplicate the entry."""
    out_dir = tmp_path
    (out_dir / "manifest.json").write_text(json.dumps({"detectors": ["a", "b"]}))
    _register_in_manifest(out_dir, "ensemble_category_best")
    _register_in_manifest(out_dir, "ensemble_category_best")
    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert manifest["detectors"].count("ensemble_category_best") == 1


def test_build_recipe_fit_ids_filters_scoring_pool(run_dir):
    """`fit_ids` restricts the F1 calculation to the given fixture ids.
    Verifies the holdout split path works — recipe should change when the
    fit pool changes."""
    out_dir, fx_path = run_dir
    labels = ["EMAIL", "PHONE"]

    # On the full set, detA wins EMAIL (recalls both gold emails on r1 + r3).
    recipe_full, _ = build_recipe_category_best(out_dir, fx_path, labels)
    assert recipe_full["EMAIL"] == "detA"

    # Restrict the fit pool to only r2 (which has no EMAIL gold at all,
    # just a PHONE). EMAIL F1 collapses to 0 for everyone on this subset,
    # so EMAIL drops out of the recipe entirely.
    recipe_r2_only, per_label_f1 = build_recipe_category_best(
        out_dir, fx_path, labels, fit_ids={"r2"}
    )
    assert "EMAIL" not in recipe_r2_only
    assert recipe_r2_only.get("PHONE") == "detB"
    # The F1 dict still exposes the (zero) scores so callers can audit.
    assert per_label_f1["EMAIL"]["detA"] == 0.0


def test_excluded_detectors_drops_from_recipe(run_dir):
    out_dir, fx_path = run_dir
    labels = ["EMAIL", "PHONE"]
    recipe, _ = build_recipe_category_best(
        out_dir, fx_path, labels, excluded_detectors={"detB"}
    )
    # detB is excluded, so PHONE has no positive candidate and is dropped.
    assert recipe == {"EMAIL": "detA"}
