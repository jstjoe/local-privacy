"""Iterate fixtures, call each detector, stream JSONL results.

Output:
    <out_dir>/raw_<detector>.jsonl   one line per example: {id, detector, spans, latency_ms, error}
    <out_dir>/manifest.json          run config snapshot (dataset, labels, fixtures hash, detectors)

A run directory belongs to one fixtures file. Adding detectors to an existing
run dir (same fixtures) is supported and merges into its manifest; pointing a
run dir at *different* fixtures is refused unless `fresh=True`, which clears
the old results first — otherwise stale raw files would leak into reports.
"""

from __future__ import annotations

import argparse
import gc
import json
import shutil
import time
import warnings
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .datasets import (
    annotated_labels,
    get as get_dataset_config,
    names as dataset_names,
    observed_labels,
)
from .detectors import registry
from .detectors.base import Detector
from .detectors.registry import GLINER_THRESHOLD  # noqa: F401  — re-exported for back-compat
from .io import file_sha256, iter_jsonl, read_meta


def _build_detector(
    name: str,
    *,
    dataset_canonicals_set: set[str] | None,
    skyflow_entity_types: list[str] | None = None,
    opf_calibration_path: str | None = None,
    device: str = "cpu",
    options: Mapping[str, Any] | None = None,
) -> Detector:
    """Build a detector through the registry.

    `dataset_canonicals_set` is what the chosen dataset annotates — detectors
    that take per-call label sets (Skyflow entity types, GLiNER prompts)
    restrict themselves to it. `None` builds with the full vocabulary.
    """
    return registry.build(
        name,
        dataset_labels=dataset_canonicals_set,
        device=device,
        skyflow_entity_types=skyflow_entity_types,
        opf_calibration_path=opf_calibration_path,
        options=options,
    )


def _read_fixtures(path: Path):
    return iter_jsonl(path)


def resolve_labels(
    fixtures: Path, examples: list[dict], dataset: str | None
) -> tuple[str | None, str | None, set[str]]:
    """Work out (dataset name, vocab key, annotated fine labels) for a run.

    Order: an explicitly named registered dataset; else the fixtures' meta
    sidecar (written by `fixtures.materialize`); else the labels observed in
    the fixtures' gold spans.
    """
    if dataset:
        cfg = get_dataset_config(dataset)
        return cfg.name, cfg.vocab_key, annotated_labels(cfg, examples, level="fine")
    meta = read_meta(fixtures)
    if meta and meta.get("labels") is not None:
        return meta.get("dataset"), meta.get("vocab_key"), set(meta["labels"])
    return None, None, observed_labels(examples)


def _is_remote(name: str) -> bool:
    try:
        return registry.get(name).remote
    except ValueError:
        return False


def _clear_run_dir(out_dir: Path) -> None:
    for p in list(out_dir.glob("raw_*.jsonl")) + [out_dir / "manifest.json", out_dir / "report.md"]:
        if p.exists():
            p.unlink()


def _check_reused_coverage(out_dir: Path, copied: list[str], ids: set[str]) -> None:
    for det in copied:
        have = {r["id"] for r in iter_jsonl(out_dir / f"raw_{det}.jsonl")}
        missing = len(ids - have)
        if missing:
            warnings.warn(
                f"[reuse] raw_{det}.jsonl covers {len(ids) - missing}/{len(ids)} fixture ids — "
                f"it was produced on different fixtures; the report will score it on the overlap only",
                stacklevel=3,
            )


def _copy_reused_raw_files(
    reuse_from: Path | None, out_dir: Path
) -> tuple[list[str], list[str]]:
    """Copy raw_<detector>.jsonl files from `reuse_from` into out_dir.

    Files already present in out_dir are not overwritten.

    Returns (present, copied):
      present — every detector that has a raw file in out_dir after copy
      copied — detectors whose raw file was copied this call (for logging)
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    pre_existing = {p.stem.removeprefix("raw_") for p in out_dir.glob("raw_*.jsonl")}
    copied: list[str] = []
    if reuse_from and reuse_from.exists():
        for src in sorted(reuse_from.glob("raw_*.jsonl")):
            det = src.stem.removeprefix("raw_")
            if det in pre_existing:
                continue
            shutil.copy2(src, out_dir / src.name)
            copied.append(det)
    present = sorted({p.stem.removeprefix("raw_") for p in out_dir.glob("raw_*.jsonl")})
    return present, sorted(copied)


def _free_detector(det: object) -> None:
    """Best-effort release of GPU memory held by a detector instance.

    Called between iterations of the per-detector loop so the next model
    isn't allocating against VRAM already pinned by the previous one. The
    CUDA caching allocator does not return memory to the driver until
    `empty_cache()` is called, even after the Python object is gone.
    """
    close = getattr(det, "close", None)
    if callable(close):
        try:
            close()
        except Exception:  # noqa: BLE001 — cleanup must not raise
            pass
    # Drop common heavy attributes if the detector stashes them. close()
    # above is the primary path; this is a fallback for detectors that
    # don't define one.
    for attr in ("_opf", "_model", "model", "_pipe", "_pipeline", "pipeline", "_engine", "engine", "_loaders"):
        if hasattr(det, attr):
            try:
                setattr(det, attr, None)
            except Exception:  # noqa: BLE001
                pass
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and getattr(mps, "is_available", lambda: False)():
            empty = getattr(getattr(torch, "mps", None), "empty_cache", None)
            if callable(empty):
                empty()
    except ImportError:
        pass


def run(
    fixtures: Path,
    detector_names: list[str],
    out_dir: Path,
    *,
    dataset: str | None = None,
    skyflow_workers: int = 1,
    skyflow_min_interval_ms: float = 0.0,
    skyflow_entity_types: list[str] | None = None,
    opf_calibration_path: str | None = None,
    reuse_from: Path | None = None,
    device: str = "cpu",
    detector_options: Mapping[str, Mapping[str, Any]] | None = None,
    fresh: bool = False,
) -> dict:
    """Run each detector over every fixture and write raw_<name>.jsonl.

    dataset: registered dataset the fixtures came from. Optional when the
        fixtures were written by `fixtures.materialize` (read from the meta
        sidecar); otherwise labels are inferred from the gold spans.
    detector_options: per-detector overrides passed to the factory, e.g.
        `{"gliner": {"threshold": 0.5}}`.
    fresh: clear existing results in `out_dir` before running. Required when
        `out_dir` holds results for a different fixtures file.

    Returns the manifest written to `out_dir/manifest.json`.
    """
    fixtures = Path(fixtures)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    examples = list(_read_fixtures(fixtures))
    fixtures_sha = file_sha256(fixtures)

    manifest_path = out_dir / "manifest.json"
    existing: dict = {}
    if manifest_path.exists():
        try:
            existing = json.loads(manifest_path.read_text())
        except json.JSONDecodeError:
            existing = {}
    if fresh:
        _clear_run_dir(out_dir)
        existing = {}
    elif existing.get("fixtures_sha256") and existing["fixtures_sha256"] != fixtures_sha:
        raise RuntimeError(
            f"{out_dir} holds results for different fixtures ({existing.get('fixtures')}). "
            f"Use a new out_dir, or pass fresh=True (--fresh) to clear it."
        )

    # Pull in raw files from a previous run before building the manifest.
    present_detectors, copied = _copy_reused_raw_files(reuse_from, out_dir)
    if copied:
        print(f"[reuse] copied {copied} from {reuse_from}")
        _check_reused_coverage(out_dir, copied, {ex["id"] for ex in examples})

    dataset_name, vocab_key, labels = resolve_labels(fixtures, examples, dataset)

    # Merge: existing raw files + previously-recorded manifest + this run.
    merged_detectors = sorted(
        set(existing.get("detectors") or []) | set(present_detectors) | set(detector_names)
    )
    options = {k: dict(v) for k, v in (detector_options or {}).items()}
    manifest = {
        "started_at": existing.get("started_at") or datetime.now(timezone.utc).isoformat(),
        "fixtures": str(fixtures),
        "fixtures_sha256": fixtures_sha,
        "dataset": dataset_name,
        "vocab_key": vocab_key,
        "labels": sorted(labels),
        "n_examples": len(examples),
        "detectors": merged_detectors,
        "detector_options": {**(existing.get("detector_options") or {}), **options},
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))

    for name in detector_names:
        det = _build_detector(
            name,
            dataset_canonicals_set=labels,
            skyflow_entity_types=skyflow_entity_types,
            opf_calibration_path=opf_calibration_path,
            device=device,
            options=options.get(name),
        )
        try:
            _run_one(
                name, det, examples, out_dir / f"raw_{name}.jsonl",
                remote=_is_remote(name),
                workers=skyflow_workers,
                min_interval_ms=skyflow_min_interval_ms,
            )
        finally:
            # Release VRAM before building the next detector — without this
            # the next iteration OOMs on shared-GPU runtimes (see _free_detector).
            _free_detector(det)
            del det
    return manifest


def _run_one(
    name: str,
    det: Detector,
    examples: list[dict],
    out_path: Path,
    *,
    remote: bool,
    workers: int,
    min_interval_ms: float,
) -> None:
    def row(ex_: dict, result: dict) -> str:
        return json.dumps({"id": ex_["id"], "detector": name, **result}, ensure_ascii=False) + "\n"

    t0 = time.perf_counter()
    with out_path.open("w") as f:
        if remote and workers > 1:
            with ThreadPoolExecutor(max_workers=workers) as ex:
                futures = {
                    ex.submit(det.detect, ex_["text"], language=ex_.get("language")): ex_
                    for ex_ in examples
                }
                for fut in futures:  # submission order
                    f.write(row(futures[fut], fut.result()))
        else:
            throttle_s = min_interval_ms / 1000.0 if remote else 0.0
            last_call = 0.0
            for ex_ in examples:
                if throttle_s:
                    wait = throttle_s - (time.perf_counter() - last_call)
                    if wait > 0:
                        time.sleep(wait)
                last_call = time.perf_counter()
                f.write(row(ex_, det.detect(ex_["text"], language=ex_.get("language"))))
    elapsed = time.perf_counter() - t0
    print(f"[{name}] {len(examples)} examples in {elapsed:.1f}s -> {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixtures", type=Path)
    ap.add_argument("--detectors", default="opf,skyflow")
    ap.add_argument("--out", type=Path)
    ap.add_argument(
        "--dataset",
        default=None,
        choices=dataset_names(),
        help=(
            "Registered dataset the fixtures came from. Defaults to the one "
            "recorded in the fixtures' .meta.json (written by opf_eval.fixtures); "
            "without either, the labels are inferred from the gold spans. Used "
            "to derive each detector's per-call label set and the scored labels."
        ),
    )
    ap.add_argument(
        "--skyflow-workers",
        type=int,
        default=1,
        help="Concurrent Skyflow requests. Default 1 to be friendly to trial accounts.",
    )
    ap.add_argument(
        "--skyflow-min-interval-ms",
        type=float,
        default=0.0,
        help="Minimum ms between Skyflow requests (rate-limit friendly). Only applies "
        "when --skyflow-workers=1.",
    )
    ap.add_argument(
        "--skyflow-entities",
        default=None,
        help="Comma-separated Skyflow request enum values (lowercase) to constrain detection to. "
        "Applies to the 'skyflow' detector only.",
    )
    ap.add_argument(
        "--opf-calibration-path",
        default=None,
        help="Path to a Viterbi calibration JSON. Required when running the "
        "'opf_calibrated' detector; ignored otherwise.",
    )
    ap.add_argument(
        "--reuse-from",
        type=Path,
        default=None,
        help="Copy raw_<detector>.jsonl files from this prior run dir into "
        "--out before processing. Only detectors not already present are "
        "copied; only detectors named in --detectors are run. Lets you add "
        "a new detector without re-running existing ones on the same "
        "fixtures.",
    )
    ap.add_argument(
        "--device",
        choices=["cpu", "cuda", "mps", "auto"],
        default="cpu",
        help=(
            "Torch device for local PyTorch detectors (opf, gliner*, "
            "ai4privacy_modernbert, openmed). `auto` picks cuda > mps > cpu. "
            "Skyflow (HTTP) and Presidio (CPU spaCy) ignore this."
        ),
    )
    ap.add_argument(
        "--fresh",
        action="store_true",
        help="Clear existing results in --out first (required when --out holds "
        "results for different fixtures).",
    )
    ap.add_argument("--list", action="store_true", help="List registered detectors and exit.")
    args = ap.parse_args()
    if not args.list and (args.fixtures is None or args.out is None):
        ap.error("--fixtures and --out are required")
    if args.list:
        for d in registry.describe():
            flag = "" if d["available"] else "  (missing deps)"
            print(f"{d['name']:<24} vocab={d['vocab']:<9} {d['description']}{flag}")
        return
    device = args.device
    if device == "auto":
        device = autodetect_device()
        print(f"[device] auto -> {device}")
    run(
        args.fixtures,
        [d.strip() for d in args.detectors.split(",") if d.strip()],
        args.out,
        dataset=args.dataset,
        skyflow_workers=args.skyflow_workers,
        skyflow_min_interval_ms=args.skyflow_min_interval_ms,
        skyflow_entity_types=(
            [e.strip() for e in args.skyflow_entities.split(",") if e.strip()]
            if args.skyflow_entities
            else None
        ),
        opf_calibration_path=args.opf_calibration_path,
        reuse_from=args.reuse_from,
        device=device,
        fresh=args.fresh,
    )


def autodetect_device() -> str:
    """cuda > mps > cpu."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:  # noqa: BLE001
        pass
    return "cpu"


_autodetect_device = autodetect_device  # back-compat alias


if __name__ == "__main__":
    main()
