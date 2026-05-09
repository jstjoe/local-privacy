"""Iterate fixtures, call each detector, stream JSONL results.

Output:
    <out_dir>/raw_<detector>.jsonl   one line per example: {id, detector, spans, latency_ms, error}
    <out_dir>/manifest.json          run config snapshot
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from .datasets import DEFAULT_DATASET, get as get_dataset_config, names as dataset_names
from .detectors import (
    Ai4PrivacyDetector,
    GLiNERDetector,
    OpenMedDetector,
    OPFDetector,
    PresidioDetector,
    SkyflowDetector,
)
from .detectors.base import Detector
from .taxonomy import (
    canonical_to_skyflow_request_types,
    dataset_canonicals,
    gliner_prompts,
    gretel_prompts,
    gretel_to_canonical,
)


def _build_detector(
    name: str,
    *,
    dataset_canonicals_set: set[str],
    skyflow_entity_types: list[str] | None = None,
    opf_calibration_path: str | None = None,
    device: str = "cpu",
) -> Detector:
    """Build a detector. `dataset_canonicals_set` is what the chosen dataset
    annotates — used to auto-configure detectors that take per-call label
    sets (skyflow, gliner). Detectors with fixed vocabularies (opf, presidio)
    ignore it. `device` is propagated to local PyTorch detectors (`opf`,
    `gliner*`, `ai4privacy_modernbert`, `openmed`); ignored by Skyflow
    (HTTP) and Presidio (CPU spaCy).
    """
    if name == "opf":
        return OPFDetector(viterbi_calibration_path=opf_calibration_path, device=device)  # type: ignore[arg-type]
    if name == "presidio":
        return PresidioDetector()
    if name == "presidio_multilang":
        # All 6 spaCy models — needs each `<lang>_core_news_lg` installed.
        from .detectors.presidio import LANGUAGE_MODELS
        return PresidioDetector(languages=list(LANGUAGE_MODELS.keys()))
    if name == "gliner":
        # Restrict prompts to what this dataset annotates so GLiNER stops
        # over-detecting labels the gold doesn't cover.
        return GLiNERDetector(prompts=gliner_prompts(dataset_canonicals_set), device=device)
    if name == "ai4privacy_modernbert":
        return Ai4PrivacyDetector(device=device)
    if name == "gliner_gretel_small":
        return GLiNERDetector(
            model_name="gretelai/gretel-gliner-bi-small-v1.0",
            threshold=0.7,
            prompts=gretel_prompts(),
            label_to_canonical=gretel_to_canonical,
            name="gliner_gretel_small",
            device=device,
        )
    if name == "gliner_gretel_large":
        return GLiNERDetector(
            model_name="gretelai/gretel-gliner-bi-large-v1.0",
            threshold=0.7,
            prompts=gretel_prompts(),
            label_to_canonical=gretel_to_canonical,
            name="gliner_gretel_large",
            device=device,
        )
    if name == "gliner_nvidia":
        # Same vocabulary as default GLiNER, larger 570M-param base, lower
        # recommended threshold per model card.
        return GLiNERDetector(
            model_name="nvidia/gliner-PII",
            threshold=0.3,
            prompts=gliner_prompts(dataset_canonicals_set),
            name="gliner_nvidia",
            device=device,
        )
    if name == "openmed":
        return OpenMedDetector(device=device)
    if name == "opf_calibrated":
        if not opf_calibration_path:
            raise ValueError("opf_calibrated requires --opf-calibration-path")
        return OPFDetector(viterbi_calibration_path=opf_calibration_path, device=device)  # type: ignore[arg-type]
    if name == "skyflow":
        # Auto-derive entity_types from the dataset's canonical labels mapped
        # to Skyflow request types. Override with --skyflow-entities or use
        # 'skyflow_full' for the unconstrained call.
        types = skyflow_entity_types or canonical_to_skyflow_request_types(
            sorted(dataset_canonicals_set)
        )
        return SkyflowDetector(entity_types=types)
    if name == "skyflow_full":
        return SkyflowDetector(entity_types=skyflow_entity_types)
    raise ValueError(f"unknown detector: {name}")


def _read_fixtures(path: Path):
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


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


def run(
    fixtures: Path,
    detector_names: list[str],
    out_dir: Path,
    *,
    dataset: str = DEFAULT_DATASET,
    skyflow_workers: int = 1,
    skyflow_min_interval_ms: float = 0.0,
    skyflow_entity_types: list[str] | None = None,
    opf_calibration_path: str | None = None,
    reuse_from: Path | None = None,
    device: str = "cpu",
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    examples = list(_read_fixtures(fixtures))

    # Pull in raw files from a previous run before building the manifest.
    present_detectors, copied = _copy_reused_raw_files(reuse_from, out_dir)
    if copied:
        print(f"[reuse] copied {copied} from {reuse_from}")

    # Merge: existing raw files + previously-recorded manifest + this run.
    manifest_path = out_dir / "manifest.json"
    existing: dict = {}
    if manifest_path.exists():
        try:
            existing = json.loads(manifest_path.read_text())
        except json.JSONDecodeError:
            existing = {}
    merged_detectors = sorted(
        set(existing.get("detectors") or []) | set(present_detectors) | set(detector_names)
    )

    cfg = get_dataset_config(dataset)
    canonicals = dataset_canonicals(cfg.vocab_key)

    manifest = {
        "started_at": existing.get("started_at") or datetime.now(timezone.utc).isoformat(),
        "fixtures": str(fixtures),
        "dataset": dataset,
        "vocab_key": cfg.vocab_key,
        "n_examples": len(examples),
        "detectors": merged_detectors,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))

    for name in detector_names:
        det = _build_detector(
            name,
            dataset_canonicals_set=canonicals,
            skyflow_entity_types=skyflow_entity_types,
            opf_calibration_path=opf_calibration_path,
            device=device,
        )
        out_path = out_dir / f"raw_{name}.jsonl"
        t0 = time.perf_counter()
        with out_path.open("w") as f:
            if name.startswith("skyflow") and skyflow_workers > 1:
                with ThreadPoolExecutor(max_workers=skyflow_workers) as ex:
                    futures = {
                        ex.submit(det.detect, ex_["text"], language=ex_.get("language")): ex_
                        for ex_ in examples
                    }
                    for fut in futures:
                        ex_ = futures[fut]
                        result = fut.result()
                        f.write(
                            json.dumps(
                                {
                                    "id": ex_["id"],
                                    "detector": name,
                                    **result,
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
            else:
                throttle_s = (
                    skyflow_min_interval_ms / 1000.0
                    if name.startswith("skyflow")
                    else 0.0
                )
                last_call = 0.0
                for ex_ in examples:
                    if throttle_s:
                        wait = throttle_s - (time.perf_counter() - last_call)
                        if wait > 0:
                            time.sleep(wait)
                    last_call = time.perf_counter()
                    result = det.detect(ex_["text"], language=ex_.get("language"))
                    f.write(
                        json.dumps(
                            {"id": ex_["id"], "detector": name, **result},
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
        elapsed = time.perf_counter() - t0
        print(f"[{name}] {len(examples)} examples in {elapsed:.1f}s -> {out_path}")
        if hasattr(det, "close"):
            det.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixtures", required=True, type=Path)
    ap.add_argument("--detectors", default="opf,skyflow")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        choices=dataset_names(),
        help=(
            f"Source dataset for the fixtures (default: {DEFAULT_DATASET}). "
            "Used to derive each detector's per-call label set and to pick the "
            "right vocabulary for canonical-label restricted scoring."
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
    args = ap.parse_args()
    device = args.device
    if device == "auto":
        device = _autodetect_device()
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
    )


def _autodetect_device() -> str:
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


if __name__ == "__main__":
    main()
