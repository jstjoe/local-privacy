"""Saved detector results shipped with the package, so notebooks can score a
sample without a GPU.

A baseline is a directory under `opf_eval/baselines/<name>/` holding:

    baseline.json        dataset, seed, detectors, options, and a sha1 of each
                         record's text keyed by fixture id
    raw_<detector>.jsonl the runner's raw output for each detector

It stores no fixture text. The notebook still samples the dataset itself (CPU
only) and a baseline is used only for records whose id *and* text hash match,
so a changed dataset revision falls back to running the detector instead of
scoring stale predictions. Samples drawn with the same seed are prefixes of
each other, so one 200-record baseline also covers the 100-record default.

Build one (needs the models, so a GPU is best):

    python -m opf_eval.baseline build --n 200

or from a notebook after a run: `nb.export_baseline(S)`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Iterable, Sequence
from datetime import datetime, timezone
from pathlib import Path

from .io import iter_jsonl, read_meta, write_jsonl

BASELINES_DIR = Path(__file__).resolve().parent / "baselines"


def text_sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _name(dataset: str, seed: int) -> str:
    return f"{dataset}_s{seed}"


def find(dataset: str, seed: int, *, root: Path | None = None) -> Path | None:
    """The baseline directory for (dataset, seed), if one is shipped."""
    path = (root or BASELINES_DIR) / _name(dataset, seed)
    return path if (path / "baseline.json").exists() else None


def seed_run_dir(
    fixtures: Path,
    run_dir: Path,
    detectors: Iterable[str],
    *,
    detector_options: dict | None = None,
    root: Path | None = None,
) -> list[str]:
    """Write baseline results into `run_dir` for each of `detectors` that
    has none yet and is covered by a matching baseline. Returns the detectors
    seeded.

    A detector is seeded only when the baseline was produced with the same
    options and covers every fixture id with an identical text hash.
    """
    meta = read_meta(fixtures) or {}
    dataset, seed = meta.get("dataset"), meta.get("seed")
    if dataset is None or seed is None:
        return []
    base = find(dataset, seed, root=root)
    if base is None:
        return []
    info = json.loads((base / "baseline.json").read_text())
    hashes: dict[str, str] = info.get("text_sha1", {})
    records = list(iter_jsonl(fixtures))
    if any(hashes.get(r["id"]) != text_sha1(r["text"]) for r in records):
        return []
    ids = {r["id"] for r in records}
    options = detector_options or {}
    run_dir.mkdir(parents=True, exist_ok=True)
    seeded = []
    for det in detectors:
        src = base / f"raw_{det}.jsonl"
        dest = run_dir / f"raw_{det}.jsonl"
        if dest.exists() or not src.exists():
            continue
        if (info.get("detector_options") or {}).get(det, {}) != options.get(det, {}):
            continue
        rows = [r for r in iter_jsonl(src) if r["id"] in ids]
        if {r["id"] for r in rows} != ids:
            continue
        write_jsonl(dest, rows)
        seeded.append(det)
    return seeded


def export(
    fixtures: Path,
    run_dir: Path,
    *,
    detectors: Sequence[str] | None = None,
    root: Path | None = None,
) -> Path:
    """Save a run's raw results as the baseline for its (dataset, seed).

    Replaces the detectors it writes and keeps any others already there, as
    long as they cover the same records.
    """
    meta = read_meta(fixtures) or {}
    dataset, seed = meta.get("dataset"), meta.get("seed")
    if dataset is None or seed is None:
        raise ValueError(f"{fixtures} has no dataset/seed in its meta sidecar; can't key a baseline")
    manifest = json.loads((run_dir / "manifest.json").read_text())
    detectors = list(detectors or manifest["detectors"])
    records = list(iter_jsonl(fixtures))
    hashes = {r["id"]: text_sha1(r["text"]) for r in records}

    out = (root or BASELINES_DIR) / _name(dataset, seed)
    out.mkdir(parents=True, exist_ok=True)
    info_path = out / "baseline.json"
    info = json.loads(info_path.read_text()) if info_path.exists() else {}
    if info.get("text_sha1") and info["text_sha1"] != hashes:
        # Different records: start over rather than mix samples.
        for p in out.glob("raw_*.jsonl"):
            p.unlink()
        info = {}

    for det in detectors:
        src = run_dir / f"raw_{det}.jsonl"
        if not src.exists():
            raise FileNotFoundError(f"no results for {det} in {run_dir}")
        write_jsonl(out / src.name, list(iter_jsonl(src)))

    kept = sorted({p.stem.removeprefix("raw_") for p in out.glob("raw_*.jsonl")})
    options = {**(info.get("detector_options") or {}), **{
        d: (manifest.get("detector_options") or {}).get(d, {}) for d in detectors
    }}
    info = {
        "dataset": dataset,
        "seed": seed,
        "n": len(records),
        "detectors": kept,
        "detector_options": {d: options.get(d, {}) for d in kept},
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "text_sha1": hashes,
    }
    info_path.write_text(json.dumps(info, indent=1) + "\n")
    return out


def build(
    *,
    dataset: str,
    n: int,
    seed: int,
    detectors: Sequence[str],
    device: str | None = None,
    work_dir: Path | None = None,
    root: Path | None = None,
) -> Path:
    """Sample, run every detector, and export the result as the baseline."""
    from .fixtures import ensure_fixtures
    from .runner import autodetect_device, run

    work = Path(work_dir or Path.cwd() / "baseline-build")
    fixtures, _ = ensure_fixtures(work / f"{dataset}_{n}_s{seed}.jsonl", n, dataset=dataset, seed=seed)
    run_dir = work / "run"
    run(fixtures, list(detectors), run_dir, device=device or autodetect_device(), fresh=True)
    return export(fixtures, run_dir, detectors=detectors, root=root)


def main(argv: Sequence[str] | None = None) -> None:
    from .nb import DEFAULT_DATASET, DEFAULT_DETECTORS

    p = argparse.ArgumentParser(prog="python -m opf_eval.baseline", description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="sample, run detectors, save as the shipped baseline")
    b.add_argument("--dataset", default=DEFAULT_DATASET)
    b.add_argument("--n", type=int, default=200)
    b.add_argument("--seed", type=int, default=42)
    b.add_argument("--detectors", default=",".join(DEFAULT_DETECTORS))
    b.add_argument("--device", default=None)
    b.add_argument("--work-dir", type=Path, default=None)
    e = sub.add_parser("export", help="save an existing run as the shipped baseline")
    e.add_argument("--fixtures", type=Path, required=True)
    e.add_argument("--run", type=Path, required=True)
    e.add_argument("--detectors", default=None)
    args = p.parse_args(argv)

    if args.cmd == "build":
        out = build(
            dataset=args.dataset, n=args.n, seed=args.seed,
            detectors=[d for d in args.detectors.split(",") if d],
            device=args.device, work_dir=args.work_dir,
        )
    else:
        dets = [d for d in args.detectors.split(",") if d] if args.detectors else None
        out = export(args.fixtures, args.run, detectors=dets)
    print(f"baseline written: {out}")


if __name__ == "__main__":
    main()
