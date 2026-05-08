"""Throughput, cold-start, and memory measurements for OPF and Skyflow.

Distinct from the runner's per-call latency:
- cold_start: time to first detect() including model load
- throughput: requests/sec at varying concurrency
- memory: peak RSS during inference

Usage:
    python -m opf_eval.perf --fixtures eval/data/dev.jsonl --detector opf --concurrencies 1,2,4,8
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import psutil

from .detectors import OPFDetector, SkyflowDetector


def _build(name: str):
    if name == "opf":
        return OPFDetector
    if name == "skyflow":
        return SkyflowDetector
    raise ValueError(f"unknown detector: {name}")


def measure_cold_start(detector_cls) -> dict:
    """Time from constructor to first successful detect()."""
    proc = psutil.Process(os.getpid())
    rss_before = proc.memory_info().rss
    t0 = time.perf_counter()
    det = detector_cls()
    t_load = time.perf_counter() - t0
    t1 = time.perf_counter()
    det.detect("Joe at joe@example.com lives in Elgin, TX. Phone: 555-1234.")
    t_first = time.perf_counter() - t1
    rss_after = proc.memory_info().rss
    return {
        "load_ms": t_load * 1000,
        "first_detect_ms": t_first * 1000,
        "rss_growth_mb": (rss_after - rss_before) / 1024 / 1024,
    }


def measure_throughput(
    detector_cls,
    texts: list[str],
    concurrency: int,
    n_requests: int,
) -> dict:
    """Issue n_requests in parallel pools of `concurrency`. Returns rps + latency stats."""
    det = detector_cls()
    # warm
    det.detect(texts[0])

    samples = (texts * (n_requests // len(texts) + 1))[:n_requests]
    latencies: list[float] = []
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        for result in ex.map(det.detect, samples):
            latencies.append(result["latency_ms"])
    elapsed = time.perf_counter() - t0
    rps = n_requests / elapsed if elapsed > 0 else 0.0
    return {
        "concurrency": concurrency,
        "n_requests": n_requests,
        "elapsed_s": elapsed,
        "rps": rps,
        "latency_p50_ms": statistics.median(latencies),
        "latency_p95_ms": sorted(latencies)[max(0, int(0.95 * (len(latencies) - 1)))],
        "latency_max_ms": max(latencies),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixtures", required=True, type=Path)
    ap.add_argument("--detector", required=True, choices=["opf", "skyflow"])
    ap.add_argument(
        "--allow-skyflow",
        action="store_true",
        help="Required to run perf against Skyflow. Defaults off so trial-account "
        "users don't accidentally hammer the API. Set only when authorized.",
    )
    ap.add_argument("--concurrencies", default="1,2,4,8", help="comma-separated")
    ap.add_argument("--n-requests", type=int, default=50)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    if args.detector == "skyflow" and not args.allow_skyflow:
        raise SystemExit(
            "perf against Skyflow is gated. Pass --allow-skyflow only when you have "
            "headroom in your account; throughput tests can issue many requests/sec."
        )
    detector_cls = _build(args.detector)
    texts = [json.loads(l)["text"] for l in args.fixtures.read_text().splitlines() if l.strip()][:50]
    print(f"loaded {len(texts)} sample texts")

    print(f"\n=== cold start: {args.detector} ===")
    cs = measure_cold_start(detector_cls)
    print(json.dumps(cs, indent=2))

    print(f"\n=== throughput: {args.detector} ===")
    rows = []
    for c in [int(x) for x in args.concurrencies.split(",")]:
        # OPF is single-instance so concurrency >1 will queue on the model;
        # report what actually happens rather than asserting against it.
        r = measure_throughput(detector_cls, texts, c, args.n_requests)
        print(json.dumps(r, indent=2))
        rows.append(r)

    if args.out:
        args.out.write_text(json.dumps({"cold_start": cs, "throughput": rows}, indent=2))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
