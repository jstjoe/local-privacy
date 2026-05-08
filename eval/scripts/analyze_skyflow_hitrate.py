"""Per-raw-label hit rate analysis: what % of each Skyflow raw_label hits gold?

Low hit rate = mostly false positives at this raw_label = drop candidate.
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


def _iou(a, b):
    lo = max(a["start"], b["start"])
    hi = min(a["end"], b["end"])
    inter = max(0, hi - lo)
    union = (a["end"] - a["start"]) + (b["end"] - b["start"]) - inter
    return inter / union if union > 0 else 0.0


def main(skyflow_jsonl: Path, fixtures_jsonl: Path, min_count: int = 5):
    fixtures = {json.loads(l)["id"]: json.loads(l) for l in fixtures_jsonl.open() if l.strip()}
    sky_records = [json.loads(l) for l in skyflow_jsonl.open() if l.strip()]

    total: Counter = Counter()
    matched: Counter = Counter()  # canonical-label match against gold

    for rec in sky_records:
        gold = fixtures.get(rec["id"], {}).get("gold_spans") or []
        for s in rec.get("spans") or []:
            raw = s.get("raw_label") or ""
            total[raw] += 1
            for g in gold:
                if s["label"] == g["label"] and _iou(s, g) >= 0.5:
                    matched[raw] += 1
                    break

    print(f"{'raw_label':<35} {'count':>6} {'matched':>8} {'hit%':>6}  verdict")
    print("-" * 70)
    rows = sorted(total.items(), key=lambda x: -x[1])
    for raw, n in rows:
        if n < min_count:
            continue
        m = matched[raw]
        hit = 100 * m / n
        verdict = "DROP" if hit < 50 else ("WEAK" if hit < 70 else "KEEP")
        print(f"{raw:<35} {n:>6} {m:>8} {hit:>5.1f}%  {verdict}")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
