"""Identify which granular Skyflow entity types are redundant against the gold.

For each granular type (e.g. LOCATION_CITY, NAME_GIVEN), check:
- How often does it appear?
- Does a more general type (LOCATION, NAME) cover the same span when present?
- Net contribution: TPs gained from granular types vs FPs added.

Output: ranked list of granular types worth keeping vs dropping.
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

# Granular -> general mapping for grouping
GRANULAR_TO_GENERAL = {
    "NAME_GIVEN": "NAME",
    "NAME_FAMILY": "NAME",
    "NAME_MEDICAL_PROFESSIONAL": "NAME",
    "LOCATION_CITY": "LOCATION",
    "LOCATION_STATE": "LOCATION",
    "LOCATION_ZIP": "LOCATION",
    "LOCATION_COUNTRY": "LOCATION",
    "LOCATION_ADDRESS": "LOCATION",
    "LOCATION_ADDRESS_STREET": "LOCATION",
    "LOCATION_COORDINATE": "LOCATION",
    "DATE_INTERVAL": "DATE",
    "DOB": "DATE",
    "TIME": "DATE",
    "DAY": "DATE",
    "MONTH": "DATE",
    "YEAR": "DATE",
    "DRIVER_LICENSE": "ACCOUNT_NUMBER",
    "PASSPORT_NUMBER": "ACCOUNT_NUMBER",
    "SSN": "ACCOUNT_NUMBER",
    "HEALTHCARE_NUMBER": "ACCOUNT_NUMBER",
    "BANK_ACCOUNT": "ACCOUNT_NUMBER",
    "CREDIT_CARD": "ACCOUNT_NUMBER",
    "ROUTING_NUMBER": "ACCOUNT_NUMBER",
}


def _iou(a, b):
    lo = max(a["start"], b["start"])
    hi = min(a["end"], b["end"])
    inter = max(0, hi - lo)
    union = (a["end"] - a["start"]) + (b["end"] - b["start"]) - inter
    return inter / union if union > 0 else 0.0


def main(skyflow_jsonl: Path, fixtures_jsonl: Path):
    fixtures = {json.loads(l)["id"]: json.loads(l) for l in fixtures_jsonl.open() if l.strip()}
    sky_records = [json.loads(l) for l in skyflow_jsonl.open() if l.strip()]

    raw_label_counts: Counter = Counter()
    granular_overlap_with_general: Counter = Counter()  # granular -> # times overlapped by same-record general
    granular_matched_to_gold: Counter = Counter()       # granular -> matches to a gold span (by canonical)
    granular_total: Counter = Counter()

    for rec in sky_records:
        spans = rec.get("spans") or []
        gold = fixtures.get(rec["id"], {}).get("gold_spans") or []

        # Bucket spans by raw_label for this record
        by_raw: dict[str, list[dict]] = defaultdict(list)
        for s in spans:
            raw = s.get("raw_label") or ""
            raw_label_counts[raw] += 1
            by_raw[raw].append(s)

        # For each granular span, check if a general-type span covers it
        for granular, general in GRANULAR_TO_GENERAL.items():
            for g_span in by_raw.get(granular, []):
                granular_total[granular] += 1
                # Did any same-record general-type span overlap it?
                for general_span in by_raw.get(general, []):
                    if _iou(g_span, general_span) > 0.5:
                        granular_overlap_with_general[granular] += 1
                        break
                # Did this granular span hit gold?
                for gold_span in gold:
                    if g_span["label"] == gold_span["label"] and _iou(g_span, gold_span) >= 0.5:
                        granular_matched_to_gold[granular] += 1
                        break

    print("=== granular type usage and redundancy ===\n")
    print(f"{'granular':<35} {'count':>7} {'redundant%':>11} {'matches_gold':>14}")
    print("-" * 70)
    rows = []
    for granular in sorted(GRANULAR_TO_GENERAL.keys(), key=lambda x: -granular_total[x]):
        total = granular_total[granular]
        if total == 0:
            continue
        redundant = granular_overlap_with_general[granular]
        matches = granular_matched_to_gold[granular]
        redundant_pct = 100 * redundant / total if total else 0
        rows.append((granular, total, redundant_pct, matches))
        print(f"{granular:<35} {total:>7} {redundant_pct:>10.1f}% {matches:>14}")

    print("\n=== recommendation ===")
    drop = []
    keep = []
    for granular, total, redundant_pct, matches in rows:
        # Drop if mostly redundant AND doesn't add unique gold matches
        if redundant_pct > 70 and matches < total * 0.3:
            drop.append(granular)
        else:
            keep.append(granular)
    print(f"\nDROP (mostly redundant w/ general type): {drop}")
    print(f"KEEP (adds unique signal): {keep}")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
