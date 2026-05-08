"""Dump one raw PII-Masking-300k record so we can see what columns + label names exist."""

import json

from datasets import load_dataset


ds = load_dataset("ai4privacy/pii-masking-300k", split="train", streaming=False)

# columns
print("--- columns ---")
print(ds.column_names)

# first record (truncated)
print("\n--- first record (keys + truncated values) ---")
rec = ds[0]
for k, v in rec.items():
    s = repr(v)
    print(f"{k}: {s[:200]}{'...' if len(s) > 200 else ''}")

# all unique labels — sample 500 records
print("\n--- unique labels in first 500 records ---")
labels = set()
for i in range(min(500, len(ds))):
    r = ds[i]
    # try common columns
    for col in ("ner_tags_str", "ner_tags", "labels", "privacy_mask", "spans"):
        v = r.get(col)
        if v is None:
            continue
        if isinstance(v, list):
            for item in v:
                if isinstance(item, str):
                    labels.add(item)
                elif isinstance(item, dict):
                    for k in ("label", "entity_type", "type"):
                        if k in item:
                            labels.add(item[k])
print(sorted(labels))
