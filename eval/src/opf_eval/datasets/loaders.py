"""Per-dataset record loaders. Each maps a raw HF record (dict) to our
fixture shape: {id, text, gold_spans, language}.

All five ai4privacy datasets we support use the same `privacy_mask` shape
(list of {label, start, end, value}); they differ in:
- ID field name (`id` vs `uid`)
- Label vocabulary (300k uses numbered names, 200k has its own, 400k +
  openpii share OpenPII vocab)
- Language field format (300k uses full names, others use ISO codes)
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Callable

# `_to_canonical` will be supplied by the registry — it's vocab-keyed.
ToCanonical = Callable[[str], "str | None"]


# Map ai4privacy 300k full-name languages -> ISO 639-1 codes. New datasets
# already use ISO; this only normalises legacy 300k records.
_LEGACY_LANG_TO_ISO: dict[str, str] = {
    "English": "en",
    "Dutch": "nl",
    "French": "fr",
    "German": "de",
    "Italian": "it",
    "Spanish": "es",
}


def _normalize_language(raw: object) -> str | None:
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    if s in _LEGACY_LANG_TO_ISO:
        return _LEGACY_LANG_TO_ISO[s]
    # Already ISO (or unknown — pass through unchanged so callers can see it)
    return s.lower() if len(s) == 2 else s


def _record_id(rec: dict, fallback_idx: int) -> str:
    for key in ("id", "uid"):
        v = rec.get(key)
        if v is not None:
            return str(v)
    return str(fallback_idx)


def _record_text(rec: dict) -> str:
    for key in ("source_text", "unmasked_text", "text", "raw_text"):
        v = rec.get(key)
        if isinstance(v, str):
            return v
    raise KeyError(f"no text column in record: {sorted(rec)}")


def _privacy_mask_to_spans(
    raw_spans: list, to_canonical: ToCanonical
) -> list[dict]:
    """ai4privacy `privacy_mask` -> our gold_spans shape, restricted to
    labels that map to a canonical category."""
    out: list[dict] = []
    for s in raw_spans or []:
        if not isinstance(s, dict):
            continue
        label = s.get("label") or s.get("entity_type") or s.get("type")
        start = s.get("start") if "start" in s else s.get("start_index")
        end = s.get("end") if "end" in s else s.get("end_index")
        if label is None or start is None or end is None:
            continue
        canonical = to_canonical(label)
        if not canonical:
            continue
        out.append(
            {
                "label": canonical,
                "raw_label": label,
                "start": int(start),
                "end": int(end),
            }
        )
    return out


def make_loader(to_canonical: ToCanonical) -> Callable[[Iterable[dict]], Iterable[dict]]:
    """Build a record loader for a given dataset's `to_canonical` function.

    The five ai4privacy datasets we support share the `privacy_mask` shape
    so a single template suffices — only the vocab mapping differs.
    """

    def loader(raw_records: Iterable[dict]) -> Iterable[dict]:
        for idx, rec in enumerate(raw_records):
            try:
                text = _record_text(rec)
            except KeyError:
                continue
            spans = _privacy_mask_to_spans(rec.get("privacy_mask") or [], to_canonical)
            yield {
                "id": _record_id(rec, idx),
                "text": text,
                "gold_spans": spans,
                "language": _normalize_language(rec.get("language")),
            }

    return loader
