"""Record loaders: raw dataset records -> fixture records.

Three schemas cover most PII datasets:

- `privacy_mask_loader` — ai4privacy's `privacy_mask` list of
  `{label, start, end, value}`.
- `spans_loader` — any record with a text field and a list of
  character-offset spans (`{start, end, label}` under common key names).
  This is the format to use for your own data.
- `bio_loader` — token-level NER: a token list plus BIO / BIOES / BILOU tags
  (CoNLL-style datasets, SPY, PIIBench). Text and character offsets are
  rebuilt from the tokens.

Every loader maps raw labels through a taxonomy vocabulary and drops spans
whose label it can't map. Records with no usable text are skipped.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import Any, Optional, Union

from ..taxonomy import parent, to_canonical
from .base import Loader

# A vocab key, or a callable raw label -> canonical label (None = drop).
LabelMapper = Union[str, Callable[[str], Optional[str]]]


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

_TEXT_KEYS = ("source_text", "unmasked_text", "text", "raw_text", "content", "document")
_SPAN_LIST_KEYS = ("gold_spans", "spans", "entities", "privacy_mask", "annotations", "pii_spans")
_ID_KEYS = ("id", "uid", "row_id", "doc_id", "document_id", "record_id")
_LABEL_KEYS = ("label", "entity_type", "type", "entity", "tag", "category")
_START_KEYS = ("start", "start_index", "start_offset", "begin", "char_start")
_END_KEYS = ("end", "end_index", "end_offset", "char_end")


def normalize_language(raw: object) -> str | None:
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    if s in _LEGACY_LANG_TO_ISO:
        return _LEGACY_LANG_TO_ISO[s]
    # Already ISO (or unknown — pass through unchanged so callers can see it)
    return s.lower() if len(s) == 2 else s


def _first(rec: dict, keys: Sequence[str]) -> Any:
    for k in keys:
        if k in rec and rec[k] is not None:
            return rec[k]
    return None


def _mapper(labels: LabelMapper) -> Callable[[str], "str | None"]:
    if isinstance(labels, str):
        vocab = labels
        return lambda raw: to_canonical(vocab, raw)
    return labels


def gold_span(canonical: str, raw_label: str, start: int, end: int) -> dict:
    """One gold span in fixture shape (coarse `label` + `fine_label`)."""
    return {
        "label": parent(canonical),
        "fine_label": canonical,
        "raw_label": raw_label,
        "start": int(start),
        "end": int(end),
    }


def spans_from_dicts(
    raw_spans: Iterable[Any],
    to_canonical_fn: Callable[[str], "str | None"],
    *,
    label_keys: Sequence[str] = _LABEL_KEYS,
    start_keys: Sequence[str] = _START_KEYS,
    end_keys: Sequence[str] = _END_KEYS,
) -> list[dict]:
    """Span dicts under common key names -> gold spans. Unmappable labels and
    malformed entries are dropped."""
    out: list[dict] = []
    for s in raw_spans or []:
        if not isinstance(s, dict):
            continue
        label = _first(s, label_keys)
        start = _first(s, start_keys)
        end = _first(s, end_keys)
        if label is None or start is None or end is None:
            continue
        canonical = to_canonical_fn(str(label))
        if not canonical:
            continue
        out.append(gold_span(canonical, str(label), int(start), int(end)))
    return out


def spans_loader(
    labels: LabelMapper = "canonical",
    *,
    text_field: str | None = None,
    spans_field: str | None = None,
    id_field: str | None = None,
    language_field: str | None = "language",
    default_language: str | None = None,
    label_key: str | None = None,
    start_key: str | None = None,
    end_key: str | None = None,
) -> Loader:
    """Loader for records carrying text + a list of character-offset spans.

    Field names default to common spellings (`text`, `spans`/`gold_spans`/
    `entities`, `id`, span keys `label`/`start`/`end` and variants); pass
    them explicitly when your data uses something else. A span's `start`
    and `end` must be character offsets into the text, end-exclusive.
    """
    to_can = _mapper(labels)
    lk = (label_key,) if label_key else _LABEL_KEYS
    sk = (start_key,) if start_key else _START_KEYS
    ek = (end_key,) if end_key else _END_KEYS

    def loader(raw_records: Iterable[dict]) -> Iterable[dict]:
        for idx, rec in enumerate(raw_records):
            text = rec.get(text_field) if text_field else _first(rec, _TEXT_KEYS)
            if not isinstance(text, str):
                continue
            raw_spans = rec.get(spans_field) if spans_field else _first(rec, _SPAN_LIST_KEYS)
            rid = rec.get(id_field) if id_field else _first(rec, _ID_KEYS)
            lang = rec.get(language_field) if language_field else None
            yield {
                "id": str(rid) if rid is not None else str(idx),
                "text": text,
                "gold_spans": spans_from_dicts(
                    raw_spans or [], to_can, label_keys=lk, start_keys=sk, end_keys=ek
                ),
                "language": normalize_language(lang) or default_language,
            }

    return loader


def privacy_mask_loader(labels: LabelMapper) -> Loader:
    """Loader for ai4privacy datasets (all share the `privacy_mask` shape;
    they differ in id field, label vocabulary and language format)."""
    return spans_loader(labels, spans_field="privacy_mask")


# Back-compat name used by the original registry.
make_loader = privacy_mask_loader


def bio_spans(
    tokens: Sequence[str],
    tags: Sequence[str],
    *,
    whitespace: Sequence[bool] | None = None,
) -> tuple[str, list[tuple[str, int, int]]]:
    """Rebuild text + character-offset (label, start, end) spans from tokens.

    `whitespace[i]` says whether token i is followed by a space (spaCy /
    SPY style `trailing_whitespace`); default is a single space between
    every token. Tags may be BIO, BIOES or BILOU; an `I-` that doesn't
    continue the current entity starts a new one (lenient decoding).
    """
    if len(tokens) != len(tags):
        raise ValueError(f"{len(tokens)} tokens but {len(tags)} tags")
    pieces: list[str] = []
    offsets: list[tuple[int, int]] = []
    pos = 0
    for i, tok in enumerate(tokens):
        offsets.append((pos, pos + len(tok)))
        pieces.append(tok)
        pos += len(tok)
        sep = whitespace[i] if whitespace is not None else i < len(tokens) - 1
        if sep:
            pieces.append(" ")
            pos += 1
    text = "".join(pieces)

    spans: list[tuple[str, int, int]] = []
    cur: list = []  # [label, start, end]

    def close() -> None:
        if cur:
            spans.append((cur[0], cur[1], cur[2]))
            cur.clear()

    for (start, end), tag in zip(offsets, tags):
        tag = str(tag)
        if tag in ("O", "", "0"):
            close()
            continue
        prefix, _, label = tag.partition("-") if "-" in tag and tag[1:2] == "-" else ("I", "-", tag)
        prefix = prefix.upper()
        if prefix in ("B", "S", "U") or not cur or cur[0] != label:
            close()
            cur.extend([label, start, end])
        else:
            cur[2] = end
        if prefix in ("S", "U", "E", "L"):
            close()
    close()
    return text, spans


def bio_loader(
    labels: LabelMapper,
    *,
    tokens_field: str = "tokens",
    tags_field: str | None = None,
    whitespace_field: str | None = "trailing_whitespace",
    tag_names: Sequence[str] | None = None,
    id_field: str | None = None,
    language_field: str | None = "language",
    default_language: str | None = None,
) -> Loader:
    """Loader for token-level NER records (tokens + BIO-style tags).

    tags_field: defaults to the first of `ner_tags`, `labels`, `tags`.
    tag_names: when tags are integer class ids (HF `ClassLabel`), the names
        to index into, e.g. `ds.features["ner_tags"].feature.names`.
    """
    to_can = _mapper(labels)

    def loader(raw_records: Iterable[dict]) -> Iterable[dict]:
        for idx, rec in enumerate(raw_records):
            tokens = rec.get(tokens_field)
            tags = rec.get(tags_field) if tags_field else _first(rec, ("ner_tags", "labels", "tags"))
            if not tokens or tags is None:
                continue
            if tag_names is not None:
                tags = [tag_names[t] if isinstance(t, int) else t for t in tags]
            ws = rec.get(whitespace_field) if whitespace_field else None
            text, raw = bio_spans(tokens, tags, whitespace=ws)
            spans = []
            for label, start, end in raw:
                canonical = to_can(label)
                if canonical:
                    spans.append(gold_span(canonical, label, start, end))
            rid = rec.get(id_field) if id_field else _first(rec, _ID_KEYS)
            lang = rec.get(language_field) if language_field else None
            yield {
                "id": str(rid) if rid is not None else str(idx),
                "text": text,
                "gold_spans": spans,
                "language": normalize_language(lang) or default_language,
            }

    return loader


SCHEMAS: dict[str, Callable[..., Loader]] = {
    "spans": spans_loader,
    "privacy_mask": privacy_mask_loader,
    "bio": bio_loader,
}
