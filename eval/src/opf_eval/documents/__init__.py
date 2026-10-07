"""Turn files and pasted text into `Document`s: plain text plus a map from
every character back to its page, cell, line or email header.

Most detectors only take text, so every file is parsed to text one standard
way first. The map lets a detected span be shown as "page 3" or
"People!B2" instead of a bare character offset. Chunking splits long
documents for detectors with input limits without cutting through a value.

Supported files (one parser module each, heavy libraries imported lazily):

    .pdf                pypdf         one segment per page; pages without text warn "needs OCR"
    .docx               python-docx   paragraphs, tables and content controls in order, then
                                      headers/footers; tracked insertions count, deletions do not
    .xlsx .xlsm         openpyxl      one line per row, one segment per cell ("Sheet", "B7")
    .csv .tsv           csv           same layout, {"row": n, "column": header}
    .txt .md .markdown  -             as written, one segment per line
    .html .htm          html.parser   visible text only, one segment per line
    .eml                email         headers, body, then supported attachments

PDF, DOCX and XLSX need the `files` extra: `pip install 'opf-eval[files]'`.

Text-based files (TXT, MD, CSV, HTML) are read as UTF-8, or as UTF-16/32
when they start with a byte-order mark. UTF-16 without a byte-order mark is
recognised by its NUL bytes and decoded with a warning. Files that are not
valid UTF-8 are read as Windows-1252. Whatever a parser could not read cleanly, such as an
encoding fallback, a scanned PDF page or DOCX text in an unsupported layout,
lands in `Document.warnings` instead of failing silently. A file that yields
no text at all gets the warning "no text was extracted".

Example:

    from opf_eval import documents, fixtures

    docs, skipped = documents.parse_many("inputs/")
    for s in skipped:
        print("skipped", s["path"], "-", s["reason"])
    fixtures.from_documents(docs, "data/custom/inputs.jsonl")

    doc = documents.parse("inputs/people.xlsx")
    i = doc.text.index("jane.doe@example.com")
    doc.locate(i, i + 20)   # -> [{"sheet": "People", "cell": "B2"}]
"""

from __future__ import annotations

import bisect
import hashlib
import importlib
import os
import re
from collections.abc import Iterable
from pathlib import Path

from .base import Document, Segment, TextBuilder, lines_into, normalize_newlines

__all__ = [
    "SUPPORTED_SUFFIXES",
    "Document",
    "Segment",
    "chunk",
    "extract_path",
    "from_text",
    "parse",
    "parse_many",
]

SUPPORTED_SUFFIXES: dict[str, str] = {
    ".pdf": "pdf",
    ".docx": "docx",
    ".xlsx": "xlsx",
    ".xlsm": "xlsx",
    ".csv": "csv",
    ".tsv": "csv",
    ".txt": "txt",
    ".md": "md",
    ".markdown": "md",
    ".html": "html",
    ".htm": "html",
    ".eml": "eml",
}

# kind -> "module:function". Modules are imported on first use so importing
# this package never pulls in pypdf/python-docx/openpyxl.
_EXTRACTORS: dict[str, str] = {
    "pdf": "pdf:extract",
    "docx": "docx:extract",
    "xlsx": "sheets:extract_xlsx",
    "csv": "sheets:extract_csv",
    "txt": "text:extract",
    "md": "text:extract",
    "html": "html:extract",
    "eml": "email:extract",
}

# Office writes "~$name.docx" lock files next to open documents. They have a
# supported suffix but are not documents.
_LOCK_PREFIX = "~$"


def extract_path(path: str | Path, kind: str, *, depth: int = 0) -> TextBuilder:
    """Run the parser for `kind` on `path` and return its builder.

    Low level: most callers want `parse`. `depth` counts how deeply an email
    attachment is nested so recursion stays bounded.
    """
    if kind not in _EXTRACTORS:
        raise ValueError(f"no parser for kind {kind!r}; known: {sorted(_EXTRACTORS)}")
    module_name, func_name = _EXTRACTORS[kind].split(":")
    module = importlib.import_module(f"{__name__}.{module_name}")
    func = getattr(module, func_name)
    if kind == "eml":
        return func(Path(path), depth=depth)
    return func(Path(path))


def _file_sha1(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _relative_name(path: Path, root: str | Path | None) -> str:
    if root is not None:
        try:
            return path.resolve().relative_to(Path(root).resolve()).as_posix()
        except ValueError:
            pass
    return path.name


def parse(path: str | Path, *, root: str | Path | None = None) -> Document:
    """Parse one file into a `Document`, choosing the parser by suffix.

    `root` makes the document id use the path relative to that folder, so two
    files with the same name in different subfolders get different ids.
    Raises ValueError for an unsupported suffix and FileNotFoundError when the
    file does not exist. Parser errors propagate; `parse_many` records them.
    """
    path = Path(path)
    kind = SUPPORTED_SUFFIXES.get(path.suffix.lower())
    if kind is None:
        raise ValueError(
            f"unsupported file type {path.suffix or '(no suffix)'!r} for {path}; "
            f"supported: {', '.join(sorted(SUPPORTED_SUFFIXES))}"
        )
    if not path.is_file():
        raise FileNotFoundError(path)
    builder = extract_path(path, kind)
    doc_id = f"{_file_sha1(path)[:12]}:{_relative_name(path, root)}"
    doc = builder.build(id=doc_id, source=str(path), kind=kind)
    problems = doc.check()
    if problems:  # a parser bug, not a bad file — fail loudly
        raise AssertionError(f"{path}: parser produced invalid segments: {problems[:3]}")
    if not doc.text.strip() and not doc.warnings:
        # An empty result is otherwise invisible: the file just yields no
        # fixture records.
        doc.warnings.append("no text was extracted")
    return doc


def _is_hidden(path: Path, root: Path) -> bool:
    try:
        parts = path.relative_to(root).parts
    except ValueError:
        parts = (path.name,)
    return any(p.startswith(".") for p in parts)


def _walk(folder: Path, recursive: bool) -> list[Path]:
    if not recursive:
        return sorted(p for p in folder.iterdir() if p.is_file())
    files = []
    for dirpath, dirnames, filenames in os.walk(folder):
        # Prune hidden folders (.git, .ipynb_checkpoints) instead of walking them.
        dirnames[:] = sorted(d for d in dirnames if not d.startswith("."))
        files.extend(Path(dirpath) / f for f in filenames)
    return sorted(files)


def parse_many(
    paths_or_dir: str | Path | Iterable[str | Path],
    *,
    recursive: bool = True,
) -> tuple[list[Document], list[dict]]:
    """Parse a folder, a file, or a list of either. Never raises on a bad file.

    Folders are walked (recursively by default) in sorted order and hidden
    files and folders are ignored. Document ids are relative to the folder
    they were found in. Returns `(docs, skipped)` where each skipped entry is
    `{"path": str, "reason": str}`: unsupported suffixes, Office lock files,
    missing paths, duplicates and files whose parser failed (with the error,
    including a missing-extra install hint).
    """
    if isinstance(paths_or_dir, (str, os.PathLike)):
        items: list[str | Path] = [paths_or_dir]
    else:
        items = list(paths_or_dir)

    targets: list[tuple[Path, Path | None]] = []
    skipped: list[dict] = []
    for item in items:
        p = Path(item)
        if p.is_dir():
            targets.extend((f, p) for f in _walk(p, recursive) if not _is_hidden(f, p))
        elif p.is_file():
            targets.append((p, None))
        else:
            skipped.append({"path": str(p), "reason": "not found"})

    docs: list[Document] = []
    seen: dict[str, str] = {}
    for path, root in targets:
        if path.name.startswith(_LOCK_PREFIX):
            skipped.append({"path": str(path), "reason": "Office lock file"})
            continue
        if path.suffix.lower() not in SUPPORTED_SUFFIXES:
            skipped.append({
                "path": str(path),
                "reason": f"unsupported file type {path.suffix or '(no suffix)'!r}",
            })
            continue
        try:
            doc = parse(path, root=root)
        except Exception as e:  # noqa: BLE001 — one bad file must not stop the batch
            skipped.append({"path": str(path), "reason": f"{type(e).__name__}: {e}"})
            continue
        if doc.id in seen:
            skipped.append({"path": str(path), "reason": f"duplicate of {seen[doc.id]}"})
            continue
        seen[doc.id] = str(path)
        docs.append(doc)
    return docs, skipped


def from_text(text: str, *, id: str | None = None, source: str = "<pasted>") -> Document:
    """Wrap pasted text as a `Document` (kind "text", one segment per line).

    The default id is derived from the text so the same paste always gets the
    same id.
    """
    text = normalize_newlines(text)
    b = TextBuilder()
    lines_into(b, text)
    doc_id = (
        id if id is not None else f"{hashlib.sha1(text.encode('utf-8')).hexdigest()[:12]}:pasted"
    )
    return b.build(id=doc_id, source=source, kind="text")


# ------------------------------------------------------------------ chunking

_SOFT_BREAKS = ("\n", ". ", " ")
_NON_SPACE = re.compile(r"\S")


def chunk(doc: Document, *, max_chars: int = 4000) -> list[tuple[int, int]]:
    """Split `doc.text` into consecutive `(start, end)` ranges of at most
    `max_chars` characters.

    Cuts go at the last segment boundary that fits, so a page, cell or line
    is never split across chunks. Only a single segment longer than
    `max_chars` is cut inside, at the last newline, sentence end or space
    before the limit (a hard cut if there is none). The ranges cover the
    whole text with no gaps or overlaps. Empty text gives no chunks.

    A chunk never holds only whitespace when that can be avoided. Separators
    in front of the next piece of text stay with that text, and trailing
    whitespace joins the last chunk if it fits there. When the separators
    and the next segment do not fit together, the separators join the
    previous chunk if they fit there. A whitespace-only chunk is left only
    when the separators fit nowhere else or the trailing whitespace does not
    fit in the last chunk.
    """
    if max_chars < 1:
        raise ValueError("max_chars must be at least 1")
    text = doc.text
    n = len(text)
    bounds = sorted({b for seg in doc.segments for b in (seg.start, seg.end)} | {n})
    ranges: list[tuple[int, int]] = []
    pos = 0
    while pos < n:
        limit = pos + max_chars
        # First character that is not whitespace. The cut must come after it
        # or the chunk would hold nothing but separators.
        m = _NON_SPACE.search(text, pos)
        q = m.start() if m else n
        if q >= n and ranges and n - ranges[-1][0] <= max_chars:
            ranges[-1] = (ranges[-1][0], n)  # trailing whitespace joins the last chunk
            break
        if limit >= n:
            ranges.append((pos, n))
            break
        lo = q if q < limit else pos
        i = bisect.bisect_right(bounds, limit) - 1
        cut = bounds[i] if i >= 0 else pos
        if cut <= lo and lo > pos and _fits_from(bounds, lo, max_chars):
            # The separators in front push a segment that fits on its own
            # past the limit. They go to the previous chunk when there is
            # room and into a chunk of their own when there is not, so the
            # segment is not cut.
            if ranges and lo - ranges[-1][0] <= max_chars:
                ranges[-1] = (ranges[-1][0], lo)
            else:
                ranges.append((pos, lo))
            pos = lo
            continue
        if cut <= lo:
            cut = _soft_cut(text, lo, limit)
        ranges.append((pos, cut))
        pos = cut
    return ranges


def _fits_from(bounds: list[int], start: int, max_chars: int) -> bool:
    """True when a segment boundary lies in (start, start + max_chars]."""
    i = bisect.bisect_right(bounds, start + max_chars) - 1
    return i >= 0 and bounds[i] > start


def _soft_cut(text: str, pos: int, limit: int) -> int:
    """Last position in (pos, limit] just after a newline, sentence end or space."""
    for brk in _SOFT_BREAKS:
        k = text.rfind(brk, pos, limit)
        if k != -1 and k + len(brk) > pos and k + len(brk) <= limit:
            return k + len(brk)
    return limit
