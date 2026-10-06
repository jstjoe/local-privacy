"""Core types for parsed documents: `Segment`, `Document` and the text builder
every parser uses so offsets stay consistent.

A parser never hands back raw strings. It feeds pieces of extracted text and
the separators between them into a `TextBuilder`, which records one `Segment`
per piece. That keeps the invariants the rest of the harness relies on in a
single place:

1. Segments are sorted and never overlap.
2. `doc.text[seg.start:seg.end]` is exactly the extracted piece.
3. Separators ("\\n", "\\t", "\\n\\n") sit between segments and belong to none.

Example:

    b = TextBuilder()
    b.add("Jane Doe", {"line": 1})
    b.newline()
    b.add("jane@example.com", {"line": 2})
    doc = b.build(id="abc:notes.txt", source="notes.txt", kind="txt")
    doc.locate(9, 25)   # -> [{"line": 2}]
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass, field
from typing import Any


def normalize_newlines(text: str) -> str:
    """CRLF and lone CR become LF so offsets mean the same on every platform."""
    return text.replace("\r\n", "\n").replace("\r", "\n")


@dataclass(frozen=True)
class Segment:
    """A located piece of a document's text.

    `start` and `end` are offsets into `Document.text` (end-exclusive).
    `where` says where the piece came from in the original file, for example
    `{"page": 3}` or `{"sheet": "Q1", "cell": "B7"}` or
    `{"part": "header", "name": "From"}`.
    """

    start: int
    end: int
    where: dict

    def to_json(self) -> list:
        return [self.start, self.end, self.where]

    @classmethod
    def from_json(cls, row: list | tuple | dict) -> Segment:
        if isinstance(row, dict):
            return cls(int(row["start"]), int(row["end"]), dict(row["where"]))
        start, end, where = row
        return cls(int(start), int(end), dict(where))


@dataclass
class Document:
    """One parsed file (or pasted text) as plain text plus a location map.

    id: `"<sha1 of the file bytes>[:12]:<relative path>"`, stable across runs
        and machines as long as the file content and its path under the
        parse root do not change.
    source: the path as given to the parser.
    kind: "pdf" | "docx" | "xlsx" | "csv" | "txt" | "md" | "html" | "eml" |
        "text" (pasted text).
    warnings: things the parser could not extract, such as a PDF page with
        no text layer. They are reported to the user and never raised.
    """

    id: str
    source: str
    kind: str
    text: str
    segments: list[Segment]
    warnings: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._starts: list[int] | None = None

    def locate(self, start: int, end: int) -> list[dict]:
        """`where` dicts of every segment that overlaps `[start, end)`.

        A zero-width range returns the segment containing that position.
        Ranges that fall only on separators return an empty list.
        """
        if self._starts is None or len(self._starts) != len(self.segments):
            self._starts = [s.start for s in self.segments]
        if end <= start:
            end = start + 1
        # Segments are sorted and disjoint, so every overlapping segment sits
        # between the last one starting at or before `start` and the first
        # one starting at or after `end`.
        i = max(0, bisect.bisect_right(self._starts, start) - 1)
        out = []
        for seg in self.segments[i:]:
            if seg.start >= end:
                break
            if seg.end > start:
                out.append(dict(seg.where))
        return out

    def segment_text(self, seg: Segment) -> str:
        return self.text[seg.start : seg.end]

    def check(self) -> list[str]:
        """Return invariant violations (empty when the document is well formed)."""
        problems = []
        prev_end = 0
        for k, seg in enumerate(self.segments):
            if not 0 <= seg.start < seg.end <= len(self.text):
                problems.append(f"segment {k} [{seg.start}, {seg.end}) is empty or out of range")
            if seg.start < prev_end:
                problems.append(f"segment {k} [{seg.start}, {seg.end}) overlaps or is out of order")
            prev_end = max(prev_end, seg.end)
        return problems


class TextBuilder:
    """Accumulate text pieces and separators and record a segment per piece.

    Pieces have their newlines normalised before they are stored. Empty and
    whitespace-only pieces are dropped (no segment, no text) so a blank cell
    or paragraph never produces a zero-width segment.
    """

    def __init__(self) -> None:
        self._parts: list[str] = []
        self._len = 0
        self.segments: list[Segment] = []
        self.warnings: list[str] = []

    def __len__(self) -> int:
        return self._len

    @property
    def text(self) -> str:
        return "".join(self._parts)

    def _append(self, s: str) -> None:
        self._parts.append(s)
        self._len += len(s)

    def add(self, piece: Any, where: dict) -> bool:
        """Append a piece and record its segment. Returns False if it was blank."""
        piece = normalize_newlines(str(piece))
        if not piece.strip():
            return False
        start = self._len
        self._append(piece)
        self.segments.append(Segment(start, self._len, dict(where)))
        return True

    def sep(self, s: str) -> None:
        """Append a separator. Leading separators are dropped so text never
        starts with an empty line."""
        if self._len:
            self._append(s)

    def literal(self, s: str) -> None:
        """Append text that belongs to no segment, even at the very start
        (for example the "From: " label in front of a header value)."""
        self._append(s)

    def newline(self) -> None:
        self.sep("\n")

    def blank_line(self) -> None:
        self.sep("\n\n")

    def extend(self, other: TextBuilder | Document, wrap=None, *, sep: str = "\n\n") -> None:
        """Append another builder or document, shifting its segments.

        `wrap(where) -> where` rewrites each appended segment's location, for
        example to nest an email attachment's locations under its filename.
        """
        text = other.text
        if not text:
            return
        self.sep(sep)
        base = self._len
        self._append(text)
        for seg in other.segments:
            where = wrap(dict(seg.where)) if wrap else dict(seg.where)
            self.segments.append(Segment(seg.start + base, seg.end + base, where))

    def build(self, *, id: str, source: str, kind: str) -> Document:
        return Document(
            id=id,
            source=source,
            kind=kind,
            text=self.text,
            segments=list(self.segments),
            warnings=list(self.warnings),
        )


def lines_into(builder: TextBuilder, text: str, where=lambda n: {"line": n}) -> None:
    """Append `text` as-is with one segment per non-blank line.

    Line numbers are 1-based and count blank lines too, so `{"line": n}`
    matches what an editor shows.
    """
    text = normalize_newlines(text)
    if not text:
        return
    if builder._len:
        builder.newline()
    pos = len(builder)
    builder._append(text)
    for n, line in enumerate(text.split("\n"), 1):
        if line.strip():
            builder.segments.append(Segment(pos, pos + len(line), where(n)))
        pos += len(line) + 1
