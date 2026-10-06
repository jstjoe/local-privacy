"""HTML parser: visible text only, one segment per output line.

Rules:

1. Content of `script`, `style`, `head`, `title`, `template` and `noscript`
   is dropped. Both `<head>` tags are optional in HTML, so a `title` is
   dropped even without a head, and the head ends at `<body>` or at the
   first tag that cannot appear in a head (as in a browser).
2. Block elements (p, div, br, li, tr, headings, table, lists, pre and so on)
   end the current line.
3. Table cells on one row are separated by a tab so "Jane" and "Doe" in two
   cells do not run together.
4. Runs of whitespace inside a line collapse to one space. Inside `pre` the
   original line breaks are kept.
5. Entities are unescaped.

`html_to_lines` is shared with the email parser for HTML bodies.
"""

from __future__ import annotations

import re
from html.parser import HTMLParser
from pathlib import Path

from .base import TextBuilder
from .text import decode_text

SKIP_TAGS = frozenset({"script", "style", "head", "title", "template", "noscript"})
# Tags allowed inside <head>. Any other start tag closes an unclosed head the
# way a browser does, so a page with "<head>...<p>text" keeps its text.
HEAD_TAGS = frozenset({
    "html", "head", "meta", "title", "link", "style", "script", "base", "noscript", "template",
})
BLOCK_TAGS = frozenset({
    "p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "section",
    "article", "table", "ul", "ol", "pre", "blockquote", "header", "footer",
    # Not in the plan's list but they are block-level in every browser and
    # running their text into a neighbour would merge unrelated values.
    "dl", "dt", "dd", "hr", "nav", "aside", "main", "figure", "figcaption",
    "address", "form", "fieldset", "details", "summary", "caption", "body", "html",
    "option", "legend",
})
CELL_TAGS = frozenset({"td", "th"})
_WS = re.compile(r"\s+")
_SPACES = re.compile(r" {2,}")
_TAB_PAD = re.compile(r" *\t *")


class _VisibleText(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.lines: list[str] = []
        self._line: list[str] = []
        self._skip: list[str] = []
        self._pre = 0

    # -- line handling
    def _flush(self) -> None:
        line = "".join(self._line)
        self._line = []
        line = _TAB_PAD.sub("\t", _SPACES.sub(" ", line)).strip(" \t")
        if line:
            self.lines.append(line)

    # -- parser callbacks
    def handle_starttag(self, tag: str, attrs) -> None:
        if tag == "body" or (self._skip and self._skip[-1] == "head" and tag not in HEAD_TAGS):
            # `</head>` is optional in HTML. A body, or any tag that cannot
            # live in a head, ends it.
            self._skip = [t for t in self._skip if t != "head"]
        if tag in SKIP_TAGS:
            self._skip.append(tag)
            return
        if self._skip:
            return
        if tag in BLOCK_TAGS:
            self._flush()
            if tag == "pre":
                self._pre += 1
        elif tag in CELL_TAGS and self._line:
            self._line.append("\t")

    def handle_endtag(self, tag: str) -> None:
        if tag in SKIP_TAGS:
            if tag in self._skip:
                # Pop back to the matching open tag so a stray unclosed
                # inner skip tag cannot hide the rest of the page.
                while self._skip and self._skip.pop() != tag:
                    pass
            return
        if self._skip:
            return
        if tag in BLOCK_TAGS:
            self._flush()
            if tag == "pre" and self._pre:
                self._pre -= 1

    def handle_data(self, data: str) -> None:
        if self._skip or not data:
            return
        if self._pre:
            parts = data.replace("\r\n", "\n").replace("\r", "\n").split("\n")
            for k, part in enumerate(parts):
                if k:
                    self._flush()
                self._line.append(part.replace("\t", " "))
            return
        self._line.append(_WS.sub(" ", data))

    def close(self) -> None:
        super().close()
        self._flush()


def html_to_lines(markup: str) -> list[str]:
    """Visible text of an HTML string as a list of non-empty lines."""
    p = _VisibleText()
    p.feed(markup)
    p.close()
    return p.lines


def lines_to_builder(lines: list[str], where=lambda n: {"line": n}) -> TextBuilder:
    b = TextBuilder()
    for n, line in enumerate(lines, 1):
        b.newline()
        b.add(line, where(n))
    return b


def extract(path: Path) -> TextBuilder:
    markup, warnings = decode_text(Path(path).read_bytes())
    b = lines_to_builder(html_to_lines(markup))
    b.warnings.extend(warnings)
    return b
