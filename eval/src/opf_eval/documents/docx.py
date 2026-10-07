"""DOCX parser (python-docx): paragraphs and table cells in document order.

Locations use python-docx's own indexing so a `where` can be looked up
directly in the file:

    {"paragraph": i}                  doc.paragraphs[i] (0-based, blank ones counted)
    {"table": t, "row": r, "col": c}  doc.tables[t].rows[r].cells[c] (0-based)
    {"content_control": k, "paragraph": j}
    {"content_control": k, "table": t, "row": r, "col": c}
                                      the k-th content control (`w:sdt`) that
                                      sits directly in the body, counting its
                                      own paragraphs and tables from 0
    {"part": "header", "section": s}  doc.sections[s].header (also "footer")

Body content is walked in XML order so a table appears between the
paragraphs that surround it. Table rows end a line and cells of a row are
joined by a tab. A merged cell is emitted once. Headers and footers come last
after a blank line. A first-page or even-page variant adds `"variant"` to the
location. Blank paragraphs are skipped.

Paragraph text is read from the XML rather than python-docx's
`Paragraph.text`, which skips text inside tracked insertions, content
controls, simple fields and smart tags. Every `w:t` counts, `w:tab` is a tab
and `w:br`/`w:cr` a newline. Deleted text (`w:del`, `w:moveFrom`), field
instructions and the legacy fallback copy of drawings (`mc:Fallback`) are
left out. Text boxes inside a paragraph follow its text on new lines.

Anything in the body that this walk still cannot place, such as table rows
wrapped in a content control, is counted. When the extracted text has fewer
characters than the body holds, a warning says how many are missing.
"""

from __future__ import annotations

from pathlib import Path

from .base import TextBuilder

FILES_HINT = "pip install 'opf-eval[files]'"

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
MC = "{http://schemas.openxmlformats.org/markup-compatibility/2006}"
W_P, W_TBL, W_SDT, W_SDT_CONTENT = W + "p", W + "tbl", W + "sdt", W + "sdtContent"
W_T, W_TAB, W_BR, W_CR, W_NB_HYPHEN = W + "t", W + "tab", W + "br", W + "cr", W + "noBreakHyphen"
# Subtrees that hold no visible text, or text that is not part of the
# current document (deletions, the moved-from copy of moved text, and the
# VML fallback that duplicates a drawing's text box).
_SKIP = frozenset({
    W + "pPr", W + "rPr", W + "tblPr", W + "tblGrid", W + "trPr", W + "tcPr",
    W + "sdtPr", W + "sdtEndPr", W + "del", W + "moveFrom", W + "sectPr",
    MC + "Fallback",
})


def _walk_text(el, out: list[str]) -> None:
    for child in el.iterchildren():
        tag = child.tag
        if not isinstance(tag, str) or tag in _SKIP:  # comments, processing instructions
            continue
        if tag == W_T:
            out.append(child.text or "")
        elif tag == W_TAB:
            out.append("\t")
        elif tag in (W_BR, W_CR):
            out.append("\n")
        elif tag == W_NB_HYPHEN:
            out.append("-")
        elif tag == W_P:
            # A paragraph nested in this one lives in a text box.
            out.append("\n")
            _walk_text(child, out)
        else:
            _walk_text(child, out)


def paragraph_text(p) -> str:
    """Visible text of one `w:p` element."""
    out: list[str] = []
    _walk_text(p, out)
    return "".join(out)


def _block_texts(el) -> list[str]:
    """Texts of the paragraphs in a block container (body, cell, content
    control, header) in order, descending into tables and content controls."""
    texts: list[str] = []
    for child in el.iterchildren():
        tag = child.tag
        if tag == W_P:
            texts.append(paragraph_text(child))
        elif tag == W_SDT:
            content = child.find(W_SDT_CONTENT)
            if content is not None:
                texts.extend(_block_texts(content))
        elif tag == W_TBL:
            for tr in child.iterchildren(W + "tr"):
                for tc in tr.iterchildren(W + "tc"):
                    texts.extend(_block_texts(tc))
    return texts


def cell_text(tc) -> str:
    """Visible text of one `w:tc`, nested tables and content controls included."""
    return "\n".join(t for t in _block_texts(tc) if t.strip())


def _all_visible_chars(el) -> int:
    out: list[str] = []
    _walk_text(el, out)
    return sum(1 for ch in "".join(out) if not ch.isspace())


def _add_table(b: TextBuilder, table, where_of) -> None:
    # Holding the elements keeps their ids unique: lxml proxies are created
    # on access, so a bare id() could be reused after garbage collection.
    seen: dict[int, object] = {}
    for r, row in enumerate(table.rows):
        first = True
        for c, cell in enumerate(row.cells):
            # python-docx repeats a merged cell once per grid column it spans
            # and once per row of a vertical merge. Emit its text only once.
            tc = cell._tc
            if id(tc) in seen:
                continue
            seen[id(tc)] = tc
            text = cell_text(tc)
            if not text.strip():
                continue
            if first:
                b.newline()
                first = False
            else:
                b.sep("\t")
            b.add(text, where_of(r, c))


def _add_blocks(
    b: TextBuilder, elements, parent, where_of, counters: dict, *, into_controls: bool
) -> None:
    """Emit the paragraphs and tables among `elements` in order.

    `where_of(kind, index)` turns ("paragraph", i) or ("table", t) into a
    location. `counters` holds the running paragraph and table indexes. With
    `into_controls`, a nested content control is read in place and continues
    the numbering of the one around it.
    """
    from docx.table import Table

    for el in elements:
        tag = el.tag
        if tag == W_P:
            text = paragraph_text(el)
            if text.strip():
                b.newline()
                b.add(text, where_of("paragraph", counters["paragraph"]))
            counters["paragraph"] += 1
        elif tag == W_TBL:
            base = where_of("table", counters["table"])
            _add_table(b, Table(el, parent), lambda r, c, base=base: {**base, "row": r, "col": c})
            counters["table"] += 1
        elif tag == W_SDT and into_controls:
            content = el.find(W_SDT_CONTENT)
            if content is not None:
                _add_blocks(
                    b, content.iterchildren(), parent, where_of, counters, into_controls=True
                )


def extract(path: Path) -> TextBuilder:
    try:
        import docx
    except ImportError as e:  # pragma: no cover - depends on the environment
        raise ImportError(f"Parsing .docx files needs python-docx: {FILES_HINT}") from e

    document = docx.Document(str(path))
    body = document.element.body
    parent = document._body
    b = TextBuilder()
    # Direct body paragraphs and tables share one running numbering that
    # matches doc.paragraphs and doc.tables. Each content control numbers
    # its own content from 0.
    body_counters = {"paragraph": 0, "table": 0}
    control = 0
    for child in body.iterchildren():
        if child.tag in (W_P, W_TBL):
            _add_blocks(
                b,
                (child,),
                parent,
                lambda kind, i: {kind: i},
                body_counters,
                into_controls=False,
            )
        elif child.tag == W_SDT:
            content = child.find(W_SDT_CONTENT)
            if content is not None:
                _add_blocks(
                    b, content.iterchildren(), parent,
                    lambda kind, i, k=control: {"content_control": k, kind: i},
                    {"paragraph": 0, "table": 0}, into_controls=True,
                )
            control += 1

    body_chars = _all_visible_chars(body)
    got_chars = sum(1 for ch in b.text if not ch.isspace())
    if got_chars < body_chars:
        b.warnings.append(
            f"{body_chars - got_chars} characters of body text could not be extracted "
            "(content in a layout this parser does not read, "
            "such as table rows inside a content control)"
        )

    first_extra = True
    for s, section in enumerate(document.sections):
        for part, variant, hf in (
            ("header", None, section.header),
            ("header", "first", section.first_page_header),
            ("header", "even", section.even_page_header),
            ("footer", None, section.footer),
            ("footer", "first", section.first_page_footer),
            ("footer", "even", section.even_page_footer),
        ):
            # A linked header/footer has no content of its own in this
            # section; reading it would repeat the previous section's text.
            if hf.is_linked_to_previous:
                continue
            where = {"part": part, "section": s}
            if variant:
                where["variant"] = variant
            pieces = [t for t in _block_texts(hf._element) if t.strip()]
            for piece in dict.fromkeys(pieces):
                if first_extra:
                    b.blank_line()
                    first_extra = False
                else:
                    b.newline()
                b.add(piece, where)
    return b
