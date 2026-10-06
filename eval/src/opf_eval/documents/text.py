"""Plain text and Markdown parser.

The file is decoded with `decode_text` and kept exactly as written apart from
newline normalisation. Markdown syntax is not stripped because removing it
would break the offset map and detectors cope with it fine. Each non-blank
line is one segment `{"line": n}` with `n` counted from 1.
"""

from __future__ import annotations

import codecs
from pathlib import Path

from .base import TextBuilder, lines_into

# Longest BOMs first: the UTF-32-LE BOM starts with the UTF-16-LE one.
_BOMS = (
    (codecs.BOM_UTF32_LE, "utf-32-le"),
    (codecs.BOM_UTF32_BE, "utf-32-be"),
    (codecs.BOM_UTF8, "utf-8"),
    (codecs.BOM_UTF16_LE, "utf-16-le"),
    (codecs.BOM_UTF16_BE, "utf-16-be"),
)


def _utf16_without_bom(data: bytes) -> str | None:
    """"utf-16-le" or "utf-16-be" when most of the high bytes are NUL, as in
    mostly-ASCII UTF-16 text written without a byte-order mark."""
    sample = data[:4096]
    if len(sample) < 4 or b"\x00" not in sample:
        return None
    half = len(sample) // 2
    odd_nuls = sample[1::2].count(0)
    even_nuls = sample[0::2].count(0)
    if odd_nuls > 0.4 * half and even_nuls < 0.05 * half:
        return "utf-16-le"
    if even_nuls > 0.4 * half and odd_nuls < 0.05 * half:
        return "utf-16-be"
    return None


def decode_text(data: bytes) -> tuple[str, list[str]]:
    """Decode a text file's bytes and say how, as `(text, warnings)`.

    1. A UTF-8, UTF-16 or UTF-32 byte-order mark picks that encoding and is
       dropped. UTF-16 without a mark is recognised by its NUL bytes.
    2. Otherwise the bytes are read as UTF-8.
    3. Bytes that are not valid UTF-8 are read as Windows-1252, which is what
       Excel and most Windows tools write. A warning says so.
    4. When even that fails the bytes are read as UTF-8 with U+FFFD for the
       bad bytes, and a warning counts them.

    A warning is never raised: the file still parses, and the warning shows
    up in `parse_many` and fixture metadata so mangled text is not silent.
    """
    for bom, encoding in _BOMS:
        if data.startswith(bom):
            return data[len(bom):].decode(encoding, errors="replace"), []
    guessed = _utf16_without_bom(data)
    if guessed:
        return data.decode(guessed, errors="replace"), [f"no byte-order mark; decoded as {guessed.upper()}"]
    try:
        return data.decode("utf-8"), []
    except UnicodeDecodeError:
        pass
    try:
        return data.decode("cp1252"), ["text is not valid UTF-8; decoded as Windows-1252"]
    except UnicodeDecodeError:
        pass
    text = data.decode("utf-8", errors="replace")
    bad = text.count("�") - data.decode("utf-8", errors="ignore").count("�")
    return text, [f"text is not valid UTF-8; {bad} undecodable byte sequence(s) replaced with U+FFFD"]


def decode(data: bytes) -> str:
    """`decode_text` without the warnings."""
    return decode_text(data)[0]


def extract(path: Path) -> TextBuilder:
    text, warnings = decode_text(Path(path).read_bytes())
    b = TextBuilder()
    b.warnings.extend(warnings)
    lines_into(b, text)
    return b
