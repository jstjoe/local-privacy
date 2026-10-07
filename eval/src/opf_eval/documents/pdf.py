"""PDF parser (pypdf): the text layer only, one segment per page.

Pages are joined with a blank line. A page whose text layer is empty is
skipped with the warning "page N has no text layer (needs OCR)" because a
scanned page silently scoring zero PII would be worse than a visible gap.
OCR is out of scope for this harness.
"""

from __future__ import annotations

from pathlib import Path

from .base import TextBuilder, normalize_newlines

FILES_HINT = "pip install 'opf-eval[files]'"


def extract(path: Path) -> TextBuilder:
    try:
        from pypdf import PasswordType, PdfReader
    except ImportError as e:  # pragma: no cover - depends on the environment
        raise ImportError(f"Parsing .pdf files needs pypdf: {FILES_HINT}") from e

    reader = PdfReader(str(path))
    if reader.is_encrypted:
        # Many PDFs are "encrypted" with an empty user password only to set
        # permissions. Anything else cannot be read without the password.
        # pypdf reports a wrong password through the return value, not an
        # exception. Exceptions come from unsupported encryption schemes.
        try:
            result = reader.decrypt("")
        except Exception as e:
            raise ValueError(
                f"PDF is encrypted and could not be opened ({type(e).__name__}: {e})"
            ) from e
        if result == PasswordType.NOT_DECRYPTED:
            raise ValueError("PDF is encrypted and needs a password")

    b = TextBuilder()
    for n, page in enumerate(reader.pages, 1):
        try:
            text = page.extract_text() or ""
        except Exception as e:  # noqa: BLE001 — one bad page must not lose the rest
            b.warnings.append(f"page {n} could not be read ({type(e).__name__}: {e})")
            continue
        text = normalize_newlines(text).strip()
        if not text:
            b.warnings.append(f"page {n} has no text layer (needs OCR)")
            continue
        b.blank_line()
        b.add(text, {"page": n})
    return b
