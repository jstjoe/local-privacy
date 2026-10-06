"""EML parser (stdlib `email`): headers, body and supported attachments.

Layout of the extracted text:

    From: Jane Doe <jane@example.com>      segment covers the value only:
    To: bob@example.com                      {"part": "header", "name": "From"}
    Subject: Invoice 42

    <body lines>                            {"part": "body", "line": n}

    <attachment text>                       {"attachment": "scan.pdf", "page": 1}

Headers From, To, Cc, Bcc, Reply-To, Subject and Date are kept in that order.
The body prefers text/plain and falls back to text/html read with the HTML
rules. Attachments whose filename has a supported suffix are parsed with the
matching parser and appended after a blank line. Their locations are nested
under the attachment's filename, and an attachment inside an attached email
gets a path such as "fwd.eml/scan.pdf". Other attachments only add a warning.
"""

from __future__ import annotations

import email
import email.policy
import tempfile
from email.message import EmailMessage
from pathlib import Path, PurePath

from .base import TextBuilder, lines_into, normalize_newlines
from .html import html_to_lines
from .text import decode_text

HEADERS = ("From", "To", "Cc", "Bcc", "Reply-To", "Subject", "Date")
MAX_ATTACHMENT_DEPTH = 4


def _header_lines(b: TextBuilder, msg: EmailMessage) -> None:
    for name in HEADERS:
        for raw in msg.get_all(name) or ():
            # Folded headers carry their line breaks into str(); a header is
            # one logical line so collapse them.
            value = " ".join(str(raw).split())
            if not value:
                continue
            b.newline()
            b.literal(f"{name}: ")
            b.add(value, {"part": "header", "name": name})


def _body(b: TextBuilder, msg: EmailMessage) -> None:
    part = msg.get_body(preferencelist=("plain", "html"))
    if part is None:
        b.warnings.append("no text body")
        return
    try:
        content = part.get_content()
    except (LookupError, ValueError) as e:  # unknown charset or broken encoding
        # Some mail servers label bodies "unknown-8bit" or "x-unknown". The
        # bytes are still there, so decode them the way text files are.
        payload = part.get_payload(decode=True)
        if not isinstance(payload, bytes):
            b.warnings.append(f"body could not be decoded ({e})")
            return
        content, how = decode_text(payload)
        content = normalize_newlines(content)
        detail = how[0] if how else "decoded as UTF-8"
        b.warnings.append(f"body charset could not be used ({e}); {detail}")
    if not isinstance(content, str):
        b.warnings.append(f"body has non-text content type {part.get_content_type()}")
        return
    body = TextBuilder()
    if part.get_content_type() == "text/html":
        for n, line in enumerate(html_to_lines(content), 1):
            body.newline()
            body.add(line, {"part": "body", "line": n})
    else:
        lines_into(body, content.rstrip("\n"), where=lambda n: {"part": "body", "line": n})
    if body.text:
        b.extend(body)
    else:
        b.warnings.append("body is empty")


def _attachment_bytes(part: EmailMessage) -> bytes | None:
    if part.get_content_type() == "message/rfc822":
        inner = part.get_payload()
        if isinstance(inner, list) and inner:
            return inner[0].as_bytes(policy=email.policy.default)
        return None
    payload = part.get_payload(decode=True)
    return payload if isinstance(payload, bytes) else None


def _nest(name: str):
    def wrap(where: dict) -> dict:
        inner = where.get("attachment")
        if inner is not None:
            return {**where, "attachment": f"{name}/{inner}"}
        return {"attachment": name, **where}

    return wrap


def _attachments(b: TextBuilder, msg: EmailMessage, depth: int) -> None:
    # Imported here: the package __init__ imports this module's parent.
    from . import SUPPORTED_SUFFIXES, extract_path

    for part in msg.iter_attachments():
        ctype = part.get_content_type()
        filename = part.get_filename()
        if not filename and ctype == "message/rfc822":
            filename = "attached.eml"
        if not filename:
            b.warnings.append(f"unnamed attachment of type {ctype} skipped")
            continue
        # Never trust a filename from the message as a path.
        name = PurePath(filename.replace("\\", "/")).name or "attachment"
        suffix = PurePath(name).suffix.lower()
        if ctype == "message/rfc822" and suffix != ".eml":
            suffix = ".eml"
        if suffix not in SUPPORTED_SUFFIXES:
            b.warnings.append(f"attachment {name!r} skipped (unsupported type {ctype})")
            continue
        if depth >= MAX_ATTACHMENT_DEPTH:
            b.warnings.append(f"attachment {name!r} skipped (nested more than {MAX_ATTACHMENT_DEPTH} deep)")
            continue
        data = _attachment_bytes(part)
        if not data:
            b.warnings.append(f"attachment {name!r} is empty")
            continue
        with tempfile.TemporaryDirectory(prefix="opf-eml-") as tmp:
            tmp_path = Path(tmp) / f"attachment{suffix}"
            tmp_path.write_bytes(data)
            try:
                inner = extract_path(tmp_path, SUPPORTED_SUFFIXES[suffix], depth=depth + 1)
            except Exception as e:  # noqa: BLE001 — a broken attachment must not lose the email
                b.warnings.append(f"attachment {name!r} could not be parsed ({type(e).__name__}: {e})")
                continue
        b.warnings.extend(f"attachment {name!r}: {w}" for w in inner.warnings)
        if not inner.text:
            continue
        b.extend(inner, _nest(name))


def extract(path: Path, *, depth: int = 0) -> TextBuilder:
    msg = email.message_from_bytes(Path(path).read_bytes(), policy=email.policy.default)
    b = TextBuilder()
    _header_lines(b, msg)
    _body(b, msg)
    _attachments(b, msg, depth)
    return b
