"""Text-transform helpers for the /v1/sanitize endpoint and notebook demo.

Single source of truth for the rendering logic shared between the
FastAPI service (`api/src/opf_api/routes.py`) and the eval-side
notebook demo. Lives in `opf_eval` because:

- the notebook already imports `opf_eval.*`,
- the API package can import from `opf_eval` (workspace member),
- the helpers themselves are pure-Python span-splice logic with no
  FastAPI dependency.

Four renderer factories cover the four user-facing modes:

| Mode           | Factory                                | Notes                                              |
|----------------|----------------------------------------|----------------------------------------------------|
| `redact`       | `redact_renderer()`                    | Fixed-length `********` (no info leak).            |
| `label`        | `label_renderer()`                     | `[CANONICAL_LABEL]` (default mode).                |
| `label_number` | `label_number_renderer()`              | Per-label counter, duplicate-aware. Stateful.      |
| `label_token`  | `label_token_renderer(spans, client)`  | One batch insert against a Skyflow vault.          |

`render_modes` is the convenience entry point for the notebook — it
runs every requested mode at once and returns a dict for easy display.
"""

from __future__ import annotations

from typing import Callable, Iterable, Literal, Protocol

from .detectors.base import Span


Mode = Literal["redact", "label", "label_number", "label_token"]


# Fixed-length asterisk run used by `redact` mode. Always 8 chars
# regardless of the original span length, so nothing leaks.
REDACT_PLACEHOLDER = "*" * 8


class TokenizerProtocol(Protocol):
    """Anything with a `tokenize_batch` method matching `TokenVaultClient`.

    Declared structurally so neither `opf_eval` nor the notebook needs
    to depend on `opf_api` at type-check time. The actual implementation
    lives in `opf_api.vault_tokens.TokenVaultClient`.
    """

    def tokenize_batch(
        self, items: list[tuple[str, str]]
    ) -> dict[tuple[str, str], str]: ...


def splice_pieces(
    text: str,
    ordered_replacements: list[tuple[Span, str]],
) -> str:
    """Splice pre-rendered replacement strings into `text`.

    `ordered_replacements` must be sorted by span (start, end). Overlap
    handling matches `splice_spans`: an earlier-starting span wins, a
    later overlapping span is skipped. Use this when callers have
    already rendered each span (e.g. to avoid double-calling a renderer
    once for the response and once for the spliced text).
    """
    if not ordered_replacements:
        return text
    pieces: list[str] = []
    cursor = 0
    for s, replacement in ordered_replacements:
        if s["start"] < cursor:
            continue
        pieces.append(text[cursor : s["start"]])
        pieces.append(replacement)
        cursor = s["end"]
    pieces.append(text[cursor:])
    return "".join(pieces)


def splice_spans(
    text: str,
    spans: list[Span],
    render: Callable[[Span], str],
) -> str:
    """Replace each non-overlapping span in `text` with `render(span)`.

    Spans are sorted by `(start, end)`. When two spans overlap the
    earlier-starting one wins and the later one is skipped. Renders
    each span exactly once.
    """
    if not spans:
        return text
    ordered = sorted(spans, key=lambda s: (s["start"], s["end"]))
    return splice_pieces(text, [(s, render(s)) for s in ordered])


def redact_renderer() -> Callable[[Span], str]:
    """Always returns `REDACT_PLACEHOLDER` regardless of span."""

    def render(_span: Span) -> str:
        return REDACT_PLACEHOLDER

    return render


def label_renderer() -> Callable[[Span], str]:
    """`[CANONICAL_LABEL]`."""

    def render(span: Span) -> str:
        return f"[{span['label']}]"

    return render


def label_number_renderer() -> Callable[[Span], str]:
    """Per-label counter; duplicate `(label, text)` reuses its number.

    Counter state lives in this closure, so each request / each
    notebook example gets a fresh numbering — there is no cross-call
    leak.
    """
    counters: dict[str, int] = {}
    assigned: dict[tuple[str, str], int] = {}

    def render(span: Span) -> str:
        key = (span["label"], span["text"])
        if key not in assigned:
            counters[span["label"]] = counters.get(span["label"], 0) + 1
            assigned[key] = counters[span["label"]]
        return f"[{span['label']}_{assigned[key]}]"

    return render


class VaultTokenError(RuntimeError):
    """Raised when the vault tokenizer fails. The API route wraps this
    as a 502; the notebook can show the error message inline."""


def label_token_renderer(
    spans: list[Span],
    client: TokenizerProtocol,
) -> Callable[[Span], str]:
    """One batch insert for the unique `(label, text)` pairs in `spans`.

    Any vault failure raises `VaultTokenError`. Spans whose `(label,
    text)` did not get back a token (e.g. the canonical label isn't
    mapped to a vault column) fall back to `[LABEL]` per span — the
    response is still useful and the audience can see something is
    sensitive.
    """
    unique: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for s in spans:
        key = (s["label"], s["text"])
        if key not in seen:
            seen.add(key)
            unique.append(key)

    try:
        token_map = client.tokenize_batch(unique)
    except Exception as e:  # noqa: BLE001
        raise VaultTokenError(repr(e)) from e

    def render(span: Span) -> str:
        key = (span["label"], span["text"])
        tok = token_map.get(key)
        if tok is None:
            return f"[{span['label']}]"
        return f"[{span['label']}_{tok}]"

    return render


LABEL_TOKEN_NOT_CONFIGURED = "(set SKYFLOW_TOKEN_VAULT_* to enable)"


def render_modes(
    text: str,
    spans: list[Span],
    *,
    modes: Iterable[Mode],
    token_vault_client: TokenizerProtocol | None = None,
) -> dict[str, str]:
    """Run every requested mode at once and return `{mode: rendered_text}`.

    `label_token` returns `LABEL_TOKEN_NOT_CONFIGURED` when
    `token_vault_client is None`, and the error message inline on any
    vault failure (so the notebook stays running instead of bailing).
    """
    out: dict[str, str] = {}
    for mode in modes:
        if mode == "label_token" and token_vault_client is None:
            out[mode] = LABEL_TOKEN_NOT_CONFIGURED
            continue
        if mode == "redact":
            renderer = redact_renderer()
        elif mode == "label":
            renderer = label_renderer()
        elif mode == "label_number":
            renderer = label_number_renderer()
        elif mode == "label_token":
            try:
                renderer = label_token_renderer(spans, token_vault_client)  # type: ignore[arg-type]
            except VaultTokenError as e:
                out[mode] = f"(vault error: {e})"
                continue
        else:
            out[mode] = f"(unknown mode: {mode})"
            continue
        out[mode] = splice_spans(text, spans, renderer)
    return out
