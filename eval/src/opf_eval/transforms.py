"""Text-transform helpers for redaction and tokenization.

Single source of truth for the rendering logic shared between the
FastAPI service (`api/src/opf_api/routes.py`) and the eval-side
notebook demo. Lives in `opf_eval` because:

- the notebook already imports `opf_eval.*`,
- the API package can import from `opf_eval` (workspace member),
- the helpers themselves are pure-Python span-splice logic with no
  FastAPI dependency.

Three renderer factories cover the five user-facing modes:

| Mode             | Factory                          | Notes                                    |
|------------------|----------------------------------|------------------------------------------|
| `bracket`        | `placeholder_renderer("bracket")`| `[EMAIL]`                                |
| `opf_native`     | `placeholder_renderer("opf_native")`| `<RAW_LABEL>` — OPF detector only.     |
| `label`          | `placeholder_renderer("bracket")`| Same shape as `bracket`; only the field name on `TokenizeResponse` differs. |
| `label_numbered` | `label_numbered_renderer()`      | Per-label counter, duplicate-aware. State is per closure. |
| `vault_token`    | `vault_token_renderer(...)`      | One batch insert against a Skyflow vault. |

`render_modes` is the convenience entry point for the notebook — it
runs every requested mode at once and returns a dict for easy display.
"""

from __future__ import annotations

from typing import Callable, Iterable, Literal, Protocol

from .detectors.base import Span


Mode = Literal["bracket", "opf_native", "label", "label_numbered", "vault_token"]


class TokenizerProtocol(Protocol):
    """Anything with a `tokenize_batch` method matching `TokenVaultClient`.

    Declared structurally so neither `opf_eval` nor the notebook needs
    to depend on `opf_api` at type-check time. The actual implementation
    lives in `opf_api.vault_tokens.TokenVaultClient`.
    """

    def tokenize_batch(
        self, items: list[tuple[str, str]]
    ) -> dict[tuple[str, str], str]: ...


def splice_spans(
    text: str,
    spans: list[Span],
    render: Callable[[Span], str],
) -> str:
    """Replace each non-overlapping span in `text` with `render(span)`.

    Spans are sorted by `(start, end)`. When two spans overlap the
    earlier-starting one wins and the later one is skipped, mirroring
    the original `_redact_text` behaviour.
    """
    if not spans:
        return text
    ordered = sorted(spans, key=lambda s: (s["start"], s["end"]))
    pieces: list[str] = []
    cursor = 0
    for s in ordered:
        if s["start"] < cursor:
            continue
        pieces.append(text[cursor : s["start"]])
        pieces.append(render(s))
        cursor = s["end"]
    pieces.append(text[cursor:])
    return "".join(pieces)


def placeholder_renderer(fmt: str) -> Callable[[Span], str]:
    """`bracket` -> `[CANONICAL_LABEL]`. `opf_native` -> `<RAW_LABEL>` (upper).

    Anything else is treated as `bracket` to keep the surface forgiving
    for the notebook (which may pass mode strings through unchanged).
    """

    def render(span: Span) -> str:
        if fmt == "opf_native":
            return f"<{span['raw_label'].upper()}>"
        return f"[{span['label']}]"

    return render


def label_numbered_renderer() -> Callable[[Span], str]:
    """Per-label counter; duplicate `(label, text)` reuses its number.

    The counter state lives in this closure, so each request / each
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


def vault_token_renderer(
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


VAULT_TOKEN_NOT_CONFIGURED = "(set SKYFLOW_TOKEN_VAULT_* to enable)"


def render_modes(
    text: str,
    spans: list[Span],
    *,
    modes: Iterable[Mode],
    detector_name: str | None = None,
    token_vault_client: TokenizerProtocol | None = None,
) -> dict[str, str]:
    """Run every requested mode at once and return `{mode: rendered_text}`.

    - `opf_native` returns `"—"` unless `detector_name == "opf"`.
    - `vault_token` returns `VAULT_TOKEN_NOT_CONFIGURED` when
      `token_vault_client is None`, and the error message on any vault
      failure (so the notebook stays running instead of bailing).
    """
    out: dict[str, str] = {}
    for mode in modes:
        if mode == "opf_native" and detector_name != "opf":
            out[mode] = "—"
            continue
        if mode == "vault_token" and token_vault_client is None:
            out[mode] = VAULT_TOKEN_NOT_CONFIGURED
            continue
        if mode in ("bracket", "label"):
            renderer = placeholder_renderer("bracket")
        elif mode == "opf_native":
            renderer = placeholder_renderer("opf_native")
        elif mode == "label_numbered":
            renderer = label_numbered_renderer()
        elif mode == "vault_token":
            try:
                renderer = vault_token_renderer(spans, token_vault_client)  # type: ignore[arg-type]
            except VaultTokenError as e:
                out[mode] = f"(vault error: {e})"
                continue
        else:
            out[mode] = f"(unknown mode: {mode})"
            continue
        out[mode] = splice_spans(text, spans, renderer)
    return out
