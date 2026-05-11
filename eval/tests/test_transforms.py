"""Unit tests for `opf_eval.transforms`.

Mirrors the relevant assertions from `api/tests/test_routes.py` but
exercises the pure-Python helpers directly — no FastAPI in the loop.
The route tests stay as the integration cover.
"""

from __future__ import annotations

import pytest

from opf_eval.transforms import (
    VAULT_TOKEN_NOT_CONFIGURED,
    VaultTokenError,
    label_numbered_renderer,
    placeholder_renderer,
    render_modes,
    splice_spans,
    vault_token_renderer,
)


def _span(label, text, start, end, raw_label=None):
    return {
        "label": label,
        "raw_label": raw_label or label,
        "start": start,
        "end": end,
        "text": text,
    }


class FakeVaultClient:
    def __init__(self, *, raise_on_call: bool = False):
        self._raise = raise_on_call
        self.calls: list[list[tuple[str, str]]] = []

    def tokenize_batch(self, items):
        self.calls.append(list(items))
        if self._raise:
            raise RuntimeError("vault unreachable")
        seen: dict[str, int] = {}
        out: dict[tuple[str, str], str] = {}
        for label, value in items:
            seen[label] = seen.get(label, 0) + 1
            out[(label, value)] = f"T{label[:1]}{seen[label]:05d}"
        return out


# --- splice_spans -----------------------------------------------------------


def test_splice_spans_empty():
    assert splice_spans("hello", [], lambda _s: "X") == "hello"


def test_splice_spans_replaces_non_overlapping():
    text = "Joe at joe@example.com lives in Elgin, TX."
    spans = [
        _span("EMAIL", "joe@example.com", 7, 22),
        _span("ADDRESS", "Elgin, TX", 32, 41),
    ]
    out = splice_spans(text, spans, placeholder_renderer("bracket"))
    assert out == "Joe at [EMAIL] lives in [ADDRESS]."


def test_splice_spans_skips_overlap_keeps_earlier():
    spans = [
        _span("A", "joe@x.com", 0, 9),
        _span("B", "@x.com", 3, 9),  # overlaps with the first
    ]
    out = splice_spans(
        "joe@x.com tail", spans, placeholder_renderer("bracket")
    )
    assert out == "[A] tail"


# --- placeholder_renderer --------------------------------------------------


def test_placeholder_renderer_bracket():
    r = placeholder_renderer("bracket")
    assert r(_span("EMAIL", "x@y.com", 0, 7)) == "[EMAIL]"


def test_placeholder_renderer_opf_native_uppercases_raw_label():
    r = placeholder_renderer("opf_native")
    span = _span("EMAIL", "x@y.com", 0, 7, raw_label="private_email")
    assert r(span) == "<PRIVATE_EMAIL>"


# --- label_numbered_renderer ----------------------------------------------


def test_label_numbered_duplicate_reuses_number():
    r = label_numbered_renderer()
    s1 = _span("EMAIL", "a@x.com", 0, 7)
    s2 = _span("EMAIL", "b@x.com", 9, 16)
    s3 = _span("EMAIL", "a@x.com", 18, 25)  # same value as s1
    assert r(s1) == "[EMAIL_1]"
    assert r(s2) == "[EMAIL_2]"
    assert r(s3) == "[EMAIL_1]"  # duplicate -> same number


def test_label_numbered_independent_counters_per_label():
    r = label_numbered_renderer()
    assert r(_span("PERSON", "Alice", 0, 5)) == "[PERSON_1]"
    assert r(_span("EMAIL", "a@x.com", 6, 13)) == "[EMAIL_1]"
    assert r(_span("PERSON", "Bob", 14, 17)) == "[PERSON_2]"


def test_label_numbered_state_is_per_closure():
    r1 = label_numbered_renderer()
    r2 = label_numbered_renderer()
    assert r1(_span("PERSON", "Alice", 0, 5)) == "[PERSON_1]"
    # New closure starts fresh — no cross-request leak.
    assert r2(_span("PERSON", "Bob", 0, 3)) == "[PERSON_1]"


# --- vault_token_renderer --------------------------------------------------


def test_vault_token_dedupes_batch_call():
    client = FakeVaultClient()
    spans = [
        _span("EMAIL", "a@x.com", 0, 7),
        _span("EMAIL", "a@x.com", 9, 16),  # duplicate
        _span("PERSON", "Alice", 18, 23),
    ]
    vault_token_renderer(spans, client)
    assert len(client.calls) == 1
    assert client.calls[0] == [("EMAIL", "a@x.com"), ("PERSON", "Alice")]


def test_vault_token_renderer_falls_back_on_missing_token():
    class HalfClient:
        def tokenize_batch(self, items):
            # Only return a token for the first item.
            return {items[0]: "ABCDEFG"}

    spans = [
        _span("EMAIL", "a@x.com", 0, 7),
        _span("PHONE", "+1-555", 9, 15),
    ]
    r = vault_token_renderer(spans, HalfClient())
    assert r(spans[0]) == "[EMAIL_ABCDEFG]"
    assert r(spans[1]) == "[PHONE]"  # fallback


def test_vault_token_renderer_wraps_errors():
    client = FakeVaultClient(raise_on_call=True)
    with pytest.raises(VaultTokenError):
        vault_token_renderer([_span("EMAIL", "a@x.com", 0, 7)], client)


# --- render_modes ----------------------------------------------------------


def test_render_modes_bracket_and_label_numbered():
    text = "Joe at joe@example.com lives in Elgin, TX."
    spans = [
        _span("EMAIL", "joe@example.com", 7, 22),
        _span("ADDRESS", "Elgin, TX", 32, 41),
    ]
    out = render_modes(text, spans, modes=["bracket", "label_numbered"])
    assert out["bracket"] == "Joe at [EMAIL] lives in [ADDRESS]."
    assert out["label_numbered"] == "Joe at [EMAIL_1] lives in [ADDRESS_1]."


def test_render_modes_opf_native_skipped_for_non_opf():
    out = render_modes(
        "x", [_span("EMAIL", "x", 0, 1)],
        modes=["opf_native"], detector_name="presidio",
    )
    assert out["opf_native"] == "—"


def test_render_modes_opf_native_active_for_opf():
    out = render_modes(
        "x", [_span("EMAIL", "x", 0, 1, raw_label="private_email")],
        modes=["opf_native"], detector_name="opf",
    )
    assert out["opf_native"] == "<PRIVATE_EMAIL>"


def test_render_modes_vault_token_not_configured():
    out = render_modes(
        "x", [_span("EMAIL", "x", 0, 1)],
        modes=["vault_token"], token_vault_client=None,
    )
    assert out["vault_token"] == VAULT_TOKEN_NOT_CONFIGURED


def test_render_modes_vault_token_with_client():
    text = "Joe at joe@example.com."
    spans = [_span("EMAIL", "joe@example.com", 7, 22)]
    out = render_modes(
        text, spans,
        modes=["vault_token"], token_vault_client=FakeVaultClient(),
    )
    assert out["vault_token"] == "Joe at [EMAIL_TE00001]."


def test_render_modes_vault_token_error_inline():
    out = render_modes(
        "x", [_span("EMAIL", "x", 0, 1)],
        modes=["vault_token"],
        token_vault_client=FakeVaultClient(raise_on_call=True),
    )
    assert out["vault_token"].startswith("(vault error:")
