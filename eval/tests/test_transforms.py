"""Unit tests for `opf_eval.transforms`.

Mirrors the relevant assertions from `api/tests/test_routes.py` but
exercises the pure-Python helpers directly — no FastAPI in the loop.
"""

from __future__ import annotations

import pytest

from opf_eval.transforms import (
    LABEL_TOKEN_NOT_CONFIGURED,
    REDACT_PLACEHOLDER,
    VaultTokenError,
    label_number_renderer,
    label_renderer,
    label_token_renderer,
    redact_renderer,
    render_modes,
    splice_pieces,
    splice_spans,
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


# --- splice_spans ---------------------------------------------------------


def test_splice_spans_empty():
    assert splice_spans("hello", [], lambda _s: "X") == "hello"


def test_splice_spans_replaces_non_overlapping():
    text = "Joe at joe@example.com lives in Elgin, TX."
    spans = [
        _span("EMAIL", "joe@example.com", 7, 22),
        _span("ADDRESS", "Elgin, TX", 32, 41),
    ]
    out = splice_spans(text, spans, label_renderer())
    assert out == "Joe at [EMAIL] lives in [ADDRESS]."


def test_splice_spans_skips_overlap_keeps_earlier():
    spans = [
        _span("A", "joe@x.com", 0, 9),
        _span("B", "@x.com", 3, 9),
    ]
    out = splice_spans("joe@x.com tail", spans, label_renderer())
    assert out == "[A] tail"


def test_splice_pieces_uses_pre_rendered_strings():
    """splice_pieces lets callers avoid calling the renderer twice — proves
    callers can use a non-idempotent renderer (e.g. one that returns a
    fresh UUID per call) without the response disagreeing with the
    spliced text. With splice_spans + a non-idempotent renderer those
    would diverge."""
    text = "Joe at joe@example.com lives in Elgin, TX."
    spans = [
        _span("EMAIL", "joe@example.com", 7, 22),
        _span("ADDRESS", "Elgin, TX", 32, 41),
    ]
    counter = {"n": 0}

    def fresh_each_call(_span):
        counter["n"] += 1
        return f"<R{counter['n']}>"

    ordered = sorted(spans, key=lambda s: (s["start"], s["end"]))
    pairs = [(s, fresh_each_call(s)) for s in ordered]  # render exactly once
    spliced = splice_pieces(text, pairs)
    assert counter["n"] == 2  # one render per span — no double-call

    # The replacement strings in pairs are exactly what ended up in the
    # spliced text — useful for the route handler reusing them for both
    # the response and the splice.
    for _, repl in pairs:
        assert repl in spliced


# --- redact_renderer ------------------------------------------------------


def test_redact_renderer_fixed_length():
    r = redact_renderer()
    short = r(_span("EMAIL", "x", 0, 1))
    long = r(_span("EMAIL", "very long email address here", 0, 28))
    assert short == long == REDACT_PLACEHOLDER == "*" * 8


# --- label_renderer ------------------------------------------------------


def test_label_renderer():
    r = label_renderer()
    assert r(_span("EMAIL", "x@y.com", 0, 7)) == "[EMAIL]"


# --- label_number_renderer ----------------------------------------------


def test_label_number_duplicate_reuses_number():
    r = label_number_renderer()
    s1 = _span("EMAIL", "a@x.com", 0, 7)
    s2 = _span("EMAIL", "b@x.com", 9, 16)
    s3 = _span("EMAIL", "a@x.com", 18, 25)
    assert r(s1) == "[EMAIL_1]"
    assert r(s2) == "[EMAIL_2]"
    assert r(s3) == "[EMAIL_1]"


def test_label_number_independent_counters_per_label():
    r = label_number_renderer()
    assert r(_span("PERSON", "Alice", 0, 5)) == "[PERSON_1]"
    assert r(_span("EMAIL", "a@x.com", 6, 13)) == "[EMAIL_1]"
    assert r(_span("PERSON", "Bob", 14, 17)) == "[PERSON_2]"


def test_label_number_state_per_closure():
    r1 = label_number_renderer()
    r2 = label_number_renderer()
    assert r1(_span("PERSON", "Alice", 0, 5)) == "[PERSON_1]"
    assert r2(_span("PERSON", "Bob", 0, 3)) == "[PERSON_1]"


# --- label_token_renderer ------------------------------------------------


def test_label_token_dedupes_batch_call():
    client = FakeVaultClient()
    spans = [
        _span("EMAIL", "a@x.com", 0, 7),
        _span("EMAIL", "a@x.com", 9, 16),
        _span("PERSON", "Alice", 18, 23),
    ]
    label_token_renderer(spans, client)
    assert len(client.calls) == 1
    assert client.calls[0] == [("EMAIL", "a@x.com"), ("PERSON", "Alice")]


def test_label_token_falls_back_on_missing_token():
    class HalfClient:
        def tokenize_batch(self, items):
            return {items[0]: "ABCDEFG"}

    spans = [
        _span("EMAIL", "a@x.com", 0, 7),
        _span("PHONE", "+1-555", 9, 15),
    ]
    r = label_token_renderer(spans, HalfClient())
    assert r(spans[0]) == "[EMAIL_ABCDEFG]"
    assert r(spans[1]) == "[PHONE]"


def test_label_token_wraps_errors():
    client = FakeVaultClient(raise_on_call=True)
    with pytest.raises(VaultTokenError):
        label_token_renderer([_span("EMAIL", "a@x.com", 0, 7)], client)


# --- render_modes --------------------------------------------------------


def test_render_modes_all_four():
    text = "Email joe@example.com today."
    spans = [_span("EMAIL", "joe@example.com", 6, 21)]
    client = FakeVaultClient()
    out = render_modes(
        text, spans,
        modes=["redact", "label", "label_number", "label_token"],
        token_vault_client=client,
    )
    assert out["redact"] == f"Email {REDACT_PLACEHOLDER} today."
    assert out["label"] == "Email [EMAIL] today."
    assert out["label_number"] == "Email [EMAIL_1] today."
    assert out["label_token"] == "Email [EMAIL_TE00001] today."


def test_render_modes_label_token_not_configured():
    out = render_modes(
        "x", [_span("EMAIL", "x", 0, 1)],
        modes=["label_token"], token_vault_client=None,
    )
    assert out["label_token"] == LABEL_TOKEN_NOT_CONFIGURED


def test_render_modes_label_token_error_inline():
    out = render_modes(
        "x", [_span("EMAIL", "x", 0, 1)],
        modes=["label_token"],
        token_vault_client=FakeVaultClient(raise_on_call=True),
    )
    assert out["label_token"].startswith("(vault error:")
