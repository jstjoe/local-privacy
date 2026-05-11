"""End-to-end route tests using a stub detector — avoids loading real models.

The tests inject a deterministic FakeDetector into the registry and exercise
the /v1 surface plus the legacy compat shim's plumbing (without OPF model load).
"""

from __future__ import annotations

import asyncio

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from opf_api.registry import DetectorEntry
from opf_api.routes import router


class FakeDetector:
    name = "fake"

    def detect(self, text: str, **_):
        spans = []
        # Match `joe@example.com` for EMAIL.
        idx = text.find("joe@example.com")
        if idx != -1:
            spans.append(
                {
                    "label": "EMAIL",
                    "raw_label": "EMAIL_ADDRESS",
                    "start": idx,
                    "end": idx + len("joe@example.com"),
                    "text": "joe@example.com",
                }
            )
        # Match `Elgin, TX` for ADDRESS.
        idx = text.find("Elgin, TX")
        if idx != -1:
            spans.append(
                {
                    "label": "ADDRESS",
                    "raw_label": "LOCATION_ADDRESS",
                    "start": idx,
                    "end": idx + len("Elgin, TX"),
                    "text": "Elgin, TX",
                }
            )
        return {"spans": spans, "latency_ms": 0.1, "error": None}


class MultiEntityDetector:
    """Detector that returns every occurrence of a fixed entity dictionary.

    Used to exercise label_numbered counters with duplicates and multi-label
    sequences. `entities` is a list of (label, raw_label, text) tuples.
    """

    name = "multi"

    def __init__(self, entities):
        self._entities = entities

    def detect(self, text: str, **_):
        spans = []
        for label, raw_label, value in self._entities:
            start = 0
            while True:
                idx = text.find(value, start)
                if idx == -1:
                    break
                spans.append(
                    {
                        "label": label,
                        "raw_label": raw_label,
                        "start": idx,
                        "end": idx + len(value),
                        "text": value,
                    }
                )
                start = idx + len(value)
        return {"spans": spans, "latency_ms": 0.1, "error": None}


class FakeTokenVaultClient:
    """Drop-in for TokenVaultClient — assigns deterministic fake tokens
    based on first-seen order of unique (label, value) pairs.
    """

    def __init__(self, *, raise_on_call: bool = False):
        self._raise = raise_on_call
        self.calls: list[list[tuple[str, str]]] = []

    def tokenize_batch(self, items):
        self.calls.append(list(items))
        if self._raise:
            raise RuntimeError("vault unreachable")
        out = {}
        seen: dict[str, int] = {}
        for label, value in items:
            seen[label] = seen.get(label, 0) + 1
            # Pad to mimic 7-char alphanumeric.
            out[(label, value)] = f"T{label[:1]}{seen[label]:05d}"
        return out

    def close(self):
        pass


def _build_app(
    *,
    detector_instance=None,
    token_vault_client=None,
    extra_detectors=None,
) -> FastAPI:
    app = FastAPI()
    app.include_router(router, prefix="/v1")
    instance = detector_instance or FakeDetector()
    fake_entry = DetectorEntry(name="fake", factory=lambda: instance)
    fake_entry.instance = instance
    registry = {"fake": fake_entry}
    for name, inst in (extra_detectors or {}).items():
        entry = DetectorEntry(name=name, factory=lambda inst=inst: inst)
        entry.instance = inst
        registry[name] = entry
    app.state.registry = registry
    app.state.default_detector = "fake"
    app.state.schema_version = 1
    app.state.token_vault_client = token_vault_client
    return app


@pytest.fixture
def client():
    app = _build_app()
    transport = ASGITransport(app=app)
    return AsyncClient(transport=transport, base_url="http://test")


@pytest.mark.asyncio
async def test_redact_default(client: AsyncClient):
    async with client as c:
        r = await c.post("/v1/redact", json={"text": "Joe at joe@example.com lives in Elgin, TX."})
    assert r.status_code == 200
    body = r.json()
    assert body["detector"] == "fake"
    assert body["redacted_text"] == "Joe at [EMAIL] lives in [ADDRESS]."
    assert body["summary"] == {"span_count": 2, "by_label": {"EMAIL": 1, "ADDRESS": 1}}
    assert {s["label"] for s in body["detected_spans"]} == {"EMAIL", "ADDRESS"}


@pytest.mark.asyncio
async def test_redact_filter_categories(client: AsyncClient):
    async with client as c:
        r = await c.post(
            "/v1/redact",
            json={
                "text": "Joe at joe@example.com lives in Elgin, TX.",
                "categories": ["EMAIL"],
            },
        )
    assert r.status_code == 200
    body = r.json()
    assert body["redacted_text"] == "Joe at [EMAIL] lives in Elgin, TX."
    assert [s["label"] for s in body["detected_spans"]] == ["EMAIL"]


@pytest.mark.asyncio
async def test_redact_invalid_category(client: AsyncClient):
    async with client as c:
        r = await c.post(
            "/v1/redact",
            json={"text": "x", "categories": ["NOT_REAL"]},
        )
    assert r.status_code == 400
    assert "NOT_REAL" in r.json()["detail"]


@pytest.mark.asyncio
async def test_unknown_detector(client: AsyncClient):
    async with client as c:
        r = await c.post("/v1/redact", json={"text": "x", "detector": "ghost"})
    assert r.status_code == 400


@pytest.mark.asyncio
async def test_opf_native_rejected_for_non_opf(client: AsyncClient):
    async with client as c:
        r = await c.post(
            "/v1/redact",
            json={"text": "x", "placeholder_format": "opf_native"},
        )
    assert r.status_code == 400


@pytest.mark.asyncio
async def test_detect_no_redacted_text(client: AsyncClient):
    async with client as c:
        r = await c.post("/v1/detect", json={"text": "joe@example.com"})
    body = r.json()
    assert "redacted_text" not in body
    assert body["summary"]["span_count"] == 1


@pytest.mark.asyncio
async def test_list_detectors(client: AsyncClient):
    async with client as c:
        r = await c.get("/v1/detectors")
    body = r.json()
    assert body["default"] == "fake"
    assert body["detectors"][0]["name"] == "fake"
    assert body["detectors"][0]["loaded"] is True


@pytest.mark.asyncio
async def test_health(client: AsyncClient):
    async with client as c:
        r = await c.get("/v1/health")
    body = r.json()
    assert body["status"] == "ok"
    assert body["default_detector"] == "fake"
    assert "fake" in body["loaded_detectors"]


# --- /v1/tokenize -----------------------------------------------------------


def _tokenize_client(**kwargs):
    app = _build_app(**kwargs)
    transport = ASGITransport(app=app)
    return AsyncClient(transport=transport, base_url="http://test")


@pytest.mark.asyncio
async def test_tokenize_label_format():
    async with _tokenize_client() as c:
        r = await c.post(
            "/v1/tokenize",
            json={
                "text": "Joe at joe@example.com lives in Elgin, TX.",
                "token_format": "label",
            },
        )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["tokenized_text"] == "Joe at [EMAIL] lives in [ADDRESS]."
    tokens = {s["label"]: s["token"] for s in body["detected_spans"]}
    assert tokens == {"EMAIL": "[EMAIL]", "ADDRESS": "[ADDRESS]"}


@pytest.mark.asyncio
async def test_tokenize_label_numbered_duplicate_reuses_number():
    detector = MultiEntityDetector(
        [
            ("EMAIL", "EMAIL_ADDRESS", "alice@x.com"),
            ("EMAIL", "EMAIL_ADDRESS", "bob@x.com"),
            ("PERSON", "PERSON_NAME", "Alice"),
            ("PERSON", "PERSON_NAME", "Bob"),
        ]
    )
    text = "Email Alice (alice@x.com) and Bob (bob@x.com). Alice will reply from alice@x.com."
    async with _tokenize_client(detector_instance=detector) as c:
        r = await c.post(
            "/v1/tokenize",
            json={"text": text, "token_format": "label_numbered"},
        )
    assert r.status_code == 200, r.text
    body = r.json()
    out = body["tokenized_text"]
    # Alice/Bob get PERSON_1/PERSON_2 by first-appearance; their emails get
    # EMAIL_1/EMAIL_2; the repeated alice@x.com reuses EMAIL_1.
    assert (
        out
        == "Email [PERSON_1] ([EMAIL_1]) and [PERSON_2] ([EMAIL_2]). "
        "[PERSON_1] will reply from [EMAIL_1]."
    )


@pytest.mark.asyncio
async def test_tokenize_label_numbered_independent_counters():
    """Per-label counters increment independently."""
    detector = MultiEntityDetector(
        [
            ("EMAIL", "EMAIL_ADDRESS", "a@x.com"),
            ("EMAIL", "EMAIL_ADDRESS", "b@x.com"),
            ("PERSON", "PERSON_NAME", "Alice"),
        ]
    )
    text = "Alice has a@x.com and b@x.com."
    async with _tokenize_client(detector_instance=detector) as c:
        r = await c.post(
            "/v1/tokenize",
            json={"text": text, "token_format": "label_numbered"},
        )
    body = r.json()
    assert body["tokenized_text"] == "[PERSON_1] has [EMAIL_1] and [EMAIL_2]."


@pytest.mark.asyncio
async def test_tokenize_vault_token_format():
    vault = FakeTokenVaultClient()
    async with _tokenize_client(token_vault_client=vault) as c:
        r = await c.post(
            "/v1/tokenize",
            json={
                "text": "Joe at joe@example.com lives in Elgin, TX.",
                "token_format": "vault_token",
            },
        )
    assert r.status_code == 200, r.text
    body = r.json()
    # Tokens use the fake client's TE/TA prefix scheme with 5-digit suffix.
    assert "[EMAIL_TE00001]" in body["tokenized_text"]
    assert "[ADDRESS_TA00001]" in body["tokenized_text"]
    # Exactly one batch call with the two unique (label, value) pairs.
    assert len(vault.calls) == 1
    assert set(vault.calls[0]) == {
        ("EMAIL", "joe@example.com"),
        ("ADDRESS", "Elgin, TX"),
    }


@pytest.mark.asyncio
async def test_tokenize_vault_token_deduplicates_batch():
    """Duplicate (label, value) pairs send one record, not many."""
    detector = MultiEntityDetector(
        [("EMAIL", "EMAIL_ADDRESS", "a@x.com")],
    )
    vault = FakeTokenVaultClient()
    async with _tokenize_client(
        detector_instance=detector, token_vault_client=vault
    ) as c:
        r = await c.post(
            "/v1/tokenize",
            json={
                "text": "a@x.com then a@x.com again",
                "token_format": "vault_token",
            },
        )
    assert r.status_code == 200
    assert vault.calls == [[("EMAIL", "a@x.com")]]
    body = r.json()
    # Same plaintext → same token, both occurrences spliced.
    assert body["tokenized_text"].count("[EMAIL_TE00001]") == 2


@pytest.mark.asyncio
async def test_tokenize_vault_token_unconfigured_400():
    async with _tokenize_client(token_vault_client=None) as c:
        r = await c.post(
            "/v1/tokenize",
            json={"text": "joe@example.com", "token_format": "vault_token"},
        )
    assert r.status_code == 400
    assert "SKYFLOW_TOKEN_VAULT" in r.json()["detail"]


@pytest.mark.asyncio
async def test_tokenize_vault_token_failure_502():
    vault = FakeTokenVaultClient(raise_on_call=True)
    async with _tokenize_client(token_vault_client=vault) as c:
        r = await c.post(
            "/v1/tokenize",
            json={"text": "joe@example.com", "token_format": "vault_token"},
        )
    assert r.status_code == 502
    assert "vault_token" in r.json()["detail"]


@pytest.mark.asyncio
async def test_tokenize_filter_categories():
    async with _tokenize_client() as c:
        r = await c.post(
            "/v1/tokenize",
            json={
                "text": "Joe at joe@example.com lives in Elgin, TX.",
                "token_format": "label_numbered",
                "categories": ["EMAIL"],
            },
        )
    body = r.json()
    assert body["tokenized_text"] == "Joe at [EMAIL_1] lives in Elgin, TX."
    assert [s["label"] for s in body["detected_spans"]] == ["EMAIL"]


@pytest.mark.asyncio
async def test_tokenize_unknown_detector():
    async with _tokenize_client() as c:
        r = await c.post(
            "/v1/tokenize",
            json={"text": "x", "detector": "ghost", "token_format": "label"},
        )
    assert r.status_code == 400


@pytest.mark.asyncio
async def test_tokenize_no_spans_passthrough():
    async with _tokenize_client() as c:
        r = await c.post(
            "/v1/tokenize",
            json={"text": "nothing to detect here", "token_format": "label_numbered"},
        )
    body = r.json()
    assert body["tokenized_text"] == "nothing to detect here"
    assert body["detected_spans"] == []
    assert body["summary"] == {"span_count": 0, "by_label": {}}
