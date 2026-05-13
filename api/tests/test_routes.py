"""End-to-end route tests using a stub detector — avoids loading real models.

The tests inject a deterministic FakeDetector into the registry and exercise
the /v1 surface for /detect, /sanitize, /detectors, /health.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from opf_api.registry import DetectorEntry
from opf_api.routes import router


class FakeDetector:
    name = "fake"

    def detect(self, text: str, **_):
        spans = []
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
    """Returns every occurrence of a fixed entity list — used to exercise
    label_number counters with duplicates and multi-label sequences."""

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
    based on first-seen order of unique (label, value) pairs."""

    def __init__(self, *, raise_on_call: bool = False):
        self._raise = raise_on_call
        self.calls: list[list[tuple[str, str]]] = []

    def tokenize_batch(self, items):
        self.calls.append(list(items))
        if self._raise:
            raise RuntimeError("vault unreachable")
        out: dict[tuple[str, str], str] = {}
        seen: dict[str, int] = {}
        for label, value in items:
            seen[label] = seen.get(label, 0) + 1
            out[(label, value)] = f"T{label[:1]}{seen[label]:05d}"
        return out

    def close(self):
        pass


def _build_app(
    *,
    detector_instance=None,
    token_vault_client=None,
) -> FastAPI:
    app = FastAPI()
    app.include_router(router, prefix="/v1")
    instance = detector_instance or FakeDetector()
    fake_entry = DetectorEntry(name="fake", factory=lambda: instance)
    fake_entry.instance = instance
    app.state.registry = {"fake": fake_entry}
    app.state.default_detector = "fake"
    app.state.token_vault_client = token_vault_client
    return app


def _client(**kwargs):
    app = _build_app(**kwargs)
    transport = ASGITransport(app=app)
    return AsyncClient(transport=transport, base_url="http://test")


# --- /v1/detect -----------------------------------------------------------


@pytest.mark.asyncio
async def test_detect_default():
    async with _client() as c:
        r = await c.post("/v1/detect", json={"text": "Joe at joe@example.com lives in Elgin, TX."})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["detector"] == "fake"
    assert "sanitized_text" not in body
    labels = {s["label"] for s in body["detected_spans"]}
    assert labels == {"EMAIL", "ADDRESS"}
    assert body["summary"] == {"span_count": 2, "by_label": {"EMAIL": 1, "ADDRESS": 1}}


@pytest.mark.asyncio
async def test_detect_filter_categories():
    async with _client() as c:
        r = await c.post(
            "/v1/detect",
            json={
                "text": "Joe at joe@example.com lives in Elgin, TX.",
                "categories": ["EMAIL"],
            },
        )
    body = r.json()
    assert [s["label"] for s in body["detected_spans"]] == ["EMAIL"]


@pytest.mark.asyncio
async def test_detect_empty_categories_returns_zero_spans():
    # `[]` is a deliberate "match nothing" filter — distinct from `None`,
    # which means "no filter". Regression guard for that contract.
    async with _client() as c:
        r = await c.post(
            "/v1/detect",
            json={
                "text": "Joe at joe@example.com lives in Elgin, TX.",
                "categories": [],
            },
        )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["detected_spans"] == []
    assert body["summary"] == {"span_count": 0, "by_label": {}}


@pytest.mark.asyncio
async def test_detect_null_categories_returns_all_spans():
    async with _client() as c:
        r = await c.post(
            "/v1/detect",
            json={"text": "Joe at joe@example.com lives in Elgin, TX.", "categories": None},
        )
    assert r.status_code == 200, r.text
    body = r.json()
    assert {s["label"] for s in body["detected_spans"]} == {"EMAIL", "ADDRESS"}


@pytest.mark.asyncio
async def test_detect_invalid_category():
    async with _client() as c:
        r = await c.post(
            "/v1/detect",
            json={"text": "x", "categories": ["NOT_REAL"]},
        )
    assert r.status_code == 422
    detail = r.json()["detail"]
    # Pydantic returns a list of per-field errors; the bad input must be there.
    assert any("NOT_REAL" == err.get("input") for err in detail), detail


@pytest.mark.asyncio
async def test_detect_unknown_opf_option_rejected():
    async with _client() as c:
        r = await c.post(
            "/v1/detect",
            json={"text": "x", "options": {"opf": {"decod_mode": "argmax"}}},
        )
    assert r.status_code == 422
    detail = r.json()["detail"]
    assert any(
        err.get("type") == "extra_forbidden" and "decod_mode" in err.get("loc", [])
        for err in detail
    ), detail


@pytest.mark.asyncio
async def test_detect_valid_opf_options_accepted():
    async with _client() as c:
        r = await c.post(
            "/v1/detect",
            json={
                "text": "Joe at joe@example.com lives in Elgin, TX.",
                "options": {"opf": {"decode_mode": "argmax"}},
            },
        )
    assert r.status_code == 200, r.text
    body = r.json()
    assert {s["label"] for s in body["detected_spans"]} == {"EMAIL", "ADDRESS"}


@pytest.mark.asyncio
async def test_detect_unknown_detector():
    async with _client() as c:
        r = await c.post("/v1/detect", json={"text": "x", "detector": "ghost"})
    assert r.status_code == 400


# --- /v1/sanitize ---------------------------------------------------------


@pytest.mark.asyncio
async def test_sanitize_default_mode_is_label():
    async with _client() as c:
        r = await c.post(
            "/v1/sanitize",
            json={"text": "Joe at joe@example.com lives in Elgin, TX."},
        )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["mode"] == "label"
    assert body["sanitized_text"] == "Joe at [EMAIL] lives in [ADDRESS]."
    replacements = {s["label"]: s["replacement"] for s in body["detected_spans"]}
    assert replacements == {"EMAIL": "[EMAIL]", "ADDRESS": "[ADDRESS]"}


@pytest.mark.asyncio
async def test_sanitize_redact_mode_fixed_length_asterisks():
    async with _client() as c:
        r = await c.post(
            "/v1/sanitize",
            json={"text": "Joe at joe@example.com lives in Elgin, TX.", "mode": "redact"},
        )
    assert r.status_code == 200
    body = r.json()
    assert body["sanitized_text"] == "Joe at ******** lives in ********."
    for s in body["detected_spans"]:
        assert s["replacement"] == "********"


@pytest.mark.asyncio
async def test_sanitize_label_number_duplicate_reuses_number():
    detector = MultiEntityDetector(
        [
            ("EMAIL", "EMAIL_ADDRESS", "alice@x.com"),
            ("EMAIL", "EMAIL_ADDRESS", "bob@x.com"),
            ("PERSON", "PERSON_NAME", "Alice"),
            ("PERSON", "PERSON_NAME", "Bob"),
        ]
    )
    text = "Email Alice (alice@x.com) and Bob (bob@x.com). Alice will reply from alice@x.com."
    async with _client(detector_instance=detector) as c:
        r = await c.post(
            "/v1/sanitize",
            json={"text": text, "mode": "label_number"},
        )
    assert r.status_code == 200, r.text
    out = r.json()["sanitized_text"]
    assert (
        out
        == "Email [PERSON_1] ([EMAIL_1]) and [PERSON_2] ([EMAIL_2]). "
        "[PERSON_1] will reply from [EMAIL_1]."
    )


@pytest.mark.asyncio
async def test_sanitize_label_number_independent_counters():
    detector = MultiEntityDetector(
        [
            ("EMAIL", "EMAIL_ADDRESS", "a@x.com"),
            ("EMAIL", "EMAIL_ADDRESS", "b@x.com"),
            ("PERSON", "PERSON_NAME", "Alice"),
        ]
    )
    text = "Alice has a@x.com and b@x.com."
    async with _client(detector_instance=detector) as c:
        r = await c.post(
            "/v1/sanitize",
            json={"text": text, "mode": "label_number"},
        )
    assert r.json()["sanitized_text"] == "[PERSON_1] has [EMAIL_1] and [EMAIL_2]."


@pytest.mark.asyncio
async def test_sanitize_label_token():
    vault = FakeTokenVaultClient()
    async with _client(token_vault_client=vault) as c:
        r = await c.post(
            "/v1/sanitize",
            json={
                "text": "Joe at joe@example.com lives in Elgin, TX.",
                "mode": "label_token",
            },
        )
    assert r.status_code == 200, r.text
    body = r.json()
    assert "[EMAIL_TE00001]" in body["sanitized_text"]
    assert "[ADDRESS_TA00001]" in body["sanitized_text"]
    assert len(vault.calls) == 1
    assert set(vault.calls[0]) == {
        ("EMAIL", "joe@example.com"),
        ("ADDRESS", "Elgin, TX"),
    }


@pytest.mark.asyncio
async def test_sanitize_label_token_dedupe_batch():
    detector = MultiEntityDetector([("EMAIL", "EMAIL_ADDRESS", "a@x.com")])
    vault = FakeTokenVaultClient()
    async with _client(detector_instance=detector, token_vault_client=vault) as c:
        r = await c.post(
            "/v1/sanitize",
            json={"text": "a@x.com then a@x.com again", "mode": "label_token"},
        )
    assert r.status_code == 200
    assert vault.calls == [[("EMAIL", "a@x.com")]]
    body = r.json()
    assert body["sanitized_text"].count("[EMAIL_TE00001]") == 2


@pytest.mark.asyncio
async def test_sanitize_label_token_unconfigured_400():
    async with _client(token_vault_client=None) as c:
        r = await c.post(
            "/v1/sanitize",
            json={"text": "joe@example.com", "mode": "label_token"},
        )
    assert r.status_code == 400
    assert "SKYFLOW_TOKEN_VAULT" in r.json()["detail"]


@pytest.mark.asyncio
async def test_sanitize_label_token_vault_failure_502():
    vault = FakeTokenVaultClient(raise_on_call=True)
    async with _client(token_vault_client=vault) as c:
        r = await c.post(
            "/v1/sanitize",
            json={"text": "joe@example.com", "mode": "label_token"},
        )
    assert r.status_code == 502
    assert "label_token" in r.json()["detail"]


@pytest.mark.asyncio
async def test_sanitize_filter_categories():
    async with _client() as c:
        r = await c.post(
            "/v1/sanitize",
            json={
                "text": "Joe at joe@example.com lives in Elgin, TX.",
                "mode": "label_number",
                "categories": ["EMAIL"],
            },
        )
    body = r.json()
    assert body["sanitized_text"] == "Joe at [EMAIL_1] lives in Elgin, TX."
    assert [s["label"] for s in body["detected_spans"]] == ["EMAIL"]


@pytest.mark.asyncio
async def test_sanitize_unknown_detector():
    async with _client() as c:
        r = await c.post(
            "/v1/sanitize",
            json={"text": "x", "detector": "ghost", "mode": "label"},
        )
    assert r.status_code == 400


@pytest.mark.asyncio
async def test_sanitize_no_spans_passthrough():
    async with _client() as c:
        r = await c.post(
            "/v1/sanitize",
            json={"text": "nothing to detect here", "mode": "label_number"},
        )
    body = r.json()
    assert body["sanitized_text"] == "nothing to detect here"
    assert body["detected_spans"] == []
    assert body["summary"] == {"span_count": 0, "by_label": {}}


# --- /v1/detectors + /v1/health ------------------------------------------


@pytest.mark.asyncio
async def test_list_detectors():
    async with _client() as c:
        r = await c.get("/v1/detectors")
    body = r.json()
    assert body["default"] == "fake"
    assert body["detectors"][0]["name"] == "fake"
    assert body["detectors"][0]["loaded"] is True


@pytest.mark.asyncio
async def test_health():
    async with _client() as c:
        r = await c.get("/v1/health")
    body = r.json()
    assert body["status"] == "ok"
    assert body["default_detector"] == "fake"
    assert "fake" in body["loaded_detectors"]
