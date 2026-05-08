"""Skyflow Detect API client — POST /v1/detect/deidentify/string.

Verified against a live response (the OpenAPI spec is stale on field names):
    POST {vault_url}/v1/detect/deidentify/string
    body: {"text": str, "vault_id": str, "entity_types": [...]?}
    response: {
        "processed_text": str,
        "word_count": int, "character_count": int,
        "entities": [
            {
                "entity_type": "EMAIL_ADDRESS",
                "entity_scores": {"EMAIL_ADDRESS": 0.91, ...},
                "location": {
                    "start_index": 7, "end_index": 22,
                    "start_index_processed": 0, "end_index_processed": 14
                },
                "value": "joe@example.com",
                "token": "EMAIL_ADDRESS_1"
            }
        ]
    }

Reads creds from env:
    SKYFLOW_VAULT_URL      e.g. https://<vault-id>.vault.skyflowapis.com
    SKYFLOW_VAULT_ID       vault uuid (also goes in request body)
    SKYFLOW_BEARER_TOKEN   short-lived bearer
"""

from __future__ import annotations

import os
import time

import httpx

from ..taxonomy import skyflow_to_canonical
from .base import DetectorResult, Span


DETECT_PATH = "/v1/detect/deidentify/string"


class SkyflowDetector:
    name = "skyflow"

    def __init__(
        self,
        *,
        vault_url: str | None = None,
        vault_id: str | None = None,
        bearer_token: str | None = None,
        timeout: float = 30.0,
        entity_types: list[str] | None = None,
    ) -> None:
        """
        entity_types: optional Skyflow request enum values (lowercase, e.g.
            ['email_address','phone_number']). When set, Skyflow only detects
            and returns these types — useful for apples-to-apples comparison
            with detectors that have a narrower category set.
        """
        self._vault_url = vault_url or os.environ["SKYFLOW_VAULT_URL"]
        self._vault_id = vault_id or os.environ["SKYFLOW_VAULT_ID"]
        self._bearer = bearer_token or os.environ["SKYFLOW_BEARER_TOKEN"]
        self._entity_types = entity_types
        self._client = httpx.Client(
            base_url=self._vault_url.rstrip("/"),
            timeout=timeout,
            headers={"Authorization": f"Bearer {self._bearer}"},
        )

    def detect(self, text: str, **_context: object) -> DetectorResult:
        body: dict = {"text": text, "vault_id": self._vault_id}
        if self._entity_types:
            body["entity_types"] = self._entity_types
        t0 = time.perf_counter()
        try:
            r = self._client.post(DETECT_PATH, json=body)
            r.raise_for_status()
            payload = r.json()
        except Exception as e:  # noqa: BLE001
            return {"spans": [], "latency_ms": (time.perf_counter() - t0) * 1000, "error": repr(e)}
        latency_ms = (time.perf_counter() - t0) * 1000
        return {
            "spans": list(_parse_entities(text, payload)),
            "latency_ms": latency_ms,
            "error": None,
        }

    def close(self) -> None:
        self._client.close()


def _parse_entities(text: str, payload: dict) -> list[Span]:
    """Project Skyflow's entities array into our Span shape."""
    out: list[Span] = []
    for ent in payload.get("entities", []):
        raw_label = ent.get("entity_type") or ent.get("best_label") or ""
        loc = ent.get("location") or {}
        start = loc.get("start_index", loc.get("start_idx"))
        end = loc.get("end_index", loc.get("end_idx"))
        if start is None or end is None:
            continue
        start, end = int(start), int(end)
        if end <= start:
            continue
        canonical = skyflow_to_canonical(raw_label) or raw_label.upper()
        out.append(
            {
                "label": canonical,
                "raw_label": raw_label,
                "start": start,
                "end": end,
                "text": ent.get("value") or text[start:end],
            }
        )
    return out
