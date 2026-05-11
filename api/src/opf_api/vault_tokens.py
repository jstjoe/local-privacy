"""Skyflow Data API client for detector-agnostic tokenization.

Distinct from `opf_eval.detectors.skyflow`, which uses the Detect API. This
module inserts already-detected entity values into a vault configured with
deterministic format-preserving tokens (7-char alphanumeric), and returns
the tokens back to the /v1/tokenize endpoint.

Vault schema (one row per insert; only one column populated per row):

    table: entities  (configurable via SKYFLOW_TOKEN_VAULT_TABLE)
    columns: tok_person, tok_email, tok_phone, tok_address, tok_url,
             tok_date, tok_account, tok_secret, tok_username, tok_demographic,
             tok_organization, tok_occupation, tok_money, tok_vehicle,
             tok_physical
    each column: DETERMINISTIC_FPT, format regex ^[A-Za-z0-9]{7}$

Env:
    SKYFLOW_TOKEN_VAULT_URL    cluster-ID base URL, e.g.
                               https://<cluster-id>.vault.skyflowapis.com
                               (workspace.url, NOT vault-id-based)
    SKYFLOW_TOKEN_VAULT_ID     vault uuid
    SKYFLOW_TOKEN_VAULT_TABLE  default "entities"
    SKYFLOW_TOKEN_BEARER_TOKEN falls back to SKYFLOW_BEARER_TOKEN
"""

from __future__ import annotations

import logging
import os

import httpx

from opf_eval.taxonomy import CANONICAL_LABELS


logger = logging.getLogger("opf_api.vault_tokens")


def column_for_label(canonical_label: str) -> str | None:
    """Map a canonical label (e.g. 'EMAIL') to its vault column ('tok_email').

    Returns None for unknown labels; callers should fall back rather than
    block tokenization on a single unmapped span.
    """
    if canonical_label not in CANONICAL_LABELS:
        return None
    return f"tok_{canonical_label.lower()}"


class TokenVaultClient:
    """Inserts entity values into the token vault and returns deterministic tokens.

    Single batch call per /v1/tokenize request. One record per unique
    (label, value) pair; the corresponding `tok_<label>` field carries the
    plaintext, all other fields omitted.
    """

    def __init__(
        self,
        *,
        vault_url: str,
        vault_id: str,
        table: str,
        bearer_token: str,
        timeout: float = 30.0,
    ) -> None:
        self._vault_id = vault_id
        self._table = table
        self._client = httpx.Client(
            base_url=vault_url.rstrip("/"),
            timeout=timeout,
            headers={
                "Authorization": f"Bearer {bearer_token}",
                "Content-Type": "application/json",
            },
        )

    @classmethod
    def from_env(cls) -> "TokenVaultClient | None":
        """Build from SKYFLOW_TOKEN_VAULT_* env vars. Returns None when any
        required value (URL, vault ID, bearer token) is missing — treating a
        partial config as "not configured" so callers get a clear 400
        ('vault not configured') instead of a 502 from an obviously-broken
        client at request time."""
        url = os.environ.get("SKYFLOW_TOKEN_VAULT_URL")
        vault_id = os.environ.get("SKYFLOW_TOKEN_VAULT_ID")
        if not url or not vault_id:
            return None
        bearer = (
            os.environ.get("SKYFLOW_TOKEN_BEARER_TOKEN")
            or os.environ.get("SKYFLOW_BEARER_TOKEN")
        )
        if not bearer:
            logger.warning(
                "SKYFLOW_TOKEN_VAULT_URL/ID set but neither "
                "SKYFLOW_TOKEN_BEARER_TOKEN nor SKYFLOW_BEARER_TOKEN is set; "
                "treating token vault as unconfigured",
            )
            return None
        table = os.environ.get("SKYFLOW_TOKEN_VAULT_TABLE", "entities")
        return cls(vault_url=url, vault_id=vault_id, table=table, bearer_token=bearer)

    def tokenize_batch(
        self, items: list[tuple[str, str]]
    ) -> dict[tuple[str, str], str]:
        """Tokenize a batch of (canonical_label, plaintext) pairs.

        Returns a dict keyed by the same tuples; missing entries indicate
        unmapped labels or vault-side issues for that record. Raises
        httpx.HTTPError / RuntimeError on transport or HTTP failure.
        """
        if not items:
            return {}

        records: list[dict] = []
        index_to_key: list[tuple[str, str] | None] = []
        for label, value in items:
            column = column_for_label(label)
            if column is None:
                index_to_key.append(None)
                continue
            records.append({"fields": {column: value}})
            index_to_key.append((label, value))

        if not records:
            return {}

        body = {"records": records, "tokenization": True}
        path = f"/v1/vaults/{self._vault_id}/{self._table}"
        r = self._client.post(path, json=body)
        if r.status_code >= 400:
            raise RuntimeError(
                f"token vault insert failed: HTTP {r.status_code}: {r.text[:500]}"
            )
        payload = r.json()

        out: dict[tuple[str, str], str] = {}
        response_records = payload.get("records") or []
        # Filter index_to_key to the entries we actually sent (drop Nones).
        sent_keys = [k for k in index_to_key if k is not None]
        for i, rec in enumerate(response_records):
            if i >= len(sent_keys):
                break
            key = sent_keys[i]
            tokens = (rec or {}).get("tokens") or {}
            column = column_for_label(key[0])
            if column is None:
                continue
            tok = tokens.get(column)
            if isinstance(tok, str) and tok:
                out[key] = tok
        return out

    def close(self) -> None:
        self._client.close()
