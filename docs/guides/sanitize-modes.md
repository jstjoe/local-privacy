# Sanitization modes

`POST /v1/sanitize` rewrites each detected span under the chosen `mode`. Four modes, in increasing strength of identity preservation:

| `mode` | Looks like | What it preserves |
|---|---|---|
| `redact` | `********` | Nothing — fixed 8-character asterisk run regardless of span length. |
| `label` | `[EMAIL]` | Category only. Default mode. |
| `label_number` | `[EMAIL_1]` | Identity **within one request**. Per-label counter; duplicate `(label, text)` reuses its number. |
| `label_token` | `[EMAIL_MGaE1Bo]` | Identity **across requests and detectors** via a Skyflow vault. Deterministic — same plaintext maps to the same 7-char token forever. |

## Overlapping spans

The earlier-starting span wins. Later overlaps are skipped in `sanitized_text` but still appear in `detected_spans`.

## `label_token` requirements

This mode requires three env vars on the server:

- `SKYFLOW_TOKEN_VAULT_URL`
- `SKYFLOW_TOKEN_VAULT_ID`
- A bearer token: `SKYFLOW_TOKEN_BEARER_TOKEN`, falling back to `SKYFLOW_BEARER_TOKEN`.

The vault must be configured per the [token vault setup guide](../token-vault-setup.md): one table with one `tok_<label>` column per canonical label, each `DETERMINISTIC_FPT` with regex `^[A-Za-z0-9]{7}$`.

Spans whose canonical label has no matching vault column fall back to `[LABEL]` for that span only — the rest of the response is unchanged.

A request that asks for `label_token` against a server without the vault env returns `400` with a remediation message in `detail`.
