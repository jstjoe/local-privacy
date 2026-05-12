# Token vault setup for `/v1/sanitize` (`label_token` mode)

The `label_token` mode in `POST /v1/sanitize` inserts each detected
entity value into a Skyflow vault and uses the deterministic
format-preserving token Skyflow returns as the replacement. Same plaintext →
same token, forever.

This vault is **separate** from any Detect API vault you may already be
using — it stores the raw entity values keyed by canonical label, and its
columns are configured to emit 7-character alphanumeric tokens.

## Schema

One table, fifteen nullable columns — one per canonical label. Each insert
sets exactly one column. Every column uses `DETERMINISTIC_FPT` with regex
`^[A-Za-z0-9]{7}$` (7-char alphanumeric, deterministic).

| Column             | Canonical label  |
|--------------------|------------------|
| `tok_person`       | PERSON           |
| `tok_email`        | EMAIL            |
| `tok_phone`        | PHONE            |
| `tok_address`      | ADDRESS          |
| `tok_url`          | URL              |
| `tok_date`         | DATE             |
| `tok_account`      | ACCOUNT          |
| `tok_secret`       | SECRET           |
| `tok_username`     | USERNAME         |
| `tok_demographic`  | DEMOGRAPHIC      |
| `tok_organization` | ORGANIZATION     |
| `tok_occupation`   | OCCUPATION       |
| `tok_money`        | MONEY            |
| `tok_vehicle`      | VEHICLE          |
| `tok_physical`     | PHYSICAL         |

`tok_` prefix avoids SQL/policy reserved-keyword collisions (`date`, `url`,
`secret`, etc.).

Table name: `entities` (override with `SKYFLOW_TOKEN_VAULT_TABLE`).

## Create the vault

Prereqs: Skyflow account, bearer token, workspace ID.

```bash
export MANAGEMENT_URL=https://manage.skyflowapis.com
export ACCOUNT_ID=<your-account-id>
export WORKSPACE_ID=<your-workspace-id>
export TOKEN=<your-bearer-token>
```

Save the following as `token-vault-schema.json`:

```json
{
  "name": "opf_token_vault",
  "description": "Detector-agnostic PII tokenization vault",
  "workspaceID": "REPLACE_WITH_WORKSPACE_ID",
  "vaultSchema": {
    "schemas": [
      {
        "name": "entities",
        "fields": [
          {
            "name": "tok_person",
            "datatype": "DT_STRING",
            "tags": [
              {"name": "skyflow.options.default_token_policy", "values": ["DETERMINISTIC_FPT"]},
              {"name": "skyflow.options.format_preserving_regex", "values": ["^[A-Za-z0-9]{7}$"]},
              {"name": "skyflow.options.default_dlp_policy", "values": ["REDACT"]},
              {"name": "skyflow.options.sensitivity", "values": ["HIGH"]}
            ]
          },
          {
            "name": "tok_email",
            "datatype": "DT_STRING",
            "tags": [
              {"name": "skyflow.options.default_token_policy", "values": ["DETERMINISTIC_FPT"]},
              {"name": "skyflow.options.format_preserving_regex", "values": ["^[A-Za-z0-9]{7}$"]},
              {"name": "skyflow.options.default_dlp_policy", "values": ["REDACT"]},
              {"name": "skyflow.options.sensitivity", "values": ["HIGH"]}
            ]
          },
          {
            "name": "tok_phone",
            "datatype": "DT_STRING",
            "tags": [
              {"name": "skyflow.options.default_token_policy", "values": ["DETERMINISTIC_FPT"]},
              {"name": "skyflow.options.format_preserving_regex", "values": ["^[A-Za-z0-9]{7}$"]},
              {"name": "skyflow.options.default_dlp_policy", "values": ["REDACT"]},
              {"name": "skyflow.options.sensitivity", "values": ["HIGH"]}
            ]
          },
          {
            "name": "tok_address",
            "datatype": "DT_STRING",
            "tags": [
              {"name": "skyflow.options.default_token_policy", "values": ["DETERMINISTIC_FPT"]},
              {"name": "skyflow.options.format_preserving_regex", "values": ["^[A-Za-z0-9]{7}$"]},
              {"name": "skyflow.options.default_dlp_policy", "values": ["REDACT"]},
              {"name": "skyflow.options.sensitivity", "values": ["HIGH"]}
            ]
          },
          {
            "name": "tok_url",
            "datatype": "DT_STRING",
            "tags": [
              {"name": "skyflow.options.default_token_policy", "values": ["DETERMINISTIC_FPT"]},
              {"name": "skyflow.options.format_preserving_regex", "values": ["^[A-Za-z0-9]{7}$"]},
              {"name": "skyflow.options.default_dlp_policy", "values": ["REDACT"]},
              {"name": "skyflow.options.sensitivity", "values": ["LOW"]}
            ]
          },
          {
            "name": "tok_date",
            "datatype": "DT_STRING",
            "tags": [
              {"name": "skyflow.options.default_token_policy", "values": ["DETERMINISTIC_FPT"]},
              {"name": "skyflow.options.format_preserving_regex", "values": ["^[A-Za-z0-9]{7}$"]},
              {"name": "skyflow.options.default_dlp_policy", "values": ["REDACT"]},
              {"name": "skyflow.options.sensitivity", "values": ["LOW"]}
            ]
          },
          {
            "name": "tok_account",
            "datatype": "DT_STRING",
            "tags": [
              {"name": "skyflow.options.default_token_policy", "values": ["DETERMINISTIC_FPT"]},
              {"name": "skyflow.options.format_preserving_regex", "values": ["^[A-Za-z0-9]{7}$"]},
              {"name": "skyflow.options.default_dlp_policy", "values": ["REDACT"]},
              {"name": "skyflow.options.sensitivity", "values": ["HIGH"]}
            ]
          },
          {
            "name": "tok_secret",
            "datatype": "DT_STRING",
            "tags": [
              {"name": "skyflow.options.default_token_policy", "values": ["DETERMINISTIC_FPT"]},
              {"name": "skyflow.options.format_preserving_regex", "values": ["^[A-Za-z0-9]{7}$"]},
              {"name": "skyflow.options.default_dlp_policy", "values": ["REDACT"]},
              {"name": "skyflow.options.sensitivity", "values": ["HIGH"]}
            ]
          },
          {
            "name": "tok_username",
            "datatype": "DT_STRING",
            "tags": [
              {"name": "skyflow.options.default_token_policy", "values": ["DETERMINISTIC_FPT"]},
              {"name": "skyflow.options.format_preserving_regex", "values": ["^[A-Za-z0-9]{7}$"]},
              {"name": "skyflow.options.default_dlp_policy", "values": ["REDACT"]},
              {"name": "skyflow.options.sensitivity", "values": ["MEDIUM"]}
            ]
          },
          {
            "name": "tok_demographic",
            "datatype": "DT_STRING",
            "tags": [
              {"name": "skyflow.options.default_token_policy", "values": ["DETERMINISTIC_FPT"]},
              {"name": "skyflow.options.format_preserving_regex", "values": ["^[A-Za-z0-9]{7}$"]},
              {"name": "skyflow.options.default_dlp_policy", "values": ["REDACT"]},
              {"name": "skyflow.options.sensitivity", "values": ["HIGH"]}
            ]
          },
          {
            "name": "tok_organization",
            "datatype": "DT_STRING",
            "tags": [
              {"name": "skyflow.options.default_token_policy", "values": ["DETERMINISTIC_FPT"]},
              {"name": "skyflow.options.format_preserving_regex", "values": ["^[A-Za-z0-9]{7}$"]},
              {"name": "skyflow.options.default_dlp_policy", "values": ["REDACT"]},
              {"name": "skyflow.options.sensitivity", "values": ["LOW"]}
            ]
          },
          {
            "name": "tok_occupation",
            "datatype": "DT_STRING",
            "tags": [
              {"name": "skyflow.options.default_token_policy", "values": ["DETERMINISTIC_FPT"]},
              {"name": "skyflow.options.format_preserving_regex", "values": ["^[A-Za-z0-9]{7}$"]},
              {"name": "skyflow.options.default_dlp_policy", "values": ["REDACT"]},
              {"name": "skyflow.options.sensitivity", "values": ["LOW"]}
            ]
          },
          {
            "name": "tok_money",
            "datatype": "DT_STRING",
            "tags": [
              {"name": "skyflow.options.default_token_policy", "values": ["DETERMINISTIC_FPT"]},
              {"name": "skyflow.options.format_preserving_regex", "values": ["^[A-Za-z0-9]{7}$"]},
              {"name": "skyflow.options.default_dlp_policy", "values": ["REDACT"]},
              {"name": "skyflow.options.sensitivity", "values": ["MEDIUM"]}
            ]
          },
          {
            "name": "tok_vehicle",
            "datatype": "DT_STRING",
            "tags": [
              {"name": "skyflow.options.default_token_policy", "values": ["DETERMINISTIC_FPT"]},
              {"name": "skyflow.options.format_preserving_regex", "values": ["^[A-Za-z0-9]{7}$"]},
              {"name": "skyflow.options.default_dlp_policy", "values": ["REDACT"]},
              {"name": "skyflow.options.sensitivity", "values": ["MEDIUM"]}
            ]
          },
          {
            "name": "tok_physical",
            "datatype": "DT_STRING",
            "tags": [
              {"name": "skyflow.options.default_token_policy", "values": ["DETERMINISTIC_FPT"]},
              {"name": "skyflow.options.format_preserving_regex", "values": ["^[A-Za-z0-9]{7}$"]},
              {"name": "skyflow.options.default_dlp_policy", "values": ["REDACT"]},
              {"name": "skyflow.options.sensitivity", "values": ["HIGH"]}
            ]
          }
        ]
      }
    ]
  }
}
```

Replace `REPLACE_WITH_WORKSPACE_ID` then create the vault:

```bash
curl -s -X POST "$MANAGEMENT_URL/v1/vaults" \
  -H "X-SKYFLOW-ACCOUNT-ID: $ACCOUNT_ID" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d @token-vault-schema.json
```

The response only returns the new vault `ID`. To find the data-plane URL,
fetch the workspace — its `url` field is the **cluster ID** hostname shared
by every vault in that workspace:

```bash
curl -s -X GET "$MANAGEMENT_URL/v1/workspaces/$WORKSPACE_ID" \
  -H "X-SKYFLOW-ACCOUNT-ID: $ACCOUNT_ID" \
  -H "Authorization: Bearer $TOKEN" | jq '.workspace.url'
# e.g. "ebfc9bee4242.vault.skyflowapis.com"
```

Use `https://<cluster-id>.vault.skyflowapis.com` for the API env var below
(not `https://<vault-id>.vault.skyflowapis.com` — the data plane is
addressed by cluster, and the vault ID lives in the request path).

## Configure the API

Set these env vars before starting `uvicorn opf_api.main:app`:

| Variable                       | Required | Notes                                                                                       |
|--------------------------------|----------|---------------------------------------------------------------------------------------------|
| `SKYFLOW_TOKEN_VAULT_URL`      | Yes      | Cluster-ID base URL, e.g. `https://ebfc9bee4242.vault.skyflowapis.com`. From `workspace.url`. |
| `SKYFLOW_TOKEN_VAULT_ID`       | Yes      | Vault UUID returned by the create call.                                                      |
| `SKYFLOW_TOKEN_BEARER_TOKEN`   | One of   | Bearer for the Data API. Falls back to `SKYFLOW_BEARER_TOKEN`.                              |
| `SKYFLOW_BEARER_TOKEN`         | One of   | Reused from the existing Skyflow detector when set.                                         |
| `SKYFLOW_TOKEN_VAULT_TABLE`    | No       | Default `entities`.                                                                          |

Without these, `/v1/sanitize` with `mode=label_token` returns 400.
With them, it returns 502 if the vault request fails for any reason.

## Smoke-test

```bash
curl -s localhost:8000/v1/sanitize -H 'content-type: application/json' -d '{
  "text": "Email alice@x.com or call +1-415-555-0100.",
  "detector": "presidio",
  "mode": "label_token"
}' | jq

# Run again — tokens for the same plaintext must match.
```
