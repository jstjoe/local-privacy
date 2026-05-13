# Auth and server environment

The API itself is unauthenticated today — front it with whatever your deployment uses (reverse proxy, API gateway, mTLS). Per-detector credentials are read from server env vars at startup.

## Env vars

| Variable | Notes |
|---|---|
| `DEFAULT_DETECTOR` | Detector used when a request omits `detector`. Default `opf`. |
| `EAGER_LOAD` | Comma-separated detector names loaded at startup. Default equals `DEFAULT_DETECTOR`. |
| `OPF_DEVICE` | `cpu`, `cuda`, `mps`. Default `cpu`. |
| `OPF_DECODE_MODE` | `viterbi` or `argmax`. OPF-only. Default `viterbi`. |
| `SKYFLOW_VAULT_URL` / `SKYFLOW_VAULT_ID` / `SKYFLOW_BEARER_TOKEN` | Required for the `skyflow` **detector** (proxy backend). |
| `SKYFLOW_TOKEN_VAULT_URL` / `SKYFLOW_TOKEN_VAULT_ID` / `SKYFLOW_TOKEN_BEARER_TOKEN` | Required for `/v1/sanitize` `label_token` mode. See [token vault setup](../token-vault-setup.md). |

`SKYFLOW_TOKEN_BEARER_TOKEN` falls back to `SKYFLOW_BEARER_TOKEN` when unset, so a single bearer can power both the detector proxy and the token vault when they live in the same Skyflow account.

## Error codes

- `400` — unknown detector, unknown canonical category, or `label_token` mode requested without the vault env.
- `502` — detector backend errored, or token vault call failed.

Always `200` from `/v1/health` while the process is up; it does not probe detector backends.
