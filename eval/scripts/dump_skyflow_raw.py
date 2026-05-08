"""One-off: dump raw Skyflow Detect response to see actual field names."""

import json
import os

import httpx


text = "Joe at joe@example.com lives in Elgin, TX. Phone: 555-1234."

r = httpx.post(
    os.environ["SKYFLOW_VAULT_URL"].rstrip("/") + "/v1/detect/deidentify/string",
    headers={"Authorization": f"Bearer {os.environ['SKYFLOW_BEARER_TOKEN']}"},
    json={"text": text, "vault_id": os.environ["SKYFLOW_VAULT_ID"]},
    timeout=30,
)
r.raise_for_status()
print(json.dumps(r.json(), indent=2))
