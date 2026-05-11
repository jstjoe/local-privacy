# Plan 10 — Redaction + tokenization demo in the existing notebook

## Context

The `/v1/redact` and `/v1/tokenize` endpoints now support five text-transform
modes between them (`bracket`, `opf_native`, `label`, `label_numbered`,
`vault_token`). The eval side never displays the transformed text — it only
scores detection F1. We need a visual demo that walks an audience through
"what does the output look like for each mode" using the same fixtures and
detector predictions the benchmark already produces.

The demo gets appended as a new section at the end of
[notebooks/pii_detector_comparison.ipynb](notebooks/pii_detector_comparison.ipynb).
Single-notebook flow is the explicit requirement: easier to walk a room of
people through one notebook than two.

## Approach

Three pieces:

1. **Lift render logic out of `api/src/opf_api/routes.py`** into a new
   shared module so the notebook can import it without going through HTTP.
2. **Wire the existing `TokenVaultClient` into the eval side** via direct
   import. No HTTP, no duplication.
3. **Append a new "Visualize redaction + tokenization output" section**
   (cells 10.x) to the existing notebook. Reuses the already-run
   `raw_<detector>.jsonl` files so no detector re-runs.

### Piece 1 — shared transforms module

New file [eval/src/opf_eval/transforms.py](eval/src/opf_eval/transforms.py):

```python
from typing import Callable, Iterable, Protocol
from .detectors.base import Span

Mode = Literal["bracket", "opf_native", "label_numbered", "vault_token"]

class _Tokenizer(Protocol):
    def tokenize_batch(
        self, items: list[tuple[str, str]]
    ) -> dict[tuple[str, str], str]: ...

def splice_spans(text: str, spans: list[Span], render: Callable[[Span], str]) -> str: ...

def placeholder_renderer(fmt: str) -> Callable[[Span], str]:
    """bracket → [LABEL], opf_native → <RAW_LABEL>"""

def label_numbered_renderer() -> Callable[[Span], str]:
    """Per-label counter, first-appearance order; duplicate (label,text)
    reuses its number. Stateful — one per request/example."""

def vault_token_renderer(
    spans: list[Span], client: _Tokenizer
) -> Callable[[Span], str]:
    """One batch insert for unique (label,text) pairs; render uses the map."""

def render_modes(
    text: str,
    spans: list[Span],
    *,
    modes: Iterable[Mode],
    token_vault_client: _Tokenizer | None = None,
    detector_name: str | None = None,
) -> dict[str, str]:
    """One-shot helper for the notebook. Returns {mode_name: rendered_text}.
    Skips `opf_native` unless detector_name == 'opf'. Skips `vault_token`
    when client is None and emits a sentinel value the notebook can show
    as 'not configured'."""
```

Update [api/src/opf_api/routes.py](api/src/opf_api/routes.py) to import:
- `splice_spans` (was `_splice_spans`)
- `placeholder_renderer` (replaces `_placeholder_for` + the inline lambda in `_redact_text`)
- `label_numbered_renderer` (was `_build_label_numbered_renderer`)
- `vault_token_renderer` (was `_build_vault_token_renderer`)

Pure refactor — behavior identical, the existing 18 route tests must
continue to pass without modification.

`TokenVaultClient` stays in [api/src/opf_api/vault_tokens.py](api/src/opf_api/vault_tokens.py).
The eval side imports it via `from opf_api.vault_tokens import TokenVaultClient`.
Cross-package coupling is accepted: `opf-eval` already depends on `opf-api`
indirectly via the workspace, and the alternative (duplicating the client)
is worse for maintenance.

### Piece 2 — new notebook section

Append section **10** after the existing section 9 ("Saving the run") in
[notebooks/pii_detector_comparison.ipynb](notebooks/pii_detector_comparison.ipynb).

Cells (each ≤ ~25 lines of code; markdown intro before each code cell):

| # | Type | Purpose |
|---|------|---------|
| 10.0 | md | Section header + explainer: five modes, what to look for, why this matters |
| 10.1 | code | **Config**: `DEMO_N_EXAMPLES = 6`, `DEMO_MODES = ["bracket", "opf_native", "label_numbered", "vault_token"]`, `DEMO_DETECTORS = DETECTORS` (defaults to the same list the audience already ran: `presidio`, `gliner`, `gliner_nvidia`, `opf`, plus `skyflow` when creds set) |
| 10.2 | md | Optional cred cell explainer for `vault_token` |
| 10.3 | code | **Token-vault creds** — Colab secret manager style, mirrors cell 3. Reads `SKYFLOW_TOKEN_VAULT_URL`, `SKYFLOW_TOKEN_VAULT_ID`, falls back to the existing `SKYFLOW_BEARER_TOKEN`. Builds `TokenVaultClient` or sets to `None`. |
| 10.4 | code | **Load predictions**: re-read fixtures + each `raw_<detector>.jsonl` into `{id: {detector: spans}}`. Pick first `DEMO_N_EXAMPLES` from fixtures (deterministic). |
| 10.5 | code | **Render grid**: for each (example, detector) call `transforms.render_modes(...)`, build a markdown table, `display(Markdown(...))`. One block per example, one row per detector, one column per mode. |
| 10.6 | md | Mini-explainer of what the rows show |
| 10.7 | code | **Cross-example determinism check (vault_token only)**: find two demo examples that share an entity value (likely fails on small N — fall back to constructing a tiny synthetic pair if the sample doesn't have one). Run both through `vault_token` mode; assert the shared entity gets the same token. Skip with a printed notice if `TokenVaultClient` is `None`. |

#### Display format (cell 10.5 output, per example)

```
### Example 12 — pii_masking_200k

> Alice (alice@x.com) emailed Bob about the trip to Elgin, TX.

| detector | bracket | opf_native | label_numbered | vault_token |
|---|---|---|---|---|
| presidio | [PERSON] ([EMAIL]) emailed [PERSON] about the trip to [ADDRESS]. | — | [PERSON_1] ([EMAIL_1]) emailed [PERSON_2] about the trip to [ADDRESS_1]. | [PERSON_aB3xQk1] ([EMAIL_M9pZr4t]) emailed [PERSON_n7Lw2vY] about the trip to [ADDRESS_pX4Wz9j]. |
| opf      | …       | <PRIVATE_NAME> (<PRIVATE_EMAIL>) emailed <PRIVATE_NAME> about the trip to <PRIVATE_ADDRESS>. | … | … |
```

Modes the row's detector cannot supply render as `—`:
- `opf_native` is OPF-only — every other detector shows `—`.
- `vault_token` shows `(set SKYFLOW_TOKEN_VAULT_* to enable)` when the
  client is `None` rather than `—`, so the audience knows it's a config
  step not a capability gap.

### Piece 3 — tests + README

| File | Change |
|---|---|
| [eval/tests/test_transforms.py](eval/tests/test_transforms.py) | **New.** Mirror the relevant assertions from [api/tests/test_routes.py](api/tests/test_routes.py): `splice_spans` skips overlaps, `label_numbered_renderer` reuses numbers on duplicate `(label,text)`, `vault_token_renderer` calls `tokenize_batch` exactly once with the de-duplicated unique pairs, missing-label spans fall back to `[LABEL]`. Uses a fake tokenizer. |
| [api/tests/test_routes.py](api/tests/test_routes.py) | Unchanged. The existing 18 tests run against the same logic, just imported from the new location — proves the refactor is behavior-preserving. |
| [notebooks/README.md](notebooks/README.md) | Add a line in the "What this notebook does" list and a one-paragraph blurb about the new demo section. |

## Critical files

| File | Why |
|---|---|
| [api/src/opf_api/routes.py](api/src/opf_api/routes.py) | Imports change; ~50 lines of helpers get deleted (now imported from transforms). |
| [api/src/opf_api/vault_tokens.py](api/src/opf_api/vault_tokens.py) | Unchanged. Eval imports `TokenVaultClient` from here. |
| [eval/src/opf_eval/transforms.py](eval/src/opf_eval/transforms.py) | **New.** Single source of truth for splice + render. |
| [notebooks/pii_detector_comparison.ipynb](notebooks/pii_detector_comparison.ipynb) | Append section 10 with cells 10.0–10.7. |

## Reused code

- `_splice_spans` body from [api/src/opf_api/routes.py:73-89](api/src/opf_api/routes.py#L73-L89) — lifted as-is.
- `_placeholder_for` from [api/src/opf_api/routes.py:56-60](api/src/opf_api/routes.py#L56-L60) — lifted as-is.
- `_build_label_numbered_renderer` from [api/src/opf_api/routes.py:174-189](api/src/opf_api/routes.py#L174-L189) — lifted as-is.
- `_build_vault_token_renderer` from [api/src/opf_api/routes.py:192-220](api/src/opf_api/routes.py#L192-L220) — lifted, but the `HTTPException` it raises today becomes a plain `RuntimeError` (notebook context, no FastAPI). The route handler wraps the new function and re-raises as 502.
- `TokenVaultClient.from_env` from [api/src/opf_api/vault_tokens.py:67-87](api/src/opf_api/vault_tokens.py#L67-L87) — imported directly by cell 10.3.
- Fixture + raw-jsonl shapes are already exercised in cells 4–6 of the existing notebook; cell 10.4 just re-reads the files written by cell 6.

## Verification

1. `.venv/bin/pytest api/tests/test_routes.py -q` — existing 18 tests must still pass after the refactor (behavior preservation).
2. `.venv/bin/pytest eval/tests/test_transforms.py -q` — new unit tests for the lifted module.
3. **Notebook smoke run**: open the notebook, run all cells with `DETECTORS = ["presidio"]` and `DEMO_N_EXAMPLES = 3` locally (skips OPF/GLiNER downloads). Confirm the markdown table renders, `opf_native` shows `—` for presidio rows, and `vault_token` shows the config-needed sentinel when env unset.
4. **End-to-end with vault**: set `SKYFLOW_TOKEN_VAULT_URL` + `_ID` + bearer, re-run cells 10.3–10.7, confirm `[LABEL_xxxxxxx]` tokens render and the determinism check in 10.7 passes (same entity gets same token across two examples).

## Out of scope

- No metrics on transform output — this is qualitative display only.
- No reverse-tokenize / detokenize cell — separate plan.
- No per-mode latency chart — the demo is about correctness/look, not speed.
- No new Colab notebook file — strict requirement from user; append-only edit to the existing one.
- No changes to detector behavior or registry.
