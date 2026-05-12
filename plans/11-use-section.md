# Plan 11 — `# Use` section: searching sanitized documents

## Context

The notebook today walks through two halves:

- **Detection** (sections 1–9): find PII spans across detectors and score detection quality.
- **Sanitization** (sections 1–5 under `# Sanitization`): rewrite the detected spans in one of four modes.

What we don't show yet: **what happens to the sanitized text downstream**. A user looking at the four modes can read the side-by-side table and ask the right question — "OK, but which mode lets me still *use* the data?" — and we don't answer it.

A third H1 section (`# Use`) closes the loop. Pedagogical punchline: only `label_token` preserves **referential integrity across documents**. When the same plaintext appears in two different inputs, only `label_token` ensures it maps to the same replacement string in both, which is the property that makes downstream search / join / RAG actually work on sanitized text.

The demo: build a tiny corpus where known PII values recur across docs, sanitize it under each of the four modes, sanitize a natural-language query the same way, then run BM25 retrieval per mode and compare the top-k results.

## Approach

### Section structure

Insert a new H1 at the end of the notebook (after the existing `# Sanitization` block):

```
# Use

(intro: "detection finds, sanitization rewrites, but downstream tools
operate on the rewritten text — search / retrieval / RAG / joins. The
mode you pick determines what's still possible.")

## 1. Configure the search demo
   (md + code: DEMO_USE_CORPUS, DEMO_USE_QUERIES, DEMO_USE_TOP_K,
    DEMO_USE_DETECTOR. Curated 5–10 docs with controlled PII overlap.
    Defaults: ~6 docs, 2 queries, top_k=3, detector=presidio.)

## 2. Sanitize the corpus and queries under every mode
   (md + code: for each of the four modes + a "plain" baseline,
    sanitize every doc and every query. Cache per-mode results into
    a `corpus_by_mode` / `queries_by_mode` structure. label_token
    requires the vault — gracefully skip with a notice if unconfigured.)

## 3. Index each sanitized corpus and run each sanitized query
   (md + code: build one BM25 index per (corpus_mode). Run each
    sanitized query against its same-mode index — must match.
    rank-bm25 installed inline via pip in cell 1's setup.)

## 4. Compare retrieval across modes
   (md + code: one markdown table per query. Rows = modes (including
    plain baseline). Columns = top-k doc indices + snippet preview.
    Cells annotated ✓/✗ against the gold relevant set hand-defined
    next to the query in section 1's config.)

## 5. Verdict
   (md: one-paragraph wrap-up. plain = reference, redact = no signal,
    label = too broad, label_number = breaks cross-doc identity,
    label_token = referential integrity preserved. Cross-link back
    to the Sanitization section's table.)
```

### Demo data and gold labels

The corpus is hand-curated for clarity, not realism. Default:

```python
DEMO_USE_CORPUS = [
    "Alice (alice@example.com) drafted the proposal last quarter.",
    "Bob received the file from alice@example.com on Monday.",
    "Charlie owns the account charlie@example.com.",
    "The team lead is alice@example.com per the org chart.",
    "Don submitted feedback to charlie@example.com.",
    "Eve had no involvement and isn't on email yet.",
]

DEMO_USE_QUERIES = [
    {
        "text": "Find documents that mention alice@example.com.",
        "relevant_doc_ids": {0, 1, 3},
    },
    {
        "text": "Anything from charlie@example.com?",
        "relevant_doc_ids": {2, 4},
    },
]
```

Gold relevance sets live next to each query so section 4 can compute and display ✓/✗ for every retrieved hit. Trivial precision/recall calculation per (query, mode).

### Search engine: BM25

Use [`rank-bm25`](https://pypi.org/project/rank-bm25/) — small (single file, pure Python), zero dependencies, ~20 lines of usage. Install inline in cell 2 (the existing harness setup):

```python
_run(f"{PIP} install -q rank-bm25", msg="install rank-bm25 (for the Use section)")
```

Usage pattern:

```python
from rank_bm25 import BM25Okapi

# Tokenization: lowercase + split on whitespace. Crude but enough for
# a demo — what matters is that [EMAIL_u8UBDWQ] survives as one token
# in both the doc and the query.
def _tok(text):
    return text.lower().split()

bm25 = BM25Okapi([_tok(doc) for doc in sanitized_corpus])
scores = bm25.get_scores(_tok(sanitized_query))
top_k = sorted(range(len(scores)), key=scores.__getitem__, reverse=True)[:DEMO_USE_TOP_K]
```

Edge case: tokens like `[EMAIL_u8UBDWQ].` would be split as one whitespace-bounded chunk including the trailing punctuation. To get clean exact-matches we lightly normalize by stripping common trailing punctuation. Single helper, ~5 lines, kept in-cell for transparency.

### Query-time vault call

When the query mode is `label_token`, the query string itself goes through the same `TokenVaultClient` round-trip as the corpus. This is intentional and worth surfacing:

> Production analogue: a "sanitize the query" middleware that runs *before* the search index sees the query. Same vault, same tokens — that's what makes search work across the boundary.

Section 2's code makes this explicit by reusing the existing `label_token_renderer` for both corpus and queries. No new helper.

### Per-mode handling

The Sanitization section already drops `opf_native` and uses the four-mode enum `redact` / `label` / `label_number` / `label_token`. The Use section reuses that exact set + a **plain** baseline (no sanitization). The plain baseline is included as the gold-standard reference — what an attacker would see if there was no privacy layer at all — so the audience can compare the four privacy-preserving modes against the ideal retrieval.

`label_number` correctly demonstrates the "breaks across documents" property: each call to `label_number_renderer()` creates a fresh closure, so doc 0's `alice@example.com` becomes `[EMAIL_1]` and doc 1's becomes `[EMAIL_1]` too (each per-doc) — but the *query's* `[EMAIL_1]` only matches one doc's `[EMAIL_1]` by coincidence. Show that the mode's design is the problem, not a bug.

For `label_token`, the same closure semantics apply but the underlying vault provides cross-call consistency, so `alice@example.com` → `[EMAIL_u8UBDWQ]` in every doc and the query alike.

### Files touched

| File | Change |
|---|---|
| [notebooks/pii_detector_comparison.ipynb](notebooks/pii_detector_comparison.ipynb) | **New `# Use` H1** + 5 numbered subsections (intro md, config md+code, sanitize md+code, index md+code, compare md+code, verdict md). |
| [notebooks/README.md](notebooks/README.md) | One-paragraph mention of the new section + link to the BM25 dep note. |
| Setup cell (cell 3 in the current ipynb) | Add `_run(f"{PIP} install -q rank-bm25", msg="install rank-bm25 (Use section)")` after the existing installs. |
| Setup cell import-confirmation loop | Add `"rank_bm25"` to the post-install sanity check so a failing install surfaces immediately. |

### Reused code

- [eval/src/opf_eval/transforms.py:render_modes](eval/src/opf_eval/transforms.py) — section 2 calls it per doc and per query with `modes=[..., "label_token"]`. The cached `token_vault_client` from section 2 of `# Sanitization` is reused (single vault instance for the whole notebook).
- [eval/src/opf_eval/runner.py:_build_detector](eval/src/opf_eval/runner.py) — section 1 picks a detector for query sanitization. Defaults to `presidio` for the same reasons as the determinism check (fast, CPU-only).
- The Sanitization section's `TOKEN_VAULT` global — referenced directly. Section 1 of `# Use` prints "(vault not configured — `label_token` mode will be skipped)" when `TOKEN_VAULT is None` and continues with the other 3 modes + plain baseline.

## Verification

1. **Cell-by-cell run** in Colab on the `jstjoe/use-section` branch, with `HARNESS_BRANCH` pointed at the branch. End-to-end: sections 1–9 + Sanitization 1–5 + Use 1–5 all complete in under ~10 minutes on the free CPU tier. The label_token cells require a configured vault.
2. **The pedagogical assertions hold**:
   - `plain` baseline returns the relevant docs (sanity check: BM25 over un-sanitized text works).
   - `redact` retrieves uniformly random docs — every doc with PII has identical `********` tokens, so the query's `********` matches everything equally.
   - `label` retrieves every doc with an email — `[EMAIL]` is in all of them, so the query distinguishes nothing.
   - `label_number` retrieves docs whose first email happens to land at the same per-doc counter as the query's first email. Effectively random.
   - `label_token` retrieves exactly the relevant docs.
3. **Edge cases** to spot-check:
   - Query whose plaintext PII doesn't appear in any doc (e.g. `eve@example.com`): `label_token` retrieves nothing (correct), other modes retrieve the broad email match set.
   - Corpus doc with PII not in any query: doesn't appear in any retrieval. No spurious matches.
4. **Notebook tests** stay green:
   - `pytest eval/tests/test_transforms.py api/tests/test_routes.py -q` — unrelated, should pass unchanged.
5. **Stale-ref grep** before commit: `grep -n "vault_token\|label_numbered\|placeholder_format" notebooks/pii_detector_comparison.ipynb` returns nothing.

## Out of scope

- **No `/v1/use` endpoint or any API-side change.** The Use section is notebook-only. If the pattern proves useful, a follow-on plan can wrap it in an endpoint (`/v1/sanitize-and-search` or a Postgres/Pinecone integration).
- **No vector-embedding demo** in this plan. Worth a follow-on: shows that even semantic search benefits from consistent tokens, but it muddies the BM25 punchline. Keep the first pass crisp.
- **No fine-tuning of BM25 parameters** (k1, b). Defaults are fine for the demo.
- **No per-mode F1 / NDCG metrics**. The ✓/✗ table is enough to make the point; precision/recall numbers add complexity without changing the lesson.
- **No persistent index storage**. BM25 instances live in the cell's namespace.
