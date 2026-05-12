# Notebooks

## `pii_detector_comparison.ipynb`

Self-contained Colab notebook for benchmarking PII detectors against PII-Masking-300k.

### Open in Colab

[Open in Colab](https://colab.research.google.com/github/jstjoe/local-privacy/blob/main/notebooks/pii_detector_comparison.ipynb)

### Sharing with others

The setup cell clones two repos:

- `https://github.com/openai/privacy-filter` — the OPF detector source
- `https://github.com/jstjoe/local-privacy` — this harness (set as `HARNESS_REPO` in the setup cell; collaborators using a fork should update it)

### What it produces

- Markdown report rendered inline (headline, per-category, per-language)
- Optional bar charts: per-category F1 across detectors, latency comparison
- Raw JSONL outputs in `/content/results/<run_name>/`, copyable to Drive
- **Sanitization section** — side-by-side rendering of the same fixtures under four modes (`redact`, `label`, `label_number`, `label_token`) per detector, plus a per-mode determinism check. Reuses the raw JSONL files from section 5 (no extra detector runs). The `label_token` column requires a configured Skyflow token vault — see [docs/token-vault-setup.md](../docs/token-vault-setup.md); other modes work without it.
- **Use section** — tiny BM25 search demo over a 20-doc corpus with controlled PII overlap. Sanitizes both the corpus and the queries under every mode, indexes per mode, runs the queries, and prints precision/recall against hand-defined gold relevance sets. Demonstrates that only `label_token` preserves cross-document identity well enough for search / RAG / joins to keep working. Installs [`rank-bm25`](https://pypi.org/project/rank-bm25/) in the setup cell.

### Detector selection

Edit `DETECTORS` in cell 5. Available names:

- `opf` — OpenAI Privacy Filter (open-weight, local)
- `gliner` — GLiNER multilingual PII (open-weight, local)
- `gliner_nvidia` — Nvidia gliner-PII on `urchade/gliner_large-v2.1` (570M, NVIDIA Open Model License)
- `gliner_gretel_small` / `gliner_gretel_large` — Gretel bi-encoder GLiNER variants (English-only training)
- `ai4privacy_modernbert` — ai4privacy ModernBERT-base, MIT, 8 languages, OpenPII vocab
- `openmed` — OpenMed PII via `openmed.extract_pii(lang=…)`, DeBERTa-based per-language models
- `presidio` — Microsoft Presidio English-only (free, local, regex+NER)
- `presidio_multilang` — Presidio with all 6 language models
- `skyflow` / `skyflow_full` — Skyflow Detect API (requires creds; `skyflow` auto-derives `entity_types` from the chosen dataset)

### Skyflow auth in Colab

Use Colab's secret manager (key icon in left sidebar). Set:

- `SKYFLOW_VAULT_URL`
- `SKYFLOW_VAULT_ID`
- `SKYFLOW_BEARER_TOKEN`

Cell 3 of the notebook auto-loads them into env vars.
