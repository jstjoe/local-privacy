# Notebooks

## `pii_detector_comparison.ipynb`

Self-contained Colab notebook for benchmarking PII detectors against PII-Masking-300k.

### Open in Colab

Once you've pushed this repo to GitHub:

```
https://colab.research.google.com/github/<USER>/local-privacy/blob/main/notebooks/pii_detector_comparison.ipynb
```

### Sharing with others

The setup cell clones two repos:
- `https://github.com/openai/privacy-filter` — the OPF detector source
- `HARNESS_REPO` — must be edited to point at this repo's GitHub URL

Tell collaborators to either fork this repo and update `HARNESS_REPO`, or use the URL of the repo you've published.

### What it produces

- Markdown report rendered inline (headline, per-category, per-language)
- Optional bar charts: per-category F1 across detectors, latency comparison
- Raw JSONL outputs in `/content/results/<run_name>/`, copyable to Drive

### Detector selection

Edit `DETECTORS` in cell 5. Available names:
- `opf` — OpenAI Privacy Filter (open-weight, local)
- `gliner` — GLiNER multilingual PII (open-weight, local)
- `presidio` — Microsoft Presidio English-only (free, local, regex+NER)
- `presidio_multilang` — Presidio with all 6 language models
- `skyflow` / `skyflow_minimal` / `skyflow_full` — Skyflow Detect API (requires creds)

### Skyflow auth in Colab

Use Colab's secret manager (key icon in left sidebar). Set:
- `SKYFLOW_VAULT_URL`
- `SKYFLOW_VAULT_ID`
- `SKYFLOW_BEARER_TOKEN`

Cell 3 of the notebook auto-loads them into env vars.
