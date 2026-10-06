# Notebook series

Seven Colab-ready notebooks covering PII **detection**, **sanitization**, and **use** of sanitized text. Each one is short enough to follow in one sitting. Each one also runs on its own: later notebooks create any fixtures or detector results they need.

| # | notebook | covers | |
| --- | --- | --- | --- |
| 00 | [Start here](00_start_here.ipynb) | the three questions (detect / sanitize / use), the two-level label taxonomy, detector and dataset catalogue, how scoring works, a first detection | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jstjoe/local-privacy/blob/main/notebooks/00_start_here.ipynb) |
| 01 | [Datasets](01_datasets.ipynb) | deterministic samples, the fixture record format, validation, **bring your own data** (JSONL, CSV with `label_map`, BIO tokens) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jstjoe/local-privacy/blob/main/notebooks/01_datasets.ipynb) |
| 02 | [Run detectors](02_run_detectors.ipynb) | picking detectors, credentials, running, what a detector returns, latency | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jstjoe/local-privacy/blob/main/notebooks/02_run_detectors.ipynb) |
| 03 | [Score and compare](03_score_and_compare.ipynb) | the report, fair vs raw views, charts, **fine-level** scoring, error analysis (missed / spurious / mislabeled / boundary) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jstjoe/local-privacy/blob/main/notebooks/03_score_and_compare.ipynb) |
| 04 | [Composite detectors](04_composite_detectors.ipynb) | best-per-category ensembles, local-only vs with a hosted API, upper bound vs honest holdout | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jstjoe/local-privacy/blob/main/notebooks/04_composite_detectors.ipynb) |
| 05 | [Sanitization](05_sanitization.ipynb) | `redact`, `label`, `label_number`, `label_token` side by side; sanitize your own text | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jstjoe/local-privacy/blob/main/notebooks/05_sanitization.ipynb) |
| 06 | [Search on sanitized data](06_search_on_sanitized_data.ipynb) | BM25 over sanitized documents: which modes keep retrieval working, and why | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jstjoe/local-privacy/blob/main/notebooks/06_search_on_sanitized_data.ipynb) |

## How the notebooks fit together

```text
01 choose sample ──► fixtures file (+ .meta.json)
02 run detectors ──► run dir: raw_<detector>.jsonl + manifest.json
03 score  ─┐
04 compose ├─ read the run dir, or the saved baseline (no model re-runs)
05 sanitize┘
06 search  ── builds its own corpus; uses one detector live
```

The choices you make (dataset, sample size, seed, detectors) are saved as a **session** in the workspace, so each notebook picks up where the last left off. The run directory's name is derived from the sample, so changing the sample never mixes old and new results.

| where | workspace |
| --- | --- |
| Colab | `/content/pii-bench`, or `MyDrive/pii-bench` in Google Drive with `USE_DRIVE = True` |
| local checkout | `eval/` (`eval/data/`, `eval/results/runs/`) |
| anywhere | `$PII_BENCH_HOME` overrides both |

## Setup

**Colab:** open a notebook and run the first code cell. It clones this repo and [openai/privacy-filter](https://github.com/openai/privacy-filter), installs them, and sets up the kernel. It takes a few minutes the first time. A GPU runtime (*Runtime → Change runtime type → T4 GPU*) makes OPF and the larger GLiNER models 5–10× faster.

To run your fork or a PR branch, edit `HARNESS_REPO` / `HARNESS_REF` in that cell.

Each Colab notebook gets its own runtime, so by default nothing carries over between them. Two things keep that cheap:

- **`USE_DRIVE = True`** in the setup cell keeps the workspace (session, fixtures, runs) in Google Drive. Set it in every notebook you open; a run from 02 is then reused by 03–06 instead of recomputed.
- **The saved baseline.** For the default session, 03–05 load detector results shipped with the package instead of running the models, so they work on a CPU runtime. Only 02, or a change to the sample or detectors, needs a GPU. Each notebook says at the top which runtime it needs.

The baseline lives in [`eval/src/opf_eval/baselines/`](../eval/src/opf_eval/baselines/). Rebuild it after changing a default detector with `python -m opf_eval.baseline build --n 200` on a GPU machine, or `nb.export_baseline(S)` after a run.

**Locally:**

```sh
git clone https://github.com/openai/privacy-filter
uv sync --all-packages --all-extras   # includes the notebooks extra (matplotlib, bm25s)
uv run jupyter lab notebooks/
```

The setup cell does nothing when the harness is already importable.

## Secrets

Everything runs without credentials. Two optional features need secrets. On Colab, add them in **Secrets** (key icon in the left sidebar). Locally, set them in your shell or a `.env` file.

| feature | secrets |
| --- | --- |
| `skyflow` detector (02–04) | `SKYFLOW_VAULT_URL`, `SKYFLOW_VAULT_ID`, `SKYFLOW_BEARER_TOKEN` |
| `label_token` sanitization (05–06) | `SKYFLOW_TOKEN_VAULT_URL`, `SKYFLOW_TOKEN_VAULT_ID`, and `SKYFLOW_TOKEN_BEARER_TOKEN` (or `SKYFLOW_BEARER_TOKEN`). One-time vault setup: [docs/token-vault-setup.md](../docs/token-vault-setup.md) |

## The library behind the notebooks

The notebooks hold only narrative and short cells; the logic lives in `eval/src/opf_eval/`, where it's tested and shared with the CLI and API:

| module | used for |
| --- | --- |
| `nb` | setup, workspace, secrets, session, Markdown tables |
| `datasets`, `fixtures` | registering data, materializing and validating samples |
| `detectors.registry` | every detector by name; adding your own |
| `runner` | running detectors over fixtures |
| `scoring`, `report` | label scopes, SemEval scores, coverage, error examples, the Markdown report |
| `ensemble` | composite detectors |
| `plots` | charts |
| `transforms`, `demo.sanitize`, `demo.search` | sanitization modes and the search demo |

The previous single workshop notebook (`pii_detector_comparison.ipynb`) and `composite_experiments.ipynb` were split into this series; they're in git history.
