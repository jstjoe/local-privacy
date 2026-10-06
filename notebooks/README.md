# Notebook series

Eight Colab-ready notebooks covering PII **detection**, **sanitization**, and **use** of sanitized text, plus **evaluation on your own data without gold labels**. Each one is short enough to follow in one sitting. Each one also runs on its own: later notebooks create any fixtures or detector results they need.

| # | notebook | covers | |
| --- | --- | --- | --- |
| 00 | [Start here](00_start_here.ipynb) | the three questions (detect / sanitize / use), the two-level label taxonomy, detector and dataset catalogue, how scoring works, a first detection | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jstjoe/local-privacy/blob/main/notebooks/00_start_here.ipynb) |
| 01 | [Datasets](01_datasets.ipynb) | deterministic samples, the fixture record format, validation, **bring your own data** (JSONL, CSV with `label_map`, BIO tokens) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jstjoe/local-privacy/blob/main/notebooks/01_datasets.ipynb) |
| 02 | [Run detectors](02_run_detectors.ipynb) | picking detectors, credentials, running, what a detector returns, latency | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jstjoe/local-privacy/blob/main/notebooks/02_run_detectors.ipynb) |
| 03 | [Score and compare](03_score_and_compare.ipynb) | the report, fair vs raw views, charts, **fine-level** scoring, error analysis (missed / spurious / mislabeled / boundary) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jstjoe/local-privacy/blob/main/notebooks/03_score_and_compare.ipynb) |
| 04 | [Composite detectors](04_composite_detectors.ipynb) | best-per-category ensembles, local-only vs with a hosted API, upper bound vs honest holdout | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jstjoe/local-privacy/blob/main/notebooks/04_composite_detectors.ipynb) |
| 05 | [Sanitization](05_sanitization.ipynb) | `redact`, `label`, `label_number`, `label_token` side by side; sanitize your own text | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jstjoe/local-privacy/blob/main/notebooks/05_sanitization.ipynb) |
| 06 | [Search on sanitized data](06_search_on_sanitized_data.ipynb) | BM25 over sanitized documents: which modes keep retrieval working, and why | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jstjoe/local-privacy/blob/main/notebooks/06_search_on_sanitized_data.ipynb) |
| 07 | [Evaluate without labels](07_evaluate_without_labels.ipynb) | **your own text and files** (PDF, Word, Excel, CSV, text, HTML, email), estimated precision and recall from LLM silver labels, LLM review or Clef-flash review, calibration against real gold | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jstjoe/local-privacy/blob/main/notebooks/07_evaluate_without_labels.ipynb) |

## How the notebooks fit together

```text
01 choose sample ──► fixtures file (+ .meta.json)
02 run detectors ──► run dir: raw_<detector>.jsonl + manifest.json
03 score  ─┐
04 compose ├─ read the run dir, or the saved baseline (no model re-runs)
05 sanitize┘
06 search  ── builds its own corpus; uses one detector live
07 your data ──► unlabeled fixtures (gold: none) ──► run dir ──► silver labels / LLM review / Clef-flash review
```

The choices you make (dataset, sample size, seed, detectors) are saved as a **session** in the workspace, so each notebook picks up where the last left off. The run directory's name is derived from the sample, so changing the sample never mixes old and new results.

Notebook 07 works on your own files and does not save its fixtures into the shared session. Notebooks 01 to 06 therefore keep working on the dataset sample you chose.

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

Everything runs without credentials. Three optional features need secrets. On Colab, add them in **Secrets** (key icon in the left sidebar). Locally, set them in your shell or a `.env` file.

| feature | secrets |
| --- | --- |
| `skyflow` detector (02–04) | `SKYFLOW_VAULT_URL`, `SKYFLOW_VAULT_ID`, `SKYFLOW_BEARER_TOKEN` |
| `label_token` sanitization (05–06) | `SKYFLOW_TOKEN_VAULT_URL`, `SKYFLOW_TOKEN_VAULT_ID`, and `SKYFLOW_TOKEN_BEARER_TOKEN` (or `SKYFLOW_BEARER_TOKEN`). One-time vault setup: [docs/token-vault-setup.md](../docs/token-vault-setup.md) |
| hosted LLMs for silver labels and LLM review (07) | `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, or `GOOGLE_CLOUD_PROJECT` with Google sign-in for Vertex AI. `OPENAI_COMPATIBLE_API_KEY` is only for an OpenAI-compatible server that needs a key. See [Data policy and secrets for notebook 07](#data-policy-and-secrets-for-notebook-07). |

## Bring your own files

Notebook 07 scores detectors on your own text and files even though they have no gold spans. You can paste text or upload files in Colab with `nb.upload_files()` or point the notebook at a Google Drive folder with `nb.drive_folder("my-folder")`. On a laptop you copy files into the folder that `nb.inputs_dir()` returns, which is `<workspace>/inputs`. The notebook also writes a small synthetic sample corpus so it runs with no uploads.

The notebook accepts these file types:

1. PDF files with a text layer. Scanned pages are reported as "needs OCR" and skipped.
2. Word documents (`.docx`).
3. Excel workbooks (`.xlsx` and `.xlsm`) and CSV or TSV files.
4. Plain text and Markdown.
5. HTML pages.
6. Emails (`.eml`) together with their supported attachments.

Every file is parsed to plain text with a map back to its page, cell, line or email header, so each detected span is shown with its location. Long documents are split into chunks without cutting through a value. The [your-own-data guide](../eval/docs/your-own-data.md) lists the exact location shapes and parser rules.

Precision and recall then come from three estimates that you can mix:

1. **Silver labels** have one or more LLMs annotate the records. The normal report then scores detectors against those annotations and is titled "Silver-label report".
2. **LLM review** has an LLM judge each detected span and list the PII that every detector missed. Its recall is an upper bound because PII that every detector and the reviewer overlooked is invisible.
3. **Classifier review** runs Cloudflare's Clef-flash locally on a GPU with about 24 GB of memory (a Colab L4). It estimates precision and the share of sentences that still hold PII after each detector's spans are masked.

Each optional section starts with a `RUN_...` flag so the notebook runs top to bottom without keys or a GPU.

Two environment variables help with dry runs. `PII_BENCH_NOTEBOOK_STUB=1` swaps every LLM for a local stub and Clef-flash for a stub backend so the notebook runs end to end with no keys and no GPU. Its toy pattern matchers produce numbers that mean nothing. A stub run is not fully offline because the detectors still run for real. Its first run installs the spaCy model `en_core_web_lg` and downloads the GLiNER weights from Hugging Face. `PII_BENCH_NOTEBOOK_DETECTORS` takes a comma-separated list of detector names such as `presidio` and replaces the notebook's `DETECTORS` list.

## Data policy and secrets for notebook 07

Hosted LLMs see the raw text you send them. During internal development it is acceptable for Colab, OpenAI and Anthropic to see the data this notebook processes, so hosted providers are allowed by default for now. This default will flip to off before the notebook is shared more widely.

1. `PII_BENCH_ALLOW_REMOTE` gates every hosted LLM call. An unset value means allowed for now. The values `1`, `true`, `yes` and `on` allow hosted calls and any other value such as `0` forbids them. `nb.allow_remote(False)` switches hosted calls off for the current kernel.
2. `ANTHROPIC_API_KEY` is needed for Claude on the Anthropic API.
3. `OPENAI_API_KEY` is needed for OpenAI models on the OpenAI API.
4. `GOOGLE_CLOUD_PROJECT` names the GCP project for Claude or gpt-oss on Vertex AI. Vertex AI also needs Google application default credentials. The notebook calls `nb.gcloud_auth()`, which opens Colab's sign-in prompt. On a laptop run `gcloud auth application-default login` once instead. Vertex AI keeps the data inside your own GCP project under that project's terms.
5. `OPENAI_COMPATIBLE_API_KEY` is needed only for an `openai_compatible` server that asks for a key. That backend never reads `OPENAI_API_KEY` so a real OpenAI key is never sent to another host.

Add the keys in Colab **Secrets** or in a local `.env` file. `nb.load_secrets(*nb.LLM_SECRETS)` copies them into the environment without printing them. Before the hosted calls of options A and B the notebook prints which model runs where and who sees the text, together with a call count. The calibration in section 8 reruns the same labelers and reviewer on a gold sample of `CALIBRATION_N` records. It prints the same line and call count before each labeler and before the reviewer. It also skips a hosted labeler when `PII_BENCH_ALLOW_REMOTE` has been switched off since option A ran.

The LLM clients can cache answers on disk so reruns cost nothing. The cache holds PII values that the models quoted back, so keep it as private as the data itself. Clef-flash and an OpenAI-compatible server on localhost keep the text on the machine.

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
| `documents` | parsing PDF, Word, Excel, CSV, text, HTML and email files into text with a location map |
| `llm` | LLM clients for every provider, the remote-provider guard and the response cache |
| `silver`, `review` | silver labels, LLM review and Clef-flash review for data without gold labels |

The previous single workshop notebook (`pii_detector_comparison.ipynb`) and `composite_experiments.ipynb` were split into this series; they're in git history.
