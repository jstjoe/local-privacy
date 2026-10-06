# Plan 13 — Evaluate detectors on your own data without gold labels

## Status

PRs 1 to 4 are implemented on the branch `jstjoe/unlabeled-eval-notebook`. That covers file parsing and unlabeled fixtures, the LLM clients with their cache, silver labels with calibration, LLM review, Clef-flash review with the dependency bump, and notebook 07. Two parts are parked:

1. PR 5 adds the Amazon Bedrock backends and waits until an AWS account exists. `llm.make_client` refuses `anthropic_bedrock` and `openai_bedrock` with that reason.
2. TypeSafe's Jev waits until its data retention terms work for this use.

These checks are still open:

1. Clef-flash itself has not been run. Its forward path is verified only on a tiny random Qwen3.5 model on CPU, so it needs a check on an L4 with a few requests.
2. The T4 int8 path and the Apple Silicon path are untested. The T4-against-L4 comparison of probabilities still needs a GPU.
3. The Vertex AI region checks in Part 3 need a GCP project. Nobody has checked yet whether Claude Opus 5.5 or gpt-oss is offered in a US west region, or whether the Vertex OpenAI endpoint supports a JSON-schema `response_format`.
4. The OpenAI model ids come from this plan and were not checked against the live catalogues. Any id is accepted at runtime.

The user guide is [eval/docs/your-own-data.md](../eval/docs/your-own-data.md). The section "Implementation notes" at the end of this plan lists where the code differs from the design below.

## Why

Notebooks 00–06 score detectors against datasets that ship gold spans. Real data has no gold spans. A team that wants to choose a detector for its own support tickets or contracts or spreadsheets cannot measure precision and recall today.

This plan adds notebook 07 and the harness code behind it. The notebook runs selected detectors on the user's own text or files. It then estimates precision and recall in three ways that the user can mix:

1. **Silver labels from LLMs.** One or more LLMs annotate the data. The existing scoring code then treats those annotations as gold.
2. **LLM review.** An LLM judges each detected span and looks for PII that every detector missed.
3. **Classifier review.** A typed decision model answers yes/no questions about each span and each text segment. This plan starts with Cloudflare's [Clef-flash](https://blog.cloudflare.com/clef-decision-models/) because it runs locally. TypeSafe's [Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev) is parked.

The notebook also accepts files. Most detectors only take text, so files are parsed to text first in one standard way that keeps a map from each character back to its place in the file.

## Design principles

1. **Reuse the fixture and run-dir contracts.** Parsed files and pasted text become ordinary fixture records with `gold_spans: []`. The runner and `nb.ensure_run` and the saved run directory work unchanged.
2. **Silver labels are just gold from another source.** They are written as a new fixtures file whose `.meta.json` records `gold_source`. The scoring module and every report then work as they do today.
3. **Every estimate states its bias.** Review-based recall only counts misses the reviewer found. The notebook says so next to every number.
4. **Calibrate on real gold first.** Each method can run on a `pii_masking_200k` sample where true gold exists. The notebook shows how far each estimate lands from the true score before the user trusts it on unlabeled data.
5. **Every provider is labelled by who sees the data.** Hosted providers see the raw text. The notebook labels each one as local or hosted and names the operator. The `PII_BENCH_ALLOW_REMOTE` switch (set with `nb.allow_remote()`) gates every hosted call.
6. **LLM calls are cached on disk.** The cache key is the sha256 of the provider and model and system prompt and user prompt and schema. The prompts embed their prompt version, so a prompt change is a new key. Reruns cost nothing and give the same numbers.

## Data exposure policy for now

During internal development it is acceptable for Colab and OpenAI and Anthropic to see the data this notebook processes. Hosted providers are therefore allowed by default for now. The switch is the env var `PII_BENCH_ALLOW_REMOTE` that `nb.allow_remote()` sets and `llm.make_client` reads. Its default lives in `llm.DEFAULT_ALLOW_REMOTE` and `nb.ALLOW_REMOTE_DEFAULT`.

This must be tightened before the notebook is shared or rolled out more widely. That is the reason this plan adds Google Vertex AI backends now and Amazon Bedrock backends in a later phase. Both run the models inside the team's own cloud account and region under that account's data terms. Before a wider rollout the default flips to `False` and the notebook documents which providers are approved.

## Part 1 — File parsing

### Module

A new module `opf_eval/documents/` turns a file into a `Document`:

```python
@dataclass
class Segment:
    start: int          # offsets into Document.text
    end: int
    where: dict         # {"page": 3} | {"sheet": "Q1", "cell": "B7"} | {"row": 4, "column": "email"} | {"line": 7} | ...

@dataclass
class Document:
    id: str             # stable: file sha1 + relative path
    source: str         # original path
    kind: str           # "pdf" | "docx" | "xlsx" | "csv" | "txt" | "md" | "html" | "eml" | "text"
    text: str
    segments: list[Segment]
    warnings: list[str]  # e.g. "page 3 has no text layer (needs OCR)"
```

`documents.parse(path)` picks a parser by suffix and `documents.parse_many(folder)` parses a folder and lists the files it skipped. `fixtures.from_documents(docs, out_path, max_chars=4000)` writes fixture records. Long documents are split into chunks at segment or paragraph boundaries. Each chunk record carries `doc_id` and `offset` so detector spans can be mapped back to the whole document and then to a page or cell.

### Parsers for the first version

| kind | library | text rules |
| --- | --- | --- |
| PDF | `pypdf` | Text layer only. One segment per page. Pages with no text are reported as "needs OCR" and skipped. |
| DOCX | `python-docx` | Paragraphs and table cells and content controls in document order. Headers and footers are separate segments. Locations follow python-docx's 0-based indexing. |
| XLSX | `openpyxl` | One line per row with its non-empty cells joined by a tab. One segment per cell so a span maps to `Sheet!B7`. |
| CSV | stdlib `csv` | Same layout as XLSX. A span maps to `{"row": 2, "column": "email"}`. |
| TXT and MD | none | Read as UTF-8 with a fallback to a byte-order mark or UTF-16 or Windows-1252 and a warning. Markdown is kept as written. |
| HTML | stdlib `html.parser` | Visible text only. Script and style content is dropped. Block elements end a line. |
| EML | stdlib `email` | Headers `From` `To` `Cc` `Subject` as their own segments and then the text body. HTML bodies use the HTML rules. Attachments are parsed recursively when their type is supported. |

I chose thin parsers over [MarkItDown](https://github.com/microsoft/markitdown) or [Docling](https://github.com/docling-project/docling) because both emit Markdown without a map back to source locations. Markdown syntax also adds characters that detectors may tag. Either library can be added later as one more parser if layout matters more than location.

### File-native detectors

Skyflow Detect can take files directly. That path is out of scope for this plan. The `Document` shape leaves room for it because a file-native detector's output can be mapped onto the same segments later.

## Part 2 — Unlabeled fixtures

`fixtures.from_texts(texts, out_path)` and `fixtures.from_documents(docs, out_path)` write fixture records with empty gold. The meta file records `"gold": "none"` and the source paths. `from_documents` also writes a `.documents.jsonl` sidecar with each document's segment map. `validate_fixtures` accepts empty gold and marks the report `unlabeled` when the meta says so. The report refuses to print precision and recall for a run whose fixtures have no gold and points the user to Parts 4 to 6.

## Part 3 — LLM clients

A small module `opf_eval/llm/` with one protocol:

```python
class LLMClient(Protocol):
    provider: str
    model: str
    remote: bool
    operator: str       # who sees the text
    def complete_json(self, *, system: str, user: str, schema: dict,
                      name: str = "result", max_tokens: int = 8000) -> dict: ...
```

### Backends

| backend | operator | SDK and client | structured output | notes |
| --- | --- | --- | --- | --- |
| `anthropic` | Anthropic | `anthropic.Anthropic()` | `client.messages.create()` with `output_config.format` set to a JSON schema | Credentials from `ANTHROPIC_API_KEY` or an `ant auth login` profile. |
| `anthropic_bedrock` (parked, PR 5) | own AWS account | `anthropic.AnthropicBedrockMantle(aws_region=...)` | same as above | Model ids take an `anthropic.` prefix such as `anthropic.claude-opus-5-5`. AWS credentials. |
| `anthropic_vertex` | own GCP project | `anthropic.AnthropicVertex(project_id=..., region="us")` | same as above | Model ids are the bare first-party ids such as `claude-opus-5-5`. Google application default credentials. Needs the `anthropic[vertex]` extra. |
| `openai` | OpenAI | `openai.OpenAI()` | JSON schema `response_format` | Credentials from `OPENAI_API_KEY`. |
| `openai_bedrock` (parked, PR 5) | own AWS account | `openai.OpenAI(base_url="https://bedrock-runtime.<region>.amazonaws.com/openai/v1")` | validate-and-retry until JSON schema support is confirmed | Needs a Bedrock API key. See the [Bedrock OpenAI models page](https://docs.aws.eu/bedrock/latest/userguide/model-parameters-openai.html) and the [Chat Completions guide](https://docs.aws.amazon.com/bedrock/latest/userguide/inference-chat-completions.html). |
| `openai_vertex` | own GCP project | `openai.OpenAI(base_url="https://<region>-aiplatform.googleapis.com/v1/projects/<project>/locations/<region>/endpoints/openapi")` | validate-and-retry until JSON schema support is confirmed | Uses a short-lived Google access token as the API key. See the [Vertex AI OpenAI models page](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/maas/openai). |
| `openai_compatible` | whoever runs the server | `openai.OpenAI(base_url=...)` | JSON schema when the server supports it and validate-and-retry otherwise | Covers LM Studio and vLLM and Ollama. The key comes from `OPENAI_COMPATIBLE_API_KEY` and never from `OPENAI_API_KEY`. Plan 04's local LLM detector can reuse this backend. |

Claude goes through the Anthropic SDK on every platform. It does not go through an OpenAI-compatible shim. Every response is validated against the schema. A failed validation retries once with the error message and is then recorded as an error row.

Claude Opus 5.5 can end a response with `stop_reason: "refusal"`. The client records that as an error row instead of an empty answer. On the first-party API the client opts into server-side fallbacks when the installed SDK supports them. On Bedrock and Vertex that parameter is not available and a refusal stays an error row.

### Models

| family | default | also supported | where available |
| --- | --- | --- | --- |
| Claude | `claude-opus-5-5` | any Claude model id | Anthropic API and Bedrock and Vertex AI. |
| OpenAI | `gpt-6.1-sol` | `gpt-6-luna` and `gpt-5.6-sol` and `gpt-5.6-luna` | OpenAI API for all four. Bedrock lists GPT-6.1 Sol. Vertex AI lists only the open-weight gpt-oss models. |
| OpenAI open weights | `openai/gpt-oss-120b-maas` on Vertex AI | `gpt-oss-120b` and `gpt-oss-20b` | Bedrock and Vertex AI and any local OpenAI-compatible server. |

OpenAI's [model list](https://developers.openai.com/api/docs/models/all) has no GPT-6.1 Luna today. The current Luna is GPT-6 Luna. The id table above still needs a check against the live catalogues because the Bedrock and Vertex catalogues change often. Proprietary GPT models are not on Vertex AI. A team that must keep data inside its GCP project would use Claude on Vertex or gpt-oss on Vertex.

### Vertex AI configuration

1. The GCP project id comes from `GOOGLE_CLOUD_PROJECT` through `nb.load_secrets()`. It is not written into the repo because the repo is public.
2. Claude on Vertex uses the `us` multi-region by default, or `CLOUD_ML_REGION` when it is set. That keeps processing in the US without tying the notebook to one region's capacity. Whether Claude Opus 5.5 is offered in a US west region such as `us-west1` is still unchecked because no GCP project was available during PR 2.
3. gpt-oss on Vertex uses `us-central1` because that is the region the Vertex documentation lists for it. The US west check is still open for it too.
4. Credentials are Google application default credentials. In Colab `nb.gcloud_auth()` calls `google.colab.auth.authenticate_user()`. On a laptop it is `gcloud auth application-default login`.

### Bedrock is a later phase

The `anthropic_bedrock` and `openai_bedrock` backends wait until an AWS account and keys exist. They move to PR 5. The `LLMClient` protocol and the response cache do not change when they are added.

## Part 4 — Silver labels from LLMs

`opf_eval.silver.generate(fixtures, labelers, out_path, labels=..., merge_how=...)` asks each labeler for the PII in each record.

1. **What the LLM returns.** A list of `{value, label, context}` items. LLMs are unreliable at character offsets, so the harness finds offsets itself by searching for `value` in the record text with `silver.locate`. It tries an exact match, then a case-insensitive match, then a whitespace-normalised match, and then the same three after Unicode NFC normalisation. `context` is a few words around the value and picks the right occurrence when a value appears more than once. Values that cannot be found are logged as dropped.
2. **Label set.** The schema enumerates the fixtures' target labels as an `enum`. By default these are every coarse category and fine sub-type. `level="coarse"` asks only for the coarse categories.
3. **Several labelers.** With two or more labelers the user picks `union` or `majority` or `intersection`. Spans from different labelers agree when they share a coarse label and overlap with an intersection over union of at least 0.5. The notebook reports pairwise agreement as strict span F1 between labelers. Low agreement is a warning that the silver labels are noisy.
4. **Output.** A fixtures file with `gold_spans` filled and meta `gold_source: {"kind": "llm", "labelers": [...], "merge": "majority", "prompt_version": ...}`. The existing report and charts and error analysis then run unchanged. Every heading in the notebook says "silver" instead of "gold".
5. **Calibration.** `silver.calibrate(labeler, dataset="pii_masking_200k", n=50, workdir=...)` scores the labeler against real gold. The notebook shows that F1 next to the silver results.

## Part 5 — LLM review

`opf_eval.review.llm.review_run(run_dir, reviewer, fixtures=...)` estimates precision and recall without writing silver labels.

1. **Precision pass.** All detectors' spans for a record are pooled and deduplicated. The reviewer judges each unique span once as `correct` or `wrong_label` or `boundary` or `not_pii`. Each verdict is then credited to every detector that produced the span. Estimated precision for a detector is the share of its spans judged `correct`. A lenient view also counts `boundary`.
2. **Recall pass.** The reviewer sees the text with every pooled span marked. It lists PII that no detector marked. A detector's estimated recall is the true entities it found divided by every true entity the review knows of. The true entities are the pooled spans judged `correct` or `wrong_label` or `boundary` plus the reviewer-found misses. Overlapping `correct` spans count as one entity, and any other accepted span joins the single entity it overlaps most. A detector finds an entity when one of its spans in it was judged `correct`, or `boundary` in the lenient view. This is the pooling method used in information retrieval evaluation.
3. **Bias.** Misses the reviewer also overlooks are invisible. Estimated recall is therefore an upper bound. The notebook prints that sentence under the table.
4. **Output.** `review_llm_<provider>_<model>.jsonl` in the run directory holds one row per judged span and one row per found miss. A summary table gives estimated precision and recall per detector. A list of disagreements links each one back to its text.

## Part 6 — Classifier review with Clef-flash

Typed decision models answer questions with calibrated probabilities instead of free text. They are a cheaper and faster second opinion to the LLM reviewer. This plan implements Cloudflare's Clef-flash first because it runs locally. TypeSafe's Jev is parked because it offers zero data retention only to enterprise customers. The review module keeps a backend interface so Jev can be added later without changing the notebook.

### Questions

1. **Per span.** One yes/no question asks whether the marked span is personal or sensitive data. A second yes/no question asks whether the proposed label is the right type. A choice question asks which coarse label fits best. A span counts as correct at a threshold when both yes/no probabilities reach it. These give a precision estimate with a probability attached to each judgement.
2. **Per segment.** Each sentence of each line is shown with one detector's spans replaced by placeholders. A yes/no question asks "Does this text still contain personal data?". A choice question asks which type. These give a residual-PII rate per detector. Classifiers return no spans, so recall is measured per segment and not per span.
3. **Threshold.** A probability threshold turns answers into verdicts. The default is 0.5. The calibration cell shows precision and recall estimates across thresholds against real gold.

### Backends

| backend | status | where it runs | how |
| --- | --- | --- | --- |
| `clef_flash` | this plan | local | Open weights [`Cloudflare/clef-flash`](https://huggingface.co/Cloudflare/clef-flash) under Apache-2.0. Built on Qwen3.5-9B. Loaded with the model repo's own `joint_schema_model.py` loader and its separate `joint_head.safetensors` decision head. Without `torchvision` the backend loads the same backbone and head with a plain tokenizer. |
| `clef` | later | local | `Cloudflare/clef` on a 27B backbone. Needs a large GPU. |
| `jev` | parked | TypeSafe's hosted API | `typesafe-sdk` package with `TYPESAFE_API_KEY`. Parked because of the data retention terms on the [legal page](https://docs.typesafe.ai/legal.md). |

### Running Clef-flash in Colab

Clef-flash has 9.41 billion parameters stored in bf16. The weights alone take about 18.8 GB. A classification pass is one forward pass with no generation, so short segments add only a little memory on top.

| runtime | memory | verdict |
| --- | --- | --- |
| L4 | 24 GB with native bf16 | **Primary target.** The weights fit with about 3 GB to spare. Segments stay short and batches stay small. |
| T4 | 16 GB with no native bf16 | Possible only with 8-bit or 4-bit weights through `bitsandbytes`. 8-bit needs about 10.7 GiB because the embeddings and the output layer stay in 16 bits. A T4 has no native bf16, so it computes in float16 and the decision head runs in float16 too. The repo's loader passes a quantization config through to `from_pretrained`, so it needs no patch. Quantization may shift the probabilities, so PR 4 measures T4 output against L4 output on the same records before the notebook offers it. |
| TPU v5e-1 | 16 GB of HBM on one chip | **Not recommended.** It has the same memory problem as the T4. The loader targets CUDA. `bitsandbytes` does not run on TPUs. Making it work would mean porting the loader to `torch_xla` or JAX. |
| A100 | 40 GB or 80 GB | Works without changes. Paid Colab tiers only. |
| Apple Silicon | unified memory | Should work on MPS with 32 GB or more. Unverified. |

The notebook checks the runtime's GPU memory with `nb.gpu_summary()` and `review.clef.plan_load()` before loading the model. It explains the options when the runtime is too small. Both measure memory in GiB, so an L4 reports about 22.0 to 22.5 GiB and the bf16 threshold is 21 GiB. The threshold sits below the smallest L4 reading so that every L4 loads in bf16.

### Dependency change

Clef-flash needs `torch` 2.11 or later and `transformers` 5.10.2 or later. Before PR 4 the lock file had `torch` 2.11.0 and `transformers` 5.1.0. PR 4 raised `gliner` to 0.2.29 because that release accepts `transformers` below 5.17, and the lock file now has `transformers` 5.16.1. `openmed` sets no upper bound. The bump also added `protobuf` as a direct dependency because GLiNER's mdeberta tokenizer needs it and `gliner` 0.2.29 no longer pulls it in. Every default detector was rerun on the 200-record notebook baseline after the bump. One GLiNER-Nvidia span moved and that baseline file was refreshed.

The community GGUF builds of Clef-flash probably lack the decision head because it ships as a separate file. They are not used.

## Part 7 — Notebook 07

`notebooks/07_evaluate_without_labels.ipynb` follows the series' layout and session model.

1. **Setup.** Install extras and set `ALLOW_REMOTE` and load secrets.
2. **Bring your data.** Paste text or upload files or point at a Drive folder. Parse files and preview each document with its segments. List files that were skipped and why.
3. **Run detectors.** Pick detectors and run them through `nb.ensure_run`. Show spans in context with their page or cell.
4. **Silver labels (option A).** Pick one or more LLM labelers. Generate and merge. Show agreement. Score with the existing report.
5. **LLM review (option B).** Pick a reviewer. Show estimated precision and recall and the list of disagreements.
6. **Classifier review (option C).** Check GPU memory and load Clef-flash. Show span verdicts and residual-PII rates.
7. **Compare.** Put the three estimates side by side for each detector. Highlight where they disagree most.
8. **Calibrate (optional).** Rerun the chosen methods on a gold sample and show each estimate against the true score.

Options A to C are each skippable. Each prints a cost and time estimate before it calls a hosted model.

## Files

| file | change |
| --- | --- |
| `eval/src/opf_eval/documents/__init__.py` and one module per parser | new |
| `eval/src/opf_eval/fixtures.py` | add `from_texts` and `from_documents` and accept empty gold |
| `eval/src/opf_eval/llm/__init__.py` and `base.py` and `anthropic.py` and `openai.py` and `cache.py` and `schema.py` and `stub.py` | new. Each SDK module covers the first-party and Vertex clients. Bedrock waits for PR 5. |
| `eval/src/opf_eval/silver.py` | new |
| `eval/src/opf_eval/review/pool.py` and `review/llm.py` and `review/classifier.py` and `review/clef.py` | new |
| `eval/src/opf_eval/report.py` | refuse P/R without gold and label silver gold as silver. `build_report(..., detectors=[...])` and the CLI flag `--detectors a,b` limit every section to the chosen detectors. |
| `eval/src/opf_eval/nb.py` | helpers for upload and Drive folders and the remote-provider guard |
| `eval/pyproject.toml` | extras `files` and `llm` and `clef`. Raise `gliner` and add `protobuf` so `transformers` can move up for Clef-flash. |
| `eval/tests/` | parser offset round-trips on small fixture files. Offset finding and merging for silver labels. Review maths with a stub reviewer. Cache keys. |
| `notebooks/07_evaluate_without_labels.ipynb` and `notebooks/README.md` | new notebook and index row |
| `eval/docs/your-own-data.md` | new user guide |

Tests use stub LLM and classifier clients. No test calls a hosted API.

## Phasing

| PR | scope | effort |
| --- | --- | --- |
| 1 | File parsing and unlabeled fixtures and notebook sections 1 to 3 | ~1 day |
| 2 | LLM clients and cache and silver labels and calibration and notebook section 4 | ~1–1.5 days |
| 3 | LLM review and notebook section 5 | ~1 day |
| 4 | Clef-flash review and the dependency bump and notebook sections 6 to 8 | ~1.5–2 days |
| 5 | Bedrock backends for Claude and OpenAI models once an AWS account exists | ~0.5 day |

## Decisions

1. Labelling and review default to `claude-opus-5-5`.
2. The OpenAI default is `gpt-6.1-sol`. The backends also accept `gpt-6-luna` and `gpt-5.6-sol` and `gpt-5.6-luna`.
3. Clef-flash comes first. Jev is parked until its data retention terms work for this use.
4. Clef-flash targets a Colab L4 first. A T4 is tried with 8-bit weights. The TPU v5e-1 runtime is not targeted.
5. Colab and OpenAI and Anthropic may see the data during internal development. Bedrock and Vertex support exists so this can be tightened before a wider rollout.
6. Vertex AI uses the team's GCP project from `GOOGLE_CLOUD_PROJECT` with US regions as described in Part 3.
7. Bedrock is parked until an AWS account exists. Its region and authentication are decided then. It is PR 5.

## Implementation notes

The code for PRs 1 to 4 differs from the design above in the ways listed here. Each difference changes behaviour that a user or a later PR can see. The [user guide](../eval/docs/your-own-data.md) describes the code as it is.

### File parsing and unlabeled fixtures

1. TXT, MD, CSV and HTML files are no longer read as UTF-8 with replacement characters. `documents.text.decode_text` honours a UTF-8, UTF-16 or UTF-32 byte-order mark, recognises UTF-16 without a mark by its NUL bytes, and falls back to Windows-1252 for bytes that are not valid UTF-8. Every fallback adds a warning to the document.
2. DOCX locations follow python-docx's 0-based indexing. `{"paragraph": i}` is `doc.paragraphs[i]` with blank paragraphs counted and `{"table": t, "row": r, "col": c}` is `doc.tables[t].rows[r].cells[c]`. Content inside a body-level content control gets `{"content_control": k, "paragraph": j}` or `{"content_control": k, "table": t, "row": r, "col": c}`. Headers and footers are `{"part": "header", "section": s}` or `{"part": "footer", "section": s}` with `"variant": "first"` or `"variant": "even"` for those variants. A merged cell is emitted once. Paragraph text is read from the XML so tracked insertions count and deletions do not.
3. XLSX, CSV and DOCX table lines join only the non-empty cells with a tab. Excel dates at midnight render as a plain date and integral numbers render without `.0`.
4. The HTML parser treats more elements as blocks than the design listed, separates table cells on one row with a tab and keeps line breaks inside `pre`.
5. An attachment inside an attached email gets a path such as `"attached.eml/scan.pdf"` in `attachment`. Recursion stops at a depth of 4.
6. `parse_many` also skips Office lock files, missing paths and duplicate files with the same document id.
7. `fixtures.from_documents` writes a second sidecar `<fixtures file name>.documents.jsonl` with each document's segment map. `fixtures.load_documents` rebuilds the documents from it, and `fixtures.map_spans_to_documents` fills `where` from it when no documents are passed, so locations survive a Colab kernel restart. Each mapped row also carries `chunk_id`.
8. `from_documents` drops chunks that hold only whitespace, so a document's `n_chunks` can be smaller than the number of ranges `documents.chunk` returns. `from_texts` skips blank texts but keeps their number in the id sequence and records `n_blank_skipped`.

### LLM clients

1. `openai_compatible` reads its key from `OPENAI_COMPATIBLE_API_KEY` and never falls back to `OPENAI_API_KEY` because that would send a real OpenAI key to whatever host `base_url` names. For the same reason the `openai` provider refuses a `base_url`.
2. The Vertex OpenAI endpoint uses the regional host `<region>-aiplatform.googleapis.com` for a regional location, `aiplatform.googleapis.com` for `global` and `aiplatform.<us|eu>.rep.googleapis.com` for the multi-regions.
3. The cache key adds a `settings` component when a backend has settings that change its answers. Anthropic adds its effort level, `openai_compatible` adds its `base_url` and OpenAI adds `reasoning_effort` when it is set.
4. The OpenAI backends step down from JSON schema to JSON object to prompt-only output because some local servers support neither response format.

### Silver labels and the report

1. `silver.locate` tries Unicode NFC matching after the exact, case-insensitive and whitespace tiers. It uses a match inside a longer word only when no tier has a stand-alone match, so "Ann" never labels "Annual". A context that matches at every occurrence counts as no context.
2. `locate` drops items with the reasons `empty`, `unknown_label`, `not_found`, `extra_occurrence` and `overlap`.
3. Records longer than 12,000 characters are sent in pieces. A piece whose answer stops at max_tokens is cut in half and asked again, down to 500 characters.
4. Only `LLMError` becomes an error row in `silver.generate`. Any other exception stops the run.
5. The silver meta adds `gold_source.names`, `gold_source.providers`, `gold_source.remote`, `n_spans`, `n_dropped`, `labeler_files` and `agreement`.
6. The report scores silver fixtures on the silver meta's `labels` and leaves out the records in `error_ids` because their empty silver gold would otherwise count every detected span there as spurious. When every labeler failed on every record the report prints a note instead of scores.
7. `silver.calibrate` defaults to `n=50` and scores at the coarse level, and it needs a `workdir`.

### LLM review

1. The review prompt is `review-v2`. It tells the reviewer to list PII inside a candidate it judged `not_pii` as missed.
2. Recall counts entities and not spans, as Part 5 now describes. Pooled spans judged `wrong_label` or `boundary` count as true entities because the reviewer never lists PII that such a candidate already covers. The design's formula would have dropped those entities from the denominator and inflated every detector's recall.
3. A candidate with no verdict after the client's one retry is written as `unjudged` and left out of every number. A located miss that an accepted candidate already stands for is dropped with the reason `covered_by_candidate`. A miss that only partly overlaps a `boundary` candidate which belongs to another entity stays a miss.

### Classifier review and Clef-flash

1. Residual segments are one per sentence of each line, at most 600 characters, and no cut falls inside a detected span. Every detector shares the same boundaries so their residual rates have the same denominator. Each detector's mask covers all of its spans whatever their label. A segment with nothing left after masking is not sent and is written with `p_residual` 0.0.
2. Questions use the coarse categories of the target labels so that each request has at most 15 options.
3. `classifier.threshold_sweep` also returns estimated and true residual rates and the error columns `precision_error` and `residual_error`.
4. `clef.plan_load` measures memory in GiB. A CUDA card with at least 21 GiB but no native bf16 maps to `fp16` because 8-bit weights would still compute in float16. The `int8` mode computes in bfloat16 when the card has native bf16.
5. `torchvision` is deliberately not in the `clef` extra because its latest release would pull a newer `torch` for every detector. Colab ships it. Without it `ClefBackend` loads the same backbone and head with a plain tokenizer (`loader="text"`), which is all text review needs.

### Dependencies and the notebook baseline

1. `gliner` is now 0.2.29 or later and the lock file has `transformers` 5.16.1 and `torch` 2.11.0.
2. `protobuf` is a direct dependency because GLiNER's mdeberta tokenizer needs it to convert `spm.model` and `gliner` 0.2.29 no longer pulls it in.
3. The regression check reran the default detectors on the 200-record `pii_masking_200k` notebook baseline after the bump. One GLiNER-Nvidia span changed: record 16466 lost a MONEY span on "₨855k". `baselines/pii_masking_200k_s42/raw_gliner_nvidia.jsonl` was refreshed with the new output.
4. `eval/pyproject.toml` registers the `slow` pytest marker for the Clef tests that build small models or fetch files from the Hugging Face Hub. `eval/tests/conftest.py` skips those tests unless `PII_BENCH_SLOW_TESTS=1` is set, so the default test command never touches the network. Run them with `PII_BENCH_SLOW_TESTS=1 uv run --with pytest python -m pytest eval/tests -q`. The test that loads the real Clef-flash tokenizer also needs `PII_BENCH_CLEF_TOKENIZER=1` because it fetches a 20 MB file from the Hub.

### Notebook helpers

1. `nb.Session` has a `fixtures` field that points a session at an explicit fixtures file. `nb.session()` clears a saved `fixtures` override when `dataset`, `n` or `seed` is passed. Notebook 07 should still use an unsaved `nb.Session(fixtures=...)` because notebooks 02, 03 and 05 load the saved session without passing a dataset.
2. `nb.ensure_run` notices when a session's fixtures file changed and reruns the detectors.
3. `nb.show_spans_in_context` needs `doc_id=` when its rows cover more than one document.

