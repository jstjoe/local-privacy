# Plan 13 — Evaluate detectors on your own data without gold labels

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
5. **Every provider is labelled by who sees the data.** Hosted providers see the raw text. The notebook labels each one as local or hosted and names the operator. An `ALLOW_REMOTE` switch gates every hosted call.
6. **LLM calls are cached on disk.** The cache key is the provider and model and prompt version and text hash. Reruns cost nothing and give the same numbers.

## Data exposure policy for now

During internal development it is acceptable for Colab and OpenAI and Anthropic to see the data this notebook processes. `ALLOW_REMOTE` therefore defaults to `True` for now.

This must be tightened before the notebook is shared or rolled out more widely. That is the reason this plan adds Google Vertex AI backends now and Amazon Bedrock backends in a later phase. Both run the models inside the team's own cloud account and region under that account's data terms. Before a wider rollout the default flips to `False` and the notebook documents which providers are approved.

## Part 1 — File parsing

### Module

A new module `opf_eval/documents/` turns a file into a `Document`:

```python
@dataclass
class Segment:
    start: int          # offsets into Document.text
    end: int
    where: dict         # {"page": 3} | {"sheet": "Q1", "cell": "B7"} | {"part": "body"}

@dataclass
class Document:
    id: str             # stable: file sha1 + relative path
    source: str         # original path
    kind: str           # "pdf" | "docx" | "xlsx" | "csv" | "txt" | "md" | "html" | "eml"
    text: str
    segments: list[Segment]
```

`documents.parse(path)` picks a parser by suffix. `documents.to_fixtures(docs, out_path, max_chars=...)` writes fixture records. Long documents are split into chunks at segment or paragraph boundaries. Each chunk record carries `doc_id` and `offset` so detector spans can be mapped back to the whole document and then to a page or cell.

### Parsers for the first version

| kind | library | text rules |
| --- | --- | --- |
| PDF | `pypdf` | Text layer only. One segment per page. Pages with no text are reported as "needs OCR" and skipped. |
| DOCX | `python-docx` | Paragraphs and table cells in document order. Headers and footers are separate segments. |
| XLSX | `openpyxl` | One line per row with cells joined by a tab. One segment per cell so a span maps to `Sheet!B7`. |
| CSV | stdlib `csv` | Same layout as XLSX. |
| TXT and MD | none | Read as UTF-8. Markdown is kept as written. |
| HTML | stdlib `html.parser` | Visible text only. Script and style content is dropped. Block elements end a line. |
| EML | stdlib `email` | Headers `From` `To` `Cc` `Subject` as their own segments and then the text body. HTML bodies use the HTML rules. Attachments are parsed recursively when their type is supported. |

I chose thin parsers over [MarkItDown](https://github.com/microsoft/markitdown) or [Docling](https://github.com/docling-project/docling) because both emit Markdown without a map back to source locations. Markdown syntax also adds characters that detectors may tag. Either library can be added later as one more parser if layout matters more than location.

### File-native detectors

Skyflow Detect can take files directly. That path is out of scope for this plan. The `Document` shape leaves room for it because a file-native detector's output can be mapped onto the same segments later.

## Part 2 — Unlabeled fixtures

`fixtures.from_texts(texts, out_path)` and `fixtures.from_documents(docs, out_path)` write fixture records with empty gold. The meta file records `"gold": "none"` and the source paths. `validate_fixtures` accepts empty gold when the meta says so. The report refuses to print precision and recall for a run whose fixtures have no gold and points the user to Parts 4 to 6.

## Part 3 — LLM clients

A small module `opf_eval/llm/` with one protocol:

```python
class LLMClient(Protocol):
    provider: str
    model: str
    remote: bool
    def complete_json(self, system: str, user: str, schema: dict) -> dict: ...
```

### Backends

| backend | operator | SDK and client | structured output | notes |
| --- | --- | --- | --- | --- |
| `anthropic` | Anthropic | `anthropic.Anthropic()` | `client.messages.parse()` with a JSON schema | Credentials from `ANTHROPIC_API_KEY` or an `ant auth login` profile. |
| `anthropic_bedrock` | own AWS account | `anthropic.AnthropicBedrockMantle(aws_region=...)` | same as above | Model ids take an `anthropic.` prefix such as `anthropic.claude-opus-5-5`. AWS credentials. |
| `anthropic_vertex` | own GCP project | `anthropic.AnthropicVertex(project_id=..., region="global")` | same as above | Model ids are the bare first-party ids such as `claude-opus-5-5`. Google application default credentials. Needs the `anthropic[vertex]` extra. |
| `openai` | OpenAI | `openai.OpenAI()` | JSON schema `response_format` | Credentials from `OPENAI_API_KEY`. |
| `openai_bedrock` | own AWS account | `openai.OpenAI(base_url="https://bedrock-runtime.<region>.amazonaws.com/openai/v1")` | validate-and-retry until JSON schema support is confirmed | Needs a Bedrock API key. See the [Bedrock OpenAI models page](https://docs.aws.eu/bedrock/latest/userguide/model-parameters-openai.html) and the [Chat Completions guide](https://docs.aws.amazon.com/bedrock/latest/userguide/inference-chat-completions.html). |
| `openai_vertex` | own GCP project | `openai.OpenAI(base_url="https://aiplatform.googleapis.com/v1/projects/<project>/locations/<region>/endpoints/openapi")` | validate-and-retry until JSON schema support is confirmed | Uses a short-lived Google access token as the API key. See the [Vertex AI OpenAI models page](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/maas/openai). |
| `openai_compatible` | whoever runs the server | `openai.OpenAI(base_url=...)` | JSON schema when the server supports it and validate-and-retry otherwise | Covers LM Studio and vLLM and Ollama. Plan 04's local LLM detector reuses this backend. |

Claude goes through the Anthropic SDK on every platform. It does not go through an OpenAI-compatible shim. Every response is validated against the schema. A failed validation retries once with the error message and is then recorded as an error row.

Claude Opus 5.5 can end a response with `stop_reason: "refusal"`. The client records that as an error row instead of an empty answer. On the first-party API the client opts into server-side fallbacks. On Bedrock and Vertex that parameter is not available and a refusal stays an error row.

### Models

| family | default | also supported | where available |
| --- | --- | --- | --- |
| Claude | `claude-opus-5-5` | any Claude model id | Anthropic API and Bedrock and Vertex AI. |
| OpenAI | `gpt-6.1-sol` | `gpt-6-luna` and `gpt-5.6-sol` and `gpt-5.6-luna` | OpenAI API for all four. Bedrock lists GPT-6.1 Sol. Vertex AI lists only the open-weight gpt-oss models. |
| OpenAI open weights | none | `gpt-oss-120b` and `gpt-oss-20b` | Bedrock and Vertex AI and any local OpenAI-compatible server. |

OpenAI's [model list](https://developers.openai.com/api/docs/models/all) has no GPT-6.1 Luna today. The current Luna is GPT-6 Luna. The id table above should be rechecked when PR 2 starts because the Bedrock and Vertex catalogues change often. Proprietary GPT models are not on Vertex AI. A team that must keep data inside its GCP project would use Claude on Vertex or gpt-oss on Vertex.

### Vertex AI configuration

1. The GCP project id comes from `GOOGLE_CLOUD_PROJECT` through `nb.load_secrets()`. It is not written into the repo because the repo is public.
2. Claude on Vertex uses the `us` multi-region by default. That keeps processing in the US without tying the notebook to one region's capacity. PR 2 checks whether Claude Opus 5.5 is offered in a US west region such as `us-west1` and switches the default if it is.
3. gpt-oss on Vertex uses `us-central1` because that is the region the Vertex documentation lists for it. PR 2 checks for a US west region the same way.
4. Credentials are Google application default credentials. In Colab that is `google.colab.auth.authenticate_user()`. On a laptop it is `gcloud auth application-default login`.

### Bedrock is a later phase

The `anthropic_bedrock` and `openai_bedrock` backends wait until an AWS account and keys exist. They move to PR 5. The `LLMClient` protocol and the response cache do not change when they are added.

## Part 4 — Silver labels from LLMs

`opf_eval.silver.generate(fixtures, labelers, out_path, labels=...)` asks each labeler for the PII in each record.

1. **What the LLM returns.** A list of `{value, label, context}` items. LLMs are unreliable at character offsets, so the harness finds offsets itself by searching for `value` in the record text. `context` is a few words around the value and picks the right occurrence when a value appears more than once. Values that cannot be found are logged as dropped.
2. **Label set.** The schema enumerates the coarse canonical labels from the taxonomy as an `enum`. A fine-level option asks for fine labels instead.
3. **Several labelers.** With two or more labelers the user picks `union` or `majority` or `intersection`. The notebook reports pairwise agreement as span F1 between labelers. Low agreement is a warning that the silver labels are noisy.
4. **Output.** A fixtures file with `gold_spans` filled and meta `gold_source: {"kind": "llm", "labelers": [...], "merge": "majority", "prompt_version": ...}`. The existing report and charts and error analysis then run unchanged. Every heading in the notebook says "silver" instead of "gold".
5. **Calibration.** `silver.calibrate(labeler, dataset="pii_masking_200k", n=100)` scores the labeler against real gold. The notebook shows that F1 next to the silver results.

## Part 5 — LLM review

`opf_eval.review.llm_review(run_dir, fixtures, reviewer)` estimates precision and recall without writing silver labels.

1. **Precision pass.** All detectors' spans for a record are pooled and deduplicated. The reviewer judges each unique span once as `correct` or `wrong_label` or `boundary` or `not_pii`. Each verdict is then credited to every detector that produced the span. Estimated precision for a detector is the share of its spans judged `correct`. A lenient view also counts `boundary`.
2. **Recall pass.** The reviewer sees the text with every pooled span marked. It lists PII that no detector marked. A detector's estimated recall is its correct spans divided by the sum of all correct pooled spans and all reviewer-found misses. This is the pooling method used in information retrieval evaluation.
3. **Bias.** Misses the reviewer also overlooks are invisible. Estimated recall is therefore an upper bound. The notebook prints that sentence under the table.
4. **Output.** `review_llm_<reviewer>.jsonl` in the run directory holds one row per judged span and one row per found miss. A summary table gives estimated precision and recall per detector. A list of disagreements links each one back to its text.

## Part 6 — Classifier review with Clef-flash

Typed decision models answer questions with calibrated probabilities instead of free text. They are a cheaper and faster second opinion to the LLM reviewer. This plan implements Cloudflare's Clef-flash first because it runs locally. TypeSafe's Jev is parked because it offers zero data retention only to enterprise customers. The review module keeps a backend interface so Jev can be added later without changing the notebook.

### Questions

1. **Per span.** A yes/no question asks "Is `<span>` in this context personal data of type `<label>`?". A choice question asks which canonical label fits best. These give a precision estimate with a probability attached to each judgement.
2. **Per segment.** Each sentence or line is shown with detected spans replaced by placeholders. A yes/no question asks "Does this text still contain personal data?". A choice question asks which type. These give a residual-PII rate per detector. Classifiers return no spans, so recall is measured per segment and not per span.
3. **Threshold.** A probability threshold turns answers into verdicts. The default is 0.5. The calibration cell shows precision and recall estimates across thresholds against real gold.

### Backends

| backend | status | where it runs | how |
| --- | --- | --- | --- |
| `clef_flash` | this plan | local | Open weights [`Cloudflare/clef-flash`](https://huggingface.co/Cloudflare/clef-flash) under Apache-2.0. Built on Qwen3.5-9B. Loaded with the model repo's own `joint_schema_model.py` loader and its separate `joint_head.safetensors` decision head. |
| `clef` | later | local | `Cloudflare/clef` on a 27B backbone. Needs a large GPU. |
| `jev` | parked | TypeSafe's hosted API | `typesafe-sdk` package with `TYPESAFE_API_KEY`. Parked because of the data retention terms on the [legal page](https://docs.typesafe.ai/legal.md). |

### Running Clef-flash in Colab

Clef-flash has 9.41 billion parameters stored in bf16. The weights alone take about 18.8 GB. A classification pass is one forward pass with no generation, so short segments add only a little memory on top.

| runtime | memory | verdict |
| --- | --- | --- |
| L4 | 24 GB with native bf16 | **Primary target.** The weights fit with about 3 GB to spare. Segments stay short and batches stay small. |
| T4 | 16 GB with no native bf16 | Possible only with 8-bit or 4-bit weights through `bitsandbytes`. 8-bit needs about 9.5 GB. The decision head stays in full precision. The repo's loader must be patched to accept a quantization config. Quantization may shift the probabilities, so PR 4 measures T4 output against L4 output on the same records before the notebook offers it. |
| TPU v5e-1 | 16 GB of HBM on one chip | **Not recommended.** It has the same memory problem as the T4. The loader targets CUDA. `bitsandbytes` does not run on TPUs. Making it work would mean porting the loader to `torch_xla` or JAX. |
| A100 | 40 GB or 80 GB | Works without changes. Paid Colab tiers only. |
| Apple Silicon | unified memory | Should work on MPS with 32 GB or more. Unverified. |

The notebook checks the runtime's GPU memory before loading the model. It explains the options when the runtime is too small.

### Dependency change

Clef-flash needs `torch` 2.11 or later and `transformers` 5.10.2 or later. The lock file has `torch` 2.11.0 and `transformers` 5.1.0 today. PR 4 raises `transformers` and probably `gliner` to 0.2.29 or later because that release accepts `transformers` below 5.17. `openmed` sets no upper bound. Every existing detector is rerun on the 100-record sample after the bump to check that its output is unchanged. The saved notebook baseline is rebuilt if anything moves.

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
| `eval/src/opf_eval/llm/__init__.py` and `anthropic.py` and `openai.py` and `cache.py` | new. Each SDK module covers the first-party and Bedrock and Vertex clients. |
| `eval/src/opf_eval/silver.py` | new |
| `eval/src/opf_eval/review/llm.py` and `review/classifier.py` and `review/clef.py` | new |
| `eval/src/opf_eval/report.py` | refuse P/R without gold and label silver gold as silver |
| `eval/src/opf_eval/nb.py` | helpers for upload and Drive folders and the remote-provider guard |
| `eval/pyproject.toml` | extras `files` and `llm` and `clef`. Raise `transformers` for Clef-flash. |
| `eval/tests/` | parser offset round-trips on small fixture files. Offset finding and merging for silver labels. Review maths with a stub reviewer. Cache keys. |
| `notebooks/07_evaluate_without_labels.ipynb` and `notebooks/README.md` | new notebook and index row |

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
