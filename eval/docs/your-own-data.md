# Evaluate detectors on your own data without gold labels

The harness can score detectors on your own text and files even though they have no gold spans. It parses the files to text and runs the detectors as usual. It then estimates precision and recall in three ways that you can mix:

1. **Silver labels.** One or more LLMs annotate the records and the normal report scores the detectors against those annotations.
2. **LLM review.** An LLM judges every span the detectors found and lists the personal data that every detector missed.
3. **Classifier review.** Cloudflare's Clef-flash decision model runs locally and answers yes/no and choice questions about each span and each sentence.

Every estimate has a bias that the sections below state next to it. Calibrate each method on a dataset with real gold before you trust it on unlabeled data. [Notebook 07](../../notebooks/07_evaluate_without_labels.ipynb) walks through the whole flow and the design is in [plan 13](../../plans/13-unlabeled-data-notebook.md).

## Install the extras

The core package has no file parsers and no LLM SDKs. Install the extras you need:

```sh
uv sync --all-packages --all-extras         # in a checkout of this repo
pip install 'opf-eval[files,llm,clef]'      # anywhere else
```

| extra | installs | needed for |
| --- | --- | --- |
| `files` | `pypdf`, `python-docx`, `openpyxl` | PDF, DOCX and XLSX files. CSV, TSV, TXT, MD, HTML and EML use the standard library. |
| `llm` | `anthropic[vertex]`, `openai`, `google-auth` | silver labels and LLM review |
| `clef` | `transformers>=5.10.2`, `pillow`, `bitsandbytes` (Linux only) | Clef-flash review |

The `clef` extra deliberately leaves out `torchvision` because its latest release would pull a newer `torch` for every detector. Colab already ships it. Without it `ClefBackend` loads the model with a plain tokenizer, which is all text review needs.

## Parse files into documents

`opf_eval.documents` turns each file into a `Document` that holds plain text plus a list of `Segment`s. Each segment maps a character range back to its place in the file, so a detected span can be shown as "page 3" or "People!B2" instead of a bare offset.

```python
from opf_eval import documents

docs, skipped = documents.parse_many("inputs/")    # a folder, a file or a list of either
for s in skipped:
    print("skipped", s["path"], "-", s["reason"])
for doc in docs:
    print(doc.id, doc.kind, len(doc.text), doc.warnings)

doc = documents.parse("inputs/people.xlsx")
i = doc.text.index("jane.doe@example.com")
doc.locate(i, i + 20)                               # [{"sheet": "People", "cell": "B2"}]

pasted = documents.from_text("Call Jane on 555-0100")   # kind "text"
```

`parse` picks a parser by suffix and raises on an unsupported suffix or a missing file. `parse_many` never raises on a bad file. It walks folders recursively in sorted order and ignores hidden files and folders. It lists these cases in `skipped` with a reason:

1. The suffix is not supported.
2. The file is an Office lock file such as `~$report.docx`.
3. The path does not exist.
4. The file has the same id as a file already parsed.
5. The parser failed, for example because an extra is missing or a PDF needs a password.

A document id is `"<first 12 hex digits of the file's sha1>:<path relative to the parse root>"`. It stays the same across runs and machines as long as the file content and its relative path do not change.

### Supported files and location shapes

Every `where` below is exactly what `Document.locate` returns for a span in that kind of file.

| suffix | kind | `where` | text layout |
| --- | --- | --- | --- |
| `.pdf` | `pdf` | `{"page": n}` with `n` from 1 | The text layer only with one segment per page. Pages are joined by a blank line. |
| `.docx` | `docx` | `{"paragraph": i}` | Body paragraphs in document order. `i` is the 0-based index into `doc.paragraphs` with blank paragraphs counted. |
| | | `{"table": t, "row": r, "col": c}` | A table cell. The indexes are 0-based and match `doc.tables[t].rows[r].cells[c]`. Cells of a row are joined by a tab. |
| | | `{"content_control": k, "paragraph": j}` or `{"content_control": k, "table": t, "row": r, "col": c}` | Content inside the k-th content control (`w:sdt`) that sits directly in the body. Each control numbers its own paragraphs and tables from 0. |
| | | `{"part": "header", "section": s}` or `{"part": "footer", "section": s}` | Headers and footers come last after a blank line. First-page and even-page variants add `"variant": "first"` or `"variant": "even"`. |
| `.xlsx` `.xlsm` | `xlsx` | `{"sheet": "People", "cell": "B2"}` | One line per non-empty row and one segment per non-empty cell. Sheets are separated by a blank line. |
| `.csv` `.tsv` | `csv` | `{"row": 2, "column": "email"}` | The same layout as XLSX. Row 1 is the header row and `column` is the header name, or the column letter when the header cell is empty. |
| `.txt` `.md` `.markdown` | `txt` or `md` | `{"line": n}` with `n` from 1 | The file as written with one segment per non-blank line. Markdown syntax is kept. |
| `.html` `.htm` | `html` | `{"line": n}` | Visible text only with one segment per output line. |
| `.eml` | `eml` | `{"part": "header", "name": "From"}` | The headers From, To, Cc, Bcc, Reply-To, Subject and Date, each on its own line. The segment covers the value and not the "From: " label. |
| | | `{"part": "body", "line": n}` | The text/plain body, or the text/html body read with the HTML rules. |
| | | `{"attachment": "scan.pdf", ...}` | An attachment with a supported suffix, parsed with its own parser. The inner location is kept, so a page of an attached PDF is `{"attachment": "scan.pdf", "page": 1}`. An attachment inside an attached email gets a path such as `"attached.eml/scan.pdf"`. |
| pasted text | `text` | `{"line": n}` | `documents.from_text` keeps the text as given. |

These parser rules affect what detectors see:

1. **PDF.** A page with no text layer is skipped with the warning `page N has no text layer (needs OCR)`. OCR is out of scope. A PDF encrypted with an empty password opens and any other encrypted PDF is skipped.
2. **DOCX.** Paragraph text is read from the XML so tracked insertions, content controls and fields count while deleted text does not. A merged cell is emitted once. When some body text sits in a layout the parser cannot place, a warning says how many characters are missing.
3. **XLSX.** Formula cells show their cached value. Integral numbers drop the `.0` so a phone number stored as a number stays recognisable. A date at midnight shows as `1990-05-17`. Empty cells are left out of the line instead of becoming runs of tabs.
4. **CSV.** The delimiter of a `.csv` file is sniffed from comma, semicolon, tab and pipe. A `.tsv` file always uses tabs. A file without a header row still uses the values of row 1 as column names.
5. **Text encodings.** TXT, MD, CSV and HTML files are decoded by `documents.text.decode_text`. A UTF-8, UTF-16 or UTF-32 byte-order mark picks that encoding. UTF-16 without a mark is recognised by its NUL bytes. Other files are read as UTF-8 and fall back to Windows-1252 with a warning when they are not valid UTF-8.
6. **HTML.** Script, style, head, title, template and noscript content is dropped. Block elements end a line. Table cells on one row are separated by a tab. Line breaks inside `pre` are kept.
7. **EML.** Unsupported attachments only add a warning. Attachments nested more than 4 levels deep are skipped.

A file that yields no text at all gets the warning `no text was extracted`. Warnings never stop a parse. They appear in `Document.warnings` and in the fixtures metadata.

### Chunking

Many detectors have input limits, so long documents are split into chunks with `documents.chunk(doc, max_chars=4000)`. A cut falls on the last segment boundary that fits, so a page or cell or line is never split across chunks. Only a single segment longer than `max_chars` is cut inside, at its last newline or sentence end or space before the limit. The chunks cover the whole text with no gaps or overlaps.

## Write unlabeled fixtures

Parsed documents and pasted text become ordinary fixture records with `gold_spans: []`. The runner, `nb.ensure_run` and the run directory work on them unchanged.

```python
from opf_eval import fixtures

fx = fixtures.from_documents(docs, "data/custom/inputs.jsonl", max_chars=4000)
fx = fixtures.from_texts(["Call Jane on 555-0100", "..."], "data/custom/pasted.jsonl")
fixtures.is_unlabeled(fx)          # True
```

`fixtures.from_documents` writes one record per chunk: `{"id": "<doc id>#<k>", "text", "language", "gold_spans": [], "doc_id", "offset"}`. The `offset` is where the chunk starts in the document's text. Chunks that hold only whitespace are left out. `fixtures.from_texts` gives a list of strings the ids `text-0001`, `text-0002` and so on, and uses the keys of a mapping as ids. It leaves blank texts out but keeps their number so later ids do not shift.

Both functions write a `.meta.json` sidecar like this one:

```json
{"version": 2, "dataset": "custom:inputs", "source": "files", "gold": "none",
 "vocab_key": null, "labels": ["ACCOUNT", "ADDRESS", "API_KEY", "..."],
 "n_written": 12, "documents": {"<doc id>": {"source": "...", "kind": "pdf", "n_chunks": 2, "warnings": []}},
 "max_chars": 4000, "sha256": "...", "created_at": "..."}
```

`"gold": "none"` is the value of `fixtures.UNLABELED` and marks the fixtures as unlabeled. `labels` is the target label set. Detectors are asked to find these labels and silver labels and reviews are scored on them. It defaults to `fixtures.default_target_labels("fine")`, which is every coarse category and fine sub-type. `from_texts` writes an empty `documents` map and adds `n_blank_skipped`. It has no `max_chars`. `fixtures.validate_fixtures` sets `FixtureReport.unlabeled` from this sidecar and accepts the empty gold.

### The `.documents.jsonl` sidecar

`from_documents` also writes `<fixtures file name>.documents.jsonl` next to the fixtures, so `inputs.jsonl` gets `inputs.jsonl.documents.jsonl`. It holds each document's id, source, kind, text length, segment map and warnings. `fixtures.documents_path(fx)` returns its path.

The sidecar lets you put spans back on pages and cells after a Colab kernel restart without parsing the files again. `fixtures.load_documents(fx)` rebuilds the full `Document`s from the chunk texts and the sidecar. `fixtures.map_spans_to_documents(fx, run_dir, detector)` shifts each span by its chunk's offset and fills `where` from the sidecar when you pass no `docs`:

```python
rows = fixtures.map_spans_to_documents(fx, run_dir, "presidio")
# {"doc_id", "source", "detector", "label", "fine_label", "text",
#  "doc_start", "doc_end", "where": [{"page": 3}], "chunk_id"}
```

`where` is empty when neither `docs` nor the sidecar is available. `silver.generate` copies the sidecar next to the silver fixtures so the same lookup works on silver files.

## Run detectors

Point an unsaved session at the fixtures and run the detectors as in notebook 02:

```python
from opf_eval import nb

S = nb.Session(fixtures=str(fx), detectors=["presidio", "gliner"])
run_dir = nb.ensure_run(S)
```

The `fixtures` override applies only to the notebook that sets it. `nb.session(fixtures=...)` saves every other field but never `fixtures`, so notebooks 02, 03 and 05 keep loading their labeled dataset sample. `ensure_run` notices when the fixtures file changes and runs the detectors again.

Skyflow reads `SKYFLOW_VAULT_URL` and `SKYFLOW_VAULT_ID` and `SKYFLOW_BEARER_TOKEN` from the environment. Notebook 07 calls `nb.load_secrets(*nb.SKYFLOW_SECRETS)` when its `DETECTORS` list holds `skyflow` or `skyflow_full`, which copies them from Colab Secrets or `.env`, and it stops with the missing names before any detector runs. Outside the notebook call `load_secrets` yourself before `ensure_run`.

## Read the report

`report.build_report` and `python -m opf_eval.report` read the `gold` value in the fixtures sidecar:

1. **Unlabeled fixtures** (`"gold": "none"`) get a short note instead of scores. The note says there is no precision or recall and names the three ways to estimate them. It still shows detected-span counts and label coverage.
2. **Silver fixtures** (`"gold": "silver"`) get a report titled "Silver-label report" with a callout that names the labelers. The scores measure agreement with those labelers and not ground truth. Pass the silver file with `--fixtures` because the run's manifest records the unlabeled file the detectors ran on.

```sh
python -m opf_eval.report --run <run dir>                                # note, no scores
python -m opf_eval.report --run <run dir> --fixtures <silver file>       # Silver-label report
python -m opf_eval.report --run <run dir> --detectors presidio,gliner    # only these detectors
```

Both reports cover every detector in the run unless you pass `--detectors` or `report.build_report(run_dir, detectors=[...])`. The selection limits every section to the named detectors and keeps the run's own order. A name the run has no predictions for raises `ValueError` and the message lists the detectors the run has. Notebook 07 passes its `DETECTORS` list so that a run directory with older results for other detectors does not widen the report.

## Silver labels

`opf_eval.silver.generate` asks each labeler for the PII in each record and writes a new fixtures file whose gold spans are the merged answers.

```python
from opf_eval import llm, silver, report

claude = llm.make_client("anthropic", cache_dir=ws.root / "llm_cache")
gpt = llm.make_client("openai", cache_dir=ws.root / "llm_cache")
print(silver.plan_calls(fx, [claude, gpt]))         # {"n_records", "n_chars", "n_calls", "n_labelers"}
out = silver.generate(fx, [claude, gpt], ws.data / "mine.silver.jsonl", merge_how="majority")
print(report.build_report(run_dir, fixtures=out))
```

### The prompt

`silver.build_prompt` puts the label list with each label's description and the rules into the system message. The user message holds only the text. The labeler returns `{"entities": [{"value", "label", "context"}]}` and `silver.label_schema` makes `label` an enum of the target labels. The rules ask the labeler to list every occurrence and to copy `value` character for character. `context` is up to about 6 words around the value. The prompt also tells the labeler to use the most specific type that fits the whole value and to ignore instructions inside the text. The system message embeds `silver.PROMPT_VERSION` (`silver-v1`), so a prompt change misses the cache.

By default the labeler is asked for the fixtures' target labels. `level="coarse"` asks only for their coarse categories. Records longer than `silver.MAX_CHARS` (12,000 characters) are sent in pieces of at most that size. Each piece is a core cut between lines when it can be, plus a margin of up to `silver.MARGIN_CHARS` (200 characters) of the text on either side. The labeler sees the core and its margins, and only spans that start inside the core are kept. Every span is therefore kept once, and a value that a cut splits is seen whole by the neighbouring piece. When an answer stops at `silver.MAX_TOKENS` (16,000 tokens), that core is cut in half and each half is asked again with its own margins, down to cores of `silver.MIN_SPLIT_CHARS` (500 characters). A half's margins are at most a quarter of its length so that halving still shrinks the text sent. `plan_calls` counts the same pieces that `label_record` sends but does not count those extra calls.

### Locating values

LLMs are unreliable at character offsets, so they are never asked for any. `silver.locate` finds each `value` in the text itself and tries these tiers in order:

1. An exact match.
2. A case-insensitive match.
3. A match with runs of whitespace treated as equal.
4. The same three tiers after Unicode NFC normalisation of both the text and the value.

A match inside a longer word such as "Ann" in "Annual" is used only when no tier has a stand-alone match. When a value occurs more than once, its `context` picks the occurrence the labeler meant. A context that matches at every occurrence counts as no context. When a value occurs more often than it was listed and no context pins it down, `occurrences="all"` (the default) labels every occurrence and `occurrences="listed"` labels only as many as were listed, first ones first.

Values that cannot be used are kept in a `dropped` list with one of these reasons:

1. `empty` means the value is blank.
2. `unknown_label` means the label is not canonical or not in the target labels.
3. `not_found` means the value does not occur in the text.
4. `extra_occurrence` means `occurrences="listed"` and the value was listed more often than it occurs.
5. `overlap` means the span overlaps a longer one and was left out.

### Merging several labelers

With two or more labelers `silver.merge` combines their spans for each record. Two spans agree when they share a coarse label and their character ranges have an intersection over union of at least 0.5. Every span any labeler gave is a candidate. Its votes are the number of labelers with a span that agrees with the candidate itself. Agreement never chains through a third span, so a span that agrees with B does not earn a vote from C just because B agrees with C. The `merge_how` argument decides which candidates are kept:

1. `union` keeps a candidate with any vote.
2. `majority` (the default) keeps a candidate with votes from more than half of the labelers.
3. `intersection` keeps a candidate only when every labeler agrees with it.

Kept candidates that overlap are resolved by votes, then by how many labelers gave that exact boundary, then by length. A kept span takes the most common fine label of the agreeing spans. Labelers that errored on a record do not vote on it.

### Agreement

`silver.agreement` gives the strict span F1 between each pair of labelers over the records both answered. Two spans match only when their boundaries are identical and their coarse labels are equal. Low agreement means the silver labels are noisy. The rows are stored in the silver meta under `agreement`.

### Output files

`generate` writes these files next to `out_path`:

1. `<stem>.labeler_<provider>_<model>.jsonl` holds one row per record for each labeler: `{"id", "spans", "dropped", "error"}`. A row also has `fallback_models` when a server-side fallback model answered.
2. `out_path` holds the input records with `gold_spans` set to the merged silver spans.
3. Its `.meta.json` copies the input meta and adds `"gold": "silver"`, `gold_source` (labelers, models, providers, merge mode, occurrences mode and prompt version), `n_errors`, `n_dropped`, `n_fallback`, `error_ids`, `labeler_files` and `agreement`.
4. The `.documents.jsonl` sidecar is copied when the input has one.

A labeler's `LLMError` on a record becomes an error row and the record is merged from the other labelers. Records that every labeler failed on are listed in `error_ids` and the report leaves them out of the scores. Any other exception stops the run. `generate` refuses an `out_path` equal to the input fixtures.

### Calibration

`silver.calibrate(labeler, dataset="pii_masking_200k", n=50, workdir=...)` measures how far one labeler is from real gold. It materializes a dataset sample in `workdir`, strips the gold, runs `generate` on the dataset's annotated labels and scores the silver spans as predictions against the real gold with nervaluate. The result holds precision, recall and F1 for the `strict`, `exact`, `partial` and `ent_type` schemas plus `by_label`. It scores at the coarse level by default. The first call downloads a Hugging Face sample.

## LLM review

`opf_eval.review.llm.review_run` estimates precision and recall without writing silver labels.

```python
from opf_eval import llm
from opf_eval.review import llm as review_llm

reviewer = llm.make_client("anthropic", cache_dir=ws.root / "llm_cache")
print(review_llm.estimate(run_dir))                 # {"n_calls", "n_candidates", "n_chars"}
path = review_llm.review_run(run_dir, reviewer)
review_llm.summarize(run_dir, path)                 # one row per detector
review_llm.disagreements(run_dir, path)             # spans not judged correct, then misses
```

`review.pool.pooled_spans` first pools every detector's spans for a record and keeps each unique `(start, end, coarse label)` once. The reviewer sees the text and a numbered list of these candidates with their labels and a little context. It makes one call per non-blank record with an output budget of 16,000 tokens (`review_llm.MAX_TOKENS`). A record with no candidates gets a call too because the reviewer can still find misses there. When an answer stops at that budget the candidates are cut in half and each half is reviewed against the same whole text, down to a single candidate. The verdicts keep their places in the record. The misses of all halves are joined and each `(start, end)` is kept once, and then a miss is checked against every candidate's verdict. A half lists the other half's PII as missed because it cannot see those candidates, so that check drops such a miss as `covered_by_candidate`. Only a truncated answer is split. Any other failure, and a truncated answer for one candidate or for a record without candidates, becomes an error row. Splits add calls, so `estimate`'s `n_calls` is a lower bound.

### Verdicts

The reviewer gives exactly one verdict per candidate:

1. `correct` means the span is personal data with the right boundaries and the right category.
2. `wrong_label` means the span is personal data but the category is wrong. The reviewer supplies the right label.
3. `boundary` means the category is right but the span is too long or too short.
4. `not_pii` means the span is not personal data of any target category.

The reviewer also lists in `missed` the personal data that no candidate covers. That list includes PII inside a candidate it judged `not_pii` and a second entity that an over-long `boundary` candidate runs into. Missed values are located with `silver.locate`. A located miss is dropped with the reason `covered_by_candidate` in three cases:

1. A candidate judged `correct`, `wrong_label` or `boundary` holds all of it.
2. It overlaps a candidate judged `correct` or `wrong_label`.
3. It overlaps a candidate judged `boundary` that overlaps no candidate judged `correct` or `wrong_label`.

A located miss with the exact boundaries of a candidate judged `not_pii` is dropped with the reason `judged_not_pii` because the call that saw that candidate rejected it. This matters for a split record because each half lists the other half's candidates as missed.

A miss that only partly overlaps a `boundary` candidate which already belongs to another entity stays a miss. For example "Jane Doe met John" judged `boundary` next to "Jane Doe" judged `correct` does not hide a missed "John Roe", so no detector gets credit for John Roe. A candidate that still has no verdict after the client's one retry is written as `unjudged` and left out of every number.

The prompt version is `review-v3`. Version 2 added the rule that PII inside a `not_pii` candidate is listed as missed. Version 3 added the rule about a second entity inside a `boundary` candidate. The system message embeds the version, so a new version misses the cache.

### Precision and pooled recall

`review_llm.summarize` credits each verdict to every detector that produced the span. Each row has `n_spans`, the four verdict counts, `precision`, `recall`, `f1` and `missed_by_all`.

1. **Precision** is the detector's spans judged `correct` divided by its judged spans. With `lenient=True` spans judged `boundary` also count as hits.
2. **Recall** is the true entities the detector found divided by every true entity the review knows of. The true entities are the pooled spans judged `correct`, `wrong_label` or `boundary` plus the reviewer's misses. Overlapping `correct` spans count as one entity. A `wrong_label` or `boundary` span joins the single entity it overlaps most, so an over-long span cannot merge two names into one entity. A detector finds an entity when one of its spans in it was judged `correct`, or `boundary` in the lenient view.

Records the reviewer failed on are left out of every number. Records a detector errored on are left out of that detector's numbers.

**Estimated recall is an upper bound** because PII that every detector and the reviewer overlooked is invisible to the review. The denominator only holds entities that someone found. This is the pooling method from information-retrieval evaluation and it is fair for comparing detectors with each other. It is not an absolute recall.

### Output files

The review writes `run_dir/review_llm_<provider>_<model>.jsonl` with one row per judged span (`kind: "span"`), per located miss (`"missed"`), per unusable miss (`"dropped"`) and per failed record (`"error"`). A `.meta.json` sidecar records the reviewer, prompt version, labels, detectors, reviewed record ids and counts. `n_split_records` counts the records whose candidates had to be split. `summarize` and `disagreements` read only these files, so they rerun without calling the model.

For a dry run without keys use `llm.make_client("stub", responder=review_llm.dry_run_responder)`. It accepts every candidate and finds no misses, so its numbers mean nothing.

## Clef-flash review

`opf_eval.review.classifier.review_run` asks a typed decision model questions about spans and text and turns the answers into per-detector estimates. `review.clef.ClefBackend` runs [Cloudflare's Clef-flash](https://huggingface.co/Cloudflare/clef-flash) locally, so no text leaves the machine. Clef-flash is a 9.4B Qwen3.5 backbone with a small decision head under Apache-2.0. It returns a probability for every option of every question in one forward pass with no text generation.

```python
from opf_eval.review import classifier, clef

print(clef.plan_load())          # {"mode": "bf16", "dtype": "bfloat16", "reason": ..., ...}
backend = clef.ClefBackend()     # downloads about 19 GB once, then loads once per process
path = classifier.review_run(run_dir, backend)
classifier.summarize(run_dir, path)
backend.close()                  # frees the GPU memory
```

`classifier.StubBackend()` answers without a model for tests and dry runs. Requests use the Jev / SystemOne request format so another backend that speaks it can implement `classifier.DecisionBackend`.

### Questions

`classifier.span_request` asks three questions about each unique pooled span. The state shows 300 characters either side of the span with the span wrapped in ⟦ ⟧.

1. `is_pii` is a yes/no question: "Is the text inside ⟦ ⟧ personal or sensitive data?"
2. `label_ok` is a yes/no question: "Is <label> (<description>) the right type for it?"
3. `label` is a choice question: "Which type fits best?" Its options are the coarse target categories plus `NOT_PII`.

`classifier.segment_requests` asks two questions about each segment with one detector's spans replaced by `[LABEL]` placeholders:

1. `residual` is a yes/no question: "Does this text still contain personal or sensitive data that is not already replaced by a [PLACEHOLDER]?"
2. `residual_type` is a choice question over the coarse target categories plus `NONE`.

A segment is one sentence of one line, cut by `classifier.split_segments` with at most 600 characters. A full stop after a common abbreviation or an initial does not end a sentence and no cut falls inside a detected span. Every detector shares the same segment boundaries so their residual rates have the same denominator. Each detector's mask covers all of its spans whatever their label. A segment with nothing left after masking is not sent and is written with `p_residual` 0.0. The questions use coarse categories because 15 options at most keep each request short.

### Precision and residual-PII rate

`classifier.summarize(run_dir, path, *, threshold=None)` turns probabilities into estimates at a threshold. `None` means the threshold stored at review time, which defaults to 0.5. Each row has these estimates:

1. `precision` is the share of the detector's spans with `p_pii` and `p_label_ok` both at or above the threshold.
2. `precision_any_label` is the share with `p_pii` at or above the threshold.
3. `residual_rate` is the share of the detector's segments with `p_residual` at or above the threshold.

Classifiers return no spans, so the residual-PII rate is a recall signal per sentence and not per span. A lower rate is better.

`classifier.threshold_sweep(run_dir, path, gold_fixtures)` calibrates both numbers on a dataset with real gold. For each threshold in 0.3, 0.4, 0.5, 0.6, 0.7 and 0.8 it compares the estimated precision with the true precision (exact boundaries and the same coarse label) and the estimated residual rate with the true one. The `precision_error` and `residual_error` columns are estimate minus truth.

The review writes `run_dir/review_<backend name>.jsonl`, which is `review_clef_flash.jsonl` for Clef-flash, plus a `.meta.json` sidecar with the threshold and labels. A failed request becomes an error row for that request alone.

### GPU memory and load modes

The bf16 weights take about 18.8 GB. `clef.plan_load()` decides how to load the model without loading anything. Its thresholds are in GiB because that is what torch reports. An L4 reports about 22.0 to 22.5 GiB and a T4 about 14.7 to 15 GiB. The bf16 threshold of 21 GiB sits below the smallest L4 reading so that every L4 loads in bf16.

| runtime | `plan_load` mode | notes |
| --- | --- | --- |
| L4, A100, H100 | `bf16` | The primary target. CUDA with at least 21 GiB and native bf16 loads the weights as they are. |
| CUDA with at least 21 GiB but no native bf16, such as a V100 | `fp16` | The float16 weights fit. 8-bit weights would still compute in float16, so they would add quantization error and remove no overflow risk. |
| T4 and other CUDA cards with 12 to 21 GiB | `int8` | 8-bit weights through `bitsandbytes` need about 10.7 GiB because the embeddings and the output layer stay in 16 bits. They compute in bfloat16 on a card with native bf16 and in float16 otherwise. `bitsandbytes` runs only on Linux with CUDA. The probabilities may shift slightly against a bf16 run and this has not been measured yet. |
| CUDA with less than 12 GiB | `too-small` | Use an L4 or a larger runtime. |
| Apple Silicon | `bf16` with at least 32 GiB of unified memory, else `too-small` | Unverified. |
| CPU | `cpu-too-slow` | A 9.4B model on CPU takes minutes per request. |

`ClefBackend()` refuses `too-small` and `cpu-too-slow` with the reason. Pass `mode="bf16"`, `"fp16"`, `"int8"` or `"fp32"` to override it. `nb.gpu_summary()` reports the same GiB unit for the notebook's memory check. In `int8` mode the backend checks for `bitsandbytes` before it downloads the weights.

Float16 activations of a model trained in bf16 can overflow. `ClefBackend.answer` raises on any logit that is not finite, so such a request becomes an error row and never a silent probability. `backend.close()` on the last backend that uses a loaded model also empties the model's parameters. That frees the GPU memory even when the traceback of an interrupted run still holds the model.

`ClefBackend` imports the model repo's own `joint_schema_model.py` from the downloaded snapshot and uses its encoder and answer functions, so requests are encoded exactly as the model was trained. The `loader` argument picks how the model is built:

1. `"release"` uses the repo's `load_release_model`, which needs `torchvision` for its image and video processor.
2. `"text"` loads the same backbone, head and tokenizer through `AutoTokenizer` without `torchvision`.
3. `"auto"` (the default) picks `"release"` when `torchvision` is installed.

One loaded model is kept per path, device and mode for the whole process. Clef-flash needs `transformers` 5.10.2 or later for its Qwen3.5 backbone.

## LLM providers

`opf_eval.llm.make_client(provider, model=None, *, cache_dir=None, allow_remote=None, **kwargs)` builds a client that implements one protocol: `complete_json(system=, user=, schema=)` returns a dict that validates against the schema or raises `LLMError`. Every provider says who sees the text and `llm.describe(client)` prints that in one line, for example `claude-opus-5-5 via Anthropic (hosted: Anthropic sees the text)`.

| provider | who sees the text | default model | credentials |
| --- | --- | --- | --- |
| `anthropic` | Anthropic | `claude-opus-5-5` | `ANTHROPIC_API_KEY` or an `ant auth login` profile |
| `anthropic_vertex` | your GCP project (Vertex AI) | `claude-opus-5-5` | `GOOGLE_CLOUD_PROJECT` and Google application default credentials |
| `openai` | OpenAI | `gpt-6.1-sol` | `OPENAI_API_KEY` |
| `openai_vertex` | your GCP project (Vertex AI) | `openai/gpt-oss-120b-maas` | `GOOGLE_CLOUD_PROJECT` and Google application default credentials |
| `openai_compatible` | the server at `base_url`, which is local when the host is localhost | none, so `model` is required | `OPENAI_COMPATIBLE_API_KEY`, else the string "not-needed" |
| `stub` | nobody because the answers are made up locally | `stub` | none |

These rules apply to every provider:

1. Claude always goes through the Anthropic SDK with structured outputs (`output_config.format` with a JSON schema). The default `effort` is "medium". On the first-party API the client opts into server-side refusal fallbacks when the installed SDK supports them. Vertex AI does not offer them.
2. The OpenAI API gets a JSON-schema `response_format`. Vertex AI and local servers step down from JSON schema to JSON object to prompt-only output when the server rejects a response format.
3. Every answer is validated against the schema. A failed validation retries once with the validator's errors and then raises `LLMError`. Refusals, truncated answers, transport errors and any other failure inside the SDK call also raise `LLMError`, and silver labels and reviews record it as an error row instead of an empty answer.
4. Lone UTF-16 surrogates cannot be encoded as UTF-8. The clients replace them with U+FFFD in the prompts and in the answers, which keeps every offset, and `io.write_jsonl` writes a row that still holds one with ASCII escapes.
5. When a server-side fallback model answers in place of the requested Claude model, the client notes it. Silver labeler rows then carry `fallback_models` and the silver meta counts them in `n_fallback`. Review rows carry `fallback_model` and the review meta counts the records in `n_fallback_records`. The disk cache keeps the fallback model with the answer.
6. `openai_compatible` never sends `OPENAI_API_KEY` because that would hand a real OpenAI key to another host. For the same reason the `openai` provider refuses a `base_url`.
7. `OPENAI_MODELS` lists the documented OpenAI choices `gpt-6.1-sol`, `gpt-6-luna`, `gpt-5.6-sol` and `gpt-5.6-luna`. Any model id is accepted.
8. The Bedrock providers `anthropic_bedrock` and `openai_bedrock` are parked until an AWS account exists, and `make_client` refuses them with that reason.

### Vertex AI

Both Vertex providers read the project from `project_id=`, then from `GOOGLE_CLOUD_PROJECT`, then from the application default credentials. `anthropic_vertex` also reads `ANTHROPIC_VERTEX_PROJECT_ID`. Credentials are Google application default credentials. Create them with `gcloud auth application-default login` on a laptop or call `nb.gcloud_auth()` in Colab, which opens Colab's sign-in prompt.

1. `anthropic_vertex` uses `region=`, then `CLOUD_ML_REGION`, then the `us` multi-region. The multi-region keeps processing in the US without tying the notebook to one region's capacity.
2. `openai_vertex` uses `region=` and falls back to `us-central1` because that is the region the Vertex documentation lists for gpt-oss. A regional location uses the host `<region>-aiplatform.googleapis.com`, `global` uses `aiplatform.googleapis.com`, and `us` or `eu` use `aiplatform.<us|eu>.rep.googleapis.com`. The client refreshes the short-lived Google access token whenever it expires.

Whether Claude Opus 5.5 or gpt-oss is offered in a US west region is still unchecked because no GCP project was available when this was built.

### The response cache contains PII

Pass `cache_dir` to `make_client` so reruns cost nothing and give the same numbers. The notebook and the examples above use `ws.root / "llm_cache"`. Each answer is one JSON file at `<cache_dir>/<provider>/<model>/<key[:2]>/<key>.json`. The key is the sha256 of the provider, model, system prompt, user prompt, schema and schema name, plus backend settings such as Anthropic's effort level or a compatible server's URL. The prompts embed their prompt version, so a prompt change is a new key. Errors are never cached. `client.hits` and `client.misses` count cache use.

The request part of a cache file keeps only metadata and hashes. The response part quotes values from the text because a silver label is the PII value itself. Treat the cache directory as being as sensitive as the data and keep it with the rest of your private files.

## Data exposure policy

During internal development it is acceptable for Colab, OpenAI and Anthropic to see the data that notebook 07 processes. Hosted providers are therefore allowed by default for now. This must be tightened before the notebook is shared more widely, and the Vertex AI backends exist so that the models can run inside the team's own GCP project.

1. **The switch.** `PII_BENCH_ALLOW_REMOTE` gates every hosted LLM call. An unset or empty value means the current default, which is `llm.DEFAULT_ALLOW_REMOTE = True`. The values `1`, `true`, `yes` and `on` allow hosted calls and any other value forbids them, so a typo fails closed. `nb.allow_remote(False)` writes `0` for the current kernel and `nb.allow_remote()` reads the switch.
2. **The guard.** `make_client` refuses a hosted provider when the switch is off and names the operator that would have seen the data. It does this before any credentials are read.
3. **Local options.** `openai_compatible` with a localhost `base_url` and Clef-flash keep the text on the machine. On Colab the machine itself belongs to Google.
4. **Before the hosted calls.** Notebook 07 prints `llm.describe(client)` and a call count from `silver.plan_calls` or `review.llm.estimate` before the hosted calls of options A and B. The calibration in section 8 prints the same line and a call count before each silver labeler and before the reviewer. It skips a hosted labeler when `PII_BENCH_ALLOW_REMOTE` has been switched off since option A ran.
5. **Secrets.** `nb.load_secrets(*nb.LLM_SECRETS)` copies `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GOOGLE_CLOUD_PROJECT` and `OPENAI_COMPATIBLE_API_KEY` from Colab Secrets or a local `.env` file into the environment without printing them. `OPENAI_COMPATIBLE_API_KEY` is needed only for an `openai_compatible` server that asks for a key. That backend never reads `OPENAI_API_KEY` so a real OpenAI key is never sent to another host. The GCP project id is kept out of the repo because the repo is public.
