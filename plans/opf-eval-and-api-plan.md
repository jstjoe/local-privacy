# OpenAI Privacy Filter — Local Eval & API Plan

## 1. What we learned

### Model

- **OpenAI Privacy Filter (OPF)** — open-weight PII detector, Apache 2.0, released April 2026
- 1.5B total params, **50M active** (sparse MoE, top-4 of 128 experts)
- Bidirectional token classifier built on a gpt-oss-style backbone; LM head replaced with a 33-class BIOES head over 8 PII categories
- Constrained Viterbi decoder produces coherent spans in a single forward pass
- 128k context window; reports 96% F1 on PII-Masking-300k
- Categories: `account_number`, `private_address`, `private_email`, `private_person`, `private_phone`, `private_url`, `private_date`, `secret`
- **Not a generative model** — no GGUF, no Ollama, no llama.cpp, no MLX-LM. Transformers/PyTorch only.

### Mac M-series specifics

- Default runtime uses Triton-optimized MoE kernels; Triton has no stable Apple Silicon support
- **Workaround:** set `OPF_MOE_TRITON=0` to use the vanilla PyTorch MoE fallback
- With that env var, both `--device cpu` and `--device mps` work
- Without it, only `--device cpu` works
- First-run downloads ~2.8 GB to `~/.opf/privacy_filter`
- Observed ~500–700 ms per short input on CPU; MPS comparison still pending

### CLI surface (`opf redact`)

| Flag | Purpose |
| --- | --- |
| `--device cuda\|cpu\|mps` | hardware target |
| `--decode-mode viterbi\|argmax` | viterbi = quality (default); argmax = fast baseline |
| `--output-mode typed\|redacted` | typed keeps category labels; redacted collapses to one |
| `--format text\|json` | string vs. structured object |
| `--no-print-color-coded-text` | **required when piping JSON to jq** (otherwise ANSI escapes break the parser) |
| `--viterbi-calibration-path` | precision/recall operating-point tuning (default = zero biases) |
| `--discard-overlapping-predicted-spans` | overlap cleanup |
| `--n-ctx` | context window override |
| `-f / --text-file` | file input |

No native per-category filter — filter spans yourself in post-processing.

### Python API (`opf._api.OPF`)

- Constructor: `OPF(model, device, output_mode, decode_mode, trim_whitespace)`
- Main method: `redact(text, decode=None) -> RedactionResult`
- `RedactionResult.to_dict()` → `{schema_version, summary, text, detected_spans, redacted_text}`
- Detected span shape: `{label, start, end, text, placeholder}`
- `INHERIT` sentinel + decoder cache make per-call decode overrides cheap
- Module-level `redact()` shortcut exists but reloads the model each call — never use it in a server

### Observed failure modes

- **Over-redaction of surrounding context.** On `"Joe at joe@example.com lives in Elgin, TX. Phone: 555-1234."`, OPF marked `"lives in Elgin, TX"` (verb included) as `private_address` rather than just `"Elgin, TX"`.
- Consistent with model-card warnings about ambiguous local context.
- Likely addressable via Viterbi calibration tuning or fine-tuning.

---

## 2. Benchmark plan: OPF vs. Skyflow Detect API

**Goal:** measure (a) accuracy and (b) performance of local OPF against the Skyflow Detect API on representative fixtures.

### Phase 1 — Fixture set

- Start with a **PII-Masking-300k** sample (~5k examples) to anchor against OPF's reported number
- Layer in internal fixtures:
  - TeamTeacher-style educator/student text
  - Public clinical / support-transcript corpora
  - Edge cases: mixed formats, code blocks, tables, multilingual snippets
- Schema per example: `{id, text, gold_spans: [{label, start, end}]}`

### Phase 2 — Taxonomy mapping

OPF's 8 categories and Skyflow's entity types don't line up 1:1. Build an explicit mapping table before any metrics work — example:

| OPF | Skyflow |
| --- | --- |
| `private_person` | `NAME` family |
| `private_email` | `EMAIL` |
| `private_phone` | `PHONE_NUMBER` |
| `private_address` | `ADDRESS` family |
| `account_number` | `CREDIT_CARD` ∪ `BANK_ACCOUNT` ∪ … (one-to-many) |
| `secret` | no direct equivalent — exclude from per-category metrics or map to closest |

For per-category precision/recall, project both systems into a chosen canonical taxonomy. For "did either system catch this at all" metrics, compare at the span-overlap level regardless of label.

### Phase 3 — Test harness

Single Python tool with a `Detector` interface:

```python
class Detector(Protocol):
    def detect(self, text: str) -> list[Span]: ...
```

Two implementations: `OPFDetector` (wraps `opf._api.OPF`) and `SkyflowDetector` (HTTP). Driver iterates fixtures, calls each detector, records latency + spans + errors to JSONL.

### Phase 4 — Metrics

**Accuracy:**
- Span-level precision / recall / F1, both exact-match and partial-overlap (≥50% IoU)
- Per-category breakdown after taxonomy projection
- Confusion matrix across labels
- Disagreement set: OPF-caught / Skyflow-missed and vice versa (the most instructive output)

**Performance:**
- p50 / p95 / p99 latency per backend
- Throughput single-request and batch-of-N
- Skyflow is network-bound — report wall-clock plus (if available) server-side timing for fair compute comparison
- OPF cold-start time recorded once for context

### Phase 5 — Output

- Markdown report with headline table + per-category breakdowns
- Raw results JSONL committed for reproducibility
- Curated disagreement examples for qualitative review

### Caveats to call out in any writeup

- Open-weight local model vs. productionized hosted API — differences are partly architectural, partly deployment
- OPF is fine-tunable to a target distribution; Skyflow has its own customization path
- Network latency to Skyflow varies by region and is part of the real cost

---

## 3. REST API plan

**Goal:** local FastAPI server wrapping OPF as a privacy preprocessing layer; later containerized for Cloud Run.

### Architecture

```
[client]
   ↓ HTTP
[FastAPI app]
   ├─ startup: load OPF once
   ├─ POST /redact   → masked text + spans
   ├─ POST /detect   → spans only
   └─ POST /redact   (+ categories filter)
```

### Phase 1 — Minimal scaffold

```python
from fastapi import FastAPI
from pydantic import BaseModel
from opf._api import OPF

app = FastAPI()
model = OPF(device="cpu", decode_mode="viterbi", output_mode="typed")

class Req(BaseModel):
    text: str

@app.post("/redact")
def redact(r: Req):
    return model.redact(r.text).to_dict()
```

Run with `uvicorn server:app --reload`.

### Phase 2 — Real-world surface

- **Category filtering** — `categories: ["private_email", "private_phone"]` filters `detected_spans` and regenerates `redacted_text` from the filtered set
- **Per-request decode mode** — toggle argmax/viterbi without restart (decoder cache makes this cheap)
- **Per-request output mode** — typed vs. redacted
- **`/health`** — returns model load status + checkpoint version
- **`/redact/batch`** (optional) — list of texts in a single call to amortize overhead

### Phase 3 — Production concerns

- **Concurrency:** OPF instance is probably not thread-safe with the shared decoder cache. Wrap inference in an `asyncio.Lock`, or run a process pool. Single-process single-worker is fine for local dev.
- **Observability:** log `latency_ms`, span count by label, decode mode used per request
- **Auth:** API key header; skip for local-only
- **Containerization:** Dockerfile with checkpoint baked in (or downloaded at first run); set `OPF_MOE_TRITON=0` for any non-CUDA image
- **Deployment:** Cloud Run with `min-instances=1` to avoid cold starts (consistent with prior GLiNER deployment analysis)

### Phase 4 — Integration patterns worth exploring

- **Pre-LLM scrubbing layer:** `text → /redact → safe_text → LLM provider` (this is the headline use case)
- **OTel processor analog:** PII-tokenize before logs reach collectors — echoes the Skyflow OTel pattern
- **Reversible mode:** keep a local placeholder ↔ original map so LLM responses can be rehydrated client-side

---

## Open questions / next moves

- [ ] Run the load-once benchmark script on representative-length inputs to settle CPU vs. MPS
- [ ] Pick an initial fixture set for the Skyflow benchmark — start with ~100 examples before scaling
- [ ] Source or train a Viterbi calibration JSON — the zero-bias default may not be the right operating point for our data
- [ ] Investigate the over-redaction case more broadly — is the `"lives in"` issue systematic?
- [ ] Draft the OPF ↔ Skyflow taxonomy mapping table before any metrics work begins
