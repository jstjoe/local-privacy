# Plan 04 — LLM-as-detector via LM Studio (OpenAI-compatible API)

## Why

Generative LLMs do PII extraction differently from token classifiers — they reason about context, can handle ambiguous cases, and aren't bound by a fixed entity ontology. They're slow and not cheap to deploy at scale, but as a research data point they tell us the **quality ceiling** of "intelligent" PII detection.

Using LM Studio means: fully local, free, OpenAI-compatible API on `localhost:1234`. The same harness code works against any future OpenAI-compatible endpoint (vLLM, Ollama, etc.) by swapping `base_url`.

## Scope

- New `LMStudioDetector` using the `openai` Python SDK pointed at LM Studio
- Pluggable model — whichever the user has loaded in LM Studio
- Structured JSON output via `response_format` so parsing is deterministic
- One-detector-at-a-time in the runner (sequential, low concurrency to avoid VRAM thrash)

## Files

- **Add** `eval/src/opf_eval/detectors/lmstudio.py` — `LMStudioDetector` class
- **Modify** `eval/src/opf_eval/detectors/__init__.py` — export
- **Modify** `eval/src/opf_eval/runner.py` — register `lmstudio` name + `--lmstudio-model` arg
- **Modify** `eval/src/opf_eval/taxonomy.py` — add canonical-label name list for prompting
- **Modify** `eval/pyproject.toml` — add `openai>=1.40`

## Architecture

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio",  # LM Studio ignores the key but SDK requires one
)

response = client.chat.completions.create(
    model="meta-llama-3.1-8b-instruct",  # or whatever's loaded
    response_format={"type": "json_schema", "json_schema": SPAN_SCHEMA},
    messages=[{"role": "user", "content": PROMPT.format(text=text)}],
)
spans = json.loads(response.choices[0].message.content)["spans"]
```

## Prompt design

Single-shot prompt that asks for span offsets in canonical-label form. Critical: the model needs to return character offsets, not just the entity text. Most chat models are bad at this — may need to fall back to "return entity text and we re-locate it ourselves."

Two strategies, try both:

**Strategy A — model returns offsets directly:**
```
Find PII in the text below. Return JSON: {"spans": [{"label": "PERSON|EMAIL|...", "start": int, "end": int}]}
Available labels: PERSON, EMAIL, PHONE, ADDRESS, URL, DATE, ACCOUNT, SECRET, USERNAME, DEMOGRAPHIC.
Text: {text}
```

**Strategy B — model returns text, we locate (more robust):**
```
... Return JSON: {"spans": [{"label": "...", "text": "..."}]}
```
Then post-process: for each returned `text`, find first occurrence in source string, derive start/end. Handles models that can't count characters.

Start with B. If a strong model handles A reliably, switch — A is faster.

## Risks / open questions

- **Latency.** Small local model (8B) on a Mac: probably 1-3 sec/example. 100 examples = 5-10 min. 1k = 1-3 hours. 5k = uncomfortable.
- **VRAM.** Whatever LM Studio has loaded. User picks the model based on what fits.
- **Hallucinated spans.** Model invents PII that isn't in the text → must verify offsets match source. If `text[start:end] != reported_text`, drop the span.
- **Offset accuracy.** Even the verify-and-drop approach loses spans the model identified but couldn't locate. Track this as a separate stat.
- **Model dependence.** Llama 3.1 8B may be too small for reliable structured output. Qwen 2.5 14B or Llama 3.3 70B (if VRAM allows) would do better. Worth trying 2-3 models and reporting separately.
- **Per-call cost in API mode.** When this harness later runs against the actual OpenAI/Anthropic API instead of LM Studio, log token counts so we can estimate cost. Add a `usage` field to the result dict.

## Verification

```sh
# Start LM Studio, load a model (e.g. Llama 3.3 70B Instruct), enable the local server on :1234

uv sync
uv run python -m opf_eval.runner \
    --fixtures eval/data/sample_100.jsonl \
    --detectors opf,skyflow_minimal,lmstudio \
    --lmstudio-model "meta-llama-3.3-70b-instruct" \
    --out eval/results/runs/run_100_llm/
uv run python -m opf_eval.report --run eval/results/runs/run_100_llm/ --fixtures eval/data/sample_100.jsonl
```

Watch for:
- F1 vs OPF/Skyflow — is "smart but slow" actually higher quality on edge cases?
- Latency p50/p95 — confirms feasibility for production
- Span-locate failure rate (% of model-returned spans we couldn't pin to source text)
- Prompt sensitivity — re-run with a slightly different prompt, see how much F1 moves

## Effort

~3-4 hours integration + 1 short benchmark. More if multiple models tested.

## Out of scope

- Few-shot prompting (start zero-shot)
- Self-consistency / multiple completions per example
- Fine-tuning the local model for PII (separate plan)
- Long-context evaluation (PII-Masking-300k examples are short)
- Routing across multiple LM Studio models (one at a time)

## Future generalization

Same `LMStudioDetector` works against any OpenAI-compatible endpoint by passing different `base_url` + `api_key`:
- LM Studio: `http://localhost:1234/v1` + dummy key
- vLLM: `http://your-vllm-server/v1`
- Ollama: `http://localhost:11434/v1`
- Real OpenAI: `https://api.openai.com/v1` + real key
- Anthropic: not OpenAI-compatible by default — would need a separate `AnthropicDetector` or a proxy

Worth refactoring once you've used it twice. For now, name it `LMStudioDetector` with hardcoded localhost defaults; rename to `OpenAICompatibleDetector` later if you add more endpoints.
