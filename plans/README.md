# Plans

Side-quest specs for extending the OPF vs Skyflow benchmark. Each plan is brief
(roughly one Pomodoro to read), self-contained, and implementable independently.

| # | plan | effort | unblocks |
|---|---|---|---|
| 01 | [Presidio baseline](01-presidio-baseline.md) | ~2h | "is the ML detector worth it vs free regex+NER?" |
| 02 | [Fine-tune OPF](02-finetune-opf.md) | ~1 day (cloud GPU) | "can OPF beat Skyflow if trained on this data?" |
| 03 | [GLiNER baseline](03-gliner-baseline.md) | ~3-4h | "is OPF the right open-weight choice or would GLiNER do as well?" |
| 04 | [LLM-as-detector via LM Studio](04-llm-detector-lmstudio.md) | ~3-4h | "what's the quality ceiling of a generative model on this task?" |

## Suggested order

1. **#01 Presidio** first — cheapest, possibly reframes the whole comparison
2. **#02 Fine-tune OPF** — most informative, biggest "if it works, change the recommendation" upside
3. **#03 GLiNER** if interested in a third open-weight peer
4. **#04 LLM detector** for research signal, not deployment

## Conventions

- All plans assume the existing `opf_eval` harness scaffolding stays unchanged
- New detectors register themselves in `runner._build_detector()` by string name
- All add a column to `taxonomy.CANONICAL_MAP` if their entity-type vocabulary differs from existing detectors
- Reports auto-pick up new detectors from the manifest
