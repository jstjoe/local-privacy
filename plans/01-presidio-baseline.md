# Plan 01 — Add Microsoft Presidio as a baseline detector

## Why

Presidio is the de-facto open-source PII baseline (Apache 2.0, regex + spaCy NER). Adding it tells us whether OPF/Skyflow's quality justifies their cost over a free, deterministic alternative. Possible the answer is "Presidio gets 80% of PII for 0% of the cost," which would change the whole framing of the comparison.

## Scope

- New `PresidioDetector` integrated into the existing harness
- No changes to fixtures, taxonomy structure, or report layout
- Runs locally, fully offline

## Files

- **Add** `eval/src/opf_eval/detectors/presidio.py` — `PresidioDetector` class
- **Modify** `eval/src/opf_eval/detectors/__init__.py` — export
- **Modify** `eval/src/opf_eval/runner.py` — register `presidio` name in `_build_detector`
- **Modify** `eval/src/opf_eval/taxonomy.py` — add `presidio` column to `CANONICAL_MAP` + `presidio_to_canonical()` helper
- **Modify** `eval/pyproject.toml` — add `presidio-analyzer`, `presidio-anonymizer`

## Implementation notes

- Use `AnalyzerEngine()` with default recognizers. English by default; non-English needs `nlp_engine` config (see Presidio multilingual docs).
- Map Presidio's entity types to canonical labels. Known types include: `PERSON`, `EMAIL_ADDRESS`, `PHONE_NUMBER`, `LOCATION`, `URL`, `DATE_TIME`, `IP_ADDRESS`, `CREDIT_CARD`, `IBAN_CODE`, `US_SSN`, `US_PASSPORT`, `US_DRIVER_LICENSE`. Most align cleanly to existing canonicals.
- Detection returns `RecognizerResult(entity_type, start, end, score)` — convert to our Span dict.
- Latency: expect ~50-200ms per example (regex + spaCy is fast).

## Risks / open questions

- **Multilingual support.** PII-Masking-300k has 6 languages. Presidio's default English-only NLP engine will tank on Dutch/German/etc. Either (a) install per-language spaCy models and route by `record["language"]`, or (b) note the English-only constraint and report English-only F1 for Presidio.
- **Regex tuning.** Presidio can be customized heavily with `PatternRecognizer`. We start with defaults to keep "out-of-the-box vs OPF" honest.
- **No native USERNAME/SECRET.** Same OPF coverage gap.

## Verification

```sh
uv sync
uv run python -m opf_eval.runner \
    --fixtures eval/data/sample_100.jsonl \
    --detectors opf,skyflow_minimal,presidio \
    --out eval/results/runs/run_100_presidio/
uv run python -m opf_eval.report --run eval/results/runs/run_100_presidio/ --fixtures eval/data/sample_100.jsonl
```

Expect Presidio precision close to OPF (regex is exact when it matches), recall lower (no semantic NER for things outside its catalog).

## Effort

~2 hours of integration + 1 short run.

## Out of scope

- Tuning Presidio's recognizer set
- Building custom Pattern recognizers for our domain
- Anonymizer (we only need detection)
