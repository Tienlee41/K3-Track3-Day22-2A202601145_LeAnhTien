# Data Card: Sample ML Preferences

## Summary

- **Dataset name:** Sample ML Preferences
- **Source:** Course-authored examples in `data/sample_preferences.jsonl`
- **License/permission:** Educational sample bundled with this repository; no permission is
  asserted for reuse outside the course.
- **Size:** 24 English preference pairs.
- **Schema:** `prompt: str`, `chosen: str`, `rejected: str`, and optional
  `metadata: dict[str, Any]`.
- **Domain:** Introductory machine-learning education.

## Collection and labeling

Each row contrasts a factually accurate, explanatory answer (`chosen`) with a plausible but
incorrect or incomplete answer (`rejected`). The metadata rubric is `accuracy`. These are
course-authored examples, not human preference votes gathered from production traffic.

## Validation and cleaning

- Escaped the unescaped quotes around `"self-attention"` in source line 1.
- Pydantic validates required, non-empty text fields.
- Chosen/rejected near-duplicates are rejected after case, punctuation, and whitespace
  normalisation.
- The loader reports JSON/schema errors with file and line number and rejects duplicate prompts.
- An opt-in PII guard detects likely email addresses and phone numbers. No PII was observed in the
  bundled dataset.

## Splitting

The local config uses an 80/20 split with seed 42. `split_by_prompt` shuffles unique prompt groups,
then assigns whole groups to train or validation, preventing prompt leakage. This produces 19 train
and 5 validation examples.

## Known limitations and biases

- The dataset is tiny, English-only, and restricted to introductory ML facts.
- Rejected answers are usually shorter and contain conspicuous false statements. A scorer can learn
  style or length artifacts instead of genuine correctness.
- Labels come from a single synthetic/course-authoring process; there is no inter-annotator
  agreement or demographic coverage.
- The deterministic local scorer is a lexical baseline, not a calibrated reward model.

## Appropriate use

Use this dataset for testing data validation, DPO/ORPO loss implementations, and local pipeline
plumbing. Do not use its metrics as evidence of broad model quality, safety, or production readiness.

## GPU training datasets

The full submission pipeline uses two additional public Hugging Face sources:

- `bkai-foundation-models/vi-alpaca`: the first 1,000 training rows after deterministic source
  ordering, converted to `prompt/response` and stored in `data/sft/train.parquet`. It supplies the
  Vietnamese SFT-mini stage. The source is synthetic/self-instruct data and can inherit generator
  errors, verbosity, cultural imbalance, and outdated facts.
- `argilla/ultrafeedback-binarized-preferences-cleaned`: the first 2,000 training rows, converted
  from chat messages into string `prompt/chosen/rejected` columns and stored in
  `data/pref/train.parquet`. The cleaned source removes known TruthfulQA contamination, but its
  preferences remain model-generated/model-judged rather than direct votes from a representative
  population.

`data/pref/inspection.json` records three inspected pairs and verifies `chosen != rejected`. The
training slices are fixed by the committed config and are not used for the eight-prompt qualitative
comparison or benchmark smoke subsets. Neither source should be interpreted as consent for handling
private data; the local loader's PII guard remains available for additional JSONL inputs.
