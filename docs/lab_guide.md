# Lab Guide

## Task 1: Data loader
Implement robust JSONL loading. Add line-numbered error messages and duplicate checks.

## Task 1.5: (Optional) Synthetic Data Generation
Use an LLM to expand your dataset. This teaches you how to create high-quality alignment data at scale.
```bash
export OPENAI_API_KEY=your_key
python scripts/generate_data.py --count 10 --domain "python coding"
```

## Task 2: Loss function
Choose DPO or ORPO. Implement the TODO in `src/preference_lab/losses.py`.

## Task 3: Evaluation
Replace mock scores with model-derived scores or a deterministic scorer for CPU mode.

## Task 4: Report
Write a short report with dataset notes, config, metrics, and failure modes.

## Local-GPU submission track

The completed T4-tier workflow is documented in the root `README.md`. It adds Vietnamese SFT-mini,
2,000 cleaned UltraFeedback pairs, real DPO training, eight-prompt qualitative comparison, six
screenshots, five executed notebooks, and the optional benchmark/GGUF/Hugging Face deliverables.
Run `make smoke` before GPU work and `make verify` after all artifacts have been generated.
