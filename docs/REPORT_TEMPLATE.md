# Preference Alignment Experiment Report

## 1. Dataset analysis and cleaning

- **Total examples loaded:** 24
- **Validation issue found:** Source line 1 contained unescaped double quotes around
  `self-attention`, so the original row was invalid JSON.
- **Cleaning performed:** Escaped those quotes. The loader now wraps JSON and Pydantic failures with
  the source file and line number, normalises prompts for duplicate detection, and offers an opt-in
  PII guard.
- **Schema hardening:** Chosen and rejected responses that differ only by case, punctuation, or
  whitespace—or are at least 98% similar—are rejected.

### Split strategy

The experiment uses a deterministic 80/20 split with seed 42: 19 training examples and 5 validation
examples. Unique normalised prompts are shuffled and assigned as groups, so one prompt cannot occur
in both splits.

## 2. DPO and ORPO implementation

Both objectives were implemented, although the configured experiment uses DPO.

- **DPO:** Compares the policy chosen/rejected log-ratio with the corresponding reference log-ratio.
  It evaluates `-log(sigmoid(x))` as `logaddexp(0, -x)` to avoid overflow.
- **ORPO:** Adds the SFT negative log-likelihood to a preference penalty computed from the stable
  chosen/rejected log-odds ratio. `log(1 - exp(x))` uses separate `log1p` and `expm1` branches.
- **Hyperparameters:** `beta = 0.1`, `lambda_orpo = 0.1`, maximum length 512, batch size 2.
- **Input safeguards:** Both losses reject empty, non-finite, or shape-mismatched inputs and invalid
  hyperparameters.

## 3. Evaluation results

The no-download path fits a smoothed token log-odds scorer on the 19-example training split. Scores
are normalised by the square root of response length and evaluated only on the held-out prompts.

| Metric | Value |
|---|---:|
| Validation examples | 5 |
| Pairwise accuracy | 0.80 |
| Ties | 0 |
| Final DPO proxy loss | 0.3391 |

These values are deterministic for the committed data and config. They are not hard-coded: the CLI
trains the scorer, serialises it to `outputs/local_scorer.json`, scores both responses, and writes
`outputs/metrics.json`.

### Qualitative failure

- **Prompt:** “What is the purpose of a validation set in machine learning?”
- **Expected preference:** The chosen answer correctly describes development-time model selection.
- **Observed scores:** chosen `-1.2364`, rejected `-1.1161`.
- **Result:** Incorrect preference. The held-out wording includes tokens whose training-set lexical
  weights do not represent their factual meaning.

## 4. Discussion and failure modes

- The pipeline catches malformed records early, prevents split leakage, and produces repeatable
  metrics without an API key or model download.
- The main observed bias is lexical/style preference. Chosen answers are generally longer and more
  detailed, while rejected answers use recurring phrases such as “is used”; the scorer can exploit
  these artifacts.
- Pairwise accuracy on five examples has very high variance and must not be interpreted as general
  alignment quality.
- The regression prompt file covers high-risk medical advice, strict word limits, uncertainty, and
  missing troubleshooting context. The local scorer is non-generative, so these prompts are a
  pre-deployment checklist rather than evidence of passed safety behavior. A generative model must be
  tested before/after real fine-tuning.

## 5. Reproduction

```bash
pref-lab validate data/sample_preferences.jsonl
pref-lab train --config configs/local.yaml
pref-lab evaluate --config configs/local.yaml
pytest -q
ruff check src tests
mypy src
```
