# Day 22 — Preference Alignment with SFT, DPO, and ORPO

An end-to-end local-GPU alignment lab built around Qwen2.5-3B: Vietnamese SFT-mini, cleaned
UltraFeedback preparation, DPO training, reward-curve analysis, side-by-side evaluation, GGUF
deployment, and lightweight alignment benchmarks. The original NumPy DPO/ORPO implementation and
strict unit-test path remain available for CI.

## Required environment

- Python 3.10–3.12 recommended (Python 3.10+ required).
- CUDA 11.8 or 12.1+.
- NVIDIA GPU with at least 12 GB VRAM.
- About 15 GB free disk space for the cached base model and optional merged/GGUF files.

The verified local run used an RTX 5060 Ti 16 GB, CUDA 12.8, and the T4-tier configuration in
`configs/gpu.yaml`.

## Setup

Linux/macOS:

```bash
bash setup-laptop.sh
```

Windows PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File setup-laptop.ps1
.\.venv-gpu\Scripts\Activate.ps1
```

The setup installs the GPU extras, checks CUDA, and enforces the 12 GB VRAM floor. CUDA 12.8 wheels
can also be installed explicitly with `requirements-gpu.txt` on recent NVIDIA cards.

## Full pipeline

```bash
make smoke          # Python, CUDA, GPU and VRAM gate
make prepare-data   # 1k VN Alpaca + 2k cleaned UltraFeedback parquet
make train-sft      # adapters/sft-mini, r=16 and alpha=32
make train-dpo      # adapters/dpo + chosen/rejected/gap metrics
make eval           # 8 prompts x SFT/DPO + win/loss/tie
make deploy         # optional merged Q4_K_M GGUF + llama.cpp smoke
make bench          # optional IFEval/GSM8K/MMLU/AlpacaEval-lite subsets
make notebooks      # build five executed evidence notebooks
make verify         # required gates; optional bonus status is reported separately
```

On Windows without GNU Make, invoke the corresponding Python scripts shown in the `Makefile`.
Training commands accept `--reuse` to preserve completed artifacts during documentation reruns.

## Expected evidence

- `adapters/sft-mini/adapter_config.json` with `r: 16`, `lora_alpha: 32`.
- `adapters/dpo/adapter_config.json`, distinct from SFT-mini.
- `data/pref/train.parquet` with 2,000 `prompt/chosen/rejected` rows.
- `submission/screenshots/03_dpo_reward_curves.png` with separate chosen, rejected, and gap curves.
- `submission/side_by_side.md` and `04_side_by_side_table.png` with eight prompts across helpfulness
  and safety.
- `submission/evaluation.json` with a transparent deterministic manual-rubric summary.
- Five executed notebooks under `notebooks/`.
- `submission/REFLECTION.md` with the six required sections and optional benchmark analysis.
- Bonus `gguf/lab22-dpo-Q4_K_M.gguf`, benchmark report, and public Hugging Face adapter links.

## Published bonus artifacts

- SFT adapter: [tiennn/day22-qwen25-3b-sft-mini](https://huggingface.co/tiennn/day22-qwen25-3b-sft-mini)
- DPO adapter: [tiennn/day22-qwen25-3b-dpo](https://huggingface.co/tiennn/day22-qwen25-3b-dpo)
- GGUF smoke evidence: `submission/deploy.json` and `submission/llama_smoke.stdout.txt`
- Lightweight benchmark: `submission/benchmark.json`

Model weights, checkpoints, merged models, and GGUF binaries are intentionally ignored by Git. The
small adapter configs, metrics, plots, notebooks, and submission evidence are versioned; adapter
weights remain present locally and are published through the professional Hugging Face bonus.

## CPU/CI quality path

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
pip install -e '.[dev]'
pytest -q
ruff check src tests scripts
mypy src
pref-lab validate data/sample_preferences.jsonl
pref-lab evaluate --config configs/local.yaml
```

The CPU CLI performs a leakage-free train/validation split and produces deterministic local metrics
without downloading a model. It complements rather than replaces the GPU submission pipeline.

## Data and safety notes

The SFT slice comes from `bkai-foundation-models/vi-alpaca`; the preference slice comes from
`argilla/ultrafeedback-binarized-preferences-cleaned`. See `docs/data_card_template.md` and
`submission/REFLECTION.md` for limitations. The benchmark scripts are small smoke subsets and must
not be reported as official leaderboard scores.
