PYTHON ?= python

.PHONY: setup test lint typecheck format validate smoke prepare-data train-sft train-dpo eval deploy bench notebooks verify run-eval clean

setup:
	$(PYTHON) -m pip install -r requirements-gpu.txt
	$(PYTHON) -m pip install -e ".[dev]"
test:
	$(PYTHON) -m pytest -q
lint:
	$(PYTHON) -m ruff check src tests scripts
typecheck:
	$(PYTHON) -m mypy src
format:
	$(PYTHON) -m ruff format src tests scripts
validate:
	pref-lab validate data/sample_preferences.jsonl
smoke:
	$(PYTHON) scripts/gpu_smoke.py --strict
prepare-data:
	$(PYTHON) scripts/prepare_training_data.py
train-sft: prepare-data
	$(PYTHON) scripts/train_sft_gpu.py
train-dpo: prepare-data
	$(PYTHON) scripts/train_dpo_gpu.py
eval:
	$(PYTHON) scripts/compare_models.py
deploy:
	$(PYTHON) scripts/deploy_gguf.py
bench:
	$(PYTHON) scripts/run_benchmarks.py
notebooks:
	$(PYTHON) scripts/build_notebooks.py
verify:
	$(PYTHON) scripts/verify_submission.py
run-eval:
	pref-lab evaluate --config configs/local.yaml
clean:
	$(PYTHON) -c "from pathlib import Path; import shutil; [shutil.rmtree(p, ignore_errors=True) for p in map(Path, ['.pytest_cache','.ruff_cache','.mypy_cache','outputs','checkpoints'])]"
