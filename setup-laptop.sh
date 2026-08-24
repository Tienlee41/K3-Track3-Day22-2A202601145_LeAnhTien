#!/usr/bin/env bash
set -euo pipefail

python3.11 -m venv .venv-gpu
source .venv-gpu/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-gpu.txt
python -m pip install -e '.[dev]'
python scripts/gpu_smoke.py --strict
