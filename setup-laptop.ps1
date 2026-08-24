$ErrorActionPreference = "Stop"

py -3.11 -m venv .venv-gpu
& .\.venv-gpu\Scripts\python.exe -m pip install --upgrade pip
& .\.venv-gpu\Scripts\python.exe -m pip install -r requirements-gpu.txt
& .\.venv-gpu\Scripts\python.exe -m pip install -e ".[dev]"
& .\.venv-gpu\Scripts\python.exe scripts\gpu_smoke.py --strict
