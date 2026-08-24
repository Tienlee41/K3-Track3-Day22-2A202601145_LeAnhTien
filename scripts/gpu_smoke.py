from __future__ import annotations

# ruff: noqa: I001 -- compatibility import must precede third-party GPU imports

import argparse
import json
import platform
import sys

import windows_platform_compat  # noqa: F401  # must precede torch imports


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe the local CUDA training environment")
    parser.add_argument("--strict", action="store_true", help="fail unless CUDA and >=12 GB exist")
    args = parser.parse_args()

    try:
        import torch
    except ImportError as exc:
        raise SystemExit("torch is not installed; run setup-laptop first") from exc

    cuda = torch.cuda.is_available()
    total_gb = 0.0
    gpu_name = "none"
    if cuda:
        props = torch.cuda.get_device_properties(0)
        total_gb = props.total_memory / 1024**3
        gpu_name = props.name

    result = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": cuda,
        "gpu": gpu_name,
        "vram_gb": round(total_gb, 2),
    }
    print(json.dumps(result, indent=2))

    if sys.version_info < (3, 10):  # noqa: UP036 -- standalone smoke must enforce the guide
        raise SystemExit("Python >=3.10 is required")
    if args.strict and (not cuda or total_gb < 12.0):
        raise SystemExit("CUDA GPU with at least 12 GB VRAM is required")


if __name__ == "__main__":
    main()
