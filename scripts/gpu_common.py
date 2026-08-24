from __future__ import annotations

# ruff: noqa: I001 -- compatibility import must precede third-party GPU imports

import json
from pathlib import Path
import random
from typing import Any

import windows_platform_compat  # noqa: F401  # must precede torch imports

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml


def load_gpu_config(path: str | Path = "configs/gpu.yaml") -> dict[str, Any]:
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise TypeError("GPU config must be a YAML mapping")
    return config


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def require_gpu(minimum_gb: float = 12.0) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this training pipeline")
    props = torch.cuda.get_device_properties(0)
    total_gb = props.total_memory / 1024**3
    if total_gb < minimum_gb:
        raise RuntimeError(f"at least {minimum_gb:g} GB VRAM is required, found {total_gb:.2f}")
    return {
        "gpu": props.name,
        "vram_total_gb": round(total_gb, 2),
        "cuda_runtime": torch.version.cuda,
        "torch": torch.__version__,
    }


def save_json(path: str | Path, payload: Any) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return destination


def plot_gpu_card(info: dict[str, Any], destination: str | Path) -> None:
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(10, 4.5))
    axis.axis("off")
    axis.set_title("Day 22 Local GPU Smoke Test", fontsize=18, weight="bold", pad=20)
    lines = [
        f"GPU: {info['gpu']}",
        f"VRAM: {info['vram_total_gb']:.2f} GB",
        f"CUDA runtime: {info['cuda_runtime']}",
        f"PyTorch: {info['torch']}",
        "Status: PASS (CUDA available, VRAM >= 12 GB)",
    ]
    axis.text(
        0.5,
        0.48,
        "\n".join(lines),
        ha="center",
        va="center",
        fontsize=15,
        linespacing=1.5,
        bbox={"boxstyle": "round,pad=1", "facecolor": "#e8f5e9", "edgecolor": "#2e7d32"},
    )
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def peak_vram_gb() -> float:
    return round(torch.cuda.max_memory_allocated() / 1024**3, 3)
