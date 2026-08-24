from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from .schemas import PreferenceExample


def pairwise_accuracy(
    examples: Sequence[PreferenceExample],
    chosen_scores: Sequence[float],
    rejected_scores: Sequence[float],
    *,
    tie_value: float = 0.5,
) -> float:
    """Return pairwise accuracy, assigning ``tie_value`` credit to ties."""
    if len(chosen_scores) != len(examples) or len(rejected_scores) != len(examples):
        raise ValueError("examples, chosen_scores, and rejected_scores must have equal lengths")
    if not 0.0 <= tie_value <= 1.0:
        raise ValueError("tie_value must be between 0 and 1")
    if not examples:
        return 0.0
    if any(not np.isfinite(score) for score in (*chosen_scores, *rejected_scores)):
        raise ValueError("scores must contain only finite values")

    wins = sum(chosen > rejected for chosen, rejected in zip(chosen_scores, rejected_scores))
    ties = sum(chosen == rejected for chosen, rejected in zip(chosen_scores, rejected_scores))
    return float((wins + tie_value * ties) / len(examples))


def write_metrics(metrics: Mapping[str, float], output_dir: str | Path) -> Path:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    out = path / "metrics.json"
    out.write_text(json.dumps(dict(metrics), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out
