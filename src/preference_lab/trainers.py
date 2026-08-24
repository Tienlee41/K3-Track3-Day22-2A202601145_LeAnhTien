from __future__ import annotations

import json
import math
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np

from .losses import dpo_loss, orpo_loss
from .schemas import PreferenceExample

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_PATTERN.findall(text.casefold())


@dataclass(frozen=True)
class TrainingConfig:
    method: str
    beta: float = 0.1
    lambda_orpo: float = 0.1
    max_length: int = 512
    batch_size: int = 2
    output_dir: Path = Path("outputs")

    def __post_init__(self) -> None:
        if self.method not in {"dpo", "orpo", "mock"}:
            raise ValueError("method must be one of: dpo, orpo, mock")
        if self.beta <= 0.0:
            raise ValueError("beta must be positive")
        if self.lambda_orpo < 0.0:
            raise ValueError("lambda_orpo must be non-negative")
        if self.max_length <= 0 or self.batch_size <= 0:
            raise ValueError("max_length and batch_size must be positive")


@dataclass(frozen=True)
class TrainingResult:
    model_path: Path
    training_examples: int
    vocabulary_size: int
    final_loss: float


class LocalPreferenceScorer:
    """Small deterministic bag-of-words scorer for the no-download local path.

    Each token receives a smoothed chosen-vs-rejected log-odds weight. Scores
    are length-normalised so the baseline does not simply prefer longer text.
    """

    def __init__(self, weights: dict[str, float] | None = None) -> None:
        self.weights = weights or {}

    @classmethod
    def fit(
        cls,
        examples: Sequence[PreferenceExample],
        *,
        smoothing: float = 0.1,
    ) -> LocalPreferenceScorer:
        if not examples:
            raise ValueError("at least one training example is required")
        if smoothing <= 0.0:
            raise ValueError("smoothing must be positive")

        chosen_counts: Counter[str] = Counter()
        rejected_counts: Counter[str] = Counter()
        for example in examples:
            chosen_counts.update(_tokenize(example.chosen))
            rejected_counts.update(_tokenize(example.rejected))

        vocabulary = set(chosen_counts) | set(rejected_counts)
        chosen_total = sum(chosen_counts.values())
        rejected_total = sum(rejected_counts.values())
        chosen_denominator = chosen_total + smoothing * len(vocabulary)
        rejected_denominator = rejected_total + smoothing * len(vocabulary)
        weights = {
            token: math.log((chosen_counts[token] + smoothing) / chosen_denominator)
            - math.log((rejected_counts[token] + smoothing) / rejected_denominator)
            for token in sorted(vocabulary)
        }
        return cls(weights)

    def score(self, text: str) -> float:
        tokens = _tokenize(text)
        if not tokens:
            return 0.0
        return sum(self.weights.get(token, 0.0) for token in tokens) / math.sqrt(len(tokens))

    def score_pairs(self, examples: Sequence[PreferenceExample]) -> tuple[list[float], list[float]]:
        return (
            [self.score(example.chosen) for example in examples],
            [self.score(example.rejected) for example in examples],
        )

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {"format_version": 1, "weights": self.weights}
        destination.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return destination

    @classmethod
    def load(cls, path: str | Path) -> LocalPreferenceScorer:
        payload: Any = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get("format_version") != 1:
            raise ValueError("unsupported local scorer format")
        raw_weights = payload.get("weights")
        if not isinstance(raw_weights, dict) or not all(
            isinstance(key, str) and isinstance(value, (int, float))
            for key, value in raw_weights.items()
        ):
            raise ValueError("local scorer contains invalid weights")
        return cls(cast(dict[str, float], raw_weights))


class PreferenceTrainer:
    """Train and persist the deterministic local preference baseline."""

    def __init__(self, config: TrainingConfig) -> None:
        self.config = config

    def train(self, examples: Sequence[PreferenceExample]) -> TrainingResult:
        scorer = LocalPreferenceScorer.fit(examples)
        chosen_scores, rejected_scores = scorer.score_pairs(examples)
        chosen = np.asarray(chosen_scores, dtype=np.float64)
        rejected = np.asarray(rejected_scores, dtype=np.float64)

        if self.config.method == "dpo":
            zeros = np.zeros_like(chosen)
            final_loss = dpo_loss(chosen, rejected, zeros, zeros, self.config.beta)
        elif self.config.method == "orpo":
            chosen_logps = -np.logaddexp(0.0, -chosen)
            rejected_logps = -np.logaddexp(0.0, -rejected)
            final_loss = orpo_loss(
                -chosen_logps,
                chosen_logps,
                rejected_logps,
                self.config.lambda_orpo,
            )
        else:
            final_loss = 0.0

        model_path = scorer.save(self.config.output_dir / "local_scorer.json")
        result = TrainingResult(
            model_path=model_path,
            training_examples=len(examples),
            vocabulary_size=len(scorer.weights),
            final_loss=final_loss,
        )
        metrics_path = self.config.output_dir / "training_metrics.json"
        metrics_path.write_text(
            json.dumps(asdict(result) | {"model_path": str(model_path)}, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        return result
