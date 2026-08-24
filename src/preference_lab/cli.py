from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

import typer
from rich import print

from .config import load_config
from .data import load_jsonl, split_by_prompt
from .evaluate import pairwise_accuracy, write_metrics
from .schemas import PreferenceExample
from .trainers import LocalPreferenceScorer, PreferenceTrainer, TrainingConfig

app = typer.Typer(help="Preference alignment lab CLI")


def _training_config(config: dict[str, Any]) -> TrainingConfig:
    raw = config.get("training")
    paths = config.get("paths")
    if not isinstance(raw, dict) or not isinstance(paths, dict):
        raise TypeError("config requires 'training' and 'paths' mappings")
    return TrainingConfig(
        method=str(raw.get("method", "dpo")),
        beta=float(raw.get("beta", 0.1)),
        lambda_orpo=float(raw.get("lambda_orpo", 0.1)),
        max_length=int(raw.get("max_length", 512)),
        batch_size=int(raw.get("batch_size", 2)),
        output_dir=Path(str(paths.get("output_dir", "outputs"))),
    )


def _load_splits(
    config: dict[str, Any],
) -> tuple[list[PreferenceExample], list[PreferenceExample]]:
    paths = config.get("paths")
    evaluation = config.get("evaluation", {})
    if not isinstance(paths, dict) or not isinstance(evaluation, dict):
        raise TypeError("config requires valid 'paths' and 'evaluation' mappings")
    examples = load_jsonl(Path(str(paths["train_data"])))
    ratio = float(evaluation.get("validation_ratio", 0.2))
    seed = int(config.get("seed", 42))
    return split_by_prompt(examples, validation_ratio=ratio, seed=seed)


@app.command()
def validate(data: Path, pii_guard: bool = False) -> None:
    """Validate every record in a preference JSONL file."""
    examples = load_jsonl(data, pii_guard=pii_guard)
    print(f"[green]Loaded {len(examples)} preference examples[/green]")


@app.command()
def train(config: Annotated[Path, typer.Option(help="Path to the experiment YAML config")]) -> None:
    """Train and persist the deterministic local preference scorer."""
    cfg = load_config(config)
    training_examples, _ = _load_splits(cfg)
    result = PreferenceTrainer(_training_config(cfg)).train(training_examples)
    print(
        f"[green]Trained on {result.training_examples} examples; "
        f"wrote model to {result.model_path}[/green]"
    )


@app.command()
def evaluate(
    config: Annotated[Path, typer.Option(help="Path to the experiment YAML config")],
) -> None:
    """Fit on the train split and evaluate on a leakage-free validation split."""
    cfg = load_config(config)
    training_examples, validation_examples = _load_splits(cfg)
    training_config = _training_config(cfg)
    result = PreferenceTrainer(training_config).train(training_examples)
    scorer = LocalPreferenceScorer.load(result.model_path)
    chosen_scores, rejected_scores = scorer.score_pairs(validation_examples)
    tie_count = sum(chosen == rejected for chosen, rejected in zip(chosen_scores, rejected_scores))
    metrics = {
        "pairwise_accuracy": pairwise_accuracy(validation_examples, chosen_scores, rejected_scores),
        "evaluated_examples": float(len(validation_examples)),
        "tie_count": float(tie_count),
        "final_loss": result.final_loss,
    }
    out = write_metrics(metrics, training_config.output_dir)
    print(f"[green]Wrote metrics to {out}[/green]")


if __name__ == "__main__":
    app()
