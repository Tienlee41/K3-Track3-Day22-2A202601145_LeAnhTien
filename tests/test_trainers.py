from pathlib import Path

from preference_lab.schemas import PreferenceExample
from preference_lab.trainers import LocalPreferenceScorer, PreferenceTrainer, TrainingConfig


def _examples() -> list[PreferenceExample]:
    return [
        PreferenceExample(
            prompt="Explain testing",
            chosen="Testing catches defects and supports safe refactoring.",
            rejected="Testing is unnecessary busywork.",
        ),
        PreferenceExample(
            prompt="Explain validation",
            chosen="Validation detects malformed input before processing.",
            rejected="Validation makes every input correct automatically.",
        ),
    ]


def test_local_scorer_round_trip(tmp_path: Path) -> None:
    scorer = LocalPreferenceScorer.fit(_examples())
    path = scorer.save(tmp_path / "scorer.json")
    restored = LocalPreferenceScorer.load(path)
    assert restored.score("safe validation") == scorer.score("safe validation")


def test_preference_trainer_writes_artifacts(tmp_path: Path) -> None:
    config = TrainingConfig(method="dpo", output_dir=tmp_path)
    result = PreferenceTrainer(config).train(_examples())
    assert result.model_path == tmp_path / "local_scorer.json"
    assert result.model_path.exists()
    assert (tmp_path / "training_metrics.json").exists()
    assert result.final_loss >= 0.0
