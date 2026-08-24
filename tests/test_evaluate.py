from pathlib import Path

import pytest

from preference_lab.evaluate import pairwise_accuracy, write_metrics
from preference_lab.schemas import PreferenceExample


def _example() -> PreferenceExample:
    return PreferenceExample(prompt="p", chosen="a", rejected="b")


def test_pairwise_accuracy_counts_wins_and_ties() -> None:
    examples = [_example(), _example()]
    assert pairwise_accuracy(examples, [2.0, 1.0], [1.0, 1.0]) == 0.75


def test_pairwise_accuracy_validates_lengths() -> None:
    with pytest.raises(ValueError, match="equal lengths"):
        pairwise_accuracy([_example()], [], [])


def test_write_metrics_creates_json(tmp_path: Path) -> None:
    output = write_metrics({"pairwise_accuracy": 0.5}, tmp_path / "nested")
    assert output.read_text(encoding="utf-8").endswith("\n")
    assert '"pairwise_accuracy": 0.5' in output.read_text(encoding="utf-8")
