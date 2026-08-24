from pathlib import Path

from typer.testing import CliRunner

from preference_lab.cli import app

runner = CliRunner()


def test_validate_command() -> None:
    result = runner.invoke(app, ["validate", "data/sample_preferences.jsonl"])
    assert result.exit_code == 0
    assert "Loaded 24 preference examples" in result.stdout


def test_evaluate_accepts_config_option(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    output = tmp_path / "outputs"
    source = Path("data/sample_preferences.jsonl").resolve().as_posix()
    config.write_text(
        f"""seed: 42
paths:
  train_data: {source}
  output_dir: {output.as_posix()}
training:
  method: dpo
  beta: 0.1
  lambda_orpo: 0.1
  max_length: 512
  batch_size: 2
evaluation:
  validation_ratio: 0.2
""",
        encoding="utf-8",
    )
    result = runner.invoke(app, ["evaluate", "--config", str(config)])
    assert result.exit_code == 0
    assert (output / "metrics.json").exists()
