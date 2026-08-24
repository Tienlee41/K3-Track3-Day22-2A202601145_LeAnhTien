from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

NOTEBOOKS = [
    (
        "01_sft_mini.ipynb",
        "NB1 — SFT-mini",
        """import json
from pathlib import Path
config = json.loads(Path('../adapters/sft-mini/adapter_config.json').read_text(encoding='utf-8'))
metrics = json.loads(Path('../adapters/sft-mini/training_metrics.json').read_text(encoding='utf-8'))
sample = json.loads(Path('../adapters/sft-mini/sample_generation.json').read_text(encoding='utf-8'))
print('adapter:', {'r': config['r'], 'lora_alpha': config['lora_alpha']})
print('training:', {k: metrics[k] for k in ['base_model', 'samples', 'train_loss', 'runtime_seconds', 'peak_vram_gb']})
print('loss first/last:', metrics['loss_history'][0]['loss'], metrics['loss_history'][-1]['loss'])
print('sample prompt:', sample['prompt'])
print('sample response:', sample['response'])
print('coherent Vietnamese:', sample['coherent_vietnamese'])
print('loss plot exists:', Path('../submission/screenshots/02_sft_loss_curve.png').exists())""",
    ),
    (
        "02_preference_data.ipynb",
        "NB2 — Preference data preparation",
        """import json
from pathlib import Path
parquet = Path('../data/pref/train.parquet')
inspection = json.loads(Path('../data/pref/inspection.json').read_text(encoding='utf-8'))
with parquet.open('rb') as stream:
    header = stream.read(4)
    stream.seek(-4, 2)
    footer = stream.read(4)
print('parquet exists:', parquet.exists(), 'size bytes:', parquet.stat().st_size)
print('parquet signature:', header, footer)
print('shape:', (inspection['preference_rows'], len(inspection['columns'])))
print('columns:', inspection['columns'])
checks = [row['chosen'] != row['rejected'] for row in inspection['examples']]
print('three inspected chosen != rejected:', checks)
print('inspection passed:', inspection['all_inspected_pairs_differ'])""",
    ),
    (
        "03_dpo_training.ipynb",
        "NB3 — DPO training",
        """import json
from pathlib import Path
config = json.loads(Path('../adapters/dpo/adapter_config.json').read_text(encoding='utf-8'))
metrics = json.loads(Path('../adapters/dpo/training_metrics.json').read_text(encoding='utf-8'))
print('adapter:', {'r': config['r'], 'lora_alpha': config['lora_alpha']})
print('hyperparameters:', {k: metrics[k] for k in ['samples', 'beta', 'learning_rate', 'epochs']})
print('runtime/loss/VRAM:', {k: metrics[k] for k in ['runtime_seconds', 'train_loss', 'peak_vram_gb']})
print('reward gap:', metrics['initial_reward_gap'], '->', metrics['final_reward_gap'])
print('trend window:', metrics['initial_reward_gap_trend'], '->', metrics['final_reward_gap_trend'])
first, last = metrics['reward_history'][0], metrics['reward_history'][-1]
print('chosen reward:', first['chosen_reward'], '->', last['chosen_reward'])
print('rejected reward:', first['rejected_reward'], '->', last['rejected_reward'])
print('reward plot exists:', Path('../submission/screenshots/03_dpo_reward_curves.png').exists())""",
    ),
    (
        "04_compare_eval.ipynb",
        "NB4 — Side-by-side comparison",
        """import json
from pathlib import Path
evaluation = json.loads(Path('../submission/evaluation.json').read_text(encoding='utf-8'))
print('comparisons:', len(evaluation['comparisons']))
for index, item in enumerate(evaluation['comparisons'], 1):
    print(index, item['category'], '=>', item['dpo_judgment'])
print('win/loss/tie:', evaluation['summary'])
print('manual report:', Path('../submission/side_by_side.md').exists())
print('table screenshot:', Path('../submission/screenshots/04_side_by_side_table.png').exists())""",
    ),
    (
        "05_deploy_benchmark.ipynb",
        "NB5/NB6 — Optional deployment and benchmark",
        """import json
from pathlib import Path
gguf = Path('../gguf/lab22-dpo-Q4_K_M.gguf')
deploy_path = Path('../submission/deploy.json')
benchmark = json.loads(Path('../submission/benchmark.json').read_text(encoding='utf-8'))
hub_path = Path('../submission/huggingface.json')
print('GGUF:', gguf.exists(), round(gguf.stat().st_size / 1024**3, 3) if gguf.exists() else None, 'GiB')
print('llama.cpp smoke:', json.loads(deploy_path.read_text(encoding='utf-8')) if deploy_path.exists() else 'not run; optional')
print('benchmark scope:', benchmark['scope'])
print('benchmark summary:', benchmark['summary'])
print('Hugging Face bonus:', json.loads(hub_path.read_text(encoding='utf-8'))['repositories'] if hub_path.exists() else 'not submitted (Option A)')
print('benchmark screenshot:', Path('../submission/screenshots/06_benchmark_summary.png').exists())""",
    ),
]


def _stream_output(text: str, name: str = "stdout") -> dict[str, object]:
    return {"name": name, "output_type": "stream", "text": text.splitlines(keepends=True)}


def _execute(code: str, cwd: Path) -> list[dict[str, object]]:
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=cwd,
        text=True,
        encoding="utf-8",
        capture_output=True,
        check=False,
    )
    outputs = []
    if result.stdout:
        outputs.append(_stream_output(result.stdout))
    if result.stderr:
        outputs.append(_stream_output(result.stderr, "stderr"))
    if result.returncode:
        raise RuntimeError(f"Notebook cell failed with exit code {result.returncode}:\n{result.stderr}")
    return outputs


def main() -> None:
    output = Path("notebooks")
    output.mkdir(parents=True, exist_ok=True)
    interpreter = str(Path(sys.executable).resolve())
    for filename, title, code in NOTEBOOKS:
        notebook = {
            "cells": [
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": [
                        f"# {title}\n",
                        "\n",
                        "Executed evidence notebook for the Day 22 DPO alignment lab.\n",
                        f"Interpreter: `{interpreter}`\n",
                    ],
                },
                {
                    "cell_type": "code",
                    "execution_count": 1,
                    "metadata": {},
                    "outputs": _execute(code, output.resolve()),
                    "source": code.splitlines(keepends=True),
                },
            ],
            "metadata": {
                "kernelspec": {
                    "display_name": "Python 3 (.venv)",
                    "language": "python",
                    "name": "python3",
                },
                "language_info": {"name": "python", "version": sys.version.split()[0]},
            },
            "nbformat": 4,
            "nbformat_minor": 5,
        }
        target = output / filename
        target.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"Wrote executed notebook: {target}")


if __name__ == "__main__":
    main()
