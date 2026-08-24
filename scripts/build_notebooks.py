from __future__ import annotations

from pathlib import Path

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

NOTEBOOKS = [
    (
        "01_sft_mini.ipynb",
        "NB1 — SFT-mini",
        """import json
from pathlib import Path
from IPython.display import Image, display
metrics = json.loads(Path('../adapters/sft-mini/training_metrics.json').read_text(encoding='utf-8'))
keys = ['base_model', 'samples', 'train_loss', 'runtime_seconds', 'peak_vram_gb']
print({k: metrics[k] for k in keys})
sample = json.loads(Path('../adapters/sft-mini/sample_generation.json').read_text(encoding='utf-8'))
print(sample['prompt'])
print(sample['response'])
display(Image('../submission/screenshots/02_sft_loss_curve.png'))""",
    ),
    (
        "02_preference_data.ipynb",
        "NB2 — Preference data preparation",
        """import json
from pathlib import Path
import pandas as pd
frame = pd.read_parquet('../data/pref/train.parquet')
print('shape:', frame.shape)
print('columns:', list(frame.columns))
checks = [(frame.iloc[i].chosen != frame.iloc[i].rejected) for i in range(3)]
print('first 3 chosen != rejected:', checks)
inspection = json.loads(Path('../data/pref/inspection.json').read_text(encoding='utf-8'))
print('inspection passed:', inspection['all_inspected_pairs_differ'])""",
    ),
    (
        "03_dpo_training.ipynb",
        "NB3 — DPO training",
        """import json
from pathlib import Path
from IPython.display import Image, display
metrics = json.loads(Path('../adapters/dpo/training_metrics.json').read_text(encoding='utf-8'))
keys = ['samples', 'beta', 'learning_rate', 'train_loss', 'initial_reward_gap',
        'final_reward_gap', 'runtime_seconds', 'peak_vram_gb']
print({k: metrics[k] for k in keys})
display(Image('../submission/screenshots/03_dpo_reward_curves.png'))""",
    ),
    (
        "04_compare_eval.ipynb",
        "NB4 — Side-by-side comparison",
        """import json
from pathlib import Path
from IPython.display import Image, Markdown, display
evaluation = json.loads(Path('../submission/evaluation.json').read_text(encoding='utf-8'))
print('comparisons:', len(evaluation['comparisons']))
print('win/loss/tie:', evaluation['summary'])
display(Markdown(Path('../submission/side_by_side.md').read_text(encoding='utf-8')))
display(Image('../submission/screenshots/05_win_loss_tie.png'))""",
    ),
    (
        "05_deploy_benchmark.ipynb",
        "NB5/NB6 — GGUF deploy and benchmark bonus",
        """import json
from pathlib import Path
from IPython.display import Image, display
deploy = json.loads(Path('../submission/deploy.json').read_text(encoding='utf-8'))
benchmark = json.loads(Path('../submission/benchmark.json').read_text(encoding='utf-8'))
hub = json.loads(Path('../submission/huggingface.json').read_text(encoding='utf-8'))
print('GGUF:', deploy['gguf'], deploy['size_gb'], 'GB', deploy['quantization'])
print('llama.cpp response:', deploy['response'])
print('benchmark scope:', benchmark['scope'])
print('benchmark summary:', benchmark['summary'])
print('Hugging Face:', hub['repositories'])
display(Image('../submission/screenshots/06_benchmark_summary.png'))""",
    ),
]


def main() -> None:
    output = Path("notebooks")
    output.mkdir(parents=True, exist_ok=True)
    executor = ExecutePreprocessor(timeout=180, kernel_name="python3")
    for filename, title, code in NOTEBOOKS:
        notebook = nbformat.v4.new_notebook(
            cells=[
                nbformat.v4.new_markdown_cell(
                    f"# {title}\n\nExecuted evidence notebook for the Day 22 DPO alignment lab."
                ),
                nbformat.v4.new_code_cell(code),
            ],
            metadata={
                "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}
            },
        )
        executor.preprocess(notebook, {"metadata": {"path": str(output.resolve())}})
        nbformat.write(notebook, output / filename)
        print(f"Wrote executed notebook: {output / filename}")


if __name__ == "__main__":
    main()
