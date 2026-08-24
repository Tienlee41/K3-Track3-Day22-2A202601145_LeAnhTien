from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd


def _check(condition: bool, label: str, failures: list[str]) -> None:
    marker = "PASS" if condition else "FAIL"
    print(f"[{marker}] {label}")
    if not condition:
        failures.append(label)


def _words(section: str) -> int:
    return len(re.findall(r"\b\w+\b", section, flags=re.UNICODE))


def _load_json(path: str | Path) -> dict[str, object]:
    source = Path(path)
    return json.loads(source.read_text(encoding="utf-8")) if source.exists() else {}


def main() -> None:
    failures: list[str] = []
    sft_path = Path("adapters/sft-mini/adapter_config.json")
    dpo_path = Path("adapters/dpo/adapter_config.json")
    sft = json.loads(sft_path.read_text(encoding="utf-8")) if sft_path.exists() else {}
    dpo = json.loads(dpo_path.read_text(encoding="utf-8")) if dpo_path.exists() else {}
    _check(
        sft.get("lora_alpha") == 32 and sft.get("r") == 16,
        "SFT adapter r=16 alpha=32",
        failures,
    )
    _check(bool(dpo) and dpo != sft, "DPO adapter exists and is distinct", failures)

    pref_path = Path("data/pref/train.parquet")
    pref = pd.read_parquet(pref_path) if pref_path.exists() else pd.DataFrame()
    _check(
        len(pref) == 2000 and list(pref.columns) == ["prompt", "chosen", "rejected"],
        "Preference parquet has 2,000 prompt/chosen/rejected rows",
        failures,
    )
    inspection = _load_json("data/pref/inspection.json")
    _check(
        inspection.get("all_inspected_pairs_differ") is True,
        "Three inspected preference pairs have chosen != rejected",
        failures,
    )

    dpo_metrics = _load_json("adapters/dpo/training_metrics.json")
    initial_gap = dpo_metrics.get("initial_reward_gap")
    final_gap = dpo_metrics.get("final_reward_gap")
    reward_history = dpo_metrics.get("reward_history", [])
    gaps = (
        [float(point["reward_gap"]) for point in reward_history]
        if isinstance(reward_history, list)
        and all(isinstance(point, dict) and "reward_gap" in point for point in reward_history)
        else []
    )
    window = min(5, len(gaps))
    initial_trend = sum(gaps[:window]) / window if window else float("nan")
    final_trend = sum(gaps[-window:]) / window if window else float("nan")
    reward_success = (
        isinstance(initial_gap, (int, float))
        and isinstance(final_gap, (int, float))
        and isinstance(reward_history, list)
        and len(reward_history) >= 2
        and final_trend > 0
        and final_trend > initial_trend
        and all(
            isinstance(point, dict)
            and "chosen_reward" in point
            and "rejected_reward" in point
            and "reward_gap" in point
            for point in reward_history
        )
    )
    _check(reward_success, "DPO chosen/rejected curves and positive increasing gap", failures)

    screenshots = Path("submission/screenshots")
    required_images = [
        "01_gpu_smoke.png",
        "02_sft_loss_curve.png",
        "03_dpo_reward_curves.png",
        "04_side_by_side_table.png",
        "05_win_loss_tie.png",
        "06_benchmark_summary.png",
    ]
    _check(
        all((screenshots / name).exists() for name in required_images),
        "Six required screenshots",
        failures,
    )

    evaluation_path = Path("submission/evaluation.json")
    evaluation = (
        json.loads(evaluation_path.read_text(encoding="utf-8")) if evaluation_path.exists() else {}
    )
    _check(
        len(evaluation.get("comparisons", [])) >= 8,
        "At least 8 side-by-side comparisons",
        failures,
    )
    _check(sum(evaluation.get("summary", {}).values()) >= 8, "Win/loss/tie summary", failures)
    _check(Path("submission/side_by_side.md").exists(), "Markdown side-by-side report", failures)

    reflection_path = Path("submission/REFLECTION.md")
    reflection = reflection_path.read_text(encoding="utf-8") if reflection_path.exists() else ""
    sections = re.split(r"(?m)^## ", reflection)
    section_map = {part.splitlines()[0]: "\n".join(part.splitlines()[1:]) for part in sections[1:]}
    _check(len(section_map) >= 6, "Reflection has at least 6 sections", failures)
    reward_section = next(
        (body for title, body in section_map.items() if title.startswith("3.")), ""
    )
    personal_section = next(
        (body for title, body in section_map.items() if title.startswith("6.")), ""
    )
    _check(_words(reward_section) >= 150, "Reflection section 3 has >=150 words", failures)
    _check(_words(personal_section) >= 150, "Reflection section 6 has >=150 words", failures)

    notebooks = sorted(Path("notebooks").glob("*.ipynb"))
    executed = 0
    for notebook in notebooks:
        data = json.loads(notebook.read_text(encoding="utf-8"))
        if any(cell.get("execution_count") is not None for cell in data.get("cells", [])):
            executed += 1
    _check(len(notebooks) >= 5 and executed >= 5, "Five executed notebooks", failures)

    gguf = Path("gguf/lab22-dpo-Q4_K_M.gguf")
    _check(
        gguf.exists() and gguf.stat().st_size < 5 * 1024**3,
        "Bonus Q4_K_M GGUF under 5 GB",
        failures,
    )
    deploy = _load_json("submission/deploy.json")
    _check(
        deploy.get("coherent_vietnamese") is True and bool(deploy.get("response")),
        "Bonus llama.cpp Vietnamese smoke response",
        failures,
    )
    benchmark = _load_json("submission/benchmark.json")
    _check(
        all(
            name in benchmark.get("summary", {}).get("dpo", {})
            for name in ("IFEval-lite", "GSM8K-lite", "MMLU-lite", "AlpacaEval-lite")
        ),
        "Bonus four-suite benchmark report",
        failures,
    )
    hub = _load_json("submission/huggingface.json")
    _check(
        hub.get("visibility") == "public" and len(hub.get("repositories", {})) == 2,
        "Professional bonus public Hugging Face adapters",
        failures,
    )

    if failures:
        print(f"\n{len(failures)} gate(s) failed.")
        sys.exit(1)
    print("\nAll required and local bonus gates passed.")


if __name__ == "__main__":
    main()
