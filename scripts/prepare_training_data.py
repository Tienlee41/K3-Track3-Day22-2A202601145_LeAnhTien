from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from datasets import load_dataset


def _assistant_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        assistant = [
            item.get("content", "")
            for item in value
            if isinstance(item, dict) and item.get("role") == "assistant"
        ]
        if assistant:
            return str(assistant[-1]).strip()
    raise ValueError("cannot extract assistant response")


def _prompt_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        users = [
            item.get("content", "")
            for item in value
            if isinstance(item, dict) and item.get("role") == "user"
        ]
        if users:
            return str(users[-1]).strip()
    raise ValueError("cannot extract user prompt")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare SFT and DPO parquet slices")
    parser.add_argument("--config", type=Path, default=Path("configs/gpu.yaml"))
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))

    sft_cfg = config["sft"]
    sft_raw = load_dataset(sft_cfg["dataset"], split=f"train[:{sft_cfg['samples']}]")
    sft_rows = []
    for row in sft_raw:
        instruction = str(row.get("instruction", "")).strip()
        context = str(row.get("input", "")).strip()
        response = str(row.get("output", "")).strip()
        prompt = instruction if not context else f"{instruction}\n\nNgữ cảnh:\n{context}"
        if prompt and response:
            sft_rows.append({"prompt": prompt, "response": response})
    if len(sft_rows) != int(sft_cfg["samples"]):
        raise RuntimeError(f"expected {sft_cfg['samples']} SFT rows, got {len(sft_rows)}")
    sft_path = Path(config["paths"]["sft_data"])
    sft_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(sft_rows).to_parquet(sft_path, index=False)

    dpo_cfg = config["dpo"]
    pref_raw = load_dataset(dpo_cfg["dataset"], split=f"train[:{dpo_cfg['samples']}]")
    pref_rows = []
    for row in pref_raw:
        prompt = _prompt_text(row["prompt"])
        chosen = _assistant_text(row["chosen"])
        rejected = _assistant_text(row["rejected"])
        if prompt and chosen and rejected and chosen != rejected:
            pref_rows.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})
    if len(pref_rows) != int(dpo_cfg["samples"]):
        raise RuntimeError(f"expected {dpo_cfg['samples']} preference rows, got {len(pref_rows)}")
    pref_path = Path(config["paths"]["preference_data"])
    pref_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(pref_rows).to_parquet(pref_path, index=False)

    inspection = {
        "sft_rows": len(sft_rows),
        "preference_rows": len(pref_rows),
        "columns": ["prompt", "chosen", "rejected"],
        "examples": pref_rows[:3],
        "all_inspected_pairs_differ": all(
            row["chosen"] != row["rejected"] for row in pref_rows[:3]
        ),
    }
    inspection_path = pref_path.parent / "inspection.json"
    inspection_path.write_text(json.dumps(inspection, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"sft": str(sft_path), "preference": str(pref_path), **inspection}, indent=2))


if __name__ == "__main__":
    main()
