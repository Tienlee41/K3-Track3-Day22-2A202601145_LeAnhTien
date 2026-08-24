from __future__ import annotations

# ruff: noqa: I001 -- compatibility import must precede third-party GPU imports

import argparse
import json
from pathlib import Path
import time

import windows_platform_compat  # noqa: F401  # must precede torch/PEFT imports

from datasets import Dataset
import matplotlib.pyplot as plt
import pandas as pd
from peft import LoraConfig, PeftModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

from gpu_common import load_gpu_config, peak_vram_gb, require_gpu, save_json, seed_everything


def _metric(item: dict[str, object], *names: str) -> float | None:
    for name in names:
        value = item.get(name)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a DPO LoRA adapter on the SFT model")
    parser.add_argument("--config", type=Path, default=Path("configs/gpu.yaml"))
    parser.add_argument("--reuse", action="store_true")
    args = parser.parse_args()
    config = load_gpu_config(args.config)
    output = Path(config["paths"]["dpo_adapter"])
    if args.reuse and (output / "adapter_config.json").exists():
        print(f"Reusing {output}")
        return

    gpu_info = require_gpu()
    seed_everything(int(config["seed"]))
    torch.cuda.reset_peak_memory_stats()
    dpo = config["dpo"]

    frame = pd.read_parquet(config["paths"]["preference_data"])
    rows = [
        {
            "prompt": [{"role": "user", "content": row.prompt}],
            "chosen": [{"role": "assistant", "content": row.chosen}],
            "rejected": [{"role": "assistant", "content": row.rejected}],
        }
        for row in frame.itertuples(index=False)
    ]
    dataset = Dataset.from_list(rows)

    tokenizer = AutoTokenizer.from_pretrained(config["base_model"])
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    base = AutoModelForCausalLM.from_pretrained(
        config["base_model"],
        device_map={"": 0},
        dtype=torch.bfloat16,
    )
    base.config.use_cache = False
    sft_model = PeftModel.from_pretrained(base, config["paths"]["sft_adapter"])
    merged_sft = sft_model.merge_and_unload()
    if hasattr(merged_sft, "peft_config"):
        delattr(merged_sft, "peft_config")
    lora = LoraConfig(
        r=int(dpo["lora_r"]),
        lora_alpha=int(dpo["lora_alpha"]),
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    training_args = DPOConfig(
        output_dir="checkpoints/dpo",
        per_device_train_batch_size=int(dpo["batch_size"]),
        gradient_accumulation_steps=int(dpo["gradient_accumulation_steps"]),
        num_train_epochs=float(dpo["epochs"]),
        learning_rate=float(dpo["learning_rate"]),
        beta=float(dpo["beta"]),
        max_length=int(dpo["max_length"]),
        max_prompt_length=int(dpo["max_prompt_length"]),
        logging_steps=5,
        logging_first_step=True,
        save_strategy="no",
        report_to="none",
        bf16=True,
        tf32=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch_fused",
        seed=int(config["seed"]),
        precompute_ref_log_probs=True,
        precompute_ref_batch_size=4,
    )
    trainer = DPOTrainer(
        model=merged_sft,
        ref_model=None,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=lora,
    )
    started = time.perf_counter()
    reference_cache = Path("checkpoints/dpo_reference_logps.parquet")
    reference_cache_reused = reference_cache.exists()
    if reference_cache_reused:
        cached = pd.read_parquet(reference_cache)
        if len(cached) != len(trainer.train_dataset):
            raise RuntimeError("DPO reference cache row count does not match the training dataset")
        trainer.train_dataset = trainer.train_dataset.add_column(
            "ref_chosen_logps", cached["ref_chosen_logps"].tolist()
        )
        trainer.train_dataset = trainer.train_dataset.add_column(
            "ref_rejected_logps", cached["ref_rejected_logps"].tolist()
        )
        trainer._precomputed_train_ref_log_probs = True
        print(f"Reused reference log-prob cache: {reference_cache}")
    else:
        trainer.get_train_dataloader()
        reference_cache.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {
                "ref_chosen_logps": trainer.train_dataset["ref_chosen_logps"],
                "ref_rejected_logps": trainer.train_dataset["ref_rejected_logps"],
            }
        ).to_parquet(reference_cache, index=False)
        print(f"Saved reference log-prob cache: {reference_cache}")
    train_result = trainer.train()
    elapsed = time.perf_counter() - started
    output.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(output, safe_serialization=True)
    tokenizer.save_pretrained(output)

    history = []
    for item in trainer.state.log_history:
        chosen = _metric(item, "rewards/chosen", "train_rewards/chosen")
        rejected = _metric(item, "rewards/rejected", "train_rewards/rejected")
        margin = _metric(item, "rewards/margins", "train_rewards/margins")
        if chosen is not None and rejected is not None:
            history.append(
                {
                    "step": int(item.get("step", len(history) + 1)),
                    "chosen_reward": chosen,
                    "rejected_reward": rejected,
                    "reward_gap": margin if margin is not None else chosen - rejected,
                }
            )
    if not history:
        raise RuntimeError(
            "DPO trainer did not emit reward metrics; keys were: "
            + str(sorted({key for item in trainer.state.log_history for key in item}))
        )

    trend_window = min(5, len(history))
    initial_gap_trend = sum(point["reward_gap"] for point in history[:trend_window]) / trend_window
    final_gap_trend = sum(point["reward_gap"] for point in history[-trend_window:]) / trend_window
    metrics = {
        **gpu_info,
        "base_model": config["base_model"],
        "dataset": dpo["dataset"],
        "samples": len(dataset),
        "epochs": float(dpo["epochs"]),
        "beta": float(dpo["beta"]),
        "learning_rate": float(dpo["learning_rate"]),
        "batch_size": int(dpo["batch_size"]),
        "gradient_accumulation_steps": int(dpo["gradient_accumulation_steps"]),
        "reference_cache_reused": reference_cache_reused,
        "runtime_seconds": round(elapsed, 3),
        "train_loss": float(train_result.training_loss),
        "peak_vram_gb": peak_vram_gb(),
        "reward_history": history,
        "initial_reward_gap": history[0]["reward_gap"],
        "final_reward_gap": history[-1]["reward_gap"],
        "reward_gap_trend_window": trend_window,
        "initial_reward_gap_trend": initial_gap_trend,
        "final_reward_gap_trend": final_gap_trend,
    }
    save_json(output / "training_metrics.json", metrics)

    screenshot = Path(config["paths"]["screenshots"]) / "03_dpo_reward_curves.png"
    screenshot.parent.mkdir(parents=True, exist_ok=True)
    steps = [point["step"] for point in history]
    chosen_rewards = [point["chosen_reward"] for point in history]
    rejected_rewards = [point["rejected_reward"] for point in history]
    gaps = [point["reward_gap"] for point in history]
    fig, axis = plt.subplots(figsize=(11, 6))
    axis.plot(steps, chosen_rewards, label="chosen reward", linewidth=2)
    axis.plot(steps, rejected_rewards, label="rejected reward", linewidth=2)
    axis.plot(steps, gaps, label="reward gap (chosen - rejected)", linewidth=3)
    axis.axhline(0.0, color="black", linewidth=1, alpha=0.5)
    axis.set(title="DPO reward trajectories", xlabel="optimizer step", ylabel="reward")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(screenshot, dpi=180)
    plt.close(fig)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
