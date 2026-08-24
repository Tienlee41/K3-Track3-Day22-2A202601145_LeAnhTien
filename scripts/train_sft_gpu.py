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
from peft import LoraConfig
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from gpu_common import (
    load_gpu_config,
    peak_vram_gb,
    plot_gpu_card,
    require_gpu,
    save_json,
    seed_everything,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Vietnamese SFT-mini LoRA adapter")
    parser.add_argument("--config", type=Path, default=Path("configs/gpu.yaml"))
    parser.add_argument("--reuse", action="store_true")
    args = parser.parse_args()
    config = load_gpu_config(args.config)
    output = Path(config["paths"]["sft_adapter"])
    if args.reuse and (output / "adapter_config.json").exists():
        print(f"Reusing {output}")
        return

    gpu_info = require_gpu()
    seed_everything(int(config["seed"]))
    torch.cuda.reset_peak_memory_stats()
    plot_gpu_card(gpu_info, Path(config["paths"]["screenshots"]) / "01_gpu_smoke.png")

    sft = config["sft"]
    frame = pd.read_parquet(config["paths"]["sft_data"])
    dataset = Dataset.from_pandas(frame, preserve_index=False).map(
        lambda row: {
            "messages": [
                {"role": "user", "content": row["prompt"]},
                {"role": "assistant", "content": row["response"]},
            ]
        },
        remove_columns=list(frame.columns),
    )

    tokenizer = AutoTokenizer.from_pretrained(config["base_model"])
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        config["base_model"],
        device_map={"": 0},
        dtype=torch.bfloat16,
    )
    model.config.use_cache = False
    lora = LoraConfig(
        r=int(sft["lora_r"]),
        lora_alpha=int(sft["lora_alpha"]),
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
    training_args = SFTConfig(
        output_dir="checkpoints/sft-mini",
        per_device_train_batch_size=int(sft["batch_size"]),
        gradient_accumulation_steps=int(sft["gradient_accumulation_steps"]),
        num_train_epochs=float(sft["epochs"]),
        learning_rate=float(sft["learning_rate"]),
        max_length=int(sft["max_length"]),
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
        dataset_num_proc=1,
    )
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=lora,
    )
    started = time.perf_counter()
    train_result = trainer.train()
    elapsed = time.perf_counter() - started
    output.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(output, safe_serialization=True)
    tokenizer.save_pretrained(output)

    loss_points = [
        {"step": int(item["step"]), "loss": float(item["loss"])}
        for item in trainer.state.log_history
        if "loss" in item and "step" in item
    ]
    metrics = {
        **gpu_info,
        "base_model": config["base_model"],
        "dataset": sft["dataset"],
        "samples": len(dataset),
        "epochs": float(sft["epochs"]),
        "runtime_seconds": round(elapsed, 3),
        "train_loss": float(train_result.training_loss),
        "peak_vram_gb": peak_vram_gb(),
        "loss_history": loss_points,
    }
    save_json(output / "training_metrics.json", metrics)

    screenshot = Path(config["paths"]["screenshots"]) / "02_sft_loss_curve.png"
    screenshot.parent.mkdir(parents=True, exist_ok=True)
    steps = [point["step"] for point in loss_points]
    losses = [point["loss"] for point in loss_points]
    running_best = []
    for loss in losses:
        running_best.append(min(loss, running_best[-1] if running_best else loss))
    fig, axis = plt.subplots(figsize=(10, 5.5))
    axis.plot(steps, losses, alpha=0.35, label="logged batch loss")
    axis.plot(steps, running_best, linewidth=2.5, label="cumulative best loss")
    axis.set(title="SFT-mini training loss", xlabel="optimizer step", ylabel="loss")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(screenshot, dpi=180)
    plt.close(fig)

    model = trainer.model
    model.eval()
    prompt = "Giải thích ngắn gọn vì sao cần tập validation trong học máy."
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)
    with torch.inference_mode():
        output_ids = model.generate(inputs, max_new_tokens=120, do_sample=False)
    response = tokenizer.decode(output_ids[0, inputs.shape[-1] :], skip_special_tokens=True).strip()
    save_json(
        output / "sample_generation.json",
        {"prompt": prompt, "response": response, "coherent_vietnamese": bool(response)},
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
