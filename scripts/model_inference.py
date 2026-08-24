from __future__ import annotations

# ruff: noqa: I001 -- compatibility import must precede third-party GPU imports

import gc
from typing import Any

import windows_platform_compat  # noqa: F401  # must precede torch/PEFT imports

from peft import PeftModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_variant(config: dict[str, Any], variant: str) -> tuple[Any, Any]:
    if variant not in {"sft", "dpo"}:
        raise ValueError("variant must be sft or dpo")
    tokenizer = AutoTokenizer.from_pretrained(config["base_model"])
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    base = AutoModelForCausalLM.from_pretrained(
        config["base_model"],
        device_map={"": 0},
        dtype=torch.bfloat16,
    )
    sft_model = PeftModel.from_pretrained(base, config["paths"]["sft_adapter"])
    if variant == "sft":
        model = sft_model
    else:
        merged = sft_model.merge_and_unload()
        if hasattr(merged, "peft_config"):
            delattr(merged, "peft_config")
        model = PeftModel.from_pretrained(merged, config["paths"]["dpo_adapter"])
    model.eval()
    model.config.use_cache = True
    model.generation_config.do_sample = False
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None
    return model, tokenizer


def generate(model: Any, tokenizer: Any, prompt: str, max_new_tokens: int) -> str:
    messages = [{"role": "user", "content": prompt}]
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)
    attention_mask = torch.ones_like(input_ids)
    with torch.inference_mode():
        output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(output[0, input_ids.shape[-1] :], skip_special_tokens=True).strip()


def clear_cuda() -> None:
    """Release unreferenced CUDA allocations between model variants."""
    gc.collect()
    torch.cuda.empty_cache()
