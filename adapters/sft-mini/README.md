---
base_model: unsloth/Qwen2.5-3B-Instruct-bnb-4bit
library_name: peft
pipeline_tag: text-generation
language:
- vi
- en
tags:
- lora
- sft
- transformers
- trl
datasets:
- bkai-foundation-models/vi-alpaca
---

# Day 22 Qwen2.5-3B Vietnamese SFT-mini LoRA

This is the SFT stage of a local-GPU preference-alignment lab. It adapts
`unsloth/Qwen2.5-3B-Instruct-bnb-4bit` on a deterministic 1,000-example slice of
`bkai-foundation-models/vi-alpaca` before a distinct DPO adapter is trained.

## Training

| Setting | Value |
|---|---:|
| LoRA rank / alpha | 16 / 32 |
| Epochs | 1 |
| Max sequence length | 512 |
| Learning rate | 2e-4 |
| Effective batch size | 8 |
| Train loss | 0.9616 |
| Runtime | 433.5 s |
| Peak allocated VRAM | 3.099 GB |

The run used BF16 compute on an NVIDIA RTX 5060 Ti 16 GB. Full reproducibility metadata,
loss history, tokenizer files, and a Vietnamese generation smoke test are included in this folder.

## Use and limitations

Load the base model in 4-bit and attach this adapter with PEFT. This is an educational checkpoint,
not a production safety model. The small training slice does not establish broad factuality,
robustness, fairness, or medical/legal reliability. Review generated content before downstream use.

Source code and evaluation evidence:
https://github.com/Tienlee41/K3-Track3-Day22-2A202601145_LeAnhTien
