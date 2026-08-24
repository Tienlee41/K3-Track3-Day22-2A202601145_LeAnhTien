---
base_model: unsloth/Qwen2.5-3B-Instruct-bnb-4bit
library_name: peft
pipeline_tag: text-generation
language:
- vi
- en
tags:
- lora
- dpo
- preference-alignment
- transformers
- trl
datasets:
- argilla/ultrafeedback-binarized-preferences-cleaned
---

# Day 22 Qwen2.5-3B DPO LoRA

This is the preference-alignment stage of a local-GPU lab. The Vietnamese SFT-mini LoRA was merged
into `unsloth/Qwen2.5-3B-Instruct-bnb-4bit`, then this distinct adapter was trained on a deterministic
2,000-pair slice of `argilla/ultrafeedback-binarized-preferences-cleaned`.

## Training

| Setting | Value |
|---|---:|
| LoRA rank / alpha | 16 / 16 |
| DPO beta | 0.1 |
| Learning rate | 5e-7 |
| Epochs / optimizer steps | 1 / 250 |
| Max sequence / prompt length | 512 / 256 |
| Micro-batch / accumulation | 1 / 8 |
| Train loss | 0.6761 |
| Runtime including reference pass | 2,777.4 s |
| Peak allocated VRAM | 4.932 GB |

The endpoint reward gap moved from -0.0170 to +0.0135. The mean of the first five logged gaps was
-0.0077; the mean of the last five was +0.0383, a +0.0460 improvement. Curves show likelihood
displacement in parts of the run: chosen reward sometimes decreases, while rejected reward decreases
more strongly. See `training_metrics.json` for all 51 logged points.

## Use and limitations

Attach this adapter only after reproducing the SFT merge used during training. This educational run
does not establish production safety, broad factuality, or official benchmark performance. The
preference labels are model-generated/model-judged and may encode evaluator bias. Review outputs,
especially for high-stakes or multilingual use.

Source code and evaluation evidence:
https://github.com/Tienlee41/K3-Track3-Day22-2A202601145_LeAnhTien
