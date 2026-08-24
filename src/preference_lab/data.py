from __future__ import annotations

import json
import random
import re
from pathlib import Path

from pydantic import ValidationError

from .schemas import PreferenceExample

_EMAIL_PATTERN = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
_PHONE_PATTERN = re.compile(r"(?<!\d)(?:\+?\d[\d .()-]{7,}\d)(?!\d)")


def _prompt_key(prompt: str) -> str:
    return " ".join(prompt.casefold().split())


def _contains_pii(example: PreferenceExample) -> bool:
    text = f"{example.prompt}\n{example.chosen}\n{example.rejected}"
    return bool(_EMAIL_PATTERN.search(text) or _PHONE_PATTERN.search(text))


def load_jsonl(
    path: str | Path,
    *,
    reject_duplicate_prompts: bool = True,
    pii_guard: bool = False,
) -> list[PreferenceExample]:
    """Load preference examples from JSONL.

    Errors include the source line number. Duplicate prompts are rejected after
    whitespace/case normalisation. Set ``pii_guard`` to reject likely emails and
    phone numbers before data enters a training pipeline.
    """
    source = Path(path)
    examples: list[PreferenceExample] = []
    prompt_lines: dict[str, int] = {}

    with source.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{source}:{line_number}: invalid JSON - {exc.msg}") from exc
            try:
                example = PreferenceExample.model_validate(payload)
            except ValidationError as exc:
                raise ValueError(f"{source}:{line_number}: invalid schema - {exc}") from exc

            prompt_key = _prompt_key(example.prompt)
            if reject_duplicate_prompts and prompt_key in prompt_lines:
                first_line = prompt_lines[prompt_key]
                raise ValueError(
                    f"{source}:{line_number}: duplicate prompt (first seen on line {first_line})"
                )
            if pii_guard and _contains_pii(example):
                raise ValueError(f"{source}:{line_number}: possible PII detected")

            prompt_lines[prompt_key] = line_number
            examples.append(example)
    return examples


def split_by_prompt(
    examples: list[PreferenceExample],
    validation_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[list[PreferenceExample], list[PreferenceExample]]:
    """Split examples by prompt to avoid leakage.

    Prompt groups, rather than individual rows, are shuffled deterministically.
    The original row order is retained inside each resulting split.
    """
    if not 0.0 < validation_ratio < 1.0:
        raise ValueError("validation_ratio must be between 0 and 1")
    if not examples:
        return [], []

    prompt_keys = list(dict.fromkeys(_prompt_key(example.prompt) for example in examples))
    if len(prompt_keys) == 1:
        return list(examples), []

    random.Random(seed).shuffle(prompt_keys)
    validation_groups = max(1, round(len(prompt_keys) * validation_ratio))
    validation_groups = min(validation_groups, len(prompt_keys) - 1)
    validation_keys = set(prompt_keys[:validation_groups])

    train = [example for example in examples if _prompt_key(example.prompt) not in validation_keys]
    validation = [example for example in examples if _prompt_key(example.prompt) in validation_keys]
    return train, validation
