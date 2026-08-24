from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any

from pydantic import BaseModel, Field, ValidationInfo, field_validator


def _normalise_text(value: str) -> str:
    """Normalise text for duplicate detection without changing stored content."""
    return " ".join(re.findall(r"\w+", value.casefold()))


class PreferenceExample(BaseModel):
    """One preference pair for DPO/ORPO-style alignment."""

    prompt: str = Field(min_length=1)
    chosen: str = Field(min_length=1)
    rejected: str = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("prompt", "chosen", "rejected")
    @classmethod
    def strip_text(cls, value: str) -> str:
        return value.strip()

    @field_validator("rejected")
    @classmethod
    def chosen_and_rejected_must_differ(cls, rejected: str, info: ValidationInfo) -> str:
        chosen = info.data.get("chosen")
        if not isinstance(chosen, str):
            return rejected

        normalised_chosen = _normalise_text(chosen)
        normalised_rejected = _normalise_text(rejected)
        similarity = SequenceMatcher(None, normalised_chosen, normalised_rejected).ratio()
        if normalised_chosen == normalised_rejected or similarity >= 0.98:
            raise ValueError("chosen and rejected must be meaningfully different")
        return rejected
