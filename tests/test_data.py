from pathlib import Path

import pytest

from preference_lab.data import load_jsonl, split_by_prompt
from preference_lab.schemas import PreferenceExample


def test_load_sample_data() -> None:
    examples = load_jsonl("data/sample_preferences.jsonl")
    assert len(examples) == 24
    assert examples[0].chosen != examples[0].rejected


def test_error_message_includes_line_number(tmp_path: Path) -> None:
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"prompt":"a","chosen":"b","rejected":"c"}\n{oops\n', encoding="utf-8")
    with pytest.raises(ValueError, match=r"bad\.jsonl:2: invalid JSON"):
        load_jsonl(bad)


def test_duplicate_prompt_is_rejected(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.jsonl"
    duplicate.write_text(
        '{"prompt":"Prompt", "chosen":"good", "rejected":"bad"}\n'
        '{"prompt":" prompt ", "chosen":"better", "rejected":"worse"}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=r"duplicate prompt.*line 1"):
        load_jsonl(duplicate)


def test_pii_guard_is_opt_in(tmp_path: Path) -> None:
    pii = tmp_path / "pii.jsonl"
    pii.write_text(
        '{"prompt":"Email me at student@example.com","chosen":"safe", "rejected":"unsafe"}\n',
        encoding="utf-8",
    )
    assert len(load_jsonl(pii)) == 1
    with pytest.raises(ValueError, match="possible PII"):
        load_jsonl(pii, pii_guard=True)


def test_schema_rejects_normalised_duplicates() -> None:
    with pytest.raises(ValueError, match="meaningfully different"):
        PreferenceExample(prompt="p", chosen="Same answer!", rejected=" same ANSWER ")


def test_split_has_no_prompt_leakage_and_is_deterministic() -> None:
    examples = load_jsonl("data/sample_preferences.jsonl")
    train, validation = split_by_prompt(examples, validation_ratio=0.5, seed=7)
    train_again, validation_again = split_by_prompt(examples, validation_ratio=0.5, seed=7)

    assert len(train) + len(validation) == len(examples)
    assert not ({example.prompt for example in train} & {example.prompt for example in validation})
    assert [example.prompt for example in train] == [example.prompt for example in train_again]
    assert [example.prompt for example in validation] == [
        example.prompt for example in validation_again
    ]


def test_split_rejects_invalid_ratio() -> None:
    with pytest.raises(ValueError, match="validation_ratio"):
        split_by_prompt([], validation_ratio=1.0)
