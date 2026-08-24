from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise TypeError("config root must be a YAML mapping")
    return cast(dict[str, Any], config)
