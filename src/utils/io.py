from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def ensure_parent_dir(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def write_json(payload: dict[str, Any], output_path: str | Path) -> None:
    ensure_parent_dir(output_path)
    with Path(output_path).open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
