import json
import os
from pathlib import Path
from typing import Any, Dict


def get_root() -> Path:
    return Path(os.environ.get("POETRYBOT_ROOT", "/opt/poetrybot"))


def outputs_dir() -> Path:
    return get_root() / "outputs"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Dict[str, Any]) -> Path:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    return path
