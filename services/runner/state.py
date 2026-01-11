from pathlib import Path
from typing import Dict, Optional

from .utils import outputs_dir, read_json, write_json

STATE_DIR = outputs_dir() / "state"


def state_path(url_hash: str) -> Path:
    return STATE_DIR / f"{url_hash}.json"


def load_state(url_hash: str) -> Optional[Dict]:
    path = state_path(url_hash)
    if not path.exists():
        return None
    return read_json(path)


def mark_state(url_hash: str, status: str, payload: Optional[Dict] = None) -> Dict:
    state = {"url_hash": url_hash, "status": status}
    if payload:
        state.update(payload)
    write_json(state_path(url_hash), state)
    return state
