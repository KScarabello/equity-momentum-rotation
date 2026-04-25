from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


DEFAULT_STATE: Dict[str, Any] = {
    "last_rebalance_date": None,
    "last_successful_run_ts": None,
    "last_target_symbols": [],
    "last_mode": None,
    "last_cycle_key": None,
    "first_run_liquidation_done": False,
}


def load_state(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return DEFAULT_STATE.copy()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return DEFAULT_STATE.copy()
    out = DEFAULT_STATE.copy()
    if isinstance(data, dict):
        out.update(data)
    return out


def save_state(path: Path, state: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
