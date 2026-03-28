"""Append-only NDJSON debug log for Cursor debug mode."""

from __future__ import annotations

import json
import time
from pathlib import Path

_LOG_PATH = Path(__file__).resolve().parent.parent / ".cursor" / "debug-457e3f.log"
_SESSION_ID = "457e3f"


def debug_ndjson_log_path() -> Path:
    """NDJSON path used by agent_debug_log (Playwright repro / game_bridge)."""
    return _LOG_PATH


def agent_debug_log(
    *,
    run_id: str,
    hypothesis_id: str,
    location: str,
    message: str,
    data: dict,
) -> None:
    # region agent log
    payload = {
        "sessionId": _SESSION_ID,
        "runId": run_id,
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
    }
    _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    # endregion
