#!/usr/bin/env python3
"""Launch the Polyplex training control GUI (React dashboard).

Usage:
    python start_gui.py

Serves the built React app from dashboard/dist and opens it in your browser.
On a fresh machine you only need Python deps — the UI is already built.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_DIST = _ROOT / "dashboard" / "dist"


def _ensure_deps() -> None:
    missing: list[str] = []
    for mod in ("fastapi", "uvicorn", "pydantic"):
        try:
            __import__(mod)
        except ModuleNotFoundError:
            missing.append(mod)
    if missing:
        print(
            "Missing Python packages: " + ", ".join(missing) + "\n"
            f"Install with this Python ({sys.executable}):\n"
            f"  {sys.executable} -m pip install -r requirements.txt\n",
            file=sys.stderr,
        )
        raise SystemExit(1)
    if not _DIST.is_dir():
        print(
            "React dashboard is not built (dashboard/dist missing).\n"
            "On a machine with Node.js:\n"
            "  cd dashboard\n"
            "  npm install\n"
            "  npm run build\n"
            "Then copy dashboard/dist to this project (or re-run start_gui here).",
            file=sys.stderr,
        )
        raise SystemExit(1)


if __name__ == "__main__":
    _ensure_deps()
    from gui_backend import main

    main()
