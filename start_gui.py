#!/usr/bin/env python3
"""Launch the Polyplex training control GUI (modern NiceGUI web dashboard).

Usage:
    python start_gui.py

Opens in your browser. From the dashboard you can start/stop training,
set num envs / headless / watch, view live metrics + an interactive progress
graph, and replay best runs.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from gui_monitor import main

if __name__ == "__main__":
    main()
