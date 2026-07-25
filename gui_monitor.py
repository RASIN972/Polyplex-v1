#!/usr/bin/env python3
"""Compatibility entrypoint — the GUI is now the React dashboard.

Prefer::

    python start_gui.py
"""

from gui_backend import main

if __name__ == "__main__":
    main()
