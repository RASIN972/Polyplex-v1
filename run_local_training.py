#!/usr/bin/env python3
"""Start Polytrack HTTP servers when needed, then run PPO training.

By default starts **eight** servers on ports **8080–8087** (one per parallel env).
On macOS/Linux extra servers use background subprocesses; see ``utils/launch_servers.py``.

Windows / Playwright: the first ``env.reset()`` path waits for the in-game track menu
(DOM selector timeout is 120s in ``GameBridge``). Wall time before Steps tick or before
a failure can be **~3 minutes** (e.g. ~2m54s reported). If the menu never appears you
get ``RuntimeError: track menu: timed out waiting for track list after Play`` — see
``docs/WINDOWS_TRAINING.md`` (startup errors). Ctrl+C often prints long Playwright /
``multiprocessing`` tracebacks and ``BrokenPipeError`` while workers tear down; that is
expected; HTTP server processes may still be running.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent


def main() -> int:
    print(
        "\n"
        " ╔══════════════════════════════════════════════════════════════╗\n"
        " ║  Polyplex — local PPO training (8 parallel envs by default)     ║\n"
        " ║  • HTTP servers: 127.0.0.1:8080–8087 (see logs/)            ║\n"
        " ║  • Ctrl+C stops training — servers may keep running         ║\n"
        " ║  • First reset may take ~3 min (track menu) — see WINDOWS  ║\n"
        " ║    training doc for timeouts / Ctrl+C worker noise          ║\n"
        " ║  • Docs: CONTEXT.md, docs/WINDOWS_TRAINING.md               ║\n"
        " ╚══════════════════════════════════════════════════════════════╝\n",
        flush=True,
    )

    if not os.environ.get("POLYTRACK_SKIP_SERVER_LAUNCH"):
        from utils.launch_servers import ensure_servers_running

        ensure_servers_running(list(range(8080, 8088)))

    train_cmd = [
        sys.executable,
        "-u",
        str(_ROOT / "training" / "train_ppo.py"),
        *sys.argv[1:],
    ]
    env = os.environ.copy()
    env["POLYTRACK_FROM_RUN_LOCAL"] = "1"
    return int(subprocess.run(train_cmd, cwd=str(_ROOT), env=env).returncode)


if __name__ == "__main__":
    raise SystemExit(main())
