#!/usr/bin/env python3
"""Start Polytrack HTTP servers when needed, then run PPO training.

By default starts **eight** servers on ports **8080–8087** (one per parallel env).
On macOS/Linux extra servers use background subprocesses; see ``utils/launch_servers.py``.
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
