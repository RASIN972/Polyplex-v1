#!/usr/bin/env python3
"""Start Polytrack HTTP servers when needed, then run PPO training.

Default: **4** parallel envs on ports **8080–8083** (matches ``training/train_ppo.py``).
Pass ``--num-envs N`` to scale servers + workers together.

Windows / Playwright: the first ``env.reset()`` may take ~1–3 minutes while the
track menu loads. See ``docs/WINDOWS_TRAINING.md``. Ctrl+C often prints
Playwright / ``BrokenPipeError`` noise while subprocess workers tear down.
"""

from __future__ import annotations

import argparse
import multiprocessing
import os
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
_DEFAULT_NUM_ENVS = 4
_BASE_PORT = 8080


def _parse_forwarded_num_envs(argv: list[str]) -> int:
    """Read --num-envs from argv without consuming the rest (forwarded to train_ppo)."""
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--num-envs", type=int, default=_DEFAULT_NUM_ENVS)
    args, _ = p.parse_known_args(argv)
    return max(1, int(args.num_envs))


def main() -> int:
    num_envs = _parse_forwarded_num_envs(sys.argv[1:])
    ports = list(range(_BASE_PORT, _BASE_PORT + num_envs))
    port_hi = ports[-1]

    print(
        "\n"
        " ╔══════════════════════════════════════════════════════════════╗\n"
        f" ║  Polyplex — local PPO training ({num_envs} parallel envs)              ║\n"
        f" ║  • HTTP servers: 127.0.0.1:{_BASE_PORT}–{port_hi}                      ║\n"
        " ║  • PPO device: CPU (AMD GPU not used for this tiny MLP)     ║\n"
        " ║  • Ctrl+C stops training — servers may keep running         ║\n"
        " ║  • First reset may take 1–3 min (track menu) on Windows     ║\n"
        " ║  • Docs: CONTEXT.md, docs/WINDOWS_TRAINING.md               ║\n"
        " ╚══════════════════════════════════════════════════════════════╝\n",
        flush=True,
    )

    if not os.environ.get("POLYTRACK_SKIP_SERVER_LAUNCH"):
        from utils.launch_servers import ensure_servers_running

        ensure_servers_running(ports)

    train_cmd = [
        sys.executable,
        "-u",
        str(_ROOT / "training" / "train_ppo.py"),
        *sys.argv[1:],
    ]
    # Ensure train_ppo sees the same num-envs if the user omitted it.
    if "--num-envs" not in sys.argv[1:]:
        train_cmd.extend(["--num-envs", str(num_envs)])

    env = os.environ.copy()
    env["POLYTRACK_FROM_RUN_LOCAL"] = "1"
    return int(subprocess.run(train_cmd, cwd=str(_ROOT), env=env).returncode)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main())
