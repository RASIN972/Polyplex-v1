"""Poll ``GameBridge.get_state()`` every 100 ms for N iterations.

Requires Polytrack at http://127.0.0.1:8080 (e.g. ``python start_server.py``).

On the menu, speed and position stay zero until a race is active and the vehicle
exists (``car_present`` true; then physics-backed fields update).

Run from repo root: ``python -m env.test_game_bridge``

Use ``--headed --seconds 10`` to drive in the Playwright window while the terminal
prints state (headless uses a separate Chromium instance — your normal browser tab
is not polled).

Slots 10–11 verification: once a race is running, ``waypoint_rel`` and
``dist_to_checkpoint`` should both be non-zero; that confirms the bundle patch
(``ghostData.track = m``) and the rAF waypoint code are working.
"""

from __future__ import annotations

import argparse
import asyncio
import math
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from env.game_bridge import GameBridge

POLL_INTERVAL_S = 0.1


def _fmt_state(i: int, n: int, s: dict) -> str:
    """Format a state dict with waypoint fields highlighted."""
    prefix = f"{i:{len(str(n))}d}/{n}"
    if s.get("error"):
        return f"{prefix}  ERROR: {s['error']}"

    wr = s.get("waypoint_rel") or {}
    dist = s.get("dist_to_checkpoint", 0)
    waypoint_str = (
        f"  wp=({wr.get('x', 0):+6.0f},{wr.get('z', 0):+6.0f})"
        f"  dst={dist:5.0f}"
    )

    walls = s.get("wall_dists") or []
    labels = ["F", "FR", "R", "L", "FL", "B"]
    wall_str = "  walls=[" + " ".join(
        f"{labels[i]}:{walls[i]:4.0f}" if i < len(walls) else f"{labels[i]}:----"
        for i in range(6)
    ) + "]"

    pos = s.get("position", {})
    return (
        f"{prefix}"
        f"  spd={s.get('speed', 0):5.1f}"
        f"  cp={s.get('checkpoint_index', 0)}"
        f"  car={int(s.get('car_present', False))}"
        f"  go={int(s.get('has_started', False))}"
        f"{waypoint_str}"
        f"{wall_str}"
        f"  pos=({pos.get('x', 0):.0f},{pos.get('y', 0):.0f},{pos.get('z', 0):.0f})"
    )


async def _main(*, url: str, headless: bool, iterations: int) -> None:
    bridge = await GameBridge.launch(url, headless=headless)
    print(
        "Polling state every 100 ms. Drive in the game window to see waypoint_rel update.\n"
        "Slots 10-11 check: waypoint_rel x/z should be non-zero once a race is running.\n"
    )
    try:
        for i in range(iterations):
            state = await bridge.get_state()
            print(_fmt_state(i + 1, iterations, state))
            await asyncio.sleep(POLL_INTERVAL_S)
    finally:
        await bridge.close()


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Poll GameBridge.get_state() at 100 ms intervals, printing waypoint fields."
    )
    p.add_argument("--url", default="http://127.0.0.1:8080/", help="Polytrack origin")
    p.add_argument(
        "--seconds",
        type=float,
        default=5.0,
        help="How long to poll (default: 5). Sample count = ceil(seconds / 0.1).",
    )
    p.add_argument(
        "--headed",
        action="store_true",
        help="Show Chromium so you can click/drive in that window while state prints here.",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    n = max(1, math.ceil(args.seconds / POLL_INTERVAL_S))
    asyncio.run(_main(url=args.url, headless=not args.headed, iterations=n))
