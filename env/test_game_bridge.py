"""Poll ``GameBridge.get_state()`` every 100 ms for 50 iterations (~5 s wall time).

Requires Polytrack at http://127.0.0.1:8080 (e.g. ``python start_server.py``).

On the menu, speed and position stay zero until a race is active and the vehicle
exists (``car_present`` true; then physics-backed fields update).

Run from repo root: ``python -m env.test_game_bridge``

Use ``--headed --seconds 10`` to drive in the Playwright window while the terminal
prints state (headless uses a separate Chromium instance — your normal browser tab
is not polled).
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


async def _main(*, url: str, headless: bool, iterations: int) -> None:
    bridge = await GameBridge.launch(url, headless=headless)
    try:
        for i in range(iterations):
            state = await bridge.get_state()
            print(f"{i + 1:{len(str(iterations))}d}/{iterations}", state)
            await asyncio.sleep(POLL_INTERVAL_S)
    finally:
        await bridge.close()


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Poll GameBridge.get_state() at 100 ms intervals.")
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
