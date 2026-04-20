"""
Interactive repro for post-finish broken state.

Requires: polytrack HTTP server on 127.0.0.1:8080, Playwright Chromium installed.

Run:  .venv/bin/python -m utils.debug_finish_repro

Keep browser open (no Enter): POLYTRACK_DEBUG_REPRO_KEEP_ALIVE=1 .venv/bin/python -m utils.debug_finish_repro
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from env.chromium_launch_args import (
    POLYTRACK_CHROMIUM_IGNORE_DEFAULT_ARGS,
    polytrack_chromium_launch_args,
)
from env.debug_logging import agent_debug_log, debug_ndjson_log_path
from env.playwright_routes import install_polytrack_offline_routes
from env.game_bridge import FinishDebugGameBridge, collect_dom_probe
from playwright.async_api import async_playwright


async def _wait_enter() -> None:
    """Block until Enter, or forever if POLYTRACK_DEBUG_REPRO_KEEP_ALIVE=1 (no TTY stdin)."""
    if os.environ.get("POLYTRACK_DEBUG_REPRO_KEEP_ALIVE") == "1":
        print(
            "POLYTRACK_DEBUG_REPRO_KEEP_ALIVE=1 — browser stays open; "
            "Ctrl+C in this terminal when finished."
        )
        await asyncio.Event().wait()
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, input)


async def _main() -> None:
    log_path = debug_ndjson_log_path()
    print(f"NDJSON debug log (append-only): {log_path}")
    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=False,
            args=list(polytrack_chromium_launch_args(headless=False)),
            ignore_default_args=list(POLYTRACK_CHROMIUM_IGNORE_DEFAULT_ARGS),
        )
        page = await browser.new_page(viewport={"width": 1280, "height": 720})
        await install_polytrack_offline_routes(page)
        await page.goto("http://127.0.0.1:8080/", wait_until="domcontentloaded")
        print("Loaded http://127.0.0.1:8080/ — play, finish the track, then cause the broken state")
        print("(e.g. run your env reset()). When frozen/unresponsive, press Enter for snapshot A.")
        await _wait_enter()
        probe_a = await collect_dom_probe(page)
        # region agent log
        agent_debug_log(
            run_id="repro-manual-broken",
            hypothesis_id="H1-H5",
            location="debug_finish_repro.py:main",
            message="probe_user_broken_state",
            data=probe_a,
        )
        # endregion
        print(f"Snapshot A appended to {log_path}")
        print("Snapshot A logged. Press Enter to run FinishDebugGameBridge.reset() (KeyR via Playwright) …")
        await _wait_enter()
        bridge = FinishDebugGameBridge(page)
        await bridge.reset(run_id="repro-after-bridge-reset")
        print(f"Done. Log file: {log_path}")
        await browser.close()


if __name__ == "__main__":
    asyncio.run(_main())
