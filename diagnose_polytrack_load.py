#!/usr/bin/env python3
"""Diagnose why the local Polytrack copy stays on the loading screen."""

from __future__ import annotations

import argparse
import asyncio
import os
import socket
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Any

from playwright.async_api import async_playwright

ROOT = Path(__file__).resolve().parent
GAME_ROOT = ROOT / "polytrackcopy"
REQUIRED_FILES = (
    "index.html",
    "js/2176-lib-ammo.js",
    "js/9209-dist-main.bundle.js",
    "css/css-loading_ui.css",
    "css/css-menu.css",
    "css/css-theme.css",
    "images/logo.svg",
    "images/play.svg",
)


def _port_open(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.25):
            return True
    except OSError:
        return False


def _check_files() -> bool:
    ok = True
    print("File check:")
    for rel in REQUIRED_FILES:
        path = GAME_ROOT / rel
        if path.is_file():
            print(f"  OK      {rel} ({path.stat().st_size} bytes)")
        else:
            ok = False
            print(f"  MISSING {rel}")

    bundle = GAME_ROOT / "js" / "9209-dist-main.bundle.js"
    if bundle.is_file():
        text = bundle.read_text(encoding="utf-8", errors="replace")
        marker = 'uk = "/offline-polytrack-api/"'
        print(f"  {'OK' if marker in text else 'OLD?'}     offline API patch")
        ok = ok and marker in text

    index = GAME_ROOT / "index.html"
    if index.is_file():
        text = index.read_text(encoding="utf-8", errors="replace")
        ammo_ref = "2176-lib-ammo.js"
        print(f"  {'OK' if ammo_ref in text else 'MISSING'}     Ammo script reference")
        ok = ok and ammo_ref in text

    return ok


def _start_server(port: int) -> subprocess.Popen[str] | None:
    if _port_open(port):
        print(f"Server check: 127.0.0.1:{port} is already open; using existing server.")
        return None

    proc = subprocess.Popen(
        [sys.executable, "-u", str(ROOT / "start_server.py"), "--port", str(port)],
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    for _ in range(80):
        if _port_open(port):
            print(f"Server check: started local server on 127.0.0.1:{port}.")
            return proc
        if proc.poll() is not None:
            output = proc.stdout.read() if proc.stdout else ""
            raise RuntimeError(f"Server exited early:\n{output}")
        import time

        time.sleep(0.25)
    raise RuntimeError(f"Timed out waiting for 127.0.0.1:{port}")


def _check_offline_api(port: int) -> None:
    url = f"http://127.0.0.1:{port}/offline-polytrack-api/leaderboard"
    with urllib.request.urlopen(url, timeout=5) as response:
        body = response.read().decode("utf-8", errors="replace")
    print(f"Offline API check: {response.status} {body[:120]}")


async def _browser_check(port: int, headed: bool) -> dict[str, Any]:
    url = f"http://127.0.0.1:{port}/"
    failed: list[str] = []
    bad_statuses: list[str] = []
    console: list[str] = []
    page_errors: list[str] = []

    async with async_playwright() as play:
        browser = await play.chromium.launch(headless=not headed)
        page = await browser.new_page(viewport={"width": 1280, "height": 720})
        page.on("requestfailed", lambda req: failed.append(f"{req.url} :: {req.failure}"))
        page.on(
            "response",
            lambda res: bad_statuses.append(f"{res.status} {res.url}")
            if res.status >= 400
            else None,
        )
        page.on(
            "console",
            lambda msg: console.append(f"{msg.type}: {msg.text}")
            if msg.type in {"error", "warning"}
            else None,
        )
        page.on("pageerror", lambda err: page_errors.append(str(err)))

        response = await page.goto(url, wait_until="domcontentloaded", timeout=30_000)
        await page.wait_for_timeout(6_000)
        text = ""
        try:
            text = (await page.locator("body").inner_text(timeout=2_000))[:500]
        except Exception as exc:
            text = f"<could not read body text: {exc}>"
        title = await page.title()
        await browser.close()

    return {
        "status": response.status if response else None,
        "title": title,
        "body": text,
        "failed": failed[:20],
        "bad_statuses": bad_statuses[:20],
        "console": console[:20],
        "page_errors": page_errors[:20],
    }


async def _main() -> int:
    parser = argparse.ArgumentParser(description="Diagnose local Polytrack loading.")
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--headed", action="store_true", help="Show the diagnostic browser window.")
    args = parser.parse_args()

    files_ok = _check_files()
    proc = _start_server(args.port)
    try:
        _check_offline_api(args.port)
        result = await _browser_check(args.port, headed=args.headed)
        print("\nBrowser check:")
        print(f"  status: {result['status']}")
        print(f"  title: {result['title']!r}")
        print(f"  body: {result['body']!r}")
        for label in ("bad_statuses", "failed", "page_errors", "console"):
            items = result[label]
            print(f"  {label}: {len(items)}")
            for item in items:
                print(f"    {item}")

        if files_ok and not result["bad_statuses"] and not result["failed"] and not result["page_errors"]:
            print("\nPASS: local files and browser load look healthy.")
            return 0
        print("\nFAIL: see the items above, then repull/restart/hard-refresh on the PC.")
        return 1
    finally:
        if proc is not None:
            proc.terminate()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
