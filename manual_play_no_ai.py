#!/usr/bin/env python3
"""Temporary one-server manual Polytrack launcher.

This is intentionally separate from training: it starts one local HTTP server,
opens the local game in your normal browser, and does not run any AI.
"""

from __future__ import annotations

import argparse
import os
import webbrowser
from http.server import ThreadingHTTPServer
from pathlib import Path

from start_server import PolytrackRequestHandler


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one local Polytrack server for manual play.")
    parser.add_argument("--port", type=int, default=8080, help="TCP port (default: 8080)")
    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help="Directory to serve (default: <repo>/polytrackcopy)",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Start the server without opening a browser window.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent
    root = Path(args.root).resolve() if args.root else repo_root / "polytrackcopy"
    if not root.is_dir():
        raise SystemExit(f"Game folder not found: {root}")

    os.chdir(root)
    try:
        server = ThreadingHTTPServer(("127.0.0.1", args.port), PolytrackRequestHandler)
    except OSError as exc:
        raise SystemExit(
            f"Could not bind 127.0.0.1:{args.port}. "
            "Another server is probably already using that port; try --port 8090."
        ) from exc
    url = f"http://127.0.0.1:{args.port}/"

    print("\nManual Polytrack test server (no AI)")
    print(f"Serving {root}")
    print(f"Open: {url}")
    print("Press Ctrl+C in this terminal to stop the server.\n")

    if not args.no_browser:
        webbrowser.open(url)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down manual test server.")
    finally:
        server.shutdown()


if __name__ == "__main__":
    main()
