#!/usr/bin/env python3
"""Serve ``polytrackcopy`` over HTTP (default http://127.0.0.1:8080)."""

from __future__ import annotations

import argparse
import os
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse


class PolytrackRequestHandler(SimpleHTTPRequestHandler):
    """Avoid 404 spam: browsers always request /favicon.ico even with <link rel=icon>."""

    def do_GET(self) -> None:
        if urlparse(self.path).path == "/favicon.ico":
            self.send_response(HTTPStatus.NO_CONTENT)
            self.end_headers()
            return
        super().do_GET()


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve Polytrack static files.")
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="TCP port (default: 8080)",
    )
    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help="Directory to serve (default: <repo>/polytrackcopy)",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve() if args.root else Path(__file__).resolve().parent / "polytrackcopy"
    if not root.is_dir():
        raise SystemExit(f"Game folder not found: {root}")

    os.chdir(root)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), PolytrackRequestHandler)
    print(
        "\n"
        "============================================================\n"
        "  POLYTRACK HTTP SERVER — keep this terminal for the server only.\n"
        "  Run training elsewhere:\n"
        "    • Other terminal:  python training/train_ppo.py\n"
        "    • Or one command:   python run_local_training.py\n"
        "  Ctrl+C here stops ONLY the server (not training in another window).\n"
        "============================================================\n"
    )
    print(f"Serving {root} at http://127.0.0.1:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.shutdown()


if __name__ == "__main__":
    main()
