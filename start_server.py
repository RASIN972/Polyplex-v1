#!/usr/bin/env python3
"""Serve ``polytrackcopy`` over HTTP (default http://127.0.0.1:8080)."""

from __future__ import annotations

import argparse
import json
import os
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse


class PolytrackRequestHandler(SimpleHTTPRequestHandler):
    """Avoid 404 spam: browsers always request /favicon.ico even with <link rel=icon>."""

    def _send_polytrack_api_json(
        self,
        payload: object,
        *,
        status: HTTPStatus = HTTPStatus.OK,
    ) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _handle_polytrack_api_get(self, path: str) -> bool:
        if not path.startswith("/offline-polytrack-api/"):
            return False

        endpoint = path.removeprefix("/offline-polytrack-api/").strip("/")
        if endpoint == "leaderboard":
            self._send_polytrack_api_json(
                {"total": 0, "entries": [], "userEntry": None}
            )
        elif endpoint == "unverifiedRecordings":
            self._send_polytrack_api_json(
                {"unverifiedRecordings": [], "exhaustive": True}
            )
        elif endpoint == "isVerifier":
            self._send_polytrack_api_json(False)
        else:
            self._send_polytrack_api_json(
                {"error": f"offline endpoint unavailable: {endpoint}"},
                status=HTTPStatus.NOT_FOUND,
            )
        return True

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/favicon.ico":
            self.send_response(HTTPStatus.NO_CONTENT)
            self.end_headers()
            return
        if self._handle_polytrack_api_get(path):
            return
        super().do_GET()

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if not path.startswith("/offline-polytrack-api/"):
            self.send_error(HTTPStatus.METHOD_NOT_ALLOWED)
            return

        self._send_polytrack_api_json(None)


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
