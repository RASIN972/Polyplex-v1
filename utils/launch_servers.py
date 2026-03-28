#!/usr/bin/env python3
"""Start multiple Polytrack HTTP servers (one process per port), each serving ``polytrackcopy``.

Used for parallel RL: each ``PolytrackEnv`` connects to ``http://127.0.0.1:<port>/``.

Usage:
  python -m utils.launch_servers
  python -m utils.launch_servers --ports 8080 8081 8082
"""

from __future__ import annotations

import argparse
import socket
import subprocess
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent


def _port_reachable(host: str, port: int, timeout: float = 0.25) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def wait_for_ports(
    host: str,
    ports: list[int],
    *,
    deadline_s: float = 60.0,
) -> bool:
    t0 = time.monotonic()
    pending = set(ports)
    while time.monotonic() - t0 < deadline_s and pending:
        for p in list(pending):
            if _port_reachable(host, p):
                pending.discard(p)
        if pending:
            time.sleep(0.15)
    return len(pending) == 0


def ensure_servers_running(
    ports: list[int],
    *,
    log_dir: Path | None = None,
    wait: bool = True,
) -> list[subprocess.Popen]:
    """Start ``start_server.py`` for each port that is not already listening."""
    host = "127.0.0.1"
    log_dir = log_dir or (_ROOT / "logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    need = [p for p in ports if not _port_reachable(host, p)]
    if not need:
        print(f"All ports already reachable: {ports}", flush=True)
        return []

    procs: list[subprocess.Popen] = []
    for port in need:
        log_path = log_dir / f"polytrack_http_server_{port}.log"
        out = open(log_path, "a", encoding="utf-8", buffering=1)
        args = [
            sys.executable,
            "-u",
            str(_ROOT / "start_server.py"),
            "--port",
            str(port),
        ]
        if sys.platform == "win32":
            proc = subprocess.Popen(
                args,
                cwd=str(_ROOT),
                stdin=subprocess.DEVNULL,
                stdout=out,
                stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,  # type: ignore[attr-defined]
            )
            procs.append(proc)
            print(f"Started server PID {proc.pid} on {host}:{port} (log: {log_path})", flush=True)
        else:
            # Detached background process (not a child of this Python process)
            proc = subprocess.Popen(
                args,
                cwd=str(_ROOT),
                stdin=subprocess.DEVNULL,
                stdout=out,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            procs.append(proc)
            print(f"Started server PID {proc.pid} on {host}:{port} (log: {log_path})", flush=True)

    if wait:
        if not wait_for_ports(host, need, deadline_s=90.0):
            print(
                f"ERROR: Timed out waiting for ports: {need}",
                file=sys.stderr,
                flush=True,
            )
            raise RuntimeError("Servers did not bind in time")
        print(f"Listening: {need}", flush=True)
    return procs


def main() -> int:
    parser = argparse.ArgumentParser(description="Start Polytrack HTTP servers on multiple ports.")
    parser.add_argument(
        "--ports",
        type=int,
        nargs="*",
        default=list(range(8080, 8088)),
        help="Ports to bind (default: 8080–8087)",
    )
    parser.add_argument(
        "--no-wait",
        action="store_true",
        help="Do not wait for ports to open",
    )
    args = parser.parse_args()
    ports = sorted(set(args.ports))
    ensure_servers_running(ports, wait=not args.no_wait)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
