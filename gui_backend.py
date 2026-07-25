#!/usr/bin/env python3
"""Polyplex training control API + React dashboard server.

Launch with::

    python start_gui.py
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
import webbrowser
import zipfile
from pathlib import Path
from typing import Any, TextIO

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

_ROOT = Path(__file__).resolve().parent
_DEFAULT_JSON = _ROOT / "logs" / "training_live.json"
_DEFAULT_RUNS = _ROOT / "logs" / "best_runs.json"
_WATCH_LOG = _ROOT / "logs" / "watch_run.log"
_DASHBOARD_DIST = _ROOT / "dashboard" / "dist"
_WATCH_PORT = 8099
_GUI_PORT = 8088


def _fmt_hms(seconds: float) -> str:
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{sec:02d}"


def _fmt_cp_times(times: list) -> str:
    if not times:
        return "—"
    return " → ".join(f"{float(t):.1f}s" for t in times)


def _port_open(port: int) -> bool:
    try:
        socket.create_connection(("127.0.0.1", port), timeout=0.4).close()
        return True
    except OSError:
        return False


class TrainConfig(BaseModel):
    num_envs: int = Field(default=4, ge=1, le=8)
    timesteps: int = Field(default=1_000_000, ge=1000)
    headless: bool = True
    watch_live: bool = False
    dummy_vec: bool = False


class WatchRequest(BaseModel):
    run_id: int


class TrainingController:
    def __init__(
        self,
        json_path: Path,
        runs_path: Path,
        watch_port: int,
    ) -> None:
        self.json_path = json_path
        self.runs_path = runs_path
        self.watch_port = watch_port
        self.train_proc: subprocess.Popen | None = None
        self.train_via_terminal = False
        self.watch_proc: subprocess.Popen | None = None
        self.server_proc: subprocess.Popen | None = None
        self.watch_log_handle: TextIO | None = None
        self.train_status = "Idle — configure options, then Start."
        self.watch_status = ""
        self.live_label = "IDLE"
        self.live_on = False
        self.last_config = TrainConfig()

    def build_train_cmd(self, cfg: TrainConfig) -> list[str]:
        n = max(1, min(8, int(cfg.num_envs)))
        total = max(1000, int(cfg.timesteps))
        cmd = [
            sys.executable,
            "-u",
            str(_ROOT / "run_local_training.py"),
            "--num-envs",
            str(n),
            "--total-timesteps",
            str(total),
        ]
        if not cfg.headless:
            cmd.append("--headed")
        elif cfg.watch_live:
            cmd.append("--watch")
        if cfg.dummy_vec:
            cmd.extend(["--vec-env", "dummy"])
        return cmd

    def start_training(self, cfg: TrainConfig) -> dict[str, Any]:
        self._check_train_proc()
        if self.train_proc is not None and self.train_proc.poll() is None:
            self.train_status = "Training already running."
            return {"ok": False, "message": self.train_status}

        self.last_config = cfg
        cmd = self.build_train_cmd(cfg)
        n = max(1, min(8, int(cfg.num_envs)))
        env = {
            **os.environ,
            "POLYTRACK_FROM_GUI": "1",
            "PYTHONUTF8": "1",
            "PYTHONIOENCODING": "utf-8",
        }
        try:
            if sys.platform == "win32":
                self.train_proc = subprocess.Popen(
                    cmd,
                    cwd=str(_ROOT),
                    env=env,
                    creationflags=(
                        getattr(subprocess, "CREATE_NEW_CONSOLE", 0)
                        | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
                    ),
                )
                self.train_via_terminal = False
            elif sys.platform == "darwin":
                import shlex

                joined = " ".join(shlex.quote(c) for c in cmd)
                script = f"cd {shlex.quote(str(_ROOT))} && {joined}"
                self.train_proc = subprocess.Popen(
                    [
                        "osascript",
                        "-e",
                        f'tell application "Terminal" to do script {json.dumps(script)}',
                    ]
                )
                self.train_via_terminal = True
            else:
                from shutil import which

                term_cmd: list[str] | None = None
                if which("gnome-terminal"):
                    term_cmd = ["gnome-terminal", "--", *cmd]
                elif which("xterm"):
                    term_cmd = ["xterm", "-e", *cmd]
                if term_cmd is not None:
                    self.train_proc = subprocess.Popen(
                        term_cmd, cwd=str(_ROOT), env=env, start_new_session=True
                    )
                else:
                    self.train_proc = subprocess.Popen(
                        cmd, cwd=str(_ROOT), env=env, start_new_session=True
                    )
                self.train_via_terminal = False

            mode = (
                "headed"
                if not cfg.headless
                else ("watch-0" if cfg.watch_live else "headless")
            )
            pid = self.train_proc.pid if self.train_proc else "?"
            self.train_status = (
                f"Training started in a terminal · {n} envs · {mode} · pid {pid}"
            )
            self.live_on = True
            self.live_label = "TRAINING"
            return {"ok": True, "message": self.train_status}
        except Exception as exc:
            self.train_status = f"Failed to start: {exc}"
            return {"ok": False, "message": self.train_status}

    def stop_training(self) -> dict[str, Any]:
        proc = self.train_proc
        self.train_status = "Stopping training…"
        try:
            if sys.platform == "win32":
                if proc is not None and proc.poll() is None:
                    subprocess.run(
                        ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                        capture_output=True,
                        check=False,
                    )
            else:
                subprocess.run(
                    ["pkill", "-f", str(_ROOT / "run_local_training.py")],
                    capture_output=True,
                    check=False,
                )
                if proc is not None and proc.poll() is None:
                    try:
                        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                    except (ProcessLookupError, PermissionError, OSError):
                        try:
                            proc.terminate()
                        except Exception:
                            pass
            if proc is not None:
                try:
                    proc.wait(timeout=5)
                except (subprocess.TimeoutExpired, Exception):
                    pass
        except Exception as exc:
            self.train_status = f"Stop error: {exc}"
            return {"ok": False, "message": self.train_status}

        self.train_proc = None
        self.train_via_terminal = False
        self.live_on = False
        self.live_label = "STOPPED"
        self.train_status = "Training stopped."
        return {"ok": True, "message": self.train_status}

    def _check_train_proc(self) -> None:
        if self.train_proc is None:
            return
        if self.train_via_terminal:
            return
        code = self.train_proc.poll()
        if code is None:
            return
        if code == 0:
            self.train_status = "Training finished."
            self.live_on = False
            self.live_label = "DONE"
        else:
            self.train_status = (
                f"Training exited (code {code}). Check the terminal window."
            )
            self.live_on = False
            self.live_label = "ERROR"
        self.train_proc = None

    def resolve_model(self, model_rel: str) -> Path | None:
        candidates = [
            (_ROOT / model_rel).resolve(),
            (_ROOT / "checkpoints" / "elites" / Path(model_rel).name).resolve(),
            (_ROOT / "checkpoints" / Path(model_rel).name).resolve(),
        ]
        for c in list(candidates):
            if not str(c).endswith(".zip"):
                candidates.append(Path(str(c) + ".zip"))
        for c in candidates:
            if c.exists():
                return c
        return None

    def ensure_watch_server(self) -> bool:
        if _port_open(self.watch_port):
            return True
        if self.server_proc is None or self.server_proc.poll() is not None:
            self.watch_status = f"Starting game server on :{self.watch_port}…"
            try:
                kw: dict[str, Any] = {
                    "cwd": str(_ROOT),
                    "stdout": subprocess.DEVNULL,
                    "stderr": subprocess.DEVNULL,
                }
                if sys.platform == "win32":
                    kw["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
                self.server_proc = subprocess.Popen(
                    [
                        sys.executable,
                        str(_ROOT / "start_server.py"),
                        "--port",
                        str(self.watch_port),
                    ],
                    **kw,
                )
            except Exception as exc:
                self.watch_status = f"Server failed: {exc}"
                return False
        for _ in range(50):
            if _port_open(self.watch_port):
                return True
            if self.server_proc is not None and self.server_proc.poll() is not None:
                self.watch_status = "start_server.py exited early"
                return False
            time.sleep(0.12)
        self.watch_status = "Timed out waiting for game server"
        return False

    def stop_watch(self) -> dict[str, Any]:
        if self.watch_proc is not None and self.watch_proc.poll() is None:
            self.watch_proc.terminate()
            try:
                self.watch_proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.watch_proc.kill()
        self.watch_proc = None
        if self.watch_log_handle is not None:
            try:
                self.watch_log_handle.close()
            except OSError:
                pass
            self.watch_log_handle = None
        self.watch_status = "Watch stopped."
        return {"ok": True, "message": self.watch_status}

    def preflight_model(self, model_path: Path) -> str | None:
        if not model_path.exists():
            return f"Model missing: {model_path}"
        try:
            with zipfile.ZipFile(model_path, "r") as zf:
                names = set(zf.namelist())
        except zipfile.BadZipFile:
            return f"Corrupt zip: {model_path.name}"
        if (
            "data" not in names
            and "policy.pth" not in names
            and not any(n.endswith(".pth") for n in names)
        ):
            return f"Not an SB3 checkpoint: {model_path.name}"
        return None

    def tail_watch_log(self) -> str:
        try:
            text = _WATCH_LOG.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        for ln in reversed(lines):
            if not ln.startswith("$"):
                return ln[:120]
        return lines[-1][:120] if lines else ""

    def find_run(self, run_id: int) -> dict | None:
        try:
            runs_raw = json.loads(self.runs_path.read_text(encoding="utf-8"))
            if not isinstance(runs_raw, list):
                return None
        except (OSError, json.JSONDecodeError):
            return None
        for r in runs_raw:
            if int(r.get("id", -1)) == run_id:
                return r
        return None

    def watch_run(self, run_id: int) -> dict[str, Any]:
        run = self.find_run(run_id)
        if run is None:
            self.watch_status = "Select a run first."
            return {"ok": False, "message": self.watch_status}

        model_rel = str(run.get("model_path") or "")
        model_path = self.resolve_model(model_rel)
        if model_path is None:
            fallback = (_ROOT / "checkpoints" / "best_model.zip").resolve()
            if fallback.exists():
                model_path = fallback
            else:
                self.watch_status = f"Model missing: {model_rel}"
                return {"ok": False, "message": self.watch_status}

        err = self.preflight_model(model_path)
        if err:
            self.watch_status = err
            return {"ok": False, "message": err}

        if self.watch_proc is not None and self.watch_proc.poll() is None:
            self.stop_watch()
        if not self.ensure_watch_server():
            return {"ok": False, "message": self.watch_status}

        track = int(run.get("track_index", 0))
        _WATCH_LOG.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            "-u",
            str(_ROOT / "evaluate.py"),
            "--model",
            str(model_path),
            "--port",
            str(self.watch_port),
            "--track-index",
            str(track),
            "--episodes",
            "0",
            "--auto-server",
        ]
        try:
            log_f: TextIO = open(_WATCH_LOG, "w", encoding="utf-8")
            log_f.write(f"$ {' '.join(cmd)}\n\n")
            log_f.flush()
            self.watch_log_handle = log_f
            kw: dict[str, Any] = {
                "cwd": str(_ROOT),
                "stdout": log_f,
                "stderr": subprocess.STDOUT,
            }
            if sys.platform == "win32":
                kw["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            self.watch_proc = subprocess.Popen(cmd, **kw)
            self.watch_status = (
                f"Watching #{run.get('id')} — Chromium opens after model load…"
            )
            return {"ok": True, "message": self.watch_status}
        except Exception as exc:
            self.watch_status = f"Failed: {exc}"
            return {"ok": False, "message": self.watch_status}

    def _check_watch_proc(self) -> None:
        if self.watch_proc is None:
            return
        if self.watch_proc.poll() is None:
            tip = self.tail_watch_log()
            if tip and not tip.startswith("$"):
                self.watch_status = tip
            return
        code = self.watch_proc.poll()
        if self.watch_log_handle is not None:
            try:
                self.watch_log_handle.close()
            except OSError:
                pass
            self.watch_log_handle = None
        if code == 0:
            self.watch_status = "Watch finished."
        else:
            tip = self.tail_watch_log() or f"exit {code}"
            self.watch_status = f"Watch failed: {tip}"
        self.watch_proc = None

    def get_state(self) -> dict[str, Any]:
        self._check_train_proc()
        self._check_watch_proc()

        training = self.train_proc is not None and (
            self.train_via_terminal or self.train_proc.poll() is None
        )

        data: dict[str, Any] = {}
        try:
            data = json.loads(self.json_path.read_text(encoding="utf-8"))
            if training:
                self.live_on = True
                self.live_label = "LIVE"
        except (OSError, json.JSONDecodeError):
            pass

        ts = int(data.get("timesteps", 0))
        total = max(1, int(data.get("total_timesteps", 1)))
        hist = data.get("history") or {}
        last5 = data.get("last5", [])

        return {
            "live_on": self.live_on,
            "live_label": self.live_label,
            "training": training,
            "train_status": self.train_status,
            "watch_status": self.watch_status,
            "watching": self.watch_proc is not None and self.watch_proc.poll() is None,
            "config": self.last_config.model_dump(),
            "watch_port": self.watch_port,
            "metrics": {
                "best_fitness": float(
                    data.get("best_fitness", data.get("best_fitness_m", 0))
                ),
                "best_reward": float(data.get("best_reward", 0)),
                "mean_fitness": float(data.get("mean_fitness_10ep", 0)),
                "timesteps": ts,
                "total_timesteps": total,
                "progress": ts / total,
                "fps": float(data.get("fps", 0)),
                "uptime_s": float(data.get("uptime_s", 0)),
                "uptime": _fmt_hms(data.get("uptime_s", 0)),
                "episodes": int(data.get("episodes", 0)),
                "finishes": int(data.get("finishes", 0)),
                "crashes": int(data.get("crashes", 0)),
                "off_tracks": int(data.get("off_tracks", 0)),
            },
            "history": {
                "mean_fitness": [float(x) for x in (hist.get("mean_fitness") or [])],
                "mean_reward": [float(x) for x in (hist.get("mean_reward") or [])],
                "timesteps": [int(x) for x in (hist.get("timesteps") or [])],
            },
            "last5": last5,
        }

    def get_runs(self) -> list[dict[str, Any]]:
        try:
            runs_raw = json.loads(self.runs_path.read_text(encoding="utf-8"))
            if not isinstance(runs_raw, list):
                return []
        except (OSError, json.JSONDecodeError):
            return []

        rows: list[dict[str, Any]] = []
        for r in runs_raw:
            kind = r.get("kind", "?")
            tag = "★ ALL" if kind == "all_time" else f"g{r.get('generation', 0)}"
            rid = int(r.get("id", 0))
            dist = float(r.get("distance_m", r.get("fitness", 0)))
            rows.append(
                {
                    "id": rid,
                    "label": f"#{rid:03d}",
                    "tag": tag,
                    "dist": dist,
                    "reward": float(r.get("reward", 0)),
                    "checkpoints": int(r.get("checkpoints", 0)),
                    "outcome": str(r.get("outcome", "?")),
                    "times": _fmt_cp_times(r.get("checkpoint_times") or []),
                    "generation": r.get("generation"),
                    "kind": r.get("kind"),
                    "steps": r.get("steps"),
                    "model_path": r.get("model_path"),
                    "checkpoint_times": r.get("checkpoint_times") or [],
                }
            )
        return rows

    def shutdown(self) -> None:
        if self.train_proc is not None and self.train_proc.poll() is None:
            self.stop_training()
        self.stop_watch()
        if self.server_proc is not None and self.server_proc.poll() is None:
            self.server_proc.terminate()


def create_app(
    json_path: Path = _DEFAULT_JSON,
    runs_path: Path = _DEFAULT_RUNS,
    watch_port: int = _WATCH_PORT,
) -> FastAPI:
    controller = TrainingController(json_path, runs_path, watch_port)
    app = FastAPI(title="Polyplex Training Control", docs_url=None, redoc_url=None)

    @app.on_event("shutdown")
    def _shutdown() -> None:
        controller.shutdown()

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/state")
    def state() -> dict[str, Any]:
        return controller.get_state()

    @app.get("/api/runs")
    def runs() -> list[dict[str, Any]]:
        return controller.get_runs()

    @app.post("/api/training/start")
    def training_start(cfg: TrainConfig) -> dict[str, Any]:
        return controller.start_training(cfg)

    @app.post("/api/training/stop")
    def training_stop() -> dict[str, Any]:
        return controller.stop_training()

    @app.post("/api/watch/start")
    def watch_start(body: WatchRequest) -> dict[str, Any]:
        return controller.watch_run(body.run_id)

    @app.post("/api/watch/stop")
    def watch_stop() -> dict[str, Any]:
        return controller.stop_watch()

    if _DASHBOARD_DIST.is_dir():
        assets = _DASHBOARD_DIST / "assets"
        if assets.is_dir():
            app.mount("/assets", StaticFiles(directory=assets), name="assets")

        @app.get("/{full_path:path}")
        def spa(full_path: str = "") -> FileResponse:
            if full_path.startswith("api/"):
                raise HTTPException(status_code=404)
            candidate = _DASHBOARD_DIST / full_path
            if full_path and candidate.is_file():
                return FileResponse(candidate)
            return FileResponse(_DASHBOARD_DIST / "index.html")
    else:

        @app.get("/")
        def missing_build() -> dict[str, str]:
            return {
                "error": "React dashboard not built",
                "hint": "cd dashboard && npm install && npm run build",
            }

    return app


def main() -> None:
    p = argparse.ArgumentParser(description="Polyplex React training control GUI.")
    p.add_argument("--json-path", type=Path, default=_DEFAULT_JSON)
    p.add_argument("--runs-path", type=Path, default=_DEFAULT_RUNS)
    p.add_argument("--watch-port", type=int, default=_WATCH_PORT)
    p.add_argument("--port", type=int, default=_GUI_PORT)
    p.add_argument("--no-open", action="store_true")
    args = p.parse_args()

    if not _DASHBOARD_DIST.is_dir():
        print(
            "React dashboard not built yet.\n"
            "  cd dashboard\n"
            "  npm install\n"
            "  npm run build\n"
            "Then re-run: python start_gui.py",
            file=sys.stderr,
        )
        raise SystemExit(1)

    app = create_app(args.json_path, args.runs_path, args.watch_port)
    url = f"http://127.0.0.1:{args.port}"
    print(f"Polyplex dashboard → {url}")
    if not args.no_open:
        webbrowser.open(url)

    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="info")


if __name__ == "__main__":
    main()
