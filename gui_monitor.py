#!/usr/bin/env python3
"""Dashboard GUI for Polytrack RL training.

Reads:
  - logs/training_live.json  — live metrics
  - logs/best_runs.json      — elite / per-generation best runs

Double-click a best run (or select + Watch) to replay via ``evaluate.py``
with an auto-started game server on port 8099.

Usage:
    python gui_monitor.py
"""

from __future__ import annotations

import argparse
import json
import socket
import subprocess
import sys
import time
import tkinter as tk
from pathlib import Path
from tkinter import ttk
from typing import IO, TextIO

_ROOT = Path(__file__).resolve().parent
_DEFAULT_JSON = _ROOT / "logs" / "training_live.json"
_DEFAULT_RUNS = _ROOT / "logs" / "best_runs.json"
_WATCH_LOG = _ROOT / "logs" / "watch_run.log"
_WATCH_PORT = 8099

# Dashboard palette (dark glass / orbital-style)
_BG = "#0d0d0d"
_CARD = "#1a1a1a"
_ELEVATED = "#242424"
_BORDER = "#2a2a2a"
_TEXT = "#f5f5f5"
_MUTED = "#a0a0a0"
_DIM = "#555555"
_ORANGE = "#ff6b00"
_GREEN = "#22c55e"
_RED = "#ef4444"
_AMBER = "#f59e0b"
_TEAL = "#2dd4bf"

_FONT = ("Segoe UI", 11) if sys.platform == "win32" else ("SF Pro Text", 12)
_FONT_SM = ("Segoe UI", 9) if sys.platform == "win32" else ("SF Pro Text", 10)
_FONT_MONO = ("Consolas", 10) if sys.platform == "win32" else ("Menlo", 11)
_FONT_H = ("Segoe UI", 13, "bold") if sys.platform == "win32" else ("SF Pro Text", 14, "bold")
_FONT_XL = ("Segoe UI", 28, "bold") if sys.platform == "win32" else ("SF Pro Text", 30, "bold")
_FONT_BRAND = ("Segoe UI", 16, "bold") if sys.platform == "win32" else ("SF Pro Text", 18, "bold")

_OUTCOME_COLORS = {
    "finished": _GREEN,
    "crashed": _RED,
    "timeout": _AMBER,
    "off_track": _ORANGE,
}


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


class MetricCard(tk.Frame):
    """Rounded-looking metric tile with label + big value + subtitle."""

    def __init__(
        self,
        parent: tk.Widget,
        title: str,
        *,
        accent: str = _TEAL,
        width: int = 160,
        height: int = 100,
    ) -> None:
        super().__init__(parent, bg=_CARD, highlightbackground=_BORDER, highlightthickness=1)
        self.configure(width=width, height=height)
        self.pack_propagate(False)
        tk.Label(
            self, text=title.upper(), font=_FONT_SM, fg=_MUTED, bg=_CARD, anchor="w"
        ).pack(fill="x", padx=12, pady=(10, 0))
        self.value = tk.StringVar(value="—")
        tk.Label(
            self,
            textvariable=self.value,
            font=_FONT_XL,
            fg=accent,
            bg=_CARD,
            anchor="w",
        ).pack(fill="x", padx=12, pady=(2, 0))
        self.sub = tk.StringVar(value="")
        tk.Label(
            self, textvariable=self.sub, font=_FONT_SM, fg=_DIM, bg=_CARD, anchor="w"
        ).pack(fill="x", padx=12, pady=(0, 10))


class TrainingMonitorGUI:
    def __init__(
        self,
        root: tk.Tk,
        json_path: Path,
        runs_path: Path,
        poll_ms: int,
        watch_port: int,
    ) -> None:
        self._root = root
        self._json_path = json_path
        self._runs_path = runs_path
        self._poll_ms = poll_ms
        self._watch_port = watch_port
        self._runs: list[dict] = []
        self._watch_proc: subprocess.Popen | None = None
        self._server_proc: subprocess.Popen | None = None
        self._watch_log_handle: IO[str] | None = None

        root.title("Polyplex Control — Training Dashboard")
        root.resizable(True, True)
        root.configure(bg=_BG)
        root.minsize(1040, 640)

        self._build_ui()
        root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._poll()

    def _build_ui(self) -> None:
        root = self._root

        # —— Top nav ——
        nav = tk.Frame(root, bg=_ELEVATED, height=52)
        nav.pack(fill="x")
        nav.pack_propagate(False)
        tk.Label(
            nav, text="POLYPLEX", font=_FONT_BRAND, fg=_ORANGE, bg=_ELEVATED
        ).pack(side="left", padx=20, pady=12)
        tk.Label(
            nav, text="Training Control", font=_FONT, fg=_MUTED, bg=_ELEVATED
        ).pack(side="left", padx=(0, 20))

        self._live_pill = tk.Label(
            nav,
            text="  WAITING  ",
            font=_FONT_SM,
            fg=_BG,
            bg=_DIM,
            padx=8,
            pady=2,
        )
        self._live_pill.pack(side="right", padx=16)

        body = tk.Frame(root, bg=_BG)
        body.pack(fill="both", expand=True, padx=16, pady=16)

        # —— Metric row ——
        metrics = tk.Frame(body, bg=_BG)
        metrics.pack(fill="x", pady=(0, 16))

        self._card_fitness = MetricCard(metrics, "Best distance", accent=_GREEN)
        self._card_fitness.pack(side="left", padx=(0, 10))
        self._card_reward = MetricCard(metrics, "Best reward", accent=_ORANGE)
        self._card_reward.pack(side="left", padx=(0, 10))
        self._card_mean = MetricCard(metrics, "Mean distance", accent=_TEAL)
        self._card_mean.pack(side="left", padx=(0, 10))
        self._card_steps = MetricCard(metrics, "Timesteps", accent=_TEXT)
        self._card_steps.pack(side="left", padx=(0, 10))
        self._card_fps = MetricCard(metrics, "Rollout FPS", accent=_AMBER)
        self._card_fps.pack(side="left")

        # —— Progress strip ——
        prog = tk.Frame(body, bg=_CARD, highlightbackground=_BORDER, highlightthickness=1)
        prog.pack(fill="x", pady=(0, 16))
        row = tk.Frame(prog, bg=_CARD)
        row.pack(fill="x", padx=16, pady=12)
        self._progress_lbl = tk.StringVar(value="Progress — waiting for training…")
        tk.Label(
            row, textvariable=self._progress_lbl, font=_FONT, fg=_TEXT, bg=_CARD
        ).pack(side="left")
        self._uptime_lbl = tk.StringVar(value="")
        tk.Label(
            row, textvariable=self._uptime_lbl, font=_FONT_SM, fg=_MUTED, bg=_CARD
        ).pack(side="right")

        self._prog_canvas = tk.Canvas(
            prog, height=8, bg=_ELEVATED, highlightthickness=0, bd=0
        )
        self._prog_canvas.pack(fill="x", padx=16, pady=(0, 12))
        self._prog_fill = self._prog_canvas.create_rectangle(
            0, 0, 0, 8, fill=_ORANGE, outline=""
        )
        self._prog_canvas.bind("<Configure>", self._redraw_progress)
        self._prog_frac = 0.0

        # —— Main split: outcomes + recent | best runs ——
        split = tk.Frame(body, bg=_BG)
        split.pack(fill="both", expand=True)

        left = tk.Frame(split, bg=_BG)
        left.pack(side="left", fill="both", expand=False, padx=(0, 12))
        left.configure(width=340)
        left.pack_propagate(False)

        outcomes = tk.Frame(
            left, bg=_CARD, highlightbackground=_BORDER, highlightthickness=1
        )
        outcomes.pack(fill="x", pady=(0, 12))
        tk.Label(
            outcomes, text="OUTCOMES", font=_FONT_H, fg=_TEXT, bg=_CARD, anchor="w"
        ).pack(fill="x", padx=14, pady=(12, 8))

        grid = tk.Frame(outcomes, bg=_CARD)
        grid.pack(fill="x", padx=14, pady=(0, 14))
        self._fin_var = tk.StringVar(value="0")
        self._crash_var = tk.StringVar(value="0")
        self._off_var = tk.StringVar(value="0")
        for i, (title, var, color) in enumerate(
            (
                ("Finishes", self._fin_var, _GREEN),
                ("Crashes", self._crash_var, _RED),
                ("Off-track", self._off_var, _ORANGE),
            )
        ):
            cell = tk.Frame(grid, bg=_ELEVATED)
            cell.grid(row=0, column=i, padx=(0 if i == 0 else 6), sticky="nsew")
            grid.columnconfigure(i, weight=1)
            tk.Label(cell, text=title, font=_FONT_SM, fg=_MUTED, bg=_ELEVATED).pack(
                padx=8, pady=(8, 0)
            )
            tk.Label(
                cell, textvariable=var, font=_FONT_H, fg=color, bg=_ELEVATED
            ).pack(padx=8, pady=(0, 8))

        recent = tk.Frame(
            left, bg=_CARD, highlightbackground=_BORDER, highlightthickness=1
        )
        recent.pack(fill="both", expand=True)
        tk.Label(
            recent, text="LAST 5 EPISODES", font=_FONT_H, fg=_TEXT, bg=_CARD, anchor="w"
        ).pack(fill="x", padx=14, pady=(12, 6))
        self._ep_labels: list[tk.Label] = []
        for _ in range(5):
            lbl = tk.Label(
                recent, text="—", font=_FONT_MONO, fg=_DIM, bg=_CARD, anchor="w"
            )
            lbl.pack(fill="x", padx=14, pady=2)
            self._ep_labels.append(lbl)
        tk.Frame(recent, bg=_CARD, height=8).pack()

        # —— Best runs panel ——
        right = tk.Frame(
            split, bg=_CARD, highlightbackground=_BORDER, highlightthickness=1
        )
        right.pack(side="left", fill="both", expand=True)

        hdr = tk.Frame(right, bg=_CARD)
        hdr.pack(fill="x", padx=14, pady=(12, 4))
        tk.Label(
            hdr, text="BEST RUNS", font=_FONT_H, fg=_TEXT, bg=_CARD
        ).pack(side="left")
        tk.Label(
            hdr,
            text="Double-click to watch · uses port "
            f"{self._watch_port}",
            font=_FONT_SM,
            fg=_DIM,
            bg=_CARD,
        ).pack(side="right")

        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure(
            "Runs.Treeview",
            background=_ELEVATED,
            foreground=_TEXT,
            fieldbackground=_ELEVATED,
            borderwidth=0,
            rowheight=26,
            font=_FONT_MONO,
        )
        style.configure(
            "Runs.Treeview.Heading",
            background=_CARD,
            foreground=_MUTED,
            font=_FONT_SM,
            relief="flat",
        )
        style.map(
            "Runs.Treeview",
            background=[("selected", _ORANGE)],
            foreground=[("selected", _BG)],
        )

        cols = ("id", "tag", "dist", "reward", "cps", "outcome", "times")
        tree_frame = tk.Frame(right, bg=_CARD)
        tree_frame.pack(fill="both", expand=True, padx=14, pady=4)
        scroll = ttk.Scrollbar(tree_frame)
        scroll.pack(side="right", fill="y")
        self._tree = ttk.Treeview(
            tree_frame,
            columns=cols,
            show="headings",
            style="Runs.Treeview",
            yscrollcommand=scroll.set,
            selectmode="browse",
        )
        scroll.config(command=self._tree.yview)
        headings = {
            "id": ("#", 48),
            "tag": ("Tag", 64),
            "dist": ("Dist m", 72),
            "reward": ("Reward", 72),
            "cps": ("CPs", 48),
            "outcome": ("Outcome", 88),
            "times": ("CP times", 180),
        }
        for key, (label, w) in headings.items():
            self._tree.heading(key, text=label)
            self._tree.column(key, width=w, anchor="w", stretch=(key == "times"))
        self._tree.pack(side="left", fill="both", expand=True)
        self._tree.bind("<Double-1>", lambda _e: self._watch_selected())
        self._tree.bind("<<TreeviewSelect>>", lambda _e: self._show_run_detail())

        self._detail_var = tk.StringVar(
            value="Select a run to see details. Double-click to open a headed replay."
        )
        tk.Label(
            right,
            textvariable=self._detail_var,
            font=_FONT_SM,
            fg=_MUTED,
            bg=_CARD,
            justify="left",
            wraplength=560,
            anchor="w",
        ).pack(fill="x", padx=14, pady=6)

        btn_row = tk.Frame(right, bg=_CARD)
        btn_row.pack(fill="x", padx=14, pady=(4, 14))
        watch_btn = tk.Button(
            btn_row,
            text="Watch selected run",
            command=self._watch_selected,
            bg=_ORANGE,
            fg=_BG,
            activebackground="#ff8533",
            activeforeground=_BG,
            font=_FONT_H,
            relief="flat",
            padx=20,
            pady=10,
            cursor="hand2",
            borderwidth=0,
        )
        watch_btn.pack(side="left")
        stop_btn = tk.Button(
            btn_row,
            text="Stop",
            command=self._stop_watch_btn,
            bg=_ELEVATED,
            fg=_TEXT,
            activebackground=_BORDER,
            activeforeground=_TEXT,
            font=_FONT,
            relief="flat",
            padx=16,
            pady=10,
            cursor="hand2",
            borderwidth=0,
            highlightbackground=_BORDER,
            highlightthickness=1,
        )
        stop_btn.pack(side="left", padx=(10, 0))
        self._watch_status = tk.StringVar(value="")
        tk.Label(
            btn_row,
            textvariable=self._watch_status,
            font=_FONT_SM,
            fg=_MUTED,
            bg=_CARD,
            anchor="w",
        ).pack(side="left", padx=14, fill="x", expand=True)

    def _redraw_progress(self, _event: object | None = None) -> None:
        w = max(1, self._prog_canvas.winfo_width())
        self._prog_canvas.coords(self._prog_fill, 0, 0, int(w * self._prog_frac), 8)

    def _poll(self) -> None:
        self._refresh_metrics()
        self._refresh_runs()
        self._check_watch_proc()
        self._root.after(self._poll_ms, self._poll)

    def _refresh_metrics(self) -> None:
        try:
            data: dict = json.loads(self._json_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            self._live_pill.config(text="  WAITING  ", bg=_DIM, fg=_BG)
            return

        self._live_pill.config(text="  LIVE  ", bg=_GREEN, fg=_BG)
        ts = int(data.get("timesteps", 0))
        total = max(1, int(data.get("total_timesteps", 1)))
        frac = ts / total
        self._prog_frac = frac
        self._redraw_progress()
        self._progress_lbl.set(f"Progress  {ts:,} / {total:,}  ({100 * frac:.1f}%)")
        self._uptime_lbl.set(
            f"Uptime {_fmt_hms(data.get('uptime_s', 0))}  ·  "
            f"episodes {int(data.get('episodes', 0))}"
        )

        best_fit = float(data.get("best_fitness", data.get("best_fitness_m", 0)))
        mean_fit = float(data.get("mean_fitness_10ep", 0))
        best_rew = float(data.get("best_reward", 0))
        self._card_fitness.value.set(f"{best_fit:.1f}")
        self._card_fitness.sub.set("metres (all-time)")
        self._card_reward.value.set(f"{best_rew:+.2f}")
        self._card_reward.sub.set("episode reward (all-time)")
        self._card_mean.value.set(f"{mean_fit:.1f}")
        self._card_mean.sub.set("metres · last 10 ep")
        self._card_steps.value.set(f"{ts // 1000}k" if ts >= 1000 else str(ts))
        self._card_steps.sub.set(f"of {total:,}")
        self._card_fps.value.set(f"{float(data.get('fps', 0)):.0f}")
        self._card_fps.sub.set("env steps / s")

        self._fin_var.set(str(int(data.get("finishes", 0))))
        self._crash_var.set(str(int(data.get("crashes", 0))))
        self._off_var.set(str(int(data.get("off_tracks", 0))))

        last5 = data.get("last5", [])
        for i, lbl in enumerate(self._ep_labels):
            if i < len(last5):
                e = last5[i]
                outcome = str(e.get("outcome", "?"))
                color = _OUTCOME_COLORS.get(outcome, _MUTED)
                fit = float(e.get("fitness", 0))
                lbl.config(
                    text=(
                        f"ep {int(e['ep']):3d}  dist {fit:5.1f}m  "
                        f"r {float(e['reward']):+.1f}  "
                        f"cp {int(e['checkpoints'])}  {outcome}"
                    ),
                    fg=color,
                )
            else:
                lbl.config(text="—", fg=_DIM)

    def _refresh_runs(self) -> None:
        try:
            raw = self._runs_path.read_text(encoding="utf-8")
            runs: list = json.loads(raw)
            if not isinstance(runs, list):
                return
        except (OSError, json.JSONDecodeError):
            return

        if runs == self._runs:
            return
        sel = self._tree.selection()
        sel_id = sel[0] if sel else None
        self._runs = runs
        self._tree.delete(*self._tree.get_children())
        for r in runs:
            kind = r.get("kind", "?")
            tag = "★ ALL" if kind == "all_time" else f"g{r.get('generation', 0)}"
            iid = str(r.get("id", len(self._tree.get_children())))
            dist = float(r.get("distance_m", r.get("fitness", 0)))
            self._tree.insert(
                "",
                "end",
                iid=iid,
                values=(
                    f"#{r.get('id', 0):03d}",
                    tag,
                    f"{dist:.0f}",
                    f"{float(r.get('reward', 0)):+.2f}",
                    int(r.get("checkpoints", 0)),
                    str(r.get("outcome", "?")),
                    _fmt_cp_times(r.get("checkpoint_times") or []),
                ),
            )
        if sel_id and self._tree.exists(sel_id):
            self._tree.selection_set(sel_id)

    def _selected_run(self) -> dict | None:
        sel = self._tree.selection()
        if not sel:
            return None
        try:
            rid = int(sel[0])
        except ValueError:
            return None
        for r in self._runs:
            if int(r.get("id", -1)) == rid:
                return r
        return None

    def _show_run_detail(self) -> None:
        run = self._selected_run()
        if run is None:
            return
        times = _fmt_cp_times(run.get("checkpoint_times") or [])
        dist = float(run.get("distance_m", run.get("fitness", 0)))
        self._detail_var.set(
            f"#{run.get('id')}  gen={run.get('generation')}  "
            f"kind={run.get('kind')}  outcome={run.get('outcome')}\n"
            f"distance={dist:.1f} m  reward={float(run.get('reward', 0)):+.2f}  "
            f"steps={run.get('steps')}\n"
            f"checkpoint times: {times}\n"
            f"model: {run.get('model_path')}"
        )

    def _resolve_model(self, model_rel: str) -> Path | None:
        candidates = [
            (_ROOT / model_rel).resolve(),
            (_ROOT / "checkpoints" / "elites" / Path(model_rel).name).resolve(),
            (_ROOT / "checkpoints" / Path(model_rel).name).resolve(),
        ]
        # elites may be stored as path without .zip
        for c in list(candidates):
            if not str(c).endswith(".zip"):
                candidates.append(Path(str(c) + ".zip"))
        for c in candidates:
            if c.exists():
                return c
        return None

    def _ensure_watch_server(self) -> bool:
        if _port_open(self._watch_port):
            return True

        if self._server_proc is None or self._server_proc.poll() is not None:
            self._watch_status.set(f"Starting game server on :{self._watch_port}…")
            self._root.update_idletasks()
            try:
                self._server_proc = subprocess.Popen(
                    [
                        sys.executable,
                        str(_ROOT / "start_server.py"),
                        "--port",
                        str(self._watch_port),
                    ],
                    cwd=str(_ROOT),
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except Exception as exc:
                self._watch_status.set(f"Server failed: {exc}")
                return False

        for _ in range(40):
            if _port_open(self._watch_port):
                return True
            if self._server_proc is not None and self._server_proc.poll() is not None:
                self._watch_status.set("start_server.py exited early")
                return False
            time.sleep(0.15)
            self._root.update_idletasks()
        self._watch_status.set("Timed out waiting for game server")
        return False

    def _stop_watch(self) -> None:
        if self._watch_proc is not None and self._watch_proc.poll() is None:
            self._watch_proc.terminate()
            try:
                self._watch_proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._watch_proc.kill()
        self._watch_proc = None
        if self._watch_log_handle is not None:
            try:
                self._watch_log_handle.close()
            except OSError:
                pass
            self._watch_log_handle = None

    def _stop_watch_btn(self) -> None:
        self._stop_watch()
        self._watch_status.set("Watch stopped.")

    def _preflight_model(self, model_path: Path) -> str | None:
        """Return an error string if the checkpoint looks unloadable."""
        import zipfile

        if not model_path.exists():
            return f"Model missing: {model_path}"
        try:
            with zipfile.ZipFile(model_path, "r") as zf:
                names = set(zf.namelist())
        except zipfile.BadZipFile:
            return f"Corrupt zip: {model_path.name}"
        # SB3 zip layout
        if "data" not in names and "policy.pth" not in names and not any(
            n.endswith(".pth") for n in names
        ):
            return f"Not an SB3 checkpoint: {model_path.name}"
        return None

    def _watch_selected(self) -> None:
        run = self._selected_run()
        if run is None:
            self._watch_status.set("Select a run first.")
            return

        model_rel = str(run.get("model_path") or "")
        model_path = self._resolve_model(model_rel)
        if model_path is None:
            # Fall back to global best if elite file was cleaned up.
            fallback = (_ROOT / "checkpoints" / "best_model.zip").resolve()
            if fallback.exists():
                model_path = fallback
                self._watch_status.set("Elite zip missing — using checkpoints/best_model.zip")
            else:
                self._watch_status.set(f"Model missing: {model_rel}")
                return

        err = self._preflight_model(model_path)
        if err:
            self._watch_status.set(err)
            return

        # Restart cleanly if a previous watch is still open.
        if self._watch_proc is not None and self._watch_proc.poll() is None:
            self._watch_status.set("Restarting previous watch…")
            self._root.update_idletasks()
            self._stop_watch()

        if not self._ensure_watch_server():
            return

        track = int(run.get("track_index", 0))
        _WATCH_LOG.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            "-u",
            str(_ROOT / "evaluate.py"),
            "--model",
            str(model_path),
            "--port",
            str(self._watch_port),
            "--track-index",
            str(track),
            "--episodes",
            "1",
            "--auto-server",
        ]
        try:
            log_f: TextIO = open(_WATCH_LOG, "w", encoding="utf-8")
            log_f.write(f"$ {' '.join(cmd)}\n\n")
            log_f.flush()
            self._watch_log_handle = log_f
            popen_kw: dict = {
                "cwd": str(_ROOT),
                "stdout": log_f,
                "stderr": subprocess.STDOUT,
            }
            # On Windows, open a visible console so Chromium/Playwright errors aren't silent.
            if sys.platform == "win32":
                popen_kw["creationflags"] = getattr(subprocess, "CREATE_NEW_CONSOLE", 0)
                popen_kw["stdout"] = None
                popen_kw["stderr"] = None
                # Still tee a minimal launch line into the log.
                log_f.write("(Windows: evaluate.py opened in a new console window)\n")
                log_f.flush()
            self._watch_proc = subprocess.Popen(cmd, **popen_kw)
            self._watch_status.set(
                f"Watching #{run.get('id')} → headed Chromium on :{self._watch_port} "
                f"(see logs/watch_run.log if it fails)"
            )
        except Exception as exc:
            self._watch_status.set(f"Failed: {exc}")

    def _check_watch_proc(self) -> None:
        if self._watch_proc is None:
            return
        code = self._watch_proc.poll()
        if code is None:
            return
        if self._watch_log_handle is not None:
            try:
                self._watch_log_handle.close()
            except OSError:
                pass
            self._watch_log_handle = None
        if code == 0:
            self._watch_status.set("Watch finished.")
        else:
            err = ""
            try:
                err = _WATCH_LOG.read_text(encoding="utf-8")[-400:]
            except OSError:
                pass
            brief = err.strip().splitlines()[-1] if err.strip() else f"exit {code}"
            self._watch_status.set(f"Watch failed: {brief}")
        self._watch_proc = None

    def _on_close(self) -> None:
        self._stop_watch()
        if self._server_proc is not None and self._server_proc.poll() is None:
            self._server_proc.terminate()
        self._root.destroy()


def main() -> None:
    p = argparse.ArgumentParser(description="Polyplex training dashboard GUI.")
    p.add_argument("--json-path", type=Path, default=_DEFAULT_JSON)
    p.add_argument("--runs-path", type=Path, default=_DEFAULT_RUNS)
    p.add_argument("--poll-ms", type=int, default=2000)
    p.add_argument(
        "--watch-port",
        type=int,
        default=_WATCH_PORT,
        help="Dedicated port for Watch replay (default: 8099)",
    )
    args = p.parse_args()

    root = tk.Tk()
    TrainingMonitorGUI(
        root, args.json_path, args.runs_path, args.poll_ms, args.watch_port
    )
    root.mainloop()


if __name__ == "__main__":
    main()
