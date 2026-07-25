#!/usr/bin/env python3
"""Polyplex training dashboard — lavender / black orbital-style GUI.

Reads:
  - logs/training_live.json
  - logs/best_runs.json

Watch launches headed ``evaluate.py`` in the background (no blank console).
Status streams from ``logs/watch_run.log`` into the dashboard.
"""

from __future__ import annotations

import argparse
import json
import socket
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from tkinter import Canvas, Frame, Label, StringVar, Tk, ttk
from typing import TextIO

_ROOT = Path(__file__).resolve().parent
_DEFAULT_JSON = _ROOT / "logs" / "training_live.json"
_DEFAULT_RUNS = _ROOT / "logs" / "best_runs.json"
_WATCH_LOG = _ROOT / "logs" / "watch_run.log"
_WATCH_PORT = 8099

# Lavender + black (reference orbital dashboard)
_BG = "#0a0a0c"
_CARD = "#14141a"
_ELEVATED = "#1c1c24"
_BORDER = "#2a2a36"
_LAVENDER = "#c4b5fd"
_LAVENDER_DIM = "#8b7ec8"
_LAVENDER_SOFT = "#ddd6fe"
_TEXT = "#f5f5f7"
_MUTED = "#9b9bb0"
_DIM = "#5c5c70"
_ORANGE = "#ff8a3d"
_GREEN = "#34d399"
_RED = "#f87171"
_AMBER = "#fbbf24"

_UI = "Segoe UI" if sys.platform == "win32" else "Helvetica Neue"
_MONO = "Consolas" if sys.platform == "win32" else "Menlo"

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


def _round_rect(
    canvas: Canvas,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    r: float,
    **kw: object,
) -> int:
    r = min(r, (x2 - x1) / 2, (y2 - y1) / 2)
    points = [
        x1 + r, y1,
        x2 - r, y1,
        x2, y1,
        x2, y1 + r,
        x2, y2 - r,
        x2, y2,
        x2 - r, y2,
        x1 + r, y2,
        x1, y2,
        x1, y2 - r,
        x1, y1 + r,
        x1, y1,
    ]
    return canvas.create_polygon(points, smooth=True, **kw)  # type: ignore[arg-type]


class RoundedCard(Frame):
    """Card with painted rounded corners (Canvas underlay)."""

    def __init__(
        self,
        parent: Frame | Tk,
        *,
        width: int = 170,
        height: int = 108,
        radius: int = 18,
        fill: str = _CARD,
        padding: int = 14,
    ) -> None:
        super().__init__(parent, bg=_BG, width=width, height=height)
        self.pack_propagate(False)
        self._radius = radius
        self._fill = fill
        self._canvas = Canvas(self, bg=_BG, highlightthickness=0, bd=0)
        self._canvas.place(x=0, y=0, relwidth=1, relheight=1)
        self._shape: int | None = None
        self.content = Frame(self, bg=fill)
        self.content.pack(fill="both", expand=True, padx=padding, pady=padding)
        self._canvas.lower(self.content)
        self.bind("<Configure>", self._paint)
        self._canvas.bind("<Configure>", self._paint)

    def _paint(self, _e: object | None = None) -> None:
        w = max(self.winfo_width(), 2)
        h = max(self.winfo_height(), 2)
        self._canvas.delete("all")
        self._shape = _round_rect(
            self._canvas, 1, 1, w - 1, h - 1, self._radius, fill=self._fill, outline=_BORDER
        )


class MetricCard(RoundedCard):
    def __init__(
        self,
        parent: Frame,
        title: str,
        *,
        accent: str = _LAVENDER,
        width: int = 168,
    ) -> None:
        super().__init__(parent, width=width, height=110, radius=20, fill=_CARD)
        Label(
            self.content,
            text=title.upper(),
            font=(_UI, 9),
            fg=_LAVENDER_DIM,
            bg=_CARD,
            anchor="w",
        ).pack(fill="x")
        self.value = StringVar(value="—")
        Label(
            self.content,
            textvariable=self.value,
            font=(_UI, 26, "bold"),
            fg=accent,
            bg=_CARD,
            anchor="w",
        ).pack(fill="x", pady=(4, 0))
        self.sub = StringVar(value="")
        Label(
            self.content,
            textvariable=self.sub,
            font=(_UI, 9),
            fg=_DIM,
            bg=_CARD,
            anchor="w",
        ).pack(fill="x")


class PillButton(Canvas):
    def __init__(
        self,
        parent: Frame,
        text: str,
        command: object,
        *,
        primary: bool = True,
        width: int = 168,
        height: int = 40,
    ) -> None:
        super().__init__(
            parent, width=width, height=height, bg=_CARD, highlightthickness=0, bd=0
        )
        self._command = command
        self._primary = primary
        self._text = text
        self._w = width
        self._h = height
        self._draw(False)
        self.bind("<Button-1>", self._click)
        self.bind("<Enter>", lambda _e: self._draw(True))
        self.bind("<Leave>", lambda _e: self._draw(False))

    def _draw(self, hover: bool) -> None:
        self.delete("all")
        if self._primary:
            fill = "#b8a4ff" if hover else _LAVENDER
            fg = _BG
        else:
            fill = _ELEVATED if not hover else _BORDER
            fg = _LAVENDER_SOFT
        _round_rect(self, 1, 1, self._w - 1, self._h - 1, self._h / 2, fill=fill, outline="")
        self.create_text(
            self._w / 2,
            self._h / 2,
            text=self._text,
            fill=fg,
            font=(_UI, 11, "bold"),
        )

    def _click(self, _e: object) -> None:
        if callable(self._command):
            self._command()


class TrainingMonitorGUI:
    def __init__(
        self,
        root: Tk,
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
        self._watch_log_handle: TextIO | None = None
        self._prog_frac = 0.0

        root.title("Polyplex — Training Control")
        root.configure(bg=_BG)
        root.minsize(1080, 680)
        root.resizable(True, True)

        self._build_ui()
        root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._poll()

    def _build_ui(self) -> None:
        root = self._root

        # Top bar
        nav = Frame(root, bg=_BG, height=64)
        nav.pack(fill="x", padx=20, pady=(16, 0))
        nav.pack_propagate(False)
        Label(
            nav, text="Polyplex", font=(_UI, 20, "bold"), fg=_LAVENDER_SOFT, bg=_BG
        ).pack(side="left", pady=8)
        Label(
            nav, text="  Satellite training control", font=(_UI, 11), fg=_MUTED, bg=_BG
        ).pack(side="left", pady=12)

        # Center pill tabs (visual only)
        pills = Canvas(nav, width=360, height=36, bg=_BG, highlightthickness=0, bd=0)
        pills.pack(side="left", padx=40)
        _round_rect(pills, 0, 2, 350, 34, 16, fill=_ELEVATED, outline="")
        _round_rect(pills, 6, 6, 110, 30, 12, fill=_LAVENDER, outline="")
        pills.create_text(58, 18, text="Overview", fill=_BG, font=(_UI, 10, "bold"))
        pills.create_text(155, 18, text="Runs", fill=_MUTED, font=(_UI, 10))
        pills.create_text(230, 18, text="Telemetry", fill=_MUTED, font=(_UI, 10))
        pills.create_text(310, 18, text="Reports", fill=_MUTED, font=(_UI, 10))

        self._live_canvas = Canvas(nav, width=90, height=28, bg=_BG, highlightthickness=0)
        self._live_canvas.pack(side="right", pady=16)
        self._set_live_pill(False)

        body = Frame(root, bg=_BG)
        body.pack(fill="both", expand=True, padx=20, pady=16)

        # Metric cards
        metrics = Frame(body, bg=_BG)
        metrics.pack(fill="x", pady=(0, 14))
        self._card_fitness = MetricCard(metrics, "Best distance", accent=_GREEN)
        self._card_fitness.pack(side="left", padx=(0, 12))
        self._card_reward = MetricCard(metrics, "Best reward", accent=_ORANGE)
        self._card_reward.pack(side="left", padx=(0, 12))
        self._card_mean = MetricCard(metrics, "Mean distance", accent=_LAVENDER)
        self._card_mean.pack(side="left", padx=(0, 12))
        self._card_steps = MetricCard(metrics, "Timesteps", accent=_LAVENDER_SOFT)
        self._card_steps.pack(side="left", padx=(0, 12))
        self._card_fps = MetricCard(metrics, "Rollout FPS", accent=_AMBER)
        self._card_fps.pack(side="left")

        # Progress card
        prog_wrap = RoundedCard(body, height=72, radius=18, fill=_CARD)
        prog_wrap.pack(fill="x", pady=(0, 14))
        prog_wrap.configure(height=72)
        row = Frame(prog_wrap.content, bg=_CARD)
        row.pack(fill="x")
        self._progress_lbl = StringVar(value="Progress — waiting for training…")
        Label(
            row, textvariable=self._progress_lbl, font=(_UI, 11), fg=_TEXT, bg=_CARD
        ).pack(side="left")
        self._uptime_lbl = StringVar(value="")
        Label(
            row, textvariable=self._uptime_lbl, font=(_UI, 9), fg=_MUTED, bg=_CARD
        ).pack(side="right")
        self._prog_canvas = Canvas(
            prog_wrap.content, height=10, bg=_ELEVATED, highlightthickness=0, bd=0
        )
        self._prog_canvas.pack(fill="x", pady=(10, 0))
        self._prog_fill = self._prog_canvas.create_rectangle(0, 0, 0, 10, fill=_LAVENDER, outline="")
        self._prog_canvas.bind("<Configure>", self._redraw_progress)

        split = Frame(body, bg=_BG)
        split.pack(fill="both", expand=True)

        # Left column
        left = Frame(split, bg=_BG, width=340)
        left.pack(side="left", fill="y", padx=(0, 14))
        left.pack_propagate(False)

        out_card = RoundedCard(left, width=340, height=140, radius=18, fill=_CARD)
        out_card.pack(fill="x", pady=(0, 12))
        out_card.configure(width=340, height=140)
        Label(
            out_card.content, text="OUTCOMES", font=(_UI, 12, "bold"), fg=_TEXT, bg=_CARD, anchor="w"
        ).pack(fill="x", pady=(0, 8))
        grid = Frame(out_card.content, bg=_CARD)
        grid.pack(fill="x")
        self._fin_var = StringVar(value="0")
        self._crash_var = StringVar(value="0")
        self._off_var = StringVar(value="0")
        for i, (title, var, color) in enumerate(
            (
                ("Finishes", self._fin_var, _GREEN),
                ("Crashes", self._crash_var, _RED),
                ("Off-track", self._off_var, _ORANGE),
            )
        ):
            cell = Frame(grid, bg=_ELEVATED)
            cell.grid(row=0, column=i, padx=(0 if i == 0 else 6), sticky="nsew", ipadx=4, ipady=4)
            grid.columnconfigure(i, weight=1)
            Label(cell, text=title, font=(_UI, 9), fg=_MUTED, bg=_ELEVATED).pack()
            Label(cell, textvariable=var, font=(_UI, 14, "bold"), fg=color, bg=_ELEVATED).pack()

        recent = RoundedCard(left, width=340, height=280, radius=18, fill=_CARD)
        recent.pack(fill="both", expand=True)
        Label(
            recent.content,
            text="LAST 5 EPISODES",
            font=(_UI, 12, "bold"),
            fg=_TEXT,
            bg=_CARD,
            anchor="w",
        ).pack(fill="x", pady=(0, 6))
        self._ep_labels: list[Label] = []
        for _ in range(5):
            lbl = Label(
                recent.content, text="—", font=(_MONO, 10), fg=_DIM, bg=_CARD, anchor="w"
            )
            lbl.pack(fill="x", pady=2)
            self._ep_labels.append(lbl)

        # Right: best runs
        right = RoundedCard(split, radius=20, fill=_CARD)
        right.pack(side="left", fill="both", expand=True)

        hdr = Frame(right.content, bg=_CARD)
        hdr.pack(fill="x", pady=(0, 6))
        Label(hdr, text="BEST RUNS", font=(_UI, 13, "bold"), fg=_TEXT, bg=_CARD).pack(
            side="left"
        )
        Label(
            hdr,
            text=f"Double-click · Chromium on :{self._watch_port}",
            font=(_UI, 9),
            fg=_DIM,
            bg=_CARD,
        ).pack(side="right")

        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure(
            "Runs.Treeview",
            background=_ELEVATED,
            foreground=_TEXT,
            fieldbackground=_ELEVATED,
            borderwidth=0,
            rowheight=28,
            font=(_MONO, 10),
        )
        style.configure(
            "Runs.Treeview.Heading",
            background=_CARD,
            foreground=_LAVENDER_DIM,
            font=(_UI, 9, "bold"),
            relief="flat",
        )
        style.map(
            "Runs.Treeview",
            background=[("selected", _LAVENDER)],
            foreground=[("selected", _BG)],
        )

        cols = ("id", "tag", "dist", "reward", "cps", "outcome", "times")
        tree_frame = Frame(right.content, bg=_CARD)
        tree_frame.pack(fill="both", expand=True)
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
            "outcome": ("Outcome", 90),
            "times": ("CP times", 180),
        }
        for key, (label, w) in headings.items():
            self._tree.heading(key, text=label)
            self._tree.column(key, width=w, anchor="w", stretch=(key == "times"))
        self._tree.pack(side="left", fill="both", expand=True)
        self._tree.bind("<Double-1>", lambda _e: self._watch_selected())
        self._tree.bind("<<TreeviewSelect>>", lambda _e: self._show_run_detail())

        self._detail_var = StringVar(
            value="Select a run, then Watch. A Chromium window opens — no extra terminal."
        )
        Label(
            right.content,
            textvariable=self._detail_var,
            font=(_UI, 9),
            fg=_MUTED,
            bg=_CARD,
            justify="left",
            wraplength=620,
            anchor="w",
        ).pack(fill="x", pady=(8, 6))

        btn_row = Frame(right.content, bg=_CARD)
        btn_row.pack(fill="x", pady=(0, 4))
        PillButton(btn_row, "Watch selected run", self._watch_selected, primary=True).pack(
            side="left"
        )
        PillButton(
            btn_row, "Stop", self._stop_watch_btn, primary=False, width=88
        ).pack(side="left", padx=(10, 0))
        self._watch_status = StringVar(value="")
        Label(
            btn_row,
            textvariable=self._watch_status,
            font=(_UI, 9),
            fg=_LAVENDER_DIM,
            bg=_CARD,
            anchor="w",
        ).pack(side="left", padx=14, fill="x", expand=True)

    def _set_live_pill(self, live: bool) -> None:
        c = self._live_canvas
        c.delete("all")
        fill = _GREEN if live else _DIM
        fg = _BG if live else _TEXT
        text = "LIVE" if live else "WAITING"
        _round_rect(c, 2, 2, 88, 26, 12, fill=fill, outline="")
        c.create_text(45, 14, text=text, fill=fg, font=(_UI, 9, "bold"))

    def _redraw_progress(self, _e: object | None = None) -> None:
        w = max(1, self._prog_canvas.winfo_width())
        self._prog_canvas.coords(self._prog_fill, 0, 0, int(w * self._prog_frac), 10)

    def _poll(self) -> None:
        self._refresh_metrics()
        self._refresh_runs()
        self._check_watch_proc()
        self._root.after(self._poll_ms, self._poll)

    def _refresh_metrics(self) -> None:
        try:
            data: dict = json.loads(self._json_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            self._set_live_pill(False)
            return

        self._set_live_pill(True)
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
        self._card_fitness.sub.set("metres · all-time")
        self._card_reward.value.set(f"{best_rew:+.2f}")
        self._card_reward.sub.set("episode reward · all-time")
        self._card_mean.value.set(f"{mean_fit:.1f}")
        self._card_mean.sub.set("metres · last 10")
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
            runs_raw = json.loads(self._runs_path.read_text(encoding="utf-8"))
            if not isinstance(runs_raw, list):
                return
            runs: list = runs_raw
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
                kw: dict = {
                    "cwd": str(_ROOT),
                    "stdout": subprocess.DEVNULL,
                    "stderr": subprocess.DEVNULL,
                }
                if sys.platform == "win32":
                    kw["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
                self._server_proc = subprocess.Popen(
                    [
                        sys.executable,
                        str(_ROOT / "start_server.py"),
                        "--port",
                        str(self._watch_port),
                    ],
                    **kw,
                )
            except Exception as exc:
                self._watch_status.set(f"Server failed: {exc}")
                return False
        for _ in range(50):
            if _port_open(self._watch_port):
                return True
            if self._server_proc is not None and self._server_proc.poll() is not None:
                self._watch_status.set("start_server.py exited early")
                return False
            time.sleep(0.12)
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
        if not model_path.exists():
            return f"Model missing: {model_path}"
        try:
            with zipfile.ZipFile(model_path, "r") as zf:
                names = set(zf.namelist())
        except zipfile.BadZipFile:
            return f"Corrupt zip: {model_path.name}"
        if "data" not in names and "policy.pth" not in names and not any(
            n.endswith(".pth") for n in names
        ):
            return f"Not an SB3 checkpoint: {model_path.name}"
        return None

    def _tail_watch_log(self) -> str:
        try:
            text = _WATCH_LOG.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if not lines:
            return ""
        # Prefer the last non-command line
        for ln in reversed(lines):
            if not ln.startswith("$"):
                return ln[:120]
        return lines[-1][:120]

    def _watch_selected(self) -> None:
        run = self._selected_run()
        if run is None:
            self._watch_status.set("Select a run first.")
            return

        model_rel = str(run.get("model_path") or "")
        model_path = self._resolve_model(model_rel)
        if model_path is None:
            fallback = (_ROOT / "checkpoints" / "best_model.zip").resolve()
            if fallback.exists():
                model_path = fallback
            else:
                self._watch_status.set(f"Model missing: {model_rel}")
                return

        err = self._preflight_model(model_path)
        if err:
            self._watch_status.set(err)
            return

        if self._watch_proc is not None and self._watch_proc.poll() is None:
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
            # No CREATE_NEW_CONSOLE — that was the blank black terminal.
            # Chromium opens as its own window; Python stays hidden and logs here.
            kw: dict = {
                "cwd": str(_ROOT),
                "stdout": log_f,
                "stderr": subprocess.STDOUT,
            }
            if sys.platform == "win32":
                kw["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            self._watch_proc = subprocess.Popen(cmd, **kw)
            self._watch_status.set(
                f"Watching #{run.get('id')} — loading model, then Chromium opens…"
            )
        except Exception as exc:
            self._watch_status.set(f"Failed: {exc}")

    def _check_watch_proc(self) -> None:
        if self._watch_proc is None:
            return
        # Live status from log while running
        if self._watch_proc.poll() is None:
            tip = self._tail_watch_log()
            if tip and not tip.startswith("$"):
                self._watch_status.set(tip)
            return

        code = self._watch_proc.poll()
        if self._watch_log_handle is not None:
            try:
                self._watch_log_handle.close()
            except OSError:
                pass
            self._watch_log_handle = None
        if code == 0:
            self._watch_status.set("Watch finished.")
        else:
            tip = self._tail_watch_log() or f"exit {code}"
            self._watch_status.set(f"Watch failed: {tip}")
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
    p.add_argument("--poll-ms", type=int, default=1500)
    p.add_argument("--watch-port", type=int, default=_WATCH_PORT)
    args = p.parse_args()

    root = Tk()
    TrainingMonitorGUI(
        root, args.json_path, args.runs_path, args.poll_ms, args.watch_port
    )
    root.mainloop()


if __name__ == "__main__":
    main()
