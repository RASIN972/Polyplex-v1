#!/usr/bin/env python3
"""Polyplex training control dashboard (lavender / black).

Launch with::

    python start_gui.py

Features:
  - Start / stop training with num-envs, headless, watch-worker-0, dummy vec
  - Live metrics + collapsible progress graph
  - Best-runs browser + Watch replay
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
import zipfile
from pathlib import Path
from tkinter import (
    BooleanVar,
    Button,
    Canvas,
    Checkbutton,
    Frame,
    IntVar,
    Label,
    Spinbox,
    StringVar,
    Tk,
    ttk,
)
from typing import TextIO

_ROOT = Path(__file__).resolve().parent
_DEFAULT_JSON = _ROOT / "logs" / "training_live.json"
_DEFAULT_RUNS = _ROOT / "logs" / "best_runs.json"
_WATCH_LOG = _ROOT / "logs" / "watch_run.log"
_WATCH_PORT = 8099

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
        x1 + r, y1, x2 - r, y1, x2, y1, x2, y1 + r,
        x2, y2 - r, x2, y2, x2 - r, y2, x1 + r, y2,
        x1, y2, x1, y2 - r, x1, y1 + r, x1, y1,
    ]
    return canvas.create_polygon(points, smooth=True, **kw)  # type: ignore[arg-type]


def _pill_button(
    parent: Frame,
    text: str,
    command: object,
    *,
    primary: bool = True,
    danger: bool = False,
) -> Button:
    if danger:
        bg, abg, fg = _RED, "#fb7185", _BG
    elif primary:
        bg, abg, fg = _LAVENDER, "#b8a4ff", _BG
    else:
        bg, abg, fg = _ELEVATED, _BORDER, _LAVENDER_SOFT
    return Button(
        parent,
        text=text,
        command=command,  # type: ignore[arg-type]
        bg=bg,
        fg=fg,
        activebackground=abg,
        activeforeground=fg if not primary and not danger else _BG,
        font=(_UI, 11, "bold"),
        relief="flat",
        borderwidth=0,
        padx=18,
        pady=9,
        cursor="hand2",
        highlightthickness=0,
    )


class RoundedCard(Frame):
    def __init__(
        self,
        parent: Frame | Tk,
        *,
        width: int | None = None,
        height: int | None = None,
        radius: int = 18,
        fill: str = _CARD,
        padding: int = 14,
    ) -> None:
        super().__init__(parent, bg=_BG)
        if width is not None:
            self.configure(width=width)
        if height is not None:
            self.configure(height=height)
        if width is not None or height is not None:
            self.pack_propagate(False)
        self._radius = radius
        self._fill = fill
        self._canvas = Canvas(self, bg=_BG, highlightthickness=0, bd=0)
        self._canvas.place(x=0, y=0, relwidth=1, relheight=1)
        self.content = Frame(self, bg=fill)
        self.content.pack(fill="both", expand=True, padx=padding, pady=padding)
        self.content.lift()
        self.bind("<Configure>", self._paint)
        self._canvas.bind("<Configure>", self._paint)

    def _paint(self, _e: object | None = None) -> None:
        w = max(self.winfo_width(), 2)
        h = max(self.winfo_height(), 2)
        self._canvas.delete("all")
        _round_rect(
            self._canvas, 1, 1, w - 1, h - 1, self._radius, fill=self._fill, outline=_BORDER
        )


class MetricCard(RoundedCard):
    def __init__(
        self,
        parent: Frame,
        title: str,
        *,
        accent: str = _LAVENDER,
        width: int = 158,
    ) -> None:
        super().__init__(parent, width=width, height=104, radius=20, fill=_CARD)
        Label(
            self.content, text=title.upper(), font=(_UI, 9), fg=_LAVENDER_DIM, bg=_CARD, anchor="w"
        ).pack(fill="x")
        self.value = StringVar(value="—")
        Label(
            self.content,
            textvariable=self.value,
            font=(_UI, 24, "bold"),
            fg=accent,
            bg=_CARD,
            anchor="w",
        ).pack(fill="x", pady=(2, 0))
        self.sub = StringVar(value="")
        Label(
            self.content, textvariable=self.sub, font=(_UI, 9), fg=_DIM, bg=_CARD, anchor="w"
        ).pack(fill="x")


class ProgressGraph(Frame):
    """Collapsible dual-series chart (mean fitness + mean reward)."""

    def __init__(self, parent: Frame) -> None:
        super().__init__(parent, bg=_BG)
        self._expanded = BooleanVar(value=True)
        self._hist: dict = {}

        hdr = Frame(self, bg=_BG)
        hdr.pack(fill="x")
        self._toggle_btn = Button(
            hdr,
            text="▼  Progress graph",
            command=self._toggle,
            bg=_BG,
            fg=_LAVENDER_SOFT,
            activebackground=_BG,
            activeforeground=_LAVENDER,
            font=(_UI, 11, "bold"),
            relief="flat",
            borderwidth=0,
            cursor="hand2",
            anchor="w",
            highlightthickness=0,
        )
        self._toggle_btn.pack(side="left")
        Label(
            hdr,
            text="lavender = mean distance   ·   orange = mean reward",
            font=(_UI, 9),
            fg=_DIM,
            bg=_BG,
        ).pack(side="right")

        self._body = RoundedCard(self, height=200, radius=18, fill=_CARD)
        self._body.pack(fill="x", pady=(8, 0))
        self._body.configure(height=200)
        self._canvas = Canvas(
            self._body.content, height=160, bg=_ELEVATED, highlightthickness=0, bd=0
        )
        self._canvas.pack(fill="both", expand=True)
        self._canvas.bind("<Configure>", lambda _e: self.redraw())
        self._empty = Label(
            self._body.content,
            text="Start training to see progress…",
            font=(_UI, 10),
            fg=_DIM,
            bg=_ELEVATED,
        )

    def _toggle(self) -> None:
        if self._expanded.get():
            self._expanded.set(False)
            self._body.pack_forget()
            self._toggle_btn.config(text="▶  Progress graph")
        else:
            self._expanded.set(True)
            self._body.pack(fill="x", pady=(8, 0))
            self._toggle_btn.config(text="▼  Progress graph")
            self.redraw()

    def update_history(self, hist: dict) -> None:
        self._hist = hist or {}
        if self._expanded.get():
            self.redraw()

    def redraw(self) -> None:
        c = self._canvas
        c.delete("all")
        w = max(c.winfo_width(), 40)
        h = max(c.winfo_height(), 40)
        pad_l, pad_r, pad_t, pad_b = 44, 12, 12, 24
        c.create_rectangle(0, 0, w, h, fill=_ELEVATED, outline="")

        fit = [float(x) for x in (self._hist.get("mean_fitness") or [])]
        rew = [float(x) for x in (self._hist.get("mean_reward") or [])]
        ts = [int(x) for x in (self._hist.get("timesteps") or [])]
        if len(fit) < 2:
            c.create_text(
                w / 2, h / 2, text="Start training to see progress…", fill=_DIM, font=(_UI, 11)
            )
            return

        def _series(vals: list[float], color: str) -> None:
            if len(vals) < 2:
                return
            vmin, vmax = min(vals), max(vals)
            if abs(vmax - vmin) < 1e-9:
                vmax = vmin + 1.0
            n = len(vals)
            pts: list[float] = []
            for i, v in enumerate(vals):
                x = pad_l + (w - pad_l - pad_r) * (i / (n - 1))
                y = pad_t + (h - pad_t - pad_b) * (1.0 - (v - vmin) / (vmax - vmin))
                pts.extend([x, y])
            c.create_line(*pts, fill=color, width=2, smooth=True)

        # area backdrop grid
        for i in range(4):
            y = pad_t + (h - pad_t - pad_b) * i / 3
            c.create_line(pad_l, y, w - pad_r, y, fill=_BORDER)

        _series(fit, _LAVENDER)
        _series(rew, _ORANGE)

        if ts:
            c.create_text(
                pad_l, h - 8, text=f"{ts[0]:,}", fill=_DIM, font=(_UI, 8), anchor="w"
            )
            c.create_text(
                w - pad_r, h - 8, text=f"{ts[-1]:,}", fill=_DIM, font=(_UI, 8), anchor="e"
            )
        if fit:
            c.create_text(
                6, pad_t, text=f"{max(fit):.0f}", fill=_LAVENDER_DIM, font=(_UI, 8), anchor="nw"
            )
            c.create_text(
                6, h - pad_b, text=f"{min(fit):.0f}", fill=_LAVENDER_DIM, font=(_UI, 8), anchor="sw"
            )


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
        self._train_proc: subprocess.Popen | None = None
        self._train_via_terminal = False
        self._watch_log_handle: TextIO | None = None
        self._prog_frac = 0.0

        self._num_envs = IntVar(value=4)
        self._headless = BooleanVar(value=True)
        self._watch_live = BooleanVar(value=False)
        self._dummy_vec = BooleanVar(value=False)
        self._timesteps = IntVar(value=1_000_000)

        root.title("Polyplex — Training Control")
        root.configure(bg=_BG)
        root.minsize(1120, 740)
        root.resizable(True, True)

        self._build_ui()
        root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._poll()

    # ── UI ────────────────────────────────────────────────────────────
    def _build_ui(self) -> None:
        root = self._root

        nav = Frame(root, bg=_BG, height=60)
        nav.pack(fill="x", padx=20, pady=(14, 0))
        nav.pack_propagate(False)
        Label(
            nav, text="Polyplex", font=(_UI, 20, "bold"), fg=_LAVENDER_SOFT, bg=_BG
        ).pack(side="left", pady=8)
        Label(
            nav, text="  Training control", font=(_UI, 11), fg=_MUTED, bg=_BG
        ).pack(side="left", pady=14)
        self._live_canvas = Canvas(nav, width=96, height=28, bg=_BG, highlightthickness=0)
        self._live_canvas.pack(side="right", pady=16)
        self._set_live_pill(False, "IDLE")

        body = Frame(root, bg=_BG)
        body.pack(fill="both", expand=True, padx=20, pady=12)

        # Control panel
        ctrl = RoundedCard(body, height=118, radius=18, fill=_CARD)
        ctrl.pack(fill="x", pady=(0, 12))
        ctrl.configure(height=118)

        Label(
            ctrl.content, text="TRAINING", font=(_UI, 12, "bold"), fg=_TEXT, bg=_CARD, anchor="w"
        ).pack(fill="x")

        opts = Frame(ctrl.content, bg=_CARD)
        opts.pack(fill="x", pady=(8, 0))

        Label(opts, text="Envs", font=(_UI, 10), fg=_MUTED, bg=_CARD).pack(side="left")
        Spinbox(
            opts,
            from_=1,
            to=8,
            textvariable=self._num_envs,
            width=4,
            font=(_UI, 11),
            bg=_ELEVATED,
            fg=_TEXT,
            buttonbackground=_ELEVATED,
            relief="flat",
            highlightthickness=1,
            highlightbackground=_BORDER,
        ).pack(side="left", padx=(6, 14))

        Label(opts, text="Timesteps", font=(_UI, 10), fg=_MUTED, bg=_CARD).pack(side="left")
        Spinbox(
            opts,
            from_=10_000,
            to=10_000_000,
            increment=50_000,
            textvariable=self._timesteps,
            width=10,
            font=(_UI, 11),
            bg=_ELEVATED,
            fg=_TEXT,
            buttonbackground=_ELEVATED,
            relief="flat",
            highlightthickness=1,
            highlightbackground=_BORDER,
        ).pack(side="left", padx=(6, 14))

        Checkbutton(
            opts,
            text="Headless",
            variable=self._headless,
            font=(_UI, 10),
            fg=_LAVENDER_SOFT,
            bg=_CARD,
            activebackground=_CARD,
            activeforeground=_LAVENDER,
            selectcolor=_ELEVATED,
            highlightthickness=0,
        ).pack(side="left", padx=(0, 10))
        Checkbutton(
            opts,
            text="Watch env 0",
            variable=self._watch_live,
            font=(_UI, 10),
            fg=_LAVENDER_SOFT,
            bg=_CARD,
            activebackground=_CARD,
            activeforeground=_LAVENDER,
            selectcolor=_ELEVATED,
            highlightthickness=0,
        ).pack(side="left", padx=(0, 10))
        Checkbutton(
            opts,
            text="Dummy vec (debug)",
            variable=self._dummy_vec,
            font=(_UI, 10),
            fg=_MUTED,
            bg=_CARD,
            activebackground=_CARD,
            activeforeground=_LAVENDER,
            selectcolor=_ELEVATED,
            highlightthickness=0,
        ).pack(side="left", padx=(0, 10))

        btns = Frame(ctrl.content, bg=_CARD)
        btns.pack(fill="x", pady=(10, 0))
        self._start_btn = _pill_button(btns, "Start training", self._start_training, primary=True)
        self._start_btn.pack(side="left")
        self._stop_btn = _pill_button(
            btns, "Stop training", self._stop_training, primary=False, danger=True
        )
        self._stop_btn.pack(side="left", padx=(10, 0))
        self._train_status = StringVar(value="Idle — configure options, then Start.")
        Label(
            btns,
            textvariable=self._train_status,
            font=(_UI, 9),
            fg=_LAVENDER_DIM,
            bg=_CARD,
            anchor="w",
        ).pack(side="left", padx=14, fill="x", expand=True)

        # Metrics
        metrics = Frame(body, bg=_BG)
        metrics.pack(fill="x", pady=(0, 10))
        self._card_fitness = MetricCard(metrics, "Best distance", accent=_GREEN)
        self._card_fitness.pack(side="left", padx=(0, 10))
        self._card_reward = MetricCard(metrics, "Best reward", accent=_ORANGE)
        self._card_reward.pack(side="left", padx=(0, 10))
        self._card_mean = MetricCard(metrics, "Mean distance", accent=_LAVENDER)
        self._card_mean.pack(side="left", padx=(0, 10))
        self._card_steps = MetricCard(metrics, "Timesteps", accent=_LAVENDER_SOFT)
        self._card_steps.pack(side="left", padx=(0, 10))
        self._card_fps = MetricCard(metrics, "Rollout FPS", accent=_AMBER)
        self._card_fps.pack(side="left")

        # Progress bar strip
        prog_wrap = RoundedCard(body, height=64, radius=16, fill=_CARD)
        prog_wrap.pack(fill="x", pady=(0, 10))
        prog_wrap.configure(height=64)
        row = Frame(prog_wrap.content, bg=_CARD)
        row.pack(fill="x")
        self._progress_lbl = StringVar(value="Progress — not training")
        Label(
            row, textvariable=self._progress_lbl, font=(_UI, 11), fg=_TEXT, bg=_CARD
        ).pack(side="left")
        self._uptime_lbl = StringVar(value="")
        Label(
            row, textvariable=self._uptime_lbl, font=(_UI, 9), fg=_MUTED, bg=_CARD
        ).pack(side="right")
        self._prog_canvas = Canvas(
            prog_wrap.content, height=8, bg=_ELEVATED, highlightthickness=0, bd=0
        )
        self._prog_canvas.pack(fill="x", pady=(8, 0))
        self._prog_fill = self._prog_canvas.create_rectangle(
            0, 0, 0, 8, fill=_LAVENDER, outline=""
        )
        self._prog_canvas.bind("<Configure>", self._redraw_progress)

        # Collapsible graph
        self._graph = ProgressGraph(body)
        self._graph.pack(fill="x", pady=(0, 12))

        # Main split
        split = Frame(body, bg=_BG)
        split.pack(fill="both", expand=True)

        left = Frame(split, bg=_BG, width=320)
        left.pack(side="left", fill="y", padx=(0, 12))
        left.pack_propagate(False)

        out_card = RoundedCard(left, width=320, height=130, radius=18, fill=_CARD)
        out_card.pack(fill="x", pady=(0, 10))
        out_card.configure(width=320, height=130)
        Label(
            out_card.content, text="OUTCOMES", font=(_UI, 12, "bold"), fg=_TEXT, bg=_CARD, anchor="w"
        ).pack(fill="x", pady=(0, 6))
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
            cell.grid(row=0, column=i, padx=(0 if i == 0 else 6), sticky="nsew", ipady=2)
            grid.columnconfigure(i, weight=1)
            Label(cell, text=title, font=(_UI, 9), fg=_MUTED, bg=_ELEVATED).pack()
            Label(cell, textvariable=var, font=(_UI, 13, "bold"), fg=color, bg=_ELEVATED).pack()

        recent = RoundedCard(left, width=320, radius=18, fill=_CARD)
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

        right = RoundedCard(split, radius=20, fill=_CARD)
        right.pack(side="left", fill="both", expand=True)

        hdr = Frame(right.content, bg=_CARD)
        hdr.pack(fill="x", pady=(0, 6))
        Label(hdr, text="BEST RUNS", font=(_UI, 13, "bold"), fg=_TEXT, bg=_CARD).pack(
            side="left"
        )
        Label(
            hdr,
            text=f"Double-click Watch · :{self._watch_port}",
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
            "times": ("CP times", 160),
        }
        for key, (label, w) in headings.items():
            self._tree.heading(key, text=label)
            self._tree.column(key, width=w, anchor="w", stretch=(key == "times"))
        self._tree.pack(side="left", fill="both", expand=True)
        self._tree.bind("<Double-1>", lambda _e: self._watch_selected())
        self._tree.bind("<<TreeviewSelect>>", lambda _e: self._show_run_detail())

        self._detail_var = StringVar(value="Select a run to inspect or Watch.")
        Label(
            right.content,
            textvariable=self._detail_var,
            font=(_UI, 9),
            fg=_MUTED,
            bg=_CARD,
            justify="left",
            wraplength=600,
            anchor="w",
        ).pack(fill="x", pady=(8, 6))

        btn_row = Frame(right.content, bg=_CARD)
        btn_row.pack(fill="x")
        _pill_button(btn_row, "Watch selected run", self._watch_selected, primary=True).pack(
            side="left"
        )
        _pill_button(btn_row, "Stop watch", self._stop_watch_btn, primary=False).pack(
            side="left", padx=(10, 0)
        )
        self._watch_status = StringVar(value="")
        Label(
            btn_row,
            textvariable=self._watch_status,
            font=(_UI, 9),
            fg=_LAVENDER_DIM,
            bg=_CARD,
            anchor="w",
        ).pack(side="left", padx=12, fill="x", expand=True)

    # ── Training start / stop ─────────────────────────────────────────
    def _build_train_cmd(self) -> list[str]:
        n = max(1, min(8, int(self._num_envs.get())))
        total = max(1000, int(self._timesteps.get()))
        cmd = [
            sys.executable,
            "-u",
            str(_ROOT / "run_local_training.py"),
            "--num-envs",
            str(n),
            "--total-timesteps",
            str(total),
        ]
        if not self._headless.get():
            cmd.append("--headed")
        elif self._watch_live.get():
            cmd.append("--watch")
        if self._dummy_vec.get():
            cmd.extend(["--vec-env", "dummy"])
        return cmd

    def _start_training(self) -> None:
        if self._train_proc is not None and self._train_proc.poll() is None:
            self._train_status.set("Training already running.")
            return

        cmd = self._build_train_cmd()
        n = max(1, min(8, int(self._num_envs.get())))
        env = {
            **os.environ,
            "POLYTRACK_FROM_GUI": "1",
            "PYTHONUTF8": "1",
            "PYTHONIOENCODING": "utf-8",
        }
        try:
            if sys.platform == "win32":
                # Visible cmd window — training logs print there.
                self._train_proc = subprocess.Popen(
                    cmd,
                    cwd=str(_ROOT),
                    env=env,
                    creationflags=(
                        getattr(subprocess, "CREATE_NEW_CONSOLE", 0)
                        | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
                    ),
                )
            elif sys.platform == "darwin":
                import json as _json
                import shlex

                joined = " ".join(shlex.quote(c) for c in cmd)
                script = f"cd {shlex.quote(str(_ROOT))} && {joined}"
                # One Terminal tab only (do not also Popen the same cmd).
                osa = subprocess.Popen(
                    [
                        "osascript",
                        "-e",
                        f"tell application \"Terminal\" to do script {_json.dumps(script)}",
                    ]
                )
                self._train_proc = osa  # Stop may only kill osascript; use pkill fallback
                self._train_via_terminal = True
            else:
                from shutil import which

                term_cmd: list[str] | None = None
                if which("gnome-terminal"):
                    term_cmd = ["gnome-terminal", "--", *cmd]
                elif which("xterm"):
                    term_cmd = ["xterm", "-e", *cmd]
                if term_cmd is not None:
                    self._train_proc = subprocess.Popen(
                        term_cmd, cwd=str(_ROOT), env=env, start_new_session=True
                    )
                else:
                    self._train_proc = subprocess.Popen(
                        cmd, cwd=str(_ROOT), env=env, start_new_session=True
                    )
                self._train_via_terminal = False

            if sys.platform == "win32":
                self._train_via_terminal = False

            mode = "headed" if not self._headless.get() else (
                "watch-0" if self._watch_live.get() else "headless"
            )
            pid = self._train_proc.pid if self._train_proc else "?"
            self._train_status.set(
                f"Training started in a terminal window · {n} envs · {mode} · pid {pid}"
            )
            self._set_live_pill(True, "TRAINING")
            self._start_btn.config(state="disabled")
        except Exception as exc:
            self._train_status.set(f"Failed to start: {exc}")

    def _stop_training(self) -> None:
        proc = self._train_proc
        self._train_status.set("Stopping training…")
        self._root.update_idletasks()
        try:
            if sys.platform == "win32":
                if proc is not None and proc.poll() is None:
                    subprocess.run(
                        ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                        capture_output=True,
                        check=False,
                    )
            else:
                # Kill train processes started from this project.
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
            self._train_status.set(f"Stop error: {exc}")
            return
        self._train_proc = None
        self._train_via_terminal = False
        self._start_btn.config(state="normal")
        self._set_live_pill(False, "STOPPED")
        self._train_status.set("Training stopped.")

    def _check_train_proc(self) -> None:
        if self._train_proc is None:
            return
        # Terminal-launched jobs: osascript exits immediately — keep "TRAINING"
        # until user hits Stop or live JSON stops updating for a long time.
        if self._train_via_terminal:
            return
        code = self._train_proc.poll()
        if code is None:
            return
        if code == 0:
            self._train_status.set("Training finished.")
            self._set_live_pill(False, "DONE")
        else:
            self._train_status.set(f"Training exited (code {code}). Check the terminal window.")
            self._set_live_pill(False, "ERROR")
        self._train_proc = None
        self._start_btn.config(state="normal")

    # ── Live metrics ──────────────────────────────────────────────────
    def _set_live_pill(self, on: bool, text: str) -> None:
        c = self._live_canvas
        c.delete("all")
        fill = _GREEN if on else _DIM
        _round_rect(c, 2, 2, 94, 26, 12, fill=fill, outline="")
        c.create_text(48, 14, text=text, fill=_BG if on else _TEXT, font=(_UI, 8, "bold"))

    def _redraw_progress(self, _e: object | None = None) -> None:
        w = max(1, self._prog_canvas.winfo_width())
        self._prog_canvas.coords(self._prog_fill, 0, 0, int(w * self._prog_frac), 8)

    def _poll(self) -> None:
        self._refresh_metrics()
        self._refresh_runs()
        self._check_watch_proc()
        self._check_train_proc()
        self._root.after(self._poll_ms, self._poll)

    def _refresh_metrics(self) -> None:
        try:
            data: dict = json.loads(self._json_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            if self._train_proc is None or self._train_proc.poll() is not None:
                if self._train_proc is None:
                    pass
            return

        training = self._train_proc is not None and self._train_proc.poll() is None
        if training:
            self._set_live_pill(True, "LIVE")

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

        hist = data.get("history") or {}
        self._graph.update_history(hist)

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

    # ── Best-run watch ────────────────────────────────────────────────
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
        for ln in reversed(lines):
            if not ln.startswith("$"):
                return ln[:120]
        return lines[-1][:120] if lines else ""

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
            "0",
            "--auto-server",
        ]
        try:
            log_f: TextIO = open(_WATCH_LOG, "w", encoding="utf-8")
            log_f.write(f"$ {' '.join(cmd)}\n\n")
            log_f.flush()
            self._watch_log_handle = log_f
            kw: dict = {"cwd": str(_ROOT), "stdout": log_f, "stderr": subprocess.STDOUT}
            if sys.platform == "win32":
                kw["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            self._watch_proc = subprocess.Popen(cmd, **kw)
            self._watch_status.set(
                f"Watching #{run.get('id')} — Chromium opens after model load…"
            )
        except Exception as exc:
            self._watch_status.set(f"Failed: {exc}")

    def _check_watch_proc(self) -> None:
        if self._watch_proc is None:
            return
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
        if self._train_proc is not None and self._train_proc.poll() is None:
            self._stop_training()
        self._stop_watch()
        if self._server_proc is not None and self._server_proc.poll() is None:
            self._server_proc.terminate()
        self._root.destroy()


def main() -> None:
    p = argparse.ArgumentParser(description="Polyplex training control GUI.")
    p.add_argument("--json-path", type=Path, default=_DEFAULT_JSON)
    p.add_argument("--runs-path", type=Path, default=_DEFAULT_RUNS)
    p.add_argument("--poll-ms", type=int, default=1500)
    p.add_argument("--watch-port", type=int, default=_WATCH_PORT)
    args = p.parse_args()
    try:
        root = Tk()
        TrainingMonitorGUI(
            root, args.json_path, args.runs_path, args.poll_ms, args.watch_port
        )
        root.mainloop()
    except Exception:
        import traceback

        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
