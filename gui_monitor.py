#!/usr/bin/env python3
"""Polyplex training control dashboard — modern web UI (NiceGUI / Quasar).

Launch with::

    python start_gui.py

Opens in your browser with native-feeling buttons, charts, and scrollbars.
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
from typing import Any, TextIO

from nicegui import app, ui

_ROOT = Path(__file__).resolve().parent
_DEFAULT_JSON = _ROOT / "logs" / "training_live.json"
_DEFAULT_RUNS = _ROOT / "logs" / "best_runs.json"
_WATCH_LOG = _ROOT / "logs" / "watch_run.log"
_WATCH_PORT = 8099

# Theme
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

_OUTCOME_COLORS = {
    "finished": _GREEN,
    "crashed": _RED,
    "timeout": _AMBER,
    "off_track": _ORANGE,
}

_CLI_ARGS: argparse.Namespace | None = None


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


class TrainingMonitorApp:
    def __init__(
        self,
        json_path: Path,
        runs_path: Path,
        poll_ms: int,
        watch_port: int,
    ) -> None:
        self._json_path = json_path
        self._runs_path = runs_path
        self._poll_ms = poll_ms
        self._watch_port = watch_port
        self._runs: list[dict] = []
        self._selected_run_id: int | None = None
        self._watch_proc: subprocess.Popen | None = None
        self._server_proc: subprocess.Popen | None = None
        self._train_proc: subprocess.Popen | None = None
        self._train_via_terminal = False
        self._watch_log_handle: TextIO | None = None

        # Controls
        self.num_envs = 4
        self.timesteps = 1_000_000
        self.headless = True
        self.watch_live = False
        self.dummy_vec = False

        # UI refs
        self._live_pill: ui.badge | None = None
        self._start_btn: ui.button | None = None
        self._train_status: ui.label | None = None
        self._progress_lbl: ui.label | None = None
        self._uptime_lbl: ui.label | None = None
        self._progress_bar: ui.linear_progress | None = None
        self._chart: ui.echart | None = None
        self._graph_expanded = True
        self._graph_body: ui.element | None = None
        self._graph_toggle: ui.button | None = None
        self._runs_table: ui.table | None = None
        self._detail_lbl: ui.label | None = None
        self._watch_status: ui.label | None = None
        self._ep_labels: list[ui.label] = []
        self._metric_values: dict[str, ui.label] = {}
        self._metric_subs: dict[str, ui.label] = {}
        self._outcome_values: dict[str, ui.label] = {}

        self._build_ui()
        ui.timer(poll_ms / 1000.0, self._poll)
        app.on_shutdown(self._on_close)

    # ── UI ────────────────────────────────────────────────────────────
    def _build_ui(self) -> None:
        ui.query("body").style(
            f"background: {_BG}; color: {_TEXT}; "
            "font-family: 'Segoe UI', 'Helvetica Neue', system-ui, sans-serif;"
        )
        ui.add_head_html(
            f"""
            <style>
              :root {{
                --q-primary: {_LAVENDER};
                --q-secondary: {_ORANGE};
                --q-positive: {_GREEN};
                --q-negative: {_RED};
                --q-warning: {_AMBER};
                --q-dark: {_CARD};
              }}
              .poly-card {{
                background: {_CARD} !important;
                border: 1px solid {_BORDER};
                border-radius: 18px;
                padding: 16px;
              }}
              .poly-elevated {{
                background: {_ELEVATED} !important;
                border-radius: 12px;
              }}
              .poly-metric-title {{
                font-size: 11px;
                letter-spacing: 0.06em;
                color: {_LAVENDER_DIM};
                font-weight: 600;
              }}
              .poly-metric-value {{
                font-size: 28px;
                font-weight: 700;
                line-height: 1.15;
                margin-top: 4px;
              }}
              .poly-metric-sub {{
                font-size: 12px;
                color: {_DIM};
                margin-top: 2px;
              }}
              .poly-ep {{
                font-family: Consolas, Menlo, monospace;
                font-size: 12px;
                padding: 4px 0;
              }}
              .q-table {{
                background: {_ELEVATED} !important;
                color: {_TEXT} !important;
                border-radius: 12px;
              }}
              .q-table__top, .q-table__bottom, thead tr, tbody td {{
                background: {_ELEVATED} !important;
                color: {_TEXT} !important;
                border-color: {_BORDER} !important;
              }}
              .q-table thead th {{
                color: {_LAVENDER_DIM} !important;
                font-size: 11px !important;
                letter-spacing: 0.04em;
              }}
              .q-table tbody tr:hover {{
                background: {_CARD} !important;
              }}
              .q-table__selected {{
                background: {_LAVENDER}33 !important;
              }}
              .q-field__control {{
                background: {_ELEVATED} !important;
                border-radius: 10px !important;
              }}
              .q-checkbox__label, .q-toggle__label {{
                color: {_LAVENDER_SOFT} !important;
              }}
              ::-webkit-scrollbar {{ width: 8px; height: 8px; }}
              ::-webkit-scrollbar-track {{ background: {_BG}; }}
              ::-webkit-scrollbar-thumb {{
                background: {_BORDER};
                border-radius: 8px;
              }}
              ::-webkit-scrollbar-thumb:hover {{ background: {_DIM}; }}
            </style>
            """
        )

        with ui.column().classes("w-full max-w-[1280px] mx-auto q-pa-md").style(
            "gap: 12px; min-height: 100vh;"
        ):
            self._build_nav()
            self._build_controls()
            self._build_metrics()
            self._build_progress()
            self._build_graph()
            self._build_main_split()

    def _build_nav(self) -> None:
        with ui.row().classes("w-full items-center justify-between").style(
            "min-height: 48px;"
        ):
            with ui.row().classes("items-baseline").style("gap: 10px;"):
                ui.label("Polyplex").style(
                    f"font-size: 22px; font-weight: 700; color: {_LAVENDER_SOFT};"
                )
                ui.label("Training control").style(
                    f"font-size: 13px; color: {_MUTED};"
                )
            self._live_pill = ui.badge("IDLE").props("rounded").style(
                f"background: {_DIM}; color: {_TEXT}; font-weight: 700; "
                "padding: 6px 14px; letter-spacing: 0.04em;"
            )

    def _build_controls(self) -> None:
        with ui.element("div").classes("poly-card w-full"):
            ui.label("TRAINING").style(
                f"font-size: 13px; font-weight: 700; color: {_TEXT};"
            )
            with ui.row().classes("w-full items-end flex-wrap").style(
                "gap: 16px; margin-top: 12px;"
            ):
                self._num_envs_input = (
                    ui.number("Envs", value=4, min=1, max=8, step=1)
                    .props("dense outlined dark")
                    .style("width: 96px;")
                    .bind_value(self, "num_envs")
                )
                self._timesteps_input = (
                    ui.number("Timesteps", value=1_000_000, min=10_000, max=10_000_000, step=50_000)
                    .props("dense outlined dark")
                    .style("width: 160px;")
                    .bind_value(self, "timesteps")
                )
                ui.checkbox("Headless").props("dark dense").bind_value(self, "headless")
                ui.checkbox("Watch env 0").props("dark dense").bind_value(self, "watch_live")
                ui.checkbox("Dummy vec (debug)").props("dark dense").bind_value(
                    self, "dummy_vec"
                )

            with ui.row().classes("w-full items-center").style(
                "gap: 10px; margin-top: 14px;"
            ):
                self._start_btn = (
                    ui.button("Start training", on_click=self._start_training)
                    .props("unelevated no-caps")
                    .style(
                        f"background: {_LAVENDER} !important; color: {_BG} !important; "
                        "font-weight: 700; border-radius: 999px; padding: 8px 20px;"
                    )
                )
                ui.button("Stop training", on_click=self._stop_training).props(
                    "unelevated no-caps"
                ).style(
                    f"background: {_RED} !important; color: {_BG} !important; "
                    "font-weight: 700; border-radius: 999px; padding: 8px 20px;"
                )
                self._train_status = ui.label(
                    "Idle — configure options, then Start."
                ).style(f"color: {_LAVENDER_DIM}; font-size: 12px;")

    def _build_metrics(self) -> None:
        specs = [
            ("fitness", "BEST DISTANCE", _GREEN),
            ("reward", "BEST REWARD", _ORANGE),
            ("mean", "MEAN DISTANCE", _LAVENDER),
            ("steps", "TIMESTEPS", _LAVENDER_SOFT),
            ("fps", "ROLLOUT FPS", _AMBER),
        ]
        with ui.row().classes("w-full").style("gap: 10px;"):
            for key, title, accent in specs:
                with ui.element("div").classes("poly-card").style(
                    "flex: 1; min-width: 140px;"
                ):
                    ui.label(title).classes("poly-metric-title")
                    val = ui.label("—").classes("poly-metric-value").style(
                        f"color: {accent};"
                    )
                    sub = ui.label("").classes("poly-metric-sub")
                    self._metric_values[key] = val
                    self._metric_subs[key] = sub

    def _build_progress(self) -> None:
        with ui.element("div").classes("poly-card w-full"):
            with ui.row().classes("w-full items-center justify-between"):
                self._progress_lbl = ui.label("Progress — not training").style(
                    f"color: {_TEXT}; font-size: 13px; font-weight: 500;"
                )
                self._uptime_lbl = ui.label("").style(
                    f"color: {_MUTED}; font-size: 12px;"
                )
            self._progress_bar = (
                ui.linear_progress(value=0, show_value=False)
                .props("rounded color=primary track-color=grey-9")
                .classes("w-full")
                .style("height: 8px; margin-top: 10px;")
            )

    def _build_graph(self) -> None:
        with ui.column().classes("w-full").style("gap: 8px;"):
            with ui.row().classes("w-full items-center justify-between"):
                self._graph_toggle = (
                    ui.button("▼  Progress graph", on_click=self._toggle_graph)
                    .props("flat no-caps dense")
                    .style(
                        f"color: {_LAVENDER_SOFT}; font-weight: 700; font-size: 13px;"
                    )
                )
                ui.label(
                    "lavender = mean distance   ·   orange = mean reward"
                ).style(f"color: {_DIM}; font-size: 12px;")

            self._graph_body = ui.element("div").classes("poly-card w-full")
            with self._graph_body:
                options = {
                    "backgroundColor": "transparent",
                    "animation": False,
                    "grid": {
                        "left": 48,
                        "right": 16,
                        "top": 20,
                        "bottom": 28,
                    },
                    "tooltip": {"trigger": "axis"},
                    "xAxis": {
                        "type": "category",
                        "data": [],
                        "axisLabel": {"color": _DIM, "fontSize": 10},
                        "axisLine": {"lineStyle": {"color": _BORDER}},
                        "axisTick": {"show": False},
                    },
                    "yAxis": [
                        {
                            "type": "value",
                            "scale": True,
                            "axisLabel": {"color": _LAVENDER_DIM, "fontSize": 10},
                            "splitLine": {"lineStyle": {"color": _BORDER}},
                            "axisLine": {"show": False},
                        },
                        {
                            "type": "value",
                            "scale": True,
                            "axisLabel": {"color": _ORANGE, "fontSize": 10},
                            "splitLine": {"show": False},
                            "axisLine": {"show": False},
                        },
                    ],
                    "series": [
                        {
                            "name": "Mean distance",
                            "type": "line",
                            "smooth": True,
                            "showSymbol": False,
                            "data": [],
                            "lineStyle": {"color": _LAVENDER, "width": 2.5},
                            "areaStyle": {"color": f"{_LAVENDER}22"},
                        },
                        {
                            "name": "Mean reward",
                            "type": "line",
                            "smooth": True,
                            "showSymbol": False,
                            "yAxisIndex": 1,
                            "data": [],
                            "lineStyle": {"color": _ORANGE, "width": 2.5},
                        },
                    ],
                }
                self._chart = ui.echart(options).classes("w-full").style(
                    "height: 200px;"
                )

    def _build_main_split(self) -> None:
        with ui.row().classes("w-full items-stretch").style("gap: 12px;"):
            with ui.column().style("width: 320px; gap: 10px; flex-shrink: 0;"):
                with ui.element("div").classes("poly-card w-full"):
                    ui.label("OUTCOMES").style(
                        f"font-size: 13px; font-weight: 700; color: {_TEXT}; "
                        "margin-bottom: 8px;"
                    )
                    with ui.row().classes("w-full").style("gap: 8px;"):
                        for key, title, color in (
                            ("finishes", "Finishes", _GREEN),
                            ("crashes", "Crashes", _RED),
                            ("off_tracks", "Off-track", _ORANGE),
                        ):
                            with ui.element("div").classes("poly-elevated").style(
                                "flex: 1; text-align: center; padding: 10px 6px;"
                            ):
                                ui.label(title).style(
                                    f"font-size: 11px; color: {_MUTED};"
                                )
                                lbl = ui.label("0").style(
                                    f"font-size: 16px; font-weight: 700; color: {color};"
                                )
                                self._outcome_values[key] = lbl

                with ui.element("div").classes("poly-card w-full").style(
                    "flex: 1;"
                ):
                    ui.label("LAST 5 EPISODES").style(
                        f"font-size: 13px; font-weight: 700; color: {_TEXT}; "
                        "margin-bottom: 8px;"
                    )
                    self._ep_labels = []
                    for _ in range(5):
                        lbl = ui.label("—").classes("poly-ep").style(
                            f"color: {_DIM};"
                        )
                        self._ep_labels.append(lbl)

            with ui.element("div").classes("poly-card").style(
                "flex: 1; min-width: 0; display: flex; flex-direction: column; gap: 8px;"
            ):
                with ui.row().classes("w-full items-center justify-between"):
                    ui.label("BEST RUNS").style(
                        f"font-size: 14px; font-weight: 700; color: {_TEXT};"
                    )
                    ui.label(
                        f"Double-click Watch · :{self._watch_port}"
                    ).style(f"font-size: 12px; color: {_DIM};")

                columns = [
                    {"name": "id", "label": "#", "field": "id", "align": "left"},
                    {"name": "tag", "label": "Tag", "field": "tag", "align": "left"},
                    {"name": "dist", "label": "Dist m", "field": "dist", "align": "left"},
                    {
                        "name": "reward",
                        "label": "Reward",
                        "field": "reward",
                        "align": "left",
                    },
                    {"name": "cps", "label": "CPs", "field": "cps", "align": "left"},
                    {
                        "name": "outcome",
                        "label": "Outcome",
                        "field": "outcome",
                        "align": "left",
                    },
                    {
                        "name": "times",
                        "label": "CP times",
                        "field": "times",
                        "align": "left",
                    },
                ]
                self._runs_table = (
                    ui.table(columns=columns, rows=[], row_key="key", selection="single")
                    .props("dense dark flat separator=horizontal")
                    .classes("w-full")
                    .style("max-height: 360px;")
                )
                self._runs_table.on("rowClick", self._on_row_click)
                self._runs_table.on("rowDblclick", self._on_row_dblclick)

                self._detail_lbl = ui.label(
                    "Select a run to inspect or Watch."
                ).style(
                    f"color: {_MUTED}; font-size: 12px; white-space: pre-wrap;"
                )

                with ui.row().classes("w-full items-center").style("gap: 10px;"):
                    ui.button(
                        "Watch selected run", on_click=self._watch_selected
                    ).props("unelevated no-caps").style(
                        f"background: {_LAVENDER} !important; color: {_BG} !important; "
                        "font-weight: 700; border-radius: 999px; padding: 8px 18px;"
                    )
                    ui.button("Stop watch", on_click=self._stop_watch_btn).props(
                        "outline no-caps color=primary"
                    ).style(
                        f"border-radius: 999px; padding: 8px 18px; color: {_LAVENDER_SOFT};"
                    )
                    self._watch_status = ui.label("").style(
                        f"color: {_LAVENDER_DIM}; font-size: 12px;"
                    )

    def _toggle_graph(self) -> None:
        self._graph_expanded = not self._graph_expanded
        if self._graph_body is not None:
            self._graph_body.set_visibility(self._graph_expanded)
        if self._graph_toggle is not None:
            self._graph_toggle.text = (
                "▼  Progress graph" if self._graph_expanded else "▶  Progress graph"
            )

    def _set_live_pill(self, on: bool, text: str) -> None:
        if self._live_pill is None:
            return
        self._live_pill.text = text
        fill = _GREEN if on else _DIM
        fg = _BG if on else _TEXT
        self._live_pill.style(
            f"background: {fill}; color: {fg}; font-weight: 700; "
            "padding: 6px 14px; letter-spacing: 0.04em;"
        )

    # ── Training start / stop ─────────────────────────────────────────
    def _build_train_cmd(self) -> list[str]:
        n = max(1, min(8, int(self.num_envs or 1)))
        total = max(1000, int(self.timesteps or 1000))
        cmd = [
            sys.executable,
            "-u",
            str(_ROOT / "run_local_training.py"),
            "--num-envs",
            str(n),
            "--total-timesteps",
            str(total),
        ]
        if not self.headless:
            cmd.append("--headed")
        elif self.watch_live:
            cmd.append("--watch")
        if self.dummy_vec:
            cmd.extend(["--vec-env", "dummy"])
        return cmd

    def _start_training(self) -> None:
        if self._train_proc is not None and self._train_proc.poll() is None:
            if self._train_status:
                self._train_status.text = "Training already running."
            return

        cmd = self._build_train_cmd()
        n = max(1, min(8, int(self.num_envs or 1)))
        env = {
            **os.environ,
            "POLYTRACK_FROM_GUI": "1",
            "PYTHONUTF8": "1",
            "PYTHONIOENCODING": "utf-8",
        }
        try:
            if sys.platform == "win32":
                self._train_proc = subprocess.Popen(
                    cmd,
                    cwd=str(_ROOT),
                    env=env,
                    creationflags=(
                        getattr(subprocess, "CREATE_NEW_CONSOLE", 0)
                        | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
                    ),
                )
                self._train_via_terminal = False
            elif sys.platform == "darwin":
                import shlex

                joined = " ".join(shlex.quote(c) for c in cmd)
                script = f"cd {shlex.quote(str(_ROOT))} && {joined}"
                osa = subprocess.Popen(
                    [
                        "osascript",
                        "-e",
                        f'tell application "Terminal" to do script {json.dumps(script)}',
                    ]
                )
                self._train_proc = osa
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

            mode = (
                "headed"
                if not self.headless
                else ("watch-0" if self.watch_live else "headless")
            )
            pid = self._train_proc.pid if self._train_proc else "?"
            if self._train_status:
                self._train_status.text = (
                    f"Training started in a terminal window · {n} envs · {mode} · pid {pid}"
                )
            self._set_live_pill(True, "TRAINING")
            if self._start_btn is not None:
                self._start_btn.disable()
        except Exception as exc:
            if self._train_status:
                self._train_status.text = f"Failed to start: {exc}"

    def _stop_training(self) -> None:
        proc = self._train_proc
        if self._train_status:
            self._train_status.text = "Stopping training…"
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
            if self._train_status:
                self._train_status.text = f"Stop error: {exc}"
            return
        self._train_proc = None
        self._train_via_terminal = False
        if self._start_btn is not None:
            self._start_btn.enable()
        self._set_live_pill(False, "STOPPED")
        if self._train_status:
            self._train_status.text = "Training stopped."

    def _check_train_proc(self) -> None:
        if self._train_proc is None:
            return
        if self._train_via_terminal:
            return
        code = self._train_proc.poll()
        if code is None:
            return
        if code == 0:
            if self._train_status:
                self._train_status.text = "Training finished."
            self._set_live_pill(False, "DONE")
        else:
            if self._train_status:
                self._train_status.text = (
                    f"Training exited (code {code}). Check the terminal window."
                )
            self._set_live_pill(False, "ERROR")
        self._train_proc = None
        if self._start_btn is not None:
            self._start_btn.enable()

    # ── Live metrics ──────────────────────────────────────────────────
    def _poll(self) -> None:
        self._refresh_metrics()
        self._refresh_runs()
        self._check_watch_proc()
        self._check_train_proc()

    def _refresh_metrics(self) -> None:
        try:
            data: dict = json.loads(self._json_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return

        training = self._train_proc is not None and self._train_proc.poll() is None
        if training:
            self._set_live_pill(True, "LIVE")

        ts = int(data.get("timesteps", 0))
        total = max(1, int(data.get("total_timesteps", 1)))
        frac = ts / total
        if self._progress_bar is not None:
            self._progress_bar.value = frac
        if self._progress_lbl is not None:
            self._progress_lbl.text = (
                f"Progress  {ts:,} / {total:,}  ({100 * frac:.1f}%)"
            )
        if self._uptime_lbl is not None:
            self._uptime_lbl.text = (
                f"Uptime {_fmt_hms(data.get('uptime_s', 0))}  ·  "
                f"episodes {int(data.get('episodes', 0))}"
            )

        best_fit = float(data.get("best_fitness", data.get("best_fitness_m", 0)))
        mean_fit = float(data.get("mean_fitness_10ep", 0))
        best_rew = float(data.get("best_reward", 0))

        self._metric_values["fitness"].text = f"{best_fit:.1f}"
        self._metric_subs["fitness"].text = "metres · all-time"
        self._metric_values["reward"].text = f"{best_rew:+.2f}"
        self._metric_subs["reward"].text = "episode reward · all-time"
        self._metric_values["mean"].text = f"{mean_fit:.1f}"
        self._metric_subs["mean"].text = "metres · last 10"
        self._metric_values["steps"].text = f"{ts // 1000}k" if ts >= 1000 else str(ts)
        self._metric_subs["steps"].text = f"of {total:,}"
        self._metric_values["fps"].text = f"{float(data.get('fps', 0)):.0f}"
        self._metric_subs["fps"].text = "env steps / s"

        self._outcome_values["finishes"].text = str(int(data.get("finishes", 0)))
        self._outcome_values["crashes"].text = str(int(data.get("crashes", 0)))
        self._outcome_values["off_tracks"].text = str(int(data.get("off_tracks", 0)))

        hist = data.get("history") or {}
        self._update_chart(hist)

        last5 = data.get("last5", [])
        for i, lbl in enumerate(self._ep_labels):
            if i < len(last5):
                e = last5[i]
                outcome = str(e.get("outcome", "?"))
                color = _OUTCOME_COLORS.get(outcome, _MUTED)
                fit = float(e.get("fitness", 0))
                lbl.text = (
                    f"ep {int(e['ep']):3d}  dist {fit:5.1f}m  "
                    f"r {float(e['reward']):+.1f}  "
                    f"cp {int(e['checkpoints'])}  {outcome}"
                )
                lbl.style(f"color: {color};")
            else:
                lbl.text = "—"
                lbl.style(f"color: {_DIM};")

    def _update_chart(self, hist: dict) -> None:
        if self._chart is None:
            return
        fit = [float(x) for x in (hist.get("mean_fitness") or [])]
        rew = [float(x) for x in (hist.get("mean_reward") or [])]
        ts = [int(x) for x in (hist.get("timesteps") or [])]
        labels = [f"{t:,}" for t in ts] if ts else [str(i) for i in range(len(fit))]
        opts = self._chart.options
        opts["xAxis"]["data"] = labels
        opts["series"][0]["data"] = fit
        opts["series"][1]["data"] = rew
        self._chart.update()

    def _refresh_runs(self) -> None:
        try:
            runs_raw = json.loads(self._runs_path.read_text(encoding="utf-8"))
            if not isinstance(runs_raw, list):
                return
            runs: list = runs_raw
        except (OSError, json.JSONDecodeError):
            return
        if runs == self._runs or self._runs_table is None:
            return
        self._runs = runs
        rows: list[dict[str, Any]] = []
        for r in runs:
            kind = r.get("kind", "?")
            tag = "★ ALL" if kind == "all_time" else f"g{r.get('generation', 0)}"
            rid = int(r.get("id", 0))
            dist = float(r.get("distance_m", r.get("fitness", 0)))
            rows.append(
                {
                    "key": rid,
                    "id": f"#{rid:03d}",
                    "tag": tag,
                    "dist": f"{dist:.0f}",
                    "reward": f"{float(r.get('reward', 0)):+.2f}",
                    "cps": int(r.get("checkpoints", 0)),
                    "outcome": str(r.get("outcome", "?")),
                    "times": _fmt_cp_times(r.get("checkpoint_times") or []),
                }
            )
        self._runs_table.rows = rows
        self._runs_table.update()

    # ── Best-run watch ────────────────────────────────────────────────
    def _on_row_click(self, e: Any) -> None:
        try:
            row = e.args[1]
            self._selected_run_id = int(row["key"])
        except (KeyError, TypeError, ValueError, IndexError):
            return
        self._show_run_detail()

    def _on_row_dblclick(self, e: Any) -> None:
        self._on_row_click(e)
        self._watch_selected()

    def _selected_run(self) -> dict | None:
        if self._selected_run_id is None:
            # Fall back to NiceGUI selection
            if self._runs_table is not None and self._runs_table.selected:
                try:
                    self._selected_run_id = int(self._runs_table.selected[0]["key"])
                except (KeyError, TypeError, ValueError, IndexError):
                    return None
            else:
                return None
        for r in self._runs:
            if int(r.get("id", -1)) == self._selected_run_id:
                return r
        return None

    def _show_run_detail(self) -> None:
        run = self._selected_run()
        if run is None or self._detail_lbl is None:
            return
        times = _fmt_cp_times(run.get("checkpoint_times") or [])
        dist = float(run.get("distance_m", run.get("fitness", 0)))
        self._detail_lbl.text = (
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
            if self._watch_status:
                self._watch_status.text = (
                    f"Starting game server on :{self._watch_port}…"
                )
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
                if self._watch_status:
                    self._watch_status.text = f"Server failed: {exc}"
                return False
        for _ in range(50):
            if _port_open(self._watch_port):
                return True
            if self._server_proc is not None and self._server_proc.poll() is not None:
                if self._watch_status:
                    self._watch_status.text = "start_server.py exited early"
                return False
            time.sleep(0.12)
        if self._watch_status:
            self._watch_status.text = "Timed out waiting for game server"
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
        if self._watch_status:
            self._watch_status.text = "Watch stopped."

    def _preflight_model(self, model_path: Path) -> str | None:
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
            if self._watch_status:
                self._watch_status.text = "Select a run first."
            return
        model_rel = str(run.get("model_path") or "")
        model_path = self._resolve_model(model_rel)
        if model_path is None:
            fallback = (_ROOT / "checkpoints" / "best_model.zip").resolve()
            if fallback.exists():
                model_path = fallback
            else:
                if self._watch_status:
                    self._watch_status.text = f"Model missing: {model_rel}"
                return
        err = self._preflight_model(model_path)
        if err:
            if self._watch_status:
                self._watch_status.text = err
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
            kw: dict = {
                "cwd": str(_ROOT),
                "stdout": log_f,
                "stderr": subprocess.STDOUT,
            }
            if sys.platform == "win32":
                kw["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            self._watch_proc = subprocess.Popen(cmd, **kw)
            if self._watch_status:
                self._watch_status.text = (
                    f"Watching #{run.get('id')} — Chromium opens after model load…"
                )
        except Exception as exc:
            if self._watch_status:
                self._watch_status.text = f"Failed: {exc}"

    def _check_watch_proc(self) -> None:
        if self._watch_proc is None:
            return
        if self._watch_proc.poll() is None:
            tip = self._tail_watch_log()
            if tip and not tip.startswith("$") and self._watch_status:
                self._watch_status.text = tip
            return
        code = self._watch_proc.poll()
        if self._watch_log_handle is not None:
            try:
                self._watch_log_handle.close()
            except OSError:
                pass
            self._watch_log_handle = None
        if self._watch_status:
            if code == 0:
                self._watch_status.text = "Watch finished."
            else:
                tip = self._tail_watch_log() or f"exit {code}"
                self._watch_status.text = f"Watch failed: {tip}"
        self._watch_proc = None

    def _on_close(self) -> None:
        if self._train_proc is not None and self._train_proc.poll() is None:
            self._stop_training()
        self._stop_watch()
        if self._server_proc is not None and self._server_proc.poll() is None:
            self._server_proc.terminate()


def main() -> None:
    global _CLI_ARGS
    p = argparse.ArgumentParser(description="Polyplex training control GUI.")
    p.add_argument("--json-path", type=Path, default=_DEFAULT_JSON)
    p.add_argument("--runs-path", type=Path, default=_DEFAULT_RUNS)
    p.add_argument("--poll-ms", type=int, default=1500)
    p.add_argument("--watch-port", type=int, default=_WATCH_PORT)
    p.add_argument("--port", type=int, default=8088, help="NiceGUI server port")
    p.add_argument(
        "--no-open",
        action="store_true",
        help="Do not auto-open the browser",
    )
    args = p.parse_args()
    _CLI_ARGS = args

    @ui.page("/")
    def _index() -> None:
        assert _CLI_ARGS is not None
        TrainingMonitorApp(
            _CLI_ARGS.json_path,
            _CLI_ARGS.runs_path,
            _CLI_ARGS.poll_ms,
            _CLI_ARGS.watch_port,
        )

    ui.run(
        title="Polyplex — Training Control",
        port=args.port,
        reload=False,
        show=not args.no_open,
        dark=True,
        favicon="🏁",
    )


if __name__ in {"__main__", "__mp_main__"}:
    main()
