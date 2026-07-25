#!/usr/bin/env python3
"""Tkinter GUI monitor for Polytrack RL training.

Reads:
  - logs/training_live.json  — live metrics (every ~1000 steps)
  - logs/best_runs.json      — elite / per-generation best runs

Double-click a best run (or select + Watch) to replay it with ``evaluate.py``.

Usage:
    python gui_monitor.py
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tkinter as tk
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
_DEFAULT_JSON = _ROOT / "logs" / "training_live.json"
_DEFAULT_RUNS = _ROOT / "logs" / "best_runs.json"

_MONO = ("Courier New", 10) if sys.platform == "win32" else ("Menlo", 11)
_HEADING = ("Helvetica", 12, "bold")

_OUTCOME_COLORS = {
    "finished": "#22c55e",
    "crashed": "#ef4444",
    "timeout": "#f59e0b",
}


def _fmt_hms(seconds: float) -> str:
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{sec:02d}"


def _progress_bar(frac: float, width: int = 20) -> str:
    f = min(1.0, max(0.0, frac))
    filled = int(round(width * f))
    return "█" * filled + "░" * (width - filled)


def _fmt_cp_times(times: list) -> str:
    if not times:
        return "—"
    return " → ".join(f"{float(t):.1f}s" for t in times)


class TrainingMonitorGUI:
    def __init__(
        self,
        root: tk.Tk,
        json_path: Path,
        runs_path: Path,
        poll_ms: int,
    ) -> None:
        self._root = root
        self._json_path = json_path
        self._runs_path = runs_path
        self._poll_ms = poll_ms
        self._runs: list[dict] = []
        self._watch_proc: subprocess.Popen | None = None

        root.title("Polytrack RL — Training Monitor")
        root.resizable(True, True)
        root.configure(bg="#1a1a1a")
        root.minsize(820, 520)

        self._build_ui()
        self._poll()

    def _label(
        self,
        parent: tk.Widget,
        text: str = "",
        *,
        font: tuple = _MONO,
        fg: str = "#e5e5e5",
        bg: str = "#1a1a1a",
        anchor: str = "w",
        **kw: object,
    ) -> tk.Label:
        return tk.Label(parent, text=text, font=font, fg=fg, bg=bg, anchor=anchor, **kw)

    def _build_ui(self) -> None:
        root = self._root
        pad = {"padx": 12, "pady": 4}

        hdr = tk.Frame(root, bg="#111")
        hdr.pack(fill="x", pady=(0, 2))
        tk.Label(
            hdr,
            text="POLYTRACK RL — TRAINING MONITOR",
            font=("Helvetica", 13, "bold"),
            fg="#ff6b00",
            bg="#111",
        ).pack(side="left", padx=12, pady=8)
        self._status_dot = tk.Label(hdr, text="●", font=("Helvetica", 14), fg="#555", bg="#111")
        self._status_dot.pack(side="right", padx=12)

        main = tk.Frame(root, bg="#1a1a1a")
        main.pack(fill="both", expand=True, padx=8, pady=4)

        # --- Left: live metrics ---
        left = tk.Frame(main, bg="#1a1a1a")
        left.pack(side="left", fill="both", expand=True)

        self._progress_var = tk.StringVar(value="Steps: —")
        self._label(left, font=_HEADING, textvariable=self._progress_var).pack(anchor="w", **pad)
        self._bar_var = tk.StringVar(value="")
        self._label(left, font=_MONO, textvariable=self._bar_var, fg="#888").pack(anchor="w", **pad)
        self._uptime_var = tk.StringVar(value="Uptime: —")
        self._label(left, textvariable=self._uptime_var, fg="#aaa").pack(anchor="w", **pad)
        self._fps_var = tk.StringVar(value="FPS: —")
        self._label(left, textvariable=self._fps_var, fg="#aaa").pack(anchor="w", **pad)

        tk.Frame(left, bg="#333", height=1).pack(fill="x", padx=8, pady=6)

        self._label(left, "FITNESS (time-to-checkpoint)", font=_HEADING).pack(anchor="w", **pad)
        self._best_fit_var = tk.StringVar(value="Best fitness: —")
        self._label(left, textvariable=self._best_fit_var, fg="#22c55e").pack(anchor="w", **pad)
        self._mean_fit_var = tk.StringVar(value="Mean fitness: —")
        self._label(left, textvariable=self._mean_fit_var).pack(anchor="w", **pad)
        self._best_var = tk.StringVar(value="Best reward (last 10): —")
        self._label(left, textvariable=self._best_var, fg="#aaa").pack(anchor="w", **pad)

        tk.Frame(left, bg="#333", height=1).pack(fill="x", padx=8, pady=6)

        self._label(left, "OUTCOMES", font=_HEADING).pack(anchor="w", **pad)
        self._crashes_var = tk.StringVar(value="Crashes:  —")
        self._label(left, textvariable=self._crashes_var, fg="#ef4444").pack(anchor="w", **pad)
        self._finishes_var = tk.StringVar(value="Finishes: —")
        self._label(left, textvariable=self._finishes_var, fg="#22c55e").pack(anchor="w", **pad)
        self._episodes_var = tk.StringVar(value="Total eps: —")
        self._label(left, textvariable=self._episodes_var).pack(anchor="w", **pad)

        self._label(left, "LAST 5 EPISODES", font=_HEADING).pack(anchor="w", padx=12, pady=(12, 4))
        self._ep_labels: list[tk.Label] = []
        for _ in range(5):
            lbl = self._label(left, text="—", font=_MONO, fg="#888")
            lbl.pack(anchor="w", padx=12, pady=1)
            self._ep_labels.append(lbl)

        # --- Right: best runs browser ---
        right = tk.Frame(main, bg="#242424")
        right.pack(side="right", fill="both", expand=True, padx=(8, 0))

        self._label(right, "BEST RUNS (each generation / all-time)", font=_HEADING, bg="#242424").pack(
            anchor="w", padx=12, pady=(8, 4)
        )
        self._label(
            right,
            "Double-click or select + Watch to replay in a headed browser.",
            font=("Helvetica", 9),
            fg="#666",
            bg="#242424",
        ).pack(anchor="w", padx=12, pady=(0, 6))

        list_frame = tk.Frame(right, bg="#242424")
        list_frame.pack(fill="both", expand=True, padx=12, pady=4)

        scroll = tk.Scrollbar(list_frame)
        scroll.pack(side="right", fill="y")
        self._runs_list = tk.Listbox(
            list_frame,
            font=_MONO,
            bg="#1a1a1a",
            fg="#e5e5e5",
            selectbackground="#ff6b00",
            selectforeground="#0d0d0d",
            activestyle="none",
            highlightthickness=0,
            borderwidth=0,
            yscrollcommand=scroll.set,
            height=14,
        )
        self._runs_list.pack(side="left", fill="both", expand=True)
        scroll.config(command=self._runs_list.yview)
        self._runs_list.bind("<Double-Button-1>", lambda _e: self._watch_selected())
        self._runs_list.bind("<<ListboxSelect>>", lambda _e: self._show_run_detail())

        self._detail_var = tk.StringVar(value="Select a run to see checkpoint times.")
        self._label(
            right,
            textvariable=self._detail_var,
            font=_MONO,
            fg="#aaa",
            bg="#242424",
            justify="left",
            wraplength=360,
        ).pack(anchor="w", padx=12, pady=6)

        btn_row = tk.Frame(right, bg="#242424")
        btn_row.pack(fill="x", padx=12, pady=(4, 12))
        watch_btn = tk.Button(
            btn_row,
            text="Watch selected run",
            command=self._watch_selected,
            bg="#ff6b00",
            fg="#0d0d0d",
            activebackground="#ff8533",
            activeforeground="#0d0d0d",
            font=("Helvetica", 11, "bold"),
            relief="flat",
            padx=16,
            pady=8,
            cursor="hand2",
        )
        watch_btn.pack(side="left")
        self._watch_status = tk.StringVar(value="")
        self._label(
            btn_row, textvariable=self._watch_status, fg="#888", bg="#242424"
        ).pack(side="left", padx=12)

    def _poll(self) -> None:
        self._refresh_metrics()
        self._refresh_runs()
        self._root.after(self._poll_ms, self._poll)

    def _refresh_metrics(self) -> None:
        try:
            data: dict = json.loads(self._json_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            self._status_dot.config(fg="#555", text="●")
            return

        self._status_dot.config(fg="#22c55e", text="●")
        ts = data.get("timesteps", 0)
        total = data.get("total_timesteps", 1)
        frac = ts / total if total > 0 else 0.0
        self._progress_var.set(f"Steps: {ts:,} / {total:,}  ({100 * frac:.1f}%)")
        self._bar_var.set(_progress_bar(frac))
        self._uptime_var.set(f"Uptime: {_fmt_hms(data.get('uptime_s', 0))}")
        self._fps_var.set(f"FPS: {data.get('fps', 0):.1f} steps/s")

        # Keys kept for backward compat (old "best_fitness_m" or new "best_fitness").
        best_fit = data.get("best_fitness", data.get("best_fitness_m", 0))
        mean_fit = data.get("mean_fitness_10ep", 0)
        self._best_fit_var.set(f"Best fitness:  {best_fit:.1f}  (higher = faster CPs)")
        self._mean_fit_var.set(f"Mean fitness:  {mean_fit:.1f}")
        self._best_var.set(f"Best reward (last 10):  {data.get('best_reward', 0):+.2f}")
        self._crashes_var.set(f"Crashes:  {data.get('crashes', 0)}")
        self._finishes_var.set(f"Finishes: {data.get('finishes', 0)}")
        self._episodes_var.set(f"Total eps: {data.get('episodes', 0)}")

        last5 = data.get("last5", [])
        for i, lbl in enumerate(self._ep_labels):
            if i < len(last5):
                e = last5[i]
                outcome = e.get("outcome", "?")
                color = _OUTCOME_COLORS.get(outcome, "#aaa")
                fit = e.get("fitness", 0)
                lbl.config(
                    text=(
                        f"ep {e['ep']:3d}  fit:{fit:5.1f}  "
                        f"r:{e['reward']:+.1f}  cp:{e['checkpoints']}  "
                        f"{outcome[:8]}"
                    ),
                    fg=color,
                )
            else:
                lbl.config(text="—", fg="#555")

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
        self._runs = runs
        sel = self._runs_list.curselection()
        self._runs_list.delete(0, tk.END)
        for r in runs:
            kind = r.get("kind", "?")
            tag = "★" if kind == "all_time" else f"g{r.get('generation', 0)}"
            times = _fmt_cp_times(r.get("checkpoint_times") or [])
            line = (
                f"#{r.get('id', 0):03d} {tag:4s}  fit={r.get('fitness', 0):5.1f}  "
                f"cp={r.get('checkpoints', 0)}  times[{times}]"
            )
            self._runs_list.insert(tk.END, line)
        if sel:
            idx = sel[0]
            if idx < len(runs):
                self._runs_list.selection_set(idx)

    def _selected_run(self) -> dict | None:
        sel = self._runs_list.curselection()
        if not sel:
            return None
        idx = int(sel[0])
        if idx < 0 or idx >= len(self._runs):
            return None
        return self._runs[idx]

    def _show_run_detail(self) -> None:
        run = self._selected_run()
        if run is None:
            return
        times = _fmt_cp_times(run.get("checkpoint_times") or [])
        self._detail_var.set(
            f"#{run.get('id')}  gen={run.get('generation')}  kind={run.get('kind')}\n"
            f"fitness={run.get('fitness')}  reward={run.get('reward'):+.2f}  "
            f"outcome={run.get('outcome')}\n"
            f"checkpoint times: {times}\n"
            f"model: {run.get('model_path')}"
        )

    def _watch_selected(self) -> None:
        run = self._selected_run()
        if run is None:
            self._watch_status.set("Select a run first.")
            return
        model_rel = run.get("model_path") or ""
        model_path = (_ROOT / model_rel).resolve()
        if not model_path.exists():
            # Try without assuming cwd; elites may be stored relative.
            alt = _ROOT / "checkpoints" / "elites" / Path(model_rel).name
            if alt.exists():
                model_path = alt
            else:
                self._watch_status.set(f"Model missing: {model_rel}")
                return

        if self._watch_proc is not None and self._watch_proc.poll() is None:
            self._watch_status.set("Already watching — close that window first.")
            return

        track = int(run.get("track_index", 0))
        cmd = [
            sys.executable,
            str(_ROOT / "evaluate.py"),
            "--model",
            str(model_path),
            "--track-index",
            str(track),
            "--episodes",
            "1",
        ]
        try:
            self._watch_proc = subprocess.Popen(
                cmd,
                cwd=str(_ROOT),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            self._watch_status.set(f"Watching #{run.get('id')} (pid {self._watch_proc.pid})…")
        except Exception as exc:
            self._watch_status.set(f"Failed: {exc}")


def main() -> None:
    p = argparse.ArgumentParser(description="Tkinter GUI monitor for Polytrack RL training.")
    p.add_argument("--json-path", type=Path, default=_DEFAULT_JSON)
    p.add_argument("--runs-path", type=Path, default=_DEFAULT_RUNS)
    p.add_argument("--poll-ms", type=int, default=2000)
    args = p.parse_args()

    root = tk.Tk()
    TrainingMonitorGUI(root, args.json_path, args.runs_path, args.poll_ms)
    root.mainloop()


if __name__ == "__main__":
    main()
