from __future__ import annotations

import json
import os
import time
from collections import deque
from pathlib import Path

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

_LIVE_JSON_PATH = Path(os.environ.get("POLYTRACK_LIVE_JSON", "logs/training_live.json"))

_SPARK = "▁▂▃▄▅▆▇█"
_BAR_W = 15


def _fmt_hms(seconds: float) -> str:
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{sec:02d}"


def _sparkline(values: list[float]) -> str:
    if not values:
        return ""
    vmin = min(values)
    vmax = max(values)
    if abs(vmax - vmin) < 1e-12:
        return _SPARK[3] * len(values)
    out: list[str] = []
    for v in values:
        t = (v - vmin) / (vmax - vmin)
        idx = min(7, max(0, int(t * 7.999)))
        out.append(_SPARK[idx])
    return "".join(out)


def _progress_bar(fraction: float, width: int = _BAR_W) -> str:
    f = min(1.0, max(0.0, fraction))
    filled = int(round(width * f))
    filled = min(width, max(0, filled))
    return "█" * filled + "░" * (width - filled)


def _outcome_label(outcome: str) -> str:
    if outcome == "finished":
        return "FINISHED"
    if outcome == "timeout":
        return "TIMEOUT"
    if outcome == "off_track":
        return "OFF-TRACK"
    return "CRASHED"


class TrainingMonitor(BaseCallback):
    def __init__(self, total_timesteps: int) -> None:
        super().__init__()
        self._total = max(1, int(total_timesteps))
        self._t0 = 0.0
        self._ep_num = 0
        self._last_rewards: deque[float] = deque(maxlen=10)
        self._last_fitness: deque[float] = deque(maxlen=10)
        self._episodes: list[
            tuple[int, int, float, int, str, float, float]
        ] = []  # ep, steps, rew, cp, outcome, mean_sp, fitness
        self._speed_samples: list[float] = []
        self._fps_t: float = 0.0
        self._fps_steps: int = 0
        self._best_fitness = 0.0
        self._best_reward = float("-inf")
        # Rolling series for the GUI progress graph (sampled every live-json write).
        self._hist_ts: list[int] = []
        self._hist_fit: list[float] = []
        self._hist_rew: list[float] = []
        self._hist_best_fit: list[float] = []
        self._hist_max = 200

    @classmethod
    def show_bootstrap(cls, total_timesteps: int) -> None:
        ghost = object.__new__(cls)
        ghost._total = max(1, int(total_timesteps))
        ghost._t0 = time.perf_counter()
        ghost.num_timesteps = 0
        ghost._ep_num = 0
        ghost._last_rewards = deque(maxlen=10)
        ghost._last_fitness = deque(maxlen=10)
        ghost._episodes = []
        ghost._speed_samples = []
        ghost._fps_t = 0.0
        ghost._fps_steps = 0
        ghost._best_fitness = 0.0
        ghost._best_reward = float("-inf")
        ghost._hist_ts = []
        ghost._hist_fit = []
        ghost._hist_rew = []
        ghost._hist_best_fit = []
        ghost._hist_max = 200
        cls._draw(ghost)

    def _on_training_start(self) -> None:
        self._t0 = time.perf_counter()
        self._fps_t = self._t0
        self._fps_steps = 0
        self._draw()

    def _on_step(self) -> bool:
        model = self.model
        obs = getattr(model, "_last_obs", None) if model is not None else None
        if obs is not None and isinstance(obs, np.ndarray) and obs.size > 0:
            if obs.ndim > 1:
                self._speed_samples.append(float(np.mean(obs[:, 0])))
            else:
                self._speed_samples.append(float(obs[0]))

        for info in self.locals.get("infos", []):
            ep = info.get("episode")
            if ep is None:
                continue
            self._ep_num += 1
            rew = float(ep["r"])
            length = int(ep["l"])
            outcome = str(info.get("outcome", "crashed"))
            cps = int(info.get("checkpoints", 0))
            fitness = float(info.get("fitness", 0.0))
            mean_sp = (
                float(np.mean(self._speed_samples)) if self._speed_samples else 0.0
            )
            self._speed_samples.clear()
            self._last_rewards.append(rew)
            self._last_fitness.append(fitness)
            if fitness > self._best_fitness:
                self._best_fitness = fitness
            if rew > self._best_reward:
                self._best_reward = rew
            self._episodes.append(
                (self._ep_num, length, rew, cps, outcome, mean_sp, fitness)
            )

        self._fps_steps += 1
        if self.num_timesteps > 0 and self.num_timesteps % 1000 == 0:
            self._draw()
            self._write_live_json()
        return True

    def _write_live_json(self) -> None:
        """Write a small JSON snapshot to logs/training_live.json for gui_monitor.py."""
        now = time.perf_counter()
        elapsed = now - self._fps_t
        fps = self._fps_steps / elapsed if elapsed > 0.5 else 0.0
        self._fps_t = now
        self._fps_steps = 0

        spark_vals = list(self._last_rewards)
        fit_vals = list(self._last_fitness)
        best_reward_10 = max(spark_vals) if spark_vals else 0.0
        mean_reward = float(np.mean(spark_vals)) if spark_vals else 0.0
        mean_fitness = float(np.mean(fit_vals)) if fit_vals else 0.0
        best_reward_all = (
            self._best_reward if self._best_reward > float("-inf") else 0.0
        )
        n_ep = len(self._episodes)
        n_crash = sum(1 for e in self._episodes if e[4] == "crashed")
        n_fin = sum(1 for e in self._episodes if e[4] == "finished")
        n_off = sum(1 for e in self._episodes if e[4] == "off_track")
        last5 = [
            {
                "ep": e[0],
                "steps": e[1],
                "reward": round(e[2], 2),
                "checkpoints": e[3],
                "outcome": e[4],
                "fitness": round(e[6], 1),
            }
            for e in self._episodes[-5:]
        ]
        self._hist_ts.append(int(self.num_timesteps))
        self._hist_fit.append(round(mean_fitness, 2))
        self._hist_rew.append(round(mean_reward, 3))
        self._hist_best_fit.append(round(self._best_fitness, 2))
        if len(self._hist_ts) > self._hist_max:
            self._hist_ts = self._hist_ts[-self._hist_max :]
            self._hist_fit = self._hist_fit[-self._hist_max :]
            self._hist_rew = self._hist_rew[-self._hist_max :]
            self._hist_best_fit = self._hist_best_fit[-self._hist_max :]

        data = {
            "timesteps": self.num_timesteps,
            "total_timesteps": self._total,
            "episodes": n_ep,
            "uptime_s": round(now - self._t0, 1),
            "fps": round(fps, 1),
            "best_reward": round(best_reward_all, 2),
            "best_reward_10ep": round(best_reward_10, 2),
            "mean_reward_10ep": round(mean_reward, 2),
            "best_fitness": round(self._best_fitness, 1),
            "best_fitness_m": round(self._best_fitness, 1),  # legacy / distance metres
            "mean_fitness_10ep": round(mean_fitness, 1),
            "crashes": n_crash,
            "finishes": n_fin,
            "off_tracks": n_off,
            "last5": last5,
            "history": {
                "timesteps": self._hist_ts,
                "mean_fitness": self._hist_fit,
                "mean_reward": self._hist_rew,
                "best_fitness": self._hist_best_fit,
            },
        }
        try:
            _LIVE_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
            tmp = _LIVE_JSON_PATH.with_suffix(".tmp")
            tmp.write_text(json.dumps(data), encoding="utf-8")
            tmp.replace(_LIVE_JSON_PATH)
        except OSError:
            pass

    def _draw(self) -> None:
        print("\033[2J\033[H", end="")
        n_ep = len(self._episodes)
        uptime = time.perf_counter() - self._t0
        frac = self.num_timesteps / self._total
        pct = 100.0 * frac
        bar = _progress_bar(frac)
        spark_vals = list(self._last_rewards)
        spark = _sparkline(spark_vals)

        print("=" * 60)
        print("  POLYTRACK RL — TRAINING MONITOR")
        print("=" * 60)
        if self.num_timesteps == 0:
            print("  (starting — first env reset / rollout may take a minute)")
            print()
        print(
            f"  Steps:          {self.num_timesteps:,} / {self._total:,}    "
            f"[{bar}] {pct:.1f}%"
        )
        print(f"  Episodes:       {n_ep}")
        print(f"  Uptime:         {_fmt_hms(uptime)}")
        print()

        fit_vals = list(self._last_fitness)
        if fit_vals:
            print("  FITNESS — horizontal distance metres (last 10 episodes)")
            print(
                f"  best:  {max(fit_vals):.1f}m   worst: {min(fit_vals):.1f}m   "
                f"mean: {float(np.mean(fit_vals)):.1f}m   "
                f"all-time: {self._best_fitness:.1f}m"
            )
            print(f"  trend: {_sparkline(fit_vals)}   ← spark line")
            print("  (higher = farther along the track)")
        else:
            print("  FITNESS — horizontal distance metres (last 10 episodes)")
            print("  (no episodes yet)")
        print()

        if spark_vals:
            best = max(spark_vals)
            worst = min(spark_vals)
            mean_r = float(np.mean(spark_vals))
            all_r = (
                self._best_reward if self._best_reward > float("-inf") else 0.0
            )
            print("  REWARD (last 10 episodes)")
            print(
                f"  best:  {best:+.2f}   worst: {worst:+.2f}   "
                f"mean: {mean_r:+.2f}   all-time: {all_r:+.2f}"
            )
            print(f"  trend: {spark}   ← spark line")
        else:
            print("  REWARD (last 10 episodes)")
            print("  (no episodes yet)")
            print("  trend:   ← spark line")
        print()

        lengths = [e[1] for e in self._episodes]
        if lengths:
            print("  EPISODE LENGTH (steps)")
            print(
                f"  mean: {int(round(np.mean(lengths)))}   "
                f"max: {max(lengths)}   min: {min(lengths)}"
            )
        else:
            print("  EPISODE LENGTH (steps)")
            print("  mean: —   max: —   min: —")
        print()

        if self._episodes:
            mean_sp_all = float(
                np.mean([e[5] for e in self._episodes])
            )
            mean_cp = float(np.mean([e[3] for e in self._episodes]))
            n_crash = sum(1 for e in self._episodes if e[4] == "crashed")
            n_fin = sum(1 for e in self._episodes if e[4] == "finished")
            crash_pct = 100.0 * n_crash / n_ep
            fin_pct = 100.0 * n_fin / n_ep
        else:
            mean_sp_all = 0.0
            mean_cp = 0.0
            crash_pct = 0.0
            fin_pct = 0.0

        print("  AGENT BEHAVIOR")
        print(f"  avg speed score:    {mean_sp_all:.2f}")
        print(f"  checkpoints/ep:     {mean_cp:.1f}")
        print(f"  crash rate:         {crash_pct:.0f}%")
        print(f"  finish rate:        {fin_pct:.0f}%")
        print()

        print("  LAST 5 EPISODES")
        tail = self._episodes[-5:]
        for ep, ln, rw, cps, oc, _, fit in tail:
            tag = _outcome_label(oc)
            print(
                f"  ep {ep:3d}  |  fit: {fit:6.1f}  |  steps: {ln:4d}  |  "
                f"reward: {rw:+.1f}  |  cp: {cps}  |  {tag}"
            )
        if not tail:
            print("  (none yet)")
        print()
        print("=" * 60, flush=True)
