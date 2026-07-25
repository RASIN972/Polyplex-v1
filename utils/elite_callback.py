"""Elite fitness selection for PPO training (time-to-checkpoint fitness).

Fitness = sum over reached checkpoints of ``(30 - arrival_time_s)``.
Faster arrivals score higher; more checkpoints also score higher.

On each new all-time best (and at the end of each generation), the policy is
snapshotted under ``checkpoints/elites/`` and appended to
``logs/best_runs.json`` so the GUI can list and replay every best run.
"""

from __future__ import annotations

import json
import shutil
import time
from collections import deque
from pathlib import Path
from typing import Any

from stable_baselines3.common.callbacks import BaseCallback

_ROOT = Path(__file__).resolve().parent.parent
_BEST_RUNS_PATH = Path(
    __import__("os").environ.get("POLYTRACK_BEST_RUNS", str(_ROOT / "logs" / "best_runs.json"))
)


class EliteFitnessCallback(BaseCallback):
    """Save elites by time-to-checkpoint fitness; log runs for the GUI."""

    def __init__(
        self,
        save_path: str | Path,
        *,
        track_index: int = 0,
        verbose: int = 1,
    ) -> None:
        super().__init__(verbose)
        self._save_dir = Path(save_path)
        self._elites_dir = self._save_dir / "elites"
        self._track_index = int(track_index)
        self._best_fitness = float("-inf")
        self._elite_count = 0
        self._recent_fitness: deque[float] = deque(maxlen=20)
        self._generation = 0
        self._gen_best: float = float("-inf")
        self._gen_worst: float = float("inf")
        self._gen_ep_count = 0
        self._gen_best_info: dict[str, Any] | None = None
        # One "generation" ≈ a batch of finished episodes.
        self._gen_episode_budget = 16
        self._runs: list[dict[str, Any]] = []

    def _on_training_start(self) -> None:
        self._save_dir.mkdir(parents=True, exist_ok=True)
        self._elites_dir.mkdir(parents=True, exist_ok=True)
        _BEST_RUNS_PATH.parent.mkdir(parents=True, exist_ok=True)
        # Start a fresh session log (keep a backup of the previous file if present).
        if _BEST_RUNS_PATH.exists():
            bak = _BEST_RUNS_PATH.with_suffix(".prev.json")
            try:
                shutil.copy2(_BEST_RUNS_PATH, bak)
            except OSError:
                pass
        self._runs = []
        self._write_runs()

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            ep = info.get("episode")
            if ep is None:
                continue

            fitness = float(info.get("fitness", 0.0))
            reward = float(ep.get("r", 0.0))
            cp_times = info.get("checkpoint_times") or []
            if not isinstance(cp_times, list):
                cp_times = []
            cps = int(info.get("checkpoints", len(cp_times)))
            outcome = str(info.get("outcome", "crashed"))

            ep_info = {
                "fitness": fitness,
                "reward": reward,
                "checkpoint_times": [float(t) for t in cp_times],
                "checkpoints": cps,
                "outcome": outcome,
                "steps": int(ep.get("l", 0)),
            }

            self._recent_fitness.append(fitness)
            self._gen_ep_count += 1
            if fitness >= self._gen_best:
                self._gen_best = fitness
                self._gen_best_info = dict(ep_info)
            if fitness < self._gen_worst:
                self._gen_worst = fitness

            if fitness > self._best_fitness:
                self._best_fitness = fitness
                self._elite_count += 1
                self._save_elite(ep_info, kind="all_time")

            if self._gen_ep_count >= self._gen_episode_budget:
                self._end_generation()

        return True

    def _end_generation(self) -> None:
        self._generation += 1
        if self._gen_best_info is not None and self._gen_best > float("-inf"):
            # Snapshot policy at generation boundary as this iteration's best run.
            self._save_elite(self._gen_best_info, kind="generation")

        if self.verbose:
            mean_f = (
                sum(self._recent_fitness) / len(self._recent_fitness)
                if self._recent_fitness
                else 0.0
            )
            times = (
                self._gen_best_info.get("checkpoint_times", [])
                if self._gen_best_info
                else []
            )
            print(
                f"[elite] gen {self._generation}: "
                f"best_fit={self._gen_best:.1f}  worst={self._gen_worst:.1f}  "
                f"mean20={mean_f:.1f}  all-time={self._best_fitness:.1f}  "
                f"best_cp_times={times}",
                flush=True,
            )
        self._gen_ep_count = 0
        self._gen_best = float("-inf")
        self._gen_worst = float("inf")
        self._gen_best_info = None

    def _save_elite(self, ep_info: dict[str, Any], *, kind: str) -> None:
        assert self.model is not None
        fitness = float(ep_info["fitness"])
        run_id = len(self._runs) + 1
        stamp = time.strftime("%Y%m%d_%H%M%S")
        slug = f"elite_{run_id:03d}_gen{self._generation}_{kind}_fit{fitness:.1f}_{stamp}"
        model_rel = Path("checkpoints") / "elites" / slug
        model_abs = self._elites_dir / slug
        self.model.save(str(model_abs))

        # Always refresh the global best / latest pointers on all-time highs.
        if kind == "all_time":
            best_path = self._save_dir / "best_model"
            self.model.save(str(best_path))
            latest_path = self._save_dir / "latest"
            self.model.save(str(latest_path))
            meta = {
                "best_fitness": round(fitness, 2),
                "checkpoint_times": ep_info.get("checkpoint_times", []),
                "checkpoints": ep_info.get("checkpoints", 0),
                "best_episode_reward": round(float(ep_info["reward"]), 4),
                "total_timesteps": int(self.num_timesteps),
                "elite_saves": self._elite_count,
                "track_index": self._track_index,
                "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "generation": self._generation,
                "model_path": str(model_rel) + ".zip",
            }
            (self._save_dir / "best_model.json").write_text(
                json.dumps(meta, indent=2), encoding="utf-8"
            )

        run = {
            "id": run_id,
            "kind": kind,
            "generation": self._generation,
            "fitness": round(fitness, 2),
            "checkpoint_times": ep_info.get("checkpoint_times", []),
            "checkpoints": int(ep_info.get("checkpoints", 0)),
            "reward": round(float(ep_info["reward"]), 4),
            "outcome": ep_info.get("outcome", "?"),
            "steps": int(ep_info.get("steps", 0)),
            "timesteps": int(self.num_timesteps),
            "track_index": self._track_index,
            "model_path": str(model_rel) + ".zip",
            "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        self._runs.append(run)
        self._write_runs()

        if self.verbose:
            label = "NEW BEST" if kind == "all_time" else f"gen-{self._generation} peak"
            print(
                f"[elite] {label} fitness={fitness:.1f} "
                f"cp_times={ep_info.get('checkpoint_times', [])} "
                f"→ {model_abs}.zip",
                flush=True,
            )

    def _write_runs(self) -> None:
        try:
            tmp = _BEST_RUNS_PATH.with_suffix(".tmp")
            tmp.write_text(json.dumps(self._runs, indent=2), encoding="utf-8")
            tmp.replace(_BEST_RUNS_PATH)
        except OSError:
            pass
