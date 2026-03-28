"""Picklable ``SubprocVecEnv`` factories (required for Windows ``spawn`` workers)."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

import gymnasium as gym
from stable_baselines3.common.monitor import Monitor

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _ensure_project_on_path() -> None:
    if str(_PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT))


def make_polytrack_monitored_env(
    port: int,
    track_index: int,
    headless: bool,
    monitor_file: str,
) -> Callable[[], gym.Env]:
    """Return a no-arg callable that builds ``Monitor(PolytrackEnv(...))``."""

    def _init() -> gym.Env:
        _ensure_project_on_path()
        from env.polytrack_env import PolytrackEnv

        e = PolytrackEnv(
            port=port,
            headless=headless,
            track_menu_index=track_index,
        )
        return Monitor(e, filename=monitor_file)

    return _init
