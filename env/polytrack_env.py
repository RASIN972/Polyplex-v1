from __future__ import annotations

import asyncio
import os
import time
from typing import Any, SupportsFloat, cast

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from env.game_bridge import GameBridge

ACTION_MAP: dict[int, list[str]] = {
    0: [],
    1: ["w"],
    2: ["s"],
    3: ["a"],
    4: ["d"],
    5: ["w", "a"],
    6: ["w", "d"],
    7: ["s", "a"],
    8: ["s", "d"],
}

STEP_WAIT_S = 0.05
MAX_TRACK_LENGTH = 2000.0
READY_TIMEOUT_S = float(os.environ.get("POLYTRACK_READY_TIMEOUT_S", "300"))
EPISODE_TIME_LIMIT_S = 60.0
MAX_CRASHES = 3
PI_F = float(np.pi)
DEFAULT_TRACK_MENU_INDEX = 0


def _polytrack_debug_chain() -> bool:
    v = os.environ.get("POLYTRACK_DEBUG_CHAIN", "").strip().lower()
    return v in ("1", "true", "yes")


def _polytrack_debug_max_steps() -> int:
    try:
        return max(0, int(os.environ.get("POLYTRACK_DEBUG_MAX_STEPS", "10")))
    except ValueError:
        return 10


def _debug_state_line(s: dict[str, Any]) -> str:
    if s.get("error"):
        return f"error={s.get('error')!r}"
    return (
        f"speed={s.get('speed')!r} has_started={s.get('has_started')!r} "
        f"car_present={s.get('car_present')!r} cp={s.get('checkpoint_index')!r} "
        f"te={s.get('time_elapsed')!r} crashed_or_reset={s.get('crashed_or_reset')!r}"
    )


class PolytrackEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        port: int,
        *,
        headless: bool = True,
        track_menu_index: int = DEFAULT_TRACK_MENU_INDEX,
    ) -> None:
        super().__init__()
        self._port = int(port)
        self._url = f"http://127.0.0.1:{self._port}/"
        self._headless = headless
        self._track_menu_index = track_menu_index
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._bridge: GameBridge | None = None

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(13,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(9)

        self._last_pos: np.ndarray | None = None
        self._last_euler: np.ndarray | None = None
        self._last_cp: int = 0
        self._cp_game_time: float = 0.0
        self._crash_count: int = 0
        self._checkpoint_hits: int = 0
        self._dbg_step_count: int = 0
        self._dbg_done_count: int = 0
        self._chain_step_seq: int = 0

    def _finalize_obs(self, arr: np.ndarray) -> np.ndarray:
        o = np.ascontiguousarray(arr, dtype=np.float32)
        if o.shape != (13,):
            raise ValueError(f"PolytrackEnv: obs must be shape (13,), got {o.shape} dtype={o.dtype}")
        return o

    def _run(self, coro: Any) -> Any:
        return self._loop.run_until_complete(coro)

    async def _wait_for_game_ready(self) -> None:
        """Wait until vehicle exists and RL harness is valid (no auto-throttle)."""
        deadline = time.monotonic() + READY_TIMEOUT_S
        last_nudge_m = 0.0
        while time.monotonic() < deadline:
            assert self._bridge is not None
            s = await self._bridge.get_state()
            if s.get("error"):
                await asyncio.sleep(0.1)
                continue
            if s.get("car_present") and s.get("ready"):
                return
            now_m = time.monotonic()
            if now_m - last_nudge_m >= 2.5:
                await self._bridge.nudge_race_start()
                last_nudge_m = now_m
            await asyncio.sleep(0.1)
        raise TimeoutError("PolytrackEnv: _wait_for_game_ready timed out")

    def _obs_from_state(
        self,
        s: dict[str, Any],
        dt: float,
    ) -> np.ndarray:
        pos = np.array(
            [
                float(s["position"]["x"]),
                float(s["position"]["y"]),
                float(s["position"]["z"]),
            ],
            dtype=np.float32,
        )
        euler = np.array(
            [
                float(s["rotation"]["x"]),
                float(s["rotation"]["y"]),
                float(s["rotation"]["z"]),
            ],
            dtype=np.float32,
        )
        if self._last_pos is None or self._last_euler is None or dt <= 0:
            vel = np.zeros(3, dtype=np.float32)
            omega = np.zeros(3, dtype=np.float32)
        else:
            vel = ((pos - self._last_pos) / dt).astype(np.float32)
            omega = ((euler - self._last_euler) / dt).astype(np.float32)

        self._last_pos = pos.copy()
        self._last_euler = euler.copy()

        speed = float(s.get("speed") or 0.0)
        sp = speed / 200.0
        obs = np.zeros(13, dtype=np.float32)
        obs[0] = np.float32(sp)
        obs[1:4] = vel / 100.0
        obs[4:7] = euler / PI_F
        obs[7:10] = omega / 10.0
        obs[10] = 0.0
        obs[11] = 0.0
        te = float(s.get("time_elapsed") or 0.0)
        tscp = te - self._cp_game_time
        obs[12] = np.float32(max(0.0, tscp) / 30.0)
        return self._finalize_obs(obs)

    def _reward(
        self,
        s: dict[str, Any],
        cp_prev: int,
        cp_now: int,
        crashed: bool,
    ) -> float:
        speed = float(s.get("speed") or 0.0)
        r = 0.0
        r += (speed / 200.0) * 0.01
        if cp_now > cp_prev:
            r += 2.0 * float(cp_now - cp_prev)
        if crashed:
            r -= 1.0
        r -= 0.001
        return r

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        dbg = _polytrack_debug_chain()
        if dbg:
            print(">>> RESET CALLED", flush=True)

        async def _go() -> dict[str, Any]:
            if self._bridge is None:
                if dbg:
                    print(">>> RESET _go: GameBridge.launch ...", flush=True)
                self._bridge = await GameBridge.launch(
                    self._url, headless=self._headless
                )
                if dbg:
                    print(">>> RESET _go: launch done", flush=True)
            else:
                if dbg:
                    print(">>> RESET _go: restart_session ...", flush=True)
                await self._bridge.restart_session(
                    self._url, headless=self._headless
                )
                if dbg:
                    print(">>> RESET _go: restart_session done", flush=True)
            assert self._bridge is not None
            if dbg:
                print(">>> RESET _go: start_track_menu_index ...", flush=True)
            await self._bridge.start_track_menu_index(self._track_menu_index)
            if dbg:
                print(
                    ">>> RESET _go: start_track_menu_index done, _wait_for_game_ready ...",
                    flush=True,
                )
            await self._wait_for_game_ready()
            if dbg:
                print(">>> RESET _go: _wait_for_game_ready done, get_state ...", flush=True)
            return await self._bridge.get_state()

        if dbg:
            print(">>> RESET: entering event loop (_run _go)", flush=True)
        s0 = self._run(_go())
        self._last_pos = None
        self._last_euler = None
        self._last_cp = int(s0.get("checkpoint_index") or 0)
        self._cp_game_time = float(s0.get("time_elapsed") or 0.0)
        self._crash_count = 0
        self._checkpoint_hits = 0
        obs = self._obs_from_state(s0, STEP_WAIT_S)
        if dbg:
            print(
                f">>> RESET DONE, obs shape: {obs.shape}, dtype: {obs.dtype}, "
                f"sample[:3]: {obs[:3]!r}",
                flush=True,
            )
            print(f">>> RESET final raw state: {_debug_state_line(s0)}", flush=True)
        return obs, {}

    def step(
        self, action: SupportsFloat | int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        a = int(cast(int, np.asarray(action).item()))
        keys = ACTION_MAP.get(a, [])

        chain_dbg = False
        if _polytrack_debug_chain():
            self._chain_step_seq += 1
            chain_dbg = self._chain_step_seq <= _polytrack_debug_max_steps()
        if chain_dbg:
            print(f">>> STEP #{self._chain_step_seq} called with action {a}", flush=True)
            print(f">>> sending keys {keys!r}", flush=True)

        async def _step() -> dict[str, Any]:
            assert self._bridge is not None
            if chain_dbg:
                print(">>> bridge.send_action(keys) ...", flush=True)
            await self._bridge.send_action(keys)
            if chain_dbg:
                print(">>> keys sent (send_action returned), sleep + get_state ...", flush=True)
            await asyncio.sleep(STEP_WAIT_S)
            if chain_dbg:
                print(">>> get_state() ...", flush=True)
            st = await self._bridge.get_state()
            if chain_dbg:
                print(f">>> state received: {_debug_state_line(st)}", flush=True)
            return st

        s = self._run(_step())
        cp = int(s.get("checkpoint_index") or 0)
        cp_prev = self._last_cp
        crashed = bool(s.get("crashed_or_reset"))
        if crashed:
            self._crash_count += 1
        if cp > cp_prev:
            self._checkpoint_hits += cp - cp_prev
            self._cp_game_time = float(s.get("time_elapsed") or 0.0)

        rew = self._reward(s, cp_prev, cp, crashed)
        obs = self._obs_from_state(s, STEP_WAIT_S)

        te = float(s.get("time_elapsed") or 0.0)
        finished = bool(s.get("has_finished"))
        terminated = finished or self._crash_count >= MAX_CRASHES
        truncated = te >= EPISODE_TIME_LIMIT_S
        if chain_dbg:
            print(
                f">>> reward: {float(rew):.4f}  terminated: {terminated}  truncated: {truncated}",
                flush=True,
            )

        self._last_cp = cp
        info: dict[str, Any] = {}
        if terminated or truncated:
            if finished:
                info["outcome"] = "finished"
            elif self._crash_count >= MAX_CRASHES:
                info["outcome"] = "crashed"
            elif truncated:
                info["outcome"] = "timeout"
            else:
                info["outcome"] = "crashed"
            info["checkpoints"] = int(self._checkpoint_hits)
            if self._dbg_done_count < 10:
                self._dbg_done_count += 1
                oc = str(info.get("outcome", ""))
                print(
                    "[polytrack_dbg] episode_end "
                    f"#{self._dbg_done_count} terminated={terminated} "
                    f"truncated={truncated} outcome={oc} "
                    f"crash_count={self._crash_count} te={te:.2f}s",
                    flush=True,
                )

        self._dbg_step_count += 1
        if self._dbg_step_count <= 20:
            spd = float(s.get("speed") or 0.0)
            print(
                "[polytrack_dbg] step "
                f"{self._dbg_step_count} reward={float(rew):.6f} "
                f"speed={spd:.2f} cp={cp} crashed_edge={crashed} "
                f"terminated={terminated} truncated={truncated}",
                flush=True,
            )

        assert obs.shape == (13,) and obs.dtype == np.float32
        return obs, float(rew), terminated, truncated, info

    def close(self) -> None:
        if self._bridge is not None:
            try:
                self._run(self._bridge.close())
            except Exception:
                pass
            self._bridge = None
        if not self._loop.is_closed():
            self._loop.close()
        super().close()
