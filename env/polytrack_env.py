from __future__ import annotations

import asyncio
import os
import time
from typing import Any, SupportsFloat, cast

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from env.game_bridge import FinishDebugGameBridge, GameBridge

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
MAX_RAY_DIST = 100.0          # max wall-raycast distance (metres), matches JS value
WALL_WARN_DIST = 15.0         # soft-penalty zone: < 15 m to nearest wall
# Lean obs: speed, yaw, yaw_rate, 6 wall rays. No velocity/euler bloat, no checkpoint slots.
OBS_SIZE = 9
READY_TIMEOUT_S = float(os.environ.get("POLYTRACK_READY_TIMEOUT_S", "300"))
EPISODE_TIME_LIMIT_S = 30.0
MAX_CRASHES = 3
PI_F = float(np.pi)
DEFAULT_TRACK_MENU_INDEX = 0
RESET_RETRIES = int(os.environ.get("POLYTRACK_RESET_RETRIES", "2"))


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
            low=-np.inf, high=np.inf, shape=(OBS_SIZE,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(9)

        self._last_yaw: float | None = None
        self._last_cp: int = 0
        self._crash_count: int = 0
        self._checkpoint_hits: int = 0
        self._dbg_step_count: int = 0
        self._dbg_done_count: int = 0
        self._chain_step_seq: int = 0
        # Fitness = sum over checkpoints of (episode_cap - arrival_time). Faster = higher.
        self._fitness: float = 0.0
        self._checkpoint_times: list[float] = []
        self._finish_scored: bool = False
        # Track whether the last episode ended via has_finished (needs FinishDebug recovery).
        self._last_episode_finished: bool = False

    def _finalize_obs(self, arr: np.ndarray) -> np.ndarray:
        o = np.ascontiguousarray(arr, dtype=np.float32)
        if o.shape != (OBS_SIZE,):
            raise ValueError(
                f"PolytrackEnv: obs must be shape ({OBS_SIZE},), got {o.shape} dtype={o.dtype}"
            )
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
        """Lean 9-d obs: speed, yaw, yaw_rate, 6 wall distances."""
        yaw = float(s["rotation"]["y"])
        if self._last_yaw is None or dt <= 0:
            yaw_rate = 0.0
        else:
            # Unwrap yaw delta into [-pi, pi] before differencing.
            dyaw = yaw - self._last_yaw
            dyaw = (dyaw + PI_F) % (2.0 * PI_F) - PI_F
            yaw_rate = dyaw / dt
        self._last_yaw = yaw

        obs = np.zeros(OBS_SIZE, dtype=np.float32)
        obs[0] = np.float32(float(s.get("speed") or 0.0) / 200.0)
        obs[1] = np.float32(yaw / PI_F)
        obs[2] = np.float32(yaw_rate / 10.0)

        # Slots 3–8: wall distances [F, FR, R, L, FL, B], normalised to [0, 1].
        raw_walls = s.get("wall_dists") or [MAX_RAY_DIST] * 6
        for i in range(6):
            d = float(raw_walls[i]) if i < len(raw_walls) else MAX_RAY_DIST
            obs[3 + i] = np.float32(np.clip(d / MAX_RAY_DIST, 0.0, 1.0))

        return self._finalize_obs(obs)

    def _update_fitness(self, s: dict[str, Any], cp_prev: int, cp_now: int) -> float:
        """Score by how fast each checkpoint is reached (lower time → higher fitness).

        For every newly reached checkpoint at game-time ``t``:
            fitness += (EPISODE_TIME_LIMIT_S - t)

        So an AI that hits CP1 at 5 s beats one that hits it at 20 s, and hitting
        more checkpoints still adds more score. Returns the fitness delta this step
        (for reward shaping).
        """
        if cp_now <= cp_prev:
            return 0.0
        te = float(s.get("time_elapsed") or 0.0)
        delta_fit = 0.0
        for _ in range(cp_now - cp_prev):
            t = max(0.0, te)
            score = max(0.0, EPISODE_TIME_LIMIT_S - t)
            self._checkpoint_times.append(round(t, 3))
            self._fitness += score
            delta_fit += score
        return delta_fit

    def _reward(
        self,
        s: dict[str, Any],
        fitness_delta: float,
        crashed: bool,
    ) -> float:
        """Favor fast checkpoint arrivals + speed; penalise walls/crashes."""
        r = 0.0

        # PRIMARY: time-to-checkpoint fitness gained this step (faster CP = more).
        r += fitness_delta * 0.15

        # Keep moving fast (helps reach checkpoints sooner).
        speed_kmh = float(s.get("speed") or 0.0)
        r += (speed_kmh / 200.0) * 0.03

        # Soft wall-proximity penalty.
        raw_walls = s.get("wall_dists") or []
        if raw_walls:
            min_wall = min(float(d) for d in raw_walls)
            if min_wall < WALL_WARN_DIST:
                closeness = 1.0 - min_wall / WALL_WARN_DIST
                r -= closeness * 0.12

        if crashed:
            r -= 1.0

        r -= 0.001  # small per-step time cost (pressure to not dawdle)
        return r

    async def _full_browser_reset(self, *, dbg: bool = False) -> dict[str, Any]:
        """Launch a fresh Chromium, navigate the menu, wait for game ready."""
        assert self._bridge is not None
        if dbg:
            print(">>> RESET _full_browser_reset: restart_session ...", flush=True)
        await self._bridge.restart_session(self._url, headless=self._headless)
        if dbg:
            print(">>> RESET _full_browser_reset: start_track_menu_index ...", flush=True)
        await self._bridge.start_track_menu_index(self._track_menu_index)
        await self._wait_for_game_ready()
        return await self._bridge.get_state()

    async def _soft_reset(self, *, dbg: bool = False, after_finish: bool = False) -> dict[str, Any]:
        """Soft reset: press R (or run FinishDebug recovery for post-finish), then wait for game ready.
        Falls back to a full browser reset on timeout/page error."""
        assert self._bridge is not None
        if after_finish:
            if dbg:
                print(">>> RESET _soft_reset: FinishDebugGameBridge recovery ...", flush=True)
            fdb = FinishDebugGameBridge(
                self._bridge._page,  # type: ignore[attr-defined]
                reenter_track_index=self._track_menu_index,
            )
            await fdb.reset(run_id="soft-reset")
        else:
            if dbg:
                print(">>> RESET _soft_reset: bridge.reset (KeyR) ...", flush=True)
            await self._bridge.reset()
        try:
            await self._wait_for_game_ready()
        except TimeoutError:
            print(
                "[polytrack_env] soft reset ready-wait timed out — falling back to full browser reset",
                flush=True,
            )
            return await self._full_browser_reset(dbg=dbg)
        return await self._bridge.get_state()

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
                # First reset: launch browser, navigate menu.
                if dbg:
                    print(">>> RESET _go: GameBridge.launch ...", flush=True)
                self._bridge = await GameBridge.launch(
                    self._url, headless=self._headless
                )
                if dbg:
                    print(">>> RESET _go: launch done, start_track_menu_index ...", flush=True)
                await self._bridge.start_track_menu_index(self._track_menu_index)
                await self._wait_for_game_ready()
                return await self._bridge.get_state()
            else:
                # Subsequent resets: try soft KeyR reset, only full-recycle on failure.
                return await self._soft_reset(
                    dbg=dbg, after_finish=self._last_episode_finished
                )

        async def _go_with_retry() -> dict[str, Any]:
            last_exc: Exception | None = None
            for attempt in range(max(1, RESET_RETRIES + 1)):
                try:
                    return await _go()
                except Exception as exc:
                    last_exc = exc
                    print(
                        f"[polytrack_env] reset attempt {attempt + 1} failed "
                        f"({type(exc).__name__}: {exc}); retrying with fresh browser ...",
                        flush=True,
                    )
                    try:
                        if self._bridge is not None:
                            await self._bridge.close()
                    except Exception:
                        pass
                    self._bridge = None
                    if attempt + 1 < max(1, RESET_RETRIES + 1):
                        # Re-launch for next attempt.
                        self._bridge = await GameBridge.launch(
                            self._url, headless=self._headless
                        )
                        await self._bridge.start_track_menu_index(self._track_menu_index)
            assert last_exc is not None
            raise last_exc

        if dbg:
            print(">>> RESET: entering event loop (_run _go)", flush=True)
        s0 = self._run(_go_with_retry())
        self._last_yaw = None
        self._last_cp = int(s0.get("checkpoint_index") or 0)
        self._crash_count = 0
        self._checkpoint_hits = 0
        self._fitness = 0.0
        self._checkpoint_times = []
        self._finish_scored = False
        self._last_episode_finished = False
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

        fitness_delta = self._update_fitness(s, cp_prev, cp)
        rew = self._reward(s, fitness_delta, crashed)
        obs = self._obs_from_state(s, STEP_WAIT_S)

        te = float(s.get("time_elapsed") or 0.0)
        finished = bool(s.get("has_finished"))
        # Finishing the lap early also scores like a final "checkpoint".
        if finished and not self._finish_scored:
            finish_score = max(0.0, EPISODE_TIME_LIMIT_S - te)
            self._fitness += finish_score
            rew += finish_score * 0.15
            self._finish_scored = True

        terminated = finished or self._crash_count >= MAX_CRASHES
        truncated = te >= EPISODE_TIME_LIMIT_S
        if chain_dbg:
            print(
                f">>> reward: {float(rew):.4f}  fitness: {self._fitness:.1f}  "
                f"cp_times={self._checkpoint_times}  "
                f"terminated: {terminated}  truncated: {truncated}",
                flush=True,
            )

        self._last_cp = cp
        info: dict[str, Any] = {
            "fitness": float(self._fitness),
            "checkpoints": int(self._checkpoint_hits),
            "checkpoint_times": list(self._checkpoint_times),
        }
        if terminated or truncated:
            self._last_episode_finished = finished
            if finished:
                info["outcome"] = "finished"
            elif self._crash_count >= MAX_CRASHES:
                info["outcome"] = "crashed"
            elif truncated:
                info["outcome"] = "timeout"
            else:
                info["outcome"] = "crashed"
            if self._dbg_done_count < 10:
                self._dbg_done_count += 1
                oc = str(info.get("outcome", ""))
                print(
                    "[polytrack_dbg] episode_end "
                    f"#{self._dbg_done_count} terminated={terminated} "
                    f"truncated={truncated} outcome={oc} "
                    f"fitness={self._fitness:.1f} "
                    f"cp_times={self._checkpoint_times} "
                    f"crash_count={self._crash_count} te={te:.2f}s",
                    flush=True,
                )

        self._dbg_step_count += 1
        if self._dbg_step_count <= 20:
            spd = float(s.get("speed") or 0.0)
            print(
                "[polytrack_dbg] step "
                f"{self._dbg_step_count} reward={float(rew):.6f} "
                f"fitness={self._fitness:.1f} speed={spd:.2f} cp={cp} "
                f"crashed_edge={crashed} terminated={terminated} truncated={truncated}",
                flush=True,
            )

        assert obs.shape == (OBS_SIZE,) and obs.dtype == np.float32
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
