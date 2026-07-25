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
# Intended control horizon at STEP_WAIT_S (not wall-clock / in-game seconds).
# With 4 Chromiums, each env step often burns 1–2s of *real game time*, so truncating
# on ``car.getTime()`` ≈ 30s only allows ~12 actions — learning collapses.
EPISODE_TIME_LIMIT_S = 30.0  # soft reference for docs / finish bonuses only
MAX_EPISODE_STEPS = int(os.environ.get("POLYTRACK_MAX_EPISODE_STEPS", "400"))
# Hard safety if the race clock somehow never resets after KeyR.
GAME_TIME_HARD_CAP_S = 120.0
MAX_CRASHES = 3
# Off-track / void (jumps allowed): terminate after sustained freefall with no track below.
Y_VOID = -25.0
AIRBORNE_OFFTRACK_STEPS = 35  # ~1.75 s at 50 ms/step — jumps are much shorter
GROUND_MISS = 95.0            # ground_dist near MAX_RAY_DIST ⇒ no track under car
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
        self._last_xz: tuple[float, float] | None = None
        self._last_cp: int = 0
        self._crash_count: int = 0
        self._checkpoint_hits: int = 0
        self._dbg_step_count: int = 0
        self._dbg_done_count: int = 0
        self._chain_step_seq: int = 0
        # Fitness = horizontal (XZ) distance travelled this episode (metres).
        self._fitness: float = 0.0
        self._checkpoint_times: list[float] = []
        self._finish_scored: bool = False
        self._horiz_distance: float = 0.0
        self._airborne_steps: int = 0
        self._episode_steps: int = 0
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

    async def _wait_for_game_ready(self, *, require_fresh_timer: bool = False) -> None:
        """Wait until vehicle exists (and optionally race timer has reset after KeyR)."""
        deadline = time.monotonic() + READY_TIMEOUT_S
        last_nudge_m = 0.0
        while time.monotonic() < deadline:
            assert self._bridge is not None
            s = await self._bridge.get_state()
            if s.get("error"):
                await asyncio.sleep(0.1)
                continue
            if s.get("car_present") and s.get("ready"):
                te = float(s.get("time_elapsed") or 0.0)
                # After soft reset, refuse to start an episode while the old 30s clock
                # is still hot — that was producing 6–15 step "timeout" episodes.
                if not require_fresh_timer or te < 2.5:
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

    def _record_checkpoint_times(
        self, s: dict[str, Any], cp_prev: int, cp_now: int
    ) -> int:
        """Log checkpoint arrival times (info only — does not affect fitness)."""
        if cp_now <= cp_prev:
            return 0
        te = float(s.get("time_elapsed") or 0.0)
        n = cp_now - cp_prev
        for _ in range(n):
            self._checkpoint_times.append(round(max(0.0, te), 3))
        return n

    def _update_fitness(self) -> float:
        """Fitness = horizontal distance (m). Returns metres gained this step."""
        prev = self._fitness
        self._fitness = float(self._horiz_distance)
        return self._fitness - prev

    def _reward(
        self,
        s: dict[str, Any],
        fitness_delta: float,
        crashed: bool,
        *,
        new_checkpoints: int = 0,
    ) -> float:
        """Favor distance travelled + speed; penalise walls/crashes/off-track."""
        r = 0.0

        # PRIMARY: metres of horizontal progress this step.
        r += fitness_delta * 0.08

        # Keep moving (helps cover distance).
        speed_kmh = float(s.get("speed") or 0.0)
        r += (speed_kmh / 200.0) * 0.03

        # Small bonus if/when checkpoints are eventually reached.
        if new_checkpoints > 0:
            r += 0.5 * float(new_checkpoints)

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

    def _update_horiz_distance(self, s: dict[str, Any]) -> None:
        """Accumulate XZ path length (ignores vertical jump motion)."""
        try:
            x = float(s["position"]["x"])
            z = float(s["position"]["z"])
        except (KeyError, TypeError, ValueError):
            return
        if self._last_xz is not None:
            dx = x - self._last_xz[0]
            dz = z - self._last_xz[1]
            step_d = float(np.hypot(dx, dz))
            # Ignore teleport/reset spikes.
            if step_d < 40.0:
                self._horiz_distance += step_d
        self._last_xz = (x, z)

    def _check_off_track(self, s: dict[str, Any]) -> bool:
        """True when the car fell off the track — not when it is mid-jump.

        Jump: briefly airborne with track mesh still below (ground_dist finite).
        Off-track: Y into the void, or long freefall with no ground ray hit.
        """
        if not s.get("has_started"):
            self._airborne_steps = 0
            return False

        try:
            y = float(s["position"]["y"])
        except (KeyError, TypeError, ValueError):
            y = 0.0

        if y < Y_VOID:
            return True
        if bool(s.get("off_track")):
            return True

        airborne = bool(s.get("airborne"))
        ground = float(s.get("ground_dist") or MAX_RAY_DIST)
        if airborne:
            self._airborne_steps += 1
        else:
            self._airborne_steps = 0

        # Long freefall with no track under the car → void / off the map.
        if (
            self._airborne_steps >= AIRBORNE_OFFTRACK_STEPS
            and ground >= GROUND_MISS
        ):
            return True
        return False

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
            await self._wait_for_game_ready(require_fresh_timer=True)
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
        self._last_xz = None
        self._last_cp = int(s0.get("checkpoint_index") or 0)
        self._crash_count = 0
        self._checkpoint_hits = 0
        self._fitness = 0.0
        self._checkpoint_times = []
        self._finish_scored = False
        self._horiz_distance = 0.0
        self._airborne_steps = 0
        self._episode_steps = 0
        self._last_episode_finished = False
        self._update_horiz_distance(s0)
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
        self._episode_steps += 1
        cp = int(s.get("checkpoint_index") or 0)
        cp_prev = self._last_cp
        crashed = bool(s.get("crashed_or_reset"))
        if crashed:
            self._crash_count += 1
        if cp > cp_prev:
            self._checkpoint_hits += cp - cp_prev

        self._update_horiz_distance(s)
        new_cps = self._record_checkpoint_times(s, cp_prev, cp)
        fitness_delta = self._update_fitness()
        off_track = self._check_off_track(s)
        rew = self._reward(
            s, fitness_delta, crashed or off_track, new_checkpoints=new_cps
        )
        obs = self._obs_from_state(s, STEP_WAIT_S)

        te = float(s.get("time_elapsed") or 0.0)
        finished = bool(s.get("has_finished"))
        # Finishing the lap: reward bonus only (fitness stays distance-based).
        if finished and not self._finish_scored:
            rew += 2.0
            self._finish_scored = True

        # Off-track → end episode immediately (soft KeyR on next reset).
        terminated = (
            finished
            or off_track
            or self._crash_count >= MAX_CRASHES
        )
        # Truncate on *action steps*, not in-game seconds (see MAX_EPISODE_STEPS).
        truncated = (
            self._episode_steps >= MAX_EPISODE_STEPS
            or te >= GAME_TIME_HARD_CAP_S
        )
        if chain_dbg:
            print(
                f">>> reward: {float(rew):.4f}  fitness: {self._fitness:.1f}  "
                f"dist={self._horiz_distance:.1f}  off_track={off_track}  "
                f"steps={self._episode_steps}  "
                f"terminated: {terminated}  truncated: {truncated}",
                flush=True,
            )

        self._last_cp = cp
        info: dict[str, Any] = {
            "fitness": float(self._fitness),
            "checkpoints": int(self._checkpoint_hits),
            "checkpoint_times": list(self._checkpoint_times),
            "distance_m": float(self._horiz_distance),
            "off_track": bool(off_track),
            "airborne": bool(s.get("airborne")),
            "episode_steps": int(self._episode_steps),
        }
        if terminated or truncated:
            self._last_episode_finished = finished
            if finished:
                info["outcome"] = "finished"
            elif off_track:
                info["outcome"] = "off_track"
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
                    f"fitness={self._fitness:.1f} dist={self._horiz_distance:.1f} "
                    f"steps={self._episode_steps} "
                    f"crash_count={self._crash_count} te={te:.2f}s",
                    flush=True,
                )

        self._dbg_step_count += 1
        if self._dbg_step_count <= 20:
            spd = float(s.get("speed") or 0.0)
            print(
                "[polytrack_dbg] step "
                f"{self._dbg_step_count} reward={float(rew):.6f} "
                f"fitness={self._fitness:.1f} dist={self._horiz_distance:.1f} "
                f"speed={spd:.2f} cp={cp} "
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
