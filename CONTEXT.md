Polytrack RL — Cursor Context File

## Project goal

Train a PPO reinforcement learning agent to play Polytrack (a browser-based low-poly racing game) as fast as possible on a specific track, eventually beating a ghost/target time. The agent controls the car via WASD keypresses and learns from game state extracted via JavaScript injection into a locally served copy of the game.

## Hardware (target training box)

- CPU: Ryzen 7 3700X (8c/16t) — parallel env rollouts
- GPU: RX 6750 XT (AMD) — no CUDA; use CPU training or ROCm if needed
- RAM: 32GB DDR4
- OS: Windows on the target machine; dev may use other OSes (e.g. macOS) — use repo-relative paths.

## Tech stack

- Python 3.11
- Playwright (async Chromium) — browser control + JS injection (`headless` or headed)
- Gymnasium — RL environment API (`env/polytrack_env.py`)
- Stable Baselines 3 — PPO (deps present; training scripts not yet added)
- PyTorch (CPU build) — neural network backend
- TensorBoard — training visualization
- `http.server` — local game (`start_server.py`)

## Project structure (repo root, e.g. Polyplex_V1)

```
├── polytrackcopy/              # cloned game (do not modify game logic)
├── env/
│   ├── game_bridge.py          # Playwright keyboard (KeyWASD) + __rlState / session recycle / track menu
│   ├── polytrack_env.py        # Gymnasium Env (Phase 2)
│   ├── test_game_bridge.py     # CLI poll of get_state() for debugging
│   ├── chromium_launch_args.py
│   ├── playwright_routes.py
│   └── debug_logging.py
├── utils/
│   ├── debug_finish_repro.py
│   └── training_monitor.py   # SB3 BaseCallback terminal dashboard
├── agent/                      # reserved
├── training/
│   ├── train_ppo.py            # single-env PPO (Monitor + DummyVecEnv)
│   └── __init__.py
├── run_local_training.py   # background HTTP server + foreground train_ppo (Ctrl+C ≠ kill server)
├── start_server.py
├── requirements.txt
├── checkpoints/
└── logs/
```

## Game details

- Source: [github.com/xanderxero/polytrackcopy](https://github.com/xanderxero/polytrackcopy) (Polytrack by Kodub, modified for AI)
- Engine: Three.js + Bullet (WASM) vehicle physics
- Controls: W accelerate, S brake/reverse, A/D steer, R restart
- Served locally; Playwright opens `http://127.0.0.1:<port>/`

### Phase 1 — vehicle state (JS)

- **Vehicle handle (read in bridge):** `window.__polytrackGhostData.advancedCar` (Bullet `em`: `getSpeedKmh`, `getPosition`, `getQuaternion`, `getNextCheckpointIndex`, `getTime`, `hasStarted`, `hasFinished`, …).
- **Injected mirror:** `window.__rlState` (updated every animation frame from `advancedCar`). Python reads this via `GameBridge.get_state()`.

### Phase 1 — localStorage (game `BC` storage, prefix `v2_`)

The minified storage class uses these keys (from `9209-dist-main.bundle.js`):

| Key / pattern | Role |
| --- | --- |
| `v2_car` | Car colors blob |
| `v2_record_<name>` | Ghost / record payloads |
| **`v2_track_<trackName>`** | **Saved track data** (`saveTrack` / `loadTrack`) |
| `v2_user` | User profile JSON |
| `v2_settings` | Settings array (tuples of setting id → string value) |
| `v2_key_bindings` | Key binding map |

There is **no separate scalar** “last selected track id” in localStorage; the UI lists tracks whose data lives under **`v2_track_*`**. RL automation: dismiss **`#ui .message-box.message`** if needed, **main-menu Play** (`button.button-image`), pick the **Nth** track row, then **track-info Play** (`#ui .menu .track-info .side-panel button.button.play`) to load the run (`GameBridge.start_track_menu_index`). Index **0** = first row (track 1 in menu order); override with `PolytrackEnv(track_menu_index=…)` or `python training/train_ppo.py --track-index N`.

### Phase 1 — reset / recycle strategy

- **Confirmed approach:** full **browser kill + new Chromium + `goto` + reinject** via `GameBridge.restart_session(url)` (used by `PolytrackEnv.reset()` after the first episode).
- **Track navigation:** `GameBridge.start_track_menu_index(index)` — pointer-lock exit, canvas focus, resume audio context, click track button `index` (default **0 = first track / track 1 in menu order**).
- **`_wait_for_game_ready()`** (env): polls `get_state()` until **`car_present` and `ready`** (vehicle exists, RL harness valid). Does **not** require motion or `has_started` — the in-game timer only starts after the player accelerates; the **first W must come from `env.step`**, not the bridge. Timeout `POLYTRACK_READY_TIMEOUT_S` (default 300 s).

## Observation space (13 floats, Box float32, normalized)

Used by `PolytrackEnv` (`MAX_TRACK_LENGTH = 2000` for future dist features):

| Index | Value | Normalization |
| --- | --- | --- |
| 0 | car speed | ÷ 200 |
| 1–3 | velocity x,y,z | ÷ 100 (finite-diff position / `STEP_WAIT_S`) |
| 4–6 | euler x,y,z (YXZ rad) | ÷ π |
| 7–9 | angular velocity proxy | ÷ 10 (finite-diff euler / step) |
| 10–11 | distance to next 2 checkpoints | ÷ `MAX_TRACK_LENGTH` (**currently 0** until track geometry is exposed) |
| 12 | time since last checkpoint (s) | ÷ 30 |

## Action space

`Discrete(9)` — maps index to WASD combination:

```python
ACTION_MAP = {
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
```

## Reward function (per step)

```text
reward = 0
reward += (speed / 200) * 0.01
reward += 2.0   # if new checkpoint this step (index increased)
reward -= 1.0   # if crashed_or_reset this step
reward -= 0.001 # time penalty every step
```

## Episode termination (`PolytrackEnv.step`)

- **`truncated`:** game `time_elapsed` ≥ **60** s.
- **`terminated`:** `has_finished` (lap / run complete) **or** **`crashed_or_reset` count ≥ 3** this episode (each step with flag True increments counter).
- Next `reset()` runs **`restart_session` + track menu index 0 + `_wait_for_game_ready`** (fresh browser after recycle).

## PPO hyperparameters (starting point)

```python
PPO(
    policy="MlpPolicy",
    policy_kwargs=dict(net_arch=[64, 64]),
    learning_rate=3e-4,
    n_steps=512,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    verbose=1,
    tensorboard_log="./logs/",
)
```

## Parallelism (implemented)

- **`training/train_ppo.py`:** `SubprocVecEnv` with **8** workers by default (`--num-envs`), **`n_steps=512` per env** → **4096** environment transitions per PPO rollout before each policy update. All env data is **concatenated** into one rollout buffer; the learner sees **every** step from **every** browser.
- **Ports:** `127.0.0.1:8080` … `8080 + num_envs - 1` (default **8080–8087**). Each `PolytrackEnv(port=…)` loads its own game origin.
- **Servers:** `python -m utils.launch_servers` or **`run_local_training.py`** (starts missing servers on 8080–8087). Logs: `logs/polytrack_http_server_<port>.log`.
- **Picklable workers:** `utils/polytrack_env_factory.py` (Windows `spawn`).
- **Windows / Ryzen + AMD:** see `docs/WINDOWS_TRAINING.md`.

## AMD GPU note

Default training: CPU. ROCm example: `pip install torch --index-url https://download.pytorch.org/whl/rocm6.2`. Do not assume CUDA; use `.to(device)` with `device = torch.device("cuda" if torch.cuda.is_available() else "cpu")`.

## Code style rules

- Minimal comments; docstrings only when logic is non-obvious
- Async Playwright: `await` in `game_bridge.py`; `PolytrackEnv` owns one `asyncio` loop and `run_until_complete` for SB3 sync API
- Keep `game_bridge` free of Gym imports
- Prefer `pathlib.Path` for filesystem paths

## Current phase

**Phase 1 — complete:** `GameBridge` (injection, `get_state`, `send_action`, `reset` = KeyR), `restart_session` (full browser recycle), `start_track_menu_index`, `FinishDebugGameBridge` / `debug_finish_repro.py` for DOM-heavy debugging. `test_game_bridge` smoke test.

**Phase 2 — complete:** `env/polytrack_env.py` — `PolytrackEnv(gymnasium.Env)`, `port` constructor arg, 13-d obs / 9 discrete actions, step cadence ~50 ms, reward + termination as above.

**Phase 3 — script:** `training/train_ppo.py` — `Monitor` + **`SubprocVecEnv`** (default **8** envs) or `--vec-env dummy` for `DummyVecEnv`; PPO (`MlpPolicy`, `net_arch=[64,64]`, **`n_steps=512` per env**, …), 1M steps, `CheckpointCallback` every 50k → `./checkpoints/`, TensorBoard → `./logs/`, `utils/training_monitor.TrainingMonitor` (ANSI dashboard every 1000 steps). Default browser is **headless**; pass **`--headed`** for visible Chromium. On episode end, `polytrack_env` sets `info["outcome"]` and `info["checkpoints"]`.

### Running on macOS (e.g. M1, 8 GB)

**Recommended:** one command starts the HTTP server in a **background process** (separate session), then training in the foreground — **Ctrl+C stops training only**, not the server:

```bash
cd /path/to/Polyplex_V1
source .venv/bin/activate
python run_local_training.py
```

Server logs: `logs/polytrack_http_server_<port>.log` (one file per port). **Ctrl+C** stops training; servers may keep running. Stop listeners: `lsof -ti :8080 | xargs kill` (repeat for 8081–8087 if needed) or use Activity Monitor / Task Manager.

**Manual servers + training:**

```bash
python -m utils.launch_servers
```

```bash
cd /path/to/Polyplex_V1
source .venv/bin/activate
python training/train_ppo.py
```

(`train_ppo.py` checks that **8080–8087** (or `base_port … + num_envs-1`) are reachable.)

If reset times out waiting for the car to be “ready”, increase seconds with env  
`POLYTRACK_READY_TIMEOUT_S` (default **300** in `polytrack_env.py`).

Optional: `python training/train_ppo.py --headless` to reduce GPU/RAM use; headed mode is heavier on an 8 GB machine—close other apps and expect swapping if memory is tight.

**Phase order**

1. ~~game_bridge + discovery~~
2. ~~`polytrack_env.py`~~
3. ~~`train_ppo.py` — single env PPO~~
4. ~~`launch_servers.py` + `SubprocVecEnv`~~
5. Ghost comparison reward

## Reward / obs implementation note

If reward or observation math changes in code, update this file. Checkpoint distances (obs slots 10–11) are placeholders until the env can read checkpoint world positions from the game or track file.
