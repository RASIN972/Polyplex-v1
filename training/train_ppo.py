from __future__ import annotations

import argparse
import multiprocessing
import os
import socket
import subprocess
import sys
import traceback
from datetime import datetime
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from utils.elite_callback import EliteFitnessCallback
from utils.polytrack_env_factory import make_polytrack_monitored_env
from utils.training_monitor import TrainingMonitor

TOTAL_TIMESTEPS = 1_000_000
CHECKPOINT_FREQ = 50_000

# PPO collects n_steps * num_envs transitions per rollout before each policy update.
N_STEPS_PER_ENV = 512
NUM_ENVS_DEFAULT = 4
BASE_PORT_DEFAULT = 8080


def _launch_gui_monitor() -> subprocess.Popen | None:
    """Spawn gui_monitor.py in a separate process. Returns the Popen or None on failure."""
    gui_script = _ROOT / "gui_monitor.py"
    if not gui_script.exists():
        return None
    # On macOS/Linux, check whether a display is available before trying Tkinter.
    if sys.platform != "win32" and not os.environ.get("DISPLAY") and sys.platform != "darwin":
        print("[gui_monitor] No DISPLAY — skipping GUI monitor (headless server).", flush=True)
        return None
    try:
        proc = subprocess.Popen(
            [sys.executable, str(gui_script)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            close_fds=True,
        )
        print(f"[gui_monitor] Launched (pid {proc.pid}). Close the window any time.", flush=True)
        return proc
    except Exception as exc:
        print(f"[gui_monitor] Could not launch GUI monitor: {exc}", flush=True)
        return None


def _require_game_servers(host: str, ports: list[int]) -> None:
    missing: list[int] = []
    for port in ports:
        try:
            socket.create_connection((host, port), timeout=2.0).close()
        except OSError:
            missing.append(port)
    if missing:
        print(
            f"\nNo game server at http://{host}:PORT/ for PORT in {missing}.\n"
            "  Start servers first, e.g.:\n"
            "    python -m utils.launch_servers\n"
            "  Or:  python run_local_training.py\n",
            file=sys.stderr,
            flush=True,
        )
        raise SystemExit(1)
    if not os.environ.get("POLYTRACK_FROM_RUN_LOCAL"):
        print(
            "\n"
            "------------------------------------------------------------\n"
            "  PPO training — start HTTP servers before training unless you use:\n"
            "    python run_local_training.py\n"
            "  (Stopping THIS process does not stop the HTTP servers.)\n"
            "------------------------------------------------------------\n",
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-port", type=int, default=BASE_PORT_DEFAULT)
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        metavar="P",
        help="Alias for --base-port (single-server legacy; parallel uses base_port..base_port+N-1)",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=NUM_ENVS_DEFAULT,
        help="Parallel Polytrack instances (each uses base_port + index)",
    )
    parser.add_argument(
        "--vec-env",
        choices=("subproc", "dummy"),
        default="subproc",
        help="subproc = parallel processes (default). dummy = single-process (debug / Windows fallback).",
    )
    parser.add_argument(
        "--headed",
        action="store_true",
        help="Show Chromium windows for ALL envs (default: headless).",
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help=(
            "Run worker 0 in a visible (headed) Chromium window so you can watch the agent live. "
            "Workers 1..N-1 remain headless. Ignored when --headed is also set."
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="PyTorch device for PPO (default: cpu — matches typical AMD/CPU training setup)",
    )
    parser.add_argument(
        "--track-index",
        type=int,
        default=0,
        help="Track row after main-menu Play (0 = first track)",
    )
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Suppress the auto-launched Tkinter GUI monitor window.",
    )
    args = parser.parse_args()

    num_envs = max(1, int(args.num_envs))
    base = args.port if args.port is not None else args.base_port
    ports = [base + i for i in range(num_envs)]

    # --headed: all envs visible. --watch: only worker 0 is headed.
    def _headless_for(worker_idx: int) -> bool:
        if args.headed:
            return False
        if args.watch and worker_idx == 0:
            return False
        return True

    _require_game_servers("127.0.0.1", ports)

    checkpoint_dir = _ROOT / "checkpoints"
    log_dir = _ROOT / "logs"
    monitor_dir = log_dir / "monitor"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    monitor_dir.mkdir(parents=True, exist_ok=True)

    if args.watch and not args.headed:
        print("  [--watch] Worker 0 will open a visible Chromium window.\n", flush=True)

    env_fns = [
        make_polytrack_monitored_env(
            port=ports[i],
            track_index=args.track_index,
            headless=_headless_for(i),
            monitor_file=str(monitor_dir / f"polytrack_{ports[i]}.csv"),
            worker_index=i,
        )
        for i in range(num_envs)
    ]

    if args.vec_env == "dummy":
        vec = DummyVecEnv([fn for fn in env_fns])
    else:
        vec = SubprocVecEnv(env_fns)

    crash_log = log_dir / "last_training_error.txt"

    rollout_size = N_STEPS_PER_ENV * num_envs
    print(
        f"PPO: {num_envs} envs × {N_STEPS_PER_ENV} steps/env = {rollout_size} transitions per rollout "
        f"(ports {ports[0]}–{ports[-1]}).\n",
        flush=True,
    )

    gui_proc: subprocess.Popen | None = None
    if not args.no_gui:
        gui_proc = _launch_gui_monitor()

    try:
        model = PPO(
            "MlpPolicy",
            vec,
            learning_rate=3e-4,
            n_steps=N_STEPS_PER_ENV,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            # Entropy bonus keeps the policy diverse (exploration); elites are
            # locked in via EliteFitnessCallback when track-distance fitness peaks.
            ent_coef=0.02,
            verbose=1,
            tensorboard_log=str(log_dir),
            # Smaller net matches the lean 9-d observation.
            policy_kwargs=dict(net_arch=[64, 64]),
            device=args.device,
        )

        checkpoint_cb = CheckpointCallback(
            save_freq=CHECKPOINT_FREQ,
            save_path=str(checkpoint_dir),
            name_prefix="ppo_polytrack",
            save_replay_buffer=False,
            save_vecnormalize=False,
        )
        elite_cb = EliteFitnessCallback(
            checkpoint_dir, track_index=args.track_index, verbose=1
        )
        monitor_cb = TrainingMonitor(TOTAL_TIMESTEPS)

        TrainingMonitor.show_bootstrap(TOTAL_TIMESTEPS)

        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=[checkpoint_cb, elite_cb, monitor_cb],
            progress_bar=False,
        )

        stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        dated_path = checkpoint_dir / f"ppo_polytrack_{stamp}"
        model.save(str(dated_path))
        print(f"Saved trained model to {dated_path}.zip", flush=True)
    except Exception as e:
        crash_log.write_text(traceback.format_exc(), encoding="utf-8")
        print(
            f"\nTraining stopped with an error. Full traceback saved to:\n  {crash_log}\n"
            "(HTTP servers on 8080+ are separate processes — stop them manually if needed.)\n",
            flush=True,
            file=sys.stderr,
        )
        if isinstance(e, TimeoutError) and "_wait_for_game_ready" in str(e):
            print(
                "Hint: the game never became ‘ready’ (car present). Try: "
                "POLYTRACK_READY_TIMEOUT_S=600 python run_local_training.py\n",
                file=sys.stderr,
                flush=True,
            )
        traceback.print_exc()
        raise
    finally:
        vec.close()
        if gui_proc is not None and gui_proc.poll() is None:
            gui_proc.terminate()


if __name__ == "__main__":
    multiprocessing.freeze_support()  # required for SubprocVecEnv on Windows
    main()
