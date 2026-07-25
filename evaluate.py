#!/usr/bin/env python3
"""Evaluate a trained Polytrack PPO model in a live headed Chromium window.

Usage:
    python evaluate.py --model checkpoints/best_model.zip --auto-server

Optional:
    --port 8099          HTTP server port (default: 8099 — avoids training ports)
    --track-index 0      Track row in the menu (default: 0)
    --episodes 0         Number of episodes; 0 = until Ctrl+C (default: 0)
    --auto-server        Start ``start_server.py`` if the port is free
    --no-deterministic   Stochastic actions
"""

from __future__ import annotations

import argparse
import socket
import subprocess
import sys
import time
import traceback
from pathlib import Path

# Immediate feedback for GUI / watch (before heavy torch imports).
print("Watch: starting evaluate.py …", flush=True)

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _port_open(port: int) -> bool:
    try:
        socket.create_connection(("127.0.0.1", port), timeout=1.0).close()
        return True
    except OSError:
        return False


def _ensure_server(port: int, *, auto: bool) -> subprocess.Popen | None:
    if _port_open(port):
        return None
    if not auto:
        print(
            f"\nNo game server at http://127.0.0.1:{port}/\n"
            "  Start one first, or pass --auto-server:\n"
            f"    python start_server.py --port {port}\n",
            file=sys.stderr,
        )
        raise SystemExit(1)

    print(f"Starting game server on port {port} …", flush=True)
    proc = subprocess.Popen(
        [sys.executable, str(_ROOT / "start_server.py"), "--port", str(port)],
        cwd=str(_ROOT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    for _ in range(40):
        if _port_open(port):
            print(f"  server ready at http://127.0.0.1:{port}/", flush=True)
            return proc
        if proc.poll() is not None:
            print("  start_server.py exited early — check polytrackcopy/", file=sys.stderr)
            raise SystemExit(1)
        time.sleep(0.25)
    print("  timed out waiting for server", file=sys.stderr)
    proc.terminate()
    raise SystemExit(1)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run a trained Polytrack PPO model in a headed browser window."
    )
    p.add_argument("--model", required=True, help="Path to .zip checkpoint (PPO.load)")
    p.add_argument(
        "--port",
        type=int,
        default=8099,
        help="Game server port (default: 8099 — separate from training 8080+)",
    )
    p.add_argument("--track-index", type=int, default=0, help="Track menu row (default: 0)")
    p.add_argument(
        "--episodes",
        type=int,
        default=0,
        help="Episodes to run; 0 = run indefinitely until Ctrl+C (default: 0)",
    )
    p.add_argument(
        "--auto-server",
        action="store_true",
        help="Start start_server.py on --port if nothing is listening",
    )
    p.add_argument(
        "--no-deterministic",
        action="store_true",
        help="Use stochastic actions (default: deterministic=True)",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    deterministic = not args.no_deterministic
    server_proc: subprocess.Popen | None = None
    env = None

    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = (_ROOT / model_path).resolve()
    if not model_path.exists():
        print(f"Model file not found: {model_path}", file=sys.stderr)
        raise SystemExit(1)

    try:
        from stable_baselines3 import PPO

        from env.polytrack_env import OBS_SIZE, PolytrackEnv

        print(f"Watch: loading model {model_path.name} …", flush=True)
        try:
            model = PPO.load(str(model_path), device="cpu")
        except Exception as exc:
            print(f"Failed to load PPO checkpoint: {exc}", file=sys.stderr)
            traceback.print_exc()
            raise SystemExit(2) from exc
        print("Watch: model loaded.", flush=True)

        obs_space = getattr(model, "observation_space", None)
        if obs_space is not None and hasattr(obs_space, "shape"):
            shape = obs_space.shape
            if shape and int(shape[0]) != OBS_SIZE:
                print(
                    f"Checkpoint obs dim {shape[0]} != current env OBS_SIZE={OBS_SIZE}.\n"
                    "  Retrain (or pick a checkpoint trained with the current obs layout).",
                    file=sys.stderr,
                )
                raise SystemExit(3)

        server_proc = _ensure_server(args.port, auto=args.auto_server)

        print(
            f"Watch: port={args.port} track={args.track_index} "
            f"deterministic={deterministic}",
            flush=True,
        )
        print("Watch: opening headed Chromium window …", flush=True)

        env = PolytrackEnv(
            port=args.port,
            headless=False,
            track_menu_index=args.track_index,
        )
        # Bind env so predict uses matching spaces (SB3 optional).
        model.set_env(env)

        ep = 0
        max_eps = args.episodes if args.episodes > 0 else float("inf")

        while ep < max_eps:
            obs, _ = env.reset()
            ep += 1
            ep_reward = 0.0
            step = 0
            t0 = time.perf_counter()
            info: dict = {}

            print(f"--- Episode {ep} ---", flush=True)
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=deterministic)
                obs, reward, terminated, truncated, info = env.step(action)
                ep_reward += float(reward)
                step += 1
                done = terminated or truncated

                speed = float(obs[0]) * 200.0
                fitness = float(info.get("fitness", 0.0))
                dist = float(info.get("distance_m", fitness))
                if step % 20 == 0:
                    print(
                        f"  step {step:4d}  speed={speed:5.1f} km/h  "
                        f"dist={dist:6.1f} m  "
                        f"ep_reward={ep_reward:+.2f}",
                        end="\r",
                        flush=True,
                    )

            elapsed = time.perf_counter() - t0
            outcome = info.get("outcome", "?")
            checkpoints = info.get("checkpoints", 0)
            fitness = float(info.get("fitness", 0.0))
            dist = float(info.get("distance_m", fitness))
            cp_times = info.get("checkpoint_times") or []
            print(
                f"  steps={step}  dist={dist:.1f}m  "
                f"reward={ep_reward:+.2f}  checkpoints={checkpoints}  "
                f"cp_times={cp_times}  outcome={outcome}  wall={elapsed:.1f}s        ",
                flush=True,
            )
    except KeyboardInterrupt:
        print("\nStopped by user.", flush=True)
    except SystemExit:
        raise
    except Exception as exc:
        print(f"\nWatch/evaluate crashed: {type(exc).__name__}: {exc}", file=sys.stderr)
        traceback.print_exc()
        raise SystemExit(1) from exc
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
        if server_proc is not None and server_proc.poll() is None:
            server_proc.terminate()
            try:
                server_proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                server_proc.kill()


if __name__ == "__main__":
    main()
