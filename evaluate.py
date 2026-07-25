#!/usr/bin/env python3
"""Evaluate a trained Polytrack PPO model in a live headed Chromium window.

Usage:
    python evaluate.py --model checkpoints/ppo_polytrack_50000_steps.zip

Optional:
    --port 8080          HTTP server port (default: 8080)
    --track-index 0      Track row in the menu (default: 0)
    --episodes 0         Number of episodes to run; 0 = run until Ctrl+C (default: 0)
    --deterministic      Use deterministic actions (default: True)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run a trained Polytrack PPO model in a headed browser window."
    )
    p.add_argument("--model", required=True, help="Path to .zip checkpoint (PPO.load)")
    p.add_argument("--port", type=int, default=8080, help="Game server port (default: 8080)")
    p.add_argument("--track-index", type=int, default=0, help="Track menu row (default: 0)")
    p.add_argument(
        "--episodes",
        type=int,
        default=0,
        help="Episodes to run; 0 = run indefinitely until Ctrl+C (default: 0)",
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

    import socket

    try:
        socket.create_connection(("127.0.0.1", args.port), timeout=2.0).close()
    except OSError:
        print(
            f"\nNo game server at http://127.0.0.1:{args.port}/\n"
            "  Start one first, e.g.:\n"
            f"    python start_server.py --port {args.port}\n",
            file=sys.stderr,
        )
        raise SystemExit(1)

    from stable_baselines3 import PPO

    from env.polytrack_env import PolytrackEnv

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Model file not found: {model_path}", file=sys.stderr)
        raise SystemExit(1)

    print(f"Loading model: {model_path}")
    print(f"  port={args.port}  track-index={args.track_index}  deterministic={deterministic}")
    print("  Press Ctrl+C to stop.\n")

    env = PolytrackEnv(
        port=args.port,
        headless=False,
        track_menu_index=args.track_index,
    )

    model = PPO.load(str(model_path), env=env)

    ep = 0
    max_eps = args.episodes if args.episodes > 0 else float("inf")

    try:
        while ep < max_eps:
            obs, _ = env.reset()
            ep += 1
            ep_reward = 0.0
            step = 0
            t0 = time.perf_counter()

            print(f"--- Episode {ep} ---")
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=deterministic)
                obs, reward, terminated, truncated, info = env.step(action)
                ep_reward += float(reward)
                step += 1
                done = terminated or truncated

                speed = float(obs[0]) * 200.0
                fitness = float(info.get("fitness", 0.0))
                if step % 20 == 0:
                    print(
                        f"  step {step:4d}  speed={speed:5.1f} km/h  "
                        f"fitness={fitness:6.1f}  ep_reward={ep_reward:+.2f}",
                        end="\r",
                    )

            elapsed = time.perf_counter() - t0
            outcome = info.get("outcome", "?")
            checkpoints = info.get("checkpoints", 0)
            fitness = float(info.get("fitness", 0.0))
            cp_times = info.get("checkpoint_times") or []
            print(
                f"  steps={step}  fitness={fitness:.1f}  reward={ep_reward:+.2f}  "
                f"checkpoints={checkpoints}  cp_times={cp_times}  "
                f"outcome={outcome}  wall={elapsed:.1f}s        "
            )
    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        env.close()


if __name__ == "__main__":
    main()
