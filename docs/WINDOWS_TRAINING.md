# Polyplex training on Windows (Ryzen 3700X + RX 6750 XT + 32 GB)

## What runs where

| Workload | Device | Notes |
|----------|--------|--------|
| **4× Chromium + Polytrack** (Playwright) | **CPU + AMD GPU (ANGLE/D3D)** | Main bottleneck. Headless on Windows uses the real GPU stack (not SwiftShader). |
| **PPO MLP (64×64)** | **CPU** (default `--device cpu`) | Tiny network — the 6750 XT does **not** help via CUDA. DirectML is optional and usually not worth it. |

**32 GB DDR4** is comfortable for **4** parallel browsers. Avoid **8** envs on a 3700X unless you are fine with high CPU load and longer track-menu waits.

## Setup

1. Install **Python 3.10+** (64-bit) from [python.org](https://www.python.org/downloads/windows/). Check “Add python.exe to PATH”.

2. Open **PowerShell** or **cmd** in the project folder (quote paths if the folder name has a space, e.g. `Polyplex_V1 PC`):

   ```bat
   cd "C:\path\to\Polyplex_V1 PC"
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   playwright install chromium
   ```

3. **Start training** (starts HTTP servers on **8080–8083**, then PPO):

   ```bat
   python run_local_training.py
   ```

   Useful variants:

   ```bat
   python run_local_training.py --num-envs 4 --watch
   python run_local_training.py --num-envs 2
   python run_local_training.py --num-envs 1 --vec-env dummy
   ```

### First-run expectations

- Monitor may sit at **0 steps** for **1–3 minutes** while each worker opens the track picker (`.track` HTTP loads).
- A Tkinter **GUI monitor** opens automatically (use `--no-gui` to suppress).
- **Ctrl+C** may print `KeyboardInterrupt`, `TargetClosedError`, `BrokenPipeError` — normal SubprocVecEnv shutdown noise. Check `logs/last_training_error.txt` for real failures.
- HTTP servers are **separate processes** and may keep running after training stops.

### If track menu times out

```bat
set POLYTRACK_TRACK_MENU_WAIT_MS=240000
python run_local_training.py --num-envs 2
```

Or debug with a visible browser:

```bat
python run_local_training.py --num-envs 1 --vec-env dummy --headed
```

Headless Windows already uses GPU/ANGLE (good for the 6750 XT). Only force software GL in VMs:

```bat
set POLYTRACK_HEADLESS_USE_SWIFTSHADER=1
```

## Options

| Goal | Command / env |
|------|----------------|
| **4 envs (default)** | `python run_local_training.py` |
| **Watch worker 0** | `python run_local_training.py --watch` |
| **No GUI monitor** | `python run_local_training.py --no-gui` |
| **Single-process debug** | `python run_local_training.py --num-envs 1 --vec-env dummy` |
| **Skip auto HTTP servers** | `set POLYTRACK_SKIP_SERVER_LAUNCH=1` then `python training\train_ppo.py` |
| **Worker stagger** | `POLYTRACK_WORKER_STAGGER_S` (default **2.5**) |
| **Track menu wait (ms)** | `POLYTRACK_TRACK_MENU_WAIT_MS` (default **180000**) |

## AMD RX 6750 XT and PyTorch

- CUDA PyTorch builds **do not** use AMD GPUs.
- Stock `requirements.txt` installs **CPU** PyTorch — correct for this project.
- Optional advanced: [`torch-directml`](https://pypi.org/project/torch-directml/) — not wired into `train_ppo.py`; the MLP is not the bottleneck.

## Stopping servers (ports 8080–8083)

```powershell
8080..8083 | ForEach-Object {
  Get-NetTCPConnection -LocalPort $_ -ErrorAction SilentlyContinue |
    Select-Object -ExpandProperty OwningProcess -Unique |
    ForEach-Object { Stop-Process -Id $_ -Force -ErrorAction SilentlyContinue }
}
```

## Hardware checklist (3700X / 6750 XT / 32 GB)

| Item | Status |
|------|--------|
| Default **4** envs | Matches CPU/RAM budget |
| Headless uses AMD GPU for WebGL | Yes (no SwiftShader by default) |
| PPO on CPU | Yes (`--device cpu`) |
| Windows `spawn` + picklable env factory | Yes (`utils/polytrack_env_factory.py` + `freeze_support`) |
| Soft KeyR resets (less Chromium thrash) | Yes |
| Evaluate / watch elites | `python evaluate.py --model checkpoints\best_model.zip` |

## Recommended first Windows command

```bat
cd "C:\path\to\Polyplex_V1 PC"
.venv\Scripts\activate
python run_local_training.py --num-envs 4
```

If the machine feels overloaded, drop to `--num-envs 2`. Use `--watch` only when you want one visible Chromium (adds GPU compositor load).
