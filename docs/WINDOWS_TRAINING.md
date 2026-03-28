# Polyplex training on Windows (e.g. Ryzen 3700X + RX 6750 XT)

## What runs where

- **8× Chromium + Polytrack** (Playwright): heavy **CPU** load; **16 threads** on a 3700X helps.
- **PPO MLP (64×64)**: tiny network — **CPU training is normal**; the RX 6750 XT rarely speeds this up unless you use a special PyTorch build (see below).

## Setup

1. Install **Python 3.10+** from [python.org](https://www.python.org/downloads/windows/) (64-bit). Check “Add python.exe to PATH”.

2. Open **PowerShell** or **cmd** in the project folder:

   ```bat
   cd path\to\Polyplex_V1
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   playwright install chromium
   ```

3. **Parallel training** (default): starts servers on **8080–8087** and uses **SubprocVecEnv**:

   ```bat
   python run_local_training.py
   ```

   Or only training (after servers are up):

   ```bat
   python -m utils.launch_servers
   python training\train_ppo.py
   ```

## Options

| Goal | Command / env |
|------|----------------|
| **8 envs, subprocesses (default)** | `python training/train_ppo.py` |
| **Single-process debug (no multiprocessing)** | `python training\train_ppo.py --num-envs 1 --vec-env dummy` |
| **Skip auto-launch of HTTP servers** | `set POLYTRACK_SKIP_SERVER_LAUNCH=1` then run `run_local_training.py` |

## AMD RX 6750 XT and PyTorch

- **NVIDIA CUDA** builds of PyTorch **do not** use AMD GPUs.
- **Official PyTorch** on Windows is **CPU** or **CUDA (NVIDIA)** for typical `pip install torch`.
- For **GPU-accelerated** tensor ops on AMD under Windows, people sometimes use **[`torch-directml`](https://pypi.org/project/torch-directml/)** (DirectML). Stable-Baselines3 + DirectML can be finicky; for this project’s **small MLP**, **CPU PyTorch is usually enough** — the bottleneck is **8× browser + game**, not the network.

To try DirectML (optional, advanced):

```bat
pip install torch-directml
```

You would need to set the device in code (not enabled in the stock `train_ppo.py`). Ask to wire it in Agent mode if you want this.

## Stopping servers

Each server is a separate process. To free **8080–8087** (PowerShell):

```powershell
8080..8087 | ForEach-Object { Get-NetTCPConnection -LocalPort $_ -ErrorAction SilentlyContinue }
# Then stop the listed PIDs, or use Resource Monitor
```

Or close the terminal windows that started them.

## Ryzen 3700X tips

- Leave **headless** default (no `--headed`) so 8 browsers stay off the main GPU compositor workload where possible.
- If RAM is tight (~16 GB), try **`--num-envs 4`** and adjust `train_ppo.py` / ports (8080–8083) plus `launch_servers` / `run_local_training` port ranges accordingly (requires small code edits).
