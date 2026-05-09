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

   **Startup timing:** The monitor can sit at **0 steps** while the **track picker** loads each track from the server (`.track` files over HTTP). The bridge waits until the in-menu **loading** UI is gone and at least one row button lays out (**~3 minutes per attempt** by default; × retries). Increase `POLYTRACK_TRACK_MENU_WAIT_MS` if your disk or CPU is busy with **8** parallel games.

   **If training stops with `RuntimeError: track menu: timed out waiting for track list after Play`:** Playwright never saw track rows as ready. **Headless on Windows** now skips SwiftShader by default (same GL path as headed); if you need software-only rendering in a VM, set `POLYTRACK_HEADLESS_USE_SWIFTSHADER=1`. Try `--headed` to confirm the UI loads. Other causes: game not fully loaded, dialog blocking Play, or too many parallel browsers — use `--num-envs 1 --vec-env dummy`, check `logs/polytrack_http_server_*.log`, open `http://127.0.0.1:8080/` manually.

   **Ctrl+C / stopping:** You may see `KeyboardInterrupt`, `TargetClosedError: ... browser has been closed`, then `BrokenPipeError` / `EOFError` from `SubprocVecEnv` while the parent process exits — that is shutdown noise after interrupt or after a worker dies. Full tracebacks from the main training failure are written to `logs/last_training_error.txt`. HTTP servers are separate processes and may need to be stopped manually (see below).

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
| **Stagger parallel env workers (seconds × worker index)** | `POLYTRACK_WORKER_STAGGER_S` (default **2.5**; set **0** to disable) |
| **Track menu: wait per attempt (ms)** | `POLYTRACK_TRACK_MENU_WAIT_MS` (default **180000** — track `.poly`/HTTP loads can be slow) |
| **Track menu: min rows before “ready”** | `POLYTRACK_TRACK_MENU_MIN_TRACKS` (default **1**) |
| **Track menu: retry open Play → list** | `POLYTRACK_TRACK_MENU_ATTEMPTS` (default **4**) |
| **Headless Windows: force software GL (SwiftShader)** | `POLYTRACK_HEADLESS_USE_SWIFTSHADER=1` (default on Windows is **off** — uses GPU/ANGLE like headed; set **1** on GPU-less VMs) |

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
