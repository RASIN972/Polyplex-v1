"""Chromium flags for Polytrack + Playwright (headless RL, WebGL, audio).

WebGL console lines like "GPU stall due to ReadPixels" come from the driver when
something uses readPixels (Three.js / effects). They are performance hints, not
hard failures; ignore for RL unless you profile GPU cost.

Headless mode uses SwiftShader (software GL) so WebGL works without a GPU.
**Headed** mode must not force SwiftShader: eight parallel software renderers on
Windows often OOM or crash; use the default GPU/ANGLE stack instead.
"""

from __future__ import annotations

import os
import sys

# Shared by headed and headless (no --use-gl here; see polytrack_chromium_launch_args).
_POLYTRACK_CHROMIUM_LAUNCH_ARGS_BASE: tuple[str, ...] = (
    "--autoplay-policy=no-user-gesture-required",
    "--no-sandbox",
    "--disable-web-security",
    "--disable-features=BlockInsecurePrivateNetworkRequests",
    "--enable-webgl",
    "--ignore-gpu-blocklist",
    "--disable-gpu-sandbox",
    "--run-virtual-display-size-always",
    "--window-size=1280,720",
)

# Headless / CI-friendly: software GL so Chromium draws off-screen without a GPU.
_SWIFTSHADER_GL: tuple[str, ...] = ("--use-gl=swiftshader",)

# Often needed in Docker / low /dev/shm on Linux.
POLYTRACK_CHROMIUM_ARG_DISABLE_DEV_SHM: str = "--disable-dev-shm-usage"


def polytrack_chromium_launch_args(*, headless: bool) -> tuple[str, ...]:
    """Launch flags for Playwright Chromium.

    Headed: real GPU/ANGLE (no SwiftShader — eight parallel software renderers often OOM on Windows).

    Headless: **Linux / CI** defaults to SwiftShader so WebGL works without a GPU. **Windows**
    desktop defaults to **no** ``--use-gl=swiftshader`` so menus/WebGL follow the normal D3D/ANGLE
    stack (SwiftShader is a common cause of “track list never appears” vs headed). Override with
    ``POLYTRACK_HEADLESS_USE_SWIFTSHADER=1`` to force software GL on Windows (e.g. VM without GPU).
    """
    args = list(_POLYTRACK_CHROMIUM_LAUNCH_ARGS_BASE)
    if headless:
        use_sw = os.environ.get("POLYTRACK_HEADLESS_USE_SWIFTSHADER", "").strip().lower()
        force_swift = use_sw in ("1", "true", "yes")
        force_no_swift = use_sw in ("0", "false", "no")
        if force_swift:
            args.extend(_SWIFTSHADER_GL)
        elif force_no_swift:
            pass
        elif sys.platform.startswith("win32"):
            pass
        else:
            args.extend(_SWIFTSHADER_GL)
    if sys.platform.startswith("linux"):
        args.append(POLYTRACK_CHROMIUM_ARG_DISABLE_DEV_SHM)
    return tuple(args)


# Backward compat: default = headless (software GL).
POLYTRACK_CHROMIUM_LAUNCH_ARGS: tuple[str, ...] = polytrack_chromium_launch_args(
    headless=True
)

POLYTRACK_CHROMIUM_IGNORE_DEFAULT_ARGS: tuple[str, ...] = ("--disable-extensions",)
