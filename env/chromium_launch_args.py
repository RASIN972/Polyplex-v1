"""Chromium flags for Polytrack + Playwright (headless RL, WebGL, audio).

WebGL console lines like "GPU stall due to ReadPixels" come from the driver when
something uses readPixels (Three.js / effects). They are performance hints, not
hard failures; ignore for RL unless you profile GPU cost.

Headless mode uses SwiftShader (software GL) so WebGL works without a GPU.
**Headed** mode must not force SwiftShader: eight parallel software renderers on
Windows often OOM or crash; use the default GPU/ANGLE stack instead.
"""

from __future__ import annotations

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
    """Launch flags for Playwright Chromium. Headed uses GPU; headless uses SwiftShader."""
    args = list(_POLYTRACK_CHROMIUM_LAUNCH_ARGS_BASE)
    if headless:
        args.extend(_SWIFTSHADER_GL)
    if sys.platform.startswith("linux"):
        args.append(POLYTRACK_CHROMIUM_ARG_DISABLE_DEV_SHM)
    return tuple(args)


# Backward compat: default = headless (software GL).
POLYTRACK_CHROMIUM_LAUNCH_ARGS: tuple[str, ...] = polytrack_chromium_launch_args(
    headless=True
)

POLYTRACK_CHROMIUM_IGNORE_DEFAULT_ARGS: tuple[str, ...] = ("--disable-extensions",)
