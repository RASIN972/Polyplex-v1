"""Chromium flags for Polytrack + Playwright (headless RL, WebGL, audio).

WebGL console lines like "GPU stall due to ReadPixels" come from the driver when
something uses readPixels (Three.js / effects). They are performance hints, not
hard failures; ignore for RL unless you profile GPU cost.
"""

from __future__ import annotations

# Headless / CI-friendly: software GL, autoplay, relaxed sandbox for containers.
POLYTRACK_CHROMIUM_LAUNCH_ARGS: tuple[str, ...] = (
    "--autoplay-policy=no-user-gesture-required",
    "--no-sandbox",
    "--disable-web-security",
    "--disable-features=BlockInsecurePrivateNetworkRequests",
    "--use-gl=swiftshader",
    "--enable-webgl",
    "--ignore-gpu-blocklist",
    "--disable-gpu-sandbox",
    "--run-virtual-display-size-always",
    "--window-size=1280,720",
)

# Often needed in Docker / low /dev/shm; add to args when appropriate.
POLYTRACK_CHROMIUM_ARG_DISABLE_DEV_SHM: str = "--disable-dev-shm-usage"

POLYTRACK_CHROMIUM_IGNORE_DEFAULT_ARGS: tuple[str, ...] = ("--disable-extensions",)
