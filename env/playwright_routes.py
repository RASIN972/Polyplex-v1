"""Playwright request routing for local Polytrack (no external services)."""

from __future__ import annotations

from playwright.async_api import Page, Route


async def _abort(_route: Route) -> None:
    await _route.abort()


async def install_polytrack_offline_routes(page: Page) -> None:
    """Optionally abort Kodub VPS HTTP(S) requests (offline / flaky-network training).

    By default Playwright reaches the live VPS normally. Polytrack may call VPS for
    leaderboards/track metadata — blocking every ``vps.kodub.com`` request was too
    aggressive and can prevent the Play → track-picker UI from populating even when
    local ``polytrackcopy`` assets are intact.

    Set ``POLYTRACK_BLOCK_REMOTE=1`` to restore strict blocking (offline / CI).
    """

    import os

    flag = os.environ.get("POLYTRACK_BLOCK_REMOTE", "").strip().lower()
    if flag not in ("1", "true", "yes", "on"):
        return

    await page.route("https://vps.kodub.com/**", _abort)
    await page.route("http://vps.kodub.com/**", _abort)
