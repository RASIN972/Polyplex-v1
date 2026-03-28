"""Playwright request routing for local Polytrack (no external services)."""

from __future__ import annotations

from playwright.async_api import Page, Route


async def _abort(_route: Route) -> None:
    await _route.abort()


async def install_polytrack_offline_routes(page: Page) -> None:
    """Abort leaderboard/API calls to Kodub VPS (timeouts in local / air-gapped training)."""
    await page.route("https://vps.kodub.com/**", _abort)
    await page.route("http://vps.kodub.com/**", _abort)
