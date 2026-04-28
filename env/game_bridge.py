"""Playwright (async) bridge to Polytrack: read vehicle state via injected JS, send WASD, reset (R).

The game bundle exposes ``window.__polytrackGhostData`` (same object as module ``ghostData``), which
the audio/physics path keeps updated with ``advancedCar`` — the Bullet vehicle (``em``) with
``getSpeedKmh``, ``getPosition``, ``getQuaternion``, ``getNextCheckpointIndex``, ``getTime``, etc.

Injected code maintains ``window.__rlState`` for RL / debugging. For the heavier post-finish DOM
recovery + KeyR flow, see ``FinishDebugGameBridge``.
"""

from __future__ import annotations

import os
import time
from typing import Any

from playwright.async_api import Browser, Page, Playwright, TimeoutError as PlaywrightTimeoutError, async_playwright

from env.chromium_launch_args import (
    POLYTRACK_CHROMIUM_IGNORE_DEFAULT_ARGS,
    polytrack_chromium_launch_args,
)
from env.debug_logging import agent_debug_log
from env.playwright_routes import install_polytrack_offline_routes

_PLAYWRIGHT_KEYCODE = {"w": "KeyW", "a": "KeyA", "s": "KeyS", "d": "KeyD"}

# --- RL: poll vehicle from window.__polytrackGhostData.advancedCar (see polytrack bundle) ---
_RL_INIT_JS = """
(() => {
  if (window.__rlState && window.__rlState._installed) return;

  const codeMap = { w: "KeyW", a: "KeyA", s: "KeyS", d: "KeyD" };

  function quatToEulerYXZ(x, y, z, w) {
    const THREE = window.__THREE__;
    if (THREE && THREE.Quaternion && THREE.Euler) {
      const q = new THREE.Quaternion(x, y, z, w);
      const e = new THREE.Euler().setFromQuaternion(q, "YXZ");
      return { x: e.x, y: e.y, z: e.z };
    }
    const sinp = 2 * (w * y - z * x);
    const pitch = Math.abs(sinp) >= 1 ? Math.sign(sinp) * (Math.PI / 2) : Math.asin(sinp);
    const roll = Math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y));
    const yaw = Math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z));
    return { x: pitch, y: yaw, z: roll };
  }

  window.__rlState = {
    _installed: true,
    ready: false,
    speed: 0,
    position: { x: 0, y: 0, z: 0 },
    rotation: { x: 0, y: 0, z: 0 },
    crashed_or_reset: false,
    checkpoint_index: 0,
    time_elapsed: 0,
    has_started: false,
    has_finished: false,
    car_present: false,
  };

  window.__rlBridge = window.__rlBridge || { held: new Set() };

  window.__rlBridge.applyKeys = (keys) => {
    const norm = keys.map((k) => String(k).toLowerCase());
    const want = new Set(norm);
    const held = window.__rlBridge.held;
    for (const k of [...held]) {
      if (!want.has(k)) {
        const code = codeMap[k];
        if (code) {
          window.dispatchEvent(
            new KeyboardEvent("keyup", { code, key: k, bubbles: true, cancelable: true })
          );
        }
        held.delete(k);
      }
    }
    for (const k of want) {
      if (!held.has(k)) {
        const code = codeMap[k];
        if (code) {
          window.dispatchEvent(
            new KeyboardEvent("keydown", { code, key: k, bubbles: true, cancelable: true })
          );
          held.add(k);
        }
      }
    }
  };

  window.__rlBridge.releaseAllKeys = () => {
    window.__rlBridge.applyKeys([]);
  };

  let lastCarRef = null;
  let lastTimeSec = -1;
  let lastCp = -1;
  let lastPos = null;
  let lastSpeed = null;
  let lastHasStarted = false;

  const TELEPORT_DIST = 38;
  const SUDDEN_STOP_SPEED_PRIOR = 55;
  const SUDDEN_STOP_SPEED_AFTER = 6;
  const SUDDEN_STOP_MAX_DIST = 14;

  function tick() {
    const gd = window.__polytrackGhostData;
    const car = gd && gd.advancedCar;
    const st = window.__rlState;

    let resetEdge = false;
    if (lastCarRef !== null && car !== null && car !== lastCarRef) {
      resetEdge = true;
      lastPos = null;
      lastSpeed = null;
    }
    if (car) lastCarRef = car;

    if (!car) {
      st.ready = !!gd;
      st.car_present = false;
      st.crashed_or_reset = resetEdge;
      lastPos = null;
      lastSpeed = null;
      lastHasStarted = false;
      requestAnimationFrame(tick);
      return;
    }

    st.car_present = true;
    st.ready = true;

    let timeSec = 0;
    try {
      const t = car.getTime && car.getTime();
      timeSec = t && typeof t.time === "number" ? t.time : 0;
    } catch (e) {
      timeSec = 0;
    }

    let cp = 0;
    try {
      cp = car.getNextCheckpointIndex ? car.getNextCheckpointIndex() : 0;
    } catch (e) {
      cp = 0;
    }

    if (lastTimeSec >= 0 && timeSec + 0.35 < lastTimeSec) resetEdge = true;
    if (lastCp >= 0 && cp < lastCp && lastCp > 0 && timeSec < 2) resetEdge = true;

    lastTimeSec = timeSec;
    lastCp = cp;

    let pos = { x: 0, y: 0, z: 0 };
    let quat = { x: 0, y: 0, z: 0, w: 1 };
    let speed = 0;
    let hasStarted = false;
    let hasFinished = false;
    try {
      const p = car.getPosition();
      if (p) pos = { x: p.x, y: p.y, z: p.z };
    } catch (e) {}
    try {
      const q = car.getQuaternion();
      if (q) quat = { x: q.x, y: q.y, z: q.z, w: q.w };
    } catch (e) {}
    try {
      speed = car.getSpeedKmh ? car.getSpeedKmh() : 0;
    } catch (e) {}
    try {
      hasStarted = car.hasStarted ? car.hasStarted() : false;
    } catch (e) {}
    try {
      hasFinished = car.hasFinished ? car.hasFinished() : false;
    } catch (e) {}

    if (lastHasStarted && hasStarted && lastPos) {
      const dx = pos.x - lastPos.x;
      const dy = pos.y - lastPos.y;
      const dz = pos.z - lastPos.z;
      const dist = Math.hypot(dx, dy, dz);
      if (dist > TELEPORT_DIST) resetEdge = true;
      if (
        lastSpeed !== null &&
        Math.abs(lastSpeed) > SUDDEN_STOP_SPEED_PRIOR &&
        Math.abs(speed) < SUDDEN_STOP_SPEED_AFTER &&
        dist < SUDDEN_STOP_MAX_DIST
      ) {
        resetEdge = true;
      }
    }

    if (hasStarted) {
      lastPos = { x: pos.x, y: pos.y, z: pos.z };
      lastSpeed = speed;
    } else {
      lastPos = null;
      lastSpeed = null;
    }
    lastHasStarted = hasStarted;

    const euler = quatToEulerYXZ(quat.x, quat.y, quat.z, quat.w);

    st.speed = speed;
    st.position = pos;
    st.rotation = euler;
    st.crashed_or_reset = resetEdge;
    st.checkpoint_index = cp;
    st.time_elapsed = timeSec;
    st.has_started = hasStarted;
    st.has_finished = hasFinished;

    requestAnimationFrame(tick);
  }

  requestAnimationFrame(tick);
})();
"""


# Snapshot DOM / focus / canvas — tests overlay, focus, visibility, pointer lock (hypotheses H1–H5).
_PROBE_JS = """
() => {
  const ui = document.getElementById("ui");
  const ann = [...document.querySelectorAll(".time-announcer")];
  const active = document.activeElement;
  let blockingLargeAuto = 0;
  const blockingSamples = [];
  if (ui) {
    ui.querySelectorAll("*").forEach((el) => {
      const cs = getComputedStyle(el);
      if (cs.pointerEvents !== "auto") return;
      const r = el.getBoundingClientRect();
      if (r.width > innerWidth * 0.45 && r.height > innerHeight * 0.45) {
        blockingLargeAuto++;
        if (blockingSamples.length < 8) {
          blockingSamples.push({
            tag: el.tagName,
            id: el.id || "",
            className: (el.className && String(el.className).slice(0, 120)) || "",
            zIndex: cs.zIndex,
          });
        }
      }
    });
  }
  const canvases = [...document.querySelectorAll("canvas")].map((c) => {
    const cs = getComputedStyle(c);
    const r = c.getBoundingClientRect();
    return {
      w: Math.round(r.width),
      h: Math.round(r.height),
      display: cs.display,
      visibility: cs.visibility,
      opacity: cs.opacity,
    };
  });
  return {
    timeAnnouncerCount: ann.length,
    timeAnnouncerPointerEvents: ann.map((el) => getComputedStyle(el).pointerEvents),
    blockingLargePointerEventsAuto: blockingLargeAuto,
    blockingSamples,
    activeTag: active ? active.tagName : null,
    activeId: active && active.id ? active.id : "",
    activeClassName: active && active.className
      ? String(active.className).slice(0, 160)
      : "",
    canvasCount: canvases.length,
    canvases,
    pointerLockElement: document.pointerLockElement
      ? document.pointerLockElement.tagName
      : null,
    bodyPointerEvents: getComputedStyle(document.body).pointerEvents,
    hiddenClassCount: document.querySelectorAll(".hidden").length,
  };
}
"""


async def collect_dom_probe(page: Page) -> dict[str, Any]:
    return await page.evaluate(_PROBE_JS)


_RECOVERY_JS = """
() => {
  const hadLock = !!document.pointerLockElement;
  if (hadLock) {
    document.exitPointerLock();
  }
  window.focus();
  const c = document.querySelector("canvas");
  if (c) {
    try {
      c.focus();
    } catch (e) {}
  }
  return { hadPointerLock: hadLock };
}
"""

_RESUME_AUDIO_CONTEXT_JS = """
async () => {
  const out = {
    candidates: [],
    resumed: [],
    howlerType: typeof window.Howler,
  };
  const add = (ctx, name) => {
    if (!ctx || typeof ctx.resume !== "function") return;
    const state = ctx.state;
    out.candidates.push({ name, state });
  };
  const tryResume = async (ctx, name) => {
    if (!ctx || typeof ctx.resume !== "function") return;
    if (ctx.state === "suspended") {
      try {
        await ctx.resume();
        out.resumed.push(name);
      } catch (e) {
        out.resumed.push(name + ":error");
      }
    }
  };
  const H = window.Howler;
  if (H) {
    add(H.ctx, "Howler.ctx");
    add(H._audioContext, "Howler._audioContext");
  }
  add(window._audioContext, "window._audioContext");
  add(window.audioContext, "window.audioContext");
  if (H && H.ctx) await tryResume(H.ctx, "Howler.ctx");
  if (H && H._audioContext) await tryResume(H._audioContext, "Howler._audioContext");
  await tryResume(window._audioContext, "window._audioContext");
  await tryResume(window.audioContext, "window.audioContext");
  return out;
}
"""


class GameBridge:
    """Headless Polytrack control: ``__rlState`` polling, WASD, KeyR reset."""

    def __init__(
        self,
        page: Page,
        *,
        _playwright: Playwright | None = None,
        _browser: Browser | None = None,
    ) -> None:
        self._page = page
        self._playwright = _playwright
        self._browser = _browser
        self._rl_installed = False
        self._keyboard_held: set[str] = set()

    @classmethod
    async def launch(
        cls,
        url: str = "http://127.0.0.1:8080/",
        *,
        headless: bool = True,
        viewport: dict[str, int] | None = None,
    ) -> GameBridge:
        """Start Chromium, open Polytrack, return a bridge. Call ``await bridge.close()`` when done."""
        pw = await async_playwright().start()
        browser = await pw.chromium.launch(
            headless=headless,
            args=list(polytrack_chromium_launch_args(headless=headless)),
            ignore_default_args=list(POLYTRACK_CHROMIUM_IGNORE_DEFAULT_ARGS),
        )
        page = await browser.new_page(viewport=viewport or {"width": 1280, "height": 720})
        await install_polytrack_offline_routes(page)
        bridge = cls(page, _playwright=pw, _browser=browser)
        await page.goto(url, wait_until="domcontentloaded", timeout=60_000)
        try:
            await page.wait_for_load_state("load", timeout=60_000)
        except PlaywrightTimeoutError:
            pass
        await bridge._ensure_rl_harness()
        return bridge

    async def _ensure_rl_harness(self) -> None:
        if self._rl_installed:
            return
        await self._page.wait_for_function(
            "() => window.__polytrackGhostData !== undefined",
            timeout=120_000,
        )
        await self._page.evaluate(_RL_INIT_JS)
        self._rl_installed = True

    async def get_state(self) -> dict[str, Any]:
        """Return a JSON-serializable snapshot of ``window.__rlState`` (game fields)."""
        await self._ensure_rl_harness()
        return await self._page.evaluate(
            """() => {
          const s = window.__rlState;
          if (!s) return { error: "no __rlState" };
          return {
            ready: s.ready,
            speed: s.speed,
            position: { ...s.position },
            rotation: { ...s.rotation },
            crashed_or_reset: s.crashed_or_reset,
            checkpoint_index: s.checkpoint_index,
            time_elapsed: s.time_elapsed,
            has_started: s.has_started,
            has_finished: s.has_finished,
            car_present: s.car_present,
          };
        }"""
        )

    async def _release_playwright_keys(self) -> None:
        for k in list(self._keyboard_held):
            code = _PLAYWRIGHT_KEYCODE.get(k)
            if code:
                try:
                    await self._page.keyboard.up(code)
                except Exception:
                    pass
        self._keyboard_held.clear()

    async def send_action(self, keys: list[str]) -> None:
        """Hold exactly the given WASD keys via Playwright (trusted input), like ``KeyR`` reset."""
        await self._ensure_rl_harness()
        want = {str(x).lower() for x in keys if str(x).lower() in _PLAYWRIGHT_KEYCODE}
        try:
            await self._page.focus("canvas#screen", timeout=3000)
        except PlaywrightTimeoutError:
            pass
        prev = self._keyboard_held
        for k in prev - want:
            code = _PLAYWRIGHT_KEYCODE[k]
            try:
                await self._page.keyboard.up(code)
            except Exception:
                pass
        for k in want - prev:
            code = _PLAYWRIGHT_KEYCODE[k]
            try:
                await self._page.keyboard.down(code)
            except Exception:
                pass
        self._keyboard_held = set(want)

    async def reset(self) -> None:
        """Press R (vehicle reset / restart in-game)."""
        await self._page.keyboard.press("KeyR")

    async def restart_session(
        self,
        url: str,
        *,
        headless: bool = True,
        viewport: dict[str, int] | None = None,
    ) -> None:
        await self.close()
        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(
            headless=headless,
            args=list(polytrack_chromium_launch_args(headless=headless)),
            ignore_default_args=list(POLYTRACK_CHROMIUM_IGNORE_DEFAULT_ARGS),
        )
        self._page = await self._browser.new_page(viewport=viewport or {"width": 1280, "height": 720})
        await install_polytrack_offline_routes(self._page)
        self._rl_installed = False
        await self._page.goto(url, wait_until="domcontentloaded", timeout=60_000)
        try:
            await self._page.wait_for_load_state("load", timeout=60_000)
        except PlaywrightTimeoutError:
            pass
        await self._ensure_rl_harness()

    async def _dismiss_message_boxes_js(self) -> bool:
        """Click Ok on open #ui .message-box dialogs using the DOM (not Playwright click).

        Polytrack's `.message` boxes hide the first button with CSS (`display: none`); the real Ok is
        the second button. Headless/Windows often report is_visible() incorrectly on the overlay, so
        we use getComputedStyle and .click() in-page. Also required before main-menu Play: token /
        leaderboard failures call show(); until Ok, menu buttons stay class ``hidden`` and the Play
        locator never matches.
        """
        clicked = await self._page.evaluate(
            """() => {
            const root = document.getElementById("ui");
            if (!root) return false;
            for (const box of root.querySelectorAll(".message-box")) {
                if (box.classList.contains("hidden")) continue;
                const boxInner = box.querySelector(".box");
                if (!boxInner) continue;
                const buttons = boxInner.querySelectorAll("button");
                for (let j = buttons.length - 1; j >= 0; j--) {
                    const b = buttons[j];
                    if (window.getComputedStyle(b).display === "none") continue;
                    b.click();
                    return true;
                }
            }
            return false;
        }"""
        )
        if clicked:
            await self._page.wait_for_timeout(400)
        return bool(clicked)

    async def _dismiss_blocking_message_boxes(self) -> bool:
        """Dismiss #ui .message-box overlays. CSS often hides the first .message button; some dialogs
        only have one button (nth(1) was wrong). Clicks the last visible button per box, then Esc.
        """
        any_clicked = False
        boxes = self._page.locator("#ui .message-box")
        n = await boxes.count()
        for i in range(n):
            box = boxes.nth(i)
            try:
                if not await box.is_visible():
                    continue
            except Exception:
                continue
            btns = box.locator(".box button")
            btn_count = await btns.count()
            for j in range(btn_count - 1, -1, -1):
                btn = btns.nth(j)
                try:
                    if not await btn.is_visible():
                        continue
                except Exception:
                    continue
                try:
                    await btn.click(
                        timeout=5000, force=True, no_wait_after=True
                    )
                    any_clicked = True
                    await self._page.wait_for_timeout(400)
                    break
                except PlaywrightTimeoutError:
                    try:
                        await btn.evaluate("el => el.click()")
                        any_clicked = True
                        await self._page.wait_for_timeout(400)
                        break
                    except Exception:
                        pass
        if not any_clicked:
            try:
                await self._page.keyboard.press("Escape")
                await self._page.wait_for_timeout(250)
            except Exception:
                pass
        return any_clicked

    def _menu_play_wait_timeout_s(self) -> float:
        return float(
            os.environ.get("POLYTRACK_MENU_WAIT_S", "300")
        )

    async def _wait_until_play_visible(self, timeout_s: float | None = None) -> None:
        if timeout_s is None:
            timeout_s = self._menu_play_wait_timeout_s()
        play = self._page.locator(
            '#ui .menu button.button-image:has(img[src*="play.svg"])'
        )
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            await self._dismiss_message_boxes_js()
            await self._dismiss_blocking_message_boxes()
            if await play.is_visible():
                return
            await self._page.wait_for_timeout(350)
        raise RuntimeError(
            "main menu: Play never became visible — dismiss any #ui .message-box "
            "dialogs (e.g. token / leaderboard) or wait for loading to finish. "
            "Try fewer envs, POLYTRACK_MENU_WAIT_S=600, or --headed to see the UI."
        )

    async def _reliable_menu_click(self, locator, *, timeout_ms: int = 60_000) -> None:
        """Headless Chromium on Windows often hangs in locator.click() after 'performing click action'.

        force + no_wait_after avoids stuck actionability / post-click waits; JS click is a fallback.
        """
        try:
            await locator.click(
                timeout=timeout_ms,
                force=True,
                no_wait_after=True,
            )
        except PlaywrightTimeoutError:
            await locator.evaluate("el => el.click()")

    async def start_track_menu_index(self, index: int = 0) -> None:
        await self._page.evaluate(_RECOVERY_JS)
        try:
            await self._page.wait_for_function(
                "() => document.pointerLockElement === null",
                timeout=2000,
            )
        except PlaywrightTimeoutError:
            pass
        try:
            await self._page.click("canvas#screen", timeout=5000)
        except PlaywrightTimeoutError:
            pass
        await self._page.evaluate(_RESUME_AUDIO_CONTEXT_JS)
        await self._wait_until_play_visible()
        play = self._page.locator(
            '#ui .menu button.button-image:has(img[src*="play.svg"])'
        )
        await self._reliable_menu_click(play)
        sel = "#ui .menu .track-selection .tracks-container .track button"
        try:
            await self._page.wait_for_selector(sel, state="visible", timeout=120_000)
        except PlaywrightTimeoutError as e:
            raise RuntimeError(
                "track menu: timed out waiting for track list after Play"
            ) from e
        buttons = self._page.locator(sel)
        n = await buttons.count()
        if n == 0:
            raise RuntimeError("track menu: no track buttons (empty list?)")
        if index < 0 or index >= n:
            raise RuntimeError(f"track menu: index {index} out of range (count={n})")
        target = buttons.nth(index)
        await target.scroll_into_view_if_needed()
        await self._reliable_menu_click(target)
        track_info_play = self._page.locator(
            "#ui .menu .track-info .side-panel button.button.play"
        )
        try:
            await track_info_play.wait_for(state="visible", timeout=60_000)
        except PlaywrightTimeoutError as e:
            raise RuntimeError(
                "track menu: track-info Play button did not appear after picking a track"
            ) from e
        await track_info_play.scroll_into_view_if_needed()
        await self._reliable_menu_click(track_info_play)
        await self._page.wait_for_timeout(800)
        try:
            await self._page.bring_to_front()
        except PlaywrightTimeoutError:
            pass
        await self._page.wait_for_timeout(400)
        await self.nudge_race_start()

    async def nudge_race_start(self) -> None:
        """Focus the game canvas and press Space (countdown / unpause often needs trusted focus)."""
        try:
            await self._page.evaluate(_RECOVERY_JS)
        except Exception:
            pass
        try:
            await self._page.focus("canvas#screen", timeout=8000)
        except PlaywrightTimeoutError:
            pass
        for _ in range(3):
            try:
                await self._page.click("canvas#screen", timeout=5000)
            except PlaywrightTimeoutError:
                pass
            for key in ("Space", "Enter"):
                try:
                    await self._page.keyboard.press(key)
                except PlaywrightTimeoutError:
                    pass
            await self._page.wait_for_timeout(200)

    async def close(self) -> None:
        """Release held keys, close the browser, and stop Playwright (when launched via ``launch``)."""
        try:
            await self._release_playwright_keys()
        except Exception:
            pass
        try:
            await self._page.evaluate(
                "() => { if (window.__rlBridge && window.__rlBridge.releaseAllKeys) window.__rlBridge.releaseAllKeys(); }"
            )
        except Exception:
            pass
        if self._browser is not None:
            await self._browser.close()
        if self._playwright is not None:
            await self._playwright.stop()


class FinishDebugGameBridge:
    """Recover from post-finish stuck state, vehicle reset (KeyR), then track picker if shown."""

    def __init__(
        self,
        page: Page,
        *,
        reenter_track_index: int | None = 0,
    ) -> None:
        self._page = page
        self._reenter_track_index = reenter_track_index

    async def _try_reenter_from_track_picker(self, *, run_id: str) -> None:
        if self._reenter_track_index is None:
            agent_debug_log(
                run_id=run_id,
                hypothesis_id="H-track-ui",
                location="game_bridge.py:_try_reenter_from_track_picker",
                message="track_picker_skip",
                data={"reason": "reenter_track_index_is_none"},
            )
            return
        idx = self._reenter_track_index
        buttons = self._page.locator(
            "#ui .menu .track-selection .tracks-container .track button"
        )
        n = await buttons.count()
        if n == 0:
            agent_debug_log(
                run_id=run_id,
                hypothesis_id="H-track-ui",
                location="game_bridge.py:_try_reenter_from_track_picker",
                message="track_picker_reenter",
                data={"clicked": False, "reason": "no_track_buttons", "count": 0},
            )
            return
        if idx >= n:
            agent_debug_log(
                run_id=run_id,
                hypothesis_id="H-track-ui",
                location="game_bridge.py:_try_reenter_from_track_picker",
                message="track_picker_reenter",
                data={"clicked": False, "reason": "index_out_of_range", "count": n, "index": idx},
            )
            return
        try:
            target = buttons.nth(idx)
            if not await target.is_visible():
                agent_debug_log(
                    run_id=run_id,
                    hypothesis_id="H-track-ui",
                    location="game_bridge.py:_try_reenter_from_track_picker",
                    message="track_picker_reenter",
                    data={"clicked": False, "reason": "button_not_visible", "count": n, "index": idx},
                )
                return
            await target.click(timeout=5000)
            await self._page.wait_for_timeout(600)
            agent_debug_log(
                run_id=run_id,
                hypothesis_id="H-track-ui",
                location="game_bridge.py:_try_reenter_from_track_picker",
                message="track_picker_reenter",
                data={"clicked": True, "count": n, "index": idx},
            )
        except PlaywrightTimeoutError:
            agent_debug_log(
                run_id=run_id,
                hypothesis_id="H-track-ui",
                location="game_bridge.py:_try_reenter_from_track_picker",
                message="track_picker_reenter",
                data={"clicked": False, "reason": "click_timeout", "count": n, "index": idx},
            )

    async def reset(self, *, run_id: str = "env-reset") -> None:
        agent_debug_log(
            run_id=run_id,
            hypothesis_id="H0",
            location="game_bridge.py:FinishDebugGameBridge.reset",
            message="reset_enter",
            data={},
        )
        before = await collect_dom_probe(self._page)
        agent_debug_log(
            run_id=run_id,
            hypothesis_id="H1-H5",
            location="game_bridge.py:FinishDebugGameBridge.reset",
            message="probe_before_keyr",
            data=before,
        )
        recovery = await self._page.evaluate(_RECOVERY_JS)
        agent_debug_log(
            run_id=run_id,
            hypothesis_id="H5",
            location="game_bridge.py:FinishDebugGameBridge.reset",
            message="post_finish_recovery_js",
            data=recovery,
        )
        try:
            await self._page.wait_for_function(
                "() => document.pointerLockElement === null",
                timeout=2000,
            )
        except PlaywrightTimeoutError:
            pass
        try:
            await self._page.click("canvas", timeout=5000)
        except PlaywrightTimeoutError:
            pass
        audio_out = await self._page.evaluate(_RESUME_AUDIO_CONTEXT_JS)
        agent_debug_log(
            run_id=run_id,
            hypothesis_id="H-audio",
            location="game_bridge.py:FinishDebugGameBridge.reset",
            message="audio_context_resume",
            data=audio_out,
        )
        await self._page.keyboard.press("KeyR")
        await self._page.wait_for_timeout(500)
        await self._try_reenter_from_track_picker(run_id=run_id)
        after = await collect_dom_probe(self._page)
        agent_debug_log(
            run_id=run_id,
            hypothesis_id="H1-H5",
            location="game_bridge.py:FinishDebugGameBridge.reset",
            message="probe_after_reset",
            data=after,
        )
