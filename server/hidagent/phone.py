"""Touch-like gestures on the phone, built from relative mouse moves.

The pointer is relative, so the server tracks where it is (in mouse units) and moves
from there; every few gestures it slams it into the top-left corner (`home`) again to
cancel drift. Coordinates are fractions of the screen (0..1, origin top-left), so they
work for any frame size.
"""

import asyncio
import json
import logging
import time
from collections import deque
from pathlib import Path

from . import vision
from .calibration import Axis, Calibration
from .device import Device, DeviceError

log = logging.getLogger(__name__)

CALIBRATION_MOVE = 200  # mouse units moved diagonally for a manual (one point) calibration; stays on screen
CALIBRATION_STOPS = (150, 350)  # automatic calibration finds the pointer at these distances from the corner
REHOME_EVERY = 10       # gestures between re-homing
HOLD = 0.25             # seconds to rest before lifting, so iOS adds no momentum (the swipe stops where it ends)
TAP_GAP = 0.15          # seconds to stay still after a click, so iOS sees a finished tap, not the start of a drag
NUDGE = 12              # mouse units the pointer is wiggled by to find it on screen
AIM_TOLERANCE = 0.008   # screen fraction; a bigger miss is corrected before pressing
MAX_MISS = 0.06         # bigger misses are still corrected, but not learned from (they may be misreadings)


class Phone:
    def __init__(self, device: Device, calibration_file: Path, screen=None):
        self.device, self._file, self.screen = device, calibration_file, screen
        # mouse units needed to cross the full screen width / height
        saved = json.loads(calibration_file.read_text()) if calibration_file.exists() else None
        self.span = {"w": saved["w"], "h": saved["h"]} if saved else None
        self._pos = None  # pointer position in mouse units; None = unknown
        self._moves = 0
        self._link = 0  # device.links when the pointer was last homed
        self._baseline = (saved or {}).get("baseline")
        self._bound = None  # a saved fit is a seed, not proof about this connection
        self._calibration_job = None
        self._cancel_requested = False
        self._paused_view = None
        self._dragging = False
        self._home_link, self._home_supported = None, None
        self.baseline_phase, self.baseline_error, self.baseline_progress = "waiting", "", ""
        # where the pointer really landed (seen on screen) for the screen fraction we aimed at, per axis;
        # a straight-line fit through them corrects every later move, so taps get better as the agent works
        self._seen = {axis: deque(map(tuple, (saved or {}).get("seen", {}).get(axis, [])), maxlen=12)
                      for axis in ("x", "y")}

    @property
    def calibrated(self) -> bool:
        if self.screen is None:
            return self.span is not None
        return bool(self._baseline and self.screen.live and self._bound == self.signature())

    def signature(self):
        if self.screen is None:
            return (self.device.links, None, None, None)
        return (self.device.links, getattr(self.screen, "source_epoch", 0), self.screen.geometry,
                getattr(self.screen, "source_key", None))

    def baseline_status(self):
        phase = self.baseline_phase
        if phase == "ready" and not self.calibrated:
            phase = "stale"
        return {"phase": phase, "progress": self.baseline_progress, "error": self.baseline_error,
                "geometry": self.screen.geometry if self.screen else None,
                "source": getattr(self.screen, "source", {}),
                "checks": (self._baseline or {}).get("checks", []) if self.calibrated else []}

    def cancel_calibration(self):
        self._cancel_requested = True
        self._paused_view = self.signature()[1:]
        if self._calibration_job:
            self._calibration_job.cancel()

    @property
    def calibration_paused(self):
        return self._paused_view == self.signature()[1:]

    async def start_calibration(self) -> bool:
        """Fit multiple observed positions and verify held-out points, including the bottom edge."""
        self._bound, self._pos = None, None
        self.baseline_phase, self.baseline_error = "calibrating", ""
        self._cancel_requested = False
        self._paused_view = None
        signature = self.signature()
        if self.screen is None or not self.screen.live or not self.screen.geometry:
            self.baseline_phase, self.baseline_error = "failed", "no live screen"
            return False
        geometry = self.screen.geometry
        same_geometry = (self._baseline or {}).get("geometry") == list(geometry)
        seed = (self.span["w"], self.span["h"]) if self.span and same_geometry else tuple(v * 1.1 for v in geometry)
        calibration = Calibration(self.device, self.screen, seed,
                                  lambda stage: setattr(self, "baseline_progress", stage))
        self._calibration_job = asyncio.create_task(calibration.run())
        try:
            baseline = await self._calibration_job
            if signature != self.signature():
                raise RuntimeError("device connection changed during calibration")
            baseline["source"] = getattr(self.screen, "source", {})
            self._baseline, self._bound = baseline, signature
            self.span = {k: round(baseline["axes"][i]["span"]) for i, k in enumerate(("w", "h"))}
            self._seen = {axis: deque(maxlen=12) for axis in ("x", "y")}
            self._pos, self._link, self._moves = calibration.pos, self.device.links, 0
            self.baseline_phase = "ready"
            self._save()
            log.info("baseline verified: geometry %s, span %s, checks %s", geometry, self.span, baseline["checks"])
            return True
        except asyncio.CancelledError:
            self.baseline_phase, self._pos = "cancelled", None
            if not self._cancel_requested:
                raise
            return False
        except Exception as error:
            self.baseline_phase, self.baseline_error, self._pos = "failed", str(error), None
            log.warning("baseline not verified: %s", error)
            return False
        finally:
            self._calibration_job = None
            if self.device.connected:
                try:
                    await self.device.send("release")
                except RuntimeError:
                    pass

    async def _find_pointer(self) -> tuple[float, float] | None:
        """Wiggle the pointer and find it anywhere on screen (the span may still be unknown)."""
        await self.screen.settled(quiet=0.2, least=0.3, most=1.5)
        before, sent = self.screen.frame, time.time()
        await self.device.send(f"move {NUDGE} 0")
        after = await self.screen.fresh(sent, 1.0)
        if after:
            await self.screen.settled(quiet=0.15, least=0.1, most=0.6)
            after = self.screen.frame
        await self.device.send(f"move -{NUDGE} 0")
        found = before and after and vision.pointer(before, after, (0.5, 0.5), 0.025, reach=(0.5, 0.5))
        return found if found and 0.03 < found[0] < 0.97 and 0.03 < found[1] < 0.97 else None

    def finish_calibration(self, x: float, y: float) -> None:
        """x, y: where the pointer ended up after start_calibration() (one point, picked by the user)."""
        self.span = {"w": round(CALIBRATION_MOVE / x), "h": round(CALIBRATION_MOVE / y)}
        self._baseline, self._bound = None, None
        for samples in self._seen.values():
            samples.clear()
        self._save()

    def _save(self) -> None:
        """Calibration plus the observed landings, so the correction survives a restart."""
        self._file.write_text(json.dumps({**self.span, "baseline": self._baseline,
                                         "seen": {k: list(v) for k, v in self._seen.items()}}))

    async def point(self, x: float, y: float, aim: bool = False) -> None:
        if not self.calibrated:
            raise RuntimeError("pointer not calibrated")
        await self._move_to(self._dx(x), self._dy(y))
        if aim and self.screen is not None:
            await self._aim(x, y)

    async def _aim(self, x: float, y: float) -> None:
        """Closed loop: find the real pointer on screen, fix a miss, and learn the error for later moves."""
        nudge = NUDGE if x < 0.9 else -NUDGE  # stay clear of the screen edge
        await self.screen.settled(quiet=0.15, least=0.1, most=0.6)  # pointer at rest
        before, sent = self.screen.frame, time.time()
        await self.device.send(f"move {nudge} 0")
        after = await self.screen.fresh(sent, 0.6)
        if after:
            await self.screen.settled(quiet=0.1, least=0.05, most=0.4)
            after = self.screen.frame
        await self.device.send(f"move {-nudge} 0")
        shift = nudge / self.span["w"]
        found = before and after and (vision.pointer(before, after, (x, y), shift)  # near the target first,
                                      or vision.pointer(before, after, (0.5, 0.5), shift, reach=(0.5, 0.5)))  # anywhere
        if not found:
            log.debug("aim (%.3f, %.3f): pointer not found (%s)", x, y,
                      "no new frame after the nudge" if not after else "no pointer-sized change near the target")
            return
        fx, fy = found
        log.debug("aim (%.3f, %.3f): pointer at (%.3f, %.3f)", x, y, fx, fy)
        if not self._baseline and max(abs(fx - x), abs(fy - y)) <= MAX_MISS:
            self._seen["x"].append((self._fix(x, "x"), fx))
            self._seen["y"].append((self._fix(y, "y"), fy))
            self._save()
        if abs(fx - x) <= AIM_TOLERANCE and abs(fy - y) <= AIM_TOLERANCE:
            return
        dx, dy = round((x - fx) * self.span["w"]), round((y - fy) * self.span["h"])  # fix this miss right away
        if dx:
            await self.device.send(f"move {dx} 0")
        if dy:
            await self.device.send(f"move 0 {dy}")
        self._pos = (self._pos[0] + dx, self._pos[1] + dy)

    def _fix(self, t: float, axis: str) -> float:
        """The fraction to aim at so the pointer lands on t, from the line through the observed landings."""
        pts = self._seen[axis]
        if len(pts) < 2:
            return t
        n = len(pts)
        mc, mf = sum(c for c, _ in pts) / n, sum(f for _, f in pts) / n
        var = sum((c - mc) ** 2 for c, _ in pts)
        a = sum((c - mc) * (f - mf) for c, f in pts) / var if var > 1e-4 else 1.0  # landed = a * aimed + b
        a = a if 0.85 < a < 1.15 else 1.0
        b = mf - a * mc
        return (t - b) / a if abs(b) < 0.1 else t

    async def _move_to(self, tx: int, ty: int, rehome: bool = True) -> None:
        if not self.calibrated:
            raise RuntimeError("device baseline is stale; wait for calibration")
        if self._link != self.device.links:  # new link: a dropped gesture may have left the button down
            await self.device.send("release")
            self._pos, self._link = None, self.device.links
        if self._pos is None or (rehome and not self._baseline and self._moves >= REHOME_EVERY):
            await self.device.send("home")
            self._pos, self._moves = (0, 0), 0
        x, y = self._pos
        self._pos = None  # unknown until the move succeeds
        if self._dragging:
            await self.device.send(f"move {tx - x} {ty - y}")
        else:
            if tx != x:
                await self.device.send(f"move {tx - x} 0")
            if ty != y:
                await self.device.send(f"move 0 {ty - y}")
        self._pos, self._moves = (tx, ty), self._moves + 1

    async def reset(self) -> None:
        """Let go of the button and re-home the pointer, whatever state a gesture left."""
        await self.device.send("release")

    async def tap(self, x: float, y: float, aim: bool = False) -> None:
        """aim=True finds the pointer on screen first (slower); worth it when a tap just missed."""
        await self.point(x, y, aim)
        await self.device.send("click")
        await asyncio.sleep(TAP_GAP)

    async def swipe(self, x1: float, y1: float, x2: float, y2: float, hold: bool = True) -> None:
        await self.drag([(x1, y1), (x2, y2)], hold)

    async def drag(self, points: list[tuple[float, float]], hold: bool = True) -> None:
        """Press at the first point, move through the rest, release: one continuous stroke.

        hold=True rests at the end first, so lists and pickers stop exactly there instead of flinging on.
        System gestures (home, back) need the flick, so they pass hold=False.
        """
        await self.point(*points[0])
        await self.device.send("press")
        self._dragging = True
        try:
            for x, y in points[1:]:
                await self._move_to(self._dx(x), self._dy(y), rehome=False)  # never home mid-drag
            if hold:
                await asyncio.sleep(HOLD)
        finally:
            self._dragging = False
            await self.device.send("release")

    async def scroll(self, x: float, y: float, lines: int) -> None:
        await self.point(x, y)
        await self.device.send(f"scroll {lines} 0")

    async def type(self, text: str) -> None:
        await self.device.send(f"type {text}")

    async def key(self, combo: str) -> None:
        await self.device.send(f"key {combo}")

    async def go_home(self) -> None:
        if self._home_link != self.device.links:
            self._home_link, self._home_supported = self.device.links, None
        if self._home_supported is not False:
            try:
                await self.device.send("homescreen")
                self._home_supported = True
                return
            except DeviceError as error:
                if str(error).strip().lower() != "err unknown command":
                    raise
                self._home_supported = False
        await self.point(0.5, 0.98)
        # Establish the physical bottom boundary before pressing, even if the fit drifts slightly.
        await self.device.send(f"move 0 {max(60, round(self.span['h'] * 0.2))}")
        self._pos = (self._dx(0.5), self._dy(1.0))
        await self.device.send("press")
        self._dragging = True
        try:
            await self._move_to(self._dx(0.5), self._dy(0.6), rehome=False)
        finally:
            self._dragging = False
            await self.device.send("release")

    async def go_back(self) -> None:
        await self.swipe(0.005, 0.45, 0.6, 0.45, hold=False)

    def _dx(self, x: float) -> int:
        if self._baseline:
            return Axis(**self._baseline["axes"][0]).units(x)
        return round(min(max(self._fix(x, "x"), 0), 1) * self.span["w"])

    def _dy(self, y: float) -> int:
        if self._baseline:
            return Axis(**self._baseline["axes"][1]).units(y)
        return round(min(max(self._fix(y, "y"), 0), 1) * self.span["h"])
