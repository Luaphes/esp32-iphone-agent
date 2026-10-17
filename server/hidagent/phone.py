"""Relative HID gestures anchored by visible pointer observations, never screen corners."""

import asyncio
import json
import math
import time
from pathlib import Path

from . import vision
from .device import Device, DeviceError
from .keyboard import prepare_text

CALIBRATION_VERSION = 2
PROBE = 24
NUDGE = 12
HOLD = .25
TAP_GAP = .15
MAX_STEP = .15


class Phone:
    def __init__(self, device: Device, calibration_file: Path, screen=None):
        self.device, self._file, self.screen = device, calibration_file, screen
        saved = json.loads(calibration_file.read_text()) if calibration_file.exists() else {}
        self.span = None
        if saved.get("version") == CALIBRATION_VERSION and all(
                isinstance(saved.get(k), (int, float)) and math.isfinite(saved[k]) and saved[k] > 0
                for k in ("w", "h", "aspect")):
            self.span = {k: saved[k] for k in ("w", "h")}
        self._aspect = saved.get("aspect")
        self._pos = None  # observed/predicted screen fraction, never a corner-origin counter
        self._link = getattr(device, "links", 0)
        self._epoch = 0
        self._pending = None
        self._commands = set()
        self._nudge_sign = 1
        self._edge_hint = None  # unverified hint may only trigger bounded inward recovery
        self._hover_hint = None

    @property
    def calibrated(self) -> bool:
        return (self.span is not None and self.screen is not None and self.screen.live
                and self._aspect is not None and abs(self.screen.aspect/self._aspect-1) < .02)

    def _context(self):
        if self.screen is None or not self.screen.live or not self.screen.frame:
            raise RuntimeError("no live screen; reconnect mirroring and retry")
        return self._epoch, getattr(self.device, "links", 0), vision.frame_size(self.screen.frame)

    def _check(self, context):
        try:
            current = self._context()
        except RuntimeError:
            self._pos, self._pending = None, None
            raise
        if current != context:
            self._pos, self._pending = None, None
            raise RuntimeError("pointer operation invalidated by stop, reconnect or screen rotation; retry calibration")

    async def _send(self, command, context):
        self._check(context)
        task = asyncio.create_task(self.device.send(command))
        self._commands.add(task)
        try:
            await task
        except asyncio.CancelledError:
            self._pos = None
            if self._epoch != context[0]:
                raise RuntimeError("pointer operation stopped; retry") from None
            raise
        except BaseException:
            self._pos = None
            raise
        finally:
            self._commands.discard(task)
        self._check(context)

    async def _frame_after(self, sent, context):
        frame = await self.screen.fresh(sent, 1.2)
        self._check(context)
        if frame:
            # A fresh AirPlay frame may still be buffered from before the HID
            # command (e.g. a blinking caret). Let the input reach the video feed.
            await self.screen.settled(quiet=.2, least=.6, most=1.2)
            self._check(context)
            frame = self.screen.frame
        return frame

    async def _find_pointer(self, near=None, context=None):
        # iPad may consume the first movement to reveal an auto-hidden pointer.
        # Retry from the now-visible frame; a one-sided blink is never an anchor.
        context = context or self._context()
        self._edge_hint = self._hover_hint = None
        # A user or another input device can move the pointer outside the cached
        # search window without changing the link or screen orientation.
        for anchor in (near, None) if near is not None else (None,):
            for direction in (1, -1):
                found = await self._probe_pointer(anchor, context, direction)
                if found is not None:
                    return found
        hint, hover = self._edge_hint, self._hover_hint
        self._edge_hint = self._hover_hint = None
        if hint is not None or hover is not None:
            # Clipping/magnetic status-bar hover can prevent A/C restoration.
            # A compact B/C motion hint is sufficient only to move inward, never
            # to click or calibrate. Require a full reversible observation after it.
            if hint is not None:
                dx, dy = [2*PROBE if v < .04 else -2*PROBE if v > .96 else 0 for v in hint]
            else:
                # iPad morphs the cursor into a hovered Dock/app icon. Its
                # reversible highlight supplies only a region to escape from.
                axis = max(range(2), key=lambda i: abs(hover[i]-.5))
                units = [0, 0]
                units[axis] = 4*PROBE if hover[axis] < .5 else -4*PROBE
                dx, dy = units
            await self._send(f"move {dx} {dy}", context)
            for direction in (1, -1):
                found = await self._probe_pointer(None, context, direction)
                if found is not None:
                    return found
        return None

    async def _probe_pointer(self, near=None, context=None, direction=1):
        """Horizontal A/B/C probe only. Unknown positions have no strict side-edge guarantee."""
        context = context or self._context()
        await self.screen.settled(quiet=.2, least=.6, most=1.2)
        self._check(context)
        before = self.screen.frame
        preferred = 1 if near and near[0] < .04 else -1 if near and near[0] > .96 else self._nudge_sign
        nudge = direction * preferred * NUDGE
        sent = time.time()
        await self._send(f"move {nudge} 0", context)
        after = await self._frame_after(sent, context)
        sent = time.time()
        await self._send(f"move {-nudge} 0", context)
        restored = await self._frame_after(sent, context)
        if not before or not after or not restored:
            return None
        # Before scale is measured this estimate only biases the anchor; relative
        # differences between calibration probes determine scale, not a corner.
        shift = nudge/self.span["w"] if self.span else math.copysign(.012, nudge)
        found = vision.pointer(before, after, near or (.5, .5), shift,
                               reach=(.12, .08) if near else (.5, .5), restored=restored)
        hover = vision.hover_hint(before, after, restored) if found is None else None
        if found is not None:
            self._nudge_sign = 1 if nudge > 0 else -1
        else:
            self._hover_hint = hover or self._hover_hint
            # Do not use a clipped/non-restored pointer as an aiming anchor.
            # The return leg can still identify which edge needs a small escape.
            edge = vision.pointer(after, restored, (.5, .5), -shift, reach=(.5, .5))
            if edge:
                edge = (edge[0]-shift, edge[1])
                if all(0 <= v <= 1 for v in edge) and any(v < .04 or v > .96 for v in edge):
                    self._edge_hint = edge
        return found

    async def start_calibration(self) -> bool:
        self._epoch += 1
        self._pos, self.span = None, None
        context = self._context()
        self._pending = context
        await self._send("release", context)
        anchor = await self._find_pointer(context=context)
        if anchor is None:
            return False
        await self._calibrate(anchor, context)
        return True

    async def finish_calibration(self, x: float, y: float) -> None:
        if self._pending is None:
            raise RuntimeError("no pending calibration; start calibration again")
        context = self._pending
        self._check(context)
        found = await self._find_pointer((x, y), context)
        if found is None and any(v < .06 or v > .94 for v in (x, y)):
            # The user identified a visible edge pointer. iPad hover can turn it
            # into the status-bar highlight; move a bounded distance inward,
            # then demand a real pointer observation before calibration/clicks.
            dx, dy = [PROBE if v < .06 else -PROBE if v > .94 else 0 for v in (x, y)]
            for _ in range(3):
                await self._send(f"move {dx} {dy}", context)
                found = await self._find_pointer((x, y), context)
                if found is not None:
                    break
        elif found is not None and max(abs(found[0]-x), abs(found[1]-y)) > .04:
            found = None
        if found is None:
            raise RuntimeError("clicked pointer could not be verified; click the visible pointer and retry")
        await self._calibrate(found, context)

    async def _calibrate(self, anchor, context):
        baseline, origin = self.screen.frame, anchor
        # A visible edge anchor can move inward; never sweep vertically while blind.
        for _ in range(16):
            if all(.04 < v < .96 for v in anchor):
                break
            axis = max(range(2), key=lambda i: abs(anchor[i]-.5))
            units = [0, 0]
            units[axis] = PROBE if anchor[axis] < .5 else -PROBE
            await self._send(f"move {units[0]} {units[1]}", context)
            found = await self._find_pointer(anchor, context)
            if (found is None or units[axis]*(found[axis]-anchor[axis]) <= 0
                    or abs(found[axis]-anchor[axis]) > .1
                    or abs(found[1-axis]-anchor[1-axis]) > .015):
                raise RuntimeError("pointer could not be followed inward; click its current position and retry")
            if vision.scene_changed(baseline, self.screen.frame, ignore=(origin, found)):
                raise RuntimeError("screen changed during calibration; restart from the current screen")
            anchor = found
        else:
            raise RuntimeError("pointer did not reach the calibration area; click its current position and retry")
        scales = {}
        for axis, name in enumerate(("w", "h")):
            start = anchor
            # Keep the measured horizontal direction: the opposite side may
            # activate an iPad hover target and hide the circular pointer.
            delta = PROBE * self._nudge_sign if axis == 0 else PROBE if start[axis] < .5 else -PROBE
            units = [0, 0]
            units[axis] = delta
            await self._send(f"move {units[0]} {units[1]}", context)
            end = await self._find_pointer(start, context)
            if end is None or not all(.03 < v < .97 for v in end):
                self._pos = None
                raise RuntimeError("calibration probe could not be tracked safely; restart calibration")
            if vision.scene_changed(baseline, self.screen.frame, ignore=(origin, end)):
                raise RuntimeError("screen changed during calibration; restart from the current screen")
            await self._send(f"move {-units[0]} {-units[1]}", context)
            restored = await self._find_pointer(end, context)
            if restored is None:
                raise RuntimeError("calibration return could not be verified; restart calibration")
            if vision.scene_changed(baseline, self.screen.frame, ignore=(origin, restored)):
                raise RuntimeError("screen changed during calibration; restart from the current screen")
            forward, backward = end[axis]-start[axis], end[axis]-restored[axis]
            if (not .003 < abs(forward) < .15 or delta*forward <= 0 or delta*backward <= 0
                    or not .75 < backward/forward < 1.25
                    or abs(end[1-axis]-start[1-axis]) > .015
                    or max(abs(a-b) for a, b in zip(start, restored, strict=True)) > .015):
                raise RuntimeError("inconsistent pointer motion during calibration; keep the screen still and retry")
            scales[name] = abs(2*delta/(forward+backward))
            anchor = restored
        self._check(context)
        self.span, self._aspect = scales, self.screen.aspect
        try:
            self._pos = await self._find_pointer(anchor, context)
        except BaseException:
            self.span = None
            raise
        if self._pos is None:
            self.span = None
            raise RuntimeError("final calibration anchor was lost; retry calibration")
        self._link, self._pending = context[1], None
        self._file.write_text(json.dumps({"version": CALIBRATION_VERSION, **self.span, "aspect": self._aspect}))

    @staticmethod
    def _validate(x, y):
        if not all(math.isfinite(v) and 0 < v < 1 for v in (x, y)):
            raise ValueError("pointer target must be inside the visible screen")

    async def point(self, x: float, y: float, aim: bool = False) -> None:
        """Every target is visually confirmed, including ordinary taps and drag starts."""
        self._validate(x, y)
        if not self.calibrated:
            raise RuntimeError("pointer not calibrated for this screen; recalibrate after device/orientation changes")
        context = self._context()
        if context[1] != self._link:
            self._pos = None
        self._link = context[1]
        await self._send("release", context)
        found = await self._find_pointer(self._pos, context)
        if found is None:
            self._pos = None
            raise RuntimeError("pointer not visible; keep the screen still or recalibrate before retrying")
        tolerance = tuple(2/n for n in context[2])
        baseline, start = self.screen.frame, found
        scales = dict(self.span)
        for attempt in range(13):
            self._check(context)
            self._pos = found
            error = (x-found[0], y-found[1])
            if all(abs(e) <= t for e, t in zip(error, tolerance, strict=True)):
                return context
            if attempt == 12:
                break
            axis = max(range(2), key=lambda i: abs(error[i])/tolerance[i])
            step = max(-MAX_STEP, min(MAX_STEP, error[axis]))
            units = [0, 0]
            units[axis] = round(step*scales[("w", "h")[axis]])
            if not units[axis]:
                break
            predicted = list(found)
            predicted[axis] += units[axis]/scales[("w", "h")[axis]]
            if not 0 < predicted[axis] < 1:
                raise RuntimeError("pointer target is too close to an edge to verify safely")
            await self._send(f"move {units[0]} {units[1]}", context)
            previous = found
            found = await self._find_pointer(tuple(predicted), context)
            if found is None:
                break
            displacement = found[axis]-previous[axis]
            issue = ("opposite direction" if displacement*step <= 0 else
                     "excessive displacement" if abs(displacement) > abs(step)*2+.01 else
                     "cross-axis displacement" if abs(found[1-axis]-previous[1-axis]) > .025 else
                     "background changed" if vision.scene_changed(
                         baseline, self.screen.frame, ignore=(start, previous, found),
                         focus=(start, (x, y))) else None)
            if issue:
                self._pos = None
                raise RuntimeError(f"unexpected pointer or screen movement ({issue}; "
                                   f"from {previous!r} to {found!r}); click cancelled, retry")
            # Small HID moves and long paced moves can have different gain.
            # Use a just-observed, validated move for this aim only, rather than
            # repeatedly overshooting with the saved calibration. Never persist it.
            if abs(displacement) > 4/context[2][axis]:
                name = ("w", "h")[axis]
                measured = abs(units[axis]/displacement)
                if .5*self.span[name] < measured < 2*self.span[name]:
                    scales[name] = measured
        self._pos = None
        raise RuntimeError("pointer target could not be confirmed; click cancelled, recalibrate or retry")

    async def reset(self) -> None:
        """Emergency release only. Stop invalidates in-flight calibration and pointer actions."""
        self._epoch += 1
        self._pos, self._pending = None, None
        tasks = list(self._commands)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        if getattr(self.device, "connected", True):
            await self.device.send("release")

    async def tap(self, x: float, y: float, aim: bool = False) -> None:
        context = await self.point(x, y, aim=True)
        await self._send("click", context)
        await asyncio.sleep(TAP_GAP)

    async def swipe(self, x1: float, y1: float, x2: float, y2: float, hold: bool = True) -> None:
        await self.drag([(x1, y1), (x2, y2)], hold)

    async def drag(self, points: list[tuple[float, float]], hold: bool = True) -> None:
        for point in points:
            self._validate(*point)
        if len(points) < 2:
            raise ValueError("drag needs at least two points")
        context = await self.point(*points[0], aim=True)
        try:
            await self._send("press", context)
            for x, y in points[1:]:
                dx = round((x-self._pos[0])*self.span["w"])
                dy = round((y-self._pos[1])*self.span["h"])
                await self._send(f"move {dx} {dy}", context)
                self._pos = (x, y)
            if hold:
                await asyncio.sleep(HOLD)
        finally:
            self._pos = None
            await self.device.send("release")

    async def scroll(self, x: float, y: float, lines: int) -> None:
        context = await self.point(x, y, aim=True)
        await self._send(f"scroll {lines} 0", context)

    async def type(self, text: str) -> dict | None:
        keys, composition = prepare_text(text)
        await self.device.send(f"type {keys}")
        return composition

    async def key(self, combo: str) -> None:
        await self.device.send(f"key {combo}")

    async def go_home(self) -> None:
        try:
            await self.device.send("homescreen")
        except DeviceError as e:
            if "unknown command" in str(e):
                raise DeviceError("board firmware lacks the Home key; flash the updated firmware") from e
            raise
        finally:
            self._pos = None

    async def go_back(self) -> None:
        await self.swipe(.005, .45, .6, .45, hold=False)
