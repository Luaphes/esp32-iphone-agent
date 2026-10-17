"""Measure a device's coordinate mapping, then check points not used in the fit."""

import math
import time
from dataclasses import asdict, dataclass

from . import vision

TOLERANCE = 0.018


@dataclass
class Axis:
    span: float
    offset: float

    def units(self, fraction):
        # Clamp the requested screen coordinate, not the corrected mouse units.
        return max(0, round((min(1, max(0, fraction)) - self.offset) * self.span))


def fit_axis(samples):
    if len(samples) < 3:
        raise ValueError("not enough pointer observations")
    units, fractions = zip(*samples, strict=True)
    mu, mf = sum(units) / len(units), sum(fractions) / len(fractions)
    variance = sum((u - mu) ** 2 for u in units)
    if variance < 100 or max(fractions) - min(fractions) < 0.15:
        raise ValueError("observations do not cover enough of the screen")
    slope = sum((u - mu) * (f - mf) for u, f in samples) / variance
    if not math.isfinite(slope) or slope <= 0:
        raise ValueError("pointer mapping is not increasing")
    axis = Axis(1 / slope, mf - slope * mu)
    residual = max(abs(u * slope + axis.offset - f) for u, f in samples)
    if not 100 < axis.span < 20000 or abs(axis.offset) > 0.2 or residual > TOLERANCE:
        raise ValueError("pointer mapping is inconsistent")
    return axis


class Calibration:
    """Only moves the released pointer; never clicks application content."""

    def __init__(self, device, screen, seed, progress=None):
        self.device, self.screen, self.seed = device, screen, seed
        self.geometry = screen.geometry
        self.pos = (0, 0)
        self.progress = progress or (lambda stage: None)
        self.samples, self.checks = [], []

    async def send(self, command):
        if not self.screen.live or self.screen.geometry != self.geometry:
            raise RuntimeError("screen disconnected or rotated during calibration")
        await self.device.send(command)

    async def move(self, x, y):
        # Diagonal HID reports have a different speed/gain from axis-aligned ones.
        # Positioning must use the same motion pattern as these measurements.
        if x != self.pos[0]:
            await self.send(f"move {x - self.pos[0]} 0")
        if y != self.pos[1]:
            await self.send(f"move 0 {y - self.pos[1]}")
        self.pos = (x, y)

    async def frame(self, since=None):
        if since is not None and not await self.screen.fresh(since, 1.5):
            raise RuntimeError("no new screen frame after pointer movement")
        await self.screen.settled(quiet=0.25, least=0.8, most=1.5)
        if not self.screen.live or self.screen.geometry != self.geometry:
            raise RuntimeError("screen disconnected or rotated during calibration")
        return self.screen.frame

    async def observe(self, near=None, span=None):
        near = near or (0.5, 0.5)
        nudge = 24 if near[0] < 0.85 else -24
        shift = nudge / (span or self.seed[0])
        a = await self.frame()
        sent = time.time()
        await self.send(f"move {nudge} 0")
        b = await self.frame(sent)
        sent = time.time()
        await self.send(f"move {-nudge} 0")
        c = await self.frame(sent)

        def locate(before):
            return (vision.pointer(before, b, near, shift)
                    or vision.pointer(before, b, (0.5, 0.5), shift, reach=(0.5, 0.5))
                    or vision.pointer(before, b, near, shift, color=True)
                    or vision.pointer(before, b, (0.5, 0.5), shift, reach=(0.5, 0.5), color=True))

        p, q = locate(a), locate(c)
        if p and q:
            if max(abs(p[i] - q[i]) for i in (0, 1)) > 0.012:
                return None
            return tuple((p[i] + q[i]) / 2 for i in (0, 1))
        return p or q

    async def run(self):
        if not self.geometry:
            raise RuntimeError("no screen dimensions available")
        await self.send("release")
        await self.send("home")
        lo = [max(60, round(v * 0.2)) for v in self.seed]
        hi = [max(lo[i] + 100, round(v * 0.75)) for i, v in enumerate(self.seed)]
        points = [(lo[0], lo[1]), (hi[0], lo[1]), (hi[0], hi[1]), (lo[0], hi[1]),
                  (round((lo[0] + hi[0]) / 2), round((lo[1] + hi[1]) / 2))]
        for i, (x, y) in enumerate(points):
            self.progress(f"measure {i + 1}/{len(points)}")
            await self.move(x, y)
            found = await self.observe()
            if found and all(0.04 < v < 0.96 for v in found):
                self.samples.append({"units": [x, y], "point": list(found)})
        axes = [fit_axis([(s["units"][i], s["point"][i]) for s in self.samples]) for i in (0, 1)]
        for i, target in enumerate([(0.5, 0.5), (0.75, 0.8), (0.5, 0.99), (0.25, 0.2)]):
            self.progress(f"verify {i + 1}/4")
            await self.move(*(axis.units(f) for axis, f in zip(axes, target, strict=True)))
            found = await self.observe(target, axes[0].span)
            if not found:
                # A wallpaper or hover target can make one location unobservable. Move to another
                # held-out point in the same height band, retaining the bottom-edge check.
                target = (min(.9, target[0] + .08), target[1] if target[1] > .95 else target[1] + .06)
                await self.move(*(axis.units(f) for axis, f in zip(axes, target, strict=True)))
                found = await self.observe(target, axes[0].span)
                if not found:
                    raise RuntimeError("could not observe the pointer at a verification point")
            error = max(abs(found[i] - target[i]) for i in (0, 1))
            self.checks.append({"target": list(target), "observed": list(found), "error": error})
            if error > TOLERANCE:
                raise RuntimeError(f"verification missed by {error:.1%} of the screen")
        return {"version": 1, "geometry": list(self.geometry), "axes": [asdict(a) for a in axes],
                "samples": self.samples, "checks": self.checks, "tolerance": TOLERANCE,
                "measured_at": time.time()}
