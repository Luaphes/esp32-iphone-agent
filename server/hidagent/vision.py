"""Small image checks on screen frames: did the screen change, and where is the pointer."""

import io

from PIL import Image, ImageChops, ImageDraw


def _gray(jpeg: bytes) -> Image.Image:
    return Image.open(io.BytesIO(jpeg)).convert("L")


def changed(before: bytes | None, after: bytes | None, ignore: tuple = (), threshold: float = 1.0) -> bool:
    """True when the screen content changed, including small things such as a checkmark.

    `ignore` lists screen points (fractions) where the pointer was or went: those spots are masked, so pointer
    moves never count. The status bar (clock) is ignored too.
    """
    if not before or not after:
        return True
    a, b = _gray(before), _gray(after)
    if a.size != b.size:
        return True
    small = ImageChops.difference(a.resize((32, 64), Image.BILINEAR), b.resize((32, 64), Image.BILINEAR)).tobytes()
    if sum(small) / len(small) > threshold:  # a big change: new screen, scrolled list, keyboard
        return True
    w, h = a.size[0] // 2, a.size[1] // 2  # fine pass for small changes, at half resolution
    diff = ImageChops.difference(a.resize((w, h), Image.BILINEAR), b.resize((w, h), Image.BILINEAR)).tobytes()
    masks = [(px * w, py * h) for px, py in ignore]
    hits = 0
    for i, v in enumerate(diff):
        if v > 25:
            x, y = i % w, i // w
            if y > 0.06 * h and all((x - mx) ** 2 + (y - my) ** 2 > (0.05 * w) ** 2 for mx, my in masks):
                hits += 1
    return hits >= 6


def scene_changed(before: bytes | None, after: bytes | None, ignore: tuple = ()) -> bool:
    """A substantial change, allowing sparse remote animation (use changed() for task progress).

    A clock hand elsewhere must not make a tap look stale. A new screen, popup or scroll changes either
    substantial brightness or more than .2% of unmasked pixels.
    """
    if not before or not after:
        return True
    a, b = _gray(before), _gray(after)
    if a.size != b.size:
        return True
    diff = ImageChops.difference(a, b)
    draw = ImageDraw.Draw(diff)
    w, h = a.size
    draw.rectangle((0, 0, w, int(.06*h)), fill=0)
    radius = .05 * w
    for x, y in ignore:
        cx, cy = x * w, y * h
        draw.ellipse((cx-radius, cy-radius, cx+radius, cy+radius), fill=0)
    small = diff.resize((32, 64), Image.BILINEAR).tobytes()
    return sum(small) / len(small) > 1 or sum(diff.histogram()[31:]) > .002 * w * h


def pointer(before: bytes, after: bytes, near: tuple[float, float], shift: float,
            reach: tuple[float, float] = (0.12, 0.08), color: bool = False) -> tuple[float, float] | None:
    """Where the pointer is in `before`, given it moved right by `shift` (fraction of width) in `after`.

    Only the pointer moved between the two frames, so the changed pixels within `reach` of `near` are the
    pointer at both spots; their centroid sits shift/2 to the right of the first one. None if unclear:
    nothing moved, or the change is too big or too spread out to be one small pointer.
    """
    a, b = ((Image.open(io.BytesIO(f)).convert("RGB") for f in (before, after))
            if color else (_gray(before), _gray(after)))
    if a.size != b.size:
        return None
    w, h = a.size
    rx, ry = reach
    left, top = max(0, int((near[0] - rx) * w)), max(int(0.05 * h), int((near[1] - ry) * h))  # skip the status bar
    box = (left, top, min(w, int((near[0] + rx) * w)), min(h, int((near[1] + ry) * h)))
    region = ImageChops.difference(a.crop(box), b.crop(box))
    if color:
        red, green, blue = region.split()
        region = ImageChops.lighter(ImageChops.lighter(red, green), blue)
    diff = region.tobytes()
    bw = box[2] - box[0]
    hits = [(i % bw, i // bw) for i, v in enumerate(diff) if v > 30]
    if not 15 <= len(hits) <= len(diff) // 4:  # nothing moved, or much more than a pointer
        return None
    xs, ys = [x for x, _ in hits], [y for _, y in hits]
    if max(xs) - min(xs) > 0.15 * w or max(ys) - min(ys) > 0.07 * h:  # not one compact spot
        return None
    cx = sum(x for x, _ in hits) / len(hits) + left
    cy = sum(y for _, y in hits) / len(hits) + top
    return cx / w - shift / 2, cy / h
