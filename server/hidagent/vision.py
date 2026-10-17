"""Small image checks on screen frames: did the screen change, and where is the pointer."""

import io

from PIL import Image, ImageChops, ImageDraw


def _gray(jpeg: bytes) -> Image.Image:
    return Image.open(io.BytesIO(jpeg)).convert("L")


def frame_size(jpeg: bytes) -> tuple[int, int]:
    return Image.open(io.BytesIO(jpeg)).size


def evidence_sheet(panels: list[tuple[str, bytes]]) -> bytes:
    """A bounded, labelled view of earlier observations and the final screen for verification."""
    images = [Image.open(io.BytesIO(frame)).convert("RGB") for _, frame in panels]
    width = min(1000, max(im.width for im in images))
    height = max(round(im.height * width / im.width) for im in images)
    header, columns = 36, min(2, len(images))
    sheet = Image.new("RGB", (columns * width, ((len(images) + columns - 1) // columns) *
                             (height + header)), "#202020")
    draw = ImageDraw.Draw(sheet)
    for i, ((label, _), im) in enumerate(zip(panels, images, strict=True)):
        x, y = (i % columns) * width, (i // columns) * (height + header)
        draw.text((x + 8, y + 8), label, fill="white", font_size=18)
        im.thumbnail((width, height), Image.Resampling.LANCZOS)
        sheet.paste(im, (x, y + header))
    output = io.BytesIO()
    sheet.save(output, "JPEG", quality=90)
    return output.getvalue()


def hover_hint(before: bytes, after: bytes, restored: bytes) -> tuple[float, float] | None:
    """Locate a reversible icon-sized hover response for escape, never for aiming/clicking."""
    a, b, c = [_gray(frame) for frame in (before, after, restored)]
    if a.size != b.size or a.size != c.size:
        return None
    w, h = a.size
    diff = ImageChops.difference(a, b)
    ImageDraw.Draw(diff).rectangle((0, 0, w, int(.06*h)), fill=0)
    box = diff.point([0]*31+[255]*225).getbbox()
    if not box:
        return None
    left, top, right, bottom = box
    if not (.04*w < right-left < .16*w and .04*h < bottom-top < .16*h):
        return None
    av, bv = a.crop(box).tobytes(), b.crop(box).tobytes()
    positive = sum(z-v > 30 for v, z in zip(av, bv, strict=True))
    negative = sum(v-z > 30 for v, z in zip(av, bv, strict=True))
    if min(positive, negative) < 100 or not .25 < positive/negative < 4:
        return None
    residual = ImageChops.difference(a.crop(box), c.crop(box))
    if sum(residual.histogram()[31:]) > .2*(positive+negative):
        return None
    return ((left+right)/(2*w), (top+bottom)/(2*h))


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
    # Mask before either pass: a large/high-contrast cursor must not count as progress.
    diff_image = ImageChops.difference(a, b)
    draw = ImageDraw.Draw(diff_image)
    width, height = a.size
    draw.rectangle((0, 0, width, int(.06 * height)), fill=0)
    radius = .05 * width
    for px, py in ignore:
        cx, cy = px * width, py * height
        draw.ellipse((cx-radius, cy-radius, cx+radius, cy+radius), fill=0)
    small = diff_image.resize((32, 64), Image.BILINEAR).tobytes()
    if sum(small) / len(small) > threshold:  # a big change: new screen, scrolled list, keyboard
        return True
    w, h = a.size[0] // 2, a.size[1] // 2  # fine pass for small changes, at half resolution
    diff = diff_image.resize((w, h), Image.BILINEAR).tobytes()
    return sum(v > 25 for v in diff) >= 6


def scene_changed(before: bytes | None, after: bytes | None, ignore: tuple = (), focus: tuple = ()) -> bool:
    """Motion guard, allowing sparse remote animation; use changed() for task progress.

    Without focus, .2% of changed pixels rejects the scene. With focus, retain
    that sensitivity locally and allow sparse remote icon updates; a global
    brightness change or >2% changed pixels still rejects a new page or popup.
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
    radius = .05*w
    for x, y in ignore:
        cx, cy = x*w, y*h
        draw.ellipse((cx-radius, cy-radius, cx+radius, cy+radius), fill=0)
    small = diff.resize((32, 64), Image.BILINEAR).tobytes()
    hits = sum(diff.histogram()[31:])
    if sum(small)/len(small) > 1 or hits > (.02 if focus else .002)*w*h:
        return True
    # A remote Dock/Handoff icon can update while the target and its surroundings
    # stay still. Retain the sensitive check around relevant positions, while
    # full-page transitions and popups are still rejected by the global pass.
    for x, y in focus:
        radius = .1*w
        box = (max(0, int(x*w-radius)), max(0, int(y*h-radius)),
               min(w, int(x*w+radius)), min(h, int(y*h+radius)))
        region = diff.crop(box)
        if sum(region.histogram()[31:]) > .002*region.width*region.height:
            return True
    return False


def _pointer_region(before: bytes, after: bytes, near: tuple[float, float], shift: float,
            reach: tuple[float, float] = (0.12, 0.08), restored: bytes | None = None) -> tuple[float, float] | None:
    """Where the pointer is in `before`, given it moved right by `shift` (fraction of width) in `after`.

    Only the pointer moved between the two frames, so the changed pixels within `reach` of `near` are the
    pointer at both spots; their centroid sits shift/2 to the right of the first one. None if unclear:
    nothing moved, or the change is too big or too spread out to be one small pointer.
    """
    a, b = _gray(before), _gray(after)
    if a.size != b.size:
        return None
    w, h = a.size
    rx, ry = reach
    left, top = max(0, int((near[0] - rx) * w)), max(0, int((near[1] - ry) * h))
    box = (left, top, min(w, int((near[0] + rx) * w)), min(h, int((near[1] + ry) * h)))
    difference = ImageChops.difference(a.crop(box), b.crop(box))
    window_area = difference.width*difference.height
    bounds = difference.point([0]*31+[255]*225).getbbox()
    if bounds is None:
        return None
    box = (left+bounds[0], top+bounds[1], left+bounds[2], top+bounds[3])
    left, top = box[:2]
    diff = difference.crop(bounds).tobytes()
    bw = box[2] - box[0]
    hits = [(i % bw, i // bw) for i, v in enumerate(diff) if v > 30]
    if not 15 <= len(hits) <= window_area // 4:  # nothing moved, or much more than a pointer
        return None
    xs, ys = [x for x, _ in hits], [y for _, y in hits]
    if max(xs) - min(xs) > 0.15 * w or max(ys) - min(ys) > 0.07 * h:  # not one compact spot
        return None
    # A translated shape both leaves and enters pixels. A blinking element only
    # appears/disappears; compactness alone used to mistake it for the pointer.
    av, bv = a.crop(box).tobytes(), b.crop(box).tobytes()
    positive = [i for i, (v, z) in enumerate(zip(av, bv, strict=True)) if z-v > 30]
    negative = [i for i, (v, z) in enumerate(zip(av, bv, strict=True)) if v-z > 30]
    if min(len(positive), len(negative)) < 8 or not .3 < len(positive)/len(negative) < 3:
        return None
    px = sum(i % bw for i in positive)/len(positive)
    nx = sum(i % bw for i in negative)/len(negative)
    py = sum(i // bw for i in positive)/len(positive)
    ny = sum(i // bw for i in negative)/len(negative)
    if abs(px-nx) < 2 or abs(py-ny) > max(3, .015*h):
        return None
    for axis in (lambda i: i % bw, lambda i: i // bw):
        extents = [max(map(axis, group))-min(map(axis, group))+1 for group in (positive, negative)]
        if min(extents)/max(extents) < .3:
            return None
    cx = sum(x for x, _ in hits) / len(hits) + left
    cy = sum(y for _, y in hits) / len(hits) + top
    found = (cx / w - shift / 2, cy / h)
    # An arrow clicks at its upper-left tip, whereas a round iPad pointer clicks
    # at its centre. The translated silhouette has a nearly vertical left edge
    # in its top quarter only for the arrow; a circle curves inward at the top.
    height = max(ys)-min(ys)+1
    top_rows = [min(x for x, y in hits if y == row) - min(xs)
                for row in range(min(ys), min(ys)+max(2, height//4))
                if any(y == row for _, y in hits)]
    shape_width = max(xs)-min(xs)+1-abs(shift)*w
    unclipped = (left+min(xs) > 2 and top+min(ys) > 2
                 and left+max(xs) < w-3 and top+max(ys) < h-3)
    if (unclipped and top_rows and 12 <= height <= .06*h and .4*height < shape_width < 1.5*height
            and sum(top_rows)/len(top_rows) < .1*height):
        found = ((left+min(xs)+max(0, -shift*w)+1)/w, (top+min(ys)+1)/h)
    spots = (found, (found[0]+shift, found[1]))
    if scene_changed(before, after, ignore=spots, focus=(found,)):
        return None
    if restored is not None:
        c = _gray(restored)
        if c.size != a.size or scene_changed(before, restored, ignore=spots, focus=(found,)):
            return None
        residual = ImageChops.difference(a.crop(box), c.crop(box)).tobytes()
        if sum(v > 30 for v in residual) > max(8, .2*len(hits)):
            return None
    return found if all(0 <= v <= 1 for v in found) else None


def pointer(before: bytes, after: bytes, near: tuple[float, float], shift: float,
            reach: tuple[float, float] = (.12, .08), restored: bytes | None = None) -> tuple[float, float] | None:
    """Verify local motion candidates independently of remote clock/widget animation."""
    a, b = _gray(before), _gray(after)
    if a.size != b.size:
        return None
    w, h = a.size
    left, top = max(0, int((near[0]-reach[0])*w)), max(0, int((near[1]-reach[1])*h))
    right, bottom = min(w, int((near[0]+reach[0])*w)), min(h, int((near[1]+reach[1])*h))
    if left >= right or top >= bottom:
        return None
    # Pool to 16px cells, then join adjacent changed cells. The two cursor lobes
    # stay together; a remote second hand forms a separate candidate.
    tile = 16
    mask = ImageChops.difference(a.crop((left, top, right, bottom)), b.crop((left, top, right, bottom)))
    grid = mask.point([0]*31+[255]*225).reduce(tile)
    cells = {(i % grid.width, i // grid.width) for i, value in enumerate(grid.tobytes()) if value}
    found = []
    while cells:
        pending = [cells.pop()]
        group = []
        while pending:
            x, y = pending.pop()
            group.append((x, y))
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    neighbour = (x+dx, y+dy)
                    if neighbour in cells:
                        cells.remove(neighbour)
                        pending.append(neighbour)
        x0 = max(left, left+(min(x for x, _ in group)-2)*tile)
        y0 = max(top, top+(min(y for _, y in group)-2)*tile)
        x1 = min(right, left+(max(x for x, _ in group)+3)*tile)
        y1 = min(bottom, top+(max(y for _, y in group)+3)*tile)
        candidate = _pointer_region(before, after, ((x0+x1)/(2*w), (y0+y1)/(2*h)), shift,
                                    reach=((x1-x0)/(2*w), (y1-y0)/(2*h)), restored=restored)
        if candidate and not any(abs(candidate[0]-p[0])*w < 4 and abs(candidate[1]-p[1])*h < 4 for p in found):
            found.append(candidate)
    return found[0] if len(found) == 1 else None
