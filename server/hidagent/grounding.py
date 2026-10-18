"""Check a proposed tap against a magnified local screenshot before moving the pointer."""

import io
import json
import re
import time

from PIL import Image


def _render(region):
    scale = 1000 / max(region.size)
    size = tuple(round(v * scale) for v in region.size)
    offset = tuple((1000 - v) // 2 for v in size)
    crop = Image.new("RGB", (1000, 1000), "white")
    crop.paste(region.resize(size, Image.Resampling.LANCZOS), offset)
    output = io.BytesIO()
    crop.save(output, format="JPEG", quality=90)
    return output.getvalue(), size, offset


def _screen_point(cx, cy, box, size, offset, geometry):
    if not all(offset[i] < value < offset[i] + size[i] for i, value in enumerate((cx, cy))):
        raise RuntimeError("intended tap target was not located inside the image content (point is in padding)")
    return tuple(round((box[i]+(value-offset[i])/size[i]*(box[i+2]-box[i]))/geometry[i]*1000)
                 for i, value in enumerate((cx, cy)))

PROMPT = """Locate the intended visible UI target in this cropped screenshot.
Return only POINT X Y, integers 0-1000 relative to THIS CROP, or NOT_FOUND.
The crop is fitted without distortion into a 1000x1000 image with blank padding: use THIS image's
pixel coordinates directly, not the full-screen coordinates. Never select the padding.
Choose the visual center of the actual icon, button or text described. Do not guess a hidden target.
If the target is an explicitly empty text area, blank note body or blank canvas (a place for the cursor, not an
existing icon or text), return POINT at the center of that blank area instead of NOT_FOUND.
If the target is not visible or is ambiguous, return NOT_FOUND."""


async def locate_tap(llm, frame: bytes, action: dict, trace) -> dict:
    """Refine a tap near the proposal; toolbar taps search the whole band, so a wrong x can be corrected."""
    image = Image.open(io.BytesIO(frame)).convert("RGB")
    w, h = image.size
    x, y = action["x"], action["y"]
    if not 0 <= x <= 1000 or not 0 <= y <= 1000:
        raise ValueError("tap coordinates must be within 0-1000")
    if action.get("top_toolbar"):  # identified from the current UI, not just a small y coordinate
        box = (0, 0, w, min(h, max(120, int(0.2*h))))
    else:
        box = (max(0, int((x-200)*w/1000)), max(0, int((y-150)*h/1000)),
               min(w, int((x+200)*w/1000)), min(h, int((y+150)*h/1000)))
    crop, size, offset = _render(image.crop(box))
    target = action.get("target") or action.get("thought", "")
    prompt = f"Current tap target only: {target}\nLocate this element, not an element mentioned in earlier context."
    started = time.time()
    reply = await llm.chat(PROMPT, prompt, crop, max_tokens=180)
    elapsed = round(time.time()-started, 2)
    trace.write(crop, kind="llm", role="grounder", model=llm.model, prompt=prompt,
                reply=reply, seconds=elapsed, crop_box=box, content_size=size, content_offset=offset,
                target=target, proposed={"x": x, "y": y})
    match = re.fullmatch(r"\s*POINT\s+(\d+)\s+(\d+)\s*", reply, re.I)
    if not match:
        # Some vision providers render a point using their own structured token wrapper.
        match = re.fullmatch(r"\s*<｜｜point｜｜>\[\[(\d+),\s*(\d+)\]\]<｜｜/point｜｜>\s*", reply)
    if not match or not all(0 < int(v) < 1000 for v in match.groups()):
        raise RuntimeError("intended tap target was not located in the crop; inspect the full screen and replan")
    cx, cy = map(int, match.groups())
    sx, sy = _screen_point(cx, cy, box, size, offset, (w, h))
    return {**action, "x": sx, "y": sy,
            "ground_s": elapsed, "proposed": {"x": x, "y": y}}


async def discover_tools(llm, frame: bytes, trace, region: tuple | None = None) -> list[dict]:
    """Neutral, visible tool candidates after a real tool-selection failure; never a global blind scan."""
    image = Image.open(io.BytesIO(frame)).convert("RGB")
    w, h = image.size
    box = region or (0, 0, w, min(h, max(120, int(.2*h))))
    crop, size, offset = _render(image.crop(box))
    area = ("observed tool-control region, including any collapsed palette launcher"
            if region else "visible top toolbar")
    prompt = (f"Inventory the current application's {area} independently of any task. "
              "List clickable tool-selection controls with literal visual appearances, from left to right. "
              "Exclude document titles, back/navigation, sharing/sending, destructive/delete/erase-content, "
              "account controls and ambiguous general menus. Do not infer that an icon is the tool a task wants. "
              "Coordinates are pixels in THIS 1000x1000 image; content is undistorted with blank padding. "
              "Return JSON only, no fences: {\"tools\":[{\"x\":123,\"y\":456,"
              "\"appearance\":\"literal icon shape\",\"kind\":\"tool\"}]}. At most 8 candidates; "
              "return an empty list if no safe tool-selection controls are visible.")
    started = time.time()
    reply = await llm.chat("Inspect only visible UI controls. Never act or invent hidden controls.",
                           prompt, crop, max_tokens=1200)
    trace.write(crop, kind="llm", role="tool inventory", model=llm.model, prompt=prompt,
                reply=reply, seconds=round(time.time()-started, 2), crop_box=box,
                content_size=size, content_offset=offset)
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", reply.strip(), flags=re.I)
    try:
        controls = json.loads(cleaned)["tools"]
        if not isinstance(controls, list):
            raise ValueError("tools must be a list")
        candidates = []
        for control in controls[:8]:
            if (not isinstance(control, dict) or control.get("kind") != "tool"
                    or not isinstance(control.get("appearance"), str) or not control["appearance"].strip()
                    or any(type(control.get(k)) is not int for k in ("x", "y"))):
                continue
            if re.search(r"\b(delete|trash|eraser?|remove|send|share|purchase|buy|cut|paste|or"
                         r"|unknown|unclear|ambiguous|dot|cursor|pointer|crosshair)\b"
                         r"|删除|清空|移除|发送|分享|购买|不明|不确定|光标|指针|圆点",
                         control["appearance"], re.I):
                continue
            try:
                x, y = _screen_point(control["x"], control["y"], box, size, offset, (w, h))
            except RuntimeError:
                continue
            if all(abs(x-c["x"]) + abs(y-c["y"]) > 25 for c in candidates):
                candidates.append({"x": x, "y": y, "appearance": control["appearance"]})
    except (ValueError, KeyError, TypeError) as e:
        raise RuntimeError(f"no usable visible tool inventory: {e}") from e
    if not candidates:
        raise RuntimeError("no safe visible tool candidates were identified")
    return sorted(candidates, key=lambda c: (c["x"], c["y"]))
