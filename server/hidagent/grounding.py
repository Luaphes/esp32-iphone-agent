"""Check a proposed tap against a magnified local screenshot before moving the pointer."""

import io
import re
import time

from PIL import Image

PROMPT = """Locate the intended visible UI target in this cropped screenshot.
Return only POINT X Y, integers 0-1000 relative to THIS CROP, or NOT_FOUND.
The crop is exactly 1000x1000 pixels: use its pixel coordinates directly, not the full-screen coordinates.
Choose the visual center of the actual icon, button or text described. Do not guess a hidden target.
If the target is an explicitly empty text area, blank note body or blank canvas (a place for the cursor, not an
existing icon or text), return POINT at the center of that blank area instead of NOT_FOUND.
If the target is not visible or is ambiguous, return NOT_FOUND."""


async def locate_tap(llm, frame: bytes, action: dict, trace) -> dict:
    """Only refine within a bounded neighborhood; no match means no click."""
    image = Image.open(io.BytesIO(frame)).convert("RGB")
    w, h = image.size
    x, y = action["x"], action["y"]
    if not 0 <= x <= 1000 or not 0 <= y <= 1000:
        raise ValueError("tap coordinates must be within 0-1000")
    box = (max(0, int((x-200)*w/1000)), max(0, int((y-150)*h/1000)),
           min(w, int((x+200)*w/1000)), min(h, int((y+150)*h/1000)))
    crop = image.crop(box).resize((1000, 1000), Image.Resampling.LANCZOS)
    output = io.BytesIO()
    crop.save(output, format="JPEG", quality=90)
    prompt = f"Intended target and purpose: {action.get('thought', '')}\nLocate the intended target in this crop."
    started = time.time()
    reply = await llm.chat(PROMPT, prompt, output.getvalue(), max_tokens=180)
    elapsed = round(time.time()-started, 2)
    trace.write(output.getvalue(), kind="llm", role="grounder", model=llm.model, prompt=prompt,
                reply=reply, seconds=elapsed, crop_box=box, proposed={"x": x, "y": y})
    match = re.fullmatch(r"\s*POINT\s+(\d+)\s+(\d+)\s*", reply, re.I)
    if not match:
        # Some vision providers render a point using their own structured token wrapper.
        match = re.fullmatch(r"\s*<｜｜point｜｜>\[\[(\d+),\s*(\d+)\]\]<｜｜/point｜｜>\s*", reply)
    if not match or not all(0 < int(v) < 1000 for v in match.groups()):
        raise RuntimeError("intended tap target was not located in the crop; inspect the full screen and replan")
    cx, cy = map(int, match.groups())
    return {**action, "x": round((box[0]+cx/1000*(box[2]-box[0]))/w*1000),
            "y": round((box[1]+cy/1000*(box[3]-box[1]))/h*1000),
            "ground_s": elapsed, "proposed": {"x": x, "y": y}}
