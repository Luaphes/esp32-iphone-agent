"""See-think-act loop: a planner model writes the plan, an actor model picks each step."""

import asyncio
import io
import json
import logging
import math
import re
import time
from collections import Counter

from PIL import Image, ImageChops, ImageDraw, ImageFilter

from . import vision
from .debug import Trace
from .device import DeviceError
from .grounding import discover_tools, locate_tap
from .llm import LLM
from .phone import Phone
from .screen import Screen

log = logging.getLogger(__name__)

PLANNER_PROMPT = """You plan tasks on an iPhone or iPad. Given the goal and the current screen,
write a short numbered plan (at most 6 steps) of what to do on screen. Plain text only, in English.
Plan only what the goal needs, from what you can see; never invent accounts, emails, logins or
screens you have not seen. Later steps can stay general (e.g. "Draw the face on the canvas").
If the needed workspace and tools are already visibly available, use them rather than reopening/resetting
fulfilled prerequisites. Observed clear regions may support new work without navigating away.
For content-creation tasks preserve existing content and choose a clearly blank area or a new document.
Do not plan to erase/delete old content merely to make room unless the user explicitly requested that edit.
Pre-existing artwork/text does not count as newly created work for this run.
Keep a coherent work item: after creating its initial structure, continue that same new item instead of
restarting a completed stage or switching to a larger pre-existing object. Follow the goal if multiple items are needed.
To open an app, prefer search over paging through home screens: tap the "Search" pill above
the dock (or swipe down from the middle of the home screen), type the app name, tap the result."""

ACTOR_PROMPT = """You operate an iPhone or iPad through a mouse pointer. You see the current screen.
Reply in this format, nothing else:
THINK: <plain English a non-technical user understands: first check your last action against the
       screen (did it do what you intended?), then what you do now and why>
EXPECT: <only the visible result of THIS next action, determined before executing it>
DO: <action>

Give exactly ONE DO line. After every action you will see a new screenshot before acting again.
For each device action, EXPECT is required. Describe a concrete visible after-condition, not the
old screen, previous failures, a later step or the whole task. For example, dismissing a popover
expects the popover to be absent; opening search expects a search field and keyboard; opening an
app expects that app's screen. Do not change this condition after seeing the result.
Never combine a tap that opens a menu or changes the toolbar with drawing, typing or done: you must
look at the new screen first.
An independent checker reviews each action against the screen. When it reports a mismatch, correct
the visible result before continuing. After three mismatches the task restarts from the home screen.

Coordinates are integers 0-1000 relative to the screenshot (x left->right, y top->bottom).
Aim at the visual center of the element you want, not its edge or its label.
Look before you act: describe what is really on the screen now, not what you expect.
For creation tasks, preserve pre-existing content and create the requested new result in a distinguishable
blank area or a new document. Do not take old artwork/text as work performed in this run.
Use the persistent content journal to keep the same newly created work item. Recorded coordinates are
normalized 0-1000 screen units, not screenshot pixels. Complete its remaining parts instead of repeatedly
starting new structures or switching to an older object merely because it looks clearer.
Do not assume where a toolbar button is because it was there on another device or app; read the
current screen. After opening a menu, look at what it contains instead of repeating the same tap.
If a toolbar tap does not select the intended tool, do not tap the same position again; try the
nearest different toolbar icon to its left or right, one at a time, and inspect what it opens.
Before path or circle, the drawing tool must be visibly selected and no menu may be open. If a menu
opened or a shape got selected instead, close it and choose the actual tool again.
A newly opened or collapsed tool palette is only an entry point: inspect its active mode and expand it
if needed before making strokes. Handwriting-to-text and text tools are different from freehand ink;
do not assume any pencil-shaped control enables the requested editing capability.
For ink marks use stroke actions rather than assuming a tap deposits ink: a tap may only move/select
the pointer. Only persistent marks remaining after the pointer moves count as created content.
Checker references to magnified/cropped details describe diagnostic evidence, not extra device UI.
Do not try to close a diagnostic view; act on the actual current device screen.
Each recent action says whether the screen changed after it. If it did NOT change, the action
may have missed; that pixel heuristic alone is not a verdict. Use the independently checked effect
and the current screen. An already-selected mode can remain valid, but creation requires new marks.
Only type when a text field is visibly focused (the keyboard is showing).
Swipes have no momentum: content moves exactly as far as the finger and stops. Prefer short swipes
(100-250 units) and check the result; repeat if needed rather than one big swipe.
Time or number wheels (e.g. an alarm's time): first tap the number in the wheel's center row and type
the digits (e.g. type 0700); check the result. Only if typing does not work, swipe inside that column:
measure the row spacing from two neighbouring numbers and swipe k times that spacing to move k rows.
Looking for an item in a list: scroll less than one screen at a time so nothing is skipped, and read every
visible item. If you reached both ends of the list and the item is not there, stop with fail and name the
closest options you saw; do not keep scrolling.
Only tap things you can actually see on this screenshot; never guess where a button might be.
If you are unsure what an element does, tap it alone (one DO line) and check the result next round.
Actions:
  tap X Y [precise]         (precise: the pointer is located and corrected before tapping, about 1 s slower;
                             use it for small or tightly packed targets: list rows, checkmarks, toggles, small icons)
  swipe X1 Y1 X2 Y2         (finger moves from X1,Y1 to X2,Y2: "swipe down" means Y2 > Y1)
  path X1 Y1 X2 Y2 X3 Y3 …  (one continuous stroke through 2-20 points, for drawing)
  circle X Y R              (draw a circle centered at X Y; R is the radius in x units)
  type TEXT                 (ASCII only; tap the input field first)
  key COMBO                 (enter, backspace, esc, cmd+a, ...)
  home | back | wait
  replan                    (the plan no longer fits the screen)
  done ANSWER               (the goal is reached; completion is independently verified from the screen)
  fail REASON

Example:
THINK: My last tap opened the home screen as intended. The App Store icon is visible, so I tap it.
EXPECT: The App Store is open and its content is visible.
DO: tap 152 362"""

VERIFY_PROMPT = """Independently verify whether the original phone task is complete from the provided screen evidence.
Do not trust the executor's claim, the plan, or earlier taps as evidence.
For text or drawings, check that the actual content is visible. For settings, check the visible state;
never infer a hidden save or a state that is not shown.
When task-initial BEFORE and current AFTER screens are provided, creation/modification requests must
show work produced in this run, not merely an already existing result. Compare the initial screen and
the executed input record with the current content; input commands alone do not prove the result.
State-only goals may already be satisfied initially. For drawing requests require actual drawn marks,
not a cursor, UI artwork, text/emoji or a pre-existing picture that happens to resemble the requested result.
For creation tasks, all required components must belong to the SAME new work item in a blank area/new
document. Do not count pre-existing partial structures toward that item unless the goal explicitly asks
to edit/complete existing content. A separate new structure plus new parts added to an old object is not
a complete newly created result. Compare object identity and the persistent content journal, not just features.
Do not invent extra features or require cosmetic perfection beyond the original goal.
Reply exactly in this format:
VERDICT: complete
REASON: <one sentence citing visible evidence>
or
VERDICT: incomplete
REASON: <what is still missing or wrong>"""

CONDITION_PROMPT = """Before a single device action executes, establish its necessary visible success conditions.
You see ONLY the BEFORE screen, original user goal, actor reasoning, proposed action and raw EXPECT.
An optional TASK INITIAL reference and persistent journal are also captured before this proposed action.
CURRENT is the action's BEFORE screen; INITIAL is earlier context. Neither is a future result.
No action has happened yet. Do not predict or invent an AFTER image. Do not choose or change the action.
Separate the intended immediate effect from unsupported predictions in EXPECT:
- Opening an application requires that target application to be visibly open. Its unseen landing subpage
  is a prediction, not a requirement, unless the user or the actor's actual immediate intent explicitly
  asks to navigate to that specific subpage. A plan or EXPECT alone cannot invent that requirement.
- Opening search requires the search interface and usable input/keyboard; an invented screen position
  for the field or keyboard is not required. Keep positions when explicitly required by the user or intent.
- Preserve target identity, requested content, tool selection, and explicitly requested pages or states.
  Never weaken selecting a drawing tool to opening any menu, or opening one app to opening any app.
  An application's permission/preferences page inside system Settings is not that application's own
  interface. For app opening require its own content/workspace/navigation, not just its name in Settings.
- A tap on a tool entry control may activate the requested mode OR expose that tool's selection palette.
  Require visible evidence of the intended mode or its relevant palette, not an invented highlight on a
  presumed toolbar icon. Keep a highlight requirement when the user explicitly requires that indicator.
  An unrelated attachment/share/text panel is not the requested tool's palette. Opening a palette does
  not by itself establish that a subsequent drawing stroke is ready: inspect the selected tool first.
  For path/circle, if ink mode is not visibly established (e.g. only text/handwriting mode is visible),
  return unclear and explain the missing prerequisite instead of assuming strokes will work.
- Dismissing a menu requires that specific menu to disappear. Preserve the promised destination state:
  if ordinary Home is requested in the goal, immediate intent OR raw EXPECT, require Home with NO search
  field, keyboard, app or other covering panel. Home icons behind search are not ordinary Home.
  Similarly, a promised unobstructed canvas must have no covering menu. These are state conditions,
  not optional layout predictions. Do not reduce a compound state change to just one clause.
- Only require effects of THIS action; do not require later task steps or preserving the old page.
- For creation, use the initial reference and persistent journal to maintain the SAME new work item.
  If the actor proposes adding components to a pre-existing object instead of its own new structure,
  or repeats an already completed stage while required parts remain, return unclear with a concrete
  correction. Do not silently retarget the action. Respect explicit goals to edit old content or make multiple items.
  Set work_item_consistent to your explicit true/false judgment of that check. False means the proposed
  action must not execute; explain which verified work item or unfinished stage it should continue.
For circle/path declare encloses_remaining_work=true ONLY when this action establishes the outer boundary
of ONE self-contained requested item and its remaining components should lie inside that boundary.
Use false for detail strokes, open/extending designs, independent repeated items, or goals requesting
multiple separate items. Do not mark an eye/detail or a retracing stroke as a new enclosing work boundary.
This declaration is for new independent work requiring clear space; do not use it for annotations or
edits that the user explicitly requests over existing content.
For each condition give its basis: goal, intent, or before (a directly visible relevant prerequisite/target).
Set ordinary_home=true when this action promises an ordinary/normal Home screen in its intent or EXPECT,
including dismissing a menu to reveal ordinary Home. Set false for opening search over Home or opening an
application. This describes the immediate destination, not the BEFORE background or eventual task goal.
The controller adds the full no-overlay Home condition when true; do not weaken it to visible background icons.
For a tap, also return target: a concise description of ONLY the currently intended visible element,
including its label/icon and distinguishing container or row. Exclude previous failed targets, future
results and guessed numeric coordinates. The locator will receive this description without the mixed
actor reasoning. Set top_toolbar=true ONLY if the target belongs to the current application's top
toolbar. A search field, search result, menu item, dialog or Home icon is not a toolbar target merely
because it is near the top. For other actions use target="" and top_toolbar=false.
Also set tool_selection=true only when this tap is intended to select an editing/manipulation tool
for subsequent work. Navigation, opening an application, sharing, dismissing a popup and changing
account settings are not tool selection. Use false for all non-tap actions.
For tool_selection set indicator_required=true ONLY when the original user explicitly requires a visual
highlight/selection indicator. A highlight predicted by the actor in THINK/EXPECT is not a user requirement.
The controller freezes a functional tool-entry condition instead of speculative appearance conditions;
retain every relevant explicit user constraint with basis goal.
List discarded unsupported predictions with a short explanation. If the immediate effect or a requirement
is ambiguous, return status unclear; do not guess or silently relax it. Never use future evidence.
Return ONLY a JSON object:
{"status":"ready","ordinary_home":false,"target":"","top_toolbar":false,"tool_selection":false,
 "indicator_required":false,
 "work_item_consistent":true,
 "encloses_remaining_work":false,
 "required":[{"condition":"...","basis":"intent"}],
 "discarded":["unsupported prediction: why"],"reason":"why these are sufficient and preserve requirements"}
or {"status":"unclear","ordinary_home":false,"required":[],"discarded":[],"reason":"what cannot be established"}.
Use 1-6 concrete, independently observable required conditions when ready. All must hold to pass."""

VALIDATE_PROMPT = """You check one action in a phone task against its predeclared expected visible result.
The image has two labeled panels: BEFORE (context only) and AFTER (the result to judge).
Judge whether AFTER satisfies the expected result, using BEFORE to understand the transition.
A successful transition can remove the previous page, menu or search results. Do not require
AFTER to preserve BEFORE: an absent dismissed popover, an open search field with keyboard, or
the destination app replacing search can be success. Do not reject observed results based on
guesses about what a gesture normally does. Judge only this action, not later steps or the whole task.
Reject wrong menus/tools and missing required content. A changed screen alone is not success.
Identify the foreground application from its actual interface, not just a visible app name. An app's
permission/preferences page inside system Settings does NOT mean that target app is open; reject it
when the action intended to open that app itself. Preserve genuinely requested Settings destinations.
Inspect any labeled AFTER change details: they are magnified original pixels, not executor claims.
When judging a tool action, inspect the actual newly visible palette/mode controls rather than assuming
which toolbar glyph is the intended tool. Even on mismatch, describe the new controls that did appear
so the next action can inspect them; do not say nothing changed when a relevant palette appeared.
Every required clause must hold. Ordinary Home requires NO search field, keyboard or covering panel;
home icons visible behind a search interface do not satisfy Home. Do not overlook a remaining overlay
just because a smaller menu disappeared. Explicitly inspect AFTER for these remaining elements.
If evidence is insufficient, use mismatch. Do not infer hidden effects or trust an actor's claim.
Reply exactly in this format:
VERDICT: ok
REASON: <visible evidence that the intended effect happened>
or
VERDICT: mismatch
REASON: <what is actually visible instead, and why it does not match>"""

META = ("t", "thought", "thought_zh", "think_s", "act_s", "shot", "_shot", "changed", "validated",
        "validation", "validate_s", "verified", "verification", "verify_s", "ground_s", "proposed",
        "explore", "expected", "evidence_id", "fresh", "raw_expected", "conditions", "discarded_predictions",
        "condition_s", "ordinary_home", "target", "top_toolbar", "tool_selection", "tool_candidate",
        "indicator_required", "pointer_park", "park_s", "declared_boundary", "start_pose",
        "pixel_observation")  # not the action

# action name -> argument names; numeric arguments are 0-1000 screen coordinates
ACTIONS = {"tap": ("x", "y"), "swipe": ("x1", "y1", "x2", "y2"), "circle": ("x", "y", "r"), "path": ("points",),
           "type": ("text",), "key": ("combo",), "done": ("answer",), "fail": ("reason",),
           "home": (), "back": (), "wait": (), "replan": ()}
TEXT_ARGS = ("text", "combo", "answer", "reason")


def parse_command(line: str) -> dict:
    """One DO line, e.g. "tap 152 362" -> {"action": "tap", "x": 152, "y": 362}."""
    name, _, rest = line.strip().partition(" ")
    name = name.lower()
    if name not in ACTIONS:
        raise ValueError(f"unknown action: {line[:100]}")
    keys = ACTIONS[name]
    if keys and keys[0] in TEXT_ARGS:
        return {"action": name, keys[0]: rest.strip().strip('"')}
    numbers = [int(float(n)) for n in re.findall(r"-?\d+(?:\.\d+)?", rest)]
    if keys == ("points",):
        if len(numbers) < 4 or len(numbers) % 2:
            raise ValueError(f"path needs 2+ x y pairs: {line[:100]}")
        return {"action": name, "points": [numbers[i:i + 2] for i in range(0, len(numbers), 2)][:20]}
    if len(numbers) != len(keys):
        raise ValueError(f"{name} needs {len(keys)} numbers: {line[:100]}")
    action = {"action": name, **dict(zip(keys, numbers, strict=True))}
    if name == "tap" and "precise" in rest.lower():
        action["precise"] = True
    return action


def parse_actions(text: str) -> list[dict]:
    """Parse a THINK / DO… reply (an optional ZH line is kept); the thought goes on the action."""
    thought = re.search(r"^\s*THINK\s*[:：]\s*(.*?)\s*$", text, re.M | re.I)
    zh = re.search(r"^\s*ZH\s*[:：]\s*(.*?)\s*$", text, re.M | re.I)
    lines = re.findall(r"^\s*DO\s*[:：]\s*(.*?)\s*$", text, re.M | re.I)
    if not lines:
        raise ValueError(f"no DO line in reply: {text[:200]}")
    if len(lines) > 1:
        raise ValueError(f"exactly one DO action is required, got {len(lines)}; act once and inspect the result")
    act = parse_command(lines[0])
    expectations = re.findall(r"^[ \t]*EXPECT[ \t]*[:：][ \t]*(.*?)[ \t]*$", text, re.M | re.I)
    if act["action"] not in ("done", "fail", "replan", "wait") and (len(expectations) != 1 or not expectations[0]):
        raise ValueError("exactly one nonempty EXPECT visible after-condition is required before a device action")
    return [{"thought": thought.group(1) if thought else "", "thought_zh": zh.group(1) if zh else "",
             "expected": expectations[0] if expectations else "", **act}]


def _changed_regions(before: Image.Image, after: Image.Image, ignore: tuple) -> list[tuple]:
    """Small effect regions, including corner palettes; exclude the moved pointer and status clock."""
    if before.size != after.size or min(after.size) < 100:
        return []
    w, h = after.size
    diff = ImageChops.difference(before, after).convert("L")
    mask = ImageDraw.Draw(diff)
    mask.rectangle((0, 0, w, .03*h), fill=0)
    for x, y in ignore:
        r = .045*w
        mask.ellipse((x*w-r, y*h-r, x*w+r, y*h+r), fill=0)
    gw, gh = 96, max(16, round(96*h/w))
    # Threshold at native resolution before pooling: averaging first erased tiny but real ink strokes.
    binary = diff.point(lambda p: 255 if p > 35 else 0).filter(ImageFilter.MaxFilter(9))
    grid = binary.resize((gw, gh), Image.Resampling.BOX).point(lambda p: 255 if p else 0)
    cells = {(i % gw, i // gw) for i, p in enumerate(grid.tobytes()) if p}
    regions = []
    while cells:
        seed = cells.pop()
        component, pending = [seed], [seed]
        while pending:
            x, y = pending.pop()
            for q in ((x-1, y), (x+1, y), (x, y-1), (x, y+1)):
                if q in cells:
                    cells.remove(q)
                    component.append(q)
                    pending.append(q)
        xs, ys = zip(*component, strict=True)
        box = (max(0, int(min(xs)*w/gw)-16), max(0, int(min(ys)*h/gh)-16),
               min(w, int((max(xs)+1)*w/gw)+16), min(h, int((max(ys)+1)*h/gh)+16))
        area = (box[2]-box[0])*(box[3]-box[1])
        if len(component) >= 4 and area < .6*w*h:
            regions.append((area, box))
    return [box for _, box in sorted(regions, reverse=True)[:2]]


def transition_image(before: bytes, after: bytes, ignore: tuple = (), *,
                     labels: tuple = ("AFTER (result)", "BEFORE (context)"), include_details: bool = True,
                     work_area: tuple | None = None) -> bytes:
    """Keep both states in one labeled image without changing the vision API."""
    images = [Image.open(io.BytesIO(frame)).convert("RGB") for frame in (before, after)]
    width, height = max(im.width for im in images), max(im.height for im in images)
    details = _changed_regions(*images, ignore) if include_details else []
    if work_area:
        details = [work_area, work_area]
    elif details:
        details = [details[0], details[0]]
    panel = Image.new("RGB", (width * 2 + (440 if details else 0), height + 32), "white")
    draw = ImageDraw.Draw(panel)
    for i, (label, im) in enumerate(zip(labels, (images[1], images[0]), strict=True)):
        draw.text((i * width + 10, 10), label, fill="black")
        panel.paste(im, (i * width, 32))
    for i, box in enumerate(details):
        y = i * (height // 2)
        label = (("AFTER requested area" if work_area else "AFTER change region")
                 if i == 0 else "BEFORE same area")
        draw.text((2*width+10, y+10), label, fill="black")
        detail = images[1 if i == 0 else 0].crop(box)
        # Enlarge small controls as well as reducing large ones, preserving their original geometry.
        scale = min(420/detail.width, (height//2-45)/detail.height)
        detail = detail.resize((max(1, round(detail.width*scale)), max(1, round(detail.height*scale))),
                               Image.Resampling.LANCZOS)
        panel.paste(detail, (2*width+10, y+35))
    output = io.BytesIO()
    panel.save(output, format="JPEG", quality=85)
    return output.getvalue()


def parse_plan(text: str) -> tuple[str, str]:
    """Split an optional EN: / ZH: plan reply; a plain reply is used for both."""
    m = re.search(r"EN\s*[:：](.*?)ZH\s*[:：](.*)", text, re.S | re.I)
    return (m.group(1).strip(), m.group(2).strip()) if m else (text.strip(), text.strip())


def _key(step: dict) -> dict:
    """The action itself, without thoughts, timings and other bookkeeping."""
    return {k: v for k, v in step.items() if k not in META}


def _toolbar_row(frame: bytes, y: int) -> bytes:
    """Visual scope around an observed toolbar target; no application identity or fixed icon positions."""
    im = Image.open(io.BytesIO(frame)).convert("L")
    w, h = im.size
    return im.crop((0, max(0, (y-30)*h//1000), w, min(h, (y+30)*h//1000))).resize((160, 16)).tobytes()


def _evidence_area(frame, boundary):
    w, h = Image.open(io.BytesIO(frame)).size
    pad = max(16, round((boundary[2]-boundary[0])*w/1000*.15))
    return (max(0, int(boundary[0]*w/1000)-pad), max(0, int(boundary[1]*h/1000)-pad),
            min(w, int(boundary[2]*w/1000)+pad), min(h, int(boundary[3]*h/1000)+pad))


def _crop_frame(frame, area):
    im = Image.open(io.BytesIO(frame)).convert('RGB').crop(area)
    scale = 500/max(im.size)
    im = im.resize((max(1, round(im.width*scale)), max(1, round(im.height*scale))), Image.Resampling.LANCZOS)
    out = io.BytesIO()
    im.save(out, 'JPEG', quality=95)
    return out.getvalue()


class BlindError(RuntimeError):
    """The screen feed is gone; acting now would be guessing."""


class StepCheckError(RuntimeError):
    """An action could not be grounded or checked; count it like a failed step check and restart later."""


class Agent:
    def __init__(self, phone: Phone, screen: Screen, actor: LLM, planner: LLM, trace: Trace,
                 max_steps: int = 40, max_failures: int = 3, max_attempts: int = 3):
        self.phone, self.screen, self.actor, self.planner = phone, screen, actor, planner
        self.trace = trace
        self.max_steps, self.max_failures, self.max_attempts = max_steps, max_failures, max_attempts
        self.task, self.plan, self.steps = "", "", []
        self.shots: dict[int, bytes] = {}  # step index -> screenshot the model decided on
        self._pointer = (0.0, 0.0)  # last pointer position (screen fractions)
        self._job: asyncio.Task | None = None
        self._initial_frame: bytes | None = None
        self._attempts, self._mismatches = 1, 0
        self._history_start = 0  # steps before this belong to an abandoned attempt
        self._last_validation = ""
        self._proposal_feedback, self._proposal_failures = "", 0
        self._creation_boundary: tuple | None = None
        self._creation_boundary_step = 0
        self._work_initial_frame: bytes | None = None
        self._toolbar_failed: int | None = None  # x of the last toolbar tap that did not do what was intended
        self._toolbar_scope: tuple[int, bytes] | None = None
        self._tool_search: dict | None = None  # scoped visible candidates and outcomes, retained across restart
        self._failure_notes: list[str] = []  # failed checks in the current attempt
        self._failed_attempts: list[tuple[int, list[str]]] = []

    @property
    def running(self) -> bool:
        return self._job is not None and not self._job.done()

    def start(self, task: str) -> None:
        if self.running:
            raise RuntimeError("agent already running")
        if not self.screen.live:
            raise RuntimeError("no phone screen: start Screen Mirroring first")
        self.task, self.plan, self.steps, self.shots = task, "", [], {}
        self._pointer = (0.0, 0.0)
        self._attempts, self._mismatches = 1, 0
        self._history_start, self._last_validation = 0, ""
        self._proposal_feedback, self._proposal_failures = "", 0
        self._creation_boundary = None
        self._creation_boundary_step = 0
        self._work_initial_frame = None
        self._toolbar_failed = None
        self._toolbar_scope = None
        self._tool_search = None
        self._failure_notes, self._failed_attempts = [], []
        self.trace.start(task)
        self._initial_frame = self.screen.frame
        self.trace.write(self._initial_frame, kind="task_initial")
        self._job = asyncio.create_task(self._run())

    def stop(self) -> None:
        if self.running:
            self._job.cancel()

    def clear(self) -> None:
        if self.running:
            raise RuntimeError("stop the agent first")
        self.task, self.plan, self.steps, self.shots = "", "", [], {}

    def _record(self, shot: bytes | None = None, **step) -> None:
        if shot:
            self.shots[len(self.steps)] = shot
        self.steps.append({"t": time.time(), "shot": bool(shot), **step})
        self.trace.write(kind="step", **step)
        log.info("agent: %s", step)

    async def _ask(self, role: str, llm: LLM, system: str, prompt: str,
                   temperature: float = 0, *, frame: bytes | None = None) -> tuple[str, float, bytes | None]:
        if not self.screen.live:  # never let the model act blind
            raise BlindError("lost the phone screen: reconnect Screen Mirroring")
        frame, started = frame if frame is not None else self.screen.frame, time.time()
        reply = await llm.chat(system, prompt, frame, temperature=temperature)
        seconds = round(time.time() - started, 2)
        self.trace.write(frame, kind="llm", role=role, model=llm.model, prompt=prompt,
                         reply=reply, seconds=seconds)
        return reply, seconds, frame

    async def _run(self) -> None:
        failures = 0
        try:
            await self._replan()
            for _ in range(self.max_steps):
                try:
                    if await self._step():
                        return
                    failures = 0
                except BlindError:
                    raise
                except Exception as e:  # a bad reply or a dropped link: recover, don't crash
                    self._record(action="hiccup", reason=str(e)[:300])
                    board_drop = isinstance(e, DeviceError) and await self.phone.device.wait_connected(15)
                    failures += 0 if board_drop else 1  # the board came back: not the agent's fault
                    if failures >= self.max_failures:
                        self._record(action="fail", reason=f"{failures} errors in a row")
                        return
                    await self.screen.settled()
                    await self._replan(stuck=True)
            self._record(action="fail", reason="step limit reached")
        except asyncio.CancelledError:
            self._record(action="stopped")
        except Exception as e:  # e.g. the planner itself is unreachable
            self._record(action="error", reason=repr(e))

    async def _step(self) -> bool:
        """One see-think-act round (exactly one action). Returns True when the task is over."""
        act = (await self._next_actions())[0]
        shot = act.pop("_shot")
        if act["action"] == "replan" or self._repeating(act):
            self._record(shot, **act)
            await self._replan(stuck=True)
            return False
        if act["action"] == "done":
            await self.screen.settled(quiet=.2, least=.2, most=1.0)
            ok, reply, seconds, _ = await self._verify(act.get("answer", ""))
            act.update(verified=ok, verification=reply, verify_s=seconds)
            self._record(shot, **act)
            if ok:
                return True
            self._record(action="hiccup", reason=f"completion not verified: {reply}"[:300])
            await self._replan(stuck=True)
            return False
        if act["action"] == "fail":
            self._record(shot, **act)
            return True
        act["evidence_id"] = len(self.steps)
        if act["action"] != "wait":
            try:
                act = await self._prepare_action(act, shot)
            except ValueError as e:
                self._proposal_failures += 1
                self._proposal_feedback = str(e)
                self._record(shot, action="hiccup", reason=f"proposal not executed: {e}",
                             evidence_id=act["evidence_id"])
                if self._proposal_failures >= self.max_failures:
                    self._record(action="fail", reason="bounded proposal corrections exhausted; no action executed")
                    return True
                return False
            self._proposal_feedback, self._proposal_failures = "", 0
        if act["action"] == "tap" and act.get("tool_selection"):
            try:
                act = await self._tool_candidate(act, shot)
            except StepCheckError as e:
                return await self._step_check_failed(act, str(e), shot)
        if act["action"] == "tap" and not act.get("explore"):
            try:
                act = await self._ground_tap(act, shot)
            except StepCheckError as e:
                return await self._step_check_failed(act, str(e), shot)
        started = time.time()
        evidence_id = act["evidence_id"]
        self.trace.write(shot, kind="action_before", evidence_id=evidence_id, expected=act["expected"],
                         action=_key(act))
        try:
            await self._do(act)
        except StepCheckError as e:
            return await self._step_check_failed(act, str(e), shot)
        if act["action"] in ("path", "circle", "tap"):
            # _do returns only after drag released; never move the pressed pointer to its parking point.
            stroke = (circle(act["x"]/1000, act["y"]/1000, act["r"]/1000, self.screen.aspect)
                      if act["action"] == "circle" else [(act["x"]/1000, act["y"]/1000)]
                      if act["action"] == "tap" else [(x/1000, y/1000) for x, y in act["points"]])
            xs, ys = zip(*stroke, strict=True)
            park = self._pointer
            if min(xs)-.08 <= park[0] <= max(xs)+.08 and min(ys)-.08 <= park[1] <= max(ys)+.08:
                park = (.97 if (min(xs)+max(xs))/2 < .5 else .03,
                        .97 if (min(ys)+max(ys))/2 < .5 else .03)
            parked_at = time.time()
            await self.phone.point(*park)
            act.update(pointer_park=park, park_s=round(time.time()-parked_at, 2))
            self.trace.write(kind="pointer_park", evidence_id=evidence_id, position=park,
                             reason="released pointer moved outside the new stroke for visual evidence")
        act["act_s"] = round(time.time() - started, 2)
        fresh = await self.screen.fresh(started, timeout=2)
        await self.screen.settled()
        after = self.screen.frame
        if not self.screen.live:
            raise BlindError("lost the phone screen after the action")
        spots = [self._pointer, act["pointer_park"]] if "pointer_park" in act else [self._pointer, *_pointer_path(act)]
        self._pointer = spots[-1]
        changed = vision.changed(shot, after, ignore=tuple(spots))
        if act['action'] in ('circle', 'path'):
            area = _evidence_area(after, self._stroke_bounds(act))
            act['pixel_observation'] = self._pixel_observation(shot, after, area)
            changed = changed or act['pixel_observation']['changed_pixels'] > 2
        act.update(changed=changed, fresh=fresh is not None)
        if act["action"] == "wait":
            self._record(shot, **act)
            return False
        if fresh is None:
            ok, reason = False, "no fresh screen frame arrived after the action; result unconfirmed"
        else:
            ok, reason, seconds, _ = await self._validate_action(act, shot, after)
            act["validate_s"] = seconds
        act.update(validated=ok, validation=reason)
        self.trace.write(after, kind="action_after", evidence_id=evidence_id, expected=act["expected"],
                         validated=ok, validation=reason, fresh=act["fresh"])
        self._record(shot, **act)
        if ok:
            self._accept_work_boundary(act, after, shot)
            self._last_validation = ""
            if act["action"] == "tap" and act.get("top_toolbar"):
                self._toolbar_failed = None
                self._toolbar_scope = None
            if act.get("tool_selection"):
                self._toolbar_failed, self._toolbar_scope, self._tool_search = None, None, None
            return False
        return await self._step_check_failed(act, reason, shot)

    async def _step_check_failed(self, act: dict, reason: str, before: bytes | None = None) -> bool:
        """Count a failed step check; after three, restart from the home screen."""
        self._mismatches += 1
        self._last_validation = reason
        if act.get("action") == "tap" and act.get("top_toolbar"):
            self._toolbar_failed = act.get("x")
            self._toolbar_scope = (act["y"], _toolbar_row(before, act["y"])) if before else None
        if act.get("action") == "tap" and act.get("tool_selection") and before:
            search = self._tool_search
            if not search or not self._same_tool_row(before, search):
                w, h = Image.open(io.BytesIO(before)).size
                region = (0, max(0, int((act['y']/1000-.15)*h)), w,
                          min(h, int((act['y']/1000+.15)*h)))
                search = self._tool_search = {"y": act["y"], "row": _toolbar_row(before, act["y"]),
                                              "tried": [], "candidates": None, "child_regions": [],
                                              "region": None if act.get('top_toolbar') else region,
                                              "target": act.get("target", "")}
            if all(abs(act["x"]-p["x"]) + abs(act["y"]-p["y"]) > 25 for p in search["tried"]):
                search["tried"].append({"x": act["x"], "y": act["y"], "result": reason[:300]})
            after = self.screen.frame if self.screen else None
            if after:
                a, b = (Image.open(io.BytesIO(f)).convert("RGB") for f in (before, after))
                w, h = b.size
                regions = _changed_regions(a, b, tuple(_pointer_path(act)))
                search["child_regions"] = [box for box in regions if box[1] > .18*h and (
                    (box[2]-box[0])*(box[3]-box[1]) < .08*w*h or (
                        (box[2]-box[0]) > 3*(box[3]-box[1])
                        and (box[2]-box[0])*(box[3]-box[1]) < .2*w*h))]
        self._failure_notes.append(f"{act.get('action')} {json.dumps(_key(act))} -> {reason}"[:300])
        self._record(action="hiccup", reason=f"step check: {reason}"[:300])
        if self._mismatches >= 3:
            return await self._restart()
        return False

    @staticmethod
    def _same_tool_row(frame: bytes, search: dict) -> bool:
        current = _toolbar_row(frame, search["y"])
        return sum(abs(a-b) for a, b in zip(search["row"], current, strict=True)) / len(current) <= 8

    async def _tool_candidate(self, act: dict, before: bytes) -> dict:
        """After a real failed tool selection, inspect current visible candidates instead of snapping back."""
        search = self._tool_search
        if not search or not self._same_tool_row(before, search):
            return act
        # Inspect compact newly exposed tool controls before moving to another unrelated parent control.
        while search.get("child_regions"):
            region = search["child_regions"].pop(0)
            try:
                children = await discover_tools(self.actor, before, self.trace, region=region)
            except (ValueError, RuntimeError) as e:
                self.trace.write(kind="tool_child_unconfirmed", region=region, reason=str(e))
                continue
            untried = [c for c in children if all(abs(c["x"]-p["x"]) + abs(c["y"]-p["y"]) > 35
                                                 for p in search["tried"])]
            if untried:
                return self._explore_control(act, before, untried[0], search, region)
        if search["candidates"] is None:
            try:
                search["candidates"] = await discover_tools(self.actor, before, self.trace,
                                                            region=search.get('region'))
            except (ValueError, RuntimeError) as e:
                raise StepCheckError(str(e)) from e
        if not self.screen.live:
            raise BlindError("lost screen during tool discovery")
        if self.screen.frame != before and vision.scene_changed(before, self.screen.frame):
            raise StepCheckError("screen changed during tool discovery; observe before selecting a candidate")
        for candidate in search["candidates"]:
            if all(abs(candidate["x"]-p["x"]) + abs(candidate["y"]-p["y"]) > 35 for p in search["tried"]):
                return self._explore_control(act, before, candidate, search)
        raise StepCheckError("all visible safe tool candidates have already failed; no blind toolbar scan")

    def _explore_control(self, act, before, candidate, search, region=None):
        if not self.screen.live:
            raise BlindError("lost screen during tool discovery")
        if self.screen.frame != before and vision.scene_changed(before, self.screen.frame):
            raise StepCheckError("screen changed during tool discovery; observe before selecting a candidate")
        refined = {**act, "x": candidate["x"], "y": candidate["y"], "precise": True,
                   "proposed": {"x": act["x"], "y": act["y"]}, "explore": True,
                   "top_toolbar": act.get("top_toolbar", False) if region is None else False,
                   "tool_candidate": candidate["appearance"]}
        self.trace.write(before, kind="tool_exploration", evidence_id=act.get("evidence_id"),
                         target=act.get("target"), candidate=candidate, tried=search["tried"],
                         expected=act["expected"], child_region=region)
        return refined

    async def _prepare_action(self, act: dict, before: bytes) -> dict:
        """Freeze necessary conditions from pre-action evidence; AFTER is never available here."""
        if act["action"] in ("circle", "path") and self._creation_boundary:
            bounds = self._stroke_bounds(act)
            outer = self._creation_boundary
            if not self._inside_work_boundary(act):
                raise ValueError(f"proposed stroke bounds {bounds} leave this run's verified work item {outer}; "
                                 "continue the same newly created item using normalized screen coordinates")
            w, h = Image.open(io.BytesIO(before)).size
            span = max((bounds[2]-bounds[0])*w/1000, (bounds[3]-bounds[1])*h/1000)
            minimum = .016*max(w, h)  # twice the normalized starting-pose uncertainty
            if span < minimum:
                raise ValueError(f"stroke span {span:.1f}px is below the verifiable input extent {minimum:.1f}px "
                                 f"({minimum/w*1000:.1f} horizontal or {minimum/h*1000:.1f} vertical normalized "
                                 "units). Propose a clearly larger mark within the same item; "
                                 "tiny motions may be treated as taps")
        prompt = (f"Original user goal: {self.task}\nActor reasoning: {act.get('thought', '')}\n"
                  f"Proposed action: {json.dumps(_key(act))}\nRaw EXPECT: {act.get('expected', '')}\n"
                  f"Persistent content journal (this run only): {self._content_journal()}\n"
                  "Establish only this action's necessary visible results before it executes.")
        evidence = (transition_image(self._initial_frame, before,
                     labels=("CURRENT (before proposed action)", "TASK INITIAL reference"), include_details=False)
                    if self._initial_frame and self._content_journal() != "[]" else before)
        elapsed, problem = 0, ""
        for attempt in range(2):
            reply, seconds, _ = await self._ask("condition builder", self.planner or self.actor,
                                               CONDITION_PROMPT, prompt, .2 * attempt, frame=evidence)
            elapsed += seconds
            try:
                contract = json.loads(reply)
                required, discarded = contract["required"], contract["discarded"]
                if contract["status"] != "ready":
                    raise ValueError(f"unclear conditions: {contract.get('reason', '')}")
                if (not isinstance(contract.get("ordinary_home"), bool)
                        or (act["action"] == "home" and not contract["ordinary_home"])
                        or not isinstance(required, list) or not 1 <= len(required) <= 6
                        or any(not isinstance(c, dict) or c.get("basis") not in ("goal", "intent", "before")
                               or not isinstance(c.get("condition"), str) or not c["condition"].strip()
                               for c in required)
                        or not isinstance(discarded, list) or any(not isinstance(c, str) for c in discarded)
                        or not isinstance(contract.get("reason"), str) or not contract["reason"].strip()):
                    raise ValueError("invalid condition contract")
                if act["action"] == "tap" and (not isinstance(contract.get("target"), str)
                                                or not contract["target"].strip()
                                                or not isinstance(contract.get("top_toolbar"), bool)
                                                or not isinstance(contract.get("tool_selection"), bool)):
                    raise ValueError("tap requires an isolated target and a current-UI toolbar classification")
                if contract.get("tool_selection") and not isinstance(contract.get("indicator_required"), bool):
                    raise ValueError("tool selection must distinguish user-required indicators from predictions")
                if act["action"] in ("circle", "path", "type") and self._content_journal() != "[]":
                    if contract.get("work_item_consistent") is not True:
                        raise ValueError(f"creation proposal conflicts with this run's work item: {contract['reason']}")
                if act["action"] in ("circle", "path"):
                    if not isinstance(contract.get("encloses_remaining_work"), bool):
                        raise ValueError("stroke must declare whether it establishes one enclosing work boundary")
                    if contract["encloses_remaining_work"] and self._creation_boundary:
                        raise ValueError("this run already established its enclosing work item; complete its parts")
            except (ValueError, KeyError, TypeError) as e:
                problem = str(e)
                self.trace.write(kind="condition_rejected", evidence_id=act.get("evidence_id"),
                                 attempt=attempt + 1, reason=problem)
                prompt += f"\nYour condition reply was rejected: {problem}. Resolve it using only BEFORE."
                continue
            if contract["ordinary_home"]:
                required.append({"condition": "The ordinary device Home screen is visible, with NO search field, "
                                 "NO keyboard, NO application and NO covering panel; background Home icons alone "
                                 "are insufficient.", "basis": "intent"})
            if contract.get("tool_selection"):
                if not contract["indicator_required"]:
                    discarded.extend(f"non-user appearance prediction replaced by tool entry: {c['condition']}"
                                     for c in required if c["basis"] != "goal")
                    required = [c for c in required if c["basis"] == "goal"]
                required.append({"condition": f"The editing capability intended by '{contract['target']}' is "
                                 "visibly active, OR its corresponding tool-selection palette/launcher is now "
                                 "visible for further selection. A different tool's controls or an unrelated "
                                 "attachment/share/navigation menu do not satisfy this condition; no unrelated "
                                 "blocking overlay may prevent selecting the intended capability.", "basis": "intent"})
            prepared = {**act, "raw_expected": act.get("expected", ""), "conditions": required,
                        "ordinary_home": contract["ordinary_home"],
                        "target": contract.get("target", ""), "top_toolbar": contract.get("top_toolbar", False),
                        "tool_selection": contract.get("tool_selection", False),
                        "indicator_required": contract.get("indicator_required", False),
                        "discarded_predictions": discarded, "condition_s": round(elapsed, 2),
                        "expected": "\n".join(f"- {c['condition'].strip()}" for c in required)}
            if contract.get("encloses_remaining_work"):
                prepared["declared_boundary"] = self._stroke_bounds(act)
                clear, stats = self._blank_work_area(before, prepared["declared_boundary"])
                self.trace.write(before, kind="work_area_check", evidence_id=act.get("evidence_id"),
                                 clear=clear, boundary=prepared["declared_boundary"], **stats)
                if not clear:
                    candidates = self._clear_work_areas(before, prepared["declared_boundary"])
                    self.trace.write(kind="clear_work_candidates", evidence_id=act.get("evidence_id"),
                                     candidates=candidates, source="current BEFORE pixels, proposed item dimensions")
                    raise ValueError("the proposed new enclosing work area contains pre-existing high-contrast "
                                     "content; preserve it. Pixel-derived clear areas fitting the proposed size: "
                                     f"{json.dumps(candidates)}. Choose clear space or a new document; "
                                     "do not erase old work")
            self.trace.write(before, kind="action_conditions", evidence_id=act.get("evidence_id"),
                             raw_expected=prepared["raw_expected"], conditions=required,
                             discarded_predictions=discarded, expected=prepared["expected"],
                             ordinary_home=prepared["ordinary_home"],
                             target=prepared["target"], top_toolbar=prepared["top_toolbar"],
                             tool_selection=prepared["tool_selection"],
                             indicator_required=prepared["indicator_required"],
                             declared_boundary=prepared.get("declared_boundary"),
                             reason=contract["reason"], seconds=prepared["condition_s"])
            return prepared
        raise ValueError(f"could not establish pre-action conditions after two replies: {problem}")

    def _stroke_bounds(self, act):
        if act["action"] == "circle":
            ry = act["r"] * self.screen.aspect
            return (act["x"]-act["r"], act["y"]-ry, act["x"]+act["r"], act["y"]+ry)
        xs, ys = zip(*act["points"], strict=True)
        return min(xs), min(ys), max(xs), max(ys)

    @staticmethod
    def _workspace_mask(frame):
        image = Image.open(io.BytesIO(frame)).convert("RGB")
        sampled = image.resize((96, 144)).tobytes()
        bins = Counter(tuple(sampled[i+j]//32 for j in range(3)) for i in range(0, len(sampled), 3))
        dominant = bins.most_common(1)[0][0]
        background = tuple(v*32+16 for v in dominant)
        channels = ImageChops.difference(image, Image.new('RGB', image.size, background)).split()
        contrast = ImageChops.lighter(ImageChops.lighter(channels[0], channels[1]), channels[2])
        return contrast.point(lambda p: 255 if p > 70 else 0), background

    @staticmethod
    def _blank_work_area(frame, boundary):
        """Conservative occupancy check for DS-declared NEW independent work, not annotations/edits."""
        mask, background = Agent._workspace_mask(frame)
        box = _evidence_area(frame, boundary)
        if box[2] <= box[0] or box[3] <= box[1]:
            return False, {"occupied_fraction": 1, "background": background}
        occupied = mask.crop(box).histogram()[255]
        fraction = occupied/max(1, (box[2]-box[0])*(box[3]-box[1]))
        return not (occupied > 25 and fraction > .003), {"occupied_fraction": round(fraction, 5),
                                                         "background": background}

    @staticmethod
    def _clear_work_areas(frame, boundary):
        """Propose observed clear regions, not fixed UI locations or task-specific geometry."""
        mask, _ = Agent._workspace_mask(frame)
        w, h = mask.size
        bw = max(1, round((boundary[2]-boundary[0])*w/1000))
        bh = max(1, round((boundary[3]-boundary[1])*h/1000))
        pad = max(16, round(.15*bw))
        original = ((boundary[0]+boundary[2])*w/2000, (boundary[1]+boundary[3])*h/2000)
        options = []
        for y in range(pad, h-bh-pad+1, max(12, bh//2)):
            for x in range(pad, w-bw-pad+1, max(12, bw//2)):
                area = (x-pad, y-pad, x+bw+pad, y+bh+pad)
                occupied = mask.crop(area).histogram()[255]
                if occupied <= 25 or occupied/((bw+2*pad)*(bh+2*pad)) <= .003:
                    distance = (x+bw/2-original[0])**2 + (y+bh/2-original[1])**2
                    options.append((distance, x, y))
        result = []
        for _, x, y in sorted(options):
            bounds = (round(x/w*1000), round(y/h*1000), round((x+bw)/w*1000), round((y+bh)/h*1000))
            center = ((bounds[0]+bounds[2])//2, (bounds[1]+bounds[3])//2)
            if Agent._blank_work_area(frame, bounds)[0] and all(abs(center[0]-c['center'][0]) >= bw/w*800
                   or abs(center[1]-c['center'][1]) >= bh/h*800 for c in result):
                result.append(dict(bounds=bounds, center=center))
                if len(result) == 3:
                    break
        return result

    @staticmethod
    def _observed_clear_spaces(frame):
        """Large low-contrast rectangles from current pixels, advisory until rechecked before input."""
        mask, _ = Agent._workspace_mask(frame)
        w, h = mask.size
        gw, gh = 48, 72
        grid = mask.resize((gw, gh), Image.Resampling.BOX).tobytes()
        heights, options = [0]*gw, []
        for y in range(gh):
            for x in range(gw):
                heights[x] = heights[x]+1 if grid[y*gw+x] == 0 else 0
            stack = []
            for x in range(gw+1):
                current = heights[x] if x < gw else 0
                while stack and heights[stack[-1]] > current:
                    height = heights[stack.pop()]
                    left = stack[-1]+1 if stack else 0
                    if height >= 4 and x-left >= 4:
                        options.append(((x-left)*height, left, y-height+1, x, y+1))
                stack.append(x)
        result = []
        for _, left, top, right, bottom in sorted(options, reverse=True):
            x1, y1, x2, y2 = left*w/gw, top*h/gh, right*w/gw, bottom*h/gh
            inset = max(16, (x2-x1)*.15)
            x1, y1, x2, y2 = x1+inset, y1+inset, x2-inset, y2-inset
            if min(x2-x1, y2-y1) < 80:
                continue
            bounds = (round(x1/w*1000), round(y1/h*1000), round(x2/w*1000), round(y2/h*1000))
            if any(max(0, min(bounds[2], r['bounds'][2])-max(bounds[0], r['bounds'][0])) *
                   max(0, min(bounds[3], r['bounds'][3])-max(bounds[1], r['bounds'][1])) >
                   .3*(bounds[2]-bounds[0])*(bounds[3]-bounds[1]) for r in result):
                continue
            if Agent._blank_work_area(frame, bounds)[0]:
                result.append(dict(bounds=bounds, center=((bounds[0]+bounds[2])//2, (bounds[1]+bounds[3])//2)))
                if len(result) == 3:
                    break
        return result

    def _workspace_observation(self):
        if self._creation_boundary or not getattr(self.screen, 'geometry', None):
            return '[]'
        return json.dumps(self._observed_clear_spaces(self.screen.frame))

    def _accept_work_boundary(self, act, after, before=None):
        if act.get("declared_boundary") and self._creation_boundary is None:
            self._creation_boundary = tuple(act["declared_boundary"])
            self._creation_boundary_step = len(self.steps)
            self._work_initial_frame = before or self._initial_frame
            self.trace.write(self._work_initial_frame, kind="work_item_start", evidence_id=act.get("evidence_id"),
                             boundary=self._creation_boundary)
            self.trace.write(after, kind="work_item_bound", evidence_id=act.get("evidence_id"),
                             boundary=self._creation_boundary, source_input=_key(act),
                             reason="DS declared an enclosing item before input; "
                                    "its creation was independently checked")

    def _inside_work_boundary(self, act):
        bounds, outer = self._stroke_bounds(act), self._creation_boundary
        return all(bounds[i] >= outer[i]-4 for i in (0, 1)) and all(bounds[i] <= outer[i]+4 for i in (2, 3))

    @staticmethod
    def _pixel_observation(before, after, area):
        a, b = (Image.open(io.BytesIO(f)).convert('RGB') for f in (before, after))
        old_mask, background = Agent._workspace_mask(before)
        channels = ImageChops.difference(b, Image.new('RGB', b.size, background)).split()
        new_mask = ImageChops.lighter(ImageChops.lighter(channels[0], channels[1]), channels[2])
        new_mask = new_mask.point(lambda p: 255 if p > 70 else 0)
        added = ImageChops.subtract(new_mask, old_mask).crop(area).histogram()[255]
        channels = ImageChops.difference(a, b).split()
        delta = ImageChops.lighter(ImageChops.lighter(channels[0], channels[1]), channels[2])
        changed = delta.crop(area).point(lambda p: 255 if p > 35 else 0).histogram()[255]
        return dict(new_contrast_pixels=added, changed_pixels=changed, source='original BEFORE/AFTER pixels')

    async def _validate_action(self, act: dict, before: bytes, after: bytes) -> tuple[bool, str, float, bytes | None]:
        """Independently check that the action produced the visible effect the actor intended."""
        if not act.get("expected", "").strip():
            return False, "missing predeclared expected visible result", 0, after
        prompt = (f"Expected visible result of this action (declared BEFORE execution): {act['expected']}\n"
                  f"Action just performed: {act['action']}\n\n"
                  "AFTER is the left/main result panel; BEFORE is context. Compare the labels and inspect "
                  "any magnified AFTER details. Evaluate only the after-condition; do not re-derive a target "
                  "position or confuse screenshot panels with original device coordinates.")
        evidence = None
        if act['action'] in ('circle', 'path'):
            scope = self._creation_boundary or act.get('declared_boundary') or self._stroke_bounds(act)
            area = _evidence_area(after, scope)
            measured = self._pixel_observation(before, after, _evidence_area(after, self._stroke_bounds(act)))
            prompt += (f"\nThis is a focused comparison of the SAME work area from the full screens: {scope}. "
                       f"Measured changes near the requested stroke: {json.dumps(measured)}. These are pixel "
                       "observations, not a success claim. Reconcile them with the images; do not call a "
                       "real added mark 'unchanged' merely because it is small. Judge its requested meaning.")
            evidence = transition_image(_crop_frame(before, area), _crop_frame(after, area), include_details=False)
        reply, seconds, _ = await self._ask("step verifier", self.planner or self.actor, VALIDATE_PROMPT, prompt,
                                           frame=evidence or transition_image(before, after,
                                               (tuple(act["pointer_park"]),) if act.get("pointer_park") else
                                               tuple(_pointer_path(act))
                                               if act["action"] == "tap" and "x" in act and "y" in act else (),
                                               work_area=_evidence_area(after, self._stroke_bounds(act))
                                               if act["action"] in ("circle", "path") else None))
        match = re.fullmatch(r"\s*VERDICT[ \t]*[:：][ \t]*(ok|mismatch)[ \t]*\r?\n"
                             r"REASON[ \t]*[:：][ \t]*(\S[^\r\n]*)[ \t]*\s*", reply, re.I)
        detail = match.group(2).strip() if match else f"invalid check response: {reply.strip()}"
        return bool(match and match.group(1).lower() == "ok"), detail, seconds, after

    async def _restart(self) -> bool:
        """After three failed step checks, return home and run a fresh attempt (bounded)."""
        if self._attempts >= self.max_attempts:
            self._record(action="fail", reason=f"three step checks failed in attempt {self._attempts}; "
                                               "no restart attempts left")
            return True
        self._attempts += 1
        self._failed_attempts.append((self._attempts - 1, self._failure_notes[-6:]))
        self._failure_notes = []
        self._record(action="restart", reason=f"three step checks failed; restarting attempt "
                                              f"{self._attempts}/{self.max_attempts} from the home screen")
        self._mismatches, self._last_validation = 0, ""
        self._toolbar_failed = None
        self._toolbar_scope = None
        self._pointer, self.plan = (0.0, 0.0), ""
        before, boundary = self.screen.frame, time.time()
        self.trace.write(before, kind="recovery_before", attempt=self._attempts, boundary=boundary)
        try:
            await self.phone.go_home()
            frame = await self._confirm_home(before, boundary)
        except Exception as e:
            self._record(action="fail", reason=f"restart could not confirm the home screen: {e}"[:300])
            return True
        await self._replan(stuck=True, frame=frame)
        self._history_start = len(self.steps)
        return False

    async def _confirm_home(self, before: bytes, boundary: float) -> bytes:
        """HID acknowledgement is not visual recovery; bounded fresh-frame checks precede planning."""
        deadline = time.monotonic() + 8
        for check in range(3):
            remaining = deadline - time.monotonic()
            if remaining <= 0 or await self.screen.fresh(boundary, timeout=remaining) is None:
                raise BlindError("no fresh recovery frame (8 second recovery budget)")
            await self.screen.settled(quiet=.35, least=.25, most=min(1, max(.25, remaining)))
            if not self.screen.live:
                raise BlindError("screen feed lost during recovery")
            frame, observed = self.screen.frame, self.screen.updated
            act = {"action": "home", "expected": "The device home screen with app icons and Dock is visible; "
                   "no application, search overlay or system panel covers it."}
            ok, reason, seconds, _ = await asyncio.wait_for(
                self._validate_action(act, before, frame), timeout=max(.01, deadline - time.monotonic()))
            self.trace.write(frame, kind="recovery_check", attempt=self._attempts, check=check + 1,
                             boundary=boundary, observed=observed, confirmed=ok, reason=reason, seconds=seconds)
            if ok:
                return frame
            boundary = observed  # a new frame is needed before checking again; never recycle the old application
        raise BlindError("home screen not confirmed within three recovery checks")

    async def _ground_tap(self, action: dict, frame: bytes) -> dict:
        """Refine a tap against the screenshot the actor decided on; never click a stale screen."""
        try:
            refined = await locate_tap(self.actor, frame, action, self.trace)
        except (RuntimeError, ValueError) as e:
            raise StepCheckError(str(e)) from e
        if not self.screen.live:
            raise BlindError("lost the phone screen: reconnect Screen Mirroring")
        if self.screen.frame != frame and vision.scene_changed(frame, self.screen.frame):
            raise StepCheckError("screen changed while locating the target; discard the tap and observe again")
        return refined

    async def _verify(self, answer: str) -> tuple[bool, str, float, bytes | None]:
        """An independent model check before accepting done; a claim alone is not evidence."""
        current = self.screen.frame
        inputs = [_key(s)
                  for s in self.steps if s.get("action") in ACTIONS and "validated" in s]
        prompt = (f"Original goal: {self.task}\n\n"
                  f"Actual input record from this run (not the actor's plan): {json.dumps(inputs)}\n\n"
                  f"Verified new work-item boundary (normalized screen units): {self._creation_boundary}\n\n"
                  "AFTER is the current result. BEFORE is the work-start image captured before this new item "
                  "was created, or the task's INITIAL screen if no item has been established. "
                  "Inspect the declared work-area crops when present; do not borrow "
                  "features from other objects. No executor or step-checker completion claims are supplied. "
                  "Verify the whole goal and, for creation, that the new work itself contains every required feature.")
        self.trace.write(current, kind="completion_screen", input_count=len(inputs), claimed_answer=answer)
        area = None
        if self._creation_boundary:
            area = _evidence_area(current, self._creation_boundary)
        reference = self._work_initial_frame or self._initial_frame
        evidence = (transition_image(reference, current, work_area=area,
                    labels=("AFTER current result", "BEFORE work start / task initial")) if reference else current)
        reply, seconds, _ = await self._ask("verifier", self.planner or self.actor, VERIFY_PROMPT, prompt,
                                           frame=evidence)
        match = re.search(r"VERDICT\s*[:：]\s*(complete|incomplete|failed)", reply, re.I)
        reason = re.search(r"REASON\s*[:：]\s*(.+)", reply, re.I | re.S)
        detail = reason.group(1).strip() if reason else reply.strip()
        violations = ([s for s in self.steps[self._creation_boundary_step:]
                       if s.get("action") in ("circle", "path") and s.get("validated") is True
                       and not self._inside_work_boundary(s)] if self._creation_boundary else [])
        if violations:
            detail = f"creation record left the verified work-item boundary; cannot accept completion. DS: {detail}"
        return bool(match and match.group(1).lower() == "complete" and not violations), detail, seconds, current

    def _content_journal(self) -> str:
        """Persistent evidence of native content operations, including those before a recovery."""
        return json.dumps([{"input": _key(s), "checked_visible_effect": s.get("validation", "")[:300]}
                           for s in self.steps if s.get("action") in ("circle", "path", "type")
                           and s.get("validated") is True], ensure_ascii=False)

    def _history(self, limit: int = 8) -> str:
        hidden = ("t", "plan", "plan_zh", "think_s", "act_s", "shot", "_shot", "thought", "thought_zh",
                  "changed", "validated", "validation", "validate_s", "verified", "verification",
                  "verify_s", "ground_s", "proposed", "explore", "raw_expected", "conditions",
                  "discarded_predictions", "condition_s", "evidence_id", "fresh", "ordinary_home", "expected",
                  "target", "top_toolbar", "tool_selection", "tool_candidate", "indicator_required",
                  "pointer_park", "park_s", "declared_boundary", "start_pose", "pixel_observation")
        effect = {True: " -> screen changed", False: " -> screen did NOT change"}
        return "\n".join(json.dumps({k: v for k, v in s.items() if k not in hidden})[:300]
                         + effect.get(s.get("changed"), "")
                         + (f" -> checked effect: {s.get('validation', '')[:240]}"
                            if s.get("validated") is True else "")
                         for s in self.steps[self._history_start:][-limit:] if "plan" not in s) or "(none)"

    def _missed(self) -> bool:
        """The previous round changed nothing on screen, so a tap may have landed off target."""
        return (bool(self.steps) and self.steps[-1].get("changed") is False
                and self.steps[-1].get("validated") is not True)

    def _repeating(self, act: dict, limit: int = 3) -> bool:
        """True when the actor picks the same action as its last limit-1 moves and the last one did nothing.

        Repeating a move that still changes the screen (e.g. scrolling down a long list) is progress, not a loop.
        """
        moves = [s for s in self.steps[self._history_start:] if "plan" not in s][-(limit - 1):]
        return (len(moves) == limit - 1 and all(_key(m) == _key(act) for m in moves)
                and moves[-1].get("changed") is not True)

    async def _replan(self, stuck: bool = False, *, frame: bytes | None = None) -> None:
        prompt = (f"Goal: {self.task}\nVerified content from THIS run: {self._content_journal()}\n"
                  f"Active work-item boundary: {self._creation_boundary}\n"
                  f"Current pixel-derived clear regions (normalized): {self._workspace_observation()}\n"
                  "Clear pixels are not proof of an editable canvas. If the required editor is visibly open, "
                  "use suitable clear space without unnecessarily navigating away; preserve all existing work.")
        if stuck:
            last = self.steps[-1] if self.steps else {}
            prompt += (f"\n\nThe previous plan got stuck (last event: {last.get('action')} {last.get('reason', '')}):"
                       f"\n{self.plan}\nLook at the current screen and plan from here, another way if needed.")
            summary = self._failure_summary()
            if summary:
                prompt += "\n\n" + summary
        reply, seconds, frame = await self._ask("planner", self.planner, PLANNER_PROMPT, prompt, frame=frame)
        self.plan, plan_zh = parse_plan(reply)
        self._record(frame, action="replan (stuck)" if stuck else "plan", plan=self.plan, plan_zh=plan_zh,
                     think_s=seconds)

    def _failure_summary(self) -> str:
        if not self._failed_attempts:
            return ""
        lines = ["Previous attempts failed. Do not repeat these methods; choose a different visible target "
                 "or input method:"]
        for attempt, notes in self._failed_attempts[-2:]:
            lines.append(f"Attempt {attempt}:")
            lines.extend(f"- {note}" for note in notes)
        return "\n".join(lines)

    async def _next_actions(self) -> list[dict]:
        if self._toolbar_scope and self.screen.frame:
            y, before = self._toolbar_scope
            after = _toolbar_row(self.screen.frame, y)
            if sum(abs(a-b) for a, b in zip(before, after, strict=True)) / len(before) > 8:
                self._toolbar_failed, self._toolbar_scope = None, None
        prompt = (f"Goal: {self.task}\n\nPlan:\n{self.plan}\n\nRecent actions:\n{self._history()}"
                  f"\nCurrent pixel-derived clear regions (normalized): {self._workspace_observation()}\n"
                  "Use these only in an actually visible editable workspace when the goal needs new content; "
                  "they are observations, not permission to draw on Home/settings. "
                  "New work space is rechecked before input."
                  f"\n\nPersistent content journal for THIS run (normalized 0-1000 coordinates):\n"
                  f"{self._content_journal()}\nContinue the same intended new work item; do not switch to "
                  "pre-existing objects or repeatedly restart a completed creation stage. "
                  "Follow the original goal when it calls for multiple items."
                  f"\nVerified enclosing work boundary (normalized): {self._creation_boundary}"
                  "\n\nNext action?")
        if self._proposal_feedback:
            prompt += (f"\n\nYour proposed action was NOT executed: {self._proposal_feedback}\n"
                       "Resolve that missing prerequisite or correct its work-item target before acting.")
        if self._last_validation:
            prompt += (f"\n\nIndependent check of your last action: {self._last_validation}\n"
                       "The visible result did not match your intent; correct that before continuing.")
        if self._toolbar_failed is not None:
            prompt += (f"\n\nThe last toolbar tap near x={self._toolbar_failed} did not select the intended tool. "
                       "Do not tap that same x again; try a different toolbar icon, nearest left or right, "
                       "one at a time, and inspect what it opens.")
        for attempt in range(3):  # retry unparseable replies, a little less deterministically
            reply, seconds, frame = await self._ask("actor", self.actor, ACTOR_PROMPT, prompt, 0.4 * attempt)
            try:
                acts = parse_actions(reply)
            except ValueError as e:
                log.warning("rejected actor reply (%s): %s", e, reply[:200])
                prompt += (f"\n\nYour last reply was rejected: {e}. Reply with exactly ONE DO line; "
                           "never combine tool selection with drawing, typing or done.")
                continue
            if self._missed() and _key(acts[0]) == _key(self.steps[-1]):  # same move that just did nothing
                log.info("actor repeated an action that changed nothing; asking for another")
                prompt += (f"\n\nYou just proposed {json.dumps(_key(acts[0]))} again, but it changed nothing last "
                           "time. Choose a different action (another position, a shorter swipe, or another way).")
                continue
            acts[0]["think_s"] = seconds
            return [{**acts[0], "_shot": frame}]
        raise ValueError(f"actor gave no valid action: {reply[:200]}")

    async def _do(self, a: dict) -> None:
        p = lambda k: a[k] / 1000  # noqa: E731
        match a["action"]:
            case "tap":
                await self.phone.tap(p("x"), p("y"),
                                     aim=a.get("precise", False) or self._missed() or a.get("top_toolbar", False))
            case "swipe": await self.phone.swipe(p("x1"), p("y1"), p("x2"), p("y2"))
            case "path" | "circle": await self._draw_verified(a)
            case "type": await self.phone.type(a["text"])
            case "key": await self.phone.key(a["combo"])
            case "home": await self.phone.go_home()
            case "back": await self.phone.go_back()
            case "wait": pass
            case other: raise ValueError(f"unknown action {other!r}")

    async def _draw_verified(self, act):
        """Observe and anchor the starting pose before pressing; stroke geometry comes only from the actor."""
        points = (circle(act["x"]/1000, act["y"]/1000, act["r"]/1000, self.screen.aspect)
                  if act["action"] == "circle" else [(x/1000, y/1000) for x, y in act["points"]])
        if any(not 0 <= x <= 1 or not 0 <= y <= 1 for x, y in points):
            raise StepCheckError("requested stroke leaves the visible screen; no press was sent")
        x, y = points[0]
        moved_at = time.time()
        await self.phone.point(x, y)
        await self.screen.fresh(moved_at, timeout=1)
        corrections = []
        for attempt in range(3):
            if not self.screen.live or not self.phone.calibrated:
                raise BlindError("screen or device baseline lost before stroke; no press was sent")
            await self.screen.settled(quiet=.15, least=.1, most=.6)
            sent = time.time()
            nudge = 12 if x < .9 else -12
            await self.phone.device.send(f"move {nudge} 0")
            after = await self.screen.fresh(sent, timeout=.8)
            if after:
                await self.screen.settled(quiet=.1, least=.05, most=.4)
                after = self.screen.frame
            returned = time.time()
            await self.phone.device.send(f"move {-nudge} 0")
            await self.screen.fresh(returned, timeout=.8)
            await self.screen.settled(quiet=.1, least=.1, most=.5)
            current = self.screen.frame
            shift, found = nudge/self.phone.span['w'], None
            if current and after:
                for color in (False, True):
                    found = (vision.pointer(current, after, (x, y), shift, color=color)
                             or vision.pointer(current, after, (.5, .5), shift, reach=(.5, .5), color=color))
                    if found:
                        break
            if found and max(abs(found[0]-x), abs(found[1]-y)) <= .008:
                # Anchor logical coordinates to observed reality, not an accumulated, drifting HID sum.
                self.phone._pos = (self.phone._dx(x), self.phone._dy(y))
                act['start_pose'] = dict(target=(x, y), observed=found, attempts=attempt+1,
                                         corrections=corrections, confirmed=True)
                self.trace.write(self.screen.frame, kind="stroke_start_pose", evidence_id=act.get('evidence_id'),
                                 **act['start_pose'])
                await self.phone.drag(points)
                return
            if found:
                dx = round((x-found[0])*self.phone.span['w'])
                dy = round((y-found[1])*self.phone.span['h'])
                corrections.append(dict(observed=found, move=(dx, dy)))
                corrected_at = time.time()
                if dx: await self.phone.device.send(f"move {dx} 0")
                if dy: await self.phone.device.send(f"move 0 {dy}")
                await self.screen.fresh(corrected_at, timeout=.8)
        self.phone._pos = None
        self.trace.write(kind="stroke_start_pose", evidence_id=act.get('evidence_id'), target=(x, y),
                         confirmed=False, attempts=3, corrections=corrections)
        raise StepCheckError("stroke start not confirmed after three observations; no press was sent")


def _pointer_path(act: dict) -> list[tuple[float, float]]:
    """Screen points (fractions) the pointer passes through for one action."""
    f = lambda x, y: (x / 1000, y / 1000)  # noqa: E731
    match act["action"]:
        case "tap": return [f(act["x"], act["y"])]
        case "swipe": return [f(act["x1"], act["y1"]), f(act["x2"], act["y2"])]
        case "path": return [f(*p) for p in act["points"]]
        case "circle": return [f(act["x"] + act["r"], act["y"])]
        case "home": return [(0.5, 0.995), (0.5, 0.6)]
        case "back": return [(0.005, 0.45), (0.6, 0.45)]
        case _: return []


def circle(x: float, y: float, r: float, aspect: float, n: int = 24) -> list[tuple[float, float]]:
    """Points around a circle; r is a fraction of the width, aspect = width / height."""
    return [(x + r * math.cos(2 * math.pi * i / n), y + r * aspect * math.sin(2 * math.pi * i / n))
            for i in range(n + 1)]
