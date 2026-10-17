"""See-think-act loop: a planner model writes the plan, an actor model picks each step."""

import asyncio
import json
import logging
import math
import re
import time

from . import vision
from .debug import Trace
from .device import DeviceError
from .grounding import locate_tap
from .llm import LLM
from .phone import Phone
from .screen import Screen

log = logging.getLogger(__name__)

PLANNER_PROMPT = """You plan tasks on an iPhone or iPad. Given the goal and the current screen,
write a short numbered plan (at most 6 steps) of what to do on screen. Plain text only, in English.
Plan only what the goal needs, from what you can see; never invent accounts, emails, logins or
screens you have not seen. Later steps can stay general (e.g. "Draw the face on the canvas").
To open an app, prefer search over paging through home screens: tap the "Search" pill above
the dock (or swipe down from the middle of the home screen), type the app name, tap the result."""

ACTOR_PROMPT = """You operate an iPhone or iPad through a mouse pointer. You see the current screen.
Reply in this format, nothing else:
THINK: <plain English a non-technical user understands: first check your last action against the
       screen (did it do what you intended?), then what you do now and why>
DO: <action>

Give exactly ONE DO line. After every action you will see a new screenshot before acting again.
Never combine a tap that opens a menu or changes the toolbar with drawing, typing or done: you must
look at the new screen first.
An independent checker reviews each action against the screen. When it reports a mismatch, correct
the visible result before continuing. After three mismatches the task restarts from the home screen.

Coordinates are integers 0-1000 relative to the screenshot (x left->right, y top->bottom).
Aim at the visual center of the element you want, not its edge or its label.
Look before you act: describe what is really on the screen now, not what you expect.
Do not assume where a toolbar button is because it was there on another device or app; read the
current screen. After opening a menu, look at what it contains instead of repeating the same tap.
Before path or circle, the drawing tool must be visibly selected and no menu may be open. If a menu
opened or a shape got selected instead, close it and choose the actual tool again.
Each recent action says whether the screen changed after it. If it did NOT change, the action
missed or had no effect: do not assume it worked; aim again more precisely or try another way.
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
DO: tap 152 362"""

VERIFY_PROMPT = """Independently verify whether the original phone task is complete using only the current screenshot.
Do not trust the executor's claim, the plan, or earlier taps as evidence.
For text or drawings, check that the actual content is visible. For settings, check the visible state;
never infer a hidden save or a state that is not shown.
Reply exactly in this format:
VERDICT: complete
REASON: <one sentence citing visible evidence>
or
VERDICT: incomplete
REASON: <what is still missing or wrong>"""

VALIDATE_PROMPT = """You check one action in a phone task. The actor stated its intent and the action it performed.
Look at the current screenshot and judge whether the action achieved that intended visible effect.
Judge only what is visible; do not infer hidden effects and do not trust the actor's claim.
Reply exactly in this format:
VERDICT: ok
REASON: <visible evidence that the intended effect happened>
or
VERDICT: mismatch
REASON: <what is actually visible instead, and why it does not match>"""

META = ("t", "thought", "thought_zh", "think_s", "act_s", "shot", "_shot", "changed", "validated",
        "validation", "validate_s", "verified", "verification", "verify_s", "ground_s", "proposed")  # not the action

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
    return [{"thought": thought.group(1) if thought else "", "thought_zh": zh.group(1) if zh else "", **act}]


def parse_plan(text: str) -> tuple[str, str]:
    """Split an optional EN: / ZH: plan reply; a plain reply is used for both."""
    m = re.search(r"EN\s*[:：](.*?)ZH\s*[:：](.*)", text, re.S | re.I)
    return (m.group(1).strip(), m.group(2).strip()) if m else (text.strip(), text.strip())


def _key(step: dict) -> dict:
    """The action itself, without thoughts, timings and other bookkeeping."""
    return {k: v for k, v in step.items() if k not in META}


class BlindError(RuntimeError):
    """The screen feed is gone; acting now would be guessing."""


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
        self._attempts, self._mismatches = 1, 0
        self._history_start = 0  # steps before this belong to an abandoned attempt
        self._last_validation = ""
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
        self._failure_notes, self._failed_attempts = [], []
        self.trace.start(task)
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
                   temperature: float = 0) -> tuple[str, float, bytes | None]:
        if not self.screen.live:  # never let the model act blind
            raise BlindError("lost the phone screen: reconnect Screen Mirroring")
        frame, started = self.screen.frame, time.time()
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
        if act["action"] == "tap":
            act = await self._ground_tap(act, shot)
        started = time.time()
        await self._do(act)
        act["act_s"] = round(time.time() - started, 2)
        self._record(shot, **act)
        await self.screen.settled()
        spots = [self._pointer, *_pointer_path(act)]
        self._pointer = spots[-1]
        changed = vision.changed(shot, self.screen.frame, ignore=tuple(spots))
        self.steps[-1]["changed"] = changed
        if act["action"] == "wait":
            return False
        if not changed:
            ok, reason = False, "the screen did not change after the action"
        else:
            ok, reason, seconds, _ = await self._validate_action(act)
            self.steps[-1]["validated"], self.steps[-1]["validation"] = ok, reason
            self.steps[-1]["validate_s"] = seconds
        if ok:
            self._last_validation = ""
            return False
        self._mismatches += 1
        self._last_validation = reason
        self._failure_notes.append(f"{act.get('action')} {json.dumps(_key(act))} -> {reason}"[:300])
        self._record(action="hiccup", reason=f"step check: {reason}"[:300])
        if self._mismatches >= 3:
            return await self._restart()
        return False

    async def _validate_action(self, act: dict) -> tuple[bool, str, float, bytes | None]:
        """Independently check that the action produced the visible effect the actor intended."""
        prompt = (f"Actor's intent: {act.get('thought', '')}\n"
                  f"Action just performed: {json.dumps(_key(act))}\n\n"
                  "Look at the current screen and judge the intended visible effect.")
        reply, seconds, frame = await self._ask("step verifier", self.planner or self.actor, VALIDATE_PROMPT, prompt)
        match = re.search(r"VERDICT\s*[:：]\s*(ok|mismatch)", reply, re.I)
        reason = re.search(r"REASON\s*[:：]\s*(.+)", reply, re.I | re.S)
        detail = reason.group(1).strip() if reason else reply.strip()
        return bool(match and match.group(1).lower() == "ok"), detail, seconds, frame

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
        self._pointer, self.plan = (0.0, 0.0), ""
        try:
            await self.phone.go_home()
        except Exception as e:
            self._record(action="hiccup", reason=f"restart could not reach the home screen: {e}"[:300])
        await self._replan(stuck=True)
        self._history_start = len(self.steps)
        return False

    async def _ground_tap(self, action: dict, frame: bytes) -> dict:
        """Refine a tap against the screenshot the actor decided on; never click a stale screen."""
        refined = await locate_tap(self.actor, frame, action, self.trace)
        if not self.screen.live:
            raise BlindError("lost the phone screen: reconnect Screen Mirroring")
        if self.screen.frame != frame and vision.scene_changed(frame, self.screen.frame):
            raise RuntimeError("screen changed while locating the target; discard the tap and observe again")
        return refined

    async def _verify(self, answer: str) -> tuple[bool, str, float, bytes | None]:
        """An independent model check before accepting done; a claim alone is not evidence."""
        prompt = (f"Original goal: {self.task}\n\nExecutor's claimed result: {answer}\n\n"
                  f"Executed actions and observed effects:\n{self._history()}\n\n"
                  "Look at the current screen and verify the goal.")
        reply, seconds, frame = await self._ask("verifier", self.planner or self.actor, VERIFY_PROMPT, prompt)
        match = re.search(r"VERDICT\s*[:：]\s*(complete|incomplete|failed)", reply, re.I)
        reason = re.search(r"REASON\s*[:：]\s*(.+)", reply, re.I | re.S)
        detail = reason.group(1).strip() if reason else reply.strip()
        return bool(match and match.group(1).lower() == "complete"), detail, seconds, frame

    def _history(self, limit: int = 8) -> str:
        hidden = ("t", "plan", "plan_zh", "think_s", "act_s", "shot", "_shot", "thought", "thought_zh",
                  "changed", "validated", "validation", "validate_s", "verified", "verification",
                  "verify_s", "ground_s", "proposed")
        effect = {True: " -> screen changed", False: " -> screen did NOT change"}
        return "\n".join(json.dumps({k: v for k, v in s.items() if k not in hidden})[:300]
                         + effect.get(s.get("changed"), "")
                         for s in self.steps[self._history_start:][-limit:] if "plan" not in s) or "(none)"

    def _missed(self) -> bool:
        """The previous round changed nothing on screen, so a tap may have landed off target."""
        return bool(self.steps) and self.steps[-1].get("changed") is False

    def _repeating(self, act: dict, limit: int = 3) -> bool:
        """True when the actor picks the same action as its last limit-1 moves and the last one did nothing.

        Repeating a move that still changes the screen (e.g. scrolling down a long list) is progress, not a loop.
        """
        moves = [s for s in self.steps[self._history_start:] if "plan" not in s][-(limit - 1):]
        return (len(moves) == limit - 1 and all(_key(m) == _key(act) for m in moves)
                and moves[-1].get("changed") is not True)

    async def _replan(self, stuck: bool = False) -> None:
        prompt = f"Goal: {self.task}"
        if stuck:
            last = self.steps[-1] if self.steps else {}
            prompt += (f"\n\nThe previous plan got stuck (last event: {last.get('action')} {last.get('reason', '')}):"
                       f"\n{self.plan}\nLook at the current screen and plan from here, another way if needed.")
            summary = self._failure_summary()
            if summary:
                prompt += "\n\n" + summary
        reply, seconds, frame = await self._ask("planner", self.planner, PLANNER_PROMPT, prompt)
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
        prompt = (f"Goal: {self.task}\n\nPlan:\n{self.plan}\n\nRecent actions:\n{self._history()}"
                  "\n\nNext action?")
        if self._last_validation:
            prompt += (f"\n\nIndependent check of your last action: {self._last_validation}\n"
                       "The visible result did not match your intent; correct that before continuing.")
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
            case "tap": await self.phone.tap(p("x"), p("y"), aim=a.get("precise", False) or self._missed())
            case "swipe": await self.phone.swipe(p("x1"), p("y1"), p("x2"), p("y2"))
            case "path": await self.phone.drag([(x / 1000, y / 1000) for x, y in a["points"]])
            case "circle": await self.phone.drag(circle(p("x"), p("y"), p("r"), self.screen.aspect))
            case "type": await self.phone.type(a["text"])
            case "key": await self.phone.key(a["combo"])
            case "home": await self.phone.go_home()
            case "back": await self.phone.go_back()
            case "wait": pass
            case other: raise ValueError(f"unknown action {other!r}")


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
