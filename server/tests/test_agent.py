import math

import pytest

from hidagent.agent import circle, parse_actions, parse_command, parse_plan


def test_parse_single_action():
    acts = parse_actions("THINK: The Search pill is visible, so I tap it.\nDO: tap 500 814")
    assert acts == [{"thought": "The Search pill is visible, so I tap it.", "thought_zh": "",
                     "action": "tap", "x": 500, "y": 814}]


def test_parse_several_actions_keeps_thought_on_first():
    acts = parse_actions("THINK: Type and search.\nDO: type Whiteboard\nDO: key enter")
    assert [a["action"] for a in acts] == ["type", "key"]
    assert acts[0]["text"] == "Whiteboard" and acts[1]["combo"] == "enter"
    assert "thought" not in acts[1]


def test_parse_caps_at_six_actions():
    reply = "THINK: draw\n" + "\n".join(f"DO: tap {i} {i}" for i in range(10))
    assert len(parse_actions(reply)) == 6


def test_parse_path_and_circle():
    assert parse_command("path 100 200 300 400 500 600")["points"] == [[100, 200], [300, 400], [500, 600]]
    assert parse_command("circle 500 450 200") == {"action": "circle", "x": 500, "y": 450, "r": 200}


@pytest.mark.parametrize("line", ["fly 1 2", "tap 500", "path 1 2 3"])
def test_parse_rejects_bad_commands(line):
    with pytest.raises(ValueError):
        parse_command(line)


def test_parse_requires_a_do_line():
    with pytest.raises(ValueError):
        parse_actions("THINK: I am not sure what to do.")


def test_parse_plan_with_and_without_languages():
    assert parse_plan("EN:\n1. Tap Search\nZH:\n1. 点搜索") == ("1. Tap Search", "1. 点搜索")
    assert parse_plan("1. Tap Search") == ("1. Tap Search", "1. Tap Search")


def test_circle_is_closed_and_round_on_a_tall_screen():
    aspect = 0.46  # width / height of a portrait iPhone frame
    pts = circle(0.5, 0.5, 0.2, aspect)
    assert pts[0] == pytest.approx(pts[-1])
    for x, y in pts:  # equal radius in pixels: y offsets shrink by the aspect ratio
        assert math.hypot(x - 0.5, (y - 0.5) / aspect) == pytest.approx(0.2)


def test_actor_may_not_repeat_a_move_that_changed_nothing(monkeypatch):
    import asyncio

    from hidagent.agent import Agent
    from hidagent.debug import Trace

    class Screen:
        live, frame, aspect = True, b"jpeg", 0.46

    class Actor:
        model = "fake"
        def __init__(self):
            self.prompts = []
        async def chat(self, system, text, image=None, max_tokens=600, temperature=0):
            self.prompts.append(text)
            return "THINK: again\nDO: swipe 382 253 382 150" if len(self.prompts) == 1 else "THINK: other\nDO: tap 1 2"

    actor = Actor()
    async def ground(self, action, frame): return action
    monkeypatch.setattr(Agent, "_ground_tap", ground)
    agent = Agent(phone=None, screen=Screen(), actor=actor, planner=actor, trace=Trace(None))
    agent.steps = [{"action": "swipe", "x1": 382, "y1": 253, "x2": 382, "y2": 150, "changed": False}]
    acts = asyncio.run(agent._next_actions())
    assert acts[0]["action"] == "tap"
    assert "changed nothing" in actor.prompts[1]


def test_repeating_a_move_that_keeps_changing_the_screen_is_not_stuck():
    from hidagent.agent import Agent
    from hidagent.debug import Trace

    agent = Agent(phone=None, screen=None, actor=None, planner=None, trace=Trace(None))
    swipe = {"action": "swipe", "x1": 500, "y1": 800, "x2": 500, "y2": 400}
    agent.steps = [{**swipe, "changed": True}, {**swipe, "changed": True}]
    assert not agent._repeating(dict(swipe))
    agent.steps[-1]["changed"] = False  # reached the end of the list
    assert agent._repeating(dict(swipe))


def test_precise_tap_flag():
    assert parse_command("tap 120 640 precise") == {"action": "tap", "x": 120, "y": 640, "precise": True}
    assert "precise" not in parse_command("tap 120 640")


def test_chinese_candidates_must_be_seen_before_batched_enter_or_done(tmp_path):
    import asyncio
    from types import SimpleNamespace

    from hidagent.agent import Agent
    from hidagent.debug import Trace
    from hidagent.phone import Phone

    commands = []

    async def send(command): commands.append(command)
    async def settled(**kwargs): pass
    async def actions():
        return [{"action": "type", "text": "美团", "_shot": None},
                {"action": "key", "combo": "enter", "_shot": None},
                {"action": "done", "answer": "finished", "_shot": None}]

    phone = Phone(SimpleNamespace(send=send), tmp_path / "calibration.json")
    screen = SimpleNamespace(frame=None, settled=settled)
    agent = Agent(phone, screen, None, None, Trace(None))
    agent._next_actions = actions

    assert asyncio.run(agent._step()) is False
    assert commands == ["type mei'tuan"]
    assert len(agent.steps) == 1
    assert agent.steps[0]["input"]["needs_confirmation"]


@pytest.mark.parametrize("combo", ["globe", "keyboard", "ctrl+space"])
def test_keyboard_switch_must_be_seen_before_typing(tmp_path, combo):
    import asyncio
    from types import SimpleNamespace

    from hidagent.agent import Agent
    from hidagent.debug import Trace
    from hidagent.phone import Phone

    commands = []
    async def send(command): commands.append(command)
    async def settled(**kwargs): pass
    async def actions():
        return [{"action": "key", "combo": combo, "_shot": None},
                {"action": "type", "text": "美团", "_shot": None}]

    phone = Phone(SimpleNamespace(send=send), tmp_path / "calibration.json")
    agent = Agent(phone, SimpleNamespace(frame=None, settled=settled), None, None, Trace(None))
    agent._next_actions = actions
    assert asyncio.run(agent._step()) is False
    assert commands == [f"key {combo}"]


class LoopScreen:
    live, frame, aspect = True, b"before", 0.46

    def __init__(self):
        self.settles = 0

    async def settled(self, **kwargs):
        self.settles += 1


class LoopLLM:
    model = "offline-test"

    def __init__(self, *replies):
        self.replies, self.prompts, self.images = list(replies), [], []

    async def chat(self, system, text, image=None, **kwargs):
        self.prompts.append(text)
        self.images.append(image)
        return self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]


def loop_agent(monkeypatch, actor, planner=None, trace=None, **kwargs):
    from types import SimpleNamespace

    from hidagent.agent import Agent
    from hidagent.debug import Trace

    screen, commands = LoopScreen(), []

    async def tap(x, y, aim=False):
        commands.append((x, y, aim))

    async def ground(self, action, frame): return action
    monkeypatch.setattr(Agent, "_ground_tap", ground)

    monkeypatch.setattr("hidagent.agent.vision.changed", lambda before, after, **kw: before != after)
    monkeypatch.setattr("hidagent.agent.vision.evidence_sheet", lambda panels: b"evidence sheet")
    agent = Agent(SimpleNamespace(tap=tap), screen, actor, planner or LoopLLM("1. Check all conditions."),
                  trace or Trace(None), **kwargs)
    agent.task = "Enable A and B, save, then show both enabled"
    return agent, screen, commands


def test_old_batch_done_is_discarded_and_next_round_verifies_fresh_screen(monkeypatch):
    import asyncio

    actor = LoopLLM("THINK: open then finish\nDO: tap 100 200\nDO: done saved",
                    "THINK: all visible\nDO: done saved")
    planner = LoopLLM("VERDICT: complete\nREASON: A and B are visibly enabled and saved.")
    agent, screen, commands = loop_agent(monkeypatch, actor, planner)

    async def run():
        assert await agent._step() is False
        assert len(commands) == 1 and planner.prompts == []
        assert screen.settles == 1
        assert all(s["action"] != "done" for s in agent.steps)
        screen.frame = b"saved screen"
        assert await agent._step() is True

    asyncio.run(run())
    assert planner.images == [b"evidence sheet"]
    assert actor.images == [b"before", b"saved screen"]
    assert agent.steps[-1]["verified"] is True
    assert agent.task in planner.prompts[0]


def test_verifier_receives_pre_edit_evidence_across_recovery_events(monkeypatch):
    import asyncio

    panels_seen = []
    planner = LoopLLM("VERDICT: complete\nREASON: hello was replaced by World in the same note.")
    agent, screen, _ = loop_agent(monkeypatch, LoopLLM("DO: done replaced"), planner)
    monkeypatch.setattr("hidagent.agent.vision.evidence_sheet",
                        lambda panels: panels_seen.extend(panels) or b"labelled evidence")
    agent.steps = [{"action": "tap", "x": 100, "y": 200},
                   {"action": "key", "combo": "cmd+a"},
                   {"action": "type", "text": "World", "changed": True},
                   {"action": "hiccup", "reason": "pointer not visible"},
                   {"action": "replan (stuck)", "plan": "check"}]
    agent.shots = {0: b"home", 1: b"hello note", 2: b"selected hello"}
    screen.frame = b"World note"
    assert asyncio.run(agent._step()) is True
    assert [frame for _, frame in panels_seen] == [b"hello note", b"selected hello", b"World note"]
    assert "before step 3 (type)" in planner.prompts[0]
    assert planner.images == [b"labelled evidence"]
    assert agent.shots[5] == b"World note"  # UI result keeps the actual current screenshot


def test_verifier_without_history_uses_current_frame(monkeypatch):
    import asyncio

    planner = LoopLLM("VERDICT: incomplete\nREASON: World is not present.")
    agent, screen, _ = loop_agent(monkeypatch, LoopLLM("DO: done yes"), planner)
    assert asyncio.run(agent._step()) is False
    assert planner.images == [screen.frame]
    assert agent.steps[-1]["action"] == "verification (incomplete)"


def test_home_effect_waits_for_delayed_video_before_recording_a_miss(monkeypatch):
    import asyncio

    agent, screen, _ = loop_agent(monkeypatch, LoopLLM('DO: home'))
    screen.frame = b'Notes'

    async def home(): pass
    async def delayed_frame(**kwargs):
        if kwargs.get('least', 0) >= .8:
            screen.frame = b'Home screen'

    agent.phone.go_home = home
    screen.settled = delayed_frame
    assert asyncio.run(agent._step()) is False
    assert agent.steps[-1]['action'] == 'home'
    assert agent.steps[-1]['changed'] is True


def test_only_first_action_runs_even_for_two_visible_taps(monkeypatch):
    import asyncio

    agent, screen, commands = loop_agent(monkeypatch, LoopLLM("DO: tap 100 200\nDO: tap 300 400"))
    assert asyncio.run(agent._step()) is False
    assert len(commands) == 1 and screen.settles == 1


def test_incomplete_verification_feedback_survives_replan_and_has_finite_budget(monkeypatch):
    import asyncio

    actor = LoopLLM("DO: done saved")
    planner = LoopLLM("VERDICT: incomplete\nREASON: B is not visible; check B.")
    agent, screen, commands = loop_agent(monkeypatch, actor, planner, max_failures=2)

    async def run():
        assert await agent._step() is False
        await agent._replan(stuck=True)
        assert "B is not visible" in planner.prompts[-1]
        assert await agent._step() is True

    asyncio.run(run())
    assert "B is not visible" in actor.prompts[-1]
    assert commands == []
    assert agent.steps[-1]["action"] == "fail"
    assert "verification failed 2 times" in agent.steps[-1]["reason"]


@pytest.mark.parametrize("reply", ["VERDICT: complete", "looks complete", "VERDICT: complete\nREASON:"])
def test_malformed_verification_cannot_finish(monkeypatch, reply):
    import asyncio

    agent, _, _ = loop_agent(monkeypatch, LoopLLM("DO: done yes"), LoopLLM(reply))
    assert asyncio.run(agent._step()) is False
    assert agent.steps[-1]["action"] == "verification (incomplete)"


def test_miss_survives_plan_hiccup_and_wait(monkeypatch):
    from hidagent.agent import _same_action

    agent, _, _ = loop_agent(monkeypatch, LoopLLM("DO: wait"))
    miss = {"action": "tap", "x": 500, "y": 500, "changed": False}
    agent.steps = [miss, {"action": "plan", "plan": "another way"},
                   {"action": "hiccup", "reason": "parse error"}, {"action": "wait", "changed": True}]
    assert agent._missed()
    assert _same_action(miss, {"action": "tap", "x": 511, "y": 511, "precise": True})
    agent.steps.insert(1, dict(miss))
    assert agent._repeating({"action": "tap", "x": 511, "y": 511, "precise": True})


@pytest.mark.parametrize("jitter", [0, 11])
def test_same_or_nearby_failed_tap_stops_with_no_progress_before_round_limit(monkeypatch, jitter):
    import asyncio

    actor = LoopLLM("DO: tap 500 500", "DO: replan", f"DO: tap {500 + jitter} {500 + jitter} precise")
    agent, _, commands = loop_agent(monkeypatch, actor, max_steps=10)
    asyncio.run(agent._run())
    assert len(commands) == 1
    assert len(actor.prompts) < 10
    assert agent.steps[-1]["action"] == "fail"
    assert "no observable progress" in agent.steps[-1]["reason"]
    assert "round limit" not in agent.steps[-1]["reason"]
    assert "tap" in agent.planner.prompts[-1] and '"changed": false' in agent.planner.prompts[-1]


def test_distinct_checkboxes_are_not_failed_by_unchanged_images(monkeypatch):
    import asyncio

    actor = LoopLLM("DO: tap 100 100", "DO: tap 400 400", "DO: tap 800 800", "DO: done saved")
    planner = LoopLLM("VERDICT: complete\nREASON: A and B are enabled and saved.")
    agent, _, commands = loop_agent(monkeypatch, actor, planner, max_steps=10)
    asyncio.run(agent._run())
    assert len(commands) == 3
    assert all(s["changed"] is False for s in agent._physical_steps())
    assert agent.steps[-1]["action"] == "done"
    assert agent.steps[-1]["verified"] is True


def test_different_target_starts_its_own_repeat_recovery_budget(monkeypatch):
    import asyncio

    actor = LoopLLM("DO: tap 100 100", "DO: tap 111 111", "DO: tap 400 400",
                    "DO: tap 411 411", "DO: tap 800 800")
    agent, _, commands = loop_agent(monkeypatch, actor, max_failures=2)

    async def run():
        assert await agent._step() is False
        assert await agent._step() is False
        assert agent._recoveries == 0
        assert await agent._step() is False
        assert agent._recoveries == 0

    asyncio.run(run())
    assert len(commands) == 3


def test_animated_nearby_tap_loop_is_bounded_across_replan(monkeypatch):
    import asyncio

    actor = LoopLLM("DO: tap 500 500", "DO: tap 511 511 precise", "DO: tap 500 500",
                    "DO: replan", "DO: tap 511 511", "DO: tap 500 500", "DO: tap 511 511 precise")
    agent, screen, commands = loop_agent(monkeypatch, actor, max_steps=12)

    async def animate(**kwargs):
        screen.frame = screen.frame + b"animation"

    screen.settled = animate
    asyncio.run(agent._run())
    assert len(commands) == 5
    assert all(s["changed"] is True for s in agent._physical_steps())
    assert len(agent.planner.prompts) == 2
    assert agent.steps[-1]["action"] == "fail"
    assert "same nearby tap proposed 6 times" in agent.steps[-1]["reason"]


def test_changing_swipe_sequence_is_allowed_past_tap_loop_limit(monkeypatch):
    import asyncio

    actor = LoopLLM(*(["DO: swipe 500 800 500 400"] * 7), "DO: done found")
    planner = LoopLLM("VERDICT: complete\nREASON: all original conditions are visible.")
    agent, screen, commands = loop_agent(monkeypatch, actor, planner, max_steps=12)

    async def swipe(*args):
        commands.append(args)

    async def change(**kwargs):
        screen.frame = screen.frame + b"scrolled"

    agent.phone.swipe = swipe
    screen.settled = change
    asyncio.run(agent._run())
    assert len(commands) == 7
    assert agent.steps[-1]["action"] == "done"


def test_replan_reports_round_budget_and_executed_actions(monkeypatch):
    import asyncio

    agent, _, commands = loop_agent(monkeypatch, LoopLLM("DO: replan"), max_steps=2)
    asyncio.run(agent._run())
    assert commands == []
    assert "2 rounds, 0 executed actions" in agent.steps[-1]["reason"]


def test_changed_effect_is_persisted_with_after_frame(monkeypatch, tmp_path):
    import asyncio
    import json

    from hidagent.debug import Trace

    trace = Trace(tmp_path)
    trace.start("offline trace")
    agent, _, _ = loop_agent(monkeypatch, LoopLLM("DO: tap 100 100"), trace=trace)
    asyncio.run(agent._step())
    records = [json.loads(line) for line in (trace.dir / "steps.jsonl").read_text().splitlines()]
    step = next(r for r in records if r["kind"] == "step")
    effect = next(r for r in records if r["kind"] == "effect")
    assert agent.steps[-1]["changed"] is step["changed"] is effect["changed"] is False
    assert (trace.dir / effect["frame"]).read_bytes() == b"before"


@pytest.mark.parametrize("lost_during", ["actor", "settling", "verifier"])
def test_frame_loss_prevents_action_or_success(monkeypatch, lost_during):
    import asyncio

    from hidagent.agent import BlindError

    actor = LoopLLM("DO: tap 100 100" if lost_during != "verifier" else "DO: done yes")
    planner = LoopLLM("VERDICT: complete\nREASON: all visible.")
    agent, screen, commands = loop_agent(monkeypatch, actor, planner)
    target = planner if lost_during == "verifier" else actor
    original_chat = target.chat

    async def lose_after_reply(*args, **kwargs):
        reply = await original_chat(*args, **kwargs)
        screen.live = False
        return reply

    async def lose_when_settled(**kwargs):
        screen.live = False

    if lost_during == "settling":
        screen.settled = lose_when_settled
    else:
        target.chat = lose_after_reply
    with pytest.raises(BlindError):
        asyncio.run(agent._step())
    assert len(commands) == (1 if lost_during == "settling" else 0)
    assert not any(s["action"] == "done" for s in agent.steps)
    if lost_during == "settling":
        assert agent.steps[-1]["changed"] is None
