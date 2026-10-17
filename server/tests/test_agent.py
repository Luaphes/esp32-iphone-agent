import math

import pytest

from hidagent.agent import circle, parse_actions, parse_command, parse_plan


def test_parse_single_action():
    acts = parse_actions("THINK: The Search pill is visible, so I tap it.\nDO: tap 500 814")
    assert acts == [{"thought": "The Search pill is visible, so I tap it.", "thought_zh": "",
                     "action": "tap", "x": 500, "y": 814}]


def test_parse_rejects_multiple_actions():
    with pytest.raises(ValueError, match="exactly one DO action"):
        parse_actions("THINK: Type and search.\nDO: type Whiteboard\nDO: key enter")


def test_parse_never_caps_a_batch_into_several_actions():
    reply = "THINK: draw\n" + "\n".join(f"DO: tap {i} {i}" for i in range(10))
    with pytest.raises(ValueError, match="exactly one DO action"):
        parse_actions(reply)


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


def test_actor_may_not_repeat_a_move_that_changed_nothing():
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


def test_actor_batch_is_rejected_and_retried_with_one_action():
    import asyncio

    from hidagent.agent import Agent
    from hidagent.debug import Trace

    class Screen:
        live, frame, aspect = True, b"jpeg", 0.7

    class Actor:
        model = "fake"

        def __init__(self):
            self.prompts = []

        async def chat(self, system, text, image=None, max_tokens=600, temperature=0):
            self.prompts.append(text)
            if len(self.prompts) == 1:
                return "THINK: draw\nDO: tap 1 2\nDO: circle 500 500 200\nDO: done ok"
            return "THINK: look first\nDO: tap 5 6"

    actor = Actor()
    agent = Agent(phone=None, screen=Screen(), actor=actor, planner=actor, trace=Trace(None))
    agent.task, agent.plan = "draw", "1. draw"
    acts = asyncio.run(agent._next_actions())
    assert acts[0]["action"] == "tap" and acts[0]["x"] == 5
    assert "exactly one DO action" in actor.prompts[1]


def test_done_needs_an_explicit_complete_verdict_from_the_verifier():
    import asyncio

    from hidagent.agent import Agent
    from hidagent.debug import Trace

    class Screen:
        live, frame, aspect = True, b"jpeg", 0.7

    class Model:
        model = "fake"

        def __init__(self, reply):
            self.reply = reply

        async def chat(self, system, text, image=None, max_tokens=600, temperature=0):
            return self.reply

    def verify(reply):
        agent = Agent(phone=None, screen=Screen(), actor=Model(reply), planner=Model(reply), trace=Trace(None))
        agent.task = "draw a smiley"
        return asyncio.run(agent._verify("done"))

    ok, detail, _, _ = verify("VERDICT: complete\nREASON: a smiley is visible on the canvas")
    assert ok and "smiley" in detail
    ok, detail, _, _ = verify("VERDICT: incomplete\nREASON: the canvas is blank")
    assert not ok and "blank" in detail
    ok, _, _, _ = verify("looks good to me")
    assert not ok


def test_each_action_is_checked_against_its_stated_intent():
    import asyncio

    from hidagent.agent import Agent
    from hidagent.debug import Trace

    class Screen:
        live, frame, aspect = True, b"jpeg", 0.7

    class Model:
        model = "fake"

        def __init__(self, reply):
            self.reply = reply

        async def chat(self, system, text, image=None, max_tokens=600, temperature=0):
            return self.reply

    def check(reply):
        agent = Agent(phone=None, screen=Screen(), actor=Model(reply), planner=Model(reply), trace=Trace(None))
        return asyncio.run(agent._validate_action({"action": "tap", "x": 1, "y": 2, "thought": "select the pen"}))

    ok, detail, _, _ = check("VERDICT: ok\nREASON: the pen palette is now visible")
    assert ok and "pen palette" in detail
    ok, detail, _, _ = check("VERDICT: mismatch\nREASON: the Insert menu opened instead")
    assert not ok and "Insert menu" in detail
    ok, _, _, _ = check("unclear")
    assert not ok


def test_three_mismatches_restart_from_the_home_screen():
    import asyncio

    from hidagent.agent import Agent
    from hidagent.debug import Trace

    class Screen:
        live, frame, aspect = True, b"jpeg", 0.7

    class Model:
        model = "fake"

        async def chat(self, system, text, image=None, max_tokens=600, temperature=0):
            return "VERDICT: ok\nREASON: fine"

    calls = []

    class Phone:
        async def go_home(self):
            calls.append("home")

    agent = Agent(phone=Phone(), screen=Screen(), actor=Model(), planner=Model(), trace=Trace(None))
    agent._attempts, agent._mismatches, agent.plan = 1, 3, "old"

    async def replan(stuck=False):
        calls.append(("replan", stuck))
        agent.plan = "new"

    agent._replan = replan
    assert not asyncio.run(agent._restart())
    assert calls == ["home", ("replan", True)]
    assert agent._attempts == 2 and agent._mismatches == 0
    assert agent._history_start == len(agent.steps)
    assert agent.steps[-1]["action"] == "restart"


def test_restart_is_bounded_so_a_loop_cannot_run_forever():
    import asyncio

    from hidagent.agent import Agent
    from hidagent.debug import Trace

    calls = []

    class Screen:
        live, frame, aspect = True, b"jpeg", 0.7

    class Phone:
        async def go_home(self):
            calls.append("home")

    agent = Agent(phone=Phone(), screen=Screen(), actor=None, planner=None, trace=Trace(None))
    agent._attempts, agent._mismatches = 3, 3
    assert asyncio.run(agent._restart())
    assert calls == []
    assert agent.steps[-1]["action"] == "fail"
