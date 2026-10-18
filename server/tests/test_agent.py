import math

import pytest

from hidagent.agent import circle, parse_actions, parse_command, parse_plan


def test_parse_single_action():
    acts = parse_actions("THINK: The Search pill is visible, so I tap it.\n"
                         "EXPECT: The intended destination is visible.\nDO: tap 500 814")
    assert acts == [{"thought": "The Search pill is visible, so I tap it.", "thought_zh": "",
                     "expected": "The intended destination is visible.", "action": "tap", "x": 500, "y": 814}]


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
            action = "swipe 382 253 382 150" if len(self.prompts) == 1 else "tap 1 2"
            return f"THINK: try\nEXPECT: The intended destination is visible.\nDO: {action}"

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
            return "THINK: look first\nEXPECT: The intended destination is visible.\nDO: tap 5 6"

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
        from io import BytesIO

        from PIL import Image
        out = BytesIO()
        Image.new("RGB", (20, 30)).save(out, "JPEG")
        return asyncio.run(agent._validate_action(
            {"action": "tap", "x": 1, "y": 2, "expected": "the pen palette is visible"},
            out.getvalue(), out.getvalue()))

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
    agent._failure_notes = ["tap at 620,46 -> the Insert menu opened"]

    async def replan(stuck=False, **kwargs):
        calls.append(("replan", stuck))
        agent.plan = "new"

    async def confirm(before, boundary):
        calls.append("confirm")
        return b"home"

    agent._confirm_home = confirm
    agent._replan = replan
    assert not asyncio.run(agent._restart())
    assert calls == ["home", "confirm", ("replan", True)]
    assert agent._attempts == 2 and agent._mismatches == 0
    assert agent._history_start == len(agent.steps)
    assert agent.steps[-1]["action"] == "restart"
    assert agent._failed_attempts == [(1, ["tap at 620,46 -> the Insert menu opened"])]


def test_failure_summary_tells_the_planner_what_already_failed():
    from hidagent.agent import Agent
    from hidagent.debug import Trace

    agent = Agent(phone=None, screen=None, actor=None, planner=None, trace=Trace(None))
    agent._failed_attempts = [(1, ["tap at 620,46 -> the Insert menu opened"]),
                              (2, ["tap at 692,46 -> the Discover panel opened"])]
    summary = agent._failure_summary()
    assert "Attempt 1" in summary and "Insert menu" in summary
    assert "Attempt 2" in summary and "Discover panel" in summary
    assert "Do not repeat" in summary


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


def test_grounding_failure_is_a_step_check_error():
    import asyncio
    from types import SimpleNamespace

    from hidagent.agent import Agent, StepCheckError
    from hidagent.debug import Trace

    async def locate(*args):
        raise RuntimeError("intended tap target was not located")

    import hidagent.agent as module
    original = module.locate_tap
    module.locate_tap = locate
    try:
        agent = Agent(phone=None, screen=SimpleNamespace(live=True, frame=b"old"),
                      actor=None, planner=None, trace=Trace(None))
        with pytest.raises(StepCheckError, match="not located"):
            asyncio.run(agent._ground_tap({"action": "tap", "x": 620, "y": 50}, b"old"))
    finally:
        module.locate_tap = original


def test_failed_toolbar_check_is_counted_and_hinted_to_the_actor():
    import asyncio

    from hidagent.agent import Agent
    from hidagent.debug import Trace

    class Actor:
        model = "fake"

        def __init__(self):
            self.prompts = []

        async def chat(self, system, text, image=None, max_tokens=600, temperature=0):
            self.prompts.append(text)
            return "THINK: try another icon\nEXPECT: The intended destination is visible.\nDO: tap 540 50"

    class Screen:
        live, frame, aspect = True, b"jpeg", 0.7

    restarted = []

    async def restart():
        restarted.append(True)
        return False

    actor = Actor()
    agent = Agent(phone=None, screen=Screen(), actor=actor, planner=actor, trace=Trace(None))
    agent.task, agent.plan = "draw", "1. draw"
    agent._restart = restart
    agent._mismatches, agent._toolbar_failed = 0, None

    # The failed toolbar check sets the hint and, on the third failure, triggers a restart.
    failed = {"action": "tap", "x": 620, "y": 50, "top_toolbar": True}
    assert not asyncio.run(agent._step_check_failed(failed, "the Insert menu opened"))
    assert agent._toolbar_failed == 620
    assert "the Insert menu opened" in agent._last_validation
    agent._mismatches = 2
    assert not asyncio.run(agent._step_check_failed(failed, "still wrong"))
    assert restarted == [True]

    acts = asyncio.run(agent._next_actions())
    assert acts[0]["x"] == 540
    assert 'explore' not in acts[0]  # bypass needs the pre-action current-UI classification first
    assert "Do not tap that same x again" in actor.prompts[-1]


def jpeg(color):
    from io import BytesIO

    from PIL import Image

    out = BytesIO()
    Image.new('RGB', (20, 30), color).save(out, 'JPEG')
    return out.getvalue()


def test_expect_is_required_and_is_not_part_of_the_device_command():
    from hidagent.agent import _key

    for reply in ('THINK: old context\nDO: tap 1 2',
                  'EXPECT: \nDO: tap 1 2', 'EXPECT: a\nEXPECT: b\nDO: tap 1 2'):
        with pytest.raises(ValueError, match='EXPECT'):
            parse_actions(reply)
    act = parse_actions('THINK: old menu\nEXPECT: menu is absent\nDO: tap 1 2')[0]
    assert act['expected'] == 'menu is absent'
    assert _key(act) == {'action': 'tap', 'x': 1, 'y': 2}


@pytest.mark.parametrize('reply,accepted', [
    ('VERDICT: ok\nREASON: AFTER shows the canvas without the popover.', True),
    ('VERDICT: mismatch\nREASON: AFTER still shows the popover.', False),
    ('VERDICT: ok', False), ('VERDICT: ok\nREASON: ', False),
    ('VERDICT: ok\nREASON: yes\nVERDICT: mismatch', False), ('unclear', False),
])
def test_transition_check_separates_old_context_and_rejects_bad_protocol(reply, accepted):
    import asyncio
    from types import SimpleNamespace

    from hidagent.agent import Agent
    from hidagent.debug import Trace

    class Model:
        model = 'fake'

        async def chat(self, system, text, image=None, **kwargs):
            assert 'old misleading context' not in text
            assert 'menu is absent' in text
            assert image != after  # labeled before/after evidence, not after alone
            return reply

    before, after = jpeg('red'), jpeg('white')
    model = Model()
    agent = Agent(None, SimpleNamespace(live=True, frame=after), model, model, Trace(None))
    result = asyncio.run(agent._validate_action(
        {'action': 'tap', 'expected': 'menu is absent', 'thought': 'old misleading context'}, before, after))
    assert result[0] is accepted


@pytest.mark.parametrize('mode', ['delayed', 'during_command', 'none', 'wrong', 'disconnect', 'command_error',
                                 'transition'])
def test_recovery_never_plans_before_fresh_confirmed_home(mode, tmp_path):
    import asyncio
    import time

    from hidagent.agent import Agent
    from hidagent.debug import Trace

    old, home = jpeg('red'), jpeg('blue')
    calls = []

    class Screen:
        live, frame, updated = True, old, 0

        async def fresh(self, since, timeout):
            calls.append('fresh')
            assert since > 0
            if mode == 'none' or (mode == 'wrong' and self.updated > 0):
                return None
            if mode != 'during_command':
                await asyncio.sleep(.01)
                self.frame = old if mode == 'wrong' or (mode == 'transition' and self.updated == 0) else home
                self.updated = time.time()
            assert self.updated > since
            return self.frame

        async def settled(self, **kwargs):
            calls.append('settled')
            if mode == 'disconnect':
                self.live = False

    screen = Screen()

    class Phone:
        async def go_home(self):
            calls.append('home')
            if mode == 'command_error':
                raise RuntimeError('HID failed')
            if mode == 'during_command':
                screen.frame, screen.updated = home, time.time()

    trace = Trace(tmp_path)
    trace.start('recovery fixture')
    agent = Agent(Phone(), screen, None, None, trace)

    async def validate(act, before, after):
        calls.append('confirm')
        assert 'home screen' in act['expected'] and before == old
        return after == home, 'observed', 0, after

    async def replan(stuck=False, *, frame=None):
        calls.append('plan')
        assert stuck and frame == home

    agent._validate_action, agent._replan = validate, replan
    stopped = asyncio.run(agent._restart())
    successful = mode in ('delayed', 'during_command', 'transition')
    assert stopped is not successful
    assert ('plan' in calls) is successful
    if successful:
        assert calls.index('fresh') < calls.index('confirm') < calls.index('plan')
    else:
        assert agent.steps[-1]['action'] == 'fail'
    assert agent._attempts == 2
    if mode == 'transition':
        assert calls.count('confirm') == 2


def test_step_trace_persists_final_validation_and_before_after_evidence(tmp_path, monkeypatch):
    import asyncio
    import json

    from hidagent.agent import Agent
    from hidagent.debug import Trace

    before, after = jpeg('red'), jpeg('white')

    class Screen:
        live, frame = True, before

        async def fresh(self, since, timeout):
            return self.frame

        async def settled(self):
            pass

    trace = Trace(tmp_path)
    trace.start('transition')
    screen = Screen()
    agent = Agent(None, screen, None, None, trace)

    async def next_actions():
        return [{'action': 'back', 'expected': 'menu is absent', '_shot': before}]

    async def do(act):
        screen.frame = after

    async def validate(act, before, after):
        return True, 'menu absent in AFTER', .1, after

    async def prepare(act, before):
        return act

    agent._prepare_action = prepare
    agent._next_actions, agent._do, agent._validate_action = next_actions, do, validate
    assert not asyncio.run(agent._step())
    records = [json.loads(line) for line in (trace.dir / 'steps.jsonl').read_text().splitlines()]
    pre, post, step = records[-3:]
    assert pre['kind'] == 'action_before' and post['kind'] == 'action_after'
    assert pre['evidence_id'] == post['evidence_id'] == step['evidence_id']
    assert step['validated'] is True and step['expected'] == 'menu is absent'
    assert (trace.dir / pre['frame']).read_bytes() == before
    assert (trace.dir / post['frame']).read_bytes() == after


def test_screen_fresh_handles_frames_during_action_delay_and_timeout():
    import asyncio
    import time

    from hidagent.screen import Screen

    async def exercise():
        screen = Screen()
        boundary = time.time() - 1  # distinct timestamps: two immediate clock reads can be equal
        screen.frame, screen.updated = b'arrived during command', time.time()
        assert await screen.fresh(boundary, timeout=.05) == b'arrived during command'
        boundary = time.time()

        async def deliver():
            await asyncio.sleep(.02)
            screen.frame, screen.updated = b'delayed', time.time()

        delivery = asyncio.create_task(deliver())
        assert await screen.fresh(boundary, timeout=.2) == b'delayed'
        await delivery
        assert await screen.fresh(time.time(), timeout=.03) is None

    asyncio.run(exercise())


@pytest.mark.parametrize('bad_reply', [
    'not json', '{"status":"ready","required":[],"discarded":[],"reason":"empty"}',
    '{"status":"unclear","required":[],"discarded":[],"reason":"target ambiguous"}',
    '{"status":"ready","required":[{"condition":"ok","basis":"after"}],"discarded":[],"reason":"x"}',
])
def test_unclear_conditions_are_bounded_and_do_not_execute_or_spend_failure_budget(bad_reply, tmp_path):
    import asyncio
    from types import SimpleNamespace

    from hidagent.agent import Agent
    from hidagent.debug import Trace

    class Model:
        model, calls = 'fake', 0

        async def chat(self, system, text, image=None, **kwargs):
            self.calls += 1
            assert image == before
            return bad_reply

    before = jpeg('white')
    model = Model()
    trace = Trace(tmp_path)
    trace.start('unclear preconditions')
    agent = Agent(None, SimpleNamespace(live=True, frame=before), model, model, trace)

    async def next_actions():
        return [{'action': 'home', 'expected': 'Home', '_shot': before}]

    async def forbidden(act):
        pytest.fail('an unconfirmed action must not execute')

    agent._next_actions, agent._do = next_actions, forbidden
    async def exercise():
        assert await agent._step() is False
        assert await agent._step() is False
        assert await agent._step() is True

    asyncio.run(exercise())
    assert model.calls == 6 and agent._mismatches == 0 and agent._attempts == 1
    assert agent.steps[-1]['action'] == 'fail' and 'no action executed' in agent.steps[-1]['reason']


def test_conditions_are_frozen_before_action_with_raw_expect_and_evidence(tmp_path):
    import asyncio
    import json
    from types import SimpleNamespace

    from hidagent.agent import Agent, _key
    from hidagent.debug import Trace

    before, after = jpeg('red'), jpeg('blue')

    class Model:
        model = 'fake'

        async def chat(self, system, text, image=None, **kwargs):
            assert image == before
            assert 'Original user goal: open search' in text
            return json.dumps({'status': 'ready', 'ordinary_home': False, 'required': [
                {'condition': 'search field and keyboard are visible', 'basis': 'intent'}],
                'discarded': ['unsupported keyboard position'], 'reason': 'position was only a prediction'})

    trace = Trace(tmp_path)
    trace.start('freeze')
    model = Model()
    agent = Agent(None, SimpleNamespace(live=True, frame=after), model, model, trace)
    agent.task = 'open search'
    original = {'action': 'swipe', 'x1': 500, 'y1': 400, 'x2': 500, 'y2': 600,
                'expected': 'keyboard at top', 'evidence_id': 1}
    prepared = asyncio.run(agent._prepare_action(original, before))
    assert original['expected'] == 'keyboard at top'  # no mutation of the raw declaration
    assert prepared['raw_expected'] == original['expected']
    assert prepared['expected'] == '- search field and keyboard are visible'
    assert _key(prepared) == _key(original)
    records = list(map(json.loads, (trace.dir / 'steps.jsonl').read_text().splitlines()))
    frozen = records[-1]
    assert frozen['kind'] == 'action_conditions' and frozen['evidence_id'] == 1
    assert frozen['conditions'] == prepared['conditions']
    assert (trace.dir / frozen['frame']).read_bytes() == before


def test_promised_home_keeps_search_and_keyboard_absence_as_hard_conditions():
    import asyncio
    import json
    from types import SimpleNamespace

    from hidagent.agent import Agent
    from hidagent.debug import Trace

    class Model:
        model = 'fake'

        async def chat(self, *args, **kwargs):
            return json.dumps({'status': 'ready', 'ordinary_home': True,
                               'target': 'empty background outside the menu', 'top_toolbar': False,
                               'tool_selection': False,
                               'required': [{'condition': 'context menu is gone', 'basis': 'intent'}],
                               'discarded': [], 'reason': 'menu dismissal promises ordinary Home'})

    frame = jpeg('white')
    agent = Agent(None, SimpleNamespace(live=True, frame=frame), Model(), Model(), Trace(None))
    prepared = asyncio.run(agent._prepare_action({'action': 'tap', 'x': 1, 'y': 2,
                                                 'expected': 'menu gone and ordinary Home'}, frame))
    assert 'NO search field' in prepared['expected'] and 'NO keyboard' in prepared['expected']
    assert 'context menu is gone' in prepared['expected']


def test_production_step_freezes_contract_on_disk_before_device_action(tmp_path):
    import asyncio
    import json

    from hidagent.agent import Agent
    from hidagent.debug import Trace

    before, after, calls = jpeg('red'), jpeg('white'), []

    class Screen:
        live, frame = True, before

        async def fresh(self, *args, **kwargs):
            return self.frame

        async def settled(self):
            pass

    class Model:
        model = 'fake'

        async def chat(self, system, text, image=None, **kwargs):
            if not calls:
                assert image == before
                calls.append('conditions')
                return json.dumps({'status': 'ready', 'ordinary_home': False,
                                   'required': [{'condition': 'popover absent', 'basis': 'intent'}],
                                   'discarded': ['unnecessary position'], 'reason': 'dismiss popover'})
            calls.append('verify')
            assert calls == ['conditions', 'device', 'verify']
            assert 'popover absent' in text and 'unnecessary position' not in text
            return 'VERDICT: ok\nREASON: AFTER has no popover.'

    trace = Trace(tmp_path)
    trace.start('production ordering')
    screen, model = Screen(), Model()
    agent = Agent(None, screen, model, model, trace)

    async def next_actions():
        return [{'action': 'back', 'expected': 'popover absent at an unnecessary position', '_shot': before}]

    async def do(act):
        records = list(map(json.loads, (trace.dir / 'steps.jsonl').read_text().splitlines()))
        frozen = next(r for r in records if r.get('kind') == 'action_conditions')
        assert frozen['expected'] == act['expected'] == '- popover absent'
        assert (trace.dir / frozen['frame']).read_bytes() == before
        calls.append('device')
        screen.frame = after

    agent._next_actions, agent._do = next_actions, do
    assert asyncio.run(agent._step()) is False
    assert calls == ['conditions', 'device', 'verify'] and agent.steps[-1]['validated'] is True


def test_verbose_conditions_do_not_truncate_the_actor_action_history():
    from hidagent.agent import Agent
    from hidagent.debug import Trace

    agent = Agent(None, None, None, None, Trace(None))
    agent.steps = [{'expected': 'very verbose condition ' * 100, 'raw_expected': 'prediction',
                    'conditions': [{'condition': 'state', 'basis': 'intent'}], 'ordinary_home': False,
                    'action': 'tap', 'x': 123, 'y': 456, 'changed': True}]
    history = agent._history()
    assert '"action": "tap"' in history and '"x": 123' in history and '"y": 456' in history
    assert 'verbose' not in history and 'prediction' not in history and 'ordinary_home' not in history


def test_search_results_near_top_do_not_seed_toolbar_exploration():
    import asyncio

    from hidagent.agent import Agent
    from hidagent.debug import Trace

    agent = Agent(None, None, None, None, Trace(None))
    asyncio.run(agent._step_check_failed({'action': 'tap', 'x': 250, 'y': 160, 'top_toolbar': False},
                                        'search result did not open'))
    assert agent._toolbar_failed is None and agent._toolbar_scope is None


def test_tool_candidates_are_scoped_and_never_repeat_known_failures(monkeypatch):
    import asyncio
    from types import SimpleNamespace

    from hidagent.agent import Agent, StepCheckError
    from hidagent.debug import Trace

    frame = jpeg('white')
    screen = SimpleNamespace(live=True, frame=frame)
    agent = Agent(None, screen, None, None, Trace(None))
    calls = []

    async def inventory(*args, **kwargs):
        calls.append(True)
        return [{'x': 300, 'y': 50, 'appearance': 'circle'},
                {'x': 450, 'y': 50, 'appearance': 'rectangle'},
                {'x': 600, 'y': 50, 'appearance': 'diagonal line'}]

    monkeypatch.setattr('hidagent.agent.discover_tools', inventory)
    initial = {'action': 'tap', 'x': 600, 'y': 50, 'top_toolbar': True,
               'tool_selection': True, 'target': 'editing tool', 'expected': 'tool selected'}

    async def exercise():
        await agent._step_check_failed(initial, 'opened wrong menu', frame)
        first = await agent._tool_candidate(initial, frame)
        assert first['x'] == 300 and first['explore'] and first['expected'] == initial['expected']
        await agent._step_check_failed(first, 'wrong tool', frame)
        second = await agent._tool_candidate(initial, frame)
        assert second['x'] == 450 and len(calls) == 1
        # A different page cannot reuse this toolbar's cached candidates.
        assert await agent._tool_candidate(initial, jpeg('black')) == initial
        agent._tool_search['tried'].append({'x': 450, 'y': 50, 'result': 'wrong'})
        with pytest.raises(StepCheckError, match='already failed'):
            await agent._tool_candidate(initial, frame)

    asyncio.run(exercise())


def test_changed_toolbar_row_discards_old_page_exploration():
    import asyncio
    from types import SimpleNamespace

    from hidagent.agent import Agent
    from hidagent.debug import Trace

    class Actor:
        model = 'fake'

        async def chat(self, system, prompt, *args, **kwargs):
            assert 'last toolbar tap near' not in prompt
            return 'THINK: new page\nEXPECT: target opens\nDO: tap 500 50'

    agent = Agent(None, SimpleNamespace(live=True, frame=jpeg('black')), Actor(), Actor(), Trace(None))
    asyncio.run(agent._step_check_failed({'action': 'tap', 'x': 200, 'y': 50, 'top_toolbar': True},
                                        'wrong menu', jpeg('white')))
    asyncio.run(agent._next_actions())
    assert agent._toolbar_failed is None and agent._toolbar_scope is None


def test_changed_region_detail_finds_corner_palette_and_ignores_pointer():
    from PIL import Image, ImageDraw

    from hidagent.agent import _changed_regions

    before = Image.new('RGB', (400, 700), 'black')
    after = before.copy()
    draw = ImageDraw.Draw(after)
    draw.rectangle((10, 610, 60, 680), fill='white')  # small new tool palette
    draw.ellipse((190, 35, 210, 55), fill='white')  # moved pointer at the tapped toolbar button
    regions = _changed_regions(before, after, ((.5, 45/700),))
    assert any(b[0] <= 10 and b[1] <= 610 and b[2] >= 60 and b[3] >= 680 for b in regions)
    assert all(b[1] > 100 for b in regions)


def test_actor_history_includes_checked_effect_without_model_claims():
    from hidagent.agent import Agent
    from hidagent.debug import Trace

    agent = Agent(None, None, None, None, Trace(None))
    agent.steps = [{'action': 'tap', 'x': 100, 'y': 50, 'validated': True,
                    'validation': 'a collapsed tool palette is now visible', 'changed': True,
                    'thought': 'I assume it is ready'}]
    history = agent._history()
    assert 'checked effect: a collapsed tool palette' in history
    assert 'I assume' not in history


@pytest.mark.parametrize('explicit', [False, True])
def test_tool_entry_uses_functional_evidence_and_preserves_user_required_indicator(explicit):
    import asyncio
    import json
    from types import SimpleNamespace

    from hidagent.agent import Agent
    from hidagent.debug import Trace

    class Model:
        model = 'fake'

        async def chat(self, *args, **kwargs):
            return json.dumps({'status': 'ready', 'ordinary_home': False, 'target': 'editing pen control',
                               'top_toolbar': True, 'tool_selection': True, 'indicator_required': explicit,
                               'required': [{'condition': 'button must be highlighted',
                                             'basis': 'goal' if explicit else 'intent'}],
                               'discarded': [], 'reason': 'separate actual requirement from appearance guess'})

    frame = jpeg('white')
    agent = Agent(None, SimpleNamespace(live=True, frame=frame), Model(), Model(), Trace(None))
    result = asyncio.run(agent._prepare_action({'action': 'tap', 'x': 100, 'y': 50,
                                               'expected': 'button must be highlighted'}, frame))
    assert ('button must be highlighted' in result['expected']) is explicit
    assert 'corresponding tool-selection palette/launcher' in result['expected']
    assert result['raw_expected'] == 'button must be highlighted'


def test_new_compact_child_control_is_inspected_before_other_parent_tools(monkeypatch):
    import asyncio
    import io
    from types import SimpleNamespace

    from PIL import Image, ImageDraw

    from hidagent.agent import Agent
    from hidagent.debug import Trace

    im = Image.new('RGB', (400, 700), 'white')
    before_io = io.BytesIO()
    im.save(before_io, 'JPEG')
    ImageDraw.Draw(im).rectangle((10, 610, 60, 680), fill='black')
    after_io = io.BytesIO()
    im.save(after_io, 'JPEG')
    before, after = before_io.getvalue(), after_io.getvalue()
    agent = Agent(None, SimpleNamespace(live=True, frame=after), None, None, Trace(None))

    async def inventory(*args, region=None):
        assert region is not None  # do not skip to a different top-level toolbar control
        return [{'x': 90, 'y': 930, 'appearance': 'collapsed tool widget'}]

    monkeypatch.setattr('hidagent.agent.discover_tools', inventory)
    act = {'action': 'tap', 'x': 250, 'y': 50, 'top_toolbar': True,
           'tool_selection': True, 'target': 'editing tool', 'expected': 'mode or palette visible'}

    async def exercise():
        await agent._step_check_failed(act, 'new controls appeared but mode is not confirmed', before)
        chosen = await agent._tool_candidate(act, after)
        assert chosen['x'] == 90 and chosen['y'] == 930
        assert chosen['top_toolbar'] is False and chosen['explore']
        assert chosen['expected'] == act['expected']

    asyncio.run(exercise())


def test_small_stroke_is_checked_after_released_pointer_parks_outside_work():
    import asyncio
    import io

    from PIL import Image, ImageDraw

    from hidagent.agent import Agent
    from hidagent.debug import Trace

    canvas = Image.new('RGB', (400, 700), 'white')
    out = io.BytesIO()
    canvas.save(out, 'JPEG')
    before = out.getvalue()
    ImageDraw.Draw(canvas).ellipse((196, 346, 204, 354), outline='black', width=2)
    out = io.BytesIO()
    canvas.save(out, 'JPEG')
    after = out.getvalue()
    released, parks = [], []

    class Screen:
        live, frame, aspect = True, before, 400/700

        async def fresh(self, *args, **kwargs):
            return self.frame

        async def settled(self):
            pass

    class Phone:
        async def point(self, x, y):
            assert released
            parks.append((x, y))

    screen = Screen()
    agent = Agent(Phone(), screen, None, None, Trace(None))
    agent._pointer = (.9, .9)

    async def next_actions():
        return [{'action': 'circle', 'x': 500, 'y': 500, 'r': 8,
                 'expected': 'a small persistent ink circle appears', '_shot': before}]

    async def prepare(act, frame):
        return act

    async def do(act):
        screen.frame = after
        released.append(True)

    async def validate(act, before, after):
        assert act['changed'] is True and parks == [(.9, .9)]
        return True, 'small ink remains after moving the cursor', 0, after

    agent._next_actions, agent._prepare_action = next_actions, prepare
    agent._do, agent._validate_action = do, validate
    assert asyncio.run(agent._step()) is False
    assert agent.steps[-1]['validated'] is True and agent._pointer == (.9, .9)


def test_completion_receives_task_initial_and_current_evidence(tmp_path):
    import asyncio
    import json
    from types import SimpleNamespace

    from hidagent.agent import Agent
    from hidagent.debug import Trace

    before, after = jpeg('white'), jpeg('black')

    class Model:
        model = 'fake'

        async def chat(self, system, text, image=None, **kwargs):
            assert image not in (before, after)
            assert 'INITIAL screen' in text and 'Actual input record' in text
            assert 'pre-existing' in system
            return 'VERDICT: incomplete\nREASON: no newly created content is established'

    trace = Trace(tmp_path)
    trace.start('initial versus final')
    agent = Agent(None, SimpleNamespace(live=True, frame=after), Model(), Model(), trace)
    agent._initial_frame = before
    assert asyncio.run(agent._verify('already present'))[0] is False
    records = list(map(json.loads, (trace.dir/'steps.jsonl').read_text().splitlines()))
    raw = next(r for r in records if r.get('kind') == 'completion_screen')
    assert (trace.dir/raw['frame']).read_bytes() == after


def test_verified_enclosing_work_boundary_blocks_drift_without_another_model_call():
    import asyncio
    import json
    from types import SimpleNamespace

    from hidagent.agent import Agent
    from hidagent.debug import Trace

    class Model:
        model, calls = 'fake', 0

        async def chat(self, *args, **kwargs):
            self.calls += 1
            return json.dumps({'status': 'ready', 'ordinary_home': False, 'work_item_consistent': True,
                               'encloses_remaining_work': self.calls == 1,
                               'required': [{'condition': 'new ink appears', 'basis': 'intent'}],
                               'discarded': [], 'reason': 'one contained composite item'})

    frame = jpeg('white')
    model = Model()
    agent = Agent(None, SimpleNamespace(live=True, frame=frame, aspect=.5), model, model, Trace(None))

    async def exercise():
        outline = await agent._prepare_action({'action': 'circle', 'x': 200, 'y': 400, 'r': 90,
                                               'expected': 'enclosing outline'}, frame)
        assert agent._creation_boundary is None  # declaration alone does not prove creation
        agent._accept_work_boundary(outline, frame)  # called only after a successful result check
        assert agent._creation_boundary == (110, 355, 290, 445)
        with pytest.raises(ValueError, match='leave this run'):
            await agent._prepare_action({'action': 'circle', 'x': 400, 'y': 380, 'r': 8,
                                         'expected': 'detail on a different object'}, frame)
        assert model.calls == 1
        valid = await agent._prepare_action({'action': 'path', 'points': [[160, 410], [200, 430], [240, 410]],
                                             'expected': 'a detail inside the new item'}, frame)
        assert valid['action'] == 'path' and model.calls == 2

    asyncio.run(exercise())


def test_content_journal_preserves_checked_work_across_recovery_history_boundary():
    import json

    from hidagent.agent import Agent
    from hidagent.debug import Trace

    agent = Agent(None, None, None, None, Trace(None))
    agent.steps = [{'action': 'circle', 'x': 200, 'y': 400, 'r': 90, 'validated': True,
                    'validation': 'new enclosing structure on the left'},
                   {'action': 'path', 'points': [[1, 2], [3, 4]], 'validated': False, 'validation': 'no ink'}]
    agent._history_start = 2
    journal = json.loads(agent._content_journal())
    assert len(journal) == 1 and journal[0]['input']['x'] == 200


@pytest.mark.parametrize('observable', [True, False])
def test_stroke_requires_observed_start_and_reanchors_logical_pose(monkeypatch, observable):
    import asyncio

    from hidagent.agent import Agent, StepCheckError
    from hidagent.debug import Trace

    sent, drawn = [], []

    class Device:
        async def send(self, line):
            sent.append(line)

    class Phone:
        calibrated, span, _pos = True, {'w': 1000, 'h': 1000}, None
        device = Device()

        def _dx(self, x): return round(x*1000)
        def _dy(self, y): return round(y*1000)

        async def point(self, x, y):
            self._pos = (self._dx(x), self._dy(y))

        async def drag(self, points):
            assert self._pos == (500, 500)  # do not subtract the visual correction from the next segment
            drawn.append(points)

    class Screen:
        live, frame, aspect = True, b'frame', .5

        async def fresh(self, *args, **kwargs): return self.frame
        async def settled(self, **kwargs): pass

    observed = iter([(.55, .5), (.5, .5)])
    monkeypatch.setattr('hidagent.agent.vision.pointer',
                        lambda *args, **kwargs: next(observed) if observable else None)
    phone = Phone()
    agent = Agent(phone, Screen(), None, None, Trace(None))
    act = {'action': 'path', 'points': [[500, 500], [600, 600]]}
    if observable:
        asyncio.run(agent._draw_verified(act))
        assert len(drawn) == 1 and 'move -50 0' in sent
        assert act['start_pose']['confirmed'] and act['start_pose']['attempts'] == 2
    else:
        with pytest.raises(StepCheckError, match='no press was sent'):
            asyncio.run(agent._draw_verified(act))
        assert drawn == [] and phone._pos is None and len(sent) == 6


def test_independent_completion_does_not_receive_executor_or_checker_claims():
    import asyncio
    from types import SimpleNamespace

    from hidagent.agent import Agent
    from hidagent.debug import Trace

    class Model:
        model = 'fake'

        async def chat(self, system, text, image=None, **kwargs):
            assert 'POISON_CLAIM' not in text and 'POISON_CHECK' not in text
            assert '"action": "circle"' in text
            return 'VERDICT: incomplete\nREASON: required content is not actually visible'

    agent = Agent(None, SimpleNamespace(live=True, frame=jpeg('white')), Model(), Model(), Trace(None))
    agent.steps = [{'action': 'circle', 'x': 500, 'y': 500, 'r': 90,
                    'validated': True, 'validation': 'POISON_CHECK'}]
    assert asyncio.run(agent._verify('POISON_CLAIM'))[0] is False


def test_blank_work_check_rejects_existing_ink_but_not_faint_canvas_grid():
    import io

    from PIL import Image, ImageDraw

    from hidagent.agent import Agent

    im = Image.new('RGB', (400, 700), 'black')
    draw = ImageDraw.Draw(im)
    for x in range(0, 400, 20):
        for y in range(0, 700, 20):
            draw.ellipse((x, y, x+1, y+1), fill=(30, 30, 30))
    out = io.BytesIO()
    im.save(out, 'JPEG')
    bounds = (400, 440, 600, 560)
    assert Agent._blank_work_area(out.getvalue(), bounds)[0] is True
    draw.ellipse((160, 310, 240, 390), outline='white', width=3)
    out = io.BytesIO()
    im.save(out, 'JPEG')
    assert Agent._blank_work_area(out.getvalue(), bounds)[0] is False


def test_native_pixel_change_detail_retains_a_tiny_new_stroke():
    from PIL import Image, ImageDraw

    from hidagent.agent import _changed_regions

    before = Image.new('RGB', (752, 1080), 'black')
    after = before.copy()
    ImageDraw.Draw(after).rectangle((400, 400, 402, 402), fill='white')
    boxes = _changed_regions(before, after, ())
    assert any(b[0] <= 400 and b[1] <= 400 and b[2] >= 402 and b[3] >= 402 for b in boxes)


def test_clear_work_suggestions_are_derived_from_current_pixels_and_fit_requested_size():
    import io

    from PIL import Image, ImageDraw

    from hidagent.agent import Agent

    im = Image.new('RGB', (400, 700), 'black')
    ImageDraw.Draw(im).ellipse((160, 310, 240, 390), outline='white', width=3)
    out = io.BytesIO()
    im.save(out, 'JPEG')
    frame = out.getvalue()
    candidates = Agent._clear_work_areas(frame, (400, 440, 600, 560))
    assert 1 <= len(candidates) <= 3
    for candidate in candidates:
        bounds = candidate['bounds']
        assert abs((bounds[2]-bounds[0])-200) <= 2
        assert abs((bounds[3]-bounds[1])-120) <= 2
        assert Agent._blank_work_area(frame, bounds)[0]


def test_unchanged_pixels_do_not_override_a_checked_idempotent_state():
    import asyncio

    from hidagent.agent import Agent
    from hidagent.debug import Trace

    frame = jpeg('white')

    class Screen:
        live, frame = True, None
        async def fresh(self, *args, **kwargs): return self.frame
        async def settled(self): pass

    screen = Screen()
    screen.frame = frame
    agent = Agent(None, screen, None, None, Trace(None))

    async def next_actions():
        return [{'action': 'back', 'expected': 'requested state visible', '_shot': frame}]

    async def prepare(act, before): return act
    async def do(act): pass
    async def validate(*args): return True, 'requested state is already visible', 0, frame

    agent._next_actions, agent._prepare_action = next_actions, prepare
    agent._do, agent._validate_action = do, validate
    assert asyncio.run(agent._step()) is False
    assert agent.steps[-1]['changed'] is False and agent.steps[-1]['validated'] is True
    assert agent._mismatches == 0 and agent._missed() is False


def test_unverifiably_small_stroke_is_rejected_before_model_or_input():
    import asyncio
    import io
    from types import SimpleNamespace

    import pytest
    from PIL import Image

    from hidagent.agent import Agent
    from hidagent.debug import Trace

    out = io.BytesIO()
    Image.new('RGB', (752, 1080), 'black').save(out, 'JPEG')
    agent = Agent(None, SimpleNamespace(aspect=752/1080), None, None, Trace(None))
    agent._creation_boundary = (100, 100, 400, 400)
    with pytest.raises(ValueError, match='below the verifiable input extent'):
        asyncio.run(agent._prepare_action({'action': 'path', 'points': [[200, 200], [204, 204]]}, out.getvalue()))


def test_large_clear_space_observations_avoid_existing_content():
    import io

    from PIL import Image, ImageDraw

    from hidagent.agent import Agent

    image = Image.new('RGB', (752, 1080), 'black')
    ImageDraw.Draw(image).rectangle((0, 0, 751, 500), fill='white')
    out = io.BytesIO()
    image.save(out, 'JPEG')
    frame = out.getvalue()
    candidates = Agent._observed_clear_spaces(frame)
    assert candidates
    assert all(c['bounds'][1] > 460 and Agent._blank_work_area(frame, c['bounds'])[0] for c in candidates)


def test_failed_non_top_tool_uses_its_own_visible_group(monkeypatch):
    import asyncio
    from types import SimpleNamespace

    from hidagent.agent import Agent
    from hidagent.debug import Trace

    frame = jpeg('white')
    agent = Agent(None, SimpleNamespace(live=True, frame=frame), None, None, Trace(None))

    async def inventory(*args, region=None):
        assert region and region[1] > 0
        return [{'x': 250, 'y': 945, 'appearance': 'lettered nib'},
                {'x': 315, 'y': 945, 'appearance': 'plain nib'}]

    monkeypatch.setattr('hidagent.agent.discover_tools', inventory)
    act = {'action': 'tap', 'x': 250, 'y': 945, 'top_toolbar': False,
           'tool_selection': True, 'target': 'drawing tool', 'expected': 'ink tool active'}

    async def exercise():
        await agent._step_check_failed(act, 'requested capability not visible', frame)
        result = await agent._tool_candidate(act, frame)
        assert result['x'] == 315 and result['explore'] and not result['top_toolbar']

    asyncio.run(exercise())
