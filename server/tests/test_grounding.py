import asyncio
import io

import pytest
from PIL import Image

from hidagent.debug import Trace
from hidagent.grounding import locate_tap


class Locator:
    model = "test"

    def __init__(self, reply):
        self.reply = reply

    async def chat(self, system, text, image, **kwargs):
        self.system = system
        self.image = Image.open(io.BytesIO(image))
        self.prompt = text
        return self.reply


def frame():
    output = io.BytesIO()
    Image.new("RGB", (752, 1080), "white").save(output, format="JPEG")
    return output.getvalue()


@pytest.mark.parametrize("reply", ["POINT 620 580", "<｜｜point｜｜>[[620,580]]<｜｜/point｜｜>"])
def test_crop_point_maps_back_to_original_screen(reply):
    llm = Locator(reply)
    proposed = {"action": "tap", "x": 178, "y": 719, "thought": "Select the exact candidate 美团"}
    result = asyncio.run(locate_tap(llm, frame(), proposed, Trace(None)))
    assert llm.image.size == (1000, 1000)
    assert "美团" in llm.prompt
    assert "blank note body" in llm.system
    assert (result["x"], result["y"]) == (234, 743)
    assert result["proposed"] == {"x": 178, "y": 719}
    assert proposed["x"] == 178


@pytest.mark.parametrize("reply", ["NOT_FOUND", "POINT 1001 200", "POINT 0 200", "maybe POINT 200 300"])
def test_uncertain_or_outside_crop_never_becomes_a_tap(reply):
    with pytest.raises(RuntimeError, match="not located"):
        asyncio.run(locate_tap(Locator(reply), frame(), {"x": 500, "y": 500}, Trace(None)))


def test_invalid_full_screen_coordinates_do_not_call_model():
    llm = Locator("POINT 500 500")
    with pytest.raises(ValueError, match="0-1000"):
        asyncio.run(locate_tap(llm, frame(), {"x": -1, "y": 500}, Trace(None)))
    assert not hasattr(llm, "image")


def test_screen_changed_during_grounding_discards_stale_tap(monkeypatch):
    from types import SimpleNamespace

    from hidagent.agent import Agent

    old = frame()
    output = io.BytesIO()
    Image.new("RGB", (752, 1080), "black").save(output, format="JPEG")
    screen = SimpleNamespace(live=True, frame=old)

    async def locate(*args):
        screen.frame = output.getvalue()
        return {"action": "tap", "x": 500, "y": 500}

    monkeypatch.setattr("hidagent.agent.locate_tap", locate)
    agent = Agent(None, screen, None, None, Trace(None))
    with pytest.raises(RuntimeError, match="discard the tap"):
        asyncio.run(agent._ground_tap({"action": "tap", "x": 500, "y": 500}, old))
