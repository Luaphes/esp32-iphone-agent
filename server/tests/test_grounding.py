import asyncio
import io

import pytest
from PIL import Image

from hidagent.debug import Trace
from hidagent.grounding import discover_tools, locate_tap


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
    assert (result["x"], result["y"]) == (241, 743)
    assert result["proposed"] == {"x": 178, "y": 719}
    assert proposed["x"] == 178


def test_toolbar_tap_searches_the_whole_toolbar_band():
    llm = Locator("POINT 250 500")
    result = asyncio.run(locate_tap(llm, frame(), {"action": "tap", "x": 620, "y": 50,
                                                    "top_toolbar": True, "thought": "the drawing tool"}, Trace(None)))
    # The full 752-pixel toolbar is the crop; 250/1000 of it is the left-side pen (screen x=0.25).
    assert (result["x"], result["y"]) == (250, 100)
    assert result["proposed"] == {"x": 620, "y": 50}


def test_toolbar_crop_preserves_icon_shape_and_ignores_mixed_old_context():
    from PIL import ImageDraw

    im = Image.new("RGB", (752, 1080), "white")
    ImageDraw.Draw(im).rectangle((100, 50, 140, 90), fill="red")
    output = io.BytesIO()
    im.save(output, "JPEG")
    llm = Locator("POINT 250 500")
    asyncio.run(locate_tap(llm, output.getvalue(), {"x": 620, "y": 50, "top_toolbar": True,
                         "target": "current square icon", "thought": "old failed search field"}, Trace(None)))
    red = [(x, y) for y in range(1000) for x in range(1000)
           if (p := llm.image.getpixel((x, y)))[0] > 180 and p[1] < 80 and p[2] < 80]
    xs, ys = zip(*red, strict=True)
    assert abs((max(xs)-min(xs)) - (max(ys)-min(ys))) <= 2
    assert 'old failed search field' not in llm.prompt and 'current square icon' in llm.prompt


def test_padding_is_not_a_valid_target():
    with pytest.raises(RuntimeError, match="padding"):
        asyncio.run(locate_tap(Locator("POINT 250 100"), frame(),
                              {"x": 620, "y": 50, "top_toolbar": True}, Trace(None)))


def test_neutral_tool_inventory_rejects_padding_and_destructive_controls():
    llm = Locator('''{"tools":[
      {"x":300,"y":500,"appearance":"circle with tip","kind":"tool"},
      {"x":500,"y":500,"appearance":"trash can","kind":"tool"},
      {"x":700,"y":500,"appearance":"jar or container","kind":"tool"},
      {"x":400,"y":500,"appearance":"small white dot","kind":"tool"},
      {"x":600,"y":500,"appearance":"box with arrow","kind":"sharing"},
      {"x":200,"y":100,"appearance":"square in padding","kind":"tool"}]}''')
    result = asyncio.run(discover_tools(llm, frame(), Trace(None)))
    assert len(result) == 1 and result[0]['x'] == 300
    assert 'circle' in result[0]['appearance']


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
