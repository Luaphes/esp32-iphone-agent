import asyncio
import importlib
import json
from types import SimpleNamespace

import httpx
import pytest

from hidagent import cli
from hidagent.client import IPhoneClient


@pytest.fixture
def api(tmp_path, monkeypatch):
    monkeypatch.setenv("HIDAGENT_DATA", str(tmp_path))
    module = importlib.import_module("hidagent.app")
    calls = []

    class Phone:
        span = {"w": 400, "h": 800}
        calibrated = True
        async def tap(self, x, y, aim=False): calls.append(("tap", x, y))
        async def type(self, text): calls.append(("type", text))
        async def key(self, text): calls.append(("key", text))
        async def start_calibration(self): return True
        async def finish_calibration(self, x, y): calls.append(("calibrate", x, y))
        async def reset(self): pass

    class Agent:
        running, task, steps = False, "", []
        trace = SimpleNamespace(dir=None)
        actor = planner = SimpleNamespace(model="fake")
        def start(self, goal): self.running, self.task = True, goal
        def stop(self): self.running = False

    monkeypatch.setattr(module, "phone", Phone())
    monkeypatch.setattr(module, "agent", Agent())
    monkeypatch.setattr(module, "device", SimpleNamespace(connected=False))
    monkeypatch.setattr(module, "screen", SimpleNamespace(live=True, frame=b"jpeg", aspect=.46))
    monkeypatch.setattr(module, "manual_lock", asyncio.Lock())
    return module, calls


def test_client_uses_legacy_routes_and_single_current_task():
    requests = []

    def respond(request):
        requests.append((request.method, request.url.path, request.content))
        if request.url.path == "/api/status":
            return httpx.Response(200, json={"agent": {"running": False, "task": "", "steps": []}})
        return httpx.Response(200, json={"ok": True})

    with IPhoneClient("http://legacy", transport=httpx.MockTransport(respond)) as client:
        client.delegate("Open Clock")
        assert client.task_status()["running"] is False
        client.stop()
        client.action("key", text="cmd+a")
        client.calibrate(.5, .4)
        with pytest.raises(ValueError):
            client.calibrate(.5)
        with pytest.raises(ValueError):
            client.action("raw", text="home")
    assert [path for _, path, _ in requests] == [
        "/api/agent", "/api/status", "/api/agent/stop", "/api/key", "/api/calibrate/finish"]
    assert requests[0][2] == b'{"text":"Open Clock"}'


def test_cli_keeps_json_on_stdout_and_errors_on_stderr(monkeypatch, capsys):
    def respond(request):
        return httpx.Response(409, json={"detail": "phone is busy"})

    monkeypatch.setattr(cli, "IPhoneClient", lambda url: IPhoneClient(url, transport=httpx.MockTransport(respond)))
    assert cli.main(["--url", "http://legacy", "type", "Calendar"]) == 2
    output = capsys.readouterr()
    assert not output.out and '"error"' in output.err


def test_cli_status_and_screenshot_do_not_import_mcp(monkeypatch, capsys, tmp_path):
    import sys

    monkeypatch.setitem(sys.modules, "mcp", None)

    def respond(request):
        if request.url.path == "/api/frame.jpg":
            return httpx.Response(200, content=b"synthetic jpeg")
        return httpx.Response(200, json={"device": True})

    monkeypatch.setattr(cli, "IPhoneClient", lambda url: IPhoneClient(url, transport=httpx.MockTransport(respond)))
    assert cli.main(["--url", "http://legacy", "status"]) == 0
    assert json.loads(capsys.readouterr().out) == {"device": True}
    output = tmp_path / "phone.jpg"
    assert cli.main(["--url", "http://legacy", "screenshot", str(output)]) == 0
    assert output.read_bytes() == b"synthetic jpeg"
    assert json.loads(capsys.readouterr().out)["path"] == str(output)


@pytest.mark.parametrize("url", ["file:///tmp/phone", "http://legacy?key=secret", "http://legacy#fragment"])
def test_client_rejects_unsupported_urls(url):
    with pytest.raises(ValueError):
        IPhoneClient(url)


def test_direct_action_and_agent_start_cannot_interleave(api, monkeypatch):
    module, calls = api

    async def scenario():
        entered, release = asyncio.Event(), asyncio.Event()

        async def tap(x, y, aim=False):
            assert aim is True
            entered.set()
            await release.wait()
            calls.append(("tap", x, y))

        monkeypatch.setattr(module.phone, "tap", tap)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=module.app), base_url="http://legacy") as client:
            pending = asyncio.create_task(client.post("/api/tap", json={"x": .5, "y": .4}))
            await entered.wait()
            for path, body in [("/api/key", {"text": "enter"}), ("/api/agent", {"text": "Open Clock"}),
                               ("/api/calibrate/start", None), ("/api/calibrate/finish", {"x": .5, "y": .4})]:
                assert (await client.post(path, json=body)).status_code == 409
            assert not calls
            assert (await client.post("/api/agent/stop")).status_code == 200
            release.set()
            assert (await pending).status_code == 200
            assert (await client.post("/api/agent", json={"text": "Open Clock"})).status_code == 200
            assert (await client.post("/api/type", json={"text": "Calendar"})).status_code == 409
            assert calls == [("tap", .5, .4)]

    asyncio.run(scenario())


@pytest.mark.parametrize("path,body", [
    ("/api/tap", {"x": 2, "y": .5}),
    ("/api/swipe", {"x1": 0, "y1": 0, "x2": -1, "y2": .5}),
    ("/api/scroll", {"x": .5, "y": .5, "lines": 101}),
    ("/api/type", {"text": "Calendar\nhome"}),
    ("/api/key", {"text": "enter\rhome"}),
    ("/api/type", {"text": "你好😀"}),
    ("/api/type", {"text": "\u007f"}),
    ("/api/calibrate/finish", {"x": 0, "y": .5}),
])
def test_invalid_remote_input_is_rejected_before_hid(api, path, body):
    module, calls = api

    async def scenario():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=module.app), base_url="http://legacy") as client:
            assert (await client.post(path, json=body)).status_code == 422
        assert not calls

    asyncio.run(scenario())


def test_chinese_api_returns_uncommitted_candidate_status(api, monkeypatch, tmp_path):
    from hidagent.phone import Phone

    module, commands = api

    async def send(command): commands.append(command)

    monkeypatch.setattr(module, "phone", Phone(SimpleNamespace(send=send), tmp_path / "calibration.json"))

    async def scenario():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=module.app), base_url="http://legacy") as client:
            response = await client.post("/api/type", json={"text": "美团"})
            assert response.status_code == 200
            assert response.json()["input"] == {
                "mode": "pinyin", "target": "美团", "keys": "mei'tuan", "needs_confirmation": True}
        assert commands == ["type mei'tuan"]

    asyncio.run(scenario())


def test_manual_calibration_waits_for_probe_and_excludes_other_actions(api, monkeypatch):
    module, calls = api
    async def scenario():
        entered, finish = asyncio.Event(), asyncio.Event()
        async def calibrate(x, y):
            entered.set()
            await finish.wait()
            calls.append(("calibrate", x, y))
        monkeypatch.setattr(module.phone, "finish_calibration", calibrate)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=module.app), base_url="http://legacy") as client:
            pending = asyncio.create_task(client.post("/api/calibrate/finish", json={"x": .4, "y": .4}))
            await entered.wait()
            assert not pending.done()
            assert (await client.post("/api/agent", json={"text": "Open Clock"})).status_code == 409
            assert (await client.post("/api/tap", json={"x": .5, "y": .5})).status_code == 409
            finish.set()
            response = await pending
            assert response.status_code == 200 and response.json()["span"] == module.phone.span
    asyncio.run(scenario())


def test_failed_manual_probe_is_retryable_api_error(api, monkeypatch):
    module, calls = api
    async def fail(x, y): raise RuntimeError("pointer not visible; retry")
    monkeypatch.setattr(module.phone, "finish_calibration", fail)
    async def scenario():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=module.app), base_url="http://legacy") as client:
            response = await client.post("/api/calibrate/finish", json={"x": .4, "y": .4})
            assert response.status_code == 409 and "retry" in response.json()["detail"]
    asyncio.run(scenario())
