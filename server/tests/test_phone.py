import asyncio
import io
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image, ImageDraw

from hidagent.device import DeviceError
from hidagent.phone import Phone


class Board:
    links = 0
    def __init__(self, error=None):
        self.commands = []
        self.error = error

    async def send(self, command):
        self.commands.append(command)
        if self.error:
            raise DeviceError(self.error)


def test_home_does_not_need_pointer_calibration_or_an_edge_gesture(tmp_path):
    board = Board()
    phone = Phone(board, tmp_path / "calibration.json")
    phone._pos = (100, 200)

    asyncio.run(phone.go_home())

    assert board.commands == ["homescreen"]
    assert phone.span is None
    assert phone._pos is None


def test_old_firmware_requests_an_update_instead_of_retrying_the_failed_swipe(tmp_path):
    board = Board("err unknown command")
    phone = Phone(board, tmp_path / "calibration.json")

    with pytest.raises(DeviceError, match="flash the updated firmware"):
        asyncio.run(phone.go_home())

    assert board.commands == ["homescreen"]


def test_home_propagates_usb_failure_and_invalidates_pointer_position(tmp_path):
    board = Board("err phone usb asleep (unlock the phone)")
    phone = Phone(board, tmp_path / "calibration.json")
    phone._pos = (100, 200)

    with pytest.raises(DeviceError, match="phone usb asleep"):
        asyncio.run(phone.go_home())

    assert phone._pos is None


@pytest.mark.parametrize("text", ["你好abc", "café", "hello\nworld", "hello\tworld", "\x7f", "😀", "你" * 13])
def test_unsupported_keyboard_text_never_reaches_the_board(tmp_path, text):
    board = Board()
    phone = Phone(board, tmp_path / "calibration.json")

    with pytest.raises(ValueError, match="printable ASCII"):
        asyncio.run(phone.type(text))

    assert board.commands == []


def test_supported_keyboard_text_is_preserved(tmp_path):
    board = Board()
    phone = Phone(board, tmp_path / "calibration.json")

    asyncio.run(phone.type("Bluetooth 5.0!"))

    assert board.commands == ["type Bluetooth 5.0!"]


@pytest.mark.parametrize("text,keys", [("美团", "mei'tuan"), ("重庆", "chong'qing"), ("女", "nv")])
def test_chinese_typing_sends_only_pinyin_without_committing_a_candidate(tmp_path, text, keys):
    board = Board()
    phone = Phone(board, tmp_path / "calibration.json")

    result = asyncio.run(phone.type(text))

    assert board.commands == [f"type {keys}"]
    assert result["target"] == text
    assert result["needs_confirmation"] is True


def test_calibration_does_not_transfer_from_iphone_to_ipad_or_across_rotation(tmp_path):
    screen = SimulatedScreen()
    path = tmp_path / "calibration.json"
    phone = Phone(SimulatedBoard(screen), path, screen)
    assert asyncio.run(phone.start_calibration())
    assert phone.calibrated
    assert Phone(Board(), path, screen).calibrated

    screen.height = 600
    assert not phone.calibrated
    with pytest.raises(RuntimeError, match="recalibrate"):
        asyncio.run(phone.point(.5, .5))
    assert asyncio.run(phone.start_calibration())
    assert phone.calibrated
    screen.width = 800
    assert not phone.calibrated


def test_old_calibration_without_screen_shape_requires_recalibration(tmp_path):
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps({"w": 504, "h": 1096}))
    phone = Phone(Board(), path, SimpleNamespace(live=True, aspect=0.75))
    assert not phone.calibrated


class SimulatedScreen:
    live = True
    width, height = 500, 800
    x, y = .4, .4
    frames = True
    background = 30

    @property
    def aspect(self): return self.width/self.height

    @property
    def frame(self):
        im = Image.new("RGB", (self.width, self.height), (self.background, 50, 70))
        px, py = round(self.x*self.width), round(self.y*self.height)
        ImageDraw.Draw(im).ellipse((px-9, py-9, px+9, py+9), fill=(230, 230, 230))
        out = io.BytesIO()
        im.save(out, "JPEG", quality=85)
        return out.getvalue()

    async def settled(self, **kwargs): pass
    async def fresh(self, *args): return self.frame if self.frames else None


class SimulatedBoard(Board):
    def __init__(self, screen):
        super().__init__()
        self.screen = screen
        self.gain = 1
        self.on_move = None

    async def send(self, command):
        await super().send(command)
        if command.startswith("move "):
            dx, dy = map(int, command.split()[1:])
            self.screen.x = max(0, min(1, self.screen.x+self.gain*dx/1000))
            self.screen.y = max(0, min(1, self.screen.y+self.gain*dy/1600))
            if self.on_move: self.on_move(dx, dy)


def calibrated_phone(tmp_path):
    screen = SimulatedScreen()
    board = SimulatedBoard(screen)
    phone = Phone(board, tmp_path/"calibration.json", screen)
    assert asyncio.run(phone.start_calibration())
    return phone, board, screen


def test_relative_probe_calibration_and_tap_never_home(tmp_path):
    phone, board, screen = calibrated_phone(tmp_path)
    assert phone.span["w"] == pytest.approx(1000, rel=.06)
    assert phone.span["h"] == pytest.approx(1600, rel=.06)
    saved = json.loads(phone._file.read_text())
    assert saved["version"] == 2 and "seen" not in saved
    asyncio.run(phone.tap(.7, .65))
    assert (screen.x, screen.y) == pytest.approx((.7, .65), abs=.005)
    assert board.commands[-1] == "click"
    assert "home" not in board.commands
    for cmd in board.commands:
        if cmd.startswith("move "):
            dx, dy = map(int, cmd.split()[1:])
            assert not (dx and dy)


def test_legacy_corner_calibration_with_aspect_is_rejected(tmp_path):
    path = tmp_path/"calibration.json"
    path.write_text(json.dumps({"w": 1000, "h": 1600, "aspect": .625, "seen": {"x": [[0, .08]]}}))
    assert not Phone(Board(), path, SimulatedScreen()).calibrated


def test_diagonal_drag_remains_a_straight_stroke(tmp_path):
    phone, board, screen = calibrated_phone(tmp_path)
    asyncio.run(phone.drag([(.4, .4), (.6, .6)], hold=False))
    drawing = board.commands[board.commands.index("press") + 1:]
    assert len(drawing) == 2 and drawing[-1] == "release"
    _, dx, dy = drawing[0].split()
    assert int(dx) > 0 and int(dy) > 0


def test_failed_final_calibration_probe_does_not_commit_scale(tmp_path):
    screen = SimulatedScreen()
    phone = Phone(SimulatedBoard(screen), tmp_path / "calibration.json", screen)
    find = phone._find_pointer

    async def final_failure(*args, **kwargs):
        if phone.span is not None:
            raise RuntimeError("disconnected during final probe")
        return await find(*args, **kwargs)

    phone._find_pointer = final_failure
    with pytest.raises(RuntimeError, match="final probe"):
        asyncio.run(phone.start_calibration())
    assert not phone.calibrated and not phone._file.exists()


def test_unknown_pointer_fallback_has_only_finite_horizontal_probe(tmp_path):
    screen = SimulatedScreen()
    screen.frames = False
    board = SimulatedBoard(screen)
    phone = Phone(board, tmp_path/"calibration.json", screen)
    assert not asyncio.run(phone.start_calibration())
    assert board.commands == ["release", "move 12 0", "move -12 0", "move -12 0", "move 12 0"]
    screen.frames = True
    asyncio.run(phone.finish_calibration(screen.x, screen.y))
    assert phone.calibrated


@pytest.mark.parametrize("event", ["stop", "reconnect", "rotate"])
def test_manual_session_invalidated_before_probing(tmp_path, event):
    screen = SimulatedScreen()
    screen.frames = False
    board = SimulatedBoard(screen)
    phone = Phone(board, tmp_path/"calibration.json", screen)
    assert not asyncio.run(phone.start_calibration())
    if event == "stop": asyncio.run(phone.reset())
    elif event == "reconnect": board.links += 1
    else: screen.width = 600
    screen.frames = True
    before = list(board.commands)
    with pytest.raises(RuntimeError): asyncio.run(phone.finish_calibration(screen.x, screen.y))
    assert board.commands == before


def test_reset_only_releases_and_unknown_manual_finish_fails(tmp_path):
    phone = Phone(Board(), tmp_path/"calibration.json")
    asyncio.run(phone.reset())
    assert phone.device.commands == ["release"]
    with pytest.raises(RuntimeError, match="start calibration"):
        asyncio.run(phone.finish_calibration(.4, .4))


def test_manual_edge_hover_recovery_is_bounded_and_requires_observation(tmp_path):
    screen = SimulatedScreen()
    board = Board()
    phone = Phone(board, tmp_path/"calibration.json", screen)
    phone._pending = phone._context()

    async def hidden(*args, **kwargs): return None
    phone._find_pointer = hidden
    with pytest.raises(RuntimeError, match="could not be verified"):
        asyncio.run(phone.finish_calibration(.005, .005))
    assert board.commands == ["move 24 24"] * 3
    assert not phone.calibrated and not phone._file.exists()


def test_manual_edge_hover_can_resume_only_from_a_verified_anchor(tmp_path):
    screen = SimulatedScreen()
    board = Board()
    phone = Phone(board, tmp_path/"calibration.json", screen)
    phone._pending = phone._context()
    observations = iter([None, None, (.04, .04)])
    anchors = []

    async def find(*args, **kwargs): return next(observations)
    async def calibrate(anchor, context): anchors.append(anchor)
    phone._find_pointer, phone._calibrate = find, calibrate
    asyncio.run(phone.finish_calibration(.005, .005))
    assert anchors == [(.04, .04)]
    assert board.commands == ["move 24 24"] * 2


def test_stop_during_probe_prevents_following_motion(tmp_path):
    screen = SimulatedScreen()
    board = SimulatedBoard(screen)
    phone = Phone(board, tmp_path/"calibration.json", screen)
    board.on_move = lambda dx, dy: setattr(phone, "_epoch", phone._epoch+1)
    with pytest.raises(RuntimeError, match="invalidated"):
        asyncio.run(phone.start_calibration())
    assert board.commands == ["release", "move 12 0"]
    assert not phone._file.exists()


def test_missing_pointer_never_clicks_even_without_precise_flag(tmp_path):
    phone, board, screen = calibrated_phone(tmp_path)
    screen.frames = False
    with pytest.raises(RuntimeError, match="pointer not visible"):
        asyncio.run(phone.tap(.5, .5))
    assert "click" not in board.commands


def test_bad_motion_during_aim_cannot_click(tmp_path):
    phone, board, screen = calibrated_phone(tmp_path)
    board.gain = 4
    with pytest.raises(RuntimeError): asyncio.run(phone.tap(.7, .65))
    assert "click" not in board.commands


def test_screen_change_during_motion_cannot_click(tmp_path):
    phone, board, screen = calibrated_phone(tmp_path)
    board.on_move = lambda dx, dy: setattr(screen, "background", 220) if abs(dx) > 12 or dy else None
    with pytest.raises(RuntimeError, match="unexpected"):
        asyncio.run(phone.tap(.7, .65))
    assert "click" not in board.commands


def test_zero_motion_probe_does_not_save_calibration(tmp_path):
    screen = SimulatedScreen()
    board = SimulatedBoard(screen)
    phone = Phone(board, tmp_path/"calibration.json", screen)
    async def find(*args, **kwargs): return (.4, .4)
    phone._find_pointer = find
    with pytest.raises(RuntimeError, match="inconsistent"):
        asyncio.run(phone.start_calibration())
    assert not phone.calibrated and not phone._file.exists()


def test_visible_edge_anchor_is_followed_inward_without_corner_sweep(tmp_path):
    screen = SimulatedScreen()
    screen.x, screen.y = .04, .04
    board = SimulatedBoard(screen)
    phone = Phone(board, tmp_path/"calibration.json", screen)
    assert asyncio.run(phone.start_calibration())
    assert .04 < screen.x < .96 and .04 < screen.y < .96
    assert "home" not in board.commands


def test_drag_releases_when_motion_is_interrupted(tmp_path):
    phone, board, screen = calibrated_phone(tmp_path)
    def disconnect(dx, dy):
        if "press" in board.commands: board.links += 1
    board.on_move = disconnect
    with pytest.raises(RuntimeError, match="invalidated"):
        asyncio.run(phone.swipe(.4, .4, .6, .6))
    assert board.commands[-1] == "release" and phone._pos is None


def test_stop_cancels_an_already_queued_command(tmp_path):
    phone, board, screen = calibrated_phone(tmp_path)
    async def scenario():
        entered = asyncio.Event()
        original_send = board.send
        async def send(command):
            if command == "click":
                entered.set()
                await asyncio.Event().wait()
            await original_send(command)
        board.send = send
        task = asyncio.create_task(phone.tap(screen.x, screen.y))
        await entered.wait()
        await phone.reset()
        with pytest.raises(RuntimeError, match="stopped"):
            await task
        assert "click" not in board.commands and board.commands[-1] == "release"
    asyncio.run(scenario())


def test_calibration_background_change_is_not_accepted(tmp_path):
    screen = SimulatedScreen()
    board = SimulatedBoard(screen)
    phone = Phone(board, tmp_path/"calibration.json", screen)
    board.on_move = lambda dx, dy: setattr(screen, "background", 220) if abs(dx) > 12 or dy else None
    with pytest.raises(RuntimeError, match="screen changed"):
        asyncio.run(phone.start_calibration())
    assert not phone._file.exists() and not phone.calibrated


def test_twelfth_relative_move_can_finish_at_the_target(tmp_path):
    screen = SimulatedScreen()
    screen.x = screen.y = .05
    board = SimulatedBoard(screen)
    phone = Phone(board, tmp_path/"calibration.json", screen)
    phone.span, phone._aspect = {"w": 1000, "h": 1600}, screen.aspect
    async def observed(*args, **kwargs):
        return screen.x, screen.y
    phone._find_pointer = observed
    asyncio.run(phone.tap(.95, .95))
    moves = [command for command in board.commands if command.startswith("move ")]
    assert len(moves) == 12
    assert board.commands[-1] == "click"
    assert (screen.x, screen.y) == pytest.approx((.95, .95))


def test_cached_pointer_window_falls_back_to_full_screen(tmp_path):
    phone, _, _ = calibrated_phone(tmp_path)
    searches = []

    async def probe(near, context, direction):
        searches.append(near)
        return (.8, .7) if near is None else None

    phone._probe_pointer = probe
    assert asyncio.run(phone._find_pointer((.2, .2))) == (.8, .7)
    assert searches == [(.2, .2), (.2, .2), None]


@pytest.mark.parametrize("recovered", [None, (.07, .08)])
def test_edge_hint_only_allows_one_inward_escape_not_a_click(tmp_path, recovered):
    screen = SimulatedScreen()
    board = Board()
    phone = Phone(board, tmp_path/'calibration.json', screen)
    calls = 0

    async def probe(near, context, direction):
        nonlocal calls
        calls += 1
        if calls <= 2:
            phone._edge_hint = (.01, .02)
            return None
        return recovered

    phone._probe_pointer = probe
    assert asyncio.run(phone._find_pointer()) == recovered
    assert board.commands == ['move 48 48']
    assert calls <= 4
    assert not phone.calibrated and not phone._file.exists()


def test_icon_hover_escape_requires_a_new_verified_pointer(tmp_path):
    screen = SimulatedScreen()
    board = Board()
    phone = Phone(board, tmp_path/'calibration.json', screen)
    calls = 0

    async def probe(near, context, direction):
        nonlocal calls
        calls += 1
        phone._hover_hint = (.7, .92)
        return None

    phone._probe_pointer = probe
    assert asyncio.run(phone._find_pointer()) is None
    assert board.commands == ['move 0 -96']
    assert calls == 4
    assert phone._pos is None and not phone.calibrated


def test_aim_adapts_to_observed_gain_without_rewriting_calibration(tmp_path):
    phone, board, screen = calibrated_phone(tmp_path)
    saved = phone._file.read_bytes()
    board.gain = 1.2
    asyncio.run(phone.tap(.8, .75))
    assert (screen.x, screen.y) == pytest.approx((.8, .75), abs=.004)
    assert board.commands[-1] == 'click'
    assert phone._file.read_bytes() == saved


def test_real_status_bar_motion_produces_only_an_edge_recovery_hint(tmp_path):
    fixture = os.environ.get('HIDAGENT_EDGE_FIXTURE')
    if not fixture:
        pytest.skip('optional local edge-pointer fixture')
    root = Path(fixture)
    frames = [(root/f'probe-01-{k}.jpg').read_bytes() for k in 'abc']

    class Screen:
        live, frame = True, frames[0]
        async def settled(self, **kwargs): pass
        async def fresh(self, *args): return self.frame

    screen = Screen()
    board = Board()
    moves = iter(frames[1:])

    async def send(command):
        board.commands.append(command)
        screen.frame = next(moves)

    board.send = send
    phone = Phone(board, tmp_path/'calibration.json', screen)
    phone.span = {'w': 1922.9147, 'h': 1270.9152}
    assert asyncio.run(phone._probe_pointer()) is None
    assert phone._edge_hint is not None
    assert all(0 <= coordinate < .04 for coordinate in phone._edge_hint)
    assert board.commands == ['move 12 0', 'move -12 0']
    assert phone._pos is None and not phone._file.exists()
