import asyncio
import io
from types import SimpleNamespace

import pytest
from PIL import Image, ImageDraw

from hidagent.calibration import Axis, Calibration, fit_axis
from hidagent.device import DeviceError
from hidagent.phone import Phone
from hidagent.screen import Screen
from hidagent.vision import pointer


def test_fit_measures_scale_and_offset_without_clamping_the_corrected_edge():
    axis = fit_axis([(200, .08), (700, .33), (1400, .68), (1100, .53)])
    assert axis.span == pytest.approx(2000)
    assert axis.offset == pytest.approx(-.02)
    assert axis.units(.995) == 2030  # clipping at span would miss the bottom
    assert Axis(1000, 0).units(2) == 1000  # only the requested fraction is clamped


def test_bottom_edge_verification_retries_at_the_same_height_clear_of_the_home_indicator():
    class Fake(Calibration):
        async def send(self, command):
            pass

        async def move(self, x, y):
            self.pos = (x, y)

        async def observe(self, near=None, span=None):
            if near is None:  # measurement: a perfect 1000 x 2000 mapping
                return self.pos[0] / 1000, self.pos[1] / 2000
            if tuple(near) == (0.5, 0.99):  # the home indicator hides the pointer here
                return None
            if tuple(near) == (0.82, 0.99):  # the same-height retry is observable
                return (0.82, 0.99)
            return tuple(near)

    screen = SimpleNamespace(live=True, geometry=(752, 1080))
    result = asyncio.run(Fake(SimpleNamespace(), screen, (1000, 2000)).run())
    bottom = result["checks"][2]
    assert bottom["target"] == [0.82, 0.99]  # height kept, moved away from the indicator
    assert bottom["observed"] == [0.82, 0.99]


@pytest.mark.parametrize('samples', [
    [(200, .2), (400, .4)],
    [(200, .5), (400, .5), (600, .5)],
    [(200, .6), (400, .4), (600, .2)],
    [(200, .1), (400, .7), (600, .4)],
])
def test_unobservable_or_inconsistent_fit_is_rejected(samples):
    with pytest.raises(ValueError):
        fit_axis(samples)


def test_screen_identity_and_geometry_are_observed_not_guessed_from_aspect():
    screen = Screen()
    frame = io.BytesIO()
    Image.new('RGB', (752, 1080)).save(frame, 'JPEG')
    screen.frame = frame.getvalue()
    assert screen.geometry == (752, 1080)
    screen._source_line('connection request from Example (iPad16,8) with deviceID = test-device')
    assert screen.source['model'] == 'iPad16,8'
    assert screen.source['family'] == 'iPad'
    assert screen.source_key != 'test-device'
    first = screen.source_epoch
    screen._source_line('connection request from Example (iPad16,8) with deviceID = test-device')
    assert screen.source_epoch == first + 1


def test_color_difference_can_locate_a_pointer_with_the_same_brightness_as_its_background():
    def image(x):
        data = io.BytesIO()
        frame = Image.new('RGB', (400, 600), (100, 0, 0))
        ImageDraw.Draw(frame).ellipse((x-8, 192, x+8, 208), fill=(30, 30, 30))
        frame.save(data, 'JPEG', quality=95)
        return data.getvalue()

    a, b = image(200), image(212)
    assert pointer(a, b, (.5, 1/3), 12/400) is None
    found = pointer(a, b, (.5, 1/3), 12/400, color=True)
    assert found[0] == pytest.approx(.5, abs=.008)
    assert found[1] == pytest.approx(1/3, abs=.008)


@pytest.fixture
def phone(tmp_path):
    calls = []

    async def send(command):
        calls.append(command)

    device = SimpleNamespace(links=1, connected=True, send=send)
    screen = SimpleNamespace(live=True, geometry=(752, 1080), source_epoch=1, source_key='test', source={})
    p = Phone(device, tmp_path/'calibration.json', screen)
    p._baseline = {'geometry': [752, 1080], 'axes': [{'span': 1000, 'offset': 0},
                                                   {'span': 2000, 'offset': -.02}], 'checks': []}
    p.span = {'w': 1000, 'h': 2000}
    p._bound = p.signature()
    p._pos, p._link, p._moves = (300, 300), 1, 100
    return p, calls


def test_saved_calibration_is_not_proof_about_a_new_device_session(phone):
    p, _ = phone
    p._save()
    assert not Phone(p.device, p._file, p.screen).calibrated


@pytest.mark.parametrize('change', ['rotation', 'board', 'mirror'])
def test_device_or_orientation_change_invalidates_before_movement(phone, change):
    p, calls = phone
    if change == 'rotation':
        p.screen.geometry = (1080, 752)
    elif change == 'board':
        p.device.links += 1
    else:
        p.screen.source_epoch += 1
    assert not p.calibrated
    with pytest.raises(RuntimeError, match='not calibrated'):
        asyncio.run(p.point(.5, .5))
    assert calls == []


def test_normal_positioning_uses_calibration_motion_and_does_not_periodically_hit_a_corner(phone):
    p, calls = phone
    asyncio.run(p.point(.5, .5))
    assert calls == ['move 200 0', 'move 0 740']


def test_home_establishes_the_bottom_boundary_before_pressing(phone):
    p, calls = phone
    p._home_link, p._home_supported = 1, False
    asyncio.run(p.go_home())
    press = calls.index('press')
    assert calls[press-1] == 'move 0 400'
    assert calls[press+1] == 'move 0 -800'
    assert calls[-1] == 'release'
    assert not p._dragging


def test_rotation_during_a_gesture_releases_without_more_coordinates(phone):
    p, calls = phone
    p._home_link, p._home_supported = 1, False

    async def send(command):
        calls.append(command)
        if command == 'press':
            p.screen.geometry = (1080, 752)

    p.device.send = send
    with pytest.raises(RuntimeError, match='stale'):
        asyncio.run(p.go_home())
    assert calls[-2:] == ['press', 'release']


def test_system_home_key_does_not_depend_on_calibration_or_pointer_height(phone):
    p, calls = phone
    p._bound = None
    asyncio.run(p.go_home())
    assert calls == ['homescreen']


def test_unknown_home_command_falls_back_but_link_errors_are_not_hidden(phone):
    p, calls = phone

    async def send(command):
        calls.append(command)
        if command == 'homescreen':
            raise DeviceError('err unknown command')

    p.device.send = send
    asyncio.run(p.go_home())
    assert 'press' in calls
    p.device.links = 2

    async def lost(_):
        raise DeviceError('board disconnected')

    p.device.send = lost
    with pytest.raises(DeviceError, match='board disconnected'):
        asyncio.run(p.go_home())


def test_failed_recalibration_cannot_leave_an_old_ready_flag(phone, monkeypatch):
    p, calls = phone

    class Failure:
        def __init__(self, *args): pass
        async def run(self): raise RuntimeError('unconfirmed edge')

    monkeypatch.setattr('hidagent.phone.Calibration', Failure)
    assert not asyncio.run(p.start_calibration())
    assert not p.calibrated
    assert p.baseline_phase == 'failed'
    assert calls[-1] == 'release'
    assert not p._file.exists()


def test_stop_does_not_move_the_pointer_to_a_system_corner(phone):
    p, calls = phone
    asyncio.run(p.reset())
    assert calls == ['release']


def test_calibration_can_be_stopped_without_becoming_ready(phone, monkeypatch):
    p, calls = phone

    class Pending:
        def __init__(self, *args): pass
        async def run(self): await asyncio.Event().wait()

    monkeypatch.setattr('hidagent.phone.Calibration', Pending)

    async def run():
        task = asyncio.create_task(p.start_calibration())
        await asyncio.sleep(0)
        p.cancel_calibration()
        assert not await task

    asyncio.run(run())
    assert p.baseline_phase == 'cancelled'
    assert not p.calibrated
    assert calls[-1] == 'release'


def test_automatic_baseline_retries_on_a_new_connection_not_every_poll(tmp_path, monkeypatch):
    import importlib

    monkeypatch.setenv('HIDAGENT_DATA', str(tmp_path))
    module = importlib.import_module('hidagent.app')
    attempts = []
    p = SimpleNamespace(calibrated=False, session=1, calibration_paused=False)
    p.signature = lambda: p.session

    async def calibrate():
        attempts.append(p.session)
        return False

    p.start_calibration = calibrate
    ticks = 0

    class Finished(Exception):
        pass

    async def tick(_):
        nonlocal ticks
        ticks += 1
        if ticks == 4:
            p.session = 2
        if ticks == 6:
            raise Finished

    monkeypatch.setattr(module, 'phone', p)
    monkeypatch.setattr(module, 'screen', SimpleNamespace(live=True))
    monkeypatch.setattr(module, 'device', SimpleNamespace(connected=True))
    monkeypatch.setattr(module, 'agent', SimpleNamespace(running=False))
    monkeypatch.setattr(module, 'asyncio', SimpleNamespace(sleep=tick))
    monkeypatch.setattr(module, 'manual_lock', asyncio.Lock())
    with pytest.raises(Finished):
        asyncio.run(module.watch_baseline())
    assert attempts == [1, 2]


def test_explicit_stop_remains_paused_across_a_tcp_reconnect_until_the_view_changes(phone):
    p, _ = phone
    p.cancel_calibration()
    assert p.calibration_paused
    p.device.links += 1
    assert p.calibration_paused
    p.screen.source_epoch += 1
    assert not p.calibration_paused
