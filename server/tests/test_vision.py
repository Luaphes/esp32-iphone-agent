import io
import os
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from hidagent.vision import changed, evidence_sheet, hover_hint, pointer, scene_changed

W, H = 498, 1080  # a mirrored iPhone frame


def test_evidence_sheet_keeps_history_and_current_in_reading_order():
    def frame(color):
        buffer = io.BytesIO()
        Image.new("RGB", (1200, 800), color).save(buffer, "JPEG")
        return buffer.getvalue()

    sheet = Image.open(io.BytesIO(evidence_sheet([
        ("HISTORICAL 1: before edit", frame("red")),
        ("HISTORICAL 2: before type", frame("green")),
        ("CURRENT: final screen", frame("blue")),
    ])))
    assert sheet.size == (2000, 1406)
    assert sheet.getpixel((500, 350)) == pytest.approx((254, 0, 0), abs=3)
    assert sheet.getpixel((1500, 350)) == pytest.approx((0, 128, 0), abs=3)
    assert sheet.getpixel((500, 1100)) == pytest.approx((0, 0, 254), abs=3)


def screen(pointer_at=None, button=False) -> bytes:
    """A synthetic home screen: gradient background, optional button and pointer dot."""
    img = Image.new("RGB", (W, H))
    draw = ImageDraw.Draw(img)
    for y in range(H):
        draw.line([(0, y), (W, y)], fill=(40, 60 + y // 10, 120))
    if button:
        draw.rectangle([150, 400, 350, 600], fill=(250, 250, 250))
    if pointer_at:
        x, y = pointer_at
        draw.ellipse([x - 12, y - 12, x + 12, y + 12], fill=(200, 200, 200))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=75)
    return buf.getvalue()


def test_pointer_move_is_not_a_screen_change():
    assert not changed(screen((260, 856)), screen((272, 856)), ignore=((260 / W, 856 / H), (272 / W, 856 / H)))


def test_new_content_is_a_screen_change():
    assert changed(screen((260, 856)), screen((260, 856), button=True))


def test_missing_frame_counts_as_changed():
    assert changed(None, screen())


def test_pointer_is_found_from_a_nudge():
    shift = 12 / W
    found = pointer(screen((262, 856)), screen((274, 856)), near=(0.5, 0.8), shift=shift)
    assert found is not None
    assert found[0] * W == pytest.approx(262, abs=3)
    assert found[1] * H == pytest.approx(856, abs=3)


def test_pointer_not_found_when_nothing_moved():
    frame = screen((262, 856))
    assert pointer(frame, frame, near=(0.5, 0.8), shift=12 / W) is None


@pytest.mark.parametrize('delta', [-12, 12])
def test_arrow_uses_tip_as_hotspot_in_both_probe_directions(delta):
    def arrow(x):
        im = Image.new('RGB', (W, H), (40, 60, 100))
        ImageDraw.Draw(im).polygon([(x, 400), (x+17, 417), (x+9, 417), (x, 423)],
                                   fill=(240, 240, 240))
        out = io.BytesIO()
        im.save(out, 'JPEG', quality=90)
        return out.getvalue()

    a, b = arrow(260), arrow(260+delta)
    found = pointer(a, b, (.5, .4), delta/W, restored=a)
    assert found is not None
    assert (found[0]*W, found[1]*H) == pytest.approx((260, 400), abs=2)


def test_icon_hover_hint_requires_local_reversible_motion():
    def hovered(x):
        im = Image.new('RGB', (W, H), (30, 50, 70))
        ImageDraw.Draw(im).rectangle((x, 900, x+70, 970), fill='white')
        out = io.BytesIO()
        im.save(out, 'JPEG', quality=90)
        return out.getvalue()

    a, b = hovered(350), hovered(356)
    hint = hover_hint(a, b, a)
    assert hint is not None
    assert hint == pytest.approx((388/W, 935/H), abs=.01)
    assert hover_hint(a, b, b) is None
    assert hover_hint(a, a, a) is None
    assert hover_hint(a, screen(button=True), a) is None


def test_real_arrow_fixture_reports_tip_instead_of_centroid():
    fixture = os.environ.get('HIDAGENT_ARROW_FIXTURE')
    if not fixture:
        pytest.skip('optional local pointer fixture')
    root = Path(fixture)
    a, b, c = [(root/f'probe-11-{k}.jpg').read_bytes() for k in 'abc']
    found = pointer(a, b, (.5, .5), 12/1922.9147, restored=c)
    assert found is not None
    # Original screen captures: arrow tip near (770, 530), while the old
    # difference centroid was near (777, 541).
    assert (found[0]*1554, found[1]*1080) == pytest.approx((770, 530), abs=3)


def checkmark_screen(checked: bool, pointer_at=(100, 500)) -> bytes:
    img = Image.open(io.BytesIO(screen(pointer_at))).convert("RGB")
    if checked:
        ImageDraw.Draw(img).line([(430, 495), (438, 505), (452, 485)], fill=(255, 149, 0), width=4)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=75)
    return buf.getvalue()


def test_small_checkmark_counts_as_a_change():
    assert changed(checkmark_screen(False), checkmark_screen(True))


def test_masked_pointer_move_is_not_a_change():
    before, after = checkmark_screen(False, (100, 500)), checkmark_screen(False, (300, 500))
    assert not changed(before, after, ignore=((100 / W, 500 / H), (300 / W, 500 / H)))


def test_pointer_far_from_the_target_is_found_with_a_full_screen_search():
    before, after = screen((354, 136)), screen((366, 136))
    assert pointer(before, after, near=(0.9, 0.1), shift=12 / W) is None  # outside the window near the target
    found = pointer(before, after, near=(0.5, 0.5), shift=12 / W, reach=(0.5, 0.5))
    assert found[0] * W == pytest.approx(354, abs=3) and found[1] * H == pytest.approx(136, abs=3)


def test_top_edge_pointer_is_not_excluded_as_a_clock():
    a, b = screen((250, 20)), screen((262, 20))
    found = pointer(a, b, (.5, .02), 12/W, restored=a)
    assert found is not None and found[1]*H == pytest.approx(20, abs=3)


def test_compact_blink_is_not_pointer_motion():
    im = Image.open(io.BytesIO(screen())).convert("RGB")
    ImageDraw.Draw(im).rectangle((200, 400, 220, 420), fill=(240, 240, 240))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=75)
    assert pointer(screen(), buf.getvalue(), (.5, .5), .025, reach=(.5, .5), restored=screen()) is None


def test_pointer_probe_requires_return_and_unchanged_background():
    a, b = screen((260, 500)), screen((272, 500))
    assert pointer(a, b, (.5, .5), 12/W, restored=a)
    assert pointer(a, b, (.5, .5), 12/W, restored=screen((290, 500))) is None
    assert pointer(a, screen((272, 500), button=True), (.5, .5), 12/W, restored=a) is None


def test_large_cursor_is_masked_before_coarse_progress_pass():
    def large(x):
        im = Image.new("RGB", (W, H), "black")
        ImageDraw.Draw(im).ellipse((x-22, 400-22, x+22, 400+22), fill="white")
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=85)
        return buf.getvalue()
    assert not changed(large(100), large(380), ignore=((100/W, 400/H), (380/W, 400/H)))


def test_clock_change_is_masked_before_coarse_progress_pass():
    im = Image.open(io.BytesIO(screen())).convert("RGB")
    ImageDraw.Draw(im).rectangle((0, 0, W, 35), fill="white")
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=75)
    assert not changed(screen(), buf.getvalue())


def test_remote_second_hand_does_not_hide_verified_pointer():
    def animated(cursor_x, tick):
        im = Image.open(io.BytesIO(screen((cursor_x, 180)))).convert("RGB")
        ImageDraw.Draw(im).line((380, 650, 420-tick*8, 710), fill="white", width=3)
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=85)
        return buf.getvalue()
    a, b, c = animated(100, 0), animated(112, 1), animated(100, 2)
    assert changed(a, c)  # task progress still uses the sensitive detector
    assert not scene_changed(a, c)
    found = pointer(a, b, (.5, .5), 12/W, reach=(.5, .5), restored=c)
    assert found is not None
    assert (found[0]*W, found[1]*H) == pytest.approx((100, 180), abs=3)


def test_large_scene_change_still_invalidates_local_pointer_motion():
    a, b = screen((100, 180)), screen((112, 180), button=True)
    assert scene_changed(a, b, ignore=((100/W, 180/H), (112/W, 180/H)))
    assert pointer(a, b, (.5, .5), 12/W, reach=(.5, .5), restored=a) is None


def test_remote_icon_update_is_allowed_but_target_change_is_rejected():
    def icon(x, y):
        im = Image.open(io.BytesIO(screen((100, 180)))).convert('RGB')
        ImageDraw.Draw(im).rectangle((x, y, x+40, y+40), fill='white')
        out = io.BytesIO()
        im.save(out, 'JPEG', quality=85)
        return out.getvalue()

    baseline = screen((100, 180))
    remote = icon(400, 950)
    assert scene_changed(baseline, remote)
    assert not scene_changed(baseline, remote, focus=((.2, .2),))
    assert scene_changed(baseline, icon(100, 240), focus=((.2, .2),))
    assert scene_changed(baseline, screen((100, 180), button=True), focus=((.8, .8),))


def test_date_hover_cannot_become_a_pointer_anchor():
    def hover(x, highlighted):
        im = Image.new("RGB", (1554, 1080), (20, 30, 40))
        draw = ImageDraw.Draw(im)
        draw.ellipse((x-12, -7, x+12, 17), fill=(150, 150, 150))
        if highlighted:
            draw.rounded_rectangle((12, 2, 210, 35), radius=8, outline=(200, 200, 200), width=2)
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=85)
        return buf.getvalue()
    a, b, c = hover(5, False), hover(17, True), hover(5, True)
    assert pointer(a, b, (.5, .5), 12/1554, reach=(.5, .5), restored=c) is None


@pytest.mark.skipif(not os.environ.get("HIDAGENT_POINTER_FIXTURE"), reason="optional local iPad fixture")
def test_local_ipad_fixture_separates_pointer_from_widget_clock():
    # Personal screenshots remain in the operator's evidence directory, never the repository.
    directory = Path(os.environ["HIDAGENT_POINTER_FIXTURE"])
    frames = [(directory/name).read_bytes() for name in ("before.jpg", "001.jpg", "002.jpg", "003.jpg", "004.jpg")]
    cases = [(frames[0], frames[1], frames[2], (.5, .5), (.5, .5)),
             (frames[2], frames[3], frames[4], (.004, .044), (.12, .08))]
    for a, b, c, near, reach in cases:
        found = pointer(a, b, near, .012, reach=reach, restored=c)
        assert found is not None
        assert (found[0]*1554, found[1]*1080) == pytest.approx((5, 47), abs=2)
