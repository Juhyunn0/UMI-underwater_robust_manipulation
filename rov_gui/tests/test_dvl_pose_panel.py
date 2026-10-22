#!/usr/bin/env python3
"""test_dvl_pose_panel.py — the trajectory panel's half of --dvl_pose
(control/dvl_pose.py): the TAG FIX switch, the DVL-pose ghost, and the chip.

What is pinned, in the order a failure should be read:

    switch   hidden without a DVL pose, shown with one, placed on row2
             after EST; it reflects MpcStatus.dvp_tag_on and never flips
             itself (a click is a REQUEST); it stays LIVE while engaged and
             while a controller CSV is open — the one switch on the row
             meant for mid-run, where DVL and EST freeze
    face     text and colour per dvp_state; every face (ON / OFF, each
             colour, pressed) reads at 4.5:1 and is not cut, rendered with
             the real app sheet; the tooltip says what the run flies on
    ghost    drawn only with dvp_ok and a pose, through the SAME mirror and
             datum as the imu_dr ghost; its trail, legend swatch and the
             "DVL-tag" readout row; nothing violet without an estimate
    chip     the pose word after the est part, "(shadow)" when the
             controller is not flying it; nothing else on the chip moves
    window   with DR, DVL, EST and TAG FIX all showing, the --mpc window
             still fits the operator's screen; the click reaches
             bus.cmd_dvl_pose_tag and nothing else

None of this is evidence about the real vehicle: every status here is
synthetic, and --dvl_pose is HARDWARE-UNVERIFIED.

    QT_QPA_PLATFORM=offscreen \\
      ~/miniforge3/envs/robust/bin/python rov_gui/tests/test_dvl_pose_panel.py
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rov_gui.state import Conn, MpcStatus                    # noqa: E402


def _harness():
    from rov_gui.tests import test_control as TC
    return TC


def _td():
    """test_dvl's rendered-face helpers (_face, _clipped): one definition of
    "readable" for every state switch on the row."""
    from rov_gui.tests import test_dvl as TD
    return TD


def _panel():
    _harness()._app()
    from rov_gui.widgets.trajectory import TrajectoryWindow
    return TrajectoryWindow()


def _rgb(hex_):
    return tuple(int(hex_[i:i + 2], 16) for i in (1, 3, 5))


#: A DVL pose that is dead-reckoning fine, with the TAG FIX switch ON.
POSE = dict(dvp_mode="control", dvp_tag_on=True, dvp_state="dr", dvp_ok=True)


# ===================================================================== switch
def test_the_tag_fix_button_is_hidden_without_a_dvl_pose_and_shown_with_one():
    p = _panel()
    assert p.btn_tagfix.isHidden()            # built hidden
    p.add_status(MpcStatus())                 # a station without --dvl_pose
    assert p.btn_tagfix.isHidden()
    p.add_status(MpcStatus(dvp_mode="shadow", dvp_state="no_anchor"))
    assert not p.btn_tagfix.isHidden()
    p.add_status(MpcStatus(dvp_mode="control", dvp_state="tag", dvp_ok=True))
    assert not p.btn_tagfix.isHidden()
    p.add_status(MpcStatus())                 # ...and gone with the pose
    assert p.btn_tagfix.isHidden()
    # Row 2, right after EST and before REC: no new row (the window's
    # height budget), and the request/re-sync switches stay together.
    row2 = p.layout().itemAt(2).layout()
    order = [row2.itemAt(i).widget() for i in range(row2.count())]
    i = order.index(p.btn_est)
    # (2026-10-19: --state-noise's NOISE button sits after TAG FIX, hidden without the flag)
    assert order[i + 1] is p.btn_tagfix and order[i + 2] is p.btn_noise and order[i + 3] is p.btn_rec, order
    assert p.btn_tagfix.focusPolicy() == p.btn_est.focusPolicy()
    assert p.btn_tagfix.isCheckable()
    p.close()


def test_the_tag_fix_button_reflects_the_worker_and_never_flips_itself():
    p = _panel()
    asked = []
    p.dvl_pose_tag_requested.connect(asked.append)
    p.add_status(MpcStatus(**dict(POSE, dvp_state="tag")))
    assert p.btn_tagfix.isChecked() and p.btn_tagfix.text() == "TAG FIX ON"
    p.btn_tagfix.click()                      # a REQUEST for OFF...
    assert asked == [False]
    assert p.btn_tagfix.isChecked() and p.btn_tagfix.text() == "TAG FIX ON"
    # ...which the worker refused: the face never moved, and the reason is
    # on the button (the worker writes it to dvp_note)
    p.add_status(MpcStatus(**dict(
        POSE, dvp_state="tag",
        dvp_note="TAG FIX OFF refused: no DVL bottom lock")))
    assert p.btn_tagfix.isChecked() and p.btn_tagfix.text() == "TAG FIX ON"
    assert "TAG FIX OFF refused: no DVL bottom lock" in p.btn_tagfix.toolTip()
    p.btn_tagfix.click()
    assert asked == [False, False]
    # the worker agreed: only NOW does the face say OFF
    p.add_status(MpcStatus(**dict(POSE, dvp_tag_on=False)))
    assert not p.btn_tagfix.isChecked()
    assert p.btn_tagfix.text() == "TAG FIX OFF"
    p.btn_tagfix.click()                      # ON is asked for the same way
    assert asked == [False, False, True]
    assert not p.btn_tagfix.isChecked()
    assert p.btn_tagfix.text() == "TAG FIX OFF"
    p.add_status(MpcStatus(**dict(POSE, dvp_state="tag")))
    assert p.btn_tagfix.isChecked() and p.btn_tagfix.text() == "TAG FIX ON"
    p.close()


def test_a_one_shot_tag_fix_has_its_own_face():
    """Pressed while OFF it is ONE correction (2026-10-18): the face says so
    while it is in progress, and goes back to OFF when the WORKER says."""
    p = _panel()
    p.add_status(MpcStatus(**dict(POSE, dvp_tag_on=False)))
    assert p.btn_tagfix.text() == "TAG FIX OFF"
    p.add_status(MpcStatus(**dict(POSE, dvp_state="tag", dvp_oneshot=True)))
    assert p.btn_tagfix.isChecked() and p.btn_tagfix.text() == "TAG FIX ONCE…"
    assert "ONE correction" in p.btn_tagfix.toolTip()
    p.add_status(MpcStatus(**dict(POSE, dvp_tag_on=False)))
    assert not p.btn_tagfix.isChecked()
    assert p.btn_tagfix.text() == "TAG FIX OFF"
    # (a flag without the switch ON is no one-shot face)
    p.add_status(MpcStatus(**dict(POSE, dvp_tag_on=False, dvp_oneshot=True)))
    assert p.btn_tagfix.text() == "TAG FIX OFF"
    p.close()


def test_the_tag_fix_button_is_live_while_engaged_and_while_a_csv_is_open():
    """The contrast is the point: DVL and EST freeze there (one run = one
    estimator); re-anchoring on the tags is a mid-run act."""
    p = _panel()
    asked = []
    p.dvl_pose_tag_requested.connect(asked.append)
    common = dict(POSE, dvl_state="in_use", dvl_use=True, vel_src="dvl",
                  ekf_state="ready")
    for extra in (dict(engaged=True, mode="pid"), dict(csv_open=True),
                  dict(engaged=True, mode="pid", csv_open=True)):
        p.add_status(MpcStatus(**common, **extra))
        assert not p.btn_dvl.isEnabled() and not p.btn_est.isEnabled(), extra
        assert p.btn_tagfix.isEnabled(), extra
        p.btn_tagfix.click()                  # a disabled button ignores this
    assert asked == [False, False, False]
    p.add_status(MpcStatus(**common))
    assert p.btn_tagfix.isEnabled() and p.btn_dvl.isEnabled()
    p.close()


# ======================================================================= face
#: (dvp_state, dvp_ok) -> the face colour, per the switch's rule: red = no
#: pose to fly on; amber = a pose with a caveat; green = a tag state, or a
#: dead reckoning that reports itself fine; dim = nothing to say yet.
FACES = (("tag", True, "OK"), ("dr", True, "OK"),
         ("dr", False, "WARN"), ("blend", True, "WARN"),
         ("held", True, "WARN"), ("held", False, "WARN"),
         ("no_anchor", False, "WARN"),
         ("lost", False, "FAIL"), ("fault", False, "FAIL"),
         ("tag", False, "TEXT_DIM"), ("", False, "TEXT_DIM"))


def test_the_tag_fix_face_says_the_pose_state():
    from rov_gui import theme

    p = _panel()
    sets = []
    real = p.btn_tagfix.setStyleSheet
    p.btn_tagfix.setStyleSheet = lambda css: (sets.append(css), real(css))
    for st, ok, name in FACES:
        colour = getattr(theme, name)
        for on in (True, False):
            p.add_status(MpcStatus(**dict(POSE, dvp_state=st, dvp_ok=ok,
                                          dvp_tag_on=on)))
            want = p._state_btn_css(colour)
            assert p.btn_tagfix.styleSheet() == want, (st, ok, on, name)
            assert p.btn_tagfix.text() == ("TAG FIX ON" if on
                                           else "TAG FIX OFF"), (st, on)
            assert p.btn_tagfix.isChecked() is on
    # The sheet is only re-set when the colour CHANGES (the _dvl_css rule):
    # once per colour run above, never per 20 Hz status.
    runs = 1 + sum(1 for a, b in zip(FACES, FACES[1:]) if a[2] != b[2])
    assert len(sets) == runs, (len(sets), runs)
    n = len(sets)
    for _ in range(5):
        p.add_status(MpcStatus(**dict(POSE, dvp_state="", dvp_ok=False)))
    assert len(sets) == n
    p.close()


def test_the_tag_fix_tooltip_says_what_the_run_flies_on():
    p = _panel()
    p.add_status(MpcStatus(
        dvp_mode="control", dvp_tag_on=False, dvp_state="dr", dvp_ok=True,
        dvp_err_m=0.042, dvp_err_yaw_deg=1.3, dvp_since_fix_s=12.4,
        dvp_dist_m=0.61, dvp_corr_m=0.015, dvp_bias_dps=0.0123,
        dvp_bias_state="learning", dvp_note="bias from 3 tag fixes"))
    tip = p.btn_tagfix.toolTip()
    for part in ("the controller AND the interlocks fly on this pose",
                 "ON: AprilTag fixes set / correct the pose",
                 "OFF: DVL-only dead reckoning", "WHILE ENGAGED",
                 "HARDWARE-UNVERIFIED", "state: dr", "4.2 cm", "+1.3 deg",
                 "12 s", "0.61 m", "1.5 cm", "+0.012 deg/s (learning)",
                 "note: bias from 3 tag fixes"):
        assert part in tip, (part, tip)
    p.add_status(MpcStatus(dvp_mode="shadow", dvp_state="no_anchor"))
    tip = p.btn_tagfix.toolTip()
    assert ("the controller flies on the tag state; this pose is drawn and "
            "recorded beside it") in tip, tip
    # unknown is "--", never 0 (the theme.fmt rule), and no empty note line
    assert "|pose - tag| --" in tip and "note:" not in tip, tip
    p.close()


def test_every_tag_fix_face_is_readable():
    """The DVL switch's 2026-10-13 lesson (test_dvl
    test_an_in_switch_is_readable_in_every_state) applied from the start:
    rendered with the REAL app sheet, ON (filled with the state colour,
    written in TEXT_ON_FILL), OFF (the state colour on the dark face) and
    pressed, in every colour the switch can take, read at 4.5:1 (WCAG AA)
    and are not cut — "TAG FIX OFF" is the widest label on the row after
    "DVL TRIPPED". Engaged and CSV-open faces are the SAME faces: the
    switch is never frozen."""
    from rov_gui import theme

    TD = _td()
    app = _harness()._app()
    prev = app.styleSheet()
    theme.apply(app)
    p = None
    try:
        p = _panel()
        p.show()
        app.processEvents()
        seen = set()
        for st, ok, name in FACES:
            colour = getattr(theme, name)
            if name in seen:
                continue                      # one state per colour is enough
            seen.add(name)
            # (ON, OFF, and the one-shot's ON — its label the widest)
            for on, once in ((True, False), (False, False), (True, True)):
                for extra in ({}, dict(engaged=True, mode="pid",
                                       csv_open=True)):
                    for down in (False, True):
                        kw = dict(POSE, dvp_state=st, dvp_ok=ok,
                                  dvp_tag_on=on, dvp_oneshot=once, **extra)
                        p.add_status(MpcStatus(**kw))
                        app.processEvents()
                        btn = p.btn_tagfix
                        assert btn.isEnabled(), kw
                        btn.setDown(down)
                        app.processEvents()
                        try:
                            bg, fg, cr = TD._face(btn)
                            assert cr >= 4.5, (kw, down, bg, fg, round(cr, 2))
                            if on:
                                # held down too: :checked follows :pressed in
                                # the switch sheet, so ON keeps its state fill
                                assert bg == _rgb(colour), (kw, down, bg)
                                assert fg == _rgb(theme.TEXT_ON_FILL), (kw, fg)
                            elif down:
                                assert bg == _rgb(theme.ACCENT), (kw, bg)
                                assert fg == _rgb(theme.TEXT_ON_FILL), (kw, fg)
                            else:
                                face = (theme.PANEL if colour == theme.FAIL
                                        else theme.PANEL_HI)
                                assert bg == _rgb(face), (kw, bg)
                                assert fg == _rgb(colour), (kw, fg)
                            assert not TD._clipped(btn), (kw, down, btn.text())
                        finally:
                            btn.setDown(False)
        assert seen == {"OK", "WARN", "FAIL", "TEXT_DIM"}
    finally:
        if p is not None:
            p.close()
        app.setStyleSheet(prev)


# ====================================================================== ghost
def _violet_px(img) -> int:
    """Pixels in the DVL pose's violet and nothing else's: blue well above
    green AND red well above green. The reference blue, the accent and the
    EKF teal have red below green; the DR amber and the DP red have no blue;
    the viridis ramp's purple end is near-black at a quarter alpha."""
    from rov_gui.recorder import _qimage_to_bgr

    bgr = _qimage_to_bgr(img).astype(int)
    b, g, r = bgr[:, :, 0], bgr[:, :, 1], bgr[:, :, 2]
    return int(((b > g + 60) & (r > g + 40)).sum())


def test_the_dvl_pose_ghost_draws_only_with_an_estimate_through_the_datum():
    from rov_gui.qt import QtGui
    from rov_gui.widgets.trajectory import TrajectoryView

    _harness()._app()
    v = TrajectoryView()
    v.resize(600, 420)
    # A non-trivial datum (rotation AND a z offset): an identity one would
    # hide a missing transform on exactly one of the series. Everything
    # stays inside the default 4 m box so the paint check below sees it.
    v.set_datum((0.2, -0.4, -0.9, math.radians(30.0)))
    s = MpcStatus(engaged=True, conn=Conn.ONLINE,
                  p_flu=(0.5, 0.0, -0.8), yaw_flu_deg=0.0,
                  p_dr_flu=(0.5, -0.20, -0.8), yaw_dr_flu_deg=12.0,
                  dr_ok=True, dr_mode="shadow",
                  p_dvp_flu=(0.5, -0.20, -0.8), yaw_dvp_flu_deg=12.0,
                  dvp_ok=True, dvp_mode="shadow", dvp_state="dr",
                  dvp_err_m=0.20)
    v.add_status(s)
    # the SAME mirror and datum as p_dr: identical inputs land identically
    assert v.p_dvp is not None and v.p_dvp == v.p_dr, (v.p_dvp, v.p_dr)
    assert v.yaw_dvp == v.yaw_dr
    assert len(v.trail_dvp) == 1 and v.trail_dvp[-1][1:] == v.trail_dr[-1][1:]
    assert abs(v.p_dvp[2] - (0.8 - 0.9)) < 1e-12, v.p_dvp   # _to_map_z(-z)
    d = math.hypot(v.p_act[0] - v.p_dvp[0], v.p_act[1] - v.p_dvp[1])
    assert abs(d - 0.20) < 1e-9, d
    assert math.dist(v.p_dvp[:2], (0.5, 0.2)) > 0.1, \
        "drawn through the mirror alone — the datum was skipped"
    assert v._readout_dvp_line() == "DVL-tag 20.0 cm"
    names = [lbl for lbl, _pen in v._legend_swatches()]
    assert "DVL" in names and names.count("tag") == 1, names

    # PAINTED, top-down and tilted: the ghost, its trail, its readout row
    for i in range(1, 6):
        v.add_status(MpcStatus(engaged=True, conn=Conn.ONLINE,
                               p_flu=(0.5 + 0.05 * i, 0.0, -0.8),
                               p_dvp_flu=(0.5 + 0.05 * i, -0.2 - 0.02 * i,
                                          -0.8),
                               dvp_ok=True, dvp_mode="shadow",
                               dvp_err_m=0.2))
    assert len(v.trail_dvp) == 6
    img = QtGui.QImage(v.size(), QtGui.QImage.Format.Format_RGB32)
    lit = {}
    for three_d in (False, True):
        v.three_d = three_d
        img.fill(QtGui.QColor("#000000"))
        v.render(img)
        lit[three_d] = _violet_px(img)
        assert lit[three_d] > 50, (three_d, lit)

    # dvp_ok False: NO marker and no new trail sample — a stopped dead
    # reckoning must not sit there looking like one tracking perfectly
    v.add_status(MpcStatus(engaged=True, conn=Conn.ONLINE,
                           p_flu=(1.3, 0.0, -0.8), p_dvp_flu=(1.3, -0.3, -0.8),
                           dvp_ok=False, dvp_mode="shadow", dvp_state="lost",
                           dvp_err_m=0.3))
    assert v.p_dvp is None and len(v.trail_dvp) == 6
    assert v._readout_dvp_line() == ""
    # ...nor with no pose at all
    v.add_status(MpcStatus(engaged=True, conn=Conn.ONLINE,
                           p_flu=(1.3, 0.0, -0.8), p_dvp_flu=None,
                           dvp_ok=True, dvp_mode="shadow"))
    assert v.p_dvp is None
    v.clear()
    assert not v.trail_dvp, "clear() must take the DVL pose trail too"
    assert "DVL" not in [lbl for lbl, _pen in v._legend_swatches()]
    for three_d in (False, True):
        v.three_d = three_d
        img.fill(QtGui.QColor("#000000"))
        v.render(img)
        assert _violet_px(img) == 0, "violet drawn with no estimate"

    # The readout needs BOTH markers and the worker's number
    v.three_d = False
    v.add_status(MpcStatus(engaged=True, conn=Conn.ONLINE,
                           p_flu=(1.0, 0.0, -0.8),
                           p_dvp_flu=(1.0, -0.1, -0.8), dvp_ok=True,
                           dvp_mode="control", dvp_err_m=None))
    assert v.p_dvp is not None and v._readout_dvp_line() == ""
    v.add_status(MpcStatus(engaged=True, conn=Conn.ONLINE, p_flu=None,
                           p_dvp_flu=(1.0, -0.1, -0.8), dvp_ok=True,
                           dvp_mode="control", dvp_err_m=0.1))
    assert v.p_dvp is not None and not v._act_fresh()
    assert v._readout_dvp_line() == "", "no hull drawn, no comparison"

    # Unengaged (no datum): world FLU -> map NED by the mirror alone, as
    # p_dr_flu is
    v.set_datum(None)
    v.add_status(MpcStatus(p_dr_flu=(0.3, 0.4, 0.5), dr_ok=True,
                           dr_mode="shadow", p_dvp_flu=(0.3, 0.4, 0.5),
                           dvp_ok=True, dvp_mode="shadow"))
    assert v.p_dvp == v.p_dr == (0.3, -0.4, -0.5), (v.p_dvp, v.p_dr)


def test_a_hidden_dr_overlay_does_not_hide_the_dvl_pose():
    """No new button: the DVL pose is drawn whenever present — it can be
    the state the vehicle is flown on — and the DR toggle is not its."""
    from rov_gui.widgets.trajectory import TrajectoryWindow

    _harness()._app()
    p = TrajectoryWindow()
    p.add_status(MpcStatus(engaged=True, conn=Conn.ONLINE,
                           p_flu=(0.0, 0.0, -1.0), dr_mode="shadow",
                           dr_ok=True, p_dr_flu=(0.1, 0.0, -1.0),
                           p_dvp_flu=(0.0, 0.1, -1.0), dvp_ok=True,
                           dvp_mode="shadow"))
    p.btn_dr.setChecked(False)
    assert not p.view._dr_visible()
    names = [lbl for lbl, _pen in p.view._legend_swatches()]
    assert "DVL" in names and "DR" not in names and "tag" in names, names
    p.view.resize(500, 320)
    p.view.grab()
    p.close()


# ======================================================================= chip
def test_the_chip_names_what_the_pose_rests_on():
    p = _panel()
    base = dict(engaged=True, mode="pid", conn=Conn.ONLINE, err_xy=0.03,
                ekf_state="ready", ekf_control=True)
    p.add_status(MpcStatus(**base))
    plain = p.chip.text()
    assert "est:EKF" in plain and "pose" not in plain, plain
    for mode, st, ok, since, want in (
            ("control", "tag", True, None, " · pose:TAG"),
            ("control", "blend", True, 0.4, " · pose:TAG+corr"),
            ("control", "dr", True, 12.4, " · pose:DVL 12s"),
            ("control", "dr", True, None, " · pose:DVL"),
            ("control", "held", False, 30.0, " · pose:HELD"),
            ("control", "lost", False, 40.0, " · pose:LOST"),
            ("control", "fault", False, None, " · pose:LOST"),
            ("control", "no_anchor", False, None, " · pose:NO ANCHOR"),
            ("shadow", "dr", True, 12.4, " · pose(shadow):DVL 12s"),
            ("shadow", "tag", True, None, " · pose(shadow):TAG"),
            ("shadow", "no_anchor", False, None,
             " · pose(shadow):NO ANCHOR")):
        p.add_status(MpcStatus(**base, dvp_mode=mode, dvp_state=st,
                               dvp_ok=ok, dvp_since_fix_s=since))
        # APPENDED after the est part; every existing part where it was
        assert p.chip.text() == plain + want, (p.chip.text(), want)
    # not engaged: the chip's own message, no pose word
    p.add_status(MpcStatus(reason="not engaged", **POSE))
    assert p.chip.text() == "not engaged"
    p.close()


# ===================================================================== window
def test_a_window_with_every_row2_switch_showing_still_fits_the_screen():
    """test_control's test_mpc_panel_lives_in_the_grid_and_fits_the_operator_
    screen budget (1366 wide, 1000 tall — the ~6 px height margin is why the
    switch is on row2), with DR, DVL, EST and TAG FIX all VISIBLE in their
    widest faces. And the window carries the click to bus.cmd_dvl_pose_tag —
    and NOT to cmd_tag_enable, the per-feed TAG detection toggle."""
    from rov_gui import theme
    from rov_gui.tests.test_offline import Opts as BaseOpts, _pump
    from rov_gui.window import MainWindow

    class Opts(BaseOpts):
        mpc = True
        nav_config = str(ROOT / "config" / "hw_nav.yaml")
        nav_geometry = None

    app = _harness()._app()
    prev = app.styleSheet()
    theme.apply(app)
    win = MainWindow(Opts())
    try:
        p = win._traj_panel
        assert p is not None
        win.resize(1600, 1000)
        win.show()
        _pump(app, 250)
        sent, tag_enable = [], []
        win.bus.cmd_dvl_pose_tag.connect(sent.append)
        win.bus.cmd_tag_enable.connect(lambda *a: tag_enable.append(a))
        win.bus.mpc_status.emit(MpcStatus(
            dvl_state="tripped", dvl_use=True, ekf_state="ready",
            ekf_control=True, dr_mode="shadow", dvp_mode="control",
            dvp_tag_on=False, dvp_state="lost", dvp_ok=False))
        _pump(app, 150)
        for b in (p.btn_dr, p.btn_dvl, p.btn_est, p.btn_tagfix):
            assert b.isVisible(), b.text()
        assert p.btn_tagfix.text() == "TAG FIX OFF"
        hint = win.minimumSizeHint()
        assert hint.width() <= 1366, f"minimum width {hint.width()} > 1366"
        assert hint.height() <= 1000, f"minimum height {hint.height()} > 1000"
        p.btn_tagfix.click()
        assert sent == [True] and not tag_enable, (sent, tag_enable)
        assert p.btn_tagfix.text() == "TAG FIX OFF"   # until the worker says
    finally:
        win.close()
        app.setStyleSheet(prev)


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"  ok    {fn.__name__}")
        except Exception as e:                                # noqa: BLE001
            failed += 1
            print(f"  FAIL  {fn.__name__}: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
