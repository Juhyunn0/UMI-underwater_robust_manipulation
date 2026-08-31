#!/usr/bin/env python3
"""
test_gantry_panel_settings.py — ~/.umi_gui_state.json persistence, no hardware.

    QT_QPA_PLATFORM=offscreen ~/miniforge3/envs/robust/bin/python src/tests/test_gantry_panel_settings.py
    QT_QPA_PLATFORM=offscreen ~/miniforge3/envs/robust/bin/python -m pytest src/tests/test_gantry_panel_settings.py -v   # if pytest is around

Regression for the 2026-08-23 bug: `_gp_save_section` REPLACED the whole section
and `_on_tab_changed` passed a bare `{"active_tab": idx}`, so a tab switch erased
`axis_sign` and `home_position_mm`. That pair is the ONLY link between the mm the
operator reads on the panel and the mm the SDK is commanded in:

    user_mm_to_abs_mm(mm, axis) = mm * axis_sign[axis] + home_offset_abs[axis]

Losing it does not fail loudly — it silently shifts every commanded target by the
home offset (one recorded run: X = -1612.43 mm,
data/20260528/20260528_212728_recording/run_metadata.json) and mirrors the axis
when the sign was -1. It fires on the NEXT launch, because the section is read at
construction (gantry_panel.py:2259, :2312) and the clobber happens on a tab
switch after that.

Qt-free by construction: `_GP_SETTINGS_PATH` is a module global, so the whole
read/modify/write path runs without building a GantryPanel. Importing the module
still pulls in PyQt5 — hence QT_QPA_PLATFORM=offscreen.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gantry_panel                                              # noqa: E402


# The section as a well-used panel leaves it. axis_sign and home_position_mm are
# the load-bearing pair; the rest are here to prove nothing else is dropped.
FULL_SECTION = {
    "active_tab": 3,
    "axis_sign": {"X": -1, "Y": 1, "Z": -1},
    "home_position_mm": {"X": -1612.43, "Y": -358.22, "Z": 0.0},
    "map_fit_mode": "pool",
    "camera_mode": "fisheye",
    "camera_detector_overlay": True,
    "left_splitter_sizes": [420, 980],
    "camera": {"device": 0, "resolution": "1280x720", "fps": 30},
}


class _settings_file:
    """Point gantry_panel._GP_SETTINGS_PATH at a throwaway file.

    A context manager rather than pytest's monkeypatch so the file runs as a
    plain script too — the house style for tests in this repo.
    """

    def __init__(self, initial=None):
        self._initial = initial
        self._saved = None
        self._tmp = None

    def __enter__(self) -> Path:
        self._tmp = tempfile.TemporaryDirectory()
        path = Path(self._tmp.name) / "umi_gui_state.json"
        if self._initial is not None:
            path.write_text(self._initial if isinstance(self._initial, str)
                            else json.dumps(self._initial, indent=2))
        self._saved = gantry_panel._GP_SETTINGS_PATH
        gantry_panel._GP_SETTINGS_PATH = path
        return path

    def __exit__(self, *exc) -> None:
        gantry_panel._GP_SETTINGS_PATH = self._saved
        self._tmp.cleanup()


def test_one_key_payload_preserves_the_rest():
    """THE regression: a bare {"active_tab": n} must not erase the section."""
    with _settings_file({"gantry_panel": dict(FULL_SECTION)}):
        gantry_panel._gp_save_section("gantry_panel", {"active_tab": 0})
        got = gantry_panel._gp_load_settings()["gantry_panel"]

    assert got["active_tab"] == 0, "the write did not land"
    assert got["axis_sign"] == FULL_SECTION["axis_sign"]
    assert got["home_position_mm"] == FULL_SECTION["home_position_mm"]
    assert set(got) == set(FULL_SECTION), f"keys dropped: {set(FULL_SECTION) - set(got)}"


def test_merge_is_one_level_deep():
    """A nested dict is replaced wholesale, not deep-merged. Every call site
    that writes axis_sign / home_position_mm / camera rebuilds the whole nested
    dict first (gantry_panel.py:2890, :4458, :5423), so a deep merge would keep
    stale per-axis entries alive forever."""
    with _settings_file({"gantry_panel": dict(FULL_SECTION)}):
        gantry_panel._gp_save_section(
            "gantry_panel", {"axis_sign": {"X": 1, "Y": 1, "Z": 1}})
        got = gantry_panel._gp_load_settings()["gantry_panel"]

    assert got["axis_sign"] == {"X": 1, "Y": 1, "Z": 1}
    assert got["home_position_mm"] == FULL_SECTION["home_position_mm"]


def test_sibling_sections_are_untouched():
    """~/.umi_gui_state.json is shared with gui_launcher.py and
    calibrate_fisheye.py — a gantry write must not disturb theirs."""
    with _settings_file({
        "gantry_panel": {"active_tab": 1},
        "gantry_runner": {"speed_mm_s": 20.0, "gantry_ip": "192.168.0.30"},
        "calibrate_fisheye": {"camera": {"device": 0}},
    }):
        gantry_panel._gp_save_section("gantry_panel", {"active_tab": 4})
        data = gantry_panel._gp_load_settings()

    assert data["gantry_runner"] == {"speed_mm_s": 20.0, "gantry_ip": "192.168.0.30"}
    assert data["calibrate_fisheye"] == {"camera": {"device": 0}}
    assert data["gantry_panel"]["active_tab"] == 4


def test_missing_file_creates_the_section():
    with _settings_file() as path:
        assert not path.exists()
        gantry_panel._gp_save_section("gantry_panel", {"active_tab": 2})
        data = gantry_panel._gp_load_settings()

    assert data == {"gantry_panel": {"active_tab": 2}}


def test_corrupt_file_does_not_raise():
    """_gp_load_settings swallows a truncated/foreign file (the file is written
    non-atomically, so a crash mid-write leaves exactly this). The save path
    must not then blow up on a non-dict top level."""
    for corrupt in ("{not json", "[1, 2, 3]", "", "null", '"a string"'):
        with _settings_file(corrupt):
            gantry_panel._gp_save_section("gantry_panel", {"active_tab": 1})
            got = gantry_panel._gp_load_settings()
        assert got["gantry_panel"]["active_tab"] == 1, f"failed on {corrupt!r}"


def test_non_dict_section_is_replaced():
    with _settings_file({"gantry_panel": "somehow a string"}):
        gantry_panel._gp_save_section("gantry_panel", {"active_tab": 1})
        got = gantry_panel._gp_load_settings()

    assert got["gantry_panel"] == {"active_tab": 1}


def main() -> int:
    tests = [
        test_one_key_payload_preserves_the_rest,
        test_merge_is_one_level_deep,
        test_sibling_sections_are_untouched,
        test_missing_file_creates_the_section,
        test_corrupt_file_does_not_raise,
        test_non_dict_section_is_replaced,
    ]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"  ok    {fn.__name__}")
        except Exception as e:                                   # noqa: BLE001
            failed += 1
            print(f"  FAIL  {fn.__name__}: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
