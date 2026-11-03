#!/usr/bin/env python3
"""
logfilter.py — which lines the MISSION LOG panel SHOWS.

Operator request 2026-10-28: one startup put 86 lines on the panel
(data/20261028/1028_152309/mission_log.txt, up to 15:10:59), and the one
thing wanted from it (the DVL's temperature) could not be found among them;
through this filter 11 of them are drawn, 12 of the session's 150. Most were the session describing its own configuration — gains, policy
paths, button maps, solver probes — and five were paragraph-long [WARN]s that
restate the mode the operator chose, amber on every launch, so that a real
warning arrived into a panel that had already been amber for a minute.

NOTHING IS DROPPED. Every line still goes to stdout and to the run folder's
``mission_log.txt`` (window.py ``_save_mission_log``): the file is the record,
the panel is what needs the operator NOW. This module decides only the second.

A WHITELIST, not a list of things to hide (operator, 2026-10-28: "화면엔 안
띄웠으면"): what the panel needs is short, the station's commentary is long
and grows every week. The price, accepted: a NEW warning is file-only until it
is named here. ``error`` is always shown, so a line that ends or blocks a run
never depends on this list.

Two channels (window.py):

* ``bus.log`` (level, message) — :func:`panel_text`. Shown: errors, and the
  :data:`PANEL_LOG` patterns.
* ``bus.mpc_event`` (message) — :func:`mission_shown`. The run's own lines,
  shown unless :data:`MISSION_QUIET` names them: the session headers that
  ``_event(defer=True)`` says at setup, and estimator state that flips every
  second while a run flies (the trajectory panel already shows HELD / LOST).

Matched on the WORDING, like ``DVP_QUIET_WARN`` (control/workers.py) — a
reworded emitter falls off the panel; test_logfilter.py pins the lines of the
2026-10-28 session that motivated this.
"""
from __future__ import annotations

import re
from pathlib import PurePath

from .state import BANNER_LINE

# workers._log re-tags "ctrl: " as "ctrl[<mode>]: " (the follower flying)
_CTRL = r"^ctrl(?:\[[^\]]*\])?: "

#: bus.log lines the panel shows besides every ``error``.
PANEL_LOG = tuple(re.compile(p) for p in (
    # the arm state (window.py, telemetry), and an ARM press that went
    # nowhere (backends/hardware.py, demo.py: COMMAND ENABLE off)
    r"^ARMED\b", r"^DISARMED\b", r"^(?:demo: )?ARM ignored\b",
    # the vehicle's own WARNING-or-worse STATUSTEXT (backends/hardware.py);
    # a leak is an error and is shown anyway
    r"^vehicle: ",
    # the DVL's temperature — each CHANGE in what it means (backends/dvl.py
    # _say_temperature). The first reading is not: the SENSORS row carries
    # the number, the log only the crossings.
    r"^dvl: HOT ", r"^dvl: temperature back to ",
    r"^dvl: the DVL itself reports HIGH TEMPERATURE",
    r"^dvl: the DVL's high-temperature flag cleared",
    r"^dvl: no temperature ",
    # the DVL switch (operator's, or trust mode's auto-IN)
    _CTRL + r"DVL (?:IN|OUT)\b",
    # the policy loaded (once; "DP policy ready in" is its duplicate)
    r"^policy: READY ",
    # an operator action refused — "engage refused: ...", "START TRAJ
    # refused — ...", "DVL switch refused ...", "mode change refused ...".
    # The word must be in the HEAD of the message: "dvl_pose: (disengaged) a
    # START now would be refused" is a forecast, not a refusal, and the
    # startup "... — rl_pwm mode will refuse" lines are configuration.
    _CTRL + r"[^:—(]*\brefused\b",
    # a LAUNCH export on the flight-ban list (2026-11-02, review F1). The ban
    # is enforced at the pick only (MpcWorker.set_rl_policy), so a
    # hw_mpc.yaml that names a banned export still builds, and this WARN
    # (workers._rl_launch_prov: "RL policy <LOW> <name> (hw_mpc.yaml
    # <low>.policy_dir, id <id>) — WARN: no-fly — <reason>"; the id may read
    # "? — not hashed: …") is the panel's word of it. Matched from the line's
    # HEAD through the no-fly tail: the same line without that tail is
    # configuration (file only), and so is a pick's own line ("RL policy
    # <LOW>: <name> (id …)", whose only WARN is the cap; a banned pick is
    # refused — the pattern above).
    _CTRL + r"RL policy \S+ .+ \(hw_mpc\.yaml \S+\.policy_dir, id .*\) — "
            r"WARN: no-fly — ",
    # a recording stopped (REC UI, the feeds, REC NAV) — shown as its folder,
    # see panel_text
    r"^[^:]*recording saved: ",
))

_SAVED = re.compile(r"^[^:]*recording saved: (\S+)")

#: bus.mpc_event lines the panel does NOT show (they still reach the file).
MISSION_QUIET = (
    # the SESSION, not a run: workers._event(defer=True), said at setup
    "ready: ", "POSE: ", "ESTIMATOR ", "imu_dr ", "object_nav ON",
    # station_bridge ON (setup) and its rung / fix-back lines (every dropout)
    "station_bridge",
    # estimator state, said again on every flip; and the mission copy of an
    # error the log channel already shows in red
    "dvl_pose", "dvl: ", "ekf: ",
    # another client asked for the parameter list (rl_pwm's refusal says it
    # when it matters)
    BANNER_LINE,
    # a profile pick — the LOW row shows it
    "MPC_Tuned profile",
    # an RL export pick (2026-11-01, MpcWorker.set_rl_policy: "RL policy
    # <LOW> ← <name> (panel pick, ...)") — the LOW row's policy widgets show
    # it; a REFUSED pick is a bus.log WARN whose head says "refused", which
    # PANEL_LOG shows
    "RL policy ",
    # a pwm transport cap change (2026-11-02, MpcWorker.set_rl_pwm_cap:
    # "RL_PWM cap <old> → <new> (panel; …)") — the LOW row's cap ▾ shows it.
    # A REFUSED change is the bus.log WARN "ctrl: RL_PWM cap refused: …",
    # shown by PANEL_LOG's refusal pattern; a pwm LOW dropped by a change
    # ("ctrl: <LOW> unavailable at cap …") is file only — the LOW row shows
    # its ERROR. NOT "RL_PWM " alone: "RL_PWM gate lost — disengaged" is a
    # run event the panel keeps.
    "RL_PWM cap ",
)


def panel_text(level: str, message: str) -> str | None:
    """What the panel shows for one ``bus.log`` line, or None: file only."""
    if level == "debug":
        return None
    if level != "error" and not any(p.search(message) for p in PANEL_LOG):
        return None
    m = _SAVED.match(message)
    if m and m.group(1) != "None":
        # One stop is several files in one run folder (the depth feed's mp4,
        # the nav folder, ...): the panel says the folder once — the payload
        # skips an identical folded line — and the file keeps every path.
        return f"recording saved: {PurePath(m.group(1)).parent}/"
    return message


def mission_shown(message: str) -> bool:
    """Whether a ``bus.mpc_event`` line goes on the panel."""
    return not message.startswith(MISSION_QUIET)
