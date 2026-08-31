#!/usr/bin/env python3
"""
theme.py — dark palette, stylesheet, and the connection-state colour language.

Written by hand rather than pulling in QDarkStyleSheet: the whole point of the
colour scheme here is the *state* language (below), and a general-purpose dark
theme has nothing to say about that. It is also one less dependency on a topside
laptop that has to work at sea.

Subsea-specific choices
-----------------------
* The background is near-black with a blue cast, not grey. A control station is
  usually run in a darkened cabin or under a hood, and a light UI next to a dark
  video feed destroys the operator's dark adaptation — you lose the ability to
  read a murky picture, which is the picture you actually get.
* Video panels are pure black behind the image so the feed's own black level is
  the darkest thing on screen.
* Exactly one saturated colour is reserved per state, and nothing else in the UI
  is allowed to use them. Amber and red mean something; if a decorative border
  is also amber, they stop meaning it.

State language, used everywhere
-------------------------------
    ONLINE      green    fresh data, within spec
    DEGRADED    amber    data is arriving but wrong (drops, low rate, errors)
    STALE       amber    was online, has gone quiet — the watchdog's verdict
    CONNECTING  blue     opening / waiting for first data
    OFFLINE     grey     never seen, or deliberately not in use
    FAULT       red      the source reported a failure

Grey for OFFLINE and red for FAULT is a deliberate split. "This module is not
plugged in" and "this module is broken" need different reactions, and a
dashboard that paints both red trains the pilot to ignore red.
"""

from __future__ import annotations

from .state import Conn

# --------------------------------------------------------------------- colours
BG = "#0b0f14"          # window
PANEL = "#121820"       # panel body
PANEL_HI = "#18202b"    # panel header / raised row
BORDER = "#22303f"
BORDER_HI = "#2e4356"
TEXT = "#c7d5e2"
TEXT_DIM = "#7b8ea1"
TEXT_FAINT = "#4d5f70"
ACCENT = "#38bdf8"      # interactive / selection
VIDEO_BG = "#000000"

OK = "#22c55e"
WARN = "#f59e0b"
FAIL = "#ef4444"
IDLE = "#64748b"
BUSY = "#38bdf8"

STATE_COLOR = {
    Conn.ONLINE: OK,
    Conn.DEGRADED: WARN,
    Conn.STALE: WARN,
    Conn.CONNECTING: BUSY,
    Conn.OFFLINE: IDLE,
    Conn.FAULT: FAIL,
}

STATE_TEXT = {
    Conn.ONLINE: "ONLINE",
    Conn.DEGRADED: "DEGRADED",
    Conn.STALE: "STALE",
    Conn.CONNECTING: "CONNECTING",
    Conn.OFFLINE: "OFFLINE",
    Conn.FAULT: "FAULT",
}

# Monospace for every number that changes. Proportional digits make a readout
# jitter sideways as it updates, which reads as noise on a glance-only display.
MONO = "DejaVu Sans Mono, Consolas, Menlo, monospace"
SANS = "Inter, Roboto, DejaVu Sans, Segoe UI, sans-serif"


def state_color(conn: Conn) -> str:
    return STATE_COLOR.get(conn, IDLE)


def state_text(conn: Conn) -> str:
    return STATE_TEXT.get(conn, "?")


def fmt(value, spec: str = ".1f", unit: str = "", dash: str = "--") -> str:
    """Format a reading, or ``--`` when it is None/NaN.

    The one place the "unknown is not zero" rule from state.py is enforced for
    display. Kept here so every panel formats missing data identically.
    """
    if value is None:
        return dash
    try:
        if value != value:            # NaN
            return dash
        text = format(float(value), spec)
    except (TypeError, ValueError):
        return str(value)
    return f"{text}{unit}" if unit else text


# ------------------------------------------------------------------ stylesheet
STYLESHEET = f"""
* {{
    font-family: {SANS};
    font-size: 12px;
    color: {TEXT};
}}
QWidget#Root, QMainWindow {{
    background: {BG};
}}
QFrame#Panel {{
    background: {PANEL};
    border: 1px solid {BORDER};
    border-radius: 6px;
}}
QFrame#Panel[alarm="true"] {{
    border: 1px solid {FAIL};
}}
QLabel#PanelTitle {{
    color: {TEXT_DIM};
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 1.2px;
    padding: 2px 0px;
}}
QLabel#Caption {{
    color: {TEXT_FAINT};
    font-size: 10px;
    letter-spacing: 0.6px;
}}
QLabel#Value {{
    font-family: {MONO};
    font-size: 13px;
    color: {TEXT};
}}
QLabel#BigValue {{
    font-family: {MONO};
    font-size: 22px;
    font-weight: 600;
    color: {TEXT};
}}
QLabel#Banner {{
    background: {WARN};
    color: #1a1206;
    font-weight: 700;
    letter-spacing: 1px;
    border-radius: 3px;
    padding: 2px 8px;
}}
/* The leak / flooding banner. Same shape as #Banner but FAIL-coloured and on
   white, because it has to win a glance against a header that already has an
   amber SIMULATED DATA label and four status pills on it. window.py blinks it
   rather than relying on colour alone. */
QLabel#Alert {{
    background: {FAIL};
    color: #ffffff;
    font-weight: 700;
    letter-spacing: 1px;
    border-radius: 3px;
    padding: 2px 8px;
}}
QLabel#Alert[dim="true"] {{
    background: {PANEL_HI};
    color: {FAIL};
}}

QPushButton {{
    background: {PANEL_HI};
    border: 1px solid {BORDER_HI};
    border-radius: 4px;
    padding: 5px 10px;
    color: {TEXT};
}}
QPushButton:hover  {{ border-color: {ACCENT}; }}
QPushButton:pressed,
QPushButton:checked {{
    background: {ACCENT};
    border-color: {ACCENT};
    color: #04121b;
    font-weight: 600;
}}
QPushButton:disabled {{
    color: {TEXT_FAINT};
    border-color: {BORDER};
    background: {PANEL};
}}
/* A button that shares its row with something else (the lights presets beside
   their slider). Only the padding changes — colour, border and the pressed /
   checked language stay the button language, which is the whole reason this is
   an objectName rule here rather than a per-widget stylesheet: a widget-level
   sheet REPLACES the application sheet for that widget and would drop these
   four back to Qt's pale default. */
QPushButton#Compact {{
    padding: 1px 3px;
    font-size: 11px;
}}
QPushButton#Danger {{
    background: #3a1214;
    border: 1px solid {FAIL};
    color: #ff9a9a;
    font-weight: 700;
    letter-spacing: 1px;
}}
QPushButton#Danger:pressed {{ background: {FAIL}; color: #1a0505; }}
QPushButton#Rec:checked {{ background: {FAIL}; border-color: {FAIL}; color: #200; }}

/* Combo boxes and spin boxes. Written out because Qt's defaults are a LIGHT
   widget: on this palette they render as pale grey slabs with near-invisible
   text, which is exactly what the mission row looked like before
   2026-08-14. They follow the button language instead — panel fill, the same
   border, accent on hover — so a control reads as a control. */
QComboBox, QAbstractSpinBox {{
    background: {PANEL_HI};
    border: 1px solid {BORDER_HI};
    border-radius: 4px;
    padding: 2px 6px;
    color: {TEXT};
    selection-background-color: {ACCENT};
    selection-color: #04121b;
}}
QComboBox:hover, QAbstractSpinBox:hover {{ border-color: {ACCENT}; }}
/* A typed-into field must be obviously the thing the keyboard is talking to:
   everything else here is NoFocus, so a focus ring is rare and meaningful. */
QAbstractSpinBox:focus {{
    border: 1px solid {ACCENT};
    background: {PANEL};
}}
QComboBox:disabled, QAbstractSpinBox:disabled {{
    color: {TEXT_FAINT};
    border-color: {BORDER};
    background: {PANEL};
}}
/* Text entry — the header's RECORDING NAME box. Same language as the combo and
   the spin boxes above, and for the same reason they were written out: left to
   Qt's default a QLineEdit is a WHITE slab, which on this palette is the
   brightest thing on the whole station and sits two centimetres from the video
   feed the operator's dark adaptation is for (operator, 2026-08-14). */
QLineEdit {{
    background: {PANEL_HI};
    border: 1px solid {BORDER_HI};
    border-radius: 4px;
    padding: 2px 6px;
    color: {TEXT};
    selection-background-color: {ACCENT};
    selection-color: #04121b;
}}
QLineEdit:hover {{ border-color: {ACCENT}; }}
QLineEdit:focus {{ border: 1px solid {ACCENT}; background: {PANEL}; }}
QLineEdit:disabled {{
    color: {TEXT_FAINT};
    border-color: {BORDER};
    background: {PANEL};
}}
/* Read-only text views (the MISSION LOG). Inset — background one step DARKER
   than the panel, like the progress bars and the plot — so a scrolling region
   reads as a well rather than as another raised control. */
QPlainTextEdit, QTextEdit {{
    background: {BG};
    border: 1px solid {BORDER};
    border-radius: 4px;
    color: {TEXT_DIM};
    selection-background-color: {ACCENT};
    selection-color: #04121b;
}}

/* Scroll bars. The application sheet styles these even inside a widget that
   carries its OWN sheet (the log view does), which is exactly why they belong
   here: unstyled, Qt paints a light-grey trough with a white handle, and that
   pale strip down the side of the mission log was the second half of the same
   2026-08-14 complaint as the white name box. Thin, no arrow buttons — a log
   is scrolled with the wheel, not by clicking 9-pixel triangles. */
QScrollBar:vertical {{
    background: transparent;
    width: 9px;
    margin: 0;
    border: none;
}}
QScrollBar:horizontal {{
    background: transparent;
    height: 9px;
    margin: 0;
    border: none;
}}
QScrollBar::handle:vertical {{
    background: {BORDER_HI};
    border-radius: 4px;
    min-height: 24px;
}}
QScrollBar::handle:horizontal {{
    background: {BORDER_HI};
    border-radius: 4px;
    min-width: 24px;
}}
QScrollBar::handle:hover {{ background: {TEXT_FAINT}; }}
QScrollBar::add-line, QScrollBar::sub-line {{
    width: 0; height: 0; border: none; background: none;
}}
QScrollBar::add-page, QScrollBar::sub-page {{ background: none; }}
QAbstractScrollArea::corner {{ background: transparent; border: none; }}
QComboBox::drop-down {{
    border: none;
    width: 16px;
}}
QComboBox::down-arrow {{
    /* Qt has no built-in glyph once the widget is restyled; a small CSS
       triangle keeps the affordance without shipping an icon file. */
    image: none;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-top: 5px solid {TEXT_DIM};
    width: 0; height: 0;
    margin-right: 4px;
}}
QComboBox::down-arrow:hover {{ border-top-color: {ACCENT}; }}
QComboBox QAbstractItemView {{
    background: {PANEL_HI};
    border: 1px solid {BORDER_HI};
    color: {TEXT};
    selection-background-color: {ACCENT};
    selection-color: #04121b;
    outline: none;
}}
QAbstractSpinBox::up-button, QAbstractSpinBox::down-button {{
    background: {PANEL};
    border-left: 1px solid {BORDER};
    width: 13px;
}}
QAbstractSpinBox::up-button:hover, QAbstractSpinBox::down-button:hover {{
    background: {BORDER_HI};
}}
QAbstractSpinBox::up-arrow {{
    image: none;
    border-left: 3px solid transparent;
    border-right: 3px solid transparent;
    border-bottom: 4px solid {TEXT_DIM};
    width: 0; height: 0;
}}
QAbstractSpinBox::down-arrow {{
    image: none;
    border-left: 3px solid transparent;
    border-right: 3px solid transparent;
    border-top: 4px solid {TEXT_DIM};
    width: 0; height: 0;
}}

QCheckBox {{ spacing: 6px; }}
QCheckBox::indicator {{
    width: 13px; height: 13px;
    border: 1px solid {BORDER_HI};
    border-radius: 3px;
    background: {PANEL};
}}
QCheckBox::indicator:checked {{ background: {OK}; border-color: {OK}; }}

QSlider::groove:horizontal {{
    height: 4px; background: {BORDER}; border-radius: 2px;
}}
QSlider::sub-page:horizontal {{ background: {ACCENT}; border-radius: 2px; }}
QSlider::handle:horizontal {{
    background: {TEXT}; width: 11px; margin: -5px 0; border-radius: 5px;
}}
QSlider::handle:horizontal:hover {{ background: {ACCENT}; }}

/* The chunk colours are DARKER than the state palette on purpose: this is the
   one bar with text printed over its fill, and #22c55e under near-white 10 px
   text is unreadable. Hue is unchanged, so it still reads as the same state. */
QProgressBar {{
    background: {BG};
    border: 1px solid {BORDER};
    border-radius: 3px;
    height: 14px;
    text-align: center;
    font-family: {MONO};
    font-size: 10px;
    font-weight: 600;
    color: #e8f0f7;
}}
QProgressBar::chunk {{ background: #15803d; border-radius: 2px; }}
QProgressBar[level="warn"]::chunk {{ background: #a35c07; }}
QProgressBar[level="crit"]::chunk {{ background: #9f2a25; }}
QProgressBar[level="idle"]::chunk {{ background: #3b4a5a; }}

QStatusBar {{
    background: {PANEL_HI};
    color: {TEXT_DIM};
    border-top: 1px solid {BORDER};
    font-family: {MONO};
    font-size: 11px;
}}
QToolTip {{
    background: {PANEL_HI};
    color: {TEXT};
    border: 1px solid {BORDER_HI};
    padding: 4px;
}}
"""


def apply(app) -> None:
    """Install the theme on a QApplication."""
    app.setStyleSheet(STYLESHEET)
