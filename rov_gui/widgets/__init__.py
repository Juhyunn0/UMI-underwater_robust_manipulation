"""rov_gui.widgets — the panels the dashboard is assembled from.

Every widget in here obeys three house rules, because the "one screen, no
scrolling" constraint is a property of the *widgets*, not of the layout:

1. **Nothing has a size hint that grows with its content.** A QLabel that gets
   a 1920x1080 pixmap asks the layout for 1920x1080, and the layout obliges by
   growing the window past the screen. The video widgets never call
   ``setPixmap``; they paint in ``paintEvent`` and declare
   ``QSizePolicy.Ignored``.
2. **Every widget renders "no data" as "--", never as 0.** See
   :func:`rov_gui.theme.fmt`.
3. **Every widget carries a connection state and shows it.** A panel that can
   only show values cannot tell the pilot the difference between "the current is
   zero" and "I stopped hearing about the current".
"""

from .indicators import (BiBar, ElidedLabel, HoldButton, HoldToConfirmButton,
                         KeyChip, MeterBar, Panel, Sparkline, StatusDot,
                         StatusPill, Tile)

__all__ = ["BiBar", "ElidedLabel", "HoldButton", "HoldToConfirmButton", "KeyChip",
           "MeterBar", "Panel", "Sparkline", "StatusDot", "StatusPill", "Tile"]
