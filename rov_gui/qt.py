#!/usr/bin/env python3
"""
qt.py — one import point for the Qt binding, and the cv2 landmine they share.

Two unrelated things live here because getting either wrong costs an afternoon.

1. Binding shim
---------------
PyQt5, PyQt6 and PySide6 differ in three places this codebase touches: the
signal/slot decorator names, which module ``QAction``/``QShortcut`` live in, and
``exec_()`` vs ``exec()``. Everything else is spelled with *scoped* enum names
(``Qt.AlignmentFlag.AlignCenter``, not ``Qt.AlignCenter``) which is the only
spelling all three accept — PyQt5 5.15 exposes the scoped form as well as the
short one, PyQt6 removed the short one. Verified 2026-08-06 against PyQt5
5.15.11 / Qt 5.15.14 in the ``robust`` env: every scoped name used by this
package resolves there.

Order of preference is PyQt5 first because that is what this repo's other Qt
GUI (``src/survey_tags_gui.py``) uses and what the ``robust`` env has installed.
``ROV_GUI_QT_API=PyQt6`` overrides.

2. cv2 hijacks QT_QPA_PLATFORM_PLUGIN_PATH
------------------------------------------
``cv2/config-3.py`` sets ``QT_QPA_PLATFORM_PLUGIN_PATH`` to *cv2's own* bundled
Qt5 plugin directory unconditionally at import — measured in the ``robust`` env
on 2026-08-06: importing cv2 sets it to
``…/site-packages/cv2/qt/plugins``, and that directory does contain a live
``platforms/libqxcb.so``. That plugin belongs to opencv's Qt build, not PyQt5's,
and pointing a PyQt process at a foreign build's platform plugin is the classic
mixed-Qt failure (either "could not load the Qt platform plugin" or a symbol
crash at startup). An environment variable cannot fix it from outside, because
cv2 overwrites whatever the caller set.

So: import cv2 through :func:`import_cv2`, which puts the variable back exactly
as it found it. Two consequences worth knowing:

* Qt reads the variable once, when the first ``QGuiApplication`` is built. Import
  order therefore also matters — build the QApplication before any cv2 import
  and the damage window closes entirely. :func:`import_cv2` covers the case
  where you cannot control the order (a backend imported lazily mid-run).
* ``c3_camera.viz`` deliberately does the *opposite* (``repair_qt_plugin_path``)
  because it wants ``cv2.imshow`` to work. Importing that module from a PyQt
  process re-points the variable at cv2's plugins; nothing here imports it before
  the QApplication exists, and the depth colouriser it provides is pulled in
  through :func:`import_cv2` afterwards, when the platform plugin is loaded and
  the variable no longer matters.
"""

from __future__ import annotations

import ctypes
import importlib
import os
import sys
from pathlib import Path

_PREFERRED = ("PyQt5", "PyQt6", "PySide6")

_want = os.environ.get("ROV_GUI_QT_API")
_candidates = [_want] if _want else list(_PREFERRED)

QtCore = QtGui = QtWidgets = None
QT_API = ""
_errors: dict[str, str] = {}
for _name in _candidates:
    try:
        QtCore = importlib.import_module(f"{_name}.QtCore")
        QtGui = importlib.import_module(f"{_name}.QtGui")
        QtWidgets = importlib.import_module(f"{_name}.QtWidgets")
        QT_API = _name
        break
    except ImportError as e:  # noqa: PERF203 - three candidates, once, at import
        _errors[_name] = str(e)

if not QT_API:
    tried = "\n".join(f"    {k}: {v}" for k, v in _errors.items())
    raise ImportError(
        "rov_gui needs a Qt binding and found none.\n"
        f"  tried:\n{tried}\n"
        "  install one, e.g.:  conda activate robust && pip install PyQt5\n"
        "  (the robust env already has PyQt5 5.15.11 as of 2026-08-06)")

# --------------------------------------------------------------- name aliases
if QT_API.startswith("PyQt"):
    Signal = QtCore.pyqtSignal
    Slot = QtCore.pyqtSlot
    Property = QtCore.pyqtProperty
else:                                    # PySide6
    Signal = QtCore.Signal
    Slot = QtCore.Slot
    Property = QtCore.Property

# QAction/QShortcut moved from QtWidgets to QtGui in Qt 6.
QAction = getattr(QtGui, "QAction", None) or QtWidgets.QAction
QShortcut = getattr(QtGui, "QShortcut", None) or QtWidgets.QShortcut

Qt = QtCore.Qt
QTimer = QtCore.QTimer
QThread = QtCore.QThread
QObject = QtCore.QObject
QMetaObject = QtCore.QMetaObject
QMutex = QtCore.QMutex
QMutexLocker = QtCore.QMutexLocker
QSize = QtCore.QSize
QRect = QtCore.QRect
QRectF = QtCore.QRectF
QPointF = QtCore.QPointF
QPoint = QtCore.QPoint

QImage = QtGui.QImage
QPixmap = QtGui.QPixmap
QPainter = QtGui.QPainter
QColor = QtGui.QColor
QFont = QtGui.QFont
QPen = QtGui.QPen
QBrush = QtGui.QBrush


def binding_plugin_dir() -> str | None:
    """Where THIS binding's own Qt plugins live, if it ships them."""
    try:
        root = Path(importlib.import_module(QT_API).__file__).resolve().parent
    except Exception:                                            # noqa: BLE001
        return None
    for sub in ("Qt5/plugins", "Qt6/plugins", "Qt/plugins"):
        candidate = root / sub
        if (candidate / "platforms").is_dir():
            return str(candidate)
    return None


def sanitize_plugin_path() -> str:
    """Point QT_QPA_PLATFORM_PLUGIN_PATH at OUR Qt, or at nothing.

    Call this immediately before building the QApplication, and call it *again*
    if anything ran in between — see the module docstring for why cv2 makes this
    necessary and :func:`import_cv2` for the other half of the defence.

    This is not paranoia, it is a crash that happened (2026-08-06): the hardware
    path runs ``c3_camera.preflight`` first, its python-env check imports cv2 to
    report the version, cv2 repoints the variable at
    ``site-packages/cv2/qt/plugins``, and that directory contains exactly one
    platform plugin — ``libqxcb.so``, built against cv2's own bundled Qt. Qt
    finds it, tries to load it into PyQt5's Qt, fails, and calls ``qFatal``:

        Could not load the Qt platform plugin "xcb" in ".../cv2/qt/plugins"
        even though it was found.   -> Aborted (core dumped)

    Note the failure is platform-specific in a way that hides it from testing:
    that directory has no ``libqoffscreen.so``, so under
    ``QT_QPA_PLATFORM=offscreen`` Qt silently falls back to its own plugin
    directory and everything works. Only a real X11 session crashes.

    Returns the value the variable ends up with ("" if it was removed).
    """
    ours = binding_plugin_dir()
    current = os.environ.get("QT_QPA_PLATFORM_PLUGIN_PATH")
    if ours is None:
        # A system Qt install with no bundled plugins: anything we inherited is
        # more likely to be wrong than right, so drop it and let Qt use its
        # compiled-in path.
        if current:
            os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)
        return ""
    if current and Path(current).resolve() == Path(ours).resolve():
        return current
    os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = ours
    return ours


def preload_platform_libs() -> list[str]:
    """dlopen our binding's Qt libraries by absolute path, RTLD_GLOBAL.

    Defends against the *other* half of the same class of bug, which
    ``QT_QPA_PLATFORM_PLUGIN_PATH`` cannot touch. Measured on this desktop
    (2026-08-06):

        LD_LIBRARY_PATH=…:/usr/lib/x86_64-linux-gnu:…      (set by OpenFOAM 12)

    PyQt5's extension modules use DT_RPATH, which beats ``LD_LIBRARY_PATH``, so
    ``import PyQt5.QtCore`` correctly loads PyQt5's own ``libQt5Core.so.5``. But
    the platform plugin ``Qt5/plugins/platforms/libqxcb.so`` uses DT_RUNPATH,
    which ``LD_LIBRARY_PATH`` *beats* — so its ``libQt5XcbQpa.so.5`` resolves to
    Ubuntu's system Qt instead, and the load dies with

        undefined symbol: _ZN23QPlatformVulkanInstance22presentAboutToBeQueuedEP7QWindow,
        version Qt_5_PRIVATE_API

    i.e. the same "could not load the Qt platform plugin" message as the cv2
    problem, with a completely different cause. Loading the right library first
    fixes it, because a later ``dlopen`` binds a SONAME that is already present
    rather than searching for it again.

    Best effort and silent: on a system Qt install there is nothing to preload
    and nothing to fix. Returns the libraries it actually loaded.
    """
    if not sys.platform.startswith("linux"):
        return []
    plugins = binding_plugin_dir()
    if plugins is None:
        return []
    libdir = Path(plugins).parent / "lib"
    if not libdir.is_dir():
        return []
    loaded = []
    for stem in ("Core", "DBus", "Gui", "XcbQpa"):
        for name in (f"libQt5{stem}.so.5", f"libQt6{stem}.so.6"):
            path = libdir / name
            if not path.exists():
                continue
            try:
                ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
                loaded.append(name)
            except OSError:
                pass          # nothing to do about it here; Qt will say so
    return loaded


# Both applied at import as well as at QApplication construction: cheap, and it
# covers a caller that builds its own QApplication without going through
# window.build_and_run().
sanitize_plugin_path()
preload_platform_libs()


def run_app(app) -> int:
    """``app.exec()`` on Qt6, ``app.exec_()`` on Qt5, without the caller caring."""
    runner = getattr(app, "exec", None)
    if callable(runner):
        return int(runner())
    return int(app.exec_())


def import_cv2():
    """Import cv2 without letting it repoint this process's Qt plugin path.

    See the module docstring. Returns the module, so call sites read
    ``cv2 = import_cv2()`` and are otherwise unchanged.
    """
    saved = os.environ.get("QT_QPA_PLATFORM_PLUGIN_PATH")
    try:
        import cv2  # noqa: PLC0415 - deliberately late and guarded
    finally:
        if saved is None:
            os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)
        else:
            os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = saved
    return cv2


def qt_versions() -> dict[str, str]:
    """What actually got loaded — printed at startup so a bug report has it."""
    return {
        "api": QT_API,
        "qt": getattr(QtCore, "QT_VERSION_STR", ""),
        "binding": (getattr(QtCore, "PYQT_VERSION_STR", "")
                    or getattr(importlib.import_module(QT_API), "__version__", "")),
    }


__all__ = ["QtCore", "QtGui", "QtWidgets", "QT_API", "Signal", "Slot", "Property",
           "QAction", "QShortcut", "Qt", "QTimer", "QThread", "QObject",
           "QMetaObject", "QMutex",
           "QMutexLocker", "QSize", "QRect", "QRectF", "QPointF", "QPoint",
           "QImage",
           "QPixmap", "QPainter", "QColor", "QFont", "QPen", "QBrush",
           "run_app", "import_cv2", "qt_versions", "sanitize_plugin_path",
           "preload_platform_libs", "binding_plugin_dir"]
