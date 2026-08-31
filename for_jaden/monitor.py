#!/usr/bin/env python3
"""Separate-process live dashboard for the BlueROV2 teleop (`teleop.py --monitor`).

Six panels in one desktop window (FLU, z-up):
  ① water velocity Vx/Vy/Vz vs time      ② disturbance arrow (XY + XZ)
  ③ ROV position 3D trajectory            ④⑤⑥ ROV position XY / XZ / YZ trajectories

Position trajectories (③④⑤⑥) encode **time as color** (viridis: dark = older,
yellow = now). The dashboard runs in its OWN process (multiprocessing `spawn`) fed by
a Queue, so it never contends with MuJoCo's GLFW viewer / Qt main thread; the sim
just calls `MonitorHandle.push(sample)` once per frame.

Reuses the live-plot pattern from `src/gantry_panel.py` (deque ring buffers, pyqtgraph
PlotWidget, dark theme, per-axis colors).
"""
import os

# Force the PyQt5 binding (repo standard; PySide6 is also installed) BEFORE importing
# pyqtgraph, and align qdarkstyle/qtpy to it, so bindings never mismatch.
os.environ.setdefault("PYQTGRAPH_QT_LIB", "PyQt5")
os.environ.setdefault("QT_API", "pyqt5")

import math
import queue
import multiprocessing
from collections import deque

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore

try:
    import pyqtgraph.opengl as gl
    _GL_OK = True
except Exception:                       # missing PyOpenGL / QtOpenGL -> 2D only
    gl = None
    _GL_OK = False

REFRESH_MS = 50                          # dashboard redraw period (~20 Hz)
MAX_TRAJ_PTS = 600                       # cap plotted trajectory points (perf)
AXIS_COLORS = ("#ef5350", "#66bb6a", "#64b5f6")   # X red, Y green, Z blue (gantry)
ARROW_XY = "#26c6da"                     # cyan  (top-down XY disturbance arrow)
ARROW_XZ = "#ffb300"                     # amber (side XZ disturbance arrow)
ARROW_RANGE = 0.6                        # disturbance-vector panel half-range (m/s),
                                         # FIXED so the scale never auto-jitters; the
                                         # current+wave water velocity stays well within
                                         # (default teleop peak ~0.4 m/s).

_FALLBACK_QSS = "QWidget{background-color:#1a1a1d;color:#e6e6e6;}"


def _stride(n, n_max=MAX_TRAJ_PTS):
    """A slice that downsamples n points to <= n_max (keeps endpoints reasonable)."""
    if n <= n_max:
        return slice(None)
    return slice(None, None, int(np.ceil(n / n_max)))


def _ensure_gl_format():
    """Request a >= 2.1 default surface format so pyqtgraph.opengl's version check
    passes on real GL contexts (it compares ctx.format().version(), which defaults
    too low otherwise). Must run before the QApplication creates any GL context."""
    if not _GL_OK:
        return
    try:
        from pyqtgraph.Qt import QtGui
        fmt = QtGui.QSurfaceFormat()
        fmt.setVersion(2, 1)
        fmt.setProfile(QtGui.QSurfaceFormat.OpenGLContextProfile.CompatibilityProfile)
        QtGui.QSurfaceFormat.setDefaultFormat(fmt)
    except Exception:
        pass


if _GL_OK:
    class _SafeGLView(gl.GLViewWidget):
        """GLViewWidget that never lets a GL init/paint failure crash the dashboard
        (e.g. headless / no GL context); it reports once via on_fail and goes inert."""

        def __init__(self, on_fail=None):
            super().__init__()
            self._on_fail = on_fail
            self._failed = False

        def initializeGL(self):
            try:
                super().initializeGL()
            except Exception as e:
                self._fail(e)

        def paintGL(self, *a, **k):
            if self._failed:
                return
            try:
                super().paintGL(*a, **k)
            except Exception as e:
                self._fail(e)

        def _fail(self, e):
            if not self._failed:
                self._failed = True
                if self._on_fail:
                    self._on_fail(e)


# ---------------------------------------------------------------------------
# Dashboard widget (lives in the child process)
# ---------------------------------------------------------------------------
class MonitorDashboard(QtWidgets.QWidget):
    def __init__(self, window_s=30.0, plan=None):
        super().__init__()
        self.window_s = float(window_s)
        self.plan = None if plan is None else np.asarray(plan, float)
        self._t0 = None                          # time axis origin (first sample / rec start)
        self._rec = False                        # last-seen recording state (edge detect)
        self.cmap = pg.colormap.get("viridis")
        # 64-entry brush LUT: color trajectory points by indexing this instead of
        # allocating a fresh QColor+QBrush per point every redraw.
        self._brush_lut = [pg.mkBrush(c)
                           for c in self.cmap.map(np.linspace(0, 1, 64), mode="qcolor")]
        pg.setConfigOption("background", "#0e0e0e")
        pg.setConfigOption("foreground", "#aaaaaa")
        pg.setConfigOptions(antialias=True)

        # ring buffers (sized generously for up to ~60 Hz pushes)
        n = int(self.window_s * 60) + 64
        self.t = deque(maxlen=n)
        self.vx = deque(maxlen=n); self.vy = deque(maxlen=n); self.vz = deque(maxlen=n)
        self.px = deque(maxlen=n); self.py = deque(maxlen=n); self.pz = deque(maxlen=n)
        self._last_vtot = (0.0, 0.0, 0.0)
        self._gl_n = 0

        grid = QtWidgets.QGridLayout(self)
        grid.addWidget(self._build_velocity(), 0, 0)
        grid.addWidget(self._build_arrow(), 0, 1)
        grid.addWidget(self._build_3d(), 0, 2)
        self.pw_xy, self.sc_xy, self.ln_xy = self._build_traj("XY  (x→, y↑)", "x (m)", "y (m)")
        self.pw_xz, self.sc_xz, self.ln_xz = self._build_traj("XZ  (x→, z↑)", "x (m)", "z (m)")
        self.pw_yz, self.sc_yz, self.ln_yz = self._build_traj("YZ  (y→, z↑)", "y (m)", "z (m)")
        grid.addWidget(self.pw_xy, 1, 0)
        grid.addWidget(self.pw_xz, 1, 1)
        grid.addWidget(self.pw_yz, 1, 2)
        for c in range(3):
            grid.setColumnStretch(c, 1)
        grid.setRowStretch(0, 1)
        grid.setRowStretch(1, 1)
        self._add_plan_overlays()

    def _add_plan_overlays(self):
        """Draw the planned trajectory (dashed amber) on the XY/XZ/YZ panels, or a +
        marker for a single-point (station-keeping) target."""
        if self.plan is None or not len(self.plan):
            return
        pen = pg.mkPen("#ffb300", width=1.5, style=QtCore.Qt.PenStyle.DashLine)
        for pw, ia, ib in ((self.pw_xy, 0, 1), (self.pw_xz, 0, 2), (self.pw_yz, 1, 2)):
            a, b = self.plan[:, ia], self.plan[:, ib]
            if len(self.plan) > 1:
                pw.plot(a, b, pen=pen)
            else:
                pw.plot(a, b, pen=None, symbol="+", symbolSize=16,
                        symbolPen=pg.mkPen("#ffb300", width=2), symbolBrush=None)

    # ---- panel builders -------------------------------------------------
    def _build_velocity(self):
        pw = pg.PlotWidget(title="① Water velocity (m/s) — current + wave")
        pw.addLegend(offset=(8, 8))
        pw.showGrid(x=True, y=True, alpha=0.15)
        pw.setLabel("bottom", "time (s from start)")
        pw.setLabel("left", "velocity (m/s)")
        self.c_vx = pw.plot([], [], pen=pg.mkPen(AXIS_COLORS[0], width=2), name="Vx")
        self.c_vy = pw.plot([], [], pen=pg.mkPen(AXIS_COLORS[1], width=2), name="Vy")
        self.c_vz = pw.plot([], [], pen=pg.mkPen(AXIS_COLORS[2], width=2), name="Vz")
        return pw

    def _build_arrow(self):
        pw = pg.PlotWidget(title="② Disturbance vector (water velocity, m/s) — "
                                 "cyan XY (top), amber XZ (side)")
        pw.setAspectLocked(True)
        pw.showGrid(x=True, y=True, alpha=0.12)
        # axis meaning: both arrows share the horizontal Vx; the vertical axis is Vy
        # for the cyan XY (top-down) arrow and Vz for the amber XZ (side) arrow.
        pw.setLabel("bottom", "Vx — forward (m/s)")
        pw.setLabel("left", "Vy cyan (left) · Vz amber (up)  (m/s)")
        # FIXED scale: set the range once (this disables auto-range for both axes), so
        # the frame no longer jitters as the disturbance magnitude changes.
        pw.setXRange(-ARROW_RANGE, ARROW_RANGE, padding=0)
        pw.setYRange(-ARROW_RANGE, ARROW_RANGE, padding=0)
        pw.addLine(x=0, pen=pg.mkPen("#444", width=1))
        pw.addLine(y=0, pen=pg.mkPen("#444", width=1))
        self.sh_xy = pg.PlotCurveItem(pen=pg.mkPen(ARROW_XY, width=3))
        self.sh_xz = pg.PlotCurveItem(pen=pg.mkPen(ARROW_XZ, width=3))
        self.hd_xy = pg.ArrowItem(angle=0, headLen=14, tipAngle=28, pen=None, brush=ARROW_XY)
        self.hd_xz = pg.ArrowItem(angle=0, headLen=14, tipAngle=28, pen=None, brush=ARROW_XZ)
        self.arrow_txt = pg.TextItem(color="#dddddd", anchor=(0, 0))
        self.arrow_txt.setPos(-0.97 * ARROW_RANGE, 0.97 * ARROW_RANGE)   # fixed corner
        for it in (self.sh_xy, self.sh_xz, self.hd_xy, self.hd_xz, self.arrow_txt):
            pw.addItem(it)
        self.pw_arrow = pw
        return pw

    def _build_3d(self):
        if _GL_OK:
            try:
                view = _SafeGLView(on_fail=self._on_gl_fail)
                view.setCameraPosition(distance=2.0)
                grid = gl.GLGridItem(); grid.setSize(2, 2); grid.setSpacing(0.25, 0.25)
                view.addItem(grid)
                view.addItem(gl.GLAxisItem(size=pg.Vector(0.5, 0.5, 0.5)))
                self.gl_line = gl.GLLinePlotItem(antialias=True, width=2.0, mode="line_strip")
                self.gl_scatter = gl.GLScatterPlotItem(size=6.0)
                view.addItem(self.gl_line)
                view.addItem(self.gl_scatter)
                self.gl_view = view
                self.gl_ok = True
                return view
            except Exception as e:               # GL widget couldn't be constructed
                self.gl_ok = False
                return QtWidgets.QLabel(f"③ 3D view unavailable\n({e})")
        self.gl_ok = False
        return QtWidgets.QLabel("③ 3D view unavailable\n(no pyqtgraph.opengl)")

    def _on_gl_fail(self, e):
        """Called once if the GL context init/paint fails (e.g. headless)."""
        self.gl_ok = False
        try:
            self.gl_view.setVisible(False)
        except Exception:
            pass
        print(f"[monitor] 3D view disabled ({e})")

    def _build_traj(self, title, xlabel, ylabel):
        pw = pg.PlotWidget(title=f"{title}   color = time (old→now)")
        pw.setAspectLocked(True)
        pw.showGrid(x=True, y=True, alpha=0.15)
        pw.setLabel("bottom", xlabel)
        pw.setLabel("left", ylabel)
        line = pg.PlotCurveItem(pen=pg.mkPen("#555", width=1))
        scatter = pg.ScatterPlotItem(size=6, pen=None)
        pw.addItem(line)
        pw.addItem(scatter)
        return pw, scatter, line

    # ---- data in --------------------------------------------------------
    def add_sample(self, s):
        t = s["t"]
        rec = bool(s.get("rec", False))
        # restart the clock at 0 when recording STARTS, or when sim time jumps back (reset)
        if (rec and not self._rec) or (self.t and t < self.t[-1]):
            for d in (self.t, self.vx, self.vy, self.vz, self.px, self.py, self.pz):
                d.clear()
            self._t0 = None
        self._rec = rec
        if self._t0 is None:
            self._t0 = t                     # first sample / recording start -> origin (0)
        self.t.append(t)
        vx, vy, vz = s["vtot"]
        self.vx.append(vx); self.vy.append(vy); self.vz.append(vz)
        px, py, pz = s["pos"]
        self.px.append(px); self.py.append(py); self.pz.append(pz)
        self._last_vtot = s["vtot"]

    # ---- redraw ---------------------------------------------------------
    def redraw(self):
        if not self.t:
            return
        t = np.fromiter(self.t, float)
        i0 = int(np.searchsorted(t, t[-1] - self.window_s))
        tt = t[i0:]
        t_rel = tt - self._t0                      # time from session start (0)
        vx = np.fromiter(self.vx, float)[i0:]
        vy = np.fromiter(self.vy, float)[i0:]
        vz = np.fromiter(self.vz, float)[i0:]
        self.c_vx.setData(t_rel, vx)
        self.c_vy.setData(t_rel, vy)
        self.c_vz.setData(t_rel, vz)

        self._update_arrow(*self._last_vtot)

        span = (tt[-1] - tt[0]) if tt.size >= 2 else 0.0
        norm = (np.clip((tt - tt[0]) / span, 0.0, 1.0)
                if span > 0 else np.zeros(tt.size))      # avoid 0/0 -> NaN colors
        px = np.fromiter(self.px, float)[i0:]
        py = np.fromiter(self.py, float)[i0:]
        pz = np.fromiter(self.pz, float)[i0:]
        self._update_traj(self.sc_xy, self.ln_xy, px, py, norm)
        self._update_traj(self.sc_xz, self.ln_xz, px, pz, norm)
        self._update_traj(self.sc_yz, self.ln_yz, py, pz, norm)
        if self.gl_ok:
            self._update_3d(px, py, pz, norm)

    def _update_arrow(self, vx, vy, vz):
        self.sh_xy.setData([0.0, vx], [0.0, vy])
        self.sh_xz.setData([0.0, vx], [0.0, vz])
        self.hd_xy.setPos(vx, vy)
        self.hd_xy.setStyle(angle=180.0 - math.degrees(math.atan2(vy, vx)))
        self.hd_xz.setPos(vx, vz)
        self.hd_xz.setStyle(angle=180.0 - math.degrees(math.atan2(vz, vx)))
        # scale is FIXED (set once in _build_arrow); don't re-range per frame.
        vmag = math.sqrt(vx * vx + vy * vy + vz * vz)
        self.arrow_txt.setText(f"|v| = {vmag:.3f} m/s\n"
                               f"({vx:+.2f}, {vy:+.2f}, {vz:+.2f})")

    def _update_traj(self, scatter, line, a, b, norm):
        line.setData(a, b)
        s = _stride(a.size)
        idx = (norm[s] * 63.0).astype(int)            # norm in [0,1] -> LUT index
        scatter.setData(x=a[s], y=b[s], pen=None,
                        brush=[self._brush_lut[i] for i in idx])

    def _update_3d(self, px, py, pz, norm):
        self._gl_n += 1
        s = _stride(px.size)
        pos = np.column_stack([px[s], py[s], pz[s]])
        colors = self.cmap.map(norm[s], mode="float")
        try:
            self.gl_scatter.setData(pos=pos, color=colors, size=6.0)
            self.gl_line.setData(pos=pos, color=colors, width=2.0, antialias=True)
            if self._gl_n % 30 == 1 and len(pos):
                c = pos.mean(axis=0)
                span = float(np.ptp(pos, axis=0).max()) if len(pos) > 1 else 1.0
                self.gl_view.opts["center"] = pg.Vector(*c)
                self.gl_view.setCameraPosition(distance=max(0.5, span * 2.5))
        except Exception as e:                    # GL paint failed -> stop trying
            self.gl_ok = False
            print(f"[monitor] 3D disabled ({e})")


# ---------------------------------------------------------------------------
# Process target + parent-side handle
# ---------------------------------------------------------------------------
def _qt_exec(app):
    (app.exec if hasattr(app, "exec") else app.exec_)()


def _run(q, cfg):
    """Child-process entry: build the Qt app + dashboard, drain the queue on a timer."""
    import signal
    # Ctrl-C in the terminal hits the whole process group; let the PARENT handle it
    # and shut us down cleanly via the None sentinel (close()), not a raw SIGINT.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    _ensure_gl_format()                       # before any GL context is created
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    try:
        import qdarkstyle
        try:
            app.setStyleSheet(qdarkstyle.load_stylesheet(qt_api="pyqt5"))
        except Exception:
            app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    except Exception:
        app.setStyleSheet(_FALLBACK_QSS)

    plan = cfg.get("plan", None)
    w = MonitorDashboard(cfg.get("window_s", 30.0),
                         plan=np.asarray(plan, float) if plan is not None else None)
    w.setWindowTitle("BlueROV2 Disturbance Monitor")
    w.resize(1400, 850)
    w.show()

    timer = QtCore.QTimer()

    def drain():
        got = 0
        while True:
            try:
                item = q.get_nowait()
            except queue.Empty:
                break
            if item is None:                      # sentinel -> shut down
                timer.stop()
                app.quit()
                return
            w.add_sample(item)
            got += 1
        if got:
            w.redraw()

    timer.timeout.connect(drain)
    timer.start(REFRESH_MS)
    _qt_exec(app)


class MonitorHandle:
    """Parent-side handle held by the sim. `push()` is non-blocking and never raises
    into the sim loop; `close()` shuts the child down. Window close also ends it."""

    def __init__(self, window_s=30.0, plan=None):
        ctx = multiprocessing.get_context("spawn")
        self._q = ctx.Queue(maxsize=256)
        cfg = {"window_s": window_s,
               "plan": (np.asarray(plan, float).tolist() if plan is not None else None)}
        self._proc = ctx.Process(target=_run, args=(self._q, cfg), daemon=True)
        self._proc.start()
        self._alive = True

    def push(self, sample):
        if not self._alive:
            return
        try:
            if not self._proc.is_alive():         # user closed the window
                self._alive = False
                return
            self._q.put_nowait(sample)
        except queue.Full:
            try:                                  # live scope: drop OLDEST, keep newest
                self._q.get_nowait()
                self._q.put_nowait(sample)
            except Exception:
                pass
        except (ValueError, OSError, AssertionError):
            self._alive = False                   # queue/pipe gone -> stop pushing

    def close(self):
        try:
            self._q.put_nowait(None)
        except Exception:
            pass
        try:
            self._proc.join(timeout=1.0)
            if self._proc.is_alive():
                self._proc.terminate()
        except Exception:
            pass
        self._alive = False


# Set a >= 2.1 default surface format at import (before any QApplication), so the
# 3D view works on real GL displays. Harmless in the parent (no GL there).
_ensure_gl_format()
