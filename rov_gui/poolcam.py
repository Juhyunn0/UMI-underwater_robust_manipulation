#!/usr/bin/env python3
"""
poolcam.py — the station's handle on the pool-corner cameras (``--pool-cams``).

The cameras are CCTV: watched on the POOL CAMS tab and recorded alongside the
station's REC UI, never an input to anything. This module is the whole of what
the station knows about them, and it is shaped to keep it that way.

They run in a CHILD PROCESS, not here
-------------------------------------
``pool_cam/pool_cam.py --serve`` owns the cameras, the decoding and the four
encoders; this side only says "record from t0 into this folder" and receives
small preview images. Three reasons, each sufficient on its own:

1. **A camera cannot take the station down.** Four USB cameras on long cables
   are the most unplug-prone hardware on the rig, and a native fault in a
   capture driver kills the process it runs in. In a child, that is a dead tab
   and a line in the MISSION LOG; here it would be the window with DISARM on it.
2. **The station's threads keep their CPU.** Four 1080p30 MJPEG decodes plus
   the pipes into ffmpeg are steady work. In a process of their own they share
   no interpreter lock with the GUI thread (which runs the 20 Hz command
   heartbeat) or with the MPC/policy workers.
3. **OpenCV's thread pool stays the station's.** pool_cam pins
   ``cv2.setNumThreads(1)`` because the umi env's OpenMP build otherwise
   busy-waits every core after each resize (pool_cam.py). That setting is
   process-global; changing it here would change how the station's own
   perception code runs.

Sync does not need the same process: frames are stamped with the driver's
CLOCK_MONOTONIC timestamps, which is the clock behind ``time.monotonic()`` in
every process on the machine. REC sends this side's ``time.monotonic()`` as the
shared t0, and each camera's sidecar JSON records ``t0_monotonic``.

What crosses back is deliberately inert: status dicts, log lines and preview
QImages. Nothing here takes the DataBus — the window hands it a ``log``
callable and nothing more — and ``test_poolcam_is_isolated_from_control``
keeps this module and its panel from importing the control path.

Lifetime: the child is in its own session, so a terminal Ctrl+C is the
station's to handle, and it ends when its stdin closes — a station that exits,
or dies, makes it finish every file and quit.
"""

from __future__ import annotations

import json
import queue
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

from pool_cam import protocol

from .qt import QImage

REPO = Path(__file__).resolve().parents[1]
POOL_CAM_PY = REPO / "pool_cam" / "pool_cam.py"
DEFAULT_CONFIG = "config/pool_cams.yaml"

#: The settings a config file may hold, with the defaults used without one.
CONFIG_DEFAULTS = {"devices": {}, "slots": 4, "models": None, "size": "1920x1080",
                   "fps": 30.0, "encoder": "auto", "quality": None, "preview_hz": 10.0,
                   "power_line": 60}

#: Free space on the recording disk below which REC refuses to start the pool
#: cameras, and below which it warns. Four cameras write ~11 GB/h [유도: ~6
#: Mbit/s each, pool_cam.py DEFAULT_QUALITY note], on the SAME filesystem as the
#: station's nav/controller/policy records — those must never be the ones that
#: hit a full disk.
MIN_FREE_GB, WARN_FREE_GB = 5.0, 20.0

#: --source demo opens no hardware (rov_gui/__init__.py), so it gets this many
#: synthetic cameras at a size that keeps the demo light.
DEMO_CAMERAS, DEMO_SIZE = 4, "640x360"


def load_config(path) -> dict:
    """``config/pool_cams.yaml`` merged over :data:`CONFIG_DEFAULTS`. A missing
    file is the defaults (every pool camera found); a broken one raises."""
    cfg = dict(CONFIG_DEFAULTS)
    p = Path(path) if path else None
    if p is not None and not p.is_absolute() and not p.exists():
        p = REPO / p
    if p is not None and p.exists():
        import yaml

        data = yaml.safe_load(p.read_text()) or {}
        if not isinstance(data, dict):
            raise ValueError(f"{p}: expected a mapping at the top level")
        unknown = set(data) - set(CONFIG_DEFAULTS)
        if unknown:
            raise ValueError(f"{p}: unknown keys {sorted(unknown)}")
        # A key left empty in the file keeps its default (quality's IS empty).
        cfg.update({k: v for k, v in data.items() if v is not None})
    cfg["devices"] = dict(cfg.get("devices") or {})
    return cfg


def child_argv(opts) -> list[str]:
    """The command line for the child, from the station's options."""
    cfg = load_config(getattr(opts, "pool_cam_config", DEFAULT_CONFIG))
    argv = [sys.executable, str(POOL_CAM_PY), "--serve",
            "--fps", f"{float(cfg['fps']):g}", "--encoder", str(cfg["encoder"]),
            # The tab always has this many tiles; a camera plugged in later takes
            # the first empty one (pool_cam.py, --slots).
            "--slots", str(int(cfg["slots"]))]
    for model in cfg["models"] or ():         # None = pool_cam.py's POOL_MODELS
        argv += ["--model", str(model)]
    # Anti-flicker (pool_cam.py POWER_LINE). YAML reads a bare `off` as False.
    power = cfg["power_line"]
    argv += ["--power-line", "off" if power is False else str(power).lower()]
    if cfg["quality"] is not None:
        argv += ["--quality", str(int(cfg["quality"]))]
    if getattr(opts, "source", "demo") == "demo":
        return argv + ["--fake", str(DEMO_CAMERAS), "--size", DEMO_SIZE]
    argv += ["--size", str(cfg["size"])]
    # --pool-cam on the command line replaces the file's list outright: mixing
    # the two would leave the operator guessing which camera got which name.
    specs = list(getattr(opts, "pool_cam", None) or [])
    if not specs:
        specs = [f"{name}={dev}" for name, dev in cfg["devices"].items()]
    for spec in specs:
        argv += ["--device", str(spec)]
    return argv


class PoolCamClient:
    """Start, command and listen to the camera process. Thread-safe.

    ``log(level, message)`` is the only way out of here. The reader thread
    keeps the newest status and one preview per camera; the GUI takes them on
    its own tick (:meth:`snapshot`, :meth:`take_frame`)."""

    def __init__(self, argv: list[str], log=None, preview_hz: float = 10.0,
                 cwd: Path = REPO):
        self.argv = list(argv)
        self.cwd = cwd
        self.preview_hz = float(preview_hz)
        self._log_fn = log or (lambda level, msg: print(f"[{level}] {msg}"))
        self._lock = threading.Lock()
        self._proc: subprocess.Popen | None = None
        self._reader: threading.Thread | None = None
        # Commands go out on a thread of their own: a write to a pipe blocks
        # once it is full, and the caller here is the GUI thread, which also
        # runs the command heartbeat. A wedged camera process may cost the
        # tab its controls — never the station its thread.
        self._cmds: queue.SimpleQueue | None = None
        self._hello: dict | None = None
        self._status: dict | None = None
        self._status_seq = 0
        self._status_t: float | None = None       # monotonic arrival of the newest
        self._frames: dict[str, tuple[int, QImage]] = {}
        self._exit_note = ""
        self._stopping = False
        self._closed = False                      # stop() returned: stay silent
        self._bye = threading.Event()
        # What the station WANTS, kept here so a restarted child is told again:
        # which cameras are left out of REC, the preview sizes, and whether REC
        # is on (``recording``) and where to (``rec_dir``).
        self._armed: dict[str, bool] = {}
        self._preview: dict[str, tuple[int, int]] = {}
        self.recording = False
        self.rec_dir: Path | None = None

    # ------------------------------------------------------------ lifecycle
    def start(self) -> bool:
        """Start (or restart) the child. False if it could not be launched."""
        if self.alive:
            return True
        with self._lock:
            self._hello, self._status, self._frames = None, None, {}
            self._status_t = None
            self._exit_note, self._stopping, self._closed = "", False, False
        self._bye.clear()
        try:
            self._proc = subprocess.Popen(
                self.argv, cwd=str(self.cwd), stdin=subprocess.PIPE,
                stdout=subprocess.PIPE, stderr=None, start_new_session=True)
        except OSError as e:
            self._exit_note = f"could not start: {e}"
            self._log("error", self._exit_note)
            return False
        self._cmds = queue.SimpleQueue()
        threading.Thread(target=self._write, args=(self._proc, self._cmds),
                         name="poolcam-writer", daemon=True).start()
        self._reader = threading.Thread(target=self._read,
                                        args=(self._proc, self._cmds),
                                        name="poolcam-reader", daemon=True)
        self._reader.start()
        return True

    @property
    def alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def stop(self, timeout: float = 10.0) -> None:
        """Stop recording, let the child finish its files, and wait for it.

        After ``timeout`` the child is left to finish on its own: its stdin is
        closed, which is the signal it ends on, and it is in its own session,
        so nothing that happens to this process afterwards can cut a file
        short."""
        proc = self._proc
        if proc is None or proc.poll() is not None:
            return
        self._stopping = True
        self._send("record", on=False)
        self._send("quit")
        self._cmds.put(None)               # ...then the writer closes stdin
        self._bye.wait(timeout)
        try:
            proc.wait(timeout=max(0.5, timeout / 4))
        except subprocess.TimeoutExpired:
            self._log("warn", f"camera process still finishing files (pid {proc.pid}); "
                              "it will exit on its own")
        self.recording = False
        # Whatever the child still says after this — a late "saved ..." from a
        # daemon thread — must not reach a bus that is being torn down.
        self._closed = True

    # ------------------------------------------------------------- commands
    def start_recording(self, t0: float, outdir: Path, ui_video: str | None = None,
                        ui_started: float | None = None) -> bool:
        """Record every camera that is not left out, from monotonic ``t0``.

        Writes ``session.json`` beside the files first: which cameras were meant
        to record and which were left out, and the shared t0 — so a camera that
        never produced a file is still accounted for."""
        if not self.alive:
            self._log("warn", "not recording — the camera process is not running "
                              "(POOL CAMS tab: RESTART)")
            return False
        outdir = Path(outdir)
        try:
            free_gb = shutil.disk_usage(next(p for p in (outdir, *outdir.parents)
                                             if p.exists())).free / 1e9
        except (OSError, StopIteration):
            free_gb = None
        if free_gb is not None and free_gb < MIN_FREE_GB:
            self._log("error", f"NOT recording — {free_gb:.1f} GB free on the "
                               f"recording disk (< {MIN_FREE_GB:g} GB); the station's "
                               "own records need it more")
            return False
        if free_gb is not None and free_gb < WARN_FREE_GB:
            self._log("warn", f"only {free_gb:.1f} GB free — four cameras write "
                              "~11 GB an hour")
        with self._lock:
            hello = dict(self._hello or {})
            status = dict(self._status or {})
        slots = status.get("cams") or hello.get("cams", [])
        labels = [c.get("label") for c in slots]
        empty = {c.get("label") for c in slots
                 if c.get("empty") or (c.get("device") is None and "empty" not in c)}
        meta = {
            "what": "pool-corner cameras, recorded with the station's REC UI",
            "t0_monotonic": round(float(t0), 6),
            "clock": "CLOCK_MONOTONIC (time.monotonic(); same in every process)",
            "started": time.strftime("%Y-%m-%d %H:%M:%S"),
            "ui_video": ui_video,
            "ui_started_monotonic": (round(float(ui_started), 6)
                                     if ui_started is not None else None),
            "recording": [n for n in labels
                          if self._armed.get(n, True) and n not in empty],
            "left_out": [n for n in labels
                         if not self._armed.get(n, True) and n not in empty],
            "empty_slots": [n for n in labels if n in empty],
            "cameras": [{k: c.get(k) for k in ("label", "device", "usb", "size")}
                        for c in slots],
            "codec": hello.get("codec"),
            "note": ("each <cam>_<stamp>.json beside this file has the camera's "
                     "own t0_monotonic: frame k shows t0_monotonic + k/fps. A "
                     "camera let back in mid-recording starts a new file at its "
                     "own t0; one that drops out and returns does too, and so "
                     "does one plugged into an empty slot during the recording."),
        }
        try:
            outdir.mkdir(parents=True, exist_ok=True)
            (outdir / "session.json").write_text(json.dumps(meta, indent=2) + "\n")
        except OSError as e:
            self._log("error", f"session.json not written: {e}")
        self.recording, self.rec_dir = True, outdir
        self._send("record", on=True, t0=float(t0), outdir=str(outdir))
        return True

    def stop_recording(self) -> None:
        if self.recording:
            self._send("record", on=False)
        self.recording = False

    def arm(self, label: str, on: bool) -> None:
        """Include (True) or leave out (False) one camera from REC."""
        self._armed[label] = bool(on)
        self._send("arm", cam=label, on=bool(on))

    def armed(self, label: str) -> bool:
        return self._armed.get(label, True)

    def set_preview(self, sizes: dict[str, tuple[int, int]]) -> None:
        """Ask for previews at these sizes ({} = none: the tab is hidden)."""
        sizes = {k: (int(w), int(h)) for k, (w, h) in sizes.items()}
        if sizes == self._preview:
            return
        self._preview = sizes
        self._send("preview", sizes={k: list(v) for k, v in sizes.items()},
                   hz=self.preview_hz)

    # ---------------------------------------------------------------- reads
    def snapshot(self) -> tuple[dict | None, dict | None, int, str]:
        """(hello, newest status, status sequence number, exit note)."""
        with self._lock:
            return self._hello, self._status, self._status_seq, self._exit_note

    def status_age(self) -> float | None:
        """Seconds since the last status (they come twice a second), or None
        before the first. A live process that has gone quiet is wedged — and
        a tab still showing its last "● REC" would be lying."""
        with self._lock:
            t = self._status_t
        return None if t is None else time.monotonic() - t

    def take_frame(self, label: str, seen: int) -> tuple[int, QImage] | None:
        """The newest preview of ``label`` if it is newer than ``seen``."""
        with self._lock:
            item = self._frames.get(label)
        if item is None or item[0] == seen:
            return None
        return item

    # ------------------------------------------------------------- internals
    def _log(self, level: str, msg: str) -> None:
        if self._closed:
            return
        try:
            self._log_fn(level, f"pool cams: {msg}")
        except Exception:                                    # noqa: BLE001
            pass

    def _send(self, cmd: str, **fields) -> None:
        """Queue a command. Never blocks (see ``_cmds``)."""
        if self._cmds is not None and self.alive:
            self._cmds.put(protocol.command(cmd, **fields))

    @staticmethod
    def _write(proc: subprocess.Popen, cmds: queue.SimpleQueue) -> None:
        while True:
            line = cmds.get()
            if line is None:
                break
            try:
                proc.stdin.write(line)
                proc.stdin.flush()
            except (OSError, ValueError):
                break               # it died; the reader reports that
        try:
            proc.stdin.close()      # EOF: the child finishes its files and exits
        except OSError:
            pass

    def _read(self, proc: subprocess.Popen, cmds: queue.SimpleQueue) -> None:
        try:
            while True:
                got = protocol.read_message(proc.stdout)
                if got is None:
                    break
                msg, payload = got
                kind = msg.get("t")
                if kind == "frame":
                    w, h = int(msg["w"]), int(msg["h"])
                    if len(payload) != w * h * 3:
                        continue
                    img = QImage(payload, w, h, 3 * w,
                                 QImage.Format.Format_BGR888).copy()
                    with self._lock:
                        self._frames[msg["cam"]] = (int(msg["seq"]), img)
                elif kind == "status":
                    with self._lock:
                        self._status = msg
                        self._status_seq += 1
                        self._status_t = time.monotonic()
                elif kind == "log":
                    self._log(msg.get("level", "info"), str(msg.get("msg", "")))
                elif kind == "hello":         # (the child logs its own summary)
                    with self._lock:
                        self._hello = msg
                    self._resend()
                elif kind == "bye":
                    self._bye.set()
        except Exception as e:                               # noqa: BLE001
            self._log("error", f"lost the camera process's stream: {e} — "
                               "stopping it")
            # Out of step: nothing more it says can be parsed. Keep reading
            # (to EOF) so it never blocks on a full pipe, and end it — a
            # process that is alive but unheard looks like a working one.
            try:
                proc.terminate()
                while proc.stdout.read(65536):
                    pass
            except (OSError, ValueError):
                pass
        code = proc.wait()
        cmds.put(None)                  # let this process's writer thread end
        self._bye.set()
        note = f"camera process exited (code {code})"
        with self._lock:
            self._exit_note = note
        if not self._stopping:
            self._log("error", note + (" — pool cams are NOT recording"
                                       if self.recording else "")
                      + " — POOL CAMS tab: RESTART")

    def _resend(self) -> None:
        """A (re)started child knows nothing: tell it what the station wants —
        including a REC that is still on (the files resume at each camera's
        next frame, in the same folder)."""
        for label, on in dict(self._armed).items():
            if not on:
                self._send("arm", cam=label, on=False)
        if self._preview:
            self._send("preview", sizes={k: list(v) for k, v in self._preview.items()},
                       hz=self.preview_hz)
        if self.recording and self.rec_dir is not None:
            self._send("record", on=True, outdir=str(self.rec_dir))
            self._log("warn", f"recording resumed after a restart -> {self.rec_dir}")


def client_from_opts(opts, log) -> PoolCamClient:
    cfg = load_config(getattr(opts, "pool_cam_config", DEFAULT_CONFIG))
    return PoolCamClient(child_argv(opts), log=log,
                         preview_hz=float(cfg.get("preview_hz") or 10.0))


__all__ = ["PoolCamClient", "client_from_opts", "child_argv", "load_config",
           "POOL_CAM_PY", "DEFAULT_CONFIG"]
