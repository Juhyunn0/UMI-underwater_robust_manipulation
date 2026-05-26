"""
GUI launchers for ``gantry_runner.py`` and ``fisheye_gantry_tagslam.py``.

Pattern: each entry script calls ``launch_gantry_gui()`` or
``launch_fisheye_gui()`` when invoked with no CLI args. The GUI form collects
parameters, builds an argv list, and runs the same script as a subprocess
with those args. This keeps Tkinter isolated from cv2/matplotlib (which the
worker scripts use) and lets the worker script's existing SIGINT handler
take over for emergency stops.

Settings persistence: last-used values are saved to
``~/.umi_gui_state.json`` keyed by form name.
"""

from __future__ import annotations

import json
import os
import queue
import signal
import subprocess
import sys
import threading
import tkinter as tk
from dataclasses import dataclass, field
from pathlib import Path
from tkinter import filedialog, ttk
from typing import Any, Callable


SETTINGS_PATH = Path.home() / ".umi_gui_state.json"


# =============================================================================
# Settings persistence
# =============================================================================
def _load_settings() -> dict[str, dict[str, Any]]:
    if not SETTINGS_PATH.exists():
        return {}
    try:
        return json.loads(SETTINGS_PATH.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def _save_settings(form_name: str, values: dict[str, Any]) -> None:
    data = _load_settings()
    data[form_name] = {k: v for k, v in values.items() if not isinstance(v, (Path,))}
    try:
        SETTINGS_PATH.write_text(json.dumps(data, indent=2))
    except OSError:
        pass


# =============================================================================
# Field specifications (declarative)
# =============================================================================
@dataclass
class Field:
    """One row of the form."""
    key: str
    label: str
    kind: str  # "str", "int", "float", "bool", "choice", "file", "dir"
    default: Any = ""
    choices: tuple[str, ...] = ()
    cli_flag: str | None = None       # If set, becomes the --flag in argv
    cli_omit_when: Any = None         # If value equals this, skip the flag
    cli_store_true: bool = False      # store_true: emit flag only if True
    width: int = 20
    tooltip: str = ""


@dataclass
class Section:
    title: str
    fields: list[Field] = field(default_factory=list)


# =============================================================================
# Form builder
# =============================================================================
class FormFrame(ttk.Frame):
    """A Tk frame that materializes a list of Section into ttk widgets and
    can return a dict of current values."""

    def __init__(self, parent: tk.Misc, sections: list[Section], saved: dict[str, Any] | None = None):
        super().__init__(parent)
        self._fields: dict[str, Field] = {}
        self._vars: dict[str, tk.Variable] = {}
        saved = saved or {}

        for section in sections:
            sec = ttk.LabelFrame(self, text=section.title, padding=(8, 6))
            sec.pack(fill=tk.X, padx=4, pady=4)
            for row, fld in enumerate(section.fields):
                self._fields[fld.key] = fld
                self._build_field(sec, fld, row, saved.get(fld.key, fld.default))

    def _build_field(self, parent: tk.Widget, fld: Field, row: int, initial: Any) -> None:
        ttk.Label(parent, text=fld.label).grid(row=row, column=0, sticky=tk.W, padx=4, pady=2)

        if fld.kind == "bool":
            var = tk.BooleanVar(value=bool(initial))
            ttk.Checkbutton(parent, variable=var).grid(row=row, column=1, sticky=tk.W, padx=4, pady=2)
        elif fld.kind == "choice":
            var = tk.StringVar(value=str(initial))
            ttk.Combobox(parent, textvariable=var, values=fld.choices,
                         state="readonly", width=fld.width).grid(row=row, column=1, sticky=tk.W, padx=4, pady=2)
        elif fld.kind in ("file", "dir"):
            var = tk.StringVar(value=str(initial))
            entry_frame = ttk.Frame(parent)
            entry_frame.grid(row=row, column=1, sticky=tk.W + tk.E, padx=4, pady=2)
            ttk.Entry(entry_frame, textvariable=var, width=fld.width).pack(side=tk.LEFT)
            mode = fld.kind
            def _browse(v=var, m=mode):
                if m == "file":
                    p = filedialog.askopenfilename()
                else:
                    p = filedialog.askdirectory()
                if p:
                    v.set(p)
            ttk.Button(entry_frame, text="…", width=3, command=_browse).pack(side=tk.LEFT, padx=(4, 0))
        else:
            var = tk.StringVar(value="" if initial is None else str(initial))
            ttk.Entry(parent, textvariable=var, width=fld.width).grid(
                row=row, column=1, sticky=tk.W, padx=4, pady=2,
            )

        self._vars[fld.key] = var
        if fld.tooltip:
            ttk.Label(parent, text=fld.tooltip, foreground="#888").grid(
                row=row, column=2, sticky=tk.W, padx=4, pady=2,
            )

    def get_values(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, fld in self._fields.items():
            raw = self._vars[key].get()
            if fld.kind == "bool":
                out[key] = bool(raw)
            elif fld.kind == "int":
                try:
                    out[key] = int(raw) if str(raw).strip() != "" else None
                except ValueError:
                    out[key] = None
            elif fld.kind == "float":
                try:
                    out[key] = float(raw) if str(raw).strip() != "" else None
                except ValueError:
                    out[key] = None
            else:
                out[key] = raw if str(raw).strip() != "" else None
        return out


def build_argv(values: dict[str, Any], fields_by_key: dict[str, Field]) -> list[str]:
    """Translate the form's values dict into a list of CLI args for subprocess."""
    argv: list[str] = []
    for key, fld in fields_by_key.items():
        if fld.cli_flag is None:
            continue
        v = values.get(key)
        if v is None:
            continue
        if fld.cli_store_true:
            if v:
                argv.append(fld.cli_flag)
            continue
        # Skip when value equals the "omit when" sentinel.
        if fld.cli_omit_when is not None and v == fld.cli_omit_when:
            continue
        if isinstance(v, bool):
            if v:
                argv.append(fld.cli_flag)
            continue
        # nargs: split on whitespace for x,y,z triplets like "0 0 0".
        argv.append(fld.cli_flag)
        for token in str(v).split():
            argv.append(token)
    return argv


def collect_fields(sections: list[Section]) -> dict[str, Field]:
    out: dict[str, Field] = {}
    for s in sections:
        for f in s.fields:
            out[f.key] = f
    return out


# =============================================================================
# Subprocess runner with Tk-friendly output pump
# =============================================================================
class ProcessRunner:
    def __init__(self, on_line: Callable[[str], None], on_done: Callable[[int], None]):
        self._on_line = on_line
        self._on_done = on_done
        self._proc: subprocess.Popen | None = None
        self._thread: threading.Thread | None = None
        self._q: queue.Queue[tuple[str, Any]] = queue.Queue()

    def start(self, argv: list[str]) -> None:
        if self.is_running():
            return
        self._proc = subprocess.Popen(
            argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            text=True,
            cwd=str(Path(__file__).resolve().parents[1]),
        )
        self._thread = threading.Thread(target=self._pump, daemon=True)
        self._thread.start()

    def _pump(self) -> None:
        assert self._proc is not None
        try:
            assert self._proc.stdout is not None
            for line in self._proc.stdout:
                self._q.put(("line", line.rstrip("\n")))
        finally:
            rc = self._proc.wait() if self._proc is not None else -1
            self._q.put(("done", rc))

    def stop(self) -> None:
        """SIGINT first (lets the script's own handler clean up), SIGTERM after 3s."""
        if not self.is_running():
            return
        assert self._proc is not None
        try:
            self._proc.send_signal(signal.SIGINT)
        except ProcessLookupError:
            return
        # Watchdog: if the process is still alive after 3s, escalate.
        def _watchdog() -> None:
            try:
                self._proc.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                try:
                    self._proc.terminate()
                except ProcessLookupError:
                    pass
        threading.Thread(target=_watchdog, daemon=True).start()

    def is_running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def drain(self, root: tk.Misc) -> None:
        """Call from the Tk main loop via root.after()."""
        try:
            while True:
                kind, payload = self._q.get_nowait()
                if kind == "line":
                    self._on_line(payload)
                elif kind == "done":
                    self._on_done(int(payload))
        except queue.Empty:
            pass
        root.after(50, lambda: self.drain(root))


# =============================================================================
# Generic launcher window
# =============================================================================
def _launch_form(
    title: str,
    form_name: str,
    sections: list[Section],
    script_path: Path,
    preamble_args: list[str] | None = None,
) -> int:
    """Build the window, run the Tk mainloop, return 0 on clean exit."""
    saved = _load_settings().get(form_name, {})
    fields_by_key = collect_fields(sections)

    root = tk.Tk()
    root.title(title)
    root.geometry("980x780")

    # Top: form (scrollable).
    canvas = tk.Canvas(root, highlightthickness=0)
    scrollbar = ttk.Scrollbar(root, orient="vertical", command=canvas.yview)
    form_holder = ttk.Frame(canvas)
    form_holder.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
    canvas.create_window((0, 0), window=form_holder, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)
    canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=8, pady=(8, 0))
    scrollbar.place(in_=canvas, relx=1.0, rely=0.0, relheight=1.0, anchor="ne")

    form = FormFrame(form_holder, sections, saved=saved)
    form.pack(fill=tk.X)

    # Mouse-wheel scrolling.
    def _on_mousewheel(event):
        canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
    canvas.bind_all("<MouseWheel>", _on_mousewheel)
    canvas.bind_all("<Button-4>", lambda e: canvas.yview_scroll(-3, "units"))
    canvas.bind_all("<Button-5>", lambda e: canvas.yview_scroll(3, "units"))

    # Buttons row.
    btn_row = ttk.Frame(root)
    btn_row.pack(side=tk.TOP, fill=tk.X, padx=8, pady=8)

    status_var = tk.StringVar(value="Idle")
    ttk.Label(btn_row, textvariable=status_var, foreground="#444").pack(side=tk.LEFT, padx=(0, 12))

    run_btn = ttk.Button(btn_row, text="Run")
    run_btn.pack(side=tk.LEFT, padx=4)
    stop_btn = ttk.Button(btn_row, text="STOP", state=tk.DISABLED)
    stop_btn.pack(side=tk.LEFT, padx=4)
    clear_btn = ttk.Button(btn_row, text="Clear log")
    clear_btn.pack(side=tk.LEFT, padx=4)

    # Output log.
    log_frame = ttk.LabelFrame(root, text="Output", padding=(4, 4))
    log_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))
    log = tk.Text(log_frame, wrap="word", height=12, bg="#111", fg="#e8e8e8",
                  insertbackground="#e8e8e8", font=("Monospace", 9))
    log_scroll = ttk.Scrollbar(log_frame, orient="vertical", command=log.yview)
    log.configure(yscrollcommand=log_scroll.set)
    log.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    log_scroll.pack(side=tk.RIGHT, fill=tk.Y)
    log.configure(state=tk.DISABLED)

    def append_log(line: str) -> None:
        log.configure(state=tk.NORMAL)
        log.insert(tk.END, line + "\n")
        log.see(tk.END)
        log.configure(state=tk.DISABLED)

    def on_done(rc: int) -> None:
        status_var.set(f"Process exited (rc={rc})")
        append_log(f"--- process exited (rc={rc}) ---")
        run_btn.configure(state=tk.NORMAL)
        stop_btn.configure(state=tk.DISABLED)

    runner = ProcessRunner(on_line=append_log, on_done=on_done)
    runner.drain(root)

    def on_run() -> None:
        values = form.get_values()
        _save_settings(form_name, values)
        cli_args = build_argv(values, fields_by_key)
        argv = [sys.executable, str(script_path)]
        if preamble_args:
            argv.extend(preamble_args)
        argv.extend(cli_args)
        append_log(f"$ {' '.join(argv)}")
        status_var.set("Running…")
        run_btn.configure(state=tk.DISABLED)
        stop_btn.configure(state=tk.NORMAL)
        runner.start(argv)

    def on_stop() -> None:
        append_log("--- STOP requested (SIGINT) ---")
        status_var.set("Stopping…")
        runner.stop()

    def on_clear() -> None:
        log.configure(state=tk.NORMAL)
        log.delete("1.0", tk.END)
        log.configure(state=tk.DISABLED)

    run_btn.configure(command=on_run)
    stop_btn.configure(command=on_stop)
    clear_btn.configure(command=on_clear)

    def on_close() -> None:
        if runner.is_running():
            runner.stop()
        root.destroy()
    root.protocol("WM_DELETE_WINDOW", on_close)

    root.mainloop()
    return 0


# =============================================================================
# Form: gantry_runner.py
# =============================================================================
def _gantry_sections() -> list[Section]:
    return [
        Section("Target", [
            Field("target_mode", "Target mode", "choice", default="Single XYZ",
                  choices=("Single XYZ", "Waypoints CSV"),
                  cli_flag=None,  # consumed below
                  tooltip="Choose single point or CSV file."),
            Field("x_mm", "X (mm)", "float", default=0.0, cli_flag="--x-mm", width=12),
            Field("y_mm", "Y (mm)", "float", default=0.0, cli_flag="--y-mm", width=12),
            Field("z_mm", "Z (mm)", "float", default=0.0, cli_flag="--z-mm", width=12),
            Field("waypoints_csv", "Waypoints CSV", "file", default="",
                  cli_flag="--waypoints-csv", width=40),
        ]),
        Section("Motion", [
            Field("speed_mm_s", "Speed (mm/s)", "float", default=20.0, cli_flag="--speed-mm-s", width=12),
            Field("acc_mm_s2", "Acceleration (mm/s²)", "float", default=50.0, cli_flag="--acc-mm-s2", width=12),
            Field("dec_mm_s2", "Deceleration (mm/s²)", "float", default=50.0, cli_flag="--dec-mm-s2", width=12),
            Field("mode", "Mode", "choice", default="line",
                  choices=("line", "sequential"), cli_flag="--mode", width=14),
        ]),
        Section("Logging", [
            Field("log_hz", "Log rate (Hz)", "float", default=100.0, cli_flag="--log-hz", width=10),
            Field("trajectory_dir", "Trajectory dir", "dir", default="data",
                  cli_flag="--trajectory-dir", width=40),
        ]),
        Section("Controller (FMC4030)", [
            Field("gantry_ip", "IP", "str", default="192.168.0.30", cli_flag="--gantry-ip", width=18),
            Field("gantry_port", "Port", "int", default=8088, cli_flag="--gantry-port", width=10),
            Field("gantry_id", "Controller ID", "int", default=1, cli_flag="--gantry-id", width=6),
        ]),
        Section("Soft limits (mm) — leave blank to use device values", [
            Field("soft_limit_min_mm", "Min  X Y Z", "str", default="",
                  cli_flag="--soft-limit-min-mm", width=24,
                  tooltip='Three numbers separated by spaces, e.g. "-200 -200 -100"'),
            Field("soft_limit_max_mm", "Max  X Y Z", "str", default="",
                  cli_flag="--soft-limit-max-mm", width=24),
        ]),
        Section("Run mode", [
            Field("dry_run", "Dry run (validate, no motion)", "bool", default=True,
                  cli_flag="--dry-run", cli_store_true=True),
        ]),
    ]


def launch_gantry_gui(script_path: Path) -> int:
    """Open the gantry_runner.py configuration form and return its exit code.

    Adds a "Connect" button that runs ``gantry_runner.py --connect-test`` with
    just the current IP/port/id values, so the user can verify the controller
    link without committing to a motion run.
    """
    sections = _gantry_sections()
    return _launch_form_with_target_pruning(
        title="Gantry Runner",
        form_name="gantry_runner",
        sections=sections,
        script_path=script_path,
        target_choice_key="target_mode",
        single_keys=("x_mm", "y_mm", "z_mm"),
        csv_keys=("waypoints_csv",),
        connect_test_script=script_path,
        connect_test_arg_keys=("gantry_ip", "gantry_port", "gantry_id"),
    )


def _launch_form_with_target_pruning(
    *,
    title: str,
    form_name: str,
    sections: list[Section],
    script_path: Path,
    target_choice_key: str,
    single_keys: tuple[str, ...],
    csv_keys: tuple[str, ...],
    connect_test_script: Path | None = None,
    connect_test_arg_keys: tuple[str, ...] = (),
) -> int:
    """A small variant of _launch_form that drops irrelevant target fields
    from argv based on the value of a 'target mode' combobox.

    If ``connect_test_script`` is set, a 'Connect' button is added that runs
    ``connect_test_script --connect-test <flags from connect_test_arg_keys>``
    so the user can verify the gantry link without committing to a full run.
    """
    saved = _load_settings().get(form_name, {})
    fields_by_key = collect_fields(sections)

    root = tk.Tk()
    root.title(title)
    root.geometry("980x780")

    canvas = tk.Canvas(root, highlightthickness=0)
    scrollbar = ttk.Scrollbar(root, orient="vertical", command=canvas.yview)
    form_holder = ttk.Frame(canvas)
    form_holder.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
    canvas.create_window((0, 0), window=form_holder, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)
    canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=8, pady=(8, 0))
    scrollbar.place(in_=canvas, relx=1.0, rely=0.0, relheight=1.0, anchor="ne")

    form = FormFrame(form_holder, sections, saved=saved)
    form.pack(fill=tk.X)

    canvas.bind_all("<MouseWheel>", lambda e: canvas.yview_scroll(int(-1 * (e.delta / 120)), "units"))
    canvas.bind_all("<Button-4>", lambda e: canvas.yview_scroll(-3, "units"))
    canvas.bind_all("<Button-5>", lambda e: canvas.yview_scroll(3, "units"))

    btn_row = ttk.Frame(root)
    btn_row.pack(side=tk.TOP, fill=tk.X, padx=8, pady=8)
    status_var = tk.StringVar(value="Idle")
    ttk.Label(btn_row, textvariable=status_var, foreground="#444").pack(side=tk.LEFT, padx=(0, 12))
    connect_btn: ttk.Button | None = None
    if connect_test_script is not None:
        connect_btn = ttk.Button(btn_row, text="Connect")
        connect_btn.pack(side=tk.LEFT, padx=4)
    run_btn = ttk.Button(btn_row, text="Run"); run_btn.pack(side=tk.LEFT, padx=4)
    stop_btn = ttk.Button(btn_row, text="STOP", state=tk.DISABLED); stop_btn.pack(side=tk.LEFT, padx=4)
    clear_btn = ttk.Button(btn_row, text="Clear log"); clear_btn.pack(side=tk.LEFT, padx=4)

    log_frame = ttk.LabelFrame(root, text="Output", padding=(4, 4))
    log_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))
    log = tk.Text(log_frame, wrap="word", height=12, bg="#111", fg="#e8e8e8",
                  insertbackground="#e8e8e8", font=("Monospace", 9))
    log_scroll = ttk.Scrollbar(log_frame, orient="vertical", command=log.yview)
    log.configure(yscrollcommand=log_scroll.set)
    log.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    log_scroll.pack(side=tk.RIGHT, fill=tk.Y)
    log.configure(state=tk.DISABLED)

    def append_log(line: str) -> None:
        log.configure(state=tk.NORMAL)
        log.insert(tk.END, line + "\n")
        log.see(tk.END)
        log.configure(state=tk.DISABLED)

    def on_done(rc: int) -> None:
        status_var.set(f"Process exited (rc={rc})")
        append_log(f"--- process exited (rc={rc}) ---")
        run_btn.configure(state=tk.NORMAL)
        stop_btn.configure(state=tk.DISABLED)
        if connect_btn is not None:
            connect_btn.configure(state=tk.NORMAL)

    runner = ProcessRunner(on_line=append_log, on_done=on_done)
    runner.drain(root)

    def on_run() -> None:
        if runner.is_running():
            return
        values = form.get_values()
        _save_settings(form_name, values)

        # Prune target-mode-irrelevant fields BEFORE building argv.
        mode = values.get(target_choice_key, "")
        pruned = dict(values)
        if mode == "Single XYZ":
            for k in csv_keys:
                pruned[k] = None
        elif mode == "Waypoints CSV":
            for k in single_keys:
                pruned[k] = None

        cli_args = build_argv(pruned, fields_by_key)
        argv = [sys.executable, str(script_path)] + cli_args
        append_log(f"$ {' '.join(argv)}")
        status_var.set("Running…")
        run_btn.configure(state=tk.DISABLED)
        stop_btn.configure(state=tk.NORMAL)
        if connect_btn is not None:
            connect_btn.configure(state=tk.DISABLED)
        runner.start(argv)

    def on_stop() -> None:
        append_log("--- STOP requested (SIGINT) ---")
        status_var.set("Stopping…")
        runner.stop()

    def on_clear() -> None:
        log.configure(state=tk.NORMAL)
        log.delete("1.0", tk.END)
        log.configure(state=tk.DISABLED)

    def on_connect() -> None:
        if runner.is_running():
            return
        if connect_test_script is None:
            return
        values = form.get_values()
        # Save so the user doesn't lose their IP when iterating.
        _save_settings(form_name, values)
        # Build a tiny argv from just the connect-test keys (gantry ip/port/id).
        sub = {k: values.get(k) for k in connect_test_arg_keys}
        cli_args = build_argv(sub, fields_by_key)
        argv = [sys.executable, str(connect_test_script), "--connect-test"] + cli_args
        append_log(f"$ {' '.join(argv)}")
        status_var.set("Connecting…")
        run_btn.configure(state=tk.DISABLED)
        stop_btn.configure(state=tk.NORMAL)
        if connect_btn is not None:
            connect_btn.configure(state=tk.DISABLED)
        runner.start(argv)

    run_btn.configure(command=on_run)
    stop_btn.configure(command=on_stop)
    clear_btn.configure(command=on_clear)
    if connect_btn is not None:
        connect_btn.configure(command=on_connect)

    def on_close() -> None:
        if runner.is_running():
            runner.stop()
        root.destroy()
    root.protocol("WM_DELETE_WINDOW", on_close)

    root.mainloop()
    return 0


# =============================================================================
# Form: fisheye_gantry_tagslam.py
# =============================================================================
def _fisheye_sections() -> list[Section]:
    return [
        Section("Camera (fisheye)", [
            Field("camera_device", "Device (index or path)", "str", default="0",
                  cli_flag="--camera-device", width=18),
            Field("camera_resolution", "Resolution (W H)", "str", default="1920 1080",
                  cli_flag="--camera-resolution", width=16,
                  tooltip='Two numbers separated by space, e.g. "1920 1080"'),
            Field("camera_fps", "FPS", "float", default=30.0, cli_flag="--camera-fps", width=8),
            Field("fisheye_calib", "Calibration YAML", "file", default="config/fisheye_calib.yaml",
                  cli_flag="--fisheye-calib", width=40),
            Field("fisheye_balance", "Undistort balance (0=tight, 1=full FOV)", "float",
                  default=0.0, cli_flag="--fisheye-balance", width=8),
        ]),
        Section("AprilTag / SLAM (essentials)", [
            Field("tag_family", "Tag family", "choice", default="tag36h11",
                  choices=("tag36h11", "tag25h9", "tag16h5"), cli_flag="--tag-family", width=14),
            Field("tag_size", "Tag edge size (m)", "float", default=0.170, cli_flag="--tag-size", width=10),
            Field("anchor_tag_id", "Anchor tag ID", "int", default=1, cli_flag="--anchor-tag-id", width=6),
            Field("water_correction_mode", "Water correction", "choice", default="none",
                  choices=("none", "scalar", "trust-region", "refractive"),
                  cli_flag="--water-correction-mode", width=14,
                  tooltip="Start with 'none' (air); switch to 'refractive' underwater."),
            Field("config", "Pool/water config YAML", "file", default="config/config.yaml",
                  cli_flag="--config", width=40),
        ]),
        Section("Output / display", [
            Field("trajectory_dir", "Output dir root", "dir", default="data",
                  cli_flag="--trajectory-dir", width=40),
            Field("record_trajectory", "Record trajectory (save frames + CSV/HTML/plot)", "bool",
                  default=True, cli_flag="--record-trajectory", cli_store_true=True),
            Field("display_width", "Display width (px)", "int", default=1600,
                  cli_flag="--display-width", width=8),
            Field("no_window", "Headless (no window)", "bool", default=False,
                  cli_flag="--no-window", cli_store_true=True),
            Field("print_every", "Console print every (s)", "float", default=0.5,
                  cli_flag="--print-every", width=8),
            Field("max_frames", "Max frames (blank = unlimited)", "int", default="",
                  cli_flag="--max-frames", width=8),
        ]),
        Section("Gantry", [
            Field("no_gantry", "Disable gantry (camera-only passive mode)", "bool",
                  default=False, cli_flag="--no-gantry", cli_store_true=True),
            Field("target_mode", "Target mode", "choice", default="Single XYZ",
                  choices=("Single XYZ", "Waypoints CSV"), cli_flag=None),
            Field("x_mm", "X (mm)", "float", default=0.0, cli_flag="--x-mm", width=12),
            Field("y_mm", "Y (mm)", "float", default=0.0, cli_flag="--y-mm", width=12),
            Field("z_mm", "Z (mm)", "float", default=0.0, cli_flag="--z-mm", width=12),
            Field("waypoints_csv", "Waypoints CSV", "file", default="",
                  cli_flag="--waypoints-csv", width=40),
            Field("speed_mm_s", "Speed (mm/s)", "float", default=20.0, cli_flag="--speed-mm-s", width=10),
            Field("acc_mm_s2", "Accel (mm/s²)", "float", default=50.0, cli_flag="--acc-mm-s2", width=10),
            Field("dec_mm_s2", "Decel (mm/s²)", "float", default=50.0, cli_flag="--dec-mm-s2", width=10),
            Field("mode", "Motion mode", "choice", default="line",
                  choices=("line", "sequential"), cli_flag="--mode", width=14),
            Field("log_hz", "Telemetry rate (Hz)", "float", default=100.0,
                  cli_flag="--log-hz", width=8),
            Field("gantry_ip", "IP", "str", default="192.168.0.30",
                  cli_flag="--gantry-ip", width=18),
            Field("gantry_port", "Port", "int", default=8088, cli_flag="--gantry-port", width=10),
            Field("gantry_id", "Controller ID", "int", default=1, cli_flag="--gantry-id", width=6),
            Field("soft_limit_min_mm", "Soft min  X Y Z", "str", default="",
                  cli_flag="--soft-limit-min-mm", width=24),
            Field("soft_limit_max_mm", "Soft max  X Y Z", "str", default="",
                  cli_flag="--soft-limit-max-mm", width=24),
        ]),
        Section("Run mode", [
            Field("dry_run", "Dry run (no motion, no camera loop)", "bool",
                  default=False, cli_flag="--dry-run", cli_store_true=True),
        ]),
    ]


def launch_fisheye_gui(script_path: Path) -> int:
    sections = _fisheye_sections()
    gantry_script = script_path.parent / "gantry_runner.py"
    return _launch_form_with_target_pruning(
        title="Fisheye + Gantry TagSLAM",
        form_name="fisheye_gantry_tagslam",
        sections=sections,
        script_path=script_path,
        target_choice_key="target_mode",
        single_keys=("x_mm", "y_mm", "z_mm"),
        csv_keys=("waypoints_csv",),
        connect_test_script=gantry_script if gantry_script.exists() else None,
        connect_test_arg_keys=("gantry_ip", "gantry_port", "gantry_id"),
    )
