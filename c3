#!/usr/bin/env bash
# c3 — run a c3_camera tool in an interpreter that has depthai 2.x, from anywhere.
#
#     ./c3 preflight                # is the rig ready? (opens nothing)
#     ./c3 collect --profile research_near
#     ./c3 collect --profile research_near --dry-run
#     ./c3 discover --probe
#     ./c3 gantry                   # FMC4030 gantry panel (jog, move, tag-map position)
#     ./c3 gantry --connect-test    # is the controller reachable? (no motion)
#     ./c3 test                     # the no-camera regression suite
#     ./c3 env                      # which interpreter, which versions
#
# This is a convenience, not the only way in. After the 2026-08-05 consolidation
# the plain form works too:
#
#     conda activate robust && python -m c3_camera.c3_collect --profile research_near
#
# What the wrapper adds over that:
#
#   1. No activate step, and it cannot pick up the wrong python. c3_camera needs
#      depthai 2.x (c3_camera/__init__.py) and this machine's DEFAULT python is
#      conda base with depthai 3.5.0, which can claim the camera merely by
#      constructing a Pipeline. Calling the interpreter by absolute path makes an
#      active conda env irrelevant.
#   2. The working directory. `python -m c3_camera.<tool>` needs the repo root on
#      sys.path, and c3_collect's default output directory is the cwd-relative
#      "c3_camera/datasets/..." — run from elsewhere and a dataset lands
#      elsewhere. This cd's, and says so when it had to.
#
# CONSEQUENCE OF THAT cd: a RELATIVE path you pass (--out ./here, --profile
# ./x.yaml) resolves against the REPO ROOT, not your shell's cwd. Use an absolute
# path if you mean somewhere else.
#
# Interpreter selection, in order:
#   $C3_PY                                       explicit override, wins over everything
#   ~/miniforge3/envs/robust/bin/python           conda robust — the consolidated env
#   ${C3_VENV:-~/.venvs/c3-depthai}/bin/python    the venv (setup_env.sh), fallback
#
# `./c3 gui` IS DIFFERENT. It runs in `rovgui-pose`, not `robust`, because the
# station now hosts the SAM2 + FoundationPose object tracker in its own process
# and that stack cannot live in `robust`:
#
#   rovgui-pose   torch 2.11+cu128, sam2, PyQt5, depthai 2.32, numpy 2.4.4, cv2 5.0
#   robust        no torch; numpy pinned <2 for gtsam; the sim/acados/SLAM env
#
# The expensive part of that stack is FoundationPose's compiled CUDA extensions
# (pytorch3d, kaolin, nvdiffrast, its own mycpp/mycuda), already built in `oakd`
# — so rovgui-pose is a clone of `oakd` plus PyQt5 and depthai 2.32, rather than
# torch bolted onto `robust`. numpy MUST be 2.x there: torch 2.11 is compiled
# against numpy 2's C ABI and fails to initialise on 1.26 regardless of what its
# metadata declares. Verified 2026-08-09: rov_gui 37/37 and c3_camera 58/58 pass
# in that env, Qt is 5.15.14 in both, and the minimum window is identical.
#
# The camera tools above stay on `robust` deliberately: they are the code with
# the most measurement invested in them, and nothing about perception requires
# moving them. Override either side with $C3_PY / $ROVGUI_PY.
#
# If rovgui-pose is absent the GUI falls back to the normal interpreter and runs
# fine WITHOUT --pose; it says so rather than failing.
#
# `./c3 gantry` stays on `robust` and is a SEPARATE PROCESS from `./c3 gui` on
# purpose. Two independent reasons, either sufficient on its own:
#   1. libFMC4030-Lib.so keeps ONE process-global DeviceManager and has no
#      internal mutex — a second registration of the same id returns -4, of the
#      same ip:port -5, and every python call site wraps its own RLock. One
#      connection per process is the library's contract, not our convention.
#   2. The panel's optional deps (pyqtgraph, qdarkstyle, qtawesome) live in
#      `robust`; the station needs `rovgui-pose` (torch, numpy 2.x). Two
#      processes means neither env has to grow toward the other.
# Nothing collides at run time either: the gantry is 192.168.0.30:8088 over
# wlan, the C3 and the ROV are 192.168.2.x over the tether, and a Ctrl-C in the
# gantry process reaches its own SIGINT handler (stop_axis(mode=2) on all three
# axes) instead of being one more thing the station has to survive.
#
# `robust` is first as of 2026-08-06, when it was moved from python 3.14.4 to
# 3.12.13. That is not a cosmetic version bump: on 3.14 there is no numpy 1.26
# wheel, so the installed numpy had been BUILT FROM SOURCE, and that combination
# silently corrupts arrays —
#
#     def f():
#         a = np.full(100_000, 3.0)
#         _ = a * a          # a[0] became 9.0: the OPERAND was overwritten
#
# because numpy's temporary-elision treats a refcount-1 operand as a throwaway
# and CPython 3.14's LOAD_FAST_BORROW stopped incrementing the refcount of a
# local pushed on the stack. On 3.12 numpy 1.26.4 installs as a real manylinux
# wheel and the probe is clean.
#
# `./c3 env` still runs that probe on whatever interpreter it picked — keep
# reading its last line rather than trusting this comment.
#
# No depthai version checking happens here — c3_camera/__init__.py already
# refuses a wrong one with a better message than this script could write.

set -euo pipefail

REPO="$(cd -P "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PREFERRED="${CONDA_PREFIX_ROOT:-$HOME/miniforge3}/envs/robust/bin/python"
FALLBACK="${C3_VENV:-$HOME/.venvs/c3-depthai}/bin/python"

PY="${C3_PY:-}"
if [[ -z "$PY" ]]; then
  if   [[ -x "$PREFERRED" ]]; then PY="$PREFERRED"
  elif [[ -x "$FALLBACK"  ]]; then PY="$FALLBACK"
  fi
fi

if [[ ! -x "$PY" ]]; then
  echo "c3: no usable interpreter." >&2
  echo "    looked at: $PREFERRED" >&2
  echo "               $FALLBACK" >&2
  echo "    fix: conda activate robust && pip install -r '$REPO/c3_camera/requirements.txt'" >&2
  echo "    or:  bash '$REPO/c3_camera/setup_env.sh'   (builds the fallback venv)" >&2
  echo "    or:  C3_PY=/path/to/python ./c3 ..." >&2
  exit 1
fi

if [[ $# -eq 0 || "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<EOF
usage: ./c3 <tool> [args...]

tools:
  preflight        is the rig ready? camera/ROV/disk   (preflight.py)
  collect          record a TUM RGB-D dataset          (c3_collect.py)
  stream           live preview / latency view         (c3_stream.py)
  bench            throughput + latency measurements   (c3_bench.py)
  discover         find the camera, report its calib   (discover_c3.py)
  replay           play back a recording               (c3_replay.py)
  sweep            option sweep over encode/depth axes (c3_option_sweep.py)
  check            validate a written dataset          (c3_dataset_check.py)
  depth-accuracy   depth-vs-tape-measure rig           (c3_depth_accuracy.py)
  encode-quality   encoder quality comparison          (c3_encode_quality.py)
  extract-rgb      pull RGB frames out of an encoded capture
  madrona          start/stop the RPi video service
  gui              topside control-station dashboard    (python -m rov_gui)
  plots            trajectory figures from a run date   (rov_gui/tools/plot_runs.py)
  gantry           FMC4030 gantry: panel (no args) or CLI (src/gantry_runner.py)
  test             run every no-camera test file
  env              print the interpreter and versions this wrapper resolved
  <module>         any other c3_camera module name, passed through

  ./c3 <tool> --help   for that tool's own flags
EOF
  exit 0
fi

tool="$1"; shift
case "$tool" in
  preflight)      mod=preflight ;;
  collect)        mod=c3_collect ;;
  stream)         mod=c3_stream ;;
  bench)          mod=c3_bench ;;
  discover)       mod=discover_c3 ;;
  replay)         mod=c3_replay ;;
  sweep)          mod=c3_option_sweep ;;
  check)          mod=c3_dataset_check ;;
  depth-accuracy) mod=c3_depth_accuracy ;;
  encode-quality) mod=c3_encode_quality ;;
  extract-rgb)    mod=c3_extract_rgb_video ;;
  madrona)        mod=madrona ;;
  gui)
    # Not a c3_camera module: rov_gui is a sibling package, and as of 2026-08-09
    # it runs in its OWN env — see the header. Still routed through this wrapper
    # for the cwd (repo root on sys.path, so `python -m rov_gui` resolves).
    GUI_PY="${ROVGUI_PY:-${CONDA_PREFIX_ROOT:-$HOME/miniforge3}/envs/rovgui-pose/bin/python}"
    if [[ ! -x "$GUI_PY" ]]; then
      echo "c3: rovgui-pose not found, falling back to $PY" >&2
      echo "    the station will run, but --pose (object tracking) needs that env:" >&2
      echo "      conda create -n rovgui-pose --clone oakd" >&2
      echo "      P=~/miniforge3/envs/rovgui-pose/bin" >&2
      echo "      \$P/pip install --force-reinstall --no-deps numpy==2.4.4" >&2
      echo "      \$P/pip install PyQt5==5.15.11 'depthai~=2.32.0' pymavlink" >&2
      GUI_PY="$PY"
    fi
    cd "$REPO"
    exec "$GUI_PY" -m rov_gui "$@"
    ;;
  plots)
    # Offline replotting of the run folders. Unlike `gui` this needs NO Qt and
    # no depthai — matplotlib + numpy + yaml only — so it stays on $PY (robust),
    # and the cd matters as much as ever: the config, the tag map and the
    # session tree are all repo-relative.
    cd "$REPO"
    exec "$PY" -m rov_gui.tools.plot_runs "$@"
    ;;
  gantry)
    # No args -> gantry_runner.py dispatches to the PyQt5 panel itself
    # (gantry_runner.py, end of main()). Any other flag -> its headless CLI
    # (--waypoints-csv / --connect-test / --dry-run / ...), which installs a
    # SIGINT handler that calls stop_axis(mode=2) on every axis. That is a real
    # emergency stop, and only a standalone process can have one.
    #
    # --mock / --mock-camera / --light are PANEL-only flags: they live in
    # gantry_panel.parse_args, not in gantry_runner.build_argparser, which
    # would argparse-error on them. So those three route straight to the panel.
    #
    # The cd is load-bearing three times over: --trajectory-dir defaults to the
    # cwd-relative Path("data"), the panel writes Path("data") directly, and the
    # tag map / fisheye calibration it reads are repo-relative.
    cd "$REPO"
    case "${1:-}" in
      --mock|--mock-camera|--light) exec "$PY" src/gantry_panel.py  "$@" ;;
      *)                            exec "$PY" src/gantry_runner.py "$@" ;;
    esac
    ;;
  env)
    cd "$REPO"
    echo "interpreter : $PY"
    # The GUI resolves separately (see the header). Report it here too, or
    # "which interpreter am I on" answers the question for only half the tools.
    GUI_PY="${ROVGUI_PY:-${CONDA_PREFIX_ROOT:-$HOME/miniforge3}/envs/rovgui-pose/bin/python}"
    if [[ -x "$GUI_PY" ]]; then
      echo "gui         : $GUI_PY"
    else
      echo "gui         : $PY  (rovgui-pose absent — --pose unavailable)"
    fi
    echo "repo        : $REPO"
    # The FMC4030 SDK is vendored but gitignored (.gitignore: vendor/), so it
    # exists on the machine that unpacked it and nowhere else. Its absence is
    # the most likely "it worked yesterday" for ./c3 gantry, so say so here
    # rather than letting the panel fail at Connect.
    GANTRY_SO="$REPO/src/gantry/vendor/ubuntu/lib/64/libFMC4030-Lib.so"
    if [[ -f "$GANTRY_SO" ]]; then
      echo "gantry lib  : $GANTRY_SO"
    else
      echo "gantry lib  : MISSING — ./c3 gantry cannot connect (vendor/ is gitignored)"
    fi
    "$PY" - <<'PY'
import sys
print(f"python      : {sys.version.split()[0]}")
for name in ("depthai", "cv2", "numpy", "yaml", "pymavlink"):
    try:
        mod = __import__(name)
        print(f"{name:<12}: {getattr(mod, '__version__', '?')}")
    except Exception as e:                                   # noqa: BLE001
        print(f"{name:<12}: MISSING ({type(e).__name__})")

# numpy temporary-elision probe. See the header of this script: on python 3.14
# with numpy 1.26 the operand of a large-array expression gets overwritten, which
# is silent and wrong rather than loud and broken.
def _probe():
    import numpy as np
    a = np.full(100_000, 3.0)
    _ = a * a
    return float(a[0])
try:
    v = _probe()
    print(f"numpy elide : {'OK' if v == 3.0 else f'*** CORRUPTS OPERANDS (a[0]={v}) ***'}")
except Exception as e:                                       # noqa: BLE001
    print(f"numpy elide : could not probe ({e})")
PY
    exit 0
    ;;
  test)
    cd "$REPO"
    rc=0
    for t in c3_camera/tests/test_*.py; do
      echo "== $t"
      "$PY" "$t" || rc=1
    done
    # src/tests are the gantry-side no-camera tests. They import gantry_panel,
    # which imports PyQt5 and builds widgets, so they need a platform plugin —
    # offscreen, because this must run over ssh and in a headless shell too.
    # (rov_gui/tests stays out: it is run on its own, see rov_gui/README.md.)
    for t in src/tests/test_*.py; do
      [[ -e "$t" ]] || continue
      echo "== $t"
      QT_QPA_PLATFORM=offscreen "$PY" "$t" || rc=1
    done
    exit $rc
    ;;
  *)              mod="$tool" ;;
esac

[[ "$PWD" -ef "$REPO" ]] || \
  echo "c3: cd $REPO  (relative paths are repo-relative from here)" >&2
cd "$REPO"
exec "$PY" -m "c3_camera.$mod" "$@"
