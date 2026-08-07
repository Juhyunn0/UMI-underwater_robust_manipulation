#!/usr/bin/env bash
# c3 — run a c3_camera tool in an interpreter that has depthai 2.x, from anywhere.
#
#     ./c3 preflight                # is the rig ready? (opens nothing)
#     ./c3 collect --profile research_near
#     ./c3 collect --profile research_near --dry-run
#     ./c3 discover --probe
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
    # Not a c3_camera module: rov_gui is a sibling package. Routed through this
    # wrapper anyway because --source hw needs exactly the same interpreter
    # (depthai 2.x) and the same cwd (repo root on sys.path) as the tools above.
    cd "$REPO"
    exec "$PY" -m rov_gui "$@"
    ;;
  env)
    cd "$REPO"
    echo "interpreter : $PY"
    echo "repo        : $REPO"
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
    exit $rc
    ;;
  *)              mod="$tool" ;;
esac

[[ "$PWD" -ef "$REPO" ]] || \
  echo "c3: cd $REPO  (relative paths are repo-relative from here)" >&2
cd "$REPO"
exec "$PY" -m "c3_camera.$mod" "$@"
