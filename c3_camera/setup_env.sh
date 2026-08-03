#!/usr/bin/env bash
# setup_env.sh — build the venv for c3_camera on the SYSTEM python3.
#
# Why not conda: this repo's conda base is python 3.14 and, more importantly,
# conda ships its own libstdc++/libGL/Qt which have collided with the system
# ones on this desktop before (cv2.imshow being the usual casualty). A venv on
# /usr/bin/python3 links against the system libraries that the X session already
# uses, so the display path just works.
#
#   bash c3_camera/setup_env.sh
#   ~/.venvs/c3-depthai/bin/python c3_camera/discover_c3.py
#
# Re-running is safe; it upgrades in place.

set -euo pipefail

VENV="${C3_VENV:-$HOME/.venvs/c3-depthai}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SYSPY="${C3_PYTHON:-/usr/bin/python3}"

echo "=== c3_camera environment setup ==="
echo "  system python : $SYSPY  ($("$SYSPY" -V 2>&1))"
echo "  venv          : $VENV"

if [[ -n "${CONDA_DEFAULT_ENV:-}" ]]; then
  echo
  echo "  NOTE: conda env '${CONDA_DEFAULT_ENV}' is active. That does not affect"
  echo "        this script (it calls $SYSPY directly), but remember to run"
  echo "        'conda deactivate' before using the venv interactively."
fi

if [[ ! -x "$SYSPY" ]]; then
  echo "ERROR: $SYSPY not found. Set C3_PYTHON to a python3.9-3.14 interpreter." >&2
  exit 1
fi

if [[ ! -d "$VENV" ]]; then
  echo "--- creating venv"
  "$SYSPY" -m venv "$VENV"
else
  echo "--- venv already exists, upgrading in place"
fi

"$VENV/bin/pip" install --upgrade --quiet pip setuptools wheel
echo "--- installing requirements"
"$VENV/bin/pip" install -r "$HERE/requirements.txt"

echo
echo "--- verifying"
"$VENV/bin/python" - <<'PY'
import sys
import cv2, numpy as np, depthai as dai
print(f"  python  {sys.version.split()[0]}")
print(f"  depthai {dai.__version__}")
print(f"  opencv  {cv2.__version__}")
print(f"  numpy   {np.__version__}")
found = dai.XLinkConnection.getAllConnectedDevices()
if found:
    for d in found:
        print(f"  visible: {d.name} mxid={d.getMxId()} {d.state.name}")
else:
    print("  no OAK device answered discovery (that is fine if it is powered off,"
          " or if broadcast is blocked — connecting by IP does not need it)")
PY

cat <<EOF

=== done ===
Next:
  $VENV/bin/python c3_camera/discover_c3.py --probe
  $VENV/bin/python c3_camera/c3_stream.py

Optional convenience:
  alias c3py='$VENV/bin/python'
EOF
