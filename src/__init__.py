"""Whisker Flow core modules for FBG sensing, motor, gantry, and experiment UIs."""

import sys
from pathlib import Path

# Make the bundled QYSEA SDK importable when running from source
_PKG_ROOT = Path(__file__).resolve().parent
_QYSEA_SDK = _PKG_ROOT / "umi_aquatic" / "qysea"
if _QYSEA_SDK.exists():
    _qysea_path = str(_QYSEA_SDK)
    if _qysea_path not in sys.path:
        sys.path.append(_qysea_path)

# Lazy imports to avoid loading C libraries at package import time
# This prevents segfaults when using python -m with Qt applications
__all__ = ["fbg", "motor", "gantry", "exp"]


def __getattr__(name):
    """Lazy-load submodules to avoid C library conflicts with Qt."""
    if name in __all__:
        import importlib
        module = importlib.import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
