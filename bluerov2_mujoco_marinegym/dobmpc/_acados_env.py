"""Point the process at the locally-built acados before acados_template loads.

The C library (libacados/libhpipm/libblasfeo/libqpOASES_e) was built into
ACADOS_DIR. The generated solver .so depends on libacados.so, which in turn
needs the other three. LD_LIBRARY_PATH is read by the loader only at process
start, so setting os.environ here is too late for the dynamic linker. Instead we
PRE-LOAD the shared libraries with ctypes RTLD_GLOBAL (in dependency order) so
their symbols are already in the global namespace when acados_template dlopen()s
the generated solver -- this makes the fast path work with NO shell exports
(teleop.py users never set LD_LIBRARY_PATH)."""
import os
import ctypes

ACADOS_DIR = os.environ.get("ACADOS_SOURCE_DIR", "/home/bdml/acados")
os.environ.setdefault("ACADOS_SOURCE_DIR", ACADOS_DIR)

_lib = os.path.join(ACADOS_DIR, "lib")
_ld = os.environ.get("LD_LIBRARY_PATH", "")
if _lib not in _ld.split(":"):
    os.environ["LD_LIBRARY_PATH"] = _lib + ((":" + _ld) if _ld else "")

# pre-load (leaves first) so the generated solver resolves without LD_LIBRARY_PATH
_preloaded = []
for _name in ("libblasfeo.so", "libhpipm.so", "libqpOASES_e.so", "libacados.so"):
    _p = os.path.join(_lib, _name)
    try:
        ctypes.CDLL(_p, mode=ctypes.RTLD_GLOBAL)
        _preloaded.append(_name)
    except OSError:
        pass


def available():
    """True if the acados C library and python template are importable."""
    if not os.path.isfile(os.path.join(_lib, "libacados.so")):
        return False
    try:
        import acados_template  # noqa: F401
        return True
    except Exception:
        return False
