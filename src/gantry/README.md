# Gantry Control Module

This package mirrors the structure of the `fbg` and `motor` modules: it exposes
importable APIs under `whisker_flow.gantry` while keeping vendor artifacts and
demo programs alongside the code.

```
gantry/
├── __init__.py                # Re-export user-facing classes
├── controller.py              # FMC4030 ctypes wrapper and helpers
├── demos/                     # Simple runnable examples
│   ├── manual_pad.py          # PyQt jog pad (re-implements Qt demo)
│   ├── whisker_dragging.py    # Ubuntu drag test using the wrapper
│   └── whisker_dragging_experiment.py  # Legacy Windows script
├── vendor/ubuntu/             # Official SDK libs + sample C demo
├── vendor/qt_demo/            # Original Qt UI project (reference only)
├── mannual/                   # Vendor PDF manuals
├── RETURN_CODES.md            # Quick reference for SDK return codes
└── README.md
```

## Python API

```python
from whisker_flow.gantry import FMC4030Controller, Axis, ControllerConfig

controller = FMC4030Controller()
controller.connect(ControllerConfig(ip="192.168.0.30", port=8088))
controller.jog_single_axis(Axis.X, position_units=50, speed_units=10, acc_units=50, dec_units=50)
```

See `controller.py` for helpers that cover the entire FMC4030 SDK: motion
primitives, IO, RS485 passthrough, Modbus operations, parameter management,
and file/script utilities.

## Demos

* `manual_pad.py` – PyQt application that mirrors the original vendor Qt demo:
  connect/disconnect, display live axis positions, and jog axes with hold-to-run
  buttons.
* `whisker_dragging.py` – Console drag test used in whisker experiments.
* `whisker_dragging_experiment.py` – Legacy Windows script that directly loads
  the vendor DLL (kept for reference).

Run any demo via:

```bash
python -m whisker_flow.gantry.demos.manual_pad
python -m whisker_flow.gantry.demos.whisker_dragging
```

## Vendor SDK

Official Ubuntu libraries live under `vendor/ubuntu/`. The `sdk_demo` subfolder
contains the C sample; run `make` there to verify the shared library linkage.

Manuals and firmware docs remain in `mannual/`. `RETURN_CODES.md` summarizes
the SDK error codes for quick reference.
