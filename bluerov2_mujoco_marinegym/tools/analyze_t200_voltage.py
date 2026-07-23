#!/usr/bin/env python3
"""Ground the realistic-thruster `voltage_scale` in the official T200 datasheet.

Reproduces the provenance for `thrusters.NOMINAL_VOLTAGE_SCALE`. The MarineGym
T200 curve in thrusters.py (max +64.13 N / -51.55 N = +6.54 / -5.26 kgf) is
effectively a HIGH-voltage fit -- its max sits at the top of Blue Robotics'
published voltage range, so `voltage_scale = 1.0` models a ~20 V thruster. A real
BlueROV2 runs a 4S Li-ion pack (nominal 14.8 V), which delivers less thrust.

This script parses the public datasheet
`marinegym_assets/T200-Public-Performance-Data-10-20V-September-2019.xlsx`
(stdlib-only: an .xlsx is a zip of XML, so no pandas/openpyxl needed), prints the
per-voltage max forward/reverse thrust, estimates the base curve's voltage, and
computes the grounded `voltage_scale` for a chosen operating voltage (default
14.8 V) as max_thrust(operating) / max_thrust(base curve).

Read-only; no plant, no MuJoCo. Run in `robust` (or any Python 3).
"""
import os
import sys
import zipfile
import re
import xml.etree.ElementTree as ET

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # package root
sys.path.insert(0, HERE)
XLSX = os.path.join(HERE, "marinegym_assets",
                    "T200-Public-Performance-Data-10-20V-September-2019.xlsx")

_NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
       "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships"}


def _read_xlsx(path):
    """Minimal stdlib .xlsx reader -> {sheet_name: [ {col_letter: value}, ... ]}."""
    z = zipfile.ZipFile(path)
    # shared strings (cells with t="s" index into this table)
    sst = []
    if "xl/sharedStrings.xml" in z.namelist():
        root = ET.fromstring(z.read("xl/sharedStrings.xml"))
        for si in root.findall("m:si", _NS):
            sst.append("".join(t.text or "" for t in si.iter("{%s}t" % _NS["m"])))
    wb = ET.fromstring(z.read("xl/workbook.xml"))
    sheets = [(s.get("name"), s.get("{%s}id" % _NS["r"]))
              for s in wb.find("m:sheets", _NS)]
    rels = ET.fromstring(z.read("xl/_rels/workbook.xml.rels"))
    rid2tgt = {r.get("Id"): r.get("Target") for r in rels}

    def parse(sheet_path):
        root = ET.fromstring(z.read("xl/" + sheet_path))
        rows = []
        for row in root.iter("{%s}row" % _NS["m"]):
            cells = {}
            for c in row.findall("m:c", _NS):
                v = c.find("m:v", _NS)
                if v is None:
                    continue
                col = re.match(r"[A-Z]+", c.get("r")).group()
                val = v.text
                if c.get("t") == "s":
                    val = sst[int(val)]
                else:
                    try:
                        val = float(val)
                    except (TypeError, ValueError):
                        pass
                cells[col] = val
            rows.append(cells)
        return rows

    out = {}
    for name, rid in sheets:
        out[name] = parse("worksheets/" + rid2tgt[rid].split("/")[-1])
    return out


def _force_extents(rows):
    """(max forward, min reverse) of the 'Force (Kg f)' column for one sheet."""
    hdr = hdr_i = None
    for i, rw in enumerate(rows[:5]):
        if sum(isinstance(v, str) for v in rw.values()) >= 3:
            hdr, hdr_i = rw, i
            break
    if hdr is None:
        return None
    fcol = next((k for k, v in hdr.items()
                 if isinstance(v, str) and "force" in v.lower()), None)
    if fcol is None:
        return None
    vals = [rw[fcol] for rw in rows[hdr_i + 1:]
            if isinstance(rw.get(fcol), (int, float))]
    return (max(vals), min(vals)) if vals else None


def _interp(x, xs, ys):
    """Linear interpolation of y at x given sorted xs (xs[i] <= x <= xs[i+1])."""
    for i in range(len(xs) - 1):
        if xs[i] <= x <= xs[i + 1]:
            t = (x - xs[i]) / (xs[i + 1] - xs[i])
            return ys[i] + t * (ys[i + 1] - ys[i])
    return ys[0] if x < xs[0] else ys[-1]


def main(operating_v=14.8):
    if not os.path.exists(XLSX):
        raise SystemExit(f"datasheet not found: {XLSX}")
    sheets = _read_xlsx(XLSX)

    volts, fwd, rev = [], [], []
    for name, rows in sheets.items():
        m = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*V\s*", name)
        if not m:
            continue
        ext = _force_extents(rows)
        if ext is None:
            continue
        volts.append(float(m.group(1)))
        fwd.append(ext[0])
        rev.append(ext[1])
    order = sorted(range(len(volts)), key=lambda i: volts[i])
    volts = [volts[i] for i in order]
    fwd = [fwd[i] for i in order]
    rev = [rev[i] for i in order]

    print("=== T200 Public Performance Data (Sep 2019) -- max thrust per voltage ===")
    print(f"{'V':>6} {'max_fwd[kgf]':>14} {'max_rev[kgf]':>14}")
    for v, f, r in zip(volts, fwd, rev):
        print(f"{v:>6.1f} {f:>14.4f} {r:>14.4f}")

    # base curve from the live thrusters.py (single source of truth)
    import thrusters as T
    base_fwd = T.T200_MAX_FWD / 9.81     # N -> kgf
    base_rev = T.T200_MAX_REV / 9.81
    base_v_fwd = _interp(base_fwd, fwd, volts)       # invert fwd table: thrust->V
    print(f"\nbase curve (thrusters.T200_MAX): {base_fwd:+.3f} / {base_rev:+.3f} kgf"
          f"  -> forward implies ~{base_v_fwd:.1f} V (rev exceeds 20 V)")

    op_fwd = _interp(operating_v, volts, fwd)
    op_rev = _interp(operating_v, volts, rev)
    s_fwd = op_fwd / base_fwd
    s_rev = op_rev / base_rev
    scale = round((s_fwd + s_rev) / 2.0, 2)
    print(f"\noperating {operating_v:.1f} V: {op_fwd:+.3f} / {op_rev:+.3f} kgf"
          f"  (interp 14<->16 V)")
    print(f"voltage_scale = operating / base: fwd {s_fwd:.3f}, rev {s_rev:.3f}"
          f"  -> single scalar ~{scale:.2f}")
    print(f"\nthrusters.NOMINAL_VOLTAGE_SCALE = {T.NOMINAL_VOLTAGE_SCALE}"
          f"   ({'MATCH' if abs(T.NOMINAL_VOLTAGE_SCALE - scale) < 0.02 else 'CHECK'})")
    # a couple of reference operating points
    for v in (16.8, 13.0):
        s = ((_interp(v, volts, fwd) / base_fwd) + (_interp(v, volts, rev) / base_rev)) / 2
        print(f"  ref: {v:>4.1f} V -> ~{s:.2f}")


if __name__ == "__main__":
    main()
