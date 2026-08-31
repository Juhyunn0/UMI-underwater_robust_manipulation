#!/usr/bin/env python3
"""
imu.py — host-side reading of the C3's on-board BNO086.

Why this IMU matters more than the ROV's
----------------------------------------
For visual-inertial SLAM the IMU must be rigidly attached to the camera and share
its clock. The C3's BNO086 satisfies both: it sits on the same board as the image
sensors, and its report timestamps come back in the **same `dai.Clock` domain as
`ImgFrame.getTimestamp()`** — verified on this camera by reading both in one
pipeline and comparing (image 189158.281 s vs accel 189158.353 s against
`dai.Clock.now()` 189158.387 s). The ROV's MAVLink IMU has neither property: it is
metres away on the Navigator board and arrives on a different clock over UDP. Log
both, but a VIO consumer should use this one.

Two measured caveats the dataset must disclose
----------------------------------------------
1. **No IMU-camera extrinsics on the device.** `getImuToCameraExtrinsics()` raises
   "IMU calibration data is not available on device yet." The factory calibration
   covers the cameras only, so T_imu_cam is unknown and a consumer must either
   calibrate it (Kalibr) or measure it mechanically.
2. **The accelerometer needs an intrinsic calibration** — but NOT the one this
   file used to describe. Static magnitude measured 11.82 m/s^2 against gravity
   9.81, accuracy flag UNRELIABLE/MEDIUM, and that was recorded here as "about
   20% high", i.e. a SCALE error. **2026-08-17: that was a misreading.** It was
   one attitude. Over a six-attitude tumble the scale comes out within 0.9% of
   unity and the real defect is a **1.80 m/s^2 (0.184 g) BIAS, almost entirely
   on IMU x** — which in the upright pose lies nearly along gravity and so adds
   straight onto it. Raw |a| across attitudes runs 8.37-11.69; corrected,
   9.806 +/- 0.050.
   [측정: sessions/low_level_controller_data/20260817/0817_101511 + _100139,
   21985 still samples, 6 attitudes; config/c3_imu_calib.json sha1 7081ff43]
   The distinction is not academic: a constant body-frame bias is removed
   completely by a static correction taken at any attitude, whereas a scale
   error keeps leaking as the attitude changes. Fit it with
   ``python -m rov_gui.tools.calib_c3_imu <jsonl> --fit accel``.
   We record the accuracy flag on every sample so this stays visible.

Both are written into the dataset metadata, because a VIO result built on
uncalibrated IMU intrinsics and a guessed extrinsic fails in ways that look like
an algorithm problem.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import depthai as dai

# One CSV row per IMU packet. accel and gyro arrive in the same packet but carry
# their own timestamps, so both are kept: a consumer that cares can interpolate,
# and one that does not can use t_device.
IMU_FIELDS = (
    "t_device", "t_host", "t_accel", "t_gyro", "seq", "seq_accel",
    "ax", "ay", "az", "accel_accuracy",
    "gx", "gy", "gz", "gyro_accuracy",
    "qi", "qj", "qk", "qw", "quat_accuracy",
    "mx", "my", "mz", "mag_accuracy",
)


@dataclass
class ImuSample:
    """One BNO086 packet, already detached from depthai's buffers.

    Units, as depthai reports them for the BNO086:
        ax, ay, az   m/s^2   (see the scale caveat in the module docstring)
        gx, gy, gz   rad/s
        mx, my, mz   micro-tesla (only when the magnetometer is enabled)
        q*           unit quaternion, ROTATION_VECTOR (only when enabled)
    """

    t_device: float          # dai.Clock domain — same timeline as ImgFrame
    t_host: float            # dai.Clock.now() when the host drained it
    t_accel: float
    t_gyro: float
    # seq is the GYROSCOPE report counter, which is contiguous on this part and is
    # therefore what continuity checks use. seq_accel is the accelerometer's own
    # counter, kept for completeness — it legitimately skips (see ImuStats).
    seq: int
    seq_accel: int = -1
    ax: float = float("nan")
    ay: float = float("nan")
    az: float = float("nan")
    accel_accuracy: str = ""
    gx: float = float("nan")
    gy: float = float("nan")
    gz: float = float("nan")
    gyro_accuracy: str = ""
    qi: float = float("nan")
    qj: float = float("nan")
    qk: float = float("nan")
    qw: float = float("nan")
    quat_accuracy: str = ""
    mx: float = float("nan")
    my: float = float("nan")
    mz: float = float("nan")
    mag_accuracy: str = ""

    def row(self) -> dict:
        return asdict(self)


def _acc_name(report) -> str:
    try:
        return str(report.accuracy).rsplit(".", 1)[-1]
    except Exception:  # noqa: BLE001
        return ""


def packet_to_sample(pkt, t_host: float) -> ImuSample | None:
    """Convert one dai.IMUPacket. Returns None if it carries nothing usable."""
    a = pkt.acceleroMeter
    g = pkt.gyroscope
    rv = pkt.rotationVector
    mf = pkt.magneticField

    if a is None and g is None:
        return None

    t_accel = a.getTimestamp().total_seconds() if a is not None else float("nan")
    t_gyro = g.getTimestamp().total_seconds() if g is not None else float("nan")
    # The accelerometer timestamp is the row's time when present: for VIO the
    # accel/gyro pair is treated as simultaneous, and picking one consistently
    # beats silently mixing the two.
    t_device = t_accel if t_accel == t_accel else t_gyro
    # Continuity comes from the gyroscope counter: measured on this BNO086, gyro
    # sequence numbers are strictly contiguous (953/953 deltas of 1) while the
    # accelerometer's skip (277 of 953 deltas were 2) even though every packet
    # carries exactly one report of each. Using the accel counter for gap
    # detection reports drops that did not happen.
    seq = g.getSequenceNum() if g is not None else a.getSequenceNum()
    seq_accel = a.getSequenceNum() if a is not None else -1

    s = ImuSample(t_device=t_device, t_host=t_host, t_accel=t_accel,
                  t_gyro=t_gyro, seq=seq, seq_accel=seq_accel)
    if a is not None:
        s.ax, s.ay, s.az, s.accel_accuracy = a.x, a.y, a.z, _acc_name(a)
    if g is not None:
        s.gx, s.gy, s.gz, s.gyro_accuracy = g.x, g.y, g.z, _acc_name(g)
    if rv is not None and (rv.real or rv.i or rv.j or rv.k):
        s.qi, s.qj, s.qk, s.qw = rv.i, rv.j, rv.k, rv.real
        s.quat_accuracy = _acc_name(rv)
    if mf is not None and (mf.x or mf.y or mf.z):
        s.mx, s.my, s.mz, s.mag_accuracy = mf.x, mf.y, mf.z, _acc_name(mf)
    return s


def read_imu_extrinsics(calib, socket=None) -> dict:
    """T_imu_cam from the device, or an explicit statement that it is missing.

    Returning the failure as data rather than raising is deliberate: 'the factory
    calibration has no IMU extrinsics' is a fact the dataset must record, not an
    error that should stop a dive.
    """
    if socket is None:
        socket = dai.CameraBoardSocket.CAM_A
    out: dict = {"available": False}
    for name in ("getImuToCameraExtrinsics", "getCameraToImuExtrinsics"):
        try:
            M = getattr(calib, name)(socket)
            out[name] = [[float(v) for v in row] for row in M]
            out["available"] = True
        except Exception as e:  # noqa: BLE001
            out[name] = None
            out.setdefault("error", f"{type(e).__name__}: {e}")
    if not out["available"]:
        out["note"] = (
            "The device holds no IMU extrinsics, so the transform between the "
            "BNO086 and the cameras is UNKNOWN. Visual-inertial use requires "
            "calibrating it (e.g. Kalibr) or measuring it mechanically. Camera "
            "intrinsics and the stereo extrinsics in calibration.json are "
            "unaffected and remain valid."
        )
    return out


class ImuStats:
    """Rate and continuity accounting, so the achieved rate can be reported.

    Continuity uses the gyroscope sequence number. The accelerometer's counter
    advances irregularly on this part — measured 277 of 953 consecutive deltas
    equal to 2 with no data actually missing — so counting gaps on it invents
    drops. The gyro counter was contiguous across every sample measured.

    Achieved rate also depends on batching. With four lossless image streams
    running, a 200 Hz request yielded 130 Hz (35% lost) at batch threshold 1, and
    199.5 Hz with 0 lost at threshold 10 — the loss was per-message XLink
    overhead, not bandwidth. The achieved rate is written into the dataset
    metadata rather than the requested one, so a consumer never has to trust that
    the request was honoured.
    """

    def __init__(self, window: int = 400):
        self.count = 0
        self.dropped = 0
        self._last_seq: int | None = None
        self._ts: list[float] = []
        self.window = window
        self.last: ImuSample | None = None

    def update(self, s: ImuSample) -> None:
        self.count += 1
        self.last = s
        if self._last_seq is not None and s.seq > self._last_seq + 1:
            self.dropped += s.seq - self._last_seq - 1
        self._last_seq = s.seq
        self._ts.append(s.t_device)
        if len(self._ts) > self.window:
            del self._ts[: len(self._ts) - self.window]

    @property
    def hz(self) -> float:
        """Achieved rate from the device timestamps, not from arrival times."""
        if len(self._ts) < 2:
            return 0.0
        span = self._ts[-1] - self._ts[0]
        return (len(self._ts) - 1) / span if span > 0 else 0.0

    @property
    def accel_magnitude(self) -> float:
        """|a| of the latest sample — should read ~9.81 at rest. It does not on
        this unit (~11.8), which is why it is on the HUD."""
        s = self.last
        if s is None or s.ax != s.ax:
            return float("nan")
        return (s.ax ** 2 + s.ay ** 2 + s.az ** 2) ** 0.5

    def hud_line(self) -> str:
        mag = self.accel_magnitude
        acc = self.last.accel_accuracy if self.last else "-"
        return (f"imu    {self.hz:6.1f} Hz  {self.count:7d} samples  "
                f"|a| {mag:5.2f} m/s2 (g=9.81)  acc={acc}  dropped {self.dropped}")

    def summary(self) -> dict:
        return {
            "samples": self.count,
            "dropped": self.dropped,
            "rate_hz": round(self.hz, 1),
            "last_accel_magnitude": (None if self.accel_magnitude != self.accel_magnitude
                                     else round(self.accel_magnitude, 3)),
            "last_accel_accuracy": self.last.accel_accuracy if self.last else None,
        }
