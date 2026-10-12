# FOV_AUDIT — Is the C3's factory calibration in air or underwater?

> **English translation.** Source of truth: [`FOV_AUDIT.md`](FOV_AUDIT.md) (Korean). Snapshot of 2026-10-12, source blob `67339dcaf4`; later edits to the original are not reflected here.

**Conclusion: it is an underwater calibration.** All three cameras agree with the flat-port Snell
prediction to within 3.5°, and deviate from the in-air spec by up to 41.9°.

- Script: [`calib/fov_audit.py`](fov_audit.py)
- Regenerate the artifacts: `python calib/fov_audit.py --audit` / `--selftest`
- Run environment: `~/.venvs/c3-depthai` (python 3.10.12 / opencv 4.11.0 / numpy 1.26.4)
- Target data: `c3_camera/datasets/*/calibration.json` — **all 10 dumps are one identical set**
  (2026-07-29 ~ 2026-08-03, no sign of a reflash). Representative: `dataset_20260803_162223/`

---

## Why this document exists

Until 2026-08-04, the evidence on this question was a comment hard-coded in our source
(`c3_camera/c3_collect.py:330` — `"IN-AIR calibration..."`). That string cited nothing, and as it
was stamped into the `note` field of every dataset's `calibration.json`, it came to look like a
fact reported by the device. **It was a wrong assumption.** This document replaces that comment
with a measurement.

## Method

The calibration is just K and a distortion vector; it does not state the field of view. But
together the two determine "where on the sensor a ray from a given direction lands", so the field
of view can be recovered. The angle θ from the optical axis is swept and **forward-projected** with
`cv2.projectPoints`, and bisection finds the θ at which the image point lands exactly on the image
boundary.

```
r_u = tan(theta)                                    ideal (undistorted) radius
r_d = r_u * (1 + k1 s + k2 s^2 + k3 s^3)            OpenCV rational model
          / (1 + k4 s + k5 s^2 + k6 s^3),  s = r_u^2
u   = cx + fx * r_d
```

**Not using back-projection (`cv2.undistortPoints`) is deliberate.** That route inverts the
distortion iteratively, and on a strongly distorted wide-angle lens it can silently fail to
converge — and that is exactly the region being tested now. The forward direction is a closed-form
polynomial evaluation, so it has no such failure mode. During the sweep it is also checked whether
the projection is monotonic; if monotonicity breaks, the model has left its valid domain, so no
value is returned and the **validity limit is reported** instead.

With a flat port, Snell's law holds at the window surface:

```
sin(theta_air) = n * sin(theta_water),    n = 1.333
```

Hence **a camera calibrated underwater reports a narrower field of view than the same lens in air.**
This asymmetry is the signal that separates the two hypotheses, and for this lens the two hypotheses
lie about 41° apart — a size that no calibration-quality error can account for.

## Results

```
camera                          size        fx  measured  air spec  water pred   delta  valid<=
-----------------------------------------------------------------------------------------------
CAM_A (colour IMX378)      3840x2160    3080.3     63.7d     95.0d       67.2d   -3.5d    85.0d
CAM_B (mono left OV9282)    1280x800     757.1     85.6d    127.0d       84.3d   +1.2d    85.0d
CAM_C (mono right OV9282)    1280x800     760.3     85.1d    127.0d       84.3d   +0.8d    85.0d

reverse check — measured in-water FOV converted back to air via Snell:
  CAM_A: 63.7d -> 89.4d  (spec 95d,  delta -5.6d)
  CAM_B: 85.6d -> 129.7d (spec 127d, delta +2.7d)
  CAM_C: 85.1d -> 128.7d (spec 127d, delta +1.7d)

worst |measured - water prediction| =  3.5 deg
worst |measured - air spec|         = 41.9 deg
VERDICT: UNDERWATER calibration
```

**The three cameras independently** support the underwater hypothesis. CAM_A (colour) has a
separate sensor, a separate lens and a separate resolution, yet gives the same conclusion.

### Caution on the FOV definition

The HFOV in the table above takes as its half-width the distance from the principal point to the
**nearest** left/right boundary, i.e. `min(cx, W-1-cx)` (a conservative definition that guarantees
the full field with certainty). For CAM_A, `cx = 1905.16` lies left of centre (1919.5), so this
choice changes the value — measured against the right boundary it comes out at **64.4°**, and the
difference from the underwater prediction becomes −2.8°. The conclusion is unaffected.

### Provenance of the in-air spec

`AIR_SPEC_HFOV = {CAM_A: 95, CAM_B: 127, CAM_C: 127}` are the Luxonis OAK-D (Pro) W's
**spec-sheet values, not measurements.** These values are hypotheses under verification, not
evidence. (The first version wrongly took CAM_A as 108°, which made CAM_A look as if it matched
nothing. That mismatch was given the explanation "4:3→16:9 crop", but **the explanation itself was
wrong** — a 16:9 crop cuts off the top and bottom, so it preserves the horizontal field of view. It
was a nonexistent reason attached to a value that did not fit; once the spec was corrected, CAM_A
became the third piece of evidence.)

## Control experiments — auditing the script itself

Since the entire conclusion rests on this one script, `--selftest` runs four controls. The most
important is **C2**: it directly disproves "doesn't this script spit out ~85° whatever you feed
it?"

```
C1  pinhole, zero distortion — does the bisection recover an exact FOV?
  [PASS] pinhole 60d: got 59.96d, want 59.96+-0.01d
  [PASS] pinhole 95d: got 94.96d, want 94.96+-0.01d
  [PASS] pinhole 127d: got 126.96d, want 126.96+-0.01d

C2  CRITICAL — can a *genuinely 127 deg* wide camera read back as 127 deg?
  [PASS] synthetic 127d fisheye: got 127.00d, want 127.00+-2.0d (fitted fx=576.6, valid<=85d)

C3  closed loop on the real C3 distortion: apply Snell -> refit -> remeasure
  [PASS] CAM_B refitted to air: got 129.75d, want 127.00+-4.0d (fx 757 -> 568)
  [PASS] CAM_C refitted to air: got 128.68d, want 127.00+-4.0d (fx 760 -> 570)

C4  guard — a model driven past its valid domain must be reported, not returned
  [PASS] non-monotonic model flagged: valid only up to 23.5d

ALL CONTROLS PASSED
```

- **C1** shows that the bisection and projection math is unbiased.
- **C2** is the key control. An equidistant fisheye (true HFOV 127°) is built and fitted to the OpenCV rational
  model; measured with the same script, it comes out at **127.00°**. In other words, both the script and the model
  are capable of representing 127° — **the 85° value is not a structural artifact of the code or the model.**
  (The control originally wanted was to run the same script on a stock in-air OAK-D W dump, but
  no such dump can be obtained right now, so equivalent power to disprove was secured with a synthetic camera.
  Once an actual bench unit is available, run it with `--calib` right away to strengthen this control.)
- **C3** is a closed loop that uses the actually measured distortion as is. Applying Snell to the C3's
  underwater model to build the in-air counterpart of the same physical lens, and refitting the rational model, gives
  **129.75° / 128.68°**, 2.7°/1.7° off the 127° spec. Incidentally, `fx 757 -> 568`,
  an exact 1.333× relation. This value agrees with the 129.7° that the user calculated independently by hand.

## What the results mean

Capturing **in air** with the underwater calibration gives

```
Z_reported = (fx_water / fx_air) * Z_true  ~  1.33 * Z_true      (central region)
```

That is, depth comes out about 33% too far. **It is not a clean scalar, though.** Even the
distortion model is for the mismatched medium, so additional radial error accrues the further out
toward the periphery. The "valid<=85d" in the table above is also the limit of the range over which
the two models are comparable.

`c3_camera/datasets/*` and `recordings/*` were all captured indoors (in air), so the depth scale of
the existing C3 RGB-D data carries a systematic error. Conversely, **in actual water this
calibration is correct.**

## What has not been confirmed yet

- **EEPROM metadata.** Reading `getEepromData()`'s `batchTime` / `boardCustom` /
  `productName` could pin down the provenance, but the camera is not on the network right
  now (192.168.2.191 ping 100% loss). Once it is back, read them and add them to this document.
- **Stock in-air OAK-D W dump.** A control that would replace C2 above with a real unit rather than a synthetic one.
- **Only 8 distortion coefficients are being saved.** `c3_camera/device.py:399` truncates with `dist[:8]`.
  If the device holds 14 (rational + thin-prism/tilt), peripheral accuracy suffers.
  A re-dump is needed.
- **No stereo extrinsics.** There is no `device.extrinsics`, so the rectified `P1` cannot be built.

## Vendor-side statements (independent evidence)

- Blue Robotics product page: *"Each unit is individually calibrated for underwater
  operation to ensure accurate depth and scale in marine environments."*
- Blue Robotics forum, Tony White: *"the calibration Marine Situ performs takes place
  entirely underwater. It uses the standard checkerboard approach with openCV"*
