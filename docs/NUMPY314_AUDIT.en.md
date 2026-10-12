# NUMPY314_AUDIT — audit of outputs from the python 3.14 period

> **English translation.** Source of truth: [`NUMPY314_AUDIT.md`](NUMPY314_AUDIT.md) (Korean). Snapshot of 2026-10-12, source blob `5ae6f95620`; later edits to the original are not reflected here.
>
> Provenance tags are kept as in the original: `[측정 …]` measured (artifact cited) · `[유도 …]` derived · `[예측 …]` predicted, not measured · `[스펙 …]` vendor spec · `실측` measured · `미실측` not measured · `산출물 없음` no artifact found · `예시/유도값` example or derived value.

**Conclusion: none of the saved artifacts was confirmed to be contaminated.** Two functions that produce corrupted
results were found, but neither has ever left its output on disk.

Audit date 2026-08-06 · window covered **2026-05-04 21:28 ~ 2026-08-06** (94 days) ·
method: run the same code in the old environment `robust314_bad` (py3.14.4) vs the new environment `robust` (py3.12.13)
and cross-check the artifacts **by content**.

## What went wrong

When `robust` was on python 3.14, numpy 1.26.4 had no cp314 wheel, so it had been built from source.
In that combination, **arithmetic operations wrote the result over the operand buffer.** The return value is correct,
but the input is destroyed, so code that reads that variable again later is silently wrong.

The trigger conditions were confirmed by measurement (`robust314_bad`):

```
overwritten    : a*b(→a), b*a(→b), a+b(→a), a*2.0(→a), 2.0*a(→a), -a(→a)
                 (a+1)*b overwrites both a and b
safe           : unary ufuncs such as np.sqrt(a), fancy indexing a[m], a.sum(), np.multiply(a,b,out=c)

size threshold : 256 KB  (regardless of dtype — starts at 32768 elements for float64, 65536 for float32)
scope          : only for function local variables. Does not reproduce for module globals
form-dependent : even the same operation sometimes does not reproduce when written all on one line
actual error   : only when the overwritten variable is "read again after that"
```

The cause: numpy's temporary-array reuse treats an operand with refcount 1 as a temporary that can be discarded,
while CPython 3.14's `LOAD_FAST_BORROW` does not increment the refcount when it pushes a local variable onto the stack,
so **a live local variable looks like a temporary**. On 3.14, **even arrays passed as parameters**
look like refcount 1, so static inference does not work — which is why every verdict was made by execution cross-check.

## Audit method and its limits

1. **Static scan** — exhaustive AST extraction of "places inside a function where a local variable is used as an
   arithmetic operand and then read again". **630 candidates** in 182 files.
2. **Execution cross-check** — run the same command in both environments and hash-compare the artifacts by
   **content excluding paths** + stdout diff.
3. **Noise floor determined** — the two numpy builds have different SIMD kernels, so results differ by **up to 3.0e-16
   (≈1.4 ULP)**. A difference at or below this is not a bug but a build difference. This value was
   measured on `ground_truth.json` from `make_synthetic_capture` (61 differing floats, median
   1.6e-16). **Verdict criterion: a relative error must exceed 1e-14 to count as contamination.**

**Limitation**: paths that run only with the hardware present (ZED camera, gantry, C3 live) could not be
execution-cross-checked. Those parts were judged by an array-size argument (below).

## Execution cross-check results — all identical

| Target | Scale | Result |
|---|---|---|
| 13 in `marinegym/tests/` | — | identical (the same 1 failure: `test_dobmpc` casadi 4x1/6x1, a pre-existing problem) |
| `verify_acados/hydro/hydro_precise/meta` | — | identical PASS |
| `verify_eaob` | — | identical FAIL (NIS 49.5, the same value) |
| `verify_state_source` | 60 lines of output | **radRMS/NIS/NEES all identical** (only the wall clock differs) |
| 4 `c3_camera` test suites | 193 | identical (depth_accuracy 58, host_depth 45, offline 58, option_sweep 32) |
| `umi_handheld.warp` | 456 PNGs | **byte-identical** |
| `umi_handheld.extract_gripper_width` | — | byte-identical |
| `umi_handheld.record --source synthetic` | — | byte-identical |
| `tools/make_synthetic_capture` | 167 files | 166 images byte-identical; only the JSON differs, by ≤1.4 ULP → **build difference** |
| `src/tools/rebuild_trajectory_html` (tagslam plots/HTML) | 7,547-row trajectory | **byte-identical** |
| `calib/fov_audit --selftest` | — | identical |
| `gen_pool_apriltags`, `c3_dataset_check` | — | identical |

## 2 confirmed corruptions — but no saved artifacts (산출물 없음)

### 1. `c3_camera/c3_encode_quality.py:385` `ssim()`
For identical images SSIM should be 1.0, but it came out as **7.05e8**. The cause: `a*a`, `b*b`,
`a*b` overwrote `a` and `b` in turn, so the final `a*b` effectively computed `a²·b²`.
`test_encode_quality` failing 4/36 in the old environment was how this bug was first found.

**No impact**: this tool writes its results to `c3_camera/encode_quality/`, but that directory
does not exist — **it has never been run.**

### 2. `c3_camera/geometry.py:100-102` `depth_to_xyz()`
```python
x = (uu - i.cx) * z / i.fx     # <- z is overwritten here
y = (vv - i.cy) * z / i.fy     #    the next line uses the broken z
return np.stack([x, y, z], ...)  #    and the Z channel of the return value is that z too
```
With the same frame (`c3_camera/datasets/dataset_20260803_162223`):

| | Point count | z range | x range |
|---|---|---|---|
| Normal (py3.10/3.12) | 86,403 | 0.150 – 4.270 m | −2.25 – 2.38 m |
| Old robust (py3.14) | 42,146 | **0.054 – 109405.688 m** | −2.25 – **0.31** m |

**No impact**: there are only three callers — the live preview in `source.py`,
`c3_stream.py:89 save_pointcloud` (**0** saved `.ply` files = never written),
`c3_depth_accuracy.py:586` (`c3_camera/depth_accuracy/` absent = never run).

## Paths that could not be execution-cross-checked — verdict by size argument

15 files that need hardware. The largest exposure was `src/tagslam_core.py` (52 candidates, 442 artifacts
in this period). Evidence for the verdict:

- The variables at those 52 sites are **3-vectors** (`normal`, `axis`, `up_world`), **tag corner points**
  (`points`, `dirs`, `air_dirs` — 4 per tag), **LM Jacobians** (number of points × 6),
  and **per-frame scalar arrays** (`c_xm`, `g_xa`, etc.).
- The input to `robust_fit_plane` is `tag_points` (tagslam_core.py:1844) — several hundred.
- Of the trajectories saved in this period, **the longest is 7,547 rows** = 60 KB as float64. 1/4 of the 256 KB threshold.
- The offline half of the same module (plots/HTML, candidates 4280~6299) was run on that 7,547-row trajectory and
  confirmed **byte-identical**.

Therefore the tagslam online path has no array that would exceed the threshold. The same holds for the gantry GUI and the C3 live stream.

## Remaining risks

- The size argument above rests on the **saved trajectory lengths**. If there was a run that put 32,768 frames (for float64) or more
  into one array, it needs re-examination — there is none among what is currently saved.
- The old environment is preserved as `robust314_bad` (2.6 GB). If a suspect result turns up later, it can be
  reproduced in that environment and cross-checked against the new one. Once this audit is judged finished, the old environment may be deleted.

## How to reproduce

```bash
# check the trigger conditions
/home/bdml/miniforge3/envs/robust314_bad/bin/python -c '
import numpy as np
def f():
    a = np.full(100_000, 3.0); _ = a * a; return a[0]
print(f())        # 9.0 = corrupted,  3.0 = normal'

# check the current environment
./c3 env          # last line shows numpy elide : OK

# cross-check an arbitrary script in both environments (content comparison excluding paths)
scratchpad/diffrun.sh <label> "/tmp/out_{ENV}" <script> --out "{OUT}"
```
