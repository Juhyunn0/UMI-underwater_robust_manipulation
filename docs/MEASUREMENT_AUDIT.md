# MEASUREMENT_AUDIT — 문서 속 수치의 출처 감사

생성일 2026-08-04 · 원자료 [`docs/measurement_audit_rows.json`](measurement_audit_rows.json) (630행) · 검사한 수치 주장 1028개 · 문서 26개

## ⚠ 이 감사 자체의 출처 (먼저 읽을 것)

이 감사는 **9개 병렬 감사자의 단일 패스 결과이고, 계획했던 적대적 검증 패스는 실행되지
못했다** (세션 토큰 한도로 `verify` 3개 + `synthesize` 1개 에이전트가 실패). 즉:

- **`a` 판정 346건은 재검증되지 않았다.** 각 행의 `evidence` 필드는 감사자가 파일을 열어
  확인했다고 *기술한* 내용이며, 제3자가 다시 열어보지 않았다. 지금 이 감사가 고발하는
  실패 유형("확인했다고 쓰여 있지만 산출물이 없다")이 이 표 안에도 있을 수 있다.
- `b`/`c` 판정은 부재 증명이라 상대적으로 튼튼하지만, 감사자가 산출물을 **못 찾은** 것과
  산출물이 **없는** 것을 구분하지 못했을 수 있다.
- 따라서 아래 표는 **감사 결과가 아니라 감사 1차 결과**다. 재검증 전에는 `a`를
  "검증됨"으로 인용하지 말 것.

재실행: `Workflow({scriptPath: '.../measurement-audit-wf_41f2a0d2-ee5.js', resumeFromRunId: 'wf_41f2a0d2-ee5'})` — 완료된 9개 감사자는 캐시에서 즉시 돌아오고 실패한 4개만 다시 돈다.

## 후속 — 해소된 항목 (표는 감사 시점의 기록이므로 고치지 않고 여기에 적는다)

### MinZ 298.9 / 149.4 mm → **실측 300 / 150 mm** (2026-08-05)

해당 행: **305, 390, 896, 910** (그리고 1042·1096의 인용). 감사가 "formula output,
산출물 없음"으로 판정한 것이 맞았고, 이제 산출물이 생겼다.

- **측정**: `c3_camera/datasets/*/depth/`, `c3_camera/recordings/*/depth/` 전수 —
  47,270 프레임 / 3.6e9 유효 픽셀. 바닥이 정확히 **300 mm**(기본) / **150 mm**
  (`--extended`)이고 **그 아래 픽셀은 0개**. 대표 산출물:
  `datasets/dataset_20260803_102816`(300 mm에 877,378 px),
  `datasets/dataset_20260803_162223`(150 mm, extended),
  `recordings/20260729_162216`(유효 픽셀의 4.61%가 바닥).
- **따라서 `TESTING.md:484`의 미해결 질문도 닫힌다**: rectified pair는 CAM_C(fx
  760.28 @1280 → 380.14 @640)를 물려받는다. 바닥이 정확히 300으로 찍히려면
  fx*B/95 ∈ [300, 301) 즉 fx_rect ∈ [380.0, 381.3)이어야 하는데 CAM_B의 378.56은
  298을 찍었어야 하고 그런 값은 데이터에 없다.
- **반영**: `c3_depth_accuracy.py`(FX_MONO_1280 = 760.28), `test_depth_accuracy.py`,
  `TESTING.md`, `host_depth.py`, `KNOWN_ISSUES.md`. 공기 중 물리거리 ≈225 mm는
  여전히 `[유도]`로 남는다 — 줄자 검증(`c3_depth_accuracy.py`)은 **아직 한 번도
  실행된 적이 없다**.

## 왜 이 감사를 했는가

`.claude/journal/research.md`의 "+28 mm bias"가 실측치처럼 서술되어 있었으나 실제로는
코드 리뷰 중 구성한 **합성 예시**였다. 같은 파일의 "IN-AIR calibration"은 우리가
하드코딩한 **근거 없는 가정**이었고 나중에 사실로 인용되었다(→ [`calib/FOV_AUDIT.md`](../calib/FOV_AUDIT.md)에서
반증). 두 건이 나온 이상 더 있다고 가정하고 전수 감사했다.

## 판정 기준

| 판정 | 뜻 |
|---|---|
| **a** | 이 레포에 실행 산출물이 있고, 감사자가 열어서 수치를 확인했다 (경로 명시) |
| **b** | 실행/실측한 것처럼 기술되어 있으나 **산출물이 레포에 없다** |
| **c** | 예시·가정·유도값·스펙시트 값인데 **측정치처럼 서술**되어 있다 |

문서가 이미 정직하게 "추정"·"스펙"·"유도"라고 밝힌 수치는 findings가 아니라 제외했다.
걸린 것은 **서술 방식이 독자에게 실측으로 읽히게 만드는** 경우뿐이다.

## 집계

| 판정 | 건수 | 비율 |
|---|---:|---:|
| a — 산출물 있음 | 346 | 54.9% |
| b — 산출물 없음 | 232 | 36.8% |
| c — 예시가 측정치로 | 52 | 8.3% |
| **합계** | **630** | |

`b`+`c` 합계 **284건** — 그중 high 71 / medium 118 / low 95.

### 문서별

| 문서 | a | b | c | 계 |
|---|---:|---:|---:|---:|
| [bluerov2_mujoco_marinegym/docs/CONTROL_METHODOLOGY.md](../bluerov2_mujoco_marinegym/docs/CONTROL_METHODOLOGY.md) | 42 | 46 | 3 | 91 |
| [.claude/journal/consults.md](../.claude/journal/consults.md) | 38 | 20 | 11 | 69 |
| [c3_camera/README.md](../c3_camera/README.md) | 25 | 40 | 3 | 68 |
| [.claude/journal/research.md](../.claude/journal/research.md) | 29 | 13 | 5 | 47 |
| [README_fisheye_gantry.md](../README_fisheye_gantry.md) | 30 | 15 | 1 | 46 |
| [c3_camera/TESTING.md](../c3_camera/TESTING.md) | 22 | 10 | 11 | 43 |
| [.claude/journal/reviews.md](../.claude/journal/reviews.md) | 18 | 17 | 2 | 37 |
| [KNOWN_ISSUES.md](../KNOWN_ISSUES.md) | 18 | 13 | 2 | 33 |
| [bluerov2_mujoco_marinegym/README.md](../bluerov2_mujoco_marinegym/README.md) | 16 | 16 | 0 | 32 |
| [c3_camera/README_DATASET.md](../c3_camera/README_DATASET.md) | 13 | 12 | 6 | 31 |
| [bluerov2_mujoco_dobmpc/README.md](../bluerov2_mujoco_dobmpc/README.md) | 10 | 7 | 0 | 17 |
| [ISAAC_AUV_SETUP_RUNBOOK.md](../ISAAC_AUV_SETUP_RUNBOOK.md) | 12 | 3 | 1 | 16 |
| [bluerov2_mujoco_marinegym/docs/HYDRO_VERIFICATION.md](../bluerov2_mujoco_marinegym/docs/HYDRO_VERIFICATION.md) | 9 | 5 | 1 | 15 |
| [bluerov2_mujoco_marinegym/docs/04_HYDRO.md](../bluerov2_mujoco_marinegym/docs/04_HYDRO.md) | 11 | 0 | 1 | 12 |
| [claude.md](../claude.md) | 3 | 6 | 0 | 9 |
| [bluerov2_mujoco_marinegym/docs/02_MODEL.md](../bluerov2_mujoco_marinegym/docs/02_MODEL.md) | 8 | 1 | 0 | 9 |
| [bluerov2_mujoco_marinegym/docs/03_THRUSTERS.md](../bluerov2_mujoco_marinegym/docs/03_THRUSTERS.md) | 9 | 0 | 0 | 9 |
| [bluerov2_mujoco_marinegym/docs/07_DISTURBANCES.md](../bluerov2_mujoco_marinegym/docs/07_DISTURBANCES.md) | 9 | 0 | 0 | 9 |
| [bluerov2_mujoco_marinegym/docs/05_TELEOP.md](../bluerov2_mujoco_marinegym/docs/05_TELEOP.md) | 4 | 4 | 0 | 8 |
| [.claude/agents/hardware-advisor.md](../.claude/agents/hardware-advisor.md) | 0 | 2 | 4 | 6 |
| [bluerov2_mujoco_marinegym/docs/REAL_HYDRO_VERIFICATION.md](../bluerov2_mujoco_marinegym/docs/REAL_HYDRO_VERIFICATION.md) | 5 | 0 | 0 | 5 |
| [bluerov2_mujoco_marinegym/docs/06_ENVIRONMENT.md](../bluerov2_mujoco_marinegym/docs/06_ENVIRONMENT.md) | 3 | 1 | 1 | 5 |
| [bluerov2_mujoco_scratch/README.md](../bluerov2_mujoco_scratch/README.md) | 5 | 0 | 0 | 5 |
| [bluerov2_mujoco_marinegym/docs/00_OVERVIEW.md](../bluerov2_mujoco_marinegym/docs/00_OVERVIEW.md) | 4 | 0 | 0 | 4 |
| [bluerov2_mujoco_marinegym/docs/01_DECISIONS.md](../bluerov2_mujoco_marinegym/docs/01_DECISIONS.md) | 3 | 0 | 0 | 3 |
| [bluerov2_issac_paper/README.md](../bluerov2_issac_paper/README.md) | 0 | 1 | 0 | 1 |

## 최고 위험 — 결정이 얹혀 있는 미검증 수치

`b`/`c` 중 severity=high **71건**. 아래는 그중 다운스트림 영향이 가장 큰 것들이다.

**1. `.claude/journal/consults.md:27` — 판정 `b`**

> NONE(외란 0) 박스 A/B에서 코너 오차가 2.00/1.92/2.04 cm로 8 N·30 N 소수점까지 동일(코너 surge 명령 평균 0.9–1.4 N, 박스 발동 0.7%)

- 근거: No recording of the box-released A/B exists — the only 20260724 compares are the two 8 N wave sweeps, and neither meta.json carries controller.u_max. Recomputing corner error from the on-disk NONE square runs gives corner-window RMS 1.40–1.67 cm (mpc/dobmpc) / 1.95–2.36 cm (pid) and corner max 4.1–5.2 cm depending on window; 2.00/1.92/2.04 is not locatable, and no artifact records commanded surge or box-activation fraction at all (sat_freq is a different, always-zero metric).
- 영향: Entry is explicitly tagged '[직접 측정]'. These numbers are the whole basis of the square-corner-error-floor memory and a KNOWN_ISSUES limitation entry. Highest-priority re-run/attach.

**2. `.claude/journal/consults.md:29` — 판정 `b`**

> storm PID 26.14 / MPC 28.82→17.86 / DOB 17.36→7.30, gentle 11.56 / 9.51→7.63 / 3.14→1.87

- 근거: The pre-change numbers ARE on disk (wave00_storm CW mpc 28.8188, dobmpc 17.3561; wave05_gentle CW mpc 9.5094, dobmpc 3.1432). The post-change values 17.86 / 7.30 / 7.63 / 1.87 and both PID figures 26.14 / 11.56 are in no recording: no compare dir stamps controller.u_max, and the nearest post-change full sweep (compare_20260727_000850, gentle CW) gives pid 14.13 / mpc 7.75 / dobmpc 1.86 — close on MPC/DOB, far off on PID.
- 영향: Entry is tagged '[구현+검증]' and is the sole evidence for the mpc-surge-box-benchmark-artifact memory conclusion ('MPC 양단 모두 승, crossover 소멸'). The 3-heading verification run was never archived.

**3. `.claude/journal/consults.md:31` — 판정 `b`**

> storm에서 틱의 60%(mpc)·75%(dobmpc) 물림 vs gentle 7%

- 근거: The named sweep exists and I opened it, but nothing in results.csv / results_raw.csv / runs/ records constraint-activity fraction (the recorded columns are scenario…n_fail; sat_freq is 0 everywhere). No instrumented run was archived.
- 영향: This is the headline causal claim of the mpc-surge-box-benchmark-artifact memory and a KNOWN_ISSUES entry; params.py now repeats it as 'Measured effect'.

**4. `.claude/journal/consults.md:31` — 판정 `b`**

> storm PID−MPC −4.24→+8.60 cm 부호역전(MPC 42%↑), gentle +4.01→+4.87(10%↑)

- 근거: Computed PID−MPC from the named sweep: storm CW 26.5582−28.8188 = −2.26 cm, storm CDW −1.75 cm, gentle CW 14.1264−9.5094 = +4.62, gentle CDW +4.44. Neither the quoted baseline −4.24/+4.01 nor the released-box +8.60/+4.87 is present, and no box-released run exists in recordings/.
- 영향: Both the '부호역전' conclusion and the paired-heading baseline are unbacked by any archived artifact.

**5. `.claude/journal/research.md:127` — 판정 `b`**

> 핵심 제약은 XLink-over-PoE 실효 천장 ≈91.5 Mbit/s(서로 다른 4구성이 86/91.7/91.1/91.6로 수렴 → 링크 천장이지 연산 한계 아님)

- 근거: c3_camera/bench/ does not exist; c3_bench.py writes no CSV anywhere in the repo. The four cited rows are NV12-raw configurations (README.md:249-257 table); every recording on disk (c3_camera/recordings/20260729_162216 and 20260803_*) is MJPEG, so none of the four rows is reproducible. I checked all six recordings' meta.json: measured totals are 70.6, 83.1, 91.7, 76.5, 86.4, 37.3 Mbit/s — 20260803_150048 happens to read 91.7 but is a 1920x1080 MJPEG+depth config, not the '960x540 NV12 + depth' row claimed.
- 영향: Highest-leverage unbacked number in the file: 91.5 is hardcoded into c3_camera/dataset.py:961,995 (link_limited_fps), c3_collect.py:72, and sets config.py POE_BUDGET_MBPS=90; TESTING.md and README then derive fps caps from it. If it is wrong, every downstream fps/frontier number moves.

**6. `.claude/journal/research.md:127` — 판정 `b`**

> 최대 레버는 디바이스측 MJPEG(960x540서 9.1→19.1 fps, 250→123 ms)

- 근거: Source of these values is README.md:256-257 (raw-vs-MJPEG table). No bench artifact exists (c3_camera/bench/ absent). No raw/NV12 recording exists on disk to recompute the 9.1 fps / 250 ms 'raw' arm — I listed every meta.json in c3_camera/recordings/ and all six are color_encode=mjpeg.
- 영향: This is the evidence cited for making MJPEG the default in config.py. The 6:1 compression half of the same sentence IS backed (see separate row); the fps/latency pair is not.

**7. `.claude/journal/research.md:129` — 판정 `b`**

> 판정 지표는 PSNR이 아니라 ORB inlier rate(실측: 압축 8.7 kB까지 밀어도 ORB count는 1833→1816로 안 움직이는데 inlier는 1.000→0.827)

- 근거: c3_camera/encode_quality/ does not exist. grep for '1833' across all .csv/.json/.txt in the repo returns only unrelated telemetry/frames files. The numbers live only in prose: c3_encode_quality.py:423-424 ('measured on this camera's frames') and tests/test_encode_quality.py:260-261 as a hardcoded fixture. No h264/h265/mp4 file exists anywhere under c3_camera/.
- 영향: Presented as '실측'. This pair is the sole justification for choosing ORB-inlier over PSNR as the decision metric. Same provenance class as the +28 mm case: a number that exists only in a docstring written alongside the tool.

**8. `.claude/journal/research.md:129` — 판정 `b`**

> h26x는 호스트 디코드 경로 탓에 코덱 무관 상한 PSNR 45.60/SSIM 0.9962/ORBinl 0.9156(손실 0인 bit-exact 스트림으로 측정)

- 근거: No encode_quality output directory, no .h264/.h265/.mp4 anywhere under c3_camera/. Values appear only in c3_encode_quality.py:119-121 CAVEATS prose and tests/test_encode_quality.py:726 as a fixture. Note the journal quotes 3 more digits (45.60 / 0.9156) than the source, which says 'PSNR 45.6 dB, SSIM 0.9962, ORB inlier 0.916'.
- 영향: The journal explicitly says this finding '결론을 바꿨다' (changed the conclusion) — a decision rests on it. The extra precision in the journal versus the source is itself a provenance smell.

**9. `.claude/journal/reviews.md:31` — 판정 `b`**

> 추종은 소스 무관 거의 동일(DP C 0.33/1.44/1.46 cm)이나 "meas" chatter가 truth의 9–15배(chatF 0.71/10.89/1.18 N/tick), effF ~2–3배

- 근거: verify_state_source.py computes chatF/effF at runtime and prints a table; it saves nothing. No recordings dir holds a 3-source A/B. The numbers exist only in the journal and in dobmpc/params.py prose ('9-15x command chatter, measured').
- 영향: This measurement is what selected MPC_STATE_SOURCE = "meas" as the default and is quoted in the mpc-state-source-switch memory as the boundary criterion. Highest-value item to re-run and archive.

**10. `.claude/journal/reviews.md:33` — 판정 `b`**

> meas chatF 10.9/11.8/12.0 → 1.76/2.91/3.08(truth의 ~2–3배, 이전 ~15배) ... 상태추정 pose~1.6mm. w-RMSE 0.24/0.34/0.39 N(구 0.29/0.41/0.48보다 좋음)

- 근거: No stored verify_state_source or verify_eaob output contains any of these; the only archived w-RMSE numbers are the PNG panel titles (0.552/0.417/0.649 N and 1.112/0.317/0.796 N), which do not match either the old or new triple.
- 영향: params.py restates 'drops "meas" chatter to ~1.8x the truth-x0 level ... (verify_state_source, 2026-07-24)' as fact — a third figure for the same measurement.

**11. `ISAAC_AUV_SETUP_RUNBOOK.md:14` — 판정 `b`**

> (보상 0.78→~76, 에러 0)

- 근거: Parsed the TensorBoard event files under /home/bdml/IsaacLab/logs/rsl_rl/warpauv_direct/ (outside the repo). The 2048-env / 400-iter run (2026-06-23_18-30-12, params/env.yaml num_envs: 2048, params/agent.yaml max_iterations: 400) has Train/mean_reward first=3.4166 @step 0, last=76.5227 @step 399. The value 0.7784 comes from a DIFFERENT run — 2026-06-23_18-28-58, whose params say num_envs: 16 and max_iterations: 2 (a 2-iteration smoke test).
- 영향: The '0.78→~76' progression splices a 16-env 2-iteration smoke run onto the 2048-env 400-iteration run, making the training look like a ~100x improvement when the actual 400-iter run improved 22x (3.42 -> 76.5). Also: no 'error' scalar exists in any of the 12 logged tags, so '에러 0' is not backed by the logs.

**12. `KNOWN_ISSUES.md:59` — 판정 `b`**

> storm PID−MPC −4.24 → +8.60 cm 부호 역전, crossover 소멸

- 근거: Only two storm sweeps exist in the whole repo: recordings/20260724/compare_20260724_005250/wave00_storm and .../compare_20260724_160210/wave00_storm (find -name '*storm*' returns nothing else; no ablation/u_max dir anywhere). Both sweeps have BYTE-IDENTICAL mpc/dobmpc storm results (CW 28.819, CDW 28.402 in both) => both were run under the SAME surge box, so the released-30 N side was never recorded. Aggregate storm PID−MPC = −1.36 cm (005250) and −1.31 cm (160210); per-mode C −1.41/−1.47, CD −1.40/−1.46, CW −2.32/−2.26, CDW −2.12/−1.75. Scanned all 300 per-run pairs in each storm results_raw.csv: max PID−MPC = +8.04 (005250) and +6.77 (160210); nothing at +8.60, and no aggregate at −4.24.
- 영향: NEITHER endpoint of the quoted ablation reproduces from any artifact. This is the highest-severity finding: the same entry uses this number to declare all pre-2026-07-24 strong-wave rankings uncitable, and bluerov2_mujoco_marinegym/README.md:44 plus the memory file mpc-surge-box-benchmark-artifact.md both repeat it. Wording ('ablation: … 부호 역전') presents it as an executed A/B.

## 전체 원장

문서별, `b`/`c` 먼저, severity 높은 순. `claim`은 축약 인용이며 원문은 해당 줄 참조.

### [.claude/agents/hardware-advisor.md](../.claude/agents/hardware-advisor.md)  — a 0 / b 2 / c 4

| 줄 | 판정 | sev | 주장 | 산출물 / 사유 |
|---:|:---:|:---:|---|---|
| 54 | **b** | med | real-world throughput is commonly **15–50 Mbps**, often ~15–20 Mbps effective once video is flowing | Stated as observed field behaviour, in explicit contrast to the vendor number in the same sentence ('~80 Mbps … (vendor's own testing)') which IS attributed. No artifact in this repo measures tether throughput. §12 offers only a blanket, n… |
| 56 | **b** | med | Field data points: a near-identical RGB-D project got only **10 Mbps at 15 m** using cobalt connectors; a GigaBlox + 50 m Fathom tether **failed to l… | Explicitly labelled 'Field data points', i.e. measurements, but these are third-party/forum reports with no per-claim source and no artifact in this repo. Only the generic §12 forum attribution covers them. |
| 258 | **c** | hig | send **compressed stereo + color**, compute depth at the compute node → full-res depth+color in ~15 Mbps | This restates the §4 cheat-sheet's 'Stereo + color combined (compressed) ~12–16 Mbps'. Those compressed rows are estimates: the three uncompressed rows in the same table are exact arithmetic (1280×800×16×30 = 491.5 Mbps, 640×400 = 122.9, 4… |
| 104 | **c** | med | \| 2× mono stereo, H.264 (8-bit, compresses well) \| ~6–10 Mbps \| ✅ \| | Same table as above. Rows 1-3 of the cheat-sheet are exactly recomputable (I verified 491/123/58 Mbps from width×height×16×30); rows 4-6 (~6–10, ~4–6, ~12–16 Mbps) are codec estimates with no derivation and no measurement behind them. The … |
| 124 | **c** | med | **Tuned low-latency pipeline ≈ 30–100 ms.** Default/buffered settings can balloon to 150–250 ms+. | Introduced on line 122 as 'Video pipeline latency (encode → tether → decode), realistic:' with a component breakdown (encode ~10–30 ms, tether ~5–20 ms, decode ~5–20 ms) — wording that reads as observed. No latency artifact exists: c3_came… |
| 144 | **c** | low | The control loop **crosses the tether twice** (~60–150 ms) | Derived by doubling the §5 estimate range; stated as a bare parenthetical fact with no 'estimated'/'approx' framing beyond the tilde, and inheriting the unsourced §5 numbers. |

### [.claude/journal/consults.md](../.claude/journal/consults.md)  — a 38 / b 20 / c 11

| 줄 | 판정 | sev | 주장 | 산출물 / 사유 |
|---:|:---:|:---:|---|---|
| 27 | **b** | hig | NONE(외란 0) 박스 A/B에서 코너 오차가 2.00/1.92/2.04 cm로 8 N·30 N 소수점까지 동일(코너 surge 명령 평균 0.9–1.4 N, 박스 발동 0.7%) | No recording of the box-released A/B exists — the only 20260724 compares are the two 8 N wave sweeps, and neither meta.json carries controller.u_max. Recomputing corner error from the on-disk NONE square runs gives corner-window RMS 1.40–1… |
| 29 | **b** | hig | storm PID 26.14 / MPC 28.82→17.86 / DOB 17.36→7.30, gentle 11.56 / 9.51→7.63 / 3.14→1.87 | The pre-change numbers ARE on disk (wave00_storm CW mpc 28.8188, dobmpc 17.3561; wave05_gentle CW mpc 9.5094, dobmpc 3.1432). The post-change values 17.86 / 7.30 / 7.63 / 1.87 and both PID figures 26.14 / 11.56 are in no recording: no comp… |
| 31 | **b** | hig | storm에서 틱의 60%(mpc)·75%(dobmpc) 물림 vs gentle 7% | The named sweep exists and I opened it, but nothing in results.csv / results_raw.csv / runs/ records constraint-activity fraction (the recorded columns are scenario…n_fail; sat_freq is 0 everywhere). No instrumented run was archived. |
| 31 | **b** | hig | storm PID−MPC −4.24→+8.60 cm 부호역전(MPC 42%↑), gentle +4.01→+4.87(10%↑) | Computed PID−MPC from the named sweep: storm CW 26.5582−28.8188 = −2.26 cm, storm CDW −1.75 cm, gentle CW 14.1264−9.5094 = +4.62, gentle CDW +4.44. Neither the quoted baseline −4.24/+4.01 nor the released-box +8.60/+4.87 is present, and no… |
| 19 | **b** | med | 슬루 지연 1.5→3.2s·199° 오버슈트 | Per-run trajectory CSVs (with yaw_deg) exist, so this is recomputable in principle, but I could not reproduce either number. Unwrapping yaw over all CW/CDW dobmpc runs: slew segments are 2.1–2.5 s typical, 3.85 s worst; cumulative yaw ends… |
| 21 | **b** | med | PD는 일정 토크에 type-0(φ_ss=τ/(kp+B), 0.4N·m서 3.5° 실측)→PID type-1(0.0° 실측) | No recording, log, or test stores a steady-tilt measurement. The only leveling regression on disk, tests/test_heavy_gripper.py::test_allocation_and_pid_hold, asserts a position error < 0.08 m and prints no attitude number. Nothing in recor… |
| 27 | **b** | med | 코너·직선을 거의 같은 비율(33%/37%)로 줄여 ... 박스는 코너 직후 회복을 개선(위상 21.33: 8.78→3.52 cm) ... 직선이 0.6–1.3 cm | Same missing A/B as above — no 30 N-box run exists on disk, so none of these paired percentages or the phase-21.33 pair can be checked. (For scale: on-disk NONE mpc/dobmpc straight-segment RMS is 0.29–0.51 cm, consistent with but not equal… |
| 29 | **b** | med | sat_freq=0 유지·피크 스러스터 40.3 N(<64.1) | The 64.1 N bound is real (bluerov_heavy.xml:128-135 ctrlrange="-51.5507 64.1319"). The measured peak 40.3 N is in no artifact — per-thruster force is not a recorded metric in any results.csv/results_raw.csv, and the verification run was no… |
| 29 | **b** | med | verify_eaob는 기존 FAIL이 오히려 개선(A/B: NIS 245.8→100.0, NEES 398.9→122.0) | The only stored verify_eaob outputs are the five 20260723 PNGs, whose titles read NIS 137.1/NEES 384.2, NIS 25.4/NEES 77.0 (x3), NIS 17.1/NEES 15.0. None of 245.8, 100.0, 398.9 or 122.0 appears. |
| 31 | **b** | med | MPC는 <0.7 rad/s에서 모든 해상상태 3–8배 우세 | Trajectory CSVs exist so a PSD is recomputable in principle, but no PSD output, figure, or script is stored, and the 3–8x band ratio appears nowhere on disk. |
| 12 | **b** | low | tag36h11 격자(실측47 + 격자fill, ~154) | config/tag_map.yaml really does hold 47 surveyed tags (parsed: tags list length 47), so '실측47' is backed. The '~154' total is not: the only generated floors on disk/in git are tag_floor.xml@0b60a6230 ('198 tag36h11 tiles') and the current … |
| 12 | **b** | low | 반투명 수면(4m) | Opened tag_floor.xml at commit 0b60a6230: header says 'seabed z=-0.5, water surface z=1.5 (depth 2.000 m)'. gen_pool_apriltags.py:66 DEF_WATER_DEPTH = 2.0 with the comment that the *physics* h is 'left at 4 m'. No generated scene puts the … |
| 12 | **b** | low | 3000스텝 롤아웃 Δ=0(교란 포함) 역학불변 | The only dynamics-inert rollout test on disk is test_water_viz.py::test_dynamics_inert with n=2000 (test_observe.py uses n=1500). I executed it: '[ok] dynamics inert: max\|delta(qpos,qvel)\| = 0.0e+00 over 2000 steps'. Δ=0 is real; the 300… |
| 19 | **b** | low | 옆흐름 두 코너 비대칭(11.9 vs 7.6) | Scanned all CW/CDW dobmpc runs for corner-window max error. 11.9 cm reproduces (traj_square_CW_dobmpc_seed0_c209.6_w281.3.csv, corners 15.5/5.5/11.9/7.8; traj_square_CDW_dobmpc_seed0_c287.8_w096.1.csv gives 11.9/5.7/9.9/7.4), but no single… |
| 21 | **b** | low | controller.py만 수정, 회귀 5개 테스트 PASS | No test log, CI artifact, or output file exists for this run anywhere in the repo. |
| 22 | **b** | low | pitch guard를 heavy에서 지웠더니 test_controller에서 pitch가 71°까지 스윙 | test_controller.py:117 carries the comment 'The soft pitch guard cuts the transient a lot (~70° -> ~50°)', which is consistent, but no stored run output contains 71°. The test prints max_pitch_deg at run time only. |
| 29 | **b** | low | verify_acados PASS(Δu 0.0615<0.25) | The 0.25 N threshold is in verify_acados.py:63 ('PASS if dmax < 0.25'). The measured 0.0615 is not stored anywhere (verify_acados prints to stdout only). |
| 31 | **b** | low | box30에서도 스러스터 36.1 N/무포화 | No per-thruster force is recorded in any results file, and the box-30 ablation was not archived. |
| 31 | **b** | low | 6 rung뿐이라 a 유의하지 않음 p=0.218 | No statistical output, notebook, or script producing a p-value exists in the repo. |
| 31 | **b** | low | 고정 compliance 비 ω_x=0.89±0.03(2 시뮬 seed 공유) | No artifact contains a compliance-ratio computation or this value. |
| 7 | **c** | med | tether는 Fathom-X ~80Mbps(실측 15–50) | Grepped the whole repo for any tether/Fathom-X throughput measurement; none exists. c3_camera/ measures only the PoE ceiling (~91.5 Mbit/s, a different link). The 15–50 Mbit/s is a third-party/community field figure, not something executed… |
| 14 | **c** | med | 현행 게인은 "파랑 증폭"이 아니라 ωn 0.59–0.88이 대역 내라 무억제(\|S\|≈0.82–1.01, 동작점 d_eff 기준) | Searched the repo for any sensitivity/Bode computation or stored output; none exists (no script, no CSV, no figure). These are hand-computed linear-analysis numbers presented as a verification result ('검증서 정정'). |
| 15 | **c** | med | MPC 모델FF + 3s preview + 유효강성 ~90N/m | No artifact computes an effective stiffness. It is a back-derivation (disturbance force / dc offset). The same quantity is later given as 77.5 N/m (consults.md:23) and 77 N/m (consults.md:33) from the same class of data. |
| 23 | **c** | med | 힘 50 N은 물리 worst(X≈35/Y≈37/Z≈3 N) 위라 우연히 무해 | No artifact records worst-case disturbance wrench magnitudes. The only recorded disturbance traces are recordings/20260723/verify_eaob_perf_traj_*.png (60 s CDW), whose panel ranges peak at X≈13 N, Y≈2 N, Z≈10 N — nowhere near 35/37/3, and… |
| 23 | **c** | med | 2.11 N/k_p 77.5 N/m | 2.11 N is not measured: it is the model sway drag at the configured 0.20 m/s current, recomputed from params.py _LINEAR_DAMPING[1]*0.2 + _QUADRATIC_DAMPING[1]*0.04 = 6.22*0.2 + 21.66*0.04 = 2.110 N. 77.5 N/m is then 2.11/0.0273. For contra… |
| 33 | **c** | med | 2.70 cm = 전류항력 2.11 N / 암묵강성 77 N/m | 2.70 cm is measured (row above), but 2.11 N is the plant-model sway drag at the configured 0.20 m/s (params.py: 6.22*0.2 + 21.66*0.2^2 = 2.110 N) and 77 N/m is 2.11/0.0270. The measured EAOB disturbance estimate in the same runs averages \… |
| 14 | **c** | low | 코드 주석 "heave Kd=22→ζ~0.65"는 added-mass만 쓴 오계산(실제 ζ≈0.60) | The 'ζ~0.65' comment is real (git show 222ea08ed:controller.py:38). The corrected ζ≈0.60 is a closed-form recomputation with no artifact. |
| 23 | **c** | low | 스웨이 보수스택 85–100 N(=blowup 레짐)에선 물리고 | No artifact; analytic stack-up of worst-case sway forcing. |
| 23 | **c** | low | 잔여는 QN=Q 종단경계층 0.3–0.5 mm뿐 | No ablation or run on disk isolates a terminal-boundary-layer offset; this is a turnpike-argument estimate. |
| 27 | **c** | low | 필요 선회율 126–474°/s ≫ 60 | Pure kinematic derivation from the square reference and the 60°/s slew rate; no artifact and none needed. Wording embeds it in a measurement narrative. |
| 33 | **c** | low | PID ωn 2.3–2.8 rad/s가 유한수심 감쇠로 캡핑된 강제대역 0.5–1.3 rad/s를 항상 상회(평탄강성 120–128 N/m) | No artifact computes closed-loop bandwidth, forcing-band limits, or stiffness. These are analytic values derived from GAINS_HEAVY and the wave model. |
| 15 | **a** | hig | C모드 MPC dc_radial 2.75cm(하류방향 ±2°), DOB 0.16, PID는 적분 덕 1.53 | `bluerov2_mujoco_marinegym/recordings/20260702/compare_20260702_222150/results.csv` |
| 15 | **a** | hig | NONE서 이미 17.8 vs 2.1cm(8.6×) | `bluerov2_mujoco_marinegym/recordings/20260702/compare_20260702_222150/results.csv` |
| 33 | **a** | hig | C/CD의 MPC 3.0 cm은 90%가 무적분 dc offset(2.70 cm) | `bluerov2_mujoco_marinegym/recordings/20260724/compare_20260724_005250/wave00_storm/results.csv` |
| 15 | **a** | med | CW band_wave 32.8/6.7/1.9 | `bluerov2_mujoco_marinegym/recordings/20260702/compare_20260702_222150/results.csv` |
| 17 | **a** | med | 이득 상한 −17%(비파도 성분 2.83cm) | `bluerov2_mujoco_marinegym/recordings/20260702/compare_20260702_222150/results.csv` |
| 19 | **a** | med | 무외란서도 전코너 ~5cm | `bluerov2_mujoco_marinegym/recordings/20260707/compare_20260707_170230/runs/traj_square_NONE_dobmpc_seed0_c005.3_w266.9.csv` |
| 23 | **a** | med | nominal 2.73 cm = 2.11 N/k_p 77.5 N/m 정확 일치 | `bluerov2_mujoco_marinegym/recordings/20260720/compare_20260720_221845/results.csv` |
| 33 | **a** | med | NONE은 MPC 승(1.09 vs 1.51, ref preview가 코너 과도 제거) | `bluerov2_mujoco_marinegym/recordings/20260724/compare_20260724_005250/wave00_storm/results.csv` |
| 33 | **a** | med | 전류방향 정렬 −0.2±1.8° | `bluerov2_mujoco_marinegym/recordings/20260724/compare_20260724_005250/wave00_storm/results_raw.csv` |
| 33 | **a** | med | 완만 저하(7.11 cm/m-Hs), MPC는 ... 가파른 저하(12.33 cm/m-Hs) | `bluerov2_mujoco_marinegym/recordings/20260724/compare_20260724_005250/` |
| 33 | **a** | med | storm acados n_fail 77/100 | `bluerov2_mujoco_marinegym/recordings/20260724/compare_20260724_005250/wave00_storm/results_raw.csv` |
| 33 | **a** | med | wave crossover는 moderate(Hs1.0/Tp8)↔rough(Hs1.2/Tp6) 사이 | `bluerov2_mujoco_marinegym/recordings/20260724/compare_20260724_005250/` |
| 9 | **a** | low | 축별 C_a=[0.49,1.12,1.29] | `bluerov2_mujoco_marinegym/dobmpc/params.py` |
| 9 | **a** | low | h=4·Tp=12 → kh≈0.34 천해 | `bluerov2_mujoco_marinegym/config/base.yaml` |
| 11 | **a** | low | beta_bar=0(평균 헤딩 +x) + s=30 좁은 cos^{2s} 분산 | `bluerov2_mujoco_marinegym/config/base.yaml` |
| 11 | **a** | low | ROV는 해저 1m 위(zb=1) | `bluerov2_mujoco_marinegym/disturbance/waves.py` |
| 13 | **a** | low | 매 스텝 애니메이트에도 롤아웃 Δ=0(두 변종·faithful/stylized) | `bluerov2_mujoco_marinegym/tests/test_water_viz.py` |
| 14 | **a** | low | 수평등방 kp/kd/ki=131.6/90.6/38.7, heave 141.8/99.1/41.7, yaw 8.95/4.32/3.95 | `bluerov2_mujoco_marinegym/controller.py` |
| 14 | **a** | low | surge f_max 6→30, e_gate 0.5→0.15, slew 30→120 | `bluerov2_mujoco_marinegym/controller.py` |
| 14 | **a** | low | controller.py:36-48 DEFAULT_GAINS 그대로(kp 6/12/20, kd 8/12/22, ki 1/1.5/1.2, yaw 5/3/0.5) | `bluerov2_mujoco_marinegym/controller.py` |
| 14 | **a** | low | --observe 기본 current는 0이 아니라 0.2 m/s(+x) | `bluerov2_mujoco_marinegym/disturbances.py` |
| 15 | **a** | low | sat_freq=0 전구간 | `bluerov2_mujoco_marinegym/recordings/20260702/compare_20260702_222150/results.csv` |
| 15 | **a** | low | PID 500Hz vs MPC 20Hz | `bluerov2_mujoco_marinegym/dobmpc/params.py` |
| 15 | **a** | low | run은 heavy(PID Mx=My=0 → pitch 방치 ±12–42°) | `bluerov2_mujoco_marinegym/recordings/20260702/compare_20260702_222150/results.csv` |
| 17 | **a** | low | w_hat을 60 스테이지 ZOH(mpc_acados.py:171-173; per-stage p 인프라는 존재) | `bluerov2_mujoco_marinegym/dobmpc/mpc_acados.py` |
| 19 | **a** | low | 상류 edge 0.35m/s가 하류 0.05의 ~7배 | `bluerov2_mujoco_marinegym/config/base.yaml` |
| 21 | **a** | low | ki=I_eff·α·ωn³(roll2.60/pitch4.64) ... 3차극 전부 LHP ζ≈0.83-0.89 | `bluerov2_mujoco_marinegym/controller.py` |
| 22 | **a** | low | test_controller pitch 임계 heavy 재보정(뒤집힘<90°+복귀<8°) | `bluerov2_mujoco_marinegym/tests/test_controller.py` |
| 22 | **a** | low | bluerov.xml/BlueROV.yaml은 8개 hydro/thruster 검증 스크립트+heavy 관성 provenance의 fixture | `bluerov2_mujoco_marinegym/` |
| 23 | **a** | low | mpc_acados w_hat ±50 클립 | `bluerov2_mujoco_marinegym/dobmpc/mpc_acados.py` |
| 23 | **a** | low | 토크 50 N·m은 authority 8–10의 5–6배로 부적절 | `bluerov2_mujoco_marinegym/dobmpc/params.py` |
| 23 | **a** | low | 관측 DOB 잔여 1.7 mm는 EAOB lag 지배 | `bluerov2_mujoco_marinegym/recordings/20260720/compare_20260720_221845/results.csv` |
| 23 | **a** | low | calm 코너는 제약-제한(surge 8 N derate vs PID 30 N) | `bluerov2_mujoco_marinegym/dobmpc/params.py` |
| 29 | **a** | low | heavy `U_MAX[0]` 8→30 적용(rank-5 분기는 8 유지) | `bluerov2_mujoco_marinegym/dobmpc/params.py` |
| 31 | **a** | low | 주파수 분해(전 1500런 PSD) | `bluerov2_mujoco_marinegym/recordings/20260724/compare_20260724_160210/wave00_storm/results_raw.csv` |
| 31 | **a** | low | U_MAX[0]=8 N(PID f_max=30의 1/3.75) | `bluerov2_mujoco_marinegym/dobmpc/params.py` |
| 33 | **a** | low | dobmpc 반사실 97% 회복 | `bluerov2_mujoco_marinegym/recordings/20260724/compare_20260724_005250/wave00_storm/results.csv` |
| 33 | **a** | low | sat_freq=0 전 구간(스러스터 물리포화 아님, 설계 박스) | `bluerov2_mujoco_marinegym/recordings/20260724/compare_20260724_005250/` |

### [.claude/journal/research.md](../.claude/journal/research.md)  — a 29 / b 13 / c 5

| 줄 | 판정 | sev | 주장 | 산출물 / 사유 |
|---:|:---:|:---:|---|---|
| 127 | **b** | hig | 핵심 제약은 XLink-over-PoE 실효 천장 ≈91.5 Mbit/s(서로 다른 4구성이 86/91.7/91.1/91.6로 수렴 → 링크 천장이지 연산 한계 아님) | c3_camera/bench/ does not exist; c3_bench.py writes no CSV anywhere in the repo. The four cited rows are NV12-raw configurations (README.md:249-257 table); every recording on disk (c3_camera/recordings/20260729_162216 and 20260803_*) is MJ… |
| 127 | **b** | hig | 최대 레버는 디바이스측 MJPEG(960x540서 9.1→19.1 fps, 250→123 ms) | Source of these values is README.md:256-257 (raw-vs-MJPEG table). No bench artifact exists (c3_camera/bench/ absent). No raw/NV12 recording exists on disk to recompute the 9.1 fps / 250 ms 'raw' arm — I listed every meta.json in c3_camera/… |
| 129 | **b** | hig | 판정 지표는 PSNR이 아니라 ORB inlier rate(실측: 압축 8.7 kB까지 밀어도 ORB count는 1833→1816로 안 움직이는데 inlier는 1.000→0.827) | c3_camera/encode_quality/ does not exist. grep for '1833' across all .csv/.json/.txt in the repo returns only unrelated telemetry/frames files. The numbers live only in prose: c3_encode_quality.py:423-424 ('measured on this camera's frames… |
| 129 | **b** | hig | h26x는 호스트 디코드 경로 탓에 코덱 무관 상한 PSNR 45.60/SSIM 0.9962/ORBinl 0.9156(손실 0인 bit-exact 스트림으로 측정) | No encode_quality output directory, no .h264/.h265/.mp4 anywhere under c3_camera/. Values appear only in c3_encode_quality.py:119-121 CAVEATS prose and tests/test_encode_quality.py:726 as a fixture. Note the journal quotes 3 more digits (4… |
| 98 | **b** | med | 과제성능(model_1250 vs 원본 model_399): 자세오차 중앙값 73°→29°, 위치 0.235→0.206m, 엄격성공(≤0.1m·≤10°) 0%→0.26% | The training artifacts DO exist and I read them (/home/bdml/IsaacLab/logs/rsl_rl/warpauv_direct/*/events.out.tfevents.*), which confirmed every reward number in the same journal line. But there is no evaluation artifact: no eval CSV/JSON u… |
| 124 | **b** | med | calm 격차=MPC 코너 과도(edge 동등 0.20 vs 0.14 cm; effort-페널티 가설 반박) | Artifact exists (recordings/20260720/compare_20260720_221845/runs/traj_square_NONE_{pid,mpc,dobmpc}_seed0_*.csv, which carry px,py,rx,ry) but I could not locate 0.14 under ANY edge definition I tried: corner-distance thresholds 0.15/0.2/0.… |
| 127 | **b** | med | 컬러 지연 3000 ms → 38 ms(p50) | The 38 ms half is backed (see separate row). The 3000 ms Madrona/RTSP baseline is not: grep '3000' across c3_camera/ finds it only in README.md:215,223 and README_DATASET.md:6, and c3_camera/madrona.py contains no latency measurement code … |
| 128 | **b** | med | 배칭이 손실 결정 — 4스트림 동시 녹화서 batch=1이면 200 Hz 요청→130 Hz(35% 손실) | I read the imu_camera.csv + metadata.json of all 10 datasets. Achieved IMU rates on disk are 200.4, 153.0, 199.7, 199.5, 199.3, 199.5, 199.4, 199.7, 199.7, 199.6 Hz. No dataset shows ~130 Hz. The only 4-stream dataset (dataset_20260729_173… |
| 128 | **b** | med | `pair_mode=latest`라 프레임당 ~3 bundle을 다 써서 파일명 중복·덮어쓰기·인덱스 비단조(8 fps인데 26 Hz 기록, left/right 159 vs rgb 516) | No dataset on disk exhibits this. The only 4-stream dataset (c3_camera/datasets/dataset_20260729_173152) has rgb/left/right/depth = 227/227/227/227 at fps_requested 8.0, fps_achieved 8.04 — i.e. the fixed state. No dataset anywhere shows 5… |
| 129 | **b** | med | DECISION 한 줄이 산포를 안 보여 동전던지기가 결론처럼 읽힘(마진 +0.0042 vs sd 0.0089) | No encode_quality artifact. The pair exists only as a hand-built fixture in tests/test_encode_quality.py:706-709, where a synthetic row is constructed with orb_inlier_sd=0.0089 and the test asserts the formatted margin string contains '+0.… |
| 19 | **b** | low | 오늘자 로그(`kit_20260623_092407.log`, 366KB/2132줄 등 다수) 전부 "Simulation App Startup Complete" 도달 후 PhysX CUDA context에서 멈춤. warp 시작은 98ms로 정상. | Searched the filesystem: no kit_20260623_092407.log, no kit_*.log anywhere (find / -maxdepth 6). ~/.nvidia-omniverse/logs/Kit/ does not exist. The 366KB / 2132 lines / 98 ms figures cannot be re-checked. (This same log is cited again at li… |
| 127 | **b** | low | 산출물: `c3_camera/`(discover_c3/c3_stream/c3_bench + 재사용 패키지 + 카메라 없이 도는 테스트 23개) | git log shows c3_camera/ has exactly one commit (90df32adb, 2026-08-03), so no intermediate state exists. The committed tests/test_offline.py has 46 test functions; the working tree has 47. The 2026-07-29 count of 23 cannot be reconstructe… |
| 130 | **b** | low | 문서의 31개 명령 전부를 argparse에 통과시키는 하네스 작성(C3Source 스텁 + --dry-run 강제, 하드웨어 미접촉) → 30/31 통과 | The harness is not in the repo: no file matching *harness*/*doc_command*/*testing_md*, and no .py anywhere references TESTING.md except host_depth.py:459 (an unrelated line citation). I counted invocations in TESTING.md — 23 lines match `c… |
| 129 | **c** | hig | quantisation-limited rung이 gate 전부 초록으로 통과(2 m/400p dZ=141 mm, 벽이 한 disparity bin에 들어가 resid 0.00·inlier 1.00인데 bias +28 mm) | c3_camera/depth_accuracy/rungs.csv (DEFAULT_CSV of c3_depth_accuracy.py) does not exist; no c3_camera/depth_accuracy/ dir at all. c3_camera/TESTING.md:334 states outright '캡처 경로는 한 번도 하드웨어에서 돌아본 적이 없다' (the capture path has NEVER been run … |
| 131 | **c** | med | `data/camera0_main_depth`가 카메라로 등록됨(실측: available_cameras에 depth 포함, get_frame_shape가 (64,1,48) 반환) | The code half is verified — I opened /home/bdml/Desktop/UMI_underwater/affordance/pipelines/legacy/gripper_pipeline/zarr_adapter.py:52-54 and it does scan `if 'camera' in key.lower()`. But no zarr on this machine has a key named camera0_ma… |
| 11 | **c** | low | 결과: 2048 envs·400 iter·~19.7M steps GPU 학습 완주(보상 0.78→~76) | Read /home/bdml/IsaacLab/logs/rsl_rl/warpauv_direct/*/events.out.tfevents.* with tensorboard EventAccumulator. The 400-iteration run is 2026-06-23_18-30-12 (seed 42, 400 scalars): its Train/mean_reward starts at 3.417 and ends at 76.52. Th… |
| 128 | **c** | low | 병목은 디스크가 아니라 PoE 링크(NVMe 10 GB/s vs 링크 11 MB/s; 무손실 4스트림 상한 480x270서 11.8 fps) | 11.8 fps is a formula output, not a measurement: dataset.py:995 computes link_limited_fps = (91.5e6/8)/per_frame_wire. I evaluated it for 480x270 lossless 4-stream (color NV12 194400 + depth16 259200 + 2x mono 640x400 256000 = 965600 B) → … |
| 131 | **c** | low | shipped `global_max 0.32258`에서 ×3.10 게인 + clamp 포화(실측 (0,0.4993)→(0,1.0)) | global_max IS shipped and I confirmed it: 0.32257768511772156 appears in /home/bdml/Desktop/UMI_underwater/affordance/training/legacy/umi_day/train_network/config/train_multi_primitive_affordance_simple_depth.yaml:27 and affordance/configs… |
| 98 | **a** | hig | 5개 run이 TensorBoard reward 곡선상 비트 단위 동일(maxdiff=0.00) | `/home/bdml/IsaacLab/logs/rsl_rl/warpauv_direct/` |
| 98 | **a** | hig | reward 76(400it)→93(800it)→피크 97.5(@1347)→90.8(1600, 피크후 하락). ... 최고 저장 체크포인트=model_1250.pt(reward 96.2) | `/home/bdml/IsaacLab/logs/rsl_rl/warpauv_direct/` |
| 126 | **a** | hig | 코너 부호 39/39 outward→39/39 inward(선행 컷, 이론 예측 그대로) | `bluerov2_mujoco_marinegym/recordings/20260721/compare_20260721_111349/runs/` |
| 126 | **a** | hig | NONE 2.09/2.12→1.06(PID 1.38 역전, 전 5모드 radial_rms 우위) ... CDW 3.46→2.66(48/50 개선) | `bluerov2_mujoco_marinegym/recordings/20260721/compare_20260721_111349/results_raw.csv` |
| 122 | **a** | med | heavy_c3=heavy+C3 변종 신설 ... test_heavy_c3.py 전부 통과(13.2kg, −3.1N, 카메라3 전방, 그리퍼 액추에이터 없음, PID 0.0cm) | `bluerov2_mujoco_marinegym/tests/test_heavy_c3.py` |
| 124 | **a** | med | C-offset 2.73→DOB 0.17(절대속도-drag 미모델) | `bluerov2_mujoco_marinegym/recordings/20260720/compare_20260720_221845/results.csv` |
| 125 | **a** | med | n_fail census 217run/289회, mpc 14% vs dobmpc 7.7%(EAOB가 OCP 안정화) | `bluerov2_mujoco_marinegym/recordings/20260720/compare_20260720_230025/results_raw.csv` |
| 125 | **a** | med | 큰 내부침투=seed-0 공통 파랑그룹의 lap7/8 V3 이벤트(181/400 run이 t=200-210s 피크, 단일실현 아티팩트)+최심부는 n_fail 증폭(>40cm 23/23 fail) | `bluerov2_mujoco_marinegym/recordings/20260720/compare_20260720_230025/runs/` |
| 127 | **a** | med | 컬러 지연 ... 38 ms(p50), 15 요청→15.00 실측, 드롭 0% | `c3_camera/recordings/20260729_162216/meta.json` |
| 128 | **a** | med | C3에 온보드 IMU 있음 = BNO086 fw 3.9.9 ... dai.Clock.now() ≡ time.monotonic()(CLOCK_MONOTONIC, 8µs 이내) ... depth_scale 5000 → 1000 | `c3_camera/datasets/dataset_20260729_181831/metadata.json` |
| 128 | **a** | med | accel seq는 정상인데도 건너뜀(953중 277이 delta 2) → gyro seq로 연속성 판정 | `c3_camera/datasets/dataset_20260729_181831/imu_camera.csv` |
| 129 | **a** | med | 컬러 MJPEG는 링크의 14%뿐이고 depth uint16이 86%라 컬러를 0으로 만들어도 fps 이득 0 | `c3_camera/datasets/dataset_20260803_105221/metadata.json` |
| 129 | **a** | med | 카메라 없는 테스트 46→140개(46+58+36) 전부 통과 | `c3_camera/tests/` |
| 113 | **a** | low | 실제 연안/조석 난류 외력 = O(0.1–10 N) (sim 자체 계수 Xu_dot=5.5, Xuu=18.18로 재유도) | `bluerov2_mujoco_marinegym/marinegym_assets/BlueROVHeavy.yaml` |
| 116 | **a** | low | 프로젝트 EAOB도 w_dot=0(eaob.py) → 0.15s 임펄스는 가정 최대 위반 ... (eval_dp.py도 "0.15s kicks not rejected"라 기록) | `bluerov2_mujoco_marinegym/dobmpc/eval_dp.py` |
| 118 | **a** | low | ② 20–50N 크기·0.15s 지속은 환경외란 anchor 없음(엔지니어링 default) | `bluerov2_mujoco_marinegym/disturbances.py` |
| 124 | **a** | low | 코너 스파이크=일반 MPC 과도(+wave 상호작용, 이득 86→63%, 역전 없음) | `bluerov2_mujoco_marinegym/recordings/20260720/compare_20260720_221845/runs/` |
| 125 | **a** | low | CW/CDW 순서 뒤집힘(3.33 vs 3.41)으로 노이즈 확정 | `bluerov2_mujoco_marinegym/recordings/20260720/compare_20260720_230025/results.csv` |
| 126 | **a** | low | n_fail 10-13→1-2 | `bluerov2_mujoco_marinegym/recordings/20260721/compare_20260721_111349/results_raw.csv` |
| 126 | **a** | low | 최악 70.8cm는 pitch 70° ... mpc엔 신규 210cm blowup ... >20cm 사건 62/62 | `bluerov2_mujoco_marinegym/recordings/20260721/compare_20260721_111349/results_raw.csv` |
| 127 | **a** | low | C3 = OAK-D-W-POE(CAM_A IMX378 고정초점 / CAM_B·C OV9282, baseline 7.5 cm, EEPROM 캘리브 존재, IR 프로젝터 없음, bootloader 0.0.28) | `c3_camera/recordings/20260729_162216/meta.json` |
| 127 | **a** | low | 실측 압축 6:1이라 초기 10:1 추정은 대역폭 절반 과소평가 → MJPEG_RATIO에 실측값 고정 | `c3_camera/recordings/20260729_162216/meta.json` |
| 127 | **a** | low | conda base는 python 3.12.11 + depthai 3.5.0(v3) | Ran /home/bdml/miniforge3/bin/python: Python 3.12.11 (conda-forge), depthai 3.5.0. Exact. |
| 128 | **a** | low | batch=10이면 199.5 Hz 손실 0(원인=메시지당 XLink 오버헤드) | `c3_camera/datasets/dataset_20260730_173057/metadata.json` |
| 128 | **a** | low | 정지 \|a\| 11.8 m/s²(중력 대비 +20%) | `c3_camera/datasets/dataset_20260729_181831/metadata.json` |
| 129 | **a** | low | 인코더가 못 보는 축은 fold해 24→9런 | `c3_camera/c3_bench.py` |
| 130 | **a** | low | T1의 mono_res 선택이 T2의 최소거리를 직접 구속(400p MinZ 298.9 mm vs 720p/800p 597.7 mm — 두 배) | `c3_camera/tests/test_depth_accuracy.py` |
| 131 | **a** | low | depthai 2.32.0.0에 `DepthAlign.RECTIFIED_LEFT=1` 존재(실측 열거) ... cv2 4.11엔 estimatePoseSingleMarkers가 이미 제거됨(실측) | Ran in ~/.venvs/c3-depthai: depthai 2.32.0.0, dai.RawStereoDepthConfig.AlgorithmControl.DepthAlign.RECTIFIED_LEFT exist… |
| 131 | **a** | low | `goal_end_random`의 `min()`(`DS:506`)은 증명 가능한 dead code ... `goal_key` grep 0건 ... repo 전체 savgol/medfilt grep 0건 | multi_primitive_train_dataset.py:500-511: inside the `current_frame >= tail_start` branch, tail_start = episode_end - w… |

### [.claude/journal/reviews.md](../.claude/journal/reviews.md)  — a 18 / b 17 / c 2

| 줄 | 판정 | sev | 주장 | 산출물 / 사유 |
|---:|:---:|:---:|---|---|
| 31 | **b** | hig | 추종은 소스 무관 거의 동일(DP C 0.33/1.44/1.46 cm)이나 "meas" chatter가 truth의 9–15배(chatF 0.71/10.89/1.18 N/tick), effF ~2–3배 | verify_state_source.py computes chatF/effF at runtime and prints a table; it saves nothing. No recordings dir holds a 3-source A/B. The numbers exist only in the journal and in dobmpc/params.py prose ('9-15x command chatter, measured'). |
| 33 | **b** | hig | meas chatF 10.9/11.8/12.0 → 1.76/2.91/3.08(truth의 ~2–3배, 이전 ~15배) ... 상태추정 pose~1.6mm. w-RMSE 0.24/0.34/0.39 N(구 0.29/0.41/0.48보다 좋음) | No stored verify_state_source or verify_eaob output contains any of these; the only archived w-RMSE numbers are the PNG panel titles (0.552/0.417/0.649 N and 1.112/0.317/0.796 N), which do not match either the old or new triple. |
| 21 | **b** | med | pitch 텀블(46.8°/58.7° > 45°) 테스트 실패 | No test log or output file exists. The 45° threshold itself no longer exists in tests/test_controller.py (it was re-baselined to <90°/<8°, see consults.md:22), so the number cannot be reproduced by re-running either. |
| 27 | **b** | med | deadbeat 실측 6.8–13.4 ms | No artifact stores an observer time-constant measurement. dobmpc/params.py repeats it as 'w-error time constant 7-13 ms; measured 2026-07-23' — a doc comment citing the same unarchived run, with a different lower bound. |
| 27 | **b** | med | 25–76% 기각으로 파랑 추종 차단(CDW 4.5→15.5 cm) | The gate-ON plot shows 'gated 347/1200' = 28.9%, inside the quoted range but the only point evidenced; 76% appears nowhere. The tracking pair 4.5→15.5 cm is in no results.csv (compare_20260723_172539 is a PID-only run). |
| 27 | **b** | med | 헤드라인 보존: DP C dc −0.11 cm vs MPC +2.13 | No compare recording from 2026-07-23 contains dp-scenario dobmpc/mpc results (compare_20260723_172539 is PID-only, and its config has no dp block with these values). |
| 27 | **b** | med | 정직한 비용: square CD 1.02→1.21, CDW 3.09→3.51 cm | No archived compare has dobmpc square CD/CDW at these values for the pre/post retune pair; nearest on-disk dobmpc square CD values are 2.03 (20260702) and 1.09 (20260721). params.py restates 3.51 vs 3.09 as fact. |
| 29 | **b** | med | 구파랑 A/B 전부 PASS: DP C 0.33→1.46, CDW 1.51→2.22 cm; square CDW 3.51→3.92 cm | verify_state_source.py writes no files (stdout table only) and no 12-run A/B artifact exists in recordings/. Checked every compare dir: none contains these numbers. |
| 29 | **b** | med | 현 config CDW는 소스 무관 NIS 41–80 실패 | No stored verify_eaob/verify_state_source output contains NIS in the 41–80 range (archived values are 17.1, 25.4, 137.1). |
| 33 | **b** | med | sigma 스케일 스윕(pos 5/2/1/0.5/0.25 cm, square C meas)으로 chatter/tracking/NIS 곡선 확인 후 x/y 0.5 cm 선택(추종오차 ~1.2 cm의 0.4배) | The chosen value is backed: params.py EAOB_SIG_POS = [0.005, 0.005, 0.003] (x/y 0.5 cm). The sweep itself — five sigma rungs with chatter/tracking/NIS curves — has no artifact: no CSV, figure, or recordings dir. |
| 33 | **b** | med | NIS/NEES는 ~13/9로 목표 18 아래=보수적 ... 게이트를 과신(상한 24)만 판정하도록 변경 | The gate change is real and confirmed: verify_eaob.py:60-63 NIS_UPPER = 24.0, NEES_UPPER = 24.0, informational band (14,24), target ~18. The measured 13/9 pair is in no artifact (only in code comments). |
| 23 | **b** | low | _prebuild_acados가 dobmpc 전용이라 mpc-only 병렬런이 tera 경합→IPOPT 폴백(20배 느림) → mpc/dobmpc 공용 프리빌드로 수정(107.6s→5.7s/run) | No timing log, benchmark output, or meta field records per-run wall time for the before/after. (results_raw.csv has a 'wall' column, but no run pair spanning this fix is archived.) |
| 23 | **b** | low | 전부 수정 후 40런 e2e 재검증 PASS | No 40-run compare directory or log exists in recordings/ for 2026-07-07 with that shape. |
| 27 | **b** | low | CD NIS 14.4 PASS | All five archived verify_eaob PNGs show NIS 137.1, 25.4 (x3) and 17.1. No 14.4 appears; there is no CD-mode plot. |
| 27 | **b** | low | CD NEES ~9(<14)는 실기체 보수성 | No CD-mode verify_eaob artifact exists (all five PNGs are CDW-shaped runs); the value is only in params.py prose. |
| 31 | **b** | low | 발산 없음/포화 0/n_fail 0/EAOB yaw 연속(\|Δψ̂\|≤0.08) | Same missing A/B artifact; no file records a per-tick yaw-increment bound. |
| 37 | **b** | low | `bytes.find` 단일 스캔(232x 빠름, 20만 랜덤+짧은문자열 전수에서 출력 동일 확인) | No benchmark script, log, or output file for the Annex-B scan comparison exists anywhere under c3_camera/ (or the repo). The fix itself is real, but the 232x figure and the 200k-case equivalence sweep are unarchived. |
| 13 | **c** | low | 관성이 "don't care"→"~10-20% 중요"로 승격 | No sensitivity sweep exists in the repo (the review itself lists that sweep as recommendation ① — i.e. it had not been run). |
| 33 | **c** | low | run_compare 4500런 규모/2700 낭비(NONE/C/CD 파도무관 중복) | A planning estimate derived from the config's sweep dimensions, not an executed count. (The sweeps actually recorded 1500 runs per wave x 6 waves.) |
| 27 | **a** | med | τ 0.5 CDW 실패(NIS 25.4/NEES 77) | `bluerov2_mujoco_marinegym/recordings/20260723/verify_eaob_perf_traj_174602.png` |
| 27 | **a** | med | 0.2 확정(CDW 17.1/15.0 PASS | `bluerov2_mujoco_marinegym/recordings/20260723/verify_eaob_perf_traj_174953.png` |
| 7 | **a** | low | Heavy 관성 텐서 [0.3291,0.6347,0.6109] (parallel-axis 유도, farol USD 대신) | `bluerov2_mujoco_marinegym/tools/compute_heavy_inertia.py` |
| 10 | **a** | low | farol [0.21,0.245,0.245] 기각 | `bluerov2_mujoco_marinegym/rov_model.py` |
| 10 | **a** | low | 0.15kg/thruster는 예산 bookkeeping(실제 T200 ~0.344kg)이나 delta만 스케일 → 영향 작음(~0.03) | `bluerov2_mujoco_marinegym/tools/compute_heavy_inertia.py` |
| 11 | **a** | low | 회전 added mass [0.12,0.12,0.12] (BlueROVHeavy.yaml)는 등방 placeholder | `bluerov2_mujoco_marinegym/marinegym_assets/BlueROVHeavy.yaml` |
| 25 | **a** | low | 샘플러↔라이브루프 동치 회귀 테스트 추가(test_square_ref_matches_live_loop, 1랩 <1e-9) | `bluerov2_mujoco_marinegym/tests/test_dobmpc.py` |
| 27 | **a** | low | 게이트 ON 측정은 NIS 통계도 오염(137 vs 25) | `bluerov2_mujoco_marinegym/recordings/20260723/verify_eaob_perf_traj_172118.png` |
| 29 | **a** | low | base.yaml 파랑이 19:28에 강화(Hs 1.2/Tp 6) | `bluerov2_mujoco_marinegym/config/base.yaml` |
| 33 | **a** | low | base.yaml에 6-스펙트럼 sea-state 사다리(storm 1.8→gentle 0.75) 추가 | `bluerov2_mujoco_marinegym/recordings/20260724/compare_20260724_005250/meta.json` |
| 35 | **a** | low | reference square를 zorder 6/6.5로 올리자 RMS 주석 ax.text(기본 zorder 3)가 ... 아래로 내려가 ... → ax.text 2곳 zorder=10로 수정 | `bluerov2_mujoco_marinegym/experiments/run_compare.py` |
| 37 | **a** | low | 기존 186 테스트 전부 통과 | `c3_camera/tests/` |
| 37 | **a** | low | 신규 `tests/test_option_sweep.py` 32개 | `c3_camera/tests/test_option_sweep.py` |
| 37 | **a** | low | `--min-free-gb`가 `written % 40`(3개 writer 스레드가 증가시키는 카운터)라 사실상 미발동 → 2초 wall-clock 주기 | `c3_camera/c3_option_sweep.py` |
| 37 | **a** | low | close()가 첫 줄에서 플래그 세팅 후 최대 120s flush | `c3_camera/dataset.py` |
| 37 | **a** | low | `Camera.fps: {fps:g}`가 7.5 같은 실수 노드를 뱉어 ORB-SLAM3 `readParameter<int>`가 exit(-1) → 정수 고정 | `c3_camera/dataset.py` |
| 37 | **a** | low | depthai 2.32에 `ImgFrame.getFrameType()`이 없어서(hasattr False 실측) frame_type이 항상 "" | `c3_camera/source.py` |
| 37 | **a** | low | RGB-D yaml에 `Stereo.ThDepth`/`Stereo.b` 누락 → 추가(... 캘리 실패 시 공칭 75mm 명시) | `c3_camera/dataset.py` |

### [ISAAC_AUV_SETUP_RUNBOOK.md](../ISAAC_AUV_SETUP_RUNBOOK.md)  — a 12 / b 3 / c 1

| 줄 | 판정 | sev | 주장 | 산출물 / 사유 |
|---:|:---:|:---:|---|---|
| 14 | **b** | hig | (보상 0.78→~76, 에러 0) | Parsed the TensorBoard event files under /home/bdml/IsaacLab/logs/rsl_rl/warpauv_direct/ (outside the repo). The 2048-env / 400-iter run (2026-06-23_18-30-12, params/env.yaml num_envs: 2048, params/agent.yaml max_iterations: 400) has Train… |
| 186 | **b** | med | README 기준: 2048 envs로 **~400 iter**에 수렴, mean reward ~95–100. | This repeats the upstream warplab README claim. The local 2048-env / 400-iter run (2026-06-23_18-30-12) reached Train/mean_reward = 76.52 at iteration 399 (max over the run 80.26), not 95-100. Reward only reached 93-97 in the longer 800- a… |
| 69 | **b** | low | 설치 위치 제안: 홈 디렉토리 (디스크 625GB 여유로 충분) | No artifact records the disk state at writing time. `df -h ~` today reports 540G available on /dev/nvme0n1p2 (1.8T, 69% used). |
| 329 | **c** | low | IOMMU off 안 해도 **매 부팅마다 ~5–6분 P2P 검증을 기다리면** 통과한다(무한 행 아님) | 5-6 min is a unit conversion of the '~306–362s @0.2GB/s' figure on line 319, which the doc itself attributes to IsaacLab issue #1764. /proc/cmdline on this machine contains amd_iommu=off, i.e. the workaround was applied, so the wait-it-out… |
| 14 | **a** | med | WarpAUV 정책 **2048 envs · 400 iter · ~19.7M steps GPU 학습 완료** | /home/bdml/IsaacLab/logs/rsl_rl/warpauv_direct/2026-06-23_18-30-12/ has model_0.pt … model_399.pt, params/env.yaml num_… |
| 16 | **a** | low | 드라이버 = `nvidia-driver-570-open` 570.211.01 (CUDA 12.8) | nvidia-smi --query-gpu=name,driver_version reports 'NVIDIA GeForce RTX 5090, 570.211.01'. Confirmed on the live system. |
| 17 | **a** | low | `…/isaacsim/extscache/omni.warp.core-1.7.1+lx64/warp/`를 warp-lang 1.8.1 **실복사본**으로 교체 … 백업 `warp.1.7.1.bak` | grep of $ENV/lib/python3.11/site-packages/isaacsim/extscache/omni.warp.core-1.7.1+lx64/warp/config.py -> 'version: str … |
| 187 | **a** | low | `experiment_name = "warpauv_direct"` (agents/rsl_rl_ppo_cfg.py), max_iterations=400, num_steps_per_env=24 | `bluerov2_issac_paper/agents/rsl_rl_ppo_cfg.py` |
| 189 | **a** | low | num_envs 가이드 (VRAM 32GB) … 32 GB (내 PC) | nvidia-smi memory.total = 32607 MiB on the RTX 5090. |
| 193 | **a** | low | 모델이 64×64 MLP·obs 17·act 6로 매우 가벼워 | `bluerov2_issac_paper/agents/rsl_rl_ppo_cfg.py` |
| 197 | **a** | low | env cfg의 scene 기본은 `num_envs=4`(warpauv_env.py:67) | `bluerov2_issac_paper/warpauv_env.py` |
| 225 | **a** | low | `custom_workflows/play_eval.py` — 12개 axis-direction sweep | `bluerov2_issac_paper/custom_workflows/play_eval.py` |
| 234 | **a** | low | `warpauv_env.py:70-72` — Box(17), Box(6, ±1), Box(17) | `bluerov2_issac_paper/warpauv_env.py` |
| 248 | **a** | low | **차량은 WARPAUV** (22.7kg, 6-thruster fully-actuated) | `bluerov2_issac_paper/warpauv_env.py` |
| 266 | **a** | low | `isaacsim==5.0.0.0` + `torch 2.7.0+cu128` (RTX 5090 `sm_120` matmul 검증됨) | env_isaaclab python: torch 2.7.0+cu128, get_arch_list() includes 'sm_120' and 'compute_120', device 'NVIDIA GeForce RTX… |
| 268 | **a** | low | `rsl-rl-lib==2.3.3` | pip show rsl-rl-lib in env_isaaclab -> Version: 2.3.3. |

### [KNOWN_ISSUES.md](../KNOWN_ISSUES.md)  — a 18 / b 13 / c 2

| 줄 | 판정 | sev | 주장 | 산출물 / 사유 |
|---:|:---:|:---:|---|---|
| 59 | **b** | hig | storm PID−MPC −4.24 → +8.60 cm 부호 역전, crossover 소멸 | Only two storm sweeps exist in the whole repo: recordings/20260724/compare_20260724_005250/wave00_storm and .../compare_20260724_160210/wave00_storm (find -name '*storm*' returns nothing else; no ablation/u_max dir anywhere). Both sweeps h… |
| 241 | **b** | hig | **surge 박스 8→30 N에서 소수점까지 불변**(코너 명령 surge는 평균 0.9–1.4 N으로 박스 근처도 안 감, 발동 0.7%) | The per-run trajectory CSVs (recordings/20260724/compare_20260724_160210/wave05_gentle/runs/traj_*.csv) have columns t,px,py,pz,rx,ry,yaw_deg,pitch_deg,lap only — no control/thrust channel at all, so commanded surge (0.9–1.4 N) and box act… |
| 70 | **b** | med | 힘 채널 50 N은 이 환경 물리 상한(realistic X≈35 / Y≈37 / Z≈3 N)보다 위라 | COULD NOT VERIFY. These are disturbance-wrench bounds, not thruster limits, so they must come from evaluating the hydro/disturbance model; no script output, CSV, or saved sweep in the repo contains them, and I could not evaluate the model … |
| 82 | **b** | med | 프레임을 25–76% 기각해 파랑 추종 자체를 차단(CDW radRMS 4.5→15.5 cm, CD 1.35→1.43 cm) | The rejection-fraction side is partly supported: recordings/20260723/verify_eaob_perf_traj_172118.png title reads 'gated 347/1200' = 28.9%, inside the quoted 25–76%. The radRMS A/B pairs are not supported — the saved PNGs plot w_hat vs w_t… |
| 83 | **b** | med | NIS 통계도 오염(137 vs gate-off 25) | '137' IS confirmed: recordings/20260723/verify_eaob_perf_traj_172118.png title reads 'NIS 137.1 NEES 384.2 gated 347/1200'. The paired gate-off value 25 is not found — the gate-off PNG I opened (verify_eaob_perf_traj_174953.png) reads 'NIS… |
| 145 | **b** | med | heavy_gripper(13.7 kg)에서 0.2717 N — 30 N sway authority의 ~0.9%라 실효 동일 최적해, 폐루프도 검증됨(DP hold 1.3 cm) | Two of the three numbers ARE sourced: the 0.25 N gate is in verify/verify_acados.py:63 ('PASS' if dmax < 0.25 else 'CHECK'), and 13.7 kg is confirmed by running bluerov2_mujoco_marinegym/compute_payload_inertia.py (baked mass 13.6640 kg, t… |
| 186 | **b** | med | NIS 80(truth)/71(estimate) (DP), 44/42(square) — τ_dist=0.2로 검증했던 목표범위(14–24) 크게 이탈, radRMS도 DP 1.5→9.6 cm / square 3.5→15.7 cm로 악화 | The 14–24 band IS in source (verify/verify_eaob.py: NIS_RANGE = (14.0, 24.0), NIS_UPPER = 24.0) — that part is bucket a. The four NIS values and the four radRMS values are not: verify_state_source.py/verify_eaob.py write only PNGs, and the… |
| 192 | **b** | med | **NIS 245.8→100.0, NEES 398.9→122.0으로 2.5–3.3× 개선**(여전히 게이트 24 초과 = FAIL) | Dated 2026-07-24(3); the only stored verify_eaob outputs in recordings/ are the five PNGs from 2026-07-23 (recordings/20260723/verify_eaob_*_1721xx/1744xx/1745xx/1746xx/1749xx.png) — there is no 2026-07-24 verify artifact at all. The '게이트 … |
| 238 | **b** | med | 필요한 선회율 126°/s(횡력 8 N)~474°/s(30 N)가 슬루 상한 60°/s를 크게 초과 | The 60°/s slew limit IS verifiable (recordings/.../compare_20260720_230025/config.yaml: yaw_rate_deg_s: 60), and \|Δv\| = 0.212 m/s on line 236 is exactly derivable (0.15 m/s × √2, speed from the same config's 'perimeter 4 m / 0.15 m/s'). … |
| 245 | **b** | med | 코너·직선을 거의 같은 비율로 줄여(33% vs 37%) | This is a before/after ratio between the 8 N and 30 N surge boxes. As established for line 59, no 30 N-box wave recording exists anywhere in recordings/ — the only two storm/gentle sweeps both carry identical MPC results, i.e. the same box… |
| 247 | **b** | med | gentle CDW 코너 PID 24.6 / MPC 13.2 / DOB 3.8 cm | Artifact exists (recordings/20260724/compare_20260724_160210/wave05_gentle/runs, 100 traj CSVs per controller) but I could not reproduce the trio. Corner RMS over all 100 runs each, 0.10 m window, laps ≥2: PID 17.5 / MPC 10.8 / DOB 3.9 cm.… |
| 84 | **b** | low | tau_dist=0.2에선 같은 시나리오에서 게이트가 아예 발동하지 않음(0/1067) | The saved gate-off/tau=0.2 artifact (recordings/20260723/verify_eaob_perf_traj_174953.png) reports 'gated 0/1200', not 0/1067. No artifact in the repo contains the denominator 1067. |
| 172 | **b** | low | torque-free kick 0.5 rad/s → 1.5 s 만에 \|q\|>60 rad/s로 재현 | Described as a reproduction ('재현'), but there is no script, test, or saved output in the repo that runs this kick. tests/test_heavy_gripper.py guards body_iquat == identity (the regression guard the entry mentions) but does not contain the… |
| 18 | **c** | hig | Snell 예측은 **84.3°** → 물속 예측과 ±1° 이내, 공기 스펙과는 41° 차이 | 84.3° is correctly derived (2·asin(sin(127°/2)/1.333) = 84.35°) and is honestly labelled 예측, so it is not itself a finding. The '±1° 이내' agreement claim is not: my reproduction gives \|85.66−84.35\| = 1.31° for CAM_B and \|85.23−84.35\| = … |
| 30 | **c** | med | MinZ 표(400p 298.9 mm)는 수중 fx 기준이라 **공기 중 실제 최소거리는 ≈225 mm**이고, 그 거리가 299 mm로 **보고**된다 | Opened c3_camera/TESTING.md: line 17 states MinZ = fx·baseline/max_disparity, i.e. 298.9 mm is a formula output (378.56 × 75 / 95 = 298.86), not a measurement — and TESTING.md:484 explicitly says 'MinZ가 298.9인지 300.1인지 미해결'. 225 mm is like… |
| 17 | **a** | hig | **CAM_B 85.6° / CAM_C 85.1°** (HFOV) | `c3_camera/datasets/dataset_20260803_105221/calibration.json` |
| 15 | **a** | med | `datasets/*/calibration.json` 10개 전부 동일한 한 벌이고 | `c3_camera/datasets/dataset_20260729_173152/calibration.json` |
| 154 | **a** | med | mpc 최대 181.8 cm(pitch 최대 79.5°), dobmpc 최대 134.7 cm(\|pz\| 97 cm) | `bluerov2_mujoco_marinegym/recordings/20260720/compare_20260720_230025/results_raw.csv` |
| 240 | **a** | med | 코너 2.00/1.92/2.04/2.39 cm vs 직선 0.22–0.28 cm(≈10×) | `bluerov2_mujoco_marinegym/recordings/20260724/compare_20260724_160210/wave05_gentle/runs/traj_square_NONE_dobmpc_seed0_c005.3_w266.9.csv` |
| 45 | **a** | low | 같은 파일에 섞인 **VFR_HUD 1169행은 전 열이 공백**이다(11개 메시지 타입 / 7935행) | `c3_camera/datasets/dataset_20260803_105221/telemetry.csv` |
| 56 | **a** | low | `params.U_MAX[0]`을 8 → 30 N(= PID `f_max`)으로 고쳤으므로 | `bluerov2_mujoco_marinegym/dobmpc/params.py` |
| 91 | **a** | low | trim/option-(b) 단정(6 N ≈ 23° pitch, `[Fx,0,0,0]` NU=4 입력) | `bluerov2_mujoco_marinegym/tests/test_dobmpc.py` |
| 152 | **a** | low | n=200 census — 3000 run 중 217 run에서 총 289회 실패, 전부 MPC 계열이고 89–99%가 CW/CDW(run 실패율 **mpc 14.0% vs dobmpc 7.7%** | `bluerov2_mujoco_marinegym/recordings/20260720/compare_20260720_230025/results_raw.csv` |
| 155 | **a** | low | **dobmpc radial_max>40 cm는 23/23이 fail run** (클린 run 상한 37.8 cm) | `bluerov2_mujoco_marinegym/recordings/20260720/compare_20260720_230025/results_raw.csv` |
| 159 | **a** | low | RMS 집계는 강건: 제외해도 평균 −3~−8%만 이동 | `bluerov2_mujoco_marinegym/recordings/20260720/compare_20260720_230025/results_raw.csv` |
| 164 | **a** | low | 10–13→1–2런, mpc 16→7/12→4) 및 dobmpc >40 cm 꼬리 소멸 … (mpc CDW에 신규 210 cm blowup | `bluerov2_mujoco_marinegym/recordings/20260721/compare_20260721_111349/results_raw.csv` |
| 178 | **a** | low | −0.0016(0.4%) → **heavy_gripper +0.064 kg·m² (Ixx의 16.8%) / heavy_c3 +0.046 (12.4%)** | `bluerov2_mujoco_marinegym/compute_payload_inertia.py` |
| 184 | **a** | low | Hs 0.75→1.2 m, Tp 12→6 s, γ 5→2, s 30→10, ω_max 1.6→3.0으로 강화됨 | `bluerov2_mujoco_marinegym/config/base.yaml` |
| 205 | **a** | low | 정지 상태에서 \|a\| = **11.8 m/s²** (중력 9.81 대비 **+20%**) | `c3_camera/datasets/dataset_20260729_173152/imu_camera.csv` |
| 221 | **a** | low | `getTimestamp()` 차이가 **약 66 ms(15 fps에서 ≈1 프레임 간격)** 로 관측된다 | `c3_camera/recordings/20260729_162216/frames.csv` |
| 276 | **a** | low | 실현 반복 주기 264.8 s ≈ run 길이 266.7 s | `bluerov2_mujoco_marinegym/recordings/20260720/compare_20260720_230025/config.yaml` |
| 277 | **a** | low | dobmpc 400 run 중 181개가 t=200–210 s에 피크, worst vertex 66%가 V3 | `bluerov2_mujoco_marinegym/recordings/20260720/compare_20260720_230025/runs/` |
| 280 | **a** | low | 스킨 bbox = 벤더 치수 × 1.0233 (세 축 균일) | `bluerov2_mujoco_marinegym/tools/process_c3_mesh.py` |

### [README_fisheye_gantry.md](../README_fisheye_gantry.md)  — a 30 / b 15 / c 1

| 줄 | 판정 | sev | 주장 | 산출물 / 사유 |
|---:|:---:|:---:|---|---|
| 235 | **b** | hig | Smoke-test timing (mock): click → handler return in **~3 ms**, halt distance after click **≈ 0 mm**. | Explicitly framed as a smoke-test measurement of an emergency-stop path. There is no test file for the panel anywhere in the repo (find for test_*.py returns only viser_test.py, tools/nadir_ruler_test.py and two external/ iPhone streaming … |
| 412 | **b** | hig | MIN_SHARPNESS = 50.0 … COVERAGE_GRID = (4, 4) … TARGET_PER_CELL = 2 … MIN_CELLS_COVERED = 12 … SCORE_WEIGHT_SHARPNESS = 0.50 … SCORE_WEIGHT_TILT = 0.… | The doc says 'Constants at the top of calibrate_fisheye.py are exposed for easy tuning' and quotes 7 names. grep over src/ and tools/ finds only MIN_SHARPNESS (src/calibrate_fisheye.py:93 = 50.0). COVERAGE_GRID, TARGET_PER_CELL, MIN_CELLS_… |
| 972 | **b** | hig | The known-good value for this rig is the example above (`[[0,1,0],[-1,0,0],[0,0,1]]`) | Opened config/fisheye_calibration.yaml: R_gantry_to_slam = [[0.0402,-0.9991,0.0157],[-0.9983,-0.0408,-0.0428],[0.0434,-0.0139,-0.9990]]. The pre-refinement hand-set value in config/fisheye_calibration.backup_20260528_215507.yaml is [[0,-1,… |
| 242 | **b** | med | verified on offscreen Qt with focus on a `QDoubleSpinBox`: handler ran in **~3 ms** | Word 'verified' asserts an executed offscreen-Qt test. No such test exists in the repo and no output is stored. |
| 354 | **b** | med | `4×4 grid · 2 frames/cell · target ≈ 32 frames · ≥12 cells required` | Presented as a literal UI summary line. No coverage-grid code exists in src/calibrate_fisheye.py (no COVERAGE_GRID / TARGET_PER_CELL / MIN_CELLS_COVERED, no cell grouping anywhere in the 2164-line file). |
| 385 | **b** | med | Rates each survivor: sharpness 50 % + tilt 30 % + reproj 20 % | No scoring function exists. grep for SCORE_WEIGHT / 'tilt' over src/calibrate_fisheye.py returns nothing; the function list has no scorer between _on_load_finished and _start_final_calibration. |
| 432 | **b** | med | The progress dialog should show "16 / 16 cells covered, 32 frames picked". | Step 4 of a numbered smoke-test procedure, i.e. presented as an observed run. No coverage/cell machinery exists in src/calibrate_fisheye.py, so the dialog cannot show this. No test file or saved output exists anywhere in the repo for this … |
| 661 | **b** | med | flushes every 1 s, ~1–2 % CPU, ~50 MB/hour | 'flushes every 1 s' is real (src/survey_diagnostics.py:133 flush_interval_s: float = 1.0). The disk rate is contradicted by the two recorded surveys: data/20260528/20260528_195059_survey — 663.6 s of survey, 2.19 MB of diagnostics files + … |
| 728 | **b** | med | `--use-frames` (slower — ~10–90 ms/frame; a 3-min/~5000-frame run ≈ 1–7 min) | config/tag_map.yaml metadata records used_frames_redetection: false — the only surveyed map in the repo was built CSV-only, so the --use-frames path has never produced a stored artifact. No timing log exists. |
| 125 | **b** | low | default ratio 38/62, window default `1500 × 900`, minimum `1280 × 800` | src/gantry_panel.py:2245 self.resize(1500, 900) — confirms 1500x900. But :2246 setMinimumSize(1200, 700), not 1280x800. And :2508-2509 splitter.setSizes([630, 870]) with the code comment '# 42/58 split at 1500 wide' — 630/1500 = 42%, not 3… |
| 141 | **b** | low | Tick spacing adapts to the visible range (~6–10 major ticks at any zoom) | src/gantry_panel.py:1577-1587 _pick_tick_spacing docstring says 'targeting ~5-8 labeled ticks in the span'. |
| 371 | **b** | low | Record at least 10–20 seconds; 30 s gives ≥ 16 cells easily. | Reads as operating experience with the coverage grid. That grid does not exist in the code (see line-412 row), and no recording session log exists in the repo to support the '30 s -> ≥16 cells' rate. |
| 382 | **b** | low | Drops frames with sharpness < 50 and duplicates within 200 ms | 'sharpness < 50' is real (src/calibrate_fisheye.py:93 MIN_SHARPNESS=50.0, used at :353). The 200 ms duplicate window does not exist — the only dedup in the file is by resolved file path (calibrate_fisheye.py:454 '# Deduplicate by resolved … |
| 567 | **b** | low | gantry_telemetry.csv      # 100 Hz, 23 columns, monotonic + unix timestamps | 100 Hz confirmed (see separate row). Column count is wrong: data/20260528/20260528_163834_survey/gantry_telemetry.csv has 26 columns (timestamp_unix … is_moving). The README's own line 946-949 documents the schema change that added the *_d… |
| 1063 | **b** | low | Acceleration column in `gantry_telemetry.csv` is a 5-sample SMA-smoothed central finite difference of velocity | src/gantry_runner.py:95-96 SMOOTHING_WINDOW_S = 0.25, SMOOTHING_POLYORDER = 2, and :107 compute_derivative_savgol using scipy.signal.savgol_filter(deriv=1). That is a Savitzky-Golay derivative over a 0.25 s window (≈25 samples at 100 Hz), … |
| 795 | **c** | low | [tag-map] Loaded 24 tags from tag map / Survey anchor: tag 70 / Runtime anchor: tag 65 … Transformed 24 tag poses | Introduced with 'Startup logs (stderr) confirm it:', which reads as a captured session. The only tag map in the repo, config/tag_map.yaml, has anchor_tag_id: 25 and 47 tags (metadata n_tags_qualified: 47). No run in data/ produced a 24-tag… |
| 8 | **a** | low | log telemetry CSV at ~100 Hz | `data/20260528/20260528_163834_survey/gantry_telemetry.csv` |
| 23 | **a** | low | `gantry_runner.SCALE_MM_PER_UNIT` (X=8.25, Y=2.5, Z=0.5 mm/unit — copied from `src/gantry/demos/whisker_dragging.py`) | `src/gantry_runner.py` |
| 132 | **a** | low | A 200-sample trail | `src/gantry_panel.py` |
| 137 | **a** | low | XY shows ≈ 4877 × 1800 mm or 1800 × 4877 mm depending on Pool Orientation | `config/config.yaml` |
| 141 | **a** | low | All updates ride the 10 Hz status-poll | `src/gantry_panel.py` |
| 185 | **a** | low | auto-stop ~500 ms after the axis reports stopped | `src/gantry_panel.py` |
| 295 | **a** | low | Home Z toward NEGATIVE limit at 5.00 cm/s (≈ 100.00 units/s on Z → clamped to 20.0). | `src/gantry_panel.py` |
| 353 | **a** | low | Defaults: 9 × 6, 25 mm/square. | `src/calibrate_fisheye.py` |
| 361 | **a** | low | Watch **Live sharpness**: keep it green (> 150) | `src/calibrate_fisheye.py` |
| 394 | **a** | low | Green < 0.5 px … Yellow 0.5–1.0 px … Red > 1.0 px | `src/calibrate_fisheye.py` |
| 435 | **a** | low | Drag the splitter handle (8 px wide, turns blue on hover) | `src/calibrate_fisheye.py` |
| 579 | **a** | low | open the interactive survey GUI (1500×950) | `src/survey_tags_gui.py` |
| 605 | **a** | low | green `< 5 mm`, yellow `5–15 mm`, red `≥ 15 mm` | `src/survey_tags_gui.py` |
| 610 | **a** | low | If the chosen anchor is not seen within 30 s the survey aborts | `src/survey_tags_gui.py` |
| 615 | **a** | low | Residual > 20 mm → 3 s yellow banner; > 100 mm → red persistent banner | `src/survey_tags_gui.py` |
| 622 | **a** | low | Tightened detection gates before the backend: reprojection ≤ 3 px, tag area ≥ 200 px², off-nadir ≤ 30°. | `src/survey_tags_gui.py` |
| 627 | **a** | low | every tag is pinned to the z = 0 plane … at σ = 5 mm | `src/survey_tags_gui.py` |
| 629 | **a** | low | Warmup (first 30 s): a tag may only enter the graph once it has been confirmed by ≥ 5 frames that each saw ≥ 3 simultaneous tags | `src/survey_tags_gui.py` |
| 638 | **a** | low | All SDK calls share one `RLock` with the 10 Hz status poll, the 100 Hz telemetry logger | `src/gantry_panel.py` |
| 648 | **a** | low | Every 200 frames *or* 30 s … If any tag moves ≥ 5 mm it rebuilds iSAM2 … A 200 ms blue flash | `src/survey_tags_gui.py` |
| 653 | **a** | low | Tighter relinearization: `relinearizeThreshold 0.01 → 0.001` in `tagslam_core.TagSlamBackend` (10× more eager) | `src/tagslam_core.py` |
| 655 | **a** | low | if the camera goes > 60 s without coming within 30 cm of a previously-visited spot | `src/survey_tags_gui.py` |
| 667 | **a** | low | per-frame backend state (≤15 Hz) | `data/20260528/20260528_202333_survey/survey_diagnostics.csv` |
| 668 | **a** | low | `observation` (every 50th) … iSAM2 health every 60 frames … full tag-pose snapshot every 30 s | `src/survey_tags_gui.py` |
| 676 | **a** | low | a **drift-onset estimate** (first frame where residual/jump exceeded baseline+3σ for ≥5 frames) | `src/survey_diagnostics.py` |
| 759 | **a** | low | Noise model matches the live pipeline (`tag_rot_sigma=0.08 rad`, `tag_trans_sigma=0.04 m`); anchor prior `σ=1e-6` | `src/survey_tags.py` |
| 949 | **a** | low | Savitzky-Golay smooth derivative (`SMOOTHING_WINDOW_S=0.25 s`, `SMOOTHING_POLYORDER=2`) | `src/gantry_runner.py` |
| 956 | **a** | low | a **warning** if divergence exceeds 50 cm/s | `src/tagslam_core.py` |
| 1034 | **a** | low | Reads the before/after RMS and the refinement angle (typically 2–5°). | `config/fisheye_calibration.yaml` |
| 1044 | **a** | low | refuses on … a refinement angle > `--max-angle-deg` (default 15°) … or when RMS doesn't improve by > `--rms-threshold-mm` (default 5 mm) … With < 100… | `src/tools/refine_R_gantry_to_slam.py` |

### [bluerov2_issac_paper/README.md](../bluerov2_issac_paper/README.md)  — a 0 / b 1 / c 0

| 줄 | 판정 | sev | 주장 | 산출물 / 사유 |
|---:|:---:|:---:|---|---|
| 78 | **b** | med | Generally converges in about 400 iterations with 2048 environments and achieves mean total reward ~95-100. | No training artifact ships in bluerov2_issac_paper/ (only weights/2024-09-13_20-15-03_poshold_DR_2.pt, a position-hold checkpoint, no logs). The only 400-iteration 2048-env run on this machine (~/IsaacLab/logs/rsl_rl/warpauv_direct/2026-06… |

### [bluerov2_mujoco_dobmpc/README.md](../bluerov2_mujoco_dobmpc/README.md)  — a 10 / b 7 / c 0

| 줄 | 판정 | sev | 주장 | 산출물 / 사유 |
|---:|:---:|:---:|---|---|
| 76 | **b** | hig | MPC    \| 0.1045 \| 0.1019 \| 0.1430 \| 0.0559 | results/dp_constant_mujoco_rmse.txt gives mpc 0.1043 / 0.1018 / 0.1419 / 0.0559; recomputed from results/dp_constant_mujoco_mpc.npz identically. x, y, z all differ from the doc. |
| 77 | **b** | hig | DOBMPC \| **0.0074** \| **0.0031** \| **0.0150** \| **0.0015** | Header says 'Verified results (this exact code, seed 1, --N 40)'. The artifact bluerov2_mujoco_dobmpc/results/dp_constant_mujoco_rmse.txt reads dobmpc 0.0023 / 0.0020 / 0.0043 / 0.0014. I recomputed RMSE directly from results/dp_constant_m… |
| 85 | **b** | hig | MPC    \| 0.1667 \| 0.1524 \| 0.2100 \| **0.0520** | results/track_circle_mixed_mujoco_rmse.txt gives mpc 0.1622 / 0.1453 / 0.1996 / 0.0502; recomputed from results/track_circle_mixed_mujoco_mpc.npz identically. All four entries differ. |
| 86 | **b** | hig | DOBMPC \| **0.1030** \| **0.0837** \| **0.0820** \| 0.0825 | results/track_circle_mixed_mujoco_rmse.txt gives dobmpc 0.0970 / 0.0824 / 0.0526 / 0.0831; recomputed from results/track_circle_mixed_mujoco_dobmpc.npz identically. z differs by 56% (0.0820 vs 0.0526). |
| 117 | **b** | hig | `R = [0.5 0.5 0.5 0.05]` reproduces the paper's behaviour | Opened bluerov2_mujoco_dobmpc/bluerov2mj/params.py: MPC_R = np.array([0.05, 0.05, 0.05, 0.005]) — ten times smaller than the README states. The params.py comment block itself says 'R = [.05 .05 .05 .005] (~40x the paper's effective penalty… |
| 94 | **b** | med | Position-channel gains of 1.6-2.6x dominate, as in the paper. | 1.6-2.6x is exactly the MPC/DOBMPC ratio of the README's own (unreproducible) tracking table (0.1667/0.1030=1.62, 0.1524/0.0837=1.82, 0.2100/0.0820=2.56). Recomputing from the real artifact results/track_circle_mixed_mujoco_{mpc,dobmpc}.np… |
| 96 | **b** | low | the shipped default is the paper's N = 60 (closed-loop behaviour is indistinguishable, ~1.5x slower) | Only one set of logs exists in results/ and the README itself says they were produced at --N 40. There is no N=60 log in results/ (all six .npz files are the N=40 run), so neither 'indistinguishable' nor '~1.5x slower' can be checked again… |
| 9 | **a** | low | EAOB (18-state EKF) | `bluerov2_mujoco_dobmpc/bluerov2mj/eaob.py` |
| 19 | **a** | low | injected per 2 ms substep via `xfrc_applied` | `bluerov2_mujoco_dobmpc/bluerov2mj/bluerov2.xml` |
| 27 | **a** | low | CasADi/Ipopt (~0.1 s per solve) | `bluerov2_mujoco_dobmpc/results/dp_constant_mujoco_dobmpc.npz` |
| 27 | **a** | low | N = 60 × 0.05 s | `bluerov2_mujoco_dobmpc/bluerov2mj/params.py` |
| 79 | **a** | low | constant current (10 N x/y/z + 5 Nm yaw at t = 10 s), RMSE over 25 s | `bluerov2_mujoco_dobmpc/results/dp_constant_mujoco_dobmpc.npz` |
| 81 | **a** | low | Circle tracking (r = 2 m, 1 m/s) under mixed disturbance (3-6 N waves + 10 N / 3 Nm step at t = 4 s) | `bluerov2_mujoco_dobmpc/bluerov2mj/experiment.py` |
| 88 | **a** | low | The EAOB tracks step and sinusoidal disturbances within 1-2 samples with ~0.5 N noise-induced jitter | `bluerov2_mujoco_dobmpc/results/dp_constant_mujoco_dobmpc.npz` |
| 92 | **a** | low | a ~0.2 rad transient right after the disturbance step while circling at 1 m/s | `bluerov2_mujoco_dobmpc/results/track_circle_mixed_mujoco_dobmpc.npz` |
| 106 | **a** | low | EMA-filtered (α = 0.3) acceleration | `bluerov2_mujoco_dobmpc/bluerov2mj/mujoco_env.py` |
| 107 | **a** | low | ≤ 0.2 cm / 0.4° divergence over 10 s of aggressive excitation (`validate_plant.py`) | `bluerov2_mujoco_dobmpc/scripts/validate_plant.py` |

### [bluerov2_mujoco_marinegym/README.md](../bluerov2_mujoco_marinegym/README.md)  — a 16 / b 16 / c 0

| 줄 | 판정 | sev | 주장 | 산출물 / 사유 |
|---:|:---:|:---:|---|---|
| 42 | **b** | hig | so `"meas"` chatter drops from ~9–15× to ~2–3× the truth-x0 level | verify/verify_state_source.py (the cited script) writes NO artifact — I grepped it for savefig/CSV/recordings output and found none; it is stdout-only, and no log is stored. Worse, the number contradicts its own source: dobmpc/params.py:14… |
| 44 | **b** | hig | releasing it moves storm PID−MPC from −4.24 to +8.60 cm | I reproduced the BEFORE half exactly: mean radial_rms PID−MPC over the first 4 sweep headings of storm/CW in that results.csv = −4.2358 cm (matches −4.24; gentle first-4 = +4.006 cm, matching the '+4.01' companion figure in CONTROL_METHODO… |
| 44 | **b** | hig | the box was active on 60 % of ticks at storm (dobmpc 75 %) vs 7 % at gentle | No artifact and no instrumentation. I grepped dobmpc/*.py, experiments/*.py and tools/*.py for any box-activity counter (box_active / u_active / at_bound / box_freq): the only hit in the whole repo is the prose comment in dobmpc/params.py:… |
| 136 | **b** | med | DOB-MPC DP hold 1.3 cm (still), 1.4 cm (current+waves) | I read tests/test_heavy_gripper.py — the sentence attaches these numbers to that test, but the test contains no DOB-MPC run at all: its only hold check is test_allocation_and_pid_hold(), a 20 s PID hold asserting err < 0.08 m. The numbers … |
| 143 | **b** | med | verified 3000-step rollout Δ=0 vs the old gray body | No test or script in the repo runs a 3000-step A/B rollout. grep for '3000' across tests/, tools/, verify/ and the package root yields only tools/extract_meshes.py:158 (--thruster-faces default 3000). The only byte-identity rollout test is… |
| 173 | **b** | med | measured ~2.2× faster model parse | Repo-wide grep for '2.2' / '2.2x' / 'faster' in tools/, docs/*.md, *.py and .claude/journal/ returns only this README line. tools/gen_pool_apriltags.py contains no timing code and no benchmark harness exists for model parse time. The compa… |
| 198 | **b** | med | verified: old-vs-new 3000-step rollout Δ=0, incl. `--disturb` | Same search as above: no 3000-step comparison exists. tests/ contains no POOL_TAGS on-vs-off dynamics-equality test (grep 'POOL_TAGS' in tests/ hits only test_water_viz.py, which sets POOL_TAGS=1 for both arms and steps 2000). The Δ=0 conc… |
| 282 | **b** | med | model loads; mass ∈ [9, 12] kg; 6 thruster sites | I opened the test. It loads bluerov_heavy.xml (line 30) and asserts `9.0 <= total_mass <= 13.0` (line 152) and `nthr == 8` (line 153), printing 'expected 8'. Neither the [9,12] kg window nor '6 thruster sites' is what the test checks; the … |
| 586 | **b** | med | passes NIS+NEES in CDW and NIS in CD (CD NEES ~9 = deliberate real-hardware conservatism) | I opened the saved verify_eaob figures. The CDW half IS confirmed: verify_eaob_perf_traj_174953.png's title reads 'NIS 17.1  NEES 15.0  gated 0/1200', exactly params.py's tau=0.2 CDW numbers (and 174602/174455 show the rejected tau=0.5 cas… |
| 39 | **b** | low | acados SQP-RTI** (~1 ms, default) | verify/verify_acados.py prints median/p95/max solve time but writes no file; no log is stored. The underlying measurement exists only as a table row in docs/CONTROL_METHODOLOGY.md:487 ('median 0.97 ms, max 1.1 ms'). Repeated at README.md:6… |
| 42 | **b** | low | below the ~1.2 cm tracking error | Traces to a prose comment in dobmpc/params.py:147-148 ('the earlier x/y 5 cm SAT ABOVE the ~1.2 cm closed-loop tracking error'). No results.csv row was cited and verify_state_source.py saves nothing. The nearest artifact values I could fin… |
| 105 | **b** | low | ~1° residual pitch from the C3's static moment | I read tests/test_heavy_c3.py in full (the regression the README points at on the next line): it checks composition, actuator list, thruster-site preservation, buoyancy sign, camera orientation and a PID position hold — there is no pitch/a… |
| 268 | **b** | low | `python test_<name>.py` (11 files) | `ls tests/test_*.py \| wc -l` = 13. The §1 table lists only 11 of them; tests/test_heavy_c3.py and tests/test_heavy_gripper.py are present on disk but absent from both the count and the table. |
| 294 | **b** | low | 21 checks on current/env | I ran the module in the `robust` env: it prints '22 passed, 0 failed'. Static count agrees: 20 check() call sites, one of which (line 50) is inside a 3-iteration loop → 19 + 3 = 22. (The sibling claim '12 checks' at README.md:293 IS correc… |
| 336 | **b** | low | ~15 s per crossing at the default sea state | Repo-wide grep across teleop.py, docs/05_TELEOP.md and docs/CONTROL_METHODOLOGY.md finds no source for this timing; tests/test_observe.py measures drift over 1500 steps (3 s) only and asserts byte-identity, not a crossing time. No observe-… |
| 418 | **b** | low | python -m disturbance.test_waves && python -m disturbance.test_env   # 33 unit asserts first | Ran both modules: 12 passed + 22 passed = 34, not 33. |
| 91 | **a** | med | 11.5 kg / [0.3291, 0.6347, 0.6109]† \| 13.2 kg / [0.37014, 0.73153, 0.67460]§ \| 13.724 kg / [0.38154, 0.77780, 0.70954]‡ | `bluerov2_mujoco_marinegym/bluerov_heavy_gripper.xml` |
| 92 | **a** | med | ~+1.1 N \| **−3.1 N (sinks)** \| **−5.7 N (sinks)** | `bluerov2_mujoco_marinegym/marinegym_assets/BlueROVHeavyGripper.yaml` |
| 44 | **a** | low | gave the NMPC a 3.75× handicap vs the PID's `f_max`=30 N | `bluerov2_mujoco_marinegym/dobmpc/params.py` |
| 102 | **a** | low | inertia **diagonal** (`Ixz` +0.046 dropped, 12.4% of Ixx — KNOWN_ISSUES) | `bluerov2_mujoco_marinegym/tools/gen_c3_variant.py` |
| 121 | **a** | low | `gripper` at **ctrl index 8** (`d.ctrl[8] = 0…0.031` = closed…62 mm open) | `bluerov2_mujoco_marinegym/bluerov_heavy_gripper.xml` |
| 123 | **a** | low | `c3_center` 12 MP-equiv fovy 52.5°, `c3_left`/`c3_right` stereo pair, 7.5 cm baseline | `bluerov2_mujoco_marinegym/bluerov_heavy_gripper.xml` |
| 180 | **a** | low | bounded to 587 tags | `bluerov2_mujoco_marinegym/tools/gen_pool_apriltags.py` |
| 239 | **a** | low | Real ocean wavelengths (6–76 m) dwarf the pool (~1.8×4.9 m) | `bluerov2_mujoco_marinegym/config/base.yaml` |
| 287 | **a** | low | CasADi MPC model ≡ NumPy fossen model (< 1e-9) | `bluerov2_mujoco_marinegym/tests/test_dobmpc.py` |
| 290 | **a** | low | leaves a 2000-step rollout byte-identical (Δ = 0) | `bluerov2_mujoco_marinegym/tests/test_water_viz.py` |
| 321 | **a** | low | base: surge 8 N, sway 15 N, heave 20 N, yaw 6 N·m, roll 3 N·m | `bluerov2_mujoco_marinegym/teleop.py` |
| 450 | **a** | low | (Figures generated before 2026-07-28 boxed a full-run RMS instead, ~1–2 % lower — see `docs/CONTROL_METHODOLOGY.md`.) | `bluerov2_mujoco_marinegym/recordings/20260727/compare_20260727_000850/wave00_very_rough/runs/` |
| 452 | **a** | low | set `false` for huge sweeps (~0.4 MB/run) | `bluerov2_mujoco_marinegym/recordings/20260727/compare_20260727_000850/wave00_very_rough/runs/` |
| 565 | **a** | low | Takes several minutes (45 runs) | `bluerov2_mujoco_marinegym/tools/ablation_thrusters.py` |
| 585 | **a** | low | equivalence (worst max\\|Δu\\| < 0.25 N on interior states) + SQP-RTI timing vs the 50 ms @ 20 Hz budget | `bluerov2_mujoco_marinegym/verify/verify_acados.py` |
| 629 | **a** | low | an **EAOB** (18-state EKF) estimates the disturbance wrench ŵ online; the **NMPC** ... re-plans every 50 ms (N=60) | `bluerov2_mujoco_marinegym/dobmpc/eaob.py` |

### [bluerov2_mujoco_marinegym/docs/00_OVERVIEW.md](../bluerov2_mujoco_marinegym/docs/00_OVERVIEW.md)  — a 4 / b 0 / c 0

| 줄 | 판정 | sev | 주장 | 산출물 / 사유 |
|---:|:---:|:---:|---|---|
| 48 | **a** | med | **Vehicle is underactuated in pitch** (rank(allocation)=5). | `bluerov2_mujoco_marinegym/thrusters.py` |
| 39 | **a** | low | Gravity (0,0,−9.81) | `bluerov2_mujoco_marinegym/bluerov.xml` |
| 58 | **a** | low | `bluerov.xml` \| the MJCF (rigid body + 6 thruster sites + 6 force actuators) | `bluerov2_mujoco_marinegym/bluerov.xml` |
| 87 | **a** | low | What's verified today: Phases 1–4 each have a passing `test_*.py` | `bluerov2_mujoco_marinegym/tests/test_load.py` |

### [bluerov2_mujoco_marinegym/docs/01_DECISIONS.md](../bluerov2_mujoco_marinegym/docs/01_DECISIONS.md)  — a 3 / b 0 / c 0

| 줄 | 판정 | sev | 주장 | 산출물 / 사유 |
|---:|:---:|:---:|---|---|
| 38 | **a** | low | D3 — CB offset coBM = 0.01 m ... (`coBM: 0.01`) | `bluerov2_mujoco_marinegym/marinegym_assets/BlueROV.yaml` |
| 50 | **a** | low | the standard vectored-**6** layout: 4 horizontal thrusters at ±45° (surge/sway/yaw) + 2 vertical (heave/roll) ... **`BlueROVHeavy`** (8 thrusters) | `bluerov2_mujoco_marinegym/bluerov.xml` |
| 81 | **a** | low | with ν̇ one-substep-lagged + 0.3 low-pass filtered | `bluerov2_mujoco_marinegym/hydro.py` |

### [bluerov2_mujoco_marinegym/docs/02_MODEL.md](../bluerov2_mujoco_marinegym/docs/02_MODEL.md)  — a 8 / b 1 / c 0

| 줄 | 판정 | sev | 주장 | 산출물 / 사유 |
|---:|:---:|:---:|---|---|
| 91 | **b** | hig | - Mass 11.20 kg, inertia as above, 6 thruster sites, correct vectored layout. | The doc attributes this line to running `python tests/test_load.py` (line 89). I opened tests/test_load.py: line 30 sets `XML = os.path.join(HERE, "bluerov_heavy.xml")` and line 153 asserts `nthr == 8`. I ran it: it prints TOTAL MASS 11.50… |
| 18 | **a** | low | `bluerov_body.obj` (307,785 → 40,000 faces) | `bluerov2_mujoco_marinegym/meshes/bluerov_body.obj` |
| 41 | **a** | low | \| total mass \| **11.20 kg** \| inertia (diagonal, Ixx,Iyy,Izz) \| **0.30375, 0.626, 0.5769** kg·m² \| COM (body frame) \| **(0, 0, 0)** \| bodies /… | `bluerov2_mujoco_marinegym/bluerov.xml` |
| 50 | **a** | low | mass **11.5 kg** and **8** thruster sites/actuators (`thr0..thr7`) | `bluerov2_mujoco_marinegym/bluerov_heavy.xml` |
| 51 | **a** | low | Its inertia **[0.3291, 0.6347, 0.6109]** is *derived* from the bluerov2 tensor by adding the parallel-axis term | `bluerov2_mujoco_marinegym/tools/compute_heavy_inertia.py` |
| 54 | **a** | low | the farol Heavy USD's own [0.21, 0.245, 0.245] is a hand-tuned Gazebo-stability literal | `external/MarineGym/marinegym/robots/assets/usd/BlueROVHeavy/BlueROVHeavy.usd` |
| 56 | **a** | low | \| collision \| one box: center (0,0,−0.05), half-size (0.25,0.175,0.125) m \| | `bluerov2_mujoco_marinegym/bluerov.xml` |
| 66 | **a** | low | All 4 horizontal thrusters sit at **z = −0.0725 m (below the COM)** ... `thruster_0` (+0.1355, −0.100, −0.0725) ... `thruster_5` (+0.0025, +0.1105, −… | `bluerov2_mujoco_marinegym/bluerov.xml` |
| 92 | **a** | low | Zero-control free fall ... Δz ≈ −19.6 m over 2 s, matching ½·g·t² | `bluerov2_mujoco_marinegym/tests/test_load.py` |

### [bluerov2_mujoco_marinegym/docs/03_THRUSTERS.md](../bluerov2_mujoco_marinegym/docs/03_THRUSTERS.md)  — a 9 / b 0 / c 0

| 줄 | 판정 | sev | 주장 | 산출물 / 사유 |
|---:|:---:|:---:|---|---|
| 13 | **a** | low | \| max forward (u=+1) \| **+64.13 N** \| max reverse (u=−1) \| **−51.55 N** \| forward/reverse asymmetry \| **1.244** \| deadband \| `\|u\| ≤ 0.075` … | `bluerov2_mujoco_marinegym/thrusters.py` |
| 31 | **a** | low | small forces round-trip to 0 or the ~1.4 N min-spin jump | `bluerov2_mujoco_marinegym/thrusters.py` |
| 44 | **a** | low | \| max fwd (kgf) \| 2.93 \| 3.71 \| 4.53 \| 5.25 \| 6.02 \| 6.72 \|  /  \| max rev (kgf) \| −2.31 \| −2.92 \| −3.52 \| −4.07 \| −4.59 \| −5.04 \| | `bluerov2_mujoco_marinegym/tools/analyze_t200_voltage.py` |
| 49 | **a** | low | 4.81 / 3.74 kgf ⇒ `14.8V/base` = 0.74 (fwd) / 0.71 (rev) ⇒ **`NOMINAL_VOLTAGE_SCALE = 0.72`** ... full-charge 16.8 V ≈ 0.83, near-empty 13 V ≈ 0.62 | `bluerov2_mujoco_marinegym/tools/analyze_t200_voltage.py` |
| 71 | **a** | low | [[ 0.7071  0.7071 -0.7071 -0.7071  0.      0.    ] ... [ 0.1665 -0.1665 -0.175   0.175   0.      0.    ]] | `bluerov2_mujoco_marinegym/thrusters.py` |
| 87 | **a** | low | **Surge → pitch:** `My = z₀·Fx = −0.0725·Fx`. **Sway → roll:** `Mx = −z₀·Fy = +0.0725·Fy` | `bluerov2_mujoco_marinegym/tests/test_thrusters.py` |
| 98 | **a** | low | `rank(B) = 5`, not 6 ... `My ≈ −0.0725·Fx − 0.0025·Fz` | `bluerov2_mujoco_marinegym/tests/test_thrusters.py` |
| 113 | **a** | low | adds 2 more vertical thrusters (4 at the corners `(±0.12, ±0.22, −0.005)`, all +Z), making **`rank(B) = 6` — fully actuated** ... (verified: a pure p… | `bluerov2_mujoco_marinegym/bluerov_heavy.xml` |
| 118 | **a** | low | the MarineGym Heavy yaml's `force_constants` 0.8e-7 would scale thrust to ~18 % | `external/MarineGym/marinegym/robots/assets/usd/BlueROVHeavy/BlueROVHeavy.yaml` |

### [bluerov2_mujoco_marinegym/docs/04_HYDRO.md](../bluerov2_mujoco_marinegym/docs/04_HYDRO.md)  — a 11 / b 0 / c 1

| 줄 | 판정 | sev | 주장 | 산출물 / 사유 |
|---:|:---:|:---:|---|---|
| 78 | **c** | med | - Below ~Fx ≈ 5 N: stable, nearly-level glide (good for open-loop driving). | Presented as a finding under '⚠ Finding — surge↔pitch coupling beats the weak restoring'. It is not derivable from the stated restoring limit: 1.11 N·m / 0.0725 m = 15.3 N, not 5 N. tests/test_hydro.py only ever runs Fx = 2.83 N (its comme… |
| 62 | **a** | med | no thrust, 10 s: steady **vz ≈ +0.115 m/s** | `bluerov2_mujoco_marinegym/tests/test_hydro.py` |
| 10 | **a** | low | \| volume V \| 0.0113459 m³ \| ... coBM 0.01 \| added mass [5.5, 12.7, 14.57, 0.12, 0.12, 0.12] \| linear damping [4.03, 6.22, 5.18, 0.07, 0.07, 0.07… | `bluerov2_mujoco_marinegym/marinegym_assets/BlueROV.yaml` |
| 17 | **a** | low | Buoyancy **B = ρ·g·V = 110.97 N** vs weight **W = m·g = 109.87 N** → **net +1.10 N** | `bluerov2_mujoco_marinegym/marinegym_assets/BlueROV.yaml` |
| 40 | **a** | low | a **one-substep-lagged, low-pass (α=0.3) filtered** finite difference | `bluerov2_mujoco_marinegym/hydro.py` |
| 51 | **a** | low | heave added mass (14.57) **exceeds** the body mass (11.2) ... The 0.3 low-pass at dt = 2 ms | `bluerov2_mujoco_marinegym/bluerov.xml` |
| 64 | **a** | low | from 20°, no thrust: **pitch 20°→0.7°**, **roll 20°→0.2°** over 25 s | `bluerov2_mujoco_marinegym/tests/test_hydro.py` |
| 68 | **a** | low | gentle surge (Fx≈2.8 N): speed rises to a steady **≈0.32 m/s** at ~4° pitch | `bluerov2_mujoco_marinegym/tests/test_hydro.py` |
| 69 | **a** | low | Release → **horizontal speed → 0.005 m/s** | `bluerov2_mujoco_marinegym/tests/test_hydro.py` |
| 70 | **a** | low | no-hydro 0.69 m/s & 84° tilt (grows/coasts) vs hydro **0.29 m/s & 9°** | `bluerov2_mujoco_marinegym/tests/test_hydro.py` |
| 72 | **a** | low | 60 s with thrust + tilt: finite, `\|qvel\|` ≈ 0.6, no NaN | `bluerov2_mujoco_marinegym/tests/test_hydro.py` |
| 76 | **a** | low | The restoring moment max is only **B·coBM ≈ 1.11 N·m** | `bluerov2_mujoco_marinegym/tests/test_hydro.py` |

### [bluerov2_mujoco_marinegym/docs/05_TELEOP.md](../bluerov2_mujoco_marinegym/docs/05_TELEOP.md)  — a 4 / b 4 / c 0

| 줄 | 판정 | sev | 주장 | 산출물 / 사유 |
|---:|:---:|:---:|---|---|
| 45 | **b** | med | forces: `0.003 m/N`, cap `0.6 m` (buoyancy 111 N → 0.33 m ≈ vehicle size) | Opened teleop.py lines 88-97: `FORCE_SCALE = 0.006   # m per N   (2x: mid forces like thrust/kick now clearly sized)` and `FORCE_CAP = 0.6`. The scale is 0.006, not 0.003, and the code comment explicitly records the 2x change. 111 N x 0.00… |
| 46 | **b** | med | velocities: `0.5 m per m/s`, cap `0.4 m` (current 0.2 m/s → 0.10 m) | teleop.py line 93: `VEL_SCALE = 0.8   # m per m/s (current 0.2 m/s -> 0.16 m)`; line 94: `VEL_CAP = 0.4`. Code says 0.8 and 0.16 m; doc says 0.5 and 0.10 m. |
| 97 | **b** | med | a kick gives a brief red spike (e.g. ~31 N) | The sentence begins 'With disturbances on, the arrows match the physics (checked headlessly by drawing into an MjvScene)' — i.e. presented as an executed check. I grepped tests/, verify/ and tools/ for `MjvScene\|user_scn\|draw_force_arrow… |
| 90 | **b** | low | surge keeps a small uncontrollable pitch — both consistent with the rank-5 allocation. | The doc attributes this to `python teleop.py --selftest`. I ran it: it loads the default ROV_MODEL=heavy (rov_model.py line 95) and prints 8-element thrust vectors ('X: all thrust -> [0. 0. 0. 0. 0. 0. 0. 0.]'). The W (surge) row shows My … |
| 18 | **a** | low | thruster forces (clamped to [−51.55, 64.13] N) | `bluerov2_mujoco_marinegym/bluerov.xml` |
| 20 | **a** | low | surge kept gentle (`SURGE_N=8`) | `bluerov2_mujoco_marinegym/teleop.py` |
| 30 | **a** | low | ~constant ~111 N up (buoyancy) ... buoyancy ~111 N up (≈ constant) | `bluerov2_mujoco_marinegym/hydro.py` |
| 47 | **a** | low | Arrows below ~0.5 N (or 0.01 m/s) are not drawn. | `bluerov2_mujoco_marinegym/teleop.py` |

### [bluerov2_mujoco_marinegym/docs/06_ENVIRONMENT.md](../bluerov2_mujoco_marinegym/docs/06_ENVIRONMENT.md)  — a 3 / b 1 / c 1

| 줄 | 판정 | sev | 주장 | 산출물 / 사유 |
|---:|:---:|:---:|---|---|
| 14 | **b** | low | repo-root `.venv` ... Python 3.13, `mujoco` 3.9.x | `ls -d /home/bdml/Desktop/umi_underwater_robust_control/.venv` -> 'No such file or directory'. No .venv exists in the repo. |
| 51 | **c** | hig | driver **595.71.05** (CUDA 13.2) | Ran `nvidia-smi` on this machine: 'NVIDIA-SMI 570.211.01  Driver Version: 570.211.01  CUDA Version: 12.8', GPU name 'NVIDIA GeForce RTX 5090'. No file in the repo records 595.71.05 or CUDA 13.2 (grepped; verify/verify_gpu_mjx.py prints no … |
| 56 | **a** | low | \| **`robust`** \| 3.14 \| 1.26.4 (<2) \| ... base `mujoco` 3.9, gtsam 4.2.1, pyzed, casadi, opencv \| | Executed /home/bdml/miniforge3/envs/robust/bin/python: sys.version 3.14.4, numpy 1.26.4, mujoco 3.9.0. site-packages li… |
| 57 | **a** | low | \| **`robust-mjx`** \| 3.12 \| 2.4.6 \| ... jax 0.10.1 + jaxlib (cuda12, bundled `nvidia-*-cu12` 12.9 wheels), mujoco-mjx 3.9 \| | Executed /home/bdml/miniforge3/envs/robust-mjx/bin/python: sys.version 3.12.13, numpy 2.4.6, jax 0.10.1, mujoco 3.9.0, … |
| 74 | **a** | low | `jax.default_backend()=='gpu'`, `jax.devices()==[CudaDevice(id=0)]` = NVIDIA GeForce RTX 5090; a tiny MJX rollout **and** the canonical `bluerov.xml`… | `bluerov2_mujoco_marinegym/verify/verify_gpu_mjx.py` |

### [bluerov2_mujoco_marinegym/docs/07_DISTURBANCES.md](../bluerov2_mujoco_marinegym/docs/07_DISTURBANCES.md)  — a 9 / b 0 / c 0

| 줄 | 판정 | sev | 주장 | 산출물 / 사유 |
|---:|:---:|:---:|---|---|
| 27 | **a** | low | `jonswap_wave_specs(Hs, Tp, n=30, gamma=3.3, heading_deg, spread_s=4, seed)` ... Default sea **Hs=0.20 m, Tp=4.0 s** | `bluerov2_mujoco_marinegym/disturbances.py` |
| 70 | **a** | low | \| current speed \| 0.20 m/s \| 0–0.4 m/s \| ... \| kick rate \| 0.2 /s \| 0.1–0.5 /s \| \| kick magnitude \| 20–50 N \| 8–60 N \| \| kick duration \… | `bluerov2_mujoco_marinegym/disturbances.py` |
| 79 | **a** | low | Depth-decay sanity at 3 m: swell T=7 s (k=0.082) keeps ~78%; chop T=2 s (k=1.0) keeps ~5% | `bluerov2_mujoco_marinegym/disturbances.py` |
| 96 | **a** | low | after 40 s the horizontal velocity reaches the current (0.20 → 0.20 m/s) | `bluerov2_mujoco_marinegym/tests/test_disturbances.py` |
| 97 | **a** | low | field orbital speed decays with depth (0.281 → 0.143 m/s, 1→5 m) | `bluerov2_mujoco_marinegym/tests/test_disturbances.py` |
| 98 | **a** | low | the vehicle's surge oscillation is much weaker deep (0.020 @1.5 m vs 0.003 @6 m, single T=3 s wave...) | `bluerov2_mujoco_marinegym/tests/test_disturbances.py` |
| 100 | **a** | low | ~the scheduled number of jolts are detected (18 scheduled, 14 detected at rate 0.4/s over 40 s) | `bluerov2_mujoco_marinegym/tests/test_disturbances.py` |
| 102 | **a** | low | current = drift (mean 0.18, ~no osc), wave = oscillation (osc 0.057, ~no drift), kick = jolt (spike 0.31) | `bluerov2_mujoco_marinegym/tests/test_disturbances.py` |
| 104 | **a** | low | **DR**: 6 seeds → varied current speeds, all ≤ 0.4 ... **Combined**: all three for 60 s → finite, `\|qvel\|` bounded | `bluerov2_mujoco_marinegym/tests/test_disturbances.py` |

### [bluerov2_mujoco_marinegym/docs/CONTROL_METHODOLOGY.md](../bluerov2_mujoco_marinegym/docs/CONTROL_METHODOLOGY.md)  — a 42 / b 46 / c 3

| 줄 | 판정 | sev | 주장 | 산출물 / 사유 |
|---:|:---:|:---:|---|---|
| 313 | **b** | hig | NMPC.solve ≈ 83 ms (≈79%), EAOB.update ≈ 22 ms (≈21%), full tick ≈ 106 ms = 0.47× real-time; rare 2.2 s freezes | Searched the repo for any profiling output (cProfile dumps, .prof, timing logs): none exists. No script in the package emits these numbers to a file. '120 warm ticks' likewise unrecorded. |
| 435 | **b** | hig | \| DP (15 s) \| 15.0 → 13.4° \| 30.0 → 22.9° \| radial 4.9 → 6.1 cm \| 0.22 → 0.09 \| … \| square (2 laps) \| 17.8 → 12.6° \| 46.7 → 23.2° \| off-pat… | recordings/20260615/ contains only the 10-lap option-(a) square CSVs and two DOB-MPC DP CSVs (dp_dobmpc_…112355, origin_dobmpc_…113808, radial 3.43/3.45 cm, pitch max 29.2°). No 15 s DP or 2-lap square A/B pair for option-a vs option-b exi… |
| 487 | **b** | hig | median 100 ms … median 0.97 ms, max 1.1 ms \| worst-case max\|Δu\| = 0.107 N \| radial 8.6 cm … 7 freezes \| radial 7.0 cm … 0 freezes \| ~0.5× real-… | Read verify/verify_acados.py end-to-end: it prints to stdout only (no savefig, no CSV/JSON write). Nothing in recordings/ or docs/ contains these numbers. recordings/20260615/acados_vs_before.png exists but shows different quantities (off-… |
| 825 | **b** | hig | \| DP dobmpc (regression) \| 0.00 cm \| \| square dobmpc (FF + heading) \| 1.87 cm \| \| square mpc (FF) \| 3.63 cm \| \| square pid (baseline) \| 25… | No 2-lap mode-C run is stored. Closest artifact recordings/20260629/square_view/traj_square_C_{dobmpc,mpc,pid}_seed0_dir0.csv are 5-lap runs and give 1.945 / 3.494 / 26.79 cm (t≥10 s) — near but not equal. recordings/20260629/compare_20260… |
| 922 | **b** | hig | Yaw slew-to-5° at the (1,1) corner 1.85 → 1.45 s … CDW (1,1) top-edge max 17.2 → 13.2 cm (−23%); overall radial RMS unchanged (CDW 3.13→3.16, NONE 2.… | The A/B is described as 'run_viewer headless' but no paired artifact exists. recordings/20260723/square_view/traj_square_CDW_dobmpc_seed0_dir0.csv is 0 bytes (empty). recordings/20260629/square_view gives CDW dobmpc 3.134 full-run / NONE d… |
| 1335 | **b** | hig | \| pid NONE truth \| 1.49 \| 0.08 \| 0.85 \| \| pid NONE meas \| 1.47 \| 0.10 \| 3.27 \| \| pid C truth \| 1.61 \| 0.08 \| 0.85 \| \| pid C meas \| 1… | Described as a 3-lap A/B; no 3-lap recording exists. The 10-lap sweeps give different values: compare_20260724_005250 (truth PID) NONE 1.5153 / C 1.6400 with dc_radial 0.0211 / 0.0263, and compare_20260724_160210 (meas PID) NONE 1.4918 / C… |
| 1365 | **b** | hig | \| storm \| 26.32 \| 30.55 (box active 60–66 % of ticks) \| 17.72 (0.7–1.8 %) \| \| gentle \| 12.51 \| 8.50 (5–9 %) \| 7.64 (0.2–0.3 %) \| | Described as a runtime-box-override ablation on 4 paired headings; no such recording exists. The stored storm/gentle CW sweeps (compare_20260724_005250 and _160210, 101 headings each) give PID 26.50/26.56 and 14.46/14.13, MPC 28.82 and 9.5… |
| 1380 | **b** | hig | \| storm \| 26.14 \| 17.86 (was 28.82) \| 7.30 (was 17.36) \| 40.3 N \| 0.0000 \| \| gentle \| 11.56 \| 7.63 (was 9.51) \| 1.87 (was 3.14) \| 36.1 N … | The parenthetical 'was' values ALL reproduce exactly from the archived 8 N sweeps: wave00_storm CW mpc 28.8188 / dobmpc 17.3561, wave05_gentle CW mpc 9.5094 / dobmpc 3.1432 (compare_20260724_005250 and _160210). The NEW 30 N numbers (26.14… |
| 130 | **b** | med | (both coefficient sets recovered back out of the sim to 0.00 %, T4.3) | verify/verify_hydro_precise.py has T4.3 (lines 491-513) but its pass gate is literally 'D_L,D_NL within 3%' and the script only prints to stdout. No saved run output anywhere in the repo. docs/HYDRO_VERIFICATION.md line 120 restates '0.00 … |
| 521 | **b** | med | \| jitter (std) ideal→LV \| 10.4 → 12.4 cm \| … \| 2.9 → 3.5 cm \| | docs/figs/ablation_thrusters.png plots radial RMS only; no jitter panel and no CSV/JSON output from tools/ablation_thrusters.py (its only write is the savefig at line 98). |
| 534 | **b** | med | seed 3 = 39 cm, n_fail 116 … Per-seed, ideal DOB-MPC: seeds 0/1/2/4 = 4.1 / 0.7 / 0.9 / 1.4 cm, n_fail 0 | No per-seed artifact exists; the ablation figure carries one DOBMPC-ideal bar (4.0 cm). Indirect check: mean(4.1,0.7,0.9,1.4,12.82)=3.98 ≈ the figure's 4.0, i.e. the archived figure is the POST-fix run and is consistent with the per-seed l… |
| 611 | **b** | med | ideal radial 5.02 cm / jitter 4.30 cm → realistic ×0.72 radial 7.76 cm / jitter 6.17 cm, n_fail 0 | Opened all seven recordings/20260618/dp_compare_*.png: 193439 (PID 12.1), 202408 (dobmpc 5.0), 202429/202957/205253/212557/212629 (3.3, 3.3, 14.5/3.7/1.1, 3.3, 3.3). None shows 7.76 cm or any jitter figure. No CSV for this A/B. |
| 889 | **b** | med | 1-lap square NONE via run_viewer: radial RMS 1.34 cm (t>5 s) | No run_viewer 1-lap NONE artifact exists for 2026-07-03. The nearest recorded value is 1.3789 cm in compare_20260703_111951/results.csv, which is a t≥10 s steady window, and that recording keeps no runs/ CSVs, so the t>5 s figure cannot be… |
| 900 | **b** | med | DOB-MPC's largest CW/CDW error concentrates at the (1,1) upstream corner (17 cm, seed 0 / current 0° / wave 0°), a cross-track sag ~0.36 m past the c… | No seed-0/current-0°/wave-0° per-corner diagnostic is stored. recordings/20260707/compare_20260707_170230/results.csv is a 101-heading sweep (square CDW dobmpc radial_max mean 23.49 cm) and does not contain a per-corner decomposition. |
| 974 | **b** | med | PID DP hold 0.0 cm (still water, 20 s); DOB-MPC DP hold rms(12–20 s) 1.3 cm still / 1.4 cm current+waves / 22 cm with 20–50 N Poisson kicks … heavy r… | recordings/ has no 20260712 directory and no heavy_gripper run of any kind; no compare_*, no traj CSV, no figure. |
| 976 | **b** | med | acados-vs-IPOPT worst-case \|Δu\| 0.2717 N (marginally over the heavy-calibrated 0.25 N gate; ~0.9% of authority) | The 0.25 N gate is real (verify/verify_acados.py line 63). The measured 0.2717 N has no saved output — verify_acados.py only prints. KNOWN_ISSUES.md line 144 restates it, but that is another hand-written doc, not an execution artifact. |
| 990 | **b** | med | its bbox equals the vendor 575×254×457 mm exactly … scale 1.0233 accounted for (the MarineGym-derived skin is uniformly 2.3% large) … residual 1.6 mm… | tools/process_c3_mesh.py restates all four numbers in its module docstring (lines 16-20) but writes no registration report; the only emitted artifact is meshes/c3_payload_frames.json, which contains poses, not residuals. assets/CAD files/o… |
| 1091 | **b** | med | Measured at a 0.40 N·m constant torque: PD holds roll 3.53° / pitch 3.58° … Adding any Ki>0 makes the loop type-1 so φ_ss → 0 exactly (verified 0.00°… | No artifact for this bench test; nothing in recordings/ and no stored output from tests/test_heavy_gripper.py or test_controller.py. |
| 1111 | **b** | med | pitch transiently swings to ~70° WITHOUT the guard vs ~45–56° WITH it | tests/test_controller.py line 117 carries a comment 'The soft pitch guard cuts the transient a lot (~70° -> ~50°)' — a restatement in the same authorship, not a stored measurement. No test output is archived. |
| 1165 | **b** | med | in CD it gives NIS 14.4 PASS, RMSE X/Y/Z 0.29/0.41/0.48 N. CD NEES sits at ~9 | No CD-mode verify_eaob artifact exists; all five archived PNGs in recordings/20260723 are the DP CDW case. |
| 1180 | **b** | med | a consistency gate rejects 25–76% of frames … CDW radRMS 4.53→15.45 cm (gate off→on), CD 1.35→1.43 cm … at the final τ=0.2 the gate never fires in th… | No closed-loop gate A/B artifact exists. The nearest support is 'gated 0/1200' in verify_eaob_perf_traj_174953.png, which corroborates 'never fires' but not the 0/1067 count nor any of the radRMS numbers. |
| 1186 | **b** | med | DP mode C … dc offset (−0.11, −0.01) cm, radial RMS 0.33 cm, vs plain MPC +2.13 cm … Square: CD 1.02 → 1.21 cm, CDW 3.09 → 3.51 cm | verify/verify_state_source.py and verify/verify_eaob.py print to stdout (verify_eaob only savefigs the two panels); no CSV/JSON of these A/B runs is stored, and no compare_* recording matches. |
| 1227 | **b** | med | config/base.yaml's wave block was strengthened at 2026-07-23 19:28 (Hs 0.75→1.2 m, Tp 12→6 s, γ 5→2, s 30→10, ω_max 1.6→3.0) … CDW NIS = 80(truth)/71… | config/base.yaml today holds a LIST of six sea states (storm…gentle), not the single Hs 1.2 / Tp 6 block, so the 'after' state is no longer inspectable; recordings/20260703/compare_20260703_111951/config.yaml preserves the 'before' (Hs 0.7… |
| 1285 | **b** | med | verify_eaob CD w-RMSE 0.24/0.34/0.39 N, if anything slightly better than the 0.29/0.41/0.48 at the old noise | No CD verify_eaob artifact of any vintage exists (all five archived PNGs are DP CDW, all dated 2026-07-23, i.e. before this change). |
| 1292 | **b** | med | \| dp C \| 0.22 / 0.26 / 0.31 \| 10.89 → 1.76 (truth 0.70) \| \| square C \| 1.13 / 1.20 / 1.16 \| 11.81 → 2.91 (truth 1.19) \| \| square CDW \| 3.19… | No artifact; verify_state_source.py stores nothing and no compare_* recording carries chatF/effF columns (results_raw.csv columns are scenario…n_fail, with 'slew' but not chatF). |
| 1387 | **b** | med | acados↔IPOPT max\|Δu\| = 0.0615 N < 0.25 N gate, 1.04 ms median, n_fail = 0); closed-loop n_fail ≈ 1 per 5334-tick run | The 0.25 N gate is verifiable (verify/verify_acados.py line 63) and 5334 ticks = 266.7 s × 20 Hz is arithmetic. The measured 0.0615 N and 1.04 ms have no stored output. The closed-loop n_fail is roughly corroborated by the archived sweep (… |
| 1396 | **b** | med | the MPC's error PSD is 3–8× below the PID's at every frequency under 0.7 rad/s in every sea state | No PSD artifact exists — no spectral columns in results_raw.csv (band_dc / band_wave / band_high are broadband RMS, not a PSD) and no stored spectrum figure. |
| 139 | **b** | low | verified C_A = −C_Aᵀ and == sim to 1e-14 (T1.1–1.2) | verify/verify_hydro_precise.py defines T1.1/T1.2/T1.3 (lines 96-119) but writes no artifact; no stored console log or JSON in recordings/ or docs/. |
| 675 | **b** | low | sensitivity: m_v 0.10→0.344 kg gives Ixx 0.321→0.362 | tools/compute_heavy_inertia.py exists and is named as the reproducer, but writes no artifact; no stored sweep output. |
| 726 | **b** | low | Full actuation actively levels pitch (11.8° trim → 0.8°) and tightens station-keeping (5.0 → 3.3 cm) | The 5.0 → 3.3 cm half reproduces from the two dp_compare PNGs. The '0.8°' does not: the table two lines above says pitch mean +0.4°, and the pitch trace in dp_compare_20260618_212557.png sits at ~0° with a 5.4° peak. No artifact yields 0.8… |
| 779 | **b** | low | 34 unit asserts + smoke pass | No test-run log on disk; the count would have to be re-derived by counting asserts across disturbance/test_*.py. |
| 803 | **b** | low | the logged yaw_deg ramps at ≤3.1°/log (no 90° jump) | No recording is named and no traj CSV in recordings/20260629 is identified with this check; the 20260629/square_view CSVs are 5-lap runs of a later configuration. |
| 894 | **b** | low | Beyond the \|S(1.6)\| ≈ 0.41 wave-band residual | A closed-loop sensitivity magnitude derived from the gains; no artifact, no script in the repo computes or stores it. |
| 962 | **b** | low | a torque-free 0.5 rad/s pitch kick exploded to \|q\| > 60 rad/s in 1.5 s | No saved log or figure of the diaginertia/fullinertia ablation; tests/test_heavy_gripper.py guards the fix but stores no measured divergence rate. |
| 1009 | **b** | low | Verified: vertex reconstruction error 0.000 mm | No build-time verification log is stored; the guard exists in the generators but emits nothing persistent. |
| 1024 | **b** | low | PID holds the origin at 0.0 cm; without the roll/pitch-leveling PD … the C3's forward-low COM leaves a ~1° residual pitch | No heavy_c3 recording exists (recordings/ has nothing after 20260727 and no heavy_c3-tagged run). tests/test_heavy_c3.py contains a PID-hold assertion but no stored measurement. |
| 1042 | **b** | low | the C3's near-square cross section (95×89 mm) made the error a clean-looking 90° twist | Not confirmed — I did not measure meshes/c3_camera.stl, and no artifact in the repo records these dimensions. |
| 1097 | **b** | low | Verified the 3rd-order closed loop … has all roots in the LHP with dominant pair ζ≈0.83–0.89 … slow companion pole at −0.67/−0.84 rad/s → residual nu… | Pure root-locus analysis with no stored computation; no script or output in the repo produces these poles. |
| 1106 | **b** | low | rarely binds at 120 N/s — ~66 N/s at hover | The 120 N/s limit is configuration (controller.py 'slew' 120.0, also in compare meta.json), but the '~66 N/s at hover' operating point has no artifact. |
| 1142 | **b** | low | per-axis w-error time constants 6.8–13.4 ms (tau-channel Kalman gains −0.976…−0.999) | Steady-state Riccati analysis with no stored output; no script in verify/ emits these. |
| 1160 | **b** | low | cross-checked against hydro's independent `diag_wtrue` diagnostic — the two agree to ≤0.08 N | The verify_eaob PNGs plot both w_true traces (model residual and hydro) and they visually overlap, but no numeric agreement figure is printed or stored. |
| 1172 | **b** | low | residual slopes vs \|nu\| ≈ 0 with \|r\| ≤ 0.18 everywhere in the final config | verify_eaob_*_corr_*.png files exist in recordings/20260723 but I did not open them and no numeric \|r\| appears in any archived filename or results file; the correlation value is not stored in a machine-readable artifact. |
| 1212 | **b** | low | verified: max per-tick \|Δpsi_hat\| ≤ 0.12 rad across all runs (a branch jump would be ~6.3) | No stored per-tick yaw diagnostic; verify_state_source.py writes nothing. |
| 1346 | **b** | low | `_R_from_rpy` round-trips to 4e-16; injected σ measured 5.24 mm / 4.82 mm·s⁻¹ / 0.482° vs the 5/5/0.5 targets; refresh cadence 25 substeps (20 Hz) | The 25-substep cadence is configuration (dobmpc/params.py line 103, 'DT_CTRL = 0.05 … = 25 * DT_SIM'). The three measured numbers (4e-16, 5.24 mm, 4.82 mm/s, 0.482°) have no stored output. |
| 1400 | **b** | low | the sea-state ladder co-varies Hs and Tp (corr(log Hs, log ω_p) = 0.967) … the Hs coefficient is not significant on 6 rungs, p = 0.22 | The six (Hs, Tp) rungs are recorded in compare_20260724_*/meta.json, so the correlation is recomputable in principle, but no regression output is stored and I did not reproduce 0.967 or p=0.22. |
| 1430 | **b** | low | all 15 (5 modes × 3 controllers) boxed values now equal `results_raw.csv` to ≤ 1.6e-6 cm | The re-rendered figures exist in that folder but carry no machine-readable residual; the ≤1.6e-6 cm agreement is not stored anywhere and I could not reproduce it without re-running the plotting code. |
| 999 | **c** | med | The C3-BR bracket straddles the Newton-gripper tube with ~1 mm clearance — independently confirming the guessed `GRIP_POS=[0.25, 0, −0.17]` is compat… | The clearance is computed between a CAD-registered bracket and a gripper position that is itself an unverified guess: KNOWN_ISSUES.md line 270 states GRIP_POS=[0.25,0,−0.17] is still an estimate, not present in the Onshape assembly, and th… |
| 1273 | **c** | med | on real hardware the SLAM+IMU FUSION owns the smoothing (its fused output is ~1–2 cm colored, far less punishing than the sim's 5 cm white @ 20 Hz), … | No measurement, dataset or citation backs the ~1–2 cm colored figure; it is attributed to a P1 advisor consult (.claude/journal 2026-07-23) inside a results section otherwise full of measured A/B numbers. The 5 cm sim figure IS configurati… |
| 704 | **c** | low | Heavy: 6×8, rank 6 — FULLY ACTUATED (verified: a pure pitch wrench realizes My=1.000) | My=1.000 for a pure-pitch command through B·pinv(B) is an algebraic identity of any full-rank allocation, not an experimental result; no saved verification output exists. rov_model.py/thrusters.py encode the 6×8 rank-6 layout but nothing o… |
| 198 | **a** | hig | under a 0.2 m/s current the integral nulls the DC bias to ~0.5 cm. But the wave band (≈13 cm radial std), impulsive kicks (~30 cm transients), and a … | `bluerov2_mujoco_marinegym/recordings/20260615/teleop_20260615_003536.csv` |
| 299 | **a** | hig | PID 13.3 cm / −0.1 · MPC 3.6 cm / +2.3 · DOB-MPC 3.7 cm / +0.3 (cm) | `bluerov2_mujoco_marinegym/recordings/20260615/dp_compare_20260615_102623.png` |
| 368 | **a** | hig | \| PID \| 14.3 cm \| 45.0 \| 39.8 cm \| 3.8 cm \| 14.2° / 33.5° \| … \| DOB-MPC \| 2.1 cm \| 17.4 \| 12.0 cm \| 1.8 cm \| 20.5° / 67.2° \| | `bluerov2_mujoco_marinegym/recordings/20260615/square_pid_20260615_120050.csv` |
| 1163 | **a** | hig | τ = 0.5 / 0.3 / 0.2 / 0.1 → NIS 25.4 / 19.8 / 17.1 / 14.8, NEES 77 / 26.1 / 15.0 / 11.5, w-RMSE_X 1.11 / 0.74 / 0.55 / 0.48 N | `bluerov2_mujoco_marinegym/recordings/20260723/verify_eaob_perf_traj_174953.png` |
| 521 | **a** | med | \| PID \| 14.86 \| 14.74 \| 15.16 \| … \| MPC \| 5.11 \| 4.30 \| 5.12 \| | `bluerov2_mujoco_marinegym/docs/figs/ablation_thrusters.png` |
| 723 | **a** | med | \| BlueROV2 (rank-5) \| 5.0 cm \| +11.8° \| 22.9° \| … \| Heavy (full 6-DOF) \| 3.3 cm \| +0.4° \| 5.4° \| | `bluerov2_mujoco_marinegym/recordings/20260618/dp_compare_20260618_202408.png` |
| 768 | **a** | med | \| C (current) \| 16.5 \| 2.9 \| 0.39 cm \| 7.4 \| \| CDW (+drift+waves) \| 13.3 \| 3.2 \| 0.47 cm \| 6.8 \| | `bluerov2_mujoco_marinegym/recordings/20260629/compare_20260629_015206/results.csv` |
| 849 | **a** | med | \| NONE (still water) \| 19.0 \| 1.9 \| 1.9 \| \| C (current) \| 29.5 \| 16.4 \| 14.6 \| \| CDW \| 67.5 \| 31.1 \| 21.4 \| | `bluerov2_mujoco_marinegym/recordings/20260629/compare_20260629_113356/results.csv` |
| 996 | **a** | med | C3 mesh centroid `[0.199, 0.008, −0.156]` … optical axis `[1.000, 0, 0.0056]` … cameras … (`x=0.2395`, `y=0.0055±0.0375`, `z=−0.1554`) | `bluerov2_mujoco_marinegym/meshes/c3_payload_frames.json` |
| 1056 | **a** | med | the n=200 corner analysis (compare_20260720_230025) … corner/edge RMS ratio ~2.4–2.9× for dobmpc, present identically in NONE | `bluerov2_mujoco_marinegym/recordings/20260720/compare_20260720_230025/runs` |
| 1161 | **a** | med | the perf profile passed only where the w_dot=0 model holds (DP CD: NIS 14.8 PASS) and failed under waves (DP CDW: NIS 25.4, NEES 77 | `bluerov2_mujoco_marinegym/recordings/20260723/verify_eaob_perf_traj_174455.png` |
| 1345 | **a** | med | `pid/NONE/truth` reproduces the recorded sweep `radial_max` 4.402 cm byte-for-byte | `bluerov2_mujoco_marinegym/recordings/20260724/compare_20260724_005250/wave00_storm/results.csv` |
| 1398 | **a** | med | it wins the disturbance-free NONE baseline (1.089 vs 1.492 cm) | `bluerov2_mujoco_marinegym/recordings/20260724/compare_20260724_160210/wave00_storm/results.csv` |
| 1399 | **a** | med | in constant-current C mode the nominal MPC still parks 2.70 cm downstream for want of integral action (PID 0.03 cm) | `bluerov2_mujoco_marinegym/recordings/20260724/compare_20260724_160210/wave00_storm/results.csv` |
| 1416 | **a** | med | \| PID \| 4.47 \| 18.57 \| 18.24 \| 18.57 \| \| MPC \| 6.13 \| 10.69 \| 10.55 \| 10.69 \| \| DOB-MPC \| 1.92 \| 2.62 \| 2.60 \| 2.62 \| | `bluerov2_mujoco_marinegym/recordings/20260727/compare_20260727_000850/wave01_moderate/runs` |
| 152 | **a** | low | B allocation matrix entries 0.707 / 0.051 / −0.002 / 0.167 … 4 horizontal thrusters at z = −0.0725 m | `bluerov2_mujoco_marinegym/bluerov.xml` |
| 159 | **a** | low | t200_thrust(+1)= +64.13 N, t200_thrust(−1)=−51.55 N (~1.24 fwd/rev asymmetry) | `bluerov2_mujoco_marinegym/thrusters.py` |
| 420 | **a** | low | pitch state constraint: \|θ_k\| ≤ θ_max, θ_max = 0.40 rad ≈ 23° ⇒ implicit optimal surge cap ≈ 5.9 N | `bluerov2_mujoco_marinegym/dobmpc/params.py` |
| 598 | **a** | low | voltage_scale = 4.81/6.54 = 0.74 (fwd), 3.74/5.26 = 0.71 (rev) → NOMINAL_VOLTAGE_SCALE = 0.72 | `bluerov2_mujoco_marinegym/thrusters.py` |
| 771 | **a** | low | EAOB estimates w_x≈1.5 N = the 0.2 m/s drag, est_err 0.01–0.06 N); only the wave-band residual grows (band_wave 0.21→0.44 cm). On square … DOB≈MPC (1… | `bluerov2_mujoco_marinegym/recordings/20260629/compare_20260629_015206/results.csv` |
| 878 | **a** | low | horizontal isotropic kp/kd/ki = 131.6/90.6/38.7 … heave 141.8/99.1/41.7, yaw 8.95/4.32/3.95 | `bluerov2_mujoco_marinegym/controller.py` |
| 882 | **a** | low | surge f_max 6→30 N, e_gate 0.5→0.15 m, surge_slew 30→120 N/s | `bluerov2_mujoco_marinegym/recordings/20260702/compare_20260702_222150/meta.json` |
| 888 | **a** | low | `--smoke` DP radial 1.4–1.5 cm (NONE/C/CD; wave modes 25 cm are 5 s-transient numbers) | `bluerov2_mujoco_marinegym/recordings/20260703/compare_20260703_111951/results.csv` |
| 889 | **a** | low | vs 17.8 cm with the old gains in `compare_20260702_222150` | `bluerov2_mujoco_marinegym/recordings/20260702/compare_20260702_222150/results.csv` |
| 890 | **a** | low | max 4.0 cm (corner transient), \|pitch\| 0.0°, no saturation chatter | `bluerov2_mujoco_marinegym/recordings/20260703/compare_20260703_111951/results.csv` |
| 940 | **a** | low | mass 13.724 kg … displaced volume 0.0131815 m³ → net buoyancy −5.7 N (sinks) | `bluerov2_mujoco_marinegym/rov_model.py` |
| 966 | **a** | low | sway m_eff 26.42 → kp/kd/ki 143.7/99.5/42.3; heave 28.29 → 153.9/108.0/45.3; yaw I_eff 0.811 → 9.92/4.79/4.38 … rp_kp=(4.3,7.7), rp_kd=(2.6,4.6) | `bluerov2_mujoco_marinegym/controller.py` |
| 969 | **a** | low | the payload's static attitude torque (jaw weight + CB_x offset ~1.3 N·m) rivals the passive B·coBM restoring (~1.2 N·m/rad) | `bluerov2_mujoco_marinegym/marinegym_assets/BlueROVHeavyGripper.yaml` |
| 1003 | **a** | low | composite inertia diag `[0.38154, 0.77780, 0.70954]` … coBM +0.01625 m (was +0.00955) … Ixz … +0.064 kg·m² (16.8% of Ixx) | `bluerov2_mujoco_marinegym/rov_model.py` |
| 1021 | **a** | low | mass 13.2 kg, I = [0.37014, 0.73153, 0.67460] … displaced volume 0.0129237 m³ → net buoyancy −3.1 N (sinks); coBM +0.01372 … Ixz +0.046 (12.4% of Ixx) | `bluerov2_mujoco_marinegym/rov_model.py` |
| 1095 | **a** | low | ki = I_eff·α·ωn³ … I_eff roll 0.481 → 2.60, pitch 0.859 → 4.64 N·m/(rad·s) … rp_gate = 0.15 rad, \|ki·I\| < rp_i_max = (1.5, 2.0) N·m | `bluerov2_mujoco_marinegym/controller.py` |
| 1140 | **a** | low | The covariances were unit-blind DT templates (`Q_pose=R=DT⁴/4`, `Q_vel=Q_dist=DT²`, `R = 1.5625e-6·I₁₈`) | `bluerov2_mujoco_marinegym/dobmpc/params.py` |
| 1169 | **a** | low | an early sweep accidentally run WITH the gate enabled reported wildly inflated NIS (137/67/43/27 for τ 0.5→0.05) | `bluerov2_mujoco_marinegym/recordings/20260723/verify_eaob_perf_traj_172118.png` |
| 1177 | **a** | low | can reject frames at χ²(0.999,18)=42.31 (`params.EAOB_GATE_ON`) | `bluerov2_mujoco_marinegym/dobmpc/params.py` |
| 1283 | **a** | low | x/y 0.5 cm, z 0.3 cm, attitude 0.3° (yaw 0.5°), lin-vel 0.5 cm/s, gyro 0.11°/s | `bluerov2_mujoco_marinegym/dobmpc/params.py` |
| 1354 | **a** | low | On heavy (rank-6) `params.U_MAX` was `[8, 30, 30, 8, 8, 10]` — surge capped at 8 N while the PID's `f_max` is 30 N on every axis, a 3.75× handicap … … | `bluerov2_mujoco_marinegym/dobmpc/params.py` |
| 1372 | **a** | low | The recorded `sat_freq` metric is blind to it (0.0000 everywhere) | `bluerov2_mujoco_marinegym/recordings/20260724/compare_20260724_005250/wave00_storm/results.csv` |
| 1390 | **a** | low | run meta now stamps `controller.u_max` … Absent key = the old 8 N box | `bluerov2_mujoco_marinegym/experiments/run_compare.py` |
| 1407 | **a** | low | `bar_square_radial_rms.png` put PID/CDW at 18.6 cm, while the boxed number on `trajectory_compare_CDW.png` read 18.2 ± 2.4 cm | `bluerov2_mujoco_marinegym/recordings/20260727/compare_20260727_000850/wave01_moderate` |
| 1412 | **a** | low | those first 10 s — only 3.7 % of the 266.7 s record (196 / 5268 samples) | `bluerov2_mujoco_marinegym/recordings/20260727/compare_20260727_000850/wave01_moderate/runs` |
| 1431 | **a** | low | `plot_trajectories` on one run prints 17.26 / 10.88 / 2.79 cm against that run's recorded 17.26 / 10.88 / 2.79 | `bluerov2_mujoco_marinegym/recordings/20260727/compare_20260727_000850/wave01_moderate/results_raw.csv` |
| 1436 | **a** | low | Figures generated before this date box a full-run RMS, ~1–2 % below the matching bar | `bluerov2_mujoco_marinegym/recordings/20260727/compare_20260727_000850/wave01_moderate/runs` |

### [bluerov2_mujoco_marinegym/docs/HYDRO_VERIFICATION.md](../bluerov2_mujoco_marinegym/docs/HYDRO_VERIFICATION.md)  — a 9 / b 5 / c 1

| 줄 | 판정 | sev | 주장 | 산출물 / 사유 |
|---:|:---:|:---:|---|---|
| 32 | **b** | low | pitch 4.79 vs 5.16 (7%) | The prediction 5.16 s is recomputable from verify/verify_hydro.py (I_eff = 0.626+0.12, k=1.1097, zeta=0.0385 → Tn = 5.155 s). The MEASURED 4.79 s is not: verify_hydro.py:237 only plots the pendulum when ax==0, so the saved figure is roll-o… |
| 33 | **b** | low | 26.1° vs 26.8° (2.7%) | The prediction is recomputable — asin(0.5/1.10969) = 26.78° — matching 26.8. The measured 26.1° is stdout-only (verify_hydro.py:257 record(...)) and no figure is saved for the static-equilibrium test. |
| 35 | **b** | low | **0.0–0.3%** on surge/sway/heave; sign −M_A on all 6 axes | I opened the figure. It confirms the measured effective-mass curves overlay the predicted ones at 16.7 / 23.9 / 25.77 kg (= m + M_A, which I recomputed) across Ω = 0.5–5 rad/s, so the claim is qualitatively supported. But a 0.3 % offset is… |
| 36 | **b** | low | νᵀC_A(ν)ν = 0 \| **4.3e-14** | verify/verify_hydro.py:357 computes exactly this over 2000 seeded random ν and record()s it to stdout with a <1e-10 gate. No stdout log is saved anywhere in the repo and no figure carries the value, so the specific 4.3e-14 cannot be confir… |
| 97 | **b** | low | reproduces hydro's hand-typed `_coriolis_added` to **1.4 × 10⁻¹⁴** | verify/verify_hydro_precise.py:101 produces exactly this quantity (max over 5000 seeded ν, gated <1e-12), but Tier 1 saves no figure and no log. Tier-2/Tier-4 figures prove the script was executed (docs/figs/hydro_P_convergence.png and hyd… |
| 65 | **c** | low | The EMA filter's corner is ~230 rad/s | This is an analytic filter property, not a measurement, and it does not reproduce. For the EMA used (α=0.3, dt=2 ms) I computed \|H(ω)\| = α/\|1−(1−α)e^{−jωT}\|: the −3 dB corner is 180.3 rad/s; the discrete pole maps to −ln(0.7)/dt = 178.… |
| 3 | **a** | med | **Result: 32/32 checks PASS.** | `bluerov2_mujoco_marinegym/verify/verify_hydro.py` |
| 20 | **a** | med | B=ρgV=**110.97 N**, W=mg=**109.87 N**, net **+1.10 N**, restoring stiffness k=coBM·B=**1.1097 N·m/rad** | `bluerov2_mujoco_marinegym/marinegym_assets/BlueROV.yaml` |
| 29 | **a** | med | **0.00%** on all 4 axes | `bluerov2_mujoco_marinegym/docs/figs/hydro_T2_terminal.png` |
| 32 | **a** | med | roll T=3.85 s vs 3.89 (1%) | `bluerov2_mujoco_marinegym/docs/figs/hydro_T4_pendulum.png` |
| 103 | **a** | med | falls as **O(dt¹)** with observed order **p̂ = 1.000** across the ladder dt = 2 → 0.125 ms (position L2 0.564 → 0.035 mm; Richardson dt→0 ≈ 2.5 × 10⁻… | `bluerov2_mujoco_marinegym/docs/figs/hydro_P_convergence.png` |
| 116 | **a** | med | an **equivalent transport delay ≈ 5.67 ms**, essentially constant with frequency, and an effective added-mass error **< 0.013 %** in the ROV disturba… | `bluerov2_mujoco_marinegym/docs/figs/hydro_P_lagfidelity.png` |
| 37 | **a** | low | dissipated 4.59 J, **monotone** | `bluerov2_mujoco_marinegym/docs/figs/hydro_T6_energy.png` |
| 39 | **a** | low | transient divergence **0.01 cm/s**; same terminal | `bluerov2_mujoco_marinegym/docs/figs/hydro_T7_R1.png` |
| 98 | **a** | low | M = M_RB + M_A is SPD (eigenvalues 0.42–25.8) | `bluerov2_mujoco_marinegym/verify/verify_hydro_precise.py` |

### [bluerov2_mujoco_marinegym/docs/REAL_HYDRO_VERIFICATION.md](../bluerov2_mujoco_marinegym/docs/REAL_HYDRO_VERIFICATION.md)  — a 5 / b 0 / c 0

| 줄 | 판정 | sev | 주장 | 산출물 / 사유 |
|---:|:---:|:---:|---|---|
| 22 | **a** | low | mass 11.2 kg, I=[0.30375, 0.626, 0.5769], V=0.0113459 m³, coBM=0.01 m, M_A=[5.5, 12.7, 14.57, 0.12, 0.12, 0.12], D_L=[4.03, 6.22, 5.18, 0.07, 0.07, 0… | `bluerov2_mujoco_marinegym/marinegym_assets/BlueROV.yaml` |
| 36 | **a** | low | ~10–20% weaker reverse | `bluerov2_mujoco_marinegym/thrusters.py` |
| 41 | **a** | low | this replaces the `force_constants: 4.4e-7` assumption in `BlueROV.yaml` | `bluerov2_mujoco_marinegym/marinegym_assets/BlueROV.yaml` |
| 48 | **a** | low | `config/config.yaml` lists pool width **1.8 m** while the working figure has been **2.438 m** — reconcile by measurement before computing run-up dist… | `config/config.yaml` |
| 123 | **a** | low | (sim surge 5.5/11.2 ≈ 49%, plausible) | `bluerov2_mujoco_marinegym/marinegym_assets/BlueROV.yaml` |

### [bluerov2_mujoco_scratch/README.md](../bluerov2_mujoco_scratch/README.md)  — a 5 / b 0 / c 0

| 줄 | 판정 | sev | 주장 | 산출물 / 사유 |
|---:|:---:|:---:|---|---|
| 19 | **a** | low | static stability: a torque kick → roll/pitch self-right (period 2.33 s ≈ 2π/√(z_G·W/I)) | `bluerov2_mujoco_scratch/phase1_static_equilibrium.py` |
| 21 | **a** | low | terminal velocity = 2.028 cm/s | `bluerov2_mujoco_scratch/phase3_hydrodynamics.py` |
| 22 | **a** | low | set-point regulation: 1.27 m offset → 0.14 cm | `bluerov2_mujoco_scratch/phase4_mpc.py` |
| 23 | **a** | low | disturbance tracking (±0.2 N) + DOBMPC holds station (~1–2 cm) under 10 N current | `bluerov2_mujoco_scratch/phase5_eaob_dob.py` |
| 97 | **a** | low | CG sits `z_G = 0.02 m` below it | `bluerov2_mujoco_dobmpc/bluerov2mj/params.py` |

### [c3_camera/README.md](../c3_camera/README.md)  — a 25 / b 40 / c 3

| 줄 | 판정 | sev | 주장 | 산출물 / 사유 |
|---:|:---:|:---:|---|---|
| 215 | **b** | hig | 컬러 latency (p50) \| 약 **3000 ms** [Madrona (H264/RTSP, 파이 경유)] | No Madrona/RTSP measurement exists anywhere in the repo: grep for '3000 ms'/'rtsp' across c3_camera/*.py and every meta.json/metadata.json/CSV returns nothing. recordings/ and datasets/ contain only DepthAI-direct captures. |
| 231 | **b** | hig | 컬러만 960x540 NV12 \| 13.8 \| 778 kB \| 86 Mbit/s | c3_bench.py never ran (no bench/ dir, no CSV). No NV12/raw capture exists in recordings/ or datasets/ at 960x540 — every recording uses color_encode mjpeg. Frame size 778 kB is recomputable (960*540*1.5), the 13.8 fps and 86 Mbit/s are not. |
| 232 | **b** | hig | 960x540 NV12 + depth \| 9.25 \| 1.24 MB \| **91.7** Mbit/s | No raw-NV12 capture exists on disk. (recordings/20260803_150048 also reports 91.7 Mbit/s total, but that is a 1920x1080 MJPEG+depth run, not this config.) |
| 233 | **b** | hig | 1920x1080 NV12 + depth \| 3.19 \| 3.57 MB \| **91.1** Mbit/s | No raw-NV12 capture on disk; no bench CSV. |
| 234 | **b** | hig | 480x270 NV12 + depth \| 17.5 \| 655 kB \| **91.6** Mbit/s | No raw-NV12 capture at 480x270+depth on disk. datasets/dataset_20260729_173152 is the closest (480x270 NV12 colour + depth + stereo @8 fps) and reports 61.9 Mbit/s at 8 fps — not this row. |
| 254 | **b** | hig | 480x270 \| raw \| 17.5 \| 123 ms \| 121 ms \| 12.5 % | c3_bench.py never executed; no bench/ directory or CSV anywhere. No raw (color_encode none) capture at 480x270 with depth 640x360 exists in recordings/ or datasets/. |
| 255 | **b** | hig | 480x270 \| **mjpeg** \| **20.0** \| **57.6 ms** \| 107 ms \| **0 %** | No bench artifact. Nearest saved capture with 480x270 MJPEG @20 is datasets/dataset_20260729_181831, which reports colour p50 32.2 ms and depth p50 70.5 ms at 480x270 depth — different config and different numbers. |
| 256 | **b** | hig | 960x540 \| raw \| 9.1 \| 250 ms \| 405 ms \| 53.5 % | No bench artifact; no raw 960x540 capture on disk. |
| 257 | **b** | hig | 960x540 \| **mjpeg** \| **19.1** \| **123 ms** \| 124 ms \| 5 % | No bench artifact. recordings/20260803_145852 is 960x540 mjpeg @20 with depth 640x360 and reports fps 20.0, colour p50 74.2 ms, depth p50 114.5 ms, drop 0% — it does not reproduce this row. |
| 258 | **b** | hig | 1920x1080 \| raw \| 3.2 \| 353 ms \| 495 ms \| 83.5 % | No bench artifact; no raw 1080p capture on disk. |
| 259 | **b** | hig | 1920x1080 \| **mjpeg** \| **13.1** \| **147 ms** \| 154 ms \| 34.8 % | No bench artifact. recordings/20260803_150048 (1920x1080 mjpeg @20 + depth) reports fps_lifetime 16.44, colour p50 138.0 ms, drop 17.6% — close in spirit, different numbers. |
| 265 | **b** | hig | MJPEG 실측 압축률은 q90에서 **약 6:1** (960x540이 112–139 kB/frame) | An artifact exists but does NOT reproduce the range. recordings/20260729_162216 is 960x540 MJPEG at mjpeg_quality 90; I measured all 365 JPEGs on disk: min 97.9 KiB, max 133.0 KiB, mean 122.8 KiB (=100.2 / 136.2 / 125.8 kB decimal). Neithe… |
| 291 | **b** | hig | 640x360 mjpeg @12 + depth 480x270 \| 67 \| 31.8 \| 35 % \| **35.5 ms** \| 36.1 | c3_bench.py never ran; no bench CSV. No capture with this config exists in recordings/ or datasets/. |
| 292 | **b** | hig | 640x360 mjpeg @20 + depth 480x270 \| 67 \| 52.0 \| 58 % \| **35.0 ms** \| 35.8 | No bench CSV; no such capture on disk. |
| 293 | **b** | hig | 960x540 mjpeg @15 + depth 640x360 **(기본)** \| 131 \| 71.4 \| 78 % \| 39 ms \| 80 | Artifact for this exact config exists (recordings/20260729_162216/meta.json) but does not reproduce it: mean_frame_kb 124.5 (not 131), mbps_measured_total 70.6 (not 71.4), colour p50 37.5 (not 39); only p95 79.6 ≈ 80 matches. |
| 294 | **b** | hig | 1920x1080 mjpeg @12 + depth 480x270 \| 377 \| 61.9 \| 68 % \| 77 ms \| 84 | No bench CSV; no 1080p @12 capture on disk. recordings/20260803_150048 is 1080p @20 (p50 138 ms, 91.7 Mbit/s). |
| 295 | **b** | hig | 480x270 mjpeg @20 + depth 640x360 \| ~30 \| ~77 \| 84 % \| 58 ms \| 58 | No bench CSV; no such capture. (datasets/dataset_20260729_181831 is 480x270 mjpeg @20 but with 480x270 depth: colour p50 32.2 ms, 47.0 Mbit/s.) |
| 296 | **b** | hig | 960x540 mjpeg @30 + depth **480x270** \| 131 \| 90.4 \| 99 % \| 84 ms \| 96 | No bench CSV; no @30 fps capture exists anywhere in recordings/ or datasets/. |
| 297 | **b** | hig | 960x540 mjpeg @30 + depth 640x360 \| 131 \| 88.2 \| 96 % \| 120 ms \| — | No bench CSV; no @30 fps capture on disk. |
| 298 | **b** | hig | 1920x1080 mjpeg @20 \| 377 \| 포화 \| >100 % \| 147 ms \| 170 | No bench CSV. Closest artifact recordings/20260803_150048 (1080p mjpeg @20 + depth) gives colour p50 138.0 / p95 161.5, mean_frame_kb 231.1 — not 147/170/377. |
| 299 | **b** | hig | 960x540 raw @20 \| 778 \| 포화 \| >100 % \| 250 ms \| 284 | No bench CSV; no raw capture on disk at all. |
| 300 | **b** | hig | 1920x1080 raw @20 \| 3111 \| 포화 \| >100 % \| 353 ms \| 395 | No bench CSV; no raw capture on disk. 3111 kB is recomputable (1920*1080*1.5), the latencies are not. |
| 325 | **b** | hig | 전부 실제로 측정한 값이다 (예측 아님) | Framing sentence for the recommendation table at lines 327-333. Of its five rows, none is reproducible from any file in the repo (see next five entries). |
| 330 | **b** | hig | `--isp-scale 1/3 --fps 20 --depth-size 480x270` \| 20.0 fps, 컬러 **35.0 ms** (p95 35.8, max 36.1), drop 0 %, 천장 58 % | No capture with isp-scale 1/3 exists in recordings/ or datasets/ (all use 1/2, 1/4, 2/3, 3/4 or none). No bench CSV. |
| 331 | **b** | hig | `--fps 30 --depth-size 480x270` \| **28.5 fps**, 84 ms, drop 5 %, 천장 99 % | No 30 fps capture exists anywhere on disk (max requested fps in any meta.json/metadata.json is 20.0). No bench CSV. |
| 332 | **b** | hig | `--isp-scale none --fps 12 --depth-size 480x270` \| 12.0 fps, **77 ms**, drop 0 %, 천장 68 % | No 12 fps capture on disk. The only isp-scale-none run (recordings/20260803_150048, @20) dropped 17.6% at 91.7 Mbit/s. |
| 217 | **b** | med | fps \| 10 요청 → 6–8 실측 [Madrona] | No Madrona-path capture exists in the repo. Nothing in recordings/, datasets/, or any CSV records the RTSP path's fps. |
| 221 | **b** | med | 420프레임 연속 측정, 실측 링크 71.6 Mbit/s | c3_bench.py has never been run: no c3_camera/bench/, no bench.csv, no *.csv from the bench tool anywhere (find over the whole repo). The only artifact for this exact config, recordings/20260729_162216, has 365 frames on disk / 192 in the m… |
| 261 | **b** | med | 960x540에서 fps 2.1배, 지연 1/2. 풀 1080p에서는 fps 4.1배, 지연 1/2.4. | Arithmetic on the unsupported table above (19.1/9.1=2.1, 13.1/3.2=4.1). No underlying artifact exists. |
| 263 | **b** | med | `--color-encode none`(대신 약 9 fps) | Derived from the 9.1 fps row of the unsupported raw table; no raw 960x540 capture exists on disk. |
| 302 | **b** | med | 68 %(1080p, 77 ms)가 78 %(기본, 39 ms)보다 느리다 | Restates two rows of the unsupported table (lines 293-294). |
| 311 | **b** | med | 검산: 640x360(67 kB) → 30 + 6 = 36 vs 실측 35.0. 960x540(131 kB) → 30 + 11 = 41 vs 실측 39. 1080p(377 kB) → ... 실측 77. | All three '실측' values are rows of the unsupported latency table. For the one config that does have an artifact (960x540 default) the saved p50 is 37.5 ms, not 39. |
| 318 | **b** | med | **천장의 60 % 아래**(약 55 Mbit/s)면 큐잉 항이 사라지고 지터도 없어진다 (p95−p50 < 1 ms) | The '<1 ms' comes from the 35.8-35.0 row at line 292, which has no artifact. Saved captures below 60% of ceiling do behave similarly (e.g. datasets/dataset_20260729_181831 at 47.0 Mbit/s: colour p50 32.2, p95 33.4 -> 1.2 ms), but that is a… |
| 329 | **b** | med | **균형 (기본값)** \| 15.0 fps, 컬러 **39 ms**, drop 0 % | The artifact for this config (recordings/20260729_162216/meta.json) confirms 15.0 fps and 0% drop but gives colour p50 37.5 ms, not 39. The 39 ms comes from the unsaved bench run. |
| 333 | **b** | med | **무손실 픽셀** \| `--color-encode none --fps 9` \| 약 9 fps, NV12 원본 | No raw/NV12 colour capture at 960x540 exists on disk; derived from the unsupported raw table row at line 256. |
| 356 | **b** | med | **공기 중 실내 테스트에서 depth valid ≈ 21%** 였다 | I decoded the depth PNGs of the only in-air-era recording (recordings/20260729_162216, every 10th of 365 frames, cv2.IMREAD_UNCHANGED): mean valid 15.5%, range 6.9-25.8%. The two later recordings with depth give 60.6% and 60.0%. No file an… |
| 106 | **b** | low | `DEPTHAI_BOOTUP_TIMEOUT` \| `30000` ms \| 실측 부팅 약 12초인데 기본값이 15초라 | The 30000 value IS in c3_camera/__init__.py:50 (with comment 'default 15000') -> config part is fine. The '실측 부팅 약 12초' has no artifact: no timing log, no bench CSV, nothing in recordings/ or datasets/ records connect time. |
| 278 | **b** | low | 드롭 38%로 9 fps 받는 것보다 9 fps 요청해서 드롭 0%가 낫다 | No capture with 38% drop at 9 fps exists. Highest drop rates on disk are 17.6% (recordings/20260803_150048) and 32.0% (datasets/dataset_20260729_174310 at 13.6 fps). |
| 349 | **b** | low | **연결에 약 12초** 걸린다 | No boot/connect timing is recorded in any artifact. Nothing in recordings/*/meta.json or datasets/*/metadata.json logs connection duration. |
| 405 | **b** | low | 기본 mjpeg (실측 약 1/6). none은 무손실 대신 약 9 fps | Both figures restate the unsupported bench results (see lines 265 and 263). MJPEG_RATIO=6.0 exists in config.py:96, but the 9 fps has no artifact. |
| 534 | **c** | hig | 공장 캘리브레이션은 **공기 중** 캘리브레이션이다 | Stated as an established property of the device. The only 'evidence' in the repo is our own hardcoded string: c3_collect.py:330 writes note='IN-AIR calibration...' into calibration.json, and dataset.py:909 writes 'These are IN-AIR values' … |
| 306 | **c** | med | 컬러 지연 ≈ 30 ms (고정: 센서 readout + ISP + 인코딩) | This is an intercept fitted by eye to the ten table rows at lines 289-300, none of which have an artifact. It is presented inside '측정값을 전부 모아 보면' as if extracted from data. |
| 452 | **c** | low | 대략 **150 MB/분 = 9 GB/시간** 수준 | Preceded by '용량은 시작할 때 알려준다', which implies the program produced it. It did not: recorder.py:320-330 estimate_disk_mb_per_min for 640x360 mjpeg @20 + 480x270 depth gives (56.25 + 253.1*0.45)*20*60/1024 ≈ 199 MB/min, not 150. The real measu… |
| 215 | **a** | med | [DepthAI 직결 (기본 설정)] 컬러 latency (p50) **38 ms** | `c3_camera/recordings/20260729_162216/meta.json` |
| 225 | **a** | med | PoE 링크 실효 대역폭 ≈ 91.5 Mbit/s | `c3_camera/recordings/20260803_150048/meta.json` |
| 22 | **a** | low | product \| **OAK-D-W-POE** (wide-FOV OAK-D PoE) | `c3_camera/datasets/dataset_20260803_162223/metadata.json` |
| 23 | **a** | low | mx_id \| `19443010315B0B2F00` @ 192.168.2.191 | `c3_camera/recordings/20260803_145852/meta.json` |
| 24 | **a** | low | bootloader \| 0.0.28 | `c3_camera/datasets/dataset_20260803_162223/metadata.json` |
| 25 | **a** | low | CAM_A \| IMX378 컬러, native 4056x3040, **고정 초점** (autofocus 없음) | `c3_camera/datasets/dataset_20260803_162223/metadata.json` |
| 26 | **a** | low | CAM_B / CAM_C \| OV9282 mono (left/right), native 1280x800 | `c3_camera/datasets/dataset_20260803_162223/metadata.json` |
| 27 | **a** | low | stereo baseline \| 7.5 cm, EEPROM에 공장 캘리브레이션 있음 | `c3_camera/datasets/dataset_20260803_162223/calibration.json` |
| 28 | **a** | low | IR 프로젝터 \| 없음 (dot projector / flood LED 미탑재) | `c3_camera/datasets/dataset_20260803_162223/metadata.json` |
| 29 | **a** | low | 컬러 지원 모드 \| 1920x1080@60, 2024x1520@85, 1352x1012@52, 3840x2160@42, 4056x3040@30 | `c3_camera/datasets/dataset_20260803_162223/metadata.json` |
| 30 | **a** | low | mono 지원 모드 \| 640x400@255, 1280x720@143, 1280x800@129 | `c3_camera/datasets/dataset_20260803_162223/metadata.json` |
| 41 | **a** | low | 검증된 조합: **python 3.10.12 / depthai 2.32.0.0 / opencv 4.11.0 / numpy 1.26.4** | `c3_camera/setup_env.sh` |
| 46 | **a** | low | conda base는 **python 3.12.11이고 이미 `depthai 3.5.0`이 깔려 있다** | `c3_camera/__init__.py` |
| 161 | **a** | low | 기본 사다리: 컬러 2크기 × raw/MJPEG × depth on/off × depth 2그리드 = 12셀 | `c3_camera/c3_option_sweep.py` |
| 179 | **a** | low | `*_mbps`는 **마지막 120프레임** 창 | `c3_camera/metrics.py` |
| 216 | **a** | low | depth latency (p50) \| **82 ms** | `c3_camera/recordings/20260729_162216/meta.json` |
| 217 | **a** | low | **15 요청 → 15.00 실측** | `c3_camera/recordings/20260729_162216/meta.json` |
| 218 | **a** | low | 프레임 드롭 \| **0 %** | `c3_camera/recordings/20260729_162216/meta.json` |
| 244 | **a** | low | `config.py`의 `POE_BUDGET_MBPS = 90` | `c3_camera/config.py` |
| 280 | **a** | low | 기본 설정에서 depth가 55 Mbit/s로 컬러(16)의 3.5배다 | `c3_camera/recordings/20260729_162216/meta.json` |
| 281 | **a** | low | 640x360 depth16 = 461 kB인데 MJPEG 컬러는 약 130 kB | `c3_camera/recordings/20260729_162216/meta.json` |
| 320 | **a** | low | 컬러 지연 바닥은 **약 30–35 ms**이고 그 아래로는 못 내려간다 | `c3_camera/datasets/dataset_20260729_181831/metadata.json` |
| 351 | **a** | low | **depth 지연 > 컬러 지연** (82 vs 38 ms)은 정상이다 ... 프레임도 3.5배 크다 | `c3_camera/recordings/20260729_162216/meta.json` |
| 353 | **a** | low | **컬러/depth skew 약 66 ms** (≈ 1 프레임) | `c3_camera/recordings/20260729_162216/frames.csv` |
| 573 | **a** | low | 카메라 없이 도는 회귀 테스트 47개 | `c3_camera/tests/test_offline.py` |

### [c3_camera/README_DATASET.md](../c3_camera/README_DATASET.md)  — a 13 / b 12 / c 6

| 줄 | 판정 | sev | 주장 | 산출물 / 사유 |
|---:|:---:|:---:|---|---|
| 155 | **b** | hig | 카메라 IMU \| 200 Hz 요청 → **약 130 Hz 실측** | Directly contradicted by the dataset this table describes: datasets/dataset_20260729_173152/metadata.json imu = {requested_rate_hz 200, rate_hz 200.4, samples 7600}. All ten datasets record 199.3-200.4 Hz; none records 130 Hz. |
| 229 | **b** | hig | **측정 결과 ... `dai.Clock.now()`와 `time.monotonic()`은 같은 시계 (CLOCK_MONOTONIC)이고 8 마이크로초 이내로 일치합니다.** | No clock-comparison measurement is stored anywhere. The apparent corroboration in every dataset's metadata.txt ('agreeing to within 8 microseconds') is a HARDCODED string literal: dataset.py:804 emits it unconditionally, and dataset.py:7 r… |
| 6 | **b** | med | 카메라 직결 자체(지연 3000 ms → 35 ms) | Same missing Madrona baseline as README.md:215; and the direct-path figure here is 35 ms while README.md:215 says 38 ms for the same claim. The 35 ms corresponds to the unsupported bench row at README.md:292. |
| 153 | **b** | med | 디스크 \| 약 11–13 MB/s (약 21 GB/시간) | The dataset this table describes records writer.mb_per_s == 4.21 (119.8 MB in 28.5 s), not 11-13. No run in datasets/ exceeds 4.3 MB/s. The row is also internally inconsistent: 11-13 MB/s is 40-47 GB/hour, not 21. |
| 154 | **b** | med | 지연 \| 컬러 p50 88 ms, depth 89 ms | The artifact for this exact run gives colour latency_p50_ms 98.7 and depth 98.6 (metadata.json), and recomputing from frames.csv over all 227 rows gives colour p50 99.54, depth p50 99.45. 88/89 matches only the MINIMUM latency in the run (… |
| 263 | **b** | med | \| accel+gyro @500 요청 \| **486 Hz** \| | No capture at a 500 Hz IMU request exists: all ten datasets record imu.requested_rate_hz == 200. No bench/sweep output is stored. |
| 264 | **b** | med | \| accel+gyro @200 \| 194 Hz \| | Artifacts exist for exactly this configuration and disagree: all ten datasets requested 200 Hz and achieved 199.3-200.4 Hz (e.g. dataset_20260729_173152 200.4, dataset_20260730_173057 199.5), never 194. |
| 265 | **b** | med | \| accel+gyro @200 **+ rotvec @100** \| 97 Hz (전부 반토막) \| | No dataset was recorded with the rotation vector enabled (no dataset metadata records a rotvec stream), so nothing on disk reproduces this. |
| 266 | **b** | med | \| accel+gyro @200 **+ mag @100** \| 97 Hz (전부 반토막) \| | No dataset was recorded with the magnetometer enabled; nothing on disk reproduces this. |
| 267 | **b** | med | \| accel+gyro @200 **+ rotvec @200** \| **194 Hz** (손실 없음) \| | No rotation-vector capture exists on disk; and 194 Hz conflicts with the 199-200 Hz actually recorded for plain accel+gyro @200. |
| 276 | **b** | med | \| 1 [--imu-batch] \| 130 Hz 수신, **35% 손실** \| | No dataset was recorded with --imu-batch 1 (all show 199.3-200.4 Hz, consistent with the default batch of 10). Nothing on disk records a 35% IMU loss. |
| 146 | **b** | low | 기본 설정 480x270 컬러+depth + 640x400 스테레오 @8 fps, 20초 실측 | The config matches datasets/dataset_20260729_173152 exactly, but its duration is not 20 s: metadata.json duration_s 399.1-equivalent field reads 37.9 s, writer.seconds 28.5, and 304 colour frames at 8.04 fps = 37.8 s. |
| 255 | **c** | med | ROV의 Navigator IMU는 수십 cm 떨어져 있고 | Unmeasured, and it contradicts our own generated artifact: dataset.py:865 writes 'Metres away from the camera' into every metadata.txt (verified in datasets/dataset_20260803_162223/metadata.txt). No mechanical measurement exists in the rep… |
| 142 | **c** | low | 용량 \| 약 0.5 GB/시간 \| 약 21 GB/시간 | No capture reaches 21 GB/hour. The research-default 4-stream lossless run (datasets/dataset_20260729_173152) wrote 119.8 MB in 28.5 s = 4.21 MB/s = 15.2 GB/hour; the largest run (dataset_20260803_162223) wrote 1083.3 MB in 259.9 s = 4.17 M… |
| 165 | **c** | low | \| 480x270 \| 11.8 [링크 상한 fps] | Formula value, not measured: 4 lossless streams at 480x270 (+640x400 mono) = 480*270*1.5 + 480*270*2 + 2*640*400 = 965600 B = 7.725 Mbit; 91.5/7.725 = 11.84. Reproduces exactly. |
| 166 | **c** | low | \| 640x360 \| 8.7 [링크 상한 fps] | Formula: 345600 + 460800 + 512000 = 1318400 B = 10.547 Mbit; 91.5/10.547 = 8.67. Reproduces exactly; no capture at this config exists. |
| 167 | **c** | low | \| 960x540 \| 4.9 [링크 상한 fps] | Formula: 777600 + 1036800 + 512000 = 2326400 B = 18.611 Mbit; 91.5/18.611 = 4.92. Reproduces exactly; no capture at this config exists. |
| 168 | **c** | low | `--streams color,depth`로 스테레오를 빼거나(14 fps) | Formula value and ambiguous: 14 fps corresponds to 640x360 colour+depth (91.5 / 6.45 Mbit = 14.2), not to the 480x270 default the sentence follows (which gives 25.2 fps). No capture without stereo at 8 fps exists. |
| 296 | **a** | med | 정지 상태에서 \|a\| = **11.8 m/s²** (중력 9.81 대비 약 20% 높음) | `c3_camera/datasets/dataset_20260729_173152/metadata.json` |
| 71 | **a** | low | BlueOS에 이미 돌고 있는 MAVLink2Rest(포트 6040)를 폴링한다 | `c3_camera/datasets/dataset_20260803_162223/metadata.txt` |
| 111 | **a** | low | `--video-bitrate-kbps`의 기본값은 4000이고, keyframe은 기본 1초마다 | `c3_camera/config.py` |
| 123 | **a** | low | left/right ... 그것들만 약 82 Mbit/s이고, 480x270 depth 약 41 Mbit/s | `c3_camera/datasets/dataset_20260803_162223/metadata.json` |
| 140 | **a** | low | 기본 설정 \| 960x540 MJPEG @15 [review] \| 480x270 무손실 @8, depth 1:1 [research] | `c3_camera/c3_collect.py` |
| 150 | **a** | low | 프레임 \| 8.0 fps 요청 → **8.0 실측, 드롭 0** | `c3_camera/datasets/dataset_20260729_173152/metadata.json` |
| 151 | **a** | low | 이미지 \| 프레임당 4장 무손실 PNG | `c3_camera/datasets/dataset_20260729_173152/metadata.json` |
| 152 | **a** | low | 링크 \| 62 Mbit/s (실측 천장 91.5의 69%) | `c3_camera/datasets/dataset_20260729_173152/metadata.json` |
| 159 | **a** | low | PNG 인코딩도 프레임당 17 ms(4장 합계) | `c3_camera/datasets/dataset_20260729_173152/metadata.json` |
| 254 | **a** | low | BNO086 (`getConnectedIMU()`로 확인, firmware 3.9.9) | `c3_camera/datasets/dataset_20260803_162223/metadata.json` |
| 277 | **a** | low | \| 10 (기본) \| **199.5 Hz 수신, 손실 0** \| | `c3_camera/datasets/dataset_20260730_173057/metadata.json` |
| 284 | **a** | low | `metadata.txt`에는 **요청값이 아니라 실측값**이 기록된다 | `c3_camera/datasets/dataset_20260803_162223/metadata.json` |
| 289 | **a** | low | `getImuToCameraExtrinsics()` → `IMU calibration data is not available on device yet.` | `c3_camera/datasets/dataset_20260803_162223/calibration.json` |

### [c3_camera/TESTING.md](../c3_camera/TESTING.md)  — a 22 / b 10 / c 11

| 줄 | 판정 | sev | 주장 | 산출물 / 사유 |
|---:|:---:|:---:|---|---|
| 130 | **b** | hig | "960x540 mjpeg @15 + depth 640x360" = 71.4 Mbit/s (78%), p50 39 ms, p95 80 ms | c3_camera/bench/ does not exist, so no c3_bench.py CSV backs this. The one recording with EXACTLY this config is c3_camera/recordings/20260729_162216 (meta.json: isp_scale [1,2] -> 960x540, mjpeg q90, depth_out [640,360], fps 15.0). I reco… |
| 131 | **b** | hig | @20 행은 인코더 표에서 19.1 fps / drop 5% | No bench CSV (c3_camera/bench/ absent). The nearest artifact, c3_camera/recordings/20260803_145852 (meta.json: 960x540 mjpeg q90 + depth 640x360 at fps 20.0), recomputes from frames.csv to 19.98 colour fps / 20.00 depth fps with ZERO seque… |
| 499 | **b** | hig | 실측(dataset_20260803_105221)에서 컬러 MJPEG는 6.8 Mbit/s로 링크의 **14%뿐**이고 depth uint16이 41.5 Mbit/s로 **86%**를 먹는다 | I opened the named dataset. metadata.txt: colour [480,270], depth [480,270], fps_achieved 20.01, 2336 frames, mjpeg. Depth checks out: 480*270*2*8*20.0/1e6 = 41.49 Mbit/s -> 41.5 CONFIRMED. Colour does NOT: mean of the 2336 rgb/*.jpg is 37… |
| 633 | **b** | hig | PSNR **45.60 dB** / SSIM **0.9962** / ORB inlier **0.9156**이 나온다 | Prefaced by '이 호스트에서 직접 측정' (measured directly on this host). c3_camera/encode_quality/ does not exist — c3_encode_quality.py has never been run. The number lives only as a hard-coded string in c3_encode_quality.py:121 (CAVEATS) and in tes… |
| 107 | **b** | med | 1080p에서 실측 대비 약 1.4배 과대평가한다 (실측 8.25:1) | '8.25:1' appears nowhere in the repo except this line (grepped all .py/.md/.json). It is exactly 3110.4 kB raw / 377 kB, i.e. back-computed from README.md:296's '1920x1080 mjpeg @12, 377 kB/f' row — and that README row has no artifact eith… |
| 282 | **b** | med | 같은 설정 6세션에서 4.05:1 ~ 6.35:1로 흔들렸다 | c3_encode_quality.py:144 says 'measured compression on this camera ranged 4.05:1 to 6.35:1 across identical sessions' — with no session count and no artifact (encode_quality/ absent). The '6세션' figure appears only in TESTING.md. Cross-chec… |
| 501 | **b** | med | 컬러를 0바이트로 만드는 완벽한 코덱도 천장의 7.4%만 회수하고, 전체가 이미 53%라 | Both are arithmetic on the unreproducible 6.8: 6.8/91.5 = 7.43% and (6.8+41.5)/91.5 = 52.8%. Recomputed from the dataset itself: colour 5.98 + depth 41.49 = 47.47 Mbit/s -> 6.5% of ceiling recoverable and 51.9% total occupancy. |
| 596 | **b** | med | 압축 아티팩트가 위양성 검출을 만들어 검출 수가 **늘어난다** (실측: 11 → 13) | '실측' (measured). Source is a hard-coded docstring in c3_encode_quality.py:468-470 ('measured on this camera's frames, a compressed copy detected 13 tags where the reference detected 11'). c3_camera/encode_quality/ does not exist, no AprilT… |
| 543 | **b** | low | raw는 960x540@20에서 링크 천장의 138%라 ~9 fps로 떨어지는데 | The 138% is verified (bucket a): the script's own --dry-run prints 'none (raw NV12) est 124.4 Mbit/s (138% of ceiling)'. The '~9 fps' is a measurement claim traceable to README.md:258 / config.py:192 ('raw NV12 delivered 9.1 of 20 requeste… |
| 636 | **b** | low | MJPEG은 같은 실험에서 true loss와 0.1 dB 이내로 일치해 이 패널티가 없다 | Same non-existent experiment as the 45.60 dB row. Source is the CAVEATS string c3_encode_quality.py:127-128 ('as-scored matched its true loss to ~0.1 dB'), which immediately hedges that it 'assumed the encoder converts limited-range NV12 t… |
| 444 | **c** | med | 2 m/400p에서 dZ = 141 mm — 피팅하려는 원점 offset(~25 mm)보다 크다 | dZ = 141 mm is verified (c3_depth_accuracy.py --dry-run --truth-mm 2000 prints 'one disparity step = 140.9 mm'). The '~25 mm' datum offset is not: c3_depth_accuracy.py:43-45 states the offset between the datum face and the optical centre i… |
| 472 | **c** | med | 실제로는 ~37.5 mm 떨어져 있어서 | Asserted as the real CAM_A-to-stereo-reference translation. It is not measured anywhere: calibration.json in every dataset carries only device.baseline_cm = 7.5 and per-camera intrinsics — there are NO CAM_A<->CAM_B/C extrinsics on this de… |
| 473 | **c** | med | 185 mm에서 그 병진은 104 px(폭의 16.3%)로 예측된 밴드 94 px보다 **크다** | The 94 px IS verified — c3_depth_accuracy.py --dry-run --truth-mm 185 --extended prints 'in the delivered 640px frame that band is 94 px = 14.7%'. The 104 px is not measured: it is fx_colour(at 640 wide) * 37.5 / 185 = (3080.35*640/3840)*3… |
| 477 | **c** | med | 1 m 실측 예: resid 10.01, temporal 5.19, fixed 8.56 | Labelled '실측 예' (measured example) but the very next clause admits it was synthesised ('진짜 센서 노이즈 1 mm에 ripple 0으로 합성했는데도'). No artifact: c3_camera/depth_accuracy/rungs.csv does not exist and c3_depth_accuracy.py has never been run on hard… |
| 614 | **c** | med | 임계값 0.7은 미보정이라 정적 리그에서도 센서 노이즈 12 DN이면 0.682까지 떨어져 false alarm이 난다 | The 0.7 floor is real (c3_encode_quality.py:1068, static_scene_warning(floor=0.7)) -> that part is bucket a. '0.682' appears nowhere in the repo. The source docstring (c3_encode_quality.py:1078-1080) says 'measured on a perfectly static SY… |
| 655 | **c** | med | c3_collect의 h26x 데이터셋 캡처는 여전히 0 바이트를 쓴다 | Stated as current verified behaviour and used as a hard gate at line 681. I read source.py: the getFrameType() call is at line 480 (not 429 as cited), and it is already wrapped in try/except with an Annex-B fallback at lines 485-490 ('dept… |
| 273 | **c** | low | **`--mjpeg-quality`가 범위 검증되지 않는다.** 150이나 -5도 통과해서 디바이스 시작 시점에서야 실패한다 = run 하나를 태운다 | False as of current source. config.py:341-343 raises ValueError('--mjpeg-quality N is outside 0-100'). I executed it: StreamConfig(mjpeg_quality=150).resolve() -> REJECTED, mjpeg_quality=-5 -> REJECTED, 90 -> accepted. config.py's own comm… |
| 480 | **c** | low | 또 ~0.9 mm 사각지대가 있어 진짜 고정 패턴이 정확히 0으로 읽힐 수 있다 | '0.9 mm' appears nowhere in c3_depth_accuracy.py or tests/test_depth_accuracy.py (grepped). It is a property of the quadrature subtraction fixed=sqrt(max(0, resid^2-temporal^2)) at c3_depth_accuracy.py:616-618 that would have to come from … |
| 556 | **c** | low | Step 3 (선택) — 1080p 팔 (4 설정, ~2분) | I ran the script's own planner: `c3_encode_quality.py --dry-run --codec none,mjpeg,h265 --isp-scale none --mjpeg-quality 90 --bitrate-kbps 8000,16000 --keyframe 1 --streams color --fps 20 --target-fps 19 --duration 20 --reference-frames 15… |
| 571 | **c** | low | 1080p raw는 링크에서 ~3.8 fps라 기본 `--duration 10`이면 프레임이 ~38장뿐이고 | Not measured — it is 91.5 / (1920*1080*1.5*8/1e6) = 3.68, rounded up. The script's own dry-run reports 1080p raw at 497.7 Mbit/s = 553% of the 90 Mbit/s ceiling, which yields 3.6 fps against POE_BUDGET_MBPS. The repo's only measured 1080p … |
| 695 | **c** | low | 컬러를 960x540 MJPEG q90으로 올리면 21.5 + 41.5 = 62.9 Mbit/s = 천장의 **69%** | 41.5 is verified from dataset_20260803_105221. 21.5 is a projection with no traceable source: config.py's own estimator gives 960*540*1.5/6.0*8*20/1e6 = 20.7 Mbit/s (and recordings/20260803_145852/meta.json resolved.bandwidth_mbps.color is… |
| 75 | **a** | hig | 링크 천장은 ~91.5 Mbit/s (`C.POE_BUDGET_MBPS = 90.0`) | `c3_camera/recordings/20260803_150048/frames.csv` |
| 18 | **a** | med | 400p면 MinZ 298.9 mm, 720p/800p면 597.7 mm | `c3_camera/datasets/dataset_20260729_173152/calibration.json` |
| 68 | **a** | med | test_offline.py # 47/47 ... test_depth_accuracy.py # 58/58 ... test_encode_quality.py # 36/36 ... test_host_depth.py # 45/45 ... test_option_sweep.py… | `c3_camera/tests/test_offline.py` |
| 112 | **a** | med | isp 1/2 (960x540) + depth 480x270 @15 = 52% ... isp none (1920x1080) + depth 576x324 @15 = 119% | `c3_camera/config.py` |
| 86 | **a** | low | 13 combo x 32 s = **~7분**. combo당 32초 = `--duration 10 + --warmup 3 + --settle 5 + 14`(부팅 12초 + PoE 리셋) | `c3_camera/c3_bench.py` |
| 161 | **a** | low | `2/3` = 1280x720 (한 번도 측정된 적 없음), `none` = 1920x1080 @15 (README에는 @12와 @20만 있다). `1/3`(640x360)과 `1/2`(960x540)은 README의 대역폭 표에 이미 측정되어 있으니 | `c3_camera/README.md` |
| 250 | **a** | low | 프레임이 애초에 안 만들어졌다 (4k는 42 fps, 12mp는 30 fps가 상한) | `c3_camera/recordings/20260729_162216/meta.json` |
| 268 | **a** | low | h264/h265는 디바이스+호스트 blocking 큐 1.5초 깊이, none/mjpeg는 non-blocking depth-1 | `c3_camera/pipeline.py` |
| 275 | **a** | low | **`--append`는 컬럼 셋이 다르면 거부된다** (rc=2, 기존 파일 무수정) | `c3_camera/c3_bench.py` |
| 316 | **a** | low | ROI(중앙 40%)는 `0.499*Z x 0.280*Z`, full frame은 `1.247*Z x 0.701*Z`를 덮는다. 2 m rung은 ROI만 997x561 mm, full frame은 2494x1402 mm | `c3_camera/datasets/dataset_20260729_173152/calibration.json` |
| 371 | **a** | low | `robotics` preset의 confidence 245는 255 중 245라 거의 완전 관대해서 | `c3_camera/c3_depth_accuracy.py` |
| 379 | **a** | low | 400p는 2 m에서 disparity 스텝이 141 mm라 quantisation-limited 경고가 뜬다. 800p는 그 절반(70.4 mm). --fps 2 인 이유: 800p depth는 1280x720이라 5 fps면 133% OVER BUDGET | `c3_camera/c3_depth_accuracy.py` |
| 396 | **a** | low | `--depth-align color`는 stereo 화각을 H 80.4° → 63.9°로 잘라내서, 근거리 overlap 손실 밴드가 **338 mm 위에서는 프레임 밖으로 완전히 잘려나간다** | `c3_camera/c3_depth_accuracy.py` |
| 448 | **a** | low | `--fit-min-mm 450` 덕에 fit에는 안 들어가지만 | `c3_camera/c3_depth_accuracy.py` |
| 482 | **a** | low | 구멍 30%/20프레임이면 ROI의 0.08%만 남고 40%면 아예 빈 셀이 된다 | Pure arithmetic from stated assumptions: 0.7^20 = 0.000798 = 0.0798% -> 0.08%; 0.6^20 = 0.0036%, effectively empty. Con… |
| 484 | **a** | low | MinZ가 298.9인지 300.1인지 미해결. CAM_B fx는 757.12, CAM_C는 760.28(0.42% 차이) | `c3_camera/datasets/dataset_20260729_173152/calibration.json` |
| 510 | **a** | low | reference의 ORB keypoint가 500 미만이면 스크립트가 전체 캡처를 `under_textured`로 플래그하고 | `c3_camera/c3_encode_quality.py` |
| 524 | **a** | low | 출력이 20 설정 / ~10.7분이라고 하면 계획대로다 | `c3_camera/c3_encode_quality.py` |
| 538 | **a** | low | 640x360 uint16 depth는 ~74 Mbit/s라 2-30 Mbit/s짜리 컬러 사다리를 통째로 삼켜서 | `c3_camera/recordings/20260803_145852/meta.json` |
| 645 | **a** | low | **H.264는 BASELINE으로 고정되어 있다** (`pipeline.py:179`) | `c3_camera/pipeline.py` |
| 672 | **a** | low | `StreamConfig.depth_size` \| `None` ... `isp_scale` \| `(1, 2)` → 960x540 ... `mono_res` \| `"400p"` ... `extended` \| `False` ... `median` \| `"5x5"… | `c3_camera/config.py` |
| 682 | **a** | low | `c3_collect.py:223` `--min-mm` 기본값 \| `300.0` | `c3_camera/c3_collect.py` |

### [claude.md](../claude.md)  — a 3 / b 6 / c 0

| 줄 | 판정 | sev | 주장 | 산출물 / 사유 |
|---:|:---:|:---:|---|---|
| 22 | **b** | hig | M6 real data: D_true ≈ 1.2 m → none ≈ 1.0 m (R ≈ 0.83), refractive ≈ 1.2 m (R ≈ 1.0). PASS. | The 'none' half is supported by ruler_test.csv at the repo root (300 rows at tag_size_m 0.17; median raw solvePnP distance 0.9639 m ≈ 1.0 m). The refractive half is not: no run output in data/ records the water-correction mode, there are n… |
| 6 | **b** | med | Pool ≈ 4.877 m × 2.438 m × 1.143 m (45 in) deep | config/config.yaml pool block: length_m 4.877 ✓, depth_m 1.143 ✓ (= 45 in exactly), but width_m: 1.8 — NOT 2.438. The doc's width contradicts the only artifact in the repo that records pool geometry. 4.877 × 2.438 m is exactly 16 × 8 ft, w… |
| 17 | **b** | med | Intrinsics are fine (ZED factory fx=fy≈534.88 HD720; no hidden extra scale, else R would deviate from physics) | No stored intrinsics anywhere. src/zed2_underwater_tagslam.py:805 reads fx live from the ZED SDK calibration (float(left.fx)); data/*/metadata.json carries only product_name/resolution/depth_mode etc. with no calibration block (checked dat… |
| 21 | **b** | med | M5 synthetic self-test passes (--refractive-self-test): rms ≈ 1e-5 px, trans ≈ 0 mm, rot ≈ 2e-4 deg. | run_refractive_self_test() exists (src/zed2_underwater_tagslam.py:830) and is wired to --refractive-self-test (line 1121), but it only prints. There are no .log files anywhere in the repo (find -name '*.log' excluding .git/external returns… |
| 24 | **b** | med | Synthetic benchmark ≈ 41× speedup. | run_refractive_benchmark() exists (src/zed2_underwater_tagslam.py:1003) and prints 'speedup: {…}x' at line 1108 over a synthetic 20-tag frame, but no output is persisted (no .log files in the repo; grep for 'speedup' across md/csv/json/txt… |
| 26 | **b** | med | --refractive-regression-check: max_trans_delta ≈ 0.0003 mm, max_rot_delta ≈ 0.00037 deg (bounds 0.1 mm / 0.01 deg). PASS. | The BOUNDS are verified in source — src/zed2_underwater_tagslam.py:997 reads `if max_trans_error_m > 1e-4 or max_rot_error_deg > 0.01:` i.e. 0.1 mm / 0.01 deg exactly as quoted. The ACHIEVED values (0.0003 mm, 0.00037 deg) are console outp… |
| 31 | **a** | hig | Tag Z span did NOT shrink (refractive even showed ~64.9 cm vs scalar ~49.9 cm) | `data/20260518/20260518_133748_tagslam_trajectory/tag_poses.csv` |
| 13 | **a** | low | Real printed AprilTag black-square edge = 0.170 m (ruler-measured). | `config/config.yaml` |
| 17 | **a** | low | Measured (correct tag size 0.170, mode none): R ≈ 0.80 at d_water ≈ 1.0 m vs predicted ≈ 0.79. | `ruler_test.csv` |

## 표기 작업 목록 (b / c → [UNVERIFIED])

각 행의 수치는 **지우지 않는다.** 그 줄에 `[UNVERIFIED]`와 한 구절짜리 사유를 붙인다.
대상 284건. 아래는 현재 파일에서 읽어온 실제 줄 내용이므로 그대로 편집에 쓸 수 있다.

- `.claude/agents/hardware-advisor.md:54` · `b` · 산출물 없음
  - 현재: `- Fathom-X is rated **~80 Mbps over two wires** (vendor's own testing), but **real-world throughput is commonly 15–50 Mbps**, often ~15–20 Mbps effective once video is flowing.`
- `.claude/agents/hardware-advisor.md:56` · `b` · 산출물 없음
  - 현재: `- **Direct gigabit over the stock tether basically does not work**, even at short range. Field data points: a near-identical RGB-D project got only **10 Mbps at 15 m** using cobalt connectors; a GigaBlox + 50 m Fathom t…`
- `.claude/agents/hardware-advisor.md:104` · `c` · 예시/유도값이 측정치로 서술됨
  - 현재: `\| 2× mono stereo, H.264 (8-bit, compresses well) \| ~6–10 Mbps \| ✅ \|`
- `.claude/agents/hardware-advisor.md:124` · `c` · 예시/유도값이 측정치로 서술됨
  - 현재: `- **Tuned low-latency pipeline ≈ 30–100 ms.** Default/buffered settings can balloon to 150–250 ms+.`
- `.claude/agents/hardware-advisor.md:144` · `c` · 예시/유도값이 측정치로 서술됨
  - 현재: `- **Desktop compute:** camera → tether → desktop (decode + infer) → command → tether → thrusters. The control loop **crosses the tether twice** (~60–150 ms). Easy to develop/debug, big GPU. Fine for prototyping and for …`
- `.claude/agents/hardware-advisor.md:258` · `c` · 예시/유도값이 측정치로 서술됨
  - 현재: `- **Best bandwidth trick:** send **compressed stereo + color**, compute depth at the compute node → full-res depth+color in ~15 Mbps. Compression is **on the camera**, not the Pi.`
- `.claude/journal/consults.md:7` · `c` · 예시/유도값이 측정치로 서술됨
  - 현재: `- 2026-06-22 [hardware-advisor] Q: C3를 BlueROV2에 추가 시 2차 하드웨어 영향/BOM 주의점? → C3는 gigabit이지만 tether는 Fathom-X ~80Mbps(실측 15–50) 병목 → tether 가로질러 gigabit 불필요; 압축 stereo+color 전송 후 depth는 topside 재구성. onboard gigabit non-Po…`
- `.claude/journal/consults.md:12` · `b` · 산출물 없음
  - 현재: `- 2026-07-01 [simulation-advisor + slam-perception-advisor + verifier] Q: 실제 pool의 tag36h11 바닥을 MuJoCo sim에 재현 가능? → POOL_TAGS=1 opt-in wrapper(scene_bluerov[_heavy]_tags.xml)가 seabed + tag36h11 격자(실측47 + 격자fill, ~154) …`
- `.claude/journal/consults.md:12` · `b` · 산출물 없음
  - 현재: `- 2026-07-01 [simulation-advisor + slam-perception-advisor + verifier] Q: 실제 pool의 tag36h11 바닥을 MuJoCo sim에 재현 가능? → POOL_TAGS=1 opt-in wrapper(scene_bluerov[_heavy]_tags.xml)가 seabed + tag36h11 격자(실측47 + 격자fill, ~154) …`
- `.claude/journal/consults.md:12` · `b` · 산출물 없음
  - 현재: `- 2026-07-01 [simulation-advisor + slam-perception-advisor + verifier] Q: 실제 pool의 tag36h11 바닥을 MuJoCo sim에 재현 가능? → POOL_TAGS=1 opt-in wrapper(scene_bluerov[_heavy]_tags.xml)가 seabed + tag36h11 격자(실측47 + 격자fill, ~154) …`
- `.claude/journal/consults.md:14` · `c` · 예시/유도값이 측정치로 서술됨
  - 현재: `- 2026-07-02 [workflow: simulation-advisor→control-theory-advisor→verifier×2+repair] Q: PID Kp/Ki/Kd를 해석적으로 유도 가능? → 가능. hover 선형화 m_eff·a=F−d·v + 3차 극배치: Kp=m(1+2ζα)ωn², Kd=m(2ζ+α)ωn−d, Ki=mαωn³; 설계점 ωn=2(병진)/3(yaw), ζ…`
- `.claude/journal/consults.md:14` · `c` · 예시/유도값이 측정치로 서술됨
  - 현재: `- 2026-07-02 [workflow: simulation-advisor→control-theory-advisor→verifier×2+repair] Q: PID Kp/Ki/Kd를 해석적으로 유도 가능? → 가능. hover 선형화 m_eff·a=F−d·v + 3차 극배치: Kp=m(1+2ζα)ωn², Kd=m(2ζ+α)ωn−d, Ki=mαωn³; 설계점 ωn=2(병진)/3(yaw), ζ…`
- `.claude/journal/consults.md:15` · `c` · 예시/유도값이 측정치로 서술됨
  - 현재: `- 2026-07-02 [workflow: control-theory-advisor + simulation-advisor] Q: 외란 모르는 nominal MPC가 왜 PID를 이기나 (toy: x*=goal+d offset 맞나?) → toy 정확 (offset=d/kappa, R↑일수록 악화; R=0 MPC ≡ kp=1 P제어기 + constraints), 실데이터에 실재: C모드 MP…`
- `.claude/journal/consults.md:19` · `b` · 산출물 없음
  - 현재: `- 2026-07-07 [workflow: diagnose-dobmpc-corner-deviation (4 investigators + adjudicator)] Q: 왜 DOB-MPC가 CW/CDW에서 (1,1) 우상단 코너에서만 크게 이탈하나 → 세 요인의 곱: (M1) sharp-square 기준 + 60°/s yaw 슬루의 코너 기하 과도현상(무외란서도 전코너 ~5cm, run_com…`
- `.claude/journal/consults.md:19` · `b` · 산출물 없음
  - 현재: `- 2026-07-07 [workflow: diagnose-dobmpc-corner-deviation (4 investigators + adjudicator)] Q: 왜 DOB-MPC가 CW/CDW에서 (1,1) 우상단 코너에서만 크게 이탈하나 → 세 요인의 곱: (M1) sharp-square 기준 + 60°/s yaw 슬루의 코너 기하 과도현상(무외란서도 전코너 ~5cm, run_com…`
- `.claude/journal/consults.md:21` · `b` · 산출물 없음
  - 현재: `- 2026-07-21 [workflow: control-theory + Explore + verifier (5 agents, 2 CONFIRMED)] Q: heavy 기준 6-DOF 독립제어 맞나 + rank-5 surge 정형(slew/pitch-guard/clamp) 제거 + roll/pitch PD→PID → (Q1) heavy 계열은 rank-6 fully-actuated, doc…`
- `.claude/journal/consults.md:21` · `b` · 산출물 없음
  - 현재: `- 2026-07-21 [workflow: control-theory + Explore + verifier (5 agents, 2 CONFIRMED)] Q: heavy 기준 6-DOF 독립제어 맞나 + rank-5 surge 정형(slew/pitch-guard/clamp) 제거 + roll/pitch PD→PID → (Q1) heavy 계열은 rank-6 fully-actuated, doc…`
- `.claude/journal/consults.md:22` · `b` · 산출물 없음
  - 현재: `- 2026-07-21 [실행: bluerov2 변종 제거 + pitch-guard 정정] 사용자 승인(테스트는 heavy 이관, dobmpc는 보류)으로 실행: rov_model/_MODELS·_POOL_WRAP, controller GAINS_BLUEROV2·DEFAULT_GAINS 폴백, 3개 루트 스모크 테스트 heavy 이관, verify_meta 허용목록, gen_pool 래퍼,…`
- `.claude/journal/consults.md:23` · `c` · 예시/유도값이 측정치로 서술됨
  - 현재: `- 2026-07-22 [workflow: control-theory-advisor ×2 + simulation-advisor + verifier (4 agents, 11/12 verified)] Q: mpc_acados w_hat ±50 클립 적정성 + yref u=0 적절성 → 클립: authority 초과 시 u*는 클립 수준과 무관(active set 동일)이라 "큰 외란이 지워진다…`
- `.claude/journal/consults.md:23` · `c` · 예시/유도값이 측정치로 서술됨
  - 현재: `- 2026-07-22 [workflow: control-theory-advisor ×2 + simulation-advisor + verifier (4 agents, 11/12 verified)] Q: mpc_acados w_hat ±50 클립 적정성 + yref u=0 적절성 → 클립: authority 초과 시 u*는 클립 수준과 무관(active set 동일)이라 "큰 외란이 지워진다…`
- `.claude/journal/consults.md:23` · `c` · 예시/유도값이 측정치로 서술됨
  - 현재: `- 2026-07-22 [workflow: control-theory-advisor ×2 + simulation-advisor + verifier (4 agents, 11/12 verified)] Q: mpc_acados w_hat ±50 클립 적정성 + yref u=0 적절성 → 클립: authority 초과 시 u*는 클립 수준과 무관(active set 동일)이라 "큰 외란이 지워진다…`
- `.claude/journal/consults.md:23` · `c` · 예시/유도값이 측정치로 서술됨
  - 현재: `- 2026-07-22 [workflow: control-theory-advisor ×2 + simulation-advisor + verifier (4 agents, 11/12 verified)] Q: mpc_acados w_hat ±50 클립 적정성 + yref u=0 적절성 → 클립: authority 초과 시 u*는 클립 수준과 무관(active set 동일)이라 "큰 외란이 지워진다…`
- `.claude/journal/consults.md:27` · `b` · 산출물 없음
  - 현재: `- 2026-07-24 (4) [직접 측정] Q: trajectory_compare에서 DOB-MPC가 코너에서만 참조와 크게 벌어지는 이유 — 그것도 8 N surge 박스 탓인가? → **아니오**. NONE(외란 0) 박스 A/B에서 코너 오차가 2.00/1.92/2.04 cm로 8 N·30 N **소수점까지 동일**(코너 surge 명령 평균 0.9–1.4 N, 박스 발동 0.7%)…`
- `.claude/journal/consults.md:27` · `c` · 예시/유도값이 측정치로 서술됨
  - 현재: `- 2026-07-24 (4) [직접 측정] Q: trajectory_compare에서 DOB-MPC가 코너에서만 참조와 크게 벌어지는 이유 — 그것도 8 N surge 박스 탓인가? → **아니오**. NONE(외란 0) 박스 A/B에서 코너 오차가 2.00/1.92/2.04 cm로 8 N·30 N **소수점까지 동일**(코너 surge 명령 평균 0.9–1.4 N, 박스 발동 0.7%)…`
- `.claude/journal/consults.md:27` · `b` · 산출물 없음
  - 현재: `- 2026-07-24 (4) [직접 측정] Q: trajectory_compare에서 DOB-MPC가 코너에서만 참조와 크게 벌어지는 이유 — 그것도 8 N surge 박스 탓인가? → **아니오**. NONE(외란 0) 박스 A/B에서 코너 오차가 2.00/1.92/2.04 cm로 8 N·30 N **소수점까지 동일**(코너 surge 명령 평균 0.9–1.4 N, 박스 발동 0.7%)…`
- `.claude/journal/consults.md:29` · `b` · 산출물 없음
  - 현재: `- 2026-07-24 (3) [구현+검증] Q: MPC도 PID와 같은 조건(surge 30 N)으로 맞춰달라 → heavy `U_MAX[0]` 8→30 적용(rank-5 분기는 8 유지, 커플링 실재), acados 재빌드 후 컴파일된 솔버 bound 확인. 재빌드 검증(storm/gentle CW, 10lap×3헤딩): storm PID 26.14 / MPC 28.82→**17.86*…`
- `.claude/journal/consults.md:29` · `b` · 산출물 없음
  - 현재: `- 2026-07-24 (3) [구현+검증] Q: MPC도 PID와 같은 조건(surge 30 N)으로 맞춰달라 → heavy `U_MAX[0]` 8→30 적용(rank-5 분기는 8 유지, 커플링 실재), acados 재빌드 후 컴파일된 솔버 bound 확인. 재빌드 검증(storm/gentle CW, 10lap×3헤딩): storm PID 26.14 / MPC 28.82→**17.86*…`
- `.claude/journal/consults.md:29` · `b` · 산출물 없음
  - 현재: `- 2026-07-24 (3) [구현+검증] Q: MPC도 PID와 같은 조건(surge 30 N)으로 맞춰달라 → heavy `U_MAX[0]` 8→30 적용(rank-5 분기는 8 유지, 커플링 실재), acados 재빌드 후 컴파일된 솔버 bound 확인. 재빌드 검증(storm/gentle CW, 10lap×3헤딩): storm PID 26.14 / MPC 28.82→**17.86*…`
- `.claude/journal/consults.md:29` · `b` · 산출물 없음
  - 현재: `- 2026-07-24 (3) [구현+검증] Q: MPC도 PID와 같은 조건(surge 30 N)으로 맞춰달라 → heavy `U_MAX[0]` 8→30 적용(rank-5 분기는 8 유지, 커플링 실재), acados 재빌드 후 컴파일된 솔버 bound 확인. 재빌드 검증(storm/gentle CW, 10lap×3헤딩): storm PID 26.14 / MPC 28.82→**17.86*…`
- `.claude/journal/consults.md:31` · `b` · 산출물 없음
  - 현재: `- 2026-07-24 (2) [workflow: 4 data/code analysts + control-theory synthesis + 4 adversarial verifiers(4/4 REFUTED=정밀도·귀속 기각, 정성핵심 존치) + 직접 ablation] Q: 대칭-state sweep(compare_20260724_160210)에서 강파랑=PID 승, 약파랑=MPC 승의 원인 …`
- `.claude/journal/consults.md:31` · `b` · 산출물 없음
  - 현재: `- 2026-07-24 (2) [workflow: 4 data/code analysts + control-theory synthesis + 4 adversarial verifiers(4/4 REFUTED=정밀도·귀속 기각, 정성핵심 존치) + 직접 ablation] Q: 대칭-state sweep(compare_20260724_160210)에서 강파랑=PID 승, 약파랑=MPC 승의 원인 …`
- `.claude/journal/consults.md:31` · `b` · 산출물 없음
  - 현재: `- 2026-07-24 (2) [workflow: 4 data/code analysts + control-theory synthesis + 4 adversarial verifiers(4/4 REFUTED=정밀도·귀속 기각, 정성핵심 존치) + 직접 ablation] Q: 대칭-state sweep(compare_20260724_160210)에서 강파랑=PID 승, 약파랑=MPC 승의 원인 …`
- `.claude/journal/consults.md:31` · `b` · 산출물 없음
  - 현재: `- 2026-07-24 (2) [workflow: 4 data/code analysts + control-theory synthesis + 4 adversarial verifiers(4/4 REFUTED=정밀도·귀속 기각, 정성핵심 존치) + 직접 ablation] Q: 대칭-state sweep(compare_20260724_160210)에서 강파랑=PID 승, 약파랑=MPC 승의 원인 …`
- `.claude/journal/consults.md:31` · `b` · 산출물 없음
  - 현재: `- 2026-07-24 (2) [workflow: 4 data/code analysts + control-theory synthesis + 4 adversarial verifiers(4/4 REFUTED=정밀도·귀속 기각, 정성핵심 존치) + 직접 ablation] Q: 대칭-state sweep(compare_20260724_160210)에서 강파랑=PID 승, 약파랑=MPC 승의 원인 …`
- `.claude/journal/consults.md:31` · `b` · 산출물 없음
  - 현재: `- 2026-07-24 (2) [workflow: 4 data/code analysts + control-theory synthesis + 4 adversarial verifiers(4/4 REFUTED=정밀도·귀속 기각, 정성핵심 존치) + 직접 ablation] Q: 대칭-state sweep(compare_20260724_160210)에서 강파랑=PID 승, 약파랑=MPC 승의 원인 …`
- `.claude/journal/consults.md:33` · `c` · 예시/유도값이 측정치로 서술됨
  - 현재: `- 2026-07-24 [workflow: 3 data/code analysts + control-theory synthesis + 8 verifiers(4 CONFIRMED, 4 세션한도 사망→핵심 수치 직접 재확인)] Q: 20260724 wave sweep에서 MPC가 PID에 지는 이유(전반·wave 강도별 역전·NONE/C/CD state-노이즈 가설) → NONE은 MPC 승(1…`
- `.claude/journal/consults.md:33` · `c` · 예시/유도값이 측정치로 서술됨
  - 현재: `- 2026-07-24 [workflow: 3 data/code analysts + control-theory synthesis + 8 verifiers(4 CONFIRMED, 4 세션한도 사망→핵심 수치 직접 재확인)] Q: 20260724 wave sweep에서 MPC가 PID에 지는 이유(전반·wave 강도별 역전·NONE/C/CD state-노이즈 가설) → NONE은 MPC 승(1…`
- `.claude/journal/research.md:11` · `c` · 예시/유도값이 측정치로 서술됨
  - 현재: `- **진짜 해결 = 드라이버를 CUDA-13(580/595)에서 CUDA-12.8 브랜치 R570(570.211.01-open)으로** (IsaacLab #3448: PhysX-on-Blackwell엔 R570 권장). Isaac Sim5.0/torch cu128/warp 전부 CUDA12.8 빌드라 드라이버도 12.8이어야 함. 결과: 2048 envs·400 iter·~19.7M st…`
- `.claude/journal/research.md:19` · `b` · 산출물 없음
  - 현재: `- **실측 재확인:** `/proc/cmdline`에 iommu off 플래그 없음(아직 미적용). 단독 `warp 1.8.1 wp.init()` 로컬 **exit 0/~수초**(sm_120, mempool, Toolkit12.8/Driver13.0) → warp 자체 hang 아님. extscache `omni.warp.core-1.7.1+lx64/warp/config.py` versi…`
- `.claude/journal/research.md:98` · `b` · 산출물 없음
  - 현재: `- 2026-06-24 [experiment-diagnostic-analyst] Q: 여러 시드 학습(seed 42/1/2/3/4) 성능 평가 → 5개 run이 TensorBoard reward 곡선상 비트 단위 동일(maxdiff=0.00). 시드 무효 실측 확정(warpauv_env.py:155 torch.manual_seed(0)이 원인). 성능을 가른 건 iteration뿐: rew…`
- `.claude/journal/research.md:124` · `b` · 산출물 없음
  - 현재: `- 2026-07-21 [workflow: 5 investigators + 4 adversarial verifiers] Q: compare_20260720_221845(새 GAINS_HEAVY 첫 full compare, 10쌍×5모드) 5질문 분석 → calm 격차=MPC 코너 과도(edge 동등 0.20 vs 0.14 cm; effort-페널티 가설 반박), C-offset 2.73→D…`
- `.claude/journal/research.md:127` · `b` · 산출물 없음
  - 현재: `- 2026-07-29 [workflow: 4 researchers + 24 adversarial verifiers, + 실측 벤치] Q: C3(MarineSitu)를 Madrona/BlueOS RTSP 우회하고 데스크톱에서 DepthAI로 직결해 저지연 RGB-D를 받을 수 있나 → 가능, **컬러 지연 3000 ms → 38 ms(p50), 15 요청→15.00 실측, 드롭 0%**. …`
- `.claude/journal/research.md:127` · `b` · 산출물 없음
  - 현재: `- 2026-07-29 [workflow: 4 researchers + 24 adversarial verifiers, + 실측 벤치] Q: C3(MarineSitu)를 Madrona/BlueOS RTSP 우회하고 데스크톱에서 DepthAI로 직결해 저지연 RGB-D를 받을 수 있나 → 가능, **컬러 지연 3000 ms → 38 ms(p50), 15 요청→15.00 실측, 드롭 0%**. …`
- `.claude/journal/research.md:127` · `b` · 산출물 없음
  - 현재: `- 2026-07-29 [workflow: 4 researchers + 24 adversarial verifiers, + 실측 벤치] Q: C3(MarineSitu)를 Madrona/BlueOS RTSP 우회하고 데스크톱에서 DepthAI로 직결해 저지연 RGB-D를 받을 수 있나 → 가능, **컬러 지연 3000 ms → 38 ms(p50), 15 요청→15.00 실측, 드롭 0%**. …`
- `.claude/journal/research.md:127` · `b` · 산출물 없음
  - 현재: `- 2026-07-29 [workflow: 4 researchers + 24 adversarial verifiers, + 실측 벤치] Q: C3(MarineSitu)를 Madrona/BlueOS RTSP 우회하고 데스크톱에서 DepthAI로 직결해 저지연 RGB-D를 받을 수 있나 → 가능, **컬러 지연 3000 ms → 38 ms(p50), 15 요청→15.00 실측, 드롭 0%**. …`
- `.claude/journal/research.md:128` · `b` · 산출물 없음
  - 현재: `- 2026-07-29(2) [workflow: 4 researchers + verifiers (verify 중) + 하드웨어 실측] Q: C3 직결 + ArduSub MAVLink로 SLAM용 RGB-D 데이터셋 수집기를 만들 수 있나 → 완성·실측 검증(`c3_collect.py`, TUM RGB-D). **C3에 온보드 IMU 있음 = BNO086 fw 3.9.9**(이전 "IMU 미…`
- `.claude/journal/research.md:128` · `b` · 산출물 없음
  - 현재: `- 2026-07-29(2) [workflow: 4 researchers + verifiers (verify 중) + 하드웨어 실측] Q: C3 직결 + ArduSub MAVLink로 SLAM용 RGB-D 데이터셋 수집기를 만들 수 있나 → 완성·실측 검증(`c3_collect.py`, TUM RGB-D). **C3에 온보드 IMU 있음 = BNO086 fw 3.9.9**(이전 "IMU 미…`
- `.claude/journal/research.md:128` · `c` · 예시/유도값이 측정치로 서술됨
  - 현재: `- 2026-07-29(2) [workflow: 4 researchers + verifiers (verify 중) + 하드웨어 실측] Q: C3 직결 + ArduSub MAVLink로 SLAM용 RGB-D 데이터셋 수집기를 만들 수 있나 → 완성·실측 검증(`c3_collect.py`, TUM RGB-D). **C3에 온보드 IMU 있음 = BNO086 fw 3.9.9**(이전 "IMU 미…`
- `.claude/journal/research.md:129` · `c` · 예시/유도값이 측정치로 서술됨
  - 현재: `- 2026-08-03 [workflow: 3 designers + 3 builders + 3 verifiers (P3)] Q: C3의 세 가지 미결 질문(해상도x fps 프론티어 / depth 정확도·최소거리 / 인코더 화질)을 실측으로 답할 도구가 있나 → 세 도구 완비. (1) `c3_bench.py`에 **--depth-size를 진짜 스윕 축으로 승격**(+mjpeg-quality…`
- `.claude/journal/research.md:129` · `b` · 산출물 없음
  - 현재: `- 2026-08-03 [workflow: 3 designers + 3 builders + 3 verifiers (P3)] Q: C3의 세 가지 미결 질문(해상도x fps 프론티어 / depth 정확도·최소거리 / 인코더 화질)을 실측으로 답할 도구가 있나 → 세 도구 완비. (1) `c3_bench.py`에 **--depth-size를 진짜 스윕 축으로 승격**(+mjpeg-quality…`
- `.claude/journal/research.md:129` · `b` · 산출물 없음
  - 현재: `- 2026-08-03 [workflow: 3 designers + 3 builders + 3 verifiers (P3)] Q: C3의 세 가지 미결 질문(해상도x fps 프론티어 / depth 정확도·최소거리 / 인코더 화질)을 실측으로 답할 도구가 있나 → 세 도구 완비. (1) `c3_bench.py`에 **--depth-size를 진짜 스윕 축으로 승격**(+mjpeg-quality…`
- `.claude/journal/research.md:129` · `b` · 산출물 없음
  - 현재: `- 2026-08-03 [workflow: 3 designers + 3 builders + 3 verifiers (P3)] Q: C3의 세 가지 미결 질문(해상도x fps 프론티어 / depth 정확도·최소거리 / 인코더 화질)을 실측으로 답할 도구가 있나 → 세 도구 완비. (1) `c3_bench.py`에 **--depth-size를 진짜 스윕 축으로 승격**(+mjpeg-quality…`
- `.claude/journal/research.md:130` · `b` · 산출물 없음
  - 현재: `- 2026-08-03(2) [build/docs] Q: 위 세 테스트를 사용자가 처음부터 끝까지 따라 할 운용 문서가 필요 → `c3_camera/TESTING.md` 작성(한국어 산문 + 영어 명령). 순서 근거: T1이 가장 싸고(물리 셋업 0, 13 combo = 7분) 나머지의 운용점을 정하며, **T1의 mono_res 선택이 T2의 최소거리를 직접 구속**(400p MinZ 2…`
- `.claude/journal/research.md:131` · `c` · 예시/유도값이 측정치로 서술됨
  - 현재: `- 2026-08-03(3) [workflow: 7 deep-readers + 3 adversarial verifiers + 1 synth] Q: OAK-D W 핸드헬드 UMI 데이터 수집 파이프라인을 짜기 전, UMI_underwater + iPhUMI가 실제로 뭘 요구하나 → 3개 관문 질문 소스 확정. (1) **zarr 계약**: `zarr_adapter.py:42-58`이 강제하는…`
- `.claude/journal/research.md:131` · `c` · 예시/유도값이 측정치로 서술됨
  - 현재: `- 2026-08-03(3) [workflow: 7 deep-readers + 3 adversarial verifiers + 1 synth] Q: OAK-D W 핸드헬드 UMI 데이터 수집 파이프라인을 짜기 전, UMI_underwater + iPhUMI가 실제로 뭘 요구하나 → 3개 관문 질문 소스 확정. (1) **zarr 계약**: `zarr_adapter.py:42-58`이 강제하는…`
- `.claude/journal/reviews.md:13` · `c` · 예시/유도값이 측정치로 서술됨
  - 현재: `- **control 지적:** Heavy는 roll/pitch commanded(NU=6) → 관성이 "don't care"→"~10-20% 중요"로 승격. EAOB는 quasi-static 오차는 흡수하나 ω̇-상관(기동 중) 오차는 못 함.`
- `.claude/journal/reviews.md:21` · `b` · 산출물 없음
  - 현재: `- 2026-07-03 [workflow: review-pid-poleplacement-apply (transcription+regression × verifier)] Q: pole-placement PID 적용 diff(GAINS_HEAVY/GAINS_BLUEROV2 분기 + r_ref yaw-rate FF) 적대적 리뷰 → 전사·게인값·r_cmd 계산 무결; 실회귀 1건 확정(실행 재현…`
- `.claude/journal/reviews.md:23` · `b` · 산출물 없음
  - 현재: `- 2026-07-07 [workflow: review-run-compare-extension (4 dims × adversarial verify)] Q: run_compare 원샷 배치 확장 diff(pairing:grid 방향그리드 + per-run runs/ CSV·meta + trajectory_compare 전방향 오버레이 + MPC 파랑) 적대적 리뷰 → 16 findings, …`
- `.claude/journal/reviews.md:23` · `b` · 산출물 없음
  - 현재: `- 2026-07-07 [workflow: review-run-compare-extension (4 dims × adversarial verify)] Q: run_compare 원샷 배치 확장 diff(pairing:grid 방향그리드 + per-run runs/ CSV·meta + trajectory_compare 전방향 오버레이 + MPC 파랑) 적대적 리뷰 → 16 findings, …`
- `.claude/journal/reviews.md:27` · `b` · 산출물 없음
  - 현재: `- 2026-07-23 [workflow: eaob-review-fanout (R1/R4/R5 병렬 3 agents) + 수치검증 + 폐루프 A/B] Q: 외부 EAOB 리뷰(C0–C4) verify-first 검증 → 재튜닝 적용. R1–R6 전부 실코드 확정(노이즈 주입 전무·R=스칼라I·deadbeat 실측 6.8–13.4 ms·EAOB_* 소비처는 eaob.py 기본 kwargs뿐·…`
- `.claude/journal/reviews.md:27` · `b` · 산출물 없음
  - 현재: `- 2026-07-23 [workflow: eaob-review-fanout (R1/R4/R5 병렬 3 agents) + 수치검증 + 폐루프 A/B] Q: 외부 EAOB 리뷰(C0–C4) verify-first 검증 → 재튜닝 적용. R1–R6 전부 실코드 확정(노이즈 주입 전무·R=스칼라I·deadbeat 실측 6.8–13.4 ms·EAOB_* 소비처는 eaob.py 기본 kwargs뿐·…`
- `.claude/journal/reviews.md:27` · `b` · 산출물 없음
  - 현재: `- 2026-07-23 [workflow: eaob-review-fanout (R1/R4/R5 병렬 3 agents) + 수치검증 + 폐루프 A/B] Q: 외부 EAOB 리뷰(C0–C4) verify-first 검증 → 재튜닝 적용. R1–R6 전부 실코드 확정(노이즈 주입 전무·R=스칼라I·deadbeat 실측 6.8–13.4 ms·EAOB_* 소비처는 eaob.py 기본 kwargs뿐·…`
- `.claude/journal/reviews.md:27` · `b` · 산출물 없음
  - 현재: `- 2026-07-23 [workflow: eaob-review-fanout (R1/R4/R5 병렬 3 agents) + 수치검증 + 폐루프 A/B] Q: 외부 EAOB 리뷰(C0–C4) verify-first 검증 → 재튜닝 적용. R1–R6 전부 실코드 확정(노이즈 주입 전무·R=스칼라I·deadbeat 실측 6.8–13.4 ms·EAOB_* 소비처는 eaob.py 기본 kwargs뿐·…`
- `.claude/journal/reviews.md:27` · `b` · 산출물 없음
  - 현재: `- 2026-07-23 [workflow: eaob-review-fanout (R1/R4/R5 병렬 3 agents) + 수치검증 + 폐루프 A/B] Q: 외부 EAOB 리뷰(C0–C4) verify-first 검증 → 재튜닝 적용. R1–R6 전부 실코드 확정(노이즈 주입 전무·R=스칼라I·deadbeat 실측 6.8–13.4 ms·EAOB_* 소비처는 eaob.py 기본 kwargs뿐·…`
- `.claude/journal/reviews.md:27` · `b` · 산출물 없음
  - 현재: `- 2026-07-23 [workflow: eaob-review-fanout (R1/R4/R5 병렬 3 agents) + 수치검증 + 폐루프 A/B] Q: 외부 EAOB 리뷰(C0–C4) verify-first 검증 → 재튜닝 적용. R1–R6 전부 실코드 확정(노이즈 주입 전무·R=스칼라I·deadbeat 실측 6.8–13.4 ms·EAOB_* 소비처는 eaob.py 기본 kwargs뿐·…`
- `.claude/journal/reviews.md:29` · `b` · 산출물 없음
  - 현재: `- 2026-07-23 [workflow: mpc-state-source-review (R4 소비처/R3 yaw·bound 병렬 2 agents) + 폐루프 A/B 12런] Q: MPC 상태 입력 truth→EAOB 추정치 전환 spec verify-first 검증 → 적용. R1–R6 확정: x_ned는 소비처 1개(단일 스위치 지점 증명), EAOB update가 solve보다 선행(지…`
- `.claude/journal/reviews.md:29` · `b` · 산출물 없음
  - 현재: `- 2026-07-23 [workflow: mpc-state-source-review (R4 소비처/R3 yaw·bound 병렬 2 agents) + 폐루프 A/B 12런] Q: MPC 상태 입력 truth→EAOB 추정치 전환 spec verify-first 검증 → 적용. R1–R6 확정: x_ned는 소비처 1개(단일 스위치 지점 증명), EAOB update가 solve보다 선행(지…`
- `.claude/journal/reviews.md:31` · `b` · 산출물 없음
  - 현재: `- 2026-07-23 [P1 control-theory-advisor 상담 → 구현: DOB 플러그인 구조 + 3-source A/B] Q: EAOB를 순수 disturbance 플러그인으로(MPC x0 = state estimator 출력, EAOB eta/nu_hat 폐기, w_hat만 MPC로) — 사용자 지시로 우선 noisy measurement를 x0로. 구현: `MPC_STA…`
- `.claude/journal/reviews.md:31` · `b` · 산출물 없음
  - 현재: `- 2026-07-23 [P1 control-theory-advisor 상담 → 구현: DOB 플러그인 구조 + 3-source A/B] Q: EAOB를 순수 disturbance 플러그인으로(MPC x0 = state estimator 출력, EAOB eta/nu_hat 폐기, w_hat만 MPC로) — 사용자 지시로 우선 noisy measurement를 x0로. 구현: `MPC_STA…`
- `.claude/journal/reviews.md:33` · `b` · 산출물 없음
  - 현재: `- 2026-07-24 [측정노이즈 축소: sigma 스케일 스윕 + A/B 재검증] Q: noisy measurement가 추종오차보다 커서 x0 chatter 유발 → 추종오차 아래로 축소 요청. sigma 스케일 스윕(pos 5/2/1/0.5/0.25 cm, square C meas)으로 chatter/tracking/NIS 곡선 확인 후 **x/y 0.5 cm** 선택(추종오차 ~1…`
- `.claude/journal/reviews.md:33` · `b` · 산출물 없음
  - 현재: `- 2026-07-24 [측정노이즈 축소: sigma 스케일 스윕 + A/B 재검증] Q: noisy measurement가 추종오차보다 커서 x0 chatter 유발 → 추종오차 아래로 축소 요청. sigma 스케일 스윕(pos 5/2/1/0.5/0.25 cm, square C meas)으로 chatter/tracking/NIS 곡선 확인 후 **x/y 0.5 cm** 선택(추종오차 ~1…`
- `.claude/journal/reviews.md:33` · `b` · 산출물 없음
  - 현재: `- 2026-07-24 [측정노이즈 축소: sigma 스케일 스윕 + A/B 재검증] Q: noisy measurement가 추종오차보다 커서 x0 chatter 유발 → 추종오차 아래로 축소 요청. sigma 스케일 스윕(pos 5/2/1/0.5/0.25 cm, square C meas)으로 chatter/tracking/NIS 곡선 확인 후 **x/y 0.5 cm** 선택(추종오차 ~1…`
- `.claude/journal/reviews.md:33` · `c` · 예시/유도값이 측정치로 서술됨
  - 현재: `- 2026-07-24 [측정노이즈 축소: sigma 스케일 스윕 + A/B 재검증] Q: noisy measurement가 추종오차보다 커서 x0 chatter 유발 → 추종오차 아래로 축소 요청. sigma 스케일 스윕(pos 5/2/1/0.5/0.25 cm, square C meas)으로 chatter/tracking/NIS 곡선 확인 후 **x/y 0.5 cm** 선택(추종오차 ~1…`
- `.claude/journal/reviews.md:37` · `b` · 산출물 없음
  - 현재: `- 2026-08-03 [P2 워크플로 5렌즈(lifecycle/dataset-validity/axes-config/edits-blast-radius/safety-resources) + 건별 adversarial verify, 38건 제기 → 세션한도로 verify 20/38 완료] Q: 신규 `c3_camera/c3_option_sweep.py`(컬러해상도 × 압축 × depth on/o…`
- `ISAAC_AUV_SETUP_RUNBOOK.md:14` · `b` · 산출물 없음
  - 현재: `WarpAUV 정책 **2048 envs · 400 iter · ~19.7M steps GPU 학습 완료**(보상 0.78→~76, 에러 0). 4개 Blackwell 블로커를 순서대로 해결:`
- `ISAAC_AUV_SETUP_RUNBOOK.md:69` · `b` · 산출물 없음
  - 현재: `# 설치 위치 제안: 홈 디렉토리 (디스크 625GB 여유로 충분)`
- `ISAAC_AUV_SETUP_RUNBOOK.md:186` · `b` · 산출물 없음
  - 현재: `- README 기준: 2048 envs로 **~400 iter**에 수렴, mean reward ~95–100. 수렴 문제 시 action penalty↓.`
- `ISAAC_AUV_SETUP_RUNBOOK.md:329` · `c` · 예시/유도값이 측정치로 서술됨
  - 현재: `> 대안(재부팅 회피): IOMMU off 안 해도 **매 부팅마다 ~5–6분 P2P 검증을 기다리면** 통과한다(무한 행 아님).`
- `KNOWN_ISSUES.md:18` · `c` · 예시/유도값이 측정치로 서술됨
  - 현재: `Snell 예측은 **84.3°** → 물속 예측과 ±1° 이내, 공기 스펙과는 41° 차이.`
- `KNOWN_ISSUES.md:30` · `c` · 예시/유도값이 측정치로 서술됨
  - 현재: `MinZ 표(400p 298.9 mm)는 수중 fx 기준이라 **공기 중 실제 최소거리는 ≈225 mm**이고,`
- `KNOWN_ISSUES.md:59` · `b` · 산출물 없음
  - 현재: `storm PID−MPC −4.24 → +8.60 cm 부호 역전, crossover 소멸).`
- `KNOWN_ISSUES.md:70` · `b` · 산출물 없음
  - 현재: `안 물리지만, 스웨이 보수 스택(추력한계 속도로 파정 역류 과도) ≈85–100 N에서는 물리고`
- `KNOWN_ISSUES.md:82` · `b` · 산출물 없음
  - 현재: `프레임을 25–76% 기각해 파랑 추종 자체를 차단(CDW radRMS 4.5→15.5 cm, CD 1.35→1.43 cm);`
- `KNOWN_ISSUES.md:83` · `b` · 산출물 없음
  - 현재: `기각이 다시 혁신을 키우는 악순환으로 NIS 통계도 오염(137 vs gate-off 25). 최종 기본`
- `KNOWN_ISSUES.md:84` · `b` · 산출물 없음
  - 현재: `tau_dist=0.2에선 같은 시나리오에서 게이트가 아예 발동하지 않음(0/1067). 일관성 게이트는`
- `KNOWN_ISSUES.md:145` · `b` · 산출물 없음
  - 현재: `heavy_gripper(13.7 kg)에서 0.2717 N — 30 N sway authority의 ~0.9%라 실효 동일 최적해,`
- `KNOWN_ISSUES.md:172` · `b` · 산출물 없음
  - 현재: `kick 0.5 rad/s → 1.5 s 만에 \|q\|>60 rad/s로 재현).`
- `KNOWN_ISSUES.md:186` · `b` · 산출물 없음
  - 현재: `(DP), 44/42(square) — τ_dist=0.2로 검증했던 목표범위(14–24) 크게 이탈, radRMS도`
- `KNOWN_ISSUES.md:192` · `b` · 산출물 없음
  - 현재: `CDW/T=60/seed 0)에서 **NIS 245.8→100.0, NEES 398.9→122.0으로 2.5–3.3× 개선**(여전히`
- `KNOWN_ISSUES.md:238` · `b` · 산출물 없음
  - 현재: `126°/s(횡력 8 N)~474°/s(30 N)가 슬루 상한 60°/s를 크게 초과. 힘이 무한해도 기수가 경로를`
- `KNOWN_ISSUES.md:241` · `b` · 산출물 없음
  - 현재: `vs 직선 0.22–0.28 cm(≈10×). **surge 박스 8→30 N에서 소수점까지 불변**(코너 명령 surge는`
- `KNOWN_ISSUES.md:245` · `b` · 산출물 없음
  - 현재: `줄여(33% vs 37%) 코너 특유 효과가 아님; 박스가 실제 개선하는 건 코너 **직후 회복**.`
- `KNOWN_ISSUES.md:247` · `b` · 산출물 없음
  - 현재: `trajectory_compare 그림에서 유독 도드라진다(절대값으로는 3사 중 최소: gentle CDW 코너`
- `README_fisheye_gantry.md:125` · `b` · 산출물 없음
  - 현재: `The left pane stays visible no matter which tab is active, so you always see live position and the workspace map. The splitter between left/right is draggable; default ratio 38/62, window default `1500 × 900`, minimum `…`
- `README_fisheye_gantry.md:141` · `b` · 산출물 없음
  - 현재: `Tick spacing adapts to the visible range (~6–10 major ticks at any zoom). Implementation is `pyqtgraph` when installed; falls back to a hand-painted `QPainter` widget otherwise. All updates ride the 10 Hz status-poll — …`
- `README_fisheye_gantry.md:235` · `b` · 산출물 없음
  - 현재: `timing (mock): click → handler return in **~3 ms**, halt distance after`
- `README_fisheye_gantry.md:242` · `b` · 산출물 없음
  - 현재: `on offscreen Qt with focus on a `QDoubleSpinBox`: handler ran in **~3 ms**).`
- `README_fisheye_gantry.md:354` · `b` · 산출물 없음
  - 현재: ``4×4 grid · 2 frames/cell · target ≈ 32 frames · ≥12 cells required`.`
- `README_fisheye_gantry.md:371` · `b` · 산출물 없음
  - 현재: `- Record at least 10–20 seconds; 30 s gives ≥ 16 cells easily.`
- `README_fisheye_gantry.md:382` · `b` · 산출물 없음
  - 현재: `\| Preliminary calibration \| Runs `cv2.fisheye.calibrate` on one frame per cell \|`
- `README_fisheye_gantry.md:385` · `b` · 산출물 없음
  - 현재: ``
- `README_fisheye_gantry.md:412` · `b` · 산출물 없음
  - 현재: `MIN_SHARPNESS       = 50.0   # Laplacian-variance floor; lower in dim environments`
- `README_fisheye_gantry.md:432` · `b` · 산출물 없음
  - 현재: `4. The progress dialog should show "16 / 16 cells covered, 32 frames picked".`
- `README_fisheye_gantry.md:567` · `b` · 산출물 없음
  - 현재: `run_metadata.json         # CLI args, K/D/T_gantry_camera, soft limits, t0/t1`
- `README_fisheye_gantry.md:661` · `b` · 산출물 없음
  - 현재: `When SURVEYING begins, a daemon-thread recorder (`survey_diagnostics.py`, never`
- `README_fisheye_gantry.md:728` · `b` · 산출물 없음
  - 현재: `- **`--use-frames` (slower — ~10–90 ms/frame; a 3-min/~5000-frame run ≈ 1–7 min).**`
- `README_fisheye_gantry.md:795` · `c` · 예시/유도값이 측정치로 서술됨
  - 현재: `[tag-map] Loaded 24 tags from tag map`
- `README_fisheye_gantry.md:972` · `b` · 산출물 없음
  - 현재: `> **Verifying R.** Generate a run's `trajectory_interactive.html` and watch stderr. If the Velocity tab's camera (red) does not track the gantry (blue) and you see `R_gantry_to_slam may be TRANSPOSED`, replace `R` with …`
- `README_fisheye_gantry.md:1063` · `b` · 산출물 없음
  - 현재: `* Acceleration column in `gantry_telemetry.csv` is a 5-sample SMA-smoothed`
- `bluerov2_issac_paper/README.md:78` · `b` · 산출물 없음
  - 현재: `- Generally converges in about 400 iterations with 2048 environments and achieves mean total reward ~95-100. Lowering action penalty often helps if there are issues with convergence.`
- `bluerov2_mujoco_dobmpc/README.md:76` · `b` · 산출물 없음
  - 현재: `\| MPC    \| 0.1045 \| 0.1019 \| 0.1430 \| 0.0559 \|`
- `bluerov2_mujoco_dobmpc/README.md:77` · `b` · 산출물 없음
  - 현재: `\| DOBMPC \| **0.0074** \| **0.0031** \| **0.0150** \| **0.0015** \|`
- `bluerov2_mujoco_dobmpc/README.md:85` · `b` · 산출물 없음
  - 현재: `\| MPC    \| 0.1667 \| 0.1524 \| 0.2100 \| **0.0520** \|`
- `bluerov2_mujoco_dobmpc/README.md:86` · `b` · 산출물 없음
  - 현재: `\| DOBMPC \| **0.1030** \| **0.0837** \| **0.0820** \| 0.0825 \|`
- `bluerov2_mujoco_dobmpc/README.md:94` · `b` · 산출물 없음
  - 현재: `Position-channel gains of 1.6-2.6x dominate, as in the paper. Figures were`
- `bluerov2_mujoco_dobmpc/README.md:96` · `b` · 산출물 없음
  - 현재: `(closed-loop behaviour is indistinguishable, ~1.5x slower).`
- `bluerov2_mujoco_dobmpc/README.md:117` · `b` · 산출물 없음
  - 현재: `even DOBMPC stops rejecting it (we verified this). `R = [0.5 0.5 0.5`
- `bluerov2_mujoco_marinegym/README.md:39` · `b` · 산출물 없음
  - 현재: `\| NMPC solved by **acados SQP-RTI** (~1 ms, default) with IPOPT fallback \| `dobmpc/mpc_acados.py`, `dobmpc/mpc.py` \| ✅ verified \|`
- `bluerov2_mujoco_marinegym/README.md:42` · `b` · 산출물 없음
  - 현재: `\| **DOB plug-in: NMPC x0 = state-estimator output** (2026-07-23 (3), default `params.MPC_STATE_SOURCE="meas"`): the EAOB is a pure disturbance plug-in — it hands the MPC only `w_hat`; x0 + yaw-ref anchor come from the …`
- `bluerov2_mujoco_marinegym/README.md:42` · `b` · 산출물 없음
  - 현재: `\| **DOB plug-in: NMPC x0 = state-estimator output** (2026-07-23 (3), default `params.MPC_STATE_SOURCE="meas"`): the EAOB is a pure disturbance plug-in — it hands the MPC only `w_hat`; x0 + yaw-ref anchor come from the …`
- `bluerov2_mujoco_marinegym/README.md:44` · `b` · 산출물 없음
  - 현재: `\| **NMPC surge box 8 → 30 N on heavy** (2026-07-24 (3)): `params.U_MAX` surge was a rank-5 relic (the real `My=−0.0725·Fx` tumble limit applies to bluerov2 only) that gave the NMPC a 3.75× handicap vs the PID's `f_max`…`
- `bluerov2_mujoco_marinegym/README.md:44` · `b` · 산출물 없음
  - 현재: `\| **NMPC surge box 8 → 30 N on heavy** (2026-07-24 (3)): `params.U_MAX` surge was a rank-5 relic (the real `My=−0.0725·Fx` tumble limit applies to bluerov2 only) that gave the NMPC a 3.75× handicap vs the PID's `f_max`…`
- `bluerov2_mujoco_marinegym/README.md:105` · `b` · 산출물 없음
  - 현재: `visual-only (mass unknown). Uses `GAINS_HEAVY` (no active roll/pitch leveling → ~1° residual`
- `bluerov2_mujoco_marinegym/README.md:136` · `b` · 산출물 없음
  - 현재: `(composition/buoyancy/gripper/PID) + DOB-MPC DP hold 1.3 cm (still), 1.4 cm (current+waves).`
- `bluerov2_mujoco_marinegym/README.md:143` · `b` · 산출물 없음
  - 현재: `is byte-identical in physics (verified 3000-step rollout Δ=0 vs the old gray body).`
- `bluerov2_mujoco_marinegym/README.md:173` · `b` · 산출물 없음
  - 현재: `instead of ~198 textured boxes (measured ~2.2× faster model parse; the GPU texture upload win`
- `bluerov2_mujoco_marinegym/README.md:198` · `b` · 산출물 없음
  - 현재: `(verified: old-vs-new 3000-step rollout Δ=0, incl. `--disturb`). Regenerate / retune with:`
- `bluerov2_mujoco_marinegym/README.md:268` · `b` · 산출물 없음
  - 현재: `\| `python test_<name>.py` (11 files) · `python -m disturbance.test_{waves,env}` \| per-component smoke/unit tests \| 1 \|`
- `bluerov2_mujoco_marinegym/README.md:282` · `b` · 산출물 없음
  - 현재: `\| `python tests/test_load.py` \| model loads; mass ∈ [9, 12] kg; 6 thruster sites; zero-control stability, no NaN. Flags: `--seconds`, `--render out.png`, `--viewer` \|`
- `bluerov2_mujoco_marinegym/README.md:294` · `b` · 산출물 없음
  - 현재: `\| `python -m disturbance.test_env` \| 21 checks on current/env: exact Gauss–Markov discretization, C/CD/CW/CDW mode gating, identical wave phases across modes per seed, FK force = ρ·vol·C_M·a_wave \|`
- `bluerov2_mujoco_marinegym/README.md:336` · `b` · 산출물 없음
  - 현재: ``pool_water_surface` extent — horizontal edges + the waterline; ~15 s per crossing at the`
- `bluerov2_mujoco_marinegym/README.md:418` · `b` · 산출물 없음
  - 현재: `python -m disturbance.test_waves && python -m disturbance.test_env   # 33 unit asserts first`
- `bluerov2_mujoco_marinegym/README.md:586` · `b` · 산출물 없음
  - 현재: `\| `python verify/verify_eaob.py` \| EAOB observer validation: runs a disturbed DP hold twice (`profile="verify"` clean-deadbeat vs `profile="perf"` + injected sensor noise) and scores the OBSERVER — per-axis RMSE of `w…`
- `bluerov2_mujoco_marinegym/docs/02_MODEL.md:91` · `b` · 산출물 없음
  - 현재: `- Mass 11.20 kg, inertia as above, 6 thruster sites, correct vectored layout.`
- `bluerov2_mujoco_marinegym/docs/04_HYDRO.md:78` · `c` · 예시/유도값이 측정치로 서술됨
  - 현재: `- Below ~Fx ≈ 5 N: stable, nearly-level glide (good for open-loop driving).`
- `bluerov2_mujoco_marinegym/docs/05_TELEOP.md:45` · `b` · 산출물 없음
  - 현재: `- forces: `0.003 m/N`, cap `0.6 m` (buoyancy 111 N → 0.33 m ≈ vehicle size);`
- `bluerov2_mujoco_marinegym/docs/05_TELEOP.md:46` · `b` · 산출물 없음
  - 현재: `- velocities: `0.5 m per m/s`, cap `0.4 m` (current 0.2 m/s → 0.10 m).`
- `bluerov2_mujoco_marinegym/docs/05_TELEOP.md:90` · `b` · 산출물 없음
  - 현재: `with the rank-5 allocation.)`
- `bluerov2_mujoco_marinegym/docs/05_TELEOP.md:97` · `b` · 산출물 없음
  - 현재: `gives a brief red spike (e.g. ~31 N). The Phase 1–4 suites still pass — the viz does`
- `bluerov2_mujoco_marinegym/docs/06_ENVIRONMENT.md:14` · `b` · 산출물 없음
  - 현재: `Python 3.13, `mujoco` 3.9.x. Activate: `source .venv/bin/activate` from the repo`
- `bluerov2_mujoco_marinegym/docs/06_ENVIRONMENT.md:51` · `c` · 예시/유도값이 측정치로 서술됨
  - 현재: `**595.71.05** (CUDA 13.2). The runtime is split across **two conda envs** because`
- `bluerov2_mujoco_marinegym/docs/CONTROL_METHODOLOGY.md:130` · `b` · 산출물 없음
  - 현재: `(both coefficient sets recovered back out of the sim to 0.00 %, T4.3)`
- `bluerov2_mujoco_marinegym/docs/CONTROL_METHODOLOGY.md:139` · `b` · 산출물 없음
  - 현재: `└−a₂v   a₁u    0   −a₅q   a₄p     0   ┘   1e-14 (T1.1–1.2)`
- `bluerov2_mujoco_marinegym/docs/CONTROL_METHODOLOGY.md:313` · `b` · 산출물 없음
  - 현재: `50 ms): **NMPC.solve ≈ 83 ms (≈79%)**, EAOB.update ≈ 22 ms (≈21%), full tick ≈ 106 ms = **0.47×`
- `bluerov2_mujoco_marinegym/docs/CONTROL_METHODOLOGY.md:435` · `b` · 산출물 없음
  - 현재: `\| DP (15 s) \| 15.0 → 13.4° \| 30.0 → **22.9°** \| radial 4.9 → 6.1 cm \| 0.22 → **0.09** \|`
- `bluerov2_mujoco_marinegym/docs/CONTROL_METHODOLOGY.md:487` · `b` · 산출물 없음
  - 현재: `\| solve / tick (N=60) \| median **100 ms** (over the 50 ms budget) \| median **0.97 ms**, max 1.1 ms \|`
- `bluerov2_mujoco_marinegym/docs/CONTROL_METHODOLOGY.md:521` · `b` · 산출물 없음
  - 현재: `\| PID \| 14.86 \| 14.74 \| 15.16 \| 10.4 → 12.4 cm \|`
- `bluerov2_mujoco_marinegym/docs/CONTROL_METHODOLOGY.md:534` · `b` · 산출물 없음
  - 현재: `path). Per-seed, ideal DOB-MPC: seeds 0/1/2/4 = 4.1 / 0.7 / 0.9 / 1.4 cm, n_fail 0 (excellent); **seed 3`
- `bluerov2_mujoco_marinegym/docs/CONTROL_METHODOLOGY.md:611` · `b` · 산출물 없음
  - 현재: `**Result.** Closed-loop DP (dobmpc, seed 0, 20 s, disturbance ON): **ideal radial 5.02 cm / jitter 4.30 cm`
- `bluerov2_mujoco_marinegym/docs/CONTROL_METHODOLOGY.md:675` · `b` · 산출물 없음
  - 현재: `> [compute_heavy_inertia.py](../compute_heavy_inertia.py) (sensitivity: m_v 0.10→0.344 kg gives`
- `bluerov2_mujoco_marinegym/docs/CONTROL_METHODOLOGY.md:704` · `c` · 예시/유도값이 측정치로 서술됨
  - 현재: `Heavy:     6×8,  rank 6   — FULLY ACTUATED (verified: a pure pitch wrench realizes My=1.000)`
- `bluerov2_mujoco_marinegym/docs/CONTROL_METHODOLOGY.md:726` · `b` · 산출물 없음
  - 현재: `Full actuation **actively levels pitch** (11.8° trim → 0.8°) and tightens station-keeping (5.0 → 3.3 cm).`
- `bluerov2_mujoco_marinegym/docs/CONTROL_METHODOLOGY.md:779` · `b` · 산출물 없음
  - 현재: `disturbance interface. 34 unit asserts + smoke pass. Run:`
- `bluerov2_mujoco_marinegym/docs/CONTROL_METHODOLOGY.md:803` · `b` · 산출물 없음
  - 현재: ``heading_follow: true`, `yaw_rate_deg_s: 60`; viewer flags `--heading {follow,fixed}`, `--yaw-rate`. Verified:`
- `bluerov2_mujoco_marinegym/docs/CONTROL_METHODOLOGY.md:825` · `b` · 산출물 없음
  - 현재: `\| DP dobmpc (regression) \| **0.00 cm** (fully-actuated + DOB rejects the current) \|`
- `bluerov2_mujoco_marinegym/docs/CONTROL_METHODOLOGY.md:889` · `b` · 산출물 없음
  - 현재: `1-lap square NONE via run_viewer: **radial RMS 1.34 cm** (t>5 s) vs **17.8 cm** with the old gains in`
- `bluerov2_mujoco_marinegym/docs/CONTROL_METHODOLOGY.md:894` · `b` · 산출물 없음
  - 현재: `filter the D term. Beyond the \|S(1.6)\| ≈ 0.41 wave-band residual, the answer is DOB-MPC, not more PID gain.`
- `bluerov2_mujoco_marinegym/docs/CONTROL_METHODOLOGY.md:900` · `b` · 산출물 없음
  - 현재: `**Why.** On the square, DOB-MPC's largest CW/CDW error concentrates at the **(1,1) upstream`
- `bluerov2_mujoco_marinegym/docs/CONTROL_METHODOLOGY.md:922` · `b` · 산출물 없음
  - 현재: `**Result (A/B, dobmpc, seed 0, current 0° / wave 0°, run_viewer headless).** Yaw slew-to-5° at the`
- `bluerov2_mujoco_marinegym/docs/CONTROL_METHODOLOGY.md:962` · `b` · 산출물 없음
  - 현재: `term into an energy pump: a torque-free 0.5 rad/s pitch kick exploded to \|q\| > 60 rad/s in 1.5 s. Isolated by`
- `bluerov2_mujoco_marinegym/docs/CONTROL_METHODOLOGY.md:974` · `b` · 산출물 없음
  - 현재: `**Results:** PID DP hold 0.0 cm (still water, 20 s); DOB-MPC DP hold rms(12–20 s) **1.3 cm** still / **1.4 cm**`
- `bluerov2_mujoco_marinegym/docs/CONTROL_METHODOLOGY.md:976` · `b` · 산출물 없음
  - 현재: `consistent with the C-mode analysis). heavy reference on the same harness: 0.4 cm (cw). acados-vs-IPOPT`
- `bluerov2_mujoco_marinegym/docs/CONTROL_METHODOLOGY.md:990` · `b` · 산출물 없음
  - 현재: `vendor 575×254×457 mm exactly), a global voxel-occupancy grid search over translation (no ICP local minima),`
- `bluerov2_mujoco_marinegym/docs/CONTROL_METHODOLOGY.md:999` · `c` · 예시/유도값이 측정치로 서술됨
  - 현재: `bracket straddles the Newton-gripper tube with ~1 mm clearance — independently confirming the guessed`
- `bluerov2_mujoco_marinegym/docs/CONTROL_METHODOLOGY.md:1009` · `b` · 산출물 없음
  - 현재: `be `mesh_quat` itself — the old bake's ~180° principal rotation masked this (conj(q) = −q). Verified: vertex`
- `bluerov2_mujoco_marinegym/docs/CONTROL_METHODOLOGY.md:1024` · `b` · 산출물 없음
  - 현재: `COM, so `<inertial pos>` = 0. Gains: `heavy_c3` inherits `GAINS_HEAVY` (fully-actuated fallback) — PID holds the`
- `bluerov2_mujoco_marinegym/docs/CONTROL_METHODOLOGY.md:1042` · `b` · 산출물 없음
  - 현재: `"cancellation" schemes (conj(mesh_quat), then mesh_quat) double-handled the reframe; the C3's near-square cross`
- `bluerov2_mujoco_marinegym/docs/CONTROL_METHODOLOGY.md:1091` · `b` · 산출물 없음
  - 현재: `COM + jaw weight, ~0.2–0.5 N·m) leaves a steady tilt `φ_ss = τ/(kp + B_restore)`. Measured at a 0.40 N·m constant`
- `bluerov2_mujoco_marinegym/docs/CONTROL_METHODOLOGY.md:1097` · `b` · 산출물 없음
  - 현재: `rp_gate = 0.15 rad`), clamp `\|ki·I\| < rp_i_max = (1.5, 2.0)` N·m. Verified the 3rd-order closed loop`
- `bluerov2_mujoco_marinegym/docs/CONTROL_METHODOLOGY.md:1106` · `b` · 산출물 없음
  - 현재: `slew on all three body-force axes (finite actuator/command bandwidth; rarely binds at 120 N/s — ~66 N/s at hover).`
- `bluerov2_mujoco_marinegym/docs/CONTROL_METHODOLOGY.md:1111` · `b` · 산출물 없음
  - 현재: `aggressive far-offset slew — pitch transiently swings to ~70° WITHOUT the guard vs ~45–56° WITH it (recovering to`
- `bluerov2_mujoco_marinegym/docs/CONTROL_METHODOLOGY.md:1142` · `b` · 산출물 없음
  - 현재: `the real (F,H) at hover: per-axis w-error time constants 6.8–13.4 ms (tau-channel Kalman gains −0.976…−0.999),`
- `bluerov2_mujoco_marinegym/docs/CONTROL_METHODOLOGY.md:1160` · `b` · 산출물 없음
  - 현재: `states, cross-checked against hydro's independent `diag_wtrue` diagnostic — the two agree to ≤0.08 N).** At the`
- `bluerov2_mujoco_marinegym/docs/CONTROL_METHODOLOGY.md:1165` · `b` · 산출물 없음
  - 현재: `the only value passing BOTH ranges in CDW; in CD it gives NIS 14.4 PASS, RMSE X/Y/Z 0.29/0.41/0.48 N. CD NEES`
- `bluerov2_mujoco_marinegym/docs/CONTROL_METHODOLOGY.md:1172` · `b` · 산출물 없음
  - 현재: `slopes vs \|nu\| ≈ 0 with \|r\| ≤ 0.18 everywhere in the final config (no model error leaking; the small CDW Z/M`
- `bluerov2_mujoco_marinegym/docs/CONTROL_METHODOLOGY.md:1180` · `b` · 산출물 없음
  - 현재: `updates that track the disturbance — CDW radRMS 4.53→15.45 cm (gate off→on), CD 1.35→1.43 cm (measured at`
- `bluerov2_mujoco_marinegym/docs/CONTROL_METHODOLOGY.md:1186` · `b` · 산출물 없음
  - 현재: `seed 0).** DP mode C: DOB-MPC keeps its headline DC rejection under realistic sensors — dc offset`
- `bluerov2_mujoco_marinegym/docs/CONTROL_METHODOLOGY.md:1212` · `b` · 산출물 없음
  - 현재: `safe — verified: max per-tick \|Δpsi_hat\| ≤ 0.12 rad across all runs (a branch jump would be ~6.3). Side benefit:`
- `bluerov2_mujoco_marinegym/docs/CONTROL_METHODOLOGY.md:1227` · `b` · 산출물 없음
  - 현재: `config/base.yaml's wave block was strengthened at 2026-07-23 19:28 (Hs 0.75→1.2 m, Tp 12→6 s, γ 5→2, s 30→10,`
- `bluerov2_mujoco_marinegym/docs/CONTROL_METHODOLOGY.md:1273` · `c` · 예시/유도값이 측정치로 서술됨
  - 현재: `punishing than the sim's 5 cm white @ 20 Hz), so the sim's "meas" is pessimistic. This is the deliberate,`
- `bluerov2_mujoco_marinegym/docs/CONTROL_METHODOLOGY.md:1285` · `b` · 산출물 없음
  - 현재: `so w_hat quality is preserved (verify_eaob CD w-RMSE 0.24/0.34/0.39 N, if anything slightly better than the`
- `bluerov2_mujoco_marinegym/docs/CONTROL_METHODOLOGY.md:1292` · `b` · 산출물 없음
  - 현재: `\|---\|---\|---\|`
- `bluerov2_mujoco_marinegym/docs/CONTROL_METHODOLOGY.md:1335` · `b` · 산출물 없음
  - 현재: `\| pid  NONE truth \| 1.49 \| 0.08 \| 0.85 \|`
- `bluerov2_mujoco_marinegym/docs/CONTROL_METHODOLOGY.md:1346` · `b` · 산출물 없음
  - 현재: ``radial_max` 4.402 cm byte-for-byte, confirming `meas_noise=False` is the exact pre-change path. Verified:`
- `bluerov2_mujoco_marinegym/docs/CONTROL_METHODOLOGY.md:1365` · `b` · 산출물 없음
  - 현재: `\| storm  \| 26.32 \| 30.55 (box active 60–66 % of ticks) \| **17.72** (0.7–1.8 %) \|`
- `bluerov2_mujoco_marinegym/docs/CONTROL_METHODOLOGY.md:1380` · `b` · 산출물 없음
  - 현재: `\| storm  \| 26.14 \| **17.86** (was 28.82) \| **7.30** (was 17.36) \| 40.3 N \| 0.0000 \|`
- `bluerov2_mujoco_marinegym/docs/CONTROL_METHODOLOGY.md:1387` · `b` · 산출물 없음
  - 현재: `corrections the 8 N box provoked. `verify_acados` still passes (acados↔IPOPT max\|Δu\| = 0.0615 N < 0.25 N gate,`
- `bluerov2_mujoco_marinegym/docs/CONTROL_METHODOLOGY.md:1396` · `b` · 산출물 없음
  - 현재: `**What survives as real control findings** (unaffected by the box): the MPC's error PSD is 3–8× below the PID's at`
- `bluerov2_mujoco_marinegym/docs/CONTROL_METHODOLOGY.md:1400` · `b` · 산출물 없음
  - 현재: `**Not established:** that the ladder's crossover was a *frequency* phenomenon — the sea-state ladder co-varies Hs`
- `bluerov2_mujoco_marinegym/docs/CONTROL_METHODOLOGY.md:1430` · `b` · 산출물 없음
  - 현재: `its existing per-run CSVs — all 15 (5 modes × 3 controllers) boxed values now equal `results_raw.csv` to`
- `bluerov2_mujoco_marinegym/docs/HYDRO_VERIFICATION.md:32` · `b` · 산출물 없음
  - 현재: `\| **T4** \| Restoring pendulum \| restoring stiffness + effective inertia \| underdamped, ω_n=√(k/(I+M_A_rot)) \| roll T=3.85 s vs 3.89 (1%); pitch 4.79 vs 5.16 (7%) \| ✅ \|`
- `bluerov2_mujoco_marinegym/docs/HYDRO_VERIFICATION.md:33` · `b` · 산출물 없음
  - 현재: `\| \| — static equilibrium \| restoring stiffness alone \| tilt = asin(M/k) \| 26.1° vs 26.8° (2.7%) \| ✅ \|`
- `bluerov2_mujoco_marinegym/docs/HYDRO_VERIFICATION.md:35` · `b` · 산출물 없음
  - 현재: `\| **T5** \| Added mass (effective inertia, Ω=0.5–5 rad/s) \| M_A delivery through the EMA filter \| effective mass = m + M_A·Re{H(Ω)} ≈ m+M_A \| **0.0–0.3%** on surge/sway/heave; sign −M_A on all 6 axes \| ✅ \|`
- `bluerov2_mujoco_marinegym/docs/HYDRO_VERIFICATION.md:36` · `b` · 산출물 없음
  - 현재: `\| **T6** \| Coriolis passivity \| skew-symmetry of C_A \| νᵀC_A(ν)ν = 0 \| **4.3e-14** \| ✅ \|`
- `bluerov2_mujoco_marinegym/docs/HYDRO_VERIFICATION.md:65` · `c` · 예시/유도값이 측정치로 서술됨
  - 현재: `physically relevant band (Ω = 0.5–5 rad/s) its in-phase gain Re{H(Ω)} ≈ 1 and the measured effective`
- `bluerov2_mujoco_marinegym/docs/HYDRO_VERIFICATION.md:97` · `b` · 산출물 없음
  - 현재: `CasADi-symbolic residual *exactly* 0), not merely the quadratic form νᵀC_Aν = 0. M = M_RB + M_A is SPD`
- `c3_camera/README.md:106` · `b` · 산출물 없음
  - 현재: `\| `DEPTHAI_BOOTUP_TIMEOUT` \| `30000` ms \| 실측 부팅 약 12초인데 기본값이 15초라 여유가 너무 얇다 \|`
- `c3_camera/README.md:215` · `b` · 산출물 없음
  - 현재: `\| 컬러 latency (p50) \| 약 **3000 ms** \| **38 ms** \|`
- `c3_camera/README.md:217` · `b` · 산출물 없음
  - 현재: `\| fps \| 10 요청 → 6–8 실측 \| **15 요청 → 15.00 실측** \|`
- `c3_camera/README.md:221` · `b` · 산출물 없음
  - 현재: `420프레임 연속 측정, 실측 링크 71.6 Mbit/s.`
- `c3_camera/README.md:231` · `b` · 산출물 없음
  - 현재: `\| 컬러만 960x540 NV12 \| 13.8 \| 778 kB \| 86 Mbit/s \|`
- `c3_camera/README.md:232` · `b` · 산출물 없음
  - 현재: `\| 960x540 NV12 + depth \| 9.25 \| 1.24 MB \| **91.7** Mbit/s \|`
- `c3_camera/README.md:233` · `b` · 산출물 없음
  - 현재: `\| 1920x1080 NV12 + depth \| 3.19 \| 3.57 MB \| **91.1** Mbit/s \|`
- `c3_camera/README.md:234` · `b` · 산출물 없음
  - 현재: `\| 480x270 NV12 + depth \| 17.5 \| 655 kB \| **91.6** Mbit/s \|`
- `c3_camera/README.md:254` · `b` · 산출물 없음
  - 현재: `\| 480x270 \| raw \| 17.5 \| 123 ms \| 121 ms \| 12.5 % \|`
- `c3_camera/README.md:255` · `b` · 산출물 없음
  - 현재: `\| 480x270 \| **mjpeg** \| **20.0** \| **57.6 ms** \| 107 ms \| **0 %** \|`
- `c3_camera/README.md:256` · `b` · 산출물 없음
  - 현재: `\| 960x540 \| raw \| 9.1 \| 250 ms \| 405 ms \| 53.5 % \|`
- `c3_camera/README.md:257` · `b` · 산출물 없음
  - 현재: `\| 960x540 \| **mjpeg** \| **19.1** \| **123 ms** \| 124 ms \| 5 % \|`
- `c3_camera/README.md:258` · `b` · 산출물 없음
  - 현재: `\| 1920x1080 \| raw \| 3.2 \| 353 ms \| 495 ms \| 83.5 % \|`
- `c3_camera/README.md:259` · `b` · 산출물 없음
  - 현재: `\| 1920x1080 \| **mjpeg** \| **13.1** \| **147 ms** \| 154 ms \| 34.8 % \|`
- `c3_camera/README.md:261` · `b` · 산출물 없음
  - 현재: `960x540에서 fps 2.1배, 지연 1/2. 풀 1080p에서는 fps 4.1배, 지연 1/2.4.`
- `c3_camera/README.md:263` · `b` · 산출물 없음
  - 현재: ``--color-encode none`(대신 약 9 fps)을 쓴다.`
- `c3_camera/README.md:265` · `b` · 산출물 없음
  - 현재: `MJPEG 실측 압축률은 q90에서 **약 6:1** (960x540이 112–139 kB/frame). 처음에`
- `c3_camera/README.md:278` · `b` · 산출물 없음
  - 현재: `지연 편차(p95)가 줄어든다. 드롭 38%로 9 fps 받는 것보다 9 fps 요청해서`
- `c3_camera/README.md:291` · `b` · 산출물 없음
  - 현재: `\| 640x360 mjpeg @12 + depth 480x270 \| 67 \| 31.8 \| 35 % \| **35.5 ms** \| 36.1 \|`
- `c3_camera/README.md:292` · `b` · 산출물 없음
  - 현재: `\| 640x360 mjpeg @20 + depth 480x270 \| 67 \| 52.0 \| 58 % \| **35.0 ms** \| 35.8 \|`
- `c3_camera/README.md:293` · `b` · 산출물 없음
  - 현재: `\| 960x540 mjpeg @15 + depth 640x360 **(기본)** \| 131 \| 71.4 \| 78 % \| 39 ms \| 80 \|`
- `c3_camera/README.md:294` · `b` · 산출물 없음
  - 현재: `\| 1920x1080 mjpeg @12 + depth 480x270 \| 377 \| 61.9 \| 68 % \| 77 ms \| 84 \|`
- `c3_camera/README.md:295` · `b` · 산출물 없음
  - 현재: `\| 480x270 mjpeg @20 + depth 640x360 \| ~30 \| ~77 \| 84 % \| 58 ms \| 58 \|`
- `c3_camera/README.md:296` · `b` · 산출물 없음
  - 현재: `\| 960x540 mjpeg @30 + depth **480x270** \| 131 \| 90.4 \| 99 % \| 84 ms \| 96 \|`
- `c3_camera/README.md:297` · `b` · 산출물 없음
  - 현재: `\| 960x540 mjpeg @30 + depth 640x360 \| 131 \| 88.2 \| 96 % \| 120 ms \| — \|`
- `c3_camera/README.md:298` · `b` · 산출물 없음
  - 현재: `\| 1920x1080 mjpeg @20 \| 377 \| 포화 \| >100 % \| 147 ms \| 170 \|`
- `c3_camera/README.md:299` · `b` · 산출물 없음
  - 현재: `\| 960x540 raw @20 \| 778 \| 포화 \| >100 % \| 250 ms \| 284 \|`
- `c3_camera/README.md:300` · `b` · 산출물 없음
  - 현재: `\| 1920x1080 raw @20 \| 3111 \| 포화 \| >100 % \| 353 ms \| 395 \|`
- `c3_camera/README.md:302` · `b` · 산출물 없음
  - 현재: `점유율만으로는 설명이 안 된다 — 68 %(1080p, 77 ms)가 78 %(기본, 39 ms)보다`
- `c3_camera/README.md:306` · `c` · 예시/유도값이 측정치로 서술됨
  - 현재: `컬러 지연 ≈ 30 ms          (고정: 센서 readout + ISP + 인코딩)`
- `c3_camera/README.md:311` · `b` · 산출물 없음
  - 현재: `검산: 640x360(67 kB) → 30 + 6 = 36 vs 실측 35.0. 960x540(131 kB) → 30 + 11 = 41`
- `c3_camera/README.md:318` · `b` · 산출물 없음
  - 현재: `>    (p95−p50 < 1 ms). 그 위는 fps와 지연을 맞바꾸는 구간.`
- `c3_camera/README.md:325` · `b` · 산출물 없음
  - 현재: `전부 실제로 측정한 값이다 (예측 아님):`
- `c3_camera/README.md:329` · `b` · 산출물 없음
  - 현재: `\| **균형 (기본값)** \| *(없음)* \| 15.0 fps, 컬러 **39 ms**, drop 0 % \|`
- `c3_camera/README.md:330` · `b` · 산출물 없음
  - 현재: `\| **최저 지연 + 지터 없음** \| `--isp-scale 1/3 --fps 20 --depth-size 480x270` \| 20.0 fps, 컬러 **35.0 ms** (p95 35.8, max 36.1), drop 0 %, 천장 58 % \|`
- `c3_camera/README.md:331` · `b` · 산출물 없음
  - 현재: `\| **최고 fps** \| `--fps 30 --depth-size 480x270` \| **28.5 fps**, 84 ms, drop 5 %, 천장 99 % \|`
- `c3_camera/README.md:332` · `b` · 산출물 없음
  - 현재: `\| **최대 화질 (풀 1080p)** \| `--isp-scale none --fps 12 --depth-size 480x270` \| 12.0 fps, **77 ms**, drop 0 %, 천장 68 % \|`
- `c3_camera/README.md:333` · `b` · 산출물 없음
  - 현재: `\| **무손실 픽셀** \| `--color-encode none --fps 9` \| 약 9 fps, NV12 원본 \|`
- `c3_camera/README.md:349` · `b` · 산출물 없음
  - 현재: `- **연결에 약 12초** 걸린다. DepthAI가 PoE로 펌웨어를 올리고 부팅하는 시간이고`
- `c3_camera/README.md:356` · `b` · 산출물 없음
  - 현재: `- **공기 중 실내 테스트에서 depth valid ≈ 21%** 였다. 텍스처 없는 벽/역광 때문이고`
- `c3_camera/README.md:405` · `b` · 산출물 없음
  - 현재: `--color-encode mjpeg \| none        기본 mjpeg (실측 약 1/6). none은 무손실 대신 약 9 fps`
- `c3_camera/README.md:452` · `c` · 예시/유도값이 측정치로 서술됨
  - 현재: `대략 **150 MB/분 = 9 GB/시간** 수준.`
- `c3_camera/README.md:534` · `c` · 예시/유도값이 측정치로 서술됨
  - 현재: `- 공장 캘리브레이션은 **공기 중** 캘리브레이션이다. 물속에서는 포트(평면/돔)가`
- `c3_camera/README_DATASET.md:6` · `b` · 산출물 없음
  - 현재: `카메라 직결 자체(지연 3000 ms → 35 ms)와 그 실측 근거는`
- `c3_camera/README_DATASET.md:142` · `c` · 예시/유도값이 측정치로 서술됨
  - 현재: `\| 용량 \| 약 0.5 GB/시간 \| 약 21 GB/시간 \|`
- `c3_camera/README_DATASET.md:146` · `b` · 산출물 없음
  - 현재: `기본 설정 480x270 컬러+depth + 640x400 스테레오 @8 fps, 20초 실측:`
- `c3_camera/README_DATASET.md:153` · `b` · 산출물 없음
  - 현재: `\| 디스크 \| 약 11–13 MB/s (약 21 GB/시간) \|`
- `c3_camera/README_DATASET.md:154` · `b` · 산출물 없음
  - 현재: `\| 지연 \| 컬러 p50 88 ms, depth 89 ms \|`
- `c3_camera/README_DATASET.md:155` · `b` · 산출물 없음
  - 현재: `\| 카메라 IMU \| 200 Hz 요청 → **약 130 Hz 실측** (아래 참고) \|`
- `c3_camera/README_DATASET.md:165` · `c` · 예시/유도값이 측정치로 서술됨
  - 현재: `\| 640x360 \| 8.7 \|`
- `c3_camera/README_DATASET.md:166` · `c` · 예시/유도값이 측정치로 서술됨
  - 현재: `\| 960x540 \| 4.9 \|`
- `c3_camera/README_DATASET.md:167` · `c` · 예시/유도값이 측정치로 서술됨
  - 현재: ``
- `c3_camera/README_DATASET.md:168` · `c` · 예시/유도값이 측정치로 서술됨
  - 현재: `더 높은 fps가 필요하면 `--streams color,depth`로 스테레오를 빼거나(14 fps),`
- `c3_camera/README_DATASET.md:229` · `b` · 산출물 없음
  - 현재: `(CLOCK_MONOTONIC)이고 8 마이크로초 이내로 일치합니다.** 따라서 카메라 프레임,`
- `c3_camera/README_DATASET.md:255` · `c` · 예시/유도값이 측정치로 서술됨
  - 현재: `**VIO에는 이걸 쓰는 게 맞습니다.** ROV의 Navigator IMU는 수십 cm 떨어져 있고`
- `c3_camera/README_DATASET.md:263` · `b` · 산출물 없음
  - 현재: `\| accel+gyro @200 \| 194 Hz \|`
- `c3_camera/README_DATASET.md:264` · `b` · 산출물 없음
  - 현재: `\| accel+gyro @200 **+ rotvec @100** \| 97 Hz (전부 반토막) \|`
- `c3_camera/README_DATASET.md:265` · `b` · 산출물 없음
  - 현재: `\| accel+gyro @200 **+ mag @100** \| 97 Hz (전부 반토막) \|`
- `c3_camera/README_DATASET.md:266` · `b` · 산출물 없음
  - 현재: `\| accel+gyro @200 **+ rotvec @200** \| **194 Hz** (손실 없음) \|`
- `c3_camera/README_DATASET.md:267` · `b` · 산출물 없음
  - 현재: ``
- `c3_camera/README_DATASET.md:276` · `b` · 산출물 없음
  - 현재: `\| 1 \| 130 Hz 수신, **35% 손실** \|`
- `c3_camera/TESTING.md:107` · `b` · 산출물 없음
  - 현재: `과대평가한다 (실측 8.25:1). 실제 프론티어는 `color_kb_frame` 컬럼으로만 나온다.`
- `c3_camera/TESTING.md:130` · `b` · 산출물 없음
  - 현재: `#      "960x540 mjpeg @15 + depth 640x360" = 71.4 Mbit/s (78%), p50 39 ms, p95 80 ms.`
- `c3_camera/TESTING.md:131` · `b` · 산출물 없음
  - 현재: `#      @20 행은 인코더 표에서 19.1 fps / drop 5%.`
- `c3_camera/TESTING.md:273` · `c` · 예시/유도값이 측정치로 서술됨
  - 현재: `* **`--mjpeg-quality`가 범위 검증되지 않는다.** 150이나 -5도 통과해서 디바이스 시작`
- `c3_camera/TESTING.md:282` · `b` · 산출물 없음
  - 현재: `6세션에서 4.05:1 ~ 6.35:1로 흔들렸다), 탁하거나 어두운 물은 고주파 성분이 적어`
- `c3_camera/TESTING.md:444` · `c` · 예시/유도값이 측정치로 서술됨
  - 현재: `bias가 ±dZ/2를 통째로 짊어진다. 2 m/400p에서 dZ = 141 mm — 피팅하려는 원점`
- `c3_camera/TESTING.md:472` · `c` · 예시/유도값이 측정치로 서술됨
  - 현재: `실제로는 ~37.5 mm 떨어져 있어서, 185 mm에서 그 병진은 104 px(폭의 16.3%)로`
- `c3_camera/TESTING.md:473` · `c` · 예시/유도값이 측정치로 서술됨
  - 현재: `예측된 밴드 94 px보다 **크다.** 밴드는 잘리는 게 아니라 **이동**하므로`
- `c3_camera/TESTING.md:477` · `c` · 예시/유도값이 측정치로 서술됨
  - 현재: `1 m 실측 예: resid 10.01, temporal 5.19, fixed 8.56 — 진짜 센서 노이즈 1 mm에`
- `c3_camera/TESTING.md:480` · `c` · 예시/유도값이 측정치로 서술됨
  - 현재: `정확히 0으로 읽힐 수 있다.`
- `c3_camera/TESTING.md:499` · `b` · 산출물 없음
  - 현재: `> 틀렸다.** 실측(dataset_20260803_105221)에서 컬러 MJPEG는 6.8 Mbit/s로 링크의`
- `c3_camera/TESTING.md:501` · `b` · 산출물 없음
  - 현재: `> 만드는 완벽한 코덱도 천장의 7.4%만 회수하고, 전체가 이미 53%라 **fps 이득은`
- `c3_camera/TESTING.md:543` · `b` · 산출물 없음
  - 현재: `> **정상**이고 랭킹에서는 `--target-fps` 미달로 실격된다. reference를 만드는 게`
- `c3_camera/TESTING.md:556` · `c` · 예시/유도값이 측정치로 서술됨
  - 현재: `## Step 3 (선택) — 1080p 팔 (4 설정, ~2분)`
- `c3_camera/TESTING.md:571` · `c` · 예시/유도값이 측정치로 서술됨
  - 현재: `> `--duration 20 --reference-frames 15`가 필수다. 1080p raw는 링크에서 ~3.8 fps라`
- `c3_camera/TESTING.md:596` · `b` · 산출물 없음
  - 현재: `검출 수가 **늘어난다** (실측: 11 → 13). `tags_false`가 0이 아니면 그 rate는`
- `c3_camera/TESTING.md:614` · `c` · 예시/유도값이 측정치로 서술됨
  - 현재: `게인이 너무 높다. 임계값 0.7은 미보정이라 정적 리그에서도 센서 노이즈 12 DN이면`
- `c3_camera/TESTING.md:633` · `b` · 산출물 없음
  - 현재: `PSNR **45.60 dB** / SSIM **0.9962** / ORB inlier **0.9156**이 나온다. 즉 h26x`
- `c3_camera/TESTING.md:636` · `b` · 산출물 없음
  - 현재: `MJPEG은 같은 실험에서 true loss와 0.1 dB 이내로 일치해 이 패널티가 없다.`
- `c3_camera/TESTING.md:655` · `c` · 예시/유도값이 측정치로 서술됨
  - 현재: `빈 문자열이고, `dataset.py:341-344`의 첫 I-frame 게이트가 안 열린다. 이 스크립트는`
- `c3_camera/TESTING.md:695` · `c` · 예시/유도값이 측정치로 서술됨
  - 현재: `쓴다. 컬러를 960x540 MJPEG q90으로 올리면 21.5 + 41.5 = 62.9 Mbit/s = 천장의`
- `claude.md:6` · `b` · 산출물 없음
  - 현재: `Physical setup: ZED2 camera in air, ~0.17 m above the water surface, looking down through the (flat, calm) water surface at AprilTags tiled across the pool floor. Pool ≈ 4.877 m × 2.438 m × 1.143 m (45 in) deep. Water l…`
- `claude.md:17` · `b` · 산출물 없음
  - 현재: `Measured (correct tag size 0.170, mode none): R ≈ 0.80 at d_water ≈ 1.0 m vs predicted ≈ 0.79. Intrinsics are fine (ZED factory fx=fy≈534.88 HD720; no hidden extra scale, else R would deviate from physics).`
- `claude.md:21` · `b` · 산출물 없음
  - 현재: `M5 synthetic self-test passes (--refractive-self-test): rms ≈ 1e-5 px, trans ≈ 0 mm, rot ≈ 2e-4 deg.`
- `claude.md:22` · `b` · 산출물 없음
  - 현재: `M6 real data: D_true ≈ 1.2 m → none ≈ 1.0 m (R ≈ 0.83), refractive ≈ 1.2 m (R ≈ 1.0). PASS.`
- `claude.md:24` · `b` · 산출물 없음
  - 현재: `Refractive solver made real-time. The first refractive implementation was a per-tag finite-difference LM with a 48-iter inner bisection → fps collapsed (~10 → ~1 when tags appear, scaling with per-frame tag count; note:…`
- `claude.md:26` · `b` · 산출물 없음
  - 현재: `--refractive-regression-check: max_trans_delta ≈ 0.0003 mm, max_rot_delta ≈ 0.00037 deg (bounds 0.1 mm / 0.01 deg). PASS.`

## 패턴 — 어떤 종류의 수치가 틀렸나

1. **도구와 함께 쓰인 docstring 숫자.** 도구를 만들면서 "이 정도 나온다"고 적은 값이
   나중에 실측으로 인용된다. `+28 mm`, ORB inlier `1.000→0.827`, h26x 상한 `PSNR 45.60`이
   전부 이 유형이다. 산출물 디렉터리(`depth_accuracy/`, `encode_quality/`, `bench/`)가
   아예 존재하지 않는다.
2. **저널이 인용하면서 정밀도가 올라간다.** 원문 `PSNR 45.6 / ORB 0.916`이 저널에서
   `45.60 / 0.9156`이 된다. 유효숫자가 늘어나는 것은 출처 냄새다.
3. **헤지가 인용 과정에서 탈락한다.** README의 `약 3000 ms`가 저널에서 단정형
   `3000 ms → 38 ms`가 된다. 같은 값이 다른 문서에서 짝이 달라진다(38 vs 35 ms).
4. **우리 가정이 우리 산출물에 찍혀 나가 사실이 된다.** `c3_collect.py:330`의
   `IN-AIR` 문자열이 모든 `calibration.json`의 `note`가 되어, 디바이스가 말한 것처럼 보였다.
5. **절반만 맞는 주장.** IMU 배칭(`batch=10 → 199.5 Hz`는 산출물로 확인, `batch=1 → 130 Hz`는
   산출물 없음)처럼 한 문장 안에서 출처가 갈린다.

## 앞으로의 규칙

> 문서에 측정치를 쓸 때는 **산출물 경로를 같이 쓴다.** 경로를 못 쓰겠으면 그 수치는
> 측정치가 아니다 — `[예측]`, `[유도]`, `[스펙]` 중 하나로 표기한다.

적용 예:

```
나쁨:  depth bias +28 mm
좋음:  depth bias +28 mm  (c3_camera/depth_accuracy/rungs.csv, row tape=2000)
좋음:  depth bias ±dZ/2 = ±70 mm [예측: c3_depth_accuracy.py:88-99, 미실측]
```
