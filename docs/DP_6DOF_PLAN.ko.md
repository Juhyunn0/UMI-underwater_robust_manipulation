# DP 6-DoF 변형 — roll/pitch 추종 (`pos_rpy_width` + `engage.attitude_axes`) 운용 계획 (2026-09-26)

> 이 문서는 운용자용이다. 무엇이 바뀌는지, 어떻게 켜는지, **어떤 순서로 처음 날리는지**, 어떤 게이트가 막는지,
> 어떤 숫자가 [예측]인지와 그 근거, 그리고 설계가 사용자에게 넘긴 결정 셋. 코드 경계는
> `rov_gui/README.md` "6-DoF 변형" 문단, 미해결 목록은 `KNOWN_ISSUES.md` 2026-09-26 항목, 설계 원본은 세션
> 스크래치패드 `design_final.json`(D1–D13). **실기 0런 — 아래 모든 자세 수치는 산출물 경로가 없는 한 [예측]이다.**

## 0. 한 줄 요약

정책이 `[dx, dy, dz, dyaw, droll, dpitch, width]` 7-dim을 내고(5-dim의 열 0:4·6과 비트 동일 + roll/pitch 두 열),
저수준 NMPC가 roll/pitch 참조를 추종하며, 그 K/M 토크를 ArduSub MANUAL_CONTROL의 **확장축**(s = pitch, t = roll,
`enabled_extensions = 0b11`)으로 보낸다. **기본값은 전부 OFF**이고 OFF면 4-DoF 경로(제어 출력·xref·MAVLink 프레임·
resolved config)는 이전과 **바이트 동일**(테스트가 고정). 기록 파일만 schema 15 → 16으로 항상 붙는 꼬리 열/키를 얻는다.

## 1. 변형이 무엇인가 — 세 스위치와 그 조합

| 스위치 | 위치 | 기본 | 의미 |
|---|---|---|---|
| `policy.action_repr` | `config/hw_mpc.yaml` | `pos_yaw_width` | 이 미션이 나는 체크포인트 계약. ARM 때 ckpt의 `shape_meta.action.action_repr`와 **같아야** 하고 `POLICY_ACTION_REPRS_FLYABLE = (pos_yaw_width, pos_rpy_width)` 안이어야 한다. 5-dim ckpt에 `pos_rpy_width`, 7-dim ckpt에 `pos_yaw_width` → 둘 다 arm 거부("action_repr mismatch"). |
| `policy.attitude_track` | 〃 | `false` | `false` = **dropped-and-logged**: 7-dim ckpt여도 roll/pitch 열은 버리고(앵커 자세로 레벨링) `plans.jsonl rp_raw`(최상위, 디코드된 절대 roll/pitch (2,K))·`dropped_rp_deg`로 기록만 (`raw.rp`는 tracked 때만). `true` = compose가 클립한 절대 roll/pitch를 `PlanMsg.rp`로 실어 NMPC가 추종. |
| `engage.attitude_axes.enabled` | 〃 | `false` | NMPC의 K/M을 선으로 보낼지. `false`면 allocation에서 예전처럼 버린다(`roll = pitch = 0`). `true`면 `wrench_to_axes(..., attitude=True)` → `s/t` 송신. 이것만 켜고 정책을 안 날리면 "능동 레벨링" 런. |

조합의 진리표(`workers._policy_refusal` + engage 게이트, 전부 meta에 찍힌다):

* `attitude_track: true` ⇒ `action_repr == pos_rpy_width` **AND** `attitude_axes.enabled` **AND** 비행 모드 MANUAL **AND**
  LOW ∈ {`mpc`, `mpc_tuned`}(`dobmpc*`는 `attitude_axes.dobmpc_allowed: true`일 때만; `pid`/`rl`/`mpcc`는 K/M 경로가 없어 거부)
  **AND** 마운트 키(`yaw_axis_*`) + `rpy_convention` 일치 **AND** 펌웨어 게이트 4개(§4; 프로브 JSON 파싱 + `sign_proven`) **AND** `rp_residual_deg ≤ 10°` [예측].
  하나라도 빠지면 **arm 거부** — 조용한 강등은 없다.
* `attitude_axes.enabled` + STABILIZE ⇒ engage **거부**. STABILIZE에서 s/t는 lean-angle 목표라 NMPC의 토크 모델이 각도 서보로
  재해석된다(캐스케이드는 이번 컷에 구현하지 않음, §8 트림 (b)).
* 7-dim ckpt + `attitude_track: false` = 모든 7-dim 망의 **의무 첫 비행**(§3 ④).

## 2. 어떻게 선택하나 (운용 절차)

1. 체크포인트: 학습 런 `data/<날짜>/<stamp>_train_umi_depth_7d_<exp>`의 `checkpoints/selected.ckpt`를 trajectory 패널 ckpt 칸에서
   고른다(sha1은 ARM에 핀). ckpt는 `shape_meta.action {shape [7], action_repr pos_rpy_width, yaw_axis_cam_tilt_deg 43.3,
   yaw_axis_R_frd_cam, rpy_convention {order: zyx, frame: body_frd_via_R_bt, columns: [dx,dy,dz,dyaw,droll,dpitch,width]}}`를
   선언해야 한다. torch 없이 배선만 볼 땐 ckpt 칸에 `stub7`(7-dim 수평) 또는 `stub_rp`(pitch 5° 램프 [예측]).
2. `config/hw_mpc.yaml`: `policy.action_repr: pos_rpy_width`. 첫 비행은 `attitude_track: false` 그대로.
3. K/M을 보낼 단계에서만 `engage.attitude_axes.enabled: true` + `probe`/`sign_probe` 산출물 경로 기입(§4). engage 게이트는
   경로 존재가 아니라 **내용을 파싱**한다: `probe`는 `tool == rov_gui.tools.attitude_axes_probe`, `pass: true`,
   `mavlink_wire_version "2.0"`, `firmware_version`이 기체의 AUTOPILOT_VERSION과 같아야 하고, `sign_probe`는 `sign_proven: true`를
   실어야 한다. 부호 프로브가 아직 없으면 `engage.attitude_axes.require_sign_probe: false`(기본 `true`)로 런을 허용할 수 있는데,
   그때는 `cap_roll/cap_pitch` 설정값과 무관하게 **`first_water_caps [0.1, 0.15]`가 강제**되고 meta에 `sign_proven: false` +
   `caps_in_force`가 찍힌다 — 부호가 증명되지 않은 모든 런은 첫 물 캡으로 난다.
4. `./c3 gui --source hw --allow-command --mpc --fstereo --policy`, HIGH `Diffusion Policy`, LOW `MPC` 또는 `MPC_Tuned`, MANUAL.

## 3. 첫 비행 순서 — 추종 비행 전에 반드시 앞서는 두 종류

설계가 요구하는 순서(설계 needs_user_decision (a)는 ①–③과 ④의 선후만 남겼다, §7):

| 단계 | 무엇 | 정책 | K/M 송신 | 산출물(없으면 다음 단계 금지) |
|---|---|---|---|---|
| ① 벤치 프로브 | `python -m rov_gui.tools.attitude_axes_probe` — **DISARMED**, s=+300 1 s → t=+300 1 s @10 Hz, RC_CHANNELS chan1/chan2 전후 비교; preflight 먼저, armed면 거부 | 없음 | 확장 프레임만 | `data/<YYYYMMDD>/<run>_probe/attitude_axes_probe.json` (pass/fail, firmware_version, wire version) → `engage.attitude_axes.probe`; engage가 내용을 파싱해 `tool`·`pass: true`·`mavlink_wire_version "2.0"`·`firmware_version` 일치를 요구 |
| ② 수중 부호 프로브 | armed, 추종자 disengaged, roll +0.10 1 s → pitch +0.10 1 s, ATTITUDE + SERVO_OUTPUT_RAW 기록, abort \|rp\| > 10° 또는 2 s | 없음 | 캡 0.1/0.15 | `engage.attitude_axes.sign_probe` 경로 + `sign {roll, pitch}` 확정; JSON에 `sign_proven: true`가 있어야 게이트 통과(`require_sign_probe: true` 기본). **미션 종류는 이번 컷에 없다**(§8 (c)) — 수동으로 만든다. 아직 없으면 `require_sign_probe: false`로 허용되지만 `first_water_caps` 강제 + meta `sign_proven: false`·`caps_in_force` |
| ③ plain mpc 자세 STEP | 정책 없이 station에서 roll/pitch 참조 스텝(0 → 10~15°)·램프(0.35 rad/s) — 0914 yaw 스텝 시험의 자세판 | 없음 | ON (첫 물 캡) | 정상 M_cmd vs θ·k로 `roll_nm/pitch_nm` 보정, 오버슈트·정착·감쇠 → `attitude_q_scale`/`rp_ref_filter` 결정 |
| ④ dropped-and-logged 7-dim 비행 | `action_repr: pos_rpy_width`, `attitude_track: false`, `attitude_axes.enabled: false` | 7-dim | **없음** | `plans.jsonl rp_raw`, `dropped_rp_deg` 분포; tagnav 수용률 vs 측정 pitch → `rp_max_deg` 선택 근거 |
| ⑤ 능동 레벨링 | `attitude_axes.enabled: true`, 정책 없음 또는 5-dim | 선택 | ON | `rroll_deg/rpitch_deg = 0` 추종 품질, meta `run.attitude_axes.sat_ticks`, 데드밴드 리밋사이클 유무 |
| ⑥ 추종 비행 | `attitude_track: true` | 7-dim | ON | 첫 런은 스모크(§1 진리표 전부 통과) |

④는 하드웨어 액추에이션 체인과 무관하므로 ①–③보다 먼저 돌려도 된다. **⑥은 ①·②·③·④가 전부 있어야** 한다.

비행 전 오프라인(하드웨어 시간 0):
* `dp_policy_offline.py gate --action-repr pos_rpy_width` — 데모 라벨의 |droll|/|dpitch| 백분위와 자세 게이트 통과율.
  첫 산출물(0901 육상 데모 1721 윈도): `data/20260926/0926_175145_offline/gate_pos_rpy_width.log` — |dpitch| p50/p90/p99/p99.9/max
  1.17/5.53/13.07/18.31/27.12°, |droll| 0.70/2.56/5.39/8.17/10.78°, 피크 |p|,|q| p90 0.453 rad/s(두 축 합산 최대, 캡 0.35 초과 → 200 ms 격자에서 2/1721
  청크 accept→clip), rp_mag 거부 0. 즉 이 데이터의 pitch 열은 SLAM 지터가 아니고(p90 5.5°), 20° 클립은 p99.9 아래다.
* `dp_policy_offline.py replay --val-only --ckpt <7-dim>` — held-out에 roll/pitch RMS·zero-predictor·corr 행이 추가된다.
  회귀 게이트: 열 [:, :4]의 pos_rmse/yaw_rms/yaw_corr이 5-dim 베이스라인(D/A 21.0/21.1 mm, corr .67/.65; `data/20260915/0915_131900_heldout_D_vs_A/`) 잡음 안, rp 열은
  zero-predictor 미만 + corr > 0.5 [예측]. **`model` 가중치로만 선택**(EMA/BN 결함).
* `policy_session_replay.py <수중 런> --ckpt D7=<7-dim> [--attitude-track]` — 기존 수중 관측에 7-dim을 먹여 정책이 요청한
  기울기(요청 − 측정 roll/pitch RMS)를 본다. 수중 OOD는 yaw에서 이미 좌편향(2026-09-08)이었다; 두 축이 더 생겼다.

## 4. 게이트 — 무엇이 막는가

**펌웨어/링크(engage, D8)**: (1) `master.mavlink20()` — v1 dialect면 확장 인자가 TypeError; (2) `AUTOPILOT_VERSION`
`flight_sw_version ≥ engage.attitude_axes.min_firmware 4.1.2`(Sub 4.1.2 22-Feb-2024 "additional joystick axis for roll/pitch";
4.1.0/4.1.1은 s/t를 **조용히 무시** [스펙: ArduSub ReleaseNotes Sub 4.1.2, joystick.cpp 원문 미열람]), 버전 미수신("")이면 거부;
(3) `require_probe: true`면 `probe` 산출물을 **파싱**해 `tool == rov_gui.tools.attitude_axes_probe`, `pass: true`,
`mavlink_wire_version "2.0"`, `firmware_version == 기체 AUTOPILOT_VERSION`을 요구(경로만 있고 내용이 다르면 거부);
(4) `require_sign_probe: true`(기본)면 `sign_probe` 산출물이 `sign_proven: true`를 실어야 한다. `require_sign_probe: false`면 부호
미증명 상태로 arm을 허용하되 **`first_water_caps [0.1, 0.15]`가 강제**되고 meta `run.attitude_axes`에 `sign_proven: false`·
`caps_in_force`가 찍힌다 — 유효 캡은 부호가 증명되지 않은 한 언제나 첫 물 캡이다.
텔레메트리 `firmware_version / mavlink_wire_version / attitude_axes_enabled / attitude_axes_degraded / rc_chan_raw`가 meta
`run.attitude_axes`에 찍힌다.

**캡 계층(D3)**, 서로 다른 메커니즘이라 버그 하나에 살아남는다:

| 층 | 어디 | 값 | 동작 |
|---|---|---|---|
| T0 | 라벨 공간 | normalizer range + DDIM clip | 망은 데모 범위 밖을 못 낸다 |
| T1 | `compose_plan` | `rp_max_deg` 20 [예측] | 디코드된 절대 roll/pitch를 축별 **클립**, `rp_clipped_deg/n` 기록, `p_tcp`는 그대로(정책 요청), 선체 자세·`p_body`만 재유도 |
| T2 | `PlanFilter` | `rp_reject_deg` 30 / `pq_max_rad_s` 0.35 / `rp_jump_max_deg` 5 [예측] | 하드 거부 / `need`(dilation) 항 / soft(기록) |
| T3 | `set_path_plan_ned` | `rp_reject_rad` | 스키마 위반이면 ValueError |
| T4 | 선 | `cap_roll 0.2 / cap_pitch 0.3`, 첫 물 `first_water_caps [0.1, 0.15]` [예측] — 부호 미증명(`sign_proven: false`)이면 설정값과 무관하게 첫 물 캡이 강제(meta `caps_in_force`) | `clip(u[3]/roll_nm)`, `clip(u[4]/pitch_nm)` |
| — | NMPC | soft \|φ\|,\|θ\| ≤ 1.2 rad | 변경 없음 |

**인터록(D9, 측정 상태 기준)**: (i) `div_max_rp_deg` 15 [예측], 0.5 s 디바운스 → 정책을 **수평 DP 홀드로 강등**(K/M은 계속 흘러
능동 레벨링; disengage가 아닌 이유: 기운 선체에서 레벨링 토크를 잃는 것이 더 나쁨); (ii) `abort_deg` 35 [예측] 0.25 s → disengage
(중립 = s=t=0 → 수동 복원); (iii) sat-ineffective: 축이 캡에 `sat_ineffective_s` 1 s 붙어 있고 해당 자세 오차가 **줄지 않으면**
→ disengage(부호 반전·s/t 무시·포화 무진행을 모두 잡는다; "단조 증가" 조건은 상수 포화 오차를 못 잡아 기각); (iv) ARM 때
`rp_residual_deg > 10°` → 거부; (v) 확장 프레임 TypeError → 6-인자 프레임(K/M 0)으로 재송신 + `attitude_axes_degraded` → 다음 틱
disengage. 조종자에게 roll/pitch 입력은 없다 — 자세 명령을 거스르는 수단은 disengage / E-STOP뿐(MANUAL 전용이라 중립 = 토크 0).

**홀드 전환(D5)**: 11곳의 `set_target_ned` 호출자 대신 `HwDobMpc` 안에서 `_rp_hold`가 마지막 자세에서 `pq_max`로 수평까지
램프한다(스텝이면 Q=80 토크 스파이크). 홀드 꼬리는 종점 자세를 유지(T1로 유계).

## 5. [예측] 숫자와 그 근거

| 값 | 근거 | 승격 조건 |
|---|---|---|
| `roll_nm 13.2 / pitch_nm 7.2` N·m | [유도] heave_n 60 N [예측] = 수직 4×15 N × 레버암 y ±0.22 / x ±0.12 m (bluerov_heavy.xml) | ③ 스텝 시험의 M_cmd vs θ·k |
| `cap_roll 0.2 / cap_pitch 0.3`(첫 물 0.1/0.15) | cap×gain 2.6/2.2 N·m ≤ U_MAX 8; 첫 물 캡은 복원 ≥ 1.35 N·m/rad [유도] 대비 전복 불가 | ②·③ |
| `rp_max_deg 20` | (a) heave-trim surge 잔차 5.81·sin 20° ≈ 2.0 N [유도] ≈ T200 무릎, (b) C3 43.3° 하향이 20° nose-up에서 바닥 태그 유지 [예측], (c) 홀드 모멘트 ≤ pitch 캡, (d) T(η) 특이점 1.2 rad에서 멀다; 라벨 p99.9 18.3° 아래(§3 gate 산출물) | ④ tagnav 수용률 vs pitch |
| `rp_reject_deg 30`, `pq_max_rad_s 0.35`, `rp_jump_max_deg 5` | 클립 위 한 단, yaw `r_max 0.5`의 70 %, `yaw_jump_max_deg 8`의 자세판 | gate/held-out 거부율·pq 피크 분포 |
| `anchor_leash_rp_deg 3`, `div_max_rp_deg 15`, `abort_deg 35`, `sat_ineffective_s 1.0` | 2026-09-14 yaw 흔들림 기제(0.5 s 낡은 측정에 재앵커)의 자세판; 시뮬 S4 div_rp 발동 [예측 1–2 s] → 실측 0.45 s (mpc·dobmpc, `bluerov2_mujoco_marinegym/recordings/20260926/attitude_hold_211917/results.csv`, `dropped_*` 행; 참조 20° = `rp_max_deg`, 임계 15° = `div_max_rp_deg`; 0.45 s = 임계 초과 연속 10 샘플 = 스테이션 0.5 s 디바운스, 선체는 20°에 접근하지 않음; 이전 174607의 dropped 행은 임계=참조 크기라 퇴화, 인용 금지) | 시뮬 S4/S5 + ⑤ |
| `RP_RANGE_MIN_RAD 0.10`(학습 빌드 가드) | 열 범위가 이보다 작으면 near-zero 타깃 | `rp_label_stats.txt` |

회전 플랜트는 전부 placeholder(감쇠 0.07, 부가질량 0.12, ZG 0.01 vs coBM 0.01625, 페이로드 pitch 모멘트 ~1.4 N·m [스펙] 미모델)
— 그래서 ③이 정책 비행보다 먼저다. 첫 물 pitch 캡 1.08 N·m < ~1.4 N·m라 능동 레벨링이 M 포화로 상수 오차를 안을 수 있다
(sat-ineffective가 잡아야 함). pitch 데드밴드 무릎 ≈ 0.096 × 7.2 ≈ 0.7 N·m [유도] ≈ 20° 홀드 모멘트 → yaw 디더의 자세판 위험,
시뮬 S5가 첫 증거. 솔버 `U_MAX[3:5]` 8 N·m > 선 캡 → plain mpc는 큰 과도에서 지연(EAOB는 `note_applied`로 진실을 봄), meta
`controller.attitude_ref.u_max_wire_nm`에 기록.

## 6. 기록 경계 (합산 금지 기준)

* meta `schema_version` 16; 미션 CSV 끝 `rroll_deg, rpitch_deg, ax_roll, ax_pitch, rp_track`(OFF/observe면 nan/0; 참조 열은
  측정 `roll_deg/pitch_deg`와 같은 FLU 부호); `policy_plan.csv` 헤더는 항상 `…,reason,roll_deg,pitch_deg`(rp 없으면 nan);
  plans.jsonl `rp_tracked` 항상, `rp_raw`(7-dim이면 항상: 디코드된 원시 자세 (2,K)), `raw.rp`(tracked 때만: 실린 (2,J)), `anchor.rp_meas / anchor.rp_used / rp_clipped_deg / rp_clipped_n`은 있을 때.
* meta `run.attitude_axes {enabled, transport, roll_nm, pitch_nm, cap_*, slew_per_s, sign, sign_proven, caps_in_force,
  mavlink_wire_version, firmware_version, probe, sign_probe, rc_trim_at_engage, enabled_extensions_sent, degraded_at, sat_ticks}`,
  `trajectory.attitude_track`, `policy.action_repr`(핀), `controller.attitude_ref {tracked, source plan|hold_ramp|level,
  rate_source none|stitcher_rate_T_inv, …}` — `source`는 meta 기록 시점의 값(추종 런의 닫는 meta도 `level`로 끝날 수 있다), 풀링 키는
  `tracked`; `rate_source`는 스테이션(`stitcher_rate_T_inv`: 스티처가 샘플한 오일러 rate를 T⁻¹로)과 시뮬 `attitude_meta`
  (`fd_horizon_T_inv`: 샘플한 호라이즌 위 np.gradient)가 **다른 방법·다른 라벨**이라 하나의 rate_source로 합산 금지;
  `controller.allocation.attitude`. 카운터: 포화 tick은 meta `run.attitude_axes.sat_ticks`, 정책 쪽은 `policy.run`의
  `reject_rp_mag / clip_rp_rate / div_rp_max_seen_deg`.
* **합산 금지**: `attitude_track`, `attitude_axes.enabled/transport`, `run.attitude_axes.sign_proven/caps_in_force`(유효 캡이 다른 런), `action_repr`, 비행 모드, `controller.allocation.attitude`
  (dobmpc `w_hat[3:5]` 의미가 바뀜), CSV `rp_track`이 다른 런끼리.
* 외부 판독기 주의: 열 수를 세거나 바이트 비교하면 깨진다 — 이 리포의 이름 기반 판독기만 보장.

## 7. 사용자 결정이 필요한 것 셋 (설계 `needs_user_decision`)

1. **첫 비행 순서**: (a) 정책 없는 plain `mpc` 자세 STEP(①·② 뒤)이 먼저인가, (b) K/M 미송신의 dropped-and-logged 7-dim 비행이
   먼저인가. 둘 다 추종 비행보다 앞서지만, 어느 쪽이 먼저냐가 다음 하드웨어 시간을 액추에이션 체인에 쓸지 망에 쓸지를 정한다.
   설계 권고는 (a).
2. **pose10d rot6d ckpt 스모크**: 7-dim 재학습이 끝나기 전에 기존 pose10d 체크포인트를 같은 `attitude_track` 플래그로 날려
   추종자/액추에이션 체인의 무학습 스모크로 쓸지(EMA/BN 결함 + identity-normalizer 거부를 감수해야 하며 arm 허용 목록 변경 필요).
3. **축 이득**: `roll_nm/pitch_nm`를 레버암 유도값(13.2/7.2 [유도])으로 첫 물에 날릴지, yaw 축처럼 시뮬/벤치 보정 산출물을
   먼저 요구할지.

## 8. 이번 컷의 범위 트림 (구현하지 않은 것)

(a) `transport: rc_override`(FALLBACK-A) — config 검증이 "not implemented"로 거부; (b) STABILIZE lean-angle 캐스케이드 —
`attitude_axes.enabled`면 STABILIZE 거부; (c) armed 수중 부호 프로브 **미션 종류** — arm 게이트는 `sign_probe` JSON의
`sign_proven: true`만 요구하므로(`require_sign_probe: false`면 첫 물 캡 강제로 우회) §3 ②는 수동 절차; (d) PID roll/pitch 루프(S11), `run_compare` 자세 시나리오, `plot_policy_map` rp 패널,
teleop 패널 자세 바; (e) 시뮬 검증은 S2 홀드 그리드(heavy, mpc/dobmpc, NONE)까지가 필수; S3–S7은 heavy·NONE 실행
(`bluerov2_mujoco_marinegym/recordings/20260926/attitude_hold_174607` 전 시나리오 — 단 step/ramp/rate/dropped의 정상상태 값은 창이
과도를 포함해 인용 불가 → step `attitude_hold_211813`, ramp/rate `attitude_hold_211921`, dropped `attitude_hold_211917`로 인용);
C/CDW 홀드 셀·heavy_gripper 데드밴드·PID rp 루프는 미실행.
