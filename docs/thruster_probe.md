# 추진기 프로브 블록 — `rov_gui/tools/thruster_probe.py` 운용 문서 (2026-10-10)

> 운용자용. **어떤 rl_pwm 비행보다 먼저** 풀에서 돌리는 다섯 개 시퀀스(A–E)의 목적·순서·안전 목록·산출물 읽는 법·
> 모터가 FAIL일 때 할 일. 근거는 `RL_controller/logs/eval/pwm_cocontraction_analysis_20261010/REPORT.md` §0–3
> (2026-10-10 풀 실패 재분석)와 `RL_controller/README.md` §10.3 항목 4(추진기별 부호 측정 — **한 번도 수행된 적 없음**).
> **실기 0런**: 아래의 모든 임계값은 산출물 경로가 생기기 전까지 [예측]이다. 코드의 상수 옆에도 같은 태그가 붙어 있다.

## 0. 한 줄 요약

```
P=~/miniforge3/envs/rovgui-pose/bin/python
$P -m rov_gui.tools.thruster_probe --dry-run                 # 하드웨어 없이 합성 기체로 전체 논리 점검 (A)
$P -m rov_gui.tools.thruster_probe --seq A                   # 기체: ARMED + MANUAL, 프로펠러 ON, 풀 중앙, 테더 느슨
$P -m rov_gui.tools.thruster_probe --seq E,A,B,D --note "jaw empty, 16.1 V"
$P -m rov_gui.tools.thruster_probe --seq C                   # 개루프 강성 (±1 N·m)
$P -m rov_gui.tools.thruster_probe --seq C --free-decay 30   # 자유감쇠 기록 (중립 펄스만)
```

정책이 날아가는 **바로 그 전송 경로**(RC_CHANNELS_OVERRIDE ch 9..16 → Navigator의 `rl_pwm_override.lua`,
`rov_gui/control/rl_pwm.py`의 PwmGate 전 항목 통과, 스크립트 하트비트 `RLPWM` 신선)로 추진기 하나씩을 구동해
기체 자신의 ATTITUDE 각속도로 판정한다. 시퀀스 A의 산출물 `thruster_sign_probe.json`을 rl_pwm engage 게이트
(`rov_gui/control/rl_pwm.py ThrusterProbeCheck` → `rov_gui/control/thruster_probe_artifact.load_and_check`)가 읽는다.
**`all_pass: true`가 아닌 한 rl_pwm 펄스는 한 발도 나가지 않는다.** 합성 기체(`--dry-run`) 산출물은 `dry_run: true`로
표시되고 게이트가 거부한다.

## 1. 왜 이 도구가 있나

2026-10-10 재분석(REPORT.md §1)의 결론: pwm10_s1의 기수 들림은 학습 세계의 **트림 부호 오류**(−10.6°를 기수-아래로
읽었으나 실제는 기수-위 [측정 data/20260921/0921_182926/mpc_182926.csv]) + **3–5배 과한 정수압 강성**(시뮬 20.2 vs 실기
4–11 N·m/rad [측정 data/2026100{1,3,5}])이고, att7pwm_s0의 전류 폭주는 할당 영공간 n_v = [0,0,0,0,1,1,1,1]의
공동수축이다. 검증자 셋이 로그에서 추진기 부호 반전을 **반증**했지만, 수직 추진기 넷은 믹서 모드에서 heave 합으로만 검증되므로
**하나가 뒤집혀도 4축 정책은 모르고 pwm 정책만 폭주한다**(README §10.3 항목 4). 그 단일 추진기 시험이 이 도구의 A다.
B/C/D/E는 수정 계획 1단계(학습 세계 재구축)의 범위를 실측으로 다시 끌어내는 네 숫자다: n_v 누설, 강성 k, ESC 정지→기동 지연,
안정 트림·순중량.

## 2. 시퀀스와 권장 순서

권장 순서는 **E → A → B → D → C**(가장 조용한 것부터; `--seq E,A,B,D` 뒤에 `--seq C`를 따로). 모든 시퀀스는 스텝 사이와
종료·예외·Ctrl-C·중단 조건에서 **중립(8채널 1500 µs)**을 보낸다. A·B(hold 없음)·C의 스텝 사이 중립은 **잠잠해질 때까지 연장**된다
(자유 자세 주기 2.6–3.5 s [측정 REPORT §1], yaw 감쇠 τ ≈ 2 s라 3 s만으로는 직전 펄스의 운동이 다음 창에 남는다; 연장 시간은
스텝 meta `settle_extra_s`와 `detail.settle_after`에 적힌다).

| 시퀀스 | 무엇을 | 펄스 | 판정/출력 | 산출물 |
|---|---|---|---|---|
| **E** rest trim | 턱 **비운** 채 60 s 중립 펄스 | 1500 ×8 | roll/pitch 평균·표준편차(NED, +pitch = 기수 **위**), 깊이 드리프트(순중량 부호), 전류(호텔 부하) | `rest_trim.json` |
| **A** 부호 프로브 | 모터 m=1..8: +100 µs 1.0 s → 중립 3 s(+자이로가 0.5 s간 2 °/s 아래로 잠잠해질 때까지, 최대 10 s 추가 [예측]) → −100 µs 1.0 s → 같은 중립 | ±100 µs 한 개씩 | 지배축(export의 B_flu·sigma로 **실행 시 계산**: 수직 → roll, 수평 → yaw) 평균 각속도 **변화**(첫 비중립 SERVO_OUTPUT_RAW 보고 후 0.15–0.55 s 창 평균 − 명령 전 0.5 s 기준선) | `thruster_sign_probe.json` |
| **B** n_v 누설 | 수직 넷 공통모드 c_v ∈ {0,+100,+200,+300} µs, 5 s씩, 두 고도(운용자가 옮김) | T5..T8 동일 | 깊이율·pitch·roll·전류를 c_v에 회귀, 기울기 0이 공칭 | `nv_leak.json` |
| **C** 개루프 강성 | ±1 N·m pitch, ±1 N·m roll, 4 s씩(중립 4 s 사이) — 수직 넷 최소노름 할당 F = B_v⁺[0,K,M] + T200 역함수 | pitch ±61, roll ±34 µs [유도] | 정상 각도 변화 → k = \|τ\|/Δθ; `--free-decay N`은 중립만 보내며 운용자가 ~15° 기울였다 놓은 감쇠 주기 → k = I(2π/T)² | `stiffness.json` / `free_decay.json` |
| **D** ESC 지연 | 한 모터(기본 T1): 1500→1600(정지에서), 1540→1640(데드밴드 가장자리), 1600→1700(회전 중), 3회, 1.5 s on / 3 s 사전레벨 | ≤ +200 µs | 펄스 변경 → 자이로 시작(\|Δrate\| > 5 °/s) 지연, 10–90 % 상승 시간; 정지 출발이 ≥ 100 ms 길면 BLHeli_S 기동 페널티 실재 | `esc_latency.json` |

설계 메모(명세와 다른 점, 모두 드라이런에서 드러난 것):

* **B의 공통모드는 기본적으로 hold 패턴 없이** 보낸다(`--nv-hold-us 0`). 명세의 "믹서 depth-hold 패턴(+41/−41/−41/+41 [측정 믹서
  모드 hold, REPORT §3]) 위에 c_v"는 **펄스 공간에서는 wrench-free가 아니다**: T200 곡선이 볼록해서 +41+100 µs의 T5/T8가 얻는
  추력이 −41+100 µs의 T6/T7이 잃는 추력보다 훨씬 커 합성 기체에서 13.7 N 순 상승·pitch −3.5°가 나왔다 [유도 드라이런]. 영공간
  n_v는 **추력 공간**의 방향이다. 그래서 hold가 0이면 펄스 공통모드 = 추력 공통모드(명세 그대로), hold가 있으면(`--nv-hold-us 41`
  또는 `auto`) c_v를 **추력 단위로** 더하고 T200 역함수로 펄스를 되찾는다(요청/달성 추력이 산출물에 적힌다). `auto`는 중립 2.5 s와
  41 µs 패턴 2.5 s의 깊이율을 선형 보간한 레벨(30..80 µs, 데드밴드 아래면 0)이며 peg를 문 2.5–5.4 N 무거운 경우 [측정 REPORT §1]용이다.
  hold가 있을 때 스텝 사이 휴지는 hold 패턴(c_v 0)이다 — 무거운 기체가 휴지 중 깊이 띠를 벗어나기 때문.
* **D의 기본 모터는 T1(수평, yaw)**. 명세의 T5는 같은 Basic ESC지만 수직 하나가 1600/1700 레벨에서 기체를 반복당 ~0.3 m 띄워
  세 번째 반복 전에 깊이 띠를 벗어나고 1700에서 k 6.7 기준 ~26° 기울어져 25° 중단에 걸린다 [유도 드라이런]. `--esc-motor 5`로
  명세대로 돌릴 수 있고 레벨 사이에 "풀 중앙으로 재배치" 프롬프트가 뜬다. yaw는 복원 모멘트가 없어 "정상 각속도"가 실제 정상 상태다.
* C의 roll 펄스 ±34 µs는 모델 데드밴드(±30 µs)에서 4 µs 위다. 실제 ESC 데드밴드 ±25 µs [스펙 BlueRobotics Basic ESC]와
  데드밴드 근처 소신호 이득이 데이터시트의 절반 [측정 REPORT §1 항목 5]이므로 roll의 k는 **상한**으로 읽을 것; 필요하면
  `--stiffness-nm 1.5`.

## 3. 안전 목록 (실행 전·중)

실행 전 운용자가 확보할 것:

1. 기체 **ARMED, MANUAL**, 프로펠러 **장착**, 풀 중앙(COM 바닥 위 ≥ 0.6 m), 테더 느슨, **물속에 아무도 없음**(자유감쇠의 기울이기는
   장대로, 손을 프로펠러 근처에 두지 말 것).
2. 팩 ≥ 15.5 V(도구가 거부), B의 +300 µs(~25 A [예측])는 15.0 V 이상에서만.
3. Lua 오버라이드 설치(`rov_gui/tools/install_rl_pwm_lua.py`; 2026-10-01 설치 기록 `data/20261001/1001_145846_vehicle_setup/`),
   QGC 종료(14550 포트), 스테이션 GUI는 꺼 둘 것(같은 기체에 두 송신자 금지).
4. E의 결과로 순중량을 확인하고 **중성 부근으로 밸러스트**한 뒤 B를 돌릴 것(hold 0 기본). 턱은 비운다.

도구가 매 틱(20 Hz) 검사하고 **중립 5프레임 + 중립 MANUAL_CONTROL**을 보낸 뒤 종료(코드 3)하는 조건 [예측 임계값]:

| 조건 | 한계 | 플래그 |
|---|---|---|
| \|roll\| 또는 \|pitch\| | > 25° | `--tilt-abort-deg` |
| 깊이 | 0.2..0.9 m 밖, 또는 2 s 이상 없음 | `--depth-min-m`, `--depth-max-m` |
| 배터리 전류 | > 30 A | `--batt-a-max` |
| SERVO_OUTPUT_RAW 나이 | > 0.5 s | `--servo-age-max-s` |
| ATTITUDE 나이 | > 0.5 s | `--att-age-max-s` |
| 오토파일럿 HEARTBEAT | 3 s 유실 | `--hb-age-max-s` |
| DISARMED / MANUAL 아님 / PwmGate 실패(파라미터·RLPWM 하트비트·스크립트 미강제) | 즉시 | — |
| 운용자 Ctrl-C, 예외 | 즉시 | — |

프리플라이트(펄스 전, RC 9..16에 아무것도 안 보냄): HEARTBEAT·ATTITUDE·SERVO_OUTPUT_RAW·깊이·전압 수신, 게이트 파라미터 전부 읽힘
(MOT_1..8_DIRECTION = export의 `motor_direction`, SERVO1..8 = Motor1..8, SCR_ENABLE/SCR_USER1 1, RC_OPTIONS bit1 0,
RC_OVERRIDE_TIME > 0, RC9..16_OPTION 0, SYSID_MYGCS = 255), RLPWM 하트비트 존재, 위 한계 전부 통과. 그 다음 중립 프레임
1.5 s 동안 스크립트가 `RLPWM = 1`(강제 중)을 말해야 첫 비중립 펄스가 나간다. 도구는 **파라미터를 쓰지 않으며** 보내는 펄스는
최대 ±300 µs(B), 그 외 ≤ ±200 µs다. 메시지 속도는 MAV_CMD_SET_MESSAGE_INTERVAL로 ATTITUDE 25 Hz·SERVO_OUTPUT_RAW 20 Hz를
요청한다(기본 4 Hz로는 0.4 s 창에 표본이 1–2개).

## 4. 산출물 읽는 법

폴더: `data/<YYYYMMDD>/<MMDD_HHMMSS>_probe/`(드라이런은 `_probedry`). 공통 헤더 `schema 1, written, tool, dry_run, vehicle{sysid,
fw, mot_direction}, export{policy_dir, sigma, B_flu, obs_spec_sha1}`. 모든 틱은 `probe_ticks.csv`(t, seq, step, pwm1..8 송신,
servo1..8 보고, roll/pitch/yaw deg NED, p/q/r deg/s, depth m, batt V/A). `probe_run.json`은 무엇이 돌았고 무엇으로 중단됐는지.

### `thruster_sign_probe.json` (게이트가 읽는 파일 — 스키마 고정)

```json
{"schema": 1, "written": "...", "tool": "rov_gui/tools/thruster_probe.py", "dry_run": false,
 "vehicle": {"sysid": 1, "fw": "4.5.1", "mot_direction": [-1,-1,1,1,-1,1,1,-1]},
 "export": {"policy_dir": "...", "sigma": [1,1,-1,-1,1,-1,-1,1], "B_flu": [[6x8]], "obs_spec_sha1": "..."},
 "pulse_us": 100, "on_s": 1.0, "off_s": 3.0,
 "motors": [{"motor": 5, "axis": "p", "expected_sign_pos": -1, "mean_rate_pos_dps": -9.2, "mean_rate_neg_dps": 8.1,
             "verdict": "PASS", "n_samples_pos": 10, "n_samples_neg": 10, "detail": {...}}, ...],
 "all_pass": true, "operator_notes": ""}
```

* `axis`/`expected_sign_pos`: +펄스가 만들어야 할 FRD 각속도의 축·부호. pwm10_s1 export로는 T1/T3 → r−, T2/T4 → r+,
  T5/T6 → p−(우현 상승 = 음의 roll), T7/T8 → p+ [유도 obs_spec B_flu; 테스트가 thrust_pos×thrust_dirs로 독립 재계산].
* 판정: **PASS** = 두 펄스 모두 부호 일치 **그리고** \|평균\| ≥ 5 °/s **그리고** 창 표본 ≥ 3; **WEAK** = 부호는 맞으나 약함/표본 부족;
  **FAIL** = 부호 불일치, 0, 창 비어 있음(SERVO_OUTPUT_RAW가 펄스를 보여 주지 않음), 미수행(중단).
* `detail.pos/neg`: 명령 시각, 서보 보고 시각·지연(링크 지연 척도), 창의 원시 평균(`mean_raw`)과 기준선(`baseline_*_dps`), 세 축의
  기준선 차감 평균(교차 결합 진단). 기준선을 빼는 이유: yaw는 복원 모멘트가 없어 τ = I/c ≈ 2 s로 느리게 식으므로(N_|r|r 1.55
  [스펙 Wu 2018 BlueROV2]) 직전 +펄스의 회전이 3 s 뒤 −펄스 창에 남아 그대로 읽힌다 [유도 드라이런에서 재현].
* 예상 크기 [유도]: +100 µs ≈ 4.2 N, −100 µs ≈ 3.4 N(voltage_scale 0.72); 수직 하나의 roll 모멘트 0.9 N·m, I_roll 0.5–1.05 kg·m²
  → 감쇠 없이 0.4 s 뒤 20–40 °/s. 합성 기체는 8–12 °/s를 냈다. **수평 추진기의 −100 µs는 한계선이다**: yaw 모멘트 3.4 N × 0.14–0.20 m
  = 0.49–0.68 N·m, I_yaw ≈ 0.73 kg·m² → 창 평균 ~5 °/s [유도]로 문턱과 같다. SERVO_OUTPUT_RAW 보고가 20 Hz라 창이 50 ms 흔들리면
  PASS/WEAK가 갈릴 수 있다(합성 기체에서 T2 −4.8 vs −5.1 °/s). T1..T4의 음펄스 WEAK는 부호 문제가 아니라 문턱 문제일 가능성이 크다 —
  `detail`의 세 축 평균으로 부호를 확인하고 재시도; 수직 넷(0.9 N·m, roll)은 여유가 있다.

### `nv_leak.json`

고도 pass마다 `steps[]`(c_v, 펄스, 요청/달성 추력, 깊이 시작/끝/율, pitch/roll 평균, 전류 평균·최대, 전압 최소, 서보 평균)와
`regression`: `depth_rate_mps_per_100us`, `pitch_deg_per_100us`, `roll_deg_per_100us`, `batt_a_per_100us`와 환산
`leak_n_per_100us_down_positive`(= 깊이율 기울기 × 15 N·s/m [예측 rov_params heave 선형 감쇠]),
`leak_pitch/roll_nm_per_100us`(= 각도 기울기 × 6.7 N·m/rad [유도 cob_arm 0.05 m]). **공칭 예측 = 기울기 0.** `findings`에
\|누설\| ≥ 5 N/100 µs 또는 ≥ 1 N·m/100 µs가 적히면 그것이 발견이며, 수정 계획 1단계의 영공간 벌점·B 모델을 다시 봐야 한다.
전류 기울기(합성 기체 8 A/100 µs)는 공동수축의 비용 그 자체다.

### `stiffness.json` / `free_decay.json`

스텝마다 `angle_before_deg → angle_steady_deg`(스텝 마지막 1.5 s 평균), `delta_deg`, `peak_deg`·`t_peak_s`(오버슈트 → 감쇠 힌트),
`k_est_nm_per_rad`, `sign_as_expected`(false면 A와 같은 부호 문제). 기대: k 4.5면 ~13°, 20.3이면 2.8° [유도]. `k_pitch/k_roll`은
공칭 T200×0.72 추력 기준의 **상한**(실제 데드밴드 근처 추력이 절반이면 k도 절반). 자유감쇠: 축별 `period_s`(감쇠 주기), `zeta`
(로그 감소율), `omega_n_rad_s`, `k_lo/k_hi_nm_per_rad`(I_pitch 0.87–1.25, I_roll 0.49–1.05 kg·m² [유도 rov_params + 부가관성
자리표시 0.12]), `extrema`. 3개 미만의 교대 극값이면 주기 없음 — 더 크게 기울이거나 다시.

### `esc_latency.json`

`cases[]`(pre_us→on_us, 예측 정상 기울기, 반복별 `onset_s_from_cmd`/`onset_s_from_servo`/`peak_dps`/`rise_10_90_s`),
`from_rest_penalty_s` = 정지 출발 지연(서보 보고 기준) − 회전 중 출발 평균, ≥ 0.100 s면 `verdict` "BLHeli_S start-up penalty REAL"
[스펙 BR 포럼/BLHeli_S 매뉴얼 110–130 ms]. 서보 보고 기준 지연은 링크 지연을 제거한 값이다. 합성 기체(0.12 s 주입)는 0.19 s를 냈다.

### `rest_trim.json`

`roll/pitch_mean_deg ± std`(NED, +pitch = 기수 위 — 스테이션 CSV의 `pitch_deg`는 부호가 반대임에 주의 [측정 REPORT §0]),
`depth_rate_mps`와 `net_weight_sign`(heavy/buoyant/neutral within resolution, 문턱 2 mm/s [예측]), `net_weight_n_est_down_positive`
(× 15 N·s/m [예측]), `batt_a_mean`(호텔 부하), `batt_v_mean`. `operator_notes`에 턱 비움·밸러스트 상태를 적어 두라(`--note`).

## 5. 모터가 FAIL이면

1. **모든 것을 멈춘다.** rl_pwm / rl_pwm_7d는 어떤 캡에서도 날리지 않는다 — 게이트가 `all_pass: false`로 거부하지만, 그것은 마지막
   방어선이지 절차가 아니다. `--note`에 상황을 적고 산출물 폴더 경로를 KNOWN_ISSUES에 남긴다.
2. 어디가 틀렸는지 가른다: FAIL이 **부호 반대**이고 \|평균\| ≥ 5 °/s이면 (a) export의 `sigma`/`B_flu`(학습 세계가 가정한 방향)
   또는 (b) 배선/프로펠러/`MOT_n_DIRECTION` 중 하나다. 대각 쌍(5↔8, 6↔7) 둘이 동시에 FAIL이면 배선 교차 [유도 드라이런이 5↔8 스왑을
   그 형태로 잡았다]; 하나만이면 그 모터의 방향 파라미터나 프로펠러(CW/CCW). 믹서 모드(mpc_tuned)로 +heave를 주며 `SERVO_OUTPUT_RAW`와
   기체 움직임을 보면 믹서 기준 방향을 교차검증할 수 있다(2026-10-03 회귀: 위로 = T5/T8 > 1500, T6/T7 < 1500 [측정 data/20261003]).
3. 고치는 쪽을 정한다. **기체를 바꾸지 말고 export를 바꾸는 것이 원칙**(학습 세계와 믹서 적합 `logs/mixer_fit_report.md`가 기체를
   기준으로 맞춰져 있다). sigma 한 항을 뒤집으면 B_flu의 그 열 전체가 뒤집힌다 — RL_controller `configs/rov_params.yaml`
   `mixer.motor_direction`/`thrusters.axes`를 고치고 **재학습·재export**가 필요하다(정책은 그 부호로 학습됐다). 배선을 고치는 경우
   `MOT_n_DIRECTION`이 export와 다시 일치하는지 게이트가 확인한다.
4. **다시 A를 돌려** 8/8 PASS를 받는다. WEAK(부호는 맞고 약함)는 PASS가 아니다: 팩 전압, ATTITUDE 속도(25 Hz 요청이 먹었는지
   `probe_run.json msg_counts`), 기체가 바닥·벽에 닿았는지 확인하고 재시도. `n_samples`가 1–2이면 메시지 속도 문제다.
5. 그 뒤에야 B/D/E/C, 그리고 README §10.3의 순서(캡 0.25 정지 유지)로 간다.

## 6. 검증 상태

* 오프라인: `rov_gui/tests/test_thruster_probe.py` 18건 + `test_rl_pwm.py` 26건 통과(2026-10-10). 드라이런으로 전 시퀀스 실행,
  5↔8 스왑·역회전 검출, 중단 조건·중립 버스트·거부 코드·`load_and_check` 거부 항목 전부 고정.
* **실기: 0런 [예측]** — 임계값(5 °/s, 창 0.15–0.55 s, 25°, 0.2–0.9 m, 30 A), 메시지 속도 요청이 BlueOS 경로에서 먹는지, Lua 스크립트
  하트비트 타이밍, SERVO_OUTPUT_RAW 보고 지연, 자유감쇠에서 기울이기 중 25° 중단에 걸리지 않는지 — 첫 산출물이 말해 줄 것.
