# KNOWN_ISSUES — 아직 안 고친 것들

> Claude Code 세션 중 발견했지만 **아직 코드로 고치지 않은** 버그·함정·보류 사항의
> 살아있는 목록. 규칙: **고치면 그 항목을 삭제**한다 (고친 기록은 git 히스토리가 담당).
> 새로 발견하면 날짜와 함께 추가한다. 항목 형식: 증상 → 임시 대응 → 제대로 고치는 법.

## ⚠️ 운용 안전

### LOW `RL_PWM`(추진기 직접 펄스)은 0.25 캡에서 **포화·진동**한다 — 첫 수중 런 2026-10-01 17:29 (2026-09-30 등재)
- **무엇**: `rl_pwm` 모드(`rov_gui/control/rl_pwm.py`, `rl_policy.py` `HwRlPwm`, `backends/hardware.py` `set_pwm`)가 정책 `pwm10_s1`의
  펄스 8개를 RC_CHANNELS_OVERRIDE ch 9..16으로 보내고, 기체의 `rl_pwm_override.lua`가 모터 출력에 강제한다. ArduSub 믹서를 우회한다.
- **확인된 것**: 오프라인 테스트 26건(`rov_gui/tests/test_rl_pwm.py`) — 실제 pymavlink v2 프레임의 바이트가 스크립트가 읽는 위치와
  일치, 게이트 항목별 거부, sink 래치·정지 경로, 워커 게이트/정지. 데모 백엔드에서는 "전송 경로 없음"으로 거부된다.
- **첫 수중 런(2026-10-01 17:29–17:30, `data/20261001/1001_172910`, 정책 미션 3회 42/8/24 s)**: 펄스는 나갔다(`rl_pwm: policy pulses are going out on RC 9..16`,
  `pwm_age_s` 평균 0.14 s). 움직임: roll −8~+6°(주기 약 3 s), pitch가 −9~−18°에 치우친 채 1–4° 흔들림, yaw 최대 34° 흔들림(std 4–10°); 위치 추종 자체는
  rms 2–4 cm·최대 15 cm [측정: `mpc_172910/172957/173011.csv` `roll_deg`·`pitch_deg`·`yaw_deg`, px−rx 등]. (같은 날 1차 보고의 "피치 ±5–10°·yaw −16°·깊이 0.1–0.4 m"는
  기준 위치 열 `ry`/`rz`를 자세로 잘못 읽은 값 — 철회. 깊이 변화는 대부분 기준이 내려간 것이다.)
  **원인은 캡**: 추진기 하나가 캡 0.25에서 내는 추력은 5.8 N, 학습 때(1.0)는 64 N — 정책은 11배 큰 권한을 전제로 학습됐다 [유도: `rl_pwm.py PwmModel` T200 곡선].
  추진기 3·4·6이 시간의 61–91 %, 1·5가 44–64 % 캡에 붙어 있고, 캡이 **스케일이 아니라 클립**(`rl_policy.py:350`, `rl_pwm.py:79`)이라 여러
  추진기가 동시에 0.25를 넘으면 추진기 간 비율(할당)이 무너진다. 지연은 아니다(tag 0.13 s, 늦은 플랜 런당 1개).
  **방향 근거**: 같은 런에서 roll·pitch·yaw 각가속도와 상하 가속도가 모델의 uK·uM·uN·uZ와 부호가 맞는다(세 런 모두; 상관 0.1–0.4로 약함) [측정: 같은 CSV 회귀].
  명령→펄스는 실기 로그로 ArduSub 믹서와 대조돼 있다(`RL_controller/README.md` 28행, `scripts/fit_mixer_from_logs.py`). 추진기 한 기씩의 부호 프로브(§10.3 (4))는 안 했다.
  **2026-10-01 운용자 결정**: `pwm_cap` 1.0 (한 번에). 1.0에서는 아직 안 돌았다.
- **2026-10-01 설치됨**: `rov_gui/tools/install_rl_pwm_lua.py --install --yes`로 `SCR_ENABLE` 0→1, `SCR_USER1` 0→1, 스크립트를 MAVLink FTP로
  `scripts/rl_pwm_override.lua`에 올리고(읽어 되돌려 바이트 동일), 재부팅 뒤 기체가 `rl_pwm_override.lua loaded`를 보냈다 — 스크립트가 **로드는
  된다**(`mavlink:init` 인자 순서 포함 초기화 통과) [측정: `data/20261001/1001_145846_vehicle_setup/rl_pwm_lua_install.json`]. 바꾼 값과 이전 값이 그 파일에 있다.
  같은 날 `SERVO14_FUNCTION` 60(RCIN10 passthrough, 출력은 0 µs — RC10에 값을 보내는 주체 없음) → 0(사용 안 함), 운용자 요청
  [측정: `data/20261001/1001_150534_vehicle_setup/servo14_function.json`]. 이제 게이트의 파라미터 항목은 전부 통과한다
  (`install_rl_pwm_lua.py` 읽기 전용 점검). 남은 것은 스크립트 heartbeat·교전 확인 — 실제 교전에서만 볼 수 있다.
- **확인 안 된 것 [예측]**: 스크립트가 펄스를 실제로 넘겨받는지(`set_output_pwm_chan_timeout` 동작, `RLPWM: engaged`), 프레임 주기 50 ms 대
  스크립트 stale 100 ms의 여유, 추진기별 방향(`MOT_n_DIRECTION` 일치는 게이트가 보지만 물리 방향은 아님), `pwm_cap` 0.25에서 정책이 뜨는지(1.0으로 학습).
- **알려진 한계**: (1) 이 기체에서 조명·카메라·그리퍼가 RCIN9..16 passthrough에 물려 있으면 게이트가 거부한다 — 옮겨야 쓸 수 있다.
  (2) CSV에 명령 펄스 열이 없다(`pwm1..8`은 기체 보고값, `uX..uN`은 명목 추정). (3) "명령은 나가는데 추진기가 중립" 감시
  (`_watch_actuation`)는 축 기준이라 이 모드에서 꺼져 있다 — 스크립트의 `RLPWM` 값이 대신한다. (4) station bridge coast 중에는
  중립 펄스만 낸다(깊이 유지 없음). (5) sink는 무장/MANUAL을 직접 보지 않는다(워커와 스크립트가 본다); `SYSID_THISMAV`는 읽지 않는다.
  (6) 태그 상실 디바운스(최대 0.4 s) 동안은 펄스도 직전 명령 갱신도 없다.
- **입수 전 절차**: `rov_gui/control/rl_policies/pwm10_s1/README.md` — 스크립트 설치 → 프로펠러 뺀 벤치(교전·E-STOP·인계·`RLPWM: released`
  깜빡임 확인) → 추진기별 방향 → 수중은 `pwm_cap` 0.25, 관측 가능한 얕은 곳에서.
- **검토**: safety-code-reviewer 1차 FAIL(게이트 구멍 3건: 채널 순열 허용, grace 실패 미래치, 파라미터 세션 캐시) → 수정 → 2차 WARN
  ("벤치 통과 후 cap 0.25 첫 수중 런 허용", 수중 전 권고 2건: grace 타이머가 끊길 때마다 리셋, sink tick 예외 시 미래치) → 둘 다 수정.
  **마지막 수정분은 재검토를 받지 않았다.** 리뷰어는 코드를 실행하지 못했다. 남은 권고: QGC를 붙인 채 10분 이상 벤치에서 파라미터 age,
  늦은 프레임으로 인한 오작동 래치 빈도, `RCINnScaled` 번호(139+n) 대조. 기록은 `.claude/journal/reviews.md`.

### 정책 추론이 단독으로는 **19 ms**인데 스테이션에서는 0.21–0.27 s — FoundationStereo와 GPU를 나눠 쓰기 때문 (2026-10-01)
- **측정** [`data/20260930/0930_220212/diag/policy_vs_fstereo_contention.json` + `policy_alone_timing.json`, 스크립트 같은 폴더; RTX 5090, peg ViT-CLIP ckpt, `model` weights, GUI·카메라 없음]:
  정책 forward 단독 8 step 19.3 ms / 16 step 33.5 ms / 4 step 12.2 ms. 같은 프로세스에서 FoundationStereo(iters 8, scale 0.75, CUDA graph)를 쉬지 않고 돌리면
  8 step p50 152.8 ms(p90 210.0), 4 step 104.4 ms. FS를 7.5 Hz로 띄엄띄엄 돌리면 8 step p50 40.3 ms(p90 86.3).
  실기(22:02·22:16 engagement 6개)는 infer_ms p50 207–273 ms [`plan_timing.json`], FS solve p50 99 ms(정책 옆에서 쉬지 않고 돈 오프라인 벤치 76 ms — FS 단독 측정은 그 파일에 없다)·9.7–11.2 Hz [`latency_breakdown.json` — UI 녹화 10프레임을 눈으로 읽음, `depth_hud_montage.png`].
- **플랜 나이의 구성** (rl 런 `mpc_220407`, p50 0.556 s) [`latency_breakdown.json`]: 촬영→depth 완성 0.196 s(FS solve 0.099 + 카메라 전송·대기 0.098 [유도: 차]),
  정책 forward 0.207 s, 나머지 0.153 s [유도: 차 — 관측 만들기 2장·트리거·버스·20 Hz tick; 단계별 stamp가 기록에 없다].
- **뜻**: 지연의 큰 덩어리는 diffusion 계산이 아니라 GPU 대기다. `num_inference_steps` 8→4는 단독 −7 ms, FS와 겹칠 때 −48 ms(−32 %)라 절반이 되지 않는다.
- **임시 대응(구현됨, 기본 OFF, 수중 미검증, 2026-10-01)**: `--policy-fs-schedule yield|only`(기본 `free` = 위 상태 그대로) — yield는 정책이
  추론하는 동안 FS가 새 프레임을 시작하지 않고(정책은 진행 중 프레임을 최대 150 ms 기다림), only는 정책 미션 중 FS가 정책이 쓰는 2장만 계산한다.
  코드 변경 없는 다른 손잡이는 `--fstereo-scale 0.5`(FS 45.7 ms라 15 fps를 따라가며 쉰다). 실제 두 워커·두 네트워크로 잰 오프라인 벤치
  [측정: `rov_gui/tools/fstereo_bench_out/policy_fs_schedule_20261001_115008.json`, 합성 입력이라 카메라 전송 지연 없음, 측정 중 CPU load 0.9→7.5/16코어]:
  정책 forward p50 132.7 ms(free) → 19.4(yield) / 19.4(only) / 30.8(scale 0.5) / 19.6 ms(yield+0.5); 플랜 나이(emit) p50 328.8 → 224.6 / 154.5 / 134.5 / 145.6 ms;
  연속 프레임 쌍 82.5 % → 86.0 / 100 / 100 / 100 %; 미션 중 FS 12.93 → 11.93 / 3.73 / 15.0 / 15.0 장/s. 사용법·기록 키·합산 경계는 `rov_gui/README.md`
  "FoundationStereo와 GPU 나눠 쓰기" 절.
  앞서 적었던 프로토타입 수치 [`data/20260930/0930_220212/diag/policy_fs_schedules.json`]는 워커가 아니라 단순 루프의 것이고, 그 JSON의 note
  "the wait is ~0 in the station"은 측정이 아닌 가정이었다(트리거가 도착이 아니라 period 위상이라 대기는 진행 중 프레임의 남은 시간이다 — 프로토타입
  p50 43.1 / p90 69.9 ms, 워커 벤치 p50 0.0 / p90 19.5 ms). (A)의 "패널 stale 기준 1.5 s"도 틀렸다: `--fstereo`의 depth 패널은 warn 0.75 s /
  stale 2.5 s다(window.py `expect = min(expect, 4.0)` → widgets/video.py Freshness) — 결론(경고 안 뜸)은 같다.
  `plans.jsonl`의 `emit_age_s`·`trigger_age_s`·`depth_ready_age_s`·`fs_wait_ms`로 "나머지 0.15 s"를 이제 나눌 수 있다(정책 플랜 줄마다, `free` 포함).
- **제대로 고치는 법**: LOW None(observe) 런으로 실기 `fs_wait_ms`·`burst_timeouts`·플랜 나이를 먼저 보고, free / yield / only / scale 0.5 중 하나를 수중 A/B로
  골라 기본값으로 만든다(또는 스위치를 지운다). scale 0.5의 정확도 대가는 `--record-stereo`로 수중 원본을 모은 뒤 오프라인 A/B로 본다 — 아직 미실행.

### 정책 플랜이 **다음 플랜이 오기 전에 끝난다** — 기준이 끝점에 서 있는 시간이 26–78 % (2026-10-01)
- **발견**: 2026-10-01, "추론 지연만큼 앞 action을 버려야 하지 않나"라는 질문을 코드·기록으로 확인하다가.
- **시간 정렬 자체는 맞다(코드)**: 플랜의 시간축은 관측 시각에 고정되고(`workers.py` `compose_plan(..., t0=obs_t_rel)`, raw action k = 관측 시각 + k·66.7 ms),
  스티처는 "지금" 시각으로 그 축을 샘플한다(`plan_stream._PlanRef.sample`) — 지연만큼의 앞부분은 저절로 건너뛴다. raw action 0은 관측 시각의 자세다
  (|action 0| p50 0.3–0.9 mm).
- **측정** [`data/20260930/0930_220212/diag/plan_timing.json`, 스크립트 같은 폴더 `plan_timing.py`; 22:02·22:16 engagement 6개]:
  플랜 나이 p50 0.56–0.66 s(추론 전 체인 0.35–0.39 + 추론 0.21–0.27) → 설치 시점에 기준이 놓이는 raw action 번호 p50 **7.4–8.3**(앞 8개는 한 번도 안 날아간다).
  설치 뒤 남은 수명 p50 0.45–0.52 s(감속 없는 플랜), 다음 설치까지 p50 0.55–2.55 s → 다음 설치 전에 끝난 플랜 **54–94 %**.
  CSV `ref_src`: hold가 rl 런(`mpc_220407`) 26 %, mpc_tuned 5개 38–78 %; 순수 plan은 3–21 %, 나머지는 blend.
- **원인**: 청크 길이 1.0 s(16 × 66.7 ms)에 여유가 없다. 추론은 실행과 겹쳐 돈다(관측 간격 p50 0.533 s; 다음 관측이 앞 플랜 도착 전에 찍힌 주기 73–96 %).
  설치된 플랜만 보면 나이 p50 0.52–0.57 s, 남은 수명 p50 0.45–0.52 s라 **다음 플랜이 정상 도착하면 빈 구간은 p50 0.0–0.10 s(p90 ≤ 0.14 s)**뿐이고,
  **다음 플랜이 late로 버려지면 p50 0.5–1.2 s(최악 런 3.9 s)**가 빈다 [측정: `data/20260930/0930_220212/diag/plan_hole.json`] — hold 시간의 대부분은 late 탈락에서 온다.
  만료 뒤에는 `hold_tail: extrapolate`가 꺼지고 끝점 정지가 된다(`_extrapolate_tail`은 플랜이 live일 때만).
- **곁들여**: 같은 기록에서 관측 프레임 쌍 간격이 66.7 ms인 플랜과 133 ms인 플랜이 반반이다(rl 런 234 : 182) — 아래 2026-09-02 "depth 페어" 항목과 같은 문제.
  `--policy-fs-schedule only`와 `--fstereo-scale 0.5`는 오프라인 워커 벤치에서 연속 프레임 쌍 100 %였다(free 82.5 %) [측정: `rov_gui/tools/fstereo_bench_out/policy_fs_schedule_20261001_115008.json`].
- **고치는 법**: 나이를 줄인다 — 바로 위 항목의 스위치(구현됨, 기본 OFF, 수중 미검증; 벤치 플랜 나이 p50 328.8 → 134.5–224.6 ms). 그 밖에 `period_s`를
  0.35 s쯤으로 줄인다 [예측: GPU를 FS와 나눠 쓰므로 나이가 늘 수 있음, 미시험], 만료 뒤에도 외삽을 잇는다(코드), 또는 action horizon을 늘려 재학습한다(미적용).

### 정책이 "그리퍼 열기"를 내도 스테이션은 열지 않는다 — 켜면 오발 위험도 있다 (2026-09-30)
- **사실**: 2026-09-30엔 `policy.gripper: false`라 폭 채널이 버려졌다(10-01부터 true, lookahead 0.8 — 10-01 15:19 rl 런에서 첫 정책 OPEN 실행,
  `data/20261001/1001_151431/mpc_151814.csv` grip_cmd). 남은 문제는 **오발**과 턱 피드백 부재다. peg 정책은 삽입 순간에 open을 낸다: rl 런 `mpc_212757` t=77–84 s의 6개 플랜이
  1초 앞 폭 0.068–0.085 m를 냈고 그때 peg는 구멍 안이었다(`data/20260930/0930_211846/diag/open_moment.png`, 조종자가 85 s에 수동으로 엶).
- **오발**: 같은 날 `mpc_213330` t=9–20 s의 5개 플랜도 open을 냈는데 peg는 구멍 밖이었다(`data/20260930/0930_211846/diag/open_moment2.png`) [측정: `data/20260930/0930_211846/diag/plan_age_and_open_requests.json`].
- **켤 때**: `gripper: true`만으로는 안 열린다 — open은 청크의 끝(약 0.9–1.0 s 앞)에 놓이고 knot 0은 현재 폭이라, `gripper_lookahead_s` 0.0("지금" 샘플)은
  문턱(폭 0.063 m)을 넘지 못한다. 0.6–0.8 s가 필요하다(같은 파일의 09-07 주석). 턱 피드백이 없고, 오발하면 peg를 떨어뜨린다. 수중 미검증.
- **22시 재확인**: rl 런 `0930_220212/mpc_220407`에서 peg가 구멍 안에 있는 동안(`data/20260930/0930_220212/diag/ui_t303.png`, `ui_t318.png`) t=176–214 s에
  플랜 17개가 open(폭 최대 0.090 m)을 냈고 CSV `grip_cmd`는 전 구간 0이다. 설치된 플랜에 `gripper_lookahead_s`를 대입하면 0.0 s는 0회, 0.8 s는 9회
  문턱을 넘는다. 삽입 전 176 s와 같은 밤 mpc_tuned engagement 5개에서는 open 요청 0건(오발 없음) [측정: `data/20260930/0930_220212/diag/open_requests.json`].

### peg-in-hole 정책(`peg5d_vitclip_9_27_9_28`)이 수중에서 **peg를 물고도 후퇴**를 낸다 — ROV 영상 속 턱+peg가 데모와 다른 자리·모양 (2026-09-30)
- **증상**: 2026-09-30 정책 런 7개 engagement 모두 플랜이 뒤+위로 향한다. 관측 런 `data/20260930/0930_183320_observe/plans.jsonl` 1079건의
  마지막 knot(1.0 s 앞) 평균 TCP dz −0.048 m, dz<0 비율 0.958; 추종 런에서는 선체가 실제로 뒤로 0.10–0.50 m 밀렸다
  [측정: `data/20260930/0930_183320_observe/diag/action_direction_stats.json`, 각 런 `mpc_*.csv` px/py]. 턱은 peg를 물고 있었고 depth에도 peg가 보인다
  [측정: `data/20260930/0930_182033/ui_20260930_182035.mp4` C3 RGB·C3 DEPTH 패널].
- **원인(부분 규명)**: UI 녹화의 depth 패널에서 정책 입력을 복원해 다시 추론했다 (복원 obs MAE 1.4–2.1/255 [측정: `data/20260930/0930_183320_observe/diag/recon_fit_0921.json`];
  기록 플랜과 dz 상관 0.91·부호 일치 0.95, 322 플랜 [측정: `data/20260930/0930_183320_observe/diag/final_tests.json` `_check`]). 복원 obs에서 dz<0 비율은
  그대로 0.90 → **턱+peg를 지우면 0.31** → 턱+peg만 42 px 아래(데모의 자리)로 옮기면 0.66 → 거기에 proprio 폭 0.007 m까지 주면 0.53
  [측정: `data/20260930/0930_183320_observe/diag/final_tests.json`]. 즉 후퇴를 만드는 것은 **ROV 영상에 찍힌 턱+peg 자체**다. ROV는 턱이 광축 아래 0.067 m, 핸드헬드는 0.129 m라
  [설정값: `mpc_183320.meta.json policy.tcp`] 224 px obs에서 턱+peg가 약 43 px 위에 찍힌다 [유도]. proprio 폭 단독 효과는 작다(0.90 → 0.83).
- **무엇이 다른가(측정)**: depth에서 peg 축을 맞춰 보면 카메라–그리퍼 축 기울기는 두 장비가 같다(ROV 40.8°, 핸드헬드 9/28 40.8°, 9/27 35.6°).
  카메라에서 peg 축까지의 수직 거리도 같다(0.213 vs 0.215–0.217 m [유도: 같은 파일의 root 좌표]). 다른 것은 **축 방향 거리**: peg 뿌리가 ROV에서
  4.6–5.9 cm 더 앞에 있다 [측정: `data/20260930/0930_183320_observe/diag/rig_geometry.json`; 양쪽 depth가 미터 단위로 맞다는 가정]. 설정값의 TCP 오프셋으로 따로 계산해도 3.9 cm 앞 [유도: meta `policy.tcp`].
  즉 캘리브레이션(내부 파라미터·워프) 문제가 아니라 카메라–턱 간 앞뒤 거리 차이다. 자로 확인 필요(미실측).
- **고치면 되는가(예측)**: 복원 obs를 3D로 재투영해 카메라를 그리퍼 축 따라 앞으로 옮긴 시점을 만들면 dz<0 비율이 0.90 → 0.66(3 cm) → 0.23(5 cm) → 0.22(7 cm),
  proprio 폭 0.007 m까지 주면 5 cm에서 0.16·dz +0.018 m·폭 출력 0.008 m(쥔 채 유지) [예측: `data/20260930/0930_183320_observe/diag/virtual_camera.json`, 322 플랜; 단일 시점 depth 재투영이라 근사].
- **철회한 것**: 같은 날 1차 진단의 "턱이 비어 있었다"는 추론. `policy.estimator.n_drives` 0은 조이스틱 버튼 구동을 세지 않아 근거가 못 된다.
- **임시 대응(구현됨, 기본 OFF, 수중 미검증)**: `config/hw_mpc.yaml policy.obs_view_forward_m` — 정책이 보는 depth obs를 C3보다 body x로 그 거리만큼
  앞선 가상 시점에서 다시 그린다(`rov_gui/perception/policy_obs.py` `_shift_view`; 행동 변환은 그대로). 실제 빌더로 당일 322 플랜을 재추론하면
  dz<0 비율 0.90 → 0.31(0.05 m), proprio 폭 0.007 m까지 주면 0.18·dz +0.018 m [예측: `data/20260930/0930_183320_observe/diag/validate_builder_shift.json`]; 폭 0.007 포함 시
  0.03 m 0.37, 0.04 m 0.23, 0.06 m 0.16, 0.07 m 0.18 [예측: `data/20260930/0930_183320_observe/diag/validate_builder_shift_sweep.json`]. 빌드 시간 0.4 → 14.7 ms/프레임
  [측정: 같은 파일 `timing_ms`, 0921 기록 depth 97장]. 21:03–21:07의 재시도 런(`0930_210312`, `0930_210502_observe` 등)은 **스위치가 꺼진 채** 돌았다
  (meta `policy.config.obs_view_forward_m` null, `OBS VIEW SHIFT` 로그 없음) — 기록된 dz<0 비율 0.86. 그 237 플랜의 obs에 스위치를 켜면 0.32(0.05 m),
  폭 0.007 m까지 주면 0.15 [예측: `data/20260930/0930_183320_observe/diag/replay_2103.json`]. 켠 상태의 수중 런은 아직 0건.
- **켤 때 알아둘 것**: (1) **프로세스 단위**다 — 켠 채로 패널에서 캔 체크포인트를 고르면 그쪽에도 적용된다. peg 세션에만 `0.05`, 끝나면 `null`
  (`test_policy_obs.py`가 출하 설정이 `null`인지 확인한다). 경계는 기동·ARM마다 로그/events.log `OBS VIEW SHIFT`, meta `policy.worker.obs.view_shift`.
  (2) 플랜당 obs age가 약 29 ms 는다 [유도: 2 프레임 × 14.7 ms]. 09-30 추종 런은 이미 5–10 %가 late였고 같은 분포에 29 ms를 더하면 8–18 %
  [유도: `data/20260930/0930_183320_observe/diag/late_fraction_estimate.json`] — 첫 런에서 `age_at_intake_s`를 실측할 것. (3) plans.jsonl의 `depth_valid`는 채운 뒤 값이라 1.0으로 찍히고,
  프레임별 채움 수는 기록되지 않는다. (4) `policy_session_replay.py`·`policy_yaw_vs_can.py`·`depth_compare.py`는 이 스위치를 모른다 — 이동 런의
  `obs/*.png`는 이동된 영상, `depth/*.png`는 이동 전이다. (5) 가려진 뒤쪽은 배경 거리로 채운 값이지 측정이 아니다. 이동 런과 아닌 런을 합산하지 말 것.
- **같이 해야 효과가 나는 설정(미적용)**: peg ckpt용 proprio 폭 수준 — `gripper_width_open_m` 0.090, `gripper_width_closed_m`·`gripper_width_init_m` 약 0.007
  (현재 0.069/0.042는 peg 데이터의 폭 범위 0–0.090 m와 다르다). 캔 ckpt와 공유하는 값이라 손대지 않았다.
- **되돌리기**: `obs_view_forward_m: null`. 스위치 자체를 걷어내려면 `policy_obs.py`의 view shift 블록, `geometry.py`·`backends/policy.py`·`workers.py`의 키,
  `test_policy_obs.py` 7절, README 항목을 지운다. 물리적으로 맞추는 길(C3를 5 cm 앞으로 또는 그리퍼를 뒤로, 기준 그림 `data/20260930/0930_183320_observe/diag/jaw_position_reference.png`)을 택하면 이 스위치는 필요 없다.

### USB 카메라를 스트림 시작 중에 뽑으면 **커널 uvcvideo가 크래시하고 그 USB 컨트롤러가 재부팅 전까지 묶인다** (2026-10-01)
- **증상**: 18:25:40 `./c3 cctv` 카메라 프로세스가 카메라 3대를 여는 순간 포트 3-6 카메라가 USB에서 끊겼고(`usb 3-6: USB
  disconnect`), 2 s 뒤 그 장치에 대한 STREAMON에서 `BUG: kernel NULL pointer dereference` — `usb_ifnum_to_if ←
  usb_hcd_alloc_bandwidth ← usb_set_interface ← uvc_video_start_transfer`, Comm `python`(카메라 프로세스의 캡처 스레드)
  [측정: `journalctl -k` 2026-10-01 18:25:42]. 그 뒤 `kworker/…+usb_hub_wq`가 D 상태, 같은 컨트롤러(PCI `0000:13:00.0`,
  Bus 03)의 다른 카메라는 STREAMOFF(`usb_set_interface`)·포맷 조회(`uvc_v4l2_get_format`)에서 **영구 D 상태**,
  `/sys/bus/usb/devices/3-6/product` 읽기도 멈춤. 끊긴 카메라는 58 s 뒤 다른 컨트롤러(Bus 01, `0000:11:00.0` 포트 2)에서
  다시 잡혔다(손으로 옮긴 것인지, 스트림 시작 때 스스로 떨어진 것인지는 모름).
- **원인(부분)**: 크래시한 모듈은 기본 커널 드라이버가 아니라 `librealsense2-dkms` 1.3.28의 패치판 uvcvideo
  (`/lib/modules/6.8.0-138-generic/updates/dkms/uvcvideo.ko`, version `1.1.1-realsense-1.3.28`). 끊김과 STREAMON의 경쟁
  상태에서 죽은 스레드가 컨트롤러의 대역 잠금을 쥔 채 사라진 것으로 보인다 [유도: D 상태 wchan]. 기본 uvcvideo에도 같은 경쟁이
  있는지는 미확인. 이 리포 코드는 RealSense를 쓰지 않지만 시스템에 librealsense2·pyrealsense2가 깔려 있다(다른 프로젝트용일 수 있음).
- **스테이션에 미친 영향**: 없음 — 카메라가 자식 프로세스라 멈춘 건 그 프로세스뿐이고, 창은 `stop()`의 제한 시간 뒤 정상
  종료했다. 같은 프로세스였다면 스테이션 자체가 kill 불가 상태가 됐다.
- **대응**: 재부팅(D 상태는 kill로 안 풀림). 카메라는 **스트리밍 중·시작 직후에 뽑지 않는다**. 15 s 넘게 첫 프레임이 없으면
  타일이 "stuck … USB/driver fault?"라고 말한다(`pool_cam.py STUCK_S`).
- **제대로 고치는 법(미실행)**: RealSense를 이 PC에서 안 쓰면 `librealsense2-dkms`를 지워 기본 uvcvideo로 돌아가는 것을 검토
  (sudo, 다른 프로젝트 확인 먼저). 쓰면 최신 dkms 버전으로.

### 스테이션 종료 시 `cmd_estop`이 **레코더 join들 뒤에** 나간다 — MPC engaged면 그동안 명령이 계속 나갈 수 있다 (2026-09-30, safety 감사에서 발견)
- **증상(코드 읽기, 미재현)**: `rov_gui/window.py` `MainWindow.shutdown()`은 `cmd_timer`를 멈춘 뒤 UI 레코더 join(≤5 s,
  `recorder.py` `ScreenRecorder.stop`), 켜져 있는 피드 레코더마다 join(각 ≤5 s), mission log 저장, nav 기록 닫기를 하고
  **그다음에** `cmd_estop.emit()` + `backend.stop()`을 한다. 텔레옵 펌프는 멈췄으니 sink deadman(500 ms)이 중립을 잡지만,
  MPC가 engaged면 MpcWorker가 그 몇 초 동안 `cmd_pilot`을 계속 보내 deadman을 먹일 수 있다 [유도; MpcWorker 쪽은 안 읽음].
  풀 카메라(`--pool-cams`) 대기는 이미 맨 끝(중립 뒤)이라 이 문제와 무관.
- **임시 대응**: 창을 닫기 전에 E-STOP(Esc)을 누른다 — `estop()`은 MPC disengage + `cmd_estop`을 즉시 보낸다.
- **제대로 고치는 법**: `shutdown()` 맨 위에서 `self._mpc_engaged`면 `cmd_mpc_engage(False)`, 그리고 `cmd_estop.emit()` +
  `backend.stop()`을 레코더 정리보다 **먼저**. 레코더·nav·mission log는 백엔드와 무관하니 순서만 바꾸면 된다(테스트로 확인 필요).

### 학습 런이 GPU를 쓰는 동안 비행하면 정책 플랜이 **전부 late**로 버려진다 (2026-09-14)
- **증상**: `data/20260914/0914_151901/` 세 engagement에서 `plans.jsonl` 329건 **전부 `late`**(intake 시 obs age 0.84 s >
  `policy.obs_max_age_s` 0.60 s), `mpc_152047.meta.json policy.run`: received 75 / installed 0. 추론 `infer_ms` p50 **629 ms**
  (p95 707, max 952) — 09-13 실기 262 ms의 2.4배. reference는 hold 지점에서 한 번도 안 움직였고(`rz` 0, `plan_id` NaN)
  DP는 기체를 몰지 않았다.
- **원인**: `tools/run_ablations.sh D_dinov3b` 학습(12:00:50 시작)이 같은 RTX 5090을 99 % / 14.7 GB 점유한 상태에서
  스테이션 추론(rovgui-pose)이 돌았다 [측정: `nvidia-smi` 15:30, `ps` etimes]. 스테이션은 이를 감지·경고하지 않는다.
- **대책(미구현)**: 풀 세션 전에 학습을 멈추거나, 스테이션 기동 시 다른 CUDA 프로세스가 있으면 WARN(`policy: READY` 줄에
  infer_ms 첫 측정치가 이미 있으니 late 3연속이면 원인 후보로 GPU 공유를 말하게 하는 것이 최소 수정).

### [임시 — 2026-09-13에 삭제] 정책 플랜의 z가 `policy.z_hold_above_floor_m: 0.20`으로 고정돼 있다 (2026-09-12)
- **무엇**: `config/hw_mpc.yaml policy.z_hold_above_floor_m: 0.20`이 켜져 있는 동안 정책 플랜의 **모든 knot z가 태그 바닥 위 0.20 m로 고정**되고 정책 자체의 dz는 버려진다. 조종자 요청("오늘만 잠시, 내일 삭제") — yaw 응답 테스트에서 핸드헬드 시연의 하강(dz)이 참조를 바닥으로 끌지 않게.
- **어디**: `rov_gui/control/workers.py` 정책 intake(`compose_plan` 직후, `msg.p_ned[2, :] = z_ned`), `rov_gui/control/geometry.py` 기본값·검증(0 < h ≤ 1.5 m 또는 null), `hw_mpc.yaml`.
- **경계**: arm마다 로그 `ctrl: Z HOLD …`, 플랜 레코드마다 `z_hold_applied`, meta `policy.config.z_hold_above_floor_m`. **이 키가 켜진 런과 꺼진 런의 z 지표를 합산하지 말 것.**
- **되돌리기**: `hw_mpc.yaml`에서 `z_hold_above_floor_m: null`(또는 줄 삭제) → 이 항목 삭제. 코드는 두어도 무해(null이면 아무것도 안 함)하나 테스트 `test_policy_z_hold_pins_every_knot_and_says_so`와 함께 지워도 된다.
- **닫힌 루프 주의**: 관측 모드에선 그림만 바뀌지만, `--policy-observe` 없이 날리면 NMPC가 이 z를 실제로 추종한다. 높이의 프레임은 "태그 바닥 위, 양수 = 위"이고 datum 변환은 p0[2]만큼의 오프셋(테스트로 고정: p0 z −0.205 → datum +0.005).


### 단일태그 fix가 폴백 fix 직후(Δt 0.8 ms)에 소비되면 속도 추정이 47.9 m/s로 튀어 상태가 2 m 점프하고 렌치가 ±30 N 포화한다 (2026-09-14)
- **증상**: `data/20260914/0914_202238/mpc_202610.csv` rows 891–929 (t 44.6–46.5 s, 20:26:55): nav 소스가 second(RGB 폴백)→main(C3)으로
  바뀌는 순간 상태 px 0.214→1.369 m 점프, 2.36 m/s coast → uX/uY/uN ±30/±30/±10 포화 29틱, solver status 4 1회, pwm 1351–1695 µs,
  batt_a 4.5–5.0 A, 선체가 물리적으로 걷어차임(yaw +8/−16°, sway 0.27 m) 뒤 ~8 s 링잉 [측정]. 소형 사례: `mpc_202951.csv` t 53.69 s
  (1-tag fix Δt 2.4 ms → 5.3 m/s, 0.24 m 점프), `0914_181425/mpc_181425.csv` t 18.28–18.43 s (4틱 ±30 N).
  기록된 fix 자체(`nav_202611/fixes.csv` rows 1286–1288, x_ned)는 정상이고 점프는 소비된 STATE에만 있다.
- **원인(추정, [유도] — 코드 경로는 맞고 수치 재현은 안 함)**: `rov_gui/control/state_assembler.py:193-197`
  `v_new = (p_tag − prev_p) / dtc`에서 dtc = 두 fix의 t_capture 차. 폴백 fix의 t_capture는 host 도착시각이라 main fix와 거의 같은
  시각에 다른 위치(1-tag PnP 수 cm 오프셋)로 들어오면 dtc≈1 ms로 나눠 v가 수십 m/s가 되고, alpha 0.6 EMA를 거쳐
  `p += v·age`(속도 브리징, :200-204)가 한 tick에 상태를 미터 단위로 옮긴다. 소스 전환/1-tag에서만 발생.
- **임시 대응**: 없음 — 정책 런 중 C3가 태그를 못 보는 구간(202610/202718에서 틱의 56/64 %가 폴백)이 길수록 노출. 폴백 구간의
  `tag_age_s` 0.03 s는 host 도착시각 기준이라 신선도 지표로 믿지 말 것.
- **제대로 고치는 법**: (1) 소스가 바뀐 첫 fix에서 `_v_ned` 리셋(또는 prev_p를 새 소스 기준으로 재시딩), (2) dtc 하한(예: 0.5·공칭
  프레임 간격) 아래면 속도 갱신 생략, (3) |v_new| 물리 상한(예: 0.5 m/s) 클램프 + 이벤트 로그, (4) 1-tag fix는 속도 갱신에 안 쓰기.
  회귀 테스트: second→main 전환 Δt 1 ms + 4 cm 오프셋을 넣어 상태 점프 < 5 cm 확인.

### 스러스터 출력이 **1.4 s 동안 얼어** 기체가 최대 역토크에도 55° 돌았다 — 명령 경로 hiccup (2026-09-14)
- **증상**: `data/20260914/0914_190226/mpc_190230.csv` t 3.62–5.03 s: 보고 PWM 8개가 `1554/1500/1400/1654/…`로 **1.4 s 동일**(평소 0.149 s마다 갱신,
  `pwm_age_s` 0.11–0.22 s로 텔레메트리는 신선), 그 사이 명령 축은 `ax_yaw` +0.40→−0.50, `ax_sway` −0.38→+0.06으로 움직였고 `uN`은 −10 N·m
  포화. 기체 yaw는 +14°→−42° (55°, 최대 45°/s)로 돌았다가 5.1 s에 PWM이 갱신되며 복귀 [측정]. 접근 walk 중(sway −22 N)이라 위치도 0.5 m 밀림.
- **빈도**: 같은 날 세션 전체에서 "PWM 동일 ≥0.35 s + 축 명령 변화 >0.05" 구간 6개(0.35–0.75 s: `mpc_171212` 52.0 s, `mpc_172142` 1.9 s,
  `mpc_175628` 24.5 s, `mpc_175912` 18.6 s, `mpc_175227` 1.8 s; 1.4 s는 이 한 번). 18:14 정책 런 세 개에는 없음.
- **원인(추정, [유도])**: 텔레메트리는 오는데 서보 출력만 안 바뀌었으므로 ArduSub가 새 MANUAL_CONTROL을 못 받고 마지막 입력을 유지한 것
  (테더/UDP 드롭 또는 송신 스레드 정지). 제어 tick(`tick_ms` 2–6 ms)은 정상이라 워커 쪽은 아님. 500 ms deadman이 발동했다면 1500(중립)이
  됐어야 하는데 값이 유지됐다.
- **임시 대응**: 없음. 접근 walk처럼 큰 수평 추력 중에 걸리면 크게 돈다. 큰 스텝 실험은 사람 손이 E-STOP 위에 있을 때.
- **제대로 고치는 법**: 송신 측에 "보낸 축 vs 보고 PWM" 불일치 감시(예: 0.3 s 이상 PWM 정지 + 축 변화 시 WARN·이벤트 기록), MANUAL_CONTROL
  송신 타임스탬프/간격을 CSV에 남겨 드롭이 송신인지 수신인지 가르기.

### `--record-stereo` 레코더가 **첫 engagement의 트리에 고정**된다 — LOW 런타임 전환 뒤 다른 트리에 섞인다 (2026-09-11)
- **증상**: `HardwareBackend`의 fstereo 워커가 `stereo_rec`를 한 번만 arm하고(`_rec_started`, `rov_gui/backends/hardware.py`
  ~3402) 폴더는 `run_dir_fn = mpc._run_dir`로 첫 프레임에 한 번 해석된다. 2026-09-11부터 LOW를 패널에서 바꿀 수 있어
  `_run_tree()`가 engagement 사이에 물속 잎 ↔ `_observe` 잎을 오가는데(2026-09-14부터 접미사; 전엔 별도 트리), 원본 스테레오 쌍은 첫 폴더에
  계속 쌓인다 — 종류 분리가 풀링 가드인 기록 경계를 넘는다. 같은 결함의 depth 레코더(`--record-depth`)는 같은 날
  `PolicyWorker._rotate_recorder_if_moved`(폴더가 바뀌면 닫고 새로 연다, `meta policy.depth_record.previous`)로 고쳤다.
- **임시 대응**: `--record-stereo` 세션에서는 LOW를 바꾸지 말고(바꿔야 하면 스테이션 재기동), 바꿨다면 `stereo/`의 t_capture를
  각 engagement의 `mpc_*.csv` 시각과 대조해 분리.
- **제대로 고치는 법**: fstereo 워커에 "새 engagement" 트리거(PolicyState epoch이 없으니 `MpcStatus.engaged` 상승 에지나 버스
  신호)를 주고, depth 레코더와 같은 폴더-비교 회전을 넣는다. 같은 폴더 재-arm은 index.csv를 `w`로 덮으니 폴더가 **다를 때만**.

### 실기 DP가 물체 오른쪽 오프셋에 **무반응**이고, 분포 밖 장면이면 yaw 헤드가 **좌 −1.4~−2.6°/plan 기본값**으로 흘러간다 (2026-09-08)
- **증상**: 0908_112101(물체가 헤딩 오른쪽 +2.5~3.3°, 0.9 m 앞)에서 정책이 처음 18 s는 우로 조금 돌다 27 s부터 좌 −2°/plan을
  요청해 끝에는 물체 방위보다 11° 왼쪽; 오늘 15런 전부 런 평균 yaw 의도가 좌(−2.0 ± 0.4°/plan)
  [측정: data/20260908/0908_112101/plans.jsonl, 같은 폴더 diag/out_regress.txt].
- **원인(오프라인 재현 r 0.92 위의 ablation)**: (1) 캔 blob을 왼쪽으로 −20/−40/−60 px 옮기면 −2.6/−5.1/−6.3°, 오른쪽으로 +20/+40 px면
  +0.2/−0.1° — **한쪽 반응**, 기준선은 화면 중앙이 아니라 턱 기둥(u≈150/224) [측정: diag/ablate_out2.txt]. (2) proprio 전부 항등·폭·페어링·
  pitch·depth 배율·좌우반전 모두 |Δ|≤0.3°이고, 분포 밖 영상이면 무엇이든(좌우반전한 **학습** 영상도 −2.2°) −1.4~−2.6°로 흘러간다 =
  yaw 헤드의 OOD 기본값; 라이브 영상은 방향 감도를 학습 대비 ~3.5× 약화 [측정: diag/taskD_out_ablation_analysis.txt]. (3) 112101에선
  기체가 0.3 m 떠올라 바닥이 멀어진 것(inverse-depth 기울기 0.52→0.26)이 트리거 — 캔·그리퍼 없는 합성 평면만으로 같은 시계열이 재현
  [측정: diag/ablate_timeseries.png]. Newton 턱이 화면에 있는 것이 ~0.5~0.7° 기여. (4) 학습셋은 조향을 담고 있으나(방위→dyaw r 0.55)
  **손끝(+4.1° = 핸드헬드 TCP 횡오프셋 0.0355 m)** 을 겨눈 것이고 캔 방위 p99가 +8.5°라 ROV가 보인 +9~13°는 예 0건
  [측정: diag/taskB_out.txt]. 배포 ep195/model은 held-out에서 yaw corr +0.64·자기일관성 +0.94(학습은 됨; ep10 EMA의 "≈0"은 인용 금지).
- **임시 대응**: 기체를 낮고 수평하게 유지(STAB 또는 아래 데드밴드 항목 해결)해 바닥 외형을 학습 범위에 두고, 물체를 손끝 set-point(+4°)
  근처나 약간 왼쪽에 두고 시작; 실기 yaw 통계는 STAB 런(정책 yaw 미실행)과 합산 금지.
- **제대로 고치는 법**: 횡 오프셋이 큰(±15°) 시연과 ROV 턱이 보이는 관측을 수집하거나 턱 영역을 일관되게 마스킹; TCP 횡오프셋을 ROV 기하(y=0)에
  맞춰 라벨을 재생성; 분포 밖 감지(바닥 기울기·근거리 덩어리 위치)로 정책 출력을 게이트.
- **2026-09-08 검증 정정(반박 2렌즈 통과분)**: (1) "오른쪽 무반응"은 맞으나 **실기 영상 탓이 아니다** —
  학습 영상에서도 같은 한쪽 응답이 나온다(아래 "정책 yaw 응답이 한쪽뿐" 항목). (2) "분포 밖 기본값"은 절반만
  맞다: 방위 0°의 좌편향은 **시연 라벨 자체에 −1.85~−2.17°/plan** 있고(손끝이 렌즈보다 3.55 cm 오른쪽),
  실기 영상이 거기에 −2.2~−2.7을 **더** 얹는다. (3) **기수 정렬은 분포 밖이 아니다** — 방위 0°는 학습 접근
  분포의 16.5 퍼센타일(±2° 안에 19.1 %)이고, 오히려 조준점으로 권했던 +12°가 99.4 퍼센타일·+13.6° 위는
  학습 창 0개다 [측정: `.../diag/taskB_windows.npz` 재계산, `q1_refuteX_recompute_out.txt`].

### 정책 미션에서 추종자 명령이 **세 축 모두 ESC 정지대역 안**이라 기체가 표류한다 — `anchor: measured`가 오차를 못 키운다 (2026-09-08)
- **증상**: 0908_112101 지면속도 0.008 m/s(요청 0.047), +0.30 m 상승(정책은 매 플랜 하강 요청); surge |ax|<0.096 100 %(uX p50 1.4 N),
  heave 92 %(uZ ≤5.9 N), yaw 96 %; 기체 보고 pwm_dev p50 10 µs·max 20 µs < ±25 µs [스펙] [측정: 0908_112101/mpc_112101.csv, diag/taskE_out.txt].
  상승은 "추력에 맞선" 것이 아니라 추력 0 + B−W ≈ +0.1 N 표류이고, 같은 날 MANUAL 런은 같은 sub-데드밴드 명령으로 −0.006~+0.029 m/s로
  제각각 떴다(부력·트림·정책에 귀속 불가).
- **원인**: anchor measured + 1 s 플랜(도착 시 0.5 s 낡음)이라 참조가 0.5 s마다 실측 pose에 재앵커 → 추종 오차가 플랜 변위(xy p50 3 cm, z ≤10 cm)에
  묶이고, 추종자 강성 uX ≈ 35.6 N/m·e·uZ ≈ 64.9 N/m·e로는 5.8~7.5 N 문턱에 xy 0.15~0.2 m 오차가 필요해 절대 못 넘는다. hold_tail track은
  plan 구간 |uX|를 0.05→1.21 N으로 올렸을 뿐(데드밴드의 1/4). STAB은 자세루프가 수직 추진기만 34~37 µs로 깨워 상승이 적다(수평은 여전히 표류).
- **2026-09-08 조치(절반)**: 예측모델의 수평 항력을 실기 값으로 올렸다 — `hw_mpc.yaml`
  `plant.linear_damping [86.7, 133.8, ...]`(= `axis_gain/0.692`, 모델의 21.5배). 오차 0인
  0.05 m/s 참조에서 uX 2.74 → 4.80 N(axis 0.046 → 0.080) [측정: 오프라인 솔버].
  **단, 그 이득은 참조가 움직이는 동안에만 있다** — LQ 위치 강성은 sqrt(q/r)이라 항력이
  안 들어간다. 실기 기하로 다시 재면(오차 4.7 cm, v_ref 0.047, **hold_frac 0.885**)
  4.62 → **4.79 N(+3.7 %)**뿐이다 [측정: .../0908_112101/diag/drag_sweep.py]: 플랜이 3 s
  호라이즌의 12 %만 덮고 나머지는 참조가 서 있어서 피드포워드가 가려진다. 즉 항력 값은
  이제 맞지만 **기체를 움직이게 하는 열쇠는 hold tail / 플랜 나이**이고, 데드밴드
  (axis 0.096 = 5.8 N)는 항력이 아니라 이 블록이 못 채운다. 실기 미검증.
- **A/B #2 (0908_165517, 같은 장면 PID↔mpc_tuned)**: PID 0.73 m/40 s(uX p90 8.9 N, 무릎 초과 30 %) vs mpc_tuned 0.01 m/45 s
  (uX 2.5–3.5 N 고정, 0 %). 정체는 피드포워드가 아니라 **비례 스프링**: uX ≈ 40 N/m × (플랜 하나의 변위 0.086 m) = 3.1 N이
  LQ 최적이고, 속도 피드포워드는 호라이즌 87 %가 v=0 hold라 0.56 N. PID는 kd·v_ref(6–8 N)+4 N 적분기의 73회/분 펄스로 움직였고
  0.029 m/s 플랜을 받던 처음 10 s엔 똑같이 멈춰 있었다 [측정: `.../0908_165517/diag/fwd_A_out.txt`].
- **랜딩(2026-09-08, 실기 미검증)**: `policy.hold_tail: extrapolate`(플랜 끝 너머를 마지막 구간 속도로 외삽, v_max·0.3 m·박스 클립,
  live-only) + `policy.along_scale: 1.0`(정책 미션 동안만 along 가중 75→300, surge만). 오프라인 폐루프(실제 acados + 측정 데드밴드
  법칙, 그 런의 88플랜): track은 q_scale 8에서도 0.017 m/s, 배포 세팅 0.055 m/s(플랜의 64 %)·오차 1.3 cm·반전 0.16/s
  [측정: `.../0908_165517/diag/fwd_C_out.txt`]. 남는 것: 0.04 m/s 플랜에선 0.008–0.02 m/s뿐(FF 2.5 N), STAB 헤딩홀드의 T2/T4
  역부호 preload(surge axis ~0.16 전엔 대각선 한 쌍만 밈)는 하네스에 없음 — 풀 A/B는 per-thruster pwm·yaw를 같이 볼 것.
- **임시 대응(대안, 운영자 결정 필요)**: `anchor: leash` + `anchor_leash_m 0.10`(오차를 10 cm까지 키움; 반박자 하네스 0.050 m/s@0.085,
  0.029@0.040, 무릎 0.13에도 견딤 — 단 knot0 점이 선체를 떠나고 정책이 본 pose보다 10 cm 앞에서 플랜을 합성, 2026-09-07의 measured
  결정을 되돌리는 것), `policy.q_scale 4`(같은 surge, sway 1200→4800·heave 150→600 동반), STAB engage(수직만).
- **제대로 고치는 법**: 할당 단에서 데드밴드 보상(최소 PWM 피드포워드 또는 실측 추력 맵), 혹은 per-thruster 인터페이스.
- **미해명 관측**: 같은 날 MANUAL 110419는 yaw 명령 ~0(|ax_yaw| p50 0.032)인데 1.6°/s 좌회전, STAB 105945도 0.15°/s 좌 — 명령되지 않은 좌회전
  토크(테더·추진기 비대칭?)가 있어 112101 순 13° 중 최대 3.6~6°까지 기여 가능 [측정: 0908_105822/mpc_110419.csv, diag/taskE_yaw_out.txt].

### 카메라 광축과 턱이 **가로로 3.4~5 cm 어긋나** 있다 — CAD 체인엔 없다 (2026-09-08)
- **증상**: obs에서 근거리 그리퍼 덩어리의 열 중심이 세 기록 모두 **obs u 150.7 / 150.8 / 151.3**
  (sd 0.2~1.3)인데, CAD 체인(`cam_t_flu [0.30584,0,0.10501]`, jaw `[0.4165,0,-0.17]`(2026-09-08부터 x 0.502이지만 이 열 예측엔 y=0만 든다), 둘 다 y=0)은
  **광축 열 116.8**을 예측한다. 차이 +34.5 px = +10.3°. 바닥 평면에서 잰 카메라 롤(+3.0~5.9°)은 그중
  4~8 px(12~23 %)만 설명하고, 남는 +26~30 px는 **가로 오프셋 34~50 mm**를 뜻한다
  [측정: `data/20260908/0908_112101/diag/q1_jaw_out.txt`, `q1_roll_out.txt`].
- **왜 중요한가**: `policy.tcp_offset_cam_m`이 x=0(턱이 광축 위)이라는 전제로 계획→NMPC 변환이 돌고,
  정책 관측의 좌우 해석도 이 전제 위에 있다. 3.4~5 cm면 62 mm 턱 개구의 절반을 넘는다.
- **아직 확정 아님**: 덩어리 중심이 턱 자체인지(팔·마운트가 섞였는지), FS depth의 왜곡인지 미분리.
- **다음에 할 일**: 자를 물고 실측하거나, 광축 위 아는 위치에 표적을 놓고 obs 열을 확인.
- **2026-09-06 항목 정정**: "그리퍼가 육상과 반대쪽(위)에 있다"는 **반증됐다** — 그 캡처는 벤치에서
  기체가 기울어져 있었고(그 항목 자신이 경고), 첫 실수중 프레임에서는 근거리 행 중심이
  **live 0.900 ± 0.020 vs train 0.898 ± 0.081, 100 %가 학습 범위 안**이다(같은
  `depth_compare.py:408-425` near_row) [측정: `.../diag/q1_ood_report_out.txt`]. 남은 차이는 위아래가
  아니라 **1.8배 크고 14 px 오른쪽**이라는 것뿐이고, 그게 위 항목이다.
- **교차 참조 (2026-09-08, 턱 실측에서 나온 후보 설명)**: obs 격자는 rect_LEFT 모노 그리드이고 그 카메라는
  컬러 카메라에서 37.5 mm 떨어져 있다(data/20260908/0908_180453_observe/mpc_180453.meta.json
  `fstereo.rig.color.t_rectleft_to_color_mm` = [−37.5, …]; 같은 런의 policy_obs/meta.json에는 이 키가 없다); 컬러 카메라 PnP로 잰
  턱에 문 병의 축은 기체 축 **오른쪽 0.011 m**다 [측정: data/20260908/0908_180453_observe/policy_obs/rgb/000160.jpg,
  같은 런 프레임 0/80도 동일]. 37.5 + 11 ≈ 49 mm가 위 34~50 mm에 맞는다 — 후보일 뿐 미확정(기준선
  방향 부호와 "덩어리 = 턱" 여부 미확인).

### C3 외부파라미터 cam_t_flu의 x/z가 의심된다 — 렌즈가 ~9 cm 뒤·5 cm 위일 수 있다 (2026-09-08)
- **발견 경위**: 0908_180453(POLICY OBSERVE, 기체 44 s 정지, 매트 위, 기수 5.3° 상향)에서 조종사가 턱에 문
  병을 바닥 태그 58 중심에 놓았는데, 화면의 턱 링이 태그 58보다 heading 방향 0.138 m 뒤에 그려졌다
  (yaw-only 마커; 기수 상향까지 넣은 3-D로는 0.123 m) [측정: data/20260908/0908_180453_observe/
  mpc_180453.meta.json datum + policy_obs/rgb/000160.jpg 재해석, config/tag_map_full.yaml 태그 58; 재해석 스크립트·투영 그림·출력은 같은 런의 diag/ (solve_frame.py, jar_geom.py, pixskeptic_refined.py, proj160_zoom.png, *_out.txt)].
- **측정은 선다, 귀속이 열려 있다**: 태그맵·PnP(rms 2.0 px, 태그 부분집합 {48,59}/{59}로도 heading 방향
  1 cm 안정)·틸트(rp_residual 2.4°)·수직(렌즈가 매트 위 0.343 m vs CAD 체인+기수각 0.350 m, 7 mm)·마커
  변환(body_to_map)까지 전부 재현된다 [측정: data/20260908/0908_180453_observe/mpc_180453.csv
  `pnp_rms_px`·`rp_residual_deg`·`n_tags`; 부분집합 재해석과 렌즈 높이 0.343 m는 같은 런 policy_obs/rgb/000160.jpg
  재해석, 0.350 m = CAD 0.322 + 0.306·sin 5.33° [유도]]. 반증된 것은 **"렌즈가 턱 중심 110.66 mm 뒤"라는 CAD 오프셋 하나**
  — 그 턱은 360행 프레임의 행 ~420으로 투영되는데 턱은 행 265~360에 보이고, 문 병의 축은 렌즈 앞 0.196 m
  (프레임 0/80/160: 0.196/0.194/0.196, 범위 0.187~0.204)다 [측정: 위 프레임 + 0908_170428/policy_obs/rgb/000000.jpg
  (병 높이 0.084 ± 0.006 m) + 0908_175151/policy_obs/rgb/000000.jpg (열린 턱 광선: 손끝 0.215~0.234, 팜 0.165~0.180)].
  높이 무관 검증도 있다: 문 병 뚜껑의 겉보기 폭 비(60 px vs 원거리 35~36 px = 1.67~1.71)가 실측 가설(1.66)에
  맞고 CAD 턱(2.38)을 기각한다. 그런데 이것은 두 가지로 읽힌다 — **(a)** `cam_t_flu`가 맞고 턱이 COM 앞 0.502에
  있다; **(a')** `cam_t_flu.x`가 ~0.09 m 크고(렌즈 실제 x ~0.21~0.23) z가 ~0.05 m 낮으며(실제 ~0.155) 시뮬 턱
  0.4165가 대략 맞다(0.22 + 0.196 = 0.42). 두 해석 모두 턱을 **렌즈 + [0.196, 0, −0.275]**에 두므로 턱/TCP는
  렌즈에 앵커해 고쳤고(`rov_shape.LENS_TO_GRIP_FLU_M`, `policy.tcp_body_flu_m` 0.502), 이 항목은 **남은 절반**이다.
- **(a')를 가리키는 정황** (전부 간접, 이 키를 잰 것은 없다): (1) `second_cam.t_flu [0.2663, …]`는 BodyAlign이
  **C3 체인에 상대적으로** 잰 값인데, 시뮬 선체 메시(base_link 프레임)의 돔 정점은 x 0.230~0.231, 전자장비
  튜브 끝은 ~0.176이다 [유도: bluerov2_mujoco_marinegym/meshes/rov_body_white.obj / rov_body_black.obj 정점 범위]
  — 돔 카메라 렌즈가 돔 정점보다 3.5~4.5 cm 앞일 수는 없고, 조종사의 테이프는 0.18이었다(hw_nav.yaml의
  "8.6 cm short" 주석). (2) 조종 명령으로 게이트한 순수 yaw 피벗(스러스터별 PWM, surge/sway ≈ 0)에서 **보고되는
  body 원점이 yaw 피벗보다 0.117~0.130 m 뒤**다 [측정: data/20260908/0908_170428_observe/mpc_170428.csv
  n=192, 0908_175151/mpc_175151.csv n=43; 스크립트·출력 data/20260908/0908_180453_observe/diag/skeptic_lever_pwm2.py, yaw_pivot_out.txt]; 피벗이 합성 COM(+0.035)이라면 (a')는 −0.13,
  현행 값은 −0.035를 예측한다. (3) 수직: BodyAlign의 돔 카메라 z −0.019 vs 메시 전자장비 튜브 축 ~+0.03,
  체인의 밑면 −0.217 vs 메시 −0.165~−0.167(white/black) [유도: 같은 메시] — 체인이 ~5 cm 낮은데 그 앵커(턱 z −0.17)가 시뮬
  추정값이다. (4) (a')면 CAD의 110.66 mm는 그립점 ~9 cm 뒤, 즉 턱 **베이스/피벗**까지의 거리로 읽혀 "틀린
  숫자"가 아니라 "다른 점"이 된다.
- **(a')가 맞다면 결과**: 보고되는 body 원점이 실제보다 heading 방향 ~0.09 m 뒤(시뮬 턱 0.4165가
  정확하다면 0.085; 렌즈 x 0.21~0.23이면 0.076~0.096 [유도])·~0.05 m 아래 → **모든 기록
  위치**가 그만큼 밀려 있고, NMPC가 규제하는 점과 플랜트 COM의 격차는 알려진 0.043 m가 아니라 ~0.13 m
  [유도], 마커의 선체는 실제보다 ~9 cm 뒤에 그려진다(턱은 맞음). 시뮬 GRIP_POS/JAW_POS는 (a')면 대략 맞고
  (a)면 렌즈-앵커 턱보다 8.5 cm 뒤다 — 플랜트 합성은 손대지 않았다.
- **왜 안 바꿨나 / 임시 대응**: 이 값은 모든 기록 위치의 기록 경계이고(hw_nav.yaml 주석), 위 정황은 전부
  간접이다. 턱/TCP만 렌즈에 앵커했다(끝). 위치 절대값을 cm 단위로 인용할 때 이 항목을 병기할 것.
- **제대로 고치는 법**: 기체에서 줄자 두 번 — (1) 프레임 앞끝(돔 정점) → C3 렌즈: 현행 값이면 렌즈가 돔
  정점보다 ~7.5 cm **앞**, (a')면 돔 정점 ±2 cm; (2) C3 렌즈 → 그리퍼 팜. 바꾸면 같은 커밋에서
  `policy.tcp_body_flu_m = cam_t_flu + [0.196, 0, −0.275]`를 재유도(또는 `tcp_offset_cam_m`을 설정해 렌즈 위치
  소거), meta `hardware.cam_t_flu`가 경계를 기록한다.
- **2026-09-09 CAD 재export 정황 ((a') 쪽)**: 사용자의 Onshape 문서 "BlueROV2" 탭 "mujoco"(카메라·upper mount·Newton
  그리퍼 포함)를 onshape-to-robot으로 재export하고, 7월 등록(R0·C_ASM, 선체 파트 포즈가 7월과 바이트 동일)으로
  base_link에 놓았다 [유도: assets/CAD files/onshape_export_20260909/payload_frames_20260909.json; 스크립트는
  같은 등록 상수(tools/process_c3_mesh.py)]. CAD가 말하는 값: **렌즈면 중심 [0.194, 0.006, +0.133]**, 광축 40.0° 하향,
  베이스라인 수평; **턱 쌍 중심 [0.364, 0.006, −0.127]**(턱 x 0.329~0.404); 렌즈→턱 중심 [+0.170, 0, −0.260].
  현행 `cam_t_flu` [0.306, 0, 0.105]보다 렌즈가 **x 0.11 m 뒤, z 0.03 m 위**, `tcp_body_flu_m` 0.502보다 턱이
  0.10~0.14 m 뒤 — (a')의 예측(렌즈 x ~0.21~0.23, 턱 ~0.42)과 같은 방향이고, 렌즈-턱 상대값(0.17~0.21 m 앞)은
  이미지 실측 0.196과 정합한다. 한계: 원점은 7월 시뮬 스킨 등록의 COM이고 CAD의 그리퍼/마운트 배치는 사용자
  모델링(export 20분 전까지 mujoco 탭에서 인스턴스 Unfix/Drag 편집 이력)이라 **줄자 실측을 대체하지 않는다**.
  - **0908 근거를 CAD로 재검산** [유도: assets/CAD files/onshape_export_20260909/reanalysis_20260909.json]: (1) yaw 피벗 —
    CAD는 보고 원점이 COM보다 **0.111 m 뒤**라고 예측, 실측 0.117~0.130 (0.6~1.9 cm 안). (2) 돔 카메라 — BodyAlign 0.266 −
    0.111 = 0.155, CAD 선체 돔 영역 최전방 x 0.208(전체 최전방 0.226)·테이프 0.18과 2~3 cm 안. (3) 턱 — 렌즈+0.196 = 0.39,
    CAD 턱 중심 0.364·팁 0.404, 시뮬 0.4165 → 현행 0.502가 0.11 m 앞. (4) **수직은 반대 방향**: CAD 선체 최저점 z −0.164
    (시뮬 메시 −0.165~−0.167과 일치)에서 렌즈는 정지 시 바닥 위 0.297, 기수 5.33° 상향 시 0.325 m인데 0908 실측은 0.343
    (바닥 z −0.016이면 0.359) — CAD가 2~3.5 cm **낮다**(옛 체인 0.350은 0.7 cm). 즉 렌즈 z는 CAD +0.133보다 높은 +0.15~0.16
    [유도]일 가능성이 있고, upper mount의 CAD 배치가 그만큼 낮게 그려졌거나 매트/피치 가정 문제. 결론: 수평은 (a')로
    수렴(x −0.11), 수직은 +0.03~+0.05 사이에서 미정 — **줄자로 결정**: 돔 정점→렌즈는 CAD면 렌즈가 정점보다 1~3 cm
    **뒤**, 현행 체인이면 8~10 cm **앞**.

### 정책 yaw 응답이 **한쪽뿐**이다 — 오른쪽 오차는 영원히 못 고친다 (2026-09-08)
- **증상**: 캔을 좌우로 옮기며 잰 knot-15 dyaw가 조준점 **왼쪽에서는 +0.15~0.25°/°** 로 반응하지만
  **오른쪽에서는 +0.02~0.05°/°** 로 죽는다. |dyaw| ≤ 1°인 평탄역이 +8°부터 최소 +20°까지 이어진다.
  같은 스윕을 **학습 영상**에 돌려도 같은 모양이라(왼쪽 +0.146, 오른쪽 +0.020) 실기 영상 탓이 아니라
  **정책 성질**이다 [측정: `.../diag/q1_sweep1_out.txt`, `q1_sweep_train_out.txt`].
- **덧붙는 상수**: 방위 0°(기수 정렬)에서 시연 라벨 자체가 −1.85~−2.17°/plan 좌(시연자 손끝이 렌즈보다
  3.55 cm 오른쪽 = +4.1°가 라벨의 영점이라서), 학습 영상 위 정책은 −2.32°, **실기 영상 위에서는 −3.7~−4.97°**
  — 즉 실기 영상이 −2.2~−2.7°/plan을 더 얹는다 [측정: `taskB_windows.npz` 재계산, `q1_sweep_train_out.txt`,
  `q1_sweep1_out.txt`].
- **함정**: 실기 폐루프의 평형점(명령 0)은 **방위 +12~18°**인데 그 구간은 학습 창이 **0개**(방위 > +13.6°)다.
  거기서 정책이 조용한 건 맞아서가 아니라 **반응을 멈춰서**다. 반대로 기수 정렬(방위 0°)은 학습 분포의
  **16.5 퍼센타일로 분포 안쪽**이다 — 캔을 오른쪽에 두는 건 수리가 아니라 바이어스의 **수치 상쇄**다.
- **임시 대응**: 첫 시험은 캔을 시연 분포의 중심인 **방위 +4°**(0.7~1.0 m에서 오른쪽 5~7 cm)에 두고,
  고도를 유지해 캔을 관측 대역(0.2~3.0 m) 안에 붙잡아 둔다. 그래도 −3.7°/plan쯤은 남는다.
- **제대로 고치는 법**: (a) ROV의 TCP(턱이 광축 위, 위 항목의 실측 오프셋 포함)로 라벨을 재생성해 재학습,
  (b) 좌우 오차가 큰(±15°) 시연 수집, (c) 임시로는 intake 크롭을 +34.5 px 이동(방위 0에서 −4.66 → −0.98°/plan,
  단 절대 열에만 반응하므로 이동한 만큼만 벌 뿐이다 [측정: `.../diag/q1_sweep_shift2_out.txt`]).

### 배포된 EMA 가중치가 held-out에서 **1.9배 나쁘다** — 원인은 BatchNorm running stats (2026-09-06)
- **증상**: 2026-09-03 실기에서 DP 참조가 우유부단했다. 그 체크포인트가 로드하는
  `state_dicts.ema_model`이 **검증 세트에서 비-EMA `state_dicts.model`보다 크게 나쁘다**
  [측정: 검증 4개 에피소드(6/32/48/56, `get_val_mask(75, 0.05, 42)`) 219 윈도, DDIM 16스텝]:

  | 가중치 | action MSE | pos MSE | pos RMSE |
  |---|---|---|---|
  | `ema_model` (**배포 중**) | 0.001582 | 0.000956 | 30.9 mm |
  | `model` (비-EMA) | 0.000828 | 0.000435 | 20.8 mm |

  실기 설정(DDIM 8스텝)에서도 같다: 0.001535 vs 0.000842. 즉 **1.82–1.91배**.
- **분해 실험이 원인을 확정했다**: EMA 사본의 **파라미터는 그대로 두고 BatchNorm buffer
  108개만** 학습된 것으로 갈아끼우면 0.001535 → **0.000896**(pos RMSE 20.9 mm)로,
  격차의 **92%가 회수된다** [측정 2026-09-06, 같은 219 윈도]. 즉 **EMA 평균 자체는
  멀쩡하고, 통계가 전부였다**. "EMA가 나쁘게 학습됐다"는 해석은 틀렸다.
- **원인 (확정)**: `EMAModel.step`은 `named_parameters()`만 순회하므로 **buffer를 평균하지
  않는다**. 그리고 `transformer_obs_encoder.py:120`의 `if use_group_norm and not pretrained:`
  때문에 `use_group_norm: true`인데도 `pretrained: true`라서 **GroupNorm 치환이 조용히
  건너뛰어졌고 BatchNorm이 살아남았다**. 결과적으로 EMA 사본의 `running_mean/var`는
  **timm ImageNet 값 그대로**다 [측정: `num_batches_tracked` — timm `resnet34.a1_in1k`
  pretrained **374,981**, ckpt `ema_model` **374,983**(+2), ckpt `model` **395,128**
  (= +20,145 = global_step+1)]. BN buffer의 model↔ema 최대 상대차 **9847배**.
  eval()에서 BN은 running stats를 쓰므로, **비전 인코더가 depth 관측을 ImageNet RGB
  통계로 정규화하고 있다**.
- **임시 대응 (2026-09-06 구현)**: `--policy-weights model`. 기본값은 `ema_model` 그대로라
  이전 기록은 재현되고, 고른 값은 `describe()`와 런 meta에 남는다 — **다른 가중치로 돈
  런끼리는 합산 금지**.
- **제대로 고치는 법**: (1) ~~가중치 소스를 config/CLI로 노출~~ 완료; (2) 재학습 시 인코더의
  BatchNorm을 GroupNorm으로 실제 치환하거나(=`pretrained` 게이트 제거) EMA가 buffer도
  복사하게 한다; (3) 재선정은 **전체 검증 세트**로 — 현재 `selected.json`의 근거 숫자는
  `next(iter(val_dataloader))` 한 배치(32 윈도)뿐이라 전체 세트보다 1.4배 낙관적이었다.
- **2026-09-07 5-dim action 재학습(`umi_depth_5d`)은 사용자 결정으로 (2)를 손대지 않아 결함을 상속했고,
  그 덕에 같은-에폭 증거가 더 선명해졌다**: epoch 195에서 219-윈도 held-out pos RMSE
  **ema_model 33.0 mm vs model 17.3 mm**(MSE 3.6배), yaw RMS 2.87° vs 2.20°
  [측정: `<run>/heldout_pos_yaw_width_ep0195{,_model}_stride4.log`]. 옛 10-dim ckpt도 29.8 vs 21.0 mm.
  → **2026-09-07부터 배포 기본 가중치를 `model`로 바꿨다**(`geometry.default_policy_block()['weights']`,
  `config/hw_mpc.yaml`·`land_dp.yaml`의 `policy.weights: model`; `--policy-weights`로 덮어쓰기 가능).
  파리티 fixture도 배포 가중치로 다시 만든다(`dp_policy_reference.py --weights`, 기본값이 배포 선택).
  `ema_model`로 돈 이전 런과 **합산 금지**는 그대로.

### NMPC 플랜트는 기체가 **5.7 N으로 가라앉는다**고 믿는데 실기는 떠오른다 — mpc/mpc_tuned가 정책 하강 플랜 아래서 계속 상승 (2026-09-07)
- **증상**: `mpc_tuned` + policy 미션 5개가 전부 위로 흘렀다(pz 0.27–0.54 m 상승, vz p50 +0.015…+0.035 m/s) — 정책은
  1 s당 +0.055 m(p50) **아래**를 요청했고(NED +down, 154 플랜), 앵커 z 오프셋(참조−실측) p50 +0.108 m로 참조는 항상 기체 아래였다
  [측정: data/20260907/0907_180038/mpc_18*.csv, plans.jsonl].
- **원인**: 히브 지령을 회귀하면 다섯 미션 모두 `uZ ≈ −5.6 N − 60 N/m·ez`(R² 0.54–0.94; ez = rz − pz, FLU). **절편 −5.6 N(위로)**은
  `dobmpc/params.py NET_BUOYANCY = −5.71 N`(heavy_gripper가 침강한다는 CAD·벤더 합성값, 실측 0건)과 일치 — 명목 NMPC는 자기 모델의
  중력을 상쇄하므로 오차 0에서 5.7 N을 **위로** 민다. 여기에 T200 데드밴드(~5.8 N)까지 더하면 아래로 추력이 나오려면 ez ≈ −0.19 m가
  필요한데 leash 0.05 m + 플랜 하강폭이 오차를 −0.10…−0.15 m에 묶어 둔다 → 추력 0(|ax_heave| < 0.096이 88–100 % 틱), 실기 부력만큼
  상승. 미션 시작 1.4 s(station hold)도 uZ −5.1…−6.1 N(위). 실기 B−W는 같은 날 −2 N(16:53 mpc hold −6.5 N 지령, 수심 평탄)에서
  > +3 N(16:07 PID +8 N 지령에도 0.25–0.49 m 상승)까지 움직였고 −5.7 N이었던 적이 없다 [측정: 0907_16*/mpc_*.csv, 0907_15*/].
  같은 메커니즘이 16:11–16:13 mpc_tuned 미션 4개(0907_160750/mpc_1611{15,49}, 1612{38}, 1613{15}: +0.22…+0.53 m)에도 있다.
- **대응(2026-09-07, 실기 미검증)**: `hw_mpc.yaml`/`land_dp.yaml` **`vehicle_net_buoyancy_n: 0.0`**(실기 B−W, + 부양) → `HwDobMpc`가
  (모델 − 실기) = −5.71 N을 **상수 히브 렌치**로 acados 파라미터 p(= w)에 넣는다(런타임 입력, 재빌드 없음, 관측기 아님; mpc/mpc_tuned만 —
  dobmpc는 EAOB가 같은 잔차를 추정하므로 미적용). meta `controller.heave_trim` 블록이 기록 경계(없는 런 = −5.7 N 상향 FF를 날린 런);
  CSV `w2`는 EAOB 열로 유지. 모델 플랜트 예측: 중립 기체 station hold에서 무보정 +12.2 cm 위에 정지, 보정 0.00 cm
  [예측: rov_gui/tests/test_path_cost.py::test_a_neutral_vehicle_holds_depth_only_with_the_trim, 데드밴드 없는 플랜트]. 검증은 CSV
  `uZ`의 절편(정지 틱 평균)이 0 근처인지와 policy 미션의 pz 드리프트로.
- **실측(2026-09-08)**: 히브 지령이 데드밴드 안인 자유 상승 창(0908_105822/mpc_110419.csv 0–11 s, pwm_dev ≤ 22 µs)을 플랜트 히브 모델로
  적합하면 B−W = **+0.12 N**(모델 항력), +0.04 N(항력 0), +0.28 N(항력 3배) [측정+유도: 0908_105822/buoyancy_fit_20260908.txt] → 기체는
  사실상 중립이고 `vehicle_net_buoyancy_n: 0.1`로 설정. 같은 날 MANUAL 런은 그래도 0.25 m/15 s 떠올랐는데, 이는 부력 상수가 아니라
  **데드밴드**다(위치항이 ~6 N이 되는 오차 ~0.1 m까지 추력 0). 반면 STABILIZE 런 2개(mpc_105945/110306)는 히브 지령 ~0으로 깊이를
  ±7 cm 안에 잡았다 — 자세 루프가 수직 추진기를 데드밴드 밖에 세워 둔다(pwm_dev p50 35 vs MANUAL 10 µs).
- **남은 것**: 트림은 **상수**라 하루 안에 ±8 N 움직이는 실기 부력(테더·트림)을 못 따라간다 — mpc 계열엔 적분·관측기가 없으므로(사용자 결정)
  세션마다 값을 재는 수밖에 없고(PID depth hold의 정상 uZ가 데드밴드 밖이면 그 값), 데드밴드 5.8 N 안의 부력 오차는 여전히 ~0.1 m 오프셋으로
  남는다(`policy.q_scale`로 강성을 올리면 줄어듦 [유도]). `--policy-observe`(비송신)에선 무관.

### STABILIZE에서 engage — 배선됐지만 **실기 0런**, yaw는 `hold`라 스테이션이 헤딩을 못 돌린다 (2026-09-08)
- `engage.require_mode`가 목록을 받아 `[MANUAL, STABILIZE]`로 열 수 있다(핀 고정·`mode_settle_s`·dobmpc 거부·
  `stabilize.yaw_axis: hold`; README "STABILIZE에서 engage하기"). 세 자문(제어이론·수중운용·정책관측) 판정은 모두
  **conditional**: roll/pitch는 이득, yaw는 의미가 토크→rate로 바뀌어 손실. 검토는 MANUAL 단독 기본을 권했으나
  **운용자 결정으로 기본값이 `[MANUAL, STABILIZE]`**(2026-09-08): STAB 버튼 → START가 그대로 STAB 아래서 돈다.
- 미해결: (1) `hold`에선 heading_follow·circle·정책 dyaw가 동작하지 않고, `torque`는 데드존(axis ≤ 0.10) 경계 hunting이
  [예측]; 제대로 하려면 N→yaw-rate 재해석 + 데드존 역보상인데 PilotGain(위 항목)을 못 읽어 미구현(`rate` 거부).
  (2) `mode_settle_s 2.0`은 [예측]이고, 게임패드의 모드 버튼은 기체로 직행해 스테이션이 못 보므로 그 경로의 전환은 HEARTBEAT(1 Hz)
  지연만큼 늦게 인지된다(패널 경로는 요청 시점에 시계를 찍음). (3) E-STOP/데드맨의 중립 송신은 STABILIZE에서 정지가 아니다(수평·헤딩
  홀드, 추진기 활성) — E-STOP 시 error 로그로만 알리고 DISARM/MANUAL 자동 전환은 별도 안전 검토 대상.
  (4) 펌웨어 버전 미확인 — 4.1.0이면 STABILIZE 자세 목표가 '마지막 yaw 입력 시 자세'로 잠겨 전제가 깨진다. (5) 수직 추진기를
  roll/pitch PID와 나눠 써 접촉 국면 heave 권한 감소 [유도]. (6) 스테이션 UI에 STAB 상태 칩/툴팁 없음(거부 문구만).
- 첫 세션 절차와 측정 항목은 README 해당 절. 그 산출물이 생기기 전엔 STABILIZE 런의 수치를 MANUAL 런과 합산·인용 금지.

### 6-DoF 변형(roll/pitch 추종, `pos_rpy_width` + `engage.attitude_axes`)이 **실기 0런** — 펌웨어 s/t 소비·부호·플랜트가 전부 미확인 (2026-09-26)
- **무엇**: 정책이 `[dx,dy,dz,dyaw,droll,dpitch,width]` 7-dim을 내고 NMPC가 roll/pitch 참조를 추종해 K/M 토크를
  MANUAL_CONTROL 확장축(s = pitch, t = roll, `enabled_extensions=0b11`)으로 보내는 변형. 기본값은 전부 OFF
  (`policy.action_repr: pos_yaw_width`, `policy.attitude_track: false`, `engage.attitude_axes.enabled: false`)이고
  OFF면 4-DoF 경로는 바이트 동일(테스트 고정). 설계·첫 비행 순서는 `docs/DP_6DOF_PLAN.ko.md`. 아래는 켜기 전에 남은 것.
- **펌웨어가 s/t를 읽는지·어느 부호로 읽는지 미확인**: 이 ROV의 ArduSub 버전 자체가 미확인(위 STABILIZE 항목 (4))이고 s/t는
  Sub 4.1.2(2024-02-22)부터만 소비된다 — 4.1.0/4.1.1이면 **조용히 무시**. `joystick.cpp`의 s/t 줄(채널 인덱스·rpyScale·부호·
  pitchTrim/rollTrim 래치)은 원문을 읽지 않았다 [스펙 미확인]. engage 게이트(mavlink20 + AUTOPILOT_VERSION ≥ 4.1.2 + 벤치 프로브
  `python -m rov_gui.tools.attitude_axes_probe` 산출물 JSON 파싱: `tool`·`pass: true`·`mavlink_wire_version "2.0"`·`firmware_version`이
  기체와 일치)와 disarmed 프로브의 전제 "disarmed 상태의 RC_CHANNELS chan1/2가 MANUAL_CONTROL
  override를 반영한다"도 [스펙 미확인] — 안 되면 armed 상태로 물속에서 프로브를 돌려야 한다.
- **중립 프레임의 확장축 0 명시**: 확장 프레임(`enabled_extensions=0b11`)을 한 번이라도 보낸 뒤의 중립 프레임은 `enabled_extensions=0b11`
  + `s = t = 0`을 **명시**해 보낸다(비트를 뺀 프레임에서 펌웨어가 마지막 s/t를 유지하는지 0으로 보는지가 [스펙 미확인]이라 0을 명시하는
  쪽을 택함). 비트가 없을 때의 펌웨어 동작은 여전히 [스펙 미확인].
- **K/M 부호 [가정]**: `engage.attitude_axes.sign {roll: 1, pitch: 1}`는 가정. 뒤집히면 U_MAX 8 N·m까지 양의 피드백
  (복원 ~2.1 N·m/rad [유도] → 전복 가능). 방어선은 첫 물 캡 `first_water_caps [0.1, 0.15]`(≤ 1.35 N·m/rad 복원 대비 전복 불가
  [유도]) + sat-ineffective 인터록(축이 캡에 1 s 붙어 있고 오차가 안 줄면 disengage, 상수 [예측]) + **armed 수중 부호 프로브**.
  부호 프로브 미션 종류(`attitude_sign_probe`)는 **이번 컷에 없다**: arm 게이트는 `engage.attitude_axes.sign_probe` JSON의
  `sign_proven: true`를 요구하므로(`require_sign_probe: true` 기본), 그 파일은 지금은 수동으로(ATTITUDE + SERVO_OUTPUT_RAW 기록,
  roll +0.10 1 s → pitch +0.10 1 s, |rp| > 10° 또는 2 s abort) 만들고 `sign_proven: true`를 적어야 한다. `require_sign_probe: false`로
  우회하면 arm은 되지만 `first_water_caps [0.1, 0.15]`가 강제되고 meta `run.attitude_axes`에 `sign_proven: false`·`caps_in_force`가
  남는다 — 부호가 증명되지 않은 런의 유효 캡은 항상 첫 물 캡.
- **회전 플랜트가 전부 placeholder**: `hw_mpc.yaml plant.linear_damping` roll/pitch 0.07, 회전 부가질량 0.12
  (메모리 heavy-added-mass-provenance), ZG 0.01 고정(params.py) vs heavy_gripper coBM 0.01625, 페이로드 정적 pitch 모멘트 ~1.4 N·m
  [스펙 BlueROVHeavyGripper.yaml] 미모델. 하드웨어에서 자세 루프를 닫아 본 적이 없다 → 첫 물 시험은 정책 없는 **plain mpc 자세
  스텝**(0914 yaw 스텝 시험의 자세판)이어야 하고, 그 전엔 `attitude_q_scale`·`rp_ref_filter`는 손대지 말 것.
- **축 이득 [유도] + 솔버 U_MAX vs 선 캡 불일치**: `axis_gain.roll_nm 13.2 / pitch_nm 7.2`는 heave_n 60 N [예측] × 레버암
  (y ±0.22 / x ±0.12 m, bluerov_heavy.xml)에서 유도한 값. 솔버 U_MAX[3:5] = 8 N·m인데 선이 나르는 최대는 cap×gain =
  2.6 / 2.2 N·m(첫 물 1.3 / 1.08 N·m) → 최적화기는 선이 못 나르는 토크를 계획할 수 있다. EAOB엔 `note_applied`로 캡된 값을
  알리므로 관측기는 진실을 보지만 plain mpc는 큰 자세 과도에서 추종 지연을 보인다. 첫 물 pitch 캡 1.08 N·m는 정적 페이로드
  모멘트 ~1.4 N·m **아래**라 능동 레벨링 런이 M 포화 상태로 상수 오차를 안고 앉을 수 있다(sat-ineffective가 잡아야 함).
  나중 수정: acados를 U_MAX[3:5] = cap×gain으로 재빌드(시뮬 플랜트 바이너리도 같이 바뀜). meta `controller.attitude_ref.u_max_wire_nm`에 기록.
- **pitch 축 데드밴드 리밋사이클 위험**: 무릎 ≈ 0.096 × 7.2 ≈ 0.7 N·m [유도] ≈ 20° 홀드 모멘트(0.47–0.73 N·m [유도]) —
  yaw 디더(2026-09-14)의 자세판. 깊이 유지 중 수직 추진기가 이미 무릎 위에 있는지는 부력 가정에 달렸는데 두 메모리가
  충돌한다(선체 음성 부력 vs 적합 +0.1 N 중립) → 시뮬 S5 데드밴드 플러그인이 첫 증거.
- **`transport: rc_override`(FALLBACK-A) 미구현**: `validate_engage_attitude_axes`가 "not implemented"로 거부. 4.1.2 미만
  펌웨어에서 s/t가 무시될 때의 대안 경로인데 데드맨 의미(RC_OVERRIDE_TIME [스펙 미확인] vs 500 ms)와 4-DoF 경로 자체를 바꾸므로
  자체 벤치 패리티 산출물 없이는 켜지 말 것. STABILIZE lean-angle 캐스케이드도 미구현(attitude_axes.enabled면 STABILIZE 거부).
- **자세 수치는 전부 [예측]**: rp_max 20°, rp_reject 30°, pq_max 0.35 rad/s, rp_jump 5°, leash 3°, div_rp 15°, abort 35°,
  caps 0.2/0.3. 라벨 분포의 첫 산출물은 `data/20260926/0926_175145_offline/gate_pos_rpy_width.log`(0901 육상 데모, 1721 윈도:
  |dpitch| p50/p90/p99/p99.9 1.17/5.53/13.07/18.31°, |droll| 0.70/2.56/5.39/8.17°; 피크 |p|,|q| p90 0.453 rad/s > 0.35 캡이라 200 ms
  격자에서 2/1721 청크가 accept→clip) — 7-dim 학습 런의 held-out 게이트(`dp_policy_offline.py replay --val-only`)와
  `policy_session_replay.py --ckpt D7=…`(수중 런 OOD 프로브)는 아직 안 돌았다.
- **기록 경계는 의도적으로 깨진다**: schema 16부터 미션 CSV 끝 5열(`rroll_deg,rpitch_deg,ax_roll,ax_pitch,rp_track`),
  `policy_plan.csv` 끝 `roll_deg,pitch_deg`, plans.jsonl `rp_tracked`가 항상 붙는다. 이름으로 읽는 이 리포의 도구는 무관하지만
  열 수를 세거나 바이트 비교하는 외부 판독기는 깨진다. `attitude_track`·`attitude_axes.enabled`·`action_repr`·비행 모드·
  `controller.allocation.attitude`(dobmpc w_hat[3:5] 의미가 바뀜)가 다른 런은 **합산 금지**.
- **시뮬 근거의 빈칸**: (i) [해소] "K/M이 조용히 버려질 때 dobmpc가 `div_rp 15°`를 안 넘는다"는 174607 `dropped_*` 행은
  임계 = 참조 크기(15°)라 퇴화 — 인용 금지. 비퇴화 재실행(참조 20° = `rp_max_deg`, 임계 15° = `div_max_rp_deg`)에선 mpc·dobmpc
  모두 0.45 s에 트립(임계 초과 연속 10 샘플 = 스테이션 0.5 s 디바운스; 선체는 20°에 접근하지 않음), dobmpc는 크레딧되었으나
  미전달된 K/M으로 pitch 6.83°까지 흘러 rad_max 14.5 cm(`bluerov2_mujoco_marinegym/recordings/20260926/attitude_hold_211917/
  results.csv`, `dropped_*` 행) — `dobmpc_allowed: false`를 유지하는 이유는 임계가 아니라 이 드리프트다. (ii) S2 hold 그리드는 NONE만 돌았고 C/CDW 셀은 미실행(`verify_attitude_hold.py --modes C,CDW`);
  (iii) S5 데드밴드 플러그인은 heavy(+1.1 N 근중립)만, heavy_gripper(−5.7 N 침강)의 리밋사이클은 미실행;
  (iv) `pos_rpy_width` 체크포인트가 아직 없다(`tools/run_ablations.sh depth_7d_*` 미실행) → held-out 수용·OOD 리플레이·
  `dp_policy_reference.py` 패리티 픽스처 전부 미실행; (v) `tests/test_dobmpc.py main()`은 기존 NU=4 vs 6 실패로 여전히
  red, 새 자세 테스트 6개는 이름을 줘서 개별 실행.

### ArduSub **조이스틱 gain**이 스테이션이 보내는 모든 축을 곱하는데, 읽지도 기록하지도 않는다 (2026-09-07)
- ArduSub은 MANUAL_CONTROL의 x/y/r을 `PWM = 1500 + 0.4·gain·값`, z를 `0.8·gain·JS_THR_GAIN` 배율로 RC override에
  넣는다(`ArduSub/joystick.cpp transform_manual_control_to_rc_override`, Sub-4.1 :55-150, Sub-4.5 동일). `gain`은 부팅 시
  `JS_GAIN_DEFAULT`(기본 **0.5**, 범위 0.25–1.0)이고 패드의 `gain_inc/dec` 버튼(이 스테이션 패드 표 기준 십자키 11–14)이
  즉석에서 바꾼다. 즉 `axis_gain.surge_n/heave_n 60 N`·데드밴드 문턱 axis ≈0.096·슬루 1.5/s 같은 스테이션 캘리브레이션은
  전부 **gain 0.5를 전제로 한 값**이고, 십자키 한 번에 조용히 2배가 된다. 기체는 이 값을 `NAMED_VALUE_FLOAT name="PilotGain"`으로
  주기 송신한다(`GCS_Mavlink.cpp` Sub-4.1 :145) [출처: ArduSub 소스 읽기 2026-09-07, 워크플로 stabilize-mode-llc; 실기 파라미터는 미확인].
- 영향: 런마다 유효 추력 배율이 달라도 meta에 흔적이 없어 데드밴드·축게인 관련 모든 수치 비교가 조건부다.
- 고치는 법: `hardware.py`가 `PilotGain`을 Telemetry에 싣고 `MpcWorker`가 engage 시 meta `run.pilot_gain`으로 적는다;
  gain ≠ 0.5면 engage 거부 또는 경고. 그 전엔 세션 시작 때 QGC/BlueOS의 gain 표시(STATUSTEXT `#Gain: NN%`)를 확인해 둘 것.

### 수동 REC 경로에선 meta `controller`가 **날린 제어기가 아닐 수 있다** (2026-09-07, safety 감사에서 발견)
- REC를 engage 전에 누르면 `_csv_auto=False` → disengage가 CSV를 안 닫고 → `set_mode`가 허용되며 → REC stop 시
  `_write_meta()`가 **새** 컨트롤러의 `meta()`를 적는다(`rov_gui/control/workers.py` `_write_meta`/`set_mode`). `type`·Q·
  `heave_trim.applied`가 전부 실제 비행과 어긋난 채 저장될 수 있다. 자동 REC(engage가 여닫는 경로)는 무관.
- 고치는 법: engage 시점에 `ctrl.meta()` 스냅샷을 잡아 close 때 그것을 쓴다. 그 전엔 수동 REC 런의 meta `controller`는
  `events.log`의 `ENGAGED (<mode>)` 줄과 대조해서 읽을 것.

### MPC 계열 추종자는 정책 플랜 아래서 **hold-tail 마스크 때문에 추력을 못 낸다** — `track` 1차 A/B는 부족, `q_scale` 미검증 (2026-09-07)
- **증상**: `mpc`/`mpc_tuned`를 하위 제어기로 policy를 날리면 기체가 거의 전진하지 않는다(순변위
  0.05–0.4 m/50–90 s; PID 0.2–0.9 m). 참조는 body 전방 7.5–10 cm에 일관, yaw 오차 ~1°, uX 부호 반전
  0–0.4/s → 흔들림이 아니라 **추력 부족**: |uX| p50 0.5–1.8 N, axis_surge 0.01–0.03으로 데드밴드 아래
  틱 92–100 %(PID 6–7 N, 37–52 %) [측정: data/20260907/0907_164659/
  mpc_16{4716,4835,5209}.csv, 0907_152730/mpc_153134.csv vs 0907_145206/mpc_145239.csv].
- **원인(증거 순)**: (1) `hold_tail: mask` — 플랜이 도착할 때 이미 ~0.5 s 낡아(obs age p50 0.5 s)
  3 s 호라이즌 중 위치 가중 구간이 0.25–0.4 s뿐(`hold_frac` p50 0.87–0.92); 그 비용의 최적 surge는
  신선한 플랜 0.05–0.2 N, blend 0.6–1.9 N. **같은 런에서 마스크가 풀린 `ref_src hold` 틱은 3.3–6.9 N**
  (mpc 164716 6.5 / 165209 6.9 / mpc_tuned 164835 4.5 / 153134 3.3 N; PID는 ref_src 무관 ~6.5 N).
  (2) leash 0.05 m가 replan 95–99 %에서 포화 → 오차가 못 자람(정책은 epoch당 3–9 m를 요청, anchor는
  0.05–0.48 m만 이동). (3) mpc/mpc_tuned엔 적분·외란 항 없음(EAOB 미사용, 의도) → DC 손실 학습 불가;
  `mpc_tuned`는 `along_scale 0.25`로 더 무름. 데드밴드는 물리 문턱(이 기록의 무릎 axis 0.05–0.07 ≈ 3–4 N).
- **A/B 1차 결과(2026-09-07 18:00, `hold_tail: track`)**: mpc_tuned 미션 4개에서 uX p50이 0.4–0.6 N(mask, 16:11–16:40) →
  **2.5–3.6 N**으로 5배쯤 올랐지만 여전히 데드밴드 아래(|ax_surge| > 0.096 틱 0–2 %), 순변위 0.07–0.24 m/16–28 s
  [측정: 0907_180038/mpc_18{0058,0132,0202,0253}.csv vs 0907_161726/, 0907_163702/]. 같은 런은 위의 부력 FF 문제로 위로 흘렀으므로
  전진 판정은 `vehicle_net_buoyancy_n` 적용 뒤 다시. 다음 레버는 `policy.q_scale`(2–3).
- **임시 대응(2026-09-07, 미검증)**: hw_mpc/land_dp `policy.hold_tail: mask → track`(단일 변수 A/B 1순위;
  A10이 우려한 "청크 끝마다 감속"을 관찰) + 보조 레버 `policy.q_scale`(정책 플랜 중 위치 가중치 ×s,
  런타임 cost_set, STOP/모드 전환 복원; 기본 1.0, 강성 ~√q [유도]) + `anchor_leash_m` 상향. 판정은
  CSV `ax_surge > 0.096` 틱 비율·순변위를 PID 런과 대조. dobmpc는 요청대로 손대지 않음.
- **A/B 2차(2026-09-08, `track` + `anchor: measured`)**: plan 구간 |uX| 0.05–0.22 → 1.21 N(blend 1.46, hold 0.80)으로 올랐지만 여전히
  데드밴드의 1/4이고 세 축 전부 pwm_dev ≤20 µs; 지면속도 0.008 m/s. 이제 병목은 마스크가 아니라 위 "세 축 모두 ESC 정지대역" 항목
  [측정: 0908_112101/mpc_112101.csv, diag/taskE_out.txt]. 오늘은 measured 앵커가 같이 바뀌어 track 단독 효과로 인용 금지.
- **제대로 고치는 법**: (a) 관측 나이(추론 232 ms, FS와 GPU 공유)를 줄여 플랜 수명을 되찾거나 마스크를
  플랜 시작 시각이 아니라 도착 시각 기준으로, (b) 플랜트에 실측 추력 맵(데드밴드)을 넣거나 per-thruster
  보상(직접 스러스터 인터페이스 필요), (c) 실기용 Q/R·leash·v_max를 접근 단계에 맞게 재유도.

### 정책 턱 채널: "지금" 샘플은 교착이고, 의도 샘플은 **도착 전에** 닫을 수 있다 (2026-09-07)
- **증상**: 첫 실기 접근(0907_145206, mission mpc_145239, 120 플랜)에서 기체는 물체에 도달했지만 턱은
  안 닫혔다. 1차 원인은 `policy.gripper: false`(A12 기본 OFF) → 2026-09-07 `true`로 전환. 그런데
  기록을 되돌려 보면 스위치를 켰어도 즉시 닫히지 않았을 것이다: 정책은 매 청크에서 knot 0 g≈0.95(열림),
  knot 4–5에서 g<0.30(닫힘)을 내는데(마지막 15 플랜 전부), 스테이션은 스티처의 "지금" 값(`ts[0]`)만 보고
  플랜은 0.45 s마다 교체되므로 문턱 아래 틱은 32 s 이후 7–59 %뿐이고, 새 플랜의 knot 0(0.95 >
  open_above 0.67)이 OPEN 에지를 다시 낼 수 있다 [유도: plans.jsonl 오프라인 재생, 이 세션].
  knot 0이 열림에 머무는 이유는 정책이 관측하는 폭이 open-loop 추정치라 명령 없이는 안 바뀌기 때문
  (정책은 턱이 움직이길, 스테이션은 g가 내려가길 기다림).
- **더 깊은 문제**: 정책의 닫기 의도는 t≈32 s, 최종 위치에서 **0.35 m 밖**부터 나타난다(플랜 g[-1] 평균
  0.26). 데모에서는 손이 최종 자세 ~1 s 전(≈0.3 m)에 닫기 시작하니 정책은 그 **공간적** 타이밍을 재현하는
  것인데, 기체는 데모보다 5–10배 느려(0.03–0.2 m/s) 그 0.35 m에 30 s가 걸린다. 의도대로 닫으면 턱(travel
  2 s)이 도착 28 s 전에 완전히 닫힌다. 게다가 배포 관측은 분포 밖(위 "그리퍼가 화면 반대쪽" 항목)이라
  폭 예측 자체의 신뢰도도 미확인.
- **임시 대응(2026-09-07)**: `gripper_lookahead_s` 노브(기본 0.0 = 옛 동작; 0.6 s면 같은 런의 32 s 이후
  틱 70–100 %가 문턱 아래) + CSV `grip_g` 열(schema 13)로 다음 런에서 hysteresis가 본 값을 남긴다.
  첫 CLOSE 에지 뒤에는 추정 폭이 내려가 정책이 따라오므로 교착은 스스로 풀릴 것으로 본다 [예측].
  파일럿 G/H 에지는 여전히 우선(last-writer-wins, 위 replay 항목).
- **제대로 고치는 법**: (a) 턱 결정을 시간이 아니라 **경로 진행**에 묶는다 — 플랜의 g가 떨어지는 knot의
  위치에 기체가 도달했을 때 닫기(스티처의 시간 팽창을 g에도 적용하면 자연히 그렇게 된다; 지금은 g가
  knot 격자에 그대로 실리는지 확인 필요); (b) 도착 판정(참조 정지 + 의도 g<문턱)의 AND; (c) 근본적으로는
  데모 속도에 맞는 추종(v_max ↑, 시간 팽창 ↓) 또는 폭 채널을 속도에 독립인 "거리 기반"으로 재라벨링.
  어느 쪽이든 **턱을 켠 실기 런 1회의 `grip_g`/`grip_cmd`/events 기록이 먼저**다.

### `demo_e2e.py policy`가 배포 설정으로 **red** — 임시 z-hold + demo 플랜트, 오늘 변경과 무관 (2026-09-11)
- **증상**: `QT_QPA_PLATFORM=offscreen python rov_gui/tests/demo_e2e.py policy`가 exit 1. 배포
  `hw_mpc.yaml policy.z_hold_above_floor_m: 0.20`(위 임시 항목)이 demo datum 기준으로 모든 knot z를 박스 밖
  (~0.3 m)에 두어 3연속 reject → escalated, installed 0. z-hold를 null로 해도 `hold_tail: extrapolate` +
  `anchor: measured`(둘 다 기존 미커밋 diff) 아래서 참조가 합성 플랜트를 앞질러 `diverged`로 끝난다
  [측정: 검토 에이전트 3회 실행, 로그 스크래치패드 e2e_policy_shipped.log — 세션 종료 후 사라짐; 재현은 위 명령].
  HIGH/LOW·ckpt 피커·max_run_s 500 변경(2026-09-11)은 실패 경로에 없지만, 그 변경에 대한 **녹색 e2e 베이스라인이 없다**.
- **임시 대응**: 오프라인 단위 테스트(offline 119·policy 84·control 112·policy_worker 30)와
  `rov_gui/tools/policy_observe_smoke.py` 15/15로 대신 확인. e2e는 `demo_e2e.py pid follow …` 같은 다른 셰이프만 green.
- **제대로 고치는 법**: z-hold 항목을 예정대로 삭제한 뒤, demo 플랜트 또는 e2e 기대치를 `hold_tail: extrapolate`/`anchor: measured`에
  맞게 재조정(`div_max_m` 0.25 대비 합성 플랜트의 응답)해 `demo_e2e.py policy`를 다시 green으로 만들고, 그때 LOW None 변형
  (`Opts.mpc_mode = "none"`)도 드라이버 인자로 돌릴 수 있게 한다(지금은 main()이 mpc/dobmpc/…/pid만 받는다).

### `demo_e2e.py mpc policy`가 한 번 "a plan was installed after STOP"으로 FAIL — 재현 안 됨 (2026-09-07)
- **증상**: 추종자 모드 확인 중 `demo_e2e.py mpc policy` 6회 중 1회(첫 실행, dobmpc_tuned 직후 연속 실행)가
  STOP 뒤 플랜 1개가 설치됐다며 FAIL. 같은 인자로 2회 재실행·pid/mpc_tuned/dobmpc_tuned/dobmpc 모두 OK
  ("after STOP: 0 plans emitted, 0 installed"). 하네스는 wall-clock으로 돌고 acados 빌드 직후라 타이밍
  플레이크일 가능성이 크지만, 실제 워커에서 STOP 뒤 in-flight 청크가 epoch/inactive 규칙을 뚫는 경로가
  있다면 안전 문제라 기록해 둔다.
- **임시 대응**: 없음(재현 시 그 런의 plans.jsonl `epoch`/`t_rel`과 STOP 이벤트 시각을 대조).
- **제대로 고치는 법**: demo_e2e의 STOP 검사에 플랜의 epoch·obs_t·intake 시각을 출력하게 해 다음 재현 때
  "STOP 이전 epoch의 지연 도착"인지 "STOP 이후 epoch"인지 즉시 갈리게 한다.

### 학습 리포의 val 곡선이 **EMA 가중치 기준**이라 체크포인트 선택을 거꾸로 시킨다 (2026-09-07)
- **증상**: 2026-09-07 5-dim 재학습에서 `logs.json.txt`의 32-윈도 `val_action_mse_error_pos`(= workspace의
  `sample_every` 배치)로 랭킹하니 **epoch 10**이 1위였는데, 배포 가중치(`model`)로 219-윈도 held-out을 재보면
  epoch 10이 **측정한 에폭 중 최악**(26.7 mm)이고 후반 에폭이 최고(ep150 17.2 / ep195 17.3 mm)다
  [측정: `<run>/heldout_pos_yaw_width_ep*_model_stride4.log`, run =
  umi_underwater_robust_control/data/20260907/0907_113747_train_umi_depth_5d_can_grasp_depth_5d_v0].
  워크스페이스의 val은 `policy`(= EMA 사본)로 샘플링하므로, BatchNorm 결함(아래 항목)에 오염된 가중치의
  곡선으로 **정상 가중치의 순위를 매기는 셈**이다. 같은 에폭 195에서 ema 33.0 mm vs model 17.3 mm
  (MSE 3.6배)이고, 두 곡선의 에폭 순위는 서로 뒤집혀 있다.
- **임시 대응**: 선택은 `dp_policy_offline.py replay --val-only --weights model`(219 윈도)로 하고, val 곡선은
  "수렴 확인용"으로만 본다. 2026-09-07 런의 `selected.json`에 이 프로토콜과 두 곡선을 모두 적어 두었다.
- **제대로 고치는 법**: (a) BatchNorm/EMA 결함을 고쳐(아래 항목의 (2)) 두 가중치가 같은 것을 가리키게 하거나,
  (b) 워크스페이스의 val 샘플링을 `model`로도 돌려 두 계열을 함께 로깅하거나, (c) 주석 처리된 held-out val
  루프(`train_diffusion_transformer_timm_workspace.py:273-288`)를 되살려 전체 검증 세트로 기록한다.

### 2026-09-06 이전 모든 런에 **수중 depth가 한 프레임도 없다** (2026-09-06)
- **증상**: 실기 런은 depth를 디스크에 안 남긴다 — 224 관측도, uint16 mm 맵도, **원본
  스테레오 쌍도**. 그래서 사후 확인도 오프라인 재생성도 불가능하다
  [측정: data/20260903/0903_183555/ = controller.json,
  mission_log.txt, ui_20260903_183555.{mp4,json} — depth 없음].
  유일한 시각 흔적인 `ui_*.mp4`는 **프레임별 auto-range + 64-knot 히스토그램 평활화로
  이미 색칠되고 위젯 크기로 축소된 화면**의 녹화라, 같은 색이 같은 거리가 아니다.
- **임시 대응 (지금부터)**: `--record-depth`가 `<run>/policy_obs/`에 무손실 PNG로 남긴다.
  `python -m rov_gui.tools.depth_compare pair --run <run>`로 육상 학습 관측과 **같은 색 규칙**에
  올려 비교한다.
- **~~제대로 고치는 법: 원본 스테레오 쌍도 남겨야 한다~~ → 2026-09-06 구현**:
  `--record-stereo`가 `<run>/stereo/`에 **RAW(미정류) mono 쌍 + rig + 그 런의 FS 설정**을
  무손실로 남긴다(`rov_gui/perception/stereo_record.py`). 이제 iters/scale/체크포인트를
  바꿔서, 또는 새 캘리브로 다시 정류해서 **재잠수 없이 depth를 다시 만들 수 있다**.
  남은 미비: **오프라인 재생성 도구는 아직 없다** — 저장 포맷은 자기서술적이지만
  `<run>/stereo/`를 읽어 FoundationStereo를 다시 돌리는 스크립트를 아직 안 썼다.

### depth 렌더러 통일 — **절반만** 끝났다 (2026-09-06 갱신)
- **고쳐진 쪽**: 수중 FoundationStereo 패널과 육상 FS 영상이 이제 **같은 함수**를 쓴다
  (`rov_gui/depth_colour.py`: TURBO / warm=NEAR / 고정 0.20–3.00 m / obs 역수 간격 /
  invalid 검정). 스테이션은 `--fstereo-palette umi`가 기본이고, 옛 장면적응형 그림
  (JET + 프레임별 백분위 + 64-knot 평활화)은 `--fstereo-palette adaptive`로만 나온다.
  육상은 `data_collection/make_depth_trajectory_video.py`와 `depth_compare`가 같은 모듈을
  임포트한다. 규칙은 HUD·컬러바·런 meta(`fstereo.panel`)에 적힌다.
- **남은 두 개(안 고침)**: 육상 recorder 프리뷰 TURBO/선형 mm/**warm=FAR**
  (`umi_handheld/record.py:529-536`)와 수중 device-stereo 패널 TURBO/선형 mm/**warm=FAR**
  (`rov_gui/imaging.py:105-125`, `depth_to_bgr`). 둘 다 극성이 반대라, FS 패널 스크린샷과
  나란히 놓으면 **가까운 쪽이 반대 색**이다.
- **덤(그대로 유효)**: repo 루트의 옛 `umi_ep0_foundationstereo.mp4`는 obs와 끝점은
  같지만(0.2–3.0 m) 매핑이 **선형**이다 — 1.00 m가 그 영상에선 팔레트 0.29, obs에선 0.14.
  새 3-패널 영상은 obs 간격이므로 그 옛 mp4와 섞어 보지 말 것.
- **제대로 고치는 법**: 남은 두 렌더러도 `depth_colour.depth_mm_to_bgr`로 옮기거나
  (device 패널은 0.3–6 m 범위를 유지하되 극성만 맞추는 선택지도 있다), 최소한 각 패널이
  자기 규칙을 화면에 적게 한다.

### 런 폴더 하나에 engagement가 여러 개이고, **플랜 로그가 공유된다** (2026-09-06)
- **증상**: `data/20260906/0906_191935_observe/`에 engagement가 **둘** 들어 있다 —
  `mpc_191935`(19:19:35, 14,918행, epoch 2, plan 1..1426)과 41분 뒤 `mpc_200050`
  (20:00:50, 2,391행, epoch 4, plan 1991..2203). CSV와 meta는 파일이 분리되지만
  **`plans.jsonl`(1,639 rec)과 `policy_plan.csv`(1,639 plan)는 두 런이 공유**하고,
  두 번째 런의 `t_rel`은 **0부터 다시 시작**한다.
- **왜 조용히 틀리는가**: 시간으로 플랜을 찾으면 t < 230 s 구간에서 **두 번째 engagement의
  참조가 첫 번째의 시각에 뽑힌다**. 실제로 첫 렌더가 그랬고, 두 epoch를 섞어 잰 clock offset
  스프레드가 **0 ms가 아니라 2,475 s**로 나왔다.
- **임시 대응 (구현됨)**: `render_run_scene.py` / `export_run_html.py`가 engagement를
  **데이터로 고른다**(기록된 depth 창 안에 플랜이 들어오는 쪽) + `meta.policy.epoch`와
  CSV의 `plan_id` 열로 플랜을 필터한다. `--engagement <이름조각>`으로 명시 선택.
  회귀 테스트 `test_run_scene.py::test_two_engagements_in_one_folder_are_not_mixed`.
- **제대로 고치는 법**: 스테이션이 engagement마다 폴더를 새로 열거나(`_run_dir` 핀이
  풀린 뒤 join 창이 다시 열린 것으로 보인다), 최소한 `policy_plan.csv`에 `epoch` 열을 넣어
  파일 스스로 소유권을 말하게 할 것. 지금은 **파일 이름만 보고는 어느 런인지 알 수 없다.**

### FoundationStereo 패널의 위쪽 아치·오른쪽 띠가 **빨갛다(≤0.2 m)** — alpha 0.5 정류 여백을 네트워크가 근거리로 환각하고 아무것도 안 걸러낸다 (2026-09-14)
- **증상**: `--fstereo`(native 뷰) depth 패널 상단에 가운데가 두꺼운 짙은 빨강 아치(rect rows 0~10),
  오른쪽 끝 ~6 px 세로 띠(cols 634~639), 왼쪽은 없음. 값은 상단 205→315 mm 램프, 오른쪽 **72 mm**
  (disparity 357 px @640). HUD는 `valid 100%`라고 말한다 — 그 100 %가 증상(정크가 유효로 집계).
- **원인(확정, 반박 3렌즈 0)**: `--policy`가 `--fstereo-alpha`를 0→**0.5**로 올린다
  (`rov_gui/__main__.py:1127`, 224 obs crop coverage 게이트 때문). alpha 0.5의 좌측 정류 맵
  (`initUndistortRectifyMap(K_left_live, D_left_live, R1, P1)`)은 격자의 **2.89 %**가 원본 센서 밖을 샘플
  (row 0은 raw y −10.4~+5.9 → 가운데만 밖 = 아치; col 639는 raw x 638.6~643.9; col 0은 x 7.4~13.3 = 안쪽이라
  왼쪽 띠 없음). `cv2.remap` 기본 BORDER_CONSTANT 0이 그 픽셀을 **검게** 채우고(`rov_gui/perception/fstereo.py:840-841`),
  FoundationStereo는 softmax 회귀라 "무효"를 낼 수 없어 양수 disparity를 뱉으며(`core/submodule.py:427-431`,
  clip은 `run_hierachical`에만), `depth_from_disparity`는 d≤0만 0 처리(`c3_camera/host_depth.py:168-226`,
  `Z_RANGE_MM=(0,65535)`), `stereoRectify`의 `validPixROI`는 버려진다(`host_depth.py:503`). 마스크가 어디에도 없다.
  [측정: near(<300 mm) 비율 out-of-sensor 마스크 안 0.811 vs 밖 0.015, 168/168 프레임, 번짐 1 px —
  scratchpad `margin_check.py`/`ring_bleed.txt` 결과를 data/20260914/0914_152913/policy_obs/depth/*.png +
  mpc_152951.meta.json `fstereo.rig`로 재현 가능; 52/52 native PNG 보유 런 전부 동일]
- **영향**: (1) 패널·커서 프로브(아치 0.17~0.31 m, 띠 0.07~0.09 m를 "측정"으로 표시); (2) 컬러 그리드 맵은 상하단
  여백은 FOV 밖으로 떨어지지만 오른쪽 72 mm 띠가 cols 298~399에 **물결 곡선으로 착지해 z-buffer에서 실제 표면을 덮음**
  (프레임당 0.6~1.0 %) → `--pose` FoundationPose 마스크가 오른쪽 1/4에 걸리면 `_frame_depth_quality` p5 흔들림 가능;
  (3) `--record-depth` native PNG에 정크가 그대로 기록(52 런); (4) **정책 obs**: 오른쪽 띠는 crop(x0=120) 밖, 상단 아치는
  obs **row 0 한 줄에 점선(~36 %/프레임, 픽셀의 0.16~0.20 %)**으로만 들어가되 valid=255로 위장 — 53/53 런 동일,
  육상 학습 obs(alpha 0, 소스 rows 81~369만 사용)엔 없는 형태. 플랜에 미치는 효과는 **미측정**
  (`policy_session_replay.py`로 row 0을 row 1로 덮은 obs와 비교하면 답 나옴, GPU 필요).
- **임시 대응**: 패널의 그 두 자리는 무시. 오프라인 통계는 out-of-sensor 마스크로 걸러서 계산(rig가 meta에 있음).
- **제대로 고치는 법(1순위 A)**: `StereoRig.from_stereo_pair`(`host_depth.py:517-519`)에서
  `valid_left = cv2.remap(np.full((h,w),255,np.uint8), maps_l[0], maps_l[1], cv2.INTER_LINEAR) == 255`를 만들고
  invalid를 1~2 px **dilate**(valid를 erode하지 말 것 — col 0 쪽은 여백이 없음), `StereoRig.valid_left` 필드 +
  provenance `valid_left_pct`/`margin_dilate_px`(meta 경계) → `fstereo.py:844` 직후 `depth_native[~valid_left] = 0`.
  HUD `valid_native`가 ~97 %로 정직해진다. 우측 영상 마스크는 **적용 금지**(격자 8.8 %가 매칭점이 우측 여백에 떨어지지만
  near 정크 없음, 0.016 vs 0.015). 보조로 alpha 0.2~0.3 [유도: coverage 99.1~99.6 %, 여백 0.31~0.74 %] 검토하되
  alpha 0 시절 네트워크 d≤0 3.06 %(`fstereo_bench_out/hardware_20260902_fill.txt`)가 되돌아올 수 있음(미모델).
  전제: run meta `fstereo.rig`에 **K_right/D_right/R2를 기록**(`hardware.py:708-715`는 좌측만 적음) — 없으면 오프라인 검증 전부 [유도].
  범위 제한 컷(E)은 답이 아님(15 cm 그리퍼를 지운 전례, `fstereo.py:107-119`). `--record-stereo` 없이는 BORDER_REPLICATE/네트워크 전 crop A/B 불가.


### 배포 FoundationStereo 설정이 학습 데이터와 다르다 — 의도적이지만 미해결 (2026-09-06)
- **상태**: 학습 depth는 iters **16** / scale **1.0**
  [측정: `~/Desktop/data collection/dataset_depth.zarr.zip` `.zattrs['depth_source']`],
  0903 실기는 iters **8** / scale **0.75**
  [측정: data/20260903/0903_183555/controller.json
  `fstereo.iters` / `fstereo.scale`]. 시간축 패리티(관측 쌍 간격을 학습 stride 66.7 ms에
  붙이기)를 사기 위한 **의도된 거래**이고 `__main__.py`의 `FS_RATE_NOTE`가 그렇게 적고 있다.
- **미해결**: 어느 불일치가 더 비싼지 **A/B를 아무도 안 돌렸다**. `--fstereo-scale 1.0
  --fstereo-iters 16`이 계기 패리티를 복원하지만 ~7 Hz로 떨어진다.
- **제대로 고치는 법**: `--record-depth`로 두 설정을 각각 한 런씩 남기고 육상 관측과
  `depth_compare`로 대조 — 이제 그 비교가 가능하다.


### `--pose`(SAM2 + FoundationPose)의 상류 작업공간 두 개가 **디스크에 없다** — 켜면 로드에서 멈춘다 (2026-09-30)
- **증상**: 기본값 `SAM2_LIVE_ROOT` = `~/Desktop/New Folder`(sam2_live, `rov_gui/perception/session.py:57`)와
  `FOUNDATIONPOSE_ROOT` = `~/Desktop/poseEstimation/FoundationPose`(`session.py:1492`)가 둘 다 없다(`ls`, 2026-09-30;
  휴지통 비어 있음). `--pose`를 켜면 `PoseSessionError: perception source not found at …/New Folder`, 메시 재구성은
  `reconstruction script not found …/run_nerf_single.py`. FoundationStereo와 같은 부류(리포 밖 트리)인데, `New Folder`는
  `rov_gui/README.md` "원본은 건드리지 않는다" 절에 **문서화돼 있었는데도** 지워졌다 — 문서만으론 부족하고 리포 안에 둬야
  한다(FS는 2026-09-30 `external/FoundationStereo`로 복원). FoundationPose 루트는 코드와 `oakd.sh`에만 있었다.
  평소 실행 줄(`--fstereo --policy`, `--pose` 없음)은 무관.
- **임시 대응**: `--pose`를 쓰지 않는다. 사본이 있으면 `--pose-src`/`$SAM2_LIVE_ROOT`, `$FOUNDATIONPOSE_ROOT`로 가리킨다
  (`rovgui-pose`의 `etc/conda/activate.d/oakd.sh`도 옛 경로를 export하지만 `./c3`는 env를 activate하지 않아 무관).
- **제대로 고치는 법**: 두 트리를 찾거나 다시 받아 FoundationStereo처럼 리포 안(`external/`)에 두고 기본값을 리포 상대
  경로로. sam2_live는 자체 코드라 upstream이 없을 수 있다 — 원본이 어디 있었는지부터 확인.


### FoundationStereo 로드가 **매번 GitHub에 묻는다** — 429/5xx면 캐시가 있어도 그 런은 depth가 없다 (2026-09-30)
- **증상(잠복, 아직 미발생)**: 업스트림 `external/FoundationStereo/depth_anything/dpt.py:157`이
  `torch.hub.load('facebookresearch/dinov2', …)`를 ref 없이 부르고, torch 2.11 `torch/hub.py:205-212`
  (`_parse_repo_info`)가 로드마다 `https://github.com/facebookresearch/dinov2/tree/main/`을 연다. 네트워크가 아예
  없으면(URLError) `~/.cache/torch/hub/facebookresearch_dinov2_main`으로 넘어가지만 **404 외 HTTPError(429, 5xx)는
  그대로 raise** → `FS FAULT` → FS는 다른 센서로 넘어가지 않으니 그 런은 depth가 없다. timeout 없는 `urlopen`이라 반쯤
  끊긴 네트워크에선 `FS LOADING`이 길어질 수 있다 [유도].
- **임시 대응**: 풀에서 인터넷이 불안정하면 아예 끊고 띄운다(URLError 경로 → 캐시).
- **제대로 고치는 법**: 로드 동안만 dinov2 hub 호출을 캐시 디렉터리 `source='local'`로 돌린다 — 업스트림이
  `dpt.py:155`에 바로 그 줄을 주석으로 남겨 두었다. 체크아웃은 건드리지 않는 원칙이니 `fstereo.py`의 기존 몽키패치
  방식(`_patch_for_capture`처럼)으로.


### 기체 마커의 그리퍼 기하: **턱 x만 실측**, 나머지는 여전히 추정 (2026-09-03, 2026-09-08 갱신)
- **상태**: `--policy-observe`와 새 기체 마커(선체+그리퍼+추진기+C3, 3-D, 앞뒤 헤딩 점선)는 풀에서
  돌았다(data/20260903/0903_183405_observe, 20260908/0908_180453 등). 마커의 첫 실물 대조가
  2026-09-08의 "턱에 문 병을 태그 58 중심에" 시험이고, 그것이 턱 위치 오류를 드러냈다(아래). 태그
  dropout·박스 이탈·긴 런에서 안 끊기는지, 조이스틱이 끝까지 살아 있는지는 **따로 확인한 적 없다**.
- **마커 기하의 근거 등급 (2026-09-08 갱신)**: 턱 x는 이제 **렌즈 기준 실측**이다 — 턱에 문 병을 바닥
  태그 58 중심에 놓은 프레임에서 그립점이 렌즈 앞 0.196 m(범위 0.187~0.204) [측정: data/*/*_observe/
  20260908/0908_180453/policy_obs/rgb/000160.jpg (런 자체 intrinsics로 태그 PnP) + 0908_170428/policy_obs/rgb/
  000000.jpg (병 높이 0.084 (±0.006) m) + 0908_175151/policy_obs/rgb/000000.jpg (열린 턱 광선)]. CAD 체인의 110.664 mm
  (→ 턱 0.4165)는 반증됐다(그 턱은 360행 프레임의 행 ~420으로 투영되는데 턱은 행 265~360에 보인다).
  `rov_shape.LENS_TO_GRIP_FLU_M = (0.196, 0, −0.275)`, `JAW_CENTRE_M = policy.tcp_body_flu_m = (0.502, 0, −0.17)`,
  메타 `rov_drawn_geometry.lens_to_grip_flu_m`. 2026-09-08 이전 런의 메타는 0.4165를 들고 있다(기록 경계).
  **아직 미실측**: 턱 z(CAD 275.014 mm 수직에 앵커; 이미지와 ~1 cm 안에서 일치하나 독립 실측 아님 [유도]),
  턱 y(0 가정 [예측]; 컬러 카메라는 문 병의 축을 기체 축 오른쪽 0.011 m에 본다), 손가락 길이·개구(턱 상자
  [예측]/벤더 [스펙]), 튜브 위치(sim `GRIP_POS`를 턱과 같은 +0.0855만큼 민 [유도]), footprint/shroud(아래).
  그리고 **렌즈 자체의 x/z가 의심된다**(항목 "C3 외부파라미터 cam_t_flu의 x/z가 의심된다") — 턱은 렌즈에
  앵커돼 있어 어느 쪽이든 턱은 맞지만, 선체 그림은 실제보다 ~9 cm 뒤에 그려질 수 있다.
- **추진기 shroud 반경 0.0467 m는 순환 유도**다: "수직 추진기가 기체 폭을 만든다"고 가정하고
  footprint 0.5334에서 뺀 값이라 **독립 검증이 없다**. 게다가 CAD 폭 0.5749가 따로
  돌아다니는데(그 값이면 0.0675) 어느 쪽이 맞는지 미결.
- **제대로 고치는 법**: (1) 풀의 `--policy-observe` 런에서 태그 dropout·박스 이탈·긴 런·조이스틱
  생존을 명시적으로 확인; (2) 턱 y·z와 그리퍼 튜브를 기체에서 실측(스테레오 삼각측량 또는 자)해
  `rov_shape` + `policy.tcp_body_flu_m`(또는 `tcp_offset_cam_m`)을 같이 갱신; (3) footprint를 CAD/테이프 중
  하나로 확정하고 shroud 반경을 재유도.

### C3 수중 depth 오차는 **거리에 비례해 커진다** — 배율 상수로는 원리적으로 못 고친다 (2026-08-24)
- **증상**: `--depth-scale` 하나로는 한 거리에서만 맞는다. 실측 depth-vs-MAP 원시값이
  낮은 높이 **1.28**, ~0.9 m **1.56**으로 움직인다(조종사 확인).
- **샘플링 아티팩트가 아니라는 결정적 증거**(진단 함수가 관여하지 않는 증인):
  한 메시의 세 축이 **143 x 166 x 187 mm**인데 캘리퍼 실측은 119.73 mm — 축별 1.19 /
  1.39 / 1.56, **비등방 1.31**
  [측정: data/20260823/0823_210304/mission_log.txt 21:02:29].
  상수 배율 오차는 **모든 축을 똑같이** 늘린다. 정육면체가 벽돌로 나왔다는 건 오차가
  거리에 따라 변한다는 뜻이다.
- **모델**: 오차는 **disparity 도메인**에 있다(disparity가 과소 보고 → depth가 길게,
  거리에 비례해 악화). `r(Z) ≈ 0.977 + 0.538·Z`
  [유도: c3_camera/datasets/* 108만 샘플 회귀 — **다른 stereo 설정**(extended ON,
  MinZ 150)이라 비행 설정(extended OFF, MinZ 300)에서 재적합 필요. 측정치로 인용 금지].
  Z→0에서 계수가 0.98±0.04 = **근거리에선 metric**, 오차 전체가 거리 비례.
- **배제된 것들**(각각 특정 숫자로): 순수 배율(1.56배면 fx=244 px → 반화각 52.7°인데
  Snell 물속 한계 48.75° 초과 = 존재할 수 없는 카메라) · 평면포트 굴절(부호가 반대이고
  10~30배 작다) · 정수 disparity(truncation 상한이 +1.8%인데 +56%가 필요; 게다가 실제
  출력은 1 mm 간격으로 조밀해 양자화 격자가 아예 안 보인다) · "스테레오 외부파라미터가
  공기 중 값" (매질로 안 변하는 양이라 물리적 내용 없음).
- **임시 대응**: `--depth-scale 0.64`는 **Z ≈ 1.09 m에서만 정확**하고 잔차가 양쪽에서
  부호를 바꾼다(0.5 m −20%, 1.5 m +14%, 2.0 m +31% [유도]). 유지하려면 **1.0~1.1 m
  높이에서 날 것**. depth 유래 거리(메시 치수·FoundationPose 거리·object_nav 거리)는
  **현재 어느 것도 metric이 아니다**. 컬러 AprilTag 경로는 영향 없고 유일한 기준이다.
- **제대로 고치는 법**: (1) `_check_depth_scale`이 프레임당 만드는 수백 쌍을 **기록**한다
  (지금은 중앙값 빼고 전부 버린다, window.py:801-811) → nav 폴더에 `depth_check.csv`;
  (2) 풀에서 한 번: 거리 사다리 5단(0.4~1.8 m) + **피치 팬 ±15°**(이게 없으면 z와
  화각항이 공선이라 계수가 쓰레기가 된다) + yaw 팬; (3) disparity 공간에서 회귀해
  `Z_corr = Z/(1 + c·Z)` 적용. **물에 안 들어가고 되는 선행 검사 두 개**:
  `calib.getFov(sock, useSpec=True/False)`와 `getStereoLeft/RightRectificationRotation()`
  덤프(정류 회전에 1~2° 상대 yaw가 있으면 오프셋 가설 확정), 그리고 공기 중에서 같은
  진단을 돌려 **거리 의존이 남는지**(남으면 굴절·매질 원인 완전 배제).
- **미확정**: 원인(정류 zero-point / homography 정류 / 매칭 편향)은 아무도 장치에서
  확인 안 했다. **1.7 m 이상은 캘리브 안 됨** — 모든 데이터가 0.10~0.76 m다.
  화각 의존항(`+0.474·Z·tan²θ`)이 실재하면 **어떤 r(Z) 곡선으로도 못 고치고** 수중
  스테레오 재캘리브레이션이 필요하다.

### 물체가 태그맵에서 **매트 밑/두 칸 옆**으로 찍히면 아무것도 안 막는다 — map-frame 타당성 검사 부재 (2026-08-24)
- **증상**: 2026-08-24 오전 두 런 모두 물체(태그 58 위)가 **매트 아래 51~53 cm**,
  가장 가까운 태그 11/10/52로 보고됐고, follow가 그대로 arm됐다(231 cm / 256 cm).
  [측정: data/20260824/{0824_101807,0824_101251},
  nav_*/map.json로 재투영 — 과대 배율 1.55x / 1.57x]
- **그날의 원인은 따로 고쳤다**(`--depth-scale` 미지정 → 기본값 0.64로 승격). 남는 결함은
  **원인과 무관한 방어가 없다는 것** — 바닥 밑 물체는 물리적으로 불가능한데
  `object_nav.update`는 pose_state / 카메라 거리 / jump만 보고 map-frame 좌표는 안 본다.
- **임시 대응**: trajectory 플롯에서 물체 마커가 매트 평면 근처인지 눈으로 확인.
- **제대로 고치는 법**: `ObjectAnchor.update`에 태그맵 평면 기준 z 밴드(예: 매트 위
  −0.1~+1.5 m 밖이면 거부)와 풀 볼륨 밖 거부를 추가. 임계는 tag_map의 z 분포에서 유도.

### FoundationPose **재등록 pose 자체에는 아직 연속성 게이트가 없다** (2026-08-24, 부분 수정)
- **고친 부분**: `object_nav`의 jump gate가 5회 거부 후 자동 reseed하던 것은 그대로지만,
  **follow 중 reseed가 나면 STATION으로 강등**한다(`workers.py:_tick_follow`, 깊이·헤딩
  유지·disengage 아님). 2026-08-23 run1의 1.324 m 유령 스냅이 기체를 끌던 경로는 끊겼다.
  물체 겉보기 속도가 FF 상한의 3배를 1초 넘기면 같은 강등이 걸린다(매끄럽게 미끄러지는
  유령은 jump gate를 안 건드리므로 별도 방어가 필요했다).
- **남은 결함**: SAM2 loss→global register는 **여전히 last-known pose와 아무 비교를 안 한다**
  (`rov_gui/perception/session.py:1012,1018,1045`). "pose does not fit" 포기 카운터도
  watchdog-trigger 재등록만 세고 정상 프레임 1장에 리셋된다(`session.py:1038-1041`) —
  run1은 나쁜 스냅 2번을 소비하고도 ~5–6 s 뒤에야 발화했다. 거리 게이트는 **카메라 범위만**
  보고 map-frame 타당성(풀 볼륨·바닥 근방) 검사는 전무하다.
- **임시 대응**: stale→재등록 직후 구간의 obj pose는 신뢰 금지. (메시 품질 쪽은
  2026-08-24에 게이트가 생겼다 — depth smear 비율 2.0 초과 프레임은 수집되지 않고,
  그런 캡처는 재구성 자체를 거부한다. `SMEAR_MAX_RATIO_*`, session.py.)
- **제대로 고치는 법**: 재등록 pose에 last-known 대비 위치/yaw 게이트를 perception 쪽에
  두고, 포기 카운터가 recovered-trigger 재등록도 세게 한다.

### 태그맵의 같은 ID 사본(54, 65)이 급기동 중 wrong-copy 매칭을 일으킨다 (2026-08-24)
- **증상**: `nav_213749/fixes.csv`에 "wrong-copy tag(s) dropped: 54, 65"(csv_t
  48.66–48.79, 49.76–50.13) — 정상 추종 650행 베이스라인에서는 0건, 급격한 yaw 스윙
  중에만 발생. 같은 창에서 full-frame 거부(reproj 3.0 px > 3, t=48.49)와 6.5 cm fix
  step이 동반돼 run3 말기의 x0 교란에 기여했다(주 원인 아님 — 물체 pose 붕괴의 2.3 s
  하류). 코드가 감지·드롭은 하지만, 맵에 같은 ID가 두 자리 있다는 것 자체가 공격적
  기동 때마다 무는 잠복 위험이다(사본 존재는 로그 문구에서 유도 — 물리 확인 필요).
- **임시 대응**: 54/65 물리 사본 중 하나를 제거/가리거나 맵에서 해당 ID 제외.
- **제대로 고치는 법**: 맵 빌드에서 중복 ID 거부, 또는 런타임에서 중복 ID를 상시 제외.

### `object_nav.max_distance_m: 10`이 자기 주석과 모순이고 **테스트를 깨고 있다** (2026-08-24)
- **증상**: `config/hw_mpc.yaml:312`의 값이 `10`인데 바로 옆 주석은
  "[측정: KNOWN_ISSUES 2026-08-09 — 0.3-0.8 m works, 2.4 m fails to register]"이고,
  `object_nav.py:102`의 출고 기본값은 **1.20**이다. `test_object_nav.py ::
  test_the_shipped_config_resolves`가 `max_distance_m <= 3.0`을 단정하므로 **현재 red**
  (35/36). 이 게이트는 "말도 안 되는 값"을 걸러내는 용도인데 10 m면 풀 전체보다 넓어
  사실상 아무것도 안 거른다.
- **왜 안 고쳤나**: 조종사가 의도적으로 넓힌 운용 한계라 조용히 되돌리면 워크플로가
  바뀐다. 다만 **2026-08-23 유령(카메라 0.44 m)은 1.20이든 10이든 통과**하므로 이 값이
  그 실패의 원인은 아니다.
- **제대로 고치는 법**: 실제로 쓰는 최대 거리로 정하고(1.5–3 m) 주석과 일치시키거나,
  넓혀야 할 이유를 주석에 적고 테스트 한계를 함께 올린다.

### `path_fillet_m: 0.0`이 MPCC 필렛 테스트를 깨고 있다 (2026-08-24)
- **증상**: `test_mpcc.py :: test_the_fillet_must_exceed_the_cross_track_error`가 red
  (24/25). 위의 **PID 경로추종 꼭짓점 교착**(2026-08-18) 항목이 임시 대응으로 지목한
  값이 정확히 `path_fillet_m: 0.15`인데, 현재 작업트리는 `0.0`(waypoint 모드)이다.
- **임시 대응/고치는 법**: 그 항목을 볼 것. 여기서는 **테스트 스위트가 red인 이유**만
  기록한다 — 새로 깨진 것으로 오독하지 말 것.

### 갠트리 엔코더 좌표가 태그맵에 **등록돼 있지 않다** (2026-08-23)
- **증상**: `config/fisheye_calibration.yaml`의 갠트리→지도 변환을 쓰면 답이 틀린다.
  세 군데가 동시에 깨져 있다.
  1. `R_gantry_to_slam`은 `src/tools/refine_R_gantry_to_slam.py`가
     `data/20260528/20260528_215858_recording`에 대해 fit한 값인데, **그 런의 anchor는
     태그 67**이었다(그 런 `tag_poses.csv`에서 태그 67이 원점, 1e-10). 지도의 anchor는
     25다. 둘 사이는 **179.81°** 차이다(`config/tag_map.yaml`의 태그 67 자세에서 유도).
     원인은 `src/gantry_panel.py`의 `_exp_autopick_anchor_tag`가 anchor를 **매 런
     이미지 중심에 가장 가까운 태그로 재선택**하고 `src/tagslam_core.py`의
     `_setup_tag_map`이 지도를 그 프레임으로 재표현하기 때문 — 즉 갠트리 런은 매번
     다른 세계에 떨어진다.
  2. **평행이동 캘리브가 없다.** `gantry_anchor_offset_mm` 키가 YAML에 아예 없어서
     대시보드는 "first-sample-zeroed" 폴백으로 도망간다. 원시 ‖p_est − p_gantry‖는
     p50 **1688.4 mm**(위 런 CSV로 재계산).
  3. `gantry_to_slam_scale: 1.03574608`이 **비등방**이다 — 축별 신축
     [0.9891, 1.0465, 1.0440], spread 0.057(위 런 CSV로 재계산). 굴절도 태그 크기도
     등방이라 범인이 아니다. 게다가 그 fit은 **Z 이동 0 mm**에 camera/gantry 경로비
     1.78×(툴 자체 경고 임계 1.3×)였다.
- **임시 대응**: **엔코더 대신 카메라를 쓴다.** 갠트리 패널의 `Tag map position`
  카드(`src/gantry_map_pose.py`)가 `rov_gui/control/tagnav.py`로 태그맵에 직접 PnP해서
  anchor-25 좌표를 낸다 — 위 세 값이 경로에 없다. **`R_gantry_to_slam`과
  `gantry_to_slam_scale`은 시각화 전용이며 어떤 숫자의 근거로도 인용 금지.**
  (다행히 둘 다 SLAM 해나 `config/tag_map.yaml`이나 `rov_gui/`에 들어간 적이 없어
  지도 자체는 오염되지 않았다.)
- **제대로 고치는 법**: anchor를 25로 고정한 뒤(아래 항목) **3축 모두 ≥3 m** 움직인
  궤적으로 재fit하고, 축별 신축이 등방으로 모일 때만 uniform scale을 믿는다. 그 전에
  비등방의 두 용의자를 각각 가른다 — `SCALE_MM_PER_UNIT` X=8.25 mm/unit
  (`src/gantry_runner.py`가 "whisker_dragging.py에서 그대로 복사"라고 자백한다)은
  1 m 지령 이동 + 줄자로, `fx/fy = 1.0411`(정사각 픽셀이면 1.000이어야 한다)은
  프레임 수를 늘린 재캘리브로. 평행이동은 카드와 엔코더를 같은 순간에 읽고 빼면
  나온다 — 주차 한 번이면 되는 10분짜리 절차다.

### 갠트리 Experiment 탭이 anchor를 **매 런 자동 재선택**한다 (2026-08-23)
- **증상**: `_exp_autopick_anchor_tag`(`src/gantry_panel.py`)가 "이미지 중심에 가장
  가까운 태그"를 anchor로 잡으므로, 두 런의 `camera_trajectory.csv`가 **서로 다른
  세계 좌표**에 있고 그 사실이 파일 어디에도 안 적힌다. 기본 anchor 값도 `1`인데
  (`gantry_panel.py`의 spin box, `tagslam_core.DEFAULT_ANCHOR_TAG_ID`) **태그 1은
  `config/tag_map.yaml`에 존재하지 않는다**. 덤으로
  `tagslam_core.DEFAULT_TAG_SIZE_M = 0.085`는 실제 0.170의 **정확히 절반**이라, CLI에서
  `--tag-size`를 빠뜨리면 0.5배 축척 지도가 조용히 나온다.
- **임시 대응**: 기존 런의 궤적을 비교할 땐 각 런의 anchor를 `tag_poses.csv`에서
  찾아(원점에 있는 태그) `config/tag_map.yaml`의 `world_T_anchor`로 anchor-25에
  올린 뒤 비교한다. `src/tests/test_gantry_map_pose.py`가 정확히 그렇게 한다.
  라이브 위치가 필요하면 Experiment 탭이 아니라 `Tag map position` 카드를 볼 것
  (그 경로는 anchor를 아예 안 쓴다 — `TagMap`이 파일의 `anchor_tag_id: 25`를 그대로 쓴다).
- **제대로 고치는 법**: PnP-only 모드에서는 auto-pick을 끄고 지도의 `anchor_tag_id`를
  강제한다. `DEFAULT_ANCHOR_TAG_ID`/`DEFAULT_TAG_SIZE_M`은 지도에 없는 값·절반 값이라
  기본값으로서 위험하니 없애거나 지도에서 읽게 한다.

### 갠트리 hold-to-jog가 **무한 거리**다 (2026-08-23)
- **증상**: `_start_jog`(`src/gantry_panel.py`)가
  `jog_single_axis(..., position_units=999999.0 × dir, relative=True)`를 던진다.
  X는 8.25 mm/unit이므로 **약 8.25 km**. 멈추는 것은 버튼 `released` →
  `stop_axis(mode=1)` 하나뿐이고, **발행과 정지가 둘 다 GUI 스레드에서 동기 실행**된다.
  release 시그널이 삼켜지거나 이벤트 루프가 멈추면 축은 펌웨어 소프트리밋이나 물리
  리밋스위치까지 간다. 패널은 소프트리밋을 더 이상 관리하지 않는다.
- **임시 대응**: 조그는 짧게 끊어 누른다. **ROV가 갠트리 아래 있는 동안에는 조그 대신
  waypoint CSV를 쓴다**(`./c3 gantry --waypoints-csv …`).
- **제대로 고치는 법**: `JOG_MAX_TRAVEL_MM` 상수(예 200 mm)로 이동량을 묶고
  `QTimer.singleShot(travel/speed + margin, stop)` 워치독을 건다. hold의 의미가
  "누른 만큼"에서 "한 번에 최대 N mm"로 바뀌므로 조종자 합의가 필요하다. 발행·정지를
  워커 스레드로 옮기는 안은 **권장하지 않는다** — 내부 뮤텍스가 없는 `.so`에 네 번째
  동시 호출자를 더하는 쪽이 고치려는 문제보다 나쁘다.

### `c3_camera/tests/test_option_sweep.py`가 20/32 실패한다 — API 드리프트 (2026-08-23)
- **증상**: `./c3 test`에서 `TypeError: StreamConfig.__init__() got an unexpected
  keyword argument 'mono_encode'`로 20개가 깨진다. `c3_camera/config.py`의
  `StreamConfig`에 `mono_encode`가 없는데 테스트가 계속 넘기고 있다. 커밋된 상태이며
  갠트리 작업과 무관하다(두 파일 다 HEAD와 동일).
- **임시 대응**: `./c3 test`의 이 파일 결과는 현재 신호가 아니다. 나머지 파일
  (host_depth 45/45, offline 58/58, preflight 43/43, src/tests 21/21)은 정상이므로
  그쪽만 보고 판단할 것.
- **제대로 고치는 법**: `mono_encode`가 언제 왜 빠졌는지 git 로그로 확인해서,
  테스트에서 지우거나 `StreamConfig`에 되살리거나 둘 중 하나로 정리한다.


### 지오펜스를 **완전히 제거**했다 — 풀 벽을 아는 것이 아무것도 없다 (2026-08-14)
- **무엇을 없앴나** (조종사 명시 요청): 플롯의 주황 점선 `GEOFENCE` 상자,
  START 전 경로 검사(line 양 끝점 / 사각형 네 모서리 — circle이 생긴 지금
  이 검사는 꼭짓점 집합으로는 표현이 안 되고 림을 샘플링해야 한다), 주행 중
  상자 이탈 시 자동 disengage. `hw_nav.yaml`의 `geofence_ned`·`geofence_frame`은 이제 읽지
  않고, MPC CSV의 `geofence_ok` 열도 빠졌다.
- **잃은 것**: 이제 **기체 위치를 이유로 멈추는 것이 하나도 없다.** 풀 밖으로
  9 m 나가는 line도 그대로 arm되고, 주행 중 벽 쪽으로 밀려도 컨트롤러는 계속
  간다. **circle(2026-08-17)은 입력한 태그에서 가장 멀리 가는 모양이다** —
  중심이 태그에서 R, 반대쪽 림이 2R(배포 반지름 0.5 m면 1.0 m)이라, 태그 옆에
  세워 놓고 START를 누르면 기체는 1 m 떨어진 곳까지 간다. 사람이 배치를
  확인하는 것 말고 막는 수단은 여전히 없다. 남은 보호는 E-STOP(Esc·헤더·MPC 패널) · DISENG · sink deadman(500 ms) ·
  engage 게이트(ARMED / MANUAL / 신선한 태그 fix·telemetry) · 주행 중 disarm ·
  telemetry 정지 · 모드 이탈 · 태그 상실 자동 해제 · 축 권한 상한.
- **임시 대응**: 수조 런에서는 **경로를 배치할 때 사람이 확인**하고, 조종사가
  E-STOP에 손을 두고 있을 것. 플롯의 `POOL` 실선은 여전히 벽을 그리지만
  **그림일 뿐 아무것도 강제하지 않는다**.
- **제대로 고치는 법**: 다시 필요해지면 펜스를 되살리는 것보다, 거부가 아니라
  **참조를 클램프**하는 쪽이 낫다(`geofence_clamp`가 그 용도로 있었다) — 배치를
  막지 않으면서 setpoint가 벽을 넘지 못하게 한다. git: 이 커밋 직전 상태.

### PID 경로추종이 **꼭짓점 10 cm 앞에서 영구 교착**한다 — leash × 코너 브레이크 × 데드밴드 (2026-08-18)
- **증상**: `--mpc` 패널 `pid` + `path_fillet_m: 0.0`(waypoint_vertex_stop) 사각형에서,
  기체가 한 꼭짓점 10 cm 앞에 서서 **아무 경고 없이 영원히 멈춘다.** solver_status 0,
  태그 21장·reproj 1.85 px·ambig 0·tag_age 0.15 s·축 포화 0 % — 계기는 전부 정상.
  2026-08-18 풀 세션에서 **두 런 연속 같은 꼭짓점**에서 발생, 조종사가 손으로 해제할 때까지
  각각 40 s / 45 s 정지
  (`data/20260818/0818_143802/mpc_143802.csv` s=1.93 정지,
  `.../mpc_143938.csv` s=5.92 정지 — 둘 다 tag 37 앵커 경로의 s≡2.0 m 꼭짓점, 즉
  origin tag 대각 반대편 모서리. 정지 중 hull wander p95 1.9 cm).
- **기구 (세 개가 겹쳐야 성립)**:
  1. `PathCursor.step`의 leash `cmd = min(cmd, theta + lead_m)` — 선체가 서면 setpoint도
     선다. → 위치 오차가 `path_lead_m` = 0.10 m에서 **하드 상한**을 갖는다
     (`e_along` p50 0.099, 관여 tick의 **83.6 %**가 0.095 m 초과 = leash 상시 포화).
  2. `speed_profile`이 필렛 없는 꼭짓점에서 v_ref를 creep 0.02까지 제동 → setpoint가 꼭짓점을
     2 cm 넘어서면 **접선이 다음 변으로 90° 돌아간다.** PID의 속도 FF `kd·v_ref_b`가
     surge 축에서 **0으로 사라진다**(kd_x = 59.7 N·s/m × 0.10 m/s = 5.97 N이 통째로 증발).
  3. 남은 상한: `kp·0.10 + i_max` = 51.73×0.103 + 4.0 = **9.33 N** [유도] — 실측 uX 평균
     9.47 N(sd 0.97). 직선 구간에서 움직일 때는 11.3–11.6 N이었다.
     9.47 N / `axis_gain.surge_n` 60 = ax_surge **0.158**, `pwm_dev_us` 평균 **26 µs**
     — hw_mpc.yaml이 스스로 적어 둔 T200/Basic ESC 널존 ±25 µs [스펙]와
     전 루프 실측 `speed = 0.692·(|axis| − 0.096)`의 문턱 바로 위. 실제 속도 0.000 m/s.
  → **선체가 못 가니 참조가 못 가고, 참조가 못 가니 명령이 못 커지고, 명령이 못 커지니
     선체가 못 간다.** 안정한 고정점이라 스스로는 절대 못 빠져나온다.
- **배경 조건**: 이 런은 leash가 처음부터 포화였다 — 지령 0.100 m/s에 실제 0.051 m/s
  (5.92 m / 116 s). 즉 속도 상자가 아니라 leash가 사실상의 제어법이었고, 코너 FF 손실
  2 N이 그대로 문턱을 갈랐다.
- **임시 대응**: `config/hw_mpc.yaml`에서 `path_fillet_m: 0.15`(기록상 최고 런 0817_110145가
  쓴 값)로 되돌리면 꼭짓점 자체가 사라져 v_ref도 접선도 연속이 된다. 겸해서 속도 상자를
  0.05–0.06 m/s로 낮출 것(기체가 실제로 내는 값).
- **제대로 고치는 법**: 두 가지가 따로 필요하다.
  (a) 적분 클램프 `pid.i_max_n` = [4.0, 5.0, 5.0] N이 **데드밴드 탈출에 필요한 힘보다 낮다** —
      "정상 오차로 영구 정지"의 교과서적 원인. 최소 8–10 N로 올리거나, leash 포화 상태에서만
      푸는 조건부 클램프.
  (b) **교착 워치독이 없다.** `e_along ≥ 0.95·lead_m` && `|v| < ε`가 N초 지속되면 경고/중단
      해야 한다. 지금은 MPCC 데드락 2건(memory: mpcc-contouring-control)과 똑같이
      solver status 0으로 조용히 실패한다.

### 미스 사유가 1 Hz로만 기록돼 CSV가 dropout을 15.6배 과소보고 (2026-09-06)
- **증상**: [`workers.py:290`](rov_gui/control/workers.py#L290)의
  `if t - self._last_miss_note > 1.0:` 때문에 거부된 프레임은 **초당 한 줄**만
  `fixes.csv`에 남는다. 실측: `data/20260906/
  0906_192348/nav_192348`에서 거부 **390건이 25행**으로 기록됐다(15.6배).
- **왜 위험한가**: `fixes.csv`의 **행 비율로 채택률을 계산하면 틀린다**
  (채택 행은 전수, 거부 행은 1 Hz로 솎였다). 2026-09-06 진단에서 실제로
  이 함정에 걸렸고, 진짜 숫자는 `frames.csv` + `detections.csv`를 다시 푸는
  재현으로만 나온다.
- **임시 대응**: 채택률은 재현으로 계산한다
  (`rov_gui/control/tagnav.py`에 녹화 코너를 그대로 물리면 accept/reject가
  100% 재현된다 — 채택 프레임의 `reproj_rms_px`가 |Δ| 최대 0.0051 px로 일치).
- **제대로 고치는 법**: 문구는 1 Hz로 유지하되, 사유별 **집계 카운터**를
  매 프레임 갱신해 런 종료 시 한 줄로 남긴다(또는 `frames.csv`에 사유 열 추가).


## 🐛 테스트 / 스크립트 함정

### `slam/9_9_26/`이 `slam/grasp_9_9_26/`으로 바뀌었는데 옛 경로가 남아 **ablation 러너가 run을 조용히 건너뛴다** (2026-10-01)
- **증상**: `~/Desktop/data collection/slam/`에는 이제 `grasp_9_4_26`, `grasp_9_9_26`, `peg_in_hole_9_27_26`, `peg_in_hole_9_28_26`만 있다
  (2026-10-01 `ls`; `data/20260915/0915_131900_heldout_D_vs_A/D_dinov3b_ep195.log:2`는 옛 경로로 로드됐으므로 이름 변경은 그 뒤).
  옛 경로를 가리키는 곳: `tools/run_ablations.sh:44-45`(`DS_9926_DEPTH`·`DS_9926_RGBD` → E/A/C/D, rgbd_*, depth_7d_*/rgbd_7d_* spec 전부),
  `external/UMI_aquatic/diffusion_policy/config/task/umi_rgbd_5d.yaml:115`, `umi_rgbd_7d.yaml:127`, `umi_handheld/build_dp_rgbd_zarr.py:5`
  (docstring), `external/UMI_aquatic/tests/print_rp_label_stats.py:6,48`, `dataset_rgbd.zarr.zip`의 `.zattrs`. 러너는 `:97`에서
  "!! <name>: 데이터셋 없음 … — 건너뜀"을 찍고 **다음 run으로 넘어간다**(크래시 없음) — 밤새 배치를 걸면 아침에 해당 run이 없다.
- **임시 대응**: 배치 시작 직후 러너 로그에서 "건너뜀"을 확인하거나, `slam/9_9_26 → grasp_9_9_26` symlink.
- **제대로 고치는 법**: 위 경로를 `grasp_9_9_26`으로 갱신(또는 symlink를 의도적으로 두고 여기 기록)하고, 러너가 데이터셋 없음을
  skip이 아니라 non-zero exit로 처리.

### `docs/DINOV3_ENCODER_STUDY.ko.md`의 "새 캔 0913 폐루프 베이스라인"은 **대부분 캔이 아니다** (2026-10-01)
- **증상**: `:25`, `:134`(원 수치 `docs/measurements/dinov3_encoder_study_20260913/` followup_baseline-0913-yaw.json)가 0913 8폴더를
  "새 캔"으로 묶어 A의 r +0.116, L/R 균형 3런(163921/170213/173702) pooled r +0.034를 DINOv3가 이겨야 할 기준선으로 둔다.
  그러나 8폴더 중 5개(161733/163921/165748/170213/173702)는 검은 긴 목 인형(청록 머리·주황 발)이고 **균형 3런은 전부 인형**이다
  [측정: `data/20260913/0913_163921/policy_obs/rgb/000768.jpg` 인형, `data/20260913/0913_155559/policy_obs/rgb/000037.jpg` 주황 테이프 캔 —
  2026-10-01 직접 확인; 나머지 런은 같은 날 workflow verifier가 런당 1–2장 확인]. 주황 캔이 찍힌 수중 기록은 6런
  (0911_171926_observe/173045/175144, 0913_155559/160850/161350), JPEG 5,330장뿐이고 0913 세 런은 캔이 왼쪽에 몰려 있다
  (L/C/R 37/1/0, 9/0/2, 10/0/0 — 같은 json). 0914–0921 수중 런은 인형.
- **영향**: 그 r 값은 "새 캔" 기준선이 아니라 주로 인형 기준선이다. 주황 캔으로 yaw 응답을 판정할 L/R 균형 런은 아직 없다.
- **제대로 고치는 법**: 문서 두 줄의 라벨 정정 + 주황 캔을 좌우 번갈아 둔 observe 런을 `--record-depth`로 새로 찍는다.

### `rov_gui/tests`를 돌리면 git이 추적하는 MPCC 생성 솔버가 **모델 항력(4.03)으로** 덮어써진다 (2026-09-30)
- **증상**: `rov_gui/control/_mpcc_gen/heavy_gripper/`(`.c/.o/.so` + `acados_ocp_mpcc.json`)는 커밋된 생성물이고
  HEAD는 실기 항력 `-86.7 / -133.8`로 생성돼 있다. 테스트(예: `test_path_cost.py:677`의 `HwMpcc(MpcConfig(...))`)는
  `plant:` 오버라이드 없이 `AcadosMPCC()`(`build=True, generate=True`)를 만들어 같은 폴더를 `-4.03 / -6.22`로 다시 쓴다
  (2026-09-30 전체 테스트 뒤 json `f_expl_expr` 비교로 확인). 습관대로 `git add .`하면 틀린 항력의 솔버가 커밋된다.
- **비행엔 무해**: GUI는 기동 때마다 `MpcWorker`가 `HwDobMpc`(항력 오버라이드 적용) → `HwMpcc` 순으로 **다시 생성**한다
  (`rov_gui/control/workers.py` 1170-1203). 같은 순서의 헤드리스 재생성이면 `.c/.o/.so`가 HEAD와 바이트 동일로 돌아온다.
- **임시 대응**: 테스트 뒤 `git status`에 `_mpcc_gen`이 보이면 커밋 전에 `git checkout -- rov_gui/control/_mpcc_gen/` 하거나
  GUI를 한 번 띄운다. env 경로(`Makefile`·json의 `code_gen_opts`)만 다른 건 GUI 기동이 원래 만드는 차이다.
- **제대로 고치는 법**: 테스트는 임시 디렉터리에 생성하게(`GEN_DIR` 주입)하거나, 생성물을 git에서 빼고(.gitignore) 기동 빌드에 맡긴다.


### depthai를 cv2/torch보다 **먼저** 초기화해야 USB 카메라가 열거된다 (2026-09-06)
- **증상**: 한 프로세스에서 `cv2`와 FoundationStereo(=torch+CUDA)를 먼저 import하면
  그 프로세스에서만 USB OAK-D-W가 `getAllAvailableDevices()`에 **30초 넘게 안 뜬다**.
  같은 시각에 띄운 다른 파이썬은 즉시 본다. PoE인 C3는 멀쩡해서 "USB 카메라가 고장났다"로
  읽힌다 [측정 2026-09-06: 도구 안에서 20회 × 1.5 s 재시도 내내 C3만, 별도 인터프리터는
  둘 다].
- **대응**: `import depthai` → 열거 → **두 장치를 먼저 연다** → 그 다음 무거운 import.
  `rov_gui/tools/depth_two_cameras.py`와 `depth_live_view.py`가 그 순서로 되어 있다.
- **제대로 고치는 법**: 원인을 XLink 초기화 순서로 좁히고(추정), 필요하면 진입점에서
  한 번 열거해 두는 헬퍼로 강제.

### 카메라 도구를 `kill -9` 하면 장치가 booted로 남고 워치독이 안 살린다 (2026-09-06)
- **증상**: SIGKILL 후 USB PID가 `03e7:2485`(unbooted) → `03e7:f63b`(booted)로 바뀌고
  `getAllAvailableDevices()`가 빈 배열, 직접 연결은 `X_LINK_DEVICE_ALREADY_IN_USE`.
  2분 넘게 자동 회복되지 않았다.
- **함정**: heredoc으로 띄운 프로세스는 커맨드라인이 `python -`이라 **`pkill -f <스크립트명>`이
  안 잡는다**. 그래서 "아무도 안 잡고 있는데 안 된다"처럼 보인다.
- **대응**: 인터프리터 경로로 찾아서 **SIGTERM**한다. 정상 종료가 장치를 돌려준다.
  ```
  ps aux | grep 'rovgui-pose/bin/python'
  kill -TERM <pid>
  ```
  실측: SIGTERM 6초 뒤 USB PID가 `03e7:2485`로 복귀, 뽑았다 꽂을 필요 없었다.
- **제대로 고치는 법**: 카메라를 여는 도구에 SIGTERM 핸들러를 달아 `device.close()`를
  보장하고, 도구는 `kill -9` 하지 말 것.


### `test_object_nav.py`가 먼저 돌면 `test_policy.py`가 23개 깨진다 — 순서 의존 (2026-09-06)
- **증상**: 파일 단독으로는 `test_policy.py` **42/42 통과**. 전체 스위트에서는 26개 실패.
  최소 재현: `pytest rov_gui/tests/test_object_nav.py rov_gui/tests/test_policy.py`
  → **23 failed, 55 passed**. `pytest rov_gui/tests/test_replay.py
  rov_gui/tests/test_policy.py` → 51 passed. 즉 오염원은 `test_object_nav.py`다.
- **두 갈래로 나타난다**: (1) 플랜 인테이크가 죽는다 —
  `w.replay["installed"] == 0`, `rp["late"] == 0` 등 `MpcWorker`가 플랜을 아예 안 받는다;
  (2) 벽시계 의존 — `engage refused: imu stale (0.44s)`. 후자는 앞 파일이 소비한
  wall-clock에 좌우돼서, 무관한 코드가 수 ms만 움직여도 실패/통과가 뒤집힌다.
- **내 변경 탓이 아님을 확인**: 2026-09-06 `--record-depth` 관련 파일 4개를 전부 되돌리고
  같은 명령을 돌려도 **23개가 그대로 실패**한다(되돌린 상태 23, 되돌리기 전 24 — 늘어난
  1개는 위 (2)의 imu-stale 타이밍 케이스).
- **임시 대응**: 정책 테스트는 파일 단독으로 판단한다. 전체 스위트의 실패 수를 회귀
  신호로 쓰지 말 것.
- **제대로 고치는 법**: `test_object_nav.py`가 남기는 전역 상태를 찾아 fixture로 격리하고,
  시계 의존 검사는 monotonic 주입으로 바꾼다.


### 1280×800에서 **컬러 스트림이 디바이스 프레임의 3.3%를 떨어뜨린다** (2026-08-04)
- **발견**: 2026-08-04, maxres를 8.4 → 29 fps로 고친 뒤 실기 take 검증 중.
- **증상 (시퀀스 번호로 센 정확한 값, `bench_stereo`와 같은 파이프라인, 10 s/행)**:

  | | age | 드롭 | fps |
  |---|---|---|---|
  | 컬러 ON | 153.0 ms | **3.67%** | 28.90 |
  | 컬러 OFF | 161.5 ms | **0.33%** | 29.90 |

  기본 프로파일(320×200)과 near 모드(19.7 fps)에서는 안 나타난다 — 여유가 없는 건
  1280×800 · 29 fps 조합뿐이다.
- **디스크가 아니다**: writer 6→12 스레드 + queue 128→256으로 올려도 개선되지 않았다.
  XLink/USB 대역 쪽으로 보이지만 **원인 미확정**. 참고로 장치는
  `maxUsbSpeed=SUPER_PLUS`를 요청해도 `usb=SUPER`로 붙는다.
- **임시 대응**: `record.streams:`에서 `color`를 빼면 된다(0.33%). 프레임은 타임스탬프로
  기록되므로 유실이 있어도 take 자체는 유효하다.
- **제대로 고치는 법**: 컬러 MJPEG 품질/해상도를 낮춰(97 → 80, 또는 800p 대신 720p)
  대역을 확보하고 `umi_handheld/bench_stereo.py`로 재측정. 컬러를 downstream이 실제로
  쓰기 시작하기 전까지는 결정할 근거가 없다 — 현재 아무것도 읽지 않는다.

### 1280×800의 **150 ms 잔여 지연은 해상도 자체**라 설정으로 못 줄인다 (2026-08-04)
- **증상**: stereo 입력 큐를 8 → 1로 고쳐 ~95 ms를 걷어낸 뒤에도 프레임이 호스트에
  도착할 때 이미 **153 ms** 지났다(29 fps 기준 ~4.4 프레임). 프리뷰가 그만큼 늦게 보인다.
- **원인 후보를 전부 배제했다** (전부 실측, 디바이스 고정): 호스트 루프 아님(디스플레이
  없을 때 640 kHz, `draw_overlay` 3.0 ms) · XLink 대역 아님(`streams`에서 right/color를
  빼도 153 → 159 ms로 변화 없음) · `setXLinkChunkSize(0)` 무효 · 호스트 큐 크기 무효 ·
  후처리 필터 거의 무효(median/temporal 전부 끄면 152.8 ms). 남는 변수는 해상도뿐:
  같은 필터 체인으로 **640×400은 36.1 ms**.
- **임시 대응**: 지연이 문제인 작업(정밀 티칭 등)에서는 `configs/pipeline.yaml`(320×200
  depth)을 쓴다. 1280×800을 쓰는 한 ~150 ms는 따라온다.
- **제대로 고치는 법**: 디바이스 내부 파이프라인 깊이라 depthai 설정으로는 못 만진다.
  줄이려면 프레임 주기를 줄여야 하는데(지연 ≈ 단수 × 주기) 800p에서 stereo 코어가
  29–30 fps가 한계다. 720p(같은 MinZ, 행 10% 감소)에서 21.4 fps가 측정된 적 있어
  기대하기 어렵다 — 재측정 없이 단정하지 말 것.

### `record.py`는 depthai **2.x 전용**인데 base conda 환경은 3.5.0 (2026-08-04)
- **증상**: base(`/home/bdml/miniforge3`)에서 `--source device`를 돌리면 즉시
  `AttributeError: module 'depthai.node' has no attribute 'XLinkOut'`. `dai.node.XLinkOut`,
  `dai.RawStereoDepthConfig`, `Device.getOutputQueue`가 전부 3.x에서 사라졌다
  (record.py:154-159, 177, 197, 216, 403-409).
- **임시 대응**: depthai 2.32.0.0이 있는 인터프리터로만 실행 —
  `~/.venvs/c3-depthai/bin/python`, 또는 conda `robust`(2026-08-05에 설치함, 단 위의
  numpy 항목을 먼저 읽을 것). 같은 폴더의 `test_oak_depth.py`는
  `DEPTHAI_MAJOR_VERSION`으로 양쪽을 분기하므로 base에서도 돌아가고, 그래서 두
  스크립트가 **서로 다른 스택**에서 돈다.
- **진행 (2026-08-05)**: `c3_camera` 쪽은 리포 루트 `./c3` 래퍼가 인터프리터와 작업
  디렉터리를 고정해서 해소됐다. `umi_handheld/record.py`는 **아직** 그대로다 —
  래퍼도 버전 분기도 없어서 `python -m umi_handheld.record`가 어떤 python을 잡느냐에
  달려 있다.
- **제대로 고치는 법**: record.py에도 test_oak_depth.py와 같은 버전 분기를 넣거나,
  `./c3`와 같은 래퍼를 umi_handheld에도 두어 인터프리터를 고정한다.

### `xlink_out_queue`는 적용했으나 **아직 실측하지 않았다** (2026-08-06)
- **한 일**: `build_device_pipeline`이 `XLinkOut` 노드들의 입력 큐를 DepthAI 기본값
  (8, blocking) 그대로 두고 있었다 — 체인에서 유일하게 설정도 측정도 없던 버퍼링 단계다.
  `stereo_input_queue`와 같은 방식으로 `record.xlink_out_queue: {size: 1, blocking: false}`
  키를 만들어 left/depth/right/color 전부에 적용했다.
- **왜 유망한가**: stereo **입력** 큐를 8→1로 바꿨을 때 지연이 252 → 153 ms로 떨어졌다
  (record.py:283-297). 출력 큐도 같은 종류의 자리인데 여기만 손대지 않았었다.
- **미검증**: 작업 시점에 벤치 카메라가 연결돼 있지 않아
  (`dai.Device.getAllAvailableDevices()` → none) 전후 비교를 못 했다. **효과가 있다고
  주장하지 말 것.**
- **재는 법**: `python -m umi_handheld.bench_stereo --config configs/pipeline.yaml --seconds 10`
  — 이번에 `age`(호스트 도달 시점의 프레임 나이, ms 중앙값)와 `drop%`(시퀀스 번호 결번) 열을
  추가해서 이제 잴 수 있다. `xlink_out_queue: {size: 8, blocking: true}`로 되돌린 행과
  나란히 찍어 비교한다(같은 장면·같은 디바이스여야 유효).
- **기대치**: `KNOWN_ISSUES` 아래 항목대로 1280×800의 잔여 ~150 ms는 해상도 자체다.
  640×400의 기준선은 36 ms이므로 여기서 걷어낼 수 있는 몫은 그보다 작다.

### 리포 루트의 출력 디렉터리 `zarr/`가 `import zarr`를 가로챈다 (2026-08-06)
- **증상**: `robust`에서 `python -m umi_handheld.build_zarr`를 돌리면
  `AttributeError: module 'zarr' has no attribute 'open'`. 리포 루트가 `sys.path[0]`이고
  거기에 산출물 디렉터리 `zarr/`가 있어서, 진짜 zarr가 **설치돼 있지 않을 때** 그 디렉터리가
  namespace package로 잡힌다 (`import zarr` → `_NamespacePath(['<repo>/zarr'])`).
- **더 나쁜 건 진단을 망친다는 점**: 리포 루트에서 `import zarr`가 성공하므로
  "이 환경에 zarr가 있다"로 오독된다. **`robust`에는 zarr가 없다** (`pip list` 확인:
  robust 무, `~/.venvs/c3-depthai` 2.18.3). 정규 패키지가 설치돼 있으면 namespace
  portion을 이기므로 venv에서는 리포 루트에서도 정상 동작한다.
- **임시 대응**: `build_zarr.py`는 `~/.venvs/c3-depthai/bin/python`으로 실행한다
  (실측: 리포 루트에서 `zarr 2.18.3` 정상 해석, session_syn 72프레임 빌드 성공).
  환경 확인은 리포 **밖**에서 — `cd /tmp && <python> -c "import zarr; print(zarr.__version__)"`.
- **제대로 고치는 법**: `robust`에 zarr를 설치하거나(numpy<2 제약 확인 필요), 산출물
  디렉터리를 `zarr_out/`처럼 모듈명과 겹치지 않게 바꾼다. 후자가 근본 해결이지만
  기존 7개 데이터셋 경로가 전부 바뀐다.

### C3 공장 캘리브레이션은 **수중** 값인데 코드·문서가 "IN-AIR"라고 단정 → 공기 중 촬영분 depth가 ~1.33배 과대 (2026-08-04)
- **발견**: 2026-08-04 (OAK-D W 핸드헬드 파이프라인 계획 중, 사용자 지적으로 재검증)
- **사실 1 (벤더)**: MarineSitu C3는 **개체별로 수중 캘리브레이션되어 출고**된다.
  Blue Robotics 제품 페이지 *"Each unit is individually calibrated for underwater
  operation"*, 포럼에서 Tony White가 *"the calibration Marine Situ performs takes
  place entirely underwater. It uses the standard checkerboard approach with openCV"*.
- **사실 2 (우리 EEPROM 실측)**: `datasets/*/calibration.json` 10개 전부 동일한 한 벌이고,
  거기 담긴 K+왜곡(rational 8계수)을 **순방향 투영**해 실제 화각을 재면
  **CAM_A 63.7° / CAM_B 85.6° / CAM_C 85.1°** (HFOV). 공기 중 OAK-D-W 스펙은
  **95 / 127 / 127°**, 평판포트 Snell 예측은 **67.2 / 84.3 / 84.3°** → 세 카메라 전부
  물속 예측과 3.5° 이내, 공기 스펙과는 최대 41.9° 차이.
  (산출물: `python calib/fov_audit.py --audit`, 방법·통제실험 [calib/FOV_AUDIT.md](calib/FOV_AUDIT.md))
  세 카메라가 독립적으로 같은 결론을 준다. 즉 EEPROM = **수중** 캘리브레이션.
- **증상**: 이 캘리브레이션으로 **공기 중에서** 찍으면 `Z_reported = (fx_water/fx_air)·Z_true
  ≈ 1.33 · Z_true` — 즉 **depth가 약 33% 멀게 나온다**. 정류(rectification)도 매질이
  달라 어긋난다. `c3_camera/datasets/*`와 `recordings/*`는 전부 실내(공기 중) 촬영이므로
  **기존 C3 RGB-D 데이터셋의 depth 스케일은 계통 오차를 갖는다**. 반대로 실제 물속에서는
  이 캘리브레이션이 **맞다**.
- **틀린 문구가 박혀 있는 곳**(전부 근거 없이 하드코딩된 우리 주석):
  `c3_collect.py:330`(이 문자열이 **모든 데이터셋의 calibration.json `note`로 기록됨**),
  `dataset.py:593, 649, 909`, `geometry.py:15`, `TESTING.md`(T2를 "전부 공기 중,
  수중은 미보정 시 스케일 ~1.33배"로 서술 — 부호가 반대).
- **임시 대응**: 공기 중에서 찍은 C3 depth는 중심부 기준 **÷1.33**으로 읽을 것. 단
  **깨끗한 스칼라가 아니다** — 왜곡 모델까지 매질이 어긋나 있어 주변부로 갈수록 반경
  방향 추가 오차가 붙는다. MinZ 표(400p **300 mm**)는 수중 fx 기준이므로
  **공기 중 실제 최소거리는 ≈225 mm이고 그것이 300 mm로 보고된다**.
  보고되는 바닥 300 mm(및 `--extended` 150 mm)는 이제 실측이다
  `[측정: c3_camera/datasets/*/depth/, recordings/*/depth/ — 47,270 프레임 전수,
  바닥 정확히 300/150 mm, 그 아래 0 픽셀]`. 공기 중 물리거리 ≈225 mm 쪽은 여전히
  `[유도: fx_air = fx_water/1.333 대입. 미실측 — depth_accuracy/rungs.csv 부재,
  줄자 검증은 한 번도 실행된 적 없음]`.
- **제대로 고치는 법**: (a) 위 5곳 문구를 사실대로 정정하고 `calibration.json`에
  `medium: "water"` + 근거를 명시, (b) 카메라가 네트워크에 돌아오면 EEPROM 메타
  (`getEepromData()`의 `batchTime`/`boardCustom`/`productName`)를 읽어 provenance를 못박고,
  (c) 공기 중 작업용 별도 in-air 캘리브레이션을 체커보드로 떠서 매질별로 선택 가능하게 한다.
  (a)만 해도 이 항목의 위험 대부분이 사라지므로 우선순위 높음. 고치면 이 항목 삭제.

### `config/config.yaml`의 수면·풀 깊이가 **ROV 실기록과 모순**한다 (2026-08-23)
- **증상**: `water.surface_height_m: 0.8255` / `pool.depth_m: 1.143`은 2026-05-25 최초
  커밋 이후 한 번도 안 바뀌었는데(`git log -- config/config.yaml`), ROV 실기록이 둘 다
  넘긴다.
  1. **압력센서가 개입하지 않는 하한**: 채택된 fix의 `z_ned` 최저 **−1.296 m**
     (p50 −1.138, n=22094 — `data/*/*/nav_*/fixes.csv`, 2026-08-13/14 집계).
     바닥 태그 매트가 z=0이고 +z가 아래니까, 기체가 매트 위 1.296 m에서 **잠긴 채**
     바닥 태그를 보고 있었다는 뜻 → 매트 위 물기둥이 최소 1.3 m.
  2. **압력에서**: engage마다 `StateAssembler.calibrate_z_offset`이
     `z_off = z_tag − depth`를 잡는다(`rov_gui/control/state_assembler.py:104-108`).
     `−z_off`(= 매트 위 수면 높이)를 events.log의 `datum p0` z와 같은 런
     `*_rov.jsonl`의 `GLOBAL_POSITION_INT.relative_alt`로 복원하면
     2026-08-18 = 1.386 / 1.409 / 1.415 / 1.418 / 1.440 m,
     2026-08-17 = 1.641 / 1.654 / 1.664 / 1.686 / 1.722 m.
  즉 config는 ~0.6 m 얕고, "45 in 풀"이라는 1.143 m조차 하한 1.3 m보다 작다.
  (`docs/MEASUREMENT_AUDIT.md`가 이미 `claude.md`의 풀 치수 줄을 UNVERIFIED로 찍었고
  거기 적힌 width도 config와 어긋난다.)
- **부수 발견 — 같은 물인데 두 세션이 0.24 m 다르다**: ArduSub의 depth 0점이
  **부팅마다 재설정**되기 때문이다(depth = (ground_pressure − p)/9800/`BARO_SPEC_GRAV`,
  ground_pressure는 부팅·preflight baro cal 때 재취득). 하루 안에서는 ±3–4 cm로
  일관하므로 **압력에서 나온 수면 높이는 그 세션 안에서만** 유효하다. Bar30의 절대
  정확도는 ±200 mbar(≈±2.04 m 담수)라 절대압으로는 아무것도 못 정한다
  [스펙: bluerobotics.com/store/.../bar-depth-pressure-sensor/].
- **임시 대응**: config의 두 값을 **어떤 숫자의 근거로도 인용 금지**. 굴절 보정이
  `water.surface_height_m`를 쓰므로(`src/tagslam_core.py:880`) 갠트리/ZED 굴절 결과도
  이 값에 매달려 있다 — 물이 찬 날의 굴절 재계산은 신뢰하지 말 것.
  `rov_gui/backends/hardware.py:1337`의 폴백(절대압 − 1013.25 hPa)은 Bar30 절대
  오프셋을 그대로 삼키므로 **수위 근거로 쓰면 안 된다**(GLOBAL_POSITION_INT가 살아
  있으면 그 경로는 안 타지만, 죽으면 조용히 갈아탄다).
- **제대로 고치는 법**: 줄자로 풀 깊이와 그날 수위를 재서(±3 mm, 10분) 날짜와 함께
  config에 적는다. ROV 쪽은 ① 기체를 **물 밖에서** 부팅해 0점을 대기압에 잡고,
  ② engage 때의 `z_off`와 depth를 `*.meta.json`에 기록하면(지금은 events.log +
  jsonl 조합으로만 복원 가능) 수위가 런마다 기록되는 양이 된다. Bar30의 선체 내 z
  오프셋(전자부 엔드캡)은 리포 어디에도 실측이 없어 **절대값은 ±5–10 cm이 한계**다 —
  같은 자리에 앉혀 재는 **변화량**은 lever arm이 소거돼 mm급으로 나온다.

### `dataset.py` telemetry CSV 스키마가 "첫 메시지 승자독식" — 세션마다 열이 달라짐 (2026-08-03)
- **발견**: 2026-08-03 (`c3_option_sweep.py` 작성 중 API 매핑 워크플로)
- **증상**: `DatasetWriter.mavlink_rows`가 그룹(telemetry/imu_rov/control)별로 **처음
  도착한 메시지 하나의 필드 집합으로 헤더를 확정**하고(dataset.py:510-517),
  `DictWriter(extrasaction="ignore")`(dataset.py:195)라 이후 다른 타입의 필드는
  **조용히 버려진다**. 실측: `datasets/dataset_20260803_105221/telemetry.csv`는 첫
  레코드가 AHRS2였던 탓에 열이 `altitude,lat,lng,pitch,roll,yaw` 6개뿐이고,
  같은 파일에 섞인 **VFR_HUD 1169행은 전 열이 공백**이다(11개 메시지 타입 / 7935행).
- **왜 스윕에서 더 나쁜가**: 어느 메시지가 먼저 도착하는지는 **레이스**라, 동일한
  차량 트래픽인데도 셀마다 telemetry.csv 스키마가 달라져 **셀 간 비교가 깨진다**.
- **임시 대응**: 텔레메트리를 쓸 때는 `msg_type`으로 먼저 필터링하고, 빈 열은
  "그 메시지에 그 필드가 없다"가 아니라 "헤더에서 잘렸다"로 해석할 것. 원본은
  `metadata.json`의 `mavlink.message_counts`로 교차 확인.
- **제대로 고치는 법**: 그룹별 헤더를 `ALL_MESSAGES`의 필드 합집합으로 미리 확정하거나,
  메시지 타입별로 파일을 분리(`telemetry_VFR_HUD.csv` …)한다. 고치면 이 항목 삭제.

### `mpc_tuned` meta가 **회전이 실제로 걸렸는지**를 기록하지 않는다 — 등방으로 날고도 "tuned"로 남는다 (2026-08-25)
- **발견**: 2026-08-25 (mpc_tuned formulation 추출 + 12개 주장 적대적 검증 워크플로)
- **증상**: `HwDobMpc.meta()`(mpc_bridge.py:641-643)는 `self.tuned`(모드 이름이 `_tuned`로
  끝나는가) **하나만** 보고 `cost_frame: "path (along/cross split)"` + `path_cost` 블록
  (q_along 75 / q_cross 1200)을 찍는다. 그런데 가중치 회전이 실제로 솔버에 들어가는 조건은
  그보다 좁다 — `_apply_stage_weights`(mpc_bridge.py:554-558)는 **path plan이 설치돼 있어야**
  회전하고, plan의 유일한 생산자(workers.py:1634-1641)는 `cfg.path_following`이 참이고
  `PathCursor`가 있을 때만 도달한다(workers.py:1613, 2468). 즉 `path_following: false`로
  `mpc_tuned`를 날리면 **미션 전 구간을 등방 300/300으로 날고도** meta는 tuned라고 적는다.
  path_cost.py:70-72가 경계하는 바로 그 실패 모드("런은 baseline으로 나는데 meta는 tuned")이고,
  실제로 회전이 걸렸는지를 아는 `_w_tuned` 플래그는 어디에도 기록되지 않는다.
- 런타임 확인(heavy_gripper, prebuilt `heavy_gripper_rti`, `solver.cost_get`으로 되읽기):
  built `[[300,0],[0,300]]` / plan 있는 tuned tick `[[75,0],[0,1200]]` / station tick(plan=None)
  `[[300,0],[0,300]]`. **산출물 미보존** — 재현은 위 세 상태에서 `cost_get(0,"W")[:2,:2]`.
- **부수 결함**: mpc_bridge.py:550이 문서화한 세 번째 no-op("plan이 `psi_path`를 안 갖고 있으면
  추측하지 말고 fall back")은 **죽은 코드**다. `NedPlan.__post_init__`(path_geometry.py:336-342)이
  `psi_path`를 항상 채우고, 레거시 4-인자 생성이면 `yaw_ned`를 대신 넣는다 — 등방 복귀가 아니라
  **기체 헤딩으로 조용히 회전**한다(`heading_follow: false`에선 경로 접선과 무관한 각도).
- **임시 대응**: tuned 런을 인용하기 전에 그 런의 `controller.json`에서 `path_following: true`를
  확인할 것. 현재 config는 true(hw_mpc.yaml:105)라 정상 경로이고, 아직 tuned 런 자체가 0건이다.
- **제대로 고치는 법**: `meta()`가 `_w_tuned`(또는 회전이 실제로 쓰인 tick 수)를 같이 기록하고,
  `_tuned` 모드가 `path_following: false`와 함께 무장되면 거부하거나 최소한 경고한다.
  `NedPlan`의 `psi_path` 폴백은 조용한 yaw 대입 대신 예외로 바꾼다. 고치면 이 항목 삭제.

### 📊 compare sweep 재실행 대기 — MPC surge 박스 8 → 30 N 변경 이후 (기록 경계)
- **발견/변경**: 2026-07-24 (wave-crossover 워크플로에서 벤치마크 불공정 발견 → 같은 날 수정)
- **증상**: `params.U_MAX[0]`을 8 → 30 N(= PID `f_max`)으로 고쳤으므로 **기존 recordings의 모든
  mpc/dobmpc 결과는 낡은 8 N 박스 하에서 측정된 것**. 특히 `compare_20260724_160210` 등에서
  관측된 "강파랑에서 PID가 MPC보다 낫다"는 **벤치마크 아티팩트이며 인용 금지**(ablation:
  storm PID−MPC −4.24 → +8.60 cm 부호 역전, crossover 소멸).  [UNVERIFIED: 산출물 없음 — docs/MEASUREMENT_AUDIT.md]
- **임시 대응**: 새 run meta는 `controller.u_max`를 기록하므로 신/구 기록은 구분 가능
  (키가 없으면 낡은 8 N). 경계를 넘어 결과를 합산하지 말 것.
- **제대로 고치는 법**: full sweep(`experiments/run_compare.py`, 6 sea state × 5 mode × 3 ctrl)
  재실행 후 이 항목 삭제. 재실행은 사용자 몫.

### w_hat ±50 클립 — 단일 스칼라가 N·N·m 겸용 + 발동 무음 + 상수 중복
- **발견**: 2026-07-22 (mpc_acados 리뷰 워크플로: control-theory ×2 + simulation-advisor,
  verifier 수치 검증 11/12 verified)
- **증상**: `np.clip(w_hat, ±50)`(mpc_acados.py:170, mpc.py:198)에 대해 —
  (1) 힘 채널 50 N은 이 환경 물리 상한(realistic X≈35 / Y≈37 / Z≈3 N)보다 위라 사실상
  안 물리지만, 스웨이 보수 스택(추력한계 속도로 파정 역류 과도) ≈85–100 N에서는 물리고  [UNVERIFIED: 산출물 없음 — docs/MEASUREMENT_AUDIT.md]
  이 영역이 정확히 acados blowup 레짐; (2) 토크 채널 50 N·m은 authority(8–10 N·m)의
  5–6배·물리 상한(Munk 미스매치 ≤10 N·m)의 5배 — 회전 유효관성 0.45–0.76 kg·m²라
  40 N·m대 추정 스파이크가 통과하면 66–111 rad/s² 예측 → slack/RTI blowup;
  (3) 발동 카운터/로그 없음 + 로그되는 wh0-2는 pre-clip·FLU-world라 body-frame 클립
  발동을 판별 불가(‖wh‖ 50–86.6 N 밴드는 모호, >86.6 N만 확정); (4) 상수가 두 파일에
  하드코드 중복.
- **임시 대응**: 없음(정상 미션에선 힘 클립이 거의 안 물림; 관측기 상태는 클립 안 되므로
  추정 자체는 오염 없음).
- **2026-07-23 갱신**: EAOB innovation gating을 **구현했으나 실측 후 기본 OFF로 결정**
  (`params.EAOB_GATE_ON=False`, 메커니즘·`n_gated` 카운터·meta 기록은 유지) — square
  seed-0 A/B(tau_dist=0.5 기준)에서 χ²(0.999,18) 게이트가 파랑/코너의 상시적 w_dot=0 위반
  프레임을 25–76% 기각해 파랑 추종 자체를 차단(CDW radRMS 4.5→15.5 cm, CD 1.35→1.43 cm);  [UNVERIFIED: 산출물 없음 — docs/MEASUREMENT_AUDIT.md]
  기각이 다시 혁신을 키우는 악순환으로 NIS 통계도 오염(137 vs gate-off 25). 최종 기본  [UNVERIFIED: 산출물 없음 — docs/MEASUREMENT_AUDIT.md]
  tau_dist=0.2에선 같은 시나리오에서 게이트가 아예 발동하지 않음(0/1067). 일관성 게이트는  [UNVERIFIED: 산출물 없음 — docs/MEASUREMENT_AUDIT.md]
  wave-band 모델 위반과 구조적으로 양립 불가 → 스파이크 방어는 여전히 per-axis
  `W_HAT_CLIP=[15,45,45,5,5,8]` 단일 정의 + 클립 발동 카운터 기록(mpc_acados.py:170·
  mpc.py의 ±50 하드코딩 중복은 그대로)이 담당해야 함. 근본 해결은 harmonic-EAOB.

### `tests/test_dobmpc.py` — rank-5(NU=4/option-b) 전제, bluerov2 제거로 실행 불가 (deferred)
- **발견**: 2026-07-06, **갱신 2026-07-21** (bluerov2 변종 제거)
- **증상**: `tests/test_dobmpc.py`의 trim/option-(b) 단정(6 N ≈ 23° pitch, `[Fx,0,0,0]` NU=4
  입력)은 제거된 rank-5 `bluerov2` 플랜트 기준. 문서화됐던 `ROV_MODEL=bluerov2` 우회는
  이제 ValueError(변종 없음). 기본 heavy에선 NU=6이라 option-(b) 단정이 어긋남.
- **임시 대응**: 없음 — dobmpc 정리를 미룬 상태라 이 테스트도 함께 보류.
- **제대로 고치려면**: **미룬 dobmpc NU=6-only 정리와 함께** 진행 — params.py의 NU=4/
  option-b 죽은 경로 제거 + test_dobmpc의 rank-5 trim 단정을 heavy(NU=6)용으로 재작성.

### `verify/verify_hydro.py` — bluerov2 강체 + heavy 계수 혼합 fixture, 39 FAIL (2026-08-13)
- **발견**: 2026-08-13, `for_jaden/` 인수인계 패키지의 자체 검증 중.
- **증상**: `python verify/verify_hydro.py --no-plot` → **39 FAIL / 13 PASS**
  (added mass T5, 복원 진자 T4, transfer-function TL 축, T7-R2 부력 등 광범위).
  hydro.py의 결함이 아니라 **fixture 불일치**다:
  - verify_hydro.py:32는 `bluerov.xml`(레거시 rank-5 BlueROV2)을 로드하고,
    verify_hydro.py:35-45의 ground-truth 상수도 BlueROV2 값(MASS 11.2, VOLUME 0.0113459).
  - 그런데 verify_hydro.py:61의 `H.Hydrodynamics(model, disturbance=None)`은 계수 YAML
    기본값 = `RM.YAML_PATH`를 읽고, `ROV_MODEL` 기본값이 heavy라 **BlueROVHeavy.yaml**
    (VOLUME 0.0116499)이 걸린다.
  - 부력으로 정확히 확인됨: 기대 997·9.81·0.0113459 = **110.969 N**(테스트 상수) vs
    측정 997·9.81·0.0116499 = **113.943 N**(실제 로드된 계수). 소수 셋째 자리까지 일치.
  - 2026-07-21에 `bluerov2` 변종이 레지스트리에서 제거돼 `ROV_MODEL=bluerov2` 우회도 불가
    (ValueError) — 위 `tests/test_dobmpc.py` 항목과 같은 계열의 잔재.
- **임시 대응**: 없음. hydro 수치를 이 스크립트로 검증했다고 인용하지 말 것.
  `tests/test_hydro.py`(중성부력/자기복원/항력 한계속도)는 정상 통과하므로 그쪽이 현재
  유일하게 유효한 hydro 스모크다.
- **제대로 고치는 법**: `make_sim()`이 `H.Hydrodynamics(model, disturbance=None,
  coeff_path=<marinegym_assets/BlueROV.yaml>)`로 계수를 **명시**하게 해서 bluerov.xml
  fixture와 짝을 맞춘다(ground-truth 상수는 그대로 유효). 또는 fixture 전체를 heavy로
  옮기고 상단 상수를 BlueROVHeavy.yaml 값으로 재작성한다. `verify_hydro_precise.py`도
  verify_hydro_precise.py:438에서 같은 기본-계수 경로로 sim을 만들므로 같은 불일치를 가질
  가능성이 높다 — **미확인**, 함께 점검할 것.

### dobmpc NU=4 / option-(b) 죽은 경로 (bluerov2 제거 후 도달 불가, deferred)
- **발견**: 2026-07-21 (bluerov2 변종 제거)
- **증상**: 모든 잔존 변종이 rank-6(NU=6)이라 `dobmpc/params.py`의 NU=4 분기,
  `mpc.py`/`mpc_acados.py`의 option-(b) surge→pitch 커플링 경로가 죽은 코드가 됨.
- **임시 대응**: 무해(도달 안 함) — 기능 영향 없음.
- **제대로 고치려면**: 사용자 지시로 **나중에** 일괄 단순화(NU=6 고정). test_dobmpc 재작성과
  함께.
- **2026-07-22 갱신**: **mpc.py는 정리 완료** — tau if/else 제거(`tau = u`), pitch bound의
  PITCH_AWARE 삼항 → 1.2 고정(PITCH_AWARE=False라 동작 동일), 낡은 주석/docstring 정정
  (git HEAD 대비 2000 랜덤 샘플 dynamics Δ=0 + IPOPT NMPC 스모크 검증).
  **잔여**: params.py의 NU=4 분기·PITCH_AWARE/THETA_MAX/SURGE_PITCH_COUPLING,
  mpc_acados.py:109의 option-(b) pitch bound 삼항(+docstring "u=[X,Y,Z,N] (4)"),
  dobmpc_controller.py:269-280의 rank-5 분기, test_dobmpc 재작성.

### `tools/gen_pool_apriltags.py --selftest` — `tag_floor.xml`을 덮어씀
- **발견**: 2026-07-06 (`--tag-mode plane` 개편 후에도 유효 — `run_selftest`가
  tools/gen_pool_apriltags.py:475에서 tag_floor.xml을 테스트 타일 2개짜리로 씀)
- **증상**: selftest 후 POOL_TAGS 씬이 타일 2개짜리 바닥으로 로드됨.
- **임시 대응**: selftest 후 `python tools/gen_pool_apriltags.py` full build 재실행
  (README §7에 경고 있음).
- **제대로 고치려면**: selftest는 별도 임시 파일에 쓰고 종료 시 삭제
  (기존 `_selftest_scene.xml`처럼).

### `tools/plot_wave_spreading.py` — config/base.yaml의 하드코딩 복사본
- **발견**: 2026-07-06
- **증상**: Hs/Tp/gamma/s/h/N_omega/N_beta가 스크립트 상수로 복사돼 있음(yaml을 읽지
  않음) → config를 바꾸면 슬라이드 figure가 실제 실험과 **조용히** 어긋남.
- **제대로 고치려면**: `disturbance.config.load_config`로 yaml을 직접 읽기.

### `water_viz.py` hfield 축이 **행↔열 전치** — 렌더된 파면이 x=y 대각 기준 미러 (2026-08-19)
- **증상**: `water_viz.py:79-84`는 "row = X, col = Y"로 주석·구현돼 있으나, MuJoCo
  (3.9.0)에 4×4 합성 hfield를 만들어 `mj_ray`로 직접 찔러 보면 마지막 **행**을 올렸을 때
  +Y가 뜨고 마지막 **열**을 올렸을 때 +X가 뜬다 — 실제 규약은 row→Y, col→X. 결과적으로
  화면의 파면·해류 진행 방향이 물리 방향과 **x=y 대각 기준으로 미러**된다. 48-샘플/
  0.1771 m 축이 물리적으로는 Y, 96-샘플/0.0885 m 축이 X.
- **영향**: **코스메틱 전용**. hx = hy = 4.25라 지오메트리는 그대로 맞고, hfield는
  `contype=0 conaffinity=0`이라 동역학 경로가 없다(`tests/test_water_viz.py` Δ=0 무관).
  단, 영상에서 "파도가 이쪽에서 온다"를 물리와 대조하면 어긋난다.
- **임시 대응**: 프리뷰 영상으로 파랑 **방향**을 주장하지 말 것(강도·질감만).
- **제대로 고치는 법**: `water_viz.py:79-84`에서 X를 ncol, Y를 nrow에 걸고
  `meshgrid(..., indexing="ij")` 축 순서를 맞춘 뒤, 합성 hfield + `mj_ray` 회귀 테스트 추가.

### `water_viz.py` 프리뷰가 실제 해상보다 **잔잔하게** 보인다 — 120-성분 트림 + eta 클리핑 (2026-08-19)
- **증상 2건**(둘 다 렌더 전용, 어디에도 기록돼 있지 않음):
  (a) `MAX_MODERN_COMPONENTS = 120`(`water_viz.py:40,138-150`)이 M = 1260–1848 성분 중
  진폭 상위 120개만 그린다 → elevation 분산 잔존율 **very_rough 74.7% (eta std 0.375 →
  0.324 m) / moderate 91.1% / gentle 96.8%** [유도: 이번 세션 재계산, 저장 산출물 없음].
  (b) `d = 0.5 + eta/elev`를 [0,1]로 clip(`water_viz.py:105-107`)하는데 shipped
  `elev = 0.60`(`bluerov2_mujoco_marinegym/tag_floor.xml:10`)이므로 **|eta| > 0.30 m에서
  flat-top**. very_rough의 트림 후 eta std가 ~0.32 m라 프리뷰 상당 부분이 잘린다. 그런데
  `tools/gen_pool_apriltags.py:231,569-570`은 이 값을 "half-range / max|eta| headroom"이라
  부른다 (실제 headroom은 elev/2).
- **임시 대응**: 강한 sea state 프리뷰는 `--water-hf-elev`를 2×(예: 1.2 이상)로 재생성.
- **제대로 고치는 법**: (a) 트림 잔존 분산을 `update()`에서 한 번 로깅, (b) `elev`를
  실제 반범위로 쓰도록 `d = 0.5 + eta/(2*elev)`로 고치거나 인자명을 바꿀 것.

### `water_viz.py`가 modern env의 **레이어 게이팅을 무시** — C/CD 모드에서 물결이 보인다 (2026-08-19)
- **증상**: `_eta_modern`/`_current_vec`은 `field.waves`를 직접 읽고
  `field.use_waves`를 **확인하지 않는다**(`water_viz.py:113-136,183-198`). 호출부는 마스터
  `enabled`만 넘긴다. 그래서 파도가 꺼진 모드(C, CD)에서도 수면이 출렁인다.
  `experiments/wave_preview.py:200`은 CLI에 `--mode C/CD`를 노출하고 `:151`은
  `enabled=True`를 하드코딩한다. (legacy `disturbances.py`는 `enabled`를 제대로 게이트하므로
  teleop 경로에는 없는 문제.)
- **영향**: 시각 전용이지만 **오독을 부른다**(C 모드 영상을 파랑 영상으로 착각).
- **제대로 고치는 법**: `_eta`에서 `getattr(field, "use_waves", True)`를 확인하고,
  advection도 `use_current`로 게이트.

### `tools/analyze_square3.py` / `tools/analyze_acados_vs_before.py` — 경로 하드코딩
- **발견**: 2026-07-06
- **증상**: `recordings/20260615/`의 특정 CSV 파일명이 하드코딩(`DIR`/`RUNS`/`PAIRS`
  상수) → 해당 recording이 없으면 crash, CLI 플래그 없음.
- **임시 대응**: 다른 run에 쓰려면 상단 상수 수정 (README §5에 명시).
- **제대로 고치려면**: `--dir` 인자화. 우선순위 낮음(일회성 분석 스크립트).

### `experiments/plot_trajectories.py` — docstring/코드 불일치
- **발견**: 2026-07-06
- **증상**: docstring은 error-vs-time 패널을 언급하지만 현재 코드에 없음.
  마지막-2랩 RMS(`rms_lastlaps`)는 내부에서 계산만 하고 어디에도 표시하지 않음.
  (2026-07-28에 *표시되는* RMS는 run_compare와 동일한 steady window로 통일됨 —
  이 항목은 남은 docstring 정리 + 미표시 `rms_lastlaps` 건에 한정.)
- **제대로 고치려면**: docstring 정리, 또는 패널/범례에 마지막-2랩 RMS 추가.

### `verify/verify_acados.py` 게이트가 heavy_gripper에서 근소 초과 (0.2717 > 0.25 N)
- **발견**: 2026-07-12 (heavy_gripper 변종 검증 중)
- **증상**: acados RTI vs IPOPT worst-case |Δu| 게이트 0.25 N은 heavy 기준 캘리브레이션;
  heavy_gripper(13.7 kg)에서 0.2717 N — 30 N sway authority의 ~0.9%라 실효 동일 최적해,  [UNVERIFIED: 산출물 없음 — docs/MEASUREMENT_AUDIT.md]
  폐루프도 검증됨(DP hold 1.3 cm). heavy는 여전히 PASS.
- **제대로 고치려면**: 게이트를 변종별 스케일 또는 ‖u‖ 상대비로.

### 파랑 모드에서 acados 솔버 실패(n_fail)가 드물게 자세/깊이 blowup 유발
- **발견**: 2026-07-21 (`compare_20260720_221845` 분석 워크플로; n=200
  `compare_20260720_230025`로 규모 확정)
- **증상**: n=200 census — 3000 run 중 217 run에서 총 289회 실패, 전부 MPC 계열이고
  89–99%가 CW/CDW(run 실패율 **mpc 14.0% vs dobmpc 7.7%** — EAOB FF가 OCP를 오히려
  안정화). 실패 run에서 depth/pitch 결합 극단 excursion — mpc 최대 181.8 cm(pitch 최대
  79.5°), dobmpc 최대 134.7 cm(|pz| 97 cm). **dobmpc radial_max>40 cm는 23/23이 fail run**
  (클린 run 상한 37.8 cm) → worst-case 통계를 이 클래스가 지배. 트리거는 seed-0 공통
  파랑그룹의 lap-7/8 V3 턴 이벤트이고 실패는 증폭자(원인 아님, run-level 연관만 확인 가능).
- **임시 대응**: 분석 시 `n_fail>0` run의 radial_max는 별도 취급(RMS 집계는 강건:
  제외해도 평균 −3~−8%만 이동).
- **제대로 고치려면**: (1) 실패 **시각** 로깅(현재 run당 카운트만 있어 tick-level 인과
  확정 불가), (2) 실패 시 fallback 전략 점검(mpc_acados 실패 경로), (3) traj CSV에
  w_hat·solver-status 기록 추가.
- **2026-07-21 갱신**: reference preview 도입 후 실패율 급감(공유 50 heading 기준 dobmpc
  10–13→1–2런, mpc 16→7/12→4) 및 dobmpc >40 cm 꼬리 소멸 — 그러나 이슈 자체는 잔존
  (mpc CDW에 신규 210 cm blowup; 위 세 수정은 여전히 유효).

### hydro.py는 body_iquat=identity(대각 관성)를 암묵 전제
- **발견**: 2026-07-12 (heavy_gripper NMPC 발산 근본원인 추적으로 발견)
- **증상**: `mj_objectVelocity(mjOBJ_BODY, local=1)`은 **inertial(주축) 프레임** 기준인데
  hydro는 body 프레임으로 간주해 drag를 `xmat`으로 적용. `fullinertia`로 주축이
  정렬·순열되면(Iyy>Izz>Ixx 등) drag 축이 뒤엉켜 **에너지 주입 → 폭발**(torque-free
  kick 0.5 rad/s → 1.5 s 만에 |q|>60 rad/s로 재현).  [UNVERIFIED: 산출물 없음 — docs/MEASUREMENT_AUDIT.md]
- **임시 대응**: heavy_gripper 생성 XML이 diaginertia 강제 + `tests/test_heavy_gripper.py`가
  `body_iquat==identity` 회귀 가드. 기존 변종은 원래 대각이라 무증상.
- **제대로 고치려면**: hydro가 `mjOBJ_XBODY`(body 프레임)로 측정하거나 ximat로 변환 —
  물리 파일 수정이라 별도 검증(기존 변종 byte-identical 확인) 필요.
- **2026-07-19 갱신**: C3를 실측 위치(전방-하단)로 옮기면서 버려지는 Ixz가
  −0.0016(0.4%) → **heavy_gripper +0.064 kg·m² (Ixx의 16.8%) / heavy_c3 +0.046 (12.4%)**
  로 커짐. 실기체에 존재할 roll-yaw 곱관성이 플랜트에 없다는 뜻 — hydro를 body-frame으로
  고치기 전까지는 구조적으로 못 넣는다. 위 "제대로 고치려면"의 우선순위가 올라감.

### 2026-07-23 base.yaml 파랑 강화 이후 CDW에서 EAOB perf 일관성 깨짐 (소스 무관)
- **발견**: 2026-07-23 (verify_state_source A/B 검증 중; 파랑 블록이 19:28에
  Hs 0.75→1.2 m, Tp 12→6 s, γ 5→2, s 30→10, ω_max 1.6→3.0으로 강화됨)
- **증상**: 새 해상 상태의 CDW에서 mpc_state_source와 무관하게 NIS 80(truth)/71(estimate)
  (DP), 44/42(square) — τ_dist=0.2로 검증했던 목표범위(14–24) 크게 이탈, radRMS도  [UNVERIFIED: 산출물 없음 — docs/MEASUREMENT_AUDIT.md]
  DP 1.5→9.6 cm / square 3.5→15.7 cm로 악화. w_dot=0 + τ_dist=0.2가 새 파랑 대역
  (ω_p≈1.05, 에너지 ~3 rad/s)을 못 쫓아가는 것; 구파랑 config로는 전 항목 PASS 재현.
- **임시 대응**: 없음(발산은 아님 — n_fail 0, 유한). `verify/verify_eaob.py`·
  `verify_state_source.py`가 현 config에서 exit 1로 신호.
- **2026-07-24(3) 갱신**: surge 박스 8→30 N 변경 A/B(같은 명령 `verify_eaob --no-plot`,
  CDW/T=60/seed 0)에서 **NIS 245.8→100.0, NEES 398.9→122.0으로 2.5–3.3× 개선**(여전히  [UNVERIFIED: 산출물 없음 — docs/MEASUREMENT_AUDIT.md]
  게이트 24 초과 = FAIL). 즉 이 일관성 붕괴의 일부는 **authority 부족으로 인한 큰 추종오차**
  였고(박스가 EAOB의 pseudo-measurement 채널까지 오염), 나머지는 원래 진단대로 τ_dist가
  새 파랑 대역을 못 쫓는 것. 재스윕은 30 N 박스 기준으로 할 것(옛 8 N 수치로 튜닝 금지).
- **제대로 고치려면**: 새 해상 상태 기준으로 EAOB_TAU_DIST 재스윕(0.05–0.1 예상) 또는
  harmonic-EAOB — 어느 쪽이든 실험 설계 결정이라 사용자 판단 필요.

### STATION BRIDGE(태그 dropout 사다리)가 **실기 미검증** (2026-08-18)
- **무엇**: station 모드에서 태그를 놓쳐도 disengage하지 않는다. 0~`imu_hold_s`(3 s)는
  전 축을 bridge 추정치로, 그 뒤로는 x/y/yaw를 해제하고 깊이+자세만 **무제한** 유지.
  `rov_gui/control/station_bridge.py`, 설정 `config/hw_mpc.yaml: station_bridge:`.
- **검증된 것**: 사다리 로직·화이트리스트·coast의 전선 위 축 값(allocation/cap/slew
  통과)·다른 인터록 생존·station 한정·meta/CSV 경계까지 `test_station_bridge.py` 15/15,
  기존 스위트 무회귀(test_control 79/79, test_offline 79/79, test_imu_dr 25/25).
- **검증 안 된 것**: **물에서 한 번도 안 돌렸다.** 특히 (1) 실제 dropout이 몇 초인지
  아직 모른다 — 2026-08-18 8개 런에서는 최악 0.55 s로 기존 1.1 s 예산 안이었고,
  즉 **문제가 재현된 로그를 아직 못 봤다**; (2) `xy_source: auto`가 고르는
  가속도 적분 경로는 C3 BNO086 캘리브(`config/c3_imu_calib.json`)의 품질에 전적으로
  달렸는데 그 캘리브 자체가 미검증([[imu-dead-reckoning-experiment]], 40° 틸트 미기입);
  (3) coast에서 실제로 얼마나 흘러가는지 모른다.
- **임시 대응**: 첫 수조 세션에서는 **조종사가 E-STOP에 손을 두고**, 패널이 빨갛게
  `NO TAG — COASTING`으로 바뀌면 그때부터는 사람이 판단한다. 지오펜스가 없어서
  coast 중 벽으로 흘러가도 막는 것이 없다. 되돌리려면 `station_bridge.enabled: false`
  한 줄이면 예전 거동으로 정확히 돌아간다(테스트로 고정).
- **제대로 고치는 법**: (a) 문제가 난 런의 CSV에서 `tag_age_s`/`n_tags`로 dropout
  길이 분포를 먼저 재고 `imu_hold_s`를 거기에 맞춘다. (b) 복구 로그가 남기는
  "IMU was N cm off"를 몇 번 모으면 `xy_source: imu`를 신뢰할지 말지가 **측정으로**
  결정된다 — 지금 3 s/9 cm는 [유도]다. (c) coast가 길어질 때 자동으로 무엇을 할지는
  아직 결정 안 했다(현재는 영원히 유지 + 사람).

### 물체 추종(`follow` + `object_nav`)이 **실기 미검증**이고 `--pose` 위험을 통째로 상속한다 (2026-08-21)
- **무엇**: `--pose --mpc`를 함께 켜면 클릭한 물체가 태그맵 좌표로 올라오고(패널 다이아몬드,
  `bus.object_fix`), 미션 모양 `follow`가 START 순간의 상대 자세(위치 3D + yaw)를 유지한다.
  `rov_gui/control/object_nav.py`, 설정 `config/hw_mpc.yaml: object_nav:`,
  문서 `rov_gui/README.md`의 "물체를 따라간다 — follow".
- **검증된 것**: extrinsic 소거(틀린 extrinsic을 넣어도 물체 위치가 1e-9로 불변) ·
  프레임 어긋남의 lever-arm 대가 · 오프셋 왕복과 궤도 성질 · yaw 축 고정 · 점프 거부/reseed ·
  t_capture 기반 속도 · 외삽 클램프 · 짝맞춤 우선순위 · 다섯 가지 arm 거부 ·
  arm이 기체를 안 움직임 · leash가 1 s tick에서도 유지 · 이탈 클램프 · freeze→station 강등 ·
  태그 dropout 강등 후 bridge 인계 · 네 곳 lifecycle · CSV 10열/schema 6 —
  `test_object_nav.py` 28/28, 기존 스위트 무회귀(test_control 81/81, test_offline 79/79,
  test_station_bridge 15/15, test_imu_dr 27/27), `demo_e2e.py pid follow {still,drift,orbit}`
  전부 통과(pair_exact 100 %).
- **검증 안 된 것**: **물에서도 공기 중에서도 한 번도 안 돌렸다.** 그리고 이 기능은
  아래 "`rov_gui --pose`: 물체 추적/자세추정이 실기에서 한 번도 안 돌았다 (2026-08-09)"
  항목의 위험을 **통째로 상속한다** — `--pose`가 책상 위 물체에서 `TRACKING`을 못 주면
  이 기능은 존재하지 않는 것과 같다. demo가 증명하는 것은 배관과 대수뿐이다: demo의
  `T_cam_obj`는 demo 기체 상태에서 만들어지므로 합성이 **구조적으로** 왕복하고,
  SAM2 마스크 품질·FoundationPose 지연·드롭아웃 통계·depth 노이즈는 하나도 안 나온다.
  `object_nav:`의 임계값은 **전부 [예측]**이다.
- **가장 싼 go/no-go (코드가 아니라 배치 문제일 수 있다)**: `cam_tilt_deg: 43.3`이면
  C3는 매트를 **내려다본다**. 물체가 태그 ≥2장과 **한 프레임**에, 0.3~0.8 m 거리,
  HFOV 63.9°(0.5 m에서 시야 폭 62 cm) 안에 들어와야 합성이 성립한다.
  **태그와 물체가 한 프레임에 공존 못 하면 이 마운트에서는 못 나는 기능이고, 답은
  코드가 아니라 두 번째 카메라나 재틸트다.** 테이프로 먼저 재 볼 것.
- **임시 대응 (벤치 → 풀장 순서, 각 단계가 다음을 벌어준다)**:
  벤치(공기 중, COMMAND ENABLE off, 미장착): ① 위 배치 확인 → ② 클릭하고 다이아몬드를
  본다. 물체를 손으로 20 cm 밀면 플롯에서도 움직여야 한다. **공기 중 거리는 ~1.33배
  길게 읽힌다**(C3 공장 캘리브가 **수중** 값 — `calib/FOV_AUDIT.md`, 아래 2026-08-04 항목).
  **공기 중에서 이걸 "고치면" 안 된다** — 손 병진 검사는 **비율로서** 유효하고, 기록할
  것은 그 비율이다(실제 20 cm에 그림 26.6 cm면 체인 전체가 맞고 스케일만 알려진 오프셋).
  → ③ 패널의 `pair` 슬롯이 사실상 0 ms인지. 아니면 두 메일박스의 독립 conflation이
  공유 stamp를 무력화한 것이고, extrinsic 소거가 멈춰 0.2855 m lever arm이 오차 예산에
  들어온다 — **풀장 전에** 알아야 한다. → ④ 거부 4종(모드 mpcc / lock 없음 /
  `nav_source: second` / `--pose` 없음) 발동 확인.
  풀장: ① **HOLD만, 물체 시야 안, 2분** — 산포·`pair_exact` 비율·`obj_age_s` p50/p95·
  `PoseTrack.state`가 `tracking`을 벗어나는 빈도. `object_nav:`의 모든 [예측]이 [측정]이
  되는 런이고, follow를 arm할 가치가 있는지 여기서 결정된다. → ② **정지 물체 follow**:
  기체가 **움직이면 안 된다**(오프셋을 현재 pose에서 떴으니까). 1초 안에 보인다. →
  ③ 손으로 천천히 옮기는 물체 1 m 직선 ~0.05 m/s, 칩의 `speed`/`ref_speed` 비교
  (FF가 안 먹으면 2배 이상 지연으로 보인다). **E-STOP에 손 — 벽을 아는 건 새
  `max_excursion_m`뿐이고 그건 미검증이다.** → ④ 제자리 회전(궤도 케이스, 잘못된
  `yaw_axis`가 가장 잘 드러난다) → 그 다음에야 빠른 이동과 긴 이탈.
  첫 벤치·풀장은 `object_nav.yaw_axis: none`으로 — 물체 yaw를 안 쓰면 실패 클래스
  하나가 통째로 사라진다. 되돌리려면 shape를 `follow`가 아닌 것으로 두면 되고, 표시
  절반만 쓰려면 arm하지 않으면 된다(기체에 대한 새 권한이 전혀 없다).
- **2026-08-23 첫 실기 시도 — 위의 go/no-go에서 걸렸다(코드 아님, 거리)**: 태그와 물체가
  한 프레임에 들어오기는 했으나 **물체가 1.2~1.56 m**에 있어 `object_nav.max_distance_m:
  1.20`이 관측을 거의 전부 거부했다. 근거: `pose: collecting reference views — object is
  1556 mm away`와 메시 캡처 시 `distance 1195-1256 mm`
  (`data/20260823/0823_162548/mission_log.txt`),
  16:34~16:36 세 런의 `obj_state`가 **전 행 `cold`**(한 번도 lock 안 됨), 16:38 런은
  단 한 번 lock한 뒤 **전 행 `lost`**로 `obj_age_s`가 9.6 s→202 s까지 자람
  (`0823_163414/mpc_163512.csv`, `0823_163707/mpc_163809.csv`). 그래서 플롯에
  다이아몬드가 거의 안 뜨고 `follow`는 arm 자체가 안 된다("no object lock").
  **1.2~1.5 m는 [미검증] 구간이다** — 0.3~0.8 m는 [측정: KNOWN_ISSUES 2026-08-09] 동작,
  2.4 m는 [측정] 실패, 그 사이는 아무도 재 본 적이 없다. 물체를 0.3~0.8 m로 가져오는 것이
  **먼저**이고, 그게 배치상 불가능할 때만 `max_distance_m`를 올리되 그 런은 새 [예측]
  구간에서 난 것으로 표시할 것.
- **제대로 고치는 법**: 풀장 ①~④를 통과하면 이 항목을 지우고 `object_nav:`의
  `[예측]`을 측정치로 바꾼다. ②가 안 되면 원인은 대개 배치이거나 `--pose` 자체다.

### 물체 자세의 **거리가 1.42~1.50배 길다** — 원인 미확정(메시 스케일 vs depth 캘리브) (2026-08-23)
- **증상**: 매트 위에 놓인 물체가 태그면 **아래 0.42~0.50 m**에 찍히고, 선체→물체 광선이
  태그면까지 거리 대비 **1.42~1.50배**(p10~p90 1.37~1.60). 방향은 맞고 길이만 늘어난다.
  [측정: `data/20260823/0823_174602/mpc_174638.csv`,
  `mpc_174657.csv`, `obj_state=live` 행; 태그면 위치는 각 런 meta의
  `hardware.datum_tag_frame.p0`]
- **굴절은 아니다(부호가 반대)**: 평면 포트에서 물은 상을 1.33배 크게 만들어 거리를
  **짧게** 읽히게 한다. 공중 캘리브를 물속에서 쓰면 1.33배 짧고, 수중 캘리브를 공중에서
  쓰면 1.33배 길다([[c3-calibration-is-underwater]], `calib/FOV_AUDIT.md`). 물속에서
  길게 읽히는 조합은 없다. 태그 PnP도 독립적으로 반증한다: engage 시 태그 z −1.002 m +
  압력 depth 0.41 m = 매트 위 수주 **1.41 m**로 실기록 ~1.4 m와 일치한다
  (1.45배라면 1.86 m여야 한다).
- **남은 두 후보는 이 데이터로 구별이 안 된다** — 둘 다 "메시가 크고 거리가 멀다"를 만든다:
  1. **재구성 메시 스케일**. 그날 메시는 8 views / arc 101°(로그가 `only 8 views (10+
     recommended); the mesh may be poor`라고 경고) → `88 x 175 x 239 mm`. FoundationPose는
     렌더한 메시가 관측 크기와 맞을 때까지 거리를 밀어내므로 **거리 ∝ 메시 크기**다.
     참고로 16:32 세션의 같은 계열 물체 메시는 `101 x 114 x 171 mm`(239/171 = 1.40).
  2. **depth map이 metric이 아니다**. stereo depth는 mono 쌍 + baseline이라 RGB fx와
     **별개 캘리브 경로**이고, **태그는 depth를 전혀 안 쓴다**. 여기가 1.4배면 BundleSDF
     재구성도 FoundationPose 추적도 같은 스케일 오차를 물려받고 태그만 멀쩡하다.
- **판별 수단은 붙여 놨다**: 궤적 readout의 **`depth-vs-TAG n.nnx (N tags)`** 줄
  (`window._check_depth_scale`). 시야의 매핑된 태그마다 depth map 값과 **fix에서 나온
  기하학적 카메라→태그 거리**를 비교한 중앙값이다. `1.00x`면 depth는 metric이고 범인은
  메시, `~1.3x`면 depth 경로다. `test_the_depth_map_is_cross_checked_against_the_tag_pnp`.
- **2026-08-23 18:26 런이 "일정한 스케일 오차"라는 가설을 반증했다.** 12178개 live 행
  [측정: `0823_182628/c3_depth_20260823_182645_mpc.csv`]:

  | | 중앙값 | p10 | p90 |
  |---|---|---|---|
  | 물체 z (MAP NED, 0 = 매트) | **+0.53 m** | −0.84 | +0.77 |
  | 광선의 매트 통과 배율 | **1.52x** | 0.51 | 1.75 |
  | 보고된 선체→물체 거리 | 2.41 m | | |

  배율이 0.5~1.75로 **3배 넘게 흔들린다** — 어떤 캘리브 상수도 이걸 못 만든다. 물체 z도
  매트 위 84 cm와 아래 77 cm를 오간다. **즉 자세추정 자체가 유효한 해를 못 잡고 있다**:
  같은 런의 `reg` 카운트가 67→86까지 올라가고 로그가 `pose does not fit — wrong object`와
  `lost the object`로 도배됐다. 8 views로 만든 수중 광택 물체 메시가 원인 후보 1번이다.
- **2026-08-23 21:1x 최종 확정: 스테레오 depth가 물속에서 1.56~1.71배 길다 — 조종사의
  캘리브레이션 가설이 맞았다.** 결정 증거: 깨끗한 캡처(arc 79°, 0.88~0.91 m, 게이트 안)의
  메시 `143×166×187 mm` vs **캘리퍼 실측 119.73 mm** = **1.562배**
  [측정: `0823_210304/mission_log.txt` 21:02:29 + 조종사 캘리퍼]. 가까운 캡처 메시들도
  전부 1.43~1.63배(171/195 mm)로 **일정**하고, 바닥 스윕 `depth-vs-MAP`는 **1.71x**
  (더 먼 거리 — 오차가 거리 의존일 수 있음). 컬러 PnP는 같은 물에서 mm급이므로
  **컬러 캘리브는 정상, 스테레오 depth 경로만 non-metric**이다.
  아래 "물체가 아닌 것을 삼킨다" 분석은 **부분 원인으로 강등**: 거리별 산포 증가(smear)는
  실측 사실이나, 지배 항은 이 스케일이다. 470 mm 메시 = 스케일 × smear.
- **대응(2026-08-23 추가)**: `--depth-scale K` — depth 밀리미터를 **소스에서 한 번** 곱해
  모든 소비자(커서 프로브·캡처·FoundationPose·depth-vs-MAP)가 같은 보정 스트림을 본다.
  시작값 **0.64**(=1/1.562). 적용하면 depth-vs-MAP가 ~1.0x로 내려와야 하고(그 줄이 보정의
  검증기가 된다), **메시는 반드시 다시 캡처**(옛 메시는 1.56배라 보정 depth와 안 맞는다).
  pose CAMERA 레코드에 `depth_scale_applied`로 기록된다. **진짜 수리는 수중 스테레오
  재캘리브레이션**(c3_camera 쪽 작업, 미착수).
- **경계**: 이 날짜 이전 depth 유래 거리는 전부 1.4~1.7배 길다 — 물체 위치·hold_m·메시
  치수·참조뷰 distance 전부 인용 금지.
- **(이하 과거 분석 기록)** 2026-08-23 20:00 당시 원인 후보였던 것: 재구성이 물체가 아닌
  것을 삼킨다(스케일 오류가 아니다).**
  같은 물체 여덟 번 재구성 [측정: `data/*/*_obj/model/model.obj`, oriented bbox]:

  | verts | bbox (mm) |
  |---|---|
  | 18,548 | 101 × 114 × **171** |
  | 30,147 | 87 × 112 × **195** |
  | 59,542 | 88 × 175 × **239** |
  | 101,428 | 108 × 172 × **470** |

  **짧은 두 변은 안정적**(87~108 × 104~175 mm)이고 **긴 변만** 자란다. 재구성은
  **metric RGB-D 기반이라 크기를 안다** — 틀린 건 스케일이 아니다.

  **범인은 마스크가 아니라 마스크 안의 depth이고, 그 결정 변수는 거리다.**
  저장된 모든 참조 캡처에서 마스크 안 depth의 p5~p95를 재봤다
  [측정: `data/*/*_obj/{depth_enhanced,mask}`]:

  | 캡처 거리 | 마스크 안 depth 산포 | 결과 메시 길이 |
  |---|---|---|
  | 510 mm | **102 mm** | 182 mm |
  | 626 mm | 142 mm | 239 mm |
  | 1210 mm | 284 mm | 171 mm |
  | 1369 mm | 265 mm | **470 mm** |
  | 1358 mm | **714 mm** | (버려짐) |

  물체는 약 110 mm다. **0.5 m에선 산포 ≈ 물체 크기**(정상)인데 **1.4 m에선 2~6배**라
  융합되는 포인트 클라우드가 뭉개진 덩어리이고, 메시는 그 뭉개짐이 향한 방향으로 길어진다.
  문제의 470 mm 메시는 **1083~1479 mm에서 캡처**됐다(마스크는 정상, 18장/79°).
  스테이션이 캡처 중 이미 "30-80 cm"라고 안내하지만, 업스트림 뷰 게이트는 1.5 m까지
  받아준다 — 그 사이가 이 사고 구간이다.
- **그래서 균일 리스케일도, 마스크 튜닝도 틀린 처방이다.** 2026-08-23에 리스케일을 한 번
  잘못 만들었다가 되돌렸다(맞는 두 변을 줄이게 된다). 마스크는 실제로 멀쩡했다.
- **대응(2026-08-23 추가)**: 캡처가 끝나는 순간 `PoseSession.depth_quality`가
  **마스크 안 depth 산포 vs 마스크 면적이 함의하는 물체 크기**를 찍는다. 1.5배를 넘으면
  경고: "마스크는 정상이고 그 안의 depth가 smear다 — 30-80 cm에서 다시 잡아라."
- **헤딩 의존성**: 태그 56 위 물체가 기체를 +y로 두면 57(+0.22 m y), +x로 두면 55 부근
  (+0.29 m x)에 찍힌다 — 오차가 **기체가 보는 방향**을 따라간다. extrinsic 버그가 아니다
  (`TagNav._solution`이 `t_cb`로 나누고 `compose_map_pose`가 다시 곱해 정확히 소거).
  카메라 프레임의 상수 오차(=광축 방향 거리 오차)가 `R_map_cam`에 실려 회전하는 것이고,
  "물체가 매트 아래로 들어간다"와 같은 하나의 결함이다.
- **대응(2026-08-23 추가)**: `--pose-object-size MM`. 물체 최장변을 재서 주면 **빌드 직후**
  메시 치수와 비교해 경고한다(리스케일 아님). 25% 넘게 길면 "이 메시로 날지 말 것".
  아래 downstream 증상은 전부 이것의 결과이므로, 이 경고가 뜨는 메시로 난 런은 인용 금지.
- **앞선 `depth-vs-TAG 1.44x`는 인용하지 말 것 — 그 체크에 샘플링 편향이 있었다.**
  태그 중심 5x5를 읽었는데 태그 중심은 검은 사각형, 즉 이 장면에서 스테레오가 **유일하게
  못 맞추는 패치**다. 그래서 태그가 작아질수록(=기체가 높을수록) 구멍과 경계 번짐이 늘어
  숫자가 태그 개수를 따라 움직였다: 3장 0.94x / 8장 1.17x / 14장 1.46x
  [측정: 조종사 스크린샷 4장, 18:30~18:35]. **2026-08-23 매트 전체를 훑는 방식으로 교체**
  (`window._check_depth_scale`: 픽셀 격자 → 태그면 z=0과의 교점 → 기대 Z 대비 depth Z,
  수백 샘플의 중앙값 + p10~p90 spread). 새 줄은 `depth-vs-MAP n.nnx (+/-s.ss)`이고
  **spread가 0.15를 넘으면 그 숫자는 상수가 아니다**(= 스케일 문제가 아니라 형상 문제거나
  매트 말고 다른 게 시야에 있다는 뜻). **아직 새 방식으로 읽은 런이 없다.**
- **즉시 할 수 있는 확인(코드 없이)**: C3 DEPTH 패널의 커서 프로브를 **물체 위에** 올려
  mm를 읽고, 같은 순간 파이프라인이 말하는 카메라→물체 거리(SENSORS `Object` 행)와
  비교한다. 프로브가 짧으면 메시, 같이 길면 depth다.

### `HwMpcc.set_target_ned`가 속도 피드포워드를 **버린다** — `follow`가 mpcc에서 거부되는 이유 (2026-08-21)
- **증상**: `mpcc_bridge.py`의 `set_target_ned`가 `del v_ned, r_ned`로 시작한다.
  호출마다 `ArcPath`+`speed_profile`+`window_for`를 새로 만들고 `scenario = None`으로
  지운다. 20 Hz로 움직이는 setpoint를 주면 **FF 없이 매 tick 경로를 재구축**한다.
- **왜 중요한가**: FF가 없으면 leash가 사실상의 제어법이 된다 — 실측
  `kp·lead = 51.7 × 0.35 = 18.1 N` vs `F = 86.7·v + 5.76` → `v ≈ 0.084 m/s`
  (2026-08-18, memory: approach-speed-leash-limited). DP hold에서는 무해하지만
  움직이는 목표를 쫓는 어떤 모드에도 못 쓴다.
- **임시 대응**: `HwMpcc.follow_ok = False` — `follow` 미션이 mpcc/dobmpcc에서
  **거부**된다(`getattr(..., False)`로 fail-closed). dobmpc/mpc/pid로 날 것.
- **제대로 고치는 법**: 움직이는 경로에 대한 contouring. theta가 솔버 상태인
  구조에서 경로 자체가 매 tick 바뀌면 theta의 의미가 유지되지 않으므로, 플래그가
  아니라 설계 변경이다. 그때까지는 거부가 옳다.

### `HwDobMpc.set_target_ned`의 `r_ned`가 기본 컨트롤러에서 **사실상 no-op** (2026-08-21)
- **증상**: `set_target_ned(p, yaw, v_ned, r_ned)`가 `r_ref`를 넘기지만
  `set_target(..., yaw_target=None)`이라 `yaw_target = yaw_ref`가 되고,
  `_xref_ned`에서 `delta = psi_t - psi0 = 0` → `xref[11,:]`이 **0으로 강제**된다.
  즉 yaw-rate 피드포워드는 `HwPid`에서만 살아 있다.
- **왜 중요한가**: 두 컨트롤러가 같은 인자를 받고 **다르게 무시**한다. 이 상태로
  A/B를 하면 비교가 컨트롤러 비교가 아니게 된다.
- **임시 대응**: `follow`는 `r_ned`를 **안 넘긴다**(`_issue_follow_target`). 헤딩
  레이트 제한은 워커 쪽에서 두 컨트롤러에 동일하게 건다.
- **제대로 고치는 법**: `set_target_ned`가 `r_ned != 0`일 때 `yaw_target`을 함께
  계산해 넘기거나(예: `yaw + r*preview`), 아니면 인자를 지우고 경로 계획만 쓴다.
  지금처럼 받아서 조용히 버리는 것이 최악이다.

### `mpc_tuned` / `dobmpc_tuned` (along/cross 비용 분리)가 **실기 미검증** (2026-08-18)
- **무엇**: tracking NMPC의 2×2 위치 가중치를 매 stage 경로 프레임으로 회전시켜
  종방향(along)·횡방향(cross)을 따로 벌하는 모드. `rov_gui/control/path_cost.py`,
  튜닝은 `config/hw_mpc.yaml: mpc_tuned:`.
- **검증된 것**: 대수(회전 = 오차분리, 등방이면 baseline과 기계정밀도 동일)와
  오프라인 폐루프 parity·모드전환 오염 없음까지 `rov_gui/tests/test_path_cost.py`
  15/15. 코너 컷 감소는 **오프라인 예측모델 플랜트에서만** 확인
  (fillet 0.15·0.10 m/s에서 4.8 → 1.2 mm, `rov_gui/tools/sweep_path_cost.py`).
- **검증 안 된 것**: 물에서 한 번도 안 날렸다. 그 플랜트는 ESC 데드밴드도 테더도
  없고 수평 항력이 8~12배 부족하다 — 2026-08-18 진단에서 확인된 실기의 지배적
  제약(ax_surge 0.158에서 실제 속도 0.000 m/s)을 하나도 담고 있지 않다. 즉
  **"코너 컷 −74 %"는 기하 예측이지 실기 성능 주장이 아니다.**
- **임시 대응**: 배포 기본값 `along_scale 0.25 / cross_scale 4.0`은 **[예측]
  첫 추정치**다. 기본 모드는 여전히 `dobmpc`이고 tuned는 opt-in.
  fillet 0인 waypoint 미션에서는 효과가 −15 %로 떨어지고 along_scale을 내리면
  오히려 나빠지므로(+57 %), **fillet 0.15와 같이 쓸 것**.
- **제대로 고치는 법**: P5/P6 뒤에 baseline `mpc` ↔ `mpc_tuned` 한 쌍을 같은
  세션에서 날리고 코너 구간 cross-track을 비교한다. run meta의
  `controller.cost_frame` / `controller.path_cost`가 두 기록의 경계다 —
  tuned 런의 `Q` 행은 split을 유도한 등방 baseline이지 솔버가 쓴 값이 아니다.

### `rov_gui --pose`: 물체 추적/자세추정이 **실기에서 한 번도 안 돌았다** (2026-08-09)
SAM2 추적(1단계), 메시 기반 6-DoF(2단계), 현장 재구성(3단계)을 전부 구현했고,
**저장된 데이터와 데모로만** 검증했다. C3는 네트워크에 없었고 ROV도 분리돼 있었다.

- **된 것**: SAM2 로딩 1.1 s / 36 Hz(저장 프레임), FoundationPose 등록 성공
  (`ref_views` 재생, 거리 0.431 m), 현장 재구성 전 구간이 `rovgui-pose` 인터프리터에서
  동작(저장 참조뷰 재생 → 수집 8장/75.3°/RMSE 6.2 mm → BundleSDF 25 s → 7993 verts
  51×94×167 mm), 오버레이·클릭 좌표변환·로깅, 테스트 61개.
- **안 해본 것**: 실제 C3 라이브 프레임, 수중 장면에서의 SAM2 마스크 품질, 수중 스테레오
  depth로 등록이 되는지, 30 fps 탭이 C3 워커 지연에 주는 영향, 그리고 **라이브 수집**
  (위 25 s는 저장 뷰 재생이고, 사람이 실제로 물체를 도는 시간은 포함되지 않는다).
- **확인 순서** (뒤로 갈수록 비싸다):
  1. `./c3 gui --source hw --pose --pose-no-build` — 공기 중, 책상 위 물체. TRACK 켜고
     클릭 → `TRACKING` + 마스크가 붙나. 여기서 안 되면 나머지는 볼 필요 없다.
  2. 같은 조건 + `--pose-mesh .../ref_views/model/model.obj`, 그 메시의 **실물**을
     0.5 m에 두고 클릭 → 0.7 s 뒤 축이 뜨고 `d ≈ 0.67 m`인가 (수중 캘리브라 지상에선
     ~1.33배 길게 읽히는 게 **정상**이다 — 0.5 m로 나오면 오히려 조사할 것).
  3. `--pose-no-build` 없이 (= 현장 재구성). 클릭 → 물체를 0.3~0.8 m에서 천천히 돌려
     `COLLECTING VIEWS`가 올라가나 → `RECONSTRUCTING` → 자세. 검산 줄(`bad_mask#`,
     메시 bbox)이 로그에 뜨고 치수가 실물과 맞나.
  4. C3 Depth 녹화를 켜고 `_pose.jsonl`의 `hz_measured`를 본다 — 거기 나오는 수가
     실제로 잡힌 레이트다.
  5. 그 다음에야 물속.
- **미리 알 것**: 작동거리 **0.3~0.8 m**(2.4 m면 등록 실패), HFOV **63.9°**(0.5 m에서
  시야 폭 62 cm), 그리고 내부파라미터가 **수중 캘리브**(위 2026-08-04 항목)라서 지상
  책상 테스트에서는 depth가 **~1.33배 길게** 읽힌다 — 실물 0.5 m가 `d ≈ 0.67 m`로 뜨는
  게 정상이고, 물속에 들어가면 metric이 맞는다. (2026-08-10 정정: 이 항목이 처음엔
  반대로 적혀 있었다.)
- **3단계의 진짜 미지수는 코드가 아니라 운용이다.** 물속에서 물체 주위를 **75° 궤도**로
  0.3~0.8 m를 유지하며 돌아야 한다(원본 표: 80°에서 9.7 mm, **100°에서 87 mm**). 그게
  기체로 가능한 기동인지는 위 3~5로만 알 수 있다. 안 되면 지상에서 미리 메시를 만들어
  `--pose-mesh`로 들고 들어가는 쪽이 현실적인 운용이다.

### C3 틸트 보정의 **병진 성분은 아직 미실측** (2026-08-17)
- **해결된 것**: `cam_tilt_deg: 43.3`이 들어갔고 실기로 확인됐다 —
  `rp_residual`이 **43.3° → roll −0.8 / pitch −0.2** 로 떨어졌다
  `[측정: 2026-08-17, 기체 정지·매트 위, rov_gui GUI 판독]`.
  두 독립 방증과도 일치(IMU→기체 회전 적합이 축 순열에서 40.7 / 41.3°).
- **남은 것**: 틸트를 **순수 회전**으로만 적용한다 — 카메라 원점을 축으로 돌리고
  `cam_t_flu`(레버암 0.2855 m)는 리마운트 전 값 그대로다. 힌지로 숙이면 렌즈
  중심도 실제로 움직이므로, 그만큼(수 cm 이하로 추정, **미실측**)이 위치에 상수
  오프셋으로 남는다. `rp_residual`은 회전만 보므로 이걸 잡아내지 못한다.
- **임시 대응**: 없음. cm 단위가 문제되는 결론을 내리기 전에 줄자로 렌즈 중심을
  재서 `hw_nav.yaml: cam_t_flu`에 넣을 것.
- **기록 경계**: `meta.json hardware.cam_tilt_deg`. 이 값이 다른 런끼리 위치·헤딩을
  합산하지 말 것 — 2026-08-17 이전 기록은 전부 수평 extrinsic으로 날았고, 최대
  0.21 m 오프셋 + yaw 오염을 안고 있다.

### IMU dead reckoning(`--imu-dr`)이 **실기 미검증** (2026-08-17)
- C3 BNO086만으로 state를 갱신하는 추정기(`rov_gui/control/imu_dr.py`)와 그
  closed-loop 모드는 **벤치·데모까지만** 검증됐다: 단위테스트 24개(등속·등가속
  적분이 1e-9까지 정확, 0.5bt²·(1/6)gβt³ 법칙, AHRS 정상상태 기울기 βτ, Kabsch
  회전+지연 복원), demo 백엔드 폐루프(주입 바이어스가 예측대로 되돌아옴,
  shadow/control 양쪽), 오프라인 재추정 왕복.
- **안 해본 것**: 실제 BNO086 샘플, 실제 마운팅 회전(캘리브 미실행 →
  `calibration_sha1: null`로 뜬다), 수중, 그리고 **닫힌 루프로 기체를 실제로 몰아본 것**.
- **가장 위험한 항목**: `--imu-dr control`은 컨트롤러가 표류하는 추정치 쪽으로
  기체를 **능동적으로 몬다**. 지오펜스가 없으므로(위 항목) 위치를 이유로 세우는
  수단은 조종사의 E-STOP뿐이다 — 운영자 결정으로 자동 abort는 꺼져 있다
  (`imu_dr.abort_err_m/abort_max_s: null`, 켜려면 숫자만 넣으면 됨).
- **순서**: 벤치 캘리브 2종 → 물 밖 트롤리에서 DR 궤적이 태그와 **같은 방향**을
  가리키는지 → 풀 station shadow → 풀 station control → line/square/circle.

### `demo_e2e.py line`이 드라이버 예산(60 s) 안에 안 끝난다 (2026-08-17, 기존 결함)
- **증상**: `demo_e2e.py pid line` → `FAIL: line never completed`. **DR과 무관**
  (`dr` 없이도 재현). station과 square는 통과한다. circle은 드라이버에
  아직 없다 — 넣는다면 예산은 `2*pi*R*laps/speed`가 아니라
  `2*pi*R*laps/min(speed, sqrt(a_lat*R))`로 잡아야 이 결함을 그대로 재현하지
  않는다(작은 반지름에서는 곡률 상한이 지배한다).
- **원인 추정**: demo 토이 플랜트의 추종오차가 커서(err p95 14 cm) governed
  path clock이 크게 감속 → 21 s 짜리 경로가 60 s 안에 안 끝난다. 즉 governor가
  설계대로 동작한 결과이고 드라이버 예산이 짧은 것.
- **임시 대응**: 없음. line 회귀는 `test_control.py`의
  `test_line_mission_is_placed_at_a_tag` 등이 덮는다.
- **제대로 고치는 법**: `demo_e2e`의 60 s 상한을 경로 길이에서 유도하거나, demo
  `SHAPES["line"]`를 더 짧게. 고치면 이 항목 삭제.

### DP 데이터셋: `extract_pose`가 tilt 0 extrinsic을 합성하는데 실기 C3는 43.3° 숙여 있다 (2026-09-01)
- **증상**: `umi_handheld/extract_pose.py:149`가 `--extrinsic c3` 기본 경로에서
  **cam_tilt_deg = 0(forward-level)** 등록을 합성한다. 실기 값은
  `config/hw_nav.yaml:161` `cam_tilt_deg: 43.3` [측정 2026-08-17].
  라벨은 "이 카메라가 C3 마운트에 달렸다면 BODY(FRD) 원점이 있었을 자리"인데,
  그 마운트 자세가 틀리면 **모든 위치 라벨이 자세 결합 오차만큼** 틀린다.
  같은 불일치를 리포는 이미 "최대 0.21 m 오프셋 + yaw 오염"으로 가격했다
  (KNOWN_ISSUES 'C3 틸트 보정' 항목). reprojection error에는 안 보인다.
- **임시 대응**: 없음. 아직 라벨을 만든 세션이 하나(0026)뿐이고 그건 fix 0이라
  오염된 산출물은 없다.
- **제대로 고치는 법**: `extract_pose`에 `--cam-tilt-deg`를 노출하거나
  `rov_gui/control/geometry.py`의 `NavConfig.R_t_frd_cam`(이미 tilt를 옳게 합성한다)를
  재사용하고, 쓴 값을 `poses.json`·`session.json`에 **경계 필드로** 찍는다.
  rig도 지그로 같은 각도에 물리적으로 고정한다. 틸트 병진 성분은 여전히 미실측이므로
  같은 기회에 `cam_t_flu`를 잰다. **대량 촬영 전에 고칠 것 — 사후엔 전량 재라벨이다.**

### DP 데이터셋: 촬영 stereo 설정과 비행 stereo 설정이 전부 다르고, warp 도착 격자도 배포 격자가 아니다 (2026-09-01)
- **증상**: 두 경로가 공유하는 knob이 사실상 없다.
  촬영(`configs/pipeline.yaml` stereo_depth): extended **true**, subpixel **true**(3 bit),
  median **off**, confidence 245, `depth_align: rectified_left`,
  decimation 2 + speckle + temporal + spatial(hole_filling 3) + threshold 100~3000 mm.
  비행(`rov_gui/backends/hardware.py:425-447`이 `StreamConfig`를 만들 때 아무것도 안 덮음
  → `c3_camera/config.py:179-192` 기본값): extended **False**, subpixel **False**,
  median **5x5**, preset robotics, `depth_align: "color"`, 후처리 스테이지 **없음**.
- **파생 결과**: (a) MinZ 300 mm(비행) vs ~112 mm(촬영) — 파지 직전 구간이 배포 센서엔
  물리적으로 없다; (b) 훈련 depth는 hole-fill·시간평활·3 m 절단, 배포 depth는 생것;
  (c) warp 도착 격자는 `configs/target_camera_underwater.yaml`의 **raw CAM_B**인데
  배포가 서빙하는 건 **CAM_A(colour) 정렬** depth다.
- **임시 대응**: 없음. 지금 촬영하면 데이터셋 전체가 이 불일치를 안고 굳는다.
- **제대로 고치는 법**: 설정 **하나**를 정해 양쪽에 강제하고 그 해시를 세션마다 찍는다.
  격자는 둘 중 하나 — 비행을 `depth_align=rectified_left`로 바꾸거나(단 rectified 모델은
  EEPROM에 extrinsics가 없어 수중 재캘리브 필요), 같은 EEPROM 덤프로 **CAM_A 타깃 모델**을
  만들어 warp를 그쪽으로 돌린다. `zarr.z_near_m: 0.20`도 배포 MinZ 위로 올려야 한다.

### DP 데이터셋: 기존 handheld 데모 25개의 pose 라벨 수율이 **0%로 측정**됐다 (2026-09-01)
- **증상**: `data/20260831/demonstration_0026/poses.json` = n_frames 576, **n_fix 0**,
  reject_counts `{"no tags": 576}`. 25개 세션 전부를 독립 재검출(세션당 10프레임,
  pupil_apriltags tag36h11)해도 id 0/1(그리퍼 손가락) 말고는 **0검출**이다.
  원인은 계획서가 적은 "태그 패밀리"가 아니라 **환경 태그가 아예 없는 실내 사무실 장면**.
  계획서 `docs/DP_TRAJECTORY_PLAN.ko.md:180-183`의 "미기록/추정" 두 헤지는 이 측정으로
  대체돼야 한다. 덤으로 그 25개는 워킹트리에 없다(2026-08-31 filter-branch 사고,
  `refs/original` 백업 ref에만 존재 — 복구 전까지 `git gc` 금지).
- **임시 대응**: 없음. 라벨용 데이터는 **0개**다.
- **제대로 고치는 법**: 새 촬영. `record.py`의 `TagProbe`가 지금은 그리퍼 손가락 태그만
  보므로(`record.py:714-715`가 `cfg['gripper']['aruco']`로 만든다), **매핑된 환경 태그 수**를
  라이브로 띄우고 `frames.csv`에 적게 확장한 뒤에 찍을 것. 안 그러면 이 사고가 반복된다
  (0026은 24일 뒤에야 발각됐다).

### DP 데이터셋: `build_zarr`에 pose/action 배열이 없고 gripper 채널이 전 스토어 상수 0.0 (2026-09-01)
- **증상**: `umi_handheld/build_zarr.py`가 쓰는 배열은 camera / camera_1 /
  camera0_main_depth / tracked_gripper_pct 넷뿐 — **pose도 action도 없다**(선행연구
  affordance 파이프라인용 스키마다). `zarr/` 7개 스토어 전부 `tracked_gripper_pct`가
  단일값 0.0이고 `gripper_dimensions_are_placeholders=true`이며,
  `meta/episode_ends`가 shape (1,) = **에피소드 1개짜리**라 다중 에피소드 슬라이싱이
  한 번도 돌아본 적이 없다.
- **부수 결함**: `.gitignore:34`의 `data/` 규칙이 앵커되지 않아 `zarr/*.zarr/data`가
  통째로 무시된다 → `git ls-files zarr` 28개(.zgroup/.zarray/episode_ends)뿐인데
  `git status`는 아무 말도 안 한다. 새로 clone하면 **픽셀 없는 zarr**를 받는다.
- **제대로 고치는 법**: 슬라이서(A2)를 만들 때 `poses.npy`+`gripper_width.npy`를
  물리고 배열을 추가한다. `.gitignore`는 `/data/`로 앵커한다.

### cv2 5.0의 "두꺼운 검정 + 얇은 흰" 외곽선이 긴 라벨의 꼬리를 유령으로 남긴다 (2026-09-06)
- **증상**: OpenCV 5.0.0의 `putText`는 thickness에 따라 글리프 전진폭이 달라진다
  [측정 2026-09-06, `cv2.getTextSize`, 같은 79자 문자열 @scale 0.40: **thickness 1 → 408 px,
  thickness 2/3/4 → 435 px**]. 그래서 같은 org에 검정 3 → 흰 1을 겹쳐 그리는 관용구는
  외곽선이 채움보다 점점 오른쪽으로 밀려, 문자열 끝에서 27 px만큼 **꼬리 글자가 그림자로
  남는다**. 라벨이 길수록 뚜렷하고, 짧은 라벨에서는 안 보인다.
- **해당 코드(안 고침)**: `rov_gui/tools/depth_compare.py:272-277` `_put`,
  `c3_camera/viz.py:133-143`. 둘 다 cv2 5.0.0인 env(`fstereo`/`oakd`/`rovgui-pose`)에서 돈다.
  cv2 4.x(`robust` 4.10, `umi` 4.7)에서는 재현되지 않는다.
- **임시 대응**: `data_collection/make_depth_trajectory_video.py`의 `text()`는 두께를 1로 고정하고
  ±1 px 오프셋 4장으로 외곽선을 그린다 — 전진폭이 같아지므로 유령이 없다.
- **제대로 고치는 법**: 위 두 곳도 같은 방식으로. 순수 미관 문제이고 측정값엔 영향 없다.


## 📌 알려진 한계 (당장 고칠 계획 없음, 잊지 말 것)

### 풀장 CCTV 카메라(`--pool-cams` / `./c3 cctv`): 4대 실기 확인, **`--source hw` 스테이션 런·허브 연결은 미확인** (2026-10-01)
- **확인된 것**: 재부팅 후 4대(`SPCA2650` 1 + `USB RGB Camera` 3)를 컨트롤러 `0000:13:00.0`의 PC 포트 1·2·9·10에 한 대씩
  꽂아 1080p30 동시 녹화 15 s — 4파일 `complete: true`, dropped 0, 29.5–30.1 fps; 녹화 중 카메라 프로세스+ffmpeg 0.87 코어,
  스테이션 쪽 0.02 코어 [측정: `pool_cam/bench_out/cost_20261001_real4_1080p30.txt`]. 스트리밍 중 `USB RGB Camera`는
  alt 7 = 3072 B/µframe(196.6 Mbit/s 예약)을 쓴다 [측정: sysfs `bAlternateSetting`·디스크립터 2026-10-01 19:00] — 두 대만
  합쳐도 USB 2.0 주기 전송 한도(6000 B/µframe)를 넘는데 한 컨트롤러에서 4대가 돌았으니 대역은 **PC 포트마다 따로**다 [유도].
- **모르는 것**: (a) **허브(멀티 포트 어댑터)**: 허브 뒤 카메라는 허브의 링크 하나를 나눠 쓰므로 1080p30이면 한 대만 들어갈
  것이다 [유도, 미시험] — `--list`가 같은 PC 포트를 공유하는 카메라를 경고한다. (b) `./c3 gui --source hw --pool-cams` 전체 런
  (C3·ROV·정책 GPU와 함께). (c) 4대가 모두 한 컨트롤러라, 위 uvcvideo 크래시 같은 고장이 나면 4대가 같이 멈춘다 —
  다른 컨트롤러(`0000:11:00.0` 등) 포트로 나누면 고장 범위가 나뉜다(대역 때문은 아님).

### 2026-09-14 이전에 앱이 쓴 기록 안의 경로는 여전히 `sessions/…`를 가리킨다 (2026-09-14)
- **증상**: 데이터 루트를 `data/YYYYMMDD/<run>[_kind]/` 하나로 접으면서(`tools/migrate_data_layout_20260914.py`,
  485건 이동) config 주석·docs·journal·memory의 인용은 새 경로로 고쳐 썼지만(292줄), **앱이 런타임에 쓴 기록**은
  손대지 않았다 — `_pose.jsonl`의 MESH 행 `path`/`ref_views_dir`(구 `sessions/pose_meshes/obj_*`), 옛 dryrun
  `meta.json`의 `log_dir`(`rov_gui/tools/dp_policy_out/*/`), `rov_gui/tools/fstereo_bench_out/*.json`, 각 런의
  `events.log`/`mission_log.txt` 본문. 그 경로는 이제 존재하지 않는다.
- **해결 경로**: `data/MIGRATION_20260914.json`의 `moves[]`(from→to)로 풀어 읽는다. 기록을 고쳐 쓰지 않는 이유는
  측정치 인용 규칙과 같다 — 기록은 당시 그대로여야 한다.
- 같은 이유로 `sessions/demonstration_0015/0019`, `sessions/grippercalibration_0004`, `sessions/session_syn`처럼
  이행 시점에 이미 없던 폴더의 인용(`configs/pipeline*.yaml`, `umi_handheld/extract_gripper_width.py`, journal)은
  그대로 두었다 — 원래부터 죽은 참조였고, 어디로 갔는지 확인된 바 없다.

### `replay-html`의 z 눈금자가 라이브 패널과 갈라졌다 (2026-09-07)
- **증상**: 라이브 `3D` 패널의 깊이 눈금은 2026-09-07에 **오른쪽 여백 고정 게이지**가
  됐지만(`_z_gauge`/`_paint_z_axis`), `./c3 replay-html`이 굽는 HTML은 **독립 JS 포트**
  `rov_gui/tools/replay_html_page.html:411 drawZAxis()`를 쓰고 그건 여전히 **선체의
  투영된 왼쪽 모서리 − 34 px**에 앵커한다. 즉 같은 런을 두 도구로 보면 눈금자가 한쪽은
  가만히 있고 한쪽은 기체를 따라다닌다.
- **왜 남겼나**: 다른 언어의 별도 구현이고 테스트가 없다. 사용자가 물은 것은 라이브
  GUI였다.
- **임시 대응**: 리플레이 HTML의 눈금자는 "기체 옆에 붙은 옛 눈금자"로 읽을 것. 두
  도구의 눈금 위치를 같은 것으로 인용하지 말 것.
- **제대로 고치는 법**: `_z_gauge()`의 규칙(고정 창 → 고정 픽셀 띠, 자세·pan 무관,
  확장 전용 래치)을 `drawZAxis()`에 그대로 포팅. 창 상수는 파이썬 쪽
  `Z_GAUGE_WINDOW_M`을 HTML 생성 시 주입해서 두 곳에 숫자를 두 번 쓰지 않는 것이 좋다.

### `policy_observe_smoke`의 폴백 자세가 매트 아래다 (2026-09-07)
- **증상**: `rov_gui/tools/policy_observe_smoke.py:86, :204`가 fix가 없을 때
  `_set_p_act((0.0, 0.0, 0.35))`로 떨어진다. map z는 **아래가 +**라 +0.35는 태그 평면
  **아래** — 실기록 44,548개 accepted fix는 전부 −0.856 .. −0.165 m 구간이고
  [측정: `data/2026090*/*/nav_*/fixes.csv`, 32파일] 매트
  아래로 간 적이 없다. 새 깊이 게이지가 바늘로 그 값을 찍기 시작해서 이제 스모크 샷에
  "실제로 존재한 적 없는 깊이"가 보인다.
- **임시 대응**: 스모크 샷의 깊이 숫자는 인용하지 말 것.
- **제대로 고치는 법**: 두 곳을 실측 p50인 `-0.22`로 바꾼다. 한 줄짜리지만 이번 변경의
  범위 밖이라 손대지 않았다.

### 두 카메라 폴백 + 틸트 추적은 **실기 미검증**, 틸트 속도는 [예측] (2026-09-07)
- **무엇**: `hw_nav.yaml: fallback` — C3가 `after_s`(0.2 s) 조용하면 기본 RGB의
  fix가 대신 들어간다(`control/nav_fusion.py`). RGB의 [예측] extrinsic 오차는 body
  프레임 상수 E로 학습해 나눠 주고, 학습 전엔 보류한다. PAYLOAD의 CAMERA TILT 줄이
  마운트 각을 추적한다(`control/tilt_tracker.py`).
- **검증된 것**: 합성뿐. 40° 틸트 + 5 cm 오차를 심은 RGB가 8쌍 뒤 5 mm 안으로
  복원되고, 실측 틸트가 40.0°로 나온다(`rov_gui/tests/test_control.py`
  `test_second_camera_fallback_*`, `test_nav_fusion.py`).
- **첫 실기(2026-09-07 14:52, `0907_145206/nav_145240`)에서 배운 것**: 정렬은 됐다
  (`offset 67 mm spread 11 mm`, RGB fix 63) — 막힌 건 **RGB 카메라 모델**. 왜곡항이
  없어 다태그 RGB 프레임이 3 px 게이트에서 전부 죽고, 통과한 63 중 58이 단일 태그
  (self-residual 2.1~2.4 px vs C3 0.17). 임시로 `fallback.max_reproj_px: 8.0`
  [예측]; 제대로는 REC NAV(양 카메라가 매트를 보게, 여러 위치·헤딩) →
  `rov_gui/tools/calibrate_second_cam.py --yaml` → `second_cam`에 붙여넣기 →
  게이트 복귀. **2026-09-08 완료**: `nav_111010`(RGB 3,463 프레임)로 fx 713.8 + 왜곡
  5항, 렌즈 위치·틸트 15.9° 실측 → `second_cam` 기입; 재현서 RGB 채택 43.5% → 100%,
  두 솔버 offset 2 mm. 게이트는 4.0(RGB 픽셀 피치 2배 [유도]). 남은 것: 아래 1, 3.
- **실기에서 확인해야 할 것**:
  1. `second_cam.tilt_rate_deg_s: 30` **[예측]** — 한 번도 재지 않았다. 마운트를
     LEVEL에서 DOWN으로 2 s 눌러 두고 PAYLOAD의 `measured:tags` 값을 읽으면 그
     자체가 실측이다(태그가 양쪽 카메라에 보여야 함). 그 전까지 dead-reckoned
     값의 ±는 적분한 각의 절반씩 커진다.
  2. RGB 피드의 `t_capture`는 **도착 시각**이다(RTP h264 지연 100~300 ms 미보정,
     `hardware.py _publish`). 정렬 쌍은 `pair_tol_s` 0.08 안에서 C3 fix와 맺히므로
     쌍 사이 실제 시간차가 지연만큼 난다. **실측 2026-09-08**: 그 런의 재현에서
     spread 1.7 mm(207쌍) — 그 속도에선 무시할 크기. 빠르게 움직이는 런에선
     `spread_mm`을 다시 볼 것.
  3. 핸드오버 순간의 실제 계단 크기(controller.json `nav_fallback.offset_mm`
     대비 `spread_mm`), 그리고 MPC engaged 중 핸드오버가 기체를 움직이는지.
  4. `fallback.min_pairs: 10`이 실제 프레임률에서 충분히 빨리 차는지(C3 15 Hz +
     RGB 32 fps면 ~1 s [예측]).
- **알려진 한계(설계상)**: 두 카메라를 **한 PnP에 합치지 않는다**(RGB 캘리브가
  전부 [예측]이라 좋은 C3 해를 오염시킬 뿐). 마운트가 움직이면 E를 버리고 다시
  배우므로, 움직인 직후 C3가 끊기면 그 구간은 공백이다(`allow_unaligned: true`로
  바꿀 수 있으나 계단을 감수하는 것).

### replay 미션(plan_stream): **실기 미검증** + jaw 중재 부재 (2026-08-30, 2026-09-02 갱신)
- **실기 미검증**: shape `replay`(기록 시연 재비행, `control/plan_stream.py` →
  `set_path_plan_ned`) 전체가 오프라인(9/9 + 19/19)과 demo_e2e(실제 acados,
  honest-완주 판정)까지만 검증됐다. safety-code-reviewer 감사(2026-08-30)의
  CRITICAL(딜맨이 래치된 버튼 비트 미해제)·HIGH(발산 가드 부재)는 **수정 완료**.
- **jaw 중재 부재(잔존)**: 파일럿 G/H와 replay/policy가 같은 `cmd_gripper_drive`를
  last-writer-wins로 쓴다. 완화 3중(스트림 미션은 상태 변화 에지에서만 방출 + 기본
  `gripper: false` + 딜맨이 이제 held drive를 놓음)으로 M0엔 충분하다는
  감사 판단이지만, 파일럿이 **누르고 있는** 중에 워커 에지가 덮으면 릴리스까지
  파일럿 의도가 밀린다. 제대로: sink에 jaw 전용 중재(파일럿 우선 + 워커 방출 무시
  창) — `replay.gripper: true` / `policy.gripper: true`를 상시 쓰기 전에.
  (2026-09-02부터 모든 드라이브가 `MpcWorker.on_gripper_drive`의 open-loop 폭
  추정기에도 적분되므로, 중재가 생기면 추정기도 같은 곳에서 먹여야 한다.)

### 라이브 diffusion policy(shape `policy`): TCP/턱 기하 — **x만 실측**, y·z 미실측 (2026-09-02, 2026-09-08 갱신)
- **상태**: `--policy` 스택은 2026-09-07부터 풀에서 돈다(data/20260907/0907_145206
  이후); 실기에서 드러난 문제는 이 파일의 2026-09-07/08 항목들(턱 채널, 추종자 정지대역, yaw 좌편향)에 따로
  있다. 파지 결과를 정책의 것으로 귀속하기엔 아직 이르다(아래).
- **TCP = 턱 기하**: `policy.tcp_body_flu_m: [0.502, 0, -0.17]` = `cam_t_flu` + 렌즈→그립 [0.196, 0, −0.275]
  (2026-09-08; x [측정: data/20260908/0908_180453_observe/policy_obs/rgb/000160.jpg — 턱에 문 병을
  태그 58 중심에 놓음, + 0908_170428/policy_obs/rgb/000000.jpg 병 높이 0.084 (±0.006) m + 0908_175151/policy_obs/rgb/000000.jpg
  열린 턱; 범위 0.187~0.204], z CAD 수직 [유도], y=0 [예측]). 2026-09-02~07 런의 0.4165(시뮬 JAW_POS)는 이미지가
  반증했고 **기록 경계**다(meta `policy.tcp_body_flu_m`). 렌즈에서 0.338 m / 광축 아래 약 11° [유도]라 턱은
  프레임 아래쪽(행 265~360)에 **보인다** — 핸드헬드(카메라 축 앞 0.346 m / 아래 22° [유도: 데이터셋 config
  offset_m [0.0355, 0.1293, 0.3186]])보다 광축에 가깝다. **핸드헬드와의 불일치 벡터(보정 안 함)**: 턱 − 손끝,
  C3 광학 프레임(x 우, y 하, z 전) = [−0.036, −0.062, +0.012] m(|0.072|; 옛 0.4165는 [−0.036, −0.004, −0.050],
  |0.062|) = body FLU 앞 +0.051·좌 +0.036·위 +0.037 [유도: `NavConfig.R_t_frd_cam` + `tcp_offset_from_body`].
  dp는 카메라 상대라 카메라 궤적은 이 키와 무관하고(t_bt는 yaw 일정 시 정확히 소거), 이 벡터가 "정책이
  학습대로 카메라를 놓았을 때 턱이 물체에서 어긋나는 양"이다 → 예상되는 첫 파지 실패는 **~5 cm 지나침·~4 cm
  높음·~3.6 cm 왼쪽**이지 "못 미침"이 아니다(옛 값은 3.4 cm 못 미침에 같은 좌/위; 턱이 정말 축 오른쪽
  0.011 m면 좌 2.4 cm). `tcp_offset_cam_m`으로 덮지 말 것 — 턱은 거기 있다; 보정은 별도 결정·별도 기록
  경계(예: 플랜 레벨 오프셋). 남은 것: 턱 y·z 미실측, `cam_t_flu` 자체의 x/z 의심(별도 항목; 턱은 렌즈에
  앵커돼 있어 TCP는 어느 쪽이든 맞다).
- **제대로**: C3 프레임에서 턱 중심을 스테레오 삼각측량으로 실측(`measure_rig.py --pick-tcp`류)해
  `tcp_offset_cam_m`에 넣으면 렌즈 위치가 소거된다. 그 전엔 파지 결과를 정책의 것으로 귀속하지 말 것.

### policy: 디바이스 depth는 **거부**, FoundationStereo만 parity 경로 (2026-09-02)
- `--source hw --policy`에 `--fstereo`가 없으면 기동을 **거부**한다(`--mpc`가 없는 건
  경고뿐이고, 거부가 경고보다 먼저 판정된다). C3 자체 depth는 CAM_A 정렬(63.7°
  화각)이라 정책 FOV(CAM_B 85.6°) 안에 든 부분만 덮고 [유도: 화각 비교, 면적 비율은
  미계산], ×0.64 stopgap이 걸린 온디바이스 매처다. `--policy-allow-device-depth`는
  벤치 실험용이며 경고와 함께 meta에 `depth_scale_applied: 0.64`와 coverage가 남는다.
- `--fstereo`도 훈련 스토어 설정(iters 16 / scale 1.0 [측정: `~/Desktop/data
  collection/depth/0/depth.zarr/.zattrs`])과 다르면 경고한다(2026-09-02까지는
  `--policy-allow-fs-mismatch` 없이 거부; 지금 그 플래그는 meta에 기록만 된다).
  체크포인트는 타이핑 값이 아니라 실제 로드될 기본값까지, 2026-09-30부터는 경로가 아니라
  내용(앞 8 MiB sha1; 스토어 쪽 파일이 지워졌으면 upstream 공개본 sha1과)으로 비교한다
  (2026-09-14~09-30 런은 `data/checkpoints` 심볼릭 링크 때문에 이 검사가 안 돌았다 — 고침). 이
  설정의 실측: 7.26 Hz, solve 137 ms, panel latency 243 ms [측정: rov_gui/tools/fstereo_bench_out/policy_bench_20260902_141224.json, C3 실기, alpha 0.5, scale 1.0, iters 16, 330 frames, GPU shared with a concurrent test run].

### policy: FS 7.26 Hz라 depth 페어가 훈련 stride의 **2.07배** — 계기 parity와 시간 parity가 충돌 (2026-09-02)
- 훈련 관측은 66.7 ms 간격 두 장인데, `--policy` 기본(iters 16 / scale 1.0 / alpha 0.5)의
  FoundationStereo는 7.26 Hz, solve 137 ms, panel latency 243 ms [측정: rov_gui/tools/fstereo_bench_out/policy_bench_20260902_141224.json, C3 실기, alpha 0.5, scale 1.0, iters 16, 330 frames, GPU shared with a concurrent test run]라 한 관측의 depth 두 장이 프레임 간격 138 ms = 2.07×
  떨어진다(플랜마다 `obs_pair_dt_s` 기록; 페어 상한 3.0·obs_dt = 200 ms라 skip은 안
  된다). proprio 두 행은 이제 **같은 138 ms**로 잡아 이미지-proprio 기준선은 일치하지만
  (검증 2026-09-02 수정), 네트워크가 훈련에서 본 적 없는 2배 기준선을 본다는 사실은
  남는다. `--fstereo-scale 0.5` + `--policy-allow-fs-mismatch`면 ~15 Hz 페어(≈ near
  창 안)를 얻는 대신 계기 불일치(훈련 scale 1.0)를 산다 — 어느 쪽 불일치가 싼지는
  실기 A/B 전엔 모른다 [예측]. 고치려면 20 Hz FS(모노 MJPEG, fstereo 메모리)나
  scale 0.5 재학습 depth 스토어.

### policy: proprio SNR — 66.7 ms 운동 단서가 태그 fix 잡음과 같은 자릿수 (2026-09-02)
- 정책의 상대 자세 관측은 두 행의 차이다(훈련 66.7 ms; 실기에선 depth 페어 간격과
  같은 138 ms — 위 항목). 0.08 m/s에서 그 변위는 66.7 ms에 ~5 mm, 138 ms에 ~11 mm
  [유도]인데 단일 태그 PnP의 위치 잡음은 ~1 cm 수준 [예측: demo 잡음 모델과 같은
  자릿수; 실기 잔차 fit은 아래 "`rov_gui --mpc`: 폐루프 MPC 스택 전체가 실기·수중
  미검증" 항목의 3번(EAOB 시그마) 참조]. fix 기준 히스토리(A6)와 degenerate /
  fix-lag skip으로 **양자화**와 몰래 줄어드는 단서는 막았지만 SNR 자체는 못 올린다.
  `dp_policy_offline.py noise`가 민감도를 재는 도구이고, 실기 값은 첫 런의
  plans.jsonl `lowdim`에서 읽을 것.

### policy: hold-tail cost mask(A10)가 **실기 미검증** (2026-09-02)
- `policy.hold_tail: mask`는 플랜 끝 너머 stage의 위치/선속도 가중치를 0으로
  둔다(`HwDobMpc._apply_stage_weights`) — 단 **플랜이 살아 있고 래치되지 않은 동안만**
  (`_tick_replay`의 `live` 게이트, 2026-09-02 검증 수정: 그 전엔 플랜 만료·halt 뒤에도
  호라이즌 전체가 마스크돼 x/y/z·heave 가중치가 0이었다; 만료 뒤엔 `w_stage=None`,
  종점 hold 전체 가중치). 오프라인에선 StubCtrl이 w_stage를 받는
  것과 acados가 완주하는 것까지만 확인했다. 실기에서 "브레이크 조기 밟기"가 정말
  사라지는지, 마스크가 solver 조건수를 흔들지 않는지는 미확인. 의심되면
  `hold_tail: track`(종전 거동)으로 A/B.

### policy: 검증 오차가 ~36 mm RMS — 정밀 파지엔 부족 (2026-09-02)
- selected.json의 val action pos MSE에서 [유도]한 위치 RMS ~36 mm. 62 mm 턱 [스펙]
  대비 여유가 얇다. 첫 풀 런의 기대치는 "정책이 대체로 맞는 방향으로 기체를 민다"
  까지이고, 파지 성공률은 더 많은 시연과 재학습 뒤의 일이다.

### policy/fstereo: 모델 로드 셋이 겹칠 때 **네이티브 크래시 2회** — 원인 미귀속, 직렬화로 완화 (2026-09-02)
- **증상 (둘 다 같은 날, 같은 프로세스 구성)**: (a) `rov_gui/tools/policy_dryrun.py` 실제
  체크포인트 4회 중 1회가 `DP policy: loading selected.ckpt` 직후 core dump(직전에 acados 빌드
  종료, FoundationStereo 미개입); (b) `rov_gui/tools/policy_bench_check.py` 1회차가 t≈2.4 s에
  `malloc(): invalid size (unsorted)` — DP ckpt 로드 + acados mpcc 빌드 + FoundationStereo
  로드가 겹친 순간. (b)는 JSON을 쓰고도(FS 7.27 Hz, 220 frames, grid coverage 1.0) 세션
  ready=False로 종료에서 멈춰 400 s 타임아웃으로 kill됐다
  [측정: rov_gui/tools/fstereo_bench_out/policy_bench_20260902_150726.json]. 재시도(151527)와
  참조 런(141224)은 정상(3.5~4.3 s 로드).
- **원인 미귀속**: 스택이 없다(faulthandler 미설정이었다). torch.load 둘 + 방금 빌드한 솔버
  `.so`의 dlopen이 세 스레드에서 겹치는 조합이라는 것만 안다.
- **임시 대응 (적용됨)**: `rov_gui/perception/upstream.MODEL_LOAD_LOCK`이 FoundationStereo
  로드(`FStereoSession._load`)와 정책 로드(`DpPolicySession.load`)를 **직렬화**하고,
  `rov_gui/__main__.main`이 `faulthandler.enable()`로 다음 크래시에 스레드를 남긴다. acados
  빌드는 여전히 병렬(`MpcWorker.setup`).
- **제대로 고치는 법**: 재발하면 faulthandler 스택으로 범인을 특정하고, 필요하면 acados 빌드도
  같은 잠금 뒤로 보낸다(시작 ~+3 s). 재현 시도는 `policy_bench_check.py`를 `timeout -s KILL 150`
  아래에서 돌릴 것 — 멈춘 프로세스가 카메라를 물고 있으면 다음 런이 DeviceBusy로 죽는다.

### 실기 Newton gripper: servo 채널·PWM 레인지·지속 close 거동이 **미기록** (2026-08-30)
- 리포에 있는 실측은 버튼 기능 번호 둘뿐(BTN0/15 = 77/76, 2026-08-06,
  `rov_gui/__main__.py:250`). servo9=Newton은 산문 한 줄(c3_camera/dataset.py:904),
  SERVOn_FUNCTION/MIN/MAX/TRIM은 어디에도 없다 [스펙 미확인]. 지속 close 비트에서의
  스톨 전류·열·클러치 거동도 미확인 — replay gripper의 `gripper_hold_max_s` 4 s
  auto-neutral은 UMI-U 관행이지 이 장치의 실측이 아니다 [예측].
- 제대로: QGC/mavproxy로 SERVO 파라미터 덤프를 받아 여기 기록하고, 벤치에서 지속
  close 전류를 한 번 재서 hold_max를 그 수치로 교체.

### `rov_gui --mpc`: 폐루프 MPC 스택 전체가 **실기·수중 미검증** (2026-08-12)
AprilTag PnP → EAOB+acados NMPC → MANUAL_CONTROL 폐루프(`rov_gui/control/`)는
벤치까지만 검증됐다: acados smoke(rovgui-pose, numpy 2, p50 3.1 ms), 오프라인
테스트 18/18, demo 백엔드 폐루프 square 1랩 완주(솔버 실패 0). 실기 앞에서 반드시
남는 것들:
1. **축 게인 4개가 전부 [예측]이다** (`config/hw_mpc.yaml: axis_gain`). T200
   곡선+믹서 기하로 때려잡은 값이고, MANUAL 모드 pilot gain 설정에도 좌우된다.
   → P4 스텝 캘리브레이션으로 교체하고 이 항을 갱신할 것. 틀린 채로도 안전하지는
   하다(axis_cap 0.5 + 지오펜스 + deadman), 성능만 무너진다.
2. **wall preset(`x_into_wall`)의 태그 축 가정 미확인** — 태그가 수직 벽에 정자세
   부착이라는 가정. 어긋나면 state_assembler가 매 틱 보고하는
   `rp_residual_deg`(태그 자세 vs ATTITUDE)로 잡힌다 → P3에서 정지 상태 확인.
3. **EAOB 시그마도 [예측]** — 실제 PnP/속도미분 노이즈를 P3 기록으로 피팅해
   `hw_mpc.yaml: eaob_sigmas`를 갱신할 것.
4. **MANUAL_CONTROL엔 roll/pitch 축이 없어 MPC의 K/M 출력을 버린다** — heavy의
   수동 복원모멘트에 의존. 잔잔한 수조에선 문제없어야 하지만 [예측]이다.
5. 카메라 extrinsic 원점은 heavy_c3 COM 기준(`c3_payload_frames.json`) — 그리퍼
   장착 시 COM이 이동한다. 하방 재장착(floor 기하) 시엔 extrinsic 재실측 필수.
6. **EAOB/NMPC는 고정 dt=50 ms를 가정하는데 QTimer 틱은 지터가 있다** — 관측자
   예측과 참조 샘플링이 벽시계가 아닌 고정 격자를 쓰므로, GUI 프로세스가 바쁘면
   (비디오 디코드, 화면 녹화) 실효 틱이 늘어지고 그만큼 모델 오차가 EAOB의 w로
   샌다. 벤치에선 solve p99 6.4 ms라 여유가 크지만 수중 세션에서 CSV의
   `solve_ms`·행 간격을 확인할 것. 심하면 MpcWorker에 실측 dt 전파를 추가한다.
   (2026-08-12 멀티에이전트 리뷰가 제기, 세션 한도로 미검증 — 판단 보류 항목.)

### `rov_gui`: 카메라 틸트 · 조이스틱 passthrough · depth 20 fps가 **실기 미검증** (2026-08-06)
세 가지 다 같은 이유다 — 구현한 날 C3가 크래시해서 네트워크에서 사라졌고 ROV도
분리돼 있었다. 코드는 데모 백엔드와 27개 오프라인 테스트로만 검증됐다.

1. **카메라 틸트 기능 번호가 [스펙]이다.** `BTN_FUNCTION`의
   `mount_tilt_up=22 / mount_tilt_down=23 / mount_center=21`은 ArduSub `JSButton`
   enum에서 왔고 **이 기체로 확인한 적이 없다**(리포에도 ArduPilot 소스가 없다).
   버튼 번호 9/10/7은 사용자의 QGC 배치 캡처에서 온 것이라 근거가 있다.
   - **완화됨**: 접속하면 `BTNn_FUNCTION`을 읽어 대조하고, 어긋나면 그 버튼을 **누르지
     않고** 양쪽 가능성을 로그에 적는다. 즉 틀렸을 때의 증상은 "틸트가 조용히 안 됨 +
     빨간 로그"이지 "엉뚱한 기능이 눌림"이 아니다.
   - **확인법**: ROV 연결 후 `./c3 gui --source hw --allow-command` 로그에서
     `button 10 = mount_tilt_up — tilt_up ok` 세 줄을 확인. 어긋나면 로그가 알려주는
     실제 값으로 `BTN_FUNCTION`을 고치고 이 항목에서 1번을 지운다.
2. **조이스틱 버튼 번역이 실기에서 안 돌아봤다.** 2026-08-07에 커널 번호를 그대로
   보내다 **틸트 버튼이 arm을 걸어 모터가 살아나는 사고**가 났고, SDL 번역을 넣어
   고쳤다(실기 관측 4건과 일치, `test_pad_buttons_are_translated_to_the_vehicles_numbering`).
   번역 자체는 아직 기체로 확인 못 했다.
   - **확인법**: COMMAND ENABLE만 켜고 **DISARM 상태에서** 버튼을 하나씩 눌러
     `JOY` 줄의 `btn 커널>기체` 값을 본다. **화살표 왼쪽(커널 번호)은 케이블에 따라
     다르다** — BT면 LB/RB `6>9`,`7>10` · View `10>4` · Menu `11>6`,
     USB면 LB/RB `4>9`,`5>10` · View `6>4` · Menu `7>6`(둘 다 실측 2026-08-17).
     **오른쪽(기체 번호)이 양쪽 다 같아야 맞는 것**이고, 십자키는 `>11..14`.
     그 다음에야 arm한다.
     조명은 **한 번에 한 칸**이어야 한다(두 칸이면 칩이 명령을 중복 발행하는 것 →
     `notify=False` 경로 확인).
   - **주의**: 패드 arm은 버튼 한 번이다(화면 ARM만 1.2 s hold). QGC와 같게 한 의도적
     선택이지만, **패드 arm은 모드를 물려받는다** — 화면 ARM만 MANUAL을 먼저 요청한다
     (비트마스크가 기체로 직행해서 가로챌 수 없다). 패드로 arm할 거면 모드를 먼저 볼 것.
4. **비행 모드 제어가 실기 미검증이다.** `MAV_CMD_DO_SET_MODE`로 MANUAL(19) 등을
   요청하고 ARM 버튼이 arm 직전에 MANUAL을 먼저 보낸다. 데모 백엔드로만 검증했다
   (`test_arm_puts_the_vehicle_in_manual_first`).
   - **확인법**: ARM 후 텔레옵의 `MANUAL` 버튼이 켜지는지(= 기체가 HEARTBEAT로 MANUAL을
     보고), 그리고 armed + 입력 0에서 스러스터가 1500 µs로 가만히 있는지. `STAB`를
     누르면 그때 비로소 움직여야 한다. ArduSub이 armed 중 전환을 거부하면 버튼
     하이라이트가 안 바뀌는 것으로 드러난다(요청이 아니라 기체 보고를 그리므로).
3. **영상 기본값 조합이 [유도]다.** `컬러 640×360 q80 @30 + depth 640×360 @20` =
   83.1 Mb/s(92%). 부품은 전부 실측(depth 산식 + C3 자기 프레임 재인코딩)이지만
   **조합을 링크에서 재본 적이 없다.** 92%는 여유가 크지 않다.
   - **확인법**: 카메라 복구 후 HUD에서 depth 20.0 fps / drop 0%, 컬러 30 fps /
     drop 0%, 합계 Mb/s가 85 아래인지. 드롭이 보이면 `--mjpeg-quality 75`(91%) →
     `--fps 20`(88%) 순으로 내린다.
- **부수**: 그 스윕 중 C3가 `ping was missed → Device likely crashed but did not
  reboot` 로 죽고 전원이 돌아올 때까지 네트워크에서 사라졌다. 재현 조건 불명 —
  다시 보이면 별도 항목으로 올릴 것.

### `rov_gui`: ROS 2 백엔드는 **한 번도 실행된 적이 없다** (2026-08-06)
- **사실**: 이 데스크톱에 `/opt/ros`가 없고 `robust` env에 `rclpy`도 없다(2026-08-06 확인).
  `rov_gui/backends/ros2.py`는 구조만 완성돼 있고 **import 가드까지만 검증**됐다 —
  토픽 이름, QoS 선택, `image_to_bgr`의 인코딩 처리, 퍼블리시 타이머는 전부 미실행 코드다.
- **영향**: `--source ros2`가 처음 돌아갈 때 실패해도 놀랄 일이 아니다. demo/hw 경로는
  `rov_gui/tests/test_offline.py`(14개)가 덮지만 ros2는 아무것도 덮지 않는다.
- **임시 대응**: 실패해도 창은 뜨고 모든 패널이 OFFLINE으로 남는다(워커 예외 → `bus.log`).
  즉 조용히 틀린 값을 그리지는 않는다.
- **제대로 고치는 법**: ROS 2를 소싱한 인터프리터에서 실제 토픽으로 한 번 돌리고,
  (a) `sensor_msgs/Image` step 패딩, (b) `SensorDataQoS`로 실제 conflate가 되는지,
  (c) 헤더 stamp가 호스트 시계와 동기돼 있는지(아니면 latency가 상수 오프셋으로 읽힌다)
  세 가지를 확인한 뒤 이 항목 삭제. conda env와 ROS python을 섞으면 rclpy 첫 spin에서
  ABI 크래시가 나므로 인터프리터를 하나로 골라야 한다.

### C3 BNO086 IMU: extrinsic 없음 + 가속도계 스케일 +20% → VIO 쓰려면 캘리브레이션 필수 (2026-07-29)
- **사실 1**: `getImuToCameraExtrinsics(CAM_A)` → `IMU calibration data is not available
  on device yet.` 공장 캘리브레이션은 카메라만 담고 있어 **T_imu_cam이 미지**다.
  (카메라 intrinsics·스테레오 extrinsics는 정상이니 RGB-D/스테레오는 영향 없음.)
- **사실 2 — 2026-08-17 정정: "스케일 +20%"는 오독이었다. 실제는 x축 바이어스다.**
  원래 기록은 "정지 |a| = 11.8 m/s² (중력 대비 +20%) → 스케일 미보정"이었는데,
  그건 **한 자세(똑바로 세운 상태)에서만** 잰 값이었다. 6자세 텀블로 재보니:

  | | 값 |
  |---|---|
  | raw \|a\| (자세별) | **8.37 ~ 11.69** m/s², mean 10.63 |
  | 적합된 scale | **[1.0013, 1.0071, 1.0092]** — 1에서 **0.9% 이내** |
  | 적합된 bias | **[1.803, 0.007, −0.064]** m/s² = **0.184 g, 거의 전부 IMU x** |
  | 보정 후 \|a\| | **9.806 ± 0.050** (중력 9.807) |

  `[측정: data/20260817/0817_101511/ +
  0817_100139/ 의 *_c3_imu.jsonl 2테이크, 21985 정지샘플 6자세;
  config/c3_imu_calib.json sha1 7081ff43]`

  스케일 오차는 자세에 따라 \|a\|가 8.4↔11.7로 흔들리는 패턴을 만들 수 없다. 똑바로
  세운 자세에서 중력 방향이 IMU x와 거의 나란해서(gravity dir ≈ [0.83, 0, 0.56])
  1.8 m/s² 바이어스가 그대로 더해진 것이고, 그게 11.8로 읽혔다.
- **왜 이 구분이 중요한가**: **고정 바이어스는 매 런의 정착창 정적보정이 자세와
  무관하게 완전히 제거한다.** 스케일 오차였다면 자세가 바뀔 때마다 `s·g·sinΔ`로
  다시 샜을 것이다(그 항으로 "2° 피치에서 10 s에 3.4 m"를 계산해 왔는데, 그
  항 자체가 없다). 추측항법 오차예산이 걱정하던 것보다 낫다.
- **여전히 사실**: accuracy 플래그는 UNRELIABLE, 축 방향은 벤더 미문서화 —
  다만 `R_frd_imu`는 이제 실측했다(아래).
- **영향**: visual-inertial SLAM(VIO)에서 스케일과 중력 정렬이 틀어진다. 더 나쁜 건
  **"알고리즘이 안 맞는 것처럼" 실패**해서 원인을 IMU로 의심하기 어렵다는 점.
  RGB-D SLAM / stereo SLAM만 쓸 거면 무관하다.
- **임시 대응**: `c3_collect.py`가 두 사실을 모든 데이터셋의 `metadata.txt`에 명시하고,
  샘플마다 accuracy 필드를 남긴다. `c3_dataset_check.py`도 extrinsic 부재를 경고한다.
  → 동료가 모르고 쓰는 일은 없다.
- **2026-08-17: 실행 완료. `config/c3_imu_calib.json` sha1 `0aae3e4d`.**
  accel scale/bias는 6자세 텀블 2테이크(21985 정지샘플), `R_frd_imu`는 60~100초
  wiggle 2테이크. 결과는 위 "사실 2" 표.
  **`R_fit.rms_deg`(6~8°)를 R의 오차로 읽지 말 것** — 그건 C3 원시 자이로 vs
  ArduSub **필터링된** ATTITUDE를 비교할 때의 잡음 바닥이고, 더 세게 흔들어도
  안 줄어든다(7.57 → 6.20). R의 실제 정확도는 **독립 2테이크가 0.81° 안에서
  일치**한다는 것이고, 그게 JSON의 `R_fit.repeatability`에 기록돼 있다.
  덤: 두 테이크 모두 축 순열에서 **40.7° / 41.3°** 떨어져 나왔다 — 카메라가
  실제로 ~40° 기울어져 있다는 독립적 방증(태그 쪽 `cam_tilt_deg`는 아직 미기입).
- **(역사) 처방이 도구가 된 경위**.
  `rov_gui/tools/calib_c3_imu.py`가 둘 다 잰다 —
  (a) accel scale/bias는 물 밖 텀블 ellipsoid 적합(`--fit accel`),
  (b) **IMU→기체** 회전은 C3 자이로 vs ArduSub ATTITUDE의 Kabsch 정렬(`--fit rotation`).
  (b)가 요점이다: **IMU→카메라 extrinsic이 아예 필요 없어진다**(장치에 없으니) 그리고
  카메라 틸트가 자동으로 흡수된다. 결과는 `config/c3_imu_calib.json`, sha1이 런
  meta에 박힌다. 합성 데이터 검증 완료(scale/bias 정확 복원, 회전 <0.5°, 지연
  <2 ms; `rov_gui/tests/test_imu_dr.py`), **실제 카메라로는 미실행**.
  도구는 커버리지가 모자란 텀블과 단일축 wiggle을 **거부**한다 — 자신 있게 틀린
  답을 내는 게 이 캘리브레이션의 진짜 실패 모드라서.
- **제대로 고치는 법**: 위 두 명령을 실기로 한 번 돌리고, 결과를
  `c3_collect.py`의 `calibration.json`에도 주입한 뒤 이 항목 삭제.

### C3 직결(`c3_camera/`): 컬러/depth 촬영시각 skew 약 1 프레임 — 하드웨어 동기 여부 미확인 (2026-07-29)
- **사실**: `c3_stream.py` 기본(`--pair-mode latest`)에서 컬러(CAM_A)와 depth(CAM_B/C 스테레오)의
  `getTimestamp()` 차이가 **약 66 ms(15 fps에서 ≈1 프레임 간격)** 로 관측된다. 컬러 카메라와
  mono 쌍이 같은 파이프라인 안에서 **하드웨어 동기(FSYNC)로 묶여 있지 않기 때문**으로 보인다.
- **영향**: 카메라나 장면이 움직이는 동안 RGB-D 정합이 최대 한 프레임만큼 어긋난다. 정지 상태
  파악에는 무해하지만, **움직이는 매니퓰레이션 데이터 수집·학습에는 실제로 문제가 될 수 있다**
  (66 ms × 0.2 m/s = 1.3 cm 어긋남).
- **임시 대응**: `--pair-mode timestamp` (+`--pair-tolerance-ms`)를 쓰면 두 스트림의 촬영시각이
  허용범위 안에 들어올 때만 bundle을 내보낸다. 대가는 최대 한 프레임 지연.
  `Bundle.skew_ms`가 매 프레임 실제 skew를 보고하므로 HUD에서 감시 가능.
- **제대로 고치는 법**: OAK-D-W-POE가 CAM_A와 mono 쌍의 **FSYNC 하드웨어 트리거를 지원하는지
  미확인**(이 보드의 FFC/FSYNC 배선 여부에 달림). 지원하면 파이프라인에서 동기를 켠다.
  아니면 depthai `Sync` 노드(디바이스측 정렬, 지연 증가) 또는 호스트측 보간을 검토.
  판정 전에는 이 항목 삭제하지 말 것.

### square 참조가 코너에서 동역학적으로 실현 불가 → 지울 수 없는 ~2 cm 코너 오차 바닥 (2026-07-24)
- **2026-08-16 hardware path mode 대응**: `rov_gui`의 `path_following: true`는 이제
  active-segment projection + corner gate를 쓰고, 각 꼭짓점에서 참조 속도를 0으로 만든 뒤
  실제 기체의 위치/속도/dwell capture를 확인해야 다음 변을 공개한다. 따라서 아래 문제는
  **legacy wall-clock trajectory mode와 simulator benchmark에는 계속 해당**하지만, 새 hardware
  geometric-path mode에서는 next-leg preview로 코너를 자르는 원인을 차단했다.
- **사실**: `square_setpoint`의 위치 경로는 각진 사각형이라 코너에서 참조 속도가 한 샘플
  (0.05 s) 만에 90° 뒤집힌다(|Δv|=0.212 m/s → 요구 가속도 사실상 무한). 게다가 yaw 참조는
  60°/s로 슬루(90°에 1.5 s)라 위치 참조와 **서로 모순** — 코너를 실제로 돌 때 필요한 선회율
  126°/s(횡력 8 N)~474°/s(30 N)가 슬루 상한 60°/s를 크게 초과. 힘이 무한해도 기수가 경로를  [UNVERIFIED: 산출물 없음 — docs/MEASUREMENT_AUDIT.md]
  못 따라간다. (슬루 자체는 `slew_heading` docstring에 의도된 설계로 명시돼 있음.)
- **정량**(dobmpc, gentle, **NONE 모드 = 외란 0**, lap 2–10 folded): 코너 2.00/1.92/2.04/2.39 cm
  vs 직선 0.22–0.28 cm(≈10×). **surge 박스 8→30 N에서 소수점까지 불변**(코너 명령 surge는  [UNVERIFIED: 산출물 없음 — docs/MEASUREMENT_AUDIT.md]
  평균 0.9–1.4 N으로 박스 근처도 안 감, 발동 0.7%) → **authority가 아니라 참조 기하 문제**.
  MPC는 preview+2차 비용으로 코너를 미리 돌아 안쪽으로 자르는(corner-cutting) **최적 절충**을
  하는 것이지 고장이 아님. 파랑 하에선 박스도 일부 기여하나 코너·직선을 거의 같은 비율로
  줄여(33% vs 37%) 코너 특유 효과가 아님; 박스가 실제 개선하는 건 코너 **직후 회복**.  [UNVERIFIED: 산출물 없음 — docs/MEASUREMENT_AUDIT.md]
- **영향**: DOB-MPC는 직선이 0.6–1.3 cm로 거의 완벽해서 이 코너 바닥이 오차 예산을 지배하고
  trajectory_compare 그림에서 유독 도드라진다(절대값으로는 3사 중 최소: gentle CDW 코너  [UNVERIFIED: 산출물 없음 — docs/MEASUREMENT_AUDIT.md]
  PID 24.6 / MPC 13.2 / DOB 3.8 cm). 컨트롤러 튜닝으로는 제거 불가.
- **줄이려면**: 참조 설계를 고칠 것 — 코너 필렛(원호/스플라인, 반경을 달성 가능 횡력과 yaw
  슬루율에 정합) 또는 코너 감속 프로파일, 또는 yaw 슬루 상한 상향. 어느 쪽이든 벤치마크
  정의가 바뀌므로 기존 기록과 비교 불가 → 사용자 판단 필요.

### heavy 회전 added mass = isotropic placeholder
- `[0.12, 0.12, 0.12]`는 임시값 — 문헌 근거 약함(von Benzon 30–100% 오차 보고,
  경쟁하는 0.40 세트 존재). 자체 system ID 전까지 HOLD.

### hydro는 MJX에서 안 돌아감 (`bluerov.xml` fixture로 확인)
- hydro가 CPU passive callback이라 MJX 미지원 — `verify/verify_gpu_mjx.py`의 bonus check가
  `bluerov.xml`(이제 검증 fixture) 로드로 non-gating 확인함. RL phase 전에 hydro의
  MJX 포팅 필요.

### C3-BR 마운트 브래킷 질량은 관성 합성에 미포함 (2026-07-19)
- heavy_gripper·heavy_c3의 브래킷(`meshes/c3_mount.stl`)은 **visual-only** — 재질/질량
  미상이라 `compute_payload_inertia.py` 합성에서 빠져 있음(카메라 1.7 kg 대비 수백 g 추정).
- 사용자에게 실물 브래킷 질량(또는 재질)을 받으면 C3처럼 합성에 추가할 것.

### Newton 그리퍼가 Onshape에 추가됐지만(2026-09-09 export) 시뮬 GRIP_POS/JAW_POS는 아직 추정값 (2026-07-20)
- 사용자 요청: Onshape 어셈블리에 있는 것(차체 + C3)만 반영. 그리퍼는 CAD 추가 전까지
  `heavy_c3`에서 제외. `heavy_gripper` 변종은 그리퍼가 추가될 때를 위한 config로 유지되나,
  현재 그 GRIP_POS=[0.25,0,−0.17]는 여전히 **추정값**(Onshape 미검증)이다.
- 그리퍼가 Onshape에 추가되면: export 재실행 → 브래킷처럼 실측 위치로 GRIP_POS 갱신 →
  heavy_gripper 재생성.
- **2026-09-09**: 그리퍼·upper mount·C3(40° 하향)가 들어간 mujoco 탭을 재export했다
  (`assets/CAD files/onshape_export_20260909/`, 파트 포즈→base_link는 payload_frames_20260909.json [유도]).
  CAD 턱 쌍 중심 x 0.364(팁 0.404), 하우징 x 0.127~0.316, z −0.127 → 시뮬 GRIP_POS 0.25/z −0.17보다 실린더가
  ~2.5 cm 앞·4 cm 위. **시뮬 변종·hw 설정은 아직 갱신하지 않았다**(사용자 결정 대기).
- **2026-09-08**: 실기 턱은 C3 렌즈 앞 0.196 m로 실측돼(항목 "기체 마커의 그리퍼 기하") 스테이션의
  렌즈-앵커 턱이 COM 앞 0.502가 됐다(`rov_shape.JAW_CENTRE_M`, `policy.tcp_body_flu_m`; 튜브도 같은
  +0.0855). 시뮬의 GRIP_POS 0.25 / JAW_POS 0.4165는 `cam_t_flu`가 맞다면 그보다 **8.5 cm 뒤**이고,
  `cam_t_flu`가 ~9 cm 앞으로 틀린 것이라면(항목 "C3 외부파라미터 cam_t_flu의 x/z가 의심된다") **대략
  맞다**. 플랜트 합성(합성 COM/관성)은 손대지 않았다 — 8.5 cm 이동이면 합성 COM은 수 mm 수준 [예측,
  미계산]. 줄자 측정으로 귀속이 정해지면 그때 GRIP_POS/JAW_POS를 갱신할 것.

### 방향 sweep이 seed-0 파랑 실현 하나를 공유 — worst-vertex 통계는 단일-실현 아티팩트
- 발견 2026-07-21 (`compare_20260720_230025` 코너 기하 분석): 모든 (current, wave) 헤딩쌍
  run이 **같은 seed-0 파랑 시계열**을 봄(wave-group **포락선**이 264.8 s마다 재귀 ≈ run
  길이 266.7 s) →
  특정 절대시각의 wave-group이 매 run 같은 lap/vertex를 때림(dobmpc 400 run 중 181개가
  t=200–210 s에 피크, worst vertex 66%가 V3). 방향 의존 결론은 **per-passage 상대각 통계**
  로만 뽑을 것; vertex별·시각별 주장은 multi-seed 재실행 전에는 출판 불가.
- **표현 정정 (2026-08-19)**: "실현이 264.8 s마다 **반복**된다"는 **틀렸다**. `waves.py:97`의
  `omega = linspace(omega_min, omega_max, N)`는 omega_i = omega_min + i·dOmega이고 omega_min이
  dOmega의 정수배가 아니라(gentle 0.2/0.0237288 = 59/7 = 8.4286), T = 2π/dOmega만큼 밀면 모든
  성분 위상이 **같은 상수 omega_min·T = 154.29°** 만큼 회전한다 → eta(t+T) ≈ −0.9·eta(t)
  (gentle 재현: corr −0.899, max|Δ| 0.583 m vs eta std ~0.13 m [유도: 이번 세션 재계산,
  저장 산출물 없음]). 정확한 반복은 7T = 1853.5 s. 재귀하는 것은 **군(group) 포락선**
  (|hilbert| corr ≈ 0.995)이고, 위 관측(같은 wave-group이 같은 vertex를 때림)은 그대로
  유효하다. 인용할 때 "파랑이 반복된다"고 쓰지 말 것.
- 발견 2026-07-19 (C3 위치 정합 중): 스킨 bbox = 벤더 치수 × 1.0233 (세 축 균일).
- C3/페이로드 배치는 **실측 metric**(COM 앵커) 기준이라 동역학·카메라는 정확하지만,
  렌더에서 페이로드가 스킨 대비 ~3–5 mm 어긋나 보일 수 있음(코스메틱).

### DP 프레임 감사에서 나온 스테일/오류 인용 4건((2)(5)(6)(7)) + COM 43.4 mm 격차 (2026-09-03; (1)은 2026-09-08 해소)
- 발견: 6-리더 프레임 감사(`.claude/journal/research.md` 2026-09-03 항목). 프레임 체인 자체는
  건전하다(정책=TCP, `policy_frames.py:578`이 레버암을 빼서 NMPC엔 body 원점 궤적). 문제는
  **주변 문서·주석이 폐기된 값을 계속 들고 있다**는 것.
- **(2) "body 원점 = heavy_c3 COM"(`geometry.py:331` 주석)은 틀렸다.** 실제는 **맨몸 heavy 차체
  COM**(`bluerov2_mujoco_marinegym/compute_payload_inertia.py:45-46` `COM_VEH = np.zeros(3)`).
  heavy_c3는 자기 합성 COM으로 재원점하므로 32.5 mm 다르다.
- **(3) NMPC가 규제하는 점과 플랜트 모델의 COM이 43.4 mm 어긋나 있다.** 스테이션 eta는
  맨몸 차체 COM(base_link) 포즈인데, NMPC가 적분하는 플랜트는 `heavy_gripper`
  (`config/hw_mpc.yaml:7`)이고 그 MJCF는 **합성 COM에 원점을 재설정**한다
  (`compute_payload_inertia.py:118-122`). 격차 (0.03489, 0.00099, −0.02579) m [유도].
  `compute_payload_inertia.py:118-122`가 스스로 "원점을 COM에서 떼면 모델되지 않은 회전-병진
  커플링이 NMPC를 불안정하게 만든다"고 경고한 바로 그 종류의 오차인데, **어떤 config 키도
  meta 필드도 이 경계를 기록하지 않는다**. 임시 대응: 없음(실기 영향 미측정). 제대로 고치는
  법: 의도적인지 먼저 판정 → 의도라면 meta에 기록, 아니라면 eta에 시프트를 적용하거나
  플랜트를 base_link 원점으로 재생성.
- **(5) obs 두 행 간격 "66.7 ms"는 런타임에서 거짓.** 실제 간격은 depth 페어 간격
  (`rov_gui/backends/policy.py:684` `spacing = pair_dt if how != "dup" else obs_dt`,
  ~138 ms 실측). 66.7 ms는 **학습 스트라이드**(down_sample_steps 2 / fps 30). 학습-배포
  모션큐 시간척도 불일치이므로, 문구 정정과 별개로 실제 영향 평가가 필요하다.
- **(6) 배포 TCP: x의 순환은 2026-09-08에 끊겼고, z의 순환은 남아 있다.** `tcp_body_flu_m`은 이제
  `cam_t_flu` + [0.196, 0, −0.275]이고 그 x는 CAD 체인과 독립적으로 실측됐다(턱에 문 병을 태그 58 중심에
  놓은 프레임에서 렌즈 앞 0.196 m [측정: data/20260908/0908_180453_observe/policy_obs/rgb/000160.jpg
  + 0908_170428/policy_obs/rgb/000000.jpg + 0908_175151/policy_obs/rgb/000000.jpg]) — CAD의 110.664 mm는
  반증됐다. **남은 순환은 z**: "렌즈가 턱 위 0.275 m"는 여전히 `cam_t_flu`를 유도한 그 CAD 수치이고
  (이미지가 ~1 cm 안에서 확인하지만 독립 실측은 아님), 스테레오 삼각측량은 여전히 0건이다. 게다가
  `cam_t_flu` 자체의 x/z가 의심된다(별도 항목) — 턱은 렌즈에 앵커돼 TCP는 어느 쪽이든 맞지만 body 원점은
  아니다. 제대로 고치는 법: 턱을 C3 프레임에서 스테레오 삼각측량해 `tcp_offset_cam_m`에 직접 넣을 것
  (렌즈 위치 소거).
- **(7) C3 마운트의 roll/yaw가 리마운트 이후 미재측정.** `hw_nav.yaml`의 `cam_xyaxes_flu`는
  주석 처리되어 `c3_payload_frames.json` 값이 그대로 산다(피치만 43.3°로 갱신). `R_bt = R_bc`
  이므로 roll/yaw 오차는 **정책 액션 프레임 전체를 회전**시킨다. 게다가 렌즈 위치는 2026-09-02
  CAD, 틸트는 2026-08-17 측정 — 두 반쪽이 서로 다른 마운트 시점에 고정되어 있다.

### `test_object_nav.test_the_shipped_config_resolves`가 배포 config에서 실패 (2026-09-07 발견)
- 테스트는 `object_nav.max_distance_m ≤ 3.0`을 단언하는데 `config/hw_mpc.yaml`은 **10.0**이라
  35/36으로 떨어진다(`min_distance_m 0.15 < max 10.0 ≤ 3.0` 단언). 헤딩 점선 작업 중
  `_run_meta` 호출 스위트를 돌리다 발견했고 그 변경과 무관(config diff에 distance 키 없음 =
  이미 커밋된 값). 테스트 상한을 풀 것인지 config를 3 m로 되돌릴 것인지는 사용자 결정 —
  10 m는 풀 대각선보다 길어 사실상 상한 없음이다.

### RGB를 먹는 정책 체크포인트(`umi_rgbd_5d`, RGB-only 포함)는 **평가할 도구가 없다** — 스테이션·리플레이·held-out 셋 다 depth 전용 (2026-09-23, 2026-10-01 갱신)
- **무엇**: 2026-09-23 ablation의 two-stream 모델(obs `camera0_rgb` + `camera0_depth`, task
  `external/UMI_aquatic/diffusion_policy/config/task/umi_rgbd_5d.yaml`)은 학습만 된다. 세 도구가 모두 `camera0_depth`만 먹인다
  [코드 읽기, 미실행 — 2026-10-01]:
  - 스테이션: `rov_gui/backends/policy.py:85` `IMAGE_KEY = "camera0_depth"`, `policy_obs.py`는 depth 레시피만 만든다.
    RGB+D ckpt는 로드에서 `ValueError`("exactly one image key", `dp_policy.py:586-603`)로 죽는다. **RGB-only ckpt(`camera0_rgb`
    키 1개)는 더 나쁘다** — warm-up이 contract의 image_key 자리에 0을 넣어 통과하므로 READY가 되고(`dp_policy.py:983-995`),
    그 뒤 매 tick `_check_obs`의 KeyError "obs is missing 'camera0_rgb'"(`:1019-1026`)가 `_infer_error`로 넘어가 플랜 0개·DEGRADED
    (`backends/policy.py:1618-1622, 1664-1669`). 패널 ckpt 교체(`set_ckpt`)에도 이미지 키 검사가 없다.
  - 오프라인 리플레이: `rov_gui/tools/policy_session_replay.py:273`이 `{"camera0_depth": img}` 고정. 기록된 `policy_obs/rgb/*.jpg`는
    C3 컬러 원본 640x360 JPEG q85이지 224 obs가 아니고, `cv2.imread`는 BGR로 돌려준다.
  - 육상 held-out: `rov_gui/tools/dp_policy_offline.py:385, 1111, 1173`도 `camera0_depth`만 읽는다.
- **학습 RGB의 정체**: OAK-D-W 컬러 CAM_A `rgb.mp4`(1280x720) 센터 크롭 224, 정류·warp 없음
  (`data collection/UMI_Underwater/demonstration_processing/build_dataset.py:83-92,146-150`). 소스 store는 **BGR**이고
  병합 store(`slam/grasp_9_9_26/dataset_rgbd.zarr.zip` — 옛 이름 `slam/9_9_26/`, 아래 🐛 경로 항목)는 RGB로 뒤집어 저장했다
  [측정: `umi_handheld/build_dp_rgbd_zarr.py` channel-order proof, stored-vs-BGR 0.0000 / stored-vs-RGB 10.2925]. depth와
  **픽셀 정렬이 아니다**(컬러 카메라 vs rectified-left 격자를 C3 광학으로 warp) — two-stream이라 학습엔 무관하지만 4채널
  early fusion에는 이 데이터를 쓰면 안 된다.
- **광학 차이의 크기 [유도, 2026-10-01]**: 학습 RGB 224는 육상 CAM_A(OV9782, 공기 중; `grasp_9_9_26/calibration.json` socket 0
  fx 573.08 @1280x800, 14계수)로 약 72°×72°이고, 같은 레시피로 자른 C3 컬러(IMX378, 수중 EEPROM fx 3080.35 @3840 → 640x360에서
  513.4; `calib/FOV_AUDIT.md:57`)는 약 38.6°×38.6°다 → 화면 중심 배율 약 1.8배(같은 거리 물체가 수중 obs에서 1.8배 크다),
  육상 크롭 가장자리 배럴 13.5 %. 학습 증강(crop 0.95, ±5°)으로 못 덮는다. `umi_handheld/warp.py` `WarpStage`는 depth 없이
  카메라모델끼리 remap하므로(`:74-84`) RGB도 C3 CAM_A로 warp할 수 있다 — 단 source는 1280x720 CAM_A 모델(fx 그대로, cy−40;
  `CameraModel.K`의 해상도 스케일 경로를 타면 fy가 10 % 틀어진다), target C3 CAM_A yaml은 레포에 없어 런 meta
  (`data/20260908/0908_180453_observe/mpc_180453.meta.json:707-753`)에서 만들어야 한다. warp하면 핸드헬드 손끝(광축 아래 약 22°)이
  VFOV ±19.3° 밖으로 빠지고 ROV 턱(약 11.5°)은 보이는 비대칭이 생긴다 [유도].
- **제대로 고치는 법**: (1) `policy_obs.py`에 C3 CAM_A 컬러 → (학습과 같은 warp 여부) 센터 크롭 224 RGB 빌더, BGR→RGB,
  (2) `backends/policy.py`의 `IMAGE_KEY`를 ckpt contract의 image_key로 바꾸고 로드 시점에 이미지 키를 검사, (3) `dp_policy.py`의
  `eval_transforms`를 키별로 적용(이미 `key_transforms`는 무시하도록 해둠), (4) 리플레이·held-out 도구에 `camera0_rgb` 경로,
  (5) 광학 보정 여부(naive / C3 warp)를 run meta에 남길 것. 그 전까지 RGB ckpt는 wandb val(EMA 가중치·배치 1개)밖에 볼 수 없다.

---
*마지막 갱신: 2026-10-01*

### `demo_e2e.py dobmpc policy`가 `skip_fix_lag`로 플랜 0건 — 데모 프로세스에서 dobmpc tick이 fix보다 70–140 ms 늦다 (2026-09-26)
- **증상**: `rov_gui/tests/demo_e2e.py dobmpc policy`(docstring 기본 추종자)는 PolicyWorker가 12/12 쌍을
  `skip_fix_lag`로 버려 "no PolicyPlan ever reached the bus"로 끝난다. 같은 드라이버의 `mpc policy`·`mpc station`·`pid policy`는 통과.
- **측정**: 임시 로그(복구 완료)로 본 `fix_lag_s` = depth 스탬프 − 최신 proprio 행: dobmpc 0.068–0.141 s (전부 > FIX_LAG_TOL·obs_dt = 33 ms),
  mpc 0.000–0.067 s (`scratchpad/demo_dobmpc_diag.txt`, `demo_mpc_diag.txt`, 2026-09-26; 세션 스크래치패드라 재현은 같은 임시 로그로).
  즉 게이트는 설계대로 동작하고, 데모 단일 프로세스(오프스크린 Qt + acados dobmpc + EAOB + 합성 플랜트)에서 MpcWorker tick이
  fix 스탬프를 놓치는 것이 원인 [유도]. 6-DoF 변형과 무관(변형 OFF, 4-DoF 경로).
- **경계**: 2026-09-11 저널에 이미 "`demo_e2e.py policy` 녹색 베이스라인 없음"으로 기록된 행. 실기 dobmpc 정책 런에는 다른 프로세스 배치라
  그대로 옮겨 읽지 말 것.
- **후속**: dobmpc tick 소요를 데모에서 재고(EAOB 첫 tick 비용 / 플랜트 RK4), 또는 demo_e2e의 policy 행 기본 추종자를 mpc로 고정.

### `demo_e2e.py dobmpc square`가 이 작업트리에서 red — 설정의 `plant.linear_damping` 덮어쓰기 vs 데모 플랜트, 그리고 드라이버 예산 (2026-09-26)
- **증상**: `rov_gui/tests/demo_e2e.py dobmpc square` → `[error] ctrl[dobmpc]: w_hat has been PINNED to its clip for 3 s (X=15, Y=45)`
  → `FAIL: START did not reach the square. last: approaching the start point (0.37 m)`; FAIL 뒤 드라이버가 disengage 없이
  `app.quit()`하면서 "Qt has caught an exception thrown from an event handler … terminate called without an active exception"으로
  abort(exit 134). 같은 드라이버의 `mpc station`·`mpc policy`·`pid policy`·`mpc replay`·`mpc policy rp`는 통과.
- **원인 [유도]**: `config/hw_mpc.yaml plant.linear_damping [86.7, 133.8, …]`(2026-09-08 실기값, 모델의 21.5×)는 **제어기 모델**에만
  들어가고 데모 플랜트(`rov_gui/control/plant.py`)는 marinegym 원본 항력(4.03)으로 돈다 → EAOB가 그 차이를 외란으로 추정해
  접근 중 클립에 붙는다. 덮어쓰기를 뺀 임시 설정으로 다시 돌리면 접근은 되지만 10 s 정착이 드라이버의 20 s 예산을 넘겨
  `settling over the start point (5 s)`에서 FAIL(`demo_e2e.py line`의 예산 결함과 같은 부류). 6-DoF 변형과 무관(변형 OFF, dobmpc 제어 출력은
  `test_attitude_axes`/`test_control`의 바이트 동일 검사 대상).
- **경계**: 2026-08-14 이후 dobmpc square 데모 행이 통과한 기록은 저널에 없다(마지막 녹색은 damping 덮어쓰기 이전). 실기 dobmpc 런에 옮겨 읽지 말 것.
- **후속**: 데모 플랜트가 `plant.linear_damping` 덮어쓰기를 같이 받게 하거나(데모의 목적이 "제어기 모델 = 플랜트"라면) 드라이버 예산을
  SETTLE_S만큼 늘리고, FAIL 분기에서 `cmd_mpc_engage(False)`를 보낸 뒤 quit하도록 고칠 것.
