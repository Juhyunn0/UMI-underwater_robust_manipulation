# Diffusion Policy 궤적 생성 — 계획 (v0.2, 2026-08-30)

> **상태: 결정 반영 초안.** D1–D5 + 시연 소스 + gripper + 진행 순서 + UI 통합 방식이
> 2026-08-30 사용자 확정됨(§1). 영어판 전환은 사용자 신호 대기.
> **구현 현황 (2026-08-30)**: M0 크리티컬 패스 코드 **랜딩** — `umi_handheld/extract_pose.py`
> (A0), `rov_gui/control/plan_stream.py`(B1), shape `replay`(B2, one-shot M0a + streamed
> M0b), gripper 실행 경로(B5, 기본 OFF), plans.jsonl + meta schema 8(B4), B3 실패모드
> 테스트 일부(test_replay 9/9 + test_plan_stream 19/19), demo_e2e 실솔버 완주.
> safety-code-reviewer 감사 FAIL→전항 수정(딜맨 held-drive 해제, 발산 가드 등 —
> .claude/journal/reviews.md 2026-08-30). **남은 M0 전제 = 태그 배열 앞 시연 1개 촬영**
> (§7 첫 항목). PolicyWorker(라이브 추론)와 Track A1 이후는 미착수.
> 조사 근거: 2026-08-30 워크플로(코드 리더 3 + UMI_underwater 레포 검증 + advisor 2)
> + 탐색 2건(handheld pose-라벨 경로 / gripper 명령 경로).
> 수치 표기 규칙은 repo CLAUDE.md의 측정 인용 규칙 — 경로 없는 수치는 `[예측]`/`[스펙]`/`[유도]`.

## 0. 한 줄 요약

**UMI handheld 시연으로 학습한 diffusion policy**를 GUI trajectory 패널의 새 미션으로
추가한다: obs = (depth 영상, proprioception) → 1 Hz로 미래 4 s 궤적 + gripper
(action 80-dim) 생성 → PlanFilter/Stitcher를 거쳐 기존 NMPC(`HwDobMpc`)가 20 Hz 추종,
gripper는 기존 MANUAL_CONTROL 버튼 경로로 전송. 1차 태스크: **바위 접근 + 파지**.

진행 순서(확정): **① 데모 1개 → 라벨 추출 → 실기 replay 게이트(M0) → ② 대량 수집(100–150)
→ ③ 학습 → ④ 라이브 policy 통합**. M0가 성공해야 ②에 들어간다 — replay는 학습이 없어
데모 1개면 충분하고, 그 하나로 라벨 복원·좌표 변환·feasibility·MPC 추종·gripper 실행이
전부 검증된다(일반화만 검증 안 됨 — 그건 policy의 몫).

구조의 본질은 **UMI-on-Air 패턴**: policy는 embodiment-무관한 기하(궤적)만 내고, 기체
dynamics/제약은 MPC follower가 담당. handheld(비히클 아닌 장치) 시연이 유효해지는 근거
자체가 이 구조다. **단, 차용하는 것은 이 역할 분담뿐 — policy 안에 MPC는 없다** (순수
생성 모델 → 사후 PlanFilter → NMPC의 직렬 분리). UMI-on-Air의 embodiment-aware
guidance(denoising 중 controller cost로 샘플 조향)는 1차 설계 미채택 — 실기에서 필터
기각/클립이 잦으면 v2 업그레이드 후보. UMI-U 논문이 Limitations에서 "PID를 dynamics-aware controller로 바꾸면
overshoot·target-switch 실패가 줄 것"이라 쓴 그 문장을 실행하는 위치. 논문화 시 baseline:
(a) UMI-U식 velocity-command DP, (b) 기하 미션 + NMPC(현행), (c) 제안: trajectory-DP + NMPC.

## 1. 확정 결정 로그 (2026-08-30, 사용자)

| # | 결정 | 내용 | 유예/수반되는 리스크 |
|---|---|---|---|
| D1 | Goal 채널 **제외** | 순수 depth+proprio로 시작. SAM2는 저장 영상 오프라인 처리로 나중에 가능 | 단일-바위 장면이라 target confusion 완화되나, multi-object 확장 시 **재점화** (UMI-U Table II: goal-agnostic는 background shift 0%) — v2 재고 항목 |
| D2 | 태스크 = **바위 접근+파지**, 태스크당 모델 1개 | 파지까지 포함 → gripper 출력 필요(G 참조) | grasp는 접촉 태스크 — 추종 오차와 g 문턱의 상호작용 (§6-②) |
| D3 | **Yaw 학습 포함** | Δψ per knot (unwrapped 스칼라) | yaw 자유도는 태그 가시성과 충돌 가능 — 시연 때 "태그 보는 heading" 습관 필요 (memory: station-tag-dropout-bridge) |
| D4 | 시연 = **UMI handheld 리그** | teleop/gantry 제외 | handheld↔ROV 도메인 갭 (§6-①); pose-라벨 파이프라인 신규 구축 필요 (§5-A0) |
| D5 | **w_hat 제외** | proprio 단순화 | 나중에 추가하면 데이터셋 경계 신설 + 재학습 |
| G | Gripper = **policy action에 포함** | knot마다 g 스칼라 1채널, 실행 시 문턱 처리(UMI-U 방식) | 실기 gripper는 피드백 전무 → open-loop (§5-B5) |
| M0 | **Replay 우선** | 데모 1개 replay 게이트 통과 후 대량 수집 | — (리스크를 줄이는 결정) |
| UI | **Trajectory 패널 SHAPES 엔트리로 통합** | `replay`/`policy` 두 엔트리, follow 패턴 재사용 | — (빼고 넣기 쉬움; `trajectory.kind`가 기록 경계를 공짜로 만듦) |

## 2. UMI-U 검증 스펙 (참조)

UMI_underwater 레포(커밋 fb99783) + RSS 2026 논문에서 검증한 사실 — 우리 설계의 준거:

- Policy: Chi et al. diffusion_policy 계열 `DiffusionUnetTimmPolicy` + `TimmObsEncoder`
  (CLIP ViT-B/16). Obs horizon 2, action horizon 16, 실행 8. **DDIM inference 16 steps**,
  denoiser는 1D conditional UNet.
- Action: body-frame **velocity command** 6-dim + gripper 2-dim 문턱(>0.40, 4 s
  auto-neutral). 실행 10 Hz, 추론 ~1 Hz (비동기 receding horizon + 시간 정렬 skip).
- 학습 데이터: 자기지도 수중 수집 233 성공 에피소드(policy), on-land 800 demos(affordance).
- proprio는 코드상 [compass, pitch, roll]/360 — 논문의 "vehicle depth"는 부정확(코드가 권위).

즉 "1 Hz 추론 + 청크 실행" 시간 구조는 UMI-U와 동일. 바뀌는 것은 action의 의미
(velocity → 궤적+g)와 replan 사이 루프를 닫는 주체(ZOH → 20 Hz NMPC)다.

## 3. 런타임 아키텍처

```
C3VideoWorker (기존)                               MpcWorker (기존, 50 ms tick)
  color 640x360@30 ──► nav_mb ──► TagNav (기존 PnP)      ▲            │
  depth 640x360@20 ─┐                                    │ set_reference_traj(fn)
  (×0.64 보정 1회,   ├─► policy_mb (RgbdMailbox 3번째 탭) │            │ g 문턱
   hardware.py:504) ─┘        │                          │            ▼
                              ▼                          │   bus.cmd_gripper_drive
                    PolicyWorker (신규, TimerWorker, 1 Hz)│   (기존 경로, §5-B5)
                      obs 조립: depth + proprio           │
                      DDIM 8–16 steps (rovgui-pose env,  │
                      RTX 5090, 예상 15–40 ms [예측])     │
                              │ raw plan (16×0.25 s + g) │
                              ▼                          │
                    PlanFilter → PlanStitcher (신규, 순수 numpy)
                      게이트/클립/기각 + blend — 1 Hz 이음새를
                      컨트롤러가 못 보게 만듦.  ◄── ReplaySource(파일)도 같은 입구
```

핵심 설계 결정:

- **패널 통합**: `SHAPES`(rov_gui/control/geometry.py:50)에 `replay`·`policy` 엔트리 추가.
  follow가 스트리밍 레퍼런스의 선례(workers.py:1941-2226)라 arm/refusal, START hold,
  미션 해제 4지점, meta `trajectory.kind` 기록을 전부 재사용. 컨트롤러 선택과 직교.
  **replay와 policy가 같은 설치 seam을 공유** — M0가 통합 지점을 미리 검증한다.
- **Follower는 MPCC가 아니라 NMPC(`HwDobMpc`)**. DP 출력은 시간 파라미터화 궤적(타이밍=
  학습된 속도)인데 MPCC는 spatial path + 자체 progress라 타이밍을 폐기하고, 1 Hz 스트림에선
  끝단 tangent-ray 외삽(path_geometry.py:131-162)이 호라이즌 절반을 지배. MPCC에는
  `follow_ok` 패턴으로 거부(fail-closed).
- **플러그인 지점**: 스티치 버퍼 위 `fn(ts) -> (p, yaw, v, r)` 샘플러를
  `HwDobMpc.set_reference_traj(fn)` (mpc_bridge.py:305-308)에 — 최소 침습.
  pid/*_tuned 호환은 `NedPlan` + `set_path_plan_ned` (mpc_bridge.py:404-418) 경로.
- **추론은 GUI 프로세스 안**: rovgui-pose env(torch 2.11+cu128, RTX 5090 — 실행 확인)에서
  이미 SAM2+FoundationPose가 돔. PoseWorker 패턴 복사(TimerWorker + RgbdMailbox 탭,
  want-게이트로 소비자 꺼짐 시 복사 비용 0, hardware.py:562-588; torch import는 setup()에서;
  종료는 join 프로토콜 session.py:499-529).
- **호라이즌 꼬리는 종점 v=0 홀드**로 클램프 (등속 외삽/tangent ray 금지).

## 4. 정책 스펙

```
Obs (horizon 2 frames, 0.25–0.5 s 간격):
  depth   : uint16 mm → ×0.64 보정 → inverse-depth 정규화 [0,1] (near=large,
            umi_handheld/build_zarr.py:59-69 재사용) → 리사이즈(예: 224²)
            + validity mask 채널 (invalid=0 ≠ far 혼동 방지)
            [도메인 브리지: 학습 데이터는 in-air OAK-D depth를 warp.py로 C3 수중
             optics에 재투영 — warp는 left+depth 둘 다 처리 (warp.py:177-189)]
  proprio : [z, roll, pitch, u,v,w,r (body)] — handheld에서는 pose 트랙 유도값
            (자세=PnP, 속도=유한차분), 배포에서는 state assembler 출력.
            **양쪽 의미 일치가 규약** — 절대 yaw·절대 xy는 제외(프레임 불변성).
            per-dim dataset stats로 [-1,1].

Action = 미래 궤적 + gripper, body-frame-at-t0 상대 표현:
  16 knots × 0.25 s = 4.0 s  →  (Δx, Δy, Δz, Δψ, g) × 16 = 80-dim
  4 s인 이유: replan 1 s + MPC preview 3 s (MPC_N=60 × DT_CTRL=0.05 s,
  dobmpc/params.py:103,111)를 항상 커버 — 짧으면 stale plan 꼬리를 추종.
  Δψ = t0 기준 unwrapped 스칼라 (4 s 내 |Δψ|<π 전제).
  g ∈ [0,1] — 라벨은 기존 gripper-width 추출값(extract_gripper_width.py, [0,1] 0=closed),
  실행은 문턱 처리(§5-B5). 61-stage plan은 cubic spline 리샘플, v/r은 spline 미분.

모델:
  encoder  : ResNet-18 + GroupNorm + spatial softmax (~11 M) 1순위 — depth 입력에
             CLIP ViT 사전학습 이득 거의 없음; "유사 구조"는 obs 구성·global-cond
             fusion 계승으로 충족.
  head     : 1D CNN UNet (down_dims 256/512/1024). DiT는 데이터 수천 에피소드 때 재고.
  diffusion: train DDPM 50–100, inference DDIM 8–16 steps — UMI-U가 같은 RTX 5090에서
             더 무거운 스택(Depth-Anything+affordance+DP)을 1 Hz로 돌림 [스펙: 논문].
  학습     : batch 64–128, ~150 epochs, EMA, bf16 (UMI-U 공개 config 준용).
```

**절대 NED waypoint 금지** (tag-map 원점 종속 + SLAM bias가 라벨에 박힘).
**velocity 적분 차선** (드리프트; plan 인터페이스가 위치 요구).

## 5. 작업 트랙

Track A(데이터/학습)와 Track B(제어/통합)는 병렬 진행, **M0에서 1차 합류**한다.

### M0 — 단일 데모 replay 게이트 (최우선 마일스톤)

순서: 새 프로토콜로 데모 **1개** 촬영 → A0 라벨 추출 → 실기 replay → 성공 시 대량 수집.

- **M0(a) one-shot replay**: 추출한 pose 궤적(+g 타이밍)으로 시간 파라미터화 샘플러를
  만들어 `set_reference_traj`에 한 번 설치(square와 같은 경로) — 최소 코드로
  라벨 복원·좌표 변환·feasibility·MPC 추종·gripper 실행을 검증.
- **M0(b) streamed replay**: 같은 데모를 4 s 창으로 잘라 1 Hz로 흘려 PlanFilter/Stitcher
  경로 검증 — 실데이터 mock planner로서 Track B의 통합 시험을 겸함.
- 통과 기준(제안): 궤적 추종 RMS가 기존 기하 미션 수준, gripper가 의도 시점 ±1 s [예측]
  내 작동, 필터 기각 0건.

### Track A — 데이터 → 학습

**A0. Handheld pose-라벨 파이프라인** (`umi_handheld/extract_pose.py` 신설 — 현재 없음)
글루 ~40줄이면 됨. 4개 부품이 이미 존재:
1. `CameraModel.K(w,h)` (umi_handheld/camera_model.py:104-109) — rectified-left는 왜곡 0
   pinhole (configs/source_camera_air.yaml:37-46) → undistort 불필요, `dist=None`.
2. `TagDetector` + `TagNav.solve(dets, K, None)` (rov_gui/control/tagnav.py:162-223,
   421-426) — numpy+cv2만, gtsam 불요. 오프라인 폴더 replay 선례 =
   src/gantry_map_pose.py + src/tests/test_gantry_map_pose.py:213-244
   (같은 코드 실기록 replay 정확도 p50 9–16 mm, README_fisheye_gantry.md:672-682).
3. **카메라→body 변환은 TagNav에 C3 extrinsic을 주면 한 번에**:
   `NavConfig.R_t_frd_cam("main")` (rov_gui/control/geometry.py:213-227;
   c3_payload_frames.json cam_center_bl/cam_xyaxes; cam_tilt_deg 43.3 실측 —
   record boundary, config/hw_nav.yaml:149-158).
4. zarr 확장 = `gripper_width.npy` 사이드카 패턴 복사 (build_zarr.py:120-126, 157-161)
   → `data/pose`·`data/action` 배열 + session.json에 tag-map/스케일 provenance 필드.

**함정 (필수 규칙)**:
- **태그 id 충돌**: gripper 태그 = tag36h11 **id 0/1** (21.37 mm)인데 tag_map_full.yaml에
  id 0/1(170 mm)이 존재 → 조용한 ~8× 스케일 폭발. **tag_map.yaml(47태그, 0/1 없음) 사용
  + pose 계산에서 id 0/1 명시 제외.**
- PnP는 **pre-warp** left 프레임에서 (extract_gripper_width.py:1-6과 같은 규칙);
  warp는 obs 도메인 브리지로만.
- handheld에 IMU 없음 → single-tag IPPE 모호성 중재 불가 (tagnav.py:537-557) →
  **min_tags≥2** 강제.
- **기존 demo 25개는 궤적 라벨 불가 추정** (환경 태그 가시 여부 미기록; 0001–0020은
  태그 패밀리도 DICT_4X4_50) → obs-전용/예비로 강등, 라벨용은 새 촬영만.

**A1. 대량 시연 수집** (M0 통과 후; D2 태스크 = 바위 접근+파지)
- 목표 **100–150 에피소드** [스펙: DP 계열 실기 50–200 표준, UMI-U 233]. 수량보다 다양성:
  시작 pose/거리/접근각/바위 배치.
- 프로토콜: 환경 태그 배열 상시 in-view(프레임당 ≥2), 태그 패밀리·실측 크기·사용 태그맵을
  session.json에 기록, C3 mount 자세(틸트) 모사, 파지 동작 포함(gripper-width 라벨은
  기존 추출기), 정지 head/tail 최소화.
- 제외 규칙: 태그 검출 불량 구간, 정지 구간 (안 자르면 hover를 배움).

**A2. 데이터셋 빌드**
- Slicing: obs 시각 t를 0.25–0.5 s stride로 sliding window,
  `label = pose over (t, t+4 s], 0.25 s 리샘플, body-frame-at-t 변환 + g`.
- 라벨은 **복원된 EXECUTED 궤적** — 리샘플 전 smoothing spline/SavGol 평활
  (안 하면 MPC가 PnP jitter를 성실히 추종).
- depth 규약 단일화: 학습(in-air→warp) / 배포(C3 ×0.64) 각각의 파이프라인과 경계 필드를
  dataset meta에 기록. 0.64는 1.0–1.1 m 근방만 정확한 스칼라 stopgap(오차 거리 종속,
  hardware.py:118-126) — 수중 재캘리브 시 데이터셋 경계 재설정 + 재학습.
- **replay overlay 단위시험 (학습 전 게이트)**: 로그된 obs → GT 라벨 궤적 디코딩 →
  실제 미래 궤적 위 겹침 top-down plot 자동 검증. 좌표계가 셋인 레포에서 부호 하나
  틀려도 학습은 잘 되고 실기에서 벽으로 간다 (NED↔FLU S-mirror, r_ned=−r_flu
  mpc_bridge.py:365, body-at-t0 변환).

**A3. 학습 + 오프라인 평가**
- env: rovgui-pose(torch 있음) 또는 별도 학습 env — robust에는 torch 없음.
- upstream diffusion_policy 코드베이스 재사용 여부는 A2 스키마 확정 시 결정.
- held-out 에피소드 open-loop 궤적 오차 + M0(b) 인프라로 오프라인 closed-loop 예행.

### Track B — 제어/통합 (모델 없이 진행 가능; mock planner + M0(b)가 시험대)

**B1. PlanFilter + PlanStitcher** (순수 numpy, 솔버 임포트 금지 — path_cost.py와 같은 규율)

Filter, 실행 순서대로:
1. 스키마/유한값/시각 단조.
2. 신선도: obs 시각 대비 도착 지연 > 0.7 s [예측] → 시간 원점 시프트 또는 기각.
3. **앵커 게이트**: |p_plan(t_now) − r_now| ≤ 0.3 m [예측]; 겹침 창 전체에서
   max|p_new−p_cur| ≤ 0.2 m [예측] ("가깝게 시작해 빠르게 발산" 구멍 방지).
4. 운동학: |v| ≤ v_max(실측 달성속도 기준 — **측정 필요**, U_MAX 아님), |a|,
   unwrap 후 |dψ/dt|. 경미 위반은 시간 팽창 t'=α·t로 감속(기하 보존), 크면 기각.
5. **작업공간 박스**: 지오펜스 완전 제거된 현재(memory: rov-gui-run-folder)
   **이 필터가 유일한 위치 기반 방호** — 필수.
6. 연속 기각 에스컬레이션: 1회 → 현행 plan 유지; 소진 → 종점 DP hold;
   연속 3회 [예측] → DP hold + 오퍼레이터 경보.
7. **bridge(태그 dropout) 중 도착 plan 수락 금지** — frozen/DR 상태에 조건화된 출력은
   환각. 복귀 후 첫 plan은 앵커 게이트 필수.
모든 문턱에 히스테리시스 (경계선 planner의 1 Hz track↔hold 토글 = 가진).

Stitcher — **commit-then-blend**:
```
p_ref(t) = (1-w(t))·p_old(t) + w(t)·p_new(t),  w: cosine ramp, T_blend 0.3–0.5 s
v_ref    = (1-w)·v_old + w·v_new + w_dot·(p_new − p_old)   ← 교차항 필수
```
- **새 plan은 측정 상태가 아니라 현재 레퍼런스 r(t_now)에 앵커** — 측정 앵커는 1 Hz마다
  추적오차 리셋 → EAOB/적분기 무효화 + leash·데드밴드와 결합한 래칫 드리프트.
  예외: |r−x_meas|가 leash(0.10 m) 초과 포화 중이면 leash 가장자리에 앵커
  (0818 런에서 leash 포화 = engaged tick의 83.6 % — path_cost.py 헤더,
  sessions/low_level_controller_data/20260818/0818_143802/mpc_143938.csv).
- ACT식 temporal ensembling 기각 — diffusion 멀티모달 출력에서 모드 간 평균
  (좌회피/우회피 평균 = 정면 돌진). 옵션: mode-consistency(직전 plan과 최근접 후보).
- yaw는 unwrap 후 블렌드.

**B2. GUI/워커 배선**
- `SHAPES`에 `replay`·`policy` 추가 (geometry.py:50 + MpcConfig 검증 + 패널
  `_shape_changed`/`_emit_scenario` 분기 + plot 분기).
- bus에 `policy_traj` Signal + backend 배선 (hardware.py:3148-3153 / demo.py:750-755 옆).
- MpcWorker: `on_policy_traj` slot (on_pose 패턴 — latest-wins, tick에서 적분),
  mission state는 해제 4지점 모두에서 clear, `_policy_refusal()`(사유별 한 문장),
  `_tick_policy` (tick 내 _tick_follow 옆).
- stale 스트림 (> ~1.5 s): 현행 plan freeze + 종점 zero-FF → station 강등
  (follow의 STALE/`_follow_to_station` 패턴, workers.py:2079-2275).

**B3. Mock planner 오프라인/HIL 검증** — 실패 모드 테스트 (demo 백엔드 + test_control 스타일):
1. 거울상 plan 교대(1 Hz 경계 진동) → blended ref의 0.5–1 Hz 스펙트럼 상한 assert.
2. 저속 ramp plan → T200 데드밴드 속 명령 → 최소 유효 속도 바닥(이하 hold 양자화) 확인.
3. ±π 횡단 / 한 스텝 90° yaw → unwrap + r 클립 assert.
4. 0.5 m 밖 시작 plan(재등록 점프 재현) → 기각+hold assert.
5. plan 지연/지터, 짧은 plan(꼬리 v=0), planner 사망(>T초 무plan → hold 사다리),
   leash 포화 중 블렌드.
6. **M0(b)**: 실데모를 1 Hz로 흘리는 통합 시험 (위 M0 참조).

**B4. 로깅/기록 경계** (planner vs tracker 귀속)
- run 폴더 `plans.jsonl` (plan당 1줄): plan_id, 체크포인트 해시, 추론 지연, 조건화 obs
  시각, 앵커 스냅샷(eta/nu/r_now/leash/bridge_tier), **raw plan 원문**, 설치된 plan
  (클립/α/블렌드), 필터 판정 + **margin 수치**, **g 채널 원값과 문턱 판정**.
- CSV 열 추가: 활성 plan_id, plan 상대 시각, 블렌드 w, 레퍼런스 소스(plan/blend/hold/
  bridge), gripper 명령 상태. "솔버가 실제 소비한 stage-0"은 기존 `ref_ned_at` 열 —
  0823 교훈(CSV follow_err가 실제 참조가 아니었던 사고) 반복 금지.
- meta: `schema_version` 7→8 (인라인 주석), 항상-기록 `policy` 블록(체크포인트 해시,
  추론율, horizon, splice 정책, FF cap, gripper 문턱), `trajectory.kind`에
  `replay`/`policy`, `reference_clock.strategy`에 새 값. **새 런은 기존 런과 합산 금지.**

**B5. Gripper 실행 경로** (탐색 완료 — 송신 경로는 이미 존재, 신규 배선 불필요)
- 경로: `bus.cmd_gripper_drive.emit(±1.0/0.0)` → `set_gripper_drive`
  (hardware.py:1958-1960, **latched level** — 0.0을 보내야 정지) → `_buttons()` 비트
  76/77 (hardware.py:2422-2425; 실기 BTN0/15 = 77/76, 2026-08-06 실측 __main__.py:250-251)
  → `manual_control_send` (hardware.py:2436-2452). axes와 buttons는 직교 경로라
  **MPC engage 중에도 차단 안 됨** (window.py:1371-1384 `_pilot_gate`는 axes만).
  주입 지점 = workers.py:2788 `cmd_pilot.emit` 옆에서 g 문턱 → drive emit.
- g 실행 규약(제안): g>0.6 → close hold, g<0.4 → open hold, 사이 = neutral(0.0) [예측 —
  UMI-U의 0.40 문턱 + 히스테리시스 변형]; UMI-U처럼 최대 hold 시간(예: 4 s) 후 auto-neutral.
- **갭 6건 (구현 시 반영)**:
  1. policy↔pilot jaw **중재 부재** (grip_drive는 last-writer-wins) → 중재 규칙 신설
     (engage 중 policy 우선 + 파일럿 개입 시 policy gripper 정지 등).
  2. **피드백 전무** (gripper_fb=None 하드코딩 hardware.py:2135, open-loop servo) →
     "잡혔는지" 모름; g는 open-loop 문턱+hold-duration으로.
  3. demo/ROS2 백엔드는 `cmd_gripper_drive`를 **조용히 드랍** (demo.py:765는
     cmd_gripper만 연결) → HIL 테스트 전 demo 백엔드에 drive 경로 추가.
  4. 실기 gripper servo 채널/PWM 레인지 미기록 (servo9=Newton은 dataset.py:904 산문뿐)
     [스펙 미확인].
  5. sim heavy_gripper는 position ctrl[8](0–0.031 m) — 실기 momentary hold와 인터페이스
     다름, 매핑 없음, 현재 아무도 ctrl[8]을 안 씀.
  6. 안전: `set_enabled(False)`/`estop()`은 이미 grip_drive=0 — 추가로 **hold/bridge 강등
     시 policy gripper 명령 동결**을 필터 규칙에 명시.

### Phase 6 — 실기 (M0 이후 단계별)
1. 태그 위 정지 상태에서 plan 시각화만 (추종 없이 — policy 출력 overlay).
2. 저속·작업공간 박스 좁게 + 오퍼레이터 즉시 개입 대기 (기존 이중 게이트/DISARM 규약,
   gripper는 초기 런에서 비활성 옵션).
3. 단계적으로 박스/속도 확대 → 파지 활성. 매 런 plans.jsonl 기반 planner-vs-tracker 귀속.

## 6. 리스크 (순위)

1. **Handheld↔ROV 도메인 갭** — obs: in-air OAK depth(warp 보정) vs 수중 C3 depth
   (×0.64 stopgap, 거리 종속 잔차); 시점 분포(손 vs mount); dynamics: handheld 궤적은
   ROV authority(실전달 ~1/11, memory: sim-vs-hw-square-corner)를 모름 → 필터의 speed
   clamp/reprofile + M0가 1차 검증.
2. **파지 단계의 open-loop gripper** — 피드백 전무 + 추종 오차와 g 문턱의 상호작용
   (기체가 5 cm 못 갔는데 g가 close로 넘어가는 류) → M0에서 타이밍 검증, 문턱에
   히스테리시스, 초기 실기는 gripper 비활성 옵션.
3. **1 Hz plan stitching + 추론 지연** → 측정 앵커 금지 + blend + 앵커/점프 게이트 (§5-B1).
4. **프레임/부호/스케일 규약** — 좌표계 3개 + 태그 id 0/1 충돌(~8× 스케일 함정) →
   replay overlay 게이트 + tag_map.yaml 사용 규칙.
5. **Depth train/deploy 불일치** (0.64 적용 위치·range-dependence·invalid=far confound)
   → §5-A2 단일 규약 + validity mask + meta 경계.

(차점: goal 상실 — 단일 바위라 유예, multi-object 시 재점화(D1); hover collapse —
정지 구간 trim; prev_plan copycat — 도입 시 dropout.)

## 7. 미확정 사항

- 새 시연 촬영 장소/태그 배열 구성 (어떤 태그를 몇 장, 실측 배치를 어느 맵으로 —
  풀 바닥 매트 재사용 in-air? 별도 벤치 보드+신규 맵?).
- v_max/실측 달성속도 (필터 문턱 기준 — 현재 [예측]뿐).
- 실기 gripper servo 파라미터(SERVOn_FUNCTION/PWM 레인지) 확인.
- upstream diffusion_policy 코드베이스 재사용 vs 자체 최소 구현 (A3 시점 결정).
- 수중 depth 재캘리브 일정 (되면 데이터셋 경계 재설정 + 재학습).
- pid/*_tuned도 policy 소스를 소비하게 할지 (NedPlan 경로면 저비용, 검증 범위 증가).
- 영어판 전환 시점 (사용자 신호).
