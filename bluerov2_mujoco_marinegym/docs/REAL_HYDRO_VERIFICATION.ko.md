# 실물 BlueROV2 Hydro 검증 — sim-to-real 프로토콜 (청사진)

**상태: BLUEPRINT.** 아직 하드웨어 실행 안 함 — 실물 BlueROV2 + 풀이 준비되면 실행할 프로토콜이다. sim 검증
[HYDRO_VERIFICATION.ko.md](HYDRO_VERIFICATION.ko.md)의 **실물 짝**.

## 두 가지 다른 주장
- **Claim A — 코드 정확성 (완료, [HYDRO_VERIFICATION.ko.md](HYDRO_VERIFICATION.ko.md), 32/32).** "주어진 계수
  θ_sim으로 시뮬레이터가 Fossen 방정식을 재현한다." 계수는 *입력*으로 가정한 것. **소프트웨어**에 대한 진술.
- **Claim B — 모델 충실도 / sim-to-real (이 문서).** "이 Fossen 구조 + 이 계수가 *실제* BlueROV2의 거동을
  예측한다." = **시스템 식별**: 기체에서 θ_real을 측정해 θ_sim과 비교, 그 차이가 제어에 중요한지 판단.

**정직한 전제:** `marinegym_assets/BlueROV.yaml` 값은 *일반 BlueROV2 문헌값이지 우리 기체에서 식별한 게 아님.*
우리 기체는 자체 밸러스트·트림·ZED 하우징·케이블·폼이 달라 θ_sim은 어느 정도 틀린 게 보장됨. "실물 BlueROV2
검증"이 곧 claim B이고, 이 프로토콜은 식별→비교한다. 비유: claim A는 "계산기가 맞게 계산함"을 증명, claim B는
"맞는 숫자를 넣었나"를 점검.

**θ_sim (비교 대상, `BlueROV.yaml` + `bluerov.xml`):** mass 11.2 kg, I=[0.30375, 0.626, 0.5769],
V=0.0113459 m³, coBM=0.01 m, M_A=[5.5, 12.7, 14.57, 0.12, 0.12, 0.12], D_L=[4.03, 6.22, 5.18, 0.07, 0.07,
0.07], D_NL=[18.18, 21.66, 36.99, 1.55, 1.55, 1.55]. ⇒ B=ρgV≈110.97 N, W≈109.87 N, 순부력 **+1.1 N**,
복원 k=coBM·B≈1.11 N·m/rad.

**가용 신호:** BlueROV2 IMU(자세·각속도·선가속), 깊이/압력, 명령 thrust(PWM → T200 곡선, *산포 있음*), 그리고
프로젝트 핵심 자산인 **ZED2 + AprilTag SLAM 포즈 ground-truth**(굴절보정; 저장소 `claude.md` §3) + gantry.

---

## 전제조건 (식별 런 전에 반드시)
1. **load cell/추력대로 T200 곡선 재보정 — 최대 영향 전제조건.** 모든 실험이 "known thrust"를 쓰는데 우리가
   아는 건 명령 PWM → 산포 있는 일반 T200 곡선(전압 sag·수온·개체차·역방향 ~10–20% 약함)뿐. 15% thrust 오차 =
   15% drag 오차, M_A는 더 나쁨. **우리 thruster를 우리 전압에서** 추력 vs 명령 측정 → `BlueROV.yaml`의
   `force_constants: 4.4e-7` 가정 교체, 하류 모든 피팅을 조임.
2. **동기·타임스탬프 로깅**(thrust 명령 + IMU + AprilTag 포즈). 시간동기 불량은 *동역학* 피팅(실험 4–5)을 조용히
   오염시킴. 가장 먼저 확정.
3. **테더 관리.** 테더는 모델 안 된 형상의존·지배적 힘이자 비반복성 #1 원인. 중립부력 구간·느슨한 bight·반복 간
   동일 배선·동역학 런은 최소 전개. **팽팽한 테더로 drag 식별 금지**(테더를 재는 셈).
4. **풀 기하·여유 reconcile.** `config/config.yaml`엔 풀 폭 **1.8 m**, 작업 수치는 **2.438 m** — 거리 계산 전 실측
   reconcile. 깊이가 **1.143 m**뿐 → 동역학 런은 최대한 깊고 벽에서 멀게, 자유표면/벽/blockage 효과(아래) 유의.

---

## sim 테스트 → 실물 실험 매핑

| 항 | θ_sim | 우리 HW로 식별? | 방법 | 난이도 |
|---|---|---|---|---|
| 순부력 (W−B) | +1.1 N up | **쉬움** | 공기중 무게 + 깊이유지 thrust = W−B | 쉬움 |
| 복원 (coBM/GZ) | k≈1.11 N·m/rad | **가능** | IMU 평형각 + 소각 주기 | 쉬움–중 |
| 2차 drag D_NL | [18.18,21.66,36.99,…] | **가장 깨끗** | 종단속도 sweep (AprilTag) | 중 |
| 선형 drag D_L | [4.03,6.22,5.18,…] | 가능 | 저속 / 진동 감쇠 | 중 |
| added mass M_A | [5.5,12.7,14.57,…] | **부분적** | 과도에서만 (실험 4–5) | 어려움 |
| added-mass Coriolis C_A | (도출) | 직접 불가 (구조적) | M_A로 구성됨; 결합 기동으로 간접 점검 | — |

정적·정상상태 항(부력·복원·drag) — **평상시 받는 힘의 대부분** — 은 가진 센서로 직접 측정 가능. added mass가 crux.

---

## 실험 프로토콜 (우선순위)

각각: **목적 → 계수 → setup → 가진/명령 → 로깅 → 피팅 → 예상 교란 → θ_sim 대비 합격기준.** (허용오차 ±X는 제어기의
각 계수 민감도가 파악되면 실제값 지정 — 시작값: drag ±15%, M_A는 자릿수.)

### 실험 1 — 정적 부력·질량 *(쉬움, 고신뢰)*
- **계수:** 순부력 W−B, 질량 m. **setup:** 저울로 공기중 무게(→ W=mg). 수중에서 수직 thrust로 일정 깊이 유지(깊이
  센서 평평). **로깅:** 무게; 깊이유지 수직 thrust; 또는 무추력 종단 승강 속도.
- **피팅:** 깊이유지 수직 thrust = W − B. **교란:** thrust 보정(전제조건 1).
- **합격:** 순부력이 sim +1.1 N의 ±X 내; 질량 11.2 kg의 ±2% 내.

### 실험 2 — 복원 진자(roll & pitch) *(쉬움–중 — sim T4의 실물판)*
- **계수:** 복원 강성 k = coBM·B (및 r_G−r_B 오프셋). **setup:** 중립 잠수·무추력; 소각 roll(다음 pitch)로 기울여
  릴리즈. **로깅:** IMU 자세 θ(t).
- **피팅:** 평형 자세 → r_G−r_B 수평 오프셋; 소각 주기 T = 2π√((I+M_A_rot)/k) → k (회전 added inertia도 산출,
  실험 4); log-decrement → D_L_rot.
- **교란:** 유한진폭 D_NL_rot이 감쇠 부풀림(≤3° 릴리즈); 순부력에 의한 느린 heave 드리프트.
- **합격:** k가 1.11 N·m/rad의 ±X 내; 평형 오프셋이 coBM = 0.01 m와 정합.

### 실험 3 — 종단속도 drag sweep(surge/sway/heave/yaw) *(워크호스 — sim T2 직접 재현)*
- **계수:** 축별 D_L, D_NL. **setup:** 풀 장축 따라 개방 런(전제조건 4 run-up 후), 깊고 벽에서 멀게. **명령:**
  축별 여러 레벨의 일정 thrust. **로깅:** AprilTag 포즈 → 미분해 body 속도; v_∞ 도달(포즈/깊이 평평) 후부터 측정창.
- **피팅:** 축별 `thrust = D_L·v_∞ + D_NL·v_∞·|v_∞|`; 여러 thrust 레벨로 D_L, D_NL 피팅. (sim T2 닫힌형의 실물판.)
- **교란:** thrust 보정(모든 레벨 일관 스케일 오프셋 ⇒ drag가 아니라 T200 곡선이 틀림 — 그래서 이 실험이 전제조건1
  오차도 *노출*); 짧은 run-up이 고추력 v_∞ 제한; **풀 blockage/벽으로 측정 drag가 개수면보다 체계적으로 높게** 나옴;
  테더; wake 재순환(런 사이 30–60 s 대기, 방향 교대).
- **합격:** D_L/D_NL 곡선이 θ_sim과 스케일·형상 ±X 내 일치(풀 값은 다소 높게 나옴이 예상 — 풀 아티팩트, 바다 진실
  아님).
- **가장 가치 큰 측정** — 축당 곡선 하나가 sim drag가 맞는지 + thrust 보정이 맞는지 둘 다 말해줌.

### 실험 4 — 진동 감쇠로 added inertia(heave/roll/pitch) *(중 — 복원 있는 축)*
- **계수:** M_A[2], M_A[3], M_A[4]. **setup:** 실험 2 재사용(이 축들은 복원 "시계" 보유). **로깅:** IMU/포즈 진동.
  **피팅:** 실험 2의 k로 자연주기 T = 2π√((I+M_A)/k)에서 added inertia M_A 역산; log-decrement → D_L. **교란:**
  1.14 m 깊이에서 heave M_A의 자유표면 민감성; 유한진폭 감쇠. **합격:** 이 3축 M_A가 sim의 자릿수/±X 내.
  *(surge·sway는 복원 없음 → 자연 진동 없음 → 여기서 못 미침.)*

### 실험 5 — surge/sway added mass 동역학 식별(PRBS/chirp + LS/EKF) *(어려움 — 추정 후 bound)*
- **계수:** M_A[0], M_A[1](복원 없는 병진축). **setup:** 실험 1–4로 정적·drag 고정 후, 런 안에서 왕복 가속을
  유지하는 **PRBS 또는 multi-sine chirp** thrust 명령. **로깅:** thrust, AprilTag 포즈→속도, IMU 가속(동기!).
- **피팅:** `[M_RB+M_A, D_L, D_NL]` 결합 최소제곱(또는 파라미터를 증강상태로 추정하는 EKF/UKF — 이 저장소의
  EAOB식 기계 활용)으로 나머지 고정 후 식별.
- **교란:** 짧은 run-up → 가속 구간 짧음 → **M_A와 D_L 상관 커짐** ⇒ M_A 신뢰구간 넓음; 포즈 미분 노이즈.
  **예인수조/PMM 없음**(in-phase drag와 quadrature added mass를 분리하는 gold standard는 도달 불가).
- **합격:** M_A[0], M_A[1]을 점추정이 아니라 **오차범위와 함께** 보고. 소형 ROV의 added mass ≈ 건조질량의 30–50%가
  자릿수 sanity(sim surge 5.5/11.2 ≈ 49%, 타당).

---

## 가능 vs 불가 (리포트에 정직히)
- **깨끗·고신뢰:** 부력, 복원, 축별 drag (정적+정상상태 = 평상시 힘의 대부분).
- **제약 가능:** heave/roll/pitch added mass(진동 테스트).
- **큰 오차범위 추정:** surge/sway added mass(짧은 런의 PRBS+EKF).
- **여기선 불가:** 예인수조급 M_A; Planar Motion Mechanism; 바다-유효값 — 작고 얕은 풀(1.14 m)은 벽/자유표면/
  blockage로 drag·added mass를 **체계적으로 높게** 만듦 → 풀 식별값은 개수면 진실이 **아님**. 쉬운 항은 깨끗이
  식별, M_A는 bound, 풀 값을 바다 진실로 보고하지 말 것.

## 계측 업그레이드 (영향순)
1. **load cell/추력대** — T200 곡선 재보정; 지배적 교란 제거, 모든 피팅 조임. **최우선.**
2. **DVL** — 깨끗한 body 속도, 포즈 미분 노이즈 제거(M_A에 가장 해로운 것). 비쌈; 동역학 식별을 강하게 밀 때만.
3. (하드웨어 아님) **동기 타임스탬프 로깅** — 실험 4–5의 전제.

## 합격기준 표 템플릿 (실행 후 채움)

| 항 | θ_sim | θ_real(측정) | Δ% | 판정 | 비고 |
|---|---|---|---|---|---|
| 순부력 [N] | +1.1 | | | | |
| 복원 k [N·m/rad] | 1.11 | | | | |
| D_L surge/sway/heave | 4.03/6.22/5.18 | | | | |
| D_NL surge/sway/heave | 18.18/21.66/36.99 | | | | |
| D_L/D_NL yaw | 0.07 / 1.55 | | | | |
| M_A heave/roll/pitch | 14.57/0.12/0.12 | | | | (실험 4) |
| M_A surge/sway | 5.5/12.7 | | | | (실험 5, ± 오차범위) |

## 참고문헌 (prior·sanity — 모든 발표값은 *남의 기체*)
- Fossen, *Handbook of Marine Craft Hydrodynamics and Motion Control*, 2판 (2021) — 모델·식별법.
- Wu, *Towards a BlueROV2 Open-Source Model* — 비교용 BlueROV2 파라미터 셋.
- von Benzon et al., "An Open-Source Benchmark Simulator: Control of a BlueROV2," *J. Mar. Sci. Eng.* 2022.
- BlueRobotics T200 성능 차트 — 보정해야 할 추력 산포·정/역 비대칭.
- Cai et al., "Learning to Swim," ICRA 2025 — Fossen hydro 계수의 sim-to-real 관점.

---
*짝 문서:* [HYDRO_VERIFICATION.ko.md](HYDRO_VERIFICATION.ko.md) (claim A — sim 코드는 검증됨; 여기 계수는
실물 식별 대기).
