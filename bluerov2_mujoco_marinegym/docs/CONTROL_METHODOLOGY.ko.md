# 제어 방법론 로그 — BlueROV2 강인 제어

이 시뮬레이터 제어 개발의 **날짜별 저널**입니다: 어떤 방법을 도입했고, **왜**(직전 방법이 못 하던 것),
그 **이론**, **어떻게** 구현했는지(+ 핵심 설계 결정과 그 이유), 그리고 **결과**. 위에서 아래로 읽으면
전체 논리 흐름을 한눈에 따라갈 수 있습니다.

이건 *서사*입니다 — 주제별 레퍼런스 문서([00_OVERVIEW](00_OVERVIEW.md), [03_THRUSTERS](03_THRUSTERS.md),
[04_HYDRO](04_HYDRO.md), [07_DISTURBANCES](07_DISTURBANCES.md) …)가 "무엇이 어떻게 동작하는가"라면,
이 로그는 "**왜 이 순서로 이렇게 결정했는가**"를 기록합니다.

**유지 방식:** **주요** 제어 변경/마일스톤마다 갱신(작은 버그픽스·리팩토링 제외). 갱신 시 아래에 날짜
항목을 추가하고 영어 쌍둥이 [CONTROL_METHODOLOGY.md](CONTROL_METHODOLOGY.md)와 동기화합니다. 항목 형식:
**Why(왜) → What(이론, *지배방정식 포함*) → How(구현+결정) → Result(결과)**. 모든 항목에 수식을 적는다.

---

## 환경 — 플랜트, 외란, 그리고 컨트롤러가 제어하는 것

아래 모든 항목의 공통 맥락. (참조: [04_HYDRO](04_HYDRO.md), [07_DISTURBANCES](07_DISTURBANCES.md);
플랜트 물리는 [HYDRO_VERIFICATION](HYDRO_VERIFICATION.md)에서 독립 검증됨.)

**플랜트(제어 대상).** BlueROV2, **FLU** body frame(x 전, y 좌, z 상), 중력 (0,0,−9.81). 강체 m=11.2 kg,
관성 diag(0.30375, 0.626, 0.5769), COM은 body 원점. 상태 = 자세 η + body 속도 ν; 컨트롤러가 가정하는
6-DOF Fossen 모델:

```
η = [x y z  φ θ ψ]ᵀ  (world 위치 + roll/pitch/yaw)     ν = [u v w  p q r]ᵀ  (body 선+각속도)
η̇ = J(η) ν
M ν̇ + C(ν)ν + D(ν)ν + g(η) = τ + w
  M = M_RB + M_A                  (강체 + added mass; 여기선 대각 M_A)
  C(ν) = C_RB(ν) + C_A(ν)         (Coriolis/구심)
  D(ν) = D_L + D_NL·|ν|           (선형 + 2차 drag, 대각)
  g(η) = 복원력 (부력 B를 CB = COM + coBM·ẑ_body에, 무게는 COM에; 순부력 +1.1 N 상향)
  τ = 제어 wrench,   w = 외란 wrench
```
(sim에선 M_A를 질량행렬이 아니라 외력(lag)으로 적용 — [HYDRO_VERIFICATION](HYDRO_VERIFICATION.md);
컨트롤러는 여전히 M = M_RB+M_A로 추론.)

**외란 w(환경 forcing).** 3개 FLU 층 — current+waves는 **수속도**로 들어와 상대속도를 통해 drag·added mass
둘 다 변조(Morison류); kick은 직접 외력:

```
v_water(t,d) = v_current + v_wave(t,d)
v_r = ν_lin − Rᵀ v_water                                  → D(·), M_A(·)에서 ν_lin 대신 사용
v_wave(t,d) = Σ_i U_i e^(−k_i d)[ dir_i cos(ω_i t+φ_i) + ẑ sin(ω_i t+φ_i) ],  k_i = ω_i²/g
F_kick(t)  = Poisson 충격 world-frame 외력(gust), COM에 직접 적용
```
즉 w는 **DC(current) + 진동 파도대역 + 충격(kick)** — 각 컨트롤러가 평가받는 스펙트럼. (JONSWAP
스펙트럼은 2026-06-14 평가환경 항목.)

**두 경계 — "입력"은 *플랜트*와 *컨트롤러*에서 뜻이 다름.**
**BlueROV2 플랜트의 입력은 추력(thrust) `τ`** — 6개 추진기가 만드는 body wrench(Fossen 식 우변의 `τ`)이고,
출력은 상태(η, ν). **컨트롤러의 입력**은 측정 상태 + 기준이고, **출력은 wrench 명령**인데, 이게 할당 + T200
추력곡선을 거쳐 그 추력이 *된다*. 즉 **컨트롤러 출력 = 플랜트 입력 = 추력.** 폐루프:

```
 p_ref, ψ_ref, v_ref ┐
                     ├──►[ 컨트롤러 ]──► wrench 명령  τ_c = [Fx Fy Fz 0 0 Mz]
 측정 η, ν ──────────┘                          │ 할당:  f = B⁺ τ_c   (6 추진기 힘, N)
        ▲                                       │ T200:  throttle = curve⁻¹(f) → data.ctrl
        │                                       ▼
        │                              [ 추진기 → MuJoCo + hydro ]
        │   플랜트 입력 = 추력 τ = B·f ─►  M ν̇ + C(ν)ν + D(ν)ν + g(η) = τ + w  ─► 새 η, ν
        └────────────────────────────────────────────────────────────────────────────┘
```

**컨트롤러 I/O** (아래 각 방법이 읽고/쓰는 것 — 입력은 *측정*이지 추력이 아님):

```
컨트롤러 INPUT  (매 스텝 측정):  p(world 위치), R(자세 → φ,θ,ψ), v(world 선속도),
        ω(body 각속도);  DOB-MPC는 추가로 ν̇(유한차분, EAOB용).
        + 기준: p_ref, ψ_ref, 궤적이면 v_ref.
컨트롤러 OUTPUT (= 플랜트 입력):  body wrench  τ_c = [Fx Fy Fz  Mx My Mz],  Mx = My = 0
        → 6 추진기 힘  f = B⁺ τ_c   (B = 6×6 할당, rank 5),  → data.ctrl (T200 경유)
        → 플랜트로 들어가는 실현 추력  τ = B f   (명령 불가한 pitch My는 투영 제거)
```
**Rank-5 underactuation.** 수평 추진기 4개가 COM보다 0.0725 m 아래라 surge가 pitch에 커플링:
`My ≈ −0.0725·Fx`. pitch는 명령하지 않고, 부력 복원이 커플링을 상쇄하는 트림으로 부유:
`sin θ* = 0.0725·Fx / (coBM·B)` (6 N서 ≈23°). 아래 모든 방법이 같은 할당에 τ_c를 내보내고 이 제약을 물려받음.

### 행렬 — 값·구조·출처

**값의 진짜 출처 (우리 파일이 아니라 학술 1차 출처, 검증함).**
- **유체계수 — added mass M_A, 댐핑 D_L, D_NL:** **Wu, C-J. (2018), *6-DoF Modelling and Control of a
  Remotely Operated Vehicle*, MEng 학위논문, Flinders University — Table 5.2(added mass)·5.3(linear &
  quadratic damping)**에서 tow-tank 정·동적 실험으로 식별. 아래 값들은 그 논문과 *정확히* 일치(원문 대조
  확인). 동일 세트가 peer-review 벤치마크 **von Benzon et al. (2022, *J. Mar. Sci. Eng.* 10(12):1898)**에
  재사용되고 **MarineGym**(Chu et al., IROS 2025)으로 채택 → 우리
  [`BlueROV.yaml`](../marinegym_assets/BlueROV.yaml).
- **강체 질량·관성 M_RB, 기하, 6 추진기 마운트:** **`bluerov2_description` ROS URDF**(`BlueROV.urdf`)가
  MarineGym Isaac asset 거쳐 우리 [`bluerov.xml`](../bluerov.xml). *주의:* 여기 m = 11.2 kg은 이 CAD/URDF
  출처이고 **Wu 논문은 11.5 kg** 사용 — 강체와 유체 파라미터는 *출처가 다름*(그래서 따로 인용).
- **volume(0.0113459 m³)·coBM(0.01 m):** CAD 유래, MarineGym `BlueROV.yaml`.
- **T200 추력곡선·rotor config:** **Blue Robotics 공개 T200 성능 데이터**, MarineGym `actuators/t200.py`에 피팅.

**행렬** (각 6-벡터 순서 **[surge, sway, heave, roll, pitch, yaw]**; 단위 kg / kg·m²):

```
            surge  sway  heave    roll     pitch     yaw
          ┌ 11.2    0     0        0         0        0      ┐
          │   0   11.2    0        0         0        0      │
M_RB  =   │   0     0   11.2       0         0        0      │   강체 (bluerov2_description URDF)
          │   0     0     0      0.30375     0        0      │   COM이 body 원점 ⇒ 대각,
          │   0     0     0        0       0.626      0      │   m·z_g surge–pitch 커플 없음
          └   0     0     0        0         0      0.5769   ┘

          ┌ 5.5    0      0      0      0      0    ┐   added mass (Wu 2018, Table 5.2)
          │  0   12.7     0      0      0      0    │   (Xu̇,Yv̇,Zẇ,Kṗ,Mq̇,Nṙ), 대각 —
M_A   =   │  0     0    14.57    0      0      0    │   비대각(예: Yṙ,Nv̇) 누락(MarineGym).
          │  0     0      0    0.12     0      0    │   heave 14.57 > m 11.2 ⇒ 질량행렬이 아닌
          │  0     0      0      0    0.12     0    │   EMA-lag 외력으로 적용 (HYDRO_VERIFICATION)
          └  0     0      0      0      0    0.12   ┘

M = M_RB + M_A = diag(16.70, 23.90, 25.77, 0.42375, 0.746, 0.6969)          — SPD (검증, T1.3)

D_L   = diag( 4.03,  6.22,  5.18, 0.07, 0.07, 0.07)   선형 drag    (Wu 2018, Table 5.3)
D_NL  = diag(18.18, 21.66, 36.99, 1.55, 1.55, 1.55)   2차 drag     (Wu 2018, Table 5.3)
   D(ν) = D_L + D_NL·|ν|  →  소산 힘으로 적용  −(D_L·ν + D_NL·|ν|·ν)
   (두 계수 세트 모두 sim에서 0.00 % 복원, T4.3)

C_RB(ν)·ν = [ m(qw−rv),  m(ru−pw),  m(pv−qu),  (I_z−I_y)qr,  (Iₓ−I_z)pr,  (I_y−Iₓ)pq ]ᵀ   (M_RB에서)

            ┌  0     0     0      0    −a₃w   a₂v ┐   a = (a₁…a₆) = M_A 대각
            │  0     0     0    a₃w     0    −a₁u │     = (5.5, 12.7, 14.57, 0.12, 0.12, 0.12)
C_A(ν)  =   │  0     0     0   −a₂v   a₁u     0   │   ν = [u v w  p q r]
            │  0   −a₃w   a₂v    0    −a₆r   a₅q │   skew 대칭 (Fossen 2011 Eq. 6.44);
            │ a₃w    0   −a₁u   a₆r     0    −a₄p │   C_A = −C_Aᵀ 및 sim과 1e-14 일치
            └−a₂v   a₁u    0   −a₅q   a₄p     0   ┘   검증 (T1.1–1.2)

g(η)  복원력 (FLU):  B = ρgV = 997·9.81·0.0113459 = 110.97 N  (상향, CB에)
                     W = mg  = 11.2·9.81           = 109.87 N  (하향, COM에)
                     순 = B − W = +1.10 N 상향 ;  CB = COM + coBM·ẑ_body,  coBM = 0.01 m
                     복원 모멘트 = k·sinθ_tilt ,  k = coBM·B = 1.110 N·m/rad
   (volume·coBM은 BlueROV.yaml; ρ = 997 담수; m, g는 URDF / model.opt.gravity)

τ = B · f   (플랜트 입력: 6 추진기 힘 f [N]로부터 body wrench)
        thr0    thr1    thr2    thr3    thr4    thr5
      ┌ 0.707   0.707  −0.707  −0.707   0       0     ┐ Fx
      │ 0.707  −0.707   0.707  −0.707   0       0     │ Fy
B  =  │ 0       0       0       0       1       1     │ Fz
      │ 0.051  −0.051   0.051  −0.051  −0.110   0.110 │ Mx
      │−0.051  −0.051   0.051   0.051  −0.002  −0.002 │ My  ← 수직 추진기에서 ±0.002뿐
      └ 0.167  −0.167  −0.175   0.175   0       0     ┘ Mz     ⇒ rank 5, pitch 거의 명령 불가
   열 i = [ d_i ; r_i × d_i ],  d_i = 추진기 축(site +X),  r_i = 위치 − COM.
   수평 4개 z = −0.0725 m(±45° 벡터드) ⇒ surge→pitch 커플; 수직 2개. (bluerov.xml sites)
   ·  T200 곡선(힘 ↔ throttle, 실제 드라이버 층): u∈[−1,1] → rpm(0.075 deadband, ±3900 rpm) →
   Blue Robotics 비대칭 T200 피팅으로 추력, t200_thrust(+1)=+64.13 N, t200_thrust(−1)=−51.55 N
   (~1.24 전/후진 비대칭). 할당/곡선: thrusters.py.
```

---

## 2026-06-14 — Baseline PID/PD 원점 유지 제어기

**Why.** 고급 제어에 투자하기 전에, 단순하고 모델이 거의 없는 제어기가 원점을 얼마나 잘 유지하고 외란을
얼마나 기각하는지 **정량화할 baseline**이 필요했다 — 이후 모든 방법을 재는 잣대.

**What (이론).** PID/PD 설정점 제어기는 차량을 목표 자세로 끌고 간다. P(와 D)는 목표로의 스프링–댐퍼,
**적분항**은 정상오차를 누적해 *미지의 상수* 외란(예: 정상류)을 상쇄한다 — PID가 "DC를 기각"하는 고전적
이유: 적분작용으로 감도함수 S(0)→0이라 상수 입력외란에 **정상상태 오차 0**. 반면 시간변화(파도대역) 외란이나
충격은 못 잡는다(S(jω)가 DC 근처에서만 작으므로).

**How (구현).** [controller.py](../controller.py) `PoseController`: world-frame 위치 PD에 힘을 body로
회전(이방성 게인의 "crabbing" 회피); 순부력 feed-forward(+1.1 N); 정상류 바이어스용 **gated anti-windup
적분**(설정점 근처에서만 적분 후 클램프); surge 포화 + slew 제한 + soft pitch guard. 결정: **pitch는 절대
명령하지 않음** — BlueROV2 vectored-6은 rank-5 underactuated이고 수평 추진기 4개가 COM보다 0.0725 m
아래라 surge가 pitch에 커플링(My≈−0.0725·Fx); roll/pitch는 수동 부력 복원에 맡긴다.

**Equations.**
```
e = p_ref − p                                            (world 위치 오차)
F_world = K_p e − K_d (v − v_ref) + K_i ∫e dt            (∫ gated: |e|<e_gate일 때만 적분)
F_world,z += −net_buoy                                   (부력 feed-forward)
F_body = Rᵀ F_world         (body로 회전; surge는 slew 제한 + 포화 + pitch guard)
M_z = k_pψ·wrap(ψ_ref−ψ) − k_dψ·r + k_iψ ∫e_ψ dt         (yaw PD+I)
τ = [F_body,x, F_body,y, F_body,z, 0, 0, M_z]
```
*적분이 DC를 기각하는 이유:* 폐루프 감도 `S(jω) = 1/(1+L(jω))`; 적분작용으로 `S(0) = 0` → **상수 w에
정상상태 오차 0**. 단 `|S(jω)|`는 DC 근처만 작아서 파도대역·충격은 통과.

**Result.** 원점 유지 성공; 0.2 m/s 정상류에서 적분이 DC 바이어스를 ~0.5 cm로 제거. 그러나 **파도대역
(반경 std ≈13 cm), 충격 kick(~30 cm 과도), 정상 9° 트림 pitch**가 잔존 — 모델 기반·외란 인지 제어기가
공략해야 할 바로 그 잔차. → MPC 동기.

---

## 2026-06-14 — 평가 환경: square 미션 + 불규칙 JONSWAP 파

**Why.** 제어기를 스트레스하려면 (a) *이동하는* 기준과 (b) *현실적인* 해양 외란이 필요했다. 기존 3-사인파
모델은 너무 규칙적(명확한 반복주기)이라 외란 기각을 과소 검증했다.

**What (이론).** JONSWAP 스펙트럼을 **등에너지 주파수 빈 + 빈별 랜덤 주파수**로 샘플링하면 인공적 반복주기가
사라진다("랜덤해 보이는" 핵심); `cos^(2s)` 방향 분산이 yaw 가진을 더한다. 파도는 **수속도**(깊이 감쇠
e^(−k·depth)의 궤도운동)로 진입해, 상대속도 hydro를 통해 파도 drag와 파도 added-mass를 모두 구동한다 —
추가 항 없는 Morison류 모델.

**How (구현).** [mission.py](../mission.py) `SquareMission`(approach → track → done, CSV 자동기록);
`disturbances.jonswap_wave_specs(...)`. square는 연속 이동 설정점 + 속도 feed-forward를 제어기 D항에 사용.

**Equations.**
```
JONSWAP:  S(ω) ∝ ω⁻⁵ exp(−1.25 (ω_p/ω)⁴) · γ^r,   r = exp(−(ω−ω_p)²/(2σ²ω_p²)),  ω_p = 2π/T_p
등에너지 빈 → ω_i(빈당 랜덤 ω);  a_i = (H_s/4)√(2/N);  U_i = ω_i a_i   ⇒ 4√(Σa_i²/2)=H_s
v_wave, v_r:  위 "환경" 절과 동일(성분이 v_wave를 구성; v_r이 drag + added mass 구동)
square 기준(원점=모서리, CCW, 변 S, 속도 c):  s(t) = c·(t − t₀)
   p_ref(s)가 S×S 사각형 4변을 그림;   v_ref = c · tangent(s)   (속도 feed-forward)
```

**Result.** 현실적 테스트베드. PID가 square를 추종하되 위상지연이 있고 외란 하에서 underactuated pitch
과도가 커짐을 확인 — 원점 유지에서 본 한계가 이동 중에도 동일.

---

## 2026-06-15 — MPC와 DOB-MPC (논문 이식)

**Why.** PID baseline의 한계가 정량화됐다: **DC 정상류는 기각**하지만 **파도대역**·**충격 kick**은 못 잡고,
underactuated surge→pitch 커플링이 이동 중 유지 강도를 제한한다. 그래서 (a) 작동기/상태 **제약**을 명시적으로
지키고, (b) 모델로 미래를 **선제 대응**하며, (c) 외란을 **능동 기각**하는 제어기를 원했다.

**What (이론).**
- **MPC (모델 예측 제어):** 매 제어 스텝마다 유한지평 최적제어 문제를 푼다 — N스텝에 걸쳐 추종오차 + 제어
  노력을 동역학 모델과 입력/상태 제약 하에 최소화 — *첫* 최적 입력만 적용하고 다음 스텝에 다시 푼다(receding
  horizon). **모델로 앞을 보고 제약을 자연스럽게** 처리해 PID를 능가한다. 단 *plain* MPC는 적분작용이 없어,
  상수 비모델 외란에 **이득제한 정상오차**를 남긴다.
- **DOB-MPC (외란관측기 기반 MPC):** **Extended Active Observer (EAOB)** 추가 — 증강상태 연속-이산 EKF
  (상태 = [자세 η; 속도 ν; 외란 w], 18차원, 내부모형 ẇ=0)로 **측정 + Fossen 모델로 외란 wrench w를 온라인
  추정**. 추정치 `w_hat`을 매 스텝 **MPC 예측모델에 주입**(지평 동안 유지)해, MPC가 추정 외란을 *상대로* 계획.
  이것이 plain MPC가 남기는 정상오차를 제거한다 — 게다가 "제어에 feed-forward 더하기"와 달리, w를 *예측*에
  넣으면 최적화기가 추론하는 파라미터-가변 모델이 된다.

**논문.** Hu, Li, Jiang, Han, Wen, "Disturbance Observer-Based Model Predictive Control for an
Unmanned Underwater Vehicle," *J. Mar. Sci. Eng.* 2024 ([docs PDF](Disturbance%20Observer-Based%20Model%20Predictive%20Control.pdf)).
독립 패키지 `bluerov2_mujoco_dobmpc/`의 검증된 EAOB + NMPC 수학을 재사용해 marinegym(FLU) 시뮬에 이식.

**How (구현 + 핵심 결정).** [dobmpc_controller.py](../dobmpc_controller.py) + [dobmpc/](../dobmpc/)
(fossen/eaob/mpc는 거의 그대로 복사; params와 frames만 marinegym 전용). 솔버: CasADi + IPOPT NLP, 다중
슈팅, N=60, 상태 12 / 제어 4, 해석적 Fossen RK4 예측, `w_hat`은 파라미터. 설계 결정과 *이유*:
- **프레임:** 관측기/MPC는 논문의 NED/FRD, marinegym은 FLU. 고정 `S=diag(1,−1,−1)` 켤레변환
  (`R_ned = S·R_flu·S`)으로 상태를 넣고 4-DOF wrench를 빼냄 — 오일러각 수동 부호반전 금지(미묘한 버그 원천).
  ([dobmpc/frames.py](../dobmpc/frames.py))
- **params를 marinegym `BlueROV.yaml`에서 재빌드** — 예측모델이 *이* plant와 일치해 참 외란(current/wave/
  kick)만 `w`로 남게. 두 함정: **damping 부호 반전**(marinegym은 양수 저장, Fossen은 음수 필요 — 거꾸로면
  모델이 *anti-damped*); **ZG_MASS=0**(marinegym COM이 body 원점이라 m·zg surge↔pitch *관성* 커플링이
  없음, 부력 복원 ZG=0.01은 유지). ([dobmpc/params.py](../dobmpc/params.py))
- **가속도는 유한차분, `data.qacc` 아님** — marinegym은 added mass를 *외력*으로 적용하므로 qacc에 이미
  포함돼 EKF 측정모델에서 이중계산됨.
- **Underactuation = "option (a)":** MPC의 pitch/roll *위치* 가중치를 0으로, pitch는 물리 트림으로 부유,
  정상 surge→pitch 커플링은 EAOB가 `w`로 흡수.
- **20 Hz 제어 + ZOH**(물리 substep 사이), EAOB엔 실제 유지한 *명령* NED wrench를 먹임.

**Equations.**
```
MPC — receding-horizon OCP, 매 스텝 풀고 u₀만 적용:
  min_{x,u}  Σ_{k=0}^{N−1} ‖x_k − x_ref,k‖²_Q + ‖u_k‖²_R  +  ‖x_N − x_ref,N‖²_QN
   s.t.  x_{k+1} = f_d(x_k, u_k, ŵ),   |u_k| ≤ u_max,   |ν_lin| ≤ v_max,   |φ|,|θ| ≤ 1.2 rad
  예측모델 (Fossen, ŵ는 지평 동안 상수 파라미터):
     ẋ = [ J(η)ν ;  M⁻¹( τ(u) + ŵ − C(ν)ν − D(ν)ν − g(η) ) ],   τ(u) = [u₁,u₂,u₃, 0,0, u₄]
  plain MPC는 ŵ = 0  → 상수 w에 이득제한 정상오차.

EAOB — 증강 연속-이산 EKF, 상태 x_a = [η; ν; w], 내부모형  ẇ = 0:
  예측:  ẋ_a = f(x_a, τ),   P⁺ = Φ P Φᵀ + Q,   Φ = exp(F·dt),  F = ∂f/∂x_a
  갱신:  z = [η; ν; τ],   ŵ는   h_τ(x_a) = M ν̇ + C(ν)ν + D(ν)ν + g(η) − w 를 통해 관측가능
         K = P Hᵀ(H P Hᵀ + R)⁻¹,   x_a ← x_a + K (z − h(x_a))
DOB-MPC = MPC의 예측 파라미터로 ŵ = (EAOB w-추정)을 매 스텝 주입.
```

**Result.** JONSWAP+current+kick 하의 원점 유지 비교(반경 RMS / DC 바이어스):
**PID 13.3 cm / −0.1 · MPC 3.6 cm / +2.3 · DOB-MPC 3.7 cm / +0.3** (cm). 두 MPC 변종 모두 파도대역
잔차를 PID 대비 ~5× 감소; **DOB-MPC의 EAOB가 plain MPC가 남기는 DC 바이어스를 제거(+2.3→+0.3 cm)** —
논문 핵심 결과를 현실적 불규칙 파에서 재현. 파도대역 자체는 공통 잔차(ẇ=0 모델이 4 s 파를 못 따라감) →
후속: **oscillator 외란상태**(내부모형원리 / Fossen 8장 wave filter).

---

## 2026-06-15 — DOB-MPC 런타임(렉) 진단 + acados 권고

**Why.** 맥북에서 viser 원격으로 보니 렉(슬로모션 + 가끔 프리즈)이 느껴졌다. 추측 대신 진짜 병목을 찾기 위해
프로파일링.

**무엇을 찾았나 (프로파일, 외란 하 DP, 워밍업 120틱).** 제어틱당(20 Hz, 예산 50 ms): **NMPC.solve ≈ 83 ms
(≈79%)**, EAOB.update ≈ 22 ms (≈21%), 전체 틱 ≈ 106 ms = **실시간의 0.47배**; 가끔 **2.2 s 프리즈**(IPOPT
실패 시 cold-restart). `cProfile`상 시간은 IPOPT 솔브 *내부*(`casadi.Function_call`)에 있고, 파이썬
rollout/Jacobian 조립이 아님.

**왜 느린가 (근본원인).** IPOPT는 범용 내부점 NLP 솔버로, 매 스텝 비선형 문제를 **수렴(tol 1e-5)까지** 완전히
푼다 — 여러 번의 내부점 반복, 각 반복마다 큰 희소 KKT 분해(N=60 → 상태 732 + 제어 240 + 슈팅 제약). 이
"매 스텝 완전 수렴"이 과하다: 50 ms 사이 시스템은 거의 안 변하는데 매 틱 처음부터 다시 최적화. 흔한 값싼
개선 셋은 **이미 적용됨** — warm-start ✅, 해석적(CasADi autodiff) Jacobian ✅, 경량 해석 Fossen 모델 ✅ —
그래서 프로파일이 MPC 쪽엔 남은 쉬운 개선이 없음을 증명; *NLP를 푸는 방식*만 중앙값을 움직일 수 있다.

**권고 (방법 확정, 이식 보류).** 솔브를 **acados**로:
- **Real-Time Iteration (RTI):** 매 스텝 **SQP 1회**(완전수렴 아님), 이전 스텝에서 warm-start. 시스템이
  천천히 변하므로 틱당 한 발짝이 누적돼 최적해 추종 → **반복수 고정 → 결정적·짧은 솔브, 프리즈 없음**.
- **HPIPM + (partial) condensing:** RTI의 선형 QP를 OCP의 시간-밴드 블록 구조를 활용하는 전용 QP 솔버로
  (condensing이 KKT 축소) — 범용 MUMPS보다 훨씬 빠름.
- **C 코드 생성:** 모델/미분/솔버를 네이티브 C로 → 파이썬/CasADi 오버헤드 없음.
- 예상 **~83 ms → ~2–5 ms (15–40×)**, N=60·Fossen 모델·DOB 구조(`w_hat` 파라미터) 보존. 모델이 이미
  CasADi-심볼릭이고 논문 원본도 acados라 1:1 재인코딩(재유도 아님). 부수 개선: EAOB의 유한차분 Jacobian을
  CasADi autodiff + Cholesky로(≈22→4 ms).
- **acados 단점:** C 라이브러리 빌드 + `acados_template` + 환경설정(`pip`만으론 안 됨); RTI는 1스텝 *근사*
  (워밍업 필요, 강한 비선형 과도에서 정확도 저하 가능 → IPOPT 해와 검증); 코드생성이라 수정 시 재생성;
  globalization 약함. 툴체인 회피 시 대안: 직접 만든 **LTV-QP + OSQP**(~5–15 ms, 가볍지만 덜 견고).

**Equations.**
```
현재 매 틱(IPOPT): 위 OCP를 tol 1e-5까지 수렴 — 내부점 반복 다수, 각 반복마다 희소 KKT 분해,
   크기 ~ N·(n_x+n_u) = 60·(12+4) = 960 변수 + 슈팅 제약  → ~83 ms.
RTI(acados) 대신: 틱당 Gauss-Newton SQP **1회**, shift된 직전 해 z로 warm-start:
   x_{k+1}=f_d(x_k,u_k,ŵ)를 직전 궤적 주변 선형화  →  구조화 QP 1개
   HPIPM + (partial) condensing으로 풀이(시간-밴드 KKT 블록 구조 활용)
   → 반복 1회 고정 → 결정적 ~2–5 ms (수렴 루프 없음, 프리즈 없음).
```

**Result.** 이번 턴은 분석만(코드 변경 없음). 제약 유지: **N=60**과 DOB 구조 + 정확성 보존. 구현 시 다음
단계: 이 OCP로 acados 프로토타입 → 로깅 상태에서 acados `u`가 현재 IPOPT `u`와 일치 검증 → 측정 →
`solver="acados"` 스위치로 연결(IPOPT는 레퍼런스/폴백).

---

## 2026-06-15 — 궤적 추종 비교(square): MPC의 pitch 비용

**Why.** DP 비교에서 DOB-MPC의 *원점 유지* 강점(DC 바이어스 제거)을 봤다. 이어서 **square 궤적**(1 m,
10바퀴, JONSWAP+current+kick)을 세 제어기로 돌려, *움직이는 코너형 기준* — 항상 과도 상태인 더 어려운
경우 — 에서의 거동을 봤다.

**무엇을 찾았나 (정상구간, 1바퀴 제외; 기하 off-path = 1 m 사각형까지 거리).**

| 제어 | off-path rms | off-path max | setpoint 오차 | 심도 std | **pitch rms / max** |
|---|---|---|---|---|---|
| PID | 14.3 cm | 45.0 | 39.8 cm | 3.8 cm | 14.2° / **33.5°** |
| MPC | 2.3 cm | 12.7 | 12.7 cm | 1.7 cm | 20.2° / **62.0°** |
| DOB-MPC | 2.1 cm | 17.4 | 12.0 cm | 1.8 cm | 20.5° / **67.2°** |

세 가지: **(1)** MPC/DOB-MPC가 PID보다 사각형을 **~7배 타이트**하게 추종(off-path 2 cm vs 14 cm,
setpoint 오차 12 vs 40 cm)하고 심도도 더 단단. **(2) 대가는 pitch:** MPC 변종이 **62–67°**(거의 뒤집힘)
vs PID 33°. PID는 surge 캡 + slew 제한 + pitch guard로 *추종을 희생해 pitch를 묶고*, MPC(option (a):
pitch 무벌점·무 slew)는 역류 코너를 따라가려 surge를 세게 밀어 → 타이트한 추종이 큰 pitch를 **유발**.
**(3) DOB-MPC ≈ MPC** (2.1 vs 2.3 cm) — 관측기의 차별점은 DC 바이어스 제거인데 이는 원점 유지 현상이라,
늘 움직이는 setpoint에선 이득이 미미.

**함의(방향).** MPC는 정확도↔pitch 축을 PID와 *정반대*로 트레이드하고, 67° pitch는 실제 하드웨어면 제어권
상실 위험. 이는 **option (b)** — 예측모델에 surge→pitch 커플링(`My=−0.0725·Fx`)+복원력을 넣어 MPC가 pitch를
*선제 인지*하고 스스로 억제 — 와/또는 OCP에 명시적 **pitch(또는 surge-slew) 제약**을 추가하는 방향을 구체적으로
가리킨다(7배 추종 이득을 유지하며 pitch를 PID 수준으로).

**Equations (지표 + pitch 비용).**
```
off-path 오차  = 사각형 4변에 대한  dist(p_xy, edge)의 최소값        (기하 형상 오차)
setpoint 오차  = ‖p_xy − p_ref(s)‖,   s = c·(t−t₀)                  (위상지연 포함)
underactuation 비용:  역류 코너에 MPC가 Fx를 올리면  My ≈ −0.0725·Fx
   → 트림 pitch  sin θ* = 0.0725·Fx / (coBM·B)   (option (a)에선 OCP가 안 막음 → 62–67°)
```

**Result.** 분석만(코드 변경 없음). 파일: `recordings/20260615/square_{pid,mpc,dobmpc}_*.csv`,
비교 플롯 `square_compare_*.png`.

---

## 2026-06-15 — 방향 오차 진단 → option (b): pitch-aware MPC

**Why.** "x/y/z는 잘 따라가는데 방향이 틀린다" — 회전 3채널로 분해: **pitch가 지배적 방향오차**(RMS 10–20°,
square max 62–67° = near-tumble); roll ≈1°(저가진 + roll은 제어 가능해 ~0 유지); yaw는 우리 `yaw_ref=0`
런에서 <1°(회전 궤적에서만 폭주 — 별개 이슈, 아래). pitch 근본원인: rank-5 surge→pitch 커플
`My ≈ −0.0725·Fx`가 MPC가 위치추종용 surge를 올릴 때마다 차량을 기울이는데, option (a)는 그 커플을 **surge
결정의 함수로 모델링하지 않고**(DOB-MPC는 실현 pitch 모멘트를 지평 동안 상수로 고정된 외란 `ŵ[pitch]`로만 봄)
pitch를 **제약하지도** 않음.

**What (이론).** option (b): 예측모델이 **자기 surge의 pitch를 선제 인지하고 제약.** MPC가 `surge↑ → pitch↑`를
결정변수의 명시적 함수로 내다보고, 강화된 pitch 상태 바운드가 계획 surge를 암묵적으로 캡 — PID의 수동 surge
리미터의 *최적* 등가물(단, 추종이 실제로 필요할 때만 사용).

**Equations.**
```
예측모델(NED): 커플을 surge 결정 u_surge의 함수로 주입:
   τ_My = +κ·u_surge ,   κ = SURGE_PITCH_COUPLING = 0.0725      (NED 부호 +, 게이트로 검증)
EAOB에 같은 τ_My 주입  ⇒  ŵ[pitch] → 0    (커플이 이제 모델됨, w로 이중계산 안 함)
pitch 상태 제약:  |θ_k| ≤ θ_max  ∀k ,   θ_max = 0.40 rad ≈ 23°
   ⇒ 암묵적 최적 surge 캡:  u_surge ≲ sin(θ_max)·zg·W / κ ≈ 5.9 N
```

**How (구현, 토글).** [dobmpc/mpc.py](../dobmpc/mpc.py) `_f_casadi`가 `τ_My=+κ·u0` + `|θ|` 바운드 1.2→`THETA_MAX`;
[dobmpc_controller.py](../dobmpc_controller.py)가 EAOB에 커플 포함 명령 wrench를 먹여 `w[pitch]→0`(thruster
명령은 `My=0` 유지 — rank-5 할당이 물리적으로 커플 실현); [dobmpc/params.py](../dobmpc/params.py)에 `PITCH_AWARE`
(기본 on; off면 option a)·`THETA_MAX`. `+κ` NED 부호는 `test_dobmpc.test_pitch_aware` 평형 게이트로 검증.

**Result.** DOB-MPC option-a → option-b, 외란 ON:

| 런 | pitch_rms | pitch_max | 위치 | w[pitch] |
|---|---|---|---|---|
| DP (15 s) | 15.0 → 13.4° | 30.0 → **22.9°** | 반경 4.9 → 6.1 cm | 0.22 → **0.09** |
| square (2바퀴) | 17.8 → 12.6° | 46.7 → **23.2°** | off-path 2.6 → 3.0 cm | — |

**pitch max 반감(θ_max로 캡; 풀바퀴 67° → ~23°)** 하면서 위치추종 거의 유지(off-path 여전히 PID의 14cm보다
훨씬 작음), `w[pitch]`도 감소(EAOB가 커플 흡수 안 함). 비용: solver fallback ~5% 증가(하드 pitch 제약이 NLP를
어렵게 함) — *소프트* pitch 제약은 후속 개선. 남은 방향 작업(보류): **회전 궤적의 yaw** = option (A) world-frame
`ŵ`를 예측 각 스텝 `ψ_k`로 회전(회전 중 yaw rate로 낡는 상수-body-`w` Assumption 2 해소) + yaw 가중치
150→300; roll은 이미 작음.

---

## 2026-06-15 — acados SQP-RTI 솔버 이식 (렉 수정, 구현 완료)

**Why.** 위 런타임 진단이 IPOPT NMPC를 병목으로 지목(≈83 ms/틱, 실시간의 0.47배, 2.2 s cold-restart
프리즈)하고 acados RTI를 권고했다. 이번 턴에 그 권고를 **구현**하고 IPOPT는 레퍼런스/폴백으로 유지한다.
제약 유지: **N=60**, DOB 구조, 정확성 보존(acados `u`가 검증된 IPOPT `u`와 일치해야 함).

**What (이론).** 같은 OCP, 다른 *풀이*. IPOPT의 완전수렴 대신 acados **SQP-RTI**: 틱당 Gauss-Newton SQP
**1회**, 직전 해(내부적으로 shift)로 warm-start; 선형화된 QP를 **PARTIAL_CONDENSING_HPIPM**으로 — OCP의
시간-밴드 KKT 블록 구조를 활용 — 풀고; 모델/미분/솔버를 **C**로 코드생성. 외란 wrench `ŵ`는 온라인
**parameter**로 유지(DOB 구조 보존). N=60·dt·Q/R/QN·option-(b) 제약 모두 불변.

**Equations.** (OCP는 MPC 항목과 동일 — 바뀐 것은 *풀이*)
```
지금 틱마다 (acados SQP-RTI):  직전 궤적 z⁻ 주변에서 Gauss-Newton 1스텝:
   QP:  min_Δz  ½·Δzᵀ H Δz + gᵀ Δz   s.t.  선형화된 동역학 + 제약            (H = Gauss-Newton)
        H, g 는 RK4(f(x,u,ŵ)) 선형화에서;  HPIPM + partial condensing 이 밴드형 KKT 풀이
   u₀ ← u₀⁻ + Δu₀ ,   다음 틱 위해 z⁻ ← z 로 shift        → 고정 1반복 → 결정적
적분기: ERK RK4, 구간당 2 substep (h = 25 ms) == mpc._rk4(n_int=2)
상태 제약(roll, pitch=θ_max, |v_lin|)은 SOFT(L2 slack)로 — 과도 선형화가 RTI QP를 infeasible로 만들어
   루프를 멈추지 못하게; 제어 제약은 HARD 유지. (IPOPT는 하드 상태 제약을 썼음.)
```

**How (구현).** 새 [dobmpc/mpc_acados.py](../dobmpc/mpc_acados.py) `AcadosNMPC`가 **동일한** 심볼릭
동역학 `dobmpc.mpc._f_casadi`(단일 소스)를 acados 모델로 재사용; LINEAR_LS 비용 `W=diag(Q,R)`,
`W_e=QN`에 스테이지별 시변 `yref`. 팩토리 [`mpc.make_nmpc()`](../dobmpc/mpc.py)가 acados
(`params.SOLVER="acados"`, 기본) 또는 IPOPT `NMPC`(레퍼런스/폴백 — acados import/빌드 실패 시 자동
폴백)를 반환; [dobmpc_controller.py](../dobmpc_controller.py)가 이를 호출하고 `solve(x, ŵ, xref)→u`
시그니처가 동일해 컨트롤러/EAOB/thruster 경로는 불변.
[dobmpc/_acados_env.py](../dobmpc/_acados_env.py)가 acados 공유 라이브러리를 `ctypes RTLD_GLOBAL`로
**선로딩**해 셸 `LD_LIBRARY_PATH` 없이도 fast path가 동작(teleop 사용자는 아무것도 export 안 함).
툴체인: acados를 `robust` env 안에서 `/home/bdml/acados`에 빌드(C 라이브러리 + `acados_template`
0.5.1); numpy는 <2 유지.

**Result.** 네 방향으로 검증([verify_acados.py](../verify_acados.py)):

| 항목 | IPOPT (레퍼런스) | acados SQP-RTI |
|---|---|---|
| solve / 틱 (N=60) | median **100 ms** (50 ms 예산 초과) | median **0.97 ms**, max 1.1 ms |
| 등가성 (interior 상태) | — | 최악 max\|Δu\| = **0.107 N** vs IPOPT (같은 최적해) |
| 폐루프 DP (15 s, 외란) | 반경 8.6 cm, pitch_max 22.9°, ŵ_x 3.19 N, **프리즈 7회** | 반경 7.0 cm, pitch_max **22.9°**, ŵ_x 3.11 N, **프리즈 0회** |
| 폐루프 square (1 m, 2바퀴, 외란) | ~0.5× 실시간 | 완주, pitch_rms 14°, **프리즈 0회**, **1.2× 실시간** |

**median ~103× 가속**, 결정적(cold-restart 프리즈 제거: `n_fail` 7→0), 폐루프 불변량이 검증된 IPOPT
컨트롤러와 일치(option-b pitch 캡 22.9°, EAOB 추정 `ŵ_x`, DC 류 상쇄). 회귀: `test_dobmpc.py`,
`teleop --selftest`, `test_square_mission.py` 모두 통과. 트레이드오프(권고대로): 컨트롤러 시작 시 ~1 s
코드생성 빌드; RTI feasibility 위해 **소프트** 상태 제약(IPOPT는 하드); RTI는 1스텝 근사 — 여기서 IPOPT로
검증함. IPOPT는 레퍼런스로 선택 가능(`params.SOLVER="ipopt"`). 보류: EAOB 유한차분 Jacobian을 CasADi
autodiff로 이식(≈22→4 ms).

---

## 2026-06-16 — Actuator 현실성 ablation(현실 T200 추진기) + 발견된 acados 취약성

**Why.** 기존 실험은 추진기 힘을 N으로 명령하고 정확히 실현된다고 가정(이상 force 경로). 실물 BlueROV2의
저수준 입력은 정규화 throttle/PWM이고, T200 곡선이 추력으로 바꾸며 **deadband**(~0.7 N 이하 소실 후 ~1.44 N
최소회전 점프), **정/역 비대칭**, **포화**, **모터 lag**, **전압/마모 게인오차**가 있다. 이를 모델하면 sim이
유의미하게 더 현실적이 되는지, 어느 컨트롤러가 가장 강건한지 확인.

**What (구현, opt-in).** 신규 `thrusters.ThrusterModel`(실제 드라이버 체인: T200 역산 → 모터 lag → 정방향
곡선 → `voltage_scale`)을 `set_wrench_command(actuator=)`와 컨트롤러에 옵션 전달(`actuator=None` 기본 —
이상 경로 불변). `ablation_thrusters.py`가 DP(원점, 외란 ON, 5 seed 평균)를 PID/MPC/DOB-MPC에 대해
**ideal / realistic / realistic-LV**(LV=전압 강하 ×0.85)로 실행.

**Result — actuator 현실성은 DP에 MODEST(정상 컨트롤러).** PID·MPC(solver 실패 0) 원점유지 radial RMS [cm],
5 seed 평균:

| ctrl | ideal | realistic | realistic-LV | jitter(std) ideal→LV |
|---|---|---|---|---|
| PID | 14.86 | 14.74 | 15.16 | 10.4 → 12.4 cm |
| MPC | 5.11 | 4.30 | 5.12 | 2.9 → 3.5 cm |

radial RMS는 거의 안 움직임(±7–9 cm seed 산포 내); 보이는 신호는 **jitter(위치 std)가 ~15–20 % 상승** =
deadband 한계진동. 즉 현실 추진기를 넣으면 sim이 조금 더 충실(deadband 채터 포착)하지만 **DP 컨트롤러 순위는
안 바뀜** — 유지 힘이 ~1.44 N deadband floor 근처/이상이고 ~10 ms 모터 lag이 50 ms 제어주기 안쪽이라. (작은
추진기 명령이 deadband를 더 자주 넘는 *이동* 궤적은 더 큰 스트레스 — 후속.)

**Result — ablation이 우연히 acados DOB-MPC 버그를 노출.** seed 평균(노이즈 때문에 필요)이 **seed 3에서
acados SQP-RTI가 `ACADOS_NAN_DETECTED`/`MINSTEP` 폭주(n_fail 116) → 39 cm 발산**, *actuator와 무관*(이상
경로에서도 발생)함을 드러냄. seed별 ideal DOB-MPC: seed 0/1/2/4 = 4.1 / 0.7 / 0.9 / 1.4 cm, n_fail 0(우수);
**seed 3 = 39 cm, n_fail 116**. 단일 seed(0) acados 검증이 놓침: 특정 wave/kick 실현이 EAOB `ŵ`를 RTI QP가
indefinite해지는 영역으로 몰고, 1 iteration이라 회복 불가(stale `u` 유지 → 발산 → 추가 실패). IPOPT 레퍼런스
(완전수렴)는 강건. **오픈 수정(권장): acados NaN 반복 시 해당 틱만 IPOPT 1회로 폴백**(IPOPT는 이미 레퍼런스로
빌드됨) + `ŵ` 강한 clamp / QP 정규화. 수정 전까지 seed 3의 DOB-MPC 수치는 actuator 효과가 아니라 solver 아티팩트.

**Takeaway.** 현실 추진기 모델은 opt-in sim-to-real 스트레스 테스트로 유지 가치 있음(deadband jitter + 가산형
DOB가 못 잡는 곱셈형 추력 강건성 축 추가). 단 DP에선 이상 경로 비교를 뒤집지 않음. 더 시급한 건 acados
DOB-MPC seed-3 NaN 취약성 — IPOPT 폴백으로 수정 예정.

---

## 2026-06-16 — 수정: acados DOB-MPC NaN 취약성 → IPOPT 폴백 + iterate 재초기화

**Why.** 위 ablation이 acados SQP-RTI가 seed 3에서 NaN 폭주·발산함을 발견(n_fail 116, 39 cm): 한 번의
실패가 RTI를 *오염된* iterate에서 warm-start하게 만들어 이후 모든 틱이 실패하고, 붙들린 stale `u`가 차체를
표류시킴.

**What.** acados 실패(NaN / min-step / 비유한 u₀) 시 `AcadosNMPC`가 이제 (1) **acados iterate 재초기화**
(현재 x로 평탄 궤적) → 다음 RTI가 깨끗하게 재시작, (2) **이번 틱을 IPOPT 1회로 복구**(검증된 완전수렴
레퍼런스, 첫 실패 시 lazy 생성). 이전엔 stale `u₀`를 반환 → 발산.

**How.** [dobmpc/mpc_acados.py](../dobmpc/mpc_acados.py): `fallback_ipopt=True`(기본); `_ipopt_fallback()`이
IPOPT `NMPC`를 lazy 생성; `n_fallback`이 복구 횟수 집계; `_warm=False`로 acados 깨끗한 재시작 강제. 무실패
경로는 불변(동일 0.97 ms RTI, 동일 등가성).

**Result.** seed 3 ideal DOB-MPC: **39.04 cm / n_fail 116 → 12.82 cm / n_fail 1** — 폴백 1회가 cascade를
끊음; 잔여는 이제 solver 발산이 아니라 진짜 큰 kick 과도(bounded, 복구됨). seed 0/1/2/4 불변(0.7–4.1 cm,
n_fail 0). 회귀: `test_dobmpc`, `teleop --selftest`, `verify_acados`(등가성 0.107 N, 102.6× 가속) 모두 통과.
트레이드오프: 실패한 틱은 ~100 ms IPOPT 1회 비용(드묾; 하드 실시간이면 폴백을 미리 빌드). acados DOB-MPC가
이제 5개 외란 seed 전부에서 강건.
