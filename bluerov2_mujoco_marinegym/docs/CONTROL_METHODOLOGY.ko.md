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

---

## 2026-06-18 — 현실적 T200 추진기: datasheet 근거 `voltage_scale` + teleop 미션 기본 ON

**Why.** 현실 actuator(`ThrusterModel`: 데드밴드 / 정·역 비대칭 / 모터 지연 / 전압)는 **opt-in(ablation 전용)**
이라, 실로봇(BlueROV2) 거동 예측이 목적인 자율 미션(`--square` / `--goto-origin`)이 **ideal force path**
(commanded == realized)로 돌고 있었음 — 비물리적 이상화. 또한 ablation에서 쓰던 전압 손실 `0.85`는 datasheet
근거 없는 **임의값**이었음.

**What.** (1) 전압 스케일을 공식 datasheet로 **근거화**, (2) 미션 경로에서 현실 모델을 **기본값**으로 하고
`--ideal-thrusters` opt-out 제공(수동 키보드 teleop·`eval_dp`·`ablation`은 별도/명시적 경로라 불변).

**근거(provenance).** Blue Robotics *T200 Public Performance Data 10–20 V (Sep 2019)*
(`marinegym_assets/*.xlsx`; [analyze_t200_voltage.py](../analyze_t200_voltage.py)로 재현 — stdlib zip/XML
파싱, pandas 불필요):

| V | 10 | 12 | 14 | 16 | 18 | 20 |
|---|---|---|---|---|---|---|
| max fwd (kgf) | 2.93 | 3.71 | 4.53 | 5.25 | 6.02 | 6.72 |
| max rev (kgf) | −2.31 | −2.92 | −3.52 | −4.07 | −4.59 | −5.04 |

MarineGym 곡선의 최대(`T200_MAX_FWD/REV` = +6.54 / −5.26 kgf)가 **범위 최상단** → 그 `voltage_scale = 1.0`은
사실상 **~20 V** 추진기. 실제 BlueROV2는 **4S Li-ion(공칭 14.8 V)** 운용이고, 14.8 V 최대 추력(14↔16 V 보간)은
4.81 / 3.74 kgf → `voltage_scale = 14.8V/베이스 = 4.81/6.54 = 0.74(fwd), 3.74/5.26 = 0.71(rev)` → 단일 스칼라
**`NOMINAL_VOLTAGE_SCALE = 0.72`** (만충 16.8 V ≈ 0.83, 방전 13 V ≈ 0.62). **0.85를 대체.**

**How.** [thrusters.py](../thrusters.py): `NOMINAL_VOLTAGE_SCALE = 0.72` 추가(주석에 산정 근거). `ThrusterModel
(voltage_scale=1.0)` 생성자 기본값은 **그대로 둠** — ablation의 명시적 `realistic`(V=1.0) 시나리오·다른 호출부를
조용히 바꾸지 않기 위해. [teleop.py](../teleop.py): `--ideal-thrusters`(opt-out)·`--thruster-voltage`(기본
`0.72`) 플래그 추가; 미션 분기에서 `ThrusterModel(lag=True, voltage_scale=…)` 1개를 만들어 `DOBMPCController`·
`PoseController` 양쪽에 `actuator=` 주입(배선·`reset()`은 이미 존재); 시작 로그로 활성 경로 출력; 기록
매니페스트(`.meta.json`)에 `run.thrusters = {model, lag, voltage_scale}` 기록해 ideal/realistic 혼동 방지.

**Result.** 폐루프 DP(dobmpc, seed 0, 20 s, 외란 ON): **ideal radial 5.02 cm / jitter 4.30 cm → realistic
×0.72 radial 7.76 cm / jitter 6.17 cm**, `n_fail 0` — 현실 단계가 예상대로 station-keeping을 악화(데드밴드
jitter + 가산형 DOB가 부분만 잡는 28 % 추력 부족), solver 문제 없음. `analyze_t200_voltage.py`가 표와 `0.72`를
재현(MATCH). 회귀: `eval_dp`(ideal 기본)·`ablation_thrusters` 시나리오 불변; `teleop --selftest` 통과.

**범위/정직성.** MarineGym 곡선을 14.8 V로 **재적합하지 않음**(그러면 `T200_MAX`/`ctrlrange`가 전부 바뀜) —
verbatim ~20 V 곡선을 두고 스칼라만 적용. fwd/rev 전압비(0.74 / 0.71)는 단일 0.72로 근사. 유입속도 의존
(advance ratio: 움직이면 추력 감소)·열·파울링은 **범위 밖** — 현실 모델도 여전히 *정수(bollard)* 곡선. 데드밴드를
DP보다 더 자극하는 *이동* 궤적(square) 실행이 자연스러운 후속.

---

## 2026-06-18 — BlueROV2 → BlueROV2 **Heavy**: 8추진기, 완전작동, 6-DOF MPC

**Why.** 표준 벡터드-6 BlueROV2에서 **Heavy** 구성으로 전환. 핵심은 작동성: Heavy는 **수직 추진기 2개를 추가**
(총 4개, 네 모서리)해 allocation이 **rank 6 = 완전작동** — roll·pitch를 직접 제어할 수 있어, rank-5
underactuation과 그에 따른 option-(a)/(b) pitch 우회가 **사라짐**. 두 변형 모두 유지하고 env var
**`ROV_MODEL`**(`bluerov2` 기본 | `heavy`)로 선택; 신규 [rov_model.py](../rov_model.py)가 단일 진실원이라 플랜트
(MJCF/hydro)와 컨트롤러(params/NMPC)가 모델을 두고 어긋날 수 없음.

**Provenance.** 모든 값은 MarineGym USD(`.../BlueROVHeavy.usd`, `pxr`로 파싱)에서 직접 검증. Heavy는 hydro 계수
(added mass, 선형/2차 감쇠)와 T200 추진기를 BlueROV2와 **동일**하게 유지 — 질량·관성·부력 volume·추진기 배치만
다름. (Heavy yaml의 `force_constants` 0.8e-7은 추력을 ~18%로 줄이지만, 물리적 추진기가 같으므로 검증된 T200
곡선을 유지 — [03_THRUSTERS.md](03_THRUSTERS.md).)

### 무엇이 바뀌었나 — 값 (기존 → 수정)

**강체 질량행렬 M_RB = diag(m, m, m, Ix, Iy, Iz):**
```
BlueROV2:  diag( 11.2, 11.2, 11.2,   0.30375, 0.626,  0.5769 )
Heavy:     diag( 11.5, 11.5, 11.5,   0.3291,  0.6347, 0.6109 )   (관성 유도 — 아래 참조)
```

**관성 I (대각):  bluerov2 → Heavy, 평행축으로 유도:**
```
[ 0.30375                ]        [ 0.3291                ]
[         0.626          ]   →    [        0.6347         ]   I_heavy = I_bluerov2 + Δ
[                 0.5769 ]        [                0.6109 ]   Δ = [+0.0254, +0.0086, +0.0340]
```

> **⚠ Heavy USD 관성을 안 쓰고 이 값을 유도한 이유.** MarineGym/farol Heavy USD는 `[0.21, 0.245, 0.245]`를
> 싣는데 Heavy가 더 무거운데도 BlueROV2보다 *작다* — 물리적으로 거꾸로. 이건 **Gazebo 안정화용 손튜닝 리터럴**:
> farol 원본 `bluerov_heavy_vehicle/urdf/base.xacro`가 이를 하드코딩하고 주석에 *"... otherwise your model
> will become unstable on Gazebo"* 라고 명시(바로 아래 물리 타원체 공식은 주석처리; 또 다른 dsor 출처는
> `[0.26, 0.23, 0.37]`로 출처마다 제각각). BlueROV2 `[0.30375, 0.626, 0.5769]`는 다른 URDF
> (`bluerov2_description`)에서 옴.
>
> 그래서 farol 리터럴을 믿는 대신, BlueROV2 텐서로부터 **Heavy 전용 텐서를 유도**. 수평 4개 추진기는 두 모델에서
> 위치가 동일해 BlueROV2→Heavy 관성 *차이*에서 상쇄됨; 바뀌는 건 **수직** 배치뿐 — BlueROV2의 중앙 2개
> (`±0.1105` y) → Heavy의 모서리 4개(`±0.12` x, `±0.22` y). 각 추진기를 **0.15 kg 점질량**으로 보면(모델
> 정합: +0.3 kg / +2추진기 = 11.2→11.5), 차이는 평행축 항
> `Δ = Σ_heavy m·(평행축) − Σ_bluerov2 m·(평행축)`:
> `I_heavy = I_bluerov2 + [+0.0254, +0.0086, +0.0340] = [0.3291, 0.6347, 0.6109]`. 이는 BlueROV2 기준값이
> 자기 추진기를 포함하든 안 하든 성립(선체 + 수평 4개는 상쇄). 추진기 질량 가정은
> [compute_heavy_inertia.py](../compute_heavy_inertia.py)로 재현/조정(민감도: m_v 0.10→0.344 kg → Ixx
> 0.321→0.362).
>
> **정직한 한계:** 이건 물리적으로 동기화된 *추정치*(점질량 추진기; BlueROV2 기준값 자체의 CAD-vs-공식 출처도
> 미확인 — config.yaml은 `bluerov2_description` 상류를 지칭하나 웹에서 정확 텐서 미발견), Heavy CAD 측정값이
> **아님**. 단 Heavy 전용이고 물리가 요구하는 대로 엄격히 **≥ BlueROV2** — farol 리터럴이나 BlueROV2 평면
> 재사용보다 근거가 확실히 낫다.

**부가질량 M_A = diag(Xu̇, Yv̇, Zẇ, Kṗ, Mq̇, Nṙ):  불변**
```
diag( 5.5, 12.7, 14.57, 0.12, 0.12, 0.12 )      (두 변형 동일)
```

**총 질량행렬 M = M_RB + M_A:**
```
BlueROV2:  diag( 16.70, 23.90, 25.77,  0.42375, 0.746,  0.6969 )
Heavy:     diag( 17.00, 24.20, 26.07,  0.4491,  0.7547, 0.7309 )
```

**감쇠 D_L, D_NL:  불변** — D_L = −diag(4.03, 6.22, 5.18, 0.07, 0.07, 0.07),
D_NL = −diag(18.18, 21.66, 36.99, 1.55, 1.55, 1.55).

**부력:** volume 0.0113459 → **0.0116499** m³ ⇒ B = ρgV 110.97 → **113.94** N. Heavy는 더 무겁고(W 109.87 →
112.82 N) 동시에 더 커서 **net buoyancy는 ~+1.1 N 유지**(B−W = +1.10 → +1.13 N).

**Allocation B (wrench = B·thruster_forces):**
```
BlueROV2:  6×6,  rank 5   — pitch My 독립 제어 불가 (underactuated)
Heavy:     6×8,  rank 6   — 완전작동 (검증: 순수 pitch wrench가 My=1.000 실현)
```
추진기: 수평 4개(thruster_0..3) 동일; 수직이 바뀜:
```
BlueROV2:  수직 2개 ( 0.0025, ±0.1105, −0.005)
Heavy:     수직 4개 (±0.12,   ±0.22,   −0.005)   (+Z, 네 모서리 → Fz/roll/pitch 독립)
```

**컨트롤러(NMPC) — 완전작동 활용:**
```
NU (제어 차원):    4  [X,Y,Z,N]        →  6  [X,Y,Z,K,M,N]
tau 매핑:          [X,Y,Z, 0, κ·X, N]  →  [X,Y,Z, K, M, N]   (κ = surge→pitch 커플링, 제거)
PITCH_AWARE(opt-b surge cap):  True    →  False   (이제 pitch는 명령 DOF)
MPC_Q roll/pitch 위치 가중치:  0,0      →  80,80   (MPC가 능동으로 수평 유지)
```

**결과 (DP, dobmpc, seed 0, 20 s, 외란 ON):**
| 변형 | radial RMS | pitch 평균 | pitch 최대 |
|---|---|---|---|
| BlueROV2 (rank-5) | 5.0 cm | **+11.8°** | 22.9° |
| Heavy (완전 6-DOF, 관성 유도) | **3.3 cm** | **+0.4°** | 5.4° |

완전작동이 **pitch를 능동으로 수평 유지**(trim 11.8° → 0.8°)하고 station-keeping도 개선(5.0 → 3.3 cm).

**How / blast radius.** [rov_model.py](../rov_model.py)(레지스트리); [bluerov_heavy.xml](../bluerov_heavy.xml)
+ [BlueROVHeavy.yaml](../marinegym_assets/BlueROVHeavy.yaml); [params.py](../dobmpc/params.py)(모델별
MASS/I/VOL, NU, U_MAX, MPC_Q/R, PITCH_AWARE); [mpc.py](../dobmpc/mpc.py)(NU + NU-aware tau);
[mpc_acados.py](../dobmpc/mpc_acados.py)(모델별 codegen 디렉터리); [thrusters.py](../thrusters.py)(allocation이
모델에서 6/8 추진기 발견); [hydro.py](../hydro.py)(모델별 yaml/volume);
[dobmpc_controller.py](../dobmpc_controller.py)(NU-aware wrench); [teleop.py](../teleop.py) /
[eval_dp.py](../dobmpc/eval_dp.py)(XML + ThrusterModel n). BlueROV2 경로는 수치적으로 불변(회귀: dobmpc DP
5.0 cm, test_thrusters/test_dobmpc/test_controller, `teleop --selftest` 두 변형 모두 통과).

**범위/후속.** PID 베이스라인([controller.py](../controller.py))은 두 변형 모두 surge/sway/heave/yaw만 명령
(Heavy에선 rank-6 allocation이 surge→pitch 커플링을 수동 상쇄해 PID도 더 평평) — 완전 6-DOF PID는 후속. Heavy
MJCF는 BlueROV2 시각 메시 재사용(시각만; 동역학은 Heavy inertial + 8 site). roll/pitch U_MAX(8 Nm)·Q
가중치(80)는 초기값, 튜닝 여지. actuator 현실성 ablation / verify_hydro는 아직 BlueROV2 대상 — Heavy 확장은 후속.

## 2026-06-29 — 유한수심 외란 환경 + 12-run 제어기 비교 (current/drift/waves)

얕은 수심(h=4 m) Monterey Bay·너울 우세 환경에서 PID/MPC/DOB-MPC 외란 제거를 공정 비교하기 위해
**유한수심 방향불규칙파 + 해류(평균+표류) + Froude-Krylov 관성력** 외란 환경을 추가. 기존
`disturbances.py`(deep-water, k=ω²/g)는 천해 타깃(kh≈0.34)에 부적합 → 신규 `disturbance/` 패키지.

**물리 분해 (underwater-robotics·control-theory 자문 검증).**
- Drag: hull 감쇠 D(nu_r)(상대속도)가 이미 Morison drag와 동치 → fluid 속도를 `water_velocity`로만 주입,
  별도 `0.5ρC_D A v_rel|v_rel|`는 **추가 안 함**(이중계산 방지).
- Inertia: hydro의 `−M_A·d(nu_r)/dt`가 added-mass(C_a) 부분을 이미 공급 → 누락분인 **Froude-Krylov
  ρ∀·a_wave (C_M=1)만** 외력으로 주입. (슬라이드의 평탄 C_M=1.5는 surge에만 맞아 폐기; 축별 C_a=[0.49,1.12,1.29].)
- 유한수심: `ω²=g·k·tanh(k·h)`(뉴턴해) + cosh/sinh 깊이 프로파일 → kh≈0.34에서 k 3배 오차/해저 수직속도 미소멸 해소.

**5 모드** (같은 seed → 파동 위상·GM 표류 비트동일, 레이어만 토글): NONE / C / CD / CW / CDW.
**NONE = still water** (current·drift·waves 모두 off) = 외란 없는 baseline, 나머지는 current / +drift /
+waves / +drift+waves. **kick 제외**.

**결과 (smoke, DP, seed 0, 8 s — 정성 검증):**
| mode | PID | MPC | DOB-MPC | DRR=MPC/DOB |
|---|---|---|---|---|
| C (current) | 16.5 | 2.9 | **0.39 cm** | 7.4 |
| CDW (+drift+waves) | 13.3 | 3.2 | **0.47 cm** | 6.8 |

DOB-MPC가 DC 해류/표류를 거의 완전 제거(EAOB w_x≈1.5 N 추정 = 0.2 m/s drag, est_err 0.01–0.06 N),
wave-band 잔류만 증가(band_wave 0.21→0.44 cm). square(추종)에선 nu_ref=0 구조적 lag가 지배 → DOB≈MPC(11 cm).

**How / blast radius.** 신규 [disturbance/](../disturbance/){waves,current,env,config,test_*}.py,
[experiments/run_compare.py](../experiments/run_compare.py), config/{base,scenario_square}.yaml. 기존 동역학
수정은 [hydro.py](../hydro.py)의 read-only `diag_wtrue` 진단 1곳(`w_true_world` = plant힘−still-water모델힘+FK,
힘 불변)뿐 — hydro의 duck-typed disturbance 인터페이스에 drop-in. 단위검증 34 assert + smoke 통과.
실행: `python -m experiments.run_compare --config config/base.yaml`.

**범위/후속.** 슬라이드(von Benzon) M_RB/M_A는 현 MarineGym 동정값과 달라(질량 11.2 vs 13.5 등) 별도 변종
`ROV_MODEL=bluerov2_vonbenzon`으로 일관 채택 예정(질량·관성·부피/부력·부가질량 + params + 재튜닝 + 재검증).
`fk_mode=morison_ca`(축별 완전 Morison)는 검증 sweep용으로 보존(기본 froude_krylov).

## 2026-06-29 — Heavy 기본 전환 + DOB-MPC 궤적 레퍼런스(속도 FF + heading-follow)

square 임무를 실시간(`experiments/run_viewer.py`)으로 보다가 나온 3개의 연동 변경: 완전구동 **Heavy**를 표준
기체로, square에서 ROV가 **진행방향을 바라보게**, 그리고 **DOB-MPC의 큰 square 추종오차**를 해결.

**1. Heavy를 프로젝트 전역 기본으로.** [rov_model.py](../rov_model.py): `ROV_MODEL` 기본값 `bluerov2`
→ **`heavy`**(8추진기, rank-6 완전구동, NU=6). 하위(hydro YAML, `dobmpc/params`, allocation, 변종별 acados
codegen)는 전부 `RM.*` 경유라 자동 추종 — end-to-end 확인(`heavy 8 True`, allocation 6×8 rank 6, heavy acados
RTI solver 기빌드). `test_load.py`는 `bluerov.xml` 하드코딩(bluerov2 전용 phase-1 점검)이라 영향 없음.
`ROV_MODEL=bluerov2`로 되돌림.

**2. heading-follow + 코너 스무딩**(`experiments/run_compare.py` square 분기 + `run_viewer.py`).
square에서 yaw 레퍼런스를 경로 접선 `atan2(ty,tx)`로 추종(진행방향 바라봄), **슬루-레이트 제한**으로 90° 코너
전환을 부드럽게 램프(`slew_heading`, 기본 **60 °/s** ≈ 1.5 s/코너) — 위치 경로는 날카로운 square 유지(각도만).
square 블록 config: `heading_follow: true`, `yaw_rate_deg_s: 60`; 뷰어 플래그 `--heading {follow,fixed}`,
`--yaw-rate`. 검증: 기록된 `yaw_deg`가 ≤3.1°/log로 램프(90° 도약 없음), 네 코너에서 0→90→±180→−90 진행.

**3. DOB-MPC `_xref_ned`: 정점 타일 → 수평선 궤적 레퍼런스**(핵심 수정).
큰 square 오차의 원인: NMPC 레퍼런스가 `nu_ref = 0`을 늘 두고 **`v_ref`를 무시** → 순수 DP(정점유지) 제어라
움직이는 목표를 구조적으로 lag(코드 주석이 "a follow-up"으로 보류해둔 부분).
[dobmpc_controller.py](../dobmpc_controller.py) `_xref_ned()`가 horizon step마다, 전부 **런타임 레퍼런스만
(모델·코스트 불변 → acados 재빌드 없음)**:
- **속도 FF**: `nu_ref = [ S·(R(yaw_ref)ᵀ·v_ref) ; 0 ]`(world-FLU 경로속도 → 레퍼런스 body 속도, FRD) — MPC가
  움직임을 더는 억누르지 않음;
- **위치 미리보기**: `p_k = p_ref + v_ref·(k·DT_CTRL)` 외삽(위치·속도 레퍼런스 자기일관 = v_ref로 움직이는 점);
- **yaw unwrap**: `psi_ref ← psi_now + wrap(psi_ref − psi_now)`로 NMPC 코스트(각도 wrap 없음)가 ±π를 짧은 쪽으로
  통과 — heading-follow를 MPC에서 정상화(없으면 한 코너에서 ~270° 헛돔).

`v_ref = 0`이면 `_xref_ned`가 **정확히** 기존 정점 타일로 환원(unit-assert `allclose 1e-12`) →
**DP/정점유지 결과 불변**(회귀 안전). 프레임 주의: `nu`는 body(FRD), `v_ref`는 world(FLU) → 회전 필수.
EAOB는 실제 명령 wrench를 보므로 무관.

**결과 (heavy, DP T=20 s / square 2바퀴, mode C, seed 0):**
| run | radial RMS |
|---|---|
| DP dobmpc (회귀) | **0.00 cm** (완전구동 + DOB가 해류 제거) |
| square dobmpc (FF + heading) | **1.87 cm** |
| square mpc (FF) | 3.63 cm |
| square pid (기준) | 25.1 cm |

DOB-MPC square 오차가 lag 지배(FF 전 수십 cm)에서 **~1.9 cm**로 감소; 남은 오차는 코너 transient(직선 미리보기가
코너 너머 과대예측) + 파동/해류 외란 잔차(EAOB 몫, 레퍼런스 FF와 별개).

## 2026-06-29 — NONE(잔잔한 물) baseline 모드 추가 + 비교 그림 정비

**5번째 외란 모드 `NONE` = still water**(current·drift·waves 전부 off)를 외란 없는 baseline으로 추가.
[disturbance/env.py](../disturbance/env.py)에 `use_current` 플래그를 신설:
`MODES = ("NONE", "C", "CD", "CW", "CDW")`. NONE에서 `water_velocity`/`external_wrench`는 정확히 0.
모드 코드는 PyYAML `safe_load`를 거치므로 YAML-1.1 bool/null 단어(`N`/`n`/`off`/`yes` 등)를 피해야 함 → 그래서
풀어 쓴 `NONE`. `config/base.yaml`, `config/scenario_square.yaml`, `run_viewer.py`/`plot_trajectories.py`의
`--mode` choices에 모두 반영.

**이유:** 다른 모드들은 제어기의 *고유 추종 지연*과 *외란 제거*를 한데 묶음. NONE은 전자를 분리 — 아무것도
밀지 않을 때 각 제어기가 낼 수 있는 최선을 막대그래프의 기준점으로 제공.

**결과 (square, current heading 10개, seed 0; recording `compare_20260629_113356`) — radial RMS [cm]:**
| mode | PID | MPC | DOB-MPC |
|---|---|---|---|
| **NONE (still water)** | **19.0** | **1.9** | **1.9** (std 0 — 결정론적) |
| C (current) | 29.5 | 16.4 | 14.6 |
| CDW (+drift+waves) | 67.5 | 31.1 | 21.4 |

모든 제어기에서 NONE ≤ C (당연). 잔잔한 물에선 MPC ≈ DOB-MPC(EAOB가 추정할 외란 없음, `est_err ≈ 0.04 N`,
DRR 0.99); PID의 19 cm는 외란이 전혀 없을 때의 순수 코너 추종 지연. NONE은 결정론적(돌릴 해류 없음, seed 고정)
→ 10개 heading에서 동일하므로 막대에 오차막대가 없음.

**그림:** [experiments/run_compare.py](../experiments/run_compare.py)의 `fig_bars()`를 논문용으로 재작성 —
팔레트 정비, 막대별 값 라벨, 각 모드 의미를 적은 2줄 x축 라벨(`NONE (still water)`, `C (current)`, …),
상·우 축선 제거, 가로 범례, 200 dpi. recording의 `bar_square_radial_rms.png` + `results.csv`/`results_raw.csv`를
NONE 포함해 재생성(기존 4모드 데이터는 보존, NONE 추가).

## 2026-07-03 — 해석적 극배치로 PID 재튜닝 (heavy) + yaw-rate 레퍼런스 FF

2026-07-02 적대검증 완료된 극배치 설계를 [controller.py](../controller.py)에 적용. `DEFAULT_GAINS`는 이제
**모델 변종별**: `GAINS_HEAVY`(신규) vs `GAINS_BLUEROV2`(2026-06-14 hand-tuned, 그대로 유지),
`rov_model.FULLY_ACTUATED`로 선택 — 분리의 동기가 된 구조적 속성이 바로 이것(기존 surge 6 N 캡은 rank-5
pitch 커플링 유물로, rank-6 heavy에는 불필요하지만 bluerov2에는 여전히 필요).

**설계 (heavy):** 축별 hover 선형화 `m_eff·a = F − d_lin·v`; 폐루프 특성다항식
`m·s³ + (d+Kd)·s² + Kp·s + Ki`를 `(s² + 2ζωn·s + ωn²)(s + αωn)`에 정합:

```
Kp = m_eff·(1+2ζα)·ωn²   Kd = m_eff·(2ζ+α)·ωn − d_lin   Ki = m_eff·α·ωn³
ωn = 병진 2.0 rad/s (JONSWAP 에너지 대역 0.45–1.2 위), yaw 3.0; ζ = 0.9, α = 0.2
```

→ 수평 **등방** kp/kd/ki = 131.6/90.6/38.7 (sway m_eff = 24.2 kg 기준 설계; body-x는 ωn ≈ 2.78, ζ ≈ 1.0
실현), heave 141.8/99.1/41.7, yaw 8.95/4.32/3.95. 등방화는 장식이 아니라 유효성 조건: world-frame PD는
kp_x = kp_y일 때만 yaw와 가환이고, 주 시나리오가 square + heading_follow. 게인과 함께 나가야 하는 동반변경
(없으면 설계 불성립): surge `f_max` 6→30 N, `e_gate` 0.5→0.15 m, `surge_slew` 30→120 N/s, 그리고
**yaw-rate 레퍼런스 피드포워드** — 두 컨트롤러의 `set_target()`이 `r_ref`를 받고 PID yaw 법칙은
`mz = kp·e_yaw − kd·(r − r_ref)`; [run_compare.py](../experiments/run_compare.py) /
[run_viewer.py](../experiments/run_viewer.py)가 heading-slew 속도(코너에서 60°/s, 직선에서 0)를 넘겨
D항이 명령된 회전에 저항하지 않음. `DOBMPCController.set_target`은 인터페이스 일치용으로 `r_ref`를
받기만 함(NMPC는 `yaw_ref`만 추종).

**결과 (heavy):** `--smoke` DP radial 1.4–1.5 cm (NONE/C/CD; wave 모드 25 cm는 5초 과도응답 수치);
run_viewer 1랩 square NONE: **radial RMS 1.34 cm** (t>5 s) vs 구게인의 `compare_20260702_222150`
**17.8 cm** — 약 13배 — max 4.0 cm (코너 과도), |pitch| 0.0°, 포화 채터링 없음. `verify_meta.py` 통과;
`meta.json` 사이드카에 새 게인 세트 기록됨. bluerov2 `--smoke` 무변화(레거시 경로).

**유효 범위:** sim 한정(이상 추진기, 완전 상태, 500 Hz). 하드웨어에선 ωn을 1.0–1.5 rad/s로 낮추고 D항
필터링 필요. wave 대역 잔류 |S(1.6)| ≈ 0.41 너머는 PID 게인이 아니라 DOB-MPC의 몫.

---

## 2026-07-07 — DOB-MPC yaw 레퍼런스 완성 (선회 시 yaw preview + yaw-rate FF)

**왜.** 사각형에서 DOB-MPC의 CW/CDW 최대 오차는 **(1,1) 상류 코너**에 집중(17 cm, seed 0 /
current 0° / wave 0°) — 코너 0.36 m 지난 지점의 cross-track 처짐. 원인은 절반만 완성된 기능:
2026-06-30 `_xref_ned` 업그레이드(정지-자세 → 이동-경로)가 **병진** 피드포워드(위치 preview +
선속도 `v_ref`)만 넣고 **회전** 절반을 안 만듦. yaw 레퍼런스는 호라이즌 내내 상수, 각속도
레퍼런스는 하드 `zeros(3)`; 2026-07-03의 `r_ref` yaw-rate FF는 PID에만 배선
(`DOBMPCController.set_target`는 "인터페이스 parity용으로만 받고 무시"). 그래서 NMPC가 60°/s로
슬루하는 yaw 목표를 zero-rate 위치 레귤레이터로 쫓아 → 슬루 지연(1.8 s, nominal 1.5 s)·~199°
오버슈트, 이 미스히딩 창을 상류 edge의 파도 drag가 파고듦.

**무엇.** `_xref_ned`가 선회 시 회전 레퍼런스를 채운다(기존 선속도 FF 미러링):
- **yaw-rate FF** → `xref[11,:] = -r_ref` (world-FLU +yaw-rate → NED/FRD `r`, `S=diag(1,-1,-1)`
  부호 반전; Q 가중치 10);
- **yaw-angle preview** → `xref[5,k]`가 현재 NED command yaw에서 레퍼런스 rate로 최종 edge
  heading `yaw_target`까지 ramp하되 거기서 **clamp**(Q 가중치 150) — 3 s 호라이즌 내내 상수 rate
  ramp하면 90° 선회를 180°+로 외삽해 **과회전**하므로 필수. `run_compare`/`run_viewer`가
  `yaw_target = atan2(ty, tx)`(edge tangent) 전달. `set_target(yaw_target=None)`은 현재 yaw로
  기본값 → preview 없음(안전).
- **게이트:** `r_ref == 0`(직선 / DP)이면 정확히 no-op → DP/정지유지 byte-identical
  (신규 `test_dobmpc.test_xref_yaw_preview`가 DP-등가 + `-r_ref` 부호 + clamp 검증).
- 런타임 전용(`yref`); acados 재빌드 없음. 가중치 불변.

**결과 (A/B, dobmpc, seed 0, current 0° / wave 0°, run_viewer headless).** (1,1) 코너 yaw
슬루-to-5° **1.85 → 1.45 s**(≈ nominal), NONE·CDW 공통 — 지연 메커니즘 수정됨. CDW (1,1)
top-edge **최대 17.2 → 13.2 cm (−23%)**; 전체 radial RMS 불변(CDW 3.13→3.16, NONE 2.08→2.07 —
회귀 없음; 정지수 코너별 peak ±0.3–1.7 cm, 무외란 코너 과도는 yaw 지연이 아니라 *위치*-레퍼런스
kink 지배). 잔존 ~13 cm는 **상대속도 파도 drag(M2) + EAOB 파도대역/프레임 한계(M3)** — 이 변경
범위 밖(M1/M4만). 문서의 deferred "yaw on turns"(ŵ를 예측 ψ_k마다 회전 = M3b)는 상보적이며 여전히
열림. 유닛 전체 통과(heavy: frames/predictor/EAOB/yaw-preview; bluerov2: 전체). 메모리
`dobmpc-corner-deviation` 참조.

---

## 2026-07-12 — heavy_gripper 변종: 실제 페이로드(Newton 그리퍼 + MarineSitu C3)를 세 번째 플랜트로

`ROV_MODEL=heavy_gripper` 추가 = heavy + 실험실의 실제 페이로드. **손튜닝이 아니라 합성**: 벤더 검증 질량
(Newton Subsea Gripper 공기중 524 g / 수중 267 g, 케이블 포함; MarineSitu C3 1700 g / 430 g — 둘 다
Blue Robotics 스토어 페이지, 교차 확인)을 평행축 합성([compute_payload_inertia.py](../compute_payload_inertia.py))
에 통과. MJCF는 [gen_gripper_variant.py](../gen_gripper_variant.py)가 `bluerov_heavy.xml`에서 *생성*(기준선
과 드리프트 불가). 질량 13.724 kg, I=[0.363, 0.749, 0.699], 배수량 0.0131815 m³ → **순부력 −5.7 N (가라앉음
— 트림 폼 없음, 결정사항)**; hydro added-mass/감쇠는 heavy 세트 유지(페이로드 증분은 축당 수 %~20% = 공개
세트 간 30~100% 편차 미만; DNV-RP-C205 build-up 추정치는 YAML에 문서화, 자체 system ID까지 HOLD). 조(jaw)는
미러된 슬라이드 관절 2개 + `position` 액추에이터 1개(**ctrl index 8**) — 스러스터 코드는 이름(`thr0..7`)으로
찾으므로 할당/teleop 무영향. 온보드 카메라 3개(스테레오 7.5 cm + 센터)가 전방 45° 아래 태그 바닥을 봄.

**뼈아픈 교훈 2개 (둘 다 [test_heavy_gripper.py](../test_heavy_gripper.py)가 회귀 가드):**

1. **origin == COM은 관례가 아니라 스택 전체의 가정.** 첫 빌드는 heavy 프레임을 유지한 채 COM만 3.3 cm
   이동 — PID는 버텼지만 NMPC 폐루프는 1초 내 발산. 예측 모델(`fossen.py`, `params.ZG_MASS=0`)과 hydro가
   전부 원점=COM을 가정하고, 미모델 m·r 회전-병진 커플링(0.45 kg·m vs Ixx 0.36)은 작지 않았다. 해결:
   생성 프레임을 **합성 COM에 재정렬**(전 site/geom −COM 이동; 기체-상대 기하 불변; 할당 B는 물리적으로
   맞는 새 모멘트암을 자동 획득).

2. **hydro.py는 body_iquat=identity(대각 관성)를 암묵 요구.** 합성 관성을 `fullinertia`(Ixz=−0.0016)로
   내보내자 MuJoCo가 주축을 정렬하며 관성 프레임이 body 축의 **순열**이 됨 — `mj_objectVelocity(mjOBJ_BODY)`
   는 그 순열 프레임으로 ν를 보고하는데 hydro는 `xmat`(body)로 drag를 적용. 축이 엇갈린 drag는 소산 대신
   **에너지 펌프**: 무토크 0.5 rad/s 피치 킥이 1.5 s 만에 |q|>60 rad/s로 폭발. ablation으로 격리(부력만=안정,
   drag만=폭발; heavy 대조군 안정). 해결: **diaginertia**로 출력(Ixz 0.4%는 계수 불확실성 이하); hydro
   근본 수정은 KNOWN_ISSUES에 기록.

**제어기 갱신:** `GAINS_HEAVY_GRIPPER` = 동일 pole placement를 페이로드 질량으로 재유도(sway m_eff 26.42 →
kp/kd/ki 143.7/99.5/42.3; heave 28.29 → 153.9/108.0/45.3; yaw I_eff 0.811 → 9.92/4.79/4.38) + 옵션 **roll/pitch
leveling PD**(`rp_kp=(4.3,7.7)`, `rp_kd=(2.6,4.6)`, 게인 dict 게이트라 heavy/bluerov2 동작 불변): 페이로드의
정적 자세 토크(jaw 중력 + CB_x 오프셋 ~1.3 N·m)가 수동 복원(B·coBM ~1.2 N·m/rad)과 맞먹어 수동만으론 자세가
흘러감. dobmpc params는 레지스트리에서 자동 추종(MPC 가중치 무변경 — 초기의 "자세 가중치 완화" 가설은 교훈
2의 오진이라 되돌림).

**결과:** PID DP hold 0.0 cm (still, 20 s); DOB-MPC DP hold rms(12–20 s) **1.3 cm** still / **1.4 cm**
current+waves / 22 cm (20–50 N Poisson kick 포함 — 임펄스 회복 과도; EAOB는 kick 예측 불가, 기존 분석과 일관).
동일 하니스 heavy 참조: 0.4 cm (cw). acados-vs-IPOPT worst-case |Δu| 0.2717 N (heavy 기준 0.25 N 게이트 근소
초과; authority의 ~0.9% — KNOWN_ISSUES 기록). 3변종 전체 회귀(selftest/load/thrusters/controller/hydro/
observe/water_viz) 통과; heavy·bluerov2 파일 무변경.

---

## 2026-07-19 — heavy_gripper: C3를 실측 마운트 위치(전방-하단, 렌즈 전방 수평)로 이동 — Onshape CAD 기반

2026-07-12부터 쓰던 C3 배치(전방-상단, 45° 하향, `C3_POS=[0.18, 0, 0.09]`)는 추정값이었고 실물과 전혀 달랐다.
실험실 Onshape 어셈블리(BROV2 Heavy + C3-BR 브래킷 위 C3)에서 끝까지 실측: **onshape-to-robot**(v1.8.2,
MuJoCo 출력; export는 `assets/CAD files/onshape_export/`에 보존)으로 내보낸 뒤, 차체 지오메트리를 sim
base_link 프레임에 **정합** — 회전은 순수 축 치환으로 고정(CAD가 축 정렬 상태; bbox가 벤더 치수 575×254×457 mm와
정확히 일치), 평행이동은 복셀 점유 전역 격자탐색(ICP 국소최소 회피), 스케일 1.0233 보정(MarineGym 유래 스킨이
균일하게 2.3% 큼; 배치는 COM 앵커 기준 **실측 metric**), trimmed-ICP 미세조정: 잔차 1.6 mm, 시드 간 편차
<0.1 mm, 전후 방향은 전자튜브 돔 형상으로 판별. 파이프라인 상수는 [process_c3_mesh.py](../process_c3_mesh.py)에
동결, 이제 `meshes/c3_payload_frames.json`을 생성해 생성기가 소비(`CP.C3_POS`와 불일치 시 에러).

**결과** (base_link FLU, 원점 = 차체 COM): C3 메쉬 중심 `[0.199, 0.008, −0.156]` — **전방-하단 중앙선**,
하우징이 프레임 코보다 ~9 mm 앞, 스키드 라인보다 ~3 cm 아래로 돌출; 광축 `[1.000, 0, 0.0056]`(CAD mate의
0.32° 상향 틸트 그대로 유지); 스테레오 베이스라인 수평. C3-BR 브래킷이 Newton 그리퍼 튜브를 **~1 mm 간격**으로
감싸고 지나감 — 추정값이었던 `GRIP_POS=[0.25, 0, −0.17]`가 실물 브래킷과 양립함을 독립 확인. MJCF 카메라 3개는
이제 **렌즈 평면**(`x=0.2395`, `y=0.0055±0.0375`, `z=−0.1554`)에서 전방 수평을 봄; 그리퍼 죠가 ~18 cm 앞 정면.

**모델 영향:** 합성 관성 diag `[0.38154, 0.77780, 0.70954]`(이전 `[0.363, 0.749, 0.699]`), TOTAL COM
`[+0.0349, +0.0010, −0.0258]`, coBM **+0.01625 m**(이전 +0.00955 — COM이 내려가고 CB는 거의 그대로), 순부력
−5.7 N 불변. 버려지는 비대각 성분이 Ixz −0.0016(0.4%) → **+0.064 kg·m² (Ixx의 16.8%)** 로 증가 — hydro가
body-frame 속도를 읽도록 고치기 전까지 플랜트에 못 싣는 실존 roll-yaw 곱관성(KNOWN_ISSUES). 브래킷 질량은 아직
합성 미포함(visual-only; 실물 질량 받으면 추가). 덤 수정: 메쉬 시각 방향 헬퍼가 `conj(mesh_quat)`를 썼는데
MuJoCo 규약은 `v_orig = R(mesh_quat)·v_stored + mesh_pos` → geom quat은 `mesh_quat` 그대로여야 함 — 구 베이크의
~180° 주축 회전이 버그를 가렸었음(conj(q) = −q). 검증: 정점 재구성 오차 0.000 mm; `test_heavy_gripper.py` 전부
통과; 렌더 `assets/screenshots/c3/c3new_*.png`.

---

## 2026-07-20 — heavy_c3 변종: Onshape 어셈블리에 있는 것만 반영 (heavy + C3, 그리퍼 없음)

실험실 Onshape 어셈블리에는 BlueROV2 Heavy + C3-BR 브래킷 위 MarineSitu C3만 있고 **Newton 그리퍼는 없다**(아직
CAD에 없음). 이를 그대로 반영하려고 `ROV_MODEL=heavy_c3`를 추가 — CAD에 없는 하드웨어는 심에서도 모델링하지 않음.
`heavy_gripper`에서 그리퍼만 뺀 것: `compute_payload_inertia.compose_c3()` / `buoyancy_c3()`가 차체 + C3만 합성
(브래킷은 visual-only, 질량 미상), `gen_c3_variant.py`가 `bluerov_heavy.xml`에서 `bluerov_heavy_c3.xml`을 생성
(프레임을 합성 COM에 재원점, 대각 관성, C3 메쉬 + 브래킷 + 렌즈평면 카메라 3개, 그리퍼 실린더/죠/액추에이터 **없음**).
수치: 질량 13.2 kg, I = [0.37014, 0.73153, 0.67460], COM [+0.026, +0.001, −0.020], 배수량 0.0129237 m³ →
**순부력 −3.1 N (가라앉음)**; coBM +0.01372. 관절체가 없어 합성 COM = 전체 차체 COM이므로 `<inertial pos>` = 0.
게인: `heavy_c3`는 `GAINS_HEAVY`를 상속(fully-actuated 폴백) — PID가 원점을 0.0 cm로 유지; roll/pitch 레벨링 PD가
없어(그건 `heavy_gripper`에 있음) C3의 전방-하단 COM 때문에 ~1° 피치 잔류(위치 유지엔 영향 없음). 버려지는 비대각은
Ixz +0.046(Ixx의 12.4%), heavy_gripper와 동일한 body-frame hydro 한계(KNOWN_ISSUES). 회귀 `test_heavy_c3.py`
(합성/무그리퍼/thruster-heavy동일/침강/카메라전방/PID유지) 통과; heavy·heavy_gripper·bluerov2 무변경. 렌더:
`assets/screenshots/c3/heavy_c3_*.png`. `heavy_gripper`는 그리퍼가 Onshape에 추가될 때를 위한 향후 config로 유지.

---

## 2026-07-20 (2) — 페이로드 메쉬 방향 버그: MuJoCo는 메쉬 재정렬을 geom에 합성한다 — 상쇄 쿼터니언은 절대 불필요

사용자가 Onshape 정면 뷰와 심을 비교해 C3가 ~90° 뒤틀려 렌더되는 것(패널이 정면을 안 봄)을 잡아냈다. 근본 원인
— 그리고 2026-07-19의 "덤 수정" 자체가 틀렸다는 정정: MuJoCo는 메쉬를 내부적으로 주관성축으로 재정렬하지만,
**컴파일러가 그 재정렬을 참조하는 geom의 pos/quat에 도로 합성한다**(수치 검증: XML quat = mesh_quat일 때 컴파일된
geom_quat이 정확히 mesh_quat⊗mesh_quat로 나옴). 즉 XML geom pose는 **저작된 원본 그대로의** 메쉬에 적용되며,
미리 구운 메쉬에는 **quat이 아예 필요 없다** — quat 없는 `rovc_*` 스킨이 항상 제대로 렌더된 이유이기도 하다. 이전
두 "상쇄" 방식(conj(mesh_quat), 그다음 mesh_quat)은 재정렬을 이중 처리한 것; C3 단면이 거의 정사각(95×89 mm)이라
오류가 깔끔한 90° 뒤틀림으로 보여 눈검사를 통과해버렸다. 수정: 두 생성기 모두 quat 없는 페이로드 geom + 빌드 시
가드 `_verify_mesh_geoms()`(생성된 XML을 컴파일해 컴파일된 geom pos/quat로 렌더 정점을 재구성, bbox가 의도 위치의
구운 STL과 ±1 mm 일치함을 assert) — 방향이 틀린 페이로드는 이제 빌드가 실패한다. Onshape 재-export로 어셈블리도
재확인: C3/마운트 pose가 동결 상수와 바이트 단위 동일 — **사용자의 mate는 문제없었고** 오류는 전적으로 우리
시각화 체인에 있었다. 하우징/브래킷은 Onshape 외관 색 그대로(파랑 0.231/0.380/0.706, 회색 0.753). 렌더가 사용자
정면 뷰와 일치(가로 나사 패널이 전방, 커넥터는 좌현): `assets/screenshots/c3/heavy_c3_front_full.png`. 동역학은
무변경(visual-only geom); 두 변종 테스트 스위트 통과.

---

## 2026-07-21 — MPC 참조 preview (tracking 모드): 호라이즌이 이제 코너를 본다

n=200 코너 분석(compare_20260720_230025)이 동기: mpc/dobmpc의 일상적 코너 편차(outside corner-rounding,
dobmpc corner/edge RMS 비율 ~2.4–2.9×, NONE에서도 동일)는 호라이즌 참조가 코너를 직진 관통 외삽
(`p_ref + v_ref·k·dt`, `_xref_ned`)하는 데서 나왔다 — NMPC가 턴을 미리 본 적이 없었던 것. 이 외삽은
constant-pose DP 타일을 PID-동등 `set_target()` 인터페이스 위에서 키워온 흔적이며, DP(점 안정화)와
trajectory tracking은 다른 문제이므로 이제 참조 모드를 분리했다.

**변경** — 표준 receding-horizon reference preview (교과서 tracking-MPC 정식화; rpg_mpc
`setReferenceTrajectory`, acados stage별 `yref`와 같은 패턴): `DOBMPCController.set_reference_traj(fn)`이
미션 샘플러 `fn(ts) → (p (3,K), yaw (K), v (3,K), r (K))` (world FLU)를 받아, 매 20 Hz tick마다
`fn(t + k·dt)` (k=0..N, 3 s)로 acados stage 참조를 채운다 — 호라이즌 안의 코너가 위치 preview를 꺾고,
stage별 body-속도 참조 `S·R(ψ_k)ᵀ·v_k`를 새 leg로 회전시키고, stage별 yaw/rate FF를 램프시킨다. yaw는
측정 yaw에 앵커한 stage-간 unwrap(±π short-way; 매 tick 재앵커로 10랩 CCW에도 유계). 미션 쪽:
`run_compare.make_square_ref()` — 위치/접선은 `square_setpoint` 정확값, heading 프로파일은 live 루프와
**같은** `slew_heading` 재귀로 사전계산(차량 피드백 없음 → 미래가 알려짐), 1랩 전체에서 <1e-9 일치 검증
(`test_square_ref_matches_live_loop`). `run_one`·`run_viewer`의 mpc/dobmpc square에만 배선; PID·DP·teleop은
기존 `set_target` 경로 유지(상수-샘플러 테스트로 legacy body 비트 동일 확인). OCP/모델/비용 무변경 —
런타임 참조 데이터만이라 acados 재빌드 없음; 코너-불가능한 sharp square 참조는 벤치마크 스트레스 테스트로
**의도적으로 유지**(P2 리뷰: 참조 smoothing은 과제 정의를 바꿈; 관측되는 rounding이 곧 컨트롤러의 최적
smoothing). Provenance: run meta에 `controller.ref_preview` 기록(2026-07-21 이전 run은 전부 False에 해당 —
이 경계를 넘어 square mpc/dobmpc 결과를 합산하지 말 것). P2 리뷰(simulation-advisor +
control-theory-advisor): wrong 판정 0건; 최소 수정 3건 반영(meta provenance, 샘플러 shape 게이트,
`reset()`의 샘플러 해제). 이론 예상: 코너 과도 수 배 감소 + 부호가 선제적 inside-cut으로 반전; 솔버 실패는
동일하거나 감소(vertex에서 전 stage가 한꺼번에 90° 도는 대신 tick 간 참조가 매끄럽게 변함). 비교 실험
재실행은 의도적으로 사용자 몫으로 남김.

## 2026-07-21 (2) — roll/pitch 레벨링 PD→PID, 그리고 heavy 경로에서 rank-5 surge 정형 제거

rank-6 heavy 계열을 위한 `controller.py`(`PoseController`) 두 가지 변경. 분석 + 반증 검증을 control-theory
워크플로로 수행(두 주장 집합 모두 CONFIRMED).

**(1) roll/pitch 레벨링 PD → PID.** 선택적 레벨링 루프(변종별로 `rp_kp`로 활성; `heavy_gripper`에서 ON)는
순수 PD였는데, 이는 *일정* 자세 토크에 대해 type-0라서 정상상태 틸트 `φ_ss = τ/(kp + B_restore)`를 남긴다.
비대칭 페이로드(전방 COM + 조 무게, ~0.2–0.5 N·m). 0.40 N·m 일정 토크에서 실측: PD는 roll 3.53° / pitch
3.58° 유지 — 매니퓰레이션 리치에서 cm급 말단 오차. `Ki>0`이면 루프가 **type-1**이 되어 `φ_ss → 0`
(실측 0.00°/0.00°), `Ki` 크기와 `B_restore` 추정 오차와 무관. 설계 = **옵션 (a)**: 검증된 `kp/kd`는 유지하고
`ki = I_eff·α·ωn³`만 추가(병진과 동일한 해석적 극배치, α=0.2, ωn=3): `I_eff` roll 0.481 → 2.60,
pitch 0.859 → 4.64. anti-windup은 위치/yaw 루프와 동일 — 수평 근처(`|angle| < rp_gate = 0.15 rad`)에서만
`-angle` 적분, `|ki·I| < rp_i_max = (1.5, 2.0)` N·m 클램프. 3차 폐루프 `I·s³+Kd·s²+(Kp+B)·s+Ki`의 모든 근이
LHP이고 지배극쌍 ζ≈0.83–0.89임을 검증(적분이 PD 감쇠를 **약화하지 않음**; −0.67/−0.84 rad/s의 느린 동반극을
삽입 → 잔류 틸트 ~4.5–6 s에 소거). 게인은 `GAINS_HEAVY_GRIPPER`에 추가(`rp_ki`, `rp_i_max`, `rp_gate`).
`self.use_i` **및** `rp_ki` 존재로 게이트 → `mode="pd"`와 rp_kp-only 변종은 바이트 동일 PD 유지. 프로버넌스:
`heavy`/`heavy_c3`는 레벨링 OFF 유지(불변), `heavy_gripper`의 2026-07-21 이전 PID-레벨링 런은 PD 레벨링(적분
없음)이었음 — 자세 정확도 데이터를 이 경계로 섞지 말 것.

**(2) rank-6에서 body-force 정형 일반화.** surge **전용** slew-rate 제한을 세 축 전체에 대한 **균일** slew로
일반화(유한 액추에이터/명령 대역폭; 120 N/s에서 거의 바인딩 안 됨 — hover에서 ~66 N/s). 소프트
**pitch guard**(|pitch|>15°에서 surge 감쇠)는 heavy 계열에 **유지**(`pitch_guard_deg` 게인 키로 게이트).
1차 분석(제거)에 대한 정정: rank-6 `pinv`(full-rank 6×8 `B`)는 `Fx`↔`My`를 **정적으로만** 분리(순수
`Fx → My=0`)하지만, heavy로 이관한 `test_controller`에서 **동적 Coriolis/부가질량 커플링**이 약복원 pitch 모드를
여전히 여기함이 드러남 — 공격적 far-offset 슬루에서 guard 없으면 pitch가 transient ~70°, 있으면 ~45–56°(둘 다
레벨 복귀). 즉 guard는 heavy에도 유용 — 15° 초과에서만 발동하는 dormant transient-pitch 한계(정상 소각도
운용엔 무발동; `heavy_gripper`에선 rp-PID 레벨링이 자세를 먼저 처리하고 guard는 큰 이탈에서만 백업).
진폭 클램프 불변(이미 등방 30/30/30). docstring 정정: heavy 계열은 rank-6 완전작동 → 6-DOF 독립 지령 가능.
`test_controller`의 pitch 임계값은 heavy용으로 재보정(스트레스 스타트 transient가 옛 rank-5 기준 45°를 넘음 →
게이트를 "뒤집힘 없음 <90°" + "레벨 복귀 <8°"로) + PD-vs-PID 전류 헤드룸 축소(강성 큰 heavy는 PD offset이 이미
~1 cm).

**(3) `bluerov2` 변종 제거 — heavy 전용 (2026-07-21).** 레거시 rank-5 벡터드-6 `bluerov2`를 선택 가능 변종에서
제거: `rov_model._MODELS`/`_POOL_WRAP`, 게인(`GAINS_BLUEROV2` + `DEFAULT_GAINS`의 under-actuated 폴백,
이제 `→ GAINS_HEAVY`), 문서에서 삭제; `ROV_MODEL=bluerov2`는 이제 즉시 실패. 루트 스모크 테스트 3개
(`test_load`, `test_controller`, `test_square_mission`)는 bare-heavy 플랜트로 이관. `bluerov.xml` +
`marinegym_assets/BlueROV.yaml`은 hydro/thruster verify_*/test_* 스크립트(`verify_hydro`, `verify_hydro_precise`,
`test_hydro`, `test_thrusters`, `test_disturbances`, `verify_meta`, `verify_gpu_mjx`)와 `compute_heavy_inertia`
provenance의 **고정 reference fixture로 유지** — 삭제 시 8개 스크립트가 깨짐. **보류**(사용자 지시): dobmpc
`NU=4`/option-(b) surge→pitch 죽은 경로 + `test_dobmpc`의 rank-5 trim 단정 → 추후 NU=6-only로 정리
(KNOWN_ISSUES 참조). 회귀: `test_load`, `test_controller`, `test_square_mission`, `test_hydro`, `test_thrusters`,
`test_disturbances`, `verify_meta`, `test_heavy_gripper`, `test_heavy_c3` 전부 PASS.

## 2026-07-23 — EAOB 현실적 센서 기준 재튜닝: 시그마 유도 공분산("perf" 프로파일) + 센서노이즈 주입; NIS 게이트는 구현 후 기본 OFF

EAOB 튜닝에 대한 verify-first 리뷰(외부 리뷰의 주장을 코드에서 한 줄씩 검증 후 수정) 및 재튜닝.
아래 수치는 전부 heavy 플랜트, seed 0, 20 Hz, `verify/verify_eaob.py` 실측.

**(1) 진단 — 구 튜닝은 near-deadbeat이었고, 시뮬이 관측기에 무노이즈 진리값을 먹여줬기 때문에만 "동작"했다.**
확인: marinegym 루프 어디에도 측정 노이즈가 없음(EAOB는 raw `xpos/xmat/mj_objectVelocity` + 클린 tick FD를
받음; XML `<sensor>` 없음; disturbance 패키지는 힘만 교란). 공분산은 단위 무시 DT 템플릿(`Q_pose=R=DT⁴/4`,
`Q_vel=Q_dist=DT²` — 미터·라디안·뉴턴에 같은 분산). hover에서 실제 (F,H)의 정상상태 Riccati 분석: 축별
w-오차 시정수 6.8–13.4 ms(tau 채널 칼만 이득 −0.976…−0.999) = 사실상 한 틱 만에 재고정되는 deadbeat
역동역학 디코더. 현실적 센서 노이즈가 있으면 그대로 `w_hat`에 통과된다.

**(2) 재튜닝("perf" 프로파일, 이제 기본값).** `params.py`에 센서 시그마 블록
(`EAOB_SIG_POS/ANG/LVEL/AVEL/ACC/AACC/ALLOC`)을 추가 — 시뮬 측 주입과 필터 R의 **단일 출처**. R은 채널별
블록대각; tau 의사측정 노이즈는 가속도계 노이즈를 질량행렬로 통과시킨 `R_tau = M Σ_a Mᵀ + Σ_alloc` —
N/N·m 스케일은 물리가 정하고 손튜닝 없음. `q_dist`는 스칼라 정상상태 역산 `K = 1−exp(−DT/τ_dist)`,
`q = diag(R_tau)·K²/(1−K)` — **`EAOB_TAU_DIST`(0.2 s, (3)의 반복 참조)가 유일한 노브**. 구 튜닝은
`profile="verify"`로 보존(무노이즈 배선 점검용, byte-identical 구성). `DOBMPCController`가 같은 시그마로
EAOB의 eta/nu/nudot에 가우시안 노이즈를 주입(`meas_noise`, perf에서 기본 ON; MPC는 여전히 클린 플랜트
상태를 소비 — 관측기를 격리 평가). run meta에 `controller.eaob = {profile, meas_noise, noise_seed,
tau_dist, gate, n_upd, n_gated}` 기록 — 이 키가 없는 기록은 verify/deadbeat + 클린 측정과 동등;
**이 경계를 넘어 dobmpc 결과를 합산하지 말 것**(2026-07-22 이전 sweep 전부가 경계 이전).

**(3) 검증 + τ_dist 반복(`verify/verify_eaob.py`: w_true = 클린 상태에서의 모델 잔차 M·a+C+D+g−τ,
hydro의 독립 `diag_wtrue` 진단과 교차검증 — 둘이 ≤0.08 N 일치).** 초기 설계점 0.5 s는 w_dot=0 모델이
유효한 곳(DP CD: NIS 14.8 PASS)만 통과하고 파랑에서 실패(DP CDW: NIS 25.4, NEES 77 — 파랑 대역이 w 랜덤워크를
앞지름). 단일 노브 반복, 게이트 OFF(DP CDW, T=60 s, seed 0): τ = 0.5 / 0.3 / **0.2** / 0.1 →
NIS 25.4 / 19.8 / **17.1** / 14.8, NEES 77 / 26.1 / **15.0** / 11.5, w-RMSE_X 1.11 / 0.74 / **0.55** / 0.48 N.
**기본값 τ_dist = 0.2 s** — CDW에서 두 범위를 모두 통과하는 유일한 값; CD에선 NIS 14.4 PASS,
RMSE X/Y/Z 0.29/0.41/0.48 N. CD NEES ~9(보수적, 안전한 방향)는 R/Q가 이상적 시뮬에 없는 실기체 항
(할당 오차, 프로세스 플로어)을 의도적으로 모델링하기 때문 — 줄이면 로봇이 아니라 시뮬에 맞추는 것이라 유지.
저속 외란 전용 실험은 τ를 올려도 됨(τ≈2 s면 CD RMSE 다시 절반, 대신 파랑 lag). 측정 주의: 초기 스윕 하나가
실수로 게이트 ON 상태로 돌아 NIS가 크게 부풀려짐(τ 0.5→0.05에서 137/67/43/27) — 기각이 필터를 굶기고 다음
혁신을 키우는 피드백 아티팩트; 일관성은 반드시 게이트 OFF로 측정. 운동상관 진단: 최종 구성에서 잔차 vs |nu|
기울기 ≈ 0, |r| ≤ 0.18(모델 오차 누출 없음; CDW의 작은 Z/M 상관은 파랑동조 운동이지 drag 미스매치 아님).
square의 코너 NIS 스파이크는 psi/r 채널 지배 — placeholder 회전 added mass와 정합.

**(4) χ² innovation 게이트: 구현·실측 후 기본 OFF.** `update()`가 매 틱 NIS를 계산(`last_nis`,
`last_zscore`, `n_gated`)하고 χ²(0.999,18)=42.31에서 프레임 기각 가능(`params.EAOB_GATE_ON`). 폐루프
A/B(square 2랩, seed 0)가 기본값으로는 부적합함을 확정: 파랑 대역 + 코너 기동이 w_dot=0을 **상시적으로**
위반하므로 일관성 게이트가 프레임의 25–76%를 기각해 외란을 추종하는 바로 그 업데이트를 차단 —
CDW radRMS 4.53→15.45 cm(게이트 off→on), CD 1.35→1.43 cm(τ=0.5 기준; 최종 τ=0.2에선 같은 시나리오에서
게이트 발동 0/1067 — 무해하지만 무용). 모든 파랑을 이상치로 부르는 모델 아래에서 게이트는 "스파이크"와
"파랑"을 구분할 수 없다; 스파이크 방어는 per-axis `W_HAT_CLIP`(KNOWN_ISSUES, 미해결) 몫.

**(5) 정직한 수치의 비용(최종 구성: perf, 노이즈 ON, τ=0.2, 게이트 OFF; 2랩 square / 60 s DP, seed 0).**
DP C 모드: 현실적 센서에서도 DOB-MPC의 헤드라인 DC 제거 유지 — dc 오프셋 (−0.11, −0.01) cm, radial RMS
0.33 cm, vs 일반 MPC +2.13 cm(구 0.00 cm는 무노이즈 허구였음). square: CD 1.02(verify, 클린) → 1.21 cm,
CDW 3.09 → 3.51 cm — vision급 센싱의 정직한 비용은 ≈0.2–0.4 cm. 회귀:
`test_frames`/`test_predictor`/`test_eaob_no_accel_doublecount` 새 기본값에서 PASS;
`test_casadi_matches_numpy`는 기존 실패(NU=4 전제, KNOWN_ISSUES 추적).

## 2026-07-23 (2) — MPC 상태 입력을 플랜트 진리값에서 EAOB 추정치로 전환 (mpc_state_source)

오전 재튜닝이 남긴 정직한-평가 공백을 닫는다: NMPC가 이제 클린 플랜트 상태 대신 EAOB의 eta_hat/nu_hat을
소비한다(dobmpc 모드, perf 프로파일 기본). verify-first 프로토콜; 수치는 전부 seed 0,
`verify/verify_state_source.py`.

**(1) 스위치(단일 지점, 유일성 증명).** `_control_step`의 `x_ned`는 소비처가 정확히 하나(`nmpc.solve`)인
지역 변수 — 감사 결과 다른 NMPC 상태 유입 경로 없음. `mpc_state_source="estimate"`면 솔버 x0과
yaw 참조 앵커 `_psi_ned_now`가 함께 추정치를 따른다(둘은 같은 branch여야 함: NMPC 비용에는 각도 wrap이
없고 참조는 매 틱 `_psi_ned_now`에 재앵커). 나머지는 전부 계약상 플랜트 읽기 유지: EAOB 자신의 측정
(meas/a_meas/_nu_prev_ned), `_read_state`(verify_eaob가 클린 진리값으로 의존), 모든 로깅 표면
(recorder/mission/run_compare/run_viewer/eval_dp/teleop은 `data.*` 직접 읽음). 기본값: mode="dobmpc"이고
profile="perf"일 때만 "estimate"(params.MPC_STATE_SOURCE); 일반 "mpc"는 관측기가 없고 verify 프로파일은
배선 점검이라 둘 다 "truth". run meta에 `controller.mpc_state_source` 기록(키 부재 = truth;
ref_preview·eaob에 이은 세 번째 provenance 키).

**(2) yaw 연속성엔 unwrap 누산기가 필요 없었다.** EAOB yaw 상태가 이미 연속 신호이고(적분만 되고 재wrap
없음; innovation만 wrap), 두 솔버 모두 yaw/위치 bound가 없어(acados idxbx=[3,4,6,7,8]; IPOPT는
roll/pitch/|v|만) 멀티랩 누적 yaw(2랩 ≈ 4π)가 안전 — 실측: 전 런에서 틱당 |Δpsi_hat| ≤ 0.12 rad
(branch 점프면 ~6.3). 부수 이득: 추정 경로는 RTI warm start가 매 ±π 교차마다 보던 측정 yaw 점프를
없앤다. 스타트업 워밍업도 불필요: EAOB가 첫 읽기값에서 lazy-init하므로 추정치가 정확히 진리값에서 출발
(모든 A/B 쌍에서 startup max radial이 초기 오프셋과 동일).

**(3) A/B 결과 (perf+노이즈, τ_dist=0.2, seed 0; 60 s DP / 2랩 square; truth → estimate).** 튜닝이 검증된
해상 상태(Hs 0.75/Tp 12) 기준: DP C 0.33→1.46 cm(dc −0.11→−0.45), CD는 C와 동일; DP CDW 1.51→2.22 cm;
square C 1.20→1.88, CDW 3.51→3.92 cm. 관측기 건강도는 루프 폐합에 무영향(DP CDW NIS 17.1→16.8,
NEES 15.0→14.5 — 전부 PASS; truth 행은 오전 verify_eaob 수치를 정확히 재현). chatter(틱당 Σ|ΔF| 평균):
DP C 0.71→1.18 N(추정 지터 유입, +66%)이지만 square CDW는 2.43→1.32 N — 관측기가 기동 중 상태 필터
역할을 해 추종을 약간 내주고 명령을 매끈하게 만든다; 포화 0, n_fail 0, 전 런 유한. 수용 기준 PASS
(estimate NIS 14–24, NEES ≤ 24; C/CD의 ~9 NEES는 문서화된 보수적 방향이라 비게이팅 보고).

**(4) 검증 중 발견 — 현재 base.yaml 해상 상태는 소스와 무관하게 CDW 일관성을 깬다.** config/base.yaml의
파랑 블록이 2026-07-23 19:28에 강화됨(Hs 0.75→1.2 m, Tp 12→6 s, γ 5→2, s 30→10, ω_max 1.6→3.0): 그 아래서
CDW NIS = 80(truth)/71(estimate) (DP), 44/42 (square), radRMS 9.6→10.1 / 15.7→17.5 cm — w_dot=0 +
τ_dist=0.2 관측기가 새 파랑 대역을 못 쫓아간다. 이는 상태 소스 전환과 직교(두 소스가 동일하게 실패;
구파랑 A/B는 전부 통과)하며 KNOWN_ISSUES에 등재; 새 해상 상태용 τ_dist 재튜닝(또는 harmonic-EAOB)은
이 과제 밖의 인간 결정 사항.

## 2026-07-23 (3) — DOB 플러그인 구조: NMPC x0 = state-estimator 출력("meas"), EAOB는 w_hat만 넘김

오전의 상태-소스 전환을 고전 DOBC 토폴로지로 재정립(control-theory-advisor 상담, P1, journal 2026-07-23):
EAOB는 순수 disturbance 플러그인이 되고 — 필터된 eta_hat/nu_hat은 컨트롤러가 더 이상 쓰지 않으며 NMPC x0는
state estimator에서, w_hat만 EAOB→MPC로 건너감. Endgame 근거: diffusion policy가 SLAM/AI estimator를
관측하므로 트래커 x0도 같은 estimator를 공유해야 하고(플래너/트래커 상태 일치), 그러면 "MPC vs DOB-MPC"가
w_hat 단일 변수로만 갈림.

**(1) 스위치(`params.MPC_STATE_SOURCE`, 이제 3-way).** `"meas"`(노이즈 활성 시 mpc·dobmpc 공통 새 기본값) =
루프가 실제로 가진 estimator 출력: 시뮬에선 클린 읽기 + 가우시안 센서노이즈(EAOB가 먹는 것과 같은 틱 샘플,
params.EAOB_SIG_*), 하드웨어에선 SLAM/AI-model 추정치. `"estimate"`(2026-07-23 (2) 기본값, 이제 opt-in) =
EAOB 필터 상태(offset-free-MPC 스타일). `"truth"` = 클린 플랜트(회귀/관측기 격리; 무노이즈 경로). 한 샘플이
EAOB 측정과 x0에 함께 들어가고, yaw 참조 앵커도 선택 소스를 따름. 일반 mpc도 이제 noisy x0 소비(기존 truth) —
mpc/dobmpc 비교를 단일변수로 만들기 위한 의도적 변경; mpc 베이스라인이 그만큼 이동(run meta
`controller.mpc_state_source`가 경계, 키 부재 = truth).

**(2) A/B 결과 — 안정적이나 raw-노이즈 x0는 chatter 대가가 큼(advisor 예측의 정량화).** perf+노이즈,
τ_dist=0.2, seed 0, 구파랑(Hs 0.75/Tp 12 — base.yaml 파랑 이슈와 분리), truth → meas → estimate:

| 시나리오 | radRMS [cm] | effF [N] | chatF [N/tick] | NIS |
|---|---|---|---|---|
| dp C | 0.33 / 1.44 / 1.46 | 3.0 / 8.3 / 3.4 | 0.71 / **10.89** / 1.18 | 14.4 / 14.9 / 14.4 |
| dp CDW | 1.51 / 2.58 / 2.22 | 5.2 / 10.0 / 5.6 | 0.88 / **10.66** / 1.32 | 17.1 / 18.5 / 16.8 |
| square C | 1.20 / 2.06 / 1.88 | 4.6 / 9.5 / 4.6 | 1.19 / **11.81** / 1.18 | 15.5 / 17.3 / 15.4 |
| square CDW | 3.51 / 3.98 / 3.92 | 7.5 / 11.4 / 6.9 | 2.43 / **12.02** / 1.32 | 20.1 / 22.9 / 17.9 |

추종(radRMS)은 소스 간 사실상 동일(meas ≈ estimate)하지만 "meas" chatter는 **truth의 9–15배**, "estimate"의
~9배, 제어노력은 ~2–3배: MPC가 5 cm/tick @ 20 Hz 위치 지터를 직접 쫓음. 발산 없음 — 유한, 포화 0, n_fail 0,
EAOB yaw 연속(max |Δψ̂| ≤ 0.08 rad; "meas"는 "estimate"가 유일하게 매끈히 하던 ±π 측정yaw warm-start 과도를
재도입하나 RTI가 흡수). "meas"의 NEES가 기동에서 24를 근소 초과(square C/CD 24.3/24.4; CDW 46.5 — 단 truth도
30.8이라 이는 알려진 파랑 한계) — chatter가 플랜트를 통해 추정 잔차로 되먹임.

**(3) 해석 — "상태 스무딩을 누가 소유하나"에 대한 실증적 답.** raw noisy estimator로 플러그인하는 것은
하드웨어 정직한 토폴로지가 맞지만 큰 액추에이터 chatter 세금을 문다; EAOB 필터 "estimate"는 동일 추종을
truth 수준 chatter로 냄 — 그 자체가 상태 필터이기 때문. advisor대로: 실기에선 SLAM+IMU 융합이 스무딩을
소유하며(융합 출력 ~1–2 cm 유색, 시뮬의 5 cm 백색 @ 20 Hz보다 훨씬 온화), 시뮬 "meas"는 비관적. 이것은
의도된 임시 출발점(estimator를 SLAM/학습모델 중 미정인 동안 사용자 선택); 완화책(EAOB·MPC 양쪽에 공급하는
SLAM-proxy KF, MPC rate weight, 또는 "estimate" 복귀)은 별도 인간 결정이라 여기서 적용 안 함. EAOB 튜닝
(R/Q/P0/τ_dist)은 제약대로 불변.

## 2026-07-24 — 센서 노이즈 시그마를 추종오차 아래로 축소 ("meas" x0 chatter 진정)

2026-07-23 (3) 플러그인 전환의 후속: `mpc_state_source="meas"`에서 raw x/y 5 cm(1σ) 위치 노이즈가 폐루프
추종오차(~1.2 cm)보다 컸기에 MPC가 자기 측정 지터를 쫓음(chatter 9–15배). `params.EAOB_SIG_*`(pose/vel
채널만)를 좋은 visual-inertial/SLAM+DVL+depth 스택 값으로 축소: x/y **0.5 cm**, z 0.3 cm, 자세 0.3°(yaw 0.5°),
선속도 0.5 cm/s, 자이로 0.11°/s. 이제 위치 노이즈 ≈ 추종오차의 0.4배. `ACC/AACC/ALLOC`(외란 의사측정 →
R_tau, q_dist)는 불변이라 w_hat 품질 보존(verify_eaob CD w-RMSE 0.24/0.34/0.39 N, 구노이즈 0.29/0.41/0.48보다
오히려 약간 좋음). `EAOB_SIG_*`가 주입·R 단일 출처라 R도 함께 줄어 필터는 설계상 일관/보수적 유지.

**A/B (gentle 해상, seed 0, truth → meas → estimate; 5 cm → 0.5 cm):**

| 시나리오 | radRMS [cm] | chatF [N/tick] (meas) |
|---|---|---|
| dp C | 0.22 / 0.26 / 0.31 | 10.89 → **1.76** (truth 0.70) |
| square C | 1.13 / 1.20 / 1.16 | 11.81 → **2.91** (truth 1.19) |
| square CDW | 3.19 / 3.26 / 3.26 | 12.02 → **3.08** (truth 1.93) |

"meas" 추종이 이제 truth 수준(이전 4–5배 악화), DC 편향 누출 사라짐, chatter는 truth-x0의 ~2–3배(이전 ~15배).
상태추정 오차 pose ~1.6 mm / vel ~2.7 mm/s로 감소. 일관성: NIS/NEES가 ~13/9(목표 ~18 아래) — **안전·보수적**
방향(필터가 자기를 과소신뢰; 과신은 절대 아님). verify 게이트(verify_eaob, verify_state_source)를 **과신만**
(값 ≤ 24) 게이팅하도록 변경, 하한은 보수적으로 보고 → 저노이즈 config가 더 이상 헛되이 FAIL 안 함. run meta에
`controller.eaob.sig_pos_xy`(0.005) + `mpc_state_source` 기록 — 5 cm 시절 meas-noise 기록(내부 A/B 뿐)과 구분.
주의: 극단적 정밀 DP hold에선 truth 오차(0.22 cm)가 0.5 cm 노이즈보다 작음; "추종오차보다 작게" 목표는 운영
square 미션(~1.2 cm)에서 충족. 더 낮추길(0.25 cm) 원하면 chatter를 truth floor까지 내릴 수 있음(센서 스펙 선택).

## 2026-07-24 (2) — PID도 MPC와 동일한 noisy 상태를 소비 (apples-to-apples 비교)

지금까지 compare sweep은 PID에 **ground-truth** 상태(`data.xpos/xmat/qvel`, 500 Hz)를 주고 MPC/DOB-MPC만
**noisy 20 Hz** 추정치(`mpc_state_source="meas"`)로 돌렸다. 이 비대칭이 모든 PID-vs-MPC 수치를 confound시켰다 —
2026-07-24 wave sweep 분석에서 "C/CD 격차가 state 노이즈 때문이 아니다"를 *간접적으로*(NONE 대조군 + DOB-MPC
반사실) 증명해야 했던 이유가 바로 PID만 더 좋은 입력을 몰래 받았기 때문이다. 새 `PoseController(meas_noise,
noise_seed)`가 이 간극을 닫는다: `meas_noise` ON이면 PID가 보는 상태는 MPC x0가 먹는 것과 **동일한** 가우시안
센서 모델(`params.EAOB_SIG_*`: pose 0.5 cm / z 0.3 cm, 자세 0.3°/yaw 0.5°, 선속도 0.5 cm/s, gyro 0.11°/s)로
오염되며, **제어틱(20 Hz)에서 샘플링·ZOH 홀드**된다 — 그래서 PID와 (DOB-)MPC는 동일하게 열화된 추정치를 먹고
**오직 제어법칙만** 다르다. noisy 자세는 full R로 재구성(`_R_from_rpy`, yaw/pitch/roll 추출의 정확한 역함수)해서
각도 읽기와 힘 회전 `R.T` 모두 추정 자세를 일관되게 쓴다. `meas_noise`는 MPC가 이미 갖던 두 결합 효과(가우시안
노이즈 + 20 Hz 샘플링 지연)를 함께 포함한다. 로그되는 truth(`px,py`)와 radial-RMS 지표는 항상 plant에 붙어 있어
지표는 여전히 **참** 추종오차를 잰다.

기본값 `params.PID_STATE_SOURCE = "meas"`(`MPC_STATE_SOURCE`와 대칭)라 sweep은 기본적으로 apples-to-apples;
`"truth"`는 opt-in clean-state baseline. 인터랙티브/eval 도구(teleop, eval_dp)는 `meas` 상속(이미 noisy한
자기 MPC와 일관), 결정론적 제어법칙 회귀 4개(test_controller/test_square_mission/test_heavy_gripper/
test_heavy_c3)는 `meas_noise=False` 고정 → clean state에서 제어법칙을 검증. run meta에 `controller.meas =
{state_source, noise_seed, sig_pos_xy}` 기록(키 없으면 truth); **이 날짜 이전 PID 기록은 truth 상태라 새 meas
기록과 비교 불가** — 비교하려면 sweep 재실행.

**A/B (heavy, seed 0, 3-lap square, truth → meas x0):**

| ctrl / mode | radRMS [cm] | dc [cm] | slew Σ\|dU\| [N/tick] |
|---|---|---|---|
| pid  NONE truth | 1.49 | 0.08 | 0.85 |
| pid  NONE meas  | 1.47 | 0.10 | 3.27 |
| pid  C    truth | 1.61 | 0.08 | 0.85 |
| pid  C    meas  | 1.58 | 0.10 | 3.24 |
| mpc  C   (meas) | 3.17 | **2.87** | 11.1 |

핵심 결과가 이제 추론이 아니라 **직접** 보인다: 동일한 noisy 20 Hz 상태를 먹어도 PID는 C모드 DC offset을
~0.1 cm로 유지(적분기가 영평균 노이즈를 평균화)하는 반면 nominal MPC는 2.87 cm — 즉 C/CD 격차는 **적분 작용
부재**이지 상태 품질이 아니다. noisy 상태의 비용은 커맨드 chatter뿐(slew ×3.8, 그래도 MPC보다 낮음 — PID의
120 N/s slew 리미터가 20 Hz 노이즈 스텝을 사실상 필터링); radRMS·offset은 거의 불변. `pid/NONE/truth`는 기록된
sweep의 `radial_max` 4.402 cm를 바이트 단위로 재현 → `meas_noise=False`가 변경 전 경로와 정확히 동일함을 확인.
검증: `_R_from_rpy` round-trip 4e-16; 주입 σ 실측 5.24 mm / 4.82 mm·s⁻¹ / 0.482°(타깃 5/5/0.5); refresh
주기 25 서브스텝(20 Hz), 틱 내 홀드.

## 2026-07-24 (3) — NMPC surge 입력 박스 8 → 30 N (heavy): 벤치마크가 like-for-like가 아니었다

대칭-state sweep에서 "강파랑=PID 승, 약파랑=MPC 승"이 나온 원인을 추적한 결과, 제어이론적 성질이 **아니라**
NMPC 자신의 하드 입력 박스였다. heavy(rank-6)에서 `params.U_MAX`가 `[8, 30, 30, 8, 8, 10]` — **surge만 8 N인데
PID의 `f_max`는 전 축 30 N**, 즉 3.75× 핸디캡. 8 N은 rank-5 유물이다: bluerov2에선 surge→pitch 커플링
`My = −0.0725·Fx`가 Fx ≈ 15 N에서 sin θ = 1을 넘겨 전복시키므로 실재하는 물리 한계지만, heavy에선 수직
스러스터 4개가 그 커플링을 상쇄한다. 기존 주석은 8 N을 "like-for-like compare를 위해" 유지한다고 적었으나
**정반대**였다. PID 쪽 동일 유물(surge 6 N cap)은 2026-07-03 pole-placement 때 이미 제거됨("ωn을 ≤0.54로
묶는다"). MPC 쪽만 남아 있었다.

**영향 크기**(storm CW, 10 lap × 4 페어드 헤딩, 런타임 박스 오버라이드, 나머지 전부 동일):

| | PID | MPC box 8 N | MPC box 30 N |
|---|---|---|---|
| storm  | 26.32 | 30.55 (박스 물림 60–66%) | **17.72** (0.7–1.8%) |
| gentle | 12.51 | 8.50 (5–9%) | 7.64 (0.2–0.3%) |

박스는 **storm에서 20 Hz 틱의 60%(dobmpc 75%), gentle에선 7%** 물렸다 — **진폭 의존적** 스로틀이며, 그래서
*같은* 주파수 대역인데도 해상상태에 따라 우열이 뒤집혔다(0.85–2.30 rad/s MPC/PID 오차비 storm 1.38 → gentle
0.51; LTI 대역폭 논리로는 불가능). 박스를 풀자 storm PID−MPC가 **−4.24 → +8.60 cm**로 바뀌고 gentle은 거의
불변(+4.01 → +4.87) — 즉 **"강파랑=PID 승" crossover 전체가 이 제약 하나였다**. 기존 `sat_freq` 지표는 이걸
못 잡는다(전 구간 0.0000) — 8 N은 스러스터 포화가 아니라 소프트웨어 한계이기 때문.

**적용**: heavy `U_MAX = [30, 30, 30, 8, 8, 10]`; rank-5 분기는 8 N 유지(커플링이 실재). 재빌드된 솔버로 검증
(storm/gentle CW, 10 lap × 3 페어드 헤딩):

| 해상상태 | PID | MPC | DOB-MPC | max 스러스터 | sat_freq |
|---|---|---|---|---|---|
| storm  | 26.14 | **17.86** (이전 28.82) | **7.30** (이전 17.36) | 40.3 N | 0.0000 |
| gentle | 11.56 | **7.63** (이전 9.51)   | **1.87** (이전 3.14)  | 36.1 N | 0.0000 |

이제 MPC가 **양단 모두 승**(PID−MPC storm +8.28, gentle +3.93) — crossover 소멸 — 이고, 가장 authority에
굶주렸던 **DOB-MPC가 가장 크게 개선**(storm 2.4배). 물리 포화는 여전히 없음: 스러스터 피크 40.3 N vs T200
`ctrlrange` −51.6…+64.1 N. 고정 헤딩에선 30 N 박스가 오히려 피크 스러스터를 **낮춘다**(36.1 vs 36.9 N) —
authority가 충분하면 8 N 박스가 유발하던 뒤늦은 과격 보정이 사라지기 때문. `verify_acados` 통과(acados↔IPOPT
max|Δu| 0.0615 N < 0.25 게이트, 1.04 ms, n_fail=0), 폐루프 n_fail ≈ 1/5334틱로 변화 없음.

**기록 경계**: run meta에 `controller.u_max` 기록 시작(run_compare + teleop 매니페스트, 라이브 솔버에서 읽음).
**키가 없으면 낡은 8 N**. 이 날짜 이전 모든 mpc/dobmpc 기록(20260724 sweep 포함)은 8 N 하에서 측정된 것이라
새 기록과 합산 금지 — 특히 그 기록으로 **"강파랑에서 PID가 MPC보다 낫다"고 인용하면 안 된다(아티팩트)**.
full sweep 재실행은 대기 중(KNOWN_ISSUES).

**박스와 무관하게 살아남는 진짜 결론**: MPC의 오차 PSD는 **모든** 해상상태에서 0.7 rad/s 아래 전 주파수에서
PID보다 3–8배 낮다 — Fossen 모델 + 3 s reference preview 덕분이며, 무외란 NONE에서 이기는 이유와 같다
(1.089 vs 1.492 cm). 그리고 정상류 C 모드에서 nominal MPC는 적분 작용이 없어 2.70 cm 하류로 밀린 채 유지된다
(PID 0.03 cm) — DOB-MPC의 `w_hat`이 메우는 격차. **미확립**: 이 crossover가 *주파수* 현상이라는 주장 —
사다리가 Hs와 Tp를 함께 바꾸고(corr(log Hs, log ω_p)=0.967) γ·s까지 공변해서 회귀로 분리 불가(공선인 γ를
대입하면 순위가 뒤집히고, 6 rung만으로는 Hs 계수가 유의하지 않음 p=0.22). 회귀 대신 ablation으로 결판냈다.

## 2026-07-28 — 보고되는 모든 RMS를 하나의 steady window로 통일 (궤적 박스 == 막대그래프)

**같은** 100개 런에서 나온 두 그림이 서로 다른 숫자를 말했다: `bar_square_radial_rms.png`는 PID/CDW를
**18.6 cm**로, `trajectory_compare_CDW.png`의 박스는 **18.2 ± 2.4 cm**로 적었다. 어느 쪽도 계산 오류가 아니라
**창(window)이 달랐다**. `metrics()`는 steady window(`t ≥ experiment.settle_s` = 10 s, 런의 절반으로 clamp)에서
점수를 매기고 그 값이 `results.csv` / `results_raw.csv` / 모든 막대로 간다. 반면 궤적 패널은 `runs/traj_*.csv`를
다시 읽어 **전체 구간**으로 계산했다. square 런은 참조 위에서 오차 ≈ 0으로 출발하므로, 그 초기 10 s —
266.7 s 기록의 3.7 %(196 / 5268 샘플) — 가 나머지보다 훨씬 조용해서 pooled RMS를 끌어내린다:

| 컨트롤러 (CDW) | 초기 10 s | 10 s 이후 | pooled (이전 박스) | steady (막대) |
|---|---|---|---|---|
| PID | 4.47 | 18.57 | 18.24 | **18.57** |
| MPC | 6.13 | 10.69 | 10.55 | **10.69** |
| DOB-MPC | 1.92 | 2.62 | 2.60 | **2.62** |

**적용**: 창을 한 곳에서만 정의한다 — `run_compare.steady_mask(t, settle_s)` (+ `experiment.settle_s`를 읽는
유일한 지점 `settle_start(cfg)`) — 그리고 `metrics()`와 `_draw_traj_panel`이 **둘 다** 이걸 호출한다. `settle`을
`fig_trajectory_compare` / `fig_trajectory_allmodes` / `fig_xwave_trajectory`로 전달하고, 모든 그림이 캡션에
자신의 창을 **명시**한다(`fig_bars`에 `note=` 각주 추가, 궤적 범례는 "lines = all laps · boxed RMS =
steady-window radial RMS (t ≥ 10 s, same metric as the bar chart)"). 독립 실행 도구
`experiments/plot_trajectories.py`도 같은 마스크를 `--settle` 플래그로 미러링 — `run_compare`의 `runs/` 폴더를
가리키면 제3의 숫자가 아니라 그 런의 `results.csv` 값을 그대로 재현한다. 궤적 **선**은 여전히 전체 런을 그린다;
바뀐 건 점수 매기는 구간뿐.

**검증**(시뮬 재실행 없음): `20260727/compare_20260727_000850/wave01_moderate`의 기존 per-run CSV만으로 그림을
다시 렌더 — 15개(5 모드 × 3 컨트롤러) 박스 값이 전부 `results_raw.csv`와 ≤ 1.6e-6 cm 이내 일치(그 CSV의
`%.5f` m 기록 정밀도가 바닥). `plot_trajectories`로 단일 런을 뽑으면 17.26 / 10.88 / 2.79 cm — 해당 런의 기록값과
동일. `--smoke` 전체 파이프라인 정상.

**기록 경계**: 기록된 숫자는 하나도 안 움직인다 — `results.csv`, `results_raw.csv`, `results_all_waves.csv`,
막대그래프는 원래부터 steady-window 지표였고 그대로다. 바뀐 것은 **궤적 그림의 박스**뿐. 이 날짜 **이전**에
생성된 그림의 박스는 full-run RMS라 대응 막대보다 ~1–2 % 낮으니, 캡션("full-run radial RMS (all laps)")을
확인하거나 다시 생성할 것.

## 2026-08-12 — DOB-MPC가 시뮬레이터 밖으로: rov_gui 폐루프 AprilTag MPC (`--mpc`)

이 문서가 기술하는 컨트롤러 스택에 **실기 소비자**가 생겼다: 관제 스테이션의 새 패키지
`rov_gui/control/`이 `dobmpc/`를 **그대로 import**하고(params/fossen/frames/eaob/mpc/mpc_acados —
전부 MuJoCo 무의존), sim 래퍼만 교체한다. `_read_state` → C3 컬러 스트림의 AprilTag PnP +
ArduSub ATTITUDE/SCALED_IMU2(`state_assembler.py`), `set_wrench_command` → 기존 명령 sink의
MANUAL_CONTROL 4축(`allocation.py`; ArduSub MANUAL 모드, roll/pitch 축이 없어 K/M은 버림).
참조 로직(`set_target`/`set_reference_traj`/`_xref_ned`/`_xref_ned_traj`)은
`dobmpc_controller.py:126-285`에서 `rov_gui/control/mpc_bridge.py`로 라인 단위 포팅했고, square
생성기는 `run_compare.py:114-177`의 바이트 복사본을 테스트로 고정했다.

sim 래퍼와 의도적으로 다른 결정:
* **x0는 항상 "meas"** — 하드웨어에 truth 분기는 없다; DOB plug-in 구조(2026-07-23 항)는 그대로.
* **EAOB의 tau_applied는 할당 후 추정치**(축 캡 적용, K/M=0) — 솔버 원출력을 주면 캡/드롭
  불일치가 외란으로 이중 계상된다.
* **w_hat 축별 클립 [15,45,45,5,5,8]**을 브리지에서 적용 — KNOWN_ISSUES의 스칼라 ±50 대체;
  sim 쪽 코드는 백스톱으로 유지.
* 솔버 실패마다 타임스탬프+status+|w_hat| 로그, `solve_ms`는 acados에서 직접 읽음.

2026-08-12 검증(벤치만): rovgui-pose(numpy 2)에서 acados SQP-RTI 빌드 1.3 s, EAOB+solve
p50 3.1 ms / p99 6.4 ms(n_fail 0/100, `rov_gui/control/smoke.py`); demo 백엔드 폐루프로
engage → DP 워밍업 → square 1랩 → 완주 → DP → 해제 전 과정 완료(솔버 실패 0). 수중 상태와
남은 [예측] 항목(축 게인, EAOB 시그마, 벽 태그 preset)은 같은 날짜의 KNOWN_ISSUES.md 참조.

**기록 경계**: 하드웨어 CSV(`*_mpc.csv`)는 sim의 9열 접두를 공유하지만 **다른 모집단**이다 —
meta.json에 `source: "hardware rov_gui.control"`, 태그맵 sha1, 축 게인 출처가 박힌다.
sim `runs/traj_*.csv`와 절대 합산하지 말 것.

## 2026-09-26 — 6-DoF 변형: 정책이 roll/pitch를 내고 NMPC가 추종한다 (기본 OFF)

교체가 아니라 설정 + 체크포인트로 선택하는 새 변형. 확산 정책이 `[dx, dy, dz, dyaw, droll, dpitch, width]`
(`action_repr: pos_rpy_width`, 폭 7)를 낼 수 있다 — 라벨 인코더 `umi/common/yaw_action.py::encode_pos_rpy`
(`R_bt R_rel R_bt^T`의 ZYX 오일러, dyaw 열은 5-dim 라벨과 비트 동일). 스테이션은 디코더를 `policy_frames.py`에
전사(경로 기준 패리티 테스트), `compose_plan(track_rp=True)`가 knot마다 절대 NED roll/pitch를 실어
(`PlanMsg.rp`, `policy.rp_max_deg`에서 T1 클립), 필터는 크기·오일러 각속도·점프를 게이트, 스티처는 자세+각속도를
샘플(`sample_att`), `HwDobMpc._xref_ned_plan`은 플랜에 rp가 있을 때만 xref[3:5]와 `T(phi,theta)^-1`을 거친
body rate 행 xref[9:11](+ xref[11])을 채운다. K/M 토크는 engage 게이트 `engage.attitude_axes` 뒤에서
MANUAL_CONTROL 확장 축(`s` = pitch, `t` = roll, `enabled_extensions=0b11`; MAVLink 2 전용)으로 ArduSub에
가며, `axis_gain.roll_nm 13.2 / pitch_nm 7.2` [유도]로 스케일하고 0.2 / 0.3 [예측]으로 캡. 시뮬 래퍼
(`dobmpc_controller.set_target(roll_ref, pitch_ref)`, 6-튜플 샘플러)도 같은 레퍼런스 경로;
`verify/verify_attitude_hold.py`가 hold 그리드 + step/ramp/dropped/부호반전/gain/deadband/headroom 시나리오를 돈다.

기억할 결정:
* **변형 OFF = 4-DoF 경로 바이트 동일.** `compose_plan` / `PlanFilter` / `PlanStitcher`의 golden sha256 핀
  (test_policy_frames, test_plan_stream), 동결된 레거시 복사본 대비 `_xref_ned` / `_xref_ned_plan` 타일 비트 동일
  (test_attitude_axes, 시뮬 tests/test_dobmpc.py), 23바이트 MAVLink 프레임 동일, 변형 키를 뺀
  `config/hw_mpc.yaml` 해석 결과의 키별 동일.
* **rp가 사라지면 스텝이 아니라 hold 램프** (`_rp_hold`가 tick당 `pq_max_rad_s`·DT로 감쇠).
* **rate 행은 T^-1 경유**: 0.35 rad/s 램프에서 수평 rate 레퍼런스는 피크 3.105 deg 지연, T^-1 행은 0.287 deg;
  램프 피크 오차 7.252 deg
  (`bluerov2_mujoco_marinegym/recordings/20260926/attitude_hold_211921/results.csv`, `rate_*` / `ramp_*` 행).
  방법도 라벨도 둘 — **하나의 `rate_source`로 합산 금지**: 시뮬 `attitude_meta.rate_source`는 `fd_horizon_T_inv`
  (샘플한 호라이즌 위 np.gradient), 스테이션 `HwDobMpc` 라벨은 `stitcher_rate_T_inv`(스티처가 샘플한 오일러 rate를 T^-1로).
* **정상상태 창은 t_step + settle = 20 s부터**(`steady_from_s=20`): step 15 deg, plain mpc, e_pitch RMS 0.0884 deg,
  오버슈트 0.9991 %, 1-deg 정착 1.0 s, accept 1 (`recordings/20260926/attitude_hold_211813/results.csv`, `step,mpc` 행).
  이전 `attitude_hold_174607`의 step/ramp/rate/dropped 행은 창이 과도를 포함해 정상상태 값으로 **인용 불가**(과도 값은 동일).
* **dobmpc는 계속 게이트(`dobmpc_allowed: false`)**: 토크 게인 2배 오차에서 EAOB가 수직 스러스터를 포화
  (234 tick, max|f| 52 N), 평범한 MPC는 정상 오차 +0.22 deg (`recordings/20260926/attitude_hold_174607/results.csv`,
  `gain_*` 행); K/M이 조용히 버려질 때(참조 20 deg = `policy.rp_max_deg`, 임계 15 deg = `policy.div_max_rp_deg`) mpc·dobmpc
  모두 0.45 s에 `div_rp` 트립 — 임계 초과 연속 10 샘플 = 스테이션 0.5 s 디바운스, 선체는 20 deg에 접근하지 않음 — 그리고
  dobmpc는 크레딧되었으나 미전달된 K/M으로 pitch 6.83 deg까지 흘러 rad_max 14.5 cm
  (`recordings/20260926/attitude_hold_211917/results.csv`, `dropped_*` 행). 174607의 `dropped_*` 행은 임계 = 참조 크기라
  퇴화 — 인용 금지.
* heavy hold 그리드(mpc, dobmpc; NONE): e_roll/e_pitch RMS 0.07–0.08 deg, 15 deg에서 K/M 0.29/0.30 N·m,
  sat_ticks 0 (`recordings/20260926/attitude_hold_174544/results.csv`). heavy_gripper (0,20) deg: plain mpc
  정상상태 pitch 오차 −0.15 deg(수평에서 +0.29 deg) vs dobmpc −0.04 deg — 예측했던 ZG·W vs B·coBM 정상 기울기는
  부호만 확인, 크기는 Q_att 80이 대부분 닫아 1 deg 미만 (`recordings/20260926/attitude_hold_174602_gripper/results.csv`).
* **캡 계층(바깥이 마지막)** [인용 없는 값은 예측]: T0 라벨 범위(normaliser + DDIM clip) → T1 `compose_plan` 클립
  `rp_max_deg` 20 → T2 PlanFilter(`rp_reject_deg` 30 hard, `pq_max_rad_s` 0.35 dilation, `rp_jump_max_deg` 5 soft)
  → T3 `set_path_plan_ned` ValueError(`rp_reject` 초과) → T4 선 캡 `cap_roll 0.2 / cap_pitch 0.3`(첫 물 세션은
  `first_water_caps [0.1, 0.15]`). T1/T2의 근거 라벨 통계: |dpitch| p99.9 16.7–18.6 deg, max 21.1–27.2 deg; |droll|
  p99.9 7.8–9.3 deg (`data/20260926/rp_label_stats_174358.txt`, `rp_label_stats_174535.txt`); 피크 오일러 각속도
  |p|,|q| p90 0.453 rad/s(두 축 합산 최대, 원 66.7 ms 격자)라 0.35 rad/s 게이트가 200 ms 격자에서 2/1721 청크를 dilation
  (`data/20260926/0926_175145_offline/gate_pos_rpy_width.log`).
* **MANUAL 전용.** STABILIZE에선 펌웨어가 s/t를 lean-angle 목표로 재해석해 자체 자세 루프를 돌리므로 스테이션의
  K/M 토크는 같은 축 위의 두 번째 제어기가 된다; `attitude_axes.enabled`면 STABILIZE engage를 거부한다
  (lean-angle 캐스케이드는 범위 제외).
* **순서: plain mpc → 토크 게인 보정 → dobmpc.** `roll_nm 13.2 / pitch_nm 7.2`는 heave_n × 레버암에서 유도한 값
  [유도]이지 측정이 아니다; 첫 물 시험은 정책 없는 plain mpc 자세 STEP으로 게인을 맞추고, 그 다음에야
  `dobmpc_allowed`를 검토한다 — EAOB는 선이 나르지 않는 토크를 크레딧하기 때문(위 `gain_*` / `dropped_*` 근거).

기록: 미션 CSV 끝에 `rroll_deg, rpitch_deg, ax_roll, ax_pitch, rp_track`(OFF면 nan/0), `policy_plan.csv`는
`,reason,roll_deg,pitch_deg`로 끝남, meta schema 15 → 16에 `run.attitude_axes`, `trajectory.attitude_track`,
`policy.action_repr`, `controller.attitude_ref`. 범위 제외: `transport: rc_override`, STABILIZE lean-angle
캐스케이드, PID roll/pitch 루프, run_compare 자세 시나리오, 수중 arm 부호 프로브 미션은 이번 컷에 없음.
**실기 0런**: 펌웨어의 s/t 소비, K/M 부호, 토크 게인 전부 미확인(KNOWN_ISSUES.md 2026-09-26).

## 2026-10-01 — DVL이 추정기의 세 번째 센서가 된다: AprilTag + IMU + DVL (기본 OFF, DVL 미장착)

교체가 아니라 스위치 셋 뒤의 추가다. **하드웨어가 오기 전에 썼다**: Water Linked DVL A50 프로토콜 문서와 벤더 공개
데모 장치의 12 s 캡처(`rov_gui/tests/fixtures/dvl_a50_demo_capture_20261001.jsonl`)가 근거의 전부이고, 아래에 이 기체의
측정값은 하나도 없다. 2026-10-02에 세 리뷰어(안전 / 정확성 / 추정 이론) 검토를 거쳐 고쳤고, 바뀐 항목은 ⟲로 표시했다.
같은 날 타이밍 시뮬레이션과 독립 검토 세 번을 더 거쳐 다시 고쳤다. 지적은 전부 반증 검증을 통과한 것만 반영했다
(2차: 리뷰어 넷, 28건 확인. 3차, 그 수정에 대해: 9건 확인. 4차, 다시 그 수정에 대해: 8건 — 반증 검증 에이전트가
세션 한도로 돌지 못해 주 세션이 스크립트를 다시 돌려 확인. 5차: 워커 리뷰어 3건 + 다른 리뷰어의 중간 실행에서 3건 —
5차의 반증 검증은 둘 다 끝나지 못해 주 세션이 하나씩 재현·유도한 뒤 반영. 6차, 5차 수정에 대한 리뷰어 1명:
PASS_WITH_FIXES — 중간 1건 수정, 낮음 1건 문서화, DVL 없음/OUT의 명령이 커밋된 코드와 같은지 IMU 단까지 돌려 다시 재서
차이 0. 어느 회차에도 치명 0) — 그때 바뀐 항목은 ⟲⟲.

* **무엇이 바뀌나.** **수평** body 속도의 **출처**만. `StateAssembler.step(..., vel_xy=None, disp_xy=None)`
  (`rov_gui/control/state_assembler.py`): `vel_xy`가 주어지면 그 측정 (surge, sway)가 저역통과 태그 유한차분 대신
  `nu[0:2]`와 `nudot`의 transport 항에 들어간다. 위치(x, y, yaw = 태그 PnP), z, roll/pitch, 각속도는 그대로.
  `vel_xy=None`은 연산 단위로 DVL 이전 산술과 같다
  (`test_dvl.py::test_without_a_dvl_the_assembler_is_bit_identical_to_before`, float64 `==`). 구조는 운영자 결정
  (2026-10-01): 새 EKF가 아니라 기존 assembler의 속도원. EAOB 측정 sigma는 속도원이 바뀌어도 **바꾸지 않는다**.
* ⟲ **heave 속도는 쓰지 않는다.** `nu[2]`는 태그 유한차분에 남는다. 아래의 검증은 surge·sway만 보므로, 기울어 달린
  브래킷(pitch 오차 e는 한 행을 cos e배 할 뿐 — 20°도 통과), 틀린 vz 부호 규약, 틀린 lever arm은 어떤 게이트에도
  걸리지 않은 채 heave 루프에 들어갈 뻔했다. DVL의 w는 수직 검증이 생길 날을 위해 `dvl_w`로 기록만 한다.
* ⟲ **fix-age 보간은 적분이다.** `p = p_tag(t_capture) + R · ∫ v_dvl dt`(DVL 이력, 구간 선형 —
  `DvlAid.displacement`). "최신 속도 × 나이"는 fix가 묵는 동안 속도가 계속 갱신되면 적분이 아니다(kick 때 추정치가
  앞질렀다가 되돌아왔다).
* **왜 EKF가 아닌가.** 동기가 된 논문("Learning to Swim", arXiv:2410.00120v2)은 AprilTag + DVL + IMU를 EKF로
  묶었다고만 하고 상태벡터·잡음·지연 처리는 주지 않는다. 플랫폼 논문(CUREE, arXiv:2303.03943)은 Waterlinked DVL과
  IMU로 dead reckoning을 하고 전역 fix와 EKF로 융합한다고 적는다. 옮겨 올 것은 역할 분담이다: 태그 = 절대 위치,
  DVL = 직접 속도 + 태그가 없을 때의 다리. 데이터 없이 튜닝한 EKF는 전부 [예측]이고, 지금까지 기록된 모든 런이 난
  DVL-off 경로까지 바꿔 버린다.
* **스위치 셋.** `dvl.enabled` / `--dvl`은 TCP 클라이언트를 만든다(off = 아무것도 생성 안 됨). `dvl.use` / trajectory
  패널의 DVL 버튼은 추정기가 **소비**하는지를 정한다. OUT = shadow(수신·게이트·비교·기록만, 명령은 DVL이 없을 때와
  비트 단위로 같다). 버튼은 disengage 상태이고 컨트롤러 CSV가 닫혀 있을 때만 바뀐다(운영자 결정: 한 런 = 한 추정기).
  셋째는 tick마다 자동: 데이터 게이트에 걸린 샘플은 **그 tick만** 태그 유한차분으로 내려간다.
* ⟲ **자동은 DVL을 빼기만 한다.** (1) 한 engagement가 DVL을 쓸 수 있는지는 engage 때 한 번 정한다
  (`_dvl_arm_run`: 스위치 IN, 검증 충족, trip 아님) — 미션 도중 완성된 검증은 다음 engage부터 유효. (2) DVL이
  **틀렸다는 증거**는 TRIP이고, 운영자가 스위치를 OUT → IN으로 다시 넣을 때까지(검증 게이트도 통과해야) 래치되며,
  `require_proven`과 무관하다(1차 리뷰에서 opt-out이 비행 중 감시까지 조용히 꺼 버리는 것이 발견됐다). (3) lock 상실 /
  링크 정체 / 단발 이상값은 trip이 아니고, `reentry_s` 뒤 스스로 돌아온다. ⟲⟲ (4) 이번 세션의 데이터가 한 번이라도
  DVL이 틀렸다고 한 뒤에는 — trip이든, shadow에서 본 반박 적합이든 — IN에 **그 뒤에 만든 검증**이 필요하다.
  `require_proven: false`여도 그렇다: 그 opt-out은 반증이 하나도 없는 DVL의 첫 검증만 면제한다(감사에서 trip 1초 뒤에
  아무 근거 없이 재무장이 됐다 — trip이 자기 증거를 지웠기 때문). trip은 검증과 그 쌍들을 버린다. 장치 설정 변경도
  **모든 상태에서** 그렇게 한다(shadow에서 옛 쌍 59개 + 새 쌍 1개가 더 이상 없는 스트림의 검증을 완성시켰다). 래치
  아래서 다시 만든 검증도 다음 증거에 똑같이 무너진다(첫 구현은 모든 trip을 "이미 trip 아님"으로 막았다). (5) trip은
  DVL이 **오래 가는 방식으로** 틀렸을 때만이고 그런 방식은 둘이다: **장착** — 태그 속도가 DVL이 말하는 운동을 거의 다
  보고 있고(80 % 이상, `COHERENT_MIN`) 그것을 다른 방향에 둔다(정렬 감시, 적합의 coherent 반박) — 그리고 **false
  lock** — 태그가 무엇을 봤든 DVL이 태그 속도에서 8 cm/s 넘게 1초쯤 떨어져 있다(차이 감시; 메시지가 "a false bottom
  lock rather than the mount"라고 말한다). trip이 **아닌** 것: 리포트 몇 개짜리 글리치(한 번이든 몇 초마다든), 평균 0인
  DVL 잡음, DVL만 "움직여서" 나온 적합 반박. 이것들은 래치 없이 세 가지로 **빠지고** 스스로 돌아온다: **innovation
  hold**(거부된 샘플 → 그 뒤에 들어온 **새 카메라 프레임**의 태그 속도가 동의할 때까지; 같은 프레임이 묵는 것만으로는
  안 풀린다 — "신선한 프레임" 기준이었을 때는 태그가 0.25 s 넘게 늦으면 hold가 안 풀리거나 아예 안 걸렸다, 5차; 거부된
  샘플은 풀린 뒤에도 fix-age 적분에 쓰지 않는다 — 남아 있으면 추정에 9–11 cm가 들어가고 DVL 단이 실어 날랐다, 6차),
  **alignment hold**(정렬 감시가 "DVL의 운동이 태그가 보여 주지 않는 쪽으로 간다"고 읽는 동안 — 태그가 얼마나 보든),
  **반박 hold**(적합이 반박 중이고 마지막 반박 평가가 `REFUTE_HOLD_S` 3 s 안). 문턱마다 재현된 실패가 있다: "태그가
  보고 있다" 조건이 없을 때 리포트 둘 12 cm/s나 정지 중 3 cm/s 잡음이(2차), 그 조건이 절반일 때 잡음 2–2.5 cm/s,
  3–10 s마다의 4-리포트 글리치("스케일 1.3×" 반박), 정사각형 모서리의 4-리포트 글리치 한 번이(4차) 올바르고 검증된
  장착을 래치시켰다. 반박 hold가 없을 때는 잡음 속의 틀린 장착이 반박 중에도 쓰였고(3차), 3 s 만료가 없을 때는 조용해진
  DVL이 ~52 s 묶여 있었고(4차), innovation hold가 없을 때는 게이트가 거부하던 false lock이 fix가 묵자 다시 들어갔다(4차,
  아래).
* **"검증" 게이트.** 틀린 장착은 정지 상태에서 관측되지 않고, 움직이면 부호가 뒤집힌 속도 피드백이 된다. IN은
  (1) `dvl.mount_rpy_deg` / `mount_t_flu`를 자로 재서 채우고 (2) 이번 세션 데이터가 장착을 검증할 때까지 거부된다.
  검증 = (surge, sway) 쌍에 대한 최소제곱 `v_tag_fd ≈ A v_dvl` — 태그의 raw 2점 차분(중점 시각)과 그 시각으로 보간한
  DVL 속도, 회귀 변수는 DVL(더 깨끗한 신호), DVL 속도 2 cm/s 이상 쌍만. ⟲ 판정은 표준오차
  σ = 잔차 산포 / (√n · 약한 축 rms 속도)를 단 **세 갈래**: max|A − I| + 3σ ≤ 0.25면 proven,
  max|A − I| − 3σ > 0.25면 refuted, 그 밖은 "아직 판정 불가". 절사 재적합 1회로 태그 점프를 제외하고, 절사 후 쌍이
  60개 미만이면 판정을 보류한다(1차 리뷰가 재현: 창이 최소 개수로 줄어든 순간 태그 점프 하나가 맞는 장착을 "27°
  돌아갔다"고 판정). 이미 검증된 장착을 뒤집으려면 반박 판정이 연속 15회 필요. 장착 yaw에 더할 값과 스케일을 보고한다.
  합성 프로토타입뿐이다 [예측: 태그 잡음 0.8 mm@15 Hz, DVL 잡음 3 mm/s 가정]: 8~12 cm/s surge+sway 8 s → A ±0.01,
  yaw ±0.3°; 20/90/180° 장착, 거울상 축, 1.33 스케일은 각각 이름을 붙여 거부; 전진만/정지는 판정 보류.
  검증이 **못 보는 것**: 수직축, 장착 기울기, lever arm. ⟲⟲ 같은 적합이 `lag_s`도 보고한다 — 같은 스탬프의 DVL
  속도보다 태그 속도가 운동을 얼마나 늦게 기술하는지(잔차를 DVL 가속도에 회귀) — 게이트가 아니라 커미셔닝
  숫자로, 검증 로그에 같이 찍힌다: 여기의 모든 것이 두 센서를 스탬프로 짝짓기 때문이다. 가속도 잡음 때문에 0
  쪽으로 치우치고(아래 시뮬레이션에서 50 / 100 / 150 ms가 +33 / +49 / +73 ms로 읽힘 [예측]), 단일 주파수 타원
  운동에서는 lag가 A에 회전으로 흡수돼 0으로 읽힌다 — 직선 패스에서 재고 하한으로 읽는다.
* ⟲⟲ **비행 중: trip 둘, 게이트와 hold.** 두 trip(래치)은 **검증과 같은 증거**를 먹는다 — 새 태그 fix마다 생기는
  **시간 정렬된 쌍**: 태그의 raw 2점 차분과, 그 중점 시각으로 보간한 DVL 속도. 첫 구현처럼 tick의 저역통과 태그 속도와
  최신 DVL 샘플을 그대로 비교하지 않는다. 그렇게 하면 두 지연의 차이가 통째로 불일치로 잡히고, 아래 타이밍
  시뮬레이션에서 두 지연이 ~0.15 s 이상 다르고 기체가 0.5 Hz 이상으로 진동하면 **올바른** 장착이 trip했다
  [예측: 설계 도중의 관찰, 2026-10-02]. (a) 수평 차이 **벡터**를 1 s 저역통과한 크기가 8 cm/s 초과 — 지속 오프셋,
  오래 가는 false lock. 뒤집힌 장착도 4 cm/s 이하에서는 안 보이고 [유도: 차이 = 속도의 2배] 진동은 전혀 못 본다.
  쌍 하나의 차이는 **고정된 15 cm/s**(innov_max)로 자른다: 태그 fix 하나가 튀는 것(raw 차분으로는 m/s 단위)이나
  짧은 버스트 하나로는 trip할 수 없고, 쌍마다 크게 틀리는 DVL은 약 0.8 s에 trip한다. ⟲⟲ (자르는 값이 검증 산포의 4σ
  였을 때 3 mm 태그에서는 36 cm/s가 돼 3–4 리포트 글리치 하나가 0.4 s에 trip시켰다 — 4차.) (b) **정렬** — 최근 1 s에
  대한 검증의 축소판: DVL 속도의 주축 e 가운데 rms 2 cm/s 넘게 움직이는 방향마다, 최소제곱 사상 A: tag ← DVL에 대해
  `1 − |A e − e|`(같은 운동 1, 반전 −1, 90° 회전 −0.4). 이 값이 0.5보다 **표준오차 3배만큼** 낮으면 DVL을 hold하고,
  태그가 그 운동의 80 % 이상을 보는 상태로 0.4 s 이어지면 trip — 표준오차는 그 회귀 자체의 잔차에서 나오므로 느리거나
  잡음 많은 운동은 "증거 없음"이다. body
  축이 아니라 주축인 이유: 장착이 몇 도 돌아가 있으면 surge가 DVL의 sway로 새고, 축별 기울기는 그 누설을 "sway
  불일치"로 읽는다. 0.4 s를 기다리는 이유: 새 운동의 첫 한두 쌍이 그 방향 증거의 전부인데, 스탬프가 수십 ms만
  어긋나도 그 쌍들이 0으로 읽힌다. 부호가 틀린 피드백으로 **진동**하는 station hold를 잡는 쪽이 이것이고, 검증을
  통과하는 장착은 여기에 걸릴 수 없다 [유도: yaw 14.5° = 편차 0.25]. 쌍 하나의 가중치를 1/7로 제한해서 태그가 5 Hz
  이하로 와도 동작한다(제한 없이는 유효 쌍 수가 최소치 10 아래인 9에서 멈춰 감시가 조용히 꺼져 있었다). 게이트(그
  tick만): innovation(그 tick의 태그 속도에서 15 cm/s 이상 벗어남; 검증 잔차의 4σ와 a_max × (태그 나이 − 0.25 s)만큼
  넓어짐). 그 기준 속도는 태그 상태가 없는 tick 한두 개를 넘어 살아 있다(0.25 s까지, 묵은 만큼 한계를 넓힘) — 그런
  tick마다 지우면 태그가 느릴 때 속도원이 fix 주기마다 한 번씩 바뀌었다. 그러나 **그만큼 신선한 태그 속도가 없는
  tick은 DVL을 아예 쓰지 않는다**: 진짜 태그 공백 직후의 첫 tick이 그런 tick이고, 공백 동안 잡힌 false lock이 검사
  없이 들어갈 자리였다. jump(좋은 샘플 사이 |Δv| > a_max·Δt + 5 cm/s, a_max 2.0 m/s²
  [유도: 30 N / 19 kg ≈ 1.6]). fom·altitude는 fail-closed. 임계값 전부 [예측].
* ⟲⟲ **합성 타이밍 시뮬레이션이 말하는 것 — 실측이 아니다.** `rov_gui/tests/sim_dvl_monitors.py`는 실제 `DvlAid` +
  `StateAssembler`를 **미리 정한 운동**(개루프)과 가정한 센서 숫자(태그 15 Hz·0.13 s 지연·0.8 mm, DVL 12 Hz·0.19 s
  지연·3 mm/s·팔 0.2 m, tick 20 Hz)로 돌린다. 2026-10-02 실행의 출력은
  `rov_gui/tests/fixtures/sim_dvl_monitors_20261002.txt`이고 여기 숫자는 전부 [예측]. 올바른 장착은 한 번도 trip하지
  않았다: 9가지 운동 × 12 seed, 같은 것에 태그 점프를 넣은 경우, 지연 5조합, 글리치·잡음 21가지 조건(모서리 포함 한
  번의 4-리포트 ±14 cm/s, 3 mm 태그 + 3–4 리포트 25–30 cm/s, 3–10 s마다의 1–4 리포트 글리치, DVL 잡음 1–5 cm/s,
  태그가 0.28·0.35 s 늦을 때의 글리치) × 네 가지 운동(0.3 m/s 왕복 포함), 그리고 opt-out에서 감시의 속도 바닥 근처 300 s. 글리치와 잡음은 대신 가용성을 깎는다 — 래치 없이
  빠진다: 3 s마다 글리치면 정지 중 39–60 %의 tick만 DVL, 잡음 2.5 cm/s면 48 %, 3 cm/s 이상은 거의 검증되지 않는다.
  **오래 가는** false lock은 의도대로 trip한다: +14 cm/s × 1 s 또는 +10 cm/s × 2 s(한 번이든 반복이든); +10 cm/s × 1 s,
  +14 cm/s × 0.5 s는 아니다. opt-out으로 날린 반전 축·90° 장착은 IN 뒤
  0.8–1.9 s에 trip(5+2 cm/s 대각 운동의 sway 반전은 8.6 s). opt-out에서 **못 잡는 것**: station keeping 속도(1–2 cm/s)의
  모든 틀린 장착, ~2 cm/s rms 미만으로 움직이는 반전 축; 다른 축에서 14.5° 이내의 **직선** 운동에서의 반전 축 — 속도와 무관(거기서 축 반전은 그 각도의 2배 회전과
  구별되지 않는다 [유도]); ~15 cm/s 미만에서의 30° 회전; ~8 cm/s 미만에서의 2× 스케일. 그런 칸에서는 tick의 98 %가
  틀린 DVL로 난다. 적합이 반박이라도 할 수 있는 경우(6 cm/s 진동의 2× 스케일)는 hold-back이 그것을 16 %로 줄인다.
  기본 경로에서는 전부 검증에 걸린다(직선 운동만으로는 검증이 성립하지도 않는다). 잔여 yaw 12°나 스케일 1.2×로 검증을 **통과한** 장착은 0.4 m/s
  이상에서 trip한다(검증 허용은 상대, 차이 문턱은 절대) — 이제 검증 로그가 몇 m/s부터인지와 장착을 몇 도 고칠지를
  같이 말한다. 잡음 큰 DVL(3 cm/s)은 거의 검증되지 않고 OUT에 머문다 — 적합이 DVL 자신의 잡음을 무시해서 사상이 0
  쪽으로 끌린다. 1 cm/s에서는 달라지는 것이 없다. 알려진 민감도: 촬영 스탬프 오차 + 0.5–1.5 Hz **타원** 운동이
  지속되면 올바른 장착도 trip한다(±50 ms에서 2/36·3/36, 100 ms 8/36, 150 ms 8/36; 직선 운동에서는 0건). ω로 도는
  속도의 시간 이동 δ는 겉보기 회전 ω·δ이기 때문이다. 틀린 lever arm(3, 10 cm)은 아무것도 trip시키지 않는다: 그것이
  만드는 가짜 속도는 DVL만 보는 운동이고, 3 cm/s [유도: 0.3 rad/s × 0.1 m]는 차이 문턱 아래다. 이 시뮬레이션에는 제어기가 없어서, 부호가 틀린 피드백이 운동을 키우는
  과정은 들어 있지 않다.
* **Lever arm.** `v_origin = R_frd_dvl v_dvl − ω × r`. 장치 프레임은 x 전방(LED), y 우현, z 아래
  [스펙: docs.waterlinked.com/dvl/axes]라서 LED 전방 장착은 단위행렬. 0.3 rad/s × 0.2 m = 6 cm/s [유도] — 미션
  속도와 같은 크기라 생략 불가. ⟲ ω는 ATTITUDE 이력에서 **샘플이 참이었던 시각**의 값으로 보간한다. 최신 값을 쓰면
  yaw가 변하는 동안 최대 ~7 cm/s가 남는다 [유도, 리뷰]. ⟲⟲ 팔이 0이 아닌데 그 시각의 각속도가 없으면 그 샘플은 쓰지
  않는다(보정 안 된 값은 body 원점 속도가 아니다).
* **태그 소실, STATION 전용.** `station_bridge.py`에 tier `dvl` 추가: 마지막 fix에서 DVL 속도를 적분해 x/y를 잇고
  (오토파일럿 roll/pitch, 자이로 yaw로 회전), `dvl_hold_s` 10 s [예측]까지 모든 축 유지, DVL이 끊김 없이 쓸 수 있는
  동안만. 그 뒤는 같은 경과 시간으로 기존 IMU/COAST 사다리, 되돌아가지 않음. ⟲ 태그가 없는 동안 DVL을 교차검증할
  것이 없으므로 station hold의 전제로 묶는다: DVL이 `dvl_max_speed_m_s` 0.3을 넘거나 적분 위치가 마지막 fix에서
  `dvl_max_disp_m` 0.5 넘게 벗어나면 그 dropout의 DVL 단을 닫는다. ⟲⟲ 상한 때문에 닫히면 아래 단은 그 DVL이 적분한
  위치가 아니라 **마지막 fix에서 정지 상태로** 시작한다(감사에서 추정 오차 0.34 m가 `imu_hold_s`가 끝날 때까지 제어기를
  밀었다). DVL이 그냥 사라진 경우에는 그 마지막 위치에서 이어간다. 단이 적분하지 못한 tick — 브리지가 한 tick 못 버틴
  경우, 브리지 대상이 아닌 고장 — 이 있으면 다음 새 fix까지 닫힌다. 그리고 이 단은 태그가 있던 마지막 tick에 **실제로
  DVL로 날고 있었을 때만** 열린다: 태그가 반박하고 있어서 innovation 게이트가 빼놓던 DVL이 그 태그가 사라졌다고
  믿을 만해지지는 않는다. ⟲⟲ 태그가 실제로 사라지는 순서에서도 — fix가 안 오고 assembler가 `tag_stale_s`까지 묵은 fix를
  이어 쓰는 동안 게이트 한계가 넓어진다 — innovation hold가 그런 DVL을 계속 빼 둔다(3차: 첫 dropout tick에 받아 써서
  0.5 m, 4차: 넓어진 게이트로 0.6 m). 닫힌 단이 돌아가는 "마지막 fix"는 **fix 자체의 위치**다 — 마지막 태그 tick의
  추정치에는 그 DVL의 적분이 이미 들어 있을 수 있다(4차: 0.125 m 어긋남). 대가: 태그가 보이는 동안 세게(~1 m/s²)
  차인 순간에는 정직한 DVL도 hold되고, 그때 태그가 사라지면 기존 사다리가 맡는다 — DVL이 없을 때와 같다. ⟲⟲ DVL 단이
  IMU 단에 넘기는 것은 **위치뿐**이다: IMU dead reckoner는 DVL 단 tick마다 정지 상태로, 태그 tick에는 DVL로 날던 tick이어도
  **태그 속도**로(DVL 이전 값) 앵커된다 — 그냥 끝나 버린 false lock의 속도를 물려받아 정지 기체에서 0.67–1.2 m를
  적분했다(5차). 그런 blind tick에 대한 사다리 자체의 처리는 커밋된
  코드 그대로다(hold 시계를 다시 시작한다; KNOWN_ISSUES): 중간에 시계를 계속 돌리게 했던 수정은 DVL이 아예 없는 런의
  단계를 바꾸는 것이어서 되돌렸다. ⟲ 이 단의 `nudot`은 0이 아니라 assembler와 같은
  식(가속도계 + 중력 − ω × v)으로 만든다 — EAOB에 "움직이는데 가속은 0"이라고 말하지 않기 위해. ⟲ 복귀 로그의
  dead-reckoning 오차는 dropout이 **DVL 단에서 끝났을 때만** 남긴다(아니면 그 값은 대부분 DVL 이후의 표류다).
  같은 구간의 자이로 yaw 오차도 같이 남긴다. 경로·정책·replay 미션은 여전히 stale-fix로 disengage(운영자 결정).
* **시각.** ⟲ 장치 스탬프는 최근 30 s 중 가장 빨리 도착한 리포트를 기준으로 스테이션 시계에 올린다(`DeviceClock`:
  offset = min(도착 − time_of_transmission), `t_valid = time_of_validity + offset`). 장치 시계 동기화가 필요 없고,
  TCP 정체 뒤 밀린 리포트가 방금 것으로 찍히지 않고 진짜 나이를 갖는다. 벤더 문서의 예시 리포트는 전송 시점에
  189 ms 묵었다 [스펙 예시 1건] — 태그 경로 지연과 같은 크기(추정 리뷰어의 예산: DVL ≈ 0.22–0.24 s vs 태그 FD ≈
  0.21–0.27 s [유도])라, 실측 전까지 기대 이득은 지연이 아니라 잡음이다.
* **기록.** meta schema 16 → 17: `dvl` 블록 항상 기록(use_at_engage, armed_at_engage, proof, tripped, 장치 설정과
  변경 이력, `vel_src`별·fallback 사유별 행 수). CSV 끝 15열 `vel_src, nu_u/v/w, fd_u/v/w, dvl_u/v/w, dvl_ok,
  dvl_valid, dvl_fom, dvl_alt_m, dvl_age_s`(body FRD) — 이 CSV가 body 속도를 담는 건 처음. `bridge_tier`에 `dvl`
  추가. 장치 원문은 `<stem>_dvl.jsonl`. 속도 유래 통계는 `vel_src`를 넘어, 런은 `dvl.armed_at_engage`를 넘어
  **합치지 말 것**.
* **RL 추종기.** `rl` / `rl_pwm`은 태그 유한차분 추정기를 시뮬레이션해 학습했다. DVL 속도는 본 적 없는 관측 분포다.
  막지 않고 engage 때 warn + 기록.
* **읽기 전용.** 스테이션은 `get_config`(읽기)를 연결당 한 번과 `config_poll_s` 10 s마다 보낸다 — ⟲ 첫 응답 이후
  장치의 `mounting_rotation_offset` / `speed_of_sound`가 바뀌면 trip — 그 외에는 아무것도 보내지 않는다(설정 쓰기,
  dead-reckoning 리셋, ping 트리거 없음). 장치의 `position_local` dead reckoning은 원시 로그에만 남고 소비하지 않는다.
* ⟲ **같이 고친 기존 결함 둘**(DVL 없이도 적용된다): `MpcStatus.bridge_tier`가 `--imu-dr` 블록 안에서만 채워져
  일반 런에서는 패널의 "NO TAG — COASTING" 칩이 뜰 수 없었다. 그리고 `disengage`가 닫는 meta를 쓰기 전에 사다리를
  reset해서, 디스크의 `station_bridge.dropouts` / `reached_coast` / `last_recovery`가 지금까지 모든 런에서
  0 / 0 / null로 닫혔다.

검증: `rov_gui/tests/test_dvl.py` 96/96, `rov_gui/tests/demo_e2e_dvl.py`(정상 장착·90° 틀린 장착, 실제 창 + 데모
백엔드), 위의 타이밍 시뮬레이션 — 장치 없이. 다른 `rov_gui/tests` 스위트는 이 변경 전과 같다. **실기 런 0회**: 장착값,
모든 [예측] 임계값, 지연, 잡음, 이 풀 바닥에서의 거동이 전부 미측정 (KNOWN_ISSUES.md 2026-10-01).

## 2026-10-02 — 논문의 추정기를 우리 것 옆에: robot_localization EKF를 shadow로, 그것으로 나는 스위치와 함께

운영자가 Learning-to-Swim 추정기를 "논문과 똑같은 방식으로" 만들어 2026-10-01 속도원 설계와 비교하고 싶어 했다.
**아래 어느 것도 이 기체에서의 측정이 아니다** (수중 런 0회).

* **어떤 EKF인가.** 논문(arXiv:2410.00120v2, Fig. 2: "IMU, DVL, April Tag → Fused state estimation (50Hz)", "ROS로
  구현한" EKF)과 플랫폼 논문(CUREE, arXiv:2303.03943)은 상태벡터·잡음·지연 처리를 밝히지 않고, 공개 코드
  (warplab/isaac-auv-env)는 시뮬레이터뿐이다. 그 연구실의 상태추정 저장소(warplab/robot-state-estimation, 비어 있음)
  설명이 "A state estimation codebase for marine robots using the ROS localization package"다. 그래서 "논문 방식"을
  ROS `robot_localization`의 `ekf_localization_node`로 본다 — 추론이고, 추론이라고 적는다.
* **옮긴 것** (`rov_gui/control/rl_ekf.py`, cra-ros-pkg/robot_localization noetic-devel @ 9ef26a57): 15상태 운동학
  모델과 Jacobian(`ekf.cpp::predict`; 전달함수의 중심차분과 대조, 최대 오차 3e-10), Joseph form `correct`(각도 wrap된
  innovation, Mahalanobis 게이트), `processMeasurement`(필터보다 오래된 측정은 **예측 없이** 보정), 측정 큐와 타이머
  (`integrateMeasurements`, sensor-timeout 예측, "corrected data" 미발행), 토픽별 스탬프 순서, IMU 3분할,
  `prepareTwist`(**상태의** 각속도로 lever arm, mask·공분산 회전), `prepareAcceleration`(IMU **메시지의** 자세로 중력
  제거), `smooth_lagged_data`(`revertTo` / `clearExpiredHistory`). r_l 기본값이 있는 곳은 전부 기본값(process noise,
  초기 공분산 1e-9, sensor_timeout = 1/frequency, smoothing 없음), 주파수는 논문의 50 Hz. 좌표계는 NED/FRD이고
  바뀌는 부호는 중력 하나. 50 Hz 타이머는 20 Hz tick 안에서 재생한다(`RlEkfNode.advance`): 각 입력은 **도착** 시각
  이후 첫 사이클 전에 전달된다 — ROS 콜백이 돌았을 시점.
* **선택 (논문에 없음; hw_mpc.yaml `ekf:`에 [예측]).** 태그는 x, y, yaw(+ 스테이션 z 출처가 태그면 z), 기체 IMU는
  roll, pitch, 각속도, 중력 제거된 가속도(나침반 yaw 제외), 압력 수심은 assembler와 같은 offset의 z, DVL은 측정된
  장착으로 (vx, vy) — assembler와 **같은 정보**라서 차이는 융합 방식에서만 난다. IMU 잡음 = mavros imu 플러그인
  기본값. 태그·수심 잡음과 DVL 대체값은 추정.
* **shadow 먼저, 스위치는 그 다음.** 기본 켜짐, 매 tick 기록(CSV schema 18: `est` + `ekf_*` 15열; meta `ekf`;
  `<stem>_ekf_inputs.jsonl` — `rov_gui.tools.ekf_replay`가 다른 설정으로 오프라인 재실행, 라이브 설정이면 수렴 후
  라이브 열과 1 mm 안). 패널 EST 버튼이 컨트롤러에 EKF 상태를 넘긴다(`meas_ctrl`, imu_dr 선례); `meas` — 플롯,
  px/py, 모든 인터록 — 는 태그 상태 그대로. disengage + CSV 닫힘일 때만, 세션을 EKF로 시작하는 건 `--ekf control`뿐,
  EKF로 바꾸거나 engage하려면 EKF와 태그 상태가 5 cm / 3° 안에서 일치해야 한다("여기" 타깃은 태그 상태에서 정해짐).
  EKF로 나는 런은 **끝난다**(조용히 태그로 돌아가지 않음): EKF fault, 융합하는 입력이 묵음(입력별 — r_l은 timeout
  예측으로 자기 stamp를 늘 "지금"으로 둔다), station bridge의 **첫** tick(태그 소실은 실어 나르지 않는다: bridge
  사다리는 EKF를 보정 없는 IMU 적분에 두고 coast retarget이 표류를 타깃에 써넣는다), 또는 신선한 태그 상태에서
  15 cm(x/y, z), 10°(yaw, roll/pitch), 15 cm/s(0.5 s 저역통과한 차이)를 0.25 s 넘게 벗어남. EKF는 DVL 버튼이 IN이고 trip이 없을 때만 DVL을
  융합한다(DVL의 장착 검증이 그대로 적용; 게이트 없는 필터는 `ekf_replay --dvl-all`로 오프라인).
* **리뷰 (2026-10-02).** safety-code-reviewer: FAIL → 치명(station bridge가 EKF 런을 실어 나름)과 높음 3건(작동 안
  하는 출력 나이 게이트, DVL 게이트 우회, xy/yaw만 보는 발산 검사)을 위와 같이 수정, 묵은 가속도와 낮음 6건도. r_l을
  독립 재전사한 기준과의 이식 충실도 검사: 낮음/정보 등급만 있는 faithful(smoothing on/off, 동적 잡음, 거부를 거쳐
  상태 차이 ≤ 5.5e-14); 싼 것들 수정(angles::normalize_angle wrap, 끼어든 재전달 dedupe, 나침반 결측, skip 집계).
  그 수정의 안전 재감사: PASS_WITH_FIXES, 새 중간 2건 수정 — 묵음/bridge EKF fault를 태그 경로처럼 tag_stale_hold_s로
  debounce(마지막 50 Hz 사이클 직후 도착한 프레임이 멀쩡한 런을 끝냈다), 그동안 명령을 내지 않음(bridge 상태로 대신
  날지 않음); EKF의 DVL 공급이 검증 게이트(`DvlAid.refusal`)도 확인(YAML `dvl.use: true`는 그 게이트 없이 IN). 속도
  게이트는 EKF−태그 차이 벡터를 0.5 s 저역통과해서 본다(태그 유한차분이 잡음). 3차: PASS_WITH_FIXES, hold tick에서
  명령 경로 없음; 새 중간 1건 수정 — hold tick이 태그 경로의 stale debounce를 더 이상 0으로 되돌리지 않는다(무명령 tick
  두 종류를 카운터 하나로 세서 한 에피소드가 tag_stale_hold_s로 묶임; 띄엄띄엄 늦은 프레임이 ~2.9 s까지 늘렸다 [유도]);
  hold 행은 est = hold + observe 표식.
* **합성 비교가 말하는 것** [예측: `rov_gui/tests/sim_ekf_compare.py`, 출력
  `rov_gui/tests/fixtures/sim_ekf_compare_20261002.txt`; 개루프, 센서 수치 전부 가정]: r_l 기본 지연 처리에서 EKF는
  ~0.12 s 묵은 태그 프레임으로 현재 상태를 보정해서 위치가 속도 × 지연만큼 뒤처지고(0.3 m/s 왕복: assembler 5 mm
  대 33 mm rms) yaw도 회전율 × 지연만큼 뒤처진다(회전 중 0.3° 대 1.7°). `smooth_lagged_data: true`가 그걸 없애고
  (6 mm) 속도는 모두 중 가장 좋다(0.72 cm/s 대 3.2–3.6) — 가속도계도 제때 융합되므로. mavros 가속도 잡음값은
  가속도계를 거의 완전히 믿는다: DVL 없이 0.1 m/s² 바이어스면 속도 오차 15 cm/s. r_l 기본 process noise 때문에
  필터는 태그 프레임을 거의 그대로 받는다(정지 상태에서도 1σ가 cm 단위).

검증: `rov_gui/tests/test_rl_ekf.py` 38/38, `demo_e2e_ekf.py`(실제 창); `test_dvl.py` 96/96(schema 18 헤더·번호 갱신 후); 다른 `rov_gui/tests`
스위트는 변경 전 기준선. **실기 런 0회.** KNOWN_ISSUES.md 2026-10-02.

## 2026-10-03 — roll/pitch 제어가 LOW 선택이 된다: PID_7D, MPC_Tuned_7D, RL_7D (스테이션, 실기 0런)

운용자 요청: "pid_7d, mpc_tuned_7d, RL_7d 버전" — 7-dim(`pos_rpy_width`) Diffusion Policy를 위해 pitch·roll까지 제어하는
추종자. 디스크에 7-dim 체크포인트는 없다(5-dim뿐; `run_ablations.sh peg_7d_vitclip`은 실행된 적 없음). RL은 운용자 결정으로
코드만 준비하고 학습은 보류했다.

* **스위치가 아니라 LOW가 정한다.** `geometry.LOW_7D_BASE = {pid_7d: pid, mpc_tuned_7d: mpc_tuned, rl_7d: rl}`. K/M은
  LOW가 `*_7d`이고 `engage.attitude_axes.enabled`(이제 스테이션 "능력" — sink가 기동 때 한 번 읽음)일 때만 MANUAL_CONTROL
  s/t로 나간다. 능력이 꺼져 있으면 `*_7d`는 실기 engage 거부, 켜져 있으면 다른 LOW는 모두 4-DoF로 engage(09-26~10-02에는
  스위치만 켜면 plain mpc/mpc_tuned가 K/M을 냈고 pid/rl/rl_pwm/mpcc는 거부됐다 — 그 기간 물속 런 0). 7-dim 플랜 추종 =
  `*_7d` LOW + 7-dim ckpt; `policy.attitude_track`은 이제 거부권(기본 true); 4-DoF LOW의 7-dim은 dropped-and-logged.
  meta: `controller.type` = LOW(mpc_tuned_7d는 `base_type: mpc_tuned`), `run.attitude_axes.low/low_attitude`,
  `trajectory.attitude_track_low/_veto`.
* **mpc_tuned_7d** = 09-26의 NMPC 자세 경로를 HwDobMpc 모드 `mpc_tuned`로(워커가 키를 매핑; 브리지의 `tuned`는
  `endswith("_tuned")`라 키를 그대로 넘기면 isotropic 비용으로 난다). 솔버 변경 없음.
* **pid_7d**(`rov_gui/control/pid.py HwPid7d`): x/y/z/yaw는 pid 그대로, 여기에
  `τ = k_ff[cosθr sinφr, sinθr] + Kp e + Kd(T⁻¹(기준)·기준 rate − 자이로 p,q) + I`. Kp/Kd는 같은 3차 pole placement(식에서
  k = d = 0)로 **ωn 0.5 rad/s**(ζ 0.9, α 0.2, heavy_gripper I_eff 0.50/0.90 kg·m² [유도]), Ki 0.15 별도(식의 Iαωn³은 k = 0
  가정), 복원 FF 1.35 N·m/rad를 **기준 각도**로, I는 N·m로 저장해 선 토크의 0.8배로 클램프, 실제 송신 K/M 기반 조건부 적분,
  heave trim 측정 자세로 회전, yaw D항은 Euler yaw rate, D5 램프·T3 검증은 HwDobMpc와 공유(`control/attitude_ref.py`).
  ωn은 관성이 아니라 **지연**이 정한다(ATTITUDE가 틱 시점에 ~52 ms 묵음 [측정 data/20261001/1001_154825/mpc_154825.csv] +
  ZOH + sink 위상 + 추진기 지연); `omega_derate`는 적용하지 않는다. 여유 [유도: control-theory-advisor 자문,
  `rov_gui/tests/sim_pid7d_attitude.py`로 재계산 → `rov_gui/tests/fixtures/sim_pid7d_attitude_20261003.txt`]: 총 지연 0.2 s·
  구동 이득 ×0.4~×2.5에서 PM ≥ ~50°는 복원강성 k ≤ ~2.3(roll)/3.6(pitch) N·m/rad일 때만, k ≈ 20(RL 쪽 로그 적합)이면
  3–41° — 선체 공진 + 지연이라 이득으로 못 고친다. 그래서 물에서 루프를 닫기 전에 pulse-and-release 시험(부호·지연·g/I·
  k/I·감쇠)이 먼저다. 같은 산출물의 실제 클래스 폐루프(first-water 캡): k 1.35 → 10° pitch 유지 오차 −0.03°, 오버슈트 2.7°;
  k 20 → 4–9° 정상오차가 남는데 축은 캡 아래다. 안전 감사(2026-10-03, WARN, HIGH 1건)가 같은 메커니즘이 **뒤집힌 선 부호**를
  숨긴다는 것을 보였다(PID가 캡 아래의 틀린 기울기에 앉음) → 수정: HwPid7d가 "권위 소진"(명령 클립, 적분 한계 + 오차 > 1°,
  또는 오차가 20° 게이트 밖)을 알리고 인터록 (iii)이 캡 고정과 같이 판정
  (`test_low_7d.py::test_interlock_iii_trips_pid_7d_on_an_inverted_sign_below_the_cap`). 감사 반영 추가: pid_7d 실기 engage는
  pulse-and-release 기록(`pid_7d.pulse_test`, 자유감쇠 주기 ≥ 3.2 s) 또는 기록되는 우회가 필요, 비유한 자세 → K/M 0 + n_fail,
  rl_7d K/M은 학습 캡으로 클립·기록 기준은 실제로 준(클립된) 자세.
* **rl_7d**(`rov_gui/control/rl_policy.py HwRl7d`): `obs_spec.attitude_reference.trained`가 true인 axes6 export만
  (fail-closed — 수평 학습 export와 레이아웃이 같다), `q_des = quat_from_euler_xyz(roll_ref, −pitch_ref, −yaw_ref)`를 학습
  envelope으로 클립, 4축 slew가 0이어도 roll/pitch는 slew(`allocation.slew_axes`), 캡이 학습 캡보다 낮으면 engage 거부
  (기록되는 우회 있음 — 2026-10-01 rl_pwm cap 0.25 런이 클립된 정책으로 한계진동). LOW `rl`은 axes6를 더 이상 받지 않는다.
  RL_controller(git 아님; 수정 전 사본 `RL_controller/_pre_att7_backup_20261003/`): `goals.py` 선택적 roll/pitch 기준
  (`att_*`, 끄면 난수 소비 없음: 신·구 12 조합 비트 동일, `RL_controller/logs/smoke/att7_goals_byte_identity_20261003.txt`),
  env q_des, att 목표에서 `ang_error_mode yaw`/`rew_scale_tilt > 0` 거부, `cob_trim_x_range_m`, 프리셋 `traj10att`, exporter
  `attitude_reference`, `scripts/_pipeline_att7_seed.sh`(미실행). Isaac 스모크(3 iteration, 코드 확인) 통과:
  `RL_controller/logs/smoke/smoke_att7_20261003.log`. RL_PWM(pwm10_s1)은 수평 q_des로 학습돼 수평만 유지하고 7-dim 정책의
  roll/pitch는 추종하지 않는다.

검증: `rov_gui/tests/test_low_7d.py` 19/19, 갱신된 자세 테스트, 스테이션 전체 1170 통과 / 15 실패(변경 전 기준선과 같은 15건) /
8 스킵, `demo_e2e.py pid_7d policy rp`·`mpc_tuned_7d policy rp` OK(`rov_gui/tests/fixtures/demo_e2e_*_policy_rp_20261003.log`,
장난감 플랜트 — 배선 검사이지 성능 아님), RL_controller 테스트 53/53. **실기 0런, 모든 게인 [예측].**

## 2026-10-03 (2) — RL_PWM이 roll/pitch 추종을 학습(att7pwm, 정지수) + 스테이션 LOW RL_PWM_7D (실기 0런)

운용자 요청: 파도·조류 외란 없이 RL_PWM을 지금 학습. 수평 학습된 pwm10_s1은 기울어진 q_des를 본 적이 없어 7-dim 정책의 roll/pitch가 분포 밖이었다.

* **학습**(RL_controller, 정지수): `scripts/_pipeline_att7pwm_seed.sh` = preset `traj10att`(traj10 DR + roll/pitch 목표 경로 |roll| ≤ 10°,
  |pitch| ≤ 20°, ≤ 0.35 rad/s, 30 % 수평 + 에피소드별 CB 트림; `sea:` 없음 → p_mode NONE), action_mode pwm, ang_error_mode full, 절대 기울기 항 없음,
  60° 전복 종료, 35° 태그 드롭아웃, PPO std 하한 0.05. 시드 0..2, 2048 env × 1200 it, 각 34:38–34:45 [측정 `logs/train_att7pwm_s*.log`].
* **평가**(새 `scripts/eval_attitude.py`, 정지수, nominal 동역학, 프로그램 15개 × 세계 cob_x 0/−0.029/−0.035; [측정
  `RL_controller/logs/eval/ATT_COMPARE_20261003.md`]): 재학습은 효과가 있다 — 세계 0 pitch RMS 1.46°(s0) vs 같은 기울어진 목표를 받은 수평 학습
  pwm10_s1 10.93°. 그러나 승격하기엔 부족하다: 트림 세계 정상 pitch 오차 최대 7–9°, 수평 유지 중 roll ±3° 흔들림(e_roll RMS 2.8 vs 0.6°),
  수직 |throttle| ~0.7 vs 0.3, 포화 틱 24–38 %(s0)/57–83 %(s1)/~100 %(s2). 세 시드 모두 [예측] 기준(|ss pitch| ≤ 2°, |ss roll| ≤ 1.5°) 미달.
  최선 s0를 **꺼진 후보**로 `rov_gui/control/rl_policies/att7pwm_s0/`에 둔다.
* **스테이션 LOW `rl_pwm_7d`**(`HwRlPwm7d`): rl_pwm 전송 그대로(펄스·Lua·MANUAL·기체 게이트·전송 단위 상한 `rl_pwm.pwm_cap` — HwRlPwm도 이제
  PWM_CAP_KEY로 읽음) + HwRl7d의 q_des(플랜 roll/pitch를 학습 envelope으로 클립, attitude_reference.trained fail-closed). LOW_7D_BASE 밖(s/t 게이트
  없음); 워커 펄스 비트 `_attitude_pulses`가 모든 pwm LOW에 기울기 **천장**을 켜고(그동안 rl_pwm에는 없었다 — 동작 변경), rl_pwm_7d에는 roll/pitch
  추종(`rp_track_live`: 진리표, div_rp)을 켠다. CSV `rp_track` 2 = 펄스로 추종; meta `run.attitude_axes.rp_transport/ceiling_armed/rp_track_live`;
  pwm coast 기록 정정(8개 1500 µs, 아무것도 유지 안 함). test_rl_pwm.py 수리(11건이 옛 출하 캡 0.25 고정): 26/26; 새 test_rl_pwm_7d.py 13/13;
  전체 1199 통과 / 무관한 기존 실패 4건. **실기 0런; 후보는 꺼져 있다.**
 * **리뷰 수정(같은 날, workflow wf_b9f3f143-d12, 확인 6건 전부 low)**: RL 7d 추종자(rl_7d, rl_pwm_7d)가 이제 **학습한 기준 변화율**로
   난다 — 기준을 envelope으로 클립한 뒤 틱당 최대 `attitude_reference.rate_max_rad_s`(att7 export는 0.35 rad/s) × dt만 움직이고,
   교전 시작은 학습 목표처럼 수평에서 출발한다. 그 키가 없는 export는 거부. D5 유지 램프 감쇠를 step **시작**으로 옮겨(pid_7d 포함)
   틱 뒤 CSV·인터록 (iii)이 읽는 기준 = 그 틱이 쓴 기준. CSV `rp_track` 2는 정책 펄스가 실제로 나간 틱에만(coast·게이트 유예 틱은
   8개 1500 µs라 0). 전체 1212 통과 / 같은 기존 실패 4건.
 * **켬(같은 날, 운용자 요청)**: `rl_pwm_7d.policy_dir: rov_gui/control/rl_policies/att7pwm_s0` — 출하 config로 스테이션이
   RL_PWM_7D를 빌드·선택한다(전송 캡 1.0 = 학습 캡 1.0, 추진기 방향 = sink export와 같음). 위 시뮬 기준 미달은 그대로; 실기 0런.

## 2026-10-03 (3) — 정책이 컬러를 본다: RGB-only 체크포인트에 C3 컬러 프레임을 먹인다 (naive 광학, 실기 0런)

* **왜:** RGB-only DP(`umi_rgb_5d_peg5d_rgb_dinov3b_9_27_9_28_20261001_211106`, DINOv3-B, 5-dim)를 풀장에서 3번 START했는데
  아무것도 안 그려졌다. 스테이션은 `camera0_depth`만 먹였고, 세션은 계약 자신의 키로 warm-up을 통과해 READY가 됐고, 매 predict가
  KeyError — 플랜 0개 [측정: `data/20261003/1003_151227/events.log`, `plans.jsonl`, meta `policy.worker.counters.infer_errors` 39→158].
* **바뀐 것(rov_gui):** PolicyWorker가 로드된 체크포인트 계약이 요구하는 이미지를 먹인다. `camera0_depth`는 기존 경로 그대로
  (변경 전 워커와 무작위 시나리오 8개·플랜 173개 비트 동일, 2026-10-03 리뷰). `camera0_rgb`는 C3 **컬러**(가운데 렌즈, 태그 인식과
  같은 프레임)를 `ColorObsBuilder`로 — 정사각 센터 크롭, 224 `INTER_AREA`, BGR→RGB, /255; 학습 store의 `camera0_rgb`를 그대로
  재현한다(리뷰: 학습 프레임 4장, 최대 차 0). 각 시도는 **가장 최근 태그 fix의 컬러 프레임**을 기준으로 해서(스탬프는 localizer 것을
  재사용해 동일) 이미지와 proprio 행이 같은 순간이다. depth 규칙(최신 프레임)을 쓰면 현실적인 localizer 지연에서 플랜 0개였다(리뷰
  시뮬레이션). 지원하지 않는 키·학습 채널 순서 불명/BGR·(3,R,R)가 아닌 shape·컬러 공급 없는 백엔드는 ERROR(ARM 거부)이고, 상태는
  계약을 적용한 뒤에야 READY다.
* **아닌 것:** 광학은 NAIVE(운영자 결정) — 학습은 핸드헬드 컬러(공기 중, warp 없음), 수중 C3 컬러는 224에서 약 1.8배 확대
  [유도: 초점거리 178 vs 319 px]. 기록: meta `policy.worker.obs_mode`/`obs.optics`, 플랜마다 `depth_src = c3_rgb` — warp한 RGB
  재학습 런과 합산 금지. FS 스케줄은 RGB 체크포인트에 적용 안 됨(effective `free`). 오프라인 재생·held-out 도구는 아직 depth 전용(KNOWN_ISSUES).
* **근거:** test_policy_worker 새 테스트 9, test_policy_obs 2, 데모 e2e `mpc_tuned policy rgb` 플랜 24/24 `c3_rgb`·설치
  [측정: `rov_gui/tests/fixtures/demo_e2e_mpc_tuned_policy_rgb_20261003.log`]; 안전 감사 PASS_WITH_FIXES(치명·높음 0) 반영.
  **실기 0런** — 첫 런은 LOW None(observe)으로, `obs_t − obs_fix_t[-1]`과 skip 카운터를 확인.

## 2026-10-03 (4) — `policy.action_repr: auto`: 패널에서 고른 체크포인트가 5-dim/7-dim을 정하고 ARM 때 고정; RL_PWM_7D 첫 수중 런 진단(2026-10-04)

* **바뀐 것:** 운용자 요청(7-dim을 고를 때 YAML 수정 없이). `policy.action_repr`에 설정 센티널 `auto`(hw_mpc.yaml·land_dp.yaml
  배포값; 키가 없을 때 코드 기본은 pos_yaw_width). auto면 START 때 정책 워커가 쥔 체크포인트의 계약을, ARM 거부 판정이 본 **바로 그**
  PolicyStatus 객체에서 `geometry.resolve_policy_action_repr`로 정해 `rp["action_repr_pinned"]`/`scen["action_repr"]`에 고정(그 사이 다른
  상태가 들어오면 ARM 거부). ""·pose10d는 거부(기본값 대체 없음), intake는 핀과만 비교. 명시값은 예전과 동일하게 동작·기록. 추종 여부는
  여전히 LOW가 결정(`geometry.policy_rp_tracked`), LOW 자동 전환 없음.
* **운용자 표시:** ckpt 칸 `READY · 5-dim | 7-dim`(START가 거부할 ckpt면 빨간 이유), START 전 미션 줄 예보(TRACKED 경고색 /
  DROPPED-AND-LOGGED / 수평 유지; 거부권·s/t 능력·land dry-run·contouring LOW·export 없는 LOW 반영), ARM 뒤 칩 `· 7d rp TRACKED|DROPPED`.
* **기록(schema 18 유지):** trajectory.action_repr = 핀(풀링 키), policy.action_repr = 라이브 미션 핀 → 이 CSV가 열린 동안 고정된 마지막
  핀 → null, policy.config.action_repr = "auto".
* **근거:** test_policy auto 테스트, 패널 예보 테스트, 변형 10/10 포착, 전체 1224 통과 / 기존 실패 4, 데모 [측정:
  `rov_gui/tests/fixtures/demo_e2e_mpc_tuned_policy_auto_20261004.log`(304행 rp_track 0),
  `demo_e2e_pid_7d_policy_rp_auto_20261004.log`(308/308행 rp_track 1)] — 합성 플랜트, 배선 검사.
* **RL_PWM_7D + 7-dim peg 체크포인트 첫 수중 런(2026-10-03 20:27, `data/20261003/1003_202721/`) 실패:** 교전 3회 1.7/6.3/2.5 s, 모두
  조종사 해제. 진단(기록 재계산): att7pwm_s0가 수평 기준에서도 수직 공동수축(T5/T8 위·T6/T7 아래 +230~+340 µs, 시뮬 level 프로그램과
  같음 — 부호는 같은 폴더 믹서 비행으로 정상 확인), 전류 22~42 A; 공칭 +19 N 위 명령에도 실제 침강(UI 영상 HUD 압력 깊이 0.66→0.83 m;
  peg −12 N은 DR 밖, 순력은 두 ~50 N의 차); 차분 속도에 대한 고이득이 태그 z 튐(교전 3에서 >20 mm 9회, 교전 2는 0회)을 ±90 N 수직
  반전으로 바꿈. DP는 주 원인 아님(교전 1 플랜 0, 교전 3 이탈은 첫 플랜 전); 교전 2의 +14° 기수 들림 정체는 선미 착저 가능성이 큼.
  스테이션 타이밍 정상. 전문은 KNOWN_ISSUES; 재학습엔 공동수축 벌점·payload 부력 DR·태그 튐/보조카메라 잡음 모델이 필요.

## 2026-10-04 — MPC_Tuned 프로필: 추종자 튜닝을 파일로 저장하고 DP 체크포인트처럼 패널에서 고른다 (실기 0런)

* **바뀐 것:** 운용자 요청 — 지금까지 날린 mpc_tuned 튜닝을 저장하고, DP ckpt처럼 골라서 불러와 적용, 기본값은 현재 값.
  프로필(`config/mpc_tuned_profiles/NAME.yaml`, `rov_gui/control/mpc_tune.py`)은 세 섹션: `apply`(Q/QN/R, along/cross 분할,
  policy q_scale/along_scale, vehicle_net_buoyancy_n, axis_cap, axis_slew_per_s — 키 전부 필수), `compare`(rov_model, ctrl_hz,
  N, dt, u_max, plant damping, axis_gain — **적용 안 함**: damping은 mpc/dobmpc가 공유하는 생성 solver 안에 있음), `record`(참고).
  HwDobMpc가 프로필을 **쥐고**, 모드 `mpc_tuned`에서만 **적용**(LOW mpc_tuned_7d도 그 모드로 난다): baseline W는 cost_set, heave
  trim은 외란 파라미터, policy 노브는 `policy_follow`, cap/slew는 MpcWorker가 이미 읽는 컨트롤러 훅(cap은 낮추기만). 다른 모드로
  바꾸는 순간 빌드 W와 config trim이 solver에 다시 쓰인다. 운용자 결정: heave trim도 적용 대상(프로필이 payload 가정을 함께 가짐).
* **기본값 = 오늘:** `20261004_default`는 2026-10-04 시점 hw_mpc.yaml + params.py 스냅샷(분할 0.25/4.0, q_scale 4.0,
  along_scale 3.0, trim −12 N peg 장착, cap 0.5, slew 1.5/s, params Q/QN/R). hw_mpc.yaml `mpc_tuned_profile`이 이를 가리키므로
  mpc_tuned는 이전과 똑같이 난다. 프로필을 쥔 동안 mpc_tuned는 hw_mpc.yaml의 `mpc_tuned:` 블록, `policy.q_scale/along_scale`,
  `vehicle_net_buoyancy_n`을 읽지 않는다(`--mpc-tune none`이면 예전처럼).
* **거부:** 교전 중·컨트롤러 CSV 열림·READY 전·로드/형상 불일치 파일은 거부(이전 프로필 유지 — 교체는 트랜잭션, solver 쓰기가
  실패하면 되돌림). 기동 시 지정한 프로필이 로드 실패하면 hw_mpc.yaml로 대신 날지 않고 mpc_tuned 교전을 **거부**.
* **안전 리뷰(2026-10-04, WARN, 치명 0) → 수정:** axis_slew_per_s > 0 필수 + cap처럼 조이기만(프로필이 떨림 방지 slew를 끌 수
  있었음); Q/QN 자세·위치 행 > 0 필수(x/y 가중 0이면 CSV open 때 meta에서 0 나누기); `path_cost`·`compare` 완비 필수(compare 누락은
  WARN); `name:` = 파일 이름; 프로필 trim ≠ hw_mpc.yaml trim이면 WARN(peg 함정, KNOWN_ISSUES); `save` 기본은 **쥔 프로필**의 apply
  (`--from-config`/`--from`) — 프로필을 쥔 동안 hw_mpc.yaml 블록은 mpc_tuned가 나는 값이 아니므로.
* **기록(경계):** meta `controller.mpc_tune`(profile, path, sha1, active, apply, warnings), `controller.weights_source`,
  `controller.Q/QN/R` = 이 모드가 실제로 난 baseline(전에는 항상 params.py 값), `trajectory.along_scale`·`follow_knobs_source`.
  2026-10-04 이전 런엔 `mpc_tune` 블록이 없고 hw_mpc.yaml 값으로 날았다.
* **근거:** `rov_gui/tests/test_mpc_tune.py` 16 통과(실제 acados 브리지: 위치 가중 4배 프로필이 mpc_tuned 렌치를 바꾸고, 이후
  plain mpc는 비트 단위로 동일), test_path_cost 27/28(실패 1은 기존 `|vehicle_net_buoyancy_n| ≤ 2` 단언 vs peg −12 N),
  test_low_7d + test_control 161, policy/offline/replay/attitude/station-bridge/policy-worker 361; 데모 e2e `mpc_tuned square`·
  `mpc_tuned policy` 기본 프로필로 OK(meta: profile 20261004_default, active, q_scale 4 / along 3 출처 프로필) — 합성 플랜트, 배선
  검사. **실기 0런** — 다만 기본 프로필이 이전 값을 그대로 재현하므로 다른 프로필을 고르기 전까지 동작은 같다.
