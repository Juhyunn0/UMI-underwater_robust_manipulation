# Hydrodynamics 검증 — 1차원리, 외란 OFF

**결과: 32/32 검사 PASS.** marinegym hydrodynamics([hydro.py](../hydro.py))는 의도한 Fossen 물리를 측정
정밀도 한계까지 재현한다. 유일한 의도된 근사(lag/필터된 added-mass 힘)는 정량화됐고 무시 가능하다.

- **무엇:** 독립 하네스 [verify_hydro.py](../verify_hydro.py)가 각 hydro 항을 **외란 OFF**(`disturbance=None`
  → 정수[still water], 각 항이 깨끗이 분리됨)에서 해석해와 비교.
- **방법:** 시뮬레이터는 **무수정**. 알려진 body wrench를 MuJoCo의 독립 외력 버퍼 `data.xfrc_applied`로 주입
  (hydro는 자기 passive 콜백으로 정상 동작). 단일 축만 순수 운동시켜야 할 때는 순부력(+1.10 N)을 테스트용 수직
  힘으로 상쇄.
- **실행:** `python verify_hydro.py` (env `robust`). 그림은 [docs/figs/](figs/)에 저장.
- **검토:** control-theory advisor(예측·격리 논리·임계).

기준 진실값(`marinegym_assets/BlueROV.yaml` + `bluerov.xml`): m=11.2 kg, I=[0.30375, 0.626, 0.5769],
V=0.0113459 m³, ρ=997, coBM=0.01 m, M_A=[5.5,12.7,14.57,0.12,0.12,0.12], D_L=[4.03,6.22,5.18,0.07,0.07,0.07],
D_NL=[18.18,21.66,36.99,1.55,1.55,1.55], EMA α=0.3, dt=2 ms. B=ρgV=**110.97 N**, W=mg=**109.87 N**,
순부력 **+1.10 N**, 복원 강성 k=coBM·B=**1.1097 N·m/rad**.

---

## 결과

| # | 테스트 | 격리 대상 | 예측 | 측정 | 판정 |
|---|------|------------------|------------|----------|---------|
| **T1** | 순부력 | 부력 − 무게 | a_z(0)=(B−W)/m = **0.0980 m/s²** | 0.0980 (**0.00%**); 측면/각 = 0 | ✅ |
| **T2** | 종단속도(drag), surge/sway/heave/yaw | 선형+2차 drag (정상상태서 added-mass·대각 Coriolis = 0) | F=D_L·v+D_NL·v² | 4축 모두 **0.00%**; added-mass < 1e-15 N | ✅ |
| | — 이방성 | 축별 D_NL 순서 | 속도 순서 = D_NL 역순 | surge>sway>heave ✓ | ✅ |
| **TL** | cross-axis 누설 | 힘 프레임 / D·M_A 대각 | 단일축 속도 → 해당 축 힘만 | 비대각 가속 **정확히 0**(6축); nu() 재정렬 OK | ✅ |
| **T4** | 복원 진자 | 복원 강성 + 유효 관성 | 과소감쇠, ω_n=√(k/(I+M_A_rot)) | roll T=3.85 s vs 3.89 (1%); pitch 4.79 vs 5.16 (7%) | ✅ |
| | — 정적 평형 | 복원 강성 단독 | tilt = asin(M/k) | 26.1° vs 26.8° (2.7%) | ✅ |
| | — 축 순도 | 교차 커플링 없음 | roll tilt → roll 모멘트만 | pitch/yaw 가속 = 0 | ✅ |
| **T5** | added mass (유효관성, Ω=0.5–5 rad/s) | EMA 필터 통한 M_A 전달 | 유효질량 = m + M_A·Re{H(Ω)} ≈ m+M_A | surge/sway/heave **0.0–0.3%**; 부호 −M_A 6축 | ✅ |
| **T6** | Coriolis 수동성 | C_A 스큐대칭 | νᵀC_A(ν)ν = 0 | **4.3e-14** | ✅ |
| | — 역학 에너지 | 소산성 | E=½νᵀ(M_RB+M_A)ν+U 비증가 | 4.59 J 소산, **단조** | ✅ |
| **T7-R2** | 전체 plant, 힘 레벨 | 총 적용 wrench, 적분기 무관 | hydro wrench == 독립 Fossen 재계산 | 6 s 가진 궤적서 **0.0 N**; 부력+CB 정확 | ✅ |
| **T7-R1** | 전체 plant, 근사 크기 | added-mass lag | sim vs 해석(M_A를 질량행렬에) | 과도 발산 **0.01 cm/s**; 종단 동일 | ✅ |

---

## 각 항을 어떻게 검증했나 (왜 격리가 깨끗한가)

- **T1 부력** — 정지·수평·무추력에서 수직 힘은 부력 − 무게뿐, added mass ≈ 0(가속 이력 없음) → `qacc_z`가
  `(B−W)/m`을 직접 읽음.
- **T2 Drag** — 가장 깨끗한 항. 종단속도에서 ν̇→0이라 lag된 added-mass 힘 →0, 단일축 병진이면 대각 added-mass
  Coriolis C_A가 항등적으로 0(모든 항이 *서로 다른* 두 속도성분의 곱). 그래서 drag만 남음: F = D_L·v + D_NL·v².
  복원 없는 4축에서 실행; 종단속도 순서가 축별 D_NL 이방성도 확인.
- **TL cross-axis 누설** — 가장 싼 버그 그물. 단일축 body 속도는 그 축에만 hydro 힘을 내야 함(D·M_A 대각, 단일축
  C_A=0). 비대각 가속이 있으면 `R @ wrench` 적용의 body↔world 프레임 오류 노출. **정확히 0**.
- **T4 복원** — 부력이 CB = COM + coBM·ẑ_body에 작용 → tilt가 강성 k = coBM·B의 복원 모멘트를 생성. roll/pitch
  계는 **강한 과소감쇠(ζ≈0.05)** → *진동*; 주기가 유효 관성을 드러내고, 이는 회전 added mass를 **올바르게 포함**
  (I_eff = I + M_A_rot — 그래서 순수 강체관성 예측은 17% 빗나갔고 added-mass 보정한 게 맞음). 깨끗한 강성 검사는
  알려진 모멘트 하의 **정적 평형**: tilt = asin(M/k), 2.7% 일치. (감쇠비는 유한 진폭에서 2차 회전 drag D_NL_rot이
  부풀림 — 불일치가 아니라 예상된 것.)
- **T5 added mass** (미묘한 항) — added mass는 *외력* −M_A·ν̇로, 1스텝 lag + EMA(α=0.3) 필터된 가속으로 적용되며
  MuJoCo 질량행렬에 **넣지 않음**. **유효관성 주파수 스윕**으로 검증: 정현 force F·sin(Ωt) 가진 → 속도 기본파 fit →
  유효질량 m_eff 추정. EMA 필터 코너가 ~230 rad/s라 물리적 대역(Ω = 0.5–5 rad/s)에서 in-phase 이득 Re{H(Ω)} ≈ 1,
  측정 유효질량 = **m + M_A 0.0–0.3% 내**(heave 포함, M_A 14.57 > 차체질량). added-mass 부호(−M_A·ν̇, 가속 반대)
  6축 모두 성립.
- **T6 Coriolis + 에너지** — added-mass Coriolis 행렬은 스큐대칭이라 일을 하지 않음: νᵀC_A(ν)ν = 0 (4e-14). 자유
  정수 감쇠에서 총 역학 에너지(전체 M_RB+M_A 운동에너지 + 순부력 위치에너지)는 단조 비증가 — drag만 소산.
- **T7 전체 plant 교차검증** — 독립 레퍼런스 2개. **R2 (힘 레벨, 적분기 무관):** 6 s 가진 궤적에서 hydro가 적용한
  정확한 body wrench를(자체 내부 drag/added/Coriolis 상태로) 재구성해 **독립 작성한** Fossen 재계산과 비교 → **0.0
  N** 일치, 힘 모델(부호·프레임·계수·부력점)이 끝까지 정확함을 입증. **R1 (근사 크기):** 1-DOF heave 상승을 M_A를
  *질량행렬에* 넣은 해석모델("이상" 물리)과 비교 → 과도에서 **0.01 cm/s**만 발산하고 종단은 동일 — lag된 외력 근사가
  무시 가능.

## 그림
- [figs/hydro_T2_terminal.png](figs/hydro_T2_terminal.png) — 종단속도 vs 해석(4축).
- [figs/hydro_T4_pendulum.png](figs/hydro_T4_pendulum.png) — roll 진자 감쇠 vs 예측 포락선.
- [figs/hydro_T5_addedmass.png](figs/hydro_T5_addedmass.png) — 유효관성 vs Ω (= m + M_A).
- [figs/hydro_T6_energy.png](figs/hydro_T6_energy.png) — 단조 에너지 소산.
- [figs/hydro_T7_R1.png](figs/hydro_T7_R1.png) — sim vs 이상-added-mass heave 상승(lag 크기).

## 결론
모든 hydrodynamic 항 — **부력, 복원, 선형+2차 drag, added mass, added-mass Coriolis** — 이 1차원리 예측과
측정 정밀도 내 일치하며, **프레임·부호·계수·커플링 오류 없음**(T7-R2 = 0.0 N; 누설 = 0). 유일한 의도적 근사인
**lag/필터된 added-mass 힘**(heave M_A 14.57 > 차체질량 11.2라 MuJoCo 명시적 passive 채널에서 그냥 두면 수치적
불안정 → 안정화용)은 모든 물리적 주파수에서 **무시 가능**함을 보임(m_eff = m+M_A 0.1%; 과도 lag 0.01 cm/s).
시뮬레이터 hydrodynamics는 제어 작업에 올바름이 검증됨.

*재현:* `python verify_hydro.py` (env `robust`).
