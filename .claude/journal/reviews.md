# P2 — Parallel-review syntheses

<!-- Populated when P2 (/review-change) lands. One block per review:
     ## YYYY-MM-DD <artifact>
     reviewers: <agents> · 합의: … · 이견: … · 리스크: … · 최소수정: … -->

## 2026-06-22 — Heavy 관성 텐서 [0.3291,0.6347,0.6109] (parallel-axis 유도, farol USD 대신)
reviewers: simulation-advisor · control-theory-advisor · underwater-robotics-advisor (병렬·독립)

- **합의(3/3):** (a) parallel-axis 유도법 타당, farol [0.21,0.245,0.245] 기각 정당(코너 질량 추가가 Iz를 낮출 수 없음 → 비물리·Gazebo hand-tune). (b) **in-sim non-issue** — MJCF plant `diaginertia` == predictor(params)라 closed-loop이 관성값에 불변; 논쟁은 sim-to-real에서만 유효. (c) 0.15kg/thruster는 예산 bookkeeping(실제 T200 ~0.344kg)이나 delta만 스케일 → 영향 작음(~0.03).
- **주요 발견(control+underwater 합의, sim도 지적):** dry inertia를 소수 3자리까지 다듬는 동안 **회전 added mass [0.12,0.12,0.12] (BlueROVHeavy.yaml)는 등방 placeholder로 방치** — 생략한 유체 r² 항이 계산한 ΔI와 동급 크기("두 동급 항 중 작은 쪽만 정밀화"). 권장 anisotropic ≈ Kp'0.07 / Mq'0.18 / Nr'0.22 (von Benzon&Fossen 2022 order로 주장됨 → P3에서 검증).
- **sim 지적:** 텐서 3중 복제(rov_model.py:54 · MJCF diaginertia · compute_heavy_inertia.py 출력)를 묶는 verify/test 없음; 스크립트가 baseline 하드코딩 → drift 위험.
- **control 지적:** Heavy는 roll/pitch commanded(NU=6) → 관성이 "don't care"→"~10-20% 중요"로 승격. EAOB는 quasi-static 오차는 흡수하나 ω̇-상관(기동 중) 오차는 못 함.
- **이견:** sim/control "in-sim 수정 불요(값 유지)" vs underwater "added mass 지금 고쳐라" → 화해: in-sim 깨진 것 없음, 단 added-mass가 진짜 fidelity gap이라 다음 투자는 dry inertia가 아니라 거기.
- **리스크:** 회전 added-mass 과소·등방(sim-to-real 자세 fidelity); 텐서 3중 복제 drift.
- **최소수정(영향순):** ① 민감도 스윕(plant 고정, predictor 관성 farol vs 유도 → pitch/roll·ŵ RMS 비교 → EAOB 흡수 정량화) ② BlueROVHeavy.yaml 회전 added mass anisotropic화 ③ test_heavy_inertia 3중 일치 게이트 + 스크립트 baseline import.
- **검증법:** roll/pitch step 후 ŵ의 ω̇-상관 펄스 관찰(관성오차 지문); yaw step→Iz 직접; bifilar(공기)+in-water decay 실측 bound.
