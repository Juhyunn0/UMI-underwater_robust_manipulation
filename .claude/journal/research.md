# P3 — Research + Verify outputs

<!-- Populated when P3 (/research-verify) lands. One block per question:
     ## YYYY-MM-DD <question>
     verified (+sources) · rejected (+why) · provenance -->

## 2026-06-22 — BlueROV2 회전 added mass: von Benzon/Wu 출처·수치 검증 (P2 후속)
researcher: control-theory-advisor · verifier: verifier (적대적)

- **Verified:** von Benzon et al. 2022 (JMSE 10(12):1898, DOI 10.3390/jmse10121898) 실재, **Fossen은 저자 아님**(원 리뷰의 "von Benzon & Fossen"은 오기 — Fossen은 모델 프레임워크). Eidsvik(2015) 경험식 오차 **회전 30–100%** / translational 10–20% → 회전 added mass는 order-of-magnitude only.
- **Uncertain(2차 출처로만 확증):** 회전 added mass ≈ Kp'0.189 / Mq'0.135 / Nr'0.222 kg·m²(von Benzon/Eidsvik). 여러 인용 논문이 동일 set 재현하나 **원문 표를 1차로 확인 못 함**(verifier도 PDF fetch 불가). researcher가 댄 provenance(RG fig 366613202=Hadi/Sensors)는 verifier가 **반증** — 그 그림은 PMC9824147(다른 논문) 소유.
- **Rejected:** 원 P2 주장 패턴 "roll 최소(Kp'≈0.07), Nr'≫Kp'" → **틀림**. 실데이터는 **pitch 최소·yaw 최대**(Nr'0.222 > Kp'0.189 > Mq'0.135, ~1.6×). 0.07/0.18/0.22 triple은 어느 데이터셋과도 불일치.
- **추가:** 경쟁 데이터셋 **Nr'≈0.40**(BR forum/Wu Heavy lineage) 존재 → 문헌이 회전값에 불합의(0.40을 arXiv:2405.00269에 귀속한 것도 약함).
- **판정/권고:** **지금 [0.12,0.12,0.12] 바꾸지 말 것.** ① 후보 0.189/0.135/0.222가 2차 확증뿐 ② 방법 오차 30–100%라 1.6× 비등방이 noise 내부(false precision) ③ 경쟁셋 0.40과 우열 근거 없음 ④ 우리 translational은 MarineGym set([5.5,12.7,14.57])이라 회전값만 이식하면 식별 혼합. 비등방 원하면 von Benzon **set 전체** 이식+order-of-mag 명시; 진짜 해법은 **자체 system ID**(free-decay/pendulum; docs/REAL_HYDRO_VERIFICATION.md).
- **provenance(핵심수치):** Kp'0.189/Mq'0.135/Nr'0.222 → von Benzon 2022(Eidsvik)로 *추정*, 1차 표 미확인·2차 인용으로만 확증(신뢰도 medium).

sources: vbn.aau.dk PDF · doi.org/10.3390/jmse10121898 · mdpi.com/2077-1312/10/12/1898 · arXiv:2405.00269 · BR forum 13065 · Wu2018 Flinders
