# FOV_AUDIT — C3의 공장 캘리브레이션은 공기 중인가 수중인가

**결론: 수중(underwater) 캘리브레이션이다.** 세 카메라 전부 평판포트 Snell 예측과
3.5° 안쪽으로 일치하고, 공기 중 스펙과는 최대 41.9° 벌어진다.

- 스크립트: [`calib/fov_audit.py`](fov_audit.py)
- 산출물 재생성: `python calib/fov_audit.py --audit` / `--selftest`
- 실행 환경: `~/.venvs/c3-depthai` (python 3.10.12 / opencv 4.11.0 / numpy 1.26.4)
- 대상 데이터: `c3_camera/datasets/*/calibration.json` — **10개 덤프 전부 동일한 한 벌**
  (2026-07-29 ~ 2026-08-03, 재플래시 흔적 없음). 대표: `dataset_20260803_162223/`

---

## 왜 이 문서가 있는가

2026-08-04 이전까지 이 질문의 근거는 우리 소스에 하드코딩된 주석이었다
(`c3_camera/c3_collect.py:330` — `"IN-AIR calibration..."`). 그 문자열은 아무것도
인용하지 않았고, 매 데이터셋의 `calibration.json` `note` 필드로 찍혀 나가면서
디바이스가 알려준 사실처럼 보였다. **틀린 가정이었다.** 이 문서는 그 주석을
측정으로 대체한다.

## 방법

캘리브레이션은 K와 왜곡 벡터일 뿐, 화각을 명시하지 않는다. 하지만 둘을 합치면
"어떤 방향의 광선이 센서 어디에 맺히는가"가 정해지므로 화각은 복원 가능하다.
광축에서 각도 θ를 쓸어가며 `cv2.projectPoints`로 **순방향 투영**해, 상이 이미지
경계에 정확히 닿는 θ를 이분법으로 찾는다.

```
r_u = tan(theta)                                    이상(무왜곡) 반경
r_d = r_u * (1 + k1 s + k2 s^2 + k3 s^3)            OpenCV rational 모델
          / (1 + k4 s + k5 s^2 + k6 s^3),  s = r_u^2
u   = cx + fx * r_d
```

**역투영(`cv2.undistortPoints`)을 쓰지 않은 것은 의도적이다.** 그쪽은 왜곡을 반복법으로
뒤집는데, 강하게 왜곡된 광각 렌즈에서 조용히 수렴 실패할 수 있고 그게 바로 지금
시험 대상 영역이다. 순방향은 닫힌 형태의 다항식 평가라 그런 실패 모드가 없다.
쓸기 도중 투영이 단조인지도 확인하며, 단조가 깨지면 모델이 유효 영역을 벗어난
것이므로 값을 반환하지 않고 **유효 한계를 보고**한다.

평판포트에서는 창면에서 Snell 법칙이 성립한다:

```
sin(theta_air) = n * sin(theta_water),    n = 1.333
```

따라서 **수중에서 캘리브된 카메라는 같은 렌즈의 공기 중보다 좁은 화각을 보고한다.**
이 비대칭이 두 가설을 가르는 신호이고, 이 렌즈에서는 두 가설이 약 41° 떨어져 있어
어떤 캘리브레이션 품질 오차로도 설명되지 않는 크기다.

## 결과

```
camera                          size        fx  measured  air spec  water pred   delta  valid<=
-----------------------------------------------------------------------------------------------
CAM_A (colour IMX378)      3840x2160    3080.3     63.7d     95.0d       67.2d   -3.5d    85.0d
CAM_B (mono left OV9282)    1280x800     757.1     85.6d    127.0d       84.3d   +1.2d    85.0d
CAM_C (mono right OV9282)    1280x800     760.3     85.1d    127.0d       84.3d   +0.8d    85.0d

reverse check — 측정된 수중 화각을 Snell로 공기 중으로 되돌리면:
  CAM_A: 63.7d -> 89.4d  (spec 95d,  delta -5.6d)
  CAM_B: 85.6d -> 129.7d (spec 127d, delta +2.7d)
  CAM_C: 85.1d -> 128.7d (spec 127d, delta +1.7d)

worst |measured - water prediction| =  3.5 deg
worst |measured - air spec|         = 41.9 deg
VERDICT: UNDERWATER calibration
```

**세 카메라가 독립적으로** 수중 가설을 지지한다. CAM_A(컬러)는 별개의 센서·별개의
렌즈·별개의 해상도인데도 같은 결론을 준다.

### 화각 정의에 대한 주의

위 표의 HFOV는 주점에서 **가장 가까운** 좌우 경계까지, 즉 `min(cx, W-1-cx)`를 반폭으로
쓴다(전 시야가 확실히 보장되는 보수적 정의). CAM_A는 `cx = 1905.16`이 중앙
(1919.5)보다 왼쪽이라 이 선택이 값을 바꾼다 — 오른쪽 경계 기준으로 재면 **64.4°**가
나오고 수중 예측과의 차이는 −2.8°가 된다. 결론에는 영향이 없다.

### 공기 중 스펙의 출처

`AIR_SPEC_HFOV = {CAM_A: 95, CAM_B: 127, CAM_C: 127}`은 Luxonis OAK-D (Pro) W의
**스펙시트 값이며 측정치가 아니다.** 이 값들은 검증 대상 가설이지 증거가 아니다.
(초판에서 CAM_A를 108°로 잘못 잡았고, 그 탓에 CAM_A가 어디에도 안 맞는 것처럼 보였다.
그 불일치에 "4:3→16:9 크롭" 이라는 설명을 붙였는데 **그 설명 자체가 틀렸다** — 16:9
크롭은 위아래를 잘라내므로 수평 화각을 보존한다. 안 맞는 값에 없는 이유를 붙인
것이었고, 스펙을 고치자 CAM_A는 세 번째 증거가 되었다.)

## 통제 실험 — 스크립트 자체 감사

결론 전체가 이 스크립트 하나에 얹혀 있으므로 `--selftest`가 네 개의 통제를 돌린다.
가장 중요한 것은 **C2**: "이 스크립트는 무엇을 넣든 ~85°를 뱉는 것 아니냐"를 직접 반증한다.

```
C1  pinhole, zero distortion — 이분법이 정확한 화각을 복원하는가?
  [PASS] pinhole 60d: got 59.96d, want 59.96+-0.01d
  [PASS] pinhole 95d: got 94.96d, want 94.96+-0.01d
  [PASS] pinhole 127d: got 126.96d, want 126.96+-0.01d

C2  CRITICAL — *진짜 127도* 광각 카메라가 127도로 되읽히는가?
  [PASS] synthetic 127d fisheye: got 127.00d, want 127.00+-2.0d (fitted fx=576.6, valid<=85d)

C3  실제 C3 왜곡으로 닫힌 루프: Snell 적용 -> 재피팅 -> 재측정
  [PASS] CAM_B refitted to air: got 129.75d, want 127.00+-4.0d (fx 757 -> 568)
  [PASS] CAM_C refitted to air: got 128.68d, want 127.00+-4.0d (fx 760 -> 570)

C4  guard — 유효 영역을 벗어난 모델은 반환하지 말고 보고해야 한다
  [PASS] non-monotonic model flagged: valid only up to 23.5d

ALL CONTROLS PASSED
```

- **C1**은 이분법·투영 수학이 편향 없음을 보인다.
- **C2**가 핵심 통제다. 등거리 어안(진짜 HFOV 127°)을 만들어 OpenCV rational 모델에
  피팅한 뒤 같은 스크립트로 재면 **127.00°**가 나온다. 즉 스크립트도 모델도
  127°를 표현할 능력이 있다 — **85°라는 값은 코드나 모델의 구조적 산물이 아니다.**
  (원래 원했던 통제는 순정 in-air OAK-D W 덤프에 같은 스크립트를 돌리는 것이었으나
  그런 덤프를 지금 구할 수 없어, 합성 카메라로 동등한 반증력을 확보했다.
  실제 벤치 유닛이 생기면 `--calib`로 바로 돌려 이 통제를 강화할 것.)
- **C3**은 실제 측정된 왜곡을 그대로 쓰는 닫힌 루프다. C3의 수중 모델에 Snell을
  적용해 같은 물리 렌즈의 공기 중 대응을 만들고 rational 모델을 다시 피팅하면
  **129.75° / 128.68°**, 스펙 127°와 2.7°/1.7° 차이. 부수적으로 `fx 757 -> 568`,
  정확히 1.333배 관계다. 이 값은 사용자가 독립적으로 손계산한 129.7°와 일치한다.

## 결과가 뜻하는 것

수중 캘리브레이션으로 **공기 중에서** 찍으면

```
Z_reported = (fx_water / fx_air) * Z_true  ~  1.33 * Z_true      (중심부)
```

즉 depth가 약 33% 멀게 나온다. **다만 깨끗한 스칼라가 아니다.** 왜곡 모델까지
매질이 어긋나 있으므로 주변부로 갈수록 반경 방향 추가 오차가 붙는다. 위 표의
"valid<=85d"는 두 모델이 비교 가능한 범위의 한계이기도 하다.

`c3_camera/datasets/*`와 `recordings/*`는 전부 실내(공기 중) 촬영이므로 기존 C3
RGB-D 데이터의 depth 스케일은 계통 오차를 갖는다. 반대로 **실제 물속에서는 이
캘리브레이션이 맞다.**

## 아직 확인 못 한 것

- **EEPROM 메타데이터.** `getEepromData()`의 `batchTime` / `boardCustom` /
  `productName`을 읽으면 provenance를 못박을 수 있는데, 카메라가 지금 네트워크에
  없다(192.168.2.191 ping 100% loss). 돌아오면 읽어서 이 문서에 추가할 것.
- **순정 in-air OAK-D W 덤프.** 위 C2를 합성이 아닌 실물로 대체할 통제.
- **왜곡계수 8개만 저장 중.** `c3_camera/device.py:399`가 `dist[:8]`로 자른다.
  디바이스가 14개(rational + thin-prism/tilt)를 갖고 있다면 주변부 정확도가 손해다.
  재덤프 필요.
- **스테레오 extrinsics 부재.** `device.extrinsics`가 없어 rectified `P1`을 만들 수 없다.

## 벤더 측 진술 (독립 증거)

- Blue Robotics 제품 페이지: *"Each unit is individually calibrated for underwater
  operation to ensure accurate depth and scale in marine environments."*
- Blue Robotics 포럼, Tony White: *"the calibration Marine Situ performs takes place
  entirely underwater. It uses the standard checkerboard approach with openCV"*
