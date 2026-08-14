# KNOWN_ISSUES — 아직 안 고친 것들

> Claude Code 세션 중 발견했지만 **아직 코드로 고치지 않은** 버그·함정·보류 사항의
> 살아있는 목록. 규칙: **고치면 그 항목을 삭제**한다 (고친 기록은 git 히스토리가 담당).
> 새로 발견하면 날짜와 함께 추가한다. 항목 형식: 증상 → 임시 대응 → 제대로 고치는 법.

## 🐛 테스트 / 스크립트 함정

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

## 📌 알려진 한계 (당장 고칠 계획 없음, 잊지 말 것)

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
     `JOY` 줄의 `btn 커널>기체` 값이 아래와 맞는지 먼저 본다 — LB/RB `6>9`,`7>10` ·
     View `10>4` · Menu `11>6` · 십자키 `>11..14`. 그 다음에야 arm한다.
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
- **사실 2**: 정지 상태에서 |a| = **11.8 m/s²** (중력 9.81 대비 **+20%**). raw/calibrated
  둘 다 동일, accuracy 플래그는 UNRELIABLE/MEDIUM. 즉 **IMU intrinsic(스케일/바이어스)이
  미보정**이다. 축 방향(카메라 광축 대비)도 벤더 미문서화·미확정.
- **영향**: visual-inertial SLAM(VIO)에서 스케일과 중력 정렬이 틀어진다. 더 나쁜 건
  **"알고리즘이 안 맞는 것처럼" 실패**해서 원인을 IMU로 의심하기 어렵다는 점.
  RGB-D SLAM / stereo SLAM만 쓸 거면 무관하다.
- **임시 대응**: `c3_collect.py`가 두 사실을 모든 데이터셋의 `metadata.txt`에 명시하고,
  샘플마다 accuracy 필드를 남긴다. `c3_dataset_check.py`도 extrinsic 부재를 경고한다.
  → 동료가 모르고 쓰는 일은 없다.
- **제대로 고치는 법**: (a) Kalibr류로 camera-IMU extrinsic + IMU intrinsic 동시
  캘리브레이션(체커보드, 공기 중 → 수중 재검), 또는 (b) extrinsic만 기계적 실측 +
  accel 스케일/바이어스 6-position 캘리브레이션. 완료 후 `calibration.json`에
  주입하는 경로를 만들고 이 항목 삭제.

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

### Newton 그리퍼는 아직 Onshape에 없어 heavy_c3에서 제외 (2026-07-20)
- 사용자 요청: Onshape 어셈블리에 있는 것(차체 + C3)만 반영. 그리퍼는 CAD 추가 전까지
  `heavy_c3`에서 제외. `heavy_gripper` 변종은 그리퍼가 추가될 때를 위한 config로 유지되나,
  현재 그 GRIP_POS=[0.25,0,−0.17]는 여전히 **추정값**(Onshape 미검증)이다.
- 그리퍼가 Onshape에 추가되면: export 재실행 → 브래킷처럼 실측 위치로 GRIP_POS 갱신 →
  heavy_gripper 재생성.

### 방향 sweep이 seed-0 파랑 실현 하나를 공유 — worst-vertex 통계는 단일-실현 아티팩트
- 발견 2026-07-21 (`compare_20260720_230025` 코너 기하 분석): 모든 (current, wave) 헤딩쌍
  run이 **같은 seed-0 파랑 시계열**을 봄(실현 반복 주기 264.8 s ≈ run 길이 266.7 s) →
  특정 절대시각의 wave-group이 매 run 같은 lap/vertex를 때림(dobmpc 400 run 중 181개가
  t=200–210 s에 피크, worst vertex 66%가 V3). 방향 의존 결론은 **per-passage 상대각 통계**
  로만 뽑을 것; vertex별·시각별 주장은 multi-seed 재실행 전에는 출판 불가.
- 발견 2026-07-19 (C3 위치 정합 중): 스킨 bbox = 벤더 치수 × 1.0233 (세 축 균일).
- C3/페이로드 배치는 **실측 metric**(COM 앵커) 기준이라 동역학·카메라는 정확하지만,
  렌더에서 페이로드가 스킨 대비 ~3–5 mm 어긋나 보일 수 있음(코스메틱).

---
*마지막 갱신: 2026-07-21*
