# KNOWN_ISSUES — 아직 안 고친 것들

> Claude Code 세션 중 발견했지만 **아직 코드로 고치지 않은** 버그·함정·보류 사항의
> 살아있는 목록. 규칙: **고치면 그 항목을 삭제**한다 (고친 기록은 git 히스토리가 담당).
> 새로 발견하면 날짜와 함께 추가한다. 항목 형식: 증상 → 임시 대응 → 제대로 고치는 법.

## ⚠️ 운용 안전

### C3 수중 depth 오차는 **거리에 비례해 커진다** — 배율 상수로는 원리적으로 못 고친다 (2026-08-24)
- **증상**: `--depth-scale` 하나로는 한 거리에서만 맞는다. 실측 depth-vs-MAP 원시값이
  낮은 높이 **1.28**, ~0.9 m **1.56**으로 움직인다(조종사 확인).
- **샘플링 아티팩트가 아니라는 결정적 증거**(진단 함수가 관여하지 않는 증인):
  한 메시의 세 축이 **143 x 166 x 187 mm**인데 캘리퍼 실측은 119.73 mm — 축별 1.19 /
  1.39 / 1.56, **비등방 1.31**
  [측정: sessions/low_level_controller_data/20260823/0823_210304/mission_log.txt 21:02:29].
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
  [측정: sessions/low_level_controller_data/20260824/{0824_101807,0824_101251},
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
  (`sessions/low_level_controller_data/20260818/0818_143802/mpc_143802.csv` s=1.93 정지,
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

### `config/config.yaml`의 수면·풀 깊이가 **ROV 실기록과 모순**한다 (2026-08-23)
- **증상**: `water.surface_height_m: 0.8255` / `pool.depth_m: 1.143`은 2026-05-25 최초
  커밋 이후 한 번도 안 바뀌었는데(`git log -- config/config.yaml`), ROV 실기록이 둘 다
  넘긴다.
  1. **압력센서가 개입하지 않는 하한**: 채택된 fix의 `z_ned` 최저 **−1.296 m**
     (p50 −1.138, n=22094 — `sessions/nav_runs/*/fixes.csv`, 2026-08-13/14 집계).
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
  (`sessions/low_level_controller_data/20260823/0823_162548/mission_log.txt`),
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
  [측정: `sessions/low_level_controller_data/20260823/0823_174602/mpc_174638.csv`,
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
  같은 물체 여덟 번 재구성 [측정: `sessions/pose_meshes/*/model/model.obj`, oriented bbox]:

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
  [측정: `sessions/pose_meshes/*/{depth_enhanced,mask}`]:

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

## 📌 알려진 한계 (당장 고칠 계획 없음, 잊지 말 것)

### replay 미션(plan_stream): **실기 미검증** + jaw 중재·라이브 소스 함정 2건 (2026-08-30)
- **실기 미검증**: shape `replay`(기록 시연 재비행, `control/plan_stream.py` →
  `set_path_plan_ned`) 전체가 오프라인(9/9 + 19/19)과 demo_e2e(실제 acados,
  honest-완주 판정)까지만 검증됐다. safety-code-reviewer 감사(2026-08-30)의
  CRITICAL(딜맨이 래치된 버튼 비트 미해제)·HIGH(발산 가드 부재)는 **수정 완료**.
- **jaw 중재 부재(잔존)**: 파일럿 G/H와 replay가 같은 `cmd_gripper_drive`를
  last-writer-wins로 쓴다. 완화 3중(replay는 상태 변화 에지에서만 방출 + 기본
  `replay.gripper: false` + 딜맨이 이제 held drive를 놓음)으로 M0엔 충분하다는
  감사 판단이지만, 파일럿이 **누르고 있는** 중에 replay 에지가 덮으면 릴리스까지
  파일럿 의도가 밀린다. 제대로: sink에 jaw 전용 중재(파일럿 우선 + 워커 방출 무시
  창) — `replay.gripper: true`를 상시 쓰기 전에.
- **라이브 policy 소스 함정**: PlanFilter의 신선도 게이트는 `obs_t=None`이면
  조용히 skip한다(replay엔 옳다 — 기록 시연에 관측 시각이 없다). 나중에 라이브
  diffusion-policy 소스를 붙일 때 obs_t를 안 실으면 **0.7 s stale 게이트가 통째로
  꺼진 채** 돈다. 제대로: 라이브 소스에선 obs_t 부재 = reject로 뒤집을 것
  (plan_stream.py:201-209 부근).

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

  `[측정: sessions/low_level_controller_data/20260817/0817_101511/ +
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

### Newton 그리퍼는 아직 Onshape에 없어 heavy_c3에서 제외 (2026-07-20)
- 사용자 요청: Onshape 어셈블리에 있는 것(차체 + C3)만 반영. 그리퍼는 CAD 추가 전까지
  `heavy_c3`에서 제외. `heavy_gripper` 변종은 그리퍼가 추가될 때를 위한 config로 유지되나,
  현재 그 GRIP_POS=[0.25,0,−0.17]는 여전히 **추정값**(Onshape 미검증)이다.
- 그리퍼가 Onshape에 추가되면: export 재실행 → 브래킷처럼 실측 위치로 GRIP_POS 갱신 →
  heavy_gripper 재생성.

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

---
*마지막 갱신: 2026-08-19*
