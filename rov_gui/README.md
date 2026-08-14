# rov_gui — BlueROV2 topside 관제 대시보드 (PyQt)

한 화면, 스크롤 없음. 영상 3개 + 텔레메트리 + 추진기 + 페이로드 + 텔레옵을
고정 `QGridLayout` 하나에 얹은 관제 스테이션이다. 패널은 하드웨어를 직접 만지지
않는다 — **백엔드**가 데이터 버스에 올린 것만 그린다. 그래서 같은 UI가 합성
데이터, 실제 C3 + ArduSub, ROS 2 토픽 위에서 그대로 돈다.

```
./c3 gui                             # 이걸 쓴다. 래퍼가 인터프리터·cwd를 골라준다
./c3 gui --source hw                 # C3 + ArduSub (연결 전에 preflight)
./c3 gui --source demo               # 합성 데이터. 하드웨어를 열지 않는다
```

`./c3 gui`를 권한다. 이유가 두 개다.

**1. depthai 버전.** `--source hw`는 `c3_camera`를 import하고 그건 **depthai 2.x**를
요구하는데, 이 데스크톱의 기본 python은 conda base(depthai 3.5.0)라서 Pipeline을 만드는
것만으로 카메라를 채갈 수 있다. 래퍼는 인터프리터를 절대경로로 호출한다.

**2. GUI는 이제 자기 env에서 돈다** (2026-08-09부터). `./c3`의 카메라 도구들은 `robust`를
쓰지만 `./c3 gui`는 **`rovgui-pose`** 를 쓴다 — 스테이션이 SAM2 + FoundationPose 물체
추적기를 같은 프로세스에 얹기 때문이고, 그 스택은 `robust`에 못 들어간다:

| | rovgui-pose (GUI) | robust (시뮬·acados·SLAM) |
|---|---|---|
| torch / sam2 | 2.11+cu128 / 있음 | 없음 |
| numpy | **2.4.4** | 1.26.4 (gtsam 때문에 <2) |
| depthai · cv2 · Qt | 2.32 · 5.0 · 5.15.14 | 2.32 · 4.10 · 5.15.14 |

`rovgui-pose`는 **`oakd`의 복제본 + PyQt5 + depthai 2.32**다. 비싼 건 FoundationPose의
컴파일된 CUDA 확장(pytorch3d·kaolin·nvdiffrast·자체 mycpp/mycuda)이고 그게 `oakd`에 이미
빌드돼 있어서, `robust`에 torch를 얹는 것보다 이 방향이 압도적으로 싸다.
**numpy는 반드시 2.x** — torch 2.11이 numpy 2의 C ABI에 컴파일돼 있어서 1.26에서는
메타데이터와 무관하게 초기화에 실패한다.

검증(2026-08-09): 그 env에서 `rov_gui` 37/37, `c3_camera` 58/58 통과, Qt는 양쪽 5.15.14로
같고 최소 창 크기도 986×763으로 동일하다 — 한 화면 약속이 그대로다.

`rovgui-pose`가 없으면 래퍼가 **원래 인터프리터로 폴백하고 그렇게 말한다.** 스테이션은
정상 동작하고 `--pose`(물체 추적)만 못 쓴다. 만드는 법도 그때 같이 찍어준다:

```bash
conda create -n rovgui-pose --clone oakd
P=~/miniforge3/envs/rovgui-pose/bin
$P/pip install --force-reinstall --no-deps numpy==2.4.4   # 클론이 numpy를 깨뜨린다
$P/pip install PyQt5==5.15.11 'depthai~=2.32.0' pymavlink
```

> `conda create --clone`은 **numpy를 조용히 깨뜨린다** — pip 메타데이터는 2.4.4인데 실제
> 파일은 conda의 1.26.4가 된다. `pip list`로는 안 보이고 `import numpy`로만 보인다.
> 클론 직후 `python -c "import numpy, torch"`로 반드시 확인할 것.

어느 인터프리터로 갈지는 `./c3 env`가 두 줄로 보여준다(`interpreter` = 카메라 도구,
`gui` = 스테이션). `$ROVGUI_PY`로 덮어쓸 수 있다.

## 화면 구성

```
row 0   header    (4열 span: 소스, 상태 pill 4개, 시계, REC, E-STOP)
row 1   C3 MAIN (RGB)   | ROV CAM  | SYSTEM
row 2   (2열 × 2행)      | C3 DEPTH | HEALTH  (2행 span)
row 3   TELEOP | PAYLOAD | PROPULS. | SENSORS
        status bar (UI Hz, 그린 프레임 수, conflate 수, 녹화 상태)
```

가운데 패널은 `--panel2`로 고른다: `rov`(기본, ROV 자체 RGB 카메라) / `stereo`
(C3 좌측 mono) / `none`.

영상 패널을 **더블클릭**하면 그 피드가 큰 슬롯으로 올라온다.

### C3 Depth 패널 — 커서가 곧 측정기

depth 패널 위에 마우스를 올리면 **그 픽셀의 실제 거리**가 커서 옆에 뜬다(`0.45 m`).
색을 눈으로 환산하는 게 아니라, 화면의 그 프레임과 **함께 메일박스를 타고 온 원본
uint16 밀리미터 맵**을 읽는다 — 값과 그림이 다른 프레임에서 올 수 없는 구조다.
구멍(스테레오 무효, 0 mm)은 `no data`로 뜬다. 0.00 m로 보여주면 물체가 렌즈에 붙어
있다는 뜻이 되기 때문이다.

컬러바도 커졌다(패널 높이의 절반까지): 끝 라벨 0.3 m / 6 m에 **1·2·4 m 눈금**이
있고, 눈금 위치는 색칠과 같은 선형 공식(`imaging.depth_to_bgr`)을 쓴다 — 다른 공식을
쓰면 범례가 거짓말을 한다. 커서가 depth를 읽는 동안엔 그 값이 컬러바 위에 마커로도
표시된다.

### 한 화면에 들어가게 만드는 것은 레이아웃이 아니라 size policy다

1. **영상 위젯은 내용에서 크기를 유도하지 않는다.** `QLabel.setPixmap`을 쓰면
   sizeHint가 픽스맵 크기가 되고, 1920×1080 프레임 하나가 창을 화면 밖으로
   밀어낸다. 여기서는 `QSizePolicy.Ignored` + 직접 `paintEvent`.
2. **데이터 패널은 세로로 Preferred이고 body 끝에 stretch를 둔다.** 필요한 만큼만
   요구하고 남는 높이는 영상 행에 돌려준다.
3. **`QScrollArea`가 하나도 없다.** 안 들어가면 압축되거나 `…`로 잘린다. 조종사가
   볼 수 없는 곳에 컨트롤이 숨는 것보다 낫다.

3열로 시작했더니 최소 창 높이가 **1026 px**이 나왔다 — 1366×768 노트북에서 이미
약속이 깨진 상태였다. 센서 목록을 4번째 열로 분리해서 지금은 최소
**1152×749**(카메라 틸트 섹션 포함)이고,
`test_layout_fits_one_screen_and_has_no_scrollarea`가 그 선을 지킨다.

## 스레딩과 데이터 바인딩

GUI 스레드는 **위젯만** 만진다. 소켓·디바이스·인코더는 절대 만지지 않는다.
트래픽 모양이 셋이라서 메커니즘도 셋이다 (`bus.py`).

| 트래픽 | 메커니즘 | 이유 |
|---|---|---|
| 스냅샷 (텔레메트리·추진기·링크) | `DataBus`의 queued signal, 10 Hz | 작고 느리다. 큐 연결이 스레드 경계를 안전하게 넘겨준다 |
| 영상 프레임 | `FrameMailbox` (1칸 conflating 버퍼) | 프레임당 signal은 GUI가 밀리는 순간 **무한 큐**가 된다. 메모리보다 지연이 문제 — 조종사가 10초 전 그림으로 조종하게 된다 |
| 부재(침묵) | `Freshness` 워치독 | "나 죽었다"고 emit하는 소스는 없다. 데드락·케이블 절단·카메라 정지는 전부 침묵으로 보인다 |

### 워커는 두 종류뿐이다 (`backends/base.py`)

* **`TimerWorker`** — 자기 스레드의 이벤트 루프 위에서 `QTimer`로 tick.
  명령을 **받아야** 하는 워커는 반드시 이쪽 (queued slot이 tick 사이에 배달된다).
* **`LoopWorker`** — `while not stopping`으로 **블로킹**. DepthAI 큐 대기,
  `recv_match` 같은 것. 도착할 때 깨어나는 게 폴링보다 지연이 낮다. 대신 이벤트
  루프가 안 돌기 때문에 **slot을 가질 수 없다** — 중지는 `threading.Event`.

함정 두 개를 코드에 못 박아 뒀다:

* `QTimer`는 **생성된 스레드**에 속한다. `__init__`(GUI 스레드)에서 만들면
  "워커" 타이머가 GUI 스레드에서 돈다. 코드는 멀티스레드처럼 보이고 프로파일러는
  아니라고 한다. 그래서 타이머는 전부 `on_started()`(= moveToThread 이후)에서 만든다.
* `QThread::quit()`은 큐에 들어가는 이벤트가 아니라 **즉시** 이벤트 루프를 끝낸다.
  그래서 종료 요청을 queue에 넣고 quit을 부르면 타이머가 살아있는 채로 스레드가
  죽고, 나중에 메인 스레드가 워커를 GC할 때
  `QObject::killTimer: Timers cannot be stopped from another thread`가 뜬다.
  → `request_exit()`는 `_shutdown`만 post하고, **`quit()`은 워커 스레드 안에서**
  타이머를 멈춘 뒤에 부른다.

### 명령은 반대 방향으로 같은 버스를 탄다

위젯 → `bus.cmd_*` → (queued) → 워커 slot → 소켓. GUI 스레드는 송신하지 않는다.

**명령 하트비트**: 창이 20 Hz로 현재 조종 입력을 *바뀌지 않아도* 다시 emit한다.
싱크의 데드맨(500 ms)과 중복이 아니라, 데드맨을 **작동시키는** 장치다. GUI 스레드가
멈추면 새 stamp가 안 나오고 → 싱크가 중립을 보낸다. 변화가 있을 때만 emit하면
"키를 누르고 있는 상태"와 "GUI가 멈춘 상태"가 구분되지 않는다.

### 프레임 경로 (worker → 화면)

```
decode/generate (worker)
  → depth면 colourise (worker, cv2)
  → 패널 크기로 축소 (worker; 패널이 mailbox에 목표 크기를 써 둔다)
  → bgr_to_qimage(...).copy()          ← QImage는 numpy 버퍼를 wrap만 한다. 필수
  → mailbox.put()                      ← 이전 프레임은 버리고 conflated++ 
  ...
  → UI 타이머 30 Hz: mailbox.take() → paintEvent에서 blit
```

`QImage`는 GUI 스레드 밖에서 만들어도 안전하다(implicitly shared data).
`QPixmap`은 아니다(윈도우 시스템을 건드릴 수 있다) — 워커는 QImage만 넘긴다.

## "Disconnected"를 믿을 수 있게 만드는 것

* 모든 패널이 자기 데이터의 나이를 잰다. 소스별 timeout이 다르다 (15 fps 영상은
  0.3초면 늦은 것, 2 Hz BATTERY_STATUS는 3초 뒤에도 정상).
* **정지 화면은 파괴적으로 표시한다.** 죽은 영상은 검게 되지 않고 마지막 좋은
  프레임을 영원히 보여준다 — 구조물 앞에 정지 호버링한 그림과 구분되지 않는다.
  그래서 stale이면 어둡게 깔고, 사선 해칭을 치고, "STALE 3.2 s ago"를 찍는다.
* 모르는 값은 `0`이 아니라 `--`다 (`theme.fmt`). 0.0 V는 "모른다"가 아니다.
* 색은 상태 전용이다: green ONLINE / amber DEGRADED·STALE / grey OFFLINE /
  red FAULT. **grey와 red를 나눈 것은 의도적**이다 — "안 꽂혀 있다"와 "고장났다"는
  다른 반응을 요구하고, 둘 다 빨강으로 칠하면 빨강을 무시하게 된다.

## 백엔드

| 소스 | 무엇을 여는가 | 비고 |
|---|---|---|
| `demo` | 아무것도 안 연다 | **모든 숫자가 합성**이다. 헤더에 `SIMULATED DATA` 배너가 상시 뜬다. 예외: 테더 패널은 데모에서도 **실제** NIC 카운터를 읽는다 |
| `hw` | C3(DepthAI) + ArduSub(MAVLink) | `c3_camera.source.C3Source`, `c3_camera.mavlink_log.MavlinkLogger` 재사용. 연결 전에 preflight |
| `ros2` | rclpy 노드 하나 | 이 머신엔 rclpy가 없어서 **실행 검증 못 함**. 구조만 완성 — KNOWN_ISSUES 참고 |

테더 패널의 **from ROV / to ROV**는 토프사이드 NIC의 실제 트래픽이다(`rx`/`tx`를
조종사 말로 바꾼 것). *from ROV* = 올라오는 것(C3 영상 + depth + ROV 카메라 +
텔레메트리), *to ROV* = 내려가는 것(우리 명령, 보통 1~2 Mb/s).

그 외에 이 패널이 보는 것 (`net.py`, sysfs, 권한 불필요):

* `carrier` / `operstate` → 광 미디어 컨버터의 이더넷 쪽 링크
* `speed` → 재협상 감지 (1000 → 100은 커넥터가 나빠질 때의 전형)
* `rx_errors` / `rx_crc_errors` / `rx_dropped` 증가율 → **링크가 올라와 있는데
  더러운 경우**. 영상이 끊기기 *전에* 올라오는 유일한 조기 경보다
* BlueOS로의 TCP connect RTT → 테더 아래쪽 경로

## 지연 (실측, 2026-08-06)

C3 컬러 **sensor→host 28.2 ms @ 30 fps**(1920×1080 창, depth 켠 상태). 거기까지
오는 데 고친 것 네 가지:

| 설정 (depth는 640×360) | 컬러 지연 | 컬러 fps | 링크 |
|---|---|---|---|
| 15 fps 컬러 + 15 fps depth | 33.7 ms | 15.0 | 61 Mb/s |
| **30 fps 컬러 + 30 fps depth** | **128.0 ms** | 20.2 (33% 드롭) | 81 Mb/s — 포화 |
| 30 fps 컬러 + 10 fps depth | 34.4 ms | 30.2 | 48 Mb/s |
| 30 fps 컬러 + 10 fps depth (같은 설정, 복잡한 장면) | 49.4 ms | 29.9 | 65.7 Mb/s |
| 30 fps 컬러, `--no-depth` | 33.1 ms | 30.0 | 11 Mb/s |

* **컬러와 depth는 fps를 따로 준다** (`--fps 30`, `--depth-fps 20`). 둘 다 30을
  주면 그림이 빨라지는 게 아니라 프레임이 1/3로 줄고 지연이 4배가 된다 — PoE
  ~90 Mbit/s 한계에 물리기 때문.

### depth 640×360@20을 지키려면 컬러는 fps가 아니라 **픽셀과 품질**로 줄인다

위 표의 `30+10` 두 줄은 **같은 설정**을 다른 날 잰 것이고 18 Mb/s 차이가 난다.
노이즈가 아니다 — 컬러는 MJPEG이라 장면에 따라 프레임당 46~121 kB로 변한다.
**depth는 raw uint16이라 `w·h·2·fps`로 고정**이고 장면과 무관하다(640×360@10 =
예측 36.86, 실측 36.9 Mb/s — 모델이 정확히 맞는다).

**640×360@20은 그 혼자서 73.7 Mb/s, 링크의 82%다.** 컬러에 남는 건 ~16 Mb/s뿐이고,
960×540 q90은 20 fps에서도 18.7이라 들어가지 않는다. **fps를 깎는 건 이 문제에서 가장
비효율적인 레버다** — 30→15는 절반을 돌려주지만 필요한 건 그 이상이고, 대가는 조종사가
제일 먼저 느끼는 것(움직임)이다.

품질과 해상도는 훨씬 싸다. 아래는 **C3가 직접 찍은 프레임 120장을 다시 인코딩해
실측**한 값이다(`c3_camera/datasets/dataset_20260805_174544/rgb`, 960×540 MJPEG).
호스트 재인코딩이 미덥잖아 보이지만 q90에서 **114.1 kB/frame이 나왔고 디바이스
인코더가 같은 프레임에 남긴 값은 114.2 kB/frame**이다 — 0.1% 차이라 대체재로 쓸 수 있다.

| 컬러 | kB/frame | Mb/s @20 | Mb/s @30 | + depth 640×360@20 |
|---|---|---|---|---|
| 960×540 q90 | 114.1 | 18.7 | 28.0 | **103% @20** |
| 960×540 q75 | 64.6 | 10.6 | 15.9 | 94% @20 |
| 800×450 q90 | 80.2 | 13.1 | 19.7 | 97% @20 |
| **640×360 q80** | **38.0** | 6.2 | **9.3** | **92% @30** |
| 640×360 q85 | 44.8 | 7.3 | 11.0 | 90% @20 · 94% @30 |
| 480×270 q90 | 36.9 | 6.1 | 9.1 | 92% @30 |

크기 지수는 `bytes ~ px^0.82` — 픽셀을 반으로 줄여도 바이트는 절반까지 안 준다
(고주파가 먼저 포화한다). 그래서 **품질을 먼저 깎는 게 이득**이다: q90→q80은
바이트의 1/3을 돌려주는데 패널 크기에서는 화이트보드 글씨가 그대로 읽힌다.

지금 기본값은 이 조합이다:

```
--fps 30 --isp-scale 1/3 --mjpeg-quality 80 --depth-fps 20 --depth-size 640x360
   컬러 640×360 q80 @30 = 9.3      depth 640×360 @20 = 73.7      합 83.1 Mb/s (92%)
```

컬러 **30 fps를 지키면서** depth를 640×360@20으로 유지하는 조합이다. 더 여유를
원하면 `--mjpeg-quality 75`(91%)나 `--fps 20`(88%), 그림을 더 원하면
`--isp-scale 1/2 --mjpeg-quality 75 --fps 20`(94%, 여유 없음).

예산 추정은 이제 **`--mjpeg-quality`에 반응한다**(`MJPEG_QUALITY_SCALE`, 위 실측에서
유도). 전에는 무시해서 q80으로 낮춰도 로그가 q90 값을 말했다. 예산을 넘기는 조합을
주면 시작 로그가 **경고 수준으로** 말한다 — 넘어도 실패하지 않고 프레임을 버리며
지연이 3~4배가 되므로, 말해주지 않으면 그냥 느린 카메라로 보인다.
`test_default_video_config_fits_the_c3_link`가 기본값을 예산의 **95% 아래**로 지킨다
(장면이 바뀔 여지를 남기려고 100%가 아니라 95%다).

> 표의 kB/frame은 실측, **조합 총합은 [유도]**다 — 링크에서 직접 재보진 못했다.
> 2026-08-06 스윕 중 C3가 크래시하고 네트워크에서 사라져서 `30+10`(위 65.7 Mb/s 행)
> 하나만 남았다.
* **depth는 축소한 뒤에 컬러라이즈한다** (`imaging.scale_depth`). 순서를 바꾸면
  4배 비싸고, 그 비용은 같은 워커 스레드를 쓰는 **컬러 피드의 지연**으로 나온다
  (57.8 → 32.2 ms). nearest-neighbour인 것도 의도적 — depth를 보간하면 경계에서
  아무것도 없는 거리가 만들어진다.
* **거의 1:1인 리사이즈는 건너뛴다** (`NO_RESIZE_ABOVE = 0.85`). 960×540을
  933×525로 줄이는 건 전체 픽셀을 훑고 얻는 게 없다 (57.7 → 28.2 ms). 마지막 몇
  %는 어차피 blit하는 QPainter가 한다.
* `--ui-fps 60`(기본). 30 Hz면 프레임이 다음 페인트까지 최대 33 ms 기다린다.

**~90 Mbit/s는 C3 자신의 링크(카메라→스위치) 한계이지 테더가 아니다.** 광 테더는
1000 Mbit/s이고 ROV 기본 카메라(~34 Mbit/s)와 MAVLink도 같이 나른다. 위 예산 계산에
들어가는 건 **C3 스트림뿐**이다.

## SRG/SWY/HVE/YAW 바는 **우리가 보내는 명령**이지 기체의 움직임이 아니다

그래서 STABILIZE나 DEPTH HOLD에서 PROPULSION은 움직이는데 이 바들은 0에 있다. 틀린 게
아니다 — 그 모드에서는 **오토파일럿이 만든 추력이 이 바를 거치지 않는다.** 바는 이
스테이션이 합성·스케일·클램프해서 실제로 내보낸 값이고, 모터가 실제로 뭘 하는지는
PROPULSION이 보여준다.

바 자체를 기체 움직임으로 바꾸지 않은 이유: 모터 8개의 PWM에서 body-frame 축을
되돌리려면 **기체의 할당 행렬(AP_Motors6DOF)** 이 필요한데 우리는 그걸 모른다. 시뮬의
행렬을 갖다 쓰면 그럴듯하지만 틀린 숫자가 나온다 — 이 리포가 감사까지 해가며 막는
바로 그 종류다.

대신 **말로 한다.** armed + 명령 0 + 모터가 도는 중이면 데드맨 줄이 이렇게 뜬다:

```
TX 20 Hz   AUTOPILOT DRIVING (STABILIZE) — bars show YOUR command only
```

바에 마우스를 올리면 같은 설명이 툴팁으로 나온다.

## OUTPUT 기본값은 20%다

모든 축에 걸리는 배율이고, 연결 직후 조종사가 제일 먼저 하는 일은 "뭐가 움직이나"
스틱을 툭 건드려 보는 것이다. 좁은 수조나 다이버 옆에서 그 한 번이 최대 추력의 1/3이면
안 된다.

> 여기 실제 버그가 있었다: 슬라이더 위치·라벨·배율이 **세 군데에 따로** 박혀 있었고
> `valueChanged` 연결이 `setValue` **뒤에** 있어서 시작할 때 핸들러가 안 돌았다. 기본값을
> 20으로 바꿨더니 슬라이더는 20%, 라벨은 60%, 실제 배율은 0.60이 됐다 — 조종사가 읽는
> 값의 3배가 나가는 상태. 지금은 `DEFAULT_OUTPUT_PCT` 하나에서 핸들러를 통해 셋 다
> 세팅하고, 테스트가 셋의 일치를 지킨다.

## 물체 추적 + 6-DoF pose — `--pose`

C3 RGB 피드에서 물체를 클릭하면 SAM2가 그걸 계속 따라가고, 이어서 6-DoF 자세까지 나온다.
**기본은 꺼져 있다.** 켰을 때 세 갈래 중 어디로 가는지는 플래그가 정한다:

| | 클릭하면 | GPU 시간 | 작동 거리 |
|---|---|---|---|
| `--pose` | 참조뷰 수집 → 재구성 → 자세 | 수집 + ~1–2분 | 0.3~0.8 m |
| `--pose --pose-mesh M.obj` | 그 메시로 바로 등록 → 자세 | ~0.73 s | 0.3~0.8 m |
| `--pose --pose-no-build` | 마스크만 | 없음 | 제한 없음 |

```bash
./c3 gui --source hw --allow-command --pose
```

패널 오른쪽 위 `TRACK`을 켜고 물체를 **한 번 클릭**하면 된다. 더블클릭은 여전히 피드를 큰
슬롯으로 승격시킨다.

실측(2026-08-09, RTX 5090, `--pose-model tiny`): 로딩 **1.1초**, 클릭 후 `tracking` 진입,
**36 Hz**. 모델 크기는 `--pose-model`로 바꾼다(tiny 84.5 fps/777 MB … large 37.1 fps/1523 MB,
원본 프로젝트 실측).

### 설계상 중요한 것 넷

* **끄면 진짜로 없다.** `--pose` 없이는 torch도 SAM2도 import되지 않고, `C3VideoWorker`는
  프레임을 **복사조차 하지 않는다**(`_tap_pose`가 메일박스 None을 보고 즉시 리턴).
* **레이아웃 비용 0.** `TRACK` 버튼은 REC 옆에 절대배치된 패널의 자식이고, 마스크·상태 칩은
  `paintEvent`의 페인트 패스다 — `legend=`와 같은 방식. 세로 여유가 5 px뿐이라 새 행이
  불가능한데, 실제로 최소 창은 **1152×763으로 그대로다**(테스트가 켠 상태와 끈 상태를 둘 다
  검사한다).
* **마스크가 아니라 윤곽선이 건너온다.** 640×360 bool 마스크는 프레임당 230 kB이고 GUI
  스레드가 픽셀 단위로 그려야 한다. `findContours` + `approxPolyDP`로 ~1 kB 폴리라인으로
  만들어서 `QPainterPath`로 그린다 — 같은 그림, GUI 스레드에 픽셀 작업 0.
* **클릭은 400 ms 지연된다.** Qt는 더블클릭에도 press+release를 **먼저** 보내므로, 단순
  `mousePressEvent`는 승격 제스처마다 프롬프트를 오발한다. 시스템의 `doubleClickInterval`
  만큼 미루고 더블클릭이 오면 취소한다. 지연은 안 보인다 — 물체 선택 자체가 즉각적인
  동작이 아니다.

### 좌표 변환의 함정

패널 클릭 → **소스 픽셀**은 좌표계가 셋이고 가운데가 함정이다:

```
캔버스 px --(레터박스)--> 표시된 이미지 px --(배율)--> 소스 px
```

표시된 이미지는 소스가 아니다(워커가 패널 크기로 줄여서 넣는다). 그런데 `scale_to_fit`은
축소율이 0.85 이상이면 **원본을 그대로 돌려주므로**(`NO_RESIZE_ABOVE`), 그 배율이 창 크기에
따라 1.0과 다른 값 사이를 오간다. 둘 중 하나를 가정하면 어떤 창 크기에서 조용히 틀린다.
레터박스 바 위의 클릭은 `None`으로 거부한다.

### 6-DoF pose — `--pose-mesh`

메시를 주면 클릭 후 **약 0.73초**에 자세가 붙고 그 뒤로는 40~60 Hz로 추적한다.

```bash
./c3 gui --source hw --allow-command --pose \
         --pose-mesh "~/Desktop/New Folder/ref_views/model/model.obj"
```

실측(2026-08-09, 저장된 참조뷰 + 그 메시로 재생): 메시 `13444 verts 101×135×165 mm`,
등록 성공, `T[:3,3] = [0.058, 0.109, 0.413] m`(거리 **0.431 m**), 축 투영 정상, 종료 깨끗.

화면에는 RGB 축(원하면 `--pose-box3d`로 12-edge 박스)과 아래 한 줄이 뜬다:

```
x +0.084  y +0.020  z +0.581 m    d 0.59 m    42 Hz   reg 1
```

### 현장 재구성 — 메시가 없을 때 (기본)

`--pose-mesh` 없이 `--pose`만 주면 **클릭 한 번이 전부**다. 스테이션이 알아서 참조 뷰를
모으고, 메시를 재구성하고, 그 메시로 자세를 추정한다. 원본 도구의 `--pose` 동작과 같다.

```bash
./c3 gui --source hw --allow-command --pose     # 메시 없이도 pose까지 간다
```

### 화면이 항상 단계를 말한다

C3 RGB 패널 왼쪽 위에 **두 줄**이 뜬다. 위는 **지금 무슨 단계인지**, 아래는 **왜 안
넘어가는지**. 둘 다 필요하다 — 수집 중에는 카운터가 안 올라가는 게 정상인지 고장인지를
아래 줄만이 말해준다.

```
LOADING SAM2 — 1s
READY — CLICK AN OBJECT
COLLECTING VIEWS  7/20    ORBIT 31/75°    0.45 m
    too close in angle to a view already taken (3 deg)
COLLECTING VIEWS  3/20    ORBIT 62/75°    1.24 m
    object too far (124 cm) — bring it to 30-80 cm
RECONSTRUCTING MESH — 42s
    BundleSDF is running in a child process
LOADING FOUNDATIONPOSE — 8s
REGISTERING POSE...
POSE TRACKING  44 Hz
TRACKING  30 Hz — no pose yet          ← pose는 기대하는데 아직 없음
TRACKING  30 Hz — mask only            ← --pose-no-build. 원래 자세가 없는 모드
LOST — CLICK THE OBJECT AGAIN
STOPPED
    stopped after 74 deg with only 3 usable views (need 6) — turn the object
    more slowly, keep it 30-80 cm away, then press TRACK again
```

거리를 칩에 넣은 이유: **0.3~0.8 m를 벗어나면 스테레오 노이즈 때문에 전 프레임이
거절되는데**, 화면만 봐서는 얼마나 떨어져 있는지 알 수가 없다.

**색도 같은 말을 한다.** 파랑(BUSY) = 진행 중(수집/재구성/등록), 초록 = 자세가 붙어서
추적 중, 빨강 = 멈춤. 예전엔 수집이 포기하면 조용히 초록으로 돌아가서 **정상 추적과
구분되지 않았다** — 그게 "갑자기 초록으로 바뀌었다"의 정체였고, 이제 `STOPPED`(빨강)로
남아 이유를 말하고 `TRACK`을 다시 눌러야 풀린다.

같은 전이가 **LOG 패널에도** 한 줄씩 남는다. 오버레이는 이력이 없어서, 끝나고 나면
"재구성이 시작은 했었나"를 로그로만 답할 수 있다.

**메시지는 영어로 번역해서 띄운다.** 원본 프로젝트는 조종자에게 한국어로 말하는데
(`sam2_live/capture.py`), 그 문자열들이 **수집이 왜 안 되는지에 대한 유일한 설명**이라
번역을 경계(`perception/session.py::english`)에서 한다 — 원본은 읽기 전용이므로.

산출물은 `sessions/pose_meshes/obj_<시각>/`에 남는다(`--pose-ref-dir`). 다음 세션에는 그
`model/model.obj`를 `--pose-mesh`로 주면 2분을 건너뛴다.

**`TRACK`을 끄면 진짜로 중단된다** — 재구성 자식 프로세스를 `terminate()` 한다. 기체를
조종하는 프로세스에서 2분짜리 GPU 작업을 못 멈추면 안 되기 때문에, 원본의
`subprocess.call` 대신 `Popen`으로 다시 썼다(그 외는 전부 원본과 동일).

실측(2026-08-09, 저장된 참조뷰 17장을 재생, 입력 `~/Desktop/New Folder/ref_views`):
수집이 **8장 / 회전 75.3° / ICP RMSE 6.2 mm**에서 자동 종료 → BundleSDF **25초** →
메시 **7993 verts, 51×94×167 mm**. 같은 물체의 원본 메시가 `13444 verts 101×135×165 mm`
이므로 **가장 긴 축은 2 mm 안에서 일치**하고 나머지 두 축이 작다 — 17장 중 8장만 쓴 부분
모델이라 그렇다. **라이브 수집 실측이 아니라 저장 데이터 재생이다.**

수집 규칙 두 개는 실측 근거가 있고 둘 다 상한이다:

* **누적 회전 75°** (`--pose-max-arc`). 뷰별 카메라 자세를 colored ICP로 만드는데, 오차가
  80°를 넘으면 완만히 나빠지는 게 아니라 절벽처럼 무너진다 — 원본 합성 실험 기준
  60° 8.7 mm, 80° 9.7 mm, **100° 87 mm, 120° 283 mm**. 올려도 커버리지가 아니라 틀린
  메시를 산다.
* **최소 6장** (`--pose-max-views`는 상한 20). 6장 미만이면 재구성을 **거부하고** 다시
  하라고 말한다. 10장 이상 권장, 참조 구현은 16장.

그 대가로 메시는 **찍은 각도 범위만 덮는 부분 모델**이다. 반대편에서 보면 자세 추정이
실패하고, 그건 재등록 감시가 "메시가 이 물체가 아니다"로 잡는다.

마스크만 필요하면 `--pose-no-build`로 재구성 경로 전체를 끈다 — 작동 거리 제약도, GPU
2분도 없다. 그때는 모든 자세 칸이 비는데, 추적기가 고장난 게 아니라 시킨 일을 하는 것이다.

#### 재등록 감시 — 유일한 되돌림 장치

FoundationPose의 `track_one()`은 **마스크를 안 쓴다.** 그래서 자세가 한 번 물체에서
미끄러지면 스스로 돌아올 방법이 없다. 매 프레임 메시 실루엣과 SAM2 마스크의 IoU를
비교하고, 어긋나면 다시 등록한다(`POSE_MIN_IOU 0.25`, 8프레임 연속, 최소 간격 1.5초 —
`register()` 자체가 0.73초라서). 4번 재시도해도 안 맞으면 매달리지 않고 말한다:
**"pose does not fit — wrong object, or the mesh is not this one"**.

#### 작동 거리 0.3~0.8 m

FoundationPose는 depth를 먹는다. 원본 README 기준 **2.4 m에서 스테레오 노이즈가 50 mm를
넘어 등록 자체가 실패**한다. ROV로는 매우 가까운 거리다. 마스크만 쓰는 모드(메시 없이)는
이 제약을 안 받으므로, 멀리 있는 물체를 그냥 표시만 하고 싶으면 그쪽이 맞다.

C3의 HFOV가 **63.9°**라는 것도 같이 봐야 한다 — 0.5 m에서 시야 폭이 **62 cm**라
기체가 30 cm만 움직여도 물체가 화면을 벗어난다.

### 3-D pose 맵 — 떠 있는 창

FoundationPose가 올라오는 순간(로딩/등록/추적) **별도 창이 저절로 열리고**, 닫았으면 C3
RGB 패널의 `MAP` 버튼으로 다시 연다. 메인 창은 세로 여유가 5 px뿐이라 새 패널이
불가능해서 떠 있는 창이다.

* **카메라 좌표계 그대로**: 카메라가 원점, +Z 전방, +Y 아래(OpenCV) — `T_cam_obj`와 POSE
  로그가 쓰는 바로 그 프레임이라 플롯과 로그가 다른 말을 할 수 없다.
* **볼륨 고정**: x,y ±1 m, z 0~2 m (pose가 ~2 m 밖에서는 어차피 존재하지 않는다).
  자동 스케일이면 등거리도 볼 때마다 다르게 보인다. **뷰는 마우스가 바꾼다** — 드래그
  궤도, 휠 줌, 더블클릭 리셋.
* 궤적(최근 3000점) + 현재 자세의 축 삼각대(10 cm), 0.5 m 간격 깊이 눈금.
* 새 pose가 올 때만(10 Hz) 다시 그린다. OpenGL 없음, QPainter뿐.

### 녹화 영상에 오버레이가 들어간다

`TRACK`이 켜져 있으면 **C3 RGB 피드 녹화(`REC`)에 마스크·3-D 축·단계 줄이 그대로 구워진다.**
추적 세션을 녹화했는데 추적이 안 보이면 그건 바닥을 녹화한 것이다.

```
POSE TRACKING  44 Hz
  (마스크 윤곽 + RGB 축)
x +0.058  y +0.109  z +0.413 m    d 0.43 m    44 Hz   reg 1
```

* 화면에 그리는 코드와 **같은 함수**(`_paint_pose`)다 — 영상이 조종자가 못 본 것을 보여줄
  수 없게. `fit` 인자만 다르다(녹화 프레임은 레터박스가 없어 이미지가 전부).
* **복사본에 그린다.** 캔버스가 같은 QImage를 들고 그 위에 라이브 오버레이를 또 그리므로,
  원본에 그리면 모든 선이 두 번 그려진다.
* **녹화 중이 아니면 복사조차 안 한다**, `TRACK`이 꺼져 있으면 프레임이 그대로 지나간다.

전체 UI 녹화(상단 `REC`)는 원래부터 화면 그대로라 해당 없다.

### 로깅 — C3 Depth 녹화에 얹힌다

`_rov.jsonl` / `_c3_imu.jsonl`과 같은 트리거다. 영상과 같은 stem으로:

```
c3_depth_….mp4
c3_depth_…_pose.jsonl   CAMERA(1회) · MESH(1회) · PROMPT(클릭) · EVENT(전이) · TRACK · POSE
c3_depth_…_pose.json    매니페스트 (by_message, hz_measured)
c3_depth_…_pose.csv     자세 1개당 1행 — t, frame_seq, x/y/z_m, distance_m,
                        r00..r22(회전 행), pose_hz, n_register, state
```

CSV는 **두 번째 형식이지 두 번째 소스가 아니다** — jsonl POSE 행과 같은 `T_cam_obj`에서
나온 같은 숫자, 같은 캡처 시각이다. pandas/플롯 스크립트가 바로 먹을 수 있게 평평하게
펴놓은 것뿐이다.

`TRACK` 행에는 **윤곽선 자체**가 들어간다 — SAM2를 다시 돌리지 않고 오프라인 재분석이
가능하다. 타임스탬프는 **프레임의 캡처 시각**이라 영상과 정렬된다.

`CAMERA` 행에는 내부파라미터와 **그 출처**가 같이 들어간다:

```json
{"msg":"CAMERA","fx":513.4,…,"medium":"water","rectified":false,
 "note":"UNDERWATER factory calibration (vendor-confirmed; measured HFOV
         matches the Snell-underwater prediction, not the in-air spec).
         Distances ARE metric in water. IN-AIR captures read ~1.33x LONG."}
```

`MESH` 행은 **자세가 어느 3-D 모델에 대한 진술인지**를 박는다 — 경로, `sha1`, 현장
재구성이었는지, 참조뷰 디렉터리, 그리고 재구성 직후 검산 줄들(`sc_factor`가 함의하는 장면
크기, 메시 bbox, `bad_mask#` 범위). 현장 재구성은 매번 다른 모델이라 경로만으로는 식별이
안 되고, `model.obj`라는 이름은 항상 같다.

`POSE` 행에는 `T_cam_obj` 16개(row-major)와 `pos_m`, `distance_m`가 들어가고, 매니페스트에
**규약을 글로** 박는다 — 암시하면 6개월 뒤 틀리게 읽힌다:

```
T_cam_obj maps OBJECT -> CAMERA. OpenCV optical axes: +X right, +Y DOWN,
+Z forward. Metres, row-major. NO camera->body transform is applied.
```

> **캘리브 방향에 주의 — 공장 캘리브는 수중 값이다** (벤더 확인 + 화각 실측:
> HFOV 63.7°가 in-air 스펙 95°가 아니라 Snell 수중 예측 67.2°와 일치,
> `calib/fov_audit.py`, KNOWN_ISSUES 2026-08-04). 그래서 결이 직관과 반대다:
> **물속 거리가 metric이고, 지상 책상 테스트가 ~1.33배 길게 읽힌다.** 실물 0.5 m가
> `d ≈ 0.67 m`로 뜨면 정상이다. 마스크만 쓸 땐 무관하다(픽셀 영역일 뿐).
> (2026-08-10 정정: 이 문서와 CAMERA 레코드가 처음엔 정확히 반대로 적혀 있었고,
> 그 출처는 감사가 이미 틀렸다고 표시한 `c3_camera/geometry.py`의 낡은 주석이었다.)

### 정류(rectification)를 안 하는 이유

원본 프로젝트는 raw(1280×800)와 정류(640×400) 두 좌표계를 오간다. OAK-D W의 96° 렌즈가
코너에서 **232 px** 틀어지기 때문이다. C3는 그런 렌즈가 아니다 — 실측:

```
HFOV 63.9°   핀홀 vs 왜곡: 중앙 0.13 px … 코너 최대 2.08 px = 0.23° = 0.6 m에서 2.4 mm
```

스테레오 depth 노이즈보다 한 자릿수 작다. 그래서 **RGB·depth·마스크·클릭이 전부 640×360
하나의 좌표계**에 산다. 버그 한 종류가 통째로 없어진다.

### 원본은 건드리지 않는다

`~/Desktop/New Folder`는 **읽기 전용**이다. `rov_gui/perception/session.py`가 그
라이브러리(`sam2_live.tracker` 등)를 import해서 오케스트레이션만 다시 쓴다 — 카메라와
OpenCV 창 없이. 그쪽 `camera.py`(depthai 3.x)는 **import되지 않는다**: 스테이션이 이미
카메라를 쥐고 있고, DepthAI는 장치당 파이프라인을 하나만 허용한다.
위치는 `--pose-src` 또는 `$SAM2_LIVE_ROOT`로 바꾼다.

## 폐루프 MPC — `--mpc` (AprilTag 측위 + acados NMPC)

```bash
./c3 gui --source hw --allow-command --mpc                  # 실기 (wall 기하)
./c3 gui --source hw --allow-command --mpc --nav-geometry floor
./c3 gui --source demo --mpc                                 # 합성 plant 폐루프
~/miniforge3/envs/rovgui-pose/bin/python -m rov_gui.control.smoke   # P0 벤치 게이트
```

sim의 DOB-MPC(`bluerov2_mujoco_marinegym/dobmpc/`, MuJoCo 무의존이라 **그대로
import**)를 실기에 붙인 폐루프다. 파이프라인 전체가 이 프로세스 안에 있다 —
C3도 명령 sink도 단일 소유이기 때문이다:

```
C3 컬러 프레임 ─► TagNavWorker (pupil_apriltags + solvePnP, 고정 맵, GTSAM 없음)
                     └► bus.nav_fix  (기체 pose, NED 태그월드)
ArduSub MAVLink ─► VehicleWorker (--mpc면 20 Hz 틱) ─► bus.vehicle_imu
                     └► StateAssembler: eta/nu/nudot (NED/FRD, 20 Hz)
                          └► EAOB + acados NMPC (mpc_bridge, DT_CTRL=50 ms)
                               └► wrench → MANUAL_CONTROL 4축 (K/M은 버림)
                                    └► 기존 MavlinkCommandSink — 게이트 전부 상속
```

설정은 `config/hw_nav.yaml`(태그/기하/extrinsic/지오펜스)과
`config/hw_mpc.yaml`(모드/축게인/EAOB 시그마/square)이다. 실측 전 수치는 전부
`[예측]` 태그가 붙어 있다 — 특히 **축 게인은 P4 스텝 캘리브레이션 전까지 추정치**다.

운용 규율 (MPC 패널은 **메인 그리드 안** — 2026-08-14부터 **맨 아랫줄의 2·3열
전체**(옛 PROPULSION·SENSORS 자리, ~790×376 px)를 차지하고, PROPULSION과
SENSORS는 3열 SYSTEM HEALTH 아래에 **나란히** 들어간다(세로로 쌓으면 최소
높이가 1080p 예산을 넘는다 — `test_mpc_panel_lives_...`가 고정). 플롯은 NED
탑다운에 m 단위 축·눈금이 붙고, **풀 경계(`hw_nav.yaml: pool_ned`)가 실선
테두리이자 축 스케일의 기준**이다. 이 배치 때문에 `--mpc`의 최소 창 높이는
768을 넘는다 — 기본 레이아웃의 one-screen(1366×768) 약속은 그대로이고,
`--mpc`는 1080p 운용 화면을 전제한다):

* **START(hold)가 원버튼 플로우**다: engage → 워밍업 → square 자동 시작, CSV
  기록은 engage 순간부터. 전제조건은 그대로 전부 검사된다: COMMAND ENABLE·ARMED·
  비행모드 MANUAL·acados 실시간(프로브 <25 ms)·태그 fix 신선(<0.5 s)·IMU 신선·
  지오펜스 안(+배치된 square 코너 4점까지). 하나라도 빠지면 사유를 말하고 거부.
* **HOLD(hold)는 DP 전용 engage**다(캘리브레이션/정점유지용). STOP TRAJ·DISENG·
  E-STOP은 전부 단일 클릭 — 멈춤에는 게이트가 없다.
* **AprilTag 감지는 피드별 TAG 버튼**으로 켜고 끈다. **어느 피드가 측위를
  담당하는지는 `hw_nav.yaml: nav_source`가 정한다**: `main` = C3(공장 수중
  캘리브 내장), `second` = ROV 기본 RGB(`second_cam` 블록 — **전부 [예측]**,
  fx가 틀리면 모든 거리가 그 비율로 스케일된다). 측위 피드의 태그는 초록 실선,
  비측위 피드는 희미한 외곽선. 측위 피드의 TAG를 끄면 stale-fix로 안전하게
  disengage되고, **다시 켜는 것이 datum 재영점**(아래) 제스처다.
* **(0,0)·yaw 0 = START(engage)를 누른 순간의 pose다** (미션 datum, 항상 켜짐).
  engage마다 그 자리에서 재영점되고, square·지오펜스·플롯·CSV가 전부 이 상대
  좌표계에 산다 — EAOB/NMPC도 이 좌표계에서 태어나므로 도중에 점프하지 않는다.
  절대(태그 프레임) 복원용 datum은 meta.json `hardware.datum_tag_frame`에 남는다.
  플롯 좌하단 `fix N Hz · detect M ms`가 측위 갱신률이다 — 느리면
  `hw_nav.yaml: quad_decimate`(기본 2.0)를 올릴수록 빨라지고 코너 정밀도는
  소폭 준다. 현재 미션 설정(2026-08-13): **갠트리 tagslam의 47태그 바닥 맵**
  (`config/tag_map.yaml`, 앵커 = **tag 25** = 맵 원점)을 **C3 RGB**로 multi-tag
  joint PnP(`geometry: floor`, `nav_source: main`, `min_tags: 2`) — C3는 공장
  수중 캘리브 내장이라 스케일이 정확하고, 단일 태그의 flip 모호성·중력
  게이트가 아예 없는 경로다. 어떤 피드가 측위 중인지는 플롯 좌하단
  `loc C3 RGB · fix N Hz · detect M ms`와 SENSORS의 `TagNav [C3] …` 줄이
  실시간으로 말해준다. quad_decimate는 피드별(main 1.0 / second 2.0 —
  640×360 C3는 디시메이션하면 먼 태그를 놓친다). **맵 태그 47개는 옛 tagslam
  시각화처럼 실제 크기·방향의 사각형**으로 깔리고, **지금 fix를 만든 태그만
  초록으로 칠해진다**(fix가 1 s 이상 끊기면 소등 — 죽은 측위가 태그를 켜둘 수
  없다). 비엔게이지 상태에선 fix가 마커를 직접 구동한다(ArduSub 텔레메트리
  없는 벤치에서도 움직임). 벽 단일 태그(104) 모드는 `geometry: wall`로 복귀.
  20 cm square @ 0.05 m/s, yaw는 시작 헤딩 고정.
* **actual 궤적은 나이(시간) 색**이다: viridis 램프(어두운 보라 = 오래됨 →
  노랑 = 지금), 우하단에 `-90s → now` 범례. 90 s(`trajectory.py:
  TRAIL_AGE_S`)가 지난 점은 서서히 투명해지며 지워진다 — 교차하는 경로에서도
  시간 순서가 읽히고, 옛 궤적이 화면을 덮지 않는다. reference 궤적은 파란
  단색 그대로.
* **REC NAV(플롯 패널 버튼) = 원시 측위 기록**. engage와 무관하게(수동 조종
  서베이 포함) `sessions/nav_runs/<stamp>/`에 네 파일을 남긴다: `map.json`
  (잠긴 태그맵 id→x,y,z,yaw + 태그 크기 + pool + 지오펜스), `fixes.csv`
  (**태그 프레임 원시 fix** — datum 변환 전, 실패 프레임도 **태그 id와** 사유를
  함께 기록), 그리고 `detections.csv` + `frames.csv`(**원시 관측** — 검출된
  모든 태그의 코너 4점과 그 프레임의 카메라 모델). 뒤의 둘이 맵을 다시
  만들거나 넓히는 재료다 — 풀린 fix만 남기면 코너가 버려져서 아무것도 못 한다.
  `python -m rov_gui.tools.plot_nav_run sessions/nav_runs/<stamp>`로 같은
  그림(태그 사각형 + 시간색 경로 + 풀 경계, matplotlib)을 다시 그린다.
* **중복 태그 id**(2026-08-14 실측): 풀 매트는 20×7=140칸 커스텀 배치이고
  **12개 id가 두 곳에 있다**(`hw_nav.yaml: duplicate_ids`). 맵은 id당 자세
  하나뿐이라 "다른 사본"의 코너가 섞이면 joint PnP가 터진다(실측 59~131 px).
  대응은 **confirm-or-drop**: 유일 태그로 자세를 먼저 확정하고, 중복 태그는
  맵 자리에서 `dup_confirm_px`(6 px) 안에 떨어질 때만 채택한다. 다른 사본의
  **위치를 몰라도 되는 게 핵심**(알아보기만 하면 됨)이라 재측량이 필요 없다.
  한 프레임에 같은 id가 두 번 잡히면 둘 다 폐기 — 신고 안 된 중복이 스스로
  드러나고, `min_tags`가 유일 id만 세게 된다.
* **플롯 위 마우스**: 끌면 이동(pan), 휠은 확대/축소, 더블클릭하면 원래 시야로.
  화면에 글자로 굽지 않고 위젯 툴팁에 있다(2026-08-14 요청). **지오펜스(주황
  점선)는 engage 중에만** 그린다 — 펜스는 datum 상대("START 누른 자리 ±1.2 m")라
  engage 전에는 datum이 없어서, 그리면 실제로는 절대 있지 않을 자리에 상자가
  놓인다.
* **풀 테두리는 맵에서 파생된다**: `hw_nav.yaml: pool_margin_m`(기본 0.10) =
  **가장 바깥 태그의 인쇄 가장자리에서 사방 이만큼**. 맵이 중심 좌표를 들고
  있으므로 반 태그(0.085 m)를 먼저 더한다. 맵을 다시 만들면 벽도 따라 움직인다
  (손으로 적은 사각형이 낡아 남는 일이 없다). 명시 지정이 필요하면
  `pool_ned: {x: [..], y: [..]}`가 우선한다. 현재 맵 기준 **2.025 × 4.437 m**.
* **어느 사본을 쓰는지 화면이 말한다**: 중복 id는 바닥에 사각형이 두 개인데,
  fix에 실제로 기여한 **그 사본만** 초록으로 칠해진다(`NavFix.tag_insts`).
  실측 검증: 두 사본이 한 프레임에서 동시에 채택된 경우 **0건**.
* **맵 넓히기**: `python -m rov_gui.tools.build_tag_map
  sessions/nav_runs/<stamp> --anchor config/tag_map.yaml -o
  config/tag_map_full.yaml`. 측량된 47개를 **얼어붙은 앵커**로 두고, 프레임마다
  아는 태그로 카메라 자세를 잡은 뒤 같은 프레임의 모르는 태그를 푼다. 새 태그는
  라운드 사이에만 승격되므로(관측 `--min-obs`개가 `--max-spread` 안에서 일치)
  **경로를 따라 오차가 누적되지 않는다** — 그래프 SLAM이 일그러지는 그 구조가
  아예 없다. 중복 id는 앵커에서 빼고 재발견해서 **인스턴스 두 개**로 기록한다.
  촬영 규칙 하나: **모르는 태그가 화면에 들어올 때 아는 태그도 같이 보여야
  한다.** 승격 못 한 태그는 관측 수와 함께 리포트에 뜬다.
* **플롯의 로봇 마커는 20 Hz**로 움직인다: 카메라 fix 사이를 상태조립기가
  속도추정+자이로로 브리징하고(`hw_mpc.yaml: vel_propagation`, 기본 on —
  가속도 이중적분은 BNO086 스케일/바이어스 때문에 의도적으로 안 씀), 플롯은
  fix가 아니라 MpcStatus 스트림을 그린다. fix Hz는 별도로 표시되므로 "제어가
  보는 상태"와 "측위 원천 주기"를 혼동할 일이 없다.
* **PID 모드**: 모드 콤보 dobmpc | mpc | pid. PID 게인은 sim의 pole-placed
  `GAINS_HEAVY_GRIPPER`(controller.py) × `hw_mpc.yaml: pid.omega_derate`
  (기본 0.6, kp·d²/kd·d/ki·d³) [예측]. acados 빌드가 실패해도 PID는 항상
  가용하다. 동일 CSV 스키마로 기록되므로 sim처럼 PID vs MPC vs DOB-MPC A/B가
  바로 된다(모드는 meta.json `controller.type`).
* **조종 입력은 언제나 이긴다.** 스틱/키가 움직이면 그 프레임은 기체로 가지 않고
  MPC 해제 요청이 된다(창의 `_pilot_gate` — pump·edge 두 경로가 같은 게이트를
  지난다). E-STOP(헤더·MPC 패널·Esc 어디서든)·DISARM·ENABLE off·태그 상실
  >0.5 s·지오펜스 이탈·솔버 연속 3실패는 전부 자동 disengage. MpcWorker가 죽어도
  sink deadman(500 ms)이 중립을 보내고, 창의 mpc 워치독(1.5 s)이 스스로 조종권을
  회수한다 — 조종사가 죽었을 때와 같은 마지막 방어선.
* wall 기하에서는 `heading_follow: false`(크랩 square)가 **필수**다. 코너에서
  기수를 돌리면 벽 태그가 시야에서 사라진다.

기록: engage하면 CSV가 무조건 열린다(`sessions/mpc_runs/…_mpc.csv`, 또는 진행 중인
녹화 스템에 `_mpc.csv`로 편승). 앞 9열은 sim `runs/traj_*.csv`와 동일 스키마
(`t,px,py,pz,rx,ry,yaw_deg,pitch_deg,lap`, world FLU)라 기존 분석 도구가 그대로
읽고, 뒤로 `w_hat×6·u×6·솔버 status/solve_ms·태그 수·reproj px·축 명령` 등이
붙는다. `.meta.json` 사이드카가 태그맵 sha1, extrinsic, 게인 출처까지 기록한다.

검증 상태 (2026-08-12): `rov_gui/control/smoke.py` — acados가 rovgui-pose(numpy 2)
에서 SQP-RTI 빌드 1.3 s, EAOB+solve p50 3.1 ms/p99 6.4 ms
(`sessions/` 아님, 콘솔 출력); demo 폐루프로 square 1랩 완주(솔버 실패 0).
**실기·수중은 아직 미검증** — KNOWN_ISSUES.md 참조. 수조 절차는 P3(측위만 기록)
→ P4(DP + 축게인 스텝 캘리브레이션) → P5(square, mpc) → P6(dobmpc A/B) 순서를
지킬 것.

## 배터리 잔량 — 셋 중 어느 걸 보고 있는지 화면이 말한다

ArduSub은 `BATT_CAPACITY`와 전류 모니터가 설정돼 있지 않으면 `battery_remaining`을
**−1(모름)** 로 보낸다. 그게 데모에선 95%가 뜨는데 실기에선 `--`만 뜨던 이유다.

이제 출처 우선순위대로 채운다:

| 출처 | 표시 | 신뢰도 |
|---|---|---|
| 기체 `BATTERY_STATUS.battery_remaining` | `battery 58%` | 오토파일럿 자신의 값 |
| 쿨롱 카운팅 `(capacity − consumed)/capacity` | `battery 58%  est mAh` | **[유도]** 실측 전류의 적분 |
| 전압 곡선 | `battery 58%  est V` | **[유도]** 부하에서 낮게 읽힘 |

`--battery-capacity-mah 18000`(스톡 Li-ion)을 주면 쿨롱 카운팅이 켜지고 타일이
`USED`에서 **`LEFT`(남은 mAh)** 로 바뀐다 — 다이빙 중에 암산할 게 아니라 남은 양이
바로 보여야 한다.

전압 추정의 한계 두 가지를 알고 써야 한다: **부하에서 팩이 처진다**(스러스터 8개가
당기면 1 V 가까이 떨어져서, 가장 열심히 일할 때 가장 낮게 읽힌다), 그리고 **곡선이
3.9–3.6 V에서 거의 평평하다**(용량 대부분이 거기 있어서 0.05 V 오차가 10%쯤 된다).

셀 수는 **추정하지 말고 `--battery-cells`로 박아라**(기본 4 = 스톡 4S). 전압 하나로는
**빈 4S(12.0 V)와 꽉 찬 3S를 구분할 수 없고**, 그 방향으로 틀리면 방전된 배터리가
82%로 보인다. 그래서 지정된 셀 수가 그럴듯하면 그게 이기고, 추정은 그 범위를 벗어날
때만 돌아간다(회귀 테스트가 12.0 V → 0%를 지킨다).

**제대로 된 해법은 기체 쪽이다**: `BATT_CAPACITY`와 전류 모니터를 설정하면 ArduSub이
직접 `battery_remaining`을 보내고, 그러면 이 패널은 추정을 버리고 그 값을 쓴다.

## IMU가 둘이다 — SENSORS에도 두 줄이다

`IMU (ROV)`와 `IMU (C3)`는 **다른 장치, 다른 시계, 다른 레이트**다. 한 줄로 합쳐두면
어느 쪽이 조용해졌는지 알 수 없고, 더 나쁘게는 downstream이 어느 쪽을 융합하고
있는지 알 수 없다.

| | `IMU (ROV)` | `IMU (C3)` |
|---|---|---|
| 장치 | 오토파일럿(Navigator) | C3 온보드 BNO086 |
| 경로 | MAVLink `RAW_IMU` / `SCALED_IMU2` | DepthAI IMU 큐 |
| 시계 | 기체 `time_boot_ms` | **이미지와 같은 시계** (`dai.Clock`) |
| 기본 요청 | 200 Hz (`--mavlink-rate`) | 500 Hz (`--c3-imu-rate`) |
| 실측 천장 | **~208 Hz** | **~486 Hz** |

* **ROV IMU**(실측 2026-08-06, `SET_MESSAGE_INTERVAL`): 기본 2-3 Hz · 요청 50 → 62 ·
  100 → 125 · 200 → **208** · 400/1000 → 여전히 208. 비용은 약 0.2 Mbit/s.
  레이트 올리기는 **기본으로 켜져 있다**(`--no-mavlink-set-rates`로 끔) — ArduSub
  기본값 2-3 Hz는 대시보드 불빛 말고는 쓸 데가 없고, depth 녹화 옆에 남기는 센서
  로그는 제대로 된 레이트여야 남길 값어치가 있다. 켜면 기체로 **송신**한다.
* **C3 IMU**: accel+gyro만 쓰면 ~486 Hz(`c3_camera/config.py` 실측). 융합
  `ROTATION_VECTOR`나 지자기를 켜면 **모든 IMU 스트림이 ~40 Hz로 붕괴**해서, 이
  스테이션은 accel+gyro만 스트리밍한다. 배치 임계값은 10 — 이 레이트에서 1이면
  샘플이 샌다(`c3_collect.py` 실측). `--no-c3-imu`로 끌 수 있다.
* C3가 살아 있고 ROV가 빠져 있어도 `IMU (C3)` 줄은 갱신된다. 잠수 전에 카메라 IMU를
  확인하려면 기체가 대답할 때까지 기다려야 하는 패널은 쓸모가 없다.
* 레이트 칸에는 **레이트**를 쓴다. 예전엔 detail 문자열이 이겨서 `486 Hz`가 와야 할
  자리에 `BNO086`이 떠 있었다. 예외는 detail이 이미 레이트를 담고 있을 때뿐인데
  (REST 트랜스포트의 `200 Hz sampled`), 그건 기체 송신 레이트가 아니라 **우리 폴링
  레이트**라는 단서를 떼면 안 되기 때문이다.

### ROV 카메라는 GStreamer로 받는다 (QGC와 같은 파이프라인)

FFmpeg 경로를 **모든 지표로 재봤는데 전부 깨끗했다** — 커널 수신 큐 0, 드롭 0,
20초 드리프트 +1 ms, 프레임 간격 최대 48 ms, B-프레임 없음, 스톨 테스트로 잰
체인 용량 ~430 ms. 그런데 그림은 여전히 **2초쯤 늦었고, 같은 스트림을 BlueOS
자체 뷰어와 QGC는 지연 없이** 보여줬다. 부품의 모든 계측이 정상이라고 하는데
부품이 여전히 틀렸다면, 그 부품을 변호하지 말고 **이 머신에서 실제로 되는
구현으로 갈아타는 것**이 맞다.

```
udpsrc ! rtpjitterbuffer latency=0 drop-on-latency=true
       ! rtph264depay ! h264parse ! avdec_h264
       ! videoscale ! videoconvert ! video/x-raw,format=BGR ! fdsink sync=false
```

중요한 두 줄: `rtpjitterbuffer latency=0`(지터 버퍼 자체를 없앤다 — **FFmpeg에는
대응물이 없다.** RTP 리더 스레드가 수 MB짜리 FIFO를 채우는데, 바깥에서는 크기를
정할 수도 들여다볼 수도 없다. `/proc/net/udp`가 0으로 보였던 이유가 이것이다)와
`fdsink sync=false`(타임스탬프에 맞춰 늦추지 말고 디코딩되는 대로 내보낸다).
스케일은 파이프라인에서 한다(I420에서 줄이면 BGR 변환 후보다 절반의 바이트만
움직인다). 실측 19-20 fps, 간격 p50 48 ms / 최대 80 ms.

`gst-launch-1.0`이 있으면 자동 선택되고, `--rov-cam-backend ffmpeg`으로 되돌린다.

**해상도는 패널을 따라간다**(`--rov-cam-size auto`, 기본). 파이프라인 출력 크기는
launch 시점에 고정되므로 패널이 충분히 커지면(예: 더블클릭으로 큰 슬롯에 올리면)
파이프라인을 다시 띄운다 — 464x262 → **940x528**, ~0.5초. 히스테리시스(1.25배/0.6배)와
3초 쿨다운이 있어서 창을 드래그해도 계속 재시작하지 않고, 소스의 1920x1080과 화면비를
넘지 않는다(없는 픽셀을 만들지 않는다). 작은 패널에서도 최대 디테일을 원하면
`--rov-cam-size 1280x720`처럼 고정하면 된다 — 패널이 다운스케일하면서 살짝 더
선명해지는 대신 파이프로 더 많은 바이트가 흐른다.

이전 FFmpeg 경로에서 고쳤던 것들(그대로 남아 있다): 진범은 **스트림 탐색 버퍼**였다. 기본
`probesize`/`analyzeduration`으로는 `open()`이 **1.7초** 걸리고 그동안 영상이 쌓여서
디코더가 그만큼 뒤에서 출발한다. 게다가 1080p 디코딩이 33 ms 예산보다 비싸서
스트림 자신의 PTS 대비 **15초마다 300 ms씩 더 밀렸다**(실측). `probesize;32`
`analyzeduration;0` `max_delay;0`으로 바꾸면 open 0.1초, **20초 동안 드리프트 +1 ms**.
대가는 26 → 20 fps다 — 소켓이 우리가 못 따라가는 프레임을 버리는 것이고, 라이브
뷰에서는 그게 맞는 거래다(프레임을 덜 받되 지연이 자라지 않는다). 30 fps가 필요하면
BlueOS Video Streams에서 스트림을 **1280×720**으로 낮추면 된다.

거기에 더해 **디코더 백로그도 흘려보낸다**(`_grab_latest`): `read()`는 버퍼에 쌓인
프레임을 순서대로 재생하므로, 기동 중에 한 번 밀리면 그 지연이 **영원히 유지된다**
(그 뒤로는 생산 속도와 같은 속도로 소비하니까). live edge에서는 grab이 한 프레임
간격만큼 걸리고, 버퍼에서 나온 프레임은 그보다 훨씬 빨리 돌아온다 — 그걸로 구분해서
색변환 없이(`grab`만) 최신까지 건너뛴다. FFmpeg는 `threads;1`인데, frame threading이
(threads−1) 프레임을 붙들기 때문이다(30 fps에서 3프레임 = 100 ms). 1스레드로도
이 1080p 스트림을 31.0 fps로 디코딩한다(실측).

그래도 ROV 카메라(`--panel2 rov`)는 **저지연 경로가 아니다** — Pi에서 H.264 인코딩 후
RTP로 오는 QGC와 같은 스트림이다. 상황 파악용이고, 조종은 C3 피드로 한다.
RTP 타임스탬프는 호스트 시계와 동기돼 있지 않아 지연 숫자를 **표시하지 않는다**
(만들어낸 숫자가 되므로).

## 명령 송신 — 두 개의 게이트

`c3_camera/control.py`의 원칙 그대로다: ArduSub이 항상 비행 제어기이고,
**명령 소스가 둘이면 그게 위험**이다(ArduSub은 마지막에 도착한 MANUAL_CONTROL을
따르므로 QGC와 동시에 쏘면 기체가 두 입력 사이에서 떤다).

* **명령은 `udpin:`으로 나간다** (기본 `udpin:0.0.0.0:14552`). BlueOS가 밀어주는
  포트를 바인드하고 **보낸 쪽에 되받아 쏘는** 방식 — QGC가 하는 그대로다.
  `udpout:192.168.2.2:14550`은 처음 기본값이었고 **조용히 아무것도 안 했다**:
  2026-08-06 프로브 결과 기체의 UDP 14550은 ICMP port-unreachable(BlueOS의 인바운드
  "GCS Server Link"가 disabled, 아웃바운드는 ephemeral 포트에서 나감). 따라서
  엔드포인트가 반드시 있어야 한다: `./c3 blueos_endpoint add --port 14552 --yes`.
  아직 아무것도 안 왔으면 CMD pill이 빨강 + `no vehicle on udp/14552 — nowhere to
  send`로 뜬다. 그 실패가 **보이지 않았던 것**이 원래 문제였다.
* **`--cmd-sysid`는 기체의 `SYSID_MYGCS`와 같아야 한다** (기본 255). ArduPilot은
  조종 입력을 **딱 한 GCS에서만** 받는다:

  ```c
  void GCS_MAVLINK::handle_manual_control(const mavlink_message_t &msg) {
      if (msg.sysid != sysid_my_gcs()) { return; }   // 조용히 버림
  ```

  NAK도 에러도 없다. 그런데 **COMMAND_LONG(=ARM)은 이 검사를 받지 않는다** — 그래서
  2026-08-06에 이 스테이션은 arm은 되는데 스러스터가 전부 1500 µs였다(기체
  SYSID_MYGCS=255, 싱크는 245). 그리퍼/조명도 MANUAL_CONTROL의 버튼 비트라 같이 버려진다.
  이제 싱크가 부팅 시 `SYSID_MYGCS`를 **읽어서** 다르면 CMD pill이 빨강 +
  `SYSID_MYGCS=255 != 245 — sticks ignored`로 뜬다.
* **arm 표시는 컴포넌트 1(오토파일럿)의 하트비트만 믿는다.** 이 링크에는 최소 세
  시스템이 하트비트를 쏜다 — 오토파일럿(1/1), 온보드 컴포넌트(1/194), GCS(255/240).
  1/194는 `base_mode = 0x80`을 쓰는데 그게 하필 SAFETY_ARMED 비트라서, 아무 하트비트나
  받으면 **disarmed 기체가 MOTORS ARMED로 표시됐다**(2026-08-06 실측). 틀릴 수 있는
  arm 표시는 없는 것보다 나쁘다. `c3_camera/mavlink_log.py`도 같이 고쳤다.
* 기본은 `NullCommandSink` — 아무것도 송신하지 않는다.
* `--allow-command`를 줘야 MAVLink 싱크가 **만들어지고**, 그 다음 UI의
  `COMMAND ENABLE` 스위치를 켜야 **송신한다**.
* **arm / disarm은 구현했다**(2026-08-06, 사용자 요청). 대신 비대칭 규칙 두 개:
  - **ARM은 1.2초 hold-to-confirm**이고 `COMMAND ENABLE`이 켜져 있어야 하며,
    누르는 순간 **모든 축을 0으로 만든 뒤** 명령을 보낸다(키를 누른 채 arm하면
    모터가 살아나는 순간 기체가 움직인다). 모달 "정말?" 창을 안 쓴 이유는 그게
    기체를 조종 중인 창의 키보드 포커스를 빼앗고, 조종사는 읽지 않고 닫는 걸
    학습하기 때문이다.
  - **DISARM은 클릭 한 번**이고 `COMMAND ENABLE`이 꺼져 있어도 나간다.
    정지를 스위치 뒤에 숨기는 게 더 위험한 실수다.
  - `MOTORS ARMED` 표시는 **기체의 HEARTBEAT**를 그대로 보여준다. 우리가 보냈다고
    armed로 칠하지 않는다 — 그래야 pre-arm 검사에 걸려 거부된 arm이 "아무 일도
    안 일어남"으로 정확히 보인다. 텔레메트리가 stale이면 `MOTORS ?`(노랑)다.
* **비행 모드 변경은 여전히 없다.** MANUAL / STABILIZE / DEPTH HOLD 중 무엇이냐가
  같은 축 입력의 의미를 바꾸므로 그 선택은 QGC에 남긴다.
* E-STOP은 **중립 + 송신 해제**까지고 disarm은 하지 않는다. 수중에서 disarm은
  뎁스홀드를 잃고 기체가 표류하는 것이라 항상 더 안전하지도 않다. 진짜로 모터를
  죽여야 하면 옆의 DISARM 버튼(또는 QGC)을 쓴다.
* enable 중에는 1 Hz HEARTBEAT를 보낸다(ArduSub GCS failsafe가 이 스테이션이
  죽었을 때 기체를 세우는 장치라서). disable하면 즉시 멈춘다.
* 데드맨: 입력이 `--deadman-ms`(기본 500)보다 오래되면 **중립을 보낸다**.
  "안 보내기"는 정지 명령이 아니다.
* MANUAL_CONTROL `z`축 관례는 `--z-convention`으로 **명시**한다. ArduSub 버전에
 따라 `0..1000`과 `-1000..1000`이 갈리고, 틀리면 중립을 의도했을 때 기체가 내려간다.
* **그리퍼/조명은 ArduSub 조이스틱 버튼 함수로 나가고, 두 함수의 의미가 다르다.**
  QGC > Vehicle Setup > Joystick > Button Assignment에서 읽어서 넘긴다:

  이 기체 값이 **기본값으로 박혀 있다** — 매번 넘길 필요 없다:

  | 버튼 | `BTNn_FUNCTION` | ArduSub 함수 | 의미 | 플래그(기본값) |
  |---|---|---|---|---|
  | 0 | 77 | `servo_1_max_momentary` | **누르는 동안** 열림 | `--btn-gripper-open 0` |
  | 15 | 76 | `servo_1_min_momentary` | **누르는 동안** 닫힘 | `--btn-gripper-close 15` |
  | 13 | 33 | `lights1_dimmer` | **누를 때마다** 한 칸 | `--btn-lights-down 13` |
  | 14 | 32 | `lights1_brighter` | **누를 때마다** 한 칸 | `--btn-lights-up 14` |

  기본값이어도 **시동 시 기체에 대조한다**: 싱크가 각 `BTNn_FUNCTION`을 읽어서
  위 번호와 다르면 그 비트를 **쓰지 않고** 이유를 로그로 남긴다. 다른 기체에서
  엉뚱한 버튼을 누르는 일이 없도록. 어떤 기능을 아예 안 건드리려면 `-1`을 준다.

  그래서 그리퍼는 **유지(hold)**, 조명은 **펄스(press)** 로 보낸다. 조명 슬라이더를
  움직이면 그 차이만큼 버튼 **누름을 큐에 쌓아** 100 ms on / 100 ms off로 내보낸다
  (ArduSub은 연속 메시지 사이의 *변화*를 보므로 눌렀다 떼야 다음 누름이 인식된다).
  칸 수는 기체의 `JS_LIGHTS_STEPS`와 맞춰야 하고 `--lights-steps`가 그 값이다.
  둘을 반대로 하면 — 레벨을 보내면 조명은 한 칸 가고 멈추며, 펄스를 보내면 턱이
  까딱하고 만다. 비트 번호를 안 주면 **아무것도 보내지 않고** 패널이 OFFLINE +
  `no button bit mapped`로 표시한다.
* **조명 레벨은 절대값으로 다룬다.** 슬라이더 한 칸 = 기체 한 스텝이고, 스텝 수는
  `JS_LIGHTS_STEPS`를 **기체에서 읽어서** 맞춘다(이 기체는 8, 코드 기본값이던 10이
  아니다 — "70%인데 꺼진 것처럼 보인다"의 절반이 이것이었다). 나머지 절반은 누적
  오차다: 프레스 하나만 유실돼도 우리 모델이 영구히 어긋나고 되돌릴 피드백이 없다.
  그래서 **OFF/MAX는 리싱크 지점**이다 — 끝단을 두 번 넘겨 눌러서 모델을 진실로
  되돌린다. `--lights-servo N`을 주면 그 서보 출력을 읽어 **실측 레벨**을 표시한다
  (이 기체는 SERVO13이 꺼짐일 때 1100 µs).
* **그리퍼에는 슬라이더가 없다.** momentary 서보에는 명령할 위치가 없기 때문이다.
  CLOSE/OPEN을 누르고 있는 동안 구동되고, 패널은 방향과 누른 시간을 보여준다.

## 키보드

```
W A S D / ←→↑↓     surge, sway   (화살표는 surge, yaw)
Q E                yaw
R F / PgUp PgDn    heave
space              ALL STOP
G H                그리퍼 close / open (누르고 있는 동안)
[ ]                조명 down / up
, . /              카메라 틸트 down / level / up (`,`와 `/`는 누르고 있는 동안)
Ctrl+R             UI 녹화 시작/정지        F11 전체화면      Esc  ALL STOP + 송신 해제
```

arm/disarm에는 **일부러 단축키를 두지 않았다.** 손가락이 미끄러져서 모터가 살아나는
경로를 만들지 않기 위해서다 — 마우스로 ARM을 1.2초 눌러야 한다.

버튼·슬라이더는 전부 `NoFocus`다. 포커스를 가져가는 위젯이 하나라도 있으면
화살표가 ROV 대신 슬라이더를 움직이고 스페이스가 마지막 버튼을 다시 누른다 —
조종사가 조용히 키보드를 잃는다. auto-repeat는 창 경계에서 걸러낸다(안 그러면
"누르고 있음" 상태가 OS 키 반복 속도로 깜빡인다). 창이 포커스를 잃으면
key-release 이벤트도 같이 잃으므로 **전체 정지**로 처리한다.

## 조이스틱

`/dev/input/js*`를 **커널에서 직접** 읽는다(패키지 불필요, `joystick.py`). 8바이트짜리
이벤트를 논블로킹으로 훑기 때문에 워커 스레드 없이 GUI 타이머에서 50 Hz로 폴링한다 —
조이스틱이 죽어도 UI가 멈출 방법이 없다. 뽑으면 알아서 감지하고 2초마다 다시 찾는다.

* **COMMAND ENABLE이나 arm 상태와 무관하게 항상 읽고 표시한다.** 그래야 아무것도
  움직일 수 없는 상태에서 매핑을 확인할 수 있다. 송신만 게이트가 걸린다.
* 스틱과 키보드는 **더해진 뒤 클램프**된다. 어느 쪽만으로도 최대 편향에 도달한다.
* 텔레옵 패널의 `JOY` 줄이 **축 번호와 눌린 버튼 번호를 실시간으로** 보여준다.
  매핑은 그걸 보면서 `--js-axis-*` / `--js-btn-*`로 맞추면 된다.
* **아날로그 트리거는 대기 상태가 중앙이 아니라 끝**이다. 이 데스크톱의 Xbox
  Wireless Controller는 축 4·5가 무입력에서 `-1.000`을 낸다 — 그 축을 heave에
  매핑해 뒀더니 GUI를 켜자마자 **heave가 +0.60으로 고정**됐다(조종사가 준 적 없는
  명령). 그래서 축이 처음 보고한 값이 그 축의 0이 된다. 단 **중앙 근처(|v| ≤ 0.5)가
  아닐 때만** — 진짜 가운데 있는 스틱은 진짜 0을 유지한다. 자동 0점 잡힌 축은
  시작 로그에 나온다.
* 실측 축 배치(Xbox Wireless Controller): `0,1` 왼쪽 스틱, `2,3` 오른쪽 스틱,
  `4,5` 트리거, `6,7` 십자키. 기본 매핑은 여기에 맞췄다(`surge -1`, `sway 0`,
  `yaw 2`, `heave -3`; 음수 = 반전).
* 데드존 0.08은 **재스케일**되므로 최대 편향이 1.0에 그대로 도달한다 — 안 그러면
  조종사가 바에 보이는 마지막 8%를 영영 못 쓴다.
* **PROPULSION은 갱신 레이트를 같이 띄운다.** vehicle 워커는 새 `SERVO_OUTPUT_RAW`가
  왔든 안 왔든 마지막 값을 100 ms마다 다시 publish하는데, 예전엔 그때마다 `now()`로
  도장을 찍었다. 그러면 watchdog이 **기체 데이터가 아니라 우리 심장박동**을 먹어서,
  피드가 완전히 끊겨도 pill은 초록이고 arm 전의 1500 µs가 그대로 남았다 — 실제로
  모터가 도는 동안 패널은 "안 움직임"이었다(2026-08-07 실기 보고). 지금은 레코드의
  `t_host`로 도장을 찍어서 침묵이 정상적으로 늙고, 요약줄에 `10 Hz`나
  `NOT UPDATING — reading is old`가 뜬다. 회귀 테스트 2개가 지킨다.
* 스틱을 밀었는데 기체가 안 움직이면 텔레옵 하단이 이유를 말한다:
  `TX 20 Hz  MOTORS DISARMED — nothing will move`. 바가 가득 차고 송신도 정상인데
  기체가 가만히 있는 상황에서 두 스위치 중 어느 쪽이 꺼졌는지 알려주는 유일한 줄이다.

### 버튼 매핑은 QGC와 같다 — 지도가 하나뿐이기 때문에

**패드의 버튼 비트마스크를 기체로 넘긴다**(`MANUAL_CONTROL.buttons`,
`--no-js-passthrough`로 끌 수 있음). 그러면 **어떤 버튼이 무슨 기능인지는 기체의
`BTNn_FUNCTION` 파라미터가 정한다** — QGC와 똑같은 계약이다. 이 스테이션에 위젯이
없는 기능(mode_manual/depth_hold/stabilize, gain_inc/dec, trim, shift,
input_hold_set, arm/disarm)도 QGC에서 설정한 그대로 동작한다.

#### 커널 번호 ≠ 기체 번호 (이걸 몰라서 사고가 났다)

넘기기 전에 **번호를 번역한다**(`--no-js-translate`로 끔). 커널은 HID 서술자 순서로
버튼을 매기고, **QGC는 SDL2를 통해 읽으면서 `SDL_GameControllerButton`이라는 고정
논리 배치로 재정렬한다.** 기체의 `BTNn_FUNCTION`은 그 SDL 번호로 설정돼 있다.

커널 번호를 그대로 넘기면 어떻게 되는지 2026-08-07에 실기에서 확인했다:
**조종사가 카메라 틸트(왼쪽 범퍼, 커널 6)를 눌렀는데 기체가 `BTN6_FUNCTION = arm`을
읽고 모터가 살아났다.** 실기 관측 4건이 전부 아래 표로 설명된다.

| 물리 | 커널 | → 기체(SDL) | QGC 기능 |
|---|---|---|---|
| A / B / X / Y | 0 / 1 / 3 / 4 | 0 / 1 / 2 / 3 | gripper open · manual · depth_hold · stabilize |
| LB / RB | 6 / 7 | **9 / 10** | mount_tilt_down / up |
| View / Menu | 10 / 11 | **4 / 6** | **disarm / arm** |
| Xbox | 12 | 5 | shift |
| 스틱 클릭 L/R | 13 / 14 | 7 / 8 | mount_center / input_hold_set |
| **십자키** | **축 6,7 (hat)** | **11–14** | gain_inc/dec · lights dimmer/brighter |

**십자키는 버튼이 아니라 hat(축 6,7)이다.** 그래서 눌러도 입력 표시에 아무것도 안
뜨고 기체로도 아무것도 안 갔다. SDL이 그 hat을 버튼 11–14로 바꾸므로 우리도 그렇게
한다.

지도가 없는 패드는 커널 번호를 그대로 보내고 **시작 로그가 경고한다.** 지도에 없는
버튼은 **아예 보내지 않는다** — 모르는 번호를 찍어보는 건 기체에서 모르는 기능을
누르는 것이다.

확인은 텔레옵의 `JOY` 줄에서 한다. 읽는 법:

```
btn 3>2     왼쪽 3 = 커널 번호, 오른쪽 2 = 기체 번호. X 버튼을 커널은 3으로,
            기체(SDL/QGC)는 2로 센다. 기능은 오른쪽 번호가 정한다.
btn 1       화살표가 없으면 두 번호가 같다는 뜻 (B 버튼: 커널 1 = 기체 1).
btn 11      십자키. 커널에선 버튼이 아니라 축이라 왼쪽에 쓸 번호가 없다.
```

모터가 살아날 수 있는 버튼을 누르기 전에 이 줄로 확인할 수 있어야 한다는 게 요점이다.

전에는 스테이션이 **자기만의 두 번째 패드 지도**(`--js-btn-*`, 기본값 0/3/4/5)까지
들고 있었다. 지금 `--js-btn-*`는 "그 버튼이 눌리면 화면의 어느 칩에 불이 들어오는가"만
정하고, 기본값은 `--btn-*`(= 기체 값)와 **같은 번호**다. passthrough가 켜져 있으면
칩은 표시만 하고 명령은 내지 않는다 — 안 그러면 한 번 눌러 두 번 나간다(조명 두 칸씩).

이 기체의 배치(QGC > Joystick > Button Assignment):

| # | 기능 | 물리 | # | 기능 | 물리 |
|---|---|---|---|---|---|
| 0 | `servo_1_max_momentary` 그리퍼 open | A | 8 | `input_hold_set` | RS 클릭 |
| 1 | `mode_manual` | B | 9 | `mount_tilt_down` | LB |
| 2 | `mode_depth_hold` | X | 10 | `mount_tilt_up` | RB |
| 3 | `mode_stabilize` | Y | 11 | `gain_inc` | 십자키 ↑ |
| 4 | `disarm` | View | 12 | `gain_dec` | 십자키 ↓ |
| 5 | `shift` | Xbox | 13 | `lights1_dimmer` | 십자키 ← |
| 6 | `arm` | Menu | 14 | `lights1_brighter` | 십자키 → |
| 7 | `mount_center` | LS 클릭 | 15 | `servo_1_min_momentary` 그리퍼 close | **이 패드엔 없음** |

`MANUAL_CONTROL.buttons`는 uint16이라 **16번 이상은 실을 수 없다**(그런 버튼이 눌리면
무시하고 한 번 경고한다).

#### 패드가 못 누르는 기능이 있으면 `--js-remap`

**그리퍼 close는 15번인데, 15는 SDL의 `MISC1`("Share")이고 이 패드는 커널이 버튼
0–14만 보고한다** — 즉 패드로는 누를 방법이 아예 없었다. (UI의 CLOSE 버튼과 `G` 키는
멀쩡하다. 비트 15는 우리가 직접 세울 수 있고, 못 누르는 건 *패드*뿐이다.)

그래서 **패드가 보내는 번호를 보내기 직전에 다시 쓴다**:

```
--js-remap 11:0,12:15      (기본값)  십자키 ↑ → 그리퍼 open, ↓ → 그리퍼 close
```

원래 번호 대신 **바뀐 번호만** 나간다(둘 다 보내면 한 번 눌러 두 기능이 나간다).
번역 직후에 적용하므로 그 뒤로는 번호가 한 벌뿐이다 — 비트마스크·칩·`JOY` 줄이 전부
같은 숫자를 쓴다.

> **`--btn-gripper-*`를 대신 바꾸면 안 된다.** 그건 "이 기체의 `BTNn_FUNCTION`이 어디
> 있는가"를 적는 값이지 취향이 아니다. 2026-08-07에 그걸 11/12로 바꿨더니 UI의 그리퍼
> 버튼이 그리퍼 대신 `gain_inc`를 누르게 돼서 **패드도 UI도 둘 다 안 되는** 상태가 됐다.
> 패드 버튼을 바꾸고 싶으면 `--js-remap`, 기체 배치를 바꿨으면 `--btn-*`.

패드 버튼도 **COMMAND ENABLE 게이트를 그대로 지난다.** 스위치가 꺼져 있으면 패드의
`arm`(Menu 버튼)도 나가지 않는다. 다만 스위치가 켜져 있으면 **패드는 버튼 한 번으로
arm된다** — 화면의 ARM 버튼은 1.2초 hold-to-confirm인데 패드는 아니다. 그게 QGC와
같은 동작이라 그렇게 뒀다.

## 비행 모드 — arm은 MANUAL로 들어간다

MANUAL이 아닌 모드(STABILIZE / DEPTH HOLD / POSHOLD)에서 arm하면 **오토파일럿이
자세·수심을 잡으려고 스스로 스러스터를 돌린다.** 조종 입력이 0이어도 그렇다. 그게 그
모드의 존재 이유이고, 물 밖에서는 "모터가 갑자기 혼자 돈다"로 보인다.

ArduSub은 **마지막에 있던 모드를 그대로 들고 있는다.** 그래서 모드를 건드리지 않으면
ARM 버튼이 그 모드를 물려받고, COMMAND ENABLE + ARM만으로 스러스터가 돈다
(2026-08-07 실기 보고).

그래서 **ARM 버튼이 arm 직전에 MANUAL을 먼저 요청한다**(`--arm-mode`, 기본 `MANUAL`;
`keep`이면 예전처럼 물려받는다). MANUAL은 armed 상태에서 입력이 0이면 가만히 있는
유일한 모드다.

**모드를 잠그지는 않는다.** 텔레옵 패널의 `MANUAL | STAB | DEPTH` 버튼으로 언제든,
arm 전후 모두 바꿀 수 있다 — 즉 **STABILIZE는 조종사가 누를 때만 걸린다**.

* 하이라이트되는 건 **기체가 HEARTBEAT로 보고한 모드**이지 우리가 요청한 모드가
  아니다. ArduSub은 armed 상태에서 일부 전환을 거부하는데, 요청을 표시했다면 화면은
  MANUAL인데 기체는 STABILIZE인 상태 — 즉 ARM이 모터를 돌리는 바로 그 상태 — 가 된다.
  `MOTORS ARMED`와 같은 규칙이다. 텔레메트리가 끊기면 하이라이트도 꺼진다.
* MANUAL이 아닌 모드가 걸려 있으면 버튼이 **경고색**으로 뜬다. "지금 이 기체는 혼자
  스러스터를 돌릴 수 있다"는 뜻이다.
* disarm→arm 전이에서 모드를 로그에 찍는다:

```
[warn] ARMED in STABILIZE — the autopilot will run the thrusters by itself
       to hold attitude/depth, with no stick input.
```

> 이건 원래 **일부러 구현하지 않았던 기능**이다("스테이션은 명령원이지 권한이 아니다,
> 모드는 QGC 몫"). 실기에서 뒤집혔다 — arm은 할 수 있는데 armed가 무슨 뜻인지는 못
> 정하는 스테이션은 더 안전한 게 아니라 그냥 덜 예측 가능하다.
>
> **패드의 `arm` 버튼은 예외다.** 그건 비트마스크로 기체에 직행하므로 우리가 가로챌 수
> 없고, 따라서 모드를 물려받는다(QGC와 같음). 패드로 arm할 거면 모드를 먼저 확인할 것.

## 카메라 틸트

ROV 기본 RGB 카메라의 마운트를 올리고 내린다. 조명이 아니라 **그리퍼와 같은 모양**이다:
ArduSub은 `mount_tilt_up/down`을 버튼 **repeat** 경로에서 처리하므로 비트가 켜져 있는
동안 계속 움직인다(조명의 "누를 때마다 한 칸"과 다르다). 그래서 누르고 있는 버튼과
`,` / `/` 키, 그리고 한 번 누르면 수평으로 돌아오는 `LEVEL`(`.`, `mount_center`).

각도 표시는 **기체가 보내줄 때만** 숫자다:

* `MOUNT_STATUS.pointing_a`가 오면 그 값을 도로 쓴다(기체가 믿는 각도).
* 안 오면 `--tilt-servo <ch>`를 준 경우에만 그 채널의 PWM을 `--tilt-min-deg` /
  `--tilt-max-deg` 범위로 환산해서 보여주고 **`[유도]`라고 표시한다** — PWM은 서보
  범위를 모르면 각도가 아니다.
* 둘 다 없으면 `no angle reported (open loop)`라고 쓰고 **명령을 각도인 척 그리지
  않는다**(그리퍼와 같은 규칙).

버튼 번호(9/10/7)는 이 기체의 QGC 배치에서 왔고, 기능 번호(23/22/21)는 ArduSub
`JSButton` enum **[스펙]** 이다. 접속하면 `BTNn_FUNCTION`을 읽어 대조하고, 어긋나면
**그 버튼을 누르지 않고** 양쪽 가능성(기체 배치가 다르거나 우리 표가 틀렸거나)을
같이 로그에 적는다. 하드웨어로 확인한 적은 아직 없다.

## 화면 녹화

`QWidget.grab()`으로 **UI 위젯 트리**를 찍는다(X11 루트 창이 아니라). 다른 창이
위에 뜰 일이 없고, Wayland 캡처 권한 문제가 없고, offscreen에서도 돌아서
테스트가 가능하다. 대신 위젯을 한 번 더 렌더링하므로 공짜가 아니다 → 기본 12 fps.
grab은 GUI 스레드에서, 인코딩은 별도 스레드에서, 사이의 큐는 **유한하고 넘치면
버린다**(조종 화면을 얼려서 녹화를 지키는 건 거래가 안 된다). 버린 수는
`.json` 사이드카에 기록된다 — 12 fps라고 적힌 파일이 실제로 7 fps였다면 그
파일로 잰 모든 시간이 40% 틀리기 때문이다.

### 피드별 녹화

**녹화 토글은 각 영상 패널 위에** 있다(`REC` → 누르면 `● REC C3 RGB`). 헤더에
네 개를 넣었더니 42 px에 맞추느라 "ROV"가 글자로 잘렸다 — 피드 이름이 들어갈 자리는
피드 위에 있다. 헤더에는 창 전체 녹화(`REC UI`)만 남았다.

이름은 세 곳에서 같다: 패널 제목 · 버튼 · 파일명(`c3_rgb_*.mp4`, `c3_depth_*.mp4`,
`default_rgb_*.mp4`). 창 녹화와 피드 녹화는 각각 **독립된 파일**이다. UI는 조종사가 무엇을
보고 무엇을 했는지의 기록이고, 피드는 그 그림 자체다 — 다른 것이므로 따로 켠다.

### C3 Depth를 녹화하면 센서도 같이 남는다

**depth 피드에만** 걸린 동작이다. depth 영상 하나로는 데이터셋이 안 되고, 데이터셋으로
만드는 건 "각 프레임을 찍을 때 카메라가 어떻게 움직이고 있었나"이며 그건 영상보다
훨씬 빠른 IMU·기압계·나침반에 있다. 컬러 피드는 조종사가 보는 그림이지 기하가 아니라서
따라붙지 않는다(피드마다 로그를 남기면 파일만 세 배가 된다).

영상과 **같은 stem**으로 두 파일이 더 생긴다(`sensorlog.py`):

```
c3_depth_20260806_225127.mp4          영상
c3_depth_20260806_225127_rov.jsonl    ArduSub: RAW_IMU/SCALED_IMU2, SCALED_PRESSURE(2),
                                      ATTITUDE, AHRS2, VFR_HUD, BATTERY_STATUS,
                                      SYS_STATUS, EKF_STATUS_REPORT, MOUNT_STATUS,
                                      SERVO_OUTPUT_RAW, LOCAL/GLOBAL position
c3_depth_20260806_225127_c3_imu.jsonl C3 BNO086: t_device, seq, ax/ay/az, gx/gy/gz
c3_depth_..._rov.json / _c3_imu.json  각각의 매니페스트
```

**소스마다 파일이 따로**인 건 의도다. ROV IMU와 C3 IMU는 서로 다른 장치이고 시계도
다르다 — 하나로 합치려면 시계 하나를 골라야 하고, 그 순간 다른 쪽에 대해 거짓말을
하게 된다. 각 워커가 자기 파일을 자기 스레드에서 쓴다.

> **2026-08-09 정정**: `_c3_imu.jsonl`은 2026-08-06~09 사이에는 **실기에서 한 번도 안
> 생겼다.** `C3VideoWorker`가 `LoopWorker`인데 요청을 `@Slot`으로 받고 있었고, LoopWorker는
> `run()`이 블록해서 이벤트 루프에 도달하지 않으므로 **큐잉된 슬롯이 배달되지 않는다.**
> `_rov.jsonl`은 `VehicleWorker`가 `TimerWorker`라 정상이었고, 데모 백엔드도 TimerWorker라
> 테스트가 못 잡았다 — 그래서 "반쪽짜리 데이터셋"이 조용히 나왔다.
> 지금은 평범한 스레드 안전 메서드(`request_sensor_log`) + `DirectConnection`이고,
> `test_c3_sensor_log_request_reaches_a_loopworker`와 LoopWorker 슬롯을 금지하는 AST 스캔
> 테스트가 지킨다. **그 기간에 찍은 depth 녹화에는 C3 IMU가 없다.**

CSV가 아니라 **JSONL**인 이유: 필드가 서로 다른 메시지가 대여섯 종류 섞여 들어오므로
CSV는 빈칸투성이 union 스키마거나 타입별 파일이 되고, union 스키마는 펌웨어가 필드를
추가하면 **조용히 버린다.** `pandas.read_json(path, lines=True)` 한 줄로 읽힌다.

시계: `t`는 topside `time.monotonic()`(영상 레코더와 **같은 시계**라 프레임과 맞출 때
쓰는 값), `t_unix`는 벽시계. 센서 자신의 타임스탬프(`time_boot_ms`, `t_device`)는
레코드 안에 **그대로 보존**되고, 시간이 중요한 작업에는 그쪽을 써야 한다 — `t`에는
링크 지연이 포함돼 있다.

매니페스트의 `hz_measured`를 먼저 보고 믿을지 정하면 된다. 200 Hz로 요청한 IMU가
거기서 47 Hz로 나오면 센서가 아니라 **링크가** 레이트를 정한 것이다.

피드 녹화는 **패널이 받은 그대로**(표시된 크기) 저장한다. 워커가 이미 디코딩·축소한
QImage를 인코더 스레드로 넘기므로 GUI 스레드 비용이 없지만, **센서 원본이 아니다.**
원본 해상도 + 타임스탬프가 필요한 데이터셋은 `c3_camera/c3_collect.py`가 그 용도다.

VideoWriter는 고정 fps가 필요한데 프레임은 링크 사정대로 온다. 그래서 **시작 시점의
실제 fps로 열고, 실제로 무슨 일이 있었는지를 사이드카에 적는다**:

```json
{"requested_fps": 30.0, "frames_written": 116, "frames_dropped": 0,
 "duration_s": 3.89, "effective_fps": 29.81, "size": [960, 540],
 "source": "rov_gui panel 'main', as displayed"}
```

데스크톱 전체를 GUI 부담 없이 녹화하려면 이 클래스가 아니라:

```bash
ffmpeg -f x11grab -framerate 15 -i :0.0 -c:v libx264 -preset veryfast out.mp4
```

## 테스트

```bash
QT_QPA_PLATFORM=offscreen ~/miniforge3/envs/robust/bin/python rov_gui/tests/test_offline.py
# 또는
QT_QPA_PLATFORM=offscreen ~/miniforge3/envs/robust/bin/python -m pytest rov_gui/tests/test_offline.py -v
```

14개. 데모 백엔드를 실제 버스·실제 위젯으로 2초 돌리고 결과를 검사한다:
프레임이 패널까지 도달하는지, 워치독이 침묵을 빨갛게 만드는지, 최소 창 크기가
아직 한 화면에 들어가는지, 트리에 스크롤 영역이 없는지, 녹화 파일에 프레임이
들어있는지, QImage↔numpy 왕복이 stride를 지키는지(패딩된 행은 그림을 어긋나게
자르는 고전적 버그다).

## 환경 메모

* Qt 바인딩은 PyQt5 → PyQt6 → PySide6 순으로 찾는다(`ROV_GUI_QT_API`로 고정 가능).
  검증된 조합은 `robust` env의 **PyQt5 5.15.11 / Qt 5.15.14**.
### xcb 플랫폼 플러그인이 죽는 원인 두 가지 (둘 다 코드로 막아뒀다)

증상은 **똑같이** 다음 한 줄이고 `Aborted (core dumped)`로 끝나는데, 원인이 완전히
다르다. 2026-08-06에 `--source hw` 첫 실행에서 둘 다 맞았다.

```
qt.qpa.plugin: Could not load the Qt platform plugin "xcb" in "..." even though it was found.
```

1. **cv2가 `QT_QPA_PLATFORM_PLUGIN_PATH`를 가로챈다.** `cv2/config-3.py`가 import
   시점에 이 변수를 cv2 자기 Qt5 플러그인 디렉토리로 **무조건** 덮어쓴다. 그 폴더엔
   `platforms/libqxcb.so`가 딱 하나 있는데 cv2의 Qt 빌드용이라 PyQt5의 Qt에
   로드되지 않는다. hw 경로에서는 **preflight의 `check_python_env`가 버전을 읽으려고
   cv2를 import**하면서 QApplication 생성 *전에* 이 일이 벌어진다.
   → `rov_gui.qt.sanitize_plugin_path()`가 QApplication 직전에 경로를 우리 바인딩
   것으로 되돌린다. cv2는 `import_cv2()`로만 import한다(원래 값 복구).
   `c3_camera.viz`는 정반대 일(`repair_qt_plugin_path`)을 하는데 `cv2.imshow`를
   위한 것이라 맞다 — 다만 QApplication을 만들기 전에 import하면 안 된다.
   **오프스크린 테스트로는 절대 안 잡힌다**: cv2 폴더엔 `libqoffscreen.so`가 없어서
   Qt가 조용히 자기 폴더로 폴백한다. X11에서만 죽는다.

2. **`LD_LIBRARY_PATH`가 시스템 Qt를 먼저 물린다.** 이 데스크톱 셸에는 OpenFOAM 12가
   `/usr/lib/x86_64-linux-gnu`를 `LD_LIBRARY_PATH`에 꽂아 둔다. PyQt5의 확장 모듈은
   `DT_RPATH`(LD_LIBRARY_PATH보다 우선)라 `libQt5Core`는 제대로 잡지만, 플러그인
   `Qt5/plugins/platforms/libqxcb.so`는 `DT_RUNPATH`(LD_LIBRARY_PATH가 이김)라
   `libQt5XcbQpa.so.5`가 우분투 시스템 Qt로 풀려서 죽는다:
   `undefined symbol: _ZN23QPlatformVulkanInstance22presentAboutToBeQueuedEP7QWindow`.
   → `rov_gui.qt.preload_platform_libs()`가 우리 바인딩의 Qt 라이브러리를 절대경로
   `RTLD_GLOBAL`로 먼저 dlopen한다. 나중 `dlopen`은 이미 올라온 SONAME에 붙으므로
   검색 자체가 일어나지 않는다. (`os.environ`에서 `LD_LIBRARY_PATH`를 지우는 건
   소용없다 — glibc가 프로세스 시작 시점에 검색 경로를 확정한다.)
