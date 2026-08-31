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

화면에는 RGB 축(원하면 `--pose-box3d`로 12-edge 박스)과 왼쪽 위 단계 칩이 뜬다:

```
POSE TRACKING  13 Hz    solve 210 ms    reg 1
```

**세 숫자가 서로 다른 질문에 답한다**(2026-08-23):

| 숫자 | 뜻 | 어디서 |
|---|---|---|
| `n Hz` | **새 자세가 실제로 도착하는 빈도** — 오버레이·물체 앵커·follow가 받는 그 속도 | `perception/session.py`의 `_RateMeter`가 **직접 센다** |
| `solve n ms` | GPU가 **한 프레임을 푸는 데** 걸리는 시간 | 업스트림 트래커의 `hz`를 ms로 되돌린 값 |
| `reg n` | 재등록 횟수 = 자세가 물체에서 미끄러진 횟수 | FoundationPose |

둘을 같이 보면 **느린 원인이 어디인지** 나온다: `solve` ≫ `1/rate`면 GPU가 병목,
`solve` ≪ `1/rate`면 카메라나 마스크가 병목이다.

> **2026-08-23 이전 기록과 합산 금지.** 그 전에는 칩의 `n Hz`가 **`1/평균 계산시간`**
> 이었다 — 업스트림 두 트래커(`sam2_live/tracker.py:507-512`, `sam2_live/pose.py:357-365`)가
> 둘 다 `1.0 / mean(직전 30개 job 소요시간)`을 `hz`라는 이름으로 발행하기 때문이다.
> 0.73 s짜리 등록 한 번과 20 ms짜리 추적이 **같은 30-샘플 창**에 섞여서, 물체가 가만히
> 있는데도 칩이 5 Hz → 9 Hz → 5 Hz로 흔들렸다(조종사, 2026-08-23). 갱신률은 애초에
> 그 숫자로 알 수 없었다. 파일에서도 이름을 바꿔 경계를 세웠다:
> `pose_hz` → **`pose_hz_measured`** + `solve_ms`(CSV), `sam_hz` → `sam_hz_measured` +
> `sam_solve_ms`(jsonl TRACK 행). 옛 파일에서 `pose_hz`를 이름으로 찾던 리더는
> 새 파일에서 **조용히 다른 양을 읽는 대신 실패한다.**

측정 방식: 두 트래커 모두 solve마다 **새 배열 객체**를 만들고 실패하면 이전 것을 그대로
두므로, 값 비교가 아니라 **객체 동일성**(`is`)이 "새로 나왔다"의 정확한 판정이다(가만히
있는 물체의 똑같은 자세도 새 도착으로 제대로 센다). 표본은 **프레임을 넘길 때마다 한
번** 뜬다 — 두 트래커 다 job 슬롯이 하나라 프레임당 결과가 하나를 못 넘기므로 놓칠 수
없고, `poll`(10 Hz 발행 게이트)에서 뜨면 그보다 빠른 건 전부 앨리어싱된다. 분모는
마지막 도착이 아니라 **지금**까지의 시간이라, 자세가 끊기면 숫자가 얼어붙지 않고
2 초에 걸쳐 0으로 내려간다. `test_the_pose_rate_is_an_arrival_rate_not_one_over_compute_time`.

**물체의 좌표는 이 패널에 없다**(2026-08-23, 조종사 요청으로 제거). 예전에는 아래쪽에
`x +0.084  y +0.020  z +0.581 m  d 0.59 m` 한 줄이 더 있었는데, 그건 **카메라 프레임**의
숫자다 — 이 스테이션의 다른 어떤 것도 그 프레임에서 일하지 않으므로 무엇과도 비교할 수
없고 그 위로 날 수도 없었다. 물체 위치는 이제 **한 곳**, 궤적 플롯의 readout에만
나온다(아래 "물체를 따라간다"). 칩에 남은 `n Hz`/`reg n`은 위치가 아니라 **health**다.

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
POSE TRACKING  13 Hz    solve 210 ms
TRACKING  13 Hz  solve 34 ms — no pose yet   ← pose는 기대하는데 아직 없음
TRACKING  13 Hz  solve 34 ms — mask only     ← --pose-no-build. 원래 자세가 없는 모드
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

**수집이 취소되는 두 가지 (2026-08-21, 실기)**. 로그를 보면 어느 쪽인지 나온다.

1. `stopped after N deg with only M usable views (need 6)` — **arc 예산**을 다 썼다.
   `--pose-max-arc`(기본 75°)는 **회전량**으로 소비되므로, 빨리 돌리면 대부분의 프레임이
   거부당하는 사이에 75°가 지나가 버린다. **이 값을 올리는 건 답이 아니다** — 상류
   합성 실험에서 per-view 오차가 80°에서 9.7 mm, 100°에서 87 mm, 120°에서 283 mm로
   **절벽처럼** 무너진다. 답은 더 천천히 도는 것이다.
2. `lost the object — lost — reacquiring` → `could not reacquire — click again` —
   **SAM2가 마스크를 놓쳤고 재획득에 실패**했다. 물속에서는 모션 블러와 기체 자체의
   움직임 때문에 벤치보다 훨씬 자주 놓친다. 재시도 예산이 `--pose-lost-timeout`
   (기본 15 s)인데 2026-08-21까지 **CLI로 나와 있지 않았다**. 지금은 있다:

   ```bash
   ./c3 gui --source hw --mpc --pose --pose-lost-timeout 45
   ```

   재획득 중에는 아무것도 기록되지 않고 아무것도 버려지지 않으므로, 올리는 비용은
   기다리는 시간뿐이다. `--pose-lost-grace`(기본 3 s)는 **짧은** 끊김을 오버레이가
   언제부터 "놓쳤다"고 부를지일 뿐 — 긴 끊김을 견디는 노브가 아니다.

### 물체를 3-D로 보려면 — trajectory 패널의 `3D` 버튼 (2026-08-21)

**떠 있던 `MAP` 창은 없어졌다.** 그 창은 물체를 **카메라 좌표계**로 그렸는데, 정작
알고 싶은 건 "물체가 **풀장** 어디에 있고 **기체**와 어떤 관계인가"였다. 지금은 그
질문을 trajectory 패널이 직접 답한다 — `3D` 버튼을 누르면 같은 플롯이 기울어진다.

* **같은 그림, 깊이만 추가**. 풀 경계·태그맵·궤적·참조 십자·DR 유령·물체 다이아몬드가
  전부 그대로 있고 각자의 **z**에 그려진다.
* **마커마다 태그면(z=0)까지 점선**이 내려간다. 직교투영에서 떠 있는 마커는 시선
  방향으로 모호하기 때문에, 이 발이 없으면 깊이를 읽을 수 없다.
* 드래그 = 궤도(2-D에서는 팬), 휠 = 줌, 더블클릭 = 리셋. 좌상단에 `az`/`el`이 뜬다.
* **elevation 90°에서 위에서 본 뷰와 픽셀 단위로 같다** — 버튼을 눌러도 세계가 움직이지
  않는다는 뜻이고, 테스트가 그걸 못박는다
  (`test_the_trajectory_3d_view_reduces_to_the_top_down_one`).
* 직교투영이다(원근 없음). 풀장 스케일에서 원근은 얻는 게 없고, 같은 거리가 위치에
  따라 다르게 보이게 만든다 — 조종사가 이 플롯에서 하는 판단이 바로 그 거리 판단이다.

**engage해도 마커의 깊이가 유지된다**(2026-08-23 수정). engage datum은 **z 오프셋을
가진 수평 등거리 변환**인데(`_datumize`는 `Rz @ (eta - p0)`), `_to_map`은 x/y만 되돌리고
z를 그냥 뒀다(필드 이름부터 `_z0`였다). 그래서 **뭔가 engage되는 순간** 선체·궤적·참조
십자·DR 유령이 전부 z ≈ 0, 즉 **태그 매트 위에 누운 채로** 그려졌다 — 물체만 진짜 map z를
유지했고, 칩의 `z -1.05 m`는 `_hold_z_text`가 그 한 곳에서만 손으로 오프셋을 되더해서
맞았다. 탑다운에선 안 보이고 `3D`를 켜면 로봇이 바닥에 붙어 있었다. 이제 `_to_map_z`가
**경계에서 한 번** 변환하므로 그 뒤의 모든 z는 map z이고, 그들 사이의 차(`dz`, 오차선)는
공통 오프셋이라 그대로 동작한다. `test_the_engaged_marker_keeps_its_map_depth_instead_of_the_tag_plane`.

**선체 마커는 언제나 한 곳에서만 쓴다** — `engaged`가 스위치다(2026-08-23 수정).

| 상태 | 마커·궤적을 쓰는 쪽 | z의 출처 |
|---|---|---|
| engage 안 됨 | `add_fix` — 태그 fix **원본** | 태그 PnP의 z |
| engage 됨 | `add_status` — 조립된 제어 상태 | **압력 depth + 세션 오프셋**, x/y는 fix age만큼 속도 보간, yaw는 자이로 보간 |

`MpcWorker._publish`는 상태만 있으면 engage 여부와 무관하게 `p_flu`를 채우므로, 예전에는
**engage 안 된 동안 두 쪽이 동시에** 같은 마커에 썼다 — 17 Hz와 20 Hz로, **서로 다른 양**을.
마커가 매 tick 두 값 사이를 오갔고 궤적은 양쪽에서 한 점씩 받았다. 3-D에서 `_basis3`의
화면-오른쪽 벡터는 z 성분이 0이므로 그 교대는 **완벽한 수직선**으로 그려지고, 90 초치가
쌓이면 빗살 커튼이 된다 — 2026-08-23 조종사가 "가만히 있는데 z가 왔다갔다한다"고 본 그림이
정확히 이것이다. 두 추정 중 틀린 것은 없었고, **둘을 한 계열로 그린 것이 틀렸다.**
`test_only_one_source_writes_the_vehicle_marker_at_a_time`이 고정한다.

### 녹화 영상에 오버레이가 들어간다

`TRACK`이 켜져 있으면 **C3 RGB 피드 녹화(`REC`)에 마스크·3-D 축·단계 줄이 그대로 구워진다.**
추적 세션을 녹화했는데 추적이 안 보이면 그건 바닥을 녹화한 것이다.

```
POSE TRACKING  13 Hz    solve 210 ms    reg 1
  (마스크 윤곽 + RGB 축)
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

설정은 `config/hw_nav.yaml`(태그/기하/extrinsic)과
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

* **미션은 패널 맨 윗줄에서 정한다**(2026-08-14, circle은 2026-08-17):
  `[station|line|square|circle|follow] from tag [id] [거리|반지름] [속도]`.
  **쓰지 않는 칸은 사라진다** — station은 거리/속도가, follow는 `from tag`와 거리가
  아예 안 보인다(follow는 태그 id를 **하나도 안 쓴다**: 기준이 arm 순간의 기체 pose와
  물체 pose다. 2026-08-23 전에는 회색으로 남아 "그래도 뭔가 하는 칸"처럼 보였다).
  **station** = 그 태그 위에 가서 **가만히 있기**
  (경로 없음, 방위까지 유지) — **이걸 먼저 날려라**. 점 하나를 못 잡는 기체는
  선도 못 따라가고, 그 상태로 line을 돌리면 나오는 숫자가 전부 그걸 재는 게
  된다. **line** = 그 태그 바로 위에서 시작해 `dir_deg`(기본 90° = 맵 +y)
  방향으로 입력 거리만큼 갔다가 **되돌아오는 왕복**(1 lap = 왕복 1회, 횟수는
  `hw_mpc.yaml: laps`). **square** = 그 태그를 첫 모서리로 하는 **직사각형**
  (x변 × y변을 따로 입력). **circle** = 그 태그를 **원의 맨 아래점**으로 하는
  원(거리 칸이 `r` 접두어와 함께 **반지름**이 된다; 1 lap = 1바퀴). 네 모양
  모두 태그는 경로가 **지나가는 점**이지 멀리서 도는 중심이 아니다 — 원의
  중심은 태그에서 맵 +x(플롯 위쪽)로 반지름만큼 떨어진 곳에 놓이고,
  `rot_deg`가 그 중심을 태그 둘레로 돌린다. 태그 id 0 = "지금 있는 자리에서".
  **속도(m/s)**는 line·square·circle의 목표 경로 속도이고 station에선 숨는다
  (2026-08-16 추가). 실제
  path mode는 기체를 현재 활성 곡선에 투영하고, 그 앞의 bounded lookahead를 PID와
  MPC에 공통으로 준다. 다각형에서는 입력 속도가 **직선 구간의 목표값**이고
  코너에서는 속도 프로파일이 미리 감속한다. **circle은 반대다**: 곡률이 랩
  전체에 걸려 있어 입력 속도는 기하가 허락하지 않을 수도 있는 **상한**일 뿐이다
  (바로 위 항목).
  이 필드들은 `cmd_mpc_scenario`로 hw_mpc.yaml의 `square:` 블록 위에
  덮어써지고(파일 값은 패널의 시작값), 랩 수·깊이·ramp는 파일에서 온다.
  **START를 누른 순간 미션이 스냅샷**되므로 필드는 주행 중은 물론 접근·정착
  단계에서도 잠긴다 — 그때 고친 값은 실제로 날아가지 않기 때문이다.
  * **방위는 `yaw_map_deg`**(기본 90 = 맵 +y를 봄), **line의 진행 방향
    `dir_deg`도 맵 절대 방위**(90 = 풀의 긴 방향 +y)다. 둘 다 datum 상대였다면
    engage 헤딩이 바뀔 때마다 다른 물리 방향을 뜻한다 — 실제로 2026-08-14에
    `dir_deg: 90`이 −x로 그려진 원인이 그거였다.
  * **지금 어느 단계인지 칩이 말한다**: `WARMING UP` → `GOING TO START 0.83 m
    to go` → `SETTLING 6 s` → `STATION HOLD` / `LINE` / `SQUARE`. 예전엔 접근
    중에도 `DP HOLD`라 구별이 안 됐다.
  * **접근은 설정값을 램프**한다(`engage.approach_speed_m_s` 0.10,
    `approach_lead_m` 0.35): 목표점을 2 m 순간이동시키면 위치 제어기에겐 그게
    풀-오소리티 스텝 명령이라 기체가 튀어나갔다가 출렁인다. 설정값이 기체
    앞 0.35 m를 넘지 않도록 하드 클램프도 건다.
  * **START는 멀어도 거부하지 않는다**(2026-08-14): DP로 그 태그 위까지 날아가
    `engage.settle_s`(기본 10 s)만큼 정지 대기한 뒤에 경로가 시작된다. 이동도
    같은 컨트롤러·같은 인터록이라 별도의 느슨한 이동 모드가 생기지 않는다.
    도착 판정은 `engage.start_err_max_m`(0.3 m)이고, 대기 중 그 밖으로 밀려나면
    타이머가 리셋된다. `approach_max_s`(180 s) 지나면 포기하고 제자리 유지.
    STOP/DISENG은 진행 중인 접근도 취소한다. `settle_s: 0`이면 예전처럼 즉시
    시작(원점 위에 있을 때).
  * **중복 id는 원점으로 못 쓴다**(두 곳에 있으니 "그 태그 위"가 모호). 거부한다.
  * line의 반환점에는 **코사인 속도 램프**(`ramp_s`, 기본 1 s)가 들어간다.
    삼각파로 1샘플 만에 속도를 뒤집으면 square 코너와 같은 이유로 추종오차에
    바닥이 생긴다(memory: square-corner-error-floor) — 참조가 실현 가능해야
    reference-vs-actual이 컨트롤러를 재는 숫자가 된다.
  * **지오펜스는 없다**(2026-08-14 제거, 조종사 요청). 예전에는 START 전에
    line 양 끝점 / 사각형 네 모서리를 상자에 대고 검사하고 벗어나면 거부했다.
    지금은 **경로 배치를 막는 것이 아무것도 없다** — 풀 밖으로 나가는 9 m line도
    그대로 arm된다. 남은 보호는 E-STOP·DISENG·sink deadman(500 ms)·engage 게이트
    (ARMED / MANUAL / 신선한 태그 fix·telemetry)·주행 중 disarm·telemetry 정지·
    모드 이탈·태그 상실 자동 해제, 그리고 축 권한 상한뿐이다.
* **START(hold)가 원버튼 플로우**다: engage → 워밍업 → square 자동 시작, CSV
  기록은 engage 순간부터. 전제조건은 그대로 전부 검사된다: COMMAND ENABLE·ARMED·
  비행모드 MANUAL·acados 실시간(프로브 <25 ms)·태그 fix 신선(<0.5 s)·IMU 신선.
  하나라도 빠지면 사유를 말하고 거부. (지오펜스 검사는 2026-08-14에 빠졌다.)
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
  서베이 포함) 런 폴더 안 `nav_<hhmmss>/`에 네 파일을 남긴다: `map.json`
  (잠긴 태그맵 id→x,y,z,yaw + 태그 크기 + pool + 지오펜스), `fixes.csv`
  (**태그 프레임 원시 fix** — datum 변환 전, 실패 프레임도 **태그 id와** 사유를
  함께 기록), 그리고 `detections.csv` + `frames.csv`(**원시 관측** — 검출된
  모든 태그의 코너 4점과 그 프레임의 카메라 모델). 뒤의 둘이 맵을 다시
  만들거나 넓히는 재료다 — 풀린 fix만 남기면 코너가 버려져서 아무것도 못 한다.
  `python -m rov_gui.tools.plot_nav_run <런폴더>/nav_<hhmmss>`로 같은
  그림(태그 사각형 + 시간색 경로 + 풀 경계, matplotlib)을 다시 그린다.
* **중복 태그 id**(2026-08-14 실측): 풀 매트는 20×7=140칸 커스텀 배치이고
  **12개 id가 두 곳에 있다**(`hw_nav.yaml: duplicate_ids`). 맵은 id당 자세
  하나뿐이라 "다른 사본"의 코너가 섞이면 joint PnP가 터진다(실측 59~131 px).
  대응은 **confirm-or-drop**: 유일 태그로 자세를 먼저 확정하고, 중복 태그는
  맵 자리에서 `dup_confirm_px`(6 px) 안에 떨어질 때만 채택한다. 다른 사본의
  **위치를 몰라도 되는 게 핵심**(알아보기만 하면 됨)이라 재측량이 필요 없다.
  한 프레임에 같은 id가 두 번 잡히면 둘 다 폐기 — 신고 안 된 중복이 스스로
  드러나고, `min_tags`가 유일 id만 세게 된다.
* **플롯은 항상 MAP(태그 월드) 프레임**이다(2026-08-14): 매트는 늘 축에
  정렬돼 있고 풀 사각형도 매 실행 같은 자리다. 컨트롤러는 ENGAGE-datum
  프레임(START pose가 원점, 시작 헤딩이 +x)에서 돌기 때문에, MpcStatus로
  들어오는 위치를 `view.set_datum()`이 다시 맵으로 돌려놓는다. **그 전에는
  변환이 반대여서 START를 누르는 순간 매트 전체가 홱 돌아갔다.**
* **플롯의 ROV는 실제 선체 크기**로 그린다(`hw_nav.yaml: rov_footprint_m`,
  기본 0.4318 × 0.5334 m = **17 × 21 inch**, heading 방향이 17 inch
  [측정: operator 2026-08-14]). 점이 아니라 사각형이라, 벽이나 태그를
  스치는지가 눈에 보인다. 중앙의 작은 점은 축소했을 때를 위한 것.
* **운용 깊이 범위는 실측 −1.30 ~ −0.54 m**(`sessions/nav_runs/*/fixes.csv`,
  채택된 fix의 z_ned). 태그 프레임 z라서 바닥 태그가 z=0이고 +z가 아래 —
  매트 위를 헤엄치는 기체는 **음수**다. 예전에는 지오펜스 z가 이 범위를 담는지
  테스트가 고정했지만 펜스가 사라졌으므로, 이제 이 숫자는 **아무것도 강제하지
  않는 기록**이다. `test_measured_operating_depth_band_is_still_on_record`가
  부호만 지킨다(양수면 기체가 바닥 아래라는 뜻 = 맵이나 규약이 깨진 것).
* **거부 사유는 패널에 뜬다**: 이전에는 로그 줄에만 나가서 chip이 계속
  `engaged (DP hold)`라, 조종사는 START가 씹힌 줄도 몰랐다. 이제
  `START refused: …`로 뜬다.
* **지오펜스는 그림도 규칙도 전부 사라졌다**(2026-08-14, 조종사 요청).
  engage하면 뜨던 주황 점선 `GEOFENCE` 상자, START 전 경로 검사, 주행 중
  이탈 자동 해제 셋 다 제거했다. `hw_nav.yaml`의 `geofence_ned` /
  `geofence_frame` 키는 **읽지 않는다**(남아 있어도 오류가 아니라 무시 —
  옛 설정 파일이 그대로 뜨게 하려고). MPC CSV의 `geofence_ok` 열도 빠졌으므로
  **2026-08-14 이전 CSV에는 그 열이 있다**: 위치가 아니라 이름으로 읽을 것.
  플롯에 남은 사각형은 `POOL` 하나뿐이고, 그건 벽의 **그림**이지 한계가 아니다.
* **플롯의 글자는 최소한만**(2026-08-14): 남는 건 `off … lag … lap N`과 dobmpc의
  `w_hat`뿐. 위치·측위 상태·솔버 시간·태그 수·거부 사유는 각각 SENSORS의
  TagNav 줄, 플롯 위 칩, PAYLOAD 아래 **MISSION LOG**로 옮겼다.
* **오차는 `off`(횡)와 `lag`(종)로 쪼개서 보여준다**(2026-08-14). path
  following의 stage 0은 현재 선분 투영점보다 `path_lead_m`만큼 앞에 있으므로,
  합친 `err`만 보면 경로 위에 정확히 올라가 있어도 실패처럼 읽힌다 — 실제로
  20:18 MPC line 런의 `err 12.5 cm`는 **lag 11 cm + off 1 cm**였다
  (`sessions/low_level_controller_data/20260814/2018/mpc_201843.csv`).
  부호 규약: `lag` +는 **늦은 것**, `off` +는 **진행방향 왼쪽**. CSV에도
  `e_along` / `e_cross` 두 열이 붙었다(그 이전 런에는 없음).
* **circle에는 실현 불가능한 구간이 없다**(2026-08-17). 90° 꼭짓점도
  (memory: square-corner-error-floor) 180° 반전도 없어서 곡률이 일정하다.
  대신 곡률 한계 `v <= sqrt(path_lat_accel_m_s2 * R)`가 **랩 전체에** 걸린다 —
  다각형에서는 코너에서만 물리던 제한이다. 배포값 `path_lat_accel_m_s2: 0.05`
  에서 r 0.5 m는 0.158 m/s, r 0.3 m는 0.122 m/s가 상한이라 속도 칸에 그보다
  큰 값을 넣어도 그대로 날지 않는다 — **[유도]** `v = sqrt(a_lat·R)`이고
  `rov_gui/tests/test_mpcc.py ::
  test_circle_speed_is_curvature_limited_over_the_whole_lap`가 고정한다.
  물에서 잰 값이 아니다(원은 아직 한 번도 안 날렸다). 그럴 때 워커가 로그에
  경고를 찍는다(`curvature caps this path at ... m/s`) — 조종사가 입력한
  숫자를 기하가 조용히 무시하는 상태를 만들지 않기 위해서다.
* **square는 MAP 프레임에 놓인다**(2026-08-14). `rot_deg 0`이면 변이 태그맵의
  x·y 축과 나란하고, 입력한 태그가 **min-x / min-y 모서리**(플롯 좌하단)이며
  첫 변은 거기서 +x로 나간다. 이전에는 datum 프레임에 놓여서 START 때 기체가
  향하던 각도만큼 사각형이 통째로 기울어졌다 — line의 `dir_deg`와 같은 종류의
  버그. 태그가 좌하단이 되는 건 `mirror_y`(FLU→NED 미러가 y를 뒤집어 태그를
  y가 **가장 큰** 모서리로 보내던 것을 되돌린다) 덕이다.
* **태그 번호·거리는 클릭해서 타이핑**할 수 있다(2026-08-14). 이 패널의 세 필드만
  `ClickFocus`이고 나머지는 전부 `NoFocus`인데(창이 유일한 키 핸들러라서),
  숫자가 아닌 키가 오면 **즉시 포커스를 놓고 그 키를 창으로 다시 보낸다** —
  태그 번호를 치고 곧바로 W를 눌러도 기체가 움직인다. 이게 없으면 필드가 W를
  삼켜서 "아무 이유 없이 조종이 안 되는" 상태가 된다. Enter는 확정+포커스 해제.
* **MISSION LOG**(PAYLOAD 패널 CAMERA TILT 아래): 미션 사건과 **모든 로그 줄**을
  **벽시계 시각**과 함께 남긴다(`18:29:06  going to tag 79 (1.24 m)`). 나중에
  영상·메모와 맞춰 보라고 monotonic이 아니라 벽시계를 쓴다.
  * **스크롤된다**(2026-08-14, 조종사 요청 "예전 것도 볼 수 있게"). 예전에는
    QLabel이 마지막 12줄만 들고 나머지를 버려서, 2분 전 거부 사유 — 런이
    틀어졌을 때 가장 보고 싶은 그 줄 — 가 그냥 사라졌다. 지금은 세션 전체를
    (최대 5000줄) 들고 있고, **맨 아래에 있을 때만** 새 줄을 따라간다:
    거슬러 올라가 읽는 중이면 20 Hz 상태 줄이 시야를 끌어내리지 않는다.
  * "컨트롤을 스크롤 뒤에 숨기지 않는다"는 규칙의 **유일한 예외**다. 로그는
    컨트롤이 아니고, 누를 것이 없으며, 지금 중요한 최신 줄은 항상 보인다.
  * **등급이 무게를 진다**(2026-08-23). 예전에는 전부 같은 흐린 회색이라
    `ENGAGED (mpc)`와 `pose: tracking`과 CUDA 경고가 활자상 똑같았고, 런이
    시작됐는지 찾으려면 패널 전체를 읽어야 했다.

    | 등급 | 표시 |
    |---|---|
    | `mission` | **굵게 + 액센트색 + 위에 가로줄** — 런 자체의 생애(ENGAGED / STATION / FOLLOW / DISENGAGED / 거부) |
    | `error` / `warn` | 빨강 / amber |
    | `info` | 원래의 흐린 회색 |
    | `debug` | **패널에 안 뜬다** — stdout 전용 |

  * **연속으로 같은 줄은 세어서 한 줄**이다: `16:34:21 (x4)  [WARN] mpc: engage
    refused: ...`. 개수를 시각 바로 뒤에 붙이는 건 이 패널이 줄바꿈을 안 하기
    때문이다 — 끝에 붙이면 긴 줄에서는 화면 밖으로 나가 안 보인다.
    **연속일 때만** 합쳐지므로, 사이에 다른 일이 있었으면 새 사건으로 남는다.
  * **한 사건에 한 줄.** 예전에는 engage/disengage가 미션 이벤트 한 줄 + `[WARN]`
    로그 한 줄로 **두 번** 찍혔다. 이제 미션 줄이 홀드 지점까지 싣고, 로그 사본은
    `debug`로 내려가 stdout에만 남는다. 같은 이유로 BundleSDF의 메시 점검 3줄은
    **한 줄로 합쳐지고**(`pose: mesh check — ... | ...`), 트래커 단계 줄은 note가
    단계 단어를 되풀이하면 그 부분을 잘라낸다(`pose: lost the object —
    lost — reacquiring` → `pose: lost the object — reacquiring`).
  * **파일로도 저장된다**: 녹화(REC UI / REC NAV)가 멈추거나 창이 닫힐 때
    그 런 폴더에 `mission_log.txt`로 세션 전체가 쓰인다. 멈춘 시점이 서로 달라도
    항상 **그 시점까지 전부**를 다시 쓰므로 조각으로 나뉘지 않는다.
  * 미션 사건은 `events.log`에도 들어가므로 화면과 파일이 다른 이야기를 못 한다.
  * **2026-08-14: 두 배 가까이 커졌다**(조종사 요청 "공간 좀 줄이고 로그 더 크게").
    GRIPPER / LIGHTS / CAMERA TILT 세 블록을 각각 5행에서 3행으로 줄여서 만든
    공간이다 — 상태 노트("no position feedback", "draw 0.42 A")는 자기 줄을
    버리고 캡션 줄 오른쪽으로 올라가 elide되고, 조명 슬라이더는 OFF/¼/½/MAX와
    한 행을 쓰며, 버튼 높이는 26→22 px다. **지운 건 없다.** 창 최소 높이는
    그대로라(994 px, --mpc 예산 1000) 레이아웃 테스트도 그대로다. 1920×1080에서
    로그 높이 109 → **192 px**. 더 키우려면 이 패널 안에서 또 뺏어와야 한다.
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
  <런폴더>/nav_<hhmmss> --anchor config/tag_map.yaml -o
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
  >0.5 s·솔버 연속 3실패는 전부 자동 disengage. **위치는 더 이상 아니다** —
  지오펜스 이탈 해제는 2026-08-14에 제거됐다. MpcWorker가 죽어도
  sink deadman(500 ms)이 중립을 보내고, 창의 mpc 워치독(1.5 s)이 스스로 조종권을
  회수한다 — 조종사가 죽었을 때와 같은 마지막 방어선.
* **circle을 wall 기하에서 돌릴 때는 태그가 시야에서 나갈 수 있다** [예측,
  미검증]. 크랩이라 기수는 안 돌아가지만 **위치가** 태그 기준으로 range 0..2R,
  lateral ±R만큼 움직인다(배포 r 0.5 m면 1.0 m와 ±0.5 m). 벽 태그 하나로
  버티는 기하에서는 이게 코너보다 더 센 요구다 — 태그를 놓치면
  `engage.tag_stale_hold_s` 디바운스를 거쳐 랩 도중에 자동 disengage 된다.
  floor 기하(태그 여러 개)에서 먼저 날려 볼 것.
* wall 기하에서는 `heading_follow: false`(크랩 square)가 **필수**다. 코너에서
  기수를 돌리면 벽 태그가 시야에서 사라진다.

기록: engage하면 CSV가 무조건 열린다(`…/mpc_<hhmmss>.csv` — 아래 "런 폴더" 참조,
또는 진행 중인 녹화 스템에 `_mpc.csv`로 편승). 앞 9열은 sim `runs/traj_*.csv`와 동일 스키마
(`t,px,py,pz,rx,ry,yaw_deg,pitch_deg,lap`, world FLU)라 기존 분석 도구가 그대로
읽고, 뒤로 `w_hat×6·u×6·솔버 status/solve_ms·태그 수·reproj px·축 명령` 등이
붙는다. `.meta.json` 사이드카가 태그맵 sha1, extrinsic, 게인 출처까지 기록한다.

**물성치와 상수도 같이 남는다 (2026-08-14, 조종사 요청).** CSV는 기체가 *무엇을
했는지*만 말하고, 컨트롤러가 기체를 *무엇이라고 믿었는지*는 말하지 않는다. 그래서
`meta.json`에 `plant` 블록이, 그리고 **REC를 누르면 그 녹화 폴더에
`controller.json`**이 같이 떨어진다(`rov_gui/control/plant.py` +
`MpcWorker.dump_run_meta`).

* `plant` — `M nu_dot + C(nu) nu + D(nu) nu + g(eta) = tau + w` (NED/FRD)의
  전부다: 질량·관성·부가질량과 **6×6 M 행렬**, C를 만드는 계수와 그 런의 nu에서
  평가한 `C_rb_at_nu`/`C_a_at_nu`, D의 선형·2차 계수와 `D_at_nu`, `g_at_eta`,
  `u_max`. 출처는 복사본이 아니라 컨트롤러가 실제로 import하는
  `dobmpc/params.py`+`fossen.py` 그 자체다.
  * **C와 D는 상태 의존**이므로 행렬은 *샘플*이고, 어떤 상태에서 뽑았는지가
    `evaluated_at`에 같이 적힌다. 상수인 척하지 않는다.
  * 기록된 생성 행렬은 쓰기 직전에 `fossen`의 곱과 대조되고 그 잔차가
    `check_max_abs_err`로 남는다(현재 0.0). 기록이 모델에서 조용히 어긋날 수 없다.
* `controller` — PID면 kp/kd/ki에 **더해** `omega_derate`, i_max, e_gate,
  yaw_gate, f_max, mz_max, slew까지. MPC/DOB-MPC면 N·horizon·**Q/QN/R**·u_max·
  w_hat_clip과 EAOB 튜닝(tau_dist, 시그마 전부). 게이트나 slew 하나가 궤적을
  kp만큼 바꾸므로 "내가 조정하는 것들"은 전부 들어간다.
* `mission` — 패널이 요청하던 shape/tag/거리. `trajectory`(armed scenario)는
  START 전에는 null이라, REC를 먼저 눌렀을 때 미션이 통째로 비는 걸 막는다.
* engage하지 않아도 쓴다. MPC를 한 번도 걸지 않은 손비행 서베이 패스도 "어떤
  컨트롤러가 선택돼 있었고 기체 모델이 무엇이었나"는 남아야 한다.

검증 상태 (2026-08-12): `rov_gui/control/smoke.py` — acados가 rovgui-pose(numpy 2)
에서 SQP-RTI 빌드 1.3 s, EAOB+solve p50 3.1 ms/p99 6.4 ms
(`sessions/` 아님, 콘솔 출력); demo 폐루프로 square 1랩 완주(솔버 실패 0).
**실기·수중은 아직 미검증** — KNOWN_ISSUES.md 참조. 수조 절차는 P3(측위만 기록)
→ P4(DP + 축게인 스텝 캘리브레이션) → P5(square, mpc) → P6(dobmpc A/B) 순서를
지킬 것. **circle은 P5 뒤에** — 크랩 원은 랩 내내 sway를 요구해서, 아직
캘리브 안 된 축게인을 square보다 세게 흔든다 [예측].

### 지금 어느 깊이를 잡고 있는지 패널이 말한다 (2026-08-18)

station 모드는 **원래부터 z를 잡고 있었다** — `_arm_path`가
`set_target_ned((x, y, depth), yaw)`로 3축 전부 걸고, `depth`는
`hw_mpc.yaml: square.depth_ned`가 null이면 **START를 누른 그 순간의 z**다
(`workers.py:989`). 문제는 **그 숫자가 화면 어디에도 없었다**는 것이다:
`err_xy`는 정의상 **수평**이라, 20 cm 가라앉은 채 떠 있는 hold가 완벽한 hold로
읽혔다. 그래서 MPC 패널 chip과 플롯 오차 줄에 깊이를 넣었다.

```
ENGAGED [pid] STATION HOLD · err 1 cm · 0.01 m/s · z -1.05 m (+3 cm)
ENGAGED [mpc_tuned] SQUARE · off -2 cm · lag +10 cm · 0.05/0.10 m/s · z -1.05 m (+3 cm)
플롯:  off  -2.3 cm   lag +10.5 cm   dz  +3.0 cm   lap 1
```

* `z`는 **MAP 프레임**이다(engage datum의 z를 되돌려 더한다). 실측 운용 밴드
  −1.30 ~ −0.54 m와 바로 비교된다. datum이 없으면 `z_d`로 이름을 바꿔 **datum
  상대값임을 밝힌다** — 조용히 틀린 숫자를 map인 척 내놓지 않는다.
* NED z가 **아래가 +**라, 바닥 태그 위를 헤엄치는 기체는 **음수**이고
  괄호의 `(+3 cm)`는 setpoint보다 **3 cm 더 깊다**는 뜻이다. chip의 툴팁이 이
  규약을 적어 둔다(부호는 안 적어두면 동전 던지기다).
* 깊이를 **숫자로 못 박고** 싶으면 `hw_mpc.yaml: square.depth_ned`에 값을 넣는다.
  비워 두면 지금처럼 매 START의 z를 쓴다(조종사 결정 2026-08-18).
* `rov_gui/tests/test_offline.py :: test_the_panel_says_what_depth_is_being_held`
  이 두 규약(map 프레임 · 아래가 +)을 고정한다.

### 코너 컷을 억제하는 비용함수 — `mpc_tuned` / `dobmpc_tuned` (2026-08-18)

**문제**: tracking NMPC는 위치 오차를 **월드 프레임 대각 Q**로 벌한다
(`params.MPC_Q`의 Q[0]=Q[1]=300). 등방이라는 뜻이고, 그래서 경로에서 **벗어난**
1 cm와 경로 위에서 **뒤처진** 1 cm의 값이 같다. 코너에서 최적화기는 싼 쪽을 팔아
비싼 쪽을 산다 — 즉 **코너를 잘라 먹는다**. 그건 그 비용함수에 대해 최적 행동이지
버그가 아니다(memory: square-corner-error-floor).

**모드**: `_tuned` 접미사가 붙은 두 모드는 같은 2×2 블록을 매 stage **경로
프레임으로 회전**시킨다.

```
cost = q_along * (t̂·e)²  +  q_cross * (n̂·e)²
     = eᵀ W e,      W = Rz(ψ_path) diag(q_along, q_cross) Rz(ψ_path)ᵀ
```

회전은 직교변환이라 `along_scale == cross_scale`이면 **등방 대각과 기계정밀도까지
동일**하다 — 그게 A/B를 신뢰할 수 있게 만드는 parity이고
`rov_gui/tests/test_path_cost.py`가 대수·폐루프 양쪽에서 고정한다. acados는 이미
생성된 `LINEAR_LS` OCP에 `cost_set(k, "W", W_k)`를 런타임으로 받으므로
**`mpc`와 `mpc_tuned`는 같은 컴파일 솔버**다 — 두 번째 codegen도, 빌드 대기도 없다.

* 회전각은 **경로의 접선**(`NedPlan.psi_path`)이지 기체 헤딩이 아니다.
  `heading_follow: false`면 기체는 크랩하므로 둘은 무관하고, 참조 **속도**로
  대신할 수도 없다 — 이 기능이 존재하는 이유인 그 코너에서 `speed_profile`이
  속도를 creep까지 깎아 버리기 때문이다.
* **경로 plan이 없으면 자동으로 등방으로 돌아간다.** DP hold·접근·10 s settle에는
  경로가 없고, "이 점을 지키되 한 방향으로만"은 station이 약속한 것이 아니다.
* 모드를 되돌리면 회전된 W도 되돌아간다(`_restore_base_weights`). 남아 있으면
  **다음 baseline 런이 조용히 오염**된다 — solver status는 어느 쪽이든 0이다.
* 튜닝은 `config/hw_mpc.yaml`의 `mpc_tuned:` 블록이고 **오타는 startup에서
  raise**한다(`imu_dr`와 같은 규칙: 조용히 무시되면 meta엔 tuned라고 적힌 채
  baseline 비용으로 날게 된다). run meta에 `cost_frame`과 `path_cost`가 남고,
  그 두 필드가 tuned/baseline 기록의 **경계**다 — tuned 런의 `Q` 행은 split을
  유도한 등방 baseline이지 솔버가 실제로 쓴 값이 아니므로 **합산 금지**.

**오프라인 스윕** (한 번 빌드하고 격자 전체를 돈다):

```bash
python -m rov_gui.tools.sweep_path_cost --fillet 0.15 --speed 0.10
python -m rov_gui.tools.sweep_path_cost --along 0.25 1 --cross 1 4 16 --plot figures/x.png
```

플랜트가 **컨트롤러 자신의 예측모델**이라 데드밴드도 테더도 없고 수평 항력이
8~12배 부족하다(`hw_mpc.yaml`의 whole-loop fit). 기하는 정직하게 순위를 매기지만
스러스터가 결정하는 것은 하나도 못 매긴다 — 출력은 전부 **[예측]**이다.

| 미션 | baseline cut | 1/16 tuned | p95 cross | lap 시간 |
|---|---|---|---|---|
| fillet 0.15, 0.10 m/s | 4.8 mm | **1.2 mm (−74 %)** | 4.4 → 1.0 mm | 41.1 → 41.0 s |
| fillet 0.0 (waypoint) | 3.6 mm | 3.1 mm (−15 %) | 1.6 → 1.0 mm | 45.7 → 45.7 s |

두 가지가 이 표에서 바로 읽힌다 [예측, 오프라인]:
1. **레버는 `cross_scale` 하나**다. `along_scale`을 내리는 건 코너 컷을 거의 못
   줄이고(고정 cross에서 0.1↔1.0 차이 <0.1 mm), 랩 시간과 lag만 늘린다.
   waypoint 미션에서는 **오히려 나쁘다**(0.25/1에서 +57 %).
2. **필렛이 있어야 효과가 난다.** fillet 0에서 남는 오차는 최적화기가 고른 컷이
   아니라 **참조 자체가 실현 불가능**해서 생기는 바닥이라, 가중치로는 못 옮긴다.

**실기 미검증** — 아직 물에서 한 번도 안 날렸다. KNOWN_ISSUES.md 참조.

### 접근이 느렸던 이유는 추력이 아니라 leash였다 (2026-08-18)

**실측** (2026-08-18, 접근 4회): 접근 중 `|axis|` p50 **0.18**, cap 0.50에 닿은 tick
**0.0 %**. 권한을 3분의 1도 안 쓰고 있었다. 남은 거리별 속도는

| 남은 거리 | 속도 | \|axis\| |
|---|---|---|
| 0.5–1.0 m | 0.102 m/s | 0.18 |
| 0.2–0.5 m | 0.093 m/s | 0.17 |
| 0.05–0.2 m | **0.037 m/s** | **0.066** |
| 0–0.05 m | 0.015 m/s | 0.12 |

**속도를 정한 건 `approach_speed_m_s`가 아니라 leash였다.** `_tick_approach`가
`set_target_ned`를 **속도 피드포워드 없이** 부르고 있었기 때문에(path follower는
늘 넘겨왔다) 정상상태가 이렇게 잡힌다:

```
kp·lead = 51.7 × 0.35 = 18.1 N,   실측 전루프 F = 86.7·v + 5.76
  → v = 0.084 m/s      [유도]   ← 실측 0.079~0.102 m/s
```

그래서 **셋을 같이** 고쳤다. 하나만 바꾸면 2026-08-17에 런 하나를 날린 그 실수
("루프에 안 들어간 속도 숫자")를 반복하게 된다.

* `_tick_approach`가 setpoint 자신의 속도를 `v_ned`로 넘긴다. 단
  **컨트롤러의 선견거리로 테이퍼**한다 — NMPC는 참조를 `v_ref`로 호라이즌 끝까지
  외삽하므로, 목표 한 호라이즌 안에서 전속 FF는 "지나쳐 가라"는 뜻이 된다.
  PID(1 stage, 0.05 s)는 테이퍼에 영향받지 않아 둘의 비교가 유지된다.
* `approach_speed_m_s: 0.10 → 0.20`, `approach_lead_m: 0.35 → 0.50`.
  같은 식으로 leash 상한이 0.232 m/s가 되어 0.20이 실현 가능해진다. 최대 명령은
  kp×0.50 = 25.9 N = axis 0.43으로 여전히 cap 안이다.
* **station은 settle을 건너뛴다**(`settle_station_s: 0.0`). 직관과 반대인데,
  settle 동안 setpoint는 **아직 leash에 묶여 있고** arming이 그 leash를 푼다 —
  즉 settle은 도착을 돕는 게 아니라 **느린 쪽 상태로 10 s 더 붙잡아 둔다.** 게다가
  station은 settle 뒤에 시작할 것도 없다(armed 타깃 = 접근이 걸어가던 그 점).
  path 미션은 그대로 유지한다(사각형이 30 cm 벗어난 채 시작하면 안 되니까).
  `imu_dr`가 켜져 있으면 static window가 필요하므로 **자동으로 원래 settle이 돌아온다.**

예상 효과 [예측 — 위 실측 관계식에서 유도, 물에서 재확인 필요]:

| 이동 | 전 (실측 이동 + settle) | 후 |
|---|---|---|
| 0.4 m | 13 s + 10 s | ~2 s + 0 |
| 1.3 m | 22 s + 10 s | ~7 s + 0 |

**마지막 20 cm는 여전히 느리다** — 거기서 `|axis|`가 데드밴드 문턱 0.096 아래로
떨어지는 건 게인 문제이고, 이번 변경은 거기를 건드리지 않았다(조종사 결정).
station에서는 settle을 건너뛰면서 leash가 일찍 풀리므로 그 구간도 조금 빨라진다.

### 태그를 놓쳐도 안 꺼진다 — STATION BRIDGE (2026-08-18)

**증상**: 외란 한 방에 카메라가 태그를 놓치면 약 1.1 s 뒤(`tag_stale_s` 0.7 +
`tag_stale_hold_s` 0.4) 루프가 **disengage**한다. 들리는 것보다 나쁘다 — 같이
**깊이 유지도 꺼지고**, 이 기체는 순부력 −5.7 N이라 가라앉기 시작한다. 자세와
시야가 바뀌니 재획득 확률은 오히려 **떨어진다**. 그리고 조종사는 다시 ARM해야 한다.

**`imu_dr`를 켜도 안 풀린다**, 그게 핵심이다. `MpcWorker._runtime_fault`가 먹는
건 `meas`(태그 해, 일부러 ground truth로 남겨둔 것)이지 컨트롤러가 실제로 먹는
`meas_ctrl`이 아니다. 그래서 추정기가 뭘 하든 인터록은 그대로 발동한다. 분리해야
하는 건 "태그 vs IMU"가 아니라 **"추정이 끊겼다"와 "제어를 포기한다"**이다.

**사다리** (`rov_gui/control/station_bridge.py`, **station 모드 전용**). 어떤 축을
얼마나 들고 갈 수 있는지는 하나의 질문이 아니라 넷이다 — 뒤에 있는 센서가 다르니까:

| 축 | 태그 없을 때 출처 | 드리프트 |
|---|---|---|
| z | 압력계 | **없음** (절대) |
| roll/pitch | ArduSub AHRS | **없음** (절대) |
| yaw | 자이로 적분 | 느림 (0.02 rad/s = 70 °/min) |
| x, y | 가속도 이중적분 | t², 빠름 |

```
fix 신선           평소 그대로
0 ~ imu_hold_s     전 축을 bridge 추정치로, 권한은 평소와 동일
imu_hold_s 이후    x/y/yaw 해제, 깊이+자세는 계속 — 무제한
```

마지막 단계는 **일부러 disengage가 아니다**. 드리프팅 추정치로 수평을 밀면 기체를
적극적으로 엉뚱한 곳으로 몰고 가는데, 위치를 이유로 멈추는 수단은 이제 하나도
없다(지오펜스 2026-08-14 제거). 손을 떼면 물살에 흘러갈 뿐이고 풀 속도에서 그건
느리다. 그동안 **드리프트가 없는 두 축**이 기체를 깊이와 수평에 붙잡아 두는데,
그게 태그를 다시 잡을 확률이 가장 높은 상태다. 그래서 무제한으로 둔다.
coast에서 실제로 나가는 축은 **heave 하나뿐**이다(K/M은 MANUAL_CONTROL에 축이
없어 어차피 버려진다) — `test_the_coast_tier_sends_heave_and_nothing_else`가
allocation·cap·slew를 다 통과한 **전선 위 값**으로 고정한다.

* **x/y 출처는 캘리브에 달려 있다**(`xy_source: auto`). 캘리브 파일이 있으면
  가속도계를 적분하고, 없으면 **마지막 fix에 x/y를 고정**한다. RAW C3는 실측
  1.8 m/s² 바이어스라 3 s면 8 m다 — auto는 편의 기능이 아니라 안전장치다.
  고정 경로는 **C3 IMU가 아예 없어도 동작**한다(압력계+오토파일럿만 쓴다).
* **다른 인터록은 전부 살아 있다**: disarm, 비행모드 이탈, telemetry 상실, 솔버
  실패, tick 초과, E-STOP. 그리고 bridge가 **타고 갈 센서 자신이 죽은** 두 결함
  (`imu stale`, `pressure depth stale`)과 `no vehicle imu`는 화이트리스트에
  없어서 그대로 disengage한다. 화이트리스트는 접두어가 아니라 **단어 경계**로
  맞춘다 — `startswith`만 쓰면 새 결함 문자열이 조용히 상속된다.
* **패널이 빨갛게 말한다**: `ENGAGED [pid] NO TAG — COASTING`. coast는 무제한이라
  "언제 그만둘지"를 판단하는 건 사람이고, 그러려면 화면이 색으로 말해야 한다.
* **복구할 때마다 공짜 측정치가 나온다**: 태그가 돌아온 순간 추정치가 얼마나
  벌어져 있었는지를 mission log에 남긴다(`fix back after 1.35s (tier imu, IMU
  was 4.2 cm off)`). [예측]으로만 있던 IMU 예산을 [측정]으로 바꾸는 숫자다.
* **기록 경계**: CSV에 `bridge_s` / `bridge_tier` 두 열이 붙고 meta schema가
  **5**가 됐다. `bridge_s > 0`인 행은 **태그로 난 게 아니다** — px/py가 IMU
  추정치(또는 고정된 앵커)이고, coast 행은 컨트롤러가 뭘 요구했든 surge/sway/yaw가
  0으로 나갔다. 깨끗한 tick과 **합산 금지**.

설정은 `config/hw_mpc.yaml`의 `station_bridge:`(오타는 startup에서 raise).
**실기 미검증** — KNOWN_ISSUES.md 참조.

## 물체를 따라간다 — follow (2026-08-21)

`--pose --mpc`를 **함께** 켜면 클릭한 물체가 풀장 좌표로 올라오고, 미션 모양
`follow`로 그 물체에 대한 **상대 자세를 계속 유지**한다.
`rov_gui/control/object_nav.py`.

### 왜 없었나

스테이션에는 서로 모르는 6-DoF 추정기가 둘 돌고 있었다. `--mpc`의
`TagNavWorker`는 **기체**의 맵 좌표(`bus.nav_fix`)를 내고, `--pose`의 `PoseWorker`는
**물체**의 **카메라 상대** 자세(`bus.pose`의 `T_cam_obj`)를 낸다. 둘을 잇는 코드가
하나도 없어서 pose 로그가 그걸 그대로 못박아 두었다 — *"NO camera->body transform
is applied; the vehicle's own pose is not in this file."* 그래서 물체가 풀장 어디에
있는지는 아무도 몰랐고, 물체를 기준으로 뭘 하는 것도 불가능했다.

### 프레임 체인, 그리고 왜 extrinsic이 소거되는가

```
T_cam_obj (16 floats, row-major, object->camera, OpenCV optical, m)
  R_cam_obj = M[0:3,0:3],  t_cam_obj = (T[3], T[7], T[11])

R_map_cam = R_ned_body · R_frd_cam        <- extrinsic 없음
p_map_cam = p_ned + R_ned_body · t_frd_cam
R_map_obj = R_map_cam · R_cam_obj
p_map_obj = p_map_cam + R_map_cam · t_cam_obj
```

`TagNav._solution`은 PnP 카메라 pose에서 extrinsic을 **나눠서** body pose를 만들고,
이쪽은 그걸 **다시 곱해** 넣는다. **같은 카메라 프레임**이면 이 둘은 정확한 역연산이라
extrinsic이 통째로 소거된다.

이게 중요한 이유: `cam_tilt_deg: 43.3`은 **순수 회전으로만** 적용되고, 실제 힌지가
같이 움직이는 병진 성분(`cam_t_flu`, lever arm 0.2855 m
[측정: `bluerov2_mujoco_marinegym/meshes/c3_payload_frames.json`, Onshape 2026-07-19])은
**아직 미실측**이다(KNOWN_ISSUES 2026-08-17). 같은 프레임 짝맞춤이 그 미지수를 무효화한다.

프레임이 어긋나면 소거가 깨지고 잔차는 `|t_frd_cam| · 2·sin(θ/2)` — 프레임 간 5° yaw면
2.5 cm. 그래서 **`pair_exact`가 상설 슬롯**을 갖는다(패널 모서리 readout, SENSORS의
`Object` 행, CSV의 `obj_pair_exact`). 이게 1이 아니라는 건 소거가 멈췄다는 **유일한**
경고다.

**합성에는 raw `NavFix`만 쓴다. `meas["eta"]`는 절대 안 된다.** `StateAssembler`가
돌려주는 건 하이브리드다 — z는 압력계, roll/pitch는 ArduSub AHRS, x/y는 프레임 사이
속도 전파, 게다가 datumize까지 한다. 그걸로 합성하면 소거가 z·roll·pitch에서 깨지고
lever arm이 되살아난다. 그리고 **조용히** 그렇게 된다.

### 무엇을 유지하는가 — 물체 프레임 오프셋

START를 누른 **그 순간**의 상대 자세를 뜬다. 오프셋은 **물체의 yaw 프레임**에 저장한다.

```
Rz_T  = Rz(yaw_obj)^T
d_obj = Rz_T · (p_veh_map - p_obj_map)
dyaw  = wrap_pi(yaw_veh_map - yaw_obj)
```

결과: **물체가 병진 → 기체 병진 / 물체가 yaw → 기체가 궤도 선회(같은 면을 봄) /
물체가 roll·pitch → 무반응.** 마지막이 의도적이다 — 기울어지는 물체가 기체를
병진시키면 안 되고, MANUAL_CONTROL에는 K/M 축이 아예 없어서 roll/pitch를 잡을 수도
없다(`station_bridge.release_horizontal`이 이미 기록한 사실).

**arm이 기체를 절대 움직이지 않는다.** 오프셋을 기체의 **현재** pose에서 뜨므로 arm
시점의 타깃이 곧 기체 자신의 위치다. approach도 settle도 없다 — 갈 데가 없다. 누락이
아니라 성질이고, 테스트가 못박는다
(`test_arming_follow_does_not_move_the_vehicle`).

물체의 헤딩 축은 `object_nav.yaw_axis`가 정한다. `auto`는 **첫 lock에서 가장 수평한
축을 한 번 고르고 앵커 수명 내내 고정**한다(중간에 바뀌면 헤딩 참조가 90° 계단으로
튄다). 현장 재구성 메시는 어느 축이 "x"인지가 임의라서 `geometry.yaw_from_R`(ZYX psi)은
쓰지 않는다 — 물체의 +x가 수직에 가까워지면 발산한다. **`yaw_axis: none`이면 물체 yaw를
아예 안 쓰고 오프셋을 맵 프레임에서 유지한다. 실패 클래스 하나가 통째로 사라지므로
첫 벤치·풀장 모드로 권장.**

### 네 겹의 안전장치

1. **이탈 클램프** — 목표점이 arm 지점 기준 `max_excursion_m`(기본 1.5 m) 구를 벗어나면
   그 안으로 클램프하고 패널에 `EXCURSION LIMIT`. **지오펜스 부활이 아니다**: 거부도
   정지도 아니고 **참조를 클램프**할 뿐이다. 2026-08-14에 펜스를 없앤 뒤 허용된 형태.
2. **레이트 제한 walk** — setpoint를 속도 칸(`speed`)만큼만 걸어간다. 물체 추정이 튀어도
   타깃은 걸어서 따라간다.
3. **하드 leash — step이 아니라 결과에 건다.** `|sp - p_veh| > approach_lead_m`이면
   그 비율로 줄인다. step만 제한하면 긴 tick 하나가 leash를 넘어간다
   (`_tick_approach`가 이미 배운 교훈).
4. **피드포워드** — 목표점 자체의 속도(궤도 항 포함) + 컨트롤러 선견거리로 테이퍼한
   catch-up. **이 층이 없으면 leash가 속도를 정한다**: 실측
   `kp·lead = 51.7 × 0.35 = 18.1 N` vs `F = 86.7·v + 5.76` → `v ≈ 0.084 m/s`
   (2026-08-18, memory: approach-speed-leash-limited). FF 없는 follow는 그보다 빠른
   물체를 못 따라간다.

### 신선도 사다리 — 조용한 초록은 없다

| 조건 | 동작 | 패널 |
|---|---|---|
| `age ≤ max_extrap_s` **및** `PoseTrack.state == "tracking"` | 외삽 + 전체 FF | `FOLLOWING` |
| 그 외 | **FREEZE**: 마지막 setpoint 재발행, FF 0 | `OBJECT STALE 1.2s` (amber) |
| freeze가 `hold_s` 초과 | **station hold로 강등** | `OBJECT LOST — HOLDING` (red) |

`PoseTrack.state` 검사가 핵심이다: `PoseWorker`는 파이프라인이 뭘 하든 10 Hz로
발행하므로 30초짜리 `registering`도 stamp는 완벽하게 신선하다. **age만 보면 절대 못
잡는다.**

**disengage가 아니라 강등**인 이유는 `station_bridge.py`가 이미 길게 논증했다 —
disengage는 순부력 −5.7 N인 기체의 깊이 유지를 같이 끄고, 가라앉는 기체는 시야가 바뀌어
재획득 확률이 **떨어진다**.

**태그를 놓치면 먼저 강등, 그 다음 bridge.** 세 가지 이유: (1) bridge의 전제가
"한 점 위에 정지"인데 follow는 구조적으로 움직이는 미션이다; (2) **태그를 놓치면 물체도
같이 안 보인다** — `T_map_obj`가 `NavFix`로부터 합성되므로 fix가 없으면 따라갈 대상
자체가 없고, 기체를 IMU로 이어가면서 목표를 관측 불가로 두는 건 표류하는 두 양을 빼는
것이다; (3) 그래도 안 끄는 게 옳다는 논거는 동일하다. `station_bridge.py`는 문구만
바뀌었다.

`STOP TRAJ`는 follow 중에도 눌린다(follow는 station처럼 `traj_on`을 안 켠다).
`set_traj(False)`가 이미 `self._eta`로 재타깃하므로 "추종 중단, 여기 그대로 hold"가
공짜로 된다. engage 게이트와 런타임 인터록은 **전부 그대로**다.

### arm 전 거부 — 각 사유가 자기 문장을 가진다

* `--pose` 없이 띄웠다
* `hw_nav.yaml`의 `nav_source`가 `main`이 아니다 → 물체와 fix가 **다른 카메라**를 타서
  extrinsic 소거가 깨진다(ROV RGB의 extrinsic은 전부 `[예측]`)
* 컨트롤러가 `mpcc`/`dobmpcc`다 → `HwMpcc.set_target_ned`가 `del v_ned, r_ned`로
  **속도 피드포워드를 버리고** 호출마다 경로를 재구축한다. `follow_ok = False`,
  `getattr(..., False)`로 fail-closed
* 물체 lock 없음(`registering`·범위 밖·미짝맞춤 포함)
* 헤딩 축이 수직이라 yaw 미정의 (`yaw_axis != none`일 때만)

`pair_exact`가 아닌 **loose 짝**은 거부가 아니라 **경고**다 — 열화이지 오류는 아니고,
그래도 그 순간 lever arm이 오차 예산에 들어왔다는 걸 말해줘야 한다.

### 화면

* 물체는 **다이아몬드**(태그는 사각형, 선체는 직사각형, 참조는 십자 — 일부러 네 번째
  모양). live면 채움, stale이면 속빈 amber, lost면 속빈 faint. **추정이 없으면 확신에
  찬 마커도 없다** — `p_dr`·`p_ref`가 이미 지키는 규칙과 동일.
* 헤딩 틱 0.15 m, `yaw_map`이 없으면 `?` 글리프.
* 물체 자체 궤적은 실선 1 px(viridis 아님 — 한 플롯에 나이 램프는 하나면 족하다).
* 선체 → 물체 점선 + cm 라벨. 조종사가 실제로 판단하는 숫자(0.3~0.8 m 밴드 안인가).
* 모서리 readout(2026-08-23 확장) — **물체 위치가 있는 유일한 곳**이고, 프레임은
  MAP이다(위 `px/py/pz`와 같은 프레임이라 눈으로 빼면 된다):

  ```
  obj  x +0.31  y -0.09  z -1.02 m
       live  d 0.87 m  yaw +147  age 0.08 s  pair 0 ms
  ```

  추종 중이면 한 줄 더: `follow following  err 4.1 cm  hold 0.55 m / +12`.
* **`depth-vs-MAP  1.02x  (+/-0.04)`** — 궤적 readout 위쪽, `tag-vs-ATTITUDE` 바로 아래.
  픽셀 격자를 태그면(z=0)과 교차시켜 얻은 **기대 Z**와 **C3 depth map이 말하는 Z**를
  비교한 중앙값 + p10~p90 spread다(`window._check_depth_scale`, 2026-08-23).
  **spread가 0.15를 넘으면 중앙값을 상수로 인용하지 말 것** — depth가 mis-scale이 아니라
  mis-shape이거나, 매트 아닌 것이 시야를 채우고 있다는 뜻이다.

  > 첫 버전은 **태그 중심 5x5**를 읽었는데, 태그 중심은 검은 사각형 — 이 장면에서
  > 스테레오가 유일하게 못 맞추는 패치다. 태그가 작아질수록(기체가 높을수록) 구멍과
  > 경계 번짐이 늘어 숫자가 **태그 개수를 따라** 움직였다(3장 0.94x / 8장 1.17x /
  > 14장 1.46x, 2026-08-23). 매트는 조밀한 텍스처이고 시야를 채우며 z=0에 있다는 걸
  > 태그맵이 안다 — 그래서 지금은 매트 전체에서 수백 개를 뽑는다.

  **이게 존재하는 이유**: depth 경로는 이 스테이션에서 **자기를 검증해 줄 증인이 없는
  유일한 양**이다. 태그 측위는 단안 PnP + 알려진 0.17 m 태그라 depth를 **전혀 안 쓰므로**,
  두 값의 비가 곧 depth의 절대 스케일이다.

  | 읽히는 값 | 뜻 |
  |---|---|
  | `1.00x` | depth는 metric. 물체 자세가 틀리면 그건 **메시** 탓이다 |
  | `~1.3x` | **depth 경로가 길다.** 재구성 메시도 추적 거리도 같이 부풀고, 다른 걸 손대기 전에 이것부터 |

  2026-08-23에 물체가 매트 **아래 0.45 m**에 찍히고 광선이 1.45배 길었는데, "메시가 크다"와
  "depth가 길다"가 **똑같은 증상을 만들어** 구별이 안 됐다. 이 한 줄이 그 갈림길이다.
  워커도, 추가 복사도, 새 시그널도 없다 — 코너·depth map·태그맵이 전부 이미 GUI 스레드에
  있다. 태그 3장 미만이면 아무것도 안 쓴다(한 장은 일화다).

* **거부 사유가 항상 보인다.** state가 `live`가 아니면 anchor의 `note`가 그대로 한 줄
  붙고, **`cold`(한 번도 lock 안 됨)일 때도 줄이 뜬다**:

  ```
  obj  cold  pair --
       object at 1.53 m, outside 0.15-1.20 m
  ```

  2026-08-23 이전에는 `cold`면 아무것도 안 그렸고 위치가 한 번이라도 잡히면 사유가
  숨었다 — 즉 **설명이 가장 필요한 두 경우에 플롯이 침묵했다.** 물체가 화면에 잘
  보이는데 다이아몬드가 안 뜨면 이 줄이 이유를 말한다(대개 `object_nav`의 거리 게이트).
* SENSORS에 `Object` 행(state · range · loose 짝이면 pair).
* 추종 타깃은 **공짜로** 그려진다 — `ctrl.ref_ned_at()`이 채우는 파란 십자가 그대로
  따라 움직인다. 새 필드도 새 그리기도 없다.

### 설정과 기록

`config/hw_mpc.yaml`의 `object_nav:`(오타는 startup에서 raise).
**`object_nav:`의 모든 숫자는 풀장 1단계까지 `[예측]`이다.**
0.3~0.8 m 동작 / 2.4 m 등록 실패는 [측정: KNOWN_ISSUES 2026-08-09, 저장 데이터],
공기 중 ~1.33배는 [측정: `calib/FOV_AUDIT.md`], lever arm 0.2855 m는
[측정: `c3_payload_frames.json`, Onshape 2026-07-19].

**기록 경계**: CSV에 열 10개(`obj_px,obj_py,obj_pz,obj_yaw_deg,obj_age_s,
obj_pair_dt_ms,obj_pair_exact,obj_state,follow_state,follow_err_m`)가 붙고 meta
schema가 **6**이 됐다. `obj_p*`는 `px/py/pz`·`dr_p*`와 같은 **datum-frame world FLU**라
상대벡터가 `obj_px - px`로 바로 나온다(버스의 `ObjectFix`는 MAP 프레임이고, 변환은
`_obj_row` 한 곳에서만 일어난다). **`obj_pair_exact == 1`인 행만 extrinsic-free** —
물체 위치 통계는 이 열을 넘어 합산 금지. `object_nav` 블록은 `--pose` 없이도 **항상**
기록된다(`{"enabled": false}`) — 키가 없으면 "이 빌드엔 follow가 없었다"와 "follow가
꺼져 있었다"를 구분할 수 없다.

### 오프라인에서 확인하기

```bash
./c3 gui --source demo --mpc --pose --demo-object drift   # still | drift | orbit
python rov_gui/tests/demo_e2e.py pid follow orbit
QT_QPA_PLATFORM=offscreen python rov_gui/tests/test_object_nav.py
```

demo의 물체는 **MAP 프레임에 놓고 실제 `R_t_frd_cam("main")`으로 역투영**한 것이고
합성 `NavFix`와 **capture stamp를 공유**하므로 `pair_exact`가 오프라인에서 실제로
검사된다. **demo가 증명 못 하는 것**: 실제 물체 추정에 관한 전부. SAM2 마스크 품질,
FoundationPose 지연, 드롭아웃 통계, depth 노이즈는 하나도 안 나온다.

**실기 미검증**이고, `--pose` 자체가 실기에서 한 번도 안 돌았다는 위험을 통째로
상속한다(KNOWN_ISSUES 2026-08-09). 벤치 → 풀장 순서는 KNOWN_ISSUES.md에 있다.

### 타깃을 갠트리가 든다면 (2026-08-23)

타깃 물체를 FMC4030 갠트리에 매달아 움직이는 실험이라면, 갠트리는 이 GUI에
넣지 않고 **별도 프로세스**로 띄운다:

```bash
./c3 gantry          # 왼쪽 창 — 갠트리 패널 (robust env)
./c3 gui --pose --mpc   # 오른쪽 창 — 스테이션 (rovgui-pose env)
```

임베드하지 않는 이유는 셋이고 각각 단독으로 충분하다: 벤더 `.so`가 프로세스당
연결 1개를 강제하고, 두 env가 서로 배타적이며(torch vs pyqtgraph/gtsam), 별도
프로세스라야 갠트리가 자기 Ctrl-C 비상정지를 갖는다.

갠트리 패널의 **Tag map position** 카드는 갠트리 어안 카메라가 지금
**anchor-25 태그맵 좌표 어디에 있는지**를 보여 준다 — `fixes.csv`와 같은 프레임이다.
같은 프레임인 이유가 우연이 아니다: 그 카드는 `rov_gui/control/tagnav.py`의
`TagDetector`+`TagNav`를 **그대로 import해서** 쓴다(`src/gantry_map_pose.py`).
그래서 태그맵 좌표에서 갠트리와 ROV를 나란히 놓고 볼 수 있다.

주의 두 가지: 그 카드가 보여 주는 것은 **렌즈** 위치이지 매달린 타깃 위치가
아니고(lever arm 미측정), 갠트리 **엔코더** 좌표를 태그맵에 올리는 옛 경로
(`R_gantry_to_slam`/`gantry_to_slam_scale`)는 세 군데가 깨져 있어 쓰면 안 된다.
자세한 것은 [README_fisheye_gantry.md](../README_fisheye_gantry.md)의
"ROV와 나란히 운용 — 태그맵 프레임 위치"와 KNOWN_ISSUES.

## IMU만으로 얼마나 가나 — `--imu-dr` (2026-08-17)

**질문**: 태그를 끊고 C3의 BNO086만으로 state를 갱신하면 얼마나 오래 쓸 만한가.
**방법**: 미션 정착(`engage.settle_s`, 10 s)이 끝나는 순간 태그 해에 앵커하고,
그 뒤로는 IMU만 적분한다. AprilTag 측위는 계속 돌아서 **ground truth 겸 화면의
두 번째 마커**가 된다. 두 마커가 벌어지는 속도가 곧 측정값이다.

미션 자체는 새로 만든 게 없다 — `station | line | square | circle`에 `origin_tag`,
그대로다. 바뀌는 건 state를 누가 만드느냐뿐이다.

```
./c3 gui --source hw --allow-command --mpc --imu-dr shadow
./c3 gui --source hw --allow-command --mpc --imu-dr control
```

* **shadow** — 컨트롤러는 계속 태그로 난다. 추정치는 그리고 기록하기만 한다.
  켜 둬도 명령이 **비트 단위로 동일**하다(테스트가 고정한다).
* **control** — 컨트롤러가 **추정치를 먹는다**. 커맨드라인에 이렇게 치는 건 그
  자체로 명시적이라 이 한 줄이면 된다. `--imu-dr-control`은 `hw_mpc.yaml`이
  이미 `mode: control`일 때만 필요한 별도 게이트다 — 설정 파일이 그 상태로
  남아 있다고 해서 아무도 요청하지 않은 폐루프가 뜨면 안 되기 때문.

**플롯**: 초록 실선 선체 = 태그(진실), 호박색 점선 유령 = IMU, 둘을 잇는 실선이
오차, 왼쪽 위에 `DR 42.0 cm  23 s [c3/ahrs]`. **자동 abort는 꺼져 있다**(운영자
결정) — 지오펜스도 없으므로 위치를 이유로 세우는 건 조종사의 E-STOP뿐이고, 저
숫자가 그 계기다. 켜려면 `hw_mpc.yaml`의 `imu_dr.abort_err_m`에 숫자만 넣으면 된다.

**자세는 AHRS가 기본**: 자이로 적분 + 가속도로 roll/pitch만 수평보정, yaw는 자이로
전용(철제 수조에서 지자기는 못 쓴다). 여전히 100% IMU이고, 자세오차가 t³로 커지는
항을 t²로 묶어준다. `gyro`(순수 적분)와 `vehicle`(roll/pitch만 오토파일럿)도 있다.

**깊이는 IMU가 아니다** — 압력센서 그대로다(운영자 결정). z를 적분하면 그 오차가
그대로 heave 명령이 돼서 매트에 박거나 수면으로 뜬다. IMU가 적분한 z도
`dr_pz_imu`로 같이 남으니 사후 비교는 공짜다.

### 먼저 캘리브레이션 — 안 하면 추정 자체가 성립하지 않는다

이 카메라의 가속도계에는 **1.80 m/s²(0.184 g) 바이어스**가 있고(거의 전부 IMU x),
IMU→카메라 extrinsic은 장치에 **없다**(`KNOWN_ISSUES.md`). 보정 없이는
`0.5·1.80·t²` = 10초에 **90 m**다 [유도] — 근사적으로 틀린 게 아니라 못 쓴다.
(리포가 오래 "스케일 +20%"라고 적어온 건 **오독**이었다: 똑바로 세운 자세 하나에서
잰 `|a|=11.8`인데, 그 자세에선 중력이 IMU x와 거의 나란해서 바이어스가 그대로
더해진다. 6자세로 재면 스케일은 1에서 0.9% 이내다. 구분이 중요한 이유는 **고정
바이어스는 정착창 정적보정이 자세와 무관하게 완전히 지우기** 때문이다.)

```
# 1. accel scale/bias — 물 밖에서 기체를 2분 이상 천천히 굴린다(구면 전체)
python -m rov_gui.tools.calib_c3_imu <run>/..._c3_imu.jsonl --fit accel \
    -o config/c3_imu_calib.json
# 2. IMU->기체 회전 — 60 s 손 흔들기, 3축 전부, ArduSub 켜 둔 채로
python -m rov_gui.tools.calib_c3_imu <run>/..._c3_imu.jsonl --fit rotation \
    --rov <run>/..._rov.jsonl -o config/c3_imu_calib.json
```

2번이 **IMU→기체**로 바로 가는 게 요점이다: 장치에 없는 IMU→카메라 extrinsic이
필요 없어지고, 카메라의 40° 틸트가 자동으로 흡수된다. 도구는 구면을 못 덮은
텀블과 단일축 wiggle을 **거부한다** — 자신 있게 틀린 답을 내는 게 이 캘리브레이션의
진짜 실패 모드라서. 캘리브가 없으면 런 meta의 `calibration_sha1`이 `null`로 남고
시작할 때 크게 경고한다.

### 기록과 사후 분석

CSV 꼬리에 `dr_px,dr_py,dr_pz,dr_pz_imu,dr_yaw_deg,dr_err_m,dr_err_z_m,dr_t_s,
dr_hz,dr_n,dr_ok,rp_residual_deg,roll_deg`. `dr_px`는 `px`와 **같은 규약**
(world FLU, datum 프레임)이라 그냥 빼면 드리프트다. meta의 `imu_dr` 블록이
기록 경계고, 꺼져 있어도 `{"enabled": false}`를 쓴다 — 키가 없으면 "옛 빌드"와
"껐음"을 구분할 수 없다. 런 내내 `*_c3_imu.jsonl`(원시 샘플)도 남는다.

```
python -m rov_gui.tools.plot_imu_dr <run 폴더>
python -m rov_gui.tools.plot_imu_dr <run> --from-jsonl --attitude gyro,ahrs --restart 20
```

그림 4장(맵 오버레이 / 오차 vs 경과시간 / along·cross·수직 분해 / 건강)과
`imu_dr_summary.json`(5·10·25·50·100 cm 도달시간, `0.5·b·t²`와 `(1/6)·g·β·t³`
적합 + R²). 어느 쪽 R²가 높은지가 **가속도계를 쫓을지 자이로를 쫓을지**를 말해준다.

`--from-jsonl`이 원시 로그를 남기는 이유 전부다: 같은 풀 세션을 **다른 자세 모드로,
다른 캘리브로, 다른 앵커로** 다시 추정할 수 있다. 특히 `--restart N`은 N초마다
재앵커해서 연속 곡선 하나(n=1)를 독립 구간 수십 개로 바꾼다 — 비행 중에는 운영자
결정대로 **앵커를 한 번만** 하고, p50/p95 통계는 여기서 뽑는다.

**실기 미검증** — 검증 상태와 순서는 `KNOWN_ISSUES.md`.

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
| 기본 요청 | 200 Hz (`--mavlink-rate`) | 200 Hz (`--c3-imu-rate`) |
| 실측 천장 | **~208 Hz** | **~486 Hz** (스탠드얼론) |

* **ROV IMU**(실측 2026-08-06, `SET_MESSAGE_INTERVAL`): 기본 2-3 Hz · 요청 50 → 62 ·
  100 → 125 · 200 → **208** · 400/1000 → 여전히 208. 비용은 약 0.2 Mbit/s.
  레이트 올리기는 **기본으로 켜져 있다**(`--no-mavlink-set-rates`로 끔) — ArduSub
  기본값 2-3 Hz는 대시보드 불빛 말고는 쓸 데가 없고, depth 녹화 옆에 남기는 센서
  로그는 제대로 된 레이트여야 남길 값어치가 있다. 켜면 기체로 **송신**한다.
* **C3 IMU**: accel+gyro만 쓰면 스탠드얼론 ~486 Hz(`c3_camera/pipeline.py` 실측).
  배치 임계값은 10 — 이 레이트에서 1이면 샘플이 샌다(`c3_collect.py` 실측).
  `--no-c3-imu`로 끌 수 있다.
  **기본 요청이 500 → 200으로 내려갔다 (2026-08-17)**: 더 달라고 하면 오히려 덜
  받는다. 500 요청에 이 스테이션에서 실측 **234.9 Hz**
  (`sessions/ui_recordings/c3_depth_20260809_202409_c3_imu.json`), 200 요청엔
  `pipeline.py:67` 실측 **194 Hz**. 요청을 올린 만큼 처리량을 잃는 구간이다.
  덧붙여 이 문단은 오래 **틀려 있었다** — "`ROTATION_VECTOR`를 켜면 모든 스트림이
  ~40 Hz로 붕괴한다"고 적혀 있었는데, 같은 파일의 실측표(`pipeline.py:71`)는
  **같은 레이트로 요청하면 페널티가 없다**고 말한다(accel+gyro @200 +
  ROTATION_VECTOR @200 → 194 Hz). 끌어내리는 건 지자기와, accel/gyro보다 **느리게**
  요청한 rotation vector다.
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
  매핑은 그걸 보면서 `--js-axis-*` / `--js-btn-*`로 맞추면 된다(축은 보통 자동
  감지가 맞으므로 손댈 일이 없어야 정상이다 — 아래 참조).
* **아날로그 트리거는 대기 상태가 중앙이 아니라 끝**이다. 이 데스크톱의 Xbox
  Wireless Controller는 축 4·5가 무입력에서 `-1.000`을 낸다 — 그 축을 heave에
  매핑해 뒀더니 GUI를 켜자마자 **heave가 +0.60으로 고정**됐다(조종사가 준 적 없는
  명령). 그래서 축이 처음 보고한 값이 그 축의 0이 된다. 단 **중앙 근처(|v| ≤ 0.5)가
  아닐 때만** — 진짜 가운데 있는 스틱은 진짜 0을 유지한다. 자동 0점 잡힌 축은
  시작 로그에 나온다.
* **축 번호는 케이블에 따라 바뀐다 — 그래서 박아두지 않고 패드에 물어본다.**
  같은 Xbox 패드가 드라이버에 따라 축을 다르게 매긴다:
  BT(hid-generic)는 `0,1` 왼쪽 · `2,3` 오른쪽 · `4,5` 트리거,
  **USB(xpad)는 `0,1` 왼쪽 · `3,4` 오른쪽 · `2,5` 트리거**다.
  그래서 예전 고정 기본값(`yaw 2`, `heave -3`)은 무선에서만 맞았고, 2026-08-17에
  패드를 **유선으로 꽂자 오른쪽 스틱 좌우(=yaw)가 heave를 몰고 yaw는 죽었다**
  (yaw가 축 2 = 왼쪽 트리거에 물렸는데, 트리거는 자동 0점 처리되므로 영영 0).
  지금은 `JSIOCGAXMAP`으로 **각 축의 `ABS_*` 코드를 읽어** 오른쪽 스틱을 이름으로
  찾는다(`ABS_RX`·`ABS_RY`가 둘 다 있으면 그게 오른쪽 스틱이고 `Z/RZ`는 트리거,
  없으면 `Z/RZ`가 오른쪽 스틱). 시작 로그가 무엇을 찾았는지 찍는다:
  `joystick axes (read off the device): surge=-1, sway=0, heave=-4, yaw=3`.
  `--js-axis-*`에 숫자를 주면 그게 이긴다(로그에 `*`로 표시).
  ioctl이 실패하면 예전 BT 번호로 폴백한다 — 못 나는 것보단 낫다.
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

| 물리 | 커널(BT) | 커널(USB) | → 기체(SDL) | QGC 기능 |
|---|---|---|---|---|
| A / B / X / Y | 0 / 1 / 3 / 4 | 0 / 1 / 2 / 3 | 0 / 1 / 2 / 3 | gripper open · manual · depth_hold · stabilize |
| LB / RB | 6 / 7 | 4 / 5 | **9 / 10** | mount_tilt_down / up |
| View / Menu | 10 / 11 | 6 / 7 | **4 / 6** | **disarm / arm** |
| Xbox | 12 | 8 | 5 | shift |
| 스틱 클릭 L/R | 13 / 14 | 9 / 10 | 7 / 8 | mount_center / input_hold_set |
| **십자키** | **축 6,7 (hat)** | **축 6,7 (hat)** | **11–14** | gain_inc/dec · lights dimmer/brighter |

**커널 번호는 축과 마찬가지로 케이블에 따라 다르다** — BT는 2/5/8/9가 비어 있고
USB는 0..10으로 빽빽하다(둘 다 실측). 그래서 패드 이름으로 찾는 고정 표 하나로는
양쪽을 못 덮는다. 지금은 `JSIOCGBTNMAP`으로 **각 버튼의 `BTN_*` 코드를 읽어**
SDL 번호로 옮긴다(`BTN_TL` → 9, `BTN_SELECT` → 4 …). BT 패드의 btnmap을 이 규칙에
통과시키면 실기에서 검증했던 위 표가 **빈칸까지 그대로** 재현된다(회귀 테스트가
지킨다). 이름 표는 ioctl이 실패할 때의 폴백으로만 남았다.

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

## 데모를 다시 난다 — replay (2026-08-30, 실기 미검증)

trajectory 패널의 여섯 번째 모양. 핸드헬드로 찍은 시연 하나에서 뽑은 pose 궤적을,
기체가 **지금 있는 자세를 원점 삼아** 다시 난다. diffusion policy로 가는 M0 게이트:
라벨 추출 → 좌표 변환 → feasibility → MPC 추종 → (옵션) gripper 실행이 데모 1개로
전부 검증되고, policy가 나중에 쓸 **같은 설치 경로**(plan_stream → `set_path_plan_ned`)를
미리 밟는다. 계획 문서: `docs/DP_TRAJECTORY_PLAN.ko.md`.

```bash
# 1) 시연 세션에서 pose 라벨 추출 (환경 태그가 프레임에 보여야 한다)
python -m umi_handheld.extract_pose sessions/demonstration_00NN
# 2) 스테이션에서 재생
./c3 gui --mpc --replay-session sessions/demonstration_00NN
#    → shape 콤보에서 replay 선택 → START (hold)
```

동작 순서와 규칙:

* **arm은 기체를 안 움직인다** — follow와 같은 성질. 트랙은 자기 첫 pose 기준
  상대좌표로 변환돼 있고, START 순간의 기체 pose에 앵커된다. 태그 id는 아무 데도
  안 들어간다(측위 자체는 여전히 태그가 한다).
* **속도는 시간 팽창으로 깎는다** — 데모가 `replay.v_max_m_s`(기본 0.12 [예측])보다
  빠르면 기하는 그대로 두고 시계만 α배 늘린다. 몇 배로 깎였는지는 로그 한 줄과
  meta(`plan_stream.run.time_dilation`)에 남는다.
* **PlanFilter가 컨트롤러 앞의 마지막 관문이다** (`control/plan_stream.py`, 순수
  numpy): 앵커/점프 게이트, 운동학 한계, 그리고 `replay.workspace_box_ned` —
  지오펜스가 제거된 지금 **위치 기반 방호는 이 박스뿐**이다. 판정마다 margin 숫자가
  run 폴더의 `plans.jsonl`에 남는다 (planner-vs-tracker 귀속용).
* **두 모드**: `replay.stream_period_s: 0`(기본)은 트랙 전체를 플랜 하나로 설치(M0a);
  `> 0`이면 `horizon_s` 창으로 잘라 그 주기로 흘린다(M0b) — 나중에 policy가 1 Hz로
  낼 스트림과 같은 경로를 실데이터로 시험하는 모드. 이음새는 PlanStitcher가
  commit-then-blend(cosine, `blend_s`)로 붙이고, 마지막 knot 너머는 **종점 v=0
  홀드**다(등속 외삽 ray 금지 — path 모드 코너 사고의 교훈).
* **gripper는 기본 OFF** (`replay.gripper: false`). 켜면 데모의 gripper-width 채널을
  문턱+히스테리시스로 momentary drive(-1/0/+1)에 재생한다. 이 드라이브는 sink에
  **래치되는 레벨**이라, 미션을 끝내는 모든 경로(STOP TRAJ / DISENG / E-STOP / 재arm)가
  0.0을 쏴서 턱을 놓는다. 실기 gripper는 피드백이 없으므로(open-loop) "잡혔는지"는
  기록도 화면도 말해주지 않는다 — 초기 실기는 궤적만 검증하고 턱은 끄고 날 것.
* **mpcc는 거부한다** — contouring은 스트림 플랜을 소비하지 않는다(`follow_ok`와 같은
  fail-closed 규칙). dobmpc/mpc/*_tuned/pid로 날 것.

기록 경계: meta `schema_version` 8, `trajectory.kind: replay`,
`reference_clock.strategy: plan_stream_replay`, 항상 기록되는 `plan_stream` 블록,
CSV 끝에 `plan_id,ref_src,grip_cmd` 세 열. **replay 런은 기하 미션 런과 절대 합산
금지** — 참조의 출처가 다른 가족이다. 오프라인 검증:
`rov_gui/tests/test_replay.py`(+`test_plan_stream.py`), E2E는
`demo_e2e.py dobmpc replay`(실제 acados 솔버, 2026-08-30 통과).

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

### UI 녹화에 이름을 붙인다 (2026-08-14)

헤더 `REC UI` **왼쪽의 입력칸**에 아무 때나 이름을 쳐 두면, 다음 녹화가
`ui_20260814_190532_squaretest.mp4`로 저장된다. 비워 두면 예전 그대로
`ui_20260814_190532.mp4`. 사이드카 `.json`에는 **슬러그가 아니라 친 그대로**의
문자열이 `"name"`으로 들어간다(슬러그는 공백·점·슬래시를 잃는다).

* **모달 팝업이 아니라 상시 입력칸**인 건 안전 결정이다. 이 창이 유일한 키 핸들러라서
  (아래 "키보드"), 주행 중 팝업이 뜨면 W/A/S/D를 그대로 삼킨다. 이 칸은 **클릭했을 때만**
  포커스를 갖고(`ClickFocus`, Tab으로는 안 온다), 포커스를 잡는 순간 **모든 축을 0으로
  내린다**(키 릴리스를 못 받으니 축이 물린 채로 남는 걸 막는다) — trajectory 패널의
  숫자 칸과 같은 패턴이다.
* **Esc는 언제나 E-STOP**이다. QLineEdit은 원래 Esc를 편집 취소로 삼키는데, 그러면
  조종사가 비상시에 누르는 그 키가 캐럿 위치에 따라 아무것도 안 하게 된다. 이벤트
  필터가 포커스를 놓고 **그대로 창의 E-STOP까지 통과시킨다**
  (`test_recording_name_field_never_takes_escape_away_from_estop`).
* Enter는 확정+포커스 해제. 녹화 중에는 칸이 잠긴다 — 이름은 시작 시점에 파일명으로
  구워지므로, 중간에 고치면 그 파일이 아닌 것을 가리키게 된다.
* 칸은 녹화가 끝나도 **비워지지 않는다**: 한 세션은 보통 같은 것의 여러 테이크이고,
  매번 다시 치게 만들면 테이크 이름이 어긋난다.

### 흰 위젯이 남지 않게 한다 (2026-08-14)

Qt의 기본 위젯 몇 개는 **밝은(라이트) 위젯**이라, 손대지 않으면 이 팔레트 위에서
가장 밝은 물체가 된다. 어두운 선실에서 탁한 영상을 읽으려고 만든 화면 옆에 흰
막대가 있으면 암순응이 깨진다 — 그래서 세 가지가 `theme.py`에 명시적으로 쓰여
있다(조종사 지적: "Recording 이름 적는거나, log 스크롤같은게 흰색 바"):

* **`QLineEdit`** — 녹화 이름 칸. 콤보·스핀박스와 같은 언어(패널 채움 + 같은 테두리
  + hover 시 accent)를 따른다.
* **`QScrollBar`** — MISSION LOG의 스크롤. 이건 **앱 스타일시트에 있어야만** 한다:
  스크롤바는 자식 위젯이라 위젯 자신의 스타일시트가 닿지 않고, 로그 뷰는 자기
  스타일시트를 갖고 있다. 화살표 버튼 없는 얇은 막대 — 로그는 휠로 스크롤한다.
* **`QPlainTextEdit`** — 로그 본체는 패널보다 한 단계 **어두운 우물**(프로그레스
  바·플롯과 같은 배경)이라, 스크롤되는 영역이 또 하나의 돋을새김 컨트롤로 읽히지
  않는다.

위젯 하나에 `setStyleSheet`을 거는 건 여전히 피한다 — 그러면 그 위젯에 대해 앱
시트를 **대체**해서 Qt 기본 모습으로 되돌아간다(trajectory 패널이 이걸로 한 번
당했다). 크기가 필요하면 `setFixedHeight`/`setMaximumWidth`로, 패딩만 다르면
`objectName`(`#Compact`) 규칙으로 앱 시트 안에서 해결한다.

### 한 런이 만든 것은 한 폴더에 모인다 (2026-08-14)

예전에는 START 한 번이 세 곳에 파일을 뿌렸다 — `sessions/ui_recordings`,
`sessions/nav_runs/<stamp>/`, `sessions/mpc_runs`. 같은 런인데 세 군데였고,
타임스탬프를 눈으로 맞춰야 관계를 알 수 있었다. 지금은:

```
sessions/low_level_controller_data/20260814/0814_184148/
    ui_20260814_184149_squaretest.mp4   조종사가 본 화면 (+ .json 사이드카)
    c3_depth_20260814_184150.mp4        피드 녹화 (+ 센서 JSONL, 같은 stem)
    mission_log.txt                     MISSION LOG 전체, 벽시계
    nav_184148/                         map.json fixes.csv detections.csv frames.csv
                                        controller.json  ← M/C/D + 그 런의 게인
    mpc_184151.csv  mpc_184151.meta.json  컨트롤러의 기록
    events.log                          engage / 거부 / disengage
```

* 말단 폴더 이름이 `MMDD_HHMMSS`인 이유(2026-08-14, 운용자 요청 두 번). **날짜**:
  런 폴더는 날짜 디렉터리 **밖으로 꺼내지는** 일이 잦은데(메시지에 붙이기, 플롯
  스크립트 인자, 백업) `1841`만으로는 언제 것인지 알 수 없다. 부모가 여전히
  `YYYYMMDD`를 들고 있으니 중복은 공짜다. **초**: 수조 세션은 1분에 런을 여러 번
  만드는데 분 해상도가 그걸 조용히 합쳐버렸다.
  **그 전에 만들어진 `HHMM/` 폴더는 그대로 둔다** — 옮기지 않는다. 부작용 하나:
  합류(아래)는 새 이름끼리만 일어나므로 옛 `1841/`이 새 런에 다시 열리는 일은
  없다(안전한 방향).

* 루트는 `--rec-dir`(기본 `sessions/low_level_controller_data`)과
  `hw_mpc.yaml: log_dir`. 둘 다 **쓰이는 폴더가 아니라 트리의 뿌리**다.
* 세 writer(GUI 스레드의 화면 녹화, 창의 nav 녹화, 자기 스레드의 MpcWorker CSV)가
  서로를 모른 채 같은 답을 내야 한다. 공유 런 id를 넘기려면 핸드셰이크가 필요하고,
  핸드셰이크에는 "한 writer가 놓쳐서 파일이 조용히 딴 데 떨어지는" 실패 모드가 있다.
* **초를 넣으면서 바뀐 것**: 분 버킷일 때는 세 writer가 각자 `1841`을 *계산*했고
  합류 창은 분 경계를 넘는 런 하나만을 위한 것이었다. 초 해상도에서는 :48, :49,
  :51이 서로 다른 이름을 만들므로, `run_dir()`은 이제 계산하지 않고 **오늘 폴더에서
  진행 중인 런을 찾아 합류**하고 없을 때만 새로 만든다. 여전히 파일시스템이 유일한
  상태이고 핸드셰이크는 없지만, **`JOIN_WINDOW_S`(90 s)가 이제 모든 런을 묶는
  역할**을 한다. 앞선 writer로부터 90초가 넘어서 시작하는 writer는 자기 폴더를
  갖는다(분 버킷 시절 분 경계에서 일어나던 것과 같은 일이고, 방향만 늘었다).
  합치기만 하고 절대 나누거나 옮기지 않는다.
* **"진행 중"의 판정**은 폴더 **이름 시각**(런이 시작된 때)과 **디렉터리 mtime**
  (파일이 *생성*될 때 갱신) 중 **더 늦은 쪽**이다. 이름만 보면 writer가 아직
  도착하는 중인데 90초 만에 런이 끝나버리고, mtime만 보면 `when=`을 넘기는 호출
  (테스트·리플레이)에서 mkdir된 실제 시각과 물어보는 시각이 달라 엉뚱한 답이 난다.
* `nav_<hhmmss>/`가 **하위 폴더인 이유**: `plot_nav_run`·`build_tag_map`이 "런 디렉터리"를
  받아서 그 안의 `map.json`/`fixes.csv`를 찾는다. 평평하게 풀면 두 도구가 과거·미래
  기록 전부에 대해 깨진다.
* **옛 데이터는 안 옮겼다.** `sessions/ui_recordings`, `nav_runs`, `mpc_runs`는 그대로
  있다 — config 주석·문서·메모리가 그 경로를 **측정치의 출처**로 인용하고 있고
  (`hw_nav.yaml`의 z 범위가 `nav_runs/*/fixes.csv`를 든다), 경로를 옮기면 인용이
  죽은 참조가 된다.

### `events.log`만 있는 폴더 — 왜 생겼고, 왜 이제 안 생기나

2026-08-14에 `20260814/` 밑 일곱 개(`2009 2011 2037 2039 2047 2053 2111 2118`)가
`events.log` 한 줄짜리였다. 그 줄은 전부 `MpcWorker.setup()`이 남기는 **빌드
지문**이다:

```
2026-08-14 21:18:46 ready: dobmpc, path-following, no geofence, mission=station @tag79
```

즉 **"그 시각에 스테이션을 띄웠고, 아무것도 날리지 않았다"**는 뜻이다(engage도,
녹화도 없음). `_log_event`가 `run_dir()`을 부르고 `run_dir()`은 폴더를 만들기
때문에, 그 한 줄이 폴더 하나를 통째로 만들었다. `2111`은 같은 줄이 4번의 버스트로
72줄 — 그 시각에 `setup()`이 그만큼 돌았다는 뜻이고(프로세스를 반복해서 띄웠을
때 나는 모양), **데이터가 아니라 실행 기록**이다. 지워도 아무것도 잃지 않는다.

지문 자체는 유지할 값어치가 있어서(2026-08-14에 "소스에서 이미 없앤 거부"에 수조
세션 하나를 썼다 — 돌고 있던 게 그냥 옛 프로세스였고 화면에 그렇게 말하는 게
없었다) **없애는 대신 미뤘다**: MISSION LOG에는 즉시 뜨고, 파일에는 **같이 적을
진짜 사건이 생겼을 때** 그 런 폴더 `events.log`의 첫 줄로 들어간다. 한 세션이 런을
여러 번 날리므로 **새 폴더마다** 다시 붙는다(`_log_event(defer=True)`).

### 발표용 궤적 그림 — 날짜만 넣으면 나온다 (`plot_runs.py`, 2026-08-15)

```bash
./c3 plots                              # config/traj_plots.yaml 의 dates
./c3 plots --dates 20260814 --list      # 뭘 그릴지만 먼저 본다
./c3 plots --dates 20260814_212021      # 그 CSV 하나만
./c3 plots --dates 20260814 --theme dark
python -m rov_gui.tools.plot_runs -c my_plots.yaml       # 래퍼 없이도 같다
```

`config/traj_plots.yaml`에 적는 것은 **날짜뿐**이다. 해상도를 골라 쓸 수 있다 —
`20260814`(그 날 전부) / `20260814_2120`(그 런 폴더) / `20260814_212021`(그 CSV
하나) / `all` / `latest`. 그림은 한 런에 하나씩

```
figures/trajectories/<날짜>/<날짜>_<런>_<컨트롤러>_<시분초>.png
```

로 떨어진다. 그림 위에는 **기체 + reference(검은 점선, 위) + 추적 물체(있으면)
+ 오차 하나 + 키(key)**만 있고 날짜는 없다 — 날짜는 파일 이름에 있으니 그림에
또 박으면 슬라이드 캡션과 싸운다.

**색이 두 가지를 동시에 나른다.** *색상*은 어느 트랙인지(파랑 = 기체, 주황 =
추적 물체), 같은 색상 안의 *밝기*는 언제인지(런 시작이 밝고 끝이 어둡다)를
말한다. 그래서 아래 띠는 범례도 컬러바도 아니고 둘 다다 — 트랙마다 그라디언트
한 줄, 경과시간 축 하나를 공유한다. reference에는 램프를 안 준다: 그건 측정이
아니라 **명령**이고, 물들이면 그것도 흘렀다는 뜻이 되기 때문이다
(`time_color: false`면 예전처럼 단색 + 보통 범례).

읽히게 만드는 건 두 가지다(2026-08-23: 첫 판은 색이 너무 비슷해 시간을 못 읽었다).

* **램프가 넓고 고르다.** 한 색상을 고수하는 대신 이웃 색상(analogous)까지
  걸어간다 — 파랑은 시안-파랑에서 인디고로, 주황은 앰버에서 진한 적갈로. 그리고
  `_cmap`이 램프를 **OKLab 호길이로 재매개화**해서 1분이 어디에 있든 같은 양의
  색 변화로 보이게 한다. 옛 주황 램프는 전반부를 거의 정지한 채 보냈다(구간
  ΔE 5.7, 5.1 / 13.1, 13.8) — 런의 첫 2분이 한 색이었다는 뜻이다. 지금은
  end-to-end ~53 / 49, 사분위 간격 ~14로 **측정해서** 고정했다.
* **경로 위의 라운드-타임 점**(60 s마다 하나 식)에 **초 숫자를 붙였다.** 궤적이
  같은 자리를 여러 바퀴 돌면 마지막 바퀴가 앞 네 바퀴를 덮어버리는데, 이건 어떤
  팔레트로도 못 고친다. 링을 두른 점은 그 덮임을 견디고, 라벨이 붙으면 **색을
  판단할 필요 자체가 없어진다**(간격이 곧 속도라는 건 덤). 근처에 샘플이 없는
  눈금은(놓아버린 구간, `segment: engaged`) 가까운 행으로 **당기지 않고 버린다** —
  120 s 라벨을 141 s 지점에 붙이면 맞아 보이는 자리에 틀린 시간을 적는 것이다.
  `time_ticks: false`로 끄고, `time_bands: N`이면 램프를 N단으로 계단화한다.

**파일 이름의 컨트롤러는 `meta.json`에서 온다.** 스테이션은 PID든 MPC든 로그를
`mpc_<hhmmss>.csv`로 쓴다(접두사가 가리키는 건 컨트롤러가 아니라 **워커**다) —
CSV stem을 그대로 쓰면 PID 결과가 `..._mpc_....png`로 나간다.

세 가지가 이 도구의 존재 이유다.

* **프레임.** CSV는 **그 런의 datum 프레임**이다 — ENGAGE 순간의 자세가
  (0,0,0)/yaw 0이 되고(`workers.py _datumize`) 컬럼은 world-FLU로 미러된다. 같은
  오후의 런 둘을 그대로 겹치면 실제로는 수조 반대편이었던 사각형 둘이 포개진다.
  그래서 매 런을 `meta.json`의 `hardware.datum_tag_frame`로 **맵(태그) 프레임에
  되올린다**(`_to_map_xy`와 같은 강체 변환). datum이 없는 런은 그럴듯한 자리에
  조용히 그리지 않고 **이유를 말하고 건너뛴다**(`frame: datum`으로 강제하면
  "DATUM FRAME" 워터마크가 그림에 박힌다).
* **축.** 모든 그림이 **같은 풀장 박스**에 잠긴다. autoscale은 어디에도 없다.
  박스는 스테이션이 쓰는 것과 같은 유도 — 최외곽 태그 **가장자리** + `pool_margin_m`
  (`geometry.py pool_from_tags`) — 이라 `config/tag_map_full.yaml`에서
  x [-0.723, +1.301] y [-2.120, +2.317] = 2.02 × 4.44 m가 나온다. 런의
  `nav_*/map.json`이 다른 박스를 적어놨으면 그림을 바꾸는 대신 **경고를 찍는다**.
* **추적 물체(`--pose`, `object_nav.py`).** `obj_px/py/pz`는 `px/py/pz`와
  **같은 datum-프레임 world FLU**라(workers.py:341) 기체와 똑같은 변환으로
  맵에 올라간다 — 그래서 둘이 mm 단위까지 비교 가능하다. 선이 아니라 **점**으로
  그린다: 트래커가 재시드(reseed)하므로 선을 그으면 일어나지 않은 이동을
  그리게 된다. 필터 셋은 전부 기록이 실제로 하는 일 때문에 있고, 각각 몇 개를
  뺐는지 **콘솔에 찍는다**(조용히 안 버린다) —
  `live_only`(stale/cold/lost는 마지막 값을 들고 있는 것이지 측정이 아니다),
  `pair_exact_only`(**카메라 extrinsic은 물체 pose와 태그 fix가 같은 프레임에서
  왔을 때만 소거된다**; `obj_pair_exact == 0`은 CSV가 record boundary라고
  부른다), `min_step_m`(로그 20 Hz / 트래커 ~9 Hz라 대부분의 행은 앞 값의 반복).
  축이 풀장이라 물체가 박스 **밖**으로 나가면 그 개수도 찍힌다.
* **어떤 오차 하나를 보여주느냐.** path-following 런의 기본은 **cross-track
  RMS**다. 공간 follower가 기체의 현재 선분 투영점보다 stage 0을 최대
  `path_lead_m`만큼 앞세우므로(`_path_split`) 선 위에 정확히 올라탄 기체도
  radial |p − ref|은 그 lead만큼 나온다 — 2055 런이 cross-track 3.3 cm인데
  radial 12.3 cm인 이유가 그거다.
  radial을 쓰고 싶으면 `--error radial`. **station 홀드는 경로가 없으므로**
  `path_following`이 켜져 있어도 radial로 돌아간다(고정점에 대한 "cross-track"은
  뜻이 없는 방향이다). 5바퀴짜리 참조는 **첫 바퀴만** 그린다(위상이 다른 점선
  다섯 벌이 서로의 빈칸을 메워 실선으로 보인다).

**어느 구간을 그리느냐는 `segment: auto`가 정한다.** square/line/circle은
trajectory 창이 곧 결과지만, station 홀드와 object follow는 `traj_on`을 **한 번도
켜지 않는다**(meta `run.traj_on: false`) — 거기서 trajectory를 달라는 건 아무것도
달라는 것과 같다. 그래서 auto는 trajectory → engagement → 로그 전체 순으로
떨어진다. 엄격하게 보려면 `--segment traj`.

**제목의 컨트롤러도 CSV가 정한다.** `meta.controller.type`은 런이 *끝난* 컨트롤러이고
패널은 런 도중에 바뀔 수 있다(2026-08-23에 한 CSV 안에서 dobmpc → pid → mpc_tuned로
간 런들이 있다). 행마다 있는 `mode` 열이 진실이라, 그려진 구간의 mode가 서로
다르면 제목이 `PID → MPC path-frame`처럼 **날아간 순서대로** 나열한다.

데이터가 없으면 없다고 말한다. 없는 날짜, 아무것도 안 맞는 `_시분초`, 헤더만 있는
CSV, 쓸 만한 샘플이 `min_points`에 못 미치는 런은 전부 `[none] <무엇>: <왜>` 한 줄로 나오고,
마지막에 몇 개가 안 나왔는지 합계가 붙는다. 출처는 그림에서 빼고 **파일 이름과
콘솔**로 옮겼다 — `[ok] 20260814/2055/mpc_205520 → 20260814_2055_mpc_205520.png`
한 줄이 그림마다 찍히므로 CSV는 여전히 한 번의 조회 거리다(CLAUDE.md 인용 규칙).

테스트: `rov_gui/tests/test_plot_runs.py`(합성 런 폴더, 하드웨어·Qt 없음).

## 누수 경보 (2026-08-14)

QGC에 있고 여기 없던 기능. QGC에도 누수 **위젯**은 없다 — 있는 건 기체 메시지
스트림이고, ArduSub가 거기에 대고 소리치는 것이다. 그러니 할 일은 표시등을
그리는 게 아니라 **이 스테이션이 버리고 있던 메시지를 받는 것**이었다.

**ArduSub가 누수를 알리는 통로는 정확히 하나**: `STATUSTEXT`, 문자열
`"Leak Detected"`, severity `MAV_SEVERITY_CRITICAL`(2), `Sub::failsafe_leak_check()`.
전용 메시지도, SYS_STATUS 비트도 없다 — MAVLink 규격의
`MAV_SYS_STATUS_SENSOR_LEAK`은 **반대 방향**(동반 컴퓨터가 원격 누수 센서를
오토파일럿에 알려주는 입력)이라, SYS_STATUS를 아무리 폴링해도 영원히 안 나온다.

여기서 따라오는 세 가지가 구현을 결정했다(`rov_gui/leak.py`):

* **선로 위에서는 반복이지 래치가 아니다.** 젖어 있는 동안 **20초마다** 다시 보낸다.
  그래서 토픽사이드 상태는 타임아웃이 붙은 홀드(`LEAK_HOLD_S = 50 s`)다 — 한 번
  놓친 반복이 "물이 빠졌다"가 되면 안 되니까.
* **"누수 해제" 메시지는 없다.** 기체는 자기 페일세이프를 조용히 푼다(감지기 3초
  쿨다운, dataflash 로그만). 해제는 **침묵으로만** 추론할 수 있다.
* **침묵은 모호하다.** `FS_LEAK_ENABLE=0`은 surface 동작만이 아니라 **경고 자체를**
  막고, `LEAK1_PIN=-1`(ArduSub 기본값)이면 감지기 백엔드가 아예 안 만들어진다.
  둘 중 하나면 **침수 중인 기체가 완전히 조용하다.** 그래서 설정을 못 읽은 상태의
  침묵을 `dry`라고 부르지 않는다 — `Telemetry.leak`은 3-상태고, 모르면 `None`이다
  (state.py 규칙: "모르는 것은 그럴듯한 기본값이 아니라 None").

문자열은 **정확히** 비교한다(`text.strip() == "Leak Detected"` + severity 2).
ArduSub에는 "leak"이 들어간 문자열이 둘 더 있는데(`"Leak detector 1 error. Please
set SERVO..."`, `"Leak detector 1 pin (servo 10) auto-set to GPIO"` — 소문자 d,
부팅 시 설정 알림), 부분일치로 매칭하면 **매 부팅이 침수 경보**가 된다.

화면에서:

* 헤더에 **빨간 `LEAK DETECTED` 배너**가 뜨고 깜빡인다. 정지가 아니라 깜빡임인 이유:
  헤더에는 이미 주황 배너와 색깔 pill 네 개가 있어서 정지된 빨강은 그냥 색 하나 더고,
  영상을 보고 있는 눈을 잡아채는 건 **움직임**이다(색각이상에서도 살아남는 유일한 단서).
* SENSORS의 **`Leak` 행**(맨 위로 올렸다 — "다이브 중단"을 뜻하는 행을 스크롤해야
  보이게 두면 아무도 안 본다)과, 그 아래 **`Enclosure` 행**(내부 기압, hPa).
* MISSION LOG에 `[ERROR]` 한 줄. **한 이벤트에 한 줄**이다 — 20초마다 반복되는 걸
  그대로 찍으면 로그가 같은 문장으로 가득 차서 아무도 안 읽는다.
* `FS_LEAK_ENABLE` / `LEAK1_PIN`을 커맨드 sink가 **읽어 온다**(파라미터 읽기는 이미
  `SYSID_MYGCS`용으로 있던 경로). 감지기가 안 켜져 있으면 부팅 시 로그가 그렇게 말하고
  `Leak` 행이 `detector DISABLED (FS_LEAK_ENABLE=0) — a flood would be silent`가 된다.

**내부 기압은 두 번째 신호**다. 별도 페일세이프가 `"Internal pressure critical!"`
(WARNING, 30초 주기, `FS_PRESS_MAX` 기본 105000 Pa = 1050 hPa)를 보낸다. 이걸 같이
보는 이유: 누수 패드는 **늦은** 신호다(그 지점까지 물이 닿아야 젖는다). 내부 기압은
씰이 풀리는 순간부터 움직인다. 절대값은 `SCALED_PRESSURE.press_abs`(id 29 = **내부**
기압계; `SCALED_PRESSURE2`(137)가 외부 Bar30 = 수심)에서 읽는다.

**누수가 스러스터를 끄지 않는다.** 이 기체는 설계상 음성 부력이라 추력을 끊으면
가라앉는다 — 누수는 **수면으로 올려서 회수할** 이유이지 멈출 이유가 아니다. ArduSub의
기본값(`FS_LEAK_ENABLE=1` = warn only)도 같은 판단이다. 크게 알리고 결정은 조종사에게
남긴다.

경보를 실제로 본 적이 없으면 믿을 수 없으므로, 데모에 `--demo-leak-after S`가 있다
(demo 전용 — 하드웨어 경로에는 이 플래그에서 오는 길이 없다).

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

오프라인 리플로터는 별도 파일이고 **Qt가 필요 없다**:

```bash
~/miniforge3/envs/robust/bin/python rov_gui/tests/test_plot_runs.py
```

32개. 합성 런 폴더를 만들어 datum 되올림이 `_to_map_xy`와 같은 산술인지(강체인지),
풀장 박스가 진짜 `geometry.pool_from_tags`와 같은 값인지, 두 런의 PNG가 같은
치수로 나오는지, 날짜 항목의 세 해상도(날짜 / 런 / CSV)가 각각 맞는 것만 잡는지,
PID 런의 파일 이름이 `pid`인지, **물체가 기체와 같은 변환을 타는지**(CSV의 고정
오프셋이 맵에서도 같은 거리로 남는지), 필터 셋이 각각 맞는 행을 빼는지, obj_* 열이
아예 없는 옛 런도 여전히 열리는지, `segment: auto`가 station/follow에서 engagement로
떨어지는지, feed 녹화 이름(`c3_depth_..._mpc`)에서도 시분초를 제대로 뽑는지, 그리고
**데이터가 없을 때 이유가 붙은 한 줄로 보고되는지**를 본다. 시간 램프는 눈이 아니라
**숫자로** 검사한다 — OKLab 스팬(> 45)과 사분위 간격의 균일함(편차 < 2.5), 그리고
표면 쪽 끝의 대비(>= 2:1).

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
