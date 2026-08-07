# rov_gui — BlueROV2 topside 관제 대시보드 (PyQt)

한 화면, 스크롤 없음. 영상 3개 + 텔레메트리 + 추진기 + 페이로드 + 텔레옵을
고정 `QGridLayout` 하나에 얹은 관제 스테이션이다. 패널은 하드웨어를 직접 만지지
않는다 — **백엔드**가 데이터 버스에 올린 것만 그린다. 그래서 같은 UI가 합성
데이터, 실제 C3 + ArduSub, ROS 2 토픽 위에서 그대로 돈다.

```
python -m rov_gui                    # 합성 데이터. 하드웨어를 열지 않는다
python -m rov_gui --source hw        # C3 + ArduSub (연결 전에 preflight)
python -m rov_gui --source ros2      # rclpy 토픽
./c3 gui                             # 래퍼가 인터프리터·cwd를 골라준다
```

`./c3 gui`를 권한다. `--source hw`는 `c3_camera`를 import하고 그건 **depthai
2.x**를 요구하는데, 이 데스크톱의 기본 python은 conda base(depthai 3.5.0)라서
Pipeline을 만드는 것만으로 카메라를 채갈 수 있다. 래퍼는 `robust` 인터프리터를
절대경로로 호출한다.

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
