# c3_camera — MarineSitu C3에 DepthAI로 직접 붙기

> **SLAM 데이터셋을 수집하려면** → [README_DATASET.md](README_DATASET.md)
> (`c3_collect.py`: TUM RGB-D 레이아웃 + 카메라 IMU(BNO086) + MAVLink 텔레메트리,
> 전부 하나의 타임라인). 이 문서는 카메라 직결·저지연·성능 실측을 다룬다.

BlueOS/Madrona의 H264 + RTSP 재전송 경로를 우회하고, 데스크톱에서 카메라에
**직접** XLink로 연결해서 RGB + depth를 받는다. 라즈베리파이를 영상 경로에서
완전히 빼는 것이 목적이다.

```
기존:  C3 → GigaBlox → RPi(Madrona: H264 인코딩 + RTSP 재서빙) → GigaBlox → fiber → Desktop
지금:  C3 → GigaBlox → fiber → Desktop        (DepthAI/XLink, 파이 경유 없음)
```

## 확인된 하드웨어 (추측이 아니라 실측)

`discover_c3.py --probe`가 카메라에서 직접 읽은 값:

| 항목 | 값 |
|---|---|
| product | **OAK-D-W-POE** (wide-FOV OAK-D PoE) |
| mx_id | `19443010315B0B2F00` @ 192.168.2.191 |
| bootloader | 0.0.28 |
| CAM_A | IMX378 컬러, native 4056x3040, **고정 초점** (autofocus 없음) |
| CAM_B / CAM_C | OV9282 mono (left/right), native 1280x800 |
| stereo baseline | 7.5 cm, EEPROM에 공장 캘리브레이션 있음 |
| IR 프로젝터 | 없음 (dot projector / flood LED 미탑재) |
| 컬러 지원 모드 | 1920x1080@60, 2024x1520@85, 1352x1012@52, 3840x2160@42, 4056x3040@30 |
| mono 지원 모드 | 640x400@255, 1280x720@143, 1280x800@129 |

`config.py`의 해상도 표는 위 실측값에서 나왔고, 런타임에
`validate_against_device()`가 다시 대조한다.

## 설치

```bash
bash c3_camera/setup_env.sh          # ~/.venvs/c3-depthai 생성 + 설치 + 검증
```

검증된 조합: **python 3.10.12 / depthai 2.32.0.0 / opencv 4.11.0 / numpy 1.26.4**
(Ubuntu 22.04, 시스템 python3 기반 venv)

### ⚠ conda base에서 그냥 돌리면 안 되는 이유 (중요)

이 데스크톱의 conda base는 **python 3.12.11이고 이미 `depthai 3.5.0`이 깔려
있다.** v3는 v2와 API가 다르고, 하필 **부분적으로만** 다르다:

| | conda base (depthai 3.5.0) |
|---|---|
| `dai.node.ColorCamera` | 있음 (shim) |
| `dai.node.XLinkOut` | **없음** |
| `Device.getOutputQueue` | **없음** |
| `dai.RawStereoDepthConfig` | **없음** |

즉 v2 코드가 파이프라인을 절반쯤 만들다가 깨진다. 더 나쁜 건 v3의
`dai.Pipeline(createImplicitDevice=True)`가 **기본값**이라서,
Pipeline을 생성하는 것만으로 디바이스를 찾아 열어버린다 — 이 네트워크에서는
**Madrona가 쓰고 있는 C3를 의도치 않게 낚아챌 수 있다는 뜻이다.**

그래서 `c3_camera`는 import 시점에 depthai가 2.x가 아니면 **바로 거부한다**:

```
c3_camera.DepthAIVersionError: c3_camera targets the depthai 2.x API but
found depthai 3.5.0 at /home/bdml/miniforge3/lib/python3.12/...
  Use the dedicated venv instead:
      ~/.venvs/c3-depthai/bin/python <script>
```

이 메시지가 보이면 venv 경로로 실행하면 된다.

### conda 라이브러리 충돌이 나면

venv를 쓰는 두 번째 이유는 conda가 자기 libstdc++/libGL/Qt를 들고 있어서
시스템 라이브러리와 충돌한 이력이다 (보통 `cv2.imshow`가 먼저 죽는다). venv는
X 세션이 이미 쓰는 시스템 라이브러리에 링크되므로 표시 경로가 그냥 동작한다.

venv의 절대 경로를 쓰면 conda가 활성화돼 있어도 대체로 문제없다. 그래도
`Could not load the Qt platform plugin "xcb"` 같은 게 나오면:

```bash
conda deactivate                     # 먼저 이걸 해라
~/.venvs/c3-depthai/bin/python c3_camera/c3_stream.py
```

그래도 안 되면:

```bash
unset QT_QPA_PLATFORM_PLUGIN_PATH    # conda가 심어놓은 Qt 경로 제거
QT_DEBUG_PLUGINS=1 ~/.venvs/c3-depthai/bin/python c3_camera/c3_stream.py  # 원인 확인
```

디스플레이 자체를 포기해도 측정은 된다: `--no-display`.
(이 환경에서 필요한 `libxcb-cursor0`/`libxcb-xinerama0`는 이미 설치돼 있고,
X11 세션(`DISPLAY=:1`)에서 실제로 창이 뜨는 것까지 확인했다.)

### import 시 자동으로 설정되는 DepthAI 환경변수

depthai는 환경변수를 **처음 읽을 때 캐시**하므로 `import depthai` 이후에
바꿔도 소용없다. 그래서 `c3_camera/__init__.py`가 submodule보다 먼저
`setdefault`로 깔아둔다 (명시적으로 지정한 값이 있으면 그게 우선):

| 변수 | 값 | 이유 |
|---|---|---|
| `DEPTHAI_PROTOCOL` | `tcpip` | PoE 카메라라 USB 탐색을 건너뛴다 |
| `DEPTHAI_BOOTUP_TIMEOUT` | `30000` ms | 실측 부팅 약 12초인데 기본값이 15초라 여유가 너무 얇다 |
| `DEPTHAI_CONNECT_TIMEOUT` | `15000` ms | 기본 5초 |
| `XLINK_LEVEL` | `warn` | 기본이 FATAL이라 PoE TCP 에러가 **숨겨진다** |

연결 실패를 파고들 때는 `DEPTHAI_LEVEL=debug XLINK_LEVEL=debug` 둘 다 켠다
(`DEPTHAI_LEVEL`만 켜면 XLink 소켓 에러가 안 보인다). 참고로
`DEPTHAI_LEVEL=fatal`은 **유효한 값이 아니라서 import에서 throw한다** —
`fatal`은 `XLINK_LEVEL`에만 있다.

## 쓰는 순서

```bash
V=~/.venvs/c3-depthai/bin/python

# 0. Madrona가 카메라를 쥐고 있는지 확인 (OAK는 파이프라인 1개만 허용)
$V -m c3_camera.madrona status

# 1. 카메라를 찾고 desktop에서 닿는지 확인
$V c3_camera/discover_c3.py --probe

# 2. 라이브 스트림 (RGB + depth, fps/latency HUD)
$V c3_camera/c3_stream.py

# 3. 해상도/fps 조합 스윕 → CSV
$V c3_camera/c3_bench.py --out bench.csv

# 4. 끝나면 Madrona 복구 (2번에서 껐다면)
$V -m c3_camera.madrona start
```

### Madrona를 꼭 멈춰야 하나

**카메라가 실제로 파이프라인을 쥐고 있을 때만.** 이 장비에서는 Madrona
컨테이너가 떠 있어도 카메라가 `X_LINK_BOOTLOADER`(=유휴) 상태인 경우가 있고,
그 상태면 확장을 건드리지 않고 그냥 붙는다. `discover_c3.py`가 `free` /
`OWNED` 를 명시적으로 알려주니 그걸 보고 판단한다.

`OWNED`면:

```bash
$V -m c3_camera.madrona stop --yes   # 확장 disable + 카메라가 풀릴 때까지 대기
```

`stop`은 **`--yes` 없이는 실행되지 않는다**. BlueOS의 disable은 영구 설정이라
재부팅해도 꺼진 채로 남고, BlueOS/Cockpit 쪽 영상이 계속 안 나오기 때문이다.
되돌리는 건 `madrona start` 하나뿐이니 잊지 말 것. 차량 설정을 스크립트로
건드리기 싫으면 BlueOS Extensions 페이지에서 손으로 꺼도 된다.

## 실측 성능

아래는 이 데스크톱–GigaBlox–C3 경로에서 `c3_bench.py`로 직접 측정한 값이다
(`--warmup` 이후 정상 구간만 집계).

### 기본 설정 결과 — 이 경로로 바꿔서 얻은 것

| | Madrona (H264/RTSP, 파이 경유) | **DepthAI 직결 (기본 설정)** |
|---|---|---|
| 컬러 latency (p50) | 약 **3000 ms** | **38 ms** |
| depth latency (p50) | — | **82 ms** |
| fps | 10 요청 → 6–8 실측 | **15 요청 → 15.00 실측** |
| 프레임 드롭 | — | **0 %** |

기본 설정 = `1080p → 960x540 MJPEG + 640x360 aligned depth @ 15 fps`,
420프레임 연속 측정, 실측 링크 71.6 Mbit/s.

**컬러 지연이 약 3000 ms → 38 ms.** 가설대로 파이 왕복이 병목이었다.

### PoE 링크 실효 대역폭 ≈ 91.5 Mbit/s — 여기가 유일한 진짜 제약

서로 다른 세 구성이 모두 같은 처리량에서 천장을 쳤다:

| 구성 | 실측 fps | x 프레임 크기 | = 처리량 |
|---|---|---|---|
| 컬러만 960x540 NV12 | 13.8 | 778 kB | 86 Mbit/s |
| 960x540 NV12 + depth | 9.25 | 1.24 MB | **91.7** Mbit/s |
| 1920x1080 NV12 + depth | 3.19 | 3.57 MB | **91.1** Mbit/s |
| 480x270 NV12 + depth | 17.5 | 655 kB | **91.6** Mbit/s |

전혀 다른 파이프라인이 같은 숫자로 수렴한다는 게 이게 **연산 한계가 아니라
링크 천장**이라는 증거다. PHY는 gigabit이지만 Myriad X의 XLink-over-TCP
경로가 거기까지 못 간다. 따라서 요청 fps가 아니라 이 식이 실제 fps를 정한다:

```
실제 fps ≈ 91.5 Mbit/s / (frame_bytes x 8)
```

`config.py`의 `POE_BUDGET_MBPS = 90`이 이 값(안전 여유 포함)이고,
`--dry-run`과 HUD가 요청 설정을 이 예산에 비춰 보여준다. HUD는 **추정치가
아니라 실측 Mbit/s**를 표시한다.

### raw vs MJPEG — 가장 큰 단일 개선

같은 해상도, 같은 20 fps 요청, depth 640x360 동반:

| 컬러 출력 | encode | 실측 fps | 컬러 p50 | depth p50 | drop |
|---|---|---|---|---|---|
| 480x270 | raw | 17.5 | 123 ms | 121 ms | 12.5 % |
| 480x270 | **mjpeg** | **20.0** | **57.6 ms** | 107 ms | **0 %** |
| 960x540 | raw | 9.1 | 250 ms | 405 ms | 53.5 % |
| 960x540 | **mjpeg** | **19.1** | **123 ms** | 124 ms | 5 % |
| 1920x1080 | raw | 3.2 | 353 ms | 495 ms | 83.5 % |
| 1920x1080 | **mjpeg** | **13.1** | **147 ms** | 154 ms | 34.8 % |

960x540에서 fps 2.1배, 지연 1/2. 풀 1080p에서는 fps 4.1배, 지연 1/2.4.
그래서 **MJPEG이 기본값이다.** 무손실 픽셀이 필요하면
`--color-encode none`(대신 약 9 fps)을 쓴다.

MJPEG 실측 압축률은 q90에서 **약 6:1** (960x540이 112–139 kB/frame). 처음에
10:1로 추정했다가 대역폭을 절반 가까이 과소평가했고, 그래서 지금은 측정값을
`MJPEG_RATIO`에 박아뒀다. 탁하거나 어두운 물은 고주파 성분이 적어 더 잘
압축되고, 맑은 물의 텍스처 있는 해저는 덜 압축된다 — 추정은 계획용이고
판단은 HUD의 실측값으로 한다.

### 대역폭을 줄이는 순서 (효과 큰 것부터)

1. **`--color-encode mjpeg`** — 디바이스에서 인코딩, 실측 약 1/6. 호스트 디코딩
   몇 ms. 위 표가 보여주듯 가장 효과가 크다.
2. **`--isp-scale`** — ISP에서 축소하므로 센서는 native 모드를 유지한다
   (= **화각이 유지된다**). 작은 센서 모드로 crop하면 화각이 좁아진다.
3. **`--fps`를 실제 통과량에 맞추기** — 못 통과할 프레임을 아예 만들지 않으므로
   지연 편차(p95)가 줄어든다. 드롭 38%로 9 fps 받는 것보다 9 fps 요청해서
   드롭 0%가 낫다.
4. **depth 쪽 줄이기** — 기본 설정에서 depth가 55 Mbit/s로 컬러(16)의 3.5배다.
   640x360 depth16 = 461 kB인데 MJPEG 컬러는 약 130 kB. 더 올리고 싶으면
   `--depth-size 480x270`이나 `--mono-fps`를 컬러보다 낮게 준다.

### 지연 = 고정 오버헤드 + 전송시간 + 큐잉

측정값을 전부 모아 보면 컬러 지연이 프레임 크기보다 **천장 대비 점유율**에
훨씬 깨끗하게 붙는다:

| 구성 | 컬러 kB/f | 실측 Mbit/s | 천장 대비 | 컬러 p50 | p95 |
|---|---|---|---|---|---|
| 640x360 mjpeg @12 + depth 480x270 | 67 | 31.8 | 35 % | **35.5 ms** | 36.1 |
| 640x360 mjpeg @20 + depth 480x270 | 67 | 52.0 | 58 % | **35.0 ms** | 35.8 |
| 960x540 mjpeg @15 + depth 640x360 **(기본)** | 131 | 71.4 | 78 % | 39 ms | 80 |
| 1920x1080 mjpeg @12 + depth 480x270 | 377 | 61.9 | 68 % | 77 ms | 84 |
| 480x270 mjpeg @20 + depth 640x360 | ~30 | ~77 | 84 % | 58 ms | 58 |
| 960x540 mjpeg @30 + depth **480x270** | 131 | 90.4 | 99 % | 84 ms | 96 |
| 960x540 mjpeg @30 + depth 640x360 | 131 | 88.2 | 96 % | 120 ms | — |
| 1920x1080 mjpeg @20 | 377 | 포화 | >100 % | 147 ms | 170 |
| 960x540 raw @20 | 778 | 포화 | >100 % | 250 ms | 284 |
| 1920x1080 raw @20 | 3111 | 포화 | >100 % | 353 ms | 395 |

점유율만으로는 설명이 안 된다 — 68 %(1080p, 77 ms)가 78 %(기본, 39 ms)보다
느리다. 세 항의 **합**으로 보면 전부 맞는다:

```
컬러 지연 ≈ 30 ms          (고정: 센서 readout + ISP + 인코딩)
          + frame_kB x 8 / 91.5 Mbit/s   (전송: 링크가 비어 있어도 드는 시간)
          + 큐잉             (천장에 가까워질수록 급증)
```

검산: 640x360(67 kB) → 30 + 6 = 36 vs 실측 35.0. 960x540(131 kB) → 30 + 11 = 41
vs 실측 39. 1080p(377 kB) → 30 + 33 = 63, 68 % 점유의 큐잉을 더해 실측 77.

그래서 실용 규칙은 둘이다:

> 1. **프레임을 작게** 하면 전송 항이 줄어든다 — 점유율과 무관하게 이득.
> 2. **천장의 60 % 아래**(약 55 Mbit/s)면 큐잉 항이 사라지고 지터도 없어진다
>    (p95−p50 < 1 ms). 그 위는 fps와 지연을 맞바꾸는 구간.
>
> 컬러 지연 바닥은 **약 30–35 ms**이고 그 아래로는 못 내려간다.
> `--dry-run`으로 미리 보고, HUD의 실측값으로 확인한다.

### 목적별 추천 설정

전부 실제로 측정한 값이다 (예측 아님):

| 목적 | 명령 (`c3_stream.py` 뒤에 붙임) | 실측 |
|---|---|---|
| **균형 (기본값)** | *(없음)* | 15.0 fps, 컬러 **39 ms**, drop 0 % |
| **최저 지연 + 지터 없음** | `--isp-scale 1/3 --fps 20 --depth-size 480x270` | 20.0 fps, 컬러 **35.0 ms** (p95 35.8, max 36.1), drop 0 %, 천장 58 % |
| **최고 fps** | `--fps 30 --depth-size 480x270` | **28.5 fps**, 84 ms, drop 5 %, 천장 99 % |
| **최대 화질 (풀 1080p)** | `--isp-scale none --fps 12 --depth-size 480x270` | 12.0 fps, **77 ms**, drop 0 %, 천장 68 % |
| **무손실 픽셀** | `--color-encode none --fps 9` | 약 9 fps, NV12 원본 |

시각 서보/텔레오퍼레이션이면 2번, 데이터 수집이면 4번, 정합이 중요하면
`--pair-mode timestamp`를 추가한다.

### Luxonis 문서의 지연 표에 대해

출발점이던 "PoE OAK는 4K 8 fps에서 150 ms, 10 fps에서 530 ms"는 방향은 맞지만
(해상도·fps를 낮추면 지연과 fps가 같이 개선된다) **숫자를 그대로 옮겨오지는
않았다.** 해당 표의 4K 행 대역폭 값이 같은 페이지의 자체 계산식과 어긋나고
(530/663 vs 796/846 Mbps), 측정 조건(인코딩 여부, ISP 출력 크기)이 특정되지
않는다. 그래서 위 숫자는 전부 **이 카메라·이 네트워크에서 `c3_bench.py`로 직접
측정한 값**이다. 케이블·스위치·광링크가 바뀌면 다시 재면 된다.

### 알아둘 것

- **연결에 약 12초** 걸린다. DepthAI가 PoE로 펌웨어를 올리고 부팅하는 시간이고
  실행당 1회 비용이다. 프레임 지연과는 무관하다.
- **depth 지연 > 컬러 지연** (82 vs 38 ms)은 정상이다. stereo 연산이 들어가고,
  프레임도 3.5배 크다.
- **컬러/depth skew 약 66 ms** (≈ 1 프레임). 컬러와 mono 쌍이 하드웨어 동기가
  아니라서 최대 한 프레임 간격까지 벌어진다. 정합이 중요하면
  `--pair-mode timestamp`를 쓴다 (최대 한 프레임 기다리는 대가).
- **공기 중 실내 테스트에서 depth valid ≈ 21%** 였다. 텍스처 없는 벽/역광 때문이고
  물속에서는 또 다르다. HUD의 `depth valid %`가 이걸 계속 보여준다.

## 저지연을 위해 실제로 한 것

| 항목 | 호출 | 이유 |
|---|---|---|
| 컬러를 NV12로 전송 | `ColorCamera.video` (not `preview`) | 1.5 B/px vs 3.0 — 같은 픽셀에 전송량 절반. BGR 변환은 어차피 호스트에서 한다 |
| XLink 청킹 해제 | `pipeline.setXLinkChunkSize(0)` | 프레임을 쪼개지 않고 한 번에 넘긴다 |
| 디바이스측 큐 1 + non-blocking | `XLinkOut.input.setQueueSize(1)`, `setBlocking(False)` | 링크가 밀리면 **오래된 프레임을 버린다**. 지연이 누적되지 않는다 |
| 호스트측 큐 1 + non-blocking | `getOutputQueue(name, 1, False)` | 같은 이유를 호스트에서 |
| 백로그 폐기 | `tryGetAll()` 후 마지막 것만 디코딩 | 쌓인 프레임을 재생하지 않고 현재만 본다 |
| spin 없는 대기 | `Device.getQueueEvents(...)` | 도착 시점에 깨어난다. 폴링 지연도, CPU 낭비도 없다 |
| ISP 축소 | `setIspScale(n, d)` | 화각 유지하면서 전송량 감소 |
| depth 단위 고정 | `initialConfig.setDepthUnit(MILLIMETER)` | 기본값에 의존하지 않고 계약을 코드로 명시 |
| timesync 강화 | `setTimesync(500ms, 10, True)` | latency 수치가 의미를 갖게 하는 전제 |

## latency를 어떻게 재는가

```
latency = dai.Clock.now() - frame.getTimestamp()
```

DepthAI가 디바이스에서 프레임에 타임스탬프를 찍고, timesync가 디바이스 시계를
호스트 `dai.Clock` 도메인에 맞춘다. 따라서 이건 **왕복이 아니라 단방향**
센서→호스트 구간이다.

- **포함**: 센서 readout, ISP, (depth는 stereo 연산), 디바이스측 큐,
  PoE XLink/TCP 전송, 호스트 큐 대기.
- **미포함**: 우리 쪽 컬러 변환, colorize, `cv2.imshow`. 이 잔여분은
  `Frame.age_ms()`가 커버하고 HUD에 같이 뜬다.
- `getTimestamp()`는 기본이 **노출 중앙** 기준이다. 어두운 물속에서 노출이
  길어지면 노출시간의 절반만큼 수치가 움직인다. 그래서 HUD가 노출시간도 같이
  보여준다.

**drop 판정**: 시퀀스 번호는 전송 전 디바이스에서 부여되므로, 번호에 구멍이
있으면 그 프레임은 만들어졌다가 버려진 것이다 (링크가 못 따라간 경우). "30 fps
나온다"가 실은 절반을 버린 결과인 경우를 이 값으로 구분한다.

## CLI 파라미터

`c3_stream.py`와 `c3_bench.py`가 같은 인자 표면을 공유한다 (`config.py`의
`add_stream_args`). 전체 목록은 `--help`.

주요 인자:

```
--color-res  1080p | 1352x1012 | 2024x1520 | 4k | 12mp      (IMX378 센서 모드)
--isp-scale  N/D 또는 none         예: 1/2 → 1080p가 960x540로
--color-encode mjpeg | none        기본 mjpeg (실측 약 1/6). none은 무손실 대신 약 9 fps
--mono-res   400p | 720p | 800p    (OV9282)
--fps / --mono-fps
--streams    color,depth[,left,right]
--depth-align color | right | left | center | none
--depth-preset robotics | high_accuracy | high_density | ...
--subpixel / --extended / --no-lr-check / --median / --confidence
--pair-mode  latest | timestamp    latest는 절대 기다리지 않는다(최저 지연)
--queue-size / --device-queue-size / --xlink-chunk / --small-pools
```

`c3_stream.py` 키: `q` 종료, **`v` 녹화 시작/정지**, `s` 스냅샷(PNG + depth
16-bit PNG + NPZ), `c` 포인트클라우드(PLY), `o` 뷰 전환(side/overlay/color/depth),
`p` 통계, `r` 통계 리셋, `[` `]` depth 컬러 범위, `h` HUD 토글.

## 녹화

```bash
# 켜자마자 녹화
c3py c3_camera/c3_stream.py --record

# 보면서 v 키로 시작/정지 (여러 구간을 나눠 담을 때)
c3py c3_camera/c3_stream.py
```

기본 저장 위치는 `c3_camera/recordings/<YYYYMMDD_HHMMSS>/` (`--record-dir`로 변경).

```
recordings/20260729_161500/
  meta.json      설정·디바이스·intrinsics·캘리브레이션·시계 오프셋
  frames.csv     프레임별 인덱스 → 파일명 + 촬영시각 + seq + latency
  color/000001.jpg    MJPEG일 때 재인코딩 없이 원본 그대로
  depth/000001.png    16-bit PNG, **밀리미터 무손실** (0 = 측정 없음)
```

설계상 지킨 것:

- **스트림을 절대 막지 않는다.** 16-bit PNG 인코딩이 몇 ms라 인라인으로 하면
  지연이 늘고 디바이스가 프레임을 버린다 — 이 클라이언트를 만든 이유가 사라진다.
  그래서 쓰기는 **별도 writer 스레드**가 하고, 큐가 차면 프레임을 버리되
  **개수를 세서 보고한다** (연구용 녹화에서 조용한 구멍이 최악이다).
- **depth는 무손실.** uint16 밀리미터라 손실 압축을 쓰면 측정값이 조용히 망가진다.
  더 빠르게 쓰려면 `--record-depth-format npy`(비압축, 대신 용량 약 2배).
- **MJPEG 컬러는 재인코딩하지 않는다.** 이미 JPEG 비트스트림으로 도착하니 그대로
  파일로 쓴다. 디코딩·재인코딩 없음 → CPU 거의 0, 추가 화질 손실 없음.

용량은 시작할 때 알려준다. 권장 설정(640x360 mjpeg @20 + depth 480x270)에서
대략 **150 MB/분 = 9 GB/시간** 수준.

### 녹화 확인 · 재생

**먼저 `--info`를 돌려보세요.** 녹화를 자기 자신과 대조 검증합니다 — 인덱스가
가리키는 파일이 다 있는지, depth가 정말 uint16 밀리미터인지, 디바이스 seq에
구멍이 없는지:

```bash
c3py c3_camera/c3_replay.py c3_camera/recordings/20260729_161500 --info
```

```bash
# 기록된 속도로 재생 (space 일시정지, . 다음, , 이전)
c3py c3_camera/c3_replay.py c3_camera/recordings/20260729_161500

# 공유용 MP4 (컬러+depth 나란히). 단 MP4의 depth는 컬러맵이라 데이터가 아닙니다
c3py c3_camera/c3_replay.py c3_camera/recordings/20260729_161500 --export-mp4 out.mp4
```

### 코드에서 녹화 읽기

```python
import csv, json, cv2
from pathlib import Path

d = Path("c3_camera/recordings/20260729_161500")
meta = json.loads((d / "meta.json").read_text())
rows = list(csv.DictReader((d / "frames.csv").open()))

for r in rows:
    colour = cv2.imread(str(d / r["color_file"]), cv2.IMREAD_COLOR)
    depth  = cv2.imread(str(d / r["depth_file"]), cv2.IMREAD_UNCHANGED)  # uint16 mm
    t      = float(r["color_t_device"])        # 촬영 시각 (dai.Clock 도메인)
```

`cv2.IMREAD_UNCHANGED`가 **필수**다. 빼면 OpenCV가 조용히 8-bit로 읽어서
모든 depth 값이 257배 틀어진다.

`t_device`는 monotonic 시계라 프로세스가 끝나면 절대시각 의미가 없어진다. 그래서
`meta.json`의 `clock`에 오프셋을 같이 저장한다:
`절대시각 = t_device - dai_now_s + time_time_s`.

## 코드에서 RGB-D 쓰기

```python
from c3_camera.config import StreamConfig
from c3_camera.source import C3Source

cfg = StreamConfig(color_res="1080p", isp_scale=(1, 2), mono_res="400p",
                   fps=10, depth_align="color", pair_mode="timestamp")

with C3Source(cfg) as src:
    for bundle in src.stream():
        colour = bundle.color.image        # HxWx3 uint8 BGR
        depth  = bundle.depth.image        # HxW  uint16, **millimetre**, 0 = 측정 없음
        skew   = bundle.skew_ms            # 컬러/depth 촬영 시점 차이
        xyz, rgb = bundle.pointcloud(stride=4)   # (N,3) float32 m, (N,3) uint8
```

`depth == 0`은 "거리 0"이 아니라 "측정 실패"다. 절대 유효값으로 흘리지 말 것.

포인트클라우드 좌표계는 OpenCV optical frame(+X 오른쪽, +Y 아래, +Z 전방)이다.
ROS body frame(+X 전방, +Y 왼쪽, +Z 위)으로는 고정 회전 하나면 되는데,
일부러 여기서 굽지 않았다.

## ROS 붙일 때

`Frame`이 ROS 퍼블리셔가 필요한 것만 들고 있고 ROS 의존성은 없다:

| Frame 필드 | ROS |
|---|---|
| `image` | `sensor_msgs/Image` (`bgr8`, depth는 `16UC1`) |
| `t_device` | `header.stamp` |
| `seq` | 디바이스 시퀀스 번호 (구멍 검출용) |
| `src.intrinsics[...]` | `sensor_msgs/CameraInfo` (스트리밍 해상도 기준 K, D) |

`header.stamp`에는 호스트 도착시각이 아니라 **`t_device`(촬영 시각)** 를 넣어야
한다. 링크가 한 번 밀렸을 때 TF 트리가 버티는지 아닌지가 여기서 갈린다.

## 수중에서의 단서

- 공장 캘리브레이션은 **공기 중** 캘리브레이션이다. 물속에서는 포트(평면/돔)가
  유효 초점거리를 바꾸고 방사 왜곡을 더한다. 물속 metric 작업에는 출발점일 뿐
  이고, 이 repo `src/`의 refractive 캘리브레이션 작업이 그대로 적용된다.
- baseline 7.5 cm + IR 프로젝터 없음 → 텍스처 없는 탁한 물에서는 stereo가
  구멍을 많이 낸다. HUD의 `depth valid %`로 바로 확인된다.
- 컬러는 고정 초점이다. 초점 관련 노브가 없는 게 정상이다.

## 문제 해결

| 증상 | 원인 | 조치 |
|---|---|---|
| `Could not connect to device with mx_id` / `X_LINK_DEVICE_NOT_FOUND` | 다른 파이프라인이 점유 중 (보통 Madrona) | `madrona status` → 필요하면 `madrona stop --yes`, 5–10초 대기 |
| 연결에 12초쯤 걸림 | 정상. PoE로 펌웨어 올리고 부팅하는 시간 | 실행당 1회 비용이지 프레임당 비용이 아니다 |
| 방금 닫고 다시 열면 실패 | PoE 디바이스가 리셋 후 bootloader로 돌아오는 중 | 5–10초 후 재시도 (`open_device`가 자동 재시도) |
| discovery는 실패하는데 TCP는 열림 | 스위치/방화벽/무관한 VPN이 broadcast를 막음 | `--ip`로 직접 연결 (broadcast 불필요) |
| fps가 요청보다 낮고 drop이 큼 | 링크 대역폭 한계 (~90 Mbit/s) | `--isp-scale` 축소, `--color-encode mjpeg`, `--fps` 하향 |
| depth가 대부분 0 | 텍스처 부족 / 너무 가까움 | `--extended`(근거리), `--confidence` 상향, 조명 |
| 녹화에서 `dropped > 0` | 디스크가 못 따라감 | `--record-depth-format npy`, `--record-queue` 상향, `--fps` 하향, 빠른 디스크. `frames.csv`의 seq 점프로 구멍 위치 확인 |
| 녹화한 depth 값이 이상함 | `cv2.imread`에 `IMREAD_UNCHANGED`를 안 씀 | 8-bit로 읽혀 값이 257배 틀어진다. 반드시 `cv2.IMREAD_UNCHANGED` |
| `Qt platform plugin "xcb"` | conda 라이브러리 충돌 | `conda deactivate` 후 재실행, `--no-display`로 우회 |

## 파일

| 파일 | 역할 |
|---|---|
| `config.py` | `StreamConfig` + 실측 capability 표 + 대역폭 예산 + CLI 인자 |
| `device.py` | discovery / 연결 (직접 IP, broadcast 불필요) + 진단 |
| `pipeline.py` | `StreamConfig` → 디바이스 파이프라인 |
| `source.py` | `C3Source` → `Bundle`/`Frame` (프로그램 인터페이스, ROS 이음매) |
| `metrics.py` | fps / latency / drop 집계 |
| `geometry.py` | intrinsics + depth(mm) → XYZ(m) |
| `viz.py` | depth colorize + HUD |
| `madrona.py` | BlueOS 확장 stop/start |
| `recorder.py` | 백그라운드 writer 스레드 녹화 (스트림을 막지 않음) |
| `discover_c3.py` | 탐색 스크립트 (deliverable 1) |
| `c3_stream.py` | 라이브 뷰어 + 녹화 (deliverable 2·3·4) |
| `c3_bench.py` | 해상도/fps 스윕 → CSV |
| `c3_replay.py` | 녹화 검증(`--info`) · 재생 · MP4 내보내기 |
| `tests/test_offline.py` | 카메라 없이 도는 회귀 테스트 46개 (역투영 수식, 해상도/대역폭 계산, CLI 파싱, H.264 추출, 녹화 무손실성) |
| `setup_env.sh` | venv 생성 + 설치 + 검증 |

```bash
# 카메라 없이 언제든 돌려볼 수 있는 검증
~/.venvs/c3-depthai/bin/python c3_camera/tests/test_offline.py
```
