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

## 설치와 실행 인터프리터

```bash
bash c3_camera/setup_env.sh          # ~/.venvs/c3-depthai 생성 + 설치 + 검증
./c3 env                             # 무엇이 잡혔는지, 버전은 뭔지 확인
```

이후 모든 도구는 래퍼로 부르면 된다 — 인터프리터도 작업 디렉터리도 래퍼가 책임진다:

```bash
./c3 collect --profile research_near
./c3 stream
./c3 test
```

검증된 조합 두 가지:

| | python | depthai | opencv | numpy | 상태 |
|---|---|---|---|---|---|
| conda `robust` | 3.12.13 | 2.32.0.0 | 4.10.0 | 1.26.4 | **기본** |
| `~/.venvs/c3-depthai` | 3.10.12 | 2.32.0.0 | 4.11.0 | 1.26.4 | 폴백 |

2026-08-06부터 `robust` 하나로 통합됐다 — 메인 코드(sim/SLAM)와 카메라가 같은
인터프리터를 쓴다. 그전까지 `robust`는 python 3.14였고 그 조합에서 numpy가 연산의
피연산자를 조용히 덮어썼기 때문에 venv가 1순위였다. 3.12로 내리면서 numpy 1.26.4가
소스빌드가 아닌 정식 휠로 깔렸고 그 문제는 사라졌다. `./c3 env`의 마지막 줄
`numpy elide`가 그 진단이며, 옛 환경은 `robust314_bad`로 보존돼 있다
(`KNOWN_ISSUES.md`: 그 기간에 나온 결과는 아직 감사 전이다).

venv는 지우지 않았다. `C3_PY=~/.venvs/c3-depthai/bin/python ./c3 ...` 로 언제든 쓸 수 있다.

conda 환경에서 미리보기 창(`cv2.imshow`)이 안 뜨는 문제는 `viz.py`가 자동으로
고친다 — cv2가 자기 Qt 플러그인 경로를 import 시점에 덮어쓰는데 `robust`에서는 그
디렉터리가 `plugins.disabled`로 꺼져 있어서 생기는 일이다.

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
| `DEPTHAI_BOOTUP_TIMEOUT` | `30000` ms | 실측 부팅 약 12초인데 기본값이 15초라 여유가 너무 얇다 [UNVERIFIED: 산출물 없음 — docs/MEASUREMENT_AUDIT.md] |
| `DEPTHAI_CONNECT_TIMEOUT` | `15000` ms | 기본 5초 |
| `XLINK_LEVEL` | `warn` | 기본이 FATAL이라 PoE TCP 에러가 **숨겨진다** |

연결 실패를 파고들 때는 `DEPTHAI_LEVEL=debug XLINK_LEVEL=debug` 둘 다 켠다
(`DEPTHAI_LEVEL`만 켜면 XLink 소켓 에러가 안 보인다). 참고로
`DEPTHAI_LEVEL=fatal`은 **유효한 값이 아니라서 import에서 throw한다** —
`fatal`은 `XLINK_LEVEL`에만 있다.

## 쓰는 순서

```bash
# 0. 리그가 준비됐나 — 카메라/ROV/디스크/디스플레이를 한 화면에서 (아무것도 안 엶)
./c3 preflight

# 1. 카메라를 찾고 센서·캘리브레이션까지 (여기서부터는 카메라를 엽니다)
./c3 discover --probe

# 2. 라이브 스트림 (RGB + depth, fps/latency HUD)
./c3 stream

# 3. 해상도/fps 조합 스윕 → CSV
./c3 bench --out bench.csv

# 4. 끝나면 Madrona 복구 (2번에서 껐다면)
./c3 madrona start
```

래퍼 없이 쓰려면 인터프리터를 직접 고르고 리포 루트에서 실행하면 된다 — 형태는 같다:

```bash
V=~/.venvs/c3-depthai/bin/python
$V -m c3_camera.c3_stream
```

## preflight — 어디서부터 잘못됐는지 먼저 말해준다 (`preflight.py`)

이 스택의 고장은 조종석에서 보면 전부 "안 되네"로 똑같이 보이지만 실제로는 서로 다른
다섯 군데에 살고, 어떤 건 12 s PoE 부팅을 기다린 뒤에야 나타난다. **하드웨어를 건드리는
모든 진입점은 연결 전에 이 검사를 먼저 돌린다.** 실패는 고치는 명령과 함께 출력된다.

| 검사 | 잡아내는 것 |
|---|---|
| `python + deps` | 잘못된 인터프리터(conda base = depthai 3.x), numpy operand 손상 probe |
| `host route` | VPN·엉뚱한 NIC로 트래픽이 새는 경우 |
| `camera link` | PoE·스위치·`192.168.2.1` — TCP 11490이 열려 있나 |
| `camera owner` | **누가** 쥐고 있나 (Madrona / 옆 터미널 c3 프로세스의 pid), mx_id가 그 C3가 맞나 |
| `BlueOS` · `ROV state` | 기체가 지금 살아있나 — armed/mode/전압, HEARTBEAT counter가 실제로 도는지 |
| `MAVLink endpoint` | BlueOS가 이 레코더 포트로 밀어주나 (QGC가 14550을 쥠) |
| `telemetry path` | 그 포트를 바인드할 수 있나, 패킷이 실제로 오나 |
| `disk` · `display` | 남은 공간, `imshow`가 창을 열 수 있나 |

```bash
./c3 preflight                              # 리포트만
./c3 preflight --json-out pf.json           # 기계가 읽는 형태
./c3 collect --preflight-only               # 그 도구의 설정 그대로, 검사만 하고 종료
./c3 collect --no-preflight                 # 건너뛰기
./c3 collect --preflight-strict             # 무인 실행: WARN도 중단 사유로
```

세 가지 설계 결정이 중요하다:

* **카메라를 절대 열지 않는다.** 전부 TCP / UDP 디스커버리 수준이라 밀리초가 들고 아무
  것에서도 카메라를 빼앗지 않는다. 여는 건 펌웨어 업로드 + 부팅(`device.TYPICAL_POE_BOOT_S`)에
  파이프라인 슬롯 하나를 점유하는 일이라, 그걸 하는 검사는 검사 대상보다 느리고 **그 자체가
  다음 도구가 실패하는 이유**가 된다. 센서 목록·캘리브레이션까지 필요하면
  `./c3 discover --probe`가 그 역할이다.
* **기체에 아무것도 보내지 않는다.** MAVLink 검사는 읽기만 한다.
* **원인은 한 번만 센다.** BlueOS가 죽어 있으면 그 아래 검사들이 같은 원인으로 각각
  타임아웃하지 않고 즉시 "위 참조"로 끝난다. 링크가 죽었을 때 `camera owner`도 두 번째
  중단 사유가 되지 않는다.

`WARN`은 진행을 막지 않는다 — 텔레메트리 없이 영상만 찍는 것도 정당한 실행이다. 막는
것은 `FAIL`(카메라 점유 · 링크 다운 · 디스크 부족)뿐이고, 그때 종료 코드는 `5`다.

`c3_collect`는 이 리포트를 **데이터셋 `metadata.json`에 그대로 남기고**, 깨끗하지 않았던
검사는 `metadata.txt`의 `== preflight ==` 절에도 적는다. 몇 달 뒤 폴더만 남았을 때
"telemetry.csv가 왜 비었지"의 답이 대개 거기 있다.

## 프로파일 — 플래그 벽 대신 YAML

`c3_collect`는 `--profile`을 받는다. `c3_camera/profiles/*.yaml`에 있고, 이름만 주면
된다:

```bash
./c3 collect --profile research_near              # 12개 플래그를 대체
./c3 collect --profile research_near --fps 30     # 플래그가 파일을 이긴다
./c3 collect --profile research_near --dry-run    # 값 + 출처를 전부 출력
```

우선순위는 낮은 것부터 **StreamConfig 기본값 < `--mode` 기본값 < 프로파일(부모→자식,
`extends:`) < 커맨드라인**이다. `--dry-run`이 설정마다 `default` / `mode:research` /
`profile:<파일>` / `cli` 중 어디서 왔는지 찍으므로 추측할 필요가 없다.

프로파일은 CLI가 노출하지 않는 `StreamConfig` 필드(`median`, `subpixel`,
`confidence`, `depth_preset`, `lr_check`, `pair_mode` …)도 설정할 수 있다. 오타·잘못된
섹션·`median: off`처럼 YAML이 불리언으로 읽어버리는 값은 로드 시점에 파일 이름과 함께
거부된다.

## 옵션 스윕 — 조합마다 SLAM 데이터셋을 남긴다 (`c3_option_sweep.py`)

`c3_bench.py`는 **픽셀을 안 남기기 때문에** "이 설정이 몇 fps 나오나"까지만
답한다. 실제로 궁금한 건 그 다음이다 — *그 설정으로 찍은 데이터로 SLAM이
돌아가나?* `c3_option_sweep.py`는 같은 축을 스윕하되, **셀마다
`datasets/dataset_*` 와 동일한 레이아웃의 완전한 TUM RGB-D 데이터셋**을 쓴다
(`rgb/ depth/ left/ right/`, `rgb.txt depth.txt associations.txt frames.csv`,
`calibration.json`, ORB-SLAM3 yaml, `imu_camera.csv`, MAVLink CSV,
`metadata.json/txt`). 그래서 설정을 fps가 아니라 **지도(map)** 로 판단할 수 있다.

스윕하는 네 축이 곧 네 개의 질문이다:

| 질문 | 옵션 |
|---|---|
| 컬러 해상도를 얼마로? | `--color-res` × `--isp-scale` |
| 압축할까 말까? | `--color-encode none,mjpeg,h264,h265` (+ `--mjpeg-quality`, `--video-bitrate-kbps`) |
| depth를 넣을까 말까? | `--depth on,off` |
| depth 해상도는? | `--depth-size match-color,derived,WxH` |

```bash
V=~/.venvs/c3-depthai/bin/python

# 0. 계획만 — 카메라를 건드리지 않고 조합·대역폭·디스크를 먼저 본다
$V c3_camera/c3_option_sweep.py --dry-run

# 1. 기본 사다리: 컬러 2크기 × raw/MJPEG × depth on/off × depth 2그리드 = 12셀
$V c3_camera/c3_option_sweep.py

# 2. "4K에서 압축 유무" 같은 구체적 질문
$V c3_camera/c3_option_sweep.py \
    --color-res 4k,1080p --isp-scale 2/3 --color-encode none,mjpeg \
    --mjpeg-quality 90 --depth on --depth-size match-color --fps 20

# 3. depth 해상도를 어디까지 깎아도 되나
$V c3_camera/c3_option_sweep.py --depth on --color-encode mjpeg \
    --depth-size match-color,derived,320x180
```

결과는 `c3_camera/sweeps/sweep_<타임스탬프>/` 아래에 셀별 디렉터리 + `results.csv`
+ `sweep_meta.json`. 각 셀 디렉터리는 그대로 SLAM에 넣으면 된다.

읽을 때 주의할 점 세 가지 (스크립트 docstring에도 있다):

- `*_mbps`는 **마지막 120프레임** 창, `*_mbps_run`이 캡처 전체 평균이다. 셀 간
  비교는 `*_mbps_run`으로 한다.
- "measured" 대역폭이 진짜 측정인 건 **압축 스트림뿐**이다. `--color-encode none`의
  NV12는 `w*h*1.5`, depth16은 `w*h*2`로 **계산된** 값이다 — 압축 축을 볼 때
  raw 행은 정의상 정확하고 MJPEG/H.26x 행만 실측이라는 뜻.
- `check` 열은 `c3_dataset_check.py` 판정인데 그건 RGB-D 검사기다. `--depth off`
  셀은 depth 전용 실패 2건을 빼고 `pass-mono`/`problems-mono`로, depth를 일부러
  컬러 그리드 밖에 둔 셀은 `-offgrid`를 붙여 보고한다.

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
| 컬러 latency (p50) | 약 **3000 ms** | **38 ms** [UNVERIFIED: 산출물 없음 — docs/MEASUREMENT_AUDIT.md] |
| depth latency (p50) | — | **82 ms** |
| fps | 10 요청 → 6–8 실측 | **15 요청 → 15.00 실측** [UNVERIFIED: 산출물 없음 — docs/MEASUREMENT_AUDIT.md] |
| 프레임 드롭 | — | **0 %** |

기본 설정 = `1080p → 960x540 MJPEG + 640x360 aligned depth @ 15 fps`,
420프레임 연속 측정, 실측 링크 71.6 Mbit/s.  [UNVERIFIED: 산출물 없음 — docs/MEASUREMENT_AUDIT.md]

**컬러 지연이 약 3000 ms → 38 ms.** 가설대로 파이 왕복이 병목이었다.

### PoE 링크 실효 대역폭 ≈ 91.5 Mbit/s — 여기가 유일한 진짜 제약

서로 다른 세 구성이 모두 같은 처리량에서 천장을 쳤다:

| 구성 | 실측 fps | x 프레임 크기 | = 처리량 |
|---|---|---|---|
| 컬러만 960x540 NV12 | 13.8 | 778 kB | 86 Mbit/s [UNVERIFIED: 산출물 없음 — docs/MEASUREMENT_AUDIT.md] |
| 960x540 NV12 + depth | 9.25 | 1.24 MB | **91.7** Mbit/s [UNVERIFIED: 산출물 없음 — docs/MEASUREMENT_AUDIT.md] |
| 1920x1080 NV12 + depth | 3.19 | 3.57 MB | **91.1** Mbit/s [UNVERIFIED: 산출물 없음 — docs/MEASUREMENT_AUDIT.md] |
| 480x270 NV12 + depth | 17.5 | 655 kB | **91.6** Mbit/s [UNVERIFIED: 산출물 없음 — docs/MEASUREMENT_AUDIT.md] |

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
| 480x270 | raw | 17.5 | 123 ms | 121 ms | 12.5 % [UNVERIFIED: 산출물 없음 — docs/MEASUREMENT_AUDIT.md] |
| 480x270 | **mjpeg** | **20.0** | **57.6 ms** | 107 ms | **0 %** [UNVERIFIED: 산출물 없음 — docs/MEASUREMENT_AUDIT.md] |
| 960x540 | raw | 9.1 | 250 ms | 405 ms | 53.5 % [UNVERIFIED: 산출물 없음 — docs/MEASUREMENT_AUDIT.md] |
| 960x540 | **mjpeg** | **19.1** | **123 ms** | 124 ms | 5 % [UNVERIFIED: 산출물 없음 — docs/MEASUREMENT_AUDIT.md] |
| 1920x1080 | raw | 3.2 | 353 ms | 495 ms | 83.5 % [UNVERIFIED: 산출물 없음 — docs/MEASUREMENT_AUDIT.md] |
| 1920x1080 | **mjpeg** | **13.1** | **147 ms** | 154 ms | 34.8 % [UNVERIFIED: 산출물 없음 — docs/MEASUREMENT_AUDIT.md] |

960x540에서 fps 2.1배, 지연 1/2. 풀 1080p에서는 fps 4.1배, 지연 1/2.4.  [UNVERIFIED: 산출물 없음 — docs/MEASUREMENT_AUDIT.md]
그래서 **MJPEG이 기본값이다.** 무손실 픽셀이 필요하면
`--color-encode none`(대신 약 9 fps)을 쓴다.  [UNVERIFIED: 산출물 없음 — docs/MEASUREMENT_AUDIT.md]

MJPEG 실측 압축률은 q90에서 **약 6:1** (960x540이 112–139 kB/frame). 처음에  [UNVERIFIED: 산출물 없음 — docs/MEASUREMENT_AUDIT.md]
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
   지연 편차(p95)가 줄어든다. 드롭 38%로 9 fps 받는 것보다 9 fps 요청해서  [UNVERIFIED: 산출물 없음 — docs/MEASUREMENT_AUDIT.md]
   드롭 0%가 낫다.
4. **depth 쪽 줄이기** — 기본 설정에서 depth가 55 Mbit/s로 컬러(16)의 3.5배다.
   640x360 depth16 = 461 kB인데 MJPEG 컬러는 약 130 kB. 더 올리고 싶으면
   `--depth-size 480x270`이나 `--mono-fps`를 컬러보다 낮게 준다.

### 지연 = 고정 오버헤드 + 전송시간 + 큐잉

측정값을 전부 모아 보면 컬러 지연이 프레임 크기보다 **천장 대비 점유율**에
훨씬 깨끗하게 붙는다:

| 구성 | 컬러 kB/f | 실측 Mbit/s | 천장 대비 | 컬러 p50 | p95 |
|---|---|---|---|---|---|
| 640x360 mjpeg @12 + depth 480x270 | 67 | 31.8 | 35 % | **35.5 ms** | 36.1 [UNVERIFIED: 산출물 없음 — docs/MEASUREMENT_AUDIT.md] |
| 640x360 mjpeg @20 + depth 480x270 | 67 | 52.0 | 58 % | **35.0 ms** | 35.8 [UNVERIFIED: 산출물 없음 — docs/MEASUREMENT_AUDIT.md] |
| 960x540 mjpeg @15 + depth 640x360 **(기본)** | 131 | 71.4 | 78 % | 39 ms | 80 [UNVERIFIED: 산출물 없음 — docs/MEASUREMENT_AUDIT.md] |
| 1920x1080 mjpeg @12 + depth 480x270 | 377 | 61.9 | 68 % | 77 ms | 84 [UNVERIFIED: 산출물 없음 — docs/MEASUREMENT_AUDIT.md] |
| 480x270 mjpeg @20 + depth 640x360 | ~30 | ~77 | 84 % | 58 ms | 58 [UNVERIFIED: 산출물 없음 — docs/MEASUREMENT_AUDIT.md] |
| 960x540 mjpeg @30 + depth **480x270** | 131 | 90.4 | 99 % | 84 ms | 96 [UNVERIFIED: 산출물 없음 — docs/MEASUREMENT_AUDIT.md] |
| 960x540 mjpeg @30 + depth 640x360 | 131 | 88.2 | 96 % | 120 ms | — [UNVERIFIED: 산출물 없음 — docs/MEASUREMENT_AUDIT.md] |
| 1920x1080 mjpeg @20 | 377 | 포화 | >100 % | 147 ms | 170 [UNVERIFIED: 산출물 없음 — docs/MEASUREMENT_AUDIT.md] |
| 960x540 raw @20 | 778 | 포화 | >100 % | 250 ms | 284 [UNVERIFIED: 산출물 없음 — docs/MEASUREMENT_AUDIT.md] |
| 1920x1080 raw @20 | 3111 | 포화 | >100 % | 353 ms | 395 [UNVERIFIED: 산출물 없음 — docs/MEASUREMENT_AUDIT.md] |

점유율만으로는 설명이 안 된다 — 68 %(1080p, 77 ms)가 78 %(기본, 39 ms)보다  [UNVERIFIED: 산출물 없음 — docs/MEASUREMENT_AUDIT.md]
느리다. 세 항의 **합**으로 보면 전부 맞는다:

```
컬러 지연 ≈ 30 ms          (고정: 센서 readout + ISP + 인코딩)
          + frame_kB x 8 / 91.5 Mbit/s   (전송: 링크가 비어 있어도 드는 시간)
          + 큐잉             (천장에 가까워질수록 급증)
```

검산: 640x360(67 kB) → 30 + 6 = 36 vs 실측 35.0. 960x540(131 kB) → 30 + 11 = 41  [UNVERIFIED: 산출물 없음 — docs/MEASUREMENT_AUDIT.md]
vs 실측 39. 1080p(377 kB) → 30 + 33 = 63, 68 % 점유의 큐잉을 더해 실측 77.

그래서 실용 규칙은 둘이다:

> 1. **프레임을 작게** 하면 전송 항이 줄어든다 — 점유율과 무관하게 이득.
> 2. **천장의 60 % 아래**(약 55 Mbit/s)면 큐잉 항이 사라지고 지터도 없어진다
>    (p95−p50 < 1 ms). 그 위는 fps와 지연을 맞바꾸는 구간.  [UNVERIFIED: 산출물 없음 — docs/MEASUREMENT_AUDIT.md]
>
> 컬러 지연 바닥은 **약 30–35 ms**이고 그 아래로는 못 내려간다.
> `--dry-run`으로 미리 보고, HUD의 실측값으로 확인한다.

### 목적별 추천 설정

전부 실제로 측정한 값이다 (예측 아님):  [UNVERIFIED: 산출물 없음 — docs/MEASUREMENT_AUDIT.md]

| 목적 | 명령 (`c3_stream.py` 뒤에 붙임) | 실측 |
|---|---|---|
| **균형 (기본값)** | *(없음)* | 15.0 fps, 컬러 **39 ms**, drop 0 % [UNVERIFIED: 산출물 없음 — docs/MEASUREMENT_AUDIT.md] |
| **최저 지연 + 지터 없음** | `--isp-scale 1/3 --fps 20 --depth-size 480x270` | 20.0 fps, 컬러 **35.0 ms** (p95 35.8, max 36.1), drop 0 %, 천장 58 % [UNVERIFIED: 산출물 없음 — docs/MEASUREMENT_AUDIT.md] |
| **최고 fps** | `--fps 30 --depth-size 480x270` | **28.5 fps**, 84 ms, drop 5 %, 천장 99 % [UNVERIFIED: 산출물 없음 — docs/MEASUREMENT_AUDIT.md] |
| **최대 화질 (풀 1080p)** | `--isp-scale none --fps 12 --depth-size 480x270` | 12.0 fps, **77 ms**, drop 0 %, 천장 68 % [UNVERIFIED: 산출물 없음 — docs/MEASUREMENT_AUDIT.md] |
| **무손실 픽셀** | `--color-encode none --fps 9` | 약 9 fps, NV12 원본 [UNVERIFIED: 산출물 없음 — docs/MEASUREMENT_AUDIT.md] |

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

- **연결에 약 12초** 걸린다. DepthAI가 PoE로 펌웨어를 올리고 부팅하는 시간이고  [UNVERIFIED: 산출물 없음 — docs/MEASUREMENT_AUDIT.md]
  실행당 1회 비용이다. 프레임 지연과는 무관하다.
- **depth 지연 > 컬러 지연** (82 vs 38 ms)은 정상이다. stereo 연산이 들어가고,
  프레임도 3.5배 크다.
- **컬러/depth skew 약 66 ms** (≈ 1 프레임). 컬러와 mono 쌍이 하드웨어 동기가
  아니라서 최대 한 프레임 간격까지 벌어진다. 정합이 중요하면
  `--pair-mode timestamp`를 쓴다 (최대 한 프레임 기다리는 대가).
- **공기 중 실내 테스트에서 depth valid ≈ 21%** 였다. 텍스처 없는 벽/역광 때문이고  [UNVERIFIED: 산출물 없음 — docs/MEASUREMENT_AUDIT.md]
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
대략 **150 MB/분 = 9 GB/시간** 수준.  [UNVERIFIED: 예시/유도값 — docs/MEASUREMENT_AUDIT.md]

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

- 공장 캘리브레이션은 **공기 중** 캘리브레이션이다. 물속에서는 포트(평면/돔)가  [UNVERIFIED: 예시/유도값 — docs/MEASUREMENT_AUDIT.md]
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
| `c3_bench.py` | 해상도/fps 스윕 → CSV (픽셀은 안 남김) |
| `c3_option_sweep.py` | **옵션 스윕 → 조합마다 SLAM 데이터셋 1개 + 비교 CSV** (아래 참조) |
| `c3_replay.py` | 녹화 검증(`--info`) · 재생 · MP4 내보내기 |
| `tests/test_offline.py` | 카메라 없이 도는 회귀 테스트 47개 (역투영 수식, 해상도/대역폭 계산, CLI 파싱, H.264 추출, 녹화 무손실성) |
| `setup_env.sh` | venv 생성 + 설치 + 검증 |

```bash
# 카메라 없이 언제든 돌려볼 수 있는 검증
~/.venvs/c3-depthai/bin/python c3_camera/tests/test_offline.py
```


<!-- MEASUREMENT AUDIT (2026-08-04): the following numbers appear inside code
     blocks above and could not be annotated inline without corrupting them. -->

> **[UNVERIFIED]** 아래 줄의 수치는 출처 감사에서 미검증으로 분류되었다 — 자세한 근거는 [docs/MEASUREMENT_AUDIT.md](docs/MEASUREMENT_AUDIT.md).
>
> - `c3_camera/README.md:306` — 예시/유도값 — `컬러 지연 ≈ 30 ms          (고정: 센서 readout + ISP + 인코딩)`
> - `c3_camera/README.md:405` — 산출물 없음 — `--color-encode mjpeg | none        기본 mjpeg (실측 약 1/6). none은 무손실 대신 약 9 fps`
