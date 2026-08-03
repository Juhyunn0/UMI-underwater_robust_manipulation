# c3_collect — 수중 RGB-D / visual SLAM 데이터셋 수집

C3 카메라를 DepthAI로 직결(Madrona 우회)해서 잡고, 동시에 BlueROV2의 MAVLink
텔레메트리를 기록해서, **전부 하나의 타임라인 위에** TUM RGB-D 레이아웃으로 쓴다.

카메라 직결 자체(지연 3000 ms → 35 ms)와 그 실측 근거는
[README.md](README.md)에 있다. 이 문서는 **데이터셋 수집** 부분만 다룬다.

> 각 데이터셋 폴더의 `metadata.txt`는 **영어로** 쓰인다. depth_scale·타임스탬프·
> IMU 주의사항을 전부 담고 있으니 동료에게 폴더째로 넘기면 된다.

## 0. 시작하기 전에: Madrona를 멈춰야 하는지 확인

OAK 디바이스는 **파이프라인을 하나만** 허용한다. Madrona가 카메라를 쥐고 있으면
`Could not connect to device with mx_id`가 난다. 단, **컨테이너가 떠 있어도
카메라가 유휴(`X_LINK_BOOTLOADER`)일 수 있다** — 이 경우 아무것도 안 건드리고 붙는다.

```bash
V=~/.venvs/c3-depthai/bin/python

$V c3_camera/discover_c3.py --probe        # free / OWNED 를 명시적으로 알려준다
```

`OWNED`로 나오면:

```bash
$V -m c3_camera.madrona stop --yes         # BlueOS 확장 disable + 카메라 해제 대기
# ... 촬영 후 반드시 복구
$V -m c3_camera.madrona start
```

`stop`은 `--yes` 없이는 안 된다. BlueOS의 disable은 영구 설정이라 재부팅해도
꺼진 채 남고, BlueOS/Cockpit 영상이 계속 안 나온다. GUI로 하려면 BlueOS
Extensions 페이지에서 꺼도 된다.

`c3_collect.py`는 이 상황을 감지해서 위 명령을 그대로 출력한다.

## 1. QGroundControl과 동시에 쓰기 — 반드시 읽을 것

실측한 문제다. BlueOS는 MAVLink를 `udpout → 192.168.2.1:14550`으로 **밀어주고**,
QGroundControl이 데스크톱에서 그 포트를 잡고 있다. **같은 유니캐스트 UDP
데이터그램을 두 프로세스가 함께 받을 수는 없다.**

- `SO_REUSEPORT`로 14550을 두 번 bind하면 리눅스는 데이터그램을 **복제하지 않고
  분배**한다 → 레코더가 QGC 패킷을 절반쯤 훔쳐가고, 조종 링크가 조용히 나빠진다.
  **하지 말 것.**

선택지 세 개, 권장 순서대로:

### (1) BlueOS에 두 번째 엔드포인트 추가 — 권장

```bash
$V -m c3_camera.blueos_endpoint list                    # 현재 라우팅 확인
$V -m c3_camera.blueos_endpoint add --port 14551 --yes   # udpout → 192.168.2.1:14551
$V c3_camera/c3_collect.py --mavlink udpin:0.0.0.0:14551 --mode research
```

전체 스트림을 최고 레이트로 받는다. `--persistent`가 기본이라 한 번만 하면 된다.
단 **mavlink-router 설정이 다시 로드되면서 QGC 링크가 순간 끊길 수 있으니
다이빙 전에** 해라. (`c3_collect.py --setup-endpoint`가 대신 해줄 수도 있다.)

### (2) MAVLink2Rest — 차량 설정 무변경

```bash
$V c3_camera/c3_collect.py --mavlink-transport rest --mode research
```

BlueOS에 이미 돌고 있는 MAVLink2Rest(포트 6040)를 폴링한다. **차량 설정을 전혀
건드리지 않고 QGC와 충돌도 없다.** 대신 각 메시지의 *최신 값*을 주는 구조라
모든 메시지를 캡처하는 게 아니라 **샘플링**한다. attitude/압력/배터리 맥락용으로는
충분하고, 고레이트 IMU 스트림 대용은 아니다. (어차피 VIO용 IMU는 카메라의
BNO086이므로 실무상 큰 손실은 아니다.)

### (3) 텔레메트리 없이

```bash
$V c3_camera/c3_collect.py --mavlink-transport none
```

## 2. 쓰는 법

```bash
# 설정만 계산해 보기 (카메라 안 건드림, 12초 부팅도 안 함)
$V c3_camera/c3_collect.py --mode research --dry-run

# 긴 탐색 다이빙: mp4 + 텔레메트리만
$V c3_camera/c3_collect.py --mode review

# SLAM 데이터셋
$V c3_camera/c3_collect.py --mode research

# 20 fps RGB-D: RGB는 온디바이스 H.264, depth는 무손실 uint16 PNG
$V c3_camera/c3_collect.py --mode research --mavlink-transport rest \
    --streams color,depth --color-encode h264 --video-bitrate-kbps 4000 \
    --isp-scale 1/4 --depth-size 480x270 --fps 20 \
    --extract-encoded-rgb

# 녹화 후 검증 (동료에게 넘기기 전에 반드시)
$V c3_camera/c3_dataset_check.py c3_camera/datasets/dataset_20260729_161500
```

**SPACE**로 녹화 시작/정지 → 먼저 잠수하고 원하는 구간만 담을 수 있다.
`q` 종료, `Ctrl-C`도 안전하다(writer flush → mp4 finalize → metadata 기록).
헤드리스면 `--no-display --record-now`, 정지는 `Ctrl-C`.

### 20 fps에서 H.264/H.265 RGB + 무손실 depth

`c3_collect.py`의 research 기본값은 RGB를 raw NV12로 전송한다. 이제
`--color-encode h264|h265`를 주면 C3의 `VideoEncoder`가 RGB만 하드웨어
압축하고, depth는 이전과 똑같이 **uint16 millimetre PNG로 무손실 저장**한다.
`--video-bitrate-kbps`의 기본값은 4000이고, keyframe은 기본 1초마다 들어간다.

H.264/H.265는 프레임 간 코덱이라 촬영 중에는 `rgb.h264`/`rgb.h265`와
프레임별 캡처 시각을 담은 `rgb_video.csv`로 기록한다. `--extract-encoded-rgb`를
주면 종료 후 FFmpeg로 `rgb/<timestamp>.png`를 만들어 기존 TUM index가 바로
가리키게 한다. 나중에 수동으로 해도 된다:

```bash
$V c3_camera/c3_extract_rgb_video.py c3_camera/datasets/dataset_YYYYMMDD_HHMMSS
```

**20 fps에서는 `--streams color,depth`가 중요하다.** left/right 원본 두 장을
같이 XLink로 보내면 그것들만 약 82 Mbit/s이고, 480x270 depth 약 41 Mbit/s까지
더해져 RGB를 완전히 공짜로 압축해도 이미 PoE 한도를 넘는다. left/right 전송을
빼도 StereoDepth는 디바이스 내부에서 두 mono 카메라를 계속 사용하므로 depth는
그대로 동작한다. 포기하는 것은 나중에 stereo disparity를 재계산할 raw mono
파일뿐이며 RGB-D SLAM에는 영향이 없다.

480x270 depth @20 + H.264 4 Mbit/s + RGB만 전송하는 설정의 예산은 약
**45 Mbit/s**다. IMU용 여유가 충분하다. H.264/H.265 RGB는 lossy이므로
photometric calibration 데이터에는 쓰지 말고, 특징점 기반 SLAM에서도 bitrate를
너무 낮추면 block artifact가 특징점을 해칠 수 있다.

### 두 모드

| | review | research |
|---|---|---|
| 용도 | 긴 탐색 다이빙 | SLAM 데이터셋 |
| 영상 | `rgb.mp4`만 | 무손실 PNG 4스트림 **+** 참고용 `rgb.mp4` |
| 기본 설정 | 960x540 MJPEG @15 | 480x270 무손실 @8, depth 1:1 |
| 텔레메트리 | CSV | CSV |
| 용량 | 약 0.5 GB/시간 | 약 21 GB/시간 |

## 3. 실측 성능 (research 모드)

기본 설정 480x270 컬러+depth + 640x400 스테레오 @8 fps, 20초 실측:

| 항목 | 값 |
|---|---|
| 프레임 | 8.0 fps 요청 → **8.0 실측, 드롭 0** |
| 이미지 | 프레임당 4장 무손실 PNG |
| 링크 | 62 Mbit/s (실측 천장 91.5의 69%) |
| 디스크 | 약 11–13 MB/s (약 21 GB/시간) |
| 지연 | 컬러 p50 88 ms, depth 89 ms |
| 카메라 IMU | 200 Hz 요청 → **약 130 Hz 실측** (아래 참고) |

**병목은 디스크가 아니라 PoE 링크다.** 브리프에서 디스크를 걱정하셨지만, 이
NVMe는 링크가 줄 수 있는 ~11 MB/s보다 수십 배 빠르다. PNG 인코딩도 프레임당
17 ms(4장 합계)로 writer 스레드 3개면 남는다. 무손실 4스트림의 fps 상한은
링크가 정한다:

| 해상도 (컬러=depth, 모노 640x400) | 링크 상한 fps |
|---|---|
| 480x270 | 11.8 |
| 640x360 | 8.7 |
| 960x540 | 4.9 |

더 높은 fps가 필요하면 `--streams color,depth`로 스테레오를 빼거나(14 fps),
해상도를 내린다.

## 4. 출력 형식

```
dataset_YYYYMMDD_HHMMSS/
├── rgb/     <timestamp>.png     무손실 8-bit BGR
├── depth/   <timestamp>.png     무손실 16-bit, 밀리미터, RGB와 1:1 정렬
├── left/    <timestamp>.png     무손실 모노 (raw)
├── right/   <timestamp>.png     무손실 모노 (raw)
├── rgb.txt          # timestamp filename   (TUM)
├── depth.txt        # timestamp filename   (TUM)
├── associations.txt rgb_ts rgb_file depth_ts depth_file
├── frames.csv       프레임 인덱스 + 스트림별 device/host 시각 + skew + latency
├── imu_camera.csv   C3 BNO086  ← VIO용으로 이걸 쓸 것
├── imu_rov.csv      ArduSub IMU (MAVLink)
├── telemetry.csv    attitude, 압력/깊이, 배터리, servo, 모드
├── control.csv      조종 입력 (RC_CHANNELS / MANUAL_CONTROL)
├── calibration.json 온디바이스 공장 캘리브레이션
├── orbslam3_rgbd.yaml  실제 intrinsics로 생성된 ORB-SLAM3 설정 초안
├── rgb.mp4          참고용 미리보기 (손실 압축, 데이터 아님)
└── metadata.txt/.json  depth_scale·시계·IMU 주의사항 전부 (영어)
```

- **depth는 절대 비디오로 안 넣는다.** 16-bit PNG 무손실(`--record-depth-format
  npy`도 가능). 손실 압축은 밀리미터 값을 망친다.
- **left/right도 PNG 무손실.** 동료가 disparity를 다시 계산하거나 stereo SLAM을
  돌릴 수 있어야 하고, 압축 아티팩트는 매칭을 망친다.
- 컬러 전송은 기본 NV12(1.5 B/px, 4:2:0 크로마). PNG 저장은 무손실이지만
  **크로마는 전송 단계에서 서브샘플됨** — 휘도는 풀 해상도라 특징점 기반 SLAM에는
  무영향. 풀 크로마가 필요하면 `--color-wire bgr`(대역폭 2배).

### depth_scale — 브리프의 숫자를 정정합니다

브리프에 "depth_scale은 밀리미터면 5000.0"이라고 쓰여 있는데 **이건 틀렸습니다.**
그대로 쓰면 동료의 SLAM에서 모든 깊이가 **5배 작게** 나오고, 망가진 게 아니라
그럴듯하게 보여서 발견도 안 됩니다.

```
metres = png_pixel_value / 1000.0        ← 우리 데이터 (uint16 밀리미터)
```

`5000`은 TUM 자기 시퀀스가 Kinect 데이터를 그렇게 스케일해서 저장했기 때문에
쓰는 값이지 규약이 아닙니다. 도구별 파라미터 이름과 우리 값:

| 도구 | 파라미터 | 값 |
|---|---|---|
| ORB-SLAM3 | `RGBD.DepthMapFactor` | **1000.0** |
| Open3D | `depth_scale` (나누는 방향) | **1000.0** |
| RTAB-Map | 16-bit mm 기본 인식, 별도 파라미터 불필요 | — |

생성되는 `orbslam3_rgbd.yaml`에 이미 1000.0으로 들어갑니다.

또 하나: **`depth == 0`은 "거리 0"이 아니라 "측정 실패"** 입니다. 마스킹하세요.
그리고 읽을 때 **`cv2.IMREAD_UNCHANGED` 필수** — 빼면 OpenCV가 조용히 8-bit로
읽어서 값이 257배 틀어집니다.

## 5. 타임스탬프 — 동료에게 설명할 내용

**측정 결과 이 호스트에서 `dai.Clock.now()`와 `time.monotonic()`은 같은 시계
(CLOCK_MONOTONIC)이고 8 마이크로초 이내로 일치합니다.** 따라서 카메라 프레임,
카메라 IMU, MAVLink 수신 시각이 **변환 없이 이미 한 타임라인 위에** 있습니다.

파일에 쓰이는 값은 전부 **Unix epoch 초, 소수점 6자리**이고, 세션 시작 시 한 번
측정한 상수 오프셋으로 변환됩니다 (`metadata.txt`의 `clock_offset_...`):

```
t_unix = t_monotonic + clock_offset_monotonic_to_unix
```

`frames.csv`에는 원시 monotonic 값도 같이 남으므로 잃는 정보가 없습니다.

**한 가지 비대칭은 반드시 알려주세요:**

| | 타임스탬프의 의미 |
|---|---|
| `rgb/`, `depth/`, `left/`, `right/`, `imu_camera.csv` | **촬영 시각** (디바이스에서 찍고 timesync로 호스트 도메인 유지) |
| `imu_rov.csv`, `telemetry.csv`, `control.csv` | **호스트 도착 시각** — 실제 측정 시점보다 링크·라우팅 지연(이 이더넷 경로에서 수 ms)만큼 늦음 |

수 ms보다 정밀해야 하면 각 MAVLink 행에 ArduSub 자신의 `time_boot_ms`가 있으니
`t_unix`에 대해 회귀해서 오토파일럿 시계를 복원하면 됩니다.

## 6. IMU — 어느 걸 쓸지, 그리고 두 가지 경고

**C3에는 온보드 IMU가 있습니다: BNO086** (`getConnectedIMU()`로 확인, firmware
3.9.9). 카메라와 강체로 붙어 있고 타임스탬프가 이미지와 같은 시계라
**VIO에는 이걸 쓰는 게 맞습니다.** ROV의 Navigator IMU는 수십 cm 떨어져 있고
다른 보드에서 UDP로 도착 시각만 찍혀 옵니다 — 맥락·교차검증용입니다.

### 실측한 레이트 규칙

| 설정 | 실측 |
|---|---|
| accel+gyro @500 요청 | **486 Hz** |
| accel+gyro @200 | 194 Hz |
| accel+gyro @200 **+ rotvec @100** | 97 Hz (전부 반토막) |
| accel+gyro @200 **+ mag @100** | 97 Hz (전부 반토막) |
| accel+gyro @200 **+ rotvec @200** | **194 Hz** (손실 없음) |

**규칙 1: 켜 놓은 센서 중 가장 느린 것의 레이트가 전부에 적용된다.** 그래서
`--imu-rotation-vector`는 같은 레이트로 요청하므로 공짜지만,
`--imu-magnetometer`는 100 Hz가 상한이라 **IMU 전체를 100 Hz로 묶습니다.**

**규칙 2: 배칭이 손실을 결정한다.** 무손실 4스트림을 동시에 녹화하면서:

| `--imu-batch` | 200 Hz 요청 결과 |
|---|---|
| 1 | 130 Hz 수신, **35% 손실** |
| 10 (기본) | **199.5 Hz 수신, 손실 0** |

원인은 대역폭이 아니라 **메시지당 XLink 오버헤드**였습니다. 리포트 1개당 메시지
1개로는 이미지와 경쟁할 때 못 버팁니다. 배치 1은 지연이 최소라 라이브 뷰에
맞고, 데이터셋은 완결성이 중요합니다 — **IMU 스트림에 구멍이 나면 VIO
preintegration이 깨집니다.** 그래서 `c3_collect.py` 기본값이 10입니다.

`metadata.txt`에는 **요청값이 아니라 실측값**이 기록되므로, 요청이 지켜졌는지
동료가 신뢰할 필요가 없습니다.

### ⚠ 경고 1: IMU-카메라 extrinsic이 디바이스에 없습니다

`getImuToCameraExtrinsics()` → `IMU calibration data is not available on device yet.`
공장 캘리브레이션은 카메라만 담고 있어서 **T_imu_cam이 미지**입니다. VIO를
돌리려면 동료가 Kalibr류로 캘리브레이션하거나 기계적으로 실측해야 합니다.
(카메라 intrinsics와 스테레오 extrinsics는 정상입니다.)

### ⚠ 경고 2: 가속도계 스케일이 어긋나 있습니다

정지 상태에서 |a| = **11.8 m/s²** (중력 9.81 대비 약 20% 높음). raw/calibrated
둘 다 동일하고 accuracy 플래그는 UNRELIABLE/MEDIUM. **IMU intrinsic 캘리브레이션
없이 VIO에 넣으면 안 됩니다.** 샘플마다 accuracy 필드를 남기니 숨지 않습니다.

축 방향(카메라 광축 대비)도 벤더 문서에 없고 이번에 확정하지 않았습니다.
회전도 병진과 함께 미지로 취급하세요.

이 세 가지는 전부 `metadata.txt`에 적힙니다. 캘리브레이션 안 된 IMU로 만든 VIO
결과는 "알고리즘 문제"처럼 보이는 방식으로 실패하기 때문입니다.

## 7. 제어 — 지금은 안 보냅니다

이 프로그램은 **아무것도 송신하지 않습니다**(예외: `--mavlink-set-rates`를 직접
켰을 때의 SET_MESSAGE_INTERVAL). QGroundControl로 조종하시고, 조종 입력은
MAVLink에서 읽어 `control.csv`에 기록됩니다.

- 하트비트를 보내지 않으므로 ArduSub의 GCS failsafe가 우리를 조종 GCS로
  오인할 수 없습니다.
- source_system을 QGC(255)와 다르게 씁니다.

나중에 조이스틱이나 비전 정책을 붙일 자리는 [control.py](control.py)에 인터페이스로
잡아뒀습니다(`ControlCommand`, `ControlSource`, `MavlinkCommandSink`). 지금은
의도적으로 `NotImplementedError`입니다 — 실제 차량에서 절반만 동작하는 제어 경로는
아예 없는 것보다 위험합니다.

**어느 경우에도 ArduSub가 비행 제어기입니다.** 안정화·깊이/자세 루프·arming·
failsafe는 계속 ArduSub가 합니다. 바뀌는 건 **QGroundControl이 하던 "MAVLink 명령
소스" 역할을 누가 맡는지**일 뿐입니다:

```
지금    QGroundControl → MANUAL_CONTROL → ArduSub → 추력기
다음    조이스틱        → MANUAL_CONTROL → ArduSub → 추력기
그다음  비전 정책       → MANUAL_CONTROL → ArduSub → 추력기
```

명령 소스를 붙일 때 주의: **동시에 두 소스가 명령하면** ArduSub는 마지막 도착
패킷을 따르므로 기체가 둘 사이에서 떨립니다. QGC 조종을 먼저 끊으세요.

## 8. 주요 CLI 옵션

전체는 `--help`. 카메라 옵션은 `--mode` 기본값을 덮어씁니다.

```
--mode research|review          --out DIR            --notes "..."
--dry-run                       설정·대역폭만 출력하고 종료

--isp-scale 1/4                 컬러 축소 (화각 유지)
--fps 8                         --streams color,depth,left,right
--color-wire nv12|bgr           --mono-source raw|rectified
--depth-size WxH                (research는 기본이 컬러와 동일 = 1:1)

--imu-rate 200                  --imu-rotation-vector   (같은 레이트라 공짜)
--imu-magnetometer              (전체를 100 Hz로 묶음)
--imu-calibrated                --no-camera-imu

--mavlink udpin:0.0.0.0:14551   --mavlink-transport udp|rest|none
--mavlink-set-rates             SET_MESSAGE_INTERVAL 전송 (기본 off = 완전 수동)
--mavlink-rate 50               --setup-endpoint

--writer-threads 3              --png-compression 1     --no-mp4
--min-free-gb 10                --record-now            --no-display
```

## 9. 문제 해결

| 증상 | 원인 | 조치 |
|---|---|---|
| `Could not connect to device with mx_id` | 다른 파이프라인 점유 (보통 Madrona) | `discover_c3.py` → `madrona stop --yes`, 5–10초 대기 |
| 연결에 12초 | 정상 (PoE 펌웨어 업로드+부팅) | 실행당 1회 비용 |
| MAVLink 아무것도 안 옴 | QGC가 14550 점유 / 엔드포인트 없음 | `blueos_endpoint add --port 14551 --yes` 또는 `--mavlink-transport rest` |
| fps가 요청보다 낮고 drop | 링크 천장 초과 | `--dry-run`으로 확인, `--fps`/`--isp-scale` 하향, 스테레오 제외 |
| `dropped > 0` (writer) | 디스크가 못 따라감 (이 장비에선 드묾) | `--writer-threads` 상향, `--png-compression 0` |
| depth 대부분 0 | 텍스처 부족/근접 | `--extended`, 조명. HUD의 valid % 확인 |
| `Qt xcb` 오류 | conda 라이브러리 충돌 | `conda deactivate` 후 venv 절대경로로 실행, 또는 `env -u LD_LIBRARY_PATH ...`, 또는 `--no-display` |
| conda base에서 실행 | base엔 depthai **3.5.0**(v3)이 있어 v2 코드가 반쯤 깨지고 카메라를 낚아챌 수 있음 | import 시 자동 거부됨. venv 경로로 실행 |

## 10. 파일

| 파일 | 역할 |
|---|---|
| `c3_collect.py` | 메인 수집 프로그램 (deliverable 2) |
| `discover_c3.py` | 탐색·진단 + IMU 유무 + 캘리브레이션 덤프 (deliverable 1) |
| `c3_dataset_check.py` | 녹화 검증 — 넘기기 전에 실행 |
| `dataset.py` | TUM 레이아웃 writer (스레드 풀), metadata 생성 |
| `imu.py` | BNO086 호스트측 읽기 + extrinsic 조회 |
| `mavlink_log.py` | 수동(passive) MAVLink 로거 |
| `blueos_endpoint.py` | QGC 공존용 MAVLink 엔드포인트 관리 |
| `control.py` | 향후 조이스틱/비전 제어 이음매 (지금은 비활성) |
| `madrona.py` | Madrona 확장 stop/start |
