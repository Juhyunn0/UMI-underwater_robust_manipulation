# C3 카메라 3대 테스트 운용 매뉴얼 (TESTING.md)

이 문서는 MarineSitu C3 (= Luxonis OAK-D-W-POE)에 대해 세 가지 질문에 답하는
런북이다. 위에서 아래로 그대로 실행하면 된다.

| # | 질문 | 도구 | 하드웨어 점유 | 물리 셋업 |
|---|---|---|---|---|
| 1 | 어떤 해상도/fps를 15-20 fps로 실제로 유지하나? | `c3_bench.py` | ~7분 (+선택 1.6분) | 없음 |
| 2 | depth가 얼마나 정확하고 최소 거리가 얼마인가? | `c3_depth_accuracy.py` | ~12분 (+사람 이동 ~40분) | 평평한 벽 + 줄자 |
| 3 | 인코더가 화질을 얼마나 깎나? | `c3_encode_quality.py` | ~11분 | 정적 텍스처 장면 |

**왜 이 순서인가**

1. **테스트 1이 가장 싸고 정보량이 가장 많다.** 물리 셋업이 전혀 없고, 7분이면
   끝나고, 나머지 두 테스트의 운용점(해상도·fps·depth 크기)을 정해준다.
2. **테스트 1의 `mono_res` 선택이 테스트 2의 최소 거리를 직접 구속한다.**
   `MinZ = fx * baseline / max_disparity` 이고 fx는 mono 해상도에 비례한다.
   400p면 MinZ **300 mm**(실측), 720p/800p면 600.2 mm[유도] — **두 배**다. 테스트 1에서
   "depth 디테일을 위해 mono를 720p로 올리자"고 결정하면 테스트 2의 근거리
   rung(130/185/280 mm)이 전부 구조적으로 실패한다. 그러니 mono_res를 먼저
   확정하고 그 값으로 테스트 2를 짜야 한다.
3. **테스트 2와 3은 같은 벽을 쓴다.** 테스트 2의 AprilTag 시트를 붙인 벽이
   테스트 3의 "정적 텍스처 장면"으로 그대로 재활용된다. 리그를 두 번 세우지
   않으려면 2 → 3 순서다.
4. **테스트 3은 마지막이어도 손해가 없다.** 유일하게 `--offline`으로 카메라
   없이 몇 번이든 재채점할 수 있는 테스트라, 캡처만 해두면 분석은 나중에
   고쳐가며 반복할 수 있다.

---

## 시작하기 전에 (Before you start)

**1. 인터프리터는 반드시 이것이다.** 래퍼가 골라준다:

```bash
./c3 env          # interpreter / python / depthai / numpy elide 를 찍는다
```

아래 명령들은 `$V`로 적혀 있는데, 그건 래퍼가 고르는 것과 같은 인터프리터다:

```bash
V=~/.venvs/c3-depthai/bin/python
```

2026-08-06부터 `./c3`의 1순위는 conda `robust`(python 3.12)다. 그전 robust는 3.14였고
numpy가 연산의 피연산자를 덮어썼다 — `./c3 env`의 `numpy elide` 줄이 그 진단이고,
`CORRUPTS OPERANDS`가 보이면 그 인터프리터로 측정하지 말 것.

**2. conda base는 위험하다.** conda base에는 depthai 3.x가 깔려 있고, v3는
`Pipeline(createImplicitDevice=True)`가 **생성만으로 디바이스를 연다** — 다른
프로세스가 쓰고 있는 C3를 낚아챈다. 패키지에 3.x 거부 import 가드가 있지만,
애초에 `conda activate` 상태로 이 스크립트들을 부르지 말 것. 확인:

```bash
$V -c "import depthai; print(depthai.__version__)"   # 2.x 여야 한다
```

**3. 리그가 준비됐는지 한 번에 본다.** 인터프리터 · 네트워크 경로 · 카메라 소유권 ·
ROV · 디스크 · 디스플레이를 한 화면으로 찍는다. 디바이스를 열지 않으므로 다른
프로세스가 스트리밍 중이어도 안전하고, 몇 초면 끝난다:

```bash
./c3 preflight; echo "exit=$?"
```

`READY`면 그대로 진행하면 된다. `NOT READY`면 마지막 `blocking:` 줄이 원인이고,
그 검사 밑에 `fix:`로 실행할 명령이 붙어 있다. 종료 코드는 **`0` = 준비됨**, `5` =
막는 검사가 있음.

카메라 소유권만 따로 보고 싶으면 `discover_c3.py`가 여전히 그 일을 한다 —
**`0` = 발견됐고 연결 가능**, `3` = 다른 파이프라인이 소유 중 (`X_LINK_BOOTED` —
Madrona/BlueOS 쪽에서 먼저 스트림을 멈출 것), `4` = 디스커버리 실패,
`5` = 네트워크에서 아예 안 보임.

> `--probe`는 **붙이지 말 것.** 파이프라인 없이 잠깐 연결하는 동작이라 1-2초
> 디바이스를 점유한다. 스트리머가 도는 중에는 금지. `./c3 preflight`는 그 점에서
> 다르다 — 아무것도 열지 않는다.

**4. 회귀 테스트가 초록인지 먼저 확인** (venv에 pytest가 없으므로 스크립트를
직접 실행한다 — `python -m pytest`는 `No module named pytest`로 실패한다):

```bash
./c3 test                                   # 여섯 개를 순서대로 (권장)

$V c3_camera/tests/test_offline.py          # 58/58
$V c3_camera/tests/test_depth_accuracy.py   # 58/58
$V c3_camera/tests/test_encode_quality.py   # 36/36
$V c3_camera/tests/test_host_depth.py       # 45/45
$V c3_camera/tests/test_option_sweep.py     # 32/32
$V c3_camera/tests/test_preflight.py        # 43/43
```

**5. 링크 천장은 ~91.5 Mbit/s** (`C.POE_BUDGET_MBPS = 90.0`). 케이블이나 스위치를
바꿨다면 이 숫자부터 무효다 — 테스트 1의 Stage 1a가 그 pass/fail 점검이다.

---

# 테스트 1 — 해상도 x fps 프론티어

**질문:** 15-20 fps를 실제로 **유지하는** 컬러/depth 해상도 조합은 무엇인가?

**물리 셋업:** 없음. 카메라가 켜져 있고 아무 장면이나 보고 있으면 된다.

**총 시간:** 13 combo x 32 s = **~7분** (Stage 4 선택 시 +1.6분).
combo당 32초 = `--duration 10 + --warmup 3 + --settle 5 + 14`(부팅 12초 + PoE 리셋).

## Stage 0 — 카메라를 안 쓰는 사전 선별 (0분, 공짜)

하드웨어를 32초씩 태우기 전에 조합을 먼저 걸러낸다.

```bash
V=~/.venvs/c3-depthai/bin/python
cd /home/bdml/Desktop/umi_underwater_robust_control

$V c3_camera/c3_stream.py --dry-run --color-res 1080p --isp-scale 2/3 \
    --color-encode mjpeg --mono-res 400p --fps 15 --depth-size 640x360
```

확인할 것 두 개:

* `colour output width ... is not a multiple of 32` **경고가 없을 것.**
  있으면 VideoEncoder가 거부해서 32초 run이 통째로 날아간다.
* `budget` 줄의 %. 단 **`OVER BUDGET`이라고 1080p를 사다리에서 빼지 말 것** —
  `est_mbps`는 `MJPEG_RATIO = 6.0` 고정이라 1080p에서 실측 대비 약 1.4배
  과대평가한다 (실측 8.25:1). 실제 프론티어는 `color_kb_frame` 컬럼으로만 나온다.  [UNVERIFIED: 산출물 없음 — docs/MEASUREMENT_AUDIT.md]

미리 확인해 둔 값 (전부 32-정렬 통과):

```
isp 1/2 (960x540)  + depth 480x270 @15 =  52% [ok]     @20 =  69% [tight]
isp 1/2 (960x540)  + depth 640x360 @15 =  79% [tight]  @20 = 105% [OVER]
isp 2/3 (1280x720) + depth 480x270 @15 =  65% [tight]  @20 =  87% [tight]
isp 2/3 (1280x720) + depth 576x324 @15 =  80% [tight]
isp 2/3 (1280x720) + depth 640x360 @15 =  92% [tight]  @20 = 123% [OVER]
isp none (1920x1080) + depth 480x270 @15 = 104% [OVER]
isp none (1920x1080) + depth 576x324 @15 = 119% [OVER]
```

## Stage 1 — depth 해상도 사다리 (6 combo, 3.2분)

depth는 압축이 없어서 대역폭이 산술로 확정된다 (`w*h*2*fps*8/1e6`). 그래서
스윕의 역할은 천장 찾기가 아니라 **무릎(knee)에서 어떻게 무너지는지** 보는 것이다.

```bash
mkdir -p c3_camera/bench

# 1a — anchor. 이 조합은 README 대역폭 표의 유일한 경계 실측행이다:
#      "960x540 mjpeg @15 + depth 640x360" = 71.4 Mbit/s (78%), p50 39 ms, p95 80 ms.
#      @20 행은 인코더 표에서 19.1 fps / drop 5%.
#      measured_mbps가 71.4와 5% 안에서 일치하면 링크가 그대로라는 뜻이고,
#      이후 단계의 예측이 전부 유효해진다. 어긋나면 케이블/스위치가 바뀐 것이다.
$V c3_camera/c3_bench.py \
    --color-res 1080p --isp-scale 1/2 --color-encode mjpeg --mono-res 400p \
    --fps 15,20 --depth-size 640x360 \
    --out c3_camera/bench/s1a_anchor.csv

# 1b — depth를 내려본다
$V c3_camera/c3_bench.py \
    --color-res 1080p --isp-scale 1/2 --color-encode mjpeg --mono-res 400p \
    --fps 15,20 --depth-size 480x270,320x180 \
    --out c3_camera/bench/s1b_depth.csv
```

**조기 중단 조건:** 1a의 `measured_mbps`가 71.4와 5% 안에서 일치하고, 1b의
480x270 두 행이 아래 `hold(f)` 판정식을 15와 20 둘 다 통과하면 `depth_size =
480x270`이 안전 기본값으로 확정된 것이다. 급하면 여기서 멈춰도 된다.

## Stage 2 — 컬러 해상도 사다리 (4 combo, 2.1분)

depth는 Stage 1에서 확정된 480x270으로 고정하고 컬러만 올린다.

```bash
$V c3_camera/c3_bench.py \
    --color-res 1080p --isp-scale 2/3,none --color-encode mjpeg --mono-res 400p \
    --fps 15,20 --depth-size 480x270 \
    --out c3_camera/bench/s2_colour.csv
```

`2/3` = 1280x720 (한 번도 측정된 적 없음), `none` = 1920x1080 @15 (README에는
@12와 @20만 있다). `1/3`(640x360)과 `1/2`(960x540)은 README의 대역폭 표에 이미
측정되어 있으니 넣지 말 것.

> 컬러 사다리는 **하나의 센서 모드 안에서 `--isp-scale`로만** 만들어야 한다.
> isp 축소는 화각을 보존하지만(downscale), 더 작은 센서 모드로 내려가면
> crop되어 화각이 좁아진다. 센서 모드를 섞으면 해상도 비교가 화각 비교와 뒤섞인다.

## Stage 3 — 두 축의 코너 (3 combo, 1.6분)

각 축의 상한을 따로 재면 "컬러 1280x720도 되고 depth 640x360도 된다"는 잘못된
결론이 나온다. 예산은 합산이라 동시에는 92%를 먹는다.

```bash
$V c3_camera/c3_bench.py \
    --color-res 1080p --isp-scale 2/3 --color-encode mjpeg --mono-res 400p \
    --fps 15 --depth-size 576x324,640x360 \
    --out c3_camera/bench/s3a_corner.csv

$V c3_camera/c3_bench.py \
    --color-res 1080p --isp-scale none --color-encode mjpeg --mono-res 400p \
    --fps 15 --depth-size 576x324 \
    --out c3_camera/bench/s3b_1080p_corner.csv
```

## Stage 4 (선택) — 1080p 위 센서 모드 / 4:3 전체 화각 (3 combo, 1.6분)

이 단계는 링크가 아니라 **ISP/센서 연산**을 시험한다.

```bash
$V c3_camera/c3_bench.py --color-res 4k --isp-scale 1/3 --color-encode mjpeg \
    --mono-res 400p --fps 15 --depth-size 480x270 \
    --out c3_camera/bench/s4a_4k.csv

$V c3_camera/c3_bench.py --color-res 12mp --isp-scale 6/19 --color-encode mjpeg \
    --mono-res 400p --fps 15 --depth-size 480x270 \
    --out c3_camera/bench/s4b_12mp.csv

$V c3_camera/c3_bench.py --color-res 2024x1520 --isp-scale 12/23 \
    --color-encode mjpeg --mono-res 400p --fps 15 --depth-size 480x270 \
    --out c3_camera/bench/s4c_2024.csv
```

`4k 1/3`은 `1080p 2/3`과 같은 1280x720 픽셀을 다른 센서 모드에서 만든다 — 나란히
놓으면 그 자체가 화각 실험이다. **Stage 2 표에 섞지 말 것.**

## 실행 체크리스트

| ☐ | Stage | 명령 | combo | 시간 | 답하는 것 |
|---|---|---|---|---|---|
| ☐ | 0 | `c3_stream.py --dry-run ...` | - | 0 | 32-정렬 + 예산 사전 선별 |
| ☐ | 1a | `s1a_anchor.csv` | 2 | 64 s | 링크가 아직 91.5 Mbit/s인가 |
| ☐ | 1b | `s1b_depth.csv` | 4 | 128 s | depth 480x270 / 320x180 무릎 |
| ☐ | 2 | `s2_colour.csv` | 4 | 128 s | 컬러 1280x720 / 1920x1080 |
| ☐ | 3a | `s3a_corner.csv` | 2 | 64 s | 1280x720 + depth 576/640 동시 |
| ☐ | 3b | `s3b_1080p_corner.csv` | 1 | 32 s | 1920x1080 + depth 576 동시 |
| ☐ | 4a-c | `s4a/b/c_*.csv` (선택) | 3 | 96 s | 4:3 전체 화각 / ISP 연산 한계 |

## 어느 컬럼이 답인가

**판정 컬럼:** `color_fps`, `depth_fps` (둘 다 warmup 이후의 `fps_lifetime`).
**함정 컬럼:** `color_drop_rate`, `depth_drop_rate`, `color_dropped`, `color_frames`.

판정식은 이것을 그대로 쓴다:

```
hold(f) = color_fps      >= 0.98 * f
      AND depth_fps      >= 0.98 * f
      AND color_drop_rate <= 0.005
      AND depth_drop_rate <= 0.005
      AND (color_lat_p95 - color_lat_p50) <= 10 ms
```

`depth_fps`를 반드시 같이 보는 이유: 페어링 rate는 `min(color_fps, depth_fps)`이고,
한쪽만 떨어지면 컬러/depth skew가 벌어진다.

교차검증: `color_frames`를 `requested_fps * duration`과 비교하면 늦은 metrics
reset도 잡힌다.

## 나쁜 결과를 읽는 법

* **`color_fps` 19.1 인데 `color_drop_rate` 0.05 → 성공 아님.** drop은 디바이스가
  전송 전에 부여한 시퀀스 번호의 **구멍**으로 센다. 즉 20장을 만들어 1장을 버린
  것이고, 스트림에 구멍이 뚫려 있다. SLAM/텔레오퍼레이션에는 **균일한 15 fps보다
  나쁘다.** 이럴 땐 fps를 15로 내려서 구멍 없는 스트림을 받는 쪽이 옳다.
* **link-limited 서명:** `measured_mbps`가 88-92에 붙음 + `est_mbps >= measured_mbps`
  + 두 스트림이 바이트 점유율에 비례해 **동시에** drop + `color_lat_p95`가
  `p50`보다 크게 벌어짐. → 해상도나 depth 크기를 내려야 한다.
* **compute-limited 서명 (3갈래):** `measured_mbps < 75`인데 fps 미달일 때,
  * `drop_rate ~ 0` → **센서 클램프.** 프레임이 애초에 안 만들어졌다 (4k는 42 fps,
    12mp는 30 fps가 상한). Stage 4에서 나오면 정상.
  * `depth_fps < color_fps`이고 `depth_drop_rate`만 0 초과 → **stereo 노드.**
    (`--subpixel` + `--depth-align color`가 알려진 범인)
  * `loop_hz < min(color_fps, depth_fps)` → **호스트.** 다른 프로세스를 끄고 재실행.
* **`ok=0` 행이 여러 개인데 전부 똑같아 보임:** `depth_out` 컬럼을 볼 것. 실패한
  행도 요청된 depth 크기를 담고 있어서 사다리의 어느 rung인지 구분된다.

## 테스트 1이 말해줄 수 없는 것

* **`--extended`와 `--mono-fps`가 c3_bench에 없다.** `--extended`는 MinZ를 절반으로
  줄이는 유일한 노브인데 벤치로는 측정할 수 없다 → 그건 테스트 2의 일이다.
  `--mono-fps`(depth fps를 컬러와 분리하는 4순위 대역폭 노브)도 마찬가지.
* **`est_mbps`는 이제 `--mjpeg-quality`에 반응한다** (2026-08-06 수정).
  `MJPEG_QUALITY_SCALE`이 q90=1.00 기준으로 스케일한다(q80 0.71, q75 0.58) —
  C3가 직접 찍은 프레임 120장을 재인코딩해 실측한 값에서 왔고
  (`c3_camera/datasets/dataset_20260805_174544/rgb`), q90에서 재인코딩 114.1 kB/f 대
  디바이스 인코더 114.2 kB/f로 0.1% 일치했다. 그래도 **추정은 추정**이다: 장면
  의존성(46~121 kB/f)은 어떤 모델도 못 잡으므로 판정은 실측 `color_kb_frame` /
  `color_mbps` 컬럼으로 한다.
* **인코더 행끼리 `drop_rate`와 latency를 직접 비교하면 안 된다.** 큐 정책이
  비대칭이다 (`pipeline.py:120-130`): h264/h265는 디바이스+호스트 blocking 큐
  1.5초 깊이, none/mjpeg는 non-blocking depth-1. 포화 상태에서 none/mjpeg는 압력을
  `drop_rate`로, h26x는 latency로 보여준다.
* **fold가 못 잡는 중복 2건:** (1) mono 400p + 16:9 컬러에서 `--depth-size 640x360`과
  `derived`는 같은 파이프라인이지만 다른 튜플로 취급된다. (2) `--fps 30
  --video-keyframe-frequency 0,30`도 같은 파이프라인이다 (0 → round(fps)).
* **`--mjpeg-quality`가 범위 검증되지 않는다.** 150이나 -5도 통과해서 디바이스 시작  [UNVERIFIED: 예시/유도값 — docs/MEASUREMENT_AUDIT.md]
  시점에서야 실패한다 = run 하나를 태운다.
* **`--append`는 컬럼 셋이 다르면 거부된다** (rc=2, 기존 파일 무수정). 그래서 위
  명령들은 전부 단계별로 다른 `--out`을 쓴다. 나중에 CSV를 합쳐 읽을 것.
* **화각을 측정하지 못한다.** Stage 4의 4:3 모드가 주는 실제 이득은 정지 장면
  육안 비교로 따로 확인해야 한다.
* **이 케이블, 이 스위치에서만 유효하다.** 91.5 Mbit/s는 경로의 성질이지 카메라의
  성질이 아니다.
* **공기 중 장면에서 잰 프론티어다.** MJPEG 압축률은 장면 의존적이고 (같은 설정
  6세션에서 4.05:1 ~ 6.35:1로 흔들렸다), 탁하거나 어두운 물은 고주파 성분이 적어  [UNVERIFIED: 산출물 없음 — docs/MEASUREMENT_AUDIT.md]
  **더 잘** 압축된다. 즉 수중에서는 컬러 대역폭이 여기서 잰 것보다 낮아질 공산이
  크고, 이 사다리는 그런 의미에서 **보수적**이다. 다만 방향만 알 뿐 크기는 모르니,
  풀에서 맞춘 설정이 암초 위에서 그대로 맞는다고 가정하지 말 것.

---

# 테스트 2 — depth 정확도와 최소 거리

**질문:** C3의 depth가 거리에 따라 얼마나 틀리고, 실제로 몇 cm까지 유효한가?
벤더가 주장하는 18 cm에 도달하려면 무엇이 필요한가?

**전제:** 테스트 1에서 `mono_res`를 확정했을 것. 아래 사다리는 **400p 기준**이다
(MinZ **300 mm**, `--extended`면 **150 mm** — 둘 다 실측, 아래 '이미 아는 것' 참조).
만약 테스트 1이 720p/800p를 골랐다면 MinZ가 600.2 mm[유도]이므로 **130/185/280/330 mm rung을 전부 지우고** 660 mm부터
시작해야 한다.

## 물리 셋업

* **타깃은 "움직이는 보드"가 아니라 "고정된 평평한 벽"**이다. 벽에 tag36h11
  시트를 **무광(matte)** 용지로 타일링해 붙인다 — 코팅지/광택지 금지 (정반사가
  stereo 매칭을 죽인다).
* 바닥에 벽과 **수직**으로 줄자를 붙이고, 카메라를 그 줄자를 따라 움직인다.
  벽을 쓰면 평면성이 공짜로 보장되고(보드는 휘고 기울어진다), 원거리 rung에서
  "타깃이 분석영역보다 작다"는 함정이 구조적으로 사라진다.
* 줄자는 스퀘어를 댈 수 있는 **기계적 기준면**(예: 하우징 앞면)에서 잰다.
  그 면이 무엇인지 `--datum`에 반드시 적는다.
* **조명은 실내 앰비언트.** 온보드 램프 금지 — 램프 조도는 1/Z²로 변해서 rung마다
  노출 조건이 달라진다. 벽은 Lambertian이라 앰비언트면 거리와 무관하게 같은 휘도다.
* rung 하나를 잡는 동안 리그가 움직이면 안 된다. 스크립트가
  `motion_madiff`로 검출해 refuse 한다.

## 필요한 벽 크기

ROI(중앙 40%)는 `0.499*Z x 0.280*Z`, full frame은 `1.247*Z x 0.701*Z`를 덮는다.
2 m rung은 ROI만 997x561 mm, full frame은 2494x1402 mm가 필요하다.

## Step 0 — 카메라 없이 기하 먼저 확인 (0분)

```bash
V=~/.venvs/c3-depthai/bin/python
cd /home/bdml/Desktop/umi_underwater_robust_control

$V c3_camera/c3_depth_accuracy.py --dry-run --truth-mm 300
$V c3_camera/c3_depth_accuracy.py --dry-run --truth-mm 185 --extended
```

각 rung의 MinZ, disparity 스텝 크기, overlap band 예측, ROI 여유 픽셀이 나온다.
**`--dry-run`은 절대 디바이스를 열지 않는다.**

## Step 1 — 셰이크다운 (1 rung, ~1분)

캡처 경로는 **한 번도 하드웨어에서 돌아본 적이 없다.** 사다리 전체에 커밋하기
전에 편한 거리에서 한 rung만 짧게 잡아본다.

```bash
$V c3_camera/c3_depth_accuracy.py --truth-mm 1000 --frames 15 \
    --label shakedown --datum "housing front face"
```

`ok=1`로 끝나고 `rho_mm`이 1000 근처면 진행. `ok=0`이면 출력된 `refused` 사유를
먼저 고칠 것 (대개 조준이나 조명).

## Step 2 — CORE 사다리, arm A/B (12 rung, ~7분 기계 + 이동 시간)

거리는 **바깥 루프**(사람이 카메라를 옮기고 줄자 값을 입력), arm은 **안쪽 루프**.
각 station에서 아래 두 줄을 연달아 실행하고 다음 거리로 옮긴다.

```bash
# --- station: 130 mm (음성 대조군 — 전부 실패해야 정상) ---
$V c3_camera/c3_depth_accuracy.py --truth-mm 130 --label default \
    --datum "housing front face" --median off --confidence 245
$V c3_camera/c3_depth_accuracy.py --truth-mm 130 --label ext --extended \
    --datum "housing front face" --median off --confidence 245

# --- 이후 185, 280, 330, 1000, 2000 mm 에서 같은 두 줄을 반복 ---
```

station마다 `T=` 한 줄만 바꿔서 아래 세 줄을 붙여넣으면 된다:

```bash
T=185
$V c3_camera/c3_depth_accuracy.py --truth-mm $T --label default \
    --datum "housing front face" --median off --confidence 245
$V c3_camera/c3_depth_accuracy.py --truth-mm $T --label ext --extended \
    --datum "housing front face" --median off --confidence 245
```

> `--median off`와 `--confidence 245`를 **명시적으로 고정**하는 이유: `robotics`
> preset의 confidence 245는 255 중 245라 거의 완전 관대해서 fill의 상당수가
> 쓰레기일 수 있고, median 필터는 구멍을 메워 잔차를 **낮춰서** 카메라를 실제보다
> 좋아 보이게 만든다. 안 고정하면 "extended가 fill을 깎았다" 같은 결론이 preset
> 아티팩트일 수 있다.

## Step 3 — 컨트롤 arm C, D (5 rung, ~4분)

```bash
# C — 800p far arm. 400p는 2 m에서 disparity 스텝이 141 mm라
#     quantisation-limited 경고가 뜬다. 800p는 그 절반(70.4 mm).
#     --fps 2 인 이유: 800p depth는 1280x720이라 5 fps면 133% OVER BUDGET.
for T in 660 1000 2000; do
  $V c3_camera/c3_depth_accuracy.py --truth-mm $T --label far800 \
      --mono-res 800p --fps 2 --timeout 60 --median off --confidence 245 \
      --datum "housing front face"
done

# D — overlap band 컨트롤. --depth-align none 이 아니면 밴드를 볼 수 없다.
for T in 185 280; do
  $V c3_camera/c3_depth_accuracy.py --truth-mm $T --label band --extended \
      --depth-align none --median off --confidence 245 \
      --datum "housing front face"
done
```

> **왜 arm D가 따로 필요한가:** `--depth-align color`는 stereo 화각을 H 80.4° →
> 63.9°로 잘라내서, 근거리 overlap 손실 밴드가 **338 mm 위에서는 프레임 밖으로
> 완전히 잘려나간다.** align=color 데이터만 보면 이 현상이 존재하지 않는 것처럼
> 보인다. `--dry-run`이 직접 이렇게 인쇄한다: *"the delivered frame is as wide as
> the stereo pair, so the whole band stays visible — this is the control
> configuration for measuring it."*

## Step 4 — 분석 (카메라 없음, 몇 번이든 반복 가능)

```bash
$V c3_camera/c3_depth_accuracy.py --analyze
```

## 실행 체크리스트

| ☐ | Arm | rung (mm) | 설정 | run 수 | 답하는 것 |
|---|---|---|---|---|---|
| ☐ | shakedown | 1000 | 기본, `--frames 15` | 1 | 캡처 경로가 하드웨어에서 도나 |
| ☐ | A `default` | 130,185,280,330,1000,2000 | 400p, ext off | 6 | 현행 기본값의 실제 유효 범위 |
| ☐ | B `ext` | 130,185,280,330,1000,2000 | 400p, `--extended` | 6 | 18 cm에 정말 도달하나 |
| ☐ | C `far800` | 660,1000,2000 | 800p, `--fps 2` | 3 | 양자화 절반이 원거리 bias를 줄이나 |
| ☐ | D `band` | 185,280 | ext + `--depth-align none` | 2 | overlap 밴드가 공식대로 나오나 |
| ☐ | analyze | - | `--analyze` | 0 | 스케일 a, 원점 c, ext off/on 비교 |

rung별 존재 이유: **130** = 모든 MinZ 아래 → 전부 실패해야 하는 음성 대조군.
**185** = 벤더 주장 18 cm, 사용자가 실제로 원하는 답. **280/330** = 현행 기본값
벼랑 300 mm의 아래/위 브래킷. **1000/2000** = 절편 c 적합용 원거리 앵커.

## 어느 컬럼이 답인가

* **"몇 cm까지 되나"** → `fill_roi_inlier` (0.5 이상이면 "여기서 동작한다"),
  `z_min_mm`, `min_z_pred_mm`. 세 개를 나란히 보면 예측과 실측이 맞는지 나온다.
* **"얼마나 틀리나"** → `bias_mm`, `bias_pct`. 단 `--analyze`가 원점 offset을
  피팅한 뒤의 값을 봐야 한다.
* **"얼마나 시끄러운가"** → `resid_rms_mm` (주 지표: 평면 잔차 RMS. tilt가 제거된
  순수 표면 재구성 오차). `patch_std_mm`은 대조용이지 잡음 지표가 아니다.
* **밴드** → `hole_left` / `hole_centre` / `hole_right` / `hole_asym`, 그리고
  예측인 `band_pred_px` / `band_pred_frac` / `band_delivered_frac`.
* **채택 여부** → `ok`, `warn`, `refused`. `ok=0`인 행은 `--analyze`가 fit에서 뺀다.

## 나쁜 결과를 읽는 법

* **130 mm rung이 "성공"한다 (`fill_roi` > 0.5)** → 이게 최악의 결과다. MinZ
  아래에서는 stereo 정보가 물리적으로 없으므로, 유효 depth가 나온다는 건 median /
  hole-fill이 **값을 지어내고 있다**는 뜻이고, 그러면 **모든 rung의 fill 숫자가
  무효다.** 스크립트가 이걸 명시적으로 경고한다.
* **`warn`에 `quantisation-limited`** → 잔차가 `dZ/6`보다 작다는 뜻이고, 벽이 한
  disparity bin 안에 통째로 들어갔다는 신호다. `rho`는 벽이 아니라 **bin**이고
  bias가 ±dZ/2를 통째로 짊어진다. 2 m/400p에서 dZ = 141 mm — 피팅하려는 원점  [UNVERIFIED: 예시/유도값 — docs/MEASUREMENT_AUDIT.md]
  offset(~25 mm)보다 크다. 대응: 800p arm(arm C)을 쓰거나, 벽을 몇 도 일부러
  틀어서 격자를 dither 한다.
* **`fill_roi`가 0.30-0.50 사이인데 `warn`이 비어 있다** → 알려진 사각지대다.
  MinZ 아래에서 40%가 유효하다고 보고돼도 조용히 통과한다. `--fit-min-mm 450`
  덕에 fit에는 안 들어가지만 표에는 남으니 눈으로 걸러야 한다.
* **`--analyze`가 config 간 `c` 불일치를 경고** → 광학중심은 mono 설정에 의존할 수
  없으므로, 원점 말고 다른 게 틀렸다는 뜻이다. `--datum`이 rung마다 달랐는지부터 볼 것.
* **`refused: exposure`** → 노출이 20 ms를 넘었다. 조명을 올릴 것. 스크립트는
  노출을 **고정할 수 없고 감지만 한다** (아래 한계 참조).

## 테스트 2가 말해줄 수 없는 것

* **★ 이 테스트는 물이 아니라 센서를 측정한다.** 전부 **공기 중** 측정이다. 물속에서는
  굴절 때문에 스케일이 통째로 바뀐다 (미보정 시 약 1.33배). `--analyze`가 뽑는
  `a` 계수가 정확히 그 자리이고 — 그래서 이 실험을 물속에서 반복하면 `a`가 곧
  refraction scale이 된다 — **하지만 아직 물속에서 재지 않았다.** 공기 중 bias를
  수중 성능으로 인용하지 말 것.
* **mono 노출을 수동 고정할 수 없다.** `StreamConfig`에 `mono_exposure_us` /
  `mono_iso`가 없고 `pipeline.py`에 `MonoCamera.initialControl.setManualExposure()`
  경로가 없다. 결과: 서로 다른 rung/arm이 **서로 다른 노출에서 비교될 수 있다.**
  스크립트는 노출 레벨·노출 산포·모션에 게이트를 걸어 **감지만** 한다. 이게 가장
  가치 높은 후속 작업이다.
* **`--disparity-shift`가 없다.** extended disparity보다 싸게 15 cm에 도달하는
  대안인데 구현되지 않아서 그 트레이드를 판정할 수 없다.
* **`--tag-check`가 없다.** 피팅된 원점 offset `c`에 대한 독립적인 second opinion이
  없다 — 설정 간 일관성 검사가 유일한 검증이다.
* **delivered-band 예측은 CAM_A를 stereo 기준 카메라와 같은 위치로 모델링한다.**
  실제로는 ~37.5 mm 떨어져 있어서, 185 mm에서 그 병진은 104 px(폭의 16.3%)로  [UNVERIFIED: 예시/유도값 — docs/MEASUREMENT_AUDIT.md]
  예측된 밴드 94 px보다 **크다.** 밴드는 잘리는 게 아니라 **이동**하므로  [UNVERIFIED: 예시/유도값 — docs/MEASUREMENT_AUDIT.md]
  `band_delivered_frac`은 정확히 근거리에서 신뢰할 수 없다. arm D(`align none`)가
  이래서 필요하다.
* **`fixed_rms_mm`은 400p / 1 m 이상에서 "고정 패턴"이 아니라 양자화가 지배한다.**
  1 m 실측 예: resid 10.01, temporal 5.19, fixed 8.56 — 진짜 센서 노이즈 1 mm에  [UNVERIFIED: 예시/유도값 — docs/MEASUREMENT_AUDIT.md]
  ripple 0으로 합성했는데도 그렇다 (dZ/sqrt(12) = 10.2가 새어 들어온다).
  `quant_dz_pred_mm`을 빼고 읽을 것. 또 ~0.9 mm 사각지대가 있어 진짜 고정 패턴이
  정확히 0으로 읽힐 수 있다.  [UNVERIFIED: 예시/유도값 — docs/MEASUREMENT_AUDIT.md]
* **`temporal_rms_mm`은 모든 프레임에서 유효한 픽셀만 쓴다.** 낙관적으로 편향돼
  있고, 구멍 30%/20프레임이면 ROI의 0.08%만 남고 40%면 아예 빈 셀이 된다 — 정확히
  이 연구가 관심 있는 근거리 holey rung에서.
* ~~**MinZ가 298.9인지 300.1인지 미해결.**~~ **해결됨 (2026-08-05): CAM_C.**
  녹화된 depth PNG를 전수 조사하니 바닥이 정확히 **300 mm**, `--extended`는 정확히
  **150 mm**이고 그 아래 픽셀은 **하나도 없다**
  `[측정: c3_camera/datasets/*/depth/, c3_camera/recordings/*/depth/ — 47,270 프레임,
  3.6e9 유효 픽셀, 2026-08-05]`. 바닥이 정확히 300으로 찍히려면 fx*B/95 ∈ [300, 301),
  즉 fx_rect ∈ [380.0, 381.3)이어야 하는데 CAM_C의 380.14는 그 안, CAM_B의 378.56은
  밖(298로 찍혔어야 함)이다. 300:150이 정확히 2:1인 것도 disparity 95→190 말고는
  설명되지 않는다(이 파이프라인엔 threshold 필터도 disparityShift도 없다).
  `host_depth.py`가 호스트 측 rectification에서 잰 ~380.00과도 0.04% 일치한다.
* **`--save-npy`의 `run_id`는 1초 해상도다.** 사람 속도 프로토콜에서는 무해하지만
  스크립트로 돌리면 파일명이 충돌한다.
* **분석 산출물은 텍스트와 CSV뿐이다.** `--save-npy`를 주지 않으면(기본 off) 나중에
  원자료로 재분석할 수 없고 CSV가 담은 것만 남는다.

---

# 테스트 3 — 인코더 화질

**질문:** 주어진 Mbit/s 예산에서 어느 (codec, rate) 조합이 초당 가장 많은
**쓸 수 있는** 디테일을 주는가?

> **먼저 알아둘 것:** "H.264/H.265가 대역폭 문제를 푼다"는 가설은 **현재 운용점에서는
> 틀렸다.** 실측(dataset_20260803_105221)에서 컬러 MJPEG는 6.8 Mbit/s로 링크의  [UNVERIFIED: 산출물 없음 — docs/MEASUREMENT_AUDIT.md]
> **14%뿐**이고 depth uint16이 41.5 Mbit/s로 **86%**를 먹는다. 컬러를 0바이트로
> 만드는 완벽한 코덱도 천장의 7.4%만 회수하고, 전체가 이미 53%라 **fps 이득은  [UNVERIFIED: 산출물 없음 — docs/MEASUREMENT_AUDIT.md]
> 정확히 0**이다. 인코더 질문이 binding해지는 곳은 **컬러 해상도를 올릴 때뿐**이다.

**물리 셋업:** 테스트 2의 벽을 그대로 쓴다. 요구조건 둘:

* **정적**이어야 한다 (리그 고정, 장면 안 움직임). 한 설정을 재는 동안 무엇도
  움직이면 안 된다.
* **텍스처가 풍부**해야 한다. AprilTag 시트가 이 조건을 만족한다. reference의
  ORB keypoint가 500 미만이면 스크립트가 전체 캡처를 `under_textured`로 플래그하고
  결론에서 제외한다 (`--min-ref-keypoints`).

**총 시간:** 20 설정 x 32 s = **~11분**.

## Step 0 — 사다리를 카메라 없이 계획 (0분)

```bash
V=~/.venvs/c3-depthai/bin/python
cd /home/bdml/Desktop/umi_underwater_robust_control

$V c3_camera/c3_encode_quality.py --dry-run \
    --codec none,mjpeg,h264,h265 --mjpeg-quality 80,90,97 \
    --bitrate-kbps 2000,4000,8000,16000 --keyframe 0,1
```

출력이 20 설정 / ~10.7분이라고 하면 계획대로다.

## Step 1 — 실캡처 (11분, **디바이스를 연다**)

```bash
$V c3_camera/c3_encode_quality.py \
    --codec none,mjpeg,h264,h265 --mjpeg-quality 80,90,97 \
    --bitrate-kbps 2000,4000,8000,16000 --keyframe 0,1 \
    --streams color --fps 20 --target-fps 19 \
    --scene-label static \
    --out c3_camera/encode_quality/run1
```

> `--streams color`인 이유: 640x360 uint16 depth는 ~74 Mbit/s라 2-30 Mbit/s짜리
> 컬러 사다리를 통째로 삼켜서 인코더 축이 안 보이게 된다.
>
> `--codec`에 `none`이 반드시 있어야 한다. raw arm이 fidelity 기준선이고 옵션이
> 아니다. raw는 960x540@20에서 링크 천장의 138%라 ~9 fps로 떨어지는데, 이건
> **정상**이고 랭킹에서는 `--target-fps` 미달로 실격된다. reference를 만드는 게  [UNVERIFIED: 산출물 없음 — docs/MEASUREMENT_AUDIT.md]
> 그 arm의 일이다.

## Step 2 — 재채점 (카메라 없음, 몇 번이든)

```bash
$V c3_camera/c3_encode_quality.py --offline c3_camera/encode_quality/run1 \
    --target-fps 19 --budget-mbps 25
```

`--budget-mbps`를 바꿔가며 여러 번 돌리는 게 이 스크립트의 요점이다.
`--rank-by ssim_y`로 판정 지표를 바꿔서 순위가 뒤집히는지도 확인할 것.

## Step 3 (선택) — 1080p 팔 (4 설정, ~2분)  [UNVERIFIED: 예시/유도값 — docs/MEASUREMENT_AUDIT.md]

인코더 질문이 실제로 binding해지는 유일한 운용점. **해상도는 스윕 축이 아니므로
(reference median stack이 한 해상도에서만 유효) 반드시 별도 `--out` 디렉터리를
쓰고, 나중에 `runs.csv`를 손으로 합쳐 읽어야 한다.**

```bash
$V c3_camera/c3_encode_quality.py \
    --codec none,mjpeg,h265 --isp-scale none --mjpeg-quality 90 \
    --bitrate-kbps 8000,16000 --keyframe 1 \
    --streams color --fps 20 --target-fps 19 \
    --duration 20 --reference-frames 15 \
    --out c3_camera/encode_quality/run2_1080p
```

> `--duration 20 --reference-frames 15`가 필수다. 1080p raw는 링크에서 ~3.8 fps라  [UNVERIFIED: 예시/유도값 — docs/MEASUREMENT_AUDIT.md]
> 기본 `--duration 10`이면 프레임이 ~38장뿐이고, `--reference-frames 25`가 그걸
> 거의 다 먹어서 score 단계가 명시적 에러로 멈춘다.

## 실행 체크리스트

| ☐ | Step | 명령 | 설정 수 | 시간 | 답하는 것 |
|---|---|---|---|---|---|
| ☐ | 0 | `--dry-run` | 20 | 0 | 사다리·시간 계획 |
| ☐ | 1 | `--out .../run1` | 20 | 10.7 min | 960x540에서 rate-quality 곡선 |
| ☐ | 2 | `--offline .../run1 --budget-mbps 25` | - | 0 | 예산별 승자 |
| ☐ | 2b | `--offline ... --rank-by ssim_y` | - | 0 | 지표를 바꾸면 순위가 뒤집히나 |
| ☐ | 3 | `--out .../run2_1080p` (선택) | 4 | 2.1 min | 1080p H.265 vs 540p MJPEG |
| ☐ | 4 | Step 1을 `--scene-label motion`으로 재실행 (선택) | 20 | 10.7 min | P-frame penalty |

## 어느 컬럼이 답인가

* **판정 지표: `orb_inlier_rate`** (기본 `--rank-by`). SLAM front end가 실제로
  먹는 것이고, PSNR이 동점으로 매기는 설정들을 분리한다. 산포는 `orb_inlier_sd`.
* **참고:** `ssim_y`, `psnr_y_db` (+ `_sd`, `_p05`).
* **절대 판정 지표로 쓰지 말 것:** Laplacian variance (`lap_var`). 압축이 오히려
  값을 **올릴 수** 있어서 장면에 따라 방향이 뒤집힌다. `RANKABLE_METRICS`에서
  의도적으로 빠져 있다.
* **AprilTag:** `tags_known`(알려진 ID map 대비 검출)과 `tags_false`를 **따로**
  볼 것. `tags_detected`만 보면 안 된다 — 압축 아티팩트가 위양성 검출을 만들어
  검출 수가 **늘어난다** (실측: 11 → 13). `tags_false`가 0이 아니면 그 rate는  [UNVERIFIED: 산출물 없음 — docs/MEASUREMENT_AUDIT.md]
  dataset path 부적격이다.
* **링크 비용:** `measured_mbps`, `budget_frac`, `color_kb_frame`,
  `color_kb_frame_p95`, `burstiness` (= p95/mean; 키프레임 피크가 큐를 채워
  latency 스파이크를 만드는지).
* **무결성:** `extraction_ok`, `frames_extracted` vs `frames_saved`. SSIM이 좋아도
  5번 중 1번 추출이 실패하는 코덱은 **더 좋은 코덱이 아니다.**
* **결론 한 줄:** 출력 맨 아래의 `DECISION:` 과 그 아래 마진 줄. 마진이 프레임별
  `sd` 안쪽이면 `UNRESOLVED`로 인쇄된다 — 그러면 동전 던지기다.

## 나쁜 결과를 읽는 법

* **`DECISION` 아래에 `UNRESOLVED`** → 1등과 2등의 마진이 프레임별 산포보다 작다.
  둘 중 아무거나 골라도 되고, 굳이 가르려면 `--duration`을 늘려 프레임을 더 모으거나
  더 벌어진 rate 쌍으로 다시 비교할 것.
* **`under_textured = 1`** → reference의 ORB keypoint가 500 미만. 장면 탓이지
  코덱 탓이 아니다. 텍스처가 더 많은 벽으로 옮기고 재캡처.
* **`SCENE WAS NOT STATIC (or the sensor is very noisy)`** → 리그가 움직였거나
  게인이 너무 높다. 임계값 0.7은 미보정이라 정적 리그에서도 센서 노이즈 12 DN이면  [UNVERIFIED: 예시/유도값 — docs/MEASUREMENT_AUDIT.md]
  0.682까지 떨어져 false alarm이 난다. 마운트를 확인한 다음 gain/exposure를 볼 것.
* **`extraction_ok = 0`** → h26x 스트림을 다시 못 읽는다. 화질 숫자가 아무리 좋아도
  실격.
* **상단 bitrate(8000-16000 kbps) 행들이 서로 붙어서 구분이 안 됨** → 인코더가
  아니라 **호스트 디코더 상한**을 읽고 있다 (아래 한계 참조). 이 구간의
  MJPEG-vs-H.26x 우열은 인용하지 말 것.

## 테스트 3이 말해줄 수 없는 것

* **★ reference는 같은 프레임이 아니다.** DepthAI 파이프라인은 boot에 고정되므로
  설정마다 별도 연결이고, 따라서 reference와 test는 **정적 장면의 서로 다른
  노출**이다. raw arm을 temporal median stack으로 접어서 shot noise를 없애는 게
  최선이지만, 그래도 same-frame PSNR이 아니다. **이 숫자는 설정을 RANK 하는 데만
  쓰고 절대 fidelity 주장으로 인용하지 말 것.** `CORRESPONDENCE_NOTE`가 모든 표와
  `report.md`, `meta.json`에 함께 인쇄되는 이유다.
* **h26x 행에는 코덱과 무관한 상한이 있다.** raw / mjpeg / h26x가 각각 다른 호스트
  디코드 경로를 탄다 (OpenCV NV12 / libjpeg / ffmpeg→PNG→imread). 이 호스트에서
  직접 측정: **손실이 정확히 0인 bit-exact H.264 스트림**을 같은 채점 경로에 태우면
  PSNR **45.60 dB** / SSIM **0.9962** / ORB inlier **0.9156**이 나온다. 즉 h26x  [UNVERIFIED: 산출물 없음 — docs/MEASUREMENT_AUDIT.md]
  행은 비트레이트를 아무리 올려도 저 값을 못 넘는다. **사다리 상단(8000-16000
  kbps)에서의 MJPEG-vs-H.26x 비교는 인코더가 아니라 디코더를 읽는 것이다.**
  MJPEG은 같은 실험에서 true loss와 0.1 dB 이내로 일치해 이 패널티가 없다.  [UNVERIFIED: 산출물 없음 — docs/MEASUREMENT_AUDIT.md]
* **코덱 축이 경과 시간과 confound 되어 있다.** 실행 순서가 항상
  `none → mjpeg → h264 → h265`이고 reference는 제일 먼저 도는 raw run에서 만들어진다.
  20 설정이면 마지막 h265 행은 reference로부터 **~10분** 떨어져 있고, 조명·노출·장면의
  느린 드리프트가 전부 늦게 돈 설정에 청구된다. 브래킷 방법: `--codec` 순서를
  뒤집어 한 번 더 돌리거나, raw 팔을 끝에 하나 더 잡아 두 raw 행을 비교할 것.
* **정적 장면은 inter-frame 코덱에 가장 유리한 최선의 경우다.** H.264/H.265 행은
  과대평가되어 있다. 실제 위험 — surge + backscatter에서 P-frame이 무너지는 것 —
  은 `--scene-label motion`으로 다시 돌려야 보인다 (체크리스트 Step 4).
* **H.264는 BASELINE으로 고정되어 있다** (`pipeline.py:179`). CABAC 없는 하한만
  재는 것이고, depthai 2.32에 H264_MAIN / H264_HIGH가 있으므로 profile 축을 넣으면
  결과가 달라질 수 있다.
* **해상도는 스윕 축이 아니다.** reference median stack이 한 해상도에서만 유효하기
  때문. 해상도별 upper envelope를 얻으려면 해상도마다 별도 `--out`으로 돌린 뒤
  `runs.csv`를 합쳐야 한다 (Step 3).
* **AprilTag corner σ(서브픽셀 정밀도)는 구현되지 않았다.** 구현된 것은 검출 수와
  위양성 ID 수뿐이다.
* **h26x 승자를 아직 `c3_collect.py`로 가져갈 수 없다.** `source.py:429`가
  `dai.ImgFrame`에 없는 `msg.getFrameType()`을 부르고 있어서 `frame_type`이 항상
  빈 문자열이고, `dataset.py:341-344`의 첫 I-frame 게이트가 안 열린다. 이 스크립트는  [UNVERIFIED: 예시/유도값 — docs/MEASUREMENT_AUDIT.md]
  자체 `annexb_frame_type()`으로 우회했지만 **c3_collect의 h26x 데이터셋 캡처는
  여전히 0 바이트를 쓴다.** 인코더를 dataset path에 채택하려면 `source.py`를 먼저
  고쳐야 한다.
* **caveat이 `runs.csv` / `scores.csv` / 런별 `meta.json`에는 붙지 않는다.**
  표·`report.md`·최상위 `meta.json`에만 들어간다. 그림이나 논문으로 옮겨지는 건
  대개 `runs.csv`이니 주의할 것.

---

# 결과를 어떻게 쓰나 — config.py 기본값 후보

결과가 좋게 나왔을 때 **바꿀 값**과 **바꾸기 위한 조건**이다. 조건이 충족되지
않으면 손대지 말 것.

| 위치 | 현재 값 | 바꿀 값 | 조건 (어느 테스트가 승인하나) |
|---|---|---|---|
| `config.py` `StreamConfig.depth_size` | `None` (derived → 400p에서 640x360) | `(480, 270)` | T1 Stage 1b에서 480x270이 15/20 fps 둘 다 `hold()` 통과하고 640x360@20이 통과 못 할 때 |
| `config.py` `StreamConfig.isp_scale` | `(1, 2)` → 960x540 | `(2, 3)` → 1280x720 | T1 Stage 2에서 2/3 @15가 `hold(15)` 통과하고 Stage 3a의 코너도 통과할 때 |
| `config.py` `StreamConfig.mono_res` | `"400p"` | 그대로 두기 | **T2가 근거리를 원하면 400p 유지.** 800p는 MinZ를 300 → 600.2 mm로 두 배로 만든다. arm C가 원거리 bias 개선을 크게 보여줄 때만 재고 |
| `config.py` `StreamConfig.extended` | `False` | `True` | T2 arm B의 185 mm rung이 `ok=1`이고, 300 mm 이상에서 arm A 대비 `fill_roi_inlier` 손실이 수용 가능할 때 |
| `config.py` `StreamConfig.confidence` | `None` (→ preset 245) | 명시적 값 (예: `200`) | T2에서 245가 fill을 지어내고 있다는 증거가 나올 때 (130 mm rung의 `fill_roi`) |
| `config.py` `StreamConfig.median` | `"5x5"` | 계측 경로는 `"off"` | T2에서 5x5가 잔차만 낮추고 fill 이득이 없을 때. **프리뷰 경로는 5x5 유지** |
| `config.py` `StreamConfig.mjpeg_quality` | `90` | `80` | T3에서 q80과 q90의 `orb_inlier_rate` 차이가 `orb_inlier_sd` 안쪽일 때 (= `UNRESOLVED`) |
| `config.py` `MJPEG_RATIO` | `6.0` (스칼라) | 해상도별 표 | T1의 `color_kb_frame`이 1080p에서 8:1대를 재확인할 때. 지금은 1080p 추정이 ~1.4배 보수적 |
| `config.py` `POE_BUDGET_MBPS` | `90.0` | 실측값 | T1 Stage 1a anchor가 README의 91.5와 5% 넘게 어긋날 때만 (케이블/스위치가 바뀌었다는 뜻) |
| `c3_collect.py` `RESEARCH_DEFAULTS["color_encode"]` | `"none"` | `"h265"` 또는 `"mjpeg"` | T3가 운용점에서 raw와 구별 불가를 보이고 **동시에** `source.py:429`의 `getFrameType` 버그가 고쳐졌을 때. 둘 중 하나라도 아니면 `none` 유지 |
| `c3_collect.py:223` `--min-mm` 기본값 | `300.0` | `100.0` | `extended`를 기본으로 올린 경우. 300은 extended가 여는 149-300 mm 구간을 화면에서 통째로 지운다 |

**바꿀 때의 규칙 두 가지**

1. **측정 증거 없이 바꾸지 말 것.** 특히 `c3_collect`의 인코더 기본값은,
   `metadata.json`에 근거 숫자(예: `ORB inlier 0.94 vs raw, tags_false 0`)를 함께
   기록한다는 조건에서만 방어 가능하다. `"H.265"`라고만 적힌 데이터셋은 방어할 수 없다.
2. **depth는 절대 인코딩하지 않는다.** 16 bit depth를 8 bit 비디오 코덱에
   통과시키는 것은 범주 오류다. 레포는 한 번도 그런 적이 없고, 앞으로도 그래야 한다.

**즉시 얻을 수 있는 것 (테스트 결과와 무관)**

현재 dataset 운용점(컬러 480x270 + depth 480x270 @20 fps)은 링크의 **53%만**
쓴다. 컬러를 960x540 MJPEG q90으로 올리면 21.5 + 41.5 = 62.9 Mbit/s = 천장의  [UNVERIFIED: 예시/유도값 — docs/MEASUREMENT_AUDIT.md]
**69%**로, **코덱을 하나도 안 바꾸고 화소 면적 4배**를 지금 얻을 수 있다.
69%는 jitter-free 구간(60% 이하)보다는 위지만 드롭 구간(85% 이상)은 아니다 —
T1 Stage 1b가 정확히 이 확인 런이다.


<!-- MEASUREMENT AUDIT (2026-08-04): the following numbers appear inside code
     blocks above and could not be annotated inline without corrupting them. -->

> **[UNVERIFIED]** 아래 줄의 수치는 출처 감사에서 미검증으로 분류되었다 — 자세한 근거는 [docs/MEASUREMENT_AUDIT.md](docs/MEASUREMENT_AUDIT.md).
>
> - `c3_camera/TESTING.md:130` — 산출물 없음 — `#      "960x540 mjpeg @15 + depth 640x360" = 71.4 Mbit/s (78%), p50 39 ms, p95 8`
> - `c3_camera/TESTING.md:131` — 산출물 없음 — `#      @20 행은 인코더 표에서 19.1 fps / drop 5%.`
