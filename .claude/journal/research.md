# P3 — Research + Verify outputs

<!-- Populated when P3 (/research-verify) lands. One block per question:
     ## YYYY-MM-DD <question>
     verified (+sources) · rejected (+why) · provenance -->

## 2026-06-23 — [RESOLVED ✅] Isaac Sim 5.0 + RTX 5090 학습 완주 (IOMMU 진단은 부분적이었음)
실측 결과(구현/디버깅 세션): 위 verifier들의 "IOMMU P2P가 부팅 hang의 원인" 결론은 **부분적으로만 맞음**.
- `amd_iommu=off` 적용 → kit 로그에 `IOMMU is disabled` 뜨고 P2P 검증은 사라졌으나 **부팅은 여전히 `omni.physx Using CUDA device ordinal 0`에서 행**(GPU util 0%, CPU 스핀). 즉 IOMMU P2P는 표면 증상, 그 아래 **PhysX GPU init 행**이 본질.
- `CUDA_MODULE_LOADING=EAGER` 무효. `--device cpu`는 부팅·학습 정상(전체 파이프라인 검증, 문제=GPU PhysX로 격리).
- **진짜 해결 = 드라이버를 CUDA-13(580/595)에서 CUDA-12.8 브랜치 R570(570.211.01-open)으로** (IsaacLab #3448: PhysX-on-Blackwell엔 R570 권장). Isaac Sim5.0/torch cu128/warp 전부 CUDA12.8 빌드라 드라이버도 12.8이어야 함. 결과: 2048 envs·400 iter·~19.7M steps GPU 학습 완주(보상 0.78→~76).
- warp 1.8.1 swap·amd_iommu=off는 유지(각각 cuDeviceGetUuid·P2P 해결). 최종 구성/명령은 `ISAAC_AUV_SETUP_RUNBOOK.md` 상단 + [[isaac-auv-env-feasibility]].

## 2026-06-23 — Isaac Sim 5.0 부팅 hang fix로 제안된 "TiledCamera→Camera" 적대적 검증
researcher: web(IsaacLab/IsaacSim issues, NVIDIA/warp CHANGELOG, docs.isaacsim) · 적대적 cross-check

- **판정: unlikely.** 제안 fix(TiledCamera 대신 표준 Camera)는 **부팅 hang과 무관**. IsaacLab#4951이 명시: TiledCamera hang은 **런타임**(env.reset/TiledCamera.update)이며 "warp 커널은 sm_120에서 정상, hang은 omni.replicator tiled 렌더 파이프라인". 부팅은 카메라 코드 도달 전이라 이 우회는 부팅을 succeed시키지 못함.
- **부팅 hang의 실제 원인(직전 2026-06-23 entry와 일치):** Kit는 log-init **이전이 아니라** "Simulation App Startup Complete" 도달 후 PhysX CUDA context에서 stall. 마지막 줄 "IOMMU is enabled. Running CUDA peer-to-peer bandwidth and latency validation". IsaacLab#1764: IOMMU-on P2P 검증이 ~362s(0.2GB/s, 49ms) hang. NVIDIA 공식: 베어메탈 Linux는 IOMMU-on P2P 미지원→disable 권장.
- **실측 재확인:** `/proc/cmdline`에 iommu off 플래그 없음(아직 미적용). 단독 `warp 1.8.1 wp.init()` 로컬 **exit 0/~수초**(sm_120, mempool, Toolkit12.8/Driver13.0) → warp 자체 hang 아님. extscache `omni.warp.core-1.7.1+lx64/warp/config.py` version=1.8.1, `warp.1.7.1.bak` 백업 존재 → swap 정상.
- **warp 버전 검증:** cuDeviceGetUuid/error36은 **warp 1.8.1 "Fixed"에서 해결**(GH-851 "Fix driver entry point error for cuDeviceGetUuid caused by using an incorrect version"). 1.8.1 changelog에 Blackwell/CUDA13/init-hang/mempool 언급 **없음** → 1.8.1은 cuDeviceGetUuid만 고침, 부팅 hang은 별개(=IOMMU).
- **실제 해법:** GRUB에 `iommu=off`(또는 `intel_iommu=off`/`amd_iommu=off`) 추가 → update-grub → reboot. headless여도 동일(P2P 검증은 headless 무관).
- **drive 주의:** Blackwell 부팅 crash는 **R590(595.xx) 브랜치**가 알려진 원인(forum scenedb)이나, 이 머신은 580.167.08(R580 validated) → 드라이버는 원인 아님.

sources: github.com/isaac-sim/IsaacLab/issues/4951 · /issues/1764 · /issues/2081 · github.com/isaac-sim/IsaacSim/issues/229 · github.com/NVIDIA/warp CHANGELOG(GH-851, v1.8.1) · forums.developer.nvidia.com/t/.../366252 · docs.isaacsim 5.1 requirements

## 2026-06-23 — Isaac Sim 5.0 headless 부팅 hang (RTX 5090) 진단·fix 적대적 검증
verifier (적대적) · 대상: warp GSP-vs-JIT 진단 probe 제안

- **반증(핵심):** 제안된 "GSP firmware halt vs soft JIT stall" 이분법은 이 머신엔 부적합. 실측: `nvidia-smi` 즉답(GPU idle P8), `dmesg` Xid **0건**, NVRM 에러 없음 → GSP class(#1111: nvidia-smi도 hang·SSH 끊김·hard power-cycle, 45분/3000req zero-gap 필요)와 정반대. 즉 제안의 진단 분기는 항상 "recoverable JIT/boot stall"로 떨어져 무의미.
- **반증(전제):** 프롬프트의 "Kit이 로그조차 안 쓰고 log-init 이전에 hang"는 디스크와 불일치. 오늘자 로그(`kit_20260623_092407.log`, 366KB/2132줄 등 다수) 전부 **"Simulation App Startup Complete" 도달 후** PhysX CUDA context(`omni.physx ... Using CUDA device ordinal 0`)에서 멈춤. warp 시작은 98ms로 정상.
- **검증(실제 stall point):** 멈춤 직전 마지막 줄이 `IOMMU is enabled. Running CUDA peer-to-peer bandwidth and latency validation`. NVIDIA 공식: 베어메탈 Linux에서 IOMMU-on이면 CUDA P2P 미지원→이 검증이 수 분 hang(보고 306s, BW 0.2GB/s) → **IOMMU off 권장**(intel_iommu=off/amd_iommu=off, /etc/default/grub+update-grub+reboot). `/proc/cmdline`에 off 플래그 없음 확인.
- **warp 1.8.1 swap 검증:** cuDeviceGetUuid/CUDA error36은 warp<1.8.1 × R580 드라이버 mismatch가 원인, **warp 1.8.1에서 fix**(NVIDIA/warp#940/#851, maintainer shi-eric). 유저 swap 정상(extscache `omni.warp.core-1.7.1+lx64/warp/config.py` 및 pip warp 모두 version 1.8.1; 캐시에 sm120.ptx 생성됨). 즉 warp는 더 이상 원인 아님 → 부팅 hang은 warp가 아니라 **IOMMU P2P 검증**.
- **판정:** 제안 fix = **unlikely**(부팅을 succeed시키지 못함; 진단만 하고 GSP 결론도 이 머신엔 오진). 실제 해법은 IOMMU 비활성화.

## 2026-06-22 — BlueROV2 회전 added mass: von Benzon/Wu 출처·수치 검증 (P2 후속)
researcher: control-theory-advisor · verifier: verifier (적대적)

- **Verified:** von Benzon et al. 2022 (JMSE 10(12):1898, DOI 10.3390/jmse10121898) 실재, **Fossen은 저자 아님**(원 리뷰의 "von Benzon & Fossen"은 오기 — Fossen은 모델 프레임워크). Eidsvik(2015) 경험식 오차 **회전 30–100%** / translational 10–20% → 회전 added mass는 order-of-magnitude only.
- **Uncertain(2차 출처로만 확증):** 회전 added mass ≈ Kp'0.189 / Mq'0.135 / Nr'0.222 kg·m²(von Benzon/Eidsvik). 여러 인용 논문이 동일 set 재현하나 **원문 표를 1차로 확인 못 함**(verifier도 PDF fetch 불가). researcher가 댄 provenance(RG fig 366613202=Hadi/Sensors)는 verifier가 **반증** — 그 그림은 PMC9824147(다른 논문) 소유.
- **Rejected:** 원 P2 주장 패턴 "roll 최소(Kp'≈0.07), Nr'≫Kp'" → **틀림**. 실데이터는 **pitch 최소·yaw 최대**(Nr'0.222 > Kp'0.189 > Mq'0.135, ~1.6×). 0.07/0.18/0.22 triple은 어느 데이터셋과도 불일치.
- **추가:** 경쟁 데이터셋 **Nr'≈0.40**(BR forum/Wu Heavy lineage) 존재 → 문헌이 회전값에 불합의(0.40을 arXiv:2405.00269에 귀속한 것도 약함).
- **판정/권고:** **지금 [0.12,0.12,0.12] 바꾸지 말 것.** ① 후보 0.189/0.135/0.222가 2차 확증뿐 ② 방법 오차 30–100%라 1.6× 비등방이 noise 내부(false precision) ③ 경쟁셋 0.40과 우열 근거 없음 ④ 우리 translational은 MarineGym set([5.5,12.7,14.57])이라 회전값만 이식하면 식별 혼합. 비등방 원하면 von Benzon **set 전체** 이식+order-of-mag 명시; 진짜 해법은 **자체 system ID**(free-decay/pendulum; docs/REAL_HYDRO_VERIFICATION.md).
- **provenance(핵심수치):** Kp'0.189/Mq'0.135/Nr'0.222 → von Benzon 2022(Eidsvik)로 *추정*, 1차 표 미확인·2차 인용으로만 확증(신뢰도 medium).

sources: vbn.aau.dk PDF · doi.org/10.3390/jmse10121898 · mdpi.com/2077-1312/10/12/1898 · arXiv:2405.00269 · BR forum 13065 · Wu2018 Flinders

## 2026-06-22 — Isaac Sim 4.5.0 시스템 요구사항 + RTX 5090(Blackwell sm_120) 지원 여부 (P3)
researcher: web (docs.isaacsim/forums/GitHub) · cross-verified across official docs + maintainer threads

- **Verified(공식 4.5.0 requirements page):** GPU min RTX 3070(8GB) / rec RTX 4080(16GB) / ideal RTX Ada 6000(48GB). **RT-core 필수**("GPUs without RT Cores (A100, H100) are not supported"). RAM 32/64GB, disk 50GB→500GB SSD, Linux driver **535.129.03**, Windows 537.58, Ubuntu 20.04/22.04 + Win10/11, Python **3.10**.
- **CRITICAL — Blackwell/RTX 50:** Isaac Sim **4.5.0은 Blackwell sm_120 공식 미지원**. 공식 4.5.0 req page에 50-series/Blackwell 언급 전무. 번들 PyTorch **2.5.1**(cu118/cu121)은 sm_50..sm_90까지만 컴파일 → sm_120 커널 없음 → RTX 5090에서 `CUDA error: no kernel image is available for execution on the device`. Kit/Warp/PhysX/NVRTC도 sm_120 미포함(IsaacLab #2652). NVIDIA: "compatibility problems between Blackwell GPUs and the Kit version in Isaac Sim (4.5.0 and 4.2.0)."
- **First official Blackwell support = Isaac Sim 5.0** (PyTorch 2.7.0+cu128, sm_120 across components; 4.5.0 blurry/noisy render bug도 5.0에서 fix). Maintainer: real fix는 5.0, 4.5 워크어라운드 아님.
- **4.5.0 워크어라운드(불완전):** `pip install --pre torch torchvision --index-url .../nightly/cu128` → PyTorch 텐서는 돌지만 viewport 렌더 noisy/voxel(Ubuntu), Kit/PhysX 레벨 sm_120 부재는 안 풀림. 권고: RTX 5090이면 **5.0+ 쓸 것**.
- **Install 방법 2가지:** ① binary/workstation(번들 python.sh 3.10) ② pip `isaacsim[all,extscache]==4.5.0 --extra-index-url https://pypi.nvidia.com` into conda py3.10.
- **sources:** docs.isaacsim 4.5.0 requirements.html, IsaacLab disc #1888/#2869, issue #2652, forum 336193, IsaacLab pip_installation(v2.1.0).

## 2026-06-22 — Learning-to-Swim(warplab/isaac-auv-env) 내 PC 실행 가능성 + Runbook (P3, 워크플로우 7-agent)
researcher: 병렬 web(Isaac Sim/Lab reqs) + repo deep-dive(general-purpose) · verifier: 4-claim 적대적

- **Lab↔Sim 짝(verifier REJECT):** "Isaac Lab 2.2.0 = Isaac Sim 4.5.0 공식 짝"은 **틀림**. 2.2.0의 1차 짝은 **Isaac Sim 5.0**(SIGGRAPH 2025 동시 GA); 4.5는 *backward-compat*만. 4.5.0의 canonical 짝은 Lab **2.1.0**. 단 repo README가 2.2.0+4.5.0을 직접 검증(commit 0d520b2)했으므로 그 조합 자체는 유효.
- **Blackwell 우회 핵심(verifier):** 4.5.0 번들 torch 2.5.1=sm_90까지(→`no kernel image`). **Lab 2.2.0는 torch 2.7.0+cu128 pin → sm_120 포함**이라 PyTorch 레벨 에러는 우회되나 Kit/PhysX/Warp sm_120 부재는 잔존. RTX 5090 정도(正道)=**Isaac Sim 5.0 + Lab 2.2.0**(Py3.11).
- **repo deep-dive(코드 확정):** task id `Isaac-WarpAUV-Direct-v1`(__init__.py:20) ✅; **symlink명 반드시 `isaac-auv-env`**(entry_point 일치; 폴더명 bluerov2_issac_paper/READMEdocker오타 isaac-warpauv-env 아님). Migration 4.0→4.5 **fully-applied**(obs/act/state space warpauv_env.py:70-72, articulation_enabled=False warpauv.py:17-19). requirements.txt 없음(deps=IsaacLab+pandas/cv2). 차량=WARPAUV 22.7kg 6-thruster(≠BlueROV2). 제공 weights는 단일파일→`play_poshold.py --play_checkpoint`(cv2 GUI 필요), 표준 play.py는 logs/rsl_rl/warpauv_direct/<run> 레이아웃 요구.
- **HW 판정:** RTX5090/32GB/60GB RAM/625GB/Ubuntu22.04/drv595 → Blackwell 외 전부 ✅. **결론=조건부 가능**(Sim5.0 권장). conda **robust 재사용 금지**(Py3.14·torch없음). → Runbook: `ISAAC_AUV_SETUP_RUNBOOK.md`. [memory: isaac-auv-env-feasibility]
- **sources:** IsaacLab disc #3167/#1888, release_notes, NVIDIA GA blog, forum 336193/337381, pytorch #159207, repo files.

## 2026-06-23 — Isaac Sim 5.0 Kit headless STARTUP HANG on RTX 5090 Blackwell sm_120 (warp가 성공적으로 로드될 때만) (P3)
researcher: web(IsaacSim/IsaacLab issues + NVIDIA forums + open-gpu-kernel-modules) · 증상-매칭 적대적 검증

- **증상 구분(핵심):** 우리 행은 *Kit 로그 쓰기 이전* pre-init 행(main thread hrtimer_nanosleep, ~24% CPU, no crash, **omni.warp.core가 warp를 성공 로드할 때만** 발생). 가장 유명한 IsaacLab **#4951 TiledCamera 행은 런타임**(omni.replicator tiled 파이프, 100% CPU)이라 *부팅 행이 아님* → 직접 원인 아님. scenedb 부팅 **크래시**(forum 366252, IsaacSim #651)는 *행이 아니라 segfault* → 별개.
- **가장 근접 원인(가설, medium):** sm_120에서 warp의 첫 GPU 명령/JIT가 **GSP firmware/driver 레벨에서 stall**. open-gpu-kernel-modules **#1111**: 580-open(580.126.20) Blackwell GB202에서 **silent hard hang, Xid/NVRM 로그 0** — nanosleep·무출력·무크래시 패턴과 일치. warp 1.7.1=`cuDeviceGetUuid`(IsaacSim #229), 1.14=API 에러 → **1.8.1만 init 성공**하나 그게 sm_120 첫 디바이스 호출을 유발해 stall 트리거.
- **구체 워크어라운드(부팅 통과 우선순위):** ① `WARP_CACHE_PATH`로 cache 분리 + 단독 `python -c "import warp;warp.init();warp.force_load()"`로 **sm_120 커널 사전 빌드**(부팅 중 JIT 차단) ② Kit에서 **omni.warp.core 비활성** (`--/app/extensions/disabled/...` 또는 headless .kit에서 제거) 후 별도 standalone warp 사용 — warp 로드가 트리거이므로 ③ RTX 렌더 우회: `--/renderer/enabled=false` 류 + enable_cameras=False면 `isaaclab.python.headless.kit` ④ 드라이버: 595.xx branch는 Blackwell+Isaac 공식 known-bad(forum 366252, NVIDIA staff) → 580.65.06(5.1 validated) 계열 유지/맞춤 ⑤ `CUDA_MODULE_LOADING=EAGER` / RtPso async(`--/rtx/...async...`) 비활성으로 비동기 컴파일 대기 제거.
- **확증 fix(타인):** scenedb 크래시는 **드라이버 다운그레이드(591.74)로 해결**(forum 365335/366252) — 행과 다른 실패지만 드라이버-branch 민감성 입증. TiledCamera 행 확증 fix=**TiledCamera→Camera 교체**(#4951).
- **권고 first-try:** warp 커널 사전 캐시 + Kit에서 omni.warp.core 제거(부팅에 warp 불필요; 학습 텐서는 torch가 담당) → 그래도 행이면 `--/renderer/enabled=false`로 렌더 vs warp 이분.
- **sources:** IsaacLab #4951/#4961/#2483, IsaacSim #229/#651, forum 366252/365335/370054, open-gpu-kernel-modules #1111, Isaac Sim 5.1 known_issues.html.

## 2026-06-23 — Isaac Sim 5.0 Kit boot-hang on RTX 5090(sm_120): warp 버전 진단 (P3)
researcher+verifier (web/gh API 적대적 검증)

- **환경 확인(ground):** `env_isaaclab`에 omni.warp.core-1.7.1+lx64 번들 warp을 1.8.1로 교체(`warp.1.7.1.bak` 백업 존재). pip `warp`도 1.8.1. Kit가 import하는 건 ext 내부 `warp/`(extension.toml path=".").
- **Verified:** cuDeviceGetUuid(CUDA err 36)는 warp **1.8.1**(commit 91fcd4d)에서 수정 — 메인테이너 shi-eric, warp#940. R580 드라이버+구 warp 조합 버그.
- **Verified:** `warp.types.array`는 **1.10.0**부터 깨짐(types.py가 5786줄→51줄 thin shim, `array` 클래스 미재export). 1.8.1·1.9.1은 real class 보유(types.py:1848/2249). 즉 omni-glue 호환 창 = **1.8.1–1.9.1**. (유저가 1.14에서 본 break는 실은 1.10.0부터.)
- **Verified:** warp **1.9.0**부터 CUDA 13 빌드 지원(드라이버 13.0 매칭). 공개 태그에 **v1.8.2 없음**(1.8.1→1.9.0) — IsaacSim 5.1의 "omni.warp.core 1.8.2"는 NVIDIA Kit 내부 라벨, pip `warp-lang==1.8.2` 불가.
- **Verified(권위):** IsaacLab #4951 메인테이너 — Isaac Sim 5.1.0은 **580 계열 드라이버 필수**(예 580.65.06), R590/CUDA13 불가. IsaacSim 5.1은 omni.warp 1.7.1→1.8.x로 올림.
- **판정/권고:** 부팅 행은 warp가 sm_120에 처음 접촉할 때의 init/mempool/커널캐시 stall로 추정(omni.replicator TiledCamera 행과는 다른 코드패스). 1차 시도: `warp-lang==1.9.1` 교체(types.array 유지+CUDA13). 보조: `WARP_CACHE_PATH` 지정+precompile, `wp.config.enable_mempools_at_init=False`, `verify_cuda` off. 상한선 1.9.1 초과 금지(omni glue 깨짐).
- **sources:** github.com/NVIDIA/warp issues#940, /releases(v1.8.1,1.9.x); IsaacLab#3477,#4951,#2483; IsaacSim#229; warp CHANGELOG; IsaacSim 5.1 release notes. [memory: isaac-auv-env-feasibility]

## 2026-06-23 — 제안 fix 적대적 검증: "Isaac Sim 5.1.0(warp 1.8.2 번들)로 올리면 5090 headless 부팅 행 해결" (P3)
researcher+verifier (web: NVIDIA docs/forums + warp CHANGELOG + IsaacSim/IsaacLab issues)

- **판정: UNCERTAIN** (likely-works 아님). 깨끗한 버전-매칭 스택 + 호환 드라이버라 시도는 합리적이나, fix의 **인과 주장이 근거 박약** — "5.1의 1.8.2가 행을 고친다"는 입증 안 됨.
- **핵심 반증(warp 1.8.2 ⊅ 추가 Blackwell fix):** cuDeviceGetUuid(GH-851)·NVRTC-unsupported-arch PTX(GH-858) 두 Blackwell fix는 **warp 1.8.1**에 들어감(공식 CHANGELOG, main=1.8.1에서 끝남). 유저는 **이미 1.8.1 패치 완료**. 5.1의 omni.warp.core 1.8.2는 1.8.1 위 **Z-bump bugfix**일 뿐, 별도 Blackwell *행* fix 문서화 없음. → fix의 전제("5.1이 fixed 1.8.2를 번들")가 유저가 이미 가진 것과 동급.
- **5.1은 행을 새로 고치지도, 재현하지도 않음(중립):** ① 5.1.0 TiledCamera/replicator **행 여전히 존재**(IsaacLab #4951, RTX5090 sm_120) — 단 런타임 렌더 파이프, 부팅 행 아님 ② 5.1.0 rtx.scenedb.plugin **부팅 크래시** 다수 보고(RTX 5070Ti/5080/5090/4090; forum 366252) — 단 주로 **595/R590 branch**. 유저 증상(hrtimer_nanosleep, pre-Kit-log, warp 로드 시)과 정확히 일치하는 5.1 사례 **무**.
- **드라이버는 OK:** 5.1.0 공식 Linux 검증 드라이버 = **580.65.06**(requirements.html). 유저 580.167.08 ⊇ 충족. 595/R590는 Isaac+Blackwell 공식 known-bad. → 드라이버 이유로 5.1 막히진 않음.
- **호환성 리스크:** 공개 태그에 v1.8.2 없음(1.8.1→1.9.0) → `pip install warp-lang==1.8.2` 불가. fix는 isaacsim==5.1.0.0 extscache로 omni.warp.core 1.8.2를 받는 경로라 이 함정은 회피하나, Lab 2.3.* 페어링·task 코드(4.5→5.0→5.1 마이그) 재검증 필요.
- **권고:** 5.1 업그레이드를 만능 해결책으로 보지 말 것. 먼저 **무비용 진단**(부팅에 warp 불필요): Kit에서 omni.warp.core 비활성 + warp 커널 사전 precompile(`WARP_CACHE_PATH`) + `--/renderer/enabled=false`로 render-vs-warp 이분(앞 6/23 엔트리). 이게 행을 통과시키면 5.1 불요. 5.1로 가더라도 행 원인이 다른 코드패스면 그대로 재현 가능 → blind upgrade 비권장.
- **sources:** docs.isaacsim 5.1.0 release_notes/known_issues/requirements.html; NVIDIA/warp CHANGELOG.md(GH-851/858 @1.8.1), /releases; IsaacLab #4951/#4961/#3477; IsaacSim #229; forum 366252. [memory: isaac-auv-env-feasibility]
- 2026-06-23 [verifier×3 workflow] Q: train 명령(`--num_envs 2048 --max_iterations 400 --seed 1 --headless`)이 Learning-to-Swim 논문 학습과 시드만 다른가? → REFUTED. iteration·보상·PPO 하이퍼파라미터·env 설정은 전부 커밋된 논문 기본값과 동일(env가 심볼릭링크라 byte-identical). 그러나 "시드만 다르다"는 부정확: (1) 논문 기본 시드=42(--seed 생략 시), 우리 기존 76짜리 run이 이미 seed 42였음 → 이미 논문설정 그대로 돌려 76; (2) warpauv_env.py:155의 무조건 `torch.manual_seed(0)`이 --seed 효과를 부분 무력화; (3) 결정타 — 논문은 Isaac Sim 4.5.0, 우리는 5.0(+RTX5090/R570/CUDA12.8/TF32) → 시드 무관하게 물리궤적·가중치 발산, 비트/궤적 재현 불가. 결론: 설정(방법론) 재현은 충실하나 "동일 정책"이 아니라 "비슷한 성능분포"가 목표여야 함. [memory: isaac-auv-env-feasibility]
- 2026-06-24 [experiment-diagnostic-analyst] Q: 여러 시드 학습(seed 42/1/2/3/4) 성능 평가 → 5개 run이 TensorBoard reward 곡선상 비트 단위 동일(maxdiff=0.00). 시드 무효 실측 확정(warpauv_env.py:155 torch.manual_seed(0)이 원인). 성능을 가른 건 iteration뿐: reward 76(400it)→93(800it)→피크 97.5(@1347)→90.8(1600, 피크후 하락). 논문 95-100 도달 = 정체 원인은 시드/Sim버전이 아니라 짧은 학습(400it). 최고 저장 체크포인트=model_1250.pt(reward 96.2). 과제성능(model_1250 vs 원본 model_399): 자세오차 중앙값 73°→29°, 위치 0.235→0.206m, 엄격성공(≤0.1m·≤10°) 0%→0.26%(여전히 정밀 station-keeping은 못함). 권고: 진짜 시드 ablation 하려면 line155 제거 필요; 배포엔 model_1599 아닌 model_1250. [memory: isaac-auv-env-feasibility]

## 2026-06-26 연안/조석류 0.4 m/s가 현실적인가 (해류 DR 범위 0~0.4 m/s 정당화)
- **Verified**:
  - von Benzon et al. (2022) JMSE 10(12):1898 — BlueROV2 오픈소스 벤치마크 시뮬레이터, 해류를 n-frame 상수 비회전류로 모델(Assumption 5), 사용 벡터 (0.3536, 0.3536, 0) m/s = **크기 0.5 m/s** [ResearchGate fig8 caption; PDF vbn.aau.dk/.../jmse_10_01898.pdf]. (verdict: substance verified; "0.5"는 성분 0.3536의 합성 크기)
  - Gabl et al. (2020) Data 5(3):57 — FloWave BlueROV2 실험, 해류 **상한 1 m/s**까지 sweep (대표 범위 0~1 m/s) [doi:10.3390/data5030057]. (ceiling verified; 중간 step값은 미확인)
  - NOAA: "strong tidal currents ... **eight knots or more**" (~4.1 m/s) [oceanservice.noaa.gov/facts/current.html — tutorial 페이지 아님].
  - Alderney Race 극단 조류 ~5 m/s [arXiv:2407.03827 + Phil.Trans.R.Soc.A reviews].
- **Uncertain**: BlueROV2 max forward speed — vendor GitHub spec=1 m/s(2kn), 신형 datasheet=1.5 m/s(3kn), 벤더 내부 불일치(PDF 본문 직접 확인 못함).
- **Rejected**: "BlueROV2 station-keeping 한계 ~1–1.5 kn(0.5–0.77 m/s)" — 특정 수치 출처 추적 불가(포럼/리셀러 lore, 검색LLM이 질의 echo). "벤더 공식 운용해류 한계는 미공개"라는 framing만 유효.
- **Provenance(0.4)**: soft engineering default("typical coastal/pool currents", docs/07_DISTURBANCES.md) → 문헌 검증 결과 **typical sheltered/coastal로 방어 가능**(von Benzon 0.5 m/s 벤치마크 바로 아래, Gabl 0~1 m/s의 하위 sub-band, 차량 authority 내), 단 **worst-case 아님**(강조류 1~5 m/s 범위 밖, 명시 필요). [memory: current-dr-range-justification]

## 2026-06-28 임펄스 "kick" 외란이 물리적으로 현실적/적절한가 (워크플로우 P3, 21 agents)
- **종합 판정**: kick은 어떤 단일 환경외란(난류·와류·슬래밍·항적·테더)의 충실한 모델이 **아님**. 그러나 다족로봇 push-recovery의 수중 아날로그로서 **강건성/회복 시험으로는 타당**. "관측기가 보상하는 외란"이 아니라 "회복 시험"으로 별도 보고할 것.
- **Verified(숫자)**:
  - 실제 연안/조석 난류 외력 = O(0.1–10 N) (sim 자체 계수 Xu_dot=5.5, Xuu=18.18로 재유도), kick 20–50N보다 1–2 자릿수 작음. 시간척도 초~수십초(적분 timescale ~6s, Milne 2017 RSPA), 0.15s 아님 → kick은 ~10–300배 짧음.
  - 조석 난류강도 TI ~6–10% (Thomson et al. 2012 IEEE JOE 37(3):363-374) — tens-of-% 는 tail.
  - 와류박리: 1–5N, 주기 3–8s(St·U/D), 임펄스 아님. 슬래밍: 완전침수(3m)에서 소멸.
  - DOB/ESO 설계는 d_dot≈0 가정(Chen et al. 2000 IEEE TIE 47(4):932-938; Do&Nguyen 2018 IEEE Access). 프로젝트 EAOB도 w_dot=0(eaob.py) → 0.15s 임펄스는 가정 최대 위반 → d_hat가 깔끔히 상쇄 못함(eval_dp.py도 "0.15s kicks not rejected"라 기록).
  - push-recovery 프로토콜(랜덤 시각·수평 위주 짧은 force pulse)은 표준 강건성 벤치마크 (arXiv 2407.04224 PA-LOCO, 2104.14534, 2203.01148).
- **Questionable(약점)**: ① 무토크(COM 적용) — 실제 임펄스(충돌·테더 스내치)는 off-COM이라 yaw/pitch 토크 유발, 빠진 게 가장 큰 결함 → 결과가 낙관적. ② 20–50N 크기·0.15s 지속은 환경외란 anchor 없음(엔지니어링 default), contact/worst-case probe로만 정당.
- **권고**: kick 유지하되 (1) "unmodeled impulsive perturbation/push-recovery test"로 재라벨(난류라 부르지 말 것), (2) 회복축(peak excursion+settling time)으로 별도 보고, (3) 최고가치 업그레이드=off-COM 적용으로 토크 부여, (4) 환경 realism은 kick 키우지 말고 colored gust velocity(von Karman/Dryden, tau~L/U~1–30s, TI~6–10%)를 water-velocity에 합산, (5) 옵션 상수 bias 케이스로 관측기 정상상태 상쇄 시연. [memory: kick-disturbance-realism]
- 2026-07-12 [workflow: 6 finders + 3 verifiers (P3)] Q: BlueROV2 Heavy에 Newton gripper + MarineSitu C3 부착 시 행렬(질량/관성/added mass/drag)을 어디서 구하나, 기존 구현 존재? → 기성 "페이로드-수정 행렬"은 어디에도 없음(AquaBot·UMI-U는 QYSEA+model-free 파라미터 미공개; BlueROV2+Alpha5 UVMS 오픈스택은 암 파라미터 비공개/식별만; MuJoCo ROV+그리퍼 전무). 방법론 컨센서스: 강체·부력은 벤더 질량으로 결정론적 합성(평행축), added mass/drag 증분(축당 수%~20%)은 공개 세트 간 30~100% 편차 미만이라 관행상 유지+문서화(DNV-RP-C205 build-up). 스펙 검증: gripper 524g/수중267g(구버전 616g 함정), C3 1700g/430g(Reef 스토어 공개). → heavy_gripper 변종으로 구현 완료. [memory: heavy-gripper-variant]
- 2026-07-19 [workflow: 3 researchers + 9 verifiers (P3)] Q: C3 실장착 위치를 Onshape에서 심으로 옮길 때 onshape-to-robot 쓸만한가 → 적합. v1.8.2(2026-03) MuJoCo 출력 1급 지원; 어셈블리에 `frame_<name>` mate connector를 달면 MJCF `<site>`(pos/quat)로 정확한 pose가 나옴(base body frame = 어셈블리 원점); 정적 어셈블리(FASTENED만)도 깔끔히 export. 단 `<inertial>`은 CAD 유래 fullinertia를 무조건 씀 → 절대 채택 금지(우리 hydro는 diag 필수, 검증된 합성 관성 유지), pose+mesh만 추출. 플랜 B: 어셈블리 단일 STL export(individual-parts 체크 해제, Y-up 체크 해제 — 다이얼로그 설정이 sticky라 매번 확인). [memory: onshape-c3-pose-pipeline]
echo journaled
- 2026-07-20 [build] C3 camera 방향 확인 + heavy_c3 변종: (1) C3 광축은 전방-수평이 맞음(평면 광학창=authored −Z가 mount에서 base_link +X=선체 반대쪽 향함; 둥근 캔=−Z=선체쪽). "방향 이상"은 빈 씬 검정 렌더 탓, POOL_TAGS 바닥 렌더로 해소. (2) 사용자 요청 "Onshape에 있는 것만 반영"→ 그리퍼(아직 CAD에 없음) 제외한 heavy_c3=heavy+C3 변종 신설: compute_payload_inertia.compose_c3/buoyancy_c3, gen_c3_variant.py, rov_model 등록, BlueROVHeavyC3.yaml, scene_bluerov_heavy_c3_tags.xml, test_heavy_c3.py 전부 통과(13.2kg, −3.1N, 카메라3 전방, 그리퍼 액추에이터 없음, PID 0.0cm). heavy_gripper는 그리퍼 CAD 추가 시 향후 config로 유지. [memory: onshape-c3-pose-pipeline, rov-model-variants]
- 2026-07-20 [build/debug] 사용자가 C3 렌더 90° 뒤틀림을 정면 뷰 비교로 적발 → 근본원인: MuJoCo 컴파일러는 mesh 주축 재정렬을 geom pos/quat에 합성함(컴파일된 quat = q_xml⊗mesh_quat 수치 확인). XML geom pose는 저작 원본 메쉬에 적용 → 구운 메쉬는 quat 불필요(identity). conj도 mesh_quat도 둘 다 틀렸었음. 수정: quat-less geom + 생성기 빌드타임 렌더-pose 검증(_verify_mesh_geoms). Onshape mate는 무결(재-export 바이트 동일). 스크린샷 정리(30개 삭제, 5개 폴더), C3 색 Onshape 외관 반영. [memory: mujoco-mesh-quat-convention]
