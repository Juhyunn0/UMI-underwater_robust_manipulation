# 풀장 외란 발생 장치 패널 — `./c3 disturb`

> 이 문서의 "외란"은 **외란을 만드는 장치**(갠트리에 매달린 T200 제트)를 뜻한다.
> rov_gui의 "disturbance"(DOB/EAOB가 **추정**한 외란 렌치)와는 다른 것이다.

## 무엇인가

풀장 위 3축 갠트리(FMC4030, TCP 192.168.0.30:8088)에 Damiao 회전 모터(SocketCAN
`can0`)가 달려 있고, 그 모터에 외부 T200 스러스터(Arduino Uno + ESC, 시리얼
−100..100 %)가 붙어 있다. `./c3 disturb`는 이 셋을 **한 창에서 수동으로** 조작하고
그 명령을 ROV 런과 같은 날짜 트리에 기록한다. 자동으로 도는 것은 갠트리 **패턴** 두 가지(Y-Z 원,
Y 왕복 — 아래 "패턴")뿐이고, 추력·각도 스케줄은 없다.

화면은 ROV 스테이션(`./c3 gui`)과 같은 어두운 테마·패널·상태 표시등(GANTRY / MOTOR / THRUSTER /
LOG)을 쓴다(`rov_gui/theme.py`, `rov_gui/widgets/indicators.py`).

장치를 실제로 움직이는 코드는 전부 동료의 `~/Desktop/whisker_vision`(위스커 실험
레포)에 있는 **디바이스 링크**(`motion_control.py`의 GantryLink·MotorLink·
ThrusterLink)이고, 이 패널은 그것을 **경로로 import만** 한다(수정하지 않음). 소프트
리밋 검사, 한 축씩 이동, POS_VEL 검증, 스러스터 계단 램프, 래치되는 STOP은 모두 링크
안에 있다. 갠트리 링크는 그 GantryLink에 이 리그용 메서드 다섯을 더한 것이다
(`disturbance_core.RigGantryLink`): 연결 중 축별 리밋 설치, 누르는 동안 한 번의 긴 상대
이동, 손을 떼면 보내는 **래치 없는 벤더식 정지**, 그 정지를 맨 앞에 둔 STOP 시퀀스, 그리고
패턴 경로(꼭짓점 목록을 정지될 때까지 반복하는 명령 하나).
패널이 더하는 것은 링크가 모르는 것들 — 버튼 활성 규칙, 운용 인터록, 가르친 홈(소프트
리밋), 정지가 안 먹을 때의 감시, 기록 — 뿐이다.

| 파일 | 역할 |
|---|---|
| `src/disturbance_panel.py` | PyQt5 창(`DisturbanceWindow`) |
| `src/disturbance_core.py` | Qt 없는 부분: 경로 심, `config/disturbance_rig.yaml`+CLI → `MotionSettings`, 로그(`DisturbLog`), git/ss 헬퍼 |
| `src/disturbance_preflight.py` | 연결 전 준비 상태 보고(장치를 열지 않음) |
| `config/disturbance_rig.yaml` | 포트·속도·조그 조각·램프 기본값 (CLI가 덮어씀, 쓰인 값은 meta.json에 남음). 갠트리 리밋은 여기 없다 — 현장에서 가르친다 |
| `data/disturbance_rig_limits.json` | 가르친 홈(소프트 리밋), 컨트롤러 카운터 mm. 다음 세션의 `Restore saved`용 |
| `src/tests/test_disturbance_*.py` | 시뮬레이터(whisker `motion_simulator`) 기반 헤드리스 테스트 (`./c3 test`에 포함) |

## 실행

```bash
./c3 disturb --preflight-only          # 준비 상태만 보고 (아무것도 연결하지 않음)
./c3 disturb                           # 실기. whisker conda env로 돈다
./c3 disturb --sim                     # 시뮬레이터, 하드웨어 없음
./c3 disturb --thruster-max 30         # 이 세션의 |추력| 상한
./c3 disturb --ramp-step 10 --ramp-interval 3   # 계단 램프 (기본값; 화면에서는 못 바꿈)
./c3 disturb --jog-piece 50            # 홈이 없는 쪽으로의 조그 한 조각 (기본 300 mm)
./c3 disturb --limits-file PATH        # 가르친 홈 저장 위치 (기본 data/disturbance_rig_limits.json)
./c3 disturb --no-motor-interlock      # 벤치 전용: 모터 홀드 없이 추력 허용
./c3 disturb --rig-note "jet faces +X at motor 90"   # meta.json rig_notes
./c3 env                               # whisker_vision 위치/브랜치, disturb 인터프리터 확인
```

- 인터프리터는 `whisker` env(`DISTURB_PY`로 오버라이드). `robust`에는 python-can·pyserial이
  없고, python-can 4.6.1은 `wrapt~=1.10`을 요구해 robust의 wrapt 2.2.1을 강등시키므로
  **robust에 pip install 하지 않는다**. `--sim`은 어느 env에서도 된다.
- `./c3 disturb`는 `LD_LIBRARY_PATH`(OpenFOAM/CUDA)를 비우고 실행한다. 직접 실행할 때는
  `env -u LD_LIBRARY_PATH python src/disturbance_panel.py …`.
- **`./c3 gantry`(피시아이 서베이 패널)·whisker `motion_panel.py`·`gantry_soft_limits.py`와
  동시에 쓸 수 없다**: 벤더 `.so`는 프로세스당 연결 1개, 컨트롤러는 TCP 클라이언트 1개다.
  패널은 연결 전에 `ss -tn`으로 점유를 확인하고 거부한다.

## 좌표계 — 숫자를 다른 도구와 섞지 말 것

- 패널의 mm는 **컨트롤러 카운터 그대로**다. 0은 "컨트롤러 전원을 켰을 때 축이 서 있던 자리"일 뿐
  의미가 없다. 이 컨트롤러의 펌웨어 리밋 쌍은 무효(200/200 units)이고, **스위치 홈은 쓰지 않는다**.
- **홈 = 소프트 리밋, 현장에서 가르친다(teach-in)**: 축을 한쪽 끝까지 조그로 옮기고 `−X home`을 누르면
  **지금 위치**가 X의 − 한계가 되고, 다른 끝으로 옮겨 `+X home`을 누르면 그 위치가 + 한계가 된다. 그 뒤로
  X는 그 둘 밖으로 조그되지도, Send되지도 않는다. 다시 누르면 새 자리로 옮겨진다. `Clear`는 그 축의
  두 홈을 지운다(그 축은 다시 한계 없음).
  - − 홈은 + 홈보다 1 mm 이상 아래여야 한다(아니면 거부, "Clear 먼저").
  - 한쪽만 가르쳐도 그쪽은 바로 막힌다(패널이 지킴). 양쪽이 다 있어야 링크에도 설치되고 `Send targets`가
    그 축에 열린다.
  - 홈이 **없는** 쪽으로는 소프트웨어 한계가 없다: 0 아래로도 가고, 막는 것은 물리 리밋 스위치와(+쪽)
    펌웨어 스톱뿐이다 — 축을 눈으로 보면서 움직일 것. 이동을 보냈는데 축이 움직이지 않으면(스위치·펌웨어
    스톱·고장) 조그를 스스로 끝낸다.
  - 가르친 값은 카운터 좌표라 **컨트롤러 전원을 끄면 무효**다(재투입 시 카운터가 0부터 다시 시작). TCP
    재연결로는 안 바뀐다(2026-10-26 런: 재연결 전후 X −389.978 mm,
    `data/20261026/1026_131512_disturb/samples.csv`).
  - 그래서 홈은 **마지막으로 본 위치와 함께** 저장되고, 다시 쓰일 때마다 그 위치와 대조된다:
    - 패널 안에서 Disconnect → Connect: 갠트리가 끊기 전 마지막 위치에 그대로(축마다 1 mm 이내) 있으면 홈을
      다시 설치하고, 아니면 **설치하지 않고** 빨간 "Homes NOT re-applied … controller power-cycled?"를 띄운다
      (`events.csv` `homes_dropped`). 정지가 안 먹어 전원을 끄라는 안내를 따른 뒤가 바로 이 경우다.
    - 패널을 다시 켜면 홈을 **자동으로 쓰지 않고** `Restore saved` 버튼과 저장값·시각만 보인다. 눌렀을 때
      갠트리가 저장 당시 위치에 그대로 있을 때만 복원되고, 다르면 거부한다(`teach_restore_refused`).
      그러니 Restore는 **연결 직후, 아무것도 움직이기 전에** 누른다.
    - 저장은 가르치기·Clear·조그 정지·Send 완료·Disconnect·창 닫기 때 된다. 패널이 강제 종료되면 저장 위치가
      옛것이라 Restore가 거부된다(안전한 쪽).
    - 잡을 수 없는 경우: 컨트롤러는 켜 둔 채 **드라이버 전원만** 끊었거나, 전원이 꺼진 동안 누가 손으로 축을
      옮긴 경우 카운터와 실제 위치가 어긋난다 — 그런 일이 있었으면 Clear 후 다시 가르친다.
  - 정지가 안 먹은 사건(`stuck`) 뒤에는 그 축의 홈을 **SUSPECT**로 표시한다(상태줄 빨강). 그 축을 다시
    가르치거나 Clear하면 풀린다.
  - 현재 위치가 홈 밖이면 바깥쪽으로의 조그는 거부되고 안쪽으로만 간다.
- `./c3 gantry`는 `~/.umi_gui_state.json`의 home reference·axis_sign을 적용한 **사용자 프레임**을
  쓴다. 두 도구의 좌표값은 서로 호환되지 않는다.
- 갠트리 축이 풀장에서 어느 방향인지(+X가 길이 방향인지, +Z가 어디인지)는 **미확인**이다
  (`config/config.yaml:8-10`의 주석과 값이 서로 다름). 현장에서 확인해 아래 칸에 적는다:
  - +X 방향: ________  +Y 방향: ________  +Z 방향: ________  스러스터 수몰 Z 범위: ________
- 제트 방향: 모터는 **연결할 때 서 있던 각도가 0°**다(2026-10-26부터, 아래 체크리스트 7). 그래서 모터
  각도는 세션마다 의미가 다르고, whisker 교정(`aim_calibration.json`, 2026-09-08: 역장착, 모터 90°에서
  +X)은 더 이상 이 패널의 각도에 맞지 않는다(meta.json `jet_calibration`에 참고용으로만 남음). 연결 전에
  마운트를 원하는 방향으로 두거나, 연결 뒤 저추력에서 눈으로 확인하고 `--rig-note`로 적는다.

## 현장 체크리스트 (순서대로)

1. **ESC 전원 킬스위치**를 손 닿는 곳에 둔다. 스러스터 펌웨어에는 host-loss failsafe가 없다
   (`whisker_vision/thruster_firmware/src/main.cpp:49-64`): PC가 죽으면(이 PC는 2026-09-10에
   3회 하드 크래시) 마지막 추력이 **계속 돈다**. 패널로는 막을 수 없다.
2. `./c3 gantry`·whisker 패널·`gantry_soft_limits.py`를 모두 닫는다.
3. 네트워크: USB-이더넷 동글(정적 192.168.0.35/24, NM 프로필 'Wired connection 2/3') 또는
   같은 /24의 wlan. `ping -c1 192.168.0.30`. **STOP ALL이 이 링크를 타고 가므로** 유선을 권장한다
   — WiFi가 끊기면 보낸 이동 명령은 끝까지 스스로 간다(최대 1400 mm / 50 mm/s ≈ 28 s [유도]).
4. 컨트롤러 전원을 켠/다시 켠 뒤: 펌웨어 Y 한계가 500 mm로 되돌아갔을 수 있다.
   `cd ~/Desktop/whisker_vision && python3 gantry_soft_limits.py`(인자 없음; 다른 클라이언트가
   없어야 함)로 보고, Y가 200/200 units면 `--write-y -4 320` 후 다시 확인. 패널의 한계 라벨에도
   `firmware stops: Y 500 mm`로 보이고, 그 너머의 타깃은 거부된다.
5. `./c3 disturb --preflight-only` → READY. WARN은 진행 가능, FAIL은 막힌다(`--no-preflight`로
   강제 가능하지만 이유를 알고 할 것). **장치는 묶음으로 판정한다**(2026-10-27 사용자 요청): 갠트리(`gantry client`·
   `gantry route`), 모터(`can bus`), 스러스터(`thruster port`) 중 **하나라도** 통과하면 나머지 장치의 FAIL은 WARN이 되고
   (`-> motor UNAVAILABLE: … press Connect motor`, 판정 앞에 `devices: … UNAVAILABLE` 줄) 패널이 열린다. 셋 다 실패할
   때만 막는다. 장치가 아닌 검사(interpreter, whisker_vision, data root, display)의 FAIL은 그대로 막는다. 빠진 장치는
   나중에 꽂고 그 `Connect`를 누르면 된다(실패한 연결은 오류를 띄우고 버튼이 `Connect …`로 돌아옴; can0은 whisker
   `damiao`가 Connect motor 때 `sudo -n`으로 올림). meta.json의 preflight 보고에 `devices`(available/unavailable)가 남는다.
6. `./c3 disturb` → **Connect gantry** → 한계 줄 확인(`limits mm: X – …`) → 컨트롤러 전원이 지난
   세션 뒤로 꺼지지 않았으면 `Restore saved`, 아니면 **가르치기**: 축마다 조그로 한쪽 끝(ROV·테더·사람과
   부딪히지 않는 마지막 자리)까지 가서 `−X home`, 다른 끝에서 `+X home`. 처음 몇 번은 **저속(`SPEED` 5–10)
   한 축씩, 손은 갠트리 전원 스위치 위**: 이 컨트롤러에서 손을 뗐을 때 정말 서는지 아직 실측이 없다(아래
   "검증 기록").
7. **Connect motor** (+ **Connect thruster**, 순서 무관) → 둘 다 연결되면 패널이 알아서 **Disable → (0.5 s) →
   Disable 한 번 더 → (0.5 s) → Zero here → Enable → 0.5 s 동안 각도 확인**을 한 단계씩(각각 링크 확인 뒤) 보낸다.
   그 결과 **지금 서 있는 각도가 0°**가 되고 모터가 그 자리를 홀드한다(정보 줄 `holding · 0° = where it stood
   at connect`, `events.csv` `motor_auto_zero`). 약 2 s 걸리고 [유도: 대기 0.5 s × 3 + 링크 확인; 실기 0회], 그동안 Set thrust는
   잠긴다. 마운트 주변은 비워 둔다.
   - **스러스터가 연결될 때까지 기다리는 이유**: 패널이 비정상 종료(SIGKILL·크래시)되면 아두이노는 마지막 추력을
     계속 낸다(펌웨어에 host-loss failsafe 없음). 스러스터 포트를 열면 아두이노가 리셋돼 1500 µs(0)가 되므로, 그 뒤에야
     마운트를 limp(Disable)로 만든다 — 도는 제트 아래서 마운트가 풀리지 않게(안전 리뷰 2026-10-26 H1). 그동안 정보 줄은
     `connect the thruster to zero + hold`. 같은 규칙이 **손으로 누르는 Disable**(버튼, 모터 Disconnect 대화상자의
     "Disable first")에도 걸린다: "제트가 꺼진 것을 안다" = 스러스터 링크가 ready이고 0이거나, 이번 세션의 마지막
     스러스터 연결이 에코된 0으로 닫힌 경우. 아니면 버튼이 비활성이고 정보 줄은 `connect the thruster, then Disable`
     (재검증 N1).
   - Disable을 두 번 보내는 이유: CAN에는 응답이 없어 링크의 "disabled"는 프레임을 **보냈다**는 뜻일 뿐이다. Zero는
     limp일 때만 되고(whisker `zero_here`), Enable은 그 자리를 잡는다 — 손으로 같은 순서를 했을 때 110°에서 0으로
     바뀐 뒤 Enable 후 −0.01°에 머물렀다(`data/20261026/1026_171856_disturb/samples.csv`, 실기 근거 1회).
   - 단계가 5 s 안에 확인되지 않거나, 추력이 0이 아니거나, STOP ALL이 눌리거나, **Enable 뒤 축이 1° 넘게 움직이면**
     (Disable이 안 먹어 드라이브가 옛 목표를 쫓는 경우) 시퀀스가 멈추고 — Zero를 보낸 뒤라면 Disable도 보낸다 —
     `motor_auto_failed`를 남긴다. 그 뒤엔 Enable/Disable을 손으로 한다. 이번 연결에서 0을 못 잡았으면 정보 줄이
     `NOT zeroed this connection`이다(각도는 이전 기준) — 다시 잡으려면 Disconnect → Connect motor.
   - 드라이브 상태를 모르는 채로 시퀀스가 멈췄으면 버튼이 **Disable**로 보인다(래치 중에도 누를 수 있음).
   Enable/Disable은 추력 0일 때만 된다(돌아가는 제트 아래서 마운트를 limp로 두지 않기 위해).
8. **Thruster connect**: 연결되는 것은 **아두이노(USB 시리얼)**뿐이다. 아두이노는 USB 전원으로 돌고, 리셋 뒤
   1500 µs(정지) 펄스를 내보낸다. 헤더의 `ARDUINO ONLINE`과 라벨의 `+10 %  ·  1540 µs`는 "아두이노가
   명령을 받아 그 펄스를 내고 있다"는 뜻이고, **ESC 배터리가 연결됐는지, 프로펠러가 도는지는 패널이 모른다**
   (펌웨어는 계산한 펄스 폭을 되돌려 줄 뿐 — `whisker_vision/thruster_firmware/src/main.cpp`; ESC 쪽 신호선은
   아두이노로 돌아오지 않음). 그래서 배터리가 빠져 있어도 화면은 똑같다(2026-10-26 14:09 런에서 확인).
   - **ESC 배터리는 0 %일 때만** 연결하고 ESC의 시작음(비프)을 듣는다. 명령이 0이 아닐 때 배터리를 꽂지 않는다.
   - 실제로 도는지는 눈·소리로 확인한다. 배터리 없이 Set thrust를 해도 로그에는 그 %가 남는다 —
     실험 기록에서 "제트가 실제로 켜져 있었는지"는 운용자 메모(`--rig-note`, 사진)로 남길 것.
   - 갠트리·모터는 스러스터(ESC 배터리, 아두이노 연결 여부)와 무관하게 움직인다.
9. 추력 부호 확인: 작은 값(예 ∓10 %)으로 제트 방향을 눈으로 확인하고 `--rig-note`에 적는다.
10. 로그는 첫 장치가 연결되면 **자동으로 시작**된다(`--no-autolog`로 끔). 폴더는 상태줄 `LOG …`.

## 버튼 의미

| 버튼 | 하는 일 | 조건 |
|---|---|---|
| `−X` `+X` (Y, Z도 같음) | **꾹 누르는 동안 부드럽게 계속 이동, 떼면 감속 정지.** 그쪽에 홈이 있으면 연속 이동 한 번이 **홈에서 끝나도록** 나가고, 없으면 한 조각(`jog_piece_mm`, 기본 300 mm)씩 — 계속 누르고 있으면 다음 조각이 이어진다(조각마다 잠깐 멈칫함). + 쪽 펌웨어 스톱(카운터 X 1650 / Y 500 / Z 100 mm)은 조각을 **줄이기만** 하고 끝점이 되지 않는다(카운터 좌표라 실제 위치가 정해져 있지 않음). 한 번 누름에 총 2000 mm까지(떼기 신호가 사라진 경우의 상한). 떼면 **항상** 그 축에 벤더식 감속 정지(`Stop_Single_Axis` mode 1)를 보내고 **래치하지 않는다**. **다른 축 버튼은 뗀 즉시** 누를 수 있고(2026-10-27), 뗀 그 축의 버튼만 그 축이 실제로 설 때까지(50 mm/s에서 1.43 s 이내, `data/20261026/1026_140909_disturb/events.csv`) 기다린다 — 컨트롤러는 끝나지 않은 축의 새 조그를 무시한다. 속도는 `SPEED` 칸(1–100 mm/s, 기본 50). **Z는 50 mm/s까지만**(Z의 카운터 범위가 100 mm인데 100 mm/s에서 뗀 뒤 약 10 cm를 더 감 [유도]) | 갠트리 ready, 래치 아님, 다른 명령 중 아님, 누를 축이 정지 중이 아님, **정지 중이 아닌 축은 모두 실제로 정지**(신선한 위치가 0.3 s 동안 0.1 mm 이내), 한계 끝이 아님 |
| `Send targets` | **최대 50 mm/s**(SPEED가 더 커도; 이 컨트롤러에서 Send를 한 번 돌려 본 뒤 올린다). **편집한 축만** 절대 이동(target 칸은 갠트리가 멈출 때마다 라이브값으로 다시 채워지므로, 바꾼 칸만 명령이 됨), X→Y→Z 한 축씩, 홈·펌웨어 스톱 안. 이동 중 그 축이 "시작..목표 ± 5 mm"를 벗어나면 STOP ALL(절대 이동이 상대로 읽히는 경우 등 — 이 컨트롤러에서 절대 이동은 미검증) | 편집한 축마다 두 홈이 모두 있고, 세 축 정지 |
| `−X home` / `+X home` | **지금 위치**를 그 쪽 소프트 리밋으로(아무것도 움직이지 않음). 저장되고 `events.csv` `teach`에 남는다 | 래치 아님, 조그·정지·이동 중이 아니고 세 축 정지 |
| `Clear` | 그 축의 두 홈을 지움 → 그 축은 한계 없음 | 그 축에 홈이 있고 조그·이동 중 아님 |
| `Restore saved` | 이전 세션이 저장한 홈을 다시 씀 — 갠트리가 **저장 당시 위치에 그대로 있을 때만**(아니면 거부) | 이 세션에 아직 아무 홈도 없고, 래치 아님, 세 축 정지 |
| `Recover gantry` | 이동이 끝나지 않고 busy가 남으면(타임아웃/60 s) 정지+링크 리셋 | 정체 판정 시만 |
| 패턴 `Start` / `Stop` | 아래 "패턴" | Start: 갠트리 정지·래치 아님·조그 중 아님 / Stop: 실행 중 |
| `Enable/Disable` | 모터 토크 on/off (on은 홀드). 연결 시 자동으로 0° 홀드가 되므로 보통 안 누른다. 드라이브 상태를 모르면 `Disable`로 보이고 Disable을 보낸다 | 양방향 모두 **추력 0** (+ Enable은 POS_VEL verified, 이번 연결에서 Disable 이력), 연결 시퀀스 중엔 비활성 |
| `Go θ` | 각도로 회전 — 현재 각도에서 **가까운 쪽**으로 간다(±180° 이음새를 넘어 먼 길로 돌지 않음; 속도 상한 `motor.vel_max_deg_s`, 기본 90 °/s). 0° = 연결 때 자리 | Enable + 정착 중 아님 |
| `Set thrust` | 계단 램프로 타깃 %(10 %/3 s 고정, YAML `thruster.ramp_*`) | 모터 Enable(홀드)이고 **텔레메트리가 신선**할 때 — 인터록. 막혀 있으면 이유가 주황 글씨로 보임 |
| `Turn off` | 계단으로 0까지 **내려감**(즉시 아님). 즉시 0은 STOP ALL(Esc) | |
| `STOP ALL (Esc)` | 갠트리 정지(비차단 호출이라 맨 먼저; 패턴 중이면 움직이는 축부터) → 스러스터 즉시 0 → 모터 현재 각도 **홀드**(limp 아님), 전부 래치. 예외: 연결 직후 0° 시퀀스가 Disable과 Enable 사이일 때 눌리면 모터는 이미 limp라 그대로 limp로 남는다(추력은 그때 잠겨 있음) | 항상 |
| `Reset` | 세 링크의 래치 해제. 아무것도 다시 명령하지 않음. 스러스터가 아직 0이 아니면 0을 다시 보내고 거부. **갠트리가 아직 움직이거나 위치가 신선하지 않으면 거부**("Reset refused" — 계속 움직이면 갠트리 전원, 링크가 죽었으면 Disconnect gantry). 해제 뒤에도 5 s 드리프트 감시 | 래치 중 |
| `● New log / ■ Stop log` | 새 폴더 / 현재 폴더 마감 | |

- 추력 램프 10 %/3 s(첫 계단은 바로, 0→50 %에 12 s). whisker 쪽 관찰: ESC/전원이 급변에 오작동함
  (`motion_config.py:93-95`). 2026-10-26부터 화면의 RAMP/EVERY 칸과 `Cut thrust` 버튼은 없앴다(사용자 요청) —
  값은 YAML/CLI(`--ramp-step`, `--ramp-interval`)로만 바뀌고 meta.json에 남는다. 즉시 0이 필요하면 STOP ALL.
  `_cut_thrust`는 내부에 남아 인터록(모터가 홀드를 잃으면 자동 0)에 쓰인다.
- Esc는 **패널 창에 포커스가 있을 때만** 동작한다(Qt 단축키). 모달 대화상자(모터 해제 확인)가 열려
  있으면 Esc는 대화상자로 간다 — 그래서 추력 0일 때만 열린다.
- STOP ALL 뒤 +0.3 s·+1.5 s에 스러스터 0을 **다시 보낸다**: whisker 링크의 램프 워커가 래치를
  SDK 락 밖에서 검사하고 전송은 락 안에서 하므로, 첫 0 뒤에 비0 스텝이 하나 물리적으로 나갈 수
  있다(`motion_control.py:1119-1129`). 로그의 `latched` 열이 STOP 이후의 권위 신호다.
- **왜 정지 방식을 바꿨나 (2026-10-26 실기)**: 처음 판은 손을 떼면 whisker 링크의 `stop_all()`(stop_run →
  축마다 `Stop_Single_Axis` mode 2, 래치)을 보냈다. 이 컨트롤러에서는 그 정지가 **상대 조그를 멈추지
  못했다**: 누를 때마다 300 mm 조각이 끝까지 갔다(X −90→−390, −390→−690, −690→−990 mm, 20 mm/s; 떼기,
  여러 번의 STOP ALL, TCP 재연결을 모두 지나서) — `data/20261026/1026_131512_disturb/samples.csv`·`events.csv`.
  그리고 떼기 뒤 감시가 STOP ALL을 눌러 래치해 버튼이 죽었다("한 번 누르면 다시 못 누름"). SDK 설명서는
  mode 1 = 감속 정지, mode 2 = 즉시 정지 [스펙: `src/gantry/mannual/FMC4030二次开发库详解V1.0.pdf`
  `FMC4030_Stop_Single_Axis`]이고, 벤더 Qt 데모와 `./c3 gantry`는 조그를 **mode 1**로 멈춘다. 그래서:
  - 떼기 = 그 축(그다음 나머지 두 축)에 mode 1, 래치 없음(`RigGantryLink.jog_stop`).
  - STOP ALL·창 닫기·Recover = 세 축 mode 1, 그다음 세 축 mode 2. whisker의 `stop_run`은 **뺐다**: 홈·보간
    운동에만 필요한데 이 패널은 둘 다 안 쓰고, 10-26에 "stop_run이 뒤 명령을 막았나 / mode 2가 조그에 안
    듣나"가 분리되지 않았기 때문.
  - **떼기(mode 1) 실측, 2026-10-26 14:09**: 18/18 정지, 20 mm/s 3.1 mm · 30 mm/s 0.4–6.8 mm · 50 mm/s X 5.5–16.9 /
    Y 8.6–27.2 / Z 5.7–22.5 mm, 모두 1.43 s 이내(`data/20261026/1026_140909_disturb/events.csv`의 `jog_stopped`; 거리는 뗄 때의 캐시 위치 — 최대 0.1 s
    묵음 — 에서 정지까지의 카운터 이동). **STOP ALL 안의 mode 1 → mode 2는 아직 미실측** — 저속 조그 중 Esc를 한
    번 눌러 확인할 것.
- **조그 안전 장치**:
  - 손을 떼면 링크가 "이미 멈췄다"고 해도 **항상** 정지 명령을 보낸다(링크의 busy는 "축이 서 있음"이 아니라
    "명령을 기다리는 중"이라는 뜻이라 믿지 않는다). 누른 직후 바로 떼서 이동이 아직 컨트롤러에 안 갔으면 그
    이동은 취소된다(세대 번호).
  - 누른 버튼의 눌림 상태가 사라지면(마우스가 버튼 밖으로 나감, 창이 포커스를 잃음, 버튼이 비활성화됨) 떼기와
    같이 정지한다.
  - 떼기 뒤 축이 **`1.5 s + 2×감속시간` 안에 서지 않거나**, `5 mm + 2×(감속거리 + 0.3 s 이동거리)`보다 더
    가거나(20 mm/s면 2.3 s / 25 mm, 기본 50 mm/s면 3.5 s / 85 mm, 100 mm/s면 5.5 s / 265 mm [유도: 가속 50 mm/s²,
    `disturbance_panel.py` RELEASE_*]), 정지 명령 전송이
    실패하거나, 위치 폴링이 끊기면 → **STOP ALL을 자동으로 누르고** "CUT THE GANTRY POWER" 경고를 띄운다
    (`events.csv` `stuck`). 정상으로 서면 `jog_stopped`(뗀 뒤 몇 mm·몇 s에 섰는지)가 남는다.
  - 선 뒤 5 s 동안(STOP ALL 뒤에는 Reset까지, Reset 뒤 다시 5 s) 그 자리에서 **누적** 2 mm 넘게 움직이면
    STOP ALL. 그 사이 다른 축을 눌러도 감시는 꺼지지 않는다(움직이라고 명령한 축만 빠짐).
  - 다음 명령(Send·가르치기·Restore·Reset·패턴)은 **세 축이 실제로 정지**했을 때만: 신선한(0.5 s 이내)·오류
    없는 위치가 0.3 s 동안 0.1 mm 안. 링크의 busy는 보지 않는다(정지가 안 먹어도 busy는 꺼진다). **조그만 예외**:
    방금 뗀 축이 아직 감속 중이어도 **다른 축**은 바로 조그할 수 있다. 정지 감시는 **축마다** 따로 돈다(축마다
    마감 시간·허용 거리; 두 번째 축을 떼면 같은 감시에 합쳐짐) — 한 축이라도 시간 안에 서지 않거나 너무 가면 STOP ALL.
    선 축은 그 자리에서 다시 드리프트 감시를 받는다. 두 번째 축을 떼면 그 정지(mode 1)가 아직 서는 중인 첫 축에도 가므로
    첫 축의 `jog_stopped`에 `stopped again by N later release(s)`가 붙는다(첫 정지가 무시됐어도 평범해 보이지 않게).
  - **감속 확인(2026-10-27)**: 뗀 뒤 `0.4 s + 0.4·v/a`부터는 축이 **느려지고 있어야** 한다 — 최근 0.3 s 속도가 "가속도
    50 mm/s²로 감속했을 때 남을 속도 + v의 20 % + 2 mm/s"를 넘으면 STOP ALL(`stuck … not decelerating (not taken)`).
    100 mm/s에서는 허용 거리(265 mm)가 조각의 남은 길이보다 길 수 있어서, 10-26처럼 정지가 무시되면 거리 검사만으로는
    조각 끝에서 "정상 정지"로 보였다. 속도 검사는 순항 속도 그대로면 약 1.2 s(약 120 mm) 뒤에 잡는다 [유도].
  - 정지 중 갠트리 링크 오류가 나면 "정지 안 먹음"이 아니라 "link error during release"로 따로 STOP ALL한다.
  - **50 mm/s 넘게 움직이는 중의 STOP ALL**(mode 1 → 곧바로 mode 2 = 즉시 정지)은 그 축의 가르친 홈을 **SUSPECT**로
    만든다(`homes_suspect`): 이 리그는 50 mm/s 넘게 돌려 본 적이 없고, 카운터 뒤에 엔코더가 없어 급정지로 스텝을 잃으면
    아무 감시도 모른다. 표시를 확인하고 다시 가르칠 것. 이 동작은 "한 축이 감속 중일 때 다른 축 단축 조그를 받는다"는
    컨트롤러 동작에 기대는데 실기 미확인이다(KNOWN_ISSUES 10-26 ⓑ) — 받지 않으면 새 조그가 "did not move"로 끝난다.
  - 홈이 없는 쪽으로의 조각 사이에는, 앞 조각이 오류 없이 끝나고 축이 **실제로 0.3 s 이상 서 있을 때만** 다음
    조각을 보낸다(링크가 오류를 삼키고 busy를 내린 채 컨트롤러는 아직 가는 경우, 상대 이동이 겹쳐 쌓이지 않게).
  - 컨트롤러에 나가 있는 이동은 홈 쪽이면 그 홈까지, 아니면 최대 `jog_piece_mm`이다. 정지가 안 먹거나 패널이
    강제로 죽으면(SIGKILL·충돌) 축은 **그 끝까지 간다** — 가르친 홈이 그래서 중요하다. 터미널·ssh 끊김(SIGHUP)은
    정상 종료로 처리해 바로 정지를 보낸다. 링크 자체는 상대 이동을 10 mm로 제한하므로, 패널의
    `RigGantryLink.request_jog_relative`가 같은 검사를 하고 제한만 바꿔 보낸다.
  - 창을 닫으면 갠트리 정지부터 보낸다.
- `Disconnect gantry/motor`는 추력 0일 때만: 해제가 GUI 스레드에서 최대 3 s join하는 동안 Esc가
  처리되지 않는다. 모터가 Enable 상태면 "Disable first / Keep holding / Cancel"을 묻고, Disable을 고르면
  링크가 disabled를 확인한 틱에 해제가 끝난다.
- **스러스터 링크 오류**(시리얼 쓰기 타임아웃 등; 링크는 오류를 스스로 지우지 않음): 물리 추력은 **미지**다.
  패널은 이를 "추력 0"으로 보지 않는다 — Enable/Disable·Disconnect가 모두 막히고, 라벨에 캐시 %와
  `LINK ERROR — physical thrust unknown`이 뜬다. 할 수 있는 것은 **STOP ALL**(드라이버만 있으면 동작)과
  Disconnect→Connect thruster(리셋·재아밍으로 확실한 0).
- **모터 텔레메트리가 끊기면**(전원 차단 등; 링크는 enabled 캐시를 그대로 둠) 라벨이 `[STALE]`/"telemetry
  STALE"로 바뀌고, 인터록이 켜진 상태에서 추력이 돌고 있으면 2 s 유예 뒤 **자동 Cut**(`events.csv`
  `interlock_lost`). Set thrust도 신선한 텔레메트리를 요구한다.
- 펌웨어 리밋이 **유효한** 축(예: `gantry_soft_limits.py --write-y`로 Y를 재기록한 경우, 시뮬레이터)은
  링크가 연결 시 그 값을 갖고, 패널은 가르친 홈과 그 값의 **교집합**을 쓴다(라벨에 `(fw)`).
- **STOP 뒤 감시**: 링크의 busy는 STOP 50 ms 뒤 무조건 꺼지므로 그것으론 "축이 실제로 멈췄는지"를 알 수 없다.
  패널은 STOP 뒤 2.5 s(감속 여유: 최고 100 mm/s에서 감속 2.0 s [유도: v/a] + 0.5 s) 이후에도 캐시 위치가 누적 2 mm 넘게 변하면 "still moving after STOP", 위치
  폴링이 1 s 넘게 끊기면 "telemetry STALE after STOP"을 빨간 상태로 띄운다. WiFi가 끊긴 경우엔 둘 다 못
  보는 수가 있다(폴도 같은 링크) — 유선을 권장하는 이유.
- `Reset`은 스러스터 워커가 램프 루프 안이거나 캐시가 0이 아니면 0을 다시 보내고 거부한다("press Reset
  again"). 링크 오류 상태면 best-effort로 0을 보내고 해제하되 "reconnect thruster" 경고를 띄운다.
- 창 닫기·Disconnect thruster 때 보낸 **마지막 0이 확인되지 않으면**(쓰기 오류, 에코 없음) `events.csv`
  `zero_unconfirmed`, `meta.json links_at_stop`, 그리고 터미널에 `*** thruster zero UNCONFIRMED … cut the
  ESC supply ***`가 남는다. 로그의 마지막 행이 0이어도 이 표시가 있으면 믿지 말 것.
- 로그 쓰기가 실패하면(디스크 풀 등) 로그만 조용히 멈추고(`meta.json log_failed`) 상태줄에 빨간 경고가
  남는다 — STOP 경로는 디스크와 무관하게 동작한다. UI FAULT 시 STOP ALL은 한 번만 실행된다.

## 패턴 (2026-10-26)

X는 그대로 두고 Y·Z만 천천히 움직이는 자동 운동. **Start를 누른 순간의 위치가 중심**이고, Stop / STOP ALL까지
한 바퀴씩 반복한다.

| MODE | 크기 칸 | 기본 | 범위 |
|---|---|---|---|
| `Circle  Y-Z` | RADIUS (반지름) | 30 mm | 5–50 mm |
| `Back-and-forth  Y` | RANGE (왕복 전체 폭; 100 = 중심 ±50) | 100 mm | 5–100 mm |

- 속도(경로를 따라) 기본 10 mm/s, 1–100 mm/s. 반지름 30 mm 원 한 바퀴 ≈ 19 s + 꼭짓점 감가속, 100 mm 왕복 한 번
  ≈ 20 s [둘 다 유도: 길이/속도]. 기본값은 `config/disturbance_rig.yaml` `pattern:`.
- **속도를 올려도 원은 거의 빨라지지 않는다**: 구간마다 서고 다시 출발하므로, 길이 L인 구간에서 낼 수 있는 최고 속도는
  √(가속도×L)이다. 가속도 50 mm/s²·구간 8 mm면 약 20 mm/s, 한 바퀴(반지름 30)는 속도를 아무리 올려도 약 19 s
  [유도: 구간당 2√(L/a) × 24]. Y 왕복(100 mm 구간)은 약 70 mm/s까지 오른다 [유도: √(50×100)]. 원을 더 빠르게 하려면
  가속도를 올려야 한다(아래 "알려진 한계").
- **원은 짧은 직선(약 8 mm, 반지름 30이면 24개)의 연결**이다. 컨트롤러는 한 축 직선 이동만 확실히 하므로,
  매 구간 Y·Z에 같은 시간에 끝나도록 축별 속도·가속을 나눠 보낸다 → 꼭짓점마다 아주 짧게 느려진다.
  컨트롤러 자체의 원호 보간(`Arc_2Axis`)은 축 단위(units)로 계산하는데 Y(2.5 mm/unit)와 Z(0.5 mm/unit)가
  5배 달라 mm로는 **타원**이 되고, 이 리그에서 돌려 본 적도 없어서 쓰지 않았다. 현 꼭짓점과 원의 차이는
  최대 0.26 mm [유도: r(1−cos(π/N))].
- **상자 안에서만 움직인다**: 모든 목표점은 "시작 위치 ∪ 꼭짓점"의 축별 범위(원이면 중심 ±r, 왕복이면
  중심 ±폭/2) 안의 절대 좌표다. 링크가 보내기 직전 SDK 락 안에서 다시 확인하고, 측정 위치가 상자 밖 3 mm를
  넘으면 링크가 멈추며, 패널은 5 mm를 넘으면 STOP ALL. 패널이 죽어도 컨트롤러는 진행 중인 한 구간의 끝(상자
  안)에서 선다 — 단 절대 이동이 정말 절대일 때(아래 ⓐ, 미검증).
- Start 전에 상자가 들어가는지 본다: 가르친 홈이나 + 펌웨어 스톱(Z 100 mm 등)이 상자를 자르면 거부하고
  얼마나 옮기라고 알려 준다(예: Z 99.9에서 반지름 30 → "move Z down by 29.9 mm first"). 홈이 없는 쪽은 상자
  말고는 막는 것이 없다 — **Start 전에 갠트리를 빈 공간 한가운데에 둘 것**.
- Stop = 조그 떼기와 같은 래치 없는 mode 1 감속 정지(움직이는 축부터, 세 축) + 같은 정지 감시(시간 안에 서지 않거나
  너무 가면 STOP ALL). X는 패턴 동안 드리프트 감시(2 mm).
- **운용자의 Stop이 아닌 종료는 모두 STOP ALL(래치)**: 링크 오류(링크가 상자 밖 3 mm를 본 경우 포함), 상자 밖 5 mm,
  진행 정지(꼭짓점이 `max(10 s, 3×가장 긴 구간 시간 + 5 s)` 동안 안 바뀜). 프레임이나 컨트롤러가 이상할 수 있으니
  Reset 전엔 아무것도 움직이지 않는다(안전 리뷰 M1).
- 패턴·조그·이동 중에는 `Disconnect gantry`가 거부된다(끊으면 정지를 확인할 감시도 사라짐; 10-26 실패가 바로
  "재연결을 지나서도 계속 감"이었다).
- 기록: `events.csv` `pattern_start`(모드·속도·중심·꼭짓점 수·상자), `pattern_lap`, `pattern_stop`(이유·바퀴
  수), `pattern_stopped`(정지 거리·시간). 위치는 `samples.csv` 그대로.
- **실기 1회(2026-10-27, `data/20261027/1027_163355_disturb/`)**: 원 r 30 @ 10 mm/s 한 바퀴 51 s(Stop까지), 반지름
  29.46–30.17 mm, X 변화 0, Y·Z 동시 이동; Y 왕복 100 mm @ 20 mm/s 2바퀴, 움직이는 중 Stop → 4.4 mm / 0.78 s에 정지.
  아래 ⓐ–ⓓ는 이 런이 근거가 됐고, **ⓔ(움직이는 중 STOP ALL)와 Send는 아직 실측이 없다** — 저속 Send 한 번과 움직이는
  중 Esc 한 번을 기록할 것. 원래의 다섯 가정(안전 리뷰 H2):
  ⓐ `Jog_Single_Axis` mode 2가 `Get_Axis_Current_Pos`와 같은 카운터 프레임의 **절대** 목표로 해석됨(음수 영역 포함 —
  10-26 마지막 런에서 Y −161.8 mm, `data/20261026/1026_171856_disturb/samples.csv`) ⓑ 두 축 단축 조그가 동시에 진행됨 ⓒ `Check_Axis_Is_Stop`이 절대 이동에서도 맞음 ⓓ mode 1이
  절대 이동도 멈춤 ⓔ 움직이는 중의 STOP ALL(mode 1 → mode 2)이 멈춤. 10-26까지의 런에는 `move`(Send)·`pattern`
  이벤트가 한 번도 없고 STOP ALL은 정지 상태에서만 눌렸다(`data/20261026/*_disturb/events.csv`). 시뮬레이터는 가속
  없이 즉시 서고 절대 이동이 정확해서 이 다섯을 증명하지 못한다. 그래서 **아래 순서로 한 단계씩**, 손은 갠트리 전원
  위에 두고, 각 단계의 런 폴더 경로를 적는다. 여유 거리: 절대 목표가 상대로 읽히면 축이 "카운터 목표값"만큼 간다
  (Y −161.8에서 −156.8로 Send → 약 157 mm [유도]). 그래서 1단계는 **카운터 0 근처**에서 하거나, 양 끝에서 |목표
  카운터값| + 30 mm 넘게 떨어진 곳에서 한다. Send에는 범위 감시(시작..목표 ± 5 mm → STOP ALL)가 있지만 그 STOP ALL
  자체(ⓔ)도 미검증이다.
  1. **Y** 한 축 `Send` 5 mm @ 5 mm/s(SPEED를 먼저 낮출 것 — 기본 50) → 그 위치에 도달하는지(ⓐ). 이어서 **Z**도.
  2. `Send` 50 mm @ 10 mm/s 도중 **Esc** → `samples.csv`에서 바로 서는지(ⓔ).
  3. `Back-and-forth Y` RANGE 10 @ 5 mm/s, 구간 도중 `Stop` → `pattern_stopped`의 거리·시간(ⓓ, ⓒ).
  4. `Circle Y-Z` RADIUS 5–10 @ 5 mm/s → Y·Z가 **동시에** 움직이는지(ⓑ), 원 모양인지.
  5. 그 뒤에만 기본값(반지름 30, 10 mm/s).

## 로그

`data/YYYYMMDD/MMDD_HHMMSS_disturb/` (ROV 런과 같은 트리, kind `disturb`로 분리; `rov_gui/runstore.py`)

| 파일 | 내용 |
|---|---|
| `samples.csv` | 10 Hz, 행마다 flush(크래시에도 남음): `host_epoch_s, t_monotonic_s, x_mm, y_mm, z_mm, gantry_busy, motor_deg, torque_nm, motor_enabled, motor_busy, thr_pct_cmd, pwm_us, ramp_target_pct, latched` |
| `events.csv` | 조작·거부·링크 오류: `host_epoch_s, t_monotonic_s, kind, detail` (connect/connect_refused/connect_failed/disconnect/disconnect_cancelled/jog_start/jog_stop/jog_stopped/teach/teach_clear/teach_restore/move/goto/enable/disable/motor_auto_zero/motor_auto_failed/pattern_start/pattern_lap/pattern_stop/pattern_stopped/thrust/thrust0/cut/cut_done/stop_all/stop_all_resend/reset/recover/recover_done/stuck/stall/interlock_lost/zero_unconfirmed/command_rejected/status/link_error/fault/log/close) |
| `thruster.csv` | 링크 자체 버퍼(스텝마다 + 1 Hz 하트비트): `host_epoch_s, throttle_pct, pwm_us` — **명령값** |
| `thruster.csv`의 `pwm_us` / `samples.csv`의 `pwm_us` | 아두이노가 **내보낸다고 되돌려 준** 펄스 폭(계산값) — ESC나 모터의 상태가 아님 |
| `meta.json` | 앵커 `t0_epoch_s`/`t0_monotonic_s`, hostname, 설정(`settings`: `jog_stop_mode`, `limits_file`, `companion_run` = `--companion`으로 적은 동반 ROV 런…), `gantry`(`taught_homes_mm` 가르친 홈과 시각, 연결 시 링크 리밋, 소프트 리밋, 펌웨어 스톱), 모터 Gr/limits, `links_at_stop`(마감 시 링크별 error/last_command_error), whisker_vision 브랜치·sha, 레포 sha(둘 다 창 생성 시 한 번 조회), preflight 보고, `jet_calibration`(whisker `aim_calibration.json` 복사, 참고용), `rig_notes`, `log_failed`(쓰기 실패 시) |

- **추력은 측정값이 아니다.** 펌웨어에 상태 조회가 없어 명령값만 기록된다. 아두이노가 USB 글리치로
  리셋되면 ESC는 1500 µs(0)로 재아밍하지만 패널과 로그는 옛 %를 계속 보인다.
- ROV 런과 맞추기: ROV MPC CSV의 `t`는 CSV를 연 순간부터 센 상대시간이다. 같은 PC(같은 부팅)면
  **행 단위로** 맞출 수 있다: ROV 행의 monotonic 시각 = meta `run.t0_monotonic_s` + `t` ↔ 여기
  `t_monotonic_s`. 다른 PC면 `run.t0_epoch_s` + `t` ↔ `host_epoch_s`(wall clock)이고, NTP 상태를 먼저
  확인한다. `run.t0_*`는 2026-10-27부터 있다. 그 전 런은 `*_ekf_inputs.jsonl` 첫 줄의 `t0_csv`(monotonic,
  EKF 로그가 있는 런만)를 쓰고, 그것도 없으면 `run.started`(초 단위)뿐이다.
  ROV를 rov_gui로 조종하는 동안 패턴을 돌리는 법: 패턴 Start → rov_gui 클릭. 패턴은 Stop까지 계속 돌고,
  포커스가 바뀌어 끊기는 건 hold-jog뿐이다. Esc는 창마다 따로라서 rov_gui의 Esc는 리그를 세우지 않는다.

## 문제가 생기면

- **손을 뗐는데 / STOP ALL 뒤에도 축이 계속 감**("still moving … CUT THE GANTRY POWER"): 정지 명령이
  컨트롤러에서 안 먹은 것이다. **갠트리 전원을 끈다** — 이동은 컨트롤러 안에서 그 끝(홈 또는 조각 끝)까지
  간다. 그 뒤 카운터가 0부터 다시 시작하므로 홈은 Clear 후 다시 가르친다. `events.csv`의 `stuck` 행과
  `samples.csv`를 그대로 보관할 것(정지 방식 판단의 근거).
- **비상정지 자체가 실패**(`gantry stop: …` 빨간 상태): Reset → 그래도 빨강이면 Disconnect/Connect
  gantry 후에만 모션. 갠트리 정지 시퀀스가 진행 중이면 Reset이 거부된다(다시 누른다).
- **연결 −1 실패**(비정상 종료 뒤): 잠시 기다리거나 컨트롤러 전원 재투입(그러면 4번 Y 한계 재확인).
- **`can0` DOWN**: 패널이 `sudo -n ip link set can0 …`로 올린다(NOPASSWD 규칙 있음). BUS-OFF는
  어댑터에 restart-ms가 없어 수동 `down`/`up`. 모터가 "silent on CAN"이면 DC 전원·CAN-H/L·터미네이션.
  무동작 진단은 `python3 ~/Desktop/whisker_vision/can_power_check.py`(상태 요청만). 주의:
  `motor_smoke_test.py`는 축을 **움직이고**, `motor_bringup_scan.py`는 can0를 재설정한다.
- **스러스터 "no 'ready' banner"**: 포트/펌웨어 불일치. by-id 경로(`ls -l /dev/serial/by-id/`),
  보드가 바뀌었으면 `--thruster-port`.
- **창이 안 뜸**: 셸에 `QT_QPA_PLATFORM=offscreen`이 남아 있는지(테스트 흔적).
- **UI FAULT** 상태: 슬롯에서 예외가 났고 패널이 스스로 STOP ALL을 눌렀다. 터미널의 traceback을
  보고, Reset 후 계속하거나 재시작.
- **Recover gantry/motor**가 켜지면: 이동이 데드라인 안에 끝나지 않았거나 정착하지 않은 것.
  2026-09-05 whisker 세션에서 축이 43 mm 못 미치고 링크가 영구 busy로 남은 사례가 있다.

## 알려진 한계 (KNOWN_ISSUES.md에도 등재)

- 호스트 사망 시 스러스터 failsafe 없음 → 킬스위치. 펌웨어 워치독(수신 없으면 1500 µs)과
  상태 조회 명령은 whisker_vision 소유자에게 제안할 항목.
- 보드 리셋 시 무음 0 %(위).
- **50 mm/s 초과는 실기 0회**(2026-10-27). 처음 올릴 때는 축마다 50 → 75 → 100 mm/s로, 테이프 표시를 전후로 확인하고,
  한 번은 움직이는 중 Esc — 각 단계 런 폴더 경로를 적는다(안전 리뷰 10-27 H2). 그 전까지 Z와 Send는 50 mm/s 상한.
- **가속도는 50 mm/s² 고정**(whisker `motion_config.py:73`). 100 mm/s에서 손을 떼면 약 10 cm 더 가고 [유도: v²/2a],
  원 패턴은 속도 설정과 무관하게 한 바퀴 약 19 s(반지름 30)보다 빨라지지 않는다(위 "패턴"). 가속도를 올리면 둘 다
  줄지만 이 갠트리가 어디까지 견디는지(탈조 → 카운터 어긋남)는 실측이 없다. `./c3 gantry`도 같은 50 mm/s²(5 cm/s²)를
  기본으로 쓴다(`src/gantry_panel.py:3147`).
- 모터 STOP 홀드는 **캐시된 각도**(0.1–0.3 s 지연)로 명령된다 — 빠른 회전 중 STOP은 속도×지연만큼
  되감긴다 [유도]. 그래서 UI 속도 상한을 90 °/s로 둔다.
- 모터는 창을 닫아도 **홀드 상태로 남는다**(whisker 계약: 스러스터 마운트가 흔들리지 않게).
  limp로 두려면 닫기 전에 Disable.
- whisker 링크에서 발견한 결함 5건은 패널이 우회한다(위 STOP 재전송, 인터록 자동 0, Zero 전 Disable,
  워치독, 명령값 로그). 수정은 whisker_vision 쪽 일이다.

## 검증 기록

- 2026-10-27 (10차) preflight 장치 묶음 판정(요청 4, `disturbance_preflight.relax_devices`): 셋 중 하나라도 되면 열림, 셋 다
  실패·장치 외 FAIL은 막힘, `--sim`의 skip은 장치로 치지 않음. preflight 11 통과(장치 묶음 4건 포함). 패널 코드는 그대로
  (`PF.gate`가 적용). 실기: 이 PC에서 세 장치 모두 OK로 확인만 함(다른 장치를 뺀 상태의 실기 확인은 없음).
- 2026-10-27 (9차) safety-code-reviewer: 다른 축 즉시 조그(요청 3)는 50 mm/s 이하에서 OK, **50 초과는 FAIL 취급** → 반영:
  뗀 축의 감속 확인(H1: 100 mm/s에선 허용 거리 265 mm가 조각보다 길어 무시된 정지를 놓침), Z·Send 50 mm/s 상한 + 50 초과
  STOP ALL 시 홈 SUSPECT(H2·M2), 두 번째 떼기가 첫 축을 다시 멈춘 기록(M1), 정지 중 링크 오류 별도 처리(M3), 정지 스레드
  중 버튼 비활성·상태줄에 정지 중 축 병기·조그 중 홈 저장 안 함(LOW). 테스트 시뮬레이터에 감속(50 mm/s²) 모델 추가.
  시뮬레이터 core 23, panel 74, preflight 11 통과. 50 mm/s 초과 실기 0회.
- 2026-10-27 (8차) 사용자 요청: 갠트리·패턴 속도 상한 50/30 → 100 mm/s(`RigGantryLink`이 whisker의 50 mm/s 상한 대신
  `RIG_MAX_SPEED_MM_S`; Send도 `request_move` 재정의), 뗀 뒤 약 1초 동안 **다른 축** 버튼이 막히던 것 해제(정지 감시를
  축별로; 뗀 축만 설 때까지 대기). STOP ALL 뒤 드리프트 기준 1.5 → 2.5 s. 버그 하나를 테스트로 잡음: 먼저 선 축의
  드리프트 기준이 움직이기 전 위치로 남아 오경보 → 그 축 기준을 선 자리로. 시뮬레이터 core 23, panel 71. 실기 0런.
- 2026-10-26 (7차) safety-code-reviewer 재검증: **FAIL** — 6차의 L5 수정(상태 미지 → Disable 버튼)이 스러스터 없이도 눌려
  H1을 손으로 다시 열었다(N1, HIGH). 반영: 손 Disable·Disconnect의 "Disable first"·자동 시퀀스 모두 "제트가 꺼진 것을 앎"
  (스러스터 ready+0, 또는 마지막 연결이 0 에코로 닫힘)일 때만(N1); Send 이동 중 "시작..목표 ± 5 mm" 이탈 → STOP ALL(N2,
  HIGH); 패턴 구간이 "정지"를 보고했는데 꼭짓점에서 1 mm 넘게 떨어져 있으면 오류(→ 래치)(N3); 래치 중엔 Disconnect
  gantry 허용·"busy 5 s after STOP" 문구 수정(N4); 패턴 이탈·정지(stall)·Send 이탈·N1·정지 순서 테스트 추가(N5);
  Disable 대기를 링크 확인 시점부터, Zero 뒤 축이 돌면 즉시 Disable, STOP ALL은 시퀀스 중에도 홀드(limp 강제 안 함),
  STOP 실패 메시지가 뒤 메시지에 덮이지 않게, STOP ALL도 패턴 축부터 정지, bring-up 여유 거리·측정 표기(LOW).
  시뮬레이터 core 23, panel 68, preflight 7.
- 2026-10-26 (6차) safety-code-reviewer: H1 수정 전 **FAIL** → 반영 뒤 WARN(실기 bring-up 전제). 반영: 모터 자동 Disable은
  스러스터 연결(아두이노 리셋 = 0) 뒤에만(H1), Disable 2회 + 각 0.5 s 대기 + Enable 뒤 0.5 s 각도 확인·실패 시 Disable(M2),
  패턴 비정상 종료는 STOP ALL 래치(M1), 구간마다 위치 폴링 최소 1회(M3), 패턴·조그·이동 중 Disconnect gantry 거부(M4),
  0을 못 잡은 연결은 `NOT zeroed this connection` 표시(M5), STOP 뒤 드리프트 기준 1.0 → 1.5 s(L1), 1 mm 미만 바퀴 거부 +
  꼭짓점마다 취소 확인(L2), 구간을 실제 위치에서 계산(L3), 움직이는 축부터 정지(L4), 상태 미지 시 버튼 = Disable(L5).
  H2(위 "패턴"의 단계별 bring-up)는 실기에서만 풀린다. 시뮬레이터 core 23, panel 63, preflight 7.
- 2026-10-26 (5차) 사용자 요청: 모터 연결 = 그 자리 0° 홀드(Disable (sync)·Zero here 버튼 제거), 스러스터 화면 정리
  (RAMP/EVERY·Cut 제거, `Thrust 0 (ramp)` → `Turn off`, 설명 줄 제거), 갠트리 기본 50 mm/s, 패턴(Y-Z 원, Y 왕복).
  시뮬레이터: core 23, panel 59, preflight 7 통과 — 원은 반지름 안(현 오차 이내), X 불변, 왕복 목표는 정확히
  중심 ±폭/2, Z 95에서 반지름 30은 거부, STOP ALL·링크 오류 시 종료. **실기 0런**(패턴, 자동 0°).
- 2026-10-26 (4차) 실기: 떼기 정지 18/18(위 "떼기 실측", `data/20261026/1026_140909_disturb/events.csv`). 사용자 지적 "스러스터 배터리를 안 꽂았는데
  연결됐다고 뜸" → 원인: 연결 대상은 USB 전원 아두이노이고 펄스 에코는 계산값, ESC 미감지 → 헤더 `ARDUINO`, 라벨
  `% cmd → Arduino µs`, 패널 안 설명 줄, 상태줄 "Arduino ok (ESC power not sensed)". 같은 날 14:00 런의 UI FAULT
  17건(`data/20261026/1026_140012_disturb/events.csv`)은 편집 도중의 코드로 띄운 것(`_gantry_at_rest` 이름 변경
  중간) — 슬롯 가드가 STOP ALL로 래치함. 안전 재검증 WARN → 반영: Clear가 프레임 검사 전에 기준 위치를 덮어쓰던 것
  (HIGH), Recover 정지가 안 먹으면 무한 대기(→ 5 s 뒤 STOP ALL), STOP ALL에서 한 장치의 예외가 나머지 정지를
  건너뛰던 것, 조그 중에도 다른 축 드리프트 감시, 닫을 때 경고 조건. 테스트 core 21, panel 52, preflight 7.
- 2026-10-26 (3차) safety-code-reviewer **FAIL**(CRITICAL 2, HIGH 2, MEDIUM 4, LOW 4) → 반영: + 펌웨어 스톱은
  조각만 줄임(전에는 홈 없는 +X 조그가 1650 mm까지 한 번에 감), 재연결·Restore는 마지막 위치와 대조(전원
  재투입 뒤 옛 홈이 조용히 다시 걸리던 것), 모든 다음 명령·Reset은 세 축 실제 정지 요구, STOP 시퀀스에서
  stop_run 제거, 다른 축을 눌러도 드리프트 감시 유지, 누름당 2000 mm 상한, 최저 속도 1 mm/s, 정지가 안 먹은
  축의 홈 SUSPECT 표시, 스레드 시작 실패 처리, 닫을 때 정지 미확인 경고. 시뮬레이터 "10-26 모델"은 mode 1이
  듣는다고 **가정한** 것이라 실기 증거가 아니다.
- 2026-10-26 (2차) 실기 첫 사용에서 떼기 정지·STOP ALL이 조그를 멈추지 못함(위 "왜 정지 방식을 바꿨나",
  `data/20261026/1026_131512_disturb/`) → 떼기 = 래치 없는 mode 1 정지, STOP 시퀀스 = mode 1 먼저, 스위치
  홈 대신 teach-in 홈(= 소프트 리밋)으로 교체. 시뮬레이터에 "mode 2·stop_run은 무시(관찰), mode 1은 듣는(가정)"
  컨트롤러 모델을 넣어 확인: whisker 원래 시퀀스로는 그 모델에서 정지 뒤에도 16 mm 더 감, 새 시퀀스는 멈춤 —
  모델 안의 결과일 뿐이다. **실기 mode 1 정지는 미확인.**
- 2026-10-26 조그를 "누르는 동안 한 번의 연속 이동"으로 변경(홈 전 10 mm 끊김 제거, 조각 300 mm): 시뮬레이터
  core 16, panel 42 통과.
- 2026-10-26 UI 개편(ROV 스테이션 테마, 꾹 누르는 조그, Home −/+ 버튼, 홈 전 무제한 조그) + 안전 리뷰
  (safety-code-reviewer, FAIL → 지적 7건 반영) 뒤 시뮬레이터 통과. 시뮬레이터는 홈 방향을 무시하고 정지가
  즉시라서 이 위험들의 증거가 되지 못한다(리뷰어 지적) — 실기 0런. 첫 실기에서 한 축·저속·손은 Esc 위에 두고:
  떼기 정지, `Home +` 뒤 `homed X … at ~0`과 −X가 + 스위치에서 멀어지는지 확인할 것. (이 Home −/+ 설계는
  같은 날 2차에서 teach-in 홈으로 교체됨 — 위.)
- 2026-10-22 시뮬레이터: `./c3 test` 통과(core 15, preflight 7, panel 27); 안전 감사 1차(3렌즈 → 14건 반박 검증,
  13건 확정 반영) + 2차(수정 검증 + 새 지적 10건 확정 반영) 뒤의 숫자. 실기 **0런**.
  실기 첫 스모크 순서는 계획 파일(`~/.claude/plans/gantry-adaptive-newt.md` "실기 스모크")대로:
  킬스위치 → preflight → 갠트리(한계·홈·조그·Send) → 모터(Disable→Enable→Go θ ±15°) →
  스러스터(∓20 %, Thrust 0, Cut) → 30 s 기록 → Esc 중단 → 창 닫기 추력 0 확인.
  실기 수치는 그 런의 `data/…_disturb/` 경로와 함께만 적는다.
