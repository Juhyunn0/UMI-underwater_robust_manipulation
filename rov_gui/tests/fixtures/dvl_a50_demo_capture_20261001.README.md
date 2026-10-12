# dvl_a50_demo_capture_20261001.jsonl — 출처

**무엇인가.** Water Linked가 공개 운영하는 DVL **데모 장치**(`dvl.demo.waterlinked.com`, TCP 16171)의
스트림을 2026-10-01 23:26 PDT에 12초 동안 받아 그대로 저장한 바이트다. 80,325 bytes, 179줄, 모든 줄이
`\r\n`으로 끝난다. `velocity` 60줄 + `position_local` 119줄, `format: json_v3.2`.

**무엇이 아닌가.** 이 프로젝트의 기체에 달린 DVL이 아니다(2026-10-01 현재 DVL은 장착되지 않았다).
데모 장치는 벤더가 돌리는 시뮬레이션 스트림이라 속도·고도 값 자체에는 측정 의미가 없다
(고도 20.9 m, 공분산 전부 0, `time`이 항상 100). **포맷 검증용**으로만 쓴다:
줄 구분, 필드 이름과 타입, 두 리포트 종류가 한 소켓에 섞여 온다는 것.

**같은 세션에서 관찰한 것 (이 파일에는 없고, 아래가 유일한 기록이다 — 실기에서 다시 확인할 것):**

```
clients before: {'tcp_water_linked': 0, 'tcp_pd6': 0, 'tcp_pd4': 0, 'serial': -1}
socket A got 4096 bytes; socket B got 4096 bytes (simultaneously open)
clients with 2 sockets open: {'tcp_water_linked': 2, 'tcp_pd6': 0, 'tcp_pd4': 0, 'serial': -1}
get_config with newline -> responses on A: [... "mounting_rotation_offset":44,"range_mode":"1<=4" ... "type":"response"]
responses seen on B (should be none if responses are per-client): []
velocity reports: 60 over 12 s; tov deltas ms min/max 199.006 201.768
tot - tov (us) min/max 200 2456
```

* TCP 서버는 **두 클라이언트를 동시에** 받았고, `get_config` 응답은 **요청한 소켓에만** 왔다.
* 데모 장치의 `tot − tov`는 0.2–2.5 ms였다. 실장치 문서의 예시 리포트는 189 ms다
  (docs.waterlinked.com/dvl/dvl-json-protocol). **지연은 실기에서 재야 한다.**

쓰는 곳: `rov_gui/tests/test_dvl.py` (`test_the_vendor_demo_capture_parses_line_for_line`,
`test_the_client_reads_torn_lines_asks_config_once_and_reconnects`).
