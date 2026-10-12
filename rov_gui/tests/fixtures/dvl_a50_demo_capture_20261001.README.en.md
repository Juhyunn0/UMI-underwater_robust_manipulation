# dvl_a50_demo_capture_20261001.jsonl — provenance

> **English translation.** Source of truth: [`dvl_a50_demo_capture_20261001.README.md`](dvl_a50_demo_capture_20261001.README.md) (Korean). Snapshot of 2026-10-12, source blob `780ac92e66`; later edits to the original are not reflected here.

**What it is.** These are the bytes of the stream from Water Linked's publicly operated DVL
**demo unit** (`dvl.demo.waterlinked.com`, TCP 16171), received at 2026-10-01 23:26 PDT for 12 seconds
and saved as-is. 80,325 bytes, 179 lines, every line ends with `\r\n`. `velocity` 60 lines +
`position_local` 119 lines, `format: json_v3.2`.

**What it is not.** It is not the DVL fitted to this project's vehicle (as of 2026-10-01 the DVL is
not mounted). The demo unit is a simulated stream run by the vendor, so the velocity and altitude
values themselves have no meaning as measurements (altitude 20.9 m, covariances all 0, `time` always
100). It is used only **for format verification**: line delimiting, field names and types, and the
fact that the two report kinds arrive mixed on one socket.

**Observed in the same session (not in this file; what follows is the only record — to be checked again on hardware):**

```
clients before: {'tcp_water_linked': 0, 'tcp_pd6': 0, 'tcp_pd4': 0, 'serial': -1}
socket A got 4096 bytes; socket B got 4096 bytes (simultaneously open)
clients with 2 sockets open: {'tcp_water_linked': 2, 'tcp_pd6': 0, 'tcp_pd4': 0, 'serial': -1}
get_config with newline -> responses on A: [... "mounting_rotation_offset":44,"range_mode":"1<=4" ... "type":"response"]
responses seen on B (should be none if responses are per-client): []
velocity reports: 60 over 12 s; tov deltas ms min/max 199.006 201.768
tot - tov (us) min/max 200 2456
```

* The TCP server accepted **two clients at once**, and the `get_config` response arrived **only on the requesting socket**.
* The demo unit's `tot − tov` was 0.2–2.5 ms. The example report in the real device's documentation shows 189 ms
  (docs.waterlinked.com/dvl/dvl-json-protocol). **Latency has to be measured on hardware.**

Used in: `rov_gui/tests/test_dvl.py` (`test_the_vendor_demo_capture_parses_line_for_line`,
`test_the_client_reads_torn_lines_asks_config_once_and_reconnects`).
