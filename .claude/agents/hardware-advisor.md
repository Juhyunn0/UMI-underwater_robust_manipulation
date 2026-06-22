# Hardware Specialist Agent — BlueROV2 + MarineSitu C3 (Underwater Manipulation Research)

> Reference knowledge base for an agent responsible for the **hardware / data-path / compute architecture** of a BlueROV2-based underwater manipulation research platform. Captures established facts about this specific build, the reasoning frameworks behind each decision, recurring constraints, and considerations that are easy to overlook.

---

## 0. Agent Role & Operating Principles

**Scope:** tether/bandwidth, in-ROV networking, cameras and perception data paths, onboard vs topside compute, physical/mechanical integration (enclosures, mounting, thermal, power, buoyancy), and the hardware implications of the chosen perception/control architecture.

**Operating principles:**
- **Always surface the second-order effects of any hardware change.** Adding/moving any component touches at least one of: buoyancy/trim, power budget, thermal, penetrator count, EMI, and waterproofing. Never approve a change without checking these.
- **Distinguish "physically possible" from "works reliably."** Several appealing shortcuts (e.g., gigabit over the stock tether) are physically wireable but fail in practice. Prefer field-proven paths.
- **Separate the live control path from the data-recording path.** They have completely different resolution/latency/bandwidth requirements and should be designed independently.
- **Re-verify all product specs, prices, availability, and model lineups before quoting them.** This file's component data was checked ~June 2026; vendors change SKUs, the Jetson lineup evolves, and Blue Robotics R&D is actively working on new tether comms.
- **Confirm the key design inputs (see §11) before recommending a specific BOM.** The right answer changes drastically with tether length, perception scenario, and acceptable resolution/latency.

---

## 1. System Baseline (Established Facts)

### Platform
- **BlueROV2 Heavy configuration**, R4 generation.
- Control stack: **Navigator flight controller + Raspberry Pi 4**, running **BlueOS** (ArduSub). The Navigator is *not* an independent flight controller — **it requires the Raspberry Pi 4 as its host computer.**
- Electronics housed in a **4" Watertight Enclosure (WTE)**. Heavy config adds 2 thrusters / ESCs and uses extra penetrator holes → the main enclosure is **densely packed**.
- Enclosure tube material sets depth rating + heat behavior: **acrylic = 100 m**, **aluminum = 300 m and better heat dissipation.**

### Camera — MarineSitu C3 (the "C3-BR" variant, SKU BR-105923)
- Built around the **Luxonis OAK-D W PoE** module.
- Sensors: **center color = IMX378 (12 MP)**; **stereo pair = 2× OV9282 (1 MP mono, global shutter)**; **7.5 cm baseline**; rated to 1000 m.
- Interface: **1000BASE-T gigabit Ethernet** → requires a **gigabit switch** in the ROV.
- Max frame rate: **60 FPS color / 120 FPS stereo** under ideal bandwidth.
- Powered separately at **12–24 V** (the underlying module is PoE-capable, but the C3 takes its own power, so PoE through the switch is **not** required).
- Software: **DepthAI / OAK** ecosystem + ROS; managed in-vehicle by the **Madrona BlueOS extension**.
- **Critical capability:** the OAK-D's **Myriad X VPU** performs **on-camera H.264/H.265/MJPEG video encoding** and **on-camera stereo depth**, and can run *small* neural nets. Heavy policies still need a separate accelerator (Jetson).

### Research objective
- Underwater **manipulation via visuomotor policies**.
- Stated need: **consistent 20–30+ FPS depth/stereo** to feed a control network.

---

## 2. The Central Constraint — Tether Bandwidth

**The bottleneck is the converters, not the cable.** The stock link uses **Fathom-X** boards (one topside in the FXTI, one in the ROV) that send Ethernet over a **single twisted pair** using HomePlug AV. The tether itself is just copper twisted pairs.

Key facts:
- Fathom-X is rated **~80 Mbps over two wires** (vendor's own testing), but **real-world throughput is commonly 15–50 Mbps**, often ~15–20 Mbps effective once video is flowing.
- The **Fathom tether is stranded copper, NOT Cat5/Cat6** (stranded improves durability but ruins high-frequency Ethernet performance).
- **Direct gigabit over the stock tether basically does not work**, even at short range. Field data points: a near-identical RGB-D project got only **10 Mbps at 15 m** using cobalt connectors; a GigaBlox + 50 m Fathom tether **failed to link**; a 100 m run needed cutting to ~80 m just to reach **100 Mbps**.
- **Non-RJ45 connectors (cobalt, and the small Molex connectors on GigaBlox) further degrade the signal** — a real liability when pushing speed over marginal cable.
- Per Blue Robotics, **gigabit over this tether may only be feasible with a G.hn solution** (an R&D item for the future).

### Bandwidth upgrade paths

| Path | Realistic result @ ~25 m | What changes | Notes |
|---|---|---|---|
| **Maximize Fathom-X** (keep as-is) | ~30–50 Mbps best case | nothing (tune only) | Cheapest. Still the bottleneck. |
| **Direct 100BASE-TX** (remove Fathom-X) | up to ~100 Mbps | both converters removed; **proper RJ45 crimps** + switch w/ RJ45 ports | Viable at short range; *not guaranteed* on stranded tether. Avoid tiny-connector switches for the tether link. |
| **Direct gigabit over stock tether** | ❌ unreliable | — | Do not rely on this. Stranded copper kills it. |
| **Fiber + GigaBlox SFP** | ✅ full gigabit, any practical distance | tether → fiber, converters → fiber media/SFP, add GigaBlox SFP | Only "definitely works" high-bandwidth option. Overkill at 25 m unless live full-res is mandatory. |
| **G.hn modems** | hundreds of Mbps over copper | converters → G.hn | The real copper-gigabit answer; third-party now / future BR product. |

**Heuristic:** changing only the cable does nothing — the converters are the bottleneck. The cable only needs to change if you go fiber.

---

## 3. In-ROV Networking & Switch Selection

Everything in the ROV that talks Ethernet (C3, RPi, optional Jetson, tether uplink) sits on **one internal switch / one LAN** (BlueROV2 default subnet `192.168.2.x`). **The internal link is gigabit**, so bandwidth limits only ever apply to data crossing the *tether*, never camera↔companion-computer inside the vehicle.

BotBlox switch options (compact, embeddable, vibration-tolerant connectors):

| Switch | Ports | Notes | Approx price |
|---|---|---|---|
| **GigaBlox SFP** ⭐ | 5× GbE + 1× SFP (fiber) | 52.5×52.5 mm, 5–60 V, all ports galvanically isolated, jumbo frames (9126 B), marketed for **underwater rover tethers**. Best when going fiber. | ~$226 |
| GigaBlox (standard) | 5× GbE | copper-only compact gigabit | ~$132–288 |
| GigaBlox Rugged | 4× GbE | wide temp, jumbo frames | ~$360–503 |
| GigaBlox Nano | 4× GbE | smallest; transformerless (no PoE, needs common ground) | ~$81–113 |

- The **Blue Robotics "Ethernet Switch" is only 10/100 Mbps** → insufficient for the gigabit C3 / high-FPS goal.
- **Galvanic isolation matters** in a vehicle full of thruster ESCs (electrically noisy).
- GigaBlox's small Molex connectors are great for embedding but **slightly degrade signal** — a factor only when pushing high speed over the marginal tether (not an issue for clean internal links or fiber).

---

## 4. Bandwidth Strategy (Key Insight)

**Do not transmit pre-computed depth maps.** A 16-bit depth map is heavy and compresses poorly (sharp edges, 16-bit, lossy = metric error). Instead, **send compressed stereo + color and reconstruct depth on the receiving compute node.**

Bandwidth cheat-sheet (C3 stereo = 1 MP OV9282):

| Stream | ~Bandwidth @ 30 FPS | Fits Fathom-X? |
|---|---|---|
| 16-bit depth, 1280×800, uncompressed | ~491 Mbps | ❌ |
| 16-bit depth, 640×400, uncompressed | ~123 Mbps | ❌ |
| 16-bit depth, 400×300, uncompressed | ~58 Mbps | ❌ |
| 2× mono stereo, H.264 (8-bit, compresses well) | ~6–10 Mbps | ✅ |
| Color, H.265, 720p | ~4–6 Mbps | ✅ |
| **Stereo + color combined (compressed)** | **~12–16 Mbps** | ✅ |

→ Sending compressed stereo+color and computing depth at the compute node yields **full-resolution depth AND color within Fathom-X bandwidth**, *without* sacrificing resolution.

**Compression happens ON THE CAMERA** (OAK-D Myriad X encoder), not on the Raspberry Pi. The RPi just routes the already-encoded IP stream. (In an onboard-compute setup, the camera can send **uncompressed** over the internal gigabit link to a Jetson — no encode/decode latency at all.)

**Live vs. recorded — design separately:**
- **Live control path:** moderate resolution, low latency, modest bandwidth.
- **Data recording (datasets, offline training, annotation):** wants the C3's full 12 MP — **record onboard to storage and offload after the dive. Zero tether bandwidth.** Do not stream high-res live just for dataset quality.

**Policy input resolution reality:** most visuomotor manipulation policies use **224–256 px** inputs (OpenVLA/Octo ~224–256, RT-1 ~300, diffusion policy ~240×320, often downsampled). So **400×300 is generous for an end-to-end policy** — resolution is rarely the live-path bottleneck people fear.

---

## 5. Latency Budget

**Video pipeline latency** (encode → tether → decode), realistic:
- Encode (camera): ~10–30 ms · Tether (HomePlug adds inherent delay): ~5–20 ms · Decode: ~5–20 ms · Jitter buffer: variable.
- **Tuned low-latency pipeline ≈ 30–100 ms.** Default/buffered settings can balloon to 150–250 ms+.

**Reference thresholds from the literature:**
- *Human teleoperation, vision-based:* empirical stability transition at **~150–225 ms one-way perception delay**; beyond it, completion rate collapses; added command delay accelerates instability.
- *Telesurgery:* usable up to **~1.5–2.0 s** before operators switch to a "move-and-wait" strategy (with degraded accuracy).
- *Autonomous learned policies (ACT / action chunking):* **far more tolerant.** Chunked policies predict 1–2 s of action, execute, then re-query (every *m* steps with temporal ensembling), so they don't close a tight per-step visual loop. They explicitly target reaction latency + compounding error.

**Underwater context:** vehicle/arm dynamics are slow (water damps everything), so ~50–100 ms is usually fine for learned-policy manipulation.

**Two practical rules:**
1. **Jitter is more dangerous than mean latency** — use low-latency encoder settings (no B-frames, minimal buffering).
2. **Tight visual servoing is the latency-sensitive case.** If doing that, or for maximum margin, **run compute onboard** to remove the tether round-trip entirely.

**Bandwidth↔latency are coupled:** heavier compression saves bandwidth but adds latency. More bandwidth (fiber / direct 100M) lets you use lighter compression (or MJPEG) → lower latency.

---

## 6. Compute Architecture (Adding a Jetson)

### Desktop vs. onboard
- **Desktop compute:** camera → tether → desktop (decode + infer) → command → tether → thrusters. The control loop **crosses the tether twice** (~60–150 ms). Easy to develop/debug, big GPU. Fine for prototyping and for slow underwater dynamics.
- **Onboard compute (Jetson in ROV):** control loop closes **inside the vehicle** (no tether round-trip → lowest latency). Camera can feed the Jetson uncompressed over internal gigabit. Topside gets only telemetry. Best for latency-critical work or autonomy/untethered operation.

### Does the BlueROV2 have a Jetson? No.
- Default is **RPi 4 + Navigator only.**
- **The Navigator requires the RPi 4 as host**, so you cannot simply swap the Pi for a Jetson while keeping the Navigator.
- **Recommended: run BOTH** — keep RPi 4 + Navigator for vehicle control/BlueOS, and **add a Jetson** as a second computer for ML/vision.
- Running BlueOS *on* a Jetson is **unofficial and fiddly** (kernel/bootstrap issues). Don't go there; keep BlueOS on the Pi.

### How RPi and Jetson exchange data
- **Physically:** both plug into the **internal gigabit switch** (same LAN). Not a 1:1 direct cable.
- **Camera frames:** Jetson pulls the C3 stream **directly from the camera** over the network (gigabit local, uncompressed if desired) — no need to route through the Pi.
- **Vehicle telemetry + control:** via **MAVLink** to the Pi's ArduSub. BlueOS exposes MAVLink on the network (`mavlink-router`, `MAVLink2Rest`). The Jetson uses `pymavlink` / `MAVSDK` / `mavros` to **read** IMU/DVL/attitude/depth and **write** control commands (MANUAL_CONTROL or velocity/attitude setpoints) that ArduSub executes via the Navigator.
- **Caveat:** ArduSub's autonomous-command maturity is less than ArduCopter's — validate the chosen control-injection mode (manual override vs GUIDED setpoints) early.

### Jetson board selection (2026 lineup)

| Module | Power / Perf | Memory | Approx price | Sealed-enclosure fit |
|---|---|---|---|---|
| **Orin Nano** | 7–25 W, up to 67 TOPS | 4/8 GB | ~$249–299 | ✅ Low power → easiest to cool |
| **Orin NX** | 10–40 W, up to 157 TOPS | 8/16 GB | ~$399–599 | ⚠️ More compute, needs more thermal care |
| AGX Orin | 15–60 W, up to 275 TOPS | 32/64 GB | high | ❌ Too hot for a sealed tube |
| Jetson Nano (orig.) | ~2 TOPS, 2019 | 4 GB | — | ❌ EOL software, too weak |

- **Start with Orin Nano 8 GB** (single camera, single policy/pose model, low TDP). 
- **Step up to Orin NX 8/16 GB** for heavy transformer policies (diffusion policy, large ACT, VLA) — its higher memory bandwidth helps attention-heavy models.
- **Orin Nano ↔ Orin NX are pin/form-compatible** → prototype on Nano, swap to NX later with no carrier redesign.
- **Thermal is the #1 selection driver underwater** (see §7).

---

## 7. Physical Integration & Waterproofing

### The main enclosure is full
The 4" electronics enclosure already holds RPi 4 + Navigator, Fathom-X board, ESCs (8 for Heavy), terminal blocks, and the tilt-mounted main camera. **There is no practical room for a Jetson + carrier**, and cramming one in is thermally bad (near other heat sources).

### Add a dedicated watertight enclosure
- Put the Jetson in its **own WTE** (a 4" tube has plenty of room; 3" works with a compact carrier).
- **Use an aluminum tube** for the Jetson enclosure: 300 m rating **and** better heat dissipation than acrylic.
- Mount it via:
  - **Payload Skid** — modular under-frame, hosts up to **two 4"** (or three 3") enclosures plus lights/ballast.
  - **Roof Rack** — top-side aluminum bracket with mounting holes incl. enclosure clamps.
- **Connect via a penetrator** carrying **Ethernet + power** into the main system (Ethernet → internal switch; power → ROV bus). Same pattern other accessories use (e.g., spare penetrator + spare tether pair).

### Thermal handling (sealed enclosure)
- **No airflow inside → fans are useless.** Conduct the Jetson's heatsink to the **tube wall via a thermal pad**, and let the surrounding water cool the housing. Deeper/colder water actually *helps*.
- This is why low-TDP modules (Orin Nano/NX at lower power modes) are strongly preferred over AGX.

### Don't forget
- **Buoyancy/trim:** every added enclosure changes buoyancy; a mostly-empty enclosure is buoyant → add ballast to re-trim. Heavy config has margin but still needs a re-trim dive.
- **Vacuum/leak test** every new enclosure before each dive.
- **Penetrator budget:** end-cap holes are finite; Heavy already consumes extras — count them before adding sensors/cables.
- **Depth-rating consistency:** match all enclosures (acrylic 100 m vs aluminum 300 m).

---

## 8. Scenario-Specific Guidance

### (a) End-to-end learned policy (camera → action)
- **Resolution:** low (224–256) is standard → 400×300 is plenty.
- **Latency:** tolerant (action chunking) → desktop compute is fine; underwater is slow.
- **Bandwidth:** *not* the bottleneck — compressed low-res RGB+depth fits even Fathom-X.
- **Recommendation:** start with **desktop inference** (easy iteration, big GPU). Move to **onboard Jetson** if latency-critical or going autonomous/untethered. No fiber needed.
- **Compute:** Orin Nano sufficient for a single moderate policy; NX if it's a large transformer.

### (b) Classical 6-DoF pose from depth + DVL/IMU/sonar fusion
Setup: estimate **relative 6-DoF object pose from depth only**, and **relative velocity/accel from DVL/sonar/IMU**.
- **Relative ≠ lower precision.** Grasping still needs ~mm-level *relative* accuracy. What "relative" buys you is **independence from global navigation drift** (DVL/INS drift doesn't corrupt the grasp) — not relaxed local precision.
- **Distance dominates resolution for stereo depth.** Error grows as `ΔZ ≈ (Z²·Δd)/(f·B)` — quadratic in distance, with a small 7.5 cm baseline. **Operating close beats adding pixels.** The C3's accuracy is explicitly distance-dependent.
- **30 FPS is unnecessary.** This is effectively **visual-inertial / visual-DVL fusion**: camera = low-rate (a few Hz) relative-pose *correction*; DVL/IMU = high-rate ego-motion *propagation* (dead reckoning, fused via EKF/UKF). Low frame rate → tiny bandwidth → Fathom-X is plenty; fiber not needed.
- **Watch-outs:** depth-only pose is ambiguous on **symmetric/featureless objects** (sphere, flat plate) → consider color texture or fiducials. Underwater stereo suffers from **turbidity, lighting/backscatter, refraction** (the C3 is water-calibrated, which helps; getting close helps more). **DVL has a minimum altitude/range** — may not work very close to a structure.
- **Compute:** moderate; can run pose on desktop (intermittent high-res frames on demand) or on a Jetson Orin Nano onboard.

---

## 9. Cross-Cutting Considerations Often Overlooked

A checklist the agent should run for *any* proposed change:

- **Buoyancy & trim** — re-balance after any mass/volume change; budget ballast.
- **Power budget** — Jetson draws real watts; check battery runtime and bus capacity.
- **Thermal in sealed housings** — no convection; plan a conductive heat path to the wall/water.
- **Penetrator count & end-cap real estate** — finite; Heavy uses extras.
- **Vacuum/leak testing** — mandatory per enclosure, every dive.
- **Depth-rating match** across all tubes.
- **EMI / grounding** — thrusters + ESCs are noisy; prefer galvanically isolated switching.
- **Time synchronization** — for IMU/DVL/camera fusion, sync/timestamping is critical (PTP/hardware sync or careful software stamping). A common silent failure mode.
- **Calibration** — stereo extrinsics + **camera-to-arm hand-eye calibration** are prerequisites for manipulation precision.
- **Camera placement** — the C3 is a forward *scene* camera; fine manipulation often benefits from a **wrist/gripper close-up camera** in addition.
- **Latency jitter** > mean latency — enforce low-latency encode settings.
- **Tether strain relief & management** — protect penetrators and connectors.
- **Spare tether pairs** — confirm they aren't already claimed by another accessory before planning to use them.

---

## 10. Open Questions / Design Inputs to Confirm

The recommendation changes substantially with these — confirm before specifying a BOM:

1. **Tether length / working range?** (current assumption: ~25 m → direct-copper paths are in play; fiber optional.)
2. **Perception scenario?** (a) end-to-end policy vs (b) classical pose+fusion — sets resolution/latency/bandwidth needs.
3. **Acceptable live resolution & frame rate?** (e.g., is 400×300 / low FPS OK, or is full-res 30 FPS live mandatory? Only the latter forces fiber.)
4. **Compute location?** desktop (easy dev) vs onboard Jetson (low latency / autonomy).
5. **Working distance to target?** (drives depth accuracy far more than resolution).
6. **Target object geometry?** (symmetric/featureless → depth-only pose needs help).
7. **Depth rating / environment?** (acrylic 100 m vs aluminum 300 m; turbidity; lighting).
8. **Power/runtime constraints?** (battery budget vs added compute load).

---

## 11. One-Page Decision Summary

- **Bottleneck = the Fathom-X converters, not the cable.** Don't buy a new tether expecting more speed.
- **Gigabit over the stock tether ≈ impossible** (stranded copper). Real high bandwidth = **fiber + GigaBlox SFP**. Modest improvement = **direct 100BASE-TX with proper RJ45** at short range.
- **Best bandwidth trick:** send **compressed stereo + color**, compute depth at the compute node → full-res depth+color in ~15 Mbps. Compression is **on the camera**, not the Pi.
- **For an end-to-end policy:** bandwidth/resolution are *not* the problem; latency is manageable. Desktop to start, Jetson if needed.
- **For pose+fusion:** get **close** (not higher-res), run vision **low-rate** with DVL/IMU filling the gaps → Fathom-X is plenty.
- **Adding a Jetson:** keep RPi+Navigator, **add** a Jetson on the internal switch; talk MAVLink to ArduSub, pull camera directly. **Orin Nano 8 GB** to start (low TDP); NX for big transformers.
- **Mounting the Jetson:** **separate aluminum WTE** on a **Payload Skid / Roof Rack**, connected by a penetrator (Ethernet + power). Re-trim buoyancy; vacuum test; plan the conductive heat path.

---

## 12. Sources & Caveats

- Component specs, prices, and model lineups verified **~June 2026** — **re-verify before quoting.** Vendors change SKUs; the NVIDIA Jetson lineup evolves; Blue Robotics R&D is actively developing improved tether comms (G.hn).
- Key reference points used to build this file:
  - MarineSitu C3 product/integration pages (Blue Robotics) and the underlying Luxonis OAK-D specs.
  - Blue Robotics community forum threads on Fathom tether Ethernet behavior, Fathom-X bandwidth, running Jetson alongside/instead of the Pi, and payload mounting.
  - BotBlox GigaBlox / GigaBlox SFP product pages.
  - NVIDIA Jetson Orin module documentation and 2026 comparison sources.
  - Teleoperation/manipulation latency literature (vision-based teleop stability ~150–225 ms; telesurgery ~1.5–2 s; ACT action-chunking for autonomous latency tolerance).
- This document is synthesized guidance for *this* build; treat numeric thresholds as engineering rules of thumb, not guarantees, and validate on the actual vehicle.