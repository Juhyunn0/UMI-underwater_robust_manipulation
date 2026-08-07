# CLAUDE.md — UMI Underwater Robust Control

Repo-wide working agreement for Claude Code. Project *facts* live in auto-memory
(`MEMORY.md`) and the per-area docs; this file is the **orchestration policy** —
how to route work to the sub-agents in `.claude/agents/`, and how to record it so
context survives across sessions and surfaces (CLI / VS Code / Claude cowork).

Explanatory prose in **Korean**; code, commands, and UI labels in English.

## Sub-agent orchestration

Three usage patterns share one roster of specialists (`.claude/agents/`):

- **P1 — Specialist Advisor** *(live)* — a domain **question** → dispatch the
  single best-matching advisor, then **state which one you used**.
- **P2 — Parallel Reviewer** *(live: `/review-change`)* — **review / 검토** of an
  artifact → fan out several reviewers independently → synthesize. For a git diff
  the `/code-review` skill (+`ultra`) also applies.
- **P3 — Researcher + Verifier** *(live: `/research-verify`)* — **research / 근거
  검증** → gather, then adversarially verify each claim via the `verifier` agent. For
  broad web topics the `deep-research` skill also applies.

### Intent → pattern
A **question** → P1. **"검토 / 리뷰 / review this change|plan"** → P2.
**"조사 / 찾아줘 / 출처·근거 검증 / research"** → P3. If genuinely ambiguous, ask.

### Routing matrix (P1) — pick the **most specific** match, and name it in the reply
| Question is about… | Dispatch |
|---|---|
| control/estimation **theory** — MPC/NMPC, RL, robust, DOB/EAOB, Fossen math, allocation theory, tuning, sim2real | `control-theory-advisor` |
| **this** project's MuJoCo sim — `bluerov2_mujoco_marinegym/`, hydro.py/thrusters.py/dobmpc/, model variants, verify_* | `simulation-advisor` |
| **real** underwater physics/operation — water hydro, current/wave, water effects on sensors, air-vs-water | `underwater-robotics-advisor` |
| **physical** hardware — tether/comms, cameras & IMU as devices + data path, Jetson/Pi compute, enclosures/power/buoyancy | `hardware-advisor` |
| perception/**SLAM** — AprilTag/GTSAM/iSAM2, calibration, refraction, tag-map; `tagslam_core.py` | `slam-perception-advisor` |
| **UMI** / imitation learning — Diffusion Policy/ACT/BC, data collection | `umi-manipulation-advisor` |
| critique a drafted **research plan** (1–4 wk) | `research-plan-reviewer` |
| audit **hardware-driving code** (gantry/camera/thruster) for safety | `safety-code-reviewer` |
| diagnose an **experiment's** unexpected recording/CSV | `experiment-diagnostic-analyst` |

Sub-agents never talk to each other — **you are the hub**. Advisors are read-only
unless explicitly asked to write. Trivial questions you can answer directly; dispatch
when the matching specialist would clearly do better, and say which one you used.

### Recording — so later sessions keep the context
After a **substantive** consult/review/research, append ONE dated line to the
matching journal in [`.claude/journal/`](.claude/journal/) (`consults.md` /
`reviews.md` / `research.md`):

```
- YYYY-MM-DD [agent] Q: <short question> → <key conclusion / decision> [memory: <slug>]
```

If a **durable fact or decision** emerged, ALSO write/update a memory file and add
its `MEMORY.md` index line (the journal is the chronological trail; memory is the
recall index). Skip trivial Q&A to avoid noise.

## Measurements — cite the artifact or don't call it a measurement

**Rule: whenever a number goes into a doc, journal, memory, or code comment as a
measured value, the artifact path goes with it.** If you cannot name a path, it is not
a measurement — tag it `[예측]`, `[유도]`, or `[스펙]` instead.

```
나쁨:  depth bias +28 mm
좋음:  depth bias +28 mm  (c3_camera/depth_accuracy/rungs.csv, row tape=2000)
좋음:  depth bias ±dZ/2 = ±70 mm  [예측: c3_depth_accuracy.py:88-99, 미실측]
```

Why: a 2026-08-04 audit of 1028 numeric claims across 26 docs found **284 that read as
measurements but had no artifact behind them** — including a fabricated
"+28 mm bias" (a synthetic example written while reviewing the tool) and an
unsourced "IN-AIR calibration" assumption that was later cited as device fact and
turned out to be backwards. Ledger: [`docs/MEASUREMENT_AUDIT.md`](docs/MEASUREMENT_AUDIT.md).

Three failure modes that audit found, worth guarding against specifically:
- **Docstring numbers become measurements.** A "this is roughly what you get" value
  written beside a new tool gets cited later as a result.
- **Precision grows on citation.** `PSNR 45.6` becomes `45.60`; extra significant
  figures are a provenance smell.
- **Hedges fall off on citation.** `약 3000 ms` becomes `3000 ms`, and our own
  assumptions printed into output files (`calibration.json`'s `note`) come back
  looking like the device said them.
