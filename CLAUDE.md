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
