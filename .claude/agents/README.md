# Sub-agents for UMI-Underwater Robust Manipulation Project

This folder contains specialized sub-agents that the main Claude (Claude Code) can dispatch for domain-specific verification, advice, and analysis tasks.

## Philosophy

**The user (JJ) is the project director.** Sub-agents are advisors and verifiers, NOT decision-makers. They:

- Confirm whether a research plan is sound (`research-plan-reviewer`)
- Provide domain-specific expertise on demand (`control-theory-advisor`, `slam-perception-advisor`, `umi-manipulation-advisor`)
- Check safety of motion code (`safety-code-reviewer`)
- Analyze experiment recordings for failure patterns (`experiment-diagnostic-analyst`)

They never:

- Make research direction decisions
- Autonomously coordinate with other agents
- Modify code without the user reviewing
- Make assumptions about user's priorities

## Available agents

Roles: **advisor** (answers/explains) · **reviewer** (independent critique, a P2
building block) · **researcher** (web-grounded, a P3 building block) · **analyst**
(post-hoc diagnosis). Routing & recording policy lives in repo-root [`CLAUDE.md`](../../CLAUDE.md).

| Agent | Role | When to invoke |
|---|---|---|
| `control-theory-advisor` | advisor · researcher | Control & estimation **theory** — MPC, RL, robust control, disturbance observers, Fossen dynamics, allocation theory, sim-to-real |
| `simulation-advisor` | advisor | **This** project's BlueROV2 MuJoCo sim (`bluerov2_mujoco_marinegym/`) — variants, hydro, thrusters, controllers, verify_*, how to run/extend |
| `underwater-robotics-advisor` | advisor | **Real** underwater operation — water hydrodynamics, current/wave, water effects on sensors, air-vs-water |
| `hardware-advisor` | advisor · researcher | **Physical** hardware — tether/comms, cameras & IMU as devices + data paths, Jetson/Pi compute, enclosures/power/buoyancy |
| `slam-perception-advisor` | advisor · researcher | Perception/**SLAM** — AprilTag, GTSAM, iSAM2, fisheye/stereo calibration, refraction, tag-map |
| `umi-manipulation-advisor` | advisor · researcher | **UMI**, imitation learning (Diffusion Policy/ACT/BC), manipulation policy, data collection |
| `research-plan-reviewer` | reviewer | Critique a drafted 1–4 week research plan before committing |
| `safety-code-reviewer` | reviewer | Audit gantry/camera/thruster-driving code for races, interlocks, e-stop |
| `experiment-diagnostic-analyst` | analyst | Diagnose an experiment's unexpected recording/CSV |

## How to invoke

In the main Claude Code conversation, dispatch like:

```
Please consult the control-theory-advisor sub-agent about whether a
disturbance observer makes sense before we wire MPC, given our current
gantry hardware setup.
```

The main Claude will run the sub-agent with full project context and return
a concise summary. The sub-agent reads project files but does not write
unless explicitly asked.

## Editing rules

- Each agent's domain knowledge is intentionally narrow — keep it that way
- If a topic spans multiple agents, route through the main Claude (don't let
  agents talk to each other)
- Update an agent's prompt when its domain knowledge grows (e.g., new key
  paper, new project component)
- Never let an agent make unreversible decisions (writes to config, hardware
  commands) without user confirmation

## Created

2026-05-29 — initial setup
2026-06-22 — added `hardware-advisor` (fixed missing frontmatter → now dispatchable) & `underwater-robotics-advisor` to the table; role tags + routing/recording policy moved to repo-root `CLAUDE.md`; `.claude/journal/` consult trail added
