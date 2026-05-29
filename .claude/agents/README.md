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

| Agent | When to invoke |
|---|---|
| `research-plan-reviewer` | Before committing to a 1-4 week research direction |
| `control-theory-advisor` | Questions on MPC, RL, robust control, disturbance observers |
| `slam-perception-advisor` | Questions on SLAM, AprilTag, fisheye calibration, GTSAM |
| `umi-manipulation-advisor` | Questions on UMI, imitation learning, manipulation policies |
| `safety-code-reviewer` | After writing or modifying gantry/ROV motion code |
| `experiment-diagnostic-analyst` | After an experiment produces unexpected data |

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
