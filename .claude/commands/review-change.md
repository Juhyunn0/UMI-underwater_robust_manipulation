---
description: P2 Parallel Reviewer — fan out independent reviewers over a change/plan/design, then synthesize
argument-hint: <what to review: a diff, file(s), a plan, or a described design>
---

You are running the **P2 — Parallel Reviewer** pattern. Target to review:

$ARGUMENTS

(If empty, review the current working-tree diff: run `git status` / `git diff` to
scope it, or ask what to review.)

## Steps
1. **Scope** the artifact: code change (which subsystem?), research plan,
   hardware/design choice, or model/control change? Read what's needed to understand
   it (the diff, the file(s), the doc).
2. **Pick 2–4 independent reviewer lenses** from the roster (repo-root `CLAUDE.md`
   routing matrix). Typical maps:
   - control/model/sim change → `simulation-advisor` + `control-theory-advisor`
     (+ `underwater-robotics-advisor` if real-water physics, + `safety-code-reviewer`
     if it drives hardware)
   - hardware/datapath change → `hardware-advisor` + `safety-code-reviewer`
     (+ `underwater-robotics-advisor`)
   - perception/SLAM change → `slam-perception-advisor` (+ `hardware-advisor`)
   - research plan → `research-plan-reviewer` + the relevant domain advisor
3. **Dispatch them IN PARALLEL** — one message, multiple Task calls; each reviewer
   gets the SAME artifact and reviews **independently** (do not have them coordinate).
   Ask each for: what's correct · what's questionable · what's wrong · the minimal
   fix · which `verify_*`/`test_*` would catch a regression.
4. **Dedupe & synthesize** the independent findings into one table:

   | Finding | Raised by | Severity | 합의/이견 | Minimal fix |

   Then a short verdict — top risks + the smallest change that addresses them. Where
   reviewers disagree, surface BOTH and say which you find more convincing and why.
5. **Record**: append a dated block to `.claude/journal/reviews.md`
   (`## YYYY-MM-DD <artifact>` · reviewers · 합의 · 이견 · 리스크 · 최소수정). If a
   durable decision emerged, also update auto-memory (`MEMORY.md` + a memory file).

For an exhaustive audit you may escalate to the `Workflow` tool (parallel reviewers +
adversarial cross-check + synthesis), or for a git diff specifically the
`/code-review` skill — but the default is the parallel sub-agent fan-out above.
Reviewers never talk to each other; you are the hub and the synthesizer.
