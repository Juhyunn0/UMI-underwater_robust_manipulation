---
description: P3 Researcher + Verifier — research a question, then adversarially verify each claim before reporting
argument-hint: <question or claim to research and fact-check>
---

You are running the **P3 — Researcher + Verifier** pattern. Question / claim:

$ARGUMENTS

(If empty, ask what to research.)

## Steps
1. **Research** — gather candidate answers. For broad web research prefer the
   `deep-research` skill; for a project-domain question dispatch the matching
   web-grounded advisor (`hardware-advisor`, `control-theory-advisor`,
   `underwater-robotics-advisor`, `slam-perception-advisor`, `umi-manipulation-advisor`
   — see repo-root `CLAUDE.md`). Collect each answer WITH its sources.
2. **Extract discrete claims** — break the findings into individually-checkable
   factual claims (numbers, specs, citations, provenance), each with its purported
   source.
3. **Adversarially verify** — dispatch the `verifier` agent on the claims (one call
   can take several). It tries to REFUTE each and returns verified|uncertain|rejected
   with the primary source. For a critical claim, dispatch 2–3 verifiers
   independently and require a majority.
4. **Report**:
   - **Verified** claims (+ primary source each),
   - **Uncertain / Rejected** claims (+ why, + the corrected value if known),
   - a one-line **provenance** trail for any key number.
   State your confidence; do NOT present rejected/uncertain claims as fact.
5. **Record**: append a dated block to `.claude/journal/research.md`
   (`## YYYY-MM-DD <question>` · verified(+sources) · rejected(+why) · provenance).
   Durable facts/provenance also go to auto-memory.

For an exhaustive find→verify sweep (loop-until-dry, multi-verifier voting) you may
escalate to the `Workflow` tool; the default is the dispatch above.
