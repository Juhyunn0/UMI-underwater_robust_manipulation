# Consult journal

Append-only, git-tracked trail of substantive sub-agent work, per the **Recording**
rule in the repo-root [`CLAUDE.md`](../../CLAUDE.md). Travels with the repo so the
context is available in the VS Code extension, Claude cowork, and to teammates.

One line per entry:

```
- YYYY-MM-DD [agent] Q: <short question> → <key conclusion / decision> [memory: <slug>]
```

Files:
- `consults.md` — **P1** specialist-advisor consults (live).
- `reviews.md`  — **P2** parallel-review syntheses (when `/review-change` lands).
- `research.md` — **P3** research + verify outputs (when `/research-verify` lands).

Durable facts/decisions ALSO go to auto-memory (`MEMORY.md` + `memory/*.md`), not
just here: **this journal is the chronological trail; memory is the recall index**
(auto-loaded each session). Keep entries terse; skip trivial Q&A.
