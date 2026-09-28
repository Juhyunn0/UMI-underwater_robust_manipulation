# onshape_export_20260909

onshape-to-robot 1.8.2 export of the Onshape document **"BlueROV2"**, assembly tab **"mujoco"**
(workspace link — a snapshot of the live workspace at 2026-09-09 11:42 −0700, not a pinned version).
Contents: BlueROV2 Heavy hull (182 parts), MarineSitu C3 on the "upper mount" bracket (part_1 / part_3),
Newton gripper + mount, and one part in the hull frame (part_1__2). `no_dynamics: true` — masses are epsilon
placeholders; use this ONLY for poses and meshes, never for inertia.

- `robot.xml`, `scene.xml` — tracked. `assets/` (408 STL/part files, 266 MB) — gitignored; re-run
  `onshape-to-robot .` here (needs `.env` with the Onshape API keys) to regenerate.
- `payload_frames_20260909.json` — part poses converted to the sim base_link frame (FLU, origin = the vehicle
  COM of the 2026-07-19 registration, `tools/process_c3_mesh.py` R0/C_ASM). **[유도]** — the hull pose is
  byte-identical to the July export, so the same registration applies; nothing here was tape-measured.
- `render/` — MuJoCo/Onshape renders: `export_20260909_views.png` (this export, six views in the vehicle FLU frame), the 4-row comparison `compare_onshape_exports_sim.png`, and the individual frames.
- Loading `scene.xml` in MuJoCo fails on 3 hull STLs with >200k faces (`__6`, `__147`, `__175`); decimate
  copies (see the render), do not overwrite the export.
- The July export (`../onshape_export/`) is kept as provenance for the sim's current C3 placement.
