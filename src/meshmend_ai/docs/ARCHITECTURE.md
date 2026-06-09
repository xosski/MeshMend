# MeshMend Local-First Architecture

MeshMend is being shaped as a fully local desktop miniature creation, repair, and export program. The product goal is: Blender-style editing, Meshy-style guided creation, and Lychee-style print sanity checks without paid APIs or hosted generation services.

## Design constraints

- No OpenAI, Meshy, Tripo hosted API, paid API, or required cloud backend.
- All generation adapters must run from local code and local/open-weight models.
- The app must stay useful when no ML model is installed: import, repair, scale, base, report, and export still work.
- Do not label output as “8K studio quality” unless the active adapter/export path actually supports high-resolution geometry and/or 8K texture/displacement assets.

## Package layout

```text
meshmend/
├─ app/          PySide6 desktop shell and viewport
├─ core/         mesh IO, transforms, base generation, reports
├─ ai/           swappable local generation adapters
├─ sculpt/       vertex brush/sculpt operations
├─ repair/       watertight/normals/hole/manifold repair
├─ export/       slicer-ready exports and report sidecars
├─ library/      kitbash asset metadata and future assets
└─ compliance/   originality prompt filters and forbidden symbol checks
```

## MVP data flow

```diagram
╭────────────╮      ╭────────────╮      ╭────────────╮
│ STL/OBJ/etc│─────▶│ MeshMend UI │─────▶│ repair/core │
╰────────────╯      ╰─────┬──────╯      ╰─────┬──────╯
                          │                   │
                          ▼                   ▼
                  ╭──────────────╮     ╭──────────────╮
                  │ local adapter │     │ printability │
                  │ no paid APIs  │     │ report       │
                  ╰──────┬───────╯     ╰──────┬───────╯
                         ▼                    ▼
                  ╭────────────────────────────────╮
                  │ slicer-ready STL/OBJ/GLB/PLY   │
                  ╰────────────────────────────────╯
```

## Local generation adapter contract

`meshmend.ai.LocalGenerationAdapter` exposes one method: `generate(GenerationRequest) -> trimesh.Trimesh`. The default MVP adapter wraps the repository's existing local `3dsculpter/model_service/native_generation.py` generator. Adapters can also wrap local Stable Fast 3D, Shap-E, OpenLRM, TripoSR, Wonder3D, Hunyuan, Blender geometry nodes, or future local diffusion/reconstruction stacks. The procedural placeholder adapter remains available for smoke tests and fallback.

## V1 model strategy

1. Keep the existing MeshMend local native generator adapter as the default MVP path.
2. Keep the placeholder adapter for offline testing and UI reliability.
3. Add optional local model adapters behind dependency checks.
4. Cache model settings/checkpoints in SQLite project metadata.
5. Route every generated mesh through compliance sanitization, repair, scale, base, and printability validation.
6. Add optional 2D concept generation through local ComfyUI/Stable Diffusion only.

## Printability checks

Current MVP reports manifold/watertight status, boundary edges/holes, non-manifold edges, thin-shell heuristics, floating shells, dimensions, and polygon count. V1 should add voxel/ray thickness analysis, overhang analysis, island support risk, resin drain holes, and slicer profile presets.
