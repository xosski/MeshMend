# MeshMend Studio Miniature Pipeline

The old local generation path used text/image-to-3D reconstruction and then tried
to repair the result. That is not enough for store/studio quality: image-to-3D
often creates smooth blobs, card-like sheets, floating panels, or high-poly
surfaces with no sculptural detail.

The studio pipeline is a local/offline deterministic hybrid system. Store-mode
generation is staged asset construction, not one-shot text-to-mesh:

1. **Concept profile**: `StudioMiniatureSpec` parses the prompt or loads a
   JSON spec.
2. **Part candidates**: generate 3-6 validated candidates for head/helmet,
   torso/body, arms, legs, weapons, backpack/accessories, and base.
3. **Part bundles**: each candidate writes mesh, preview SVG, metadata, anchors,
   sockets, symmetry, scale, and cleanup report.
4. **Selection**: user/automation selects one candidate per category.
5. **Final assembly**: selected parts are assembled into a miniature.
6. **Detail pass**: physical STL geometry for panel lines, rivets, seams, trim,
   helmet lenses, backpack vents, weapon barrels, fingers, and base texture.
7. **Printability pass**: duplicate/degenerate cleanup, normals, hole filling,
   artifact-shell removal, scale normalization, detail-preserving subdivision,
   and light Taubin smoothing.
8. **Miniature quality gate**: rejects low face/vertex count, non-watertight geometry,
   boundary/non-manifold edges, floating shells, outliers, large sheet/card
   artifacts, too-flat shapes, primitive-only parts, low-detail parts, unusable
   sockets, wrong scale, or missing body/limb/weapon metadata.
9. **Export**: STL/OBJ/GLB/PLY plus printability, stage, selection, and studio
   quality sidecars.

## Generate locally

```powershell
cd D:\MeshMend\src\meshmend_ai
python -m meshmend --cli `
  --studio-config .\examples\studio_sci_fi_heavy_infantry.json `
  --studio-candidates-dir .\outputs\studio_candidates `
  --output .\outputs\studio_heavy_infantry.stl
```

Or from a prompt:

```powershell
python -m meshmend --cli `
  --studio-generate "original sci-fi heavy infantry with rifle backpack vents helmet lenses rivets panel lines" `
  --scale 32mm `
  --target-faces 90000 `
  --output .\outputs\studio_prompt_heavy_infantry.glb
```

The export writes:

- `*.stl`, `*.obj`, `*.glb`, or `*.ply`
- `*.printability.json`
- `*.studio_quality.json`
- `*.studio_stages.json`
- `*.studio_selection.json`

Candidate bundles are written under:

```text
studio_candidates/
  head_helmet/head_helmet_1/head_helmet_1.stl
  head_helmet/head_helmet_1/preview.svg
  head_helmet/head_helmet_1/metadata.json
  head_helmet/head_helmet_1/cleanup_report.json
  ...
```

## Why this replaces the old store-quality path

The Hunyuan/image reconstruction path can still be used as a draft/proposal or a
future optional AI hook, but it cannot guarantee premium resin miniatures by
itself. The studio path makes all store-critical features explicit geometry and
rejects bad exports before they leave MeshMend.

## Quality gate failures

Failures are intentional. MeshMend should fail loudly rather than export a blob,
sheet, or malformed blockout as “store quality.” Common failure keys:

- `faces_below_studio_minimum`
- `vertices_below_studio_minimum`
- `mesh_not_watertight`
- `boundary_edges`
- `non_manifold_edges`
- `floating_shells`
- `sheet_artifacts`
- `too_flat_depth_ratio`
- `missing_required_components`
