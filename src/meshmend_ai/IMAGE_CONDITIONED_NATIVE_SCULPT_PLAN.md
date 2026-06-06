# Image-conditioned native sculpt backend plan

This document maps the next backend step for MeshMend's native miniature generator: an AI-assisted image-conditioned rig/detail planner that still keeps polygon generation, fusion, STL export, and print validation inside MeshMend-controlled code.

## Goal

Convert a concept image plus optional prompt into a structured native miniature plan:

```text
concept image + prompt
  -> vision evidence
  -> MiniatureSpec
  -> RigPlan
  -> DetailPlan
  -> native geometry parts
  -> voxel-fused watertight STL
  -> concept/print validators
```

AI integration is allowed to identify subject, pose, parts, style, and detail zones. AI must not be the primary polygon generator or final STL source.

## Current problem

The current `meshmend_sculpt` backend can make valid native STLs, but concept matching is still weak because image handling is mostly heuristic. A detailed image can be reduced to broad cues like `wide_silhouette` or `hard_surface`, causing wrong rigs such as generic armored humanoids when the concept is actually a rider on a beast.

The fix is a planning layer with a strict schema and validation, not more global noise/detail displacement.

## Core contracts

### VisionEvidence

Evidence extracted from the image and prompt.

```json
{
  "provider": "local_heuristic|command|hosted_vision",
  "confidence": 0.0,
  "subject": "hooded rider on armored reptilian mount",
  "archetype": "mounted_beast",
  "pose": "low prowling mount, upright rider, raised blade",
  "style_tags": ["fantasy", "grimdark", "armored", "spiked"],
  "detected_parts": [
    {"kind": "mount", "label": "four-legged reptilian beast", "confidence": 0.9},
    {"kind": "rider", "label": "hooded armored rider", "confidence": 0.85},
    {"kind": "weapon", "label": "raised curved sword", "confidence": 0.8}
  ],
  "silhouette": {
    "bbox_aspect": 1.15,
    "top_width": 0.28,
    "mid_width": 0.93,
    "bottom_width": 0.71,
    "foreground_ratio": 0.32
  },
  "warnings": []
}
```

### MiniatureSpec

High-level subject plan. This extends the current `MiniatureSpec` in `native_sculpt_backend.py`.

Required fields:

- `archetype`: `armored_humanoid`, `orc_warrior`, `mounted_beast`, `armored_mech`, later more
- `scale_mm`
- `pose`
- `armor_style`
- `weapon`
- `base_style`
- `detail_language`
- `proportions`
- `concept_cues`

### RigPlan

Native geometry part plan. This is the bridge from AI understanding to backend-controlled mesh generation.

```json
{
  "archetype": "mounted_beast",
  "pose_tags": ["low_mount", "upright_rider", "raised_blade"],
  "parts": [
    {
      "id": "beast_body",
      "kind": "beast_body",
      "primitive": "ellipsoid",
      "anchor": "base_center",
      "side": null,
      "center": [0, 0, 8.2],
      "scale": [9.4, 3.2, 3.0],
      "rotation_deg": [0, 0, 0],
      "required": true,
      "confidence": 0.95
    }
  ],
  "unsupported_parts": []
}
```

Initial supported rig vocabulary:

- `head`
- `torso`
- `pelvis`
- `left_arm`, `right_arm`
- `left_leg`, `right_leg`
- `weapon`
- `shield`
- `cape`
- `wing_left`, `wing_right`
- `tail`
- `horn`
- `shoulder_pad`
- `armor_plate`
- `beast_body`
- `beast_head`
- `beast_leg`
- `saddle`
- `rider`
- `base`
- `skull_or_rock_base_detail`

### DetailPlan

Zone-based detail instructions. This replaces global sinusoidal/random-looking relief.

```json
{
  "style": "grimdark fantasy armored mount",
  "materials": ["plate armor", "reptile scales", "cloth cloak", "bone/skulls"],
  "min_feature_mm": 0.18,
  "zones": [
    {"zone": "beast_back", "motifs": ["spine_spikes", "scale_rows"], "density": 0.8},
    {"zone": "rider_torso", "motifs": ["layered_armor", "raised_trim"], "density": 0.7},
    {"zone": "base", "motifs": ["rocks", "skulls"], "density": 0.5}
  ],
  "motifs": ["spikes", "plates", "scales", "torn_cloak", "skulls"]
}
```

## Planner providers

### 1. `local_heuristic`

Default provider. Uses PIL/numpy analysis already in the repo:

- foreground mask
- aspect ratios
- body band widths
- edge density
- asymmetry
- simple prompt keywords

This is fully offline and deterministic.

### 2. `command`

Preferred first AI integration seam. MeshMend launches an external local command that returns validated JSON only.

Environment:

```powershell
$env:MESHMEND_SCULPT_PLANNER_PROVIDER="command"
$env:MESHMEND_SCULPT_PLANNER_COMMAND="python local_vision_planner.py --image {image_path} --prompt {prompt_path} --schema {schema_path} --out {plan_path}"
```

The command may use local tools such as Ollama/LLaVA/Qwen-VL or a private wrapper. MeshMend does not accept polygons from it, only plan JSON.

### 3. `hosted_vision`

Optional later provider. It may call a hosted vision model, but it must return the same schema. It is still not a mesh backend.

## Validation gates

### Concept validation

Reject or warn when:

- no foreground subject is detected
- image appears to be a flat card/reference sheet
- subject is heavily cropped
- image is too noisy/low-contrast for planning
- no image was received for image-to-3D

### Plan validation

Reject or downgrade when:

- planner confidence is too low
- archetype is unsupported
- required parts are unsupported
- prompt and image conflict strongly
- requested detail is below printable feature size
- plan asks for too many tiny/noisy accessories

### Mesh validation

Block high-detail experimental output when:

- mesh is not watertight
- components > 1
- mesh is too flat/card-like
- no meaningful base contact exists
- face count is too low for requested mode
- face count/file size is too high for UI/download reliability

### Concept mismatch validation

Render or project the generated mesh and compare against concept evidence:

- bbox aspect
- top/mid/bottom width ratios
- mounted/quadruped vs humanoid proportions
- major side accessory asymmetry
- vertical weapon/wing/cape cues

If image says `mounted_creature_silhouette` but the mesh projection is narrow humanoid, fail with:

```text
concept_geometry_mismatch_archetype_or_silhouette
```

## Native generation architecture

The generator should move from:

```python
spec = parse_miniature_spec(request, image_path)
parts, layers = build_rigged_miniature(spec)
```

to:

```python
plan = plan_miniature(request, image_path, output_dir)
validate_plan_or_raise(plan)
parts, layers = build_planned_miniature(plan)
mesh = fuse_and_remesh(parts)
mesh = apply_detail_plan(mesh, plan.details)
validate_mesh_or_raise(mesh, plan)
```

## Implementation phases

### Phase 1: planner module and reports

Add `native_sculpt_planner.py` with:

- dataclasses for `VisionEvidence`, `RigPart`, `RigPlan`, `DetailPlan`, `MiniaturePlan`
- `plan_miniature(request, image_path, output_dir)`
- local heuristic provider moved out of `native_sculpt_backend.py`
- `miniature_plan.json` artifact written for every run

Definition of done:

- every generated model includes `native_sculpt_report.json` and `miniature_plan.json`
- reports clearly show whether image was received and what subject/archetype/parts were planned

### Phase 2: command AI planner integration

Add provider dispatch:

```text
MESHMEND_SCULPT_PLANNER_PROVIDER=local_heuristic|command
MESHMEND_SCULPT_PLANNER_COMMAND=...
```

Definition of done:

- if command fails or emits invalid JSON, backend falls back only when standard quality allows it
- high-detail experimental mode fails honestly on invalid/low-confidence plan

### Phase 3: RigPlan-driven geometry

Refactor hardcoded archetype builders to consume `RigPlan` parts.

Definition of done:

- mounted beast concepts produce beast body/head/legs/tail/rider/weapon as planned parts
- humanoid concepts do not accidentally become mounts
- major unsupported parts are reported, not ignored

### Phase 4: zone-based detail library

Implement native detail stamps:

- armor plates
- rivets
- straps
- raised trim
- scale rows
- spine spikes
- cloth folds
- torn cape edges
- skull/rock base details
- weapon bevels

Definition of done:

- details are attached to named zones, not global noise
- `min_feature_mm` clamps tiny detail for printability

### Phase 5: concept mismatch validator

Add simple mesh projection validation against image evidence.

Definition of done:

- obvious mismatches fail before user sees another unrelated STL
- failures include actionable blockers in `result.json`

## Capability labels

Until all validators pass reliably, report:

```json
"capability_tier": "experimental_image_conditioned_native_sculpt",
"store_quality_certified": false
```

Only a separately certified backend should report:

```json
"store_quality_certified": true
```

## First build target

The first target should be the concept that recently failed:

```text
hooded armored rider on a spiked reptilian mount, raised curved sword, tail, claws, armored plates, skull scenic base
```

Acceptance criteria:

- planner identifies `mounted_beast`
- weapon is `sword` or `curved_blade`, not rifle
- RigPlan includes mount body, mount head, four legs, tail, rider, raised blade, saddle, spine spikes, base details
- STL is watertight and one fused component
- result report includes concept evidence and planned parts
- if the planner cannot satisfy the concept, generation fails with a clear mismatch/blocker instead of exporting a generic humanoid
