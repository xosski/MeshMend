# MeshMend V1 Roadmap

## Local text-to-3D

- Add open-weight local adapters: Stable Fast 3D, Shap-E, OpenLRM, Wonder3D/Hunyuan variants where licenses allow.
- Normalize all adapter outputs through repair, scale, base, and printability gates.
- Add prompt-to-structured-miniature-spec parser that favors original designs and print-safe parts.

## Image-to-3D

- Import a concept/reference image and reconstruct locally through open-weight models.
- Add alpha/background cleanup, multi-view estimation where supported, and final mesh sheet/card rejection.

## Kitbash generator

- Bundle legally distinct primitive weapons, armor plates, bases, heads, backpacks, shields, and accessories.
- Add parametric part sizing, sockets, mirroring, and boolean attachment.

## Detail sculpting

- Viewport brush picking for smooth/inflate/pinch/flatten/grab/crease/detail stamp.
- Symmetry, stencil/detail alpha stamps, armor panel engraver, rivets, cloth folds, scales, chitin.

## Pose rigging

- Humanoid/creature armature presets.
- Pose presets for heroic stance, aiming, charging, spellcasting, idle, mounted.
- Mesh deformation with printability re-check after posing.

## 8K texture/displacement support

- Optional 8K texture/normal/displacement generation only for adapters and exports that actually produce those assets.
- Bake displacement to geometry for STL workflows when requested.
- PBR material preview for GLB/PLY workflows.

## Resin print validation

- True wall thickness via ray/voxel analysis.
- Overhang/support risk detection.
- Drain/vent hole helper for hollow models.
- Floating island and fragile spear/weapon warnings.

## Marketplace-ready render/export mode

- Turntable renders through Blender local bridge.
- Export bundles: STL, OBJ/MTL, GLB, report JSON, preview PNGs, license/compliance note.
- Enforce no protected logos/symbols in generated prompt metadata and kitbash assets.
