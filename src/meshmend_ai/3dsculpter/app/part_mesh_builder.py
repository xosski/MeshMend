from __future__ import annotations

import json
import math
import os
import re
from dataclasses import asdict
from typing import Any

import numpy as np
import trimesh

try:
    from portable_llm import PortableLLM
except Exception:
    PortableLLM = None

from app.llm_part_planner import MiniaturePlan, PartSpec


class PartMeshBuilder:
    """
    Converts MiniaturePlan into a better 3D miniature scaffold.
    Can optionally query Perseus/PortableLLM for archetype details.
    """

    def __init__(self, use_llm_lookup: bool = True):
        self.use_llm_lookup = use_llm_lookup
        self.llm = None

        if use_llm_lookup and PortableLLM is not None:
            try:
                self.llm = PortableLLM(strict_local_only=True, allow_online_search=True)
            except Exception:
                self.llm = None

    def build_from_plan(self, plan: MiniaturePlan) -> trimesh.Trimesh:
        archetype = self._get_archetype(plan)
        meshes: list[trimesh.Trimesh] = []
        base_height = float(plan.base.get("height_mm", 2.2)) if plan.base.get("enabled", True) else 0.0

        if plan.base.get("enabled", True):
            meshes.append(self._base(plan))

        subject = plan.subject.lower()
        all_text = f"{plan.subject} {plan.style} " + " ".join(p.name + " " + p.shape for p in plan.parts)
        all_text = all_text.lower()
        intent = str(plan.print_rules.get("generation_intent", "character_miniature")).lower()

        if intent != "character_miniature":
            object_meshes = self._build_standalone_object(plan, intent, all_text)
            object = self._scale_standalone_object_to_plan_size(object_meshes, plan, base_height, intent)
            object = self._cohesive_miniature_remesh(object, all_text)
            meshes.append(object)
            combined = trimesh.util.concatenate(meshes)
            combined.merge_vertices()
            combined.remove_unreferenced_vertices()
            try:
                combined.fill_holes()
                combined.fix_normals()
            except Exception:
                pass
            return self._subdivide_and_sculpt_surface(combined, all_text)

        character_meshes: list[trimesh.Trimesh]
        if any(k in all_text for k in ["orc", "ork", "brute", "warboss"]):
            character_meshes = self._build_orc_humanoid(plan)
        elif any(k in all_text for k in ["space marine", "power armor", "power armour", "marine", "bolter", "grimdark", "warhammer", "40k"]):
            character_meshes = self._build_power_armored_humanoid(plan, archetype)
        elif any(k in all_text for k in ["beast", "quadruped", "lizard", "bear"]):
            character_meshes = self._build_quadruped_beast(plan, archetype)
        else:
            character_meshes = self._build_generic_humanoid(plan, archetype)

        character_meshes += self._planned_part_overlays(plan, all_text)
        character_meshes += self._surface_detail_parts(plan, all_text)
        character = self._scale_character_to_plan_height(character_meshes, plan, base_height)
        character = self._cohesive_miniature_remesh(character, all_text)
        meshes.append(character)

        combined = trimesh.util.concatenate(meshes)
        combined.merge_vertices()
        combined.remove_unreferenced_vertices()

        try:
            combined.fill_holes()
            combined.fix_normals()
        except Exception:
            pass

        combined = self._subdivide_and_sculpt_surface(combined, all_text)

        return combined

    def _scale_standalone_object_to_plan_size(
        self,
        object_meshes: list[trimesh.Trimesh],
        plan: MiniaturePlan,
        base_height: float,
        intent: str,
    ) -> trimesh.Trimesh:
        """Scale non-character objects by their dominant dimension.

        A rifle or terrain piece should not be scaled to 32mm *height*; doing so
        makes long props enormous. Character minis keep height scaling, while
        standalone objects use their largest dimension unless they are bust/mask.
        """
        mesh = trimesh.util.concatenate(object_meshes)
        bounds = np.asarray(mesh.bounds, dtype=float)
        extents = np.maximum(bounds[1] - bounds[0], 1e-9)
        target = float(plan.scale_mm)
        if intent in {"wearable_object", "bust"}:
            current = float(extents[2])
        else:
            current = float(np.max(extents))
        if current > 1e-9:
            mesh.apply_scale(max(8.0, target) / current)
        mesh.apply_translation([0.0, 0.0, base_height - float(mesh.bounds[0, 2])])
        return mesh

    def _build_standalone_object(self, plan: MiniaturePlan, intent: str, all_text: str) -> list[trimesh.Trimesh]:
        """Build non-humanoid requests without adding torso/head/arms/legs.

        This is the topology escape hatch for prompts like "plague doctor mask",
        "ornate sci-fi rifle", busts, terrain, and reference-image objects.
        """
        if intent == "wearable_object":
            return self._build_mask_or_helmet(plan, all_text)
        if intent == "prop_object":
            return self._build_weapon_or_prop_object(plan, all_text)
        if intent == "bust":
            return self._build_bust_object(plan, all_text)
        if intent == "vehicle_object":
            return self._build_vehicle_object(plan, all_text)
        if intent == "terrain_object":
            return self._build_terrain_object(plan, all_text)
        return self._build_generic_standalone_subject(plan, all_text)

    def _build_mask_or_helmet(self, plan: MiniaturePlan, all_text: str) -> list[trimesh.Trimesh]:
        meshes: list[trimesh.Trimesh] = []
        shell = self._capsule([0, 0, 8.2], [3.1, 1.25, 4.2])
        front_plate = self._box("mask_front_faceplate", [0, -1.08, 7.9], [4.8, 0.42, 5.8])
        brow = self._box("mask_heavy_brow", [0, -1.42, 9.9], [4.2, 0.30, 0.42])
        chin = self._box("mask_chin_plate", [0, -1.38, 5.7], [2.4, 0.35, 0.75])
        meshes += [shell, front_plate, brow, chin]

        for x in [-1.05, 1.05]:
            lens = trimesh.creation.cylinder(radius=0.52, height=0.30, sections=32)
            lens.apply_transform(trimesh.transformations.rotation_matrix(math.pi / 2, [1, 0, 0]))
            lens.apply_translation([x, -1.72, 8.95])
            rim = trimesh.creation.torus(major_radius=0.56, minor_radius=0.075, major_sections=32, minor_sections=8)
            rim.apply_scale([1.0, 0.35, 1.0])
            rim.apply_transform(trimesh.transformations.rotation_matrix(math.pi / 2, [1, 0, 0]))
            rim.apply_translation([x, -1.90, 8.95])
            meshes += [lens, rim]

        if "plague" in all_text or "beak" in all_text:
            beak = trimesh.creation.cone(radius=0.78, height=4.8, sections=32)
            beak.apply_scale([0.82, 1.0, 0.58])
            beak.apply_transform(trimesh.transformations.rotation_matrix(math.pi / 2, [1, 0, 0]))
            beak.apply_translation([0, -3.65, 7.45])
            seam = self._box("beak_center_seam", [0, -3.75, 7.62], [0.12, 4.15, 0.14])
            meshes += [beak, seam]
            for i, y in enumerate([-1.9, -2.45, -3.0, -3.55]):
                meshes.append(self._rivet([-0.46, y, 7.95 - i * 0.09], radius=0.11))
                meshes.append(self._rivet([0.46, y, 7.95 - i * 0.09], radius=0.11))
        else:
            vent = self._box("front_filter_grille", [0, -1.78, 6.75], [1.7, 0.22, 0.75])
            meshes.append(vent)
            for x in [-0.55, -0.25, 0.05, 0.35, 0.65]:
                meshes.append(self._box("filter_vertical_slot", [x, -1.96, 6.75], [0.06, 0.08, 0.62]))

        for side in [-1.0, 1.0]:
            strap = self._box("side_mask_strap", [side * 2.65, 0.10, 7.65], [0.32, 0.34, 3.9])
            buckle = self._box("strap_buckle", [side * 2.72, -0.25, 7.8], [0.55, 0.18, 0.75])
            meshes += [strap, buckle]
        for z in [5.45, 6.2, 7.0, 8.0, 9.0, 9.8]:
            meshes.append(self._box("mask_panel_line", [0, -1.93, z], [3.65, 0.06, 0.06]))
        for side in [-1.0, 1.0]:
            # Cheek filters and hose couplers give the mask a miniature-grade
            # readable silhouette instead of just a smooth face shell.
            cheek = trimesh.creation.cylinder(radius=0.42, height=0.34, sections=28)
            cheek.apply_transform(trimesh.transformations.rotation_matrix(math.pi / 2, [1, 0, 0]))
            cheek.apply_translation([side * 1.75, -1.82, 6.75])
            meshes.append(cheek)
            for i, z in enumerate([6.48, 6.66, 6.84, 7.02]):
                meshes.append(self._box("cheek_filter_louver", [side * 1.75, -2.03, z], [0.58, 0.06, 0.045]))
            hose = self._limb_between([side * 1.72, -1.72, 6.35], [side * 2.42, -0.72, 5.35], 0.085, sections=12)
            meshes.append(hose)

        # Raised trim, stitch holes, and worn scratches: these are separate
        # physical geometry pieces so they survive STL export and slicing.
        trim_points = [(-2.15, 7.1), (-1.9, 8.0), (-1.45, 9.15), (-0.75, 10.0), (0.0, 10.25), (0.75, 10.0), (1.45, 9.15), (1.9, 8.0), (2.15, 7.1)]
        for (x0, z0), (x1, z1) in zip(trim_points, trim_points[1:]):
            meshes.append(self._limb_between([x0, -2.02, z0], [x1, -2.02, z1], 0.055, sections=8))
        for side in [-1.0, 1.0]:
            for i, z in enumerate([5.9, 6.35, 6.8, 7.25, 7.7, 8.15, 8.6, 9.05, 9.5]):
                meshes.append(self._rivet([side * (2.12 - 0.04 * i), -2.08, z], radius=0.075))
        for i, (x, z, angle) in enumerate([(-0.85, 8.35, 0.25), (0.95, 7.65, -0.18), (-1.35, 6.4, -0.12), (1.28, 9.35, 0.18)]):
            scratch = self._box("mask_fine_scratch_gouge", [x, -2.10, z], [0.78, 0.045, 0.055])
            scratch.apply_transform(trimesh.transformations.rotation_matrix(angle, [0, 0, 1]))
            meshes.append(scratch)
        return meshes

    def _build_weapon_or_prop_object(self, plan: MiniaturePlan, all_text: str) -> list[trimesh.Trimesh]:
        meshes: list[trimesh.Trimesh] = []
        if any(k in all_text for k in ["rifle", "gun", "pistol", "cannon", "bolter"]):
            meshes += self._rifle([0, 0, 6.0], length=8.8, scale=1.25)
            meshes.append(self._box("standalone_rifle_stock", [-4.35, 0, 5.75], [3.0, 0.85, 1.45]))
            meshes.append(self._box("standalone_rifle_scope", [0.9, -0.05, 7.15], [2.4, 0.45, 0.45]))
            meshes.append(self._box("standalone_box_magazine", [-0.45, 0, 4.45], [0.95, 0.7, 1.8]))
            for x in [1.8, 2.3, 2.8, 3.3, 3.8, 4.3]:
                meshes.append(self._box("weapon_heat_vent", [x, -0.52, 6.22], [0.20, 0.08, 0.12]))
        elif any(k in all_text for k in ["sword", "blade", "katana"]):
            blade = self._box("standalone_blade", [0, 0, 7.0], [0.75, 0.18, 9.5])
            point = trimesh.creation.cone(radius=0.40, height=1.1, sections=4)
            point.apply_translation([0, 0, 12.3])
            guard = self._box("standalone_crossguard", [0, 0, 2.25], [4.2, 0.34, 0.42])
            grip = self._limb_between([0, 0, -0.3], [0, 0, 2.0], 0.32, sections=20)
            meshes += [blade, point, guard, grip, self._rivet([0, 0, -0.65], radius=0.42)]
        else:
            meshes.append(self._box("standalone_prop_core", [0, 0, 6.0], [6.5, 1.4, 3.4]))
            meshes.append(self._box("prop_raised_trim", [0, -0.78, 6.75], [5.6, 0.16, 0.22]))
            meshes.append(self._box("prop_lower_trim", [0, -0.78, 5.05], [5.0, 0.16, 0.22]))
        for x in [-2.2, -1.1, 0.0, 1.1, 2.2]:
            meshes.append(self._rivet([x, -0.82, 6.05], radius=0.10))
        return meshes

    def _build_bust_object(self, plan: MiniaturePlan, all_text: str) -> list[trimesh.Trimesh]:
        meshes = [
            self._capsule([0, 0, 10.4], [2.1, 1.65, 2.8]),
            self._capsule([0, 0.05, 6.7], [3.7, 2.0, 2.2]),
            self._box("bust_collar", [0, -1.25, 6.8], [5.0, 0.36, 0.7]),
        ]
        for x in [-0.65, 0.65]:
            meshes.append(self._box("bust_eye_socket", [x, -1.5, 10.8], [0.52, 0.12, 0.24]))
        if "plague" in all_text or "beak" in all_text:
            meshes += self._build_mask_or_helmet(plan, all_text)
        return meshes

    def _build_vehicle_object(self, plan: MiniaturePlan, all_text: str) -> list[trimesh.Trimesh]:
        meshes = [
            self._box("vehicle_hull", [0, 0, 5.2], [7.5, 3.2, 2.4]),
            self._box("vehicle_upper_casemate", [0.2, -0.25, 6.75], [4.4, 2.4, 1.7]),
            *self._turret([0.9, -0.2, 8.0]),
        ]
        for side in [-1.0, 1.0]:
            for x in [-2.7, -1.6, -0.5, 0.6, 1.7, 2.8]:
                wheel = trimesh.creation.cylinder(radius=0.48, height=0.32, sections=24)
                wheel.apply_transform(trimesh.transformations.rotation_matrix(math.pi / 2, [1, 0, 0]))
                wheel.apply_translation([x, side * 1.86, 4.2])
                meshes.append(wheel)
            meshes.append(self._box("vehicle_track", [0.05, side * 1.9, 4.15], [6.8, 0.35, 1.1]))
        return meshes

    def _build_terrain_object(self, plan: MiniaturePlan, all_text: str) -> list[trimesh.Trimesh]:
        meshes = [self._box("terrain_ground_slab", [0, 0, 1.0], [9.0, 7.0, 1.2])]
        for i, x in enumerate([-3.0, -1.5, 0.0, 1.4, 2.8]):
            meshes.append(self._box("broken_wall_segment", [x, 1.15, 2.6 + (i % 3) * 0.45], [1.3, 0.55, 3.2 + (i % 2) * 1.2]))
        for i in range(12):
            ang = i * math.tau / 12
            meshes.append(self._box("terrain_rubble_chip", [math.cos(ang) * 3.0, math.sin(ang) * 2.0, 1.8], [0.7, 0.35, 0.28]))
        return meshes

    def _build_generic_standalone_subject(self, plan: MiniaturePlan, all_text: str) -> list[trimesh.Trimesh]:
        meshes = [
            self._capsule([0, 0, 6.2], [3.2, 1.7, 3.1]),
            self._box("standalone_detail_band_upper", [0, -1.42, 7.35], [4.8, 0.18, 0.22]),
            self._box("standalone_detail_band_lower", [0, -1.42, 5.25], [4.2, 0.18, 0.22]),
        ]
        for x in [-1.8, -0.9, 0.0, 0.9, 1.8]:
            meshes.append(self._rivet([x, -1.55, 6.25], radius=0.12))
        return meshes

    def _cohesive_miniature_remesh(self, mesh: trimesh.Trimesh, all_text: str) -> trimesh.Trimesh:
        """Fuse intersecting sculpt parts into a cohesive miniature shell.

        The generator intentionally builds from many semantic pieces (plates,
        vents, pouches, weapon parts). If those are only concatenated, the STL
        reads as stacked primitives even after subdivision. A voxel remesh gives
        wargaming/high-detail prompts a single sculptable skin while preserving
        the original mesh as a safe fallback if the local trimesh backend cannot
        remesh a model.
        """
        if os.environ.get("MESHMEND_ENABLE_COHESIVE_MINIATURE_REMESH", "").strip().lower() not in {"1", "true", "yes"}:
            return mesh

        wants_cohesive_sculpt = any(
            k in all_text
            for k in [
                "8k", "high detail", "highly detailed", "intricate", "display quality", "studio quality",
                "wargaming", "tabletop", "miniature", "heroic scale", "resin-printable", "resin printable",
                "grimdark", "space marine", "power armor", "power armour", "bolter", "warhammer", "40k",
            ]
        )
        if not wants_cohesive_sculpt or len(mesh.faces) < 1000:
            return mesh

        try:
            bounds = np.asarray(mesh.bounds, dtype=float)
            height = float(bounds[1, 2] - bounds[0, 2])
            if height <= 0.0:
                return mesh

            requested_pitch = float(os.environ.get("MESHMEND_COHESIVE_REMESH_PITCH_MM", "0") or 0.0)
            pitch = requested_pitch if requested_pitch > 0.0 else max(0.10, min(0.18, height / 190.0))

            voxels = mesh.voxelized(pitch)
            try:
                voxels = voxels.fill()
            except Exception:
                pass
            remeshed = voxels.marching_cubes
            if not isinstance(remeshed, trimesh.Trimesh) or len(remeshed.faces) < 1000:
                return mesh
            remeshed.apply_transform(voxels.transform)

            remeshed.merge_vertices()
            remeshed.remove_unreferenced_vertices()
            if os.environ.get("MESHMEND_DISABLE_COHESIVE_REMESH_SMOOTHING", "").strip().lower() not in {"1", "true", "yes"}:
                try:
                    trimesh.smoothing.filter_taubin(remeshed, lamb=0.35, nu=-0.38, iterations=2)
                except Exception:
                    pass
            try:
                remeshed.fill_holes()
                remeshed.fix_normals()
            except Exception:
                pass
            return remeshed
        except Exception:
            return mesh

    def _scale_character_to_plan_height(
        self,
        character_meshes: list[trimesh.Trimesh],
        plan: MiniaturePlan,
        base_height: float,
    ) -> trimesh.Trimesh:
        character = trimesh.util.concatenate(character_meshes)
        bounds = np.asarray(character.bounds, dtype=float)
        current_height = float(bounds[1, 2] - bounds[0, 2])
        target_height = max(8.0, float(plan.scale_mm) - float(base_height))
        if current_height > 1e-9:
            anchor = np.array([0.0, 0.0, base_height], dtype=float)
            character.apply_translation(-anchor)
            character.apply_scale(target_height / current_height)
            character.apply_translation(anchor)
        character.apply_translation([0.0, 0.0, base_height - float(character.bounds[0, 2])])
        return character

    def _get_archetype(self, plan: MiniaturePlan) -> dict[str, Any]:
        fallback = {
            "helmet": True,
            "large_shoulders": True,
            "chest_armor": True,
            "backpack": True,
            "rifle": "bolter" in plan.subject.lower(),
            "boots": True,
            "armor_panels": True,
        }
        fallback.update(self._design_profile(plan))

        if not self.llm:
            return fallback

        try:
            prompt = f"""
Create a JSON-only miniature sculpt archetype for this tabletop model.
Do not include copyrighted names or logos.
Focus on shapes: helmet, armor plates, weapons, backpack, boots, shoulder pads.
Return safe generic design controls only. Useful numeric keys:
shoulder_width, torso_bulk, leg_spread, armor_thickness, helmet_scale, backpack_scale,
weapon_scale, pose_twist, stance_forward, detail_density. Keep values between 0.7 and 1.8.

Subject: {plan.subject}
Style: {plan.style}
Parts: {json.dumps([asdict(p) for p in plan.parts])}

Return compact JSON only.
"""
            response = self.llm.ask(prompt) if hasattr(self.llm, "ask") else None
            if not response:
                return fallback

            match = re.search(r"\{.*\}", response, flags=re.S)
            if not match:
                return fallback

            data = json.loads(match.group(0))
            if isinstance(data, dict):
                fallback.update(data)
            return fallback
        except Exception:
            return fallback

    def _design_profile(self, plan: MiniaturePlan) -> dict[str, float | bool]:
        """Translate prompt/planner language into safe shape controls.

        This is the bridge between AI/reference planning and geometry: values here
        alter proportions and pose instead of merely adding labels or decorations.
        """
        intent_text = f"{plan.subject} {plan.style}".lower()
        text = f"{plan.subject} {plan.style} " + " ".join(
            f"{part.name} {part.shape} {' '.join(part.detail_tags)}" for part in plan.parts
        )
        text = text.lower()
        profile: dict[str, float | bool] = {
            "shoulder_width": 1.0,
            "torso_bulk": 1.0,
            "leg_spread": 1.0,
            "armor_thickness": 1.0,
            "helmet_scale": 1.0,
            "backpack_scale": 1.0,
            "weapon_scale": 1.0,
            "pose_twist": 1.0,
            "stance_forward": 1.0,
            "detail_density": 1.0,
        }
        if any(k in text for k in ["grimdark", "power armor", "power armour", "space marine", "warhammer", "40k", "bolter"]):
            profile.update({
                "shoulder_width": 1.35,
                "torso_bulk": 1.22,
                "leg_spread": 1.16,
                "armor_thickness": 1.28,
                "helmet_scale": 1.08,
                "backpack_scale": 1.28,
                "weapon_scale": 1.24,
                "pose_twist": 1.18,
                "stance_forward": 1.12,
                "detail_density": 1.35,
            })
        if any(k in intent_text for k in ["heavy", "juggernaut", "terminator", "bulky", "brutal"]):
            profile["torso_bulk"] = max(float(profile["torso_bulk"]), 1.35)
            profile["armor_thickness"] = max(float(profile["armor_thickness"]), 1.42)
            profile["shoulder_width"] = max(float(profile["shoulder_width"]), 1.42)
        if any(k in intent_text for k in ["dynamic", "charging", "running", "lunging", "action pose"]):
            profile["leg_spread"] = max(float(profile["leg_spread"]), 1.28)
            profile["pose_twist"] = max(float(profile["pose_twist"]), 1.32)
            profile["stance_forward"] = max(float(profile["stance_forward"]), 1.25)
        if any(k in intent_text for k in ["slender", "scout", "agile", "sniper"]):
            profile["torso_bulk"] = min(float(profile["torso_bulk"]), 0.9)
            profile["armor_thickness"] = min(float(profile["armor_thickness"]), 0.92)
        if any(k in text for k in ["8k", "ultra detail", "maximum detail", "high definition"]):
            profile["detail_density"] = max(float(profile["detail_density"]), 1.65)
        return profile

    @staticmethod
    def _archetype_float(archetype: dict[str, Any], key: str, default: float, min_value: float, max_value: float) -> float:
        try:
            value = float(archetype.get(key, default))
        except Exception:
            value = default
        return float(np.clip(value, min_value, max_value))

    def _planned_part_overlays(self, plan: MiniaturePlan, all_text: str) -> list[trimesh.Trimesh]:
        """Materialize AI/planner PartSpec armor/detail entries as geometry.

        Weapon parts are handled by the base builders. This handles the richer
        armor/detail/cloth parts emitted by reference-enriched planning so the AI
        can change the actual design language, not just text labels.
        """
        overlays: list[trimesh.Trimesh] = []
        for part in plan.parts:
            text = f"{part.name} {part.kind} {part.shape} {' '.join(part.detail_tags)}".lower()
            if part.kind in {"weapon", "mounted_weapon", "weapon_arm", "body", "limb"}:
                continue
            pos = [float(v) for v in (part.position or [0, 0, 12])[:3]]
            scale = [max(0.05, float(v)) for v in (part.scale or [1, 1, 1])[:3]]
            if "beak" in text or "plague" in text:
                beak = trimesh.creation.cone(radius=max(0.25, scale[0] * 0.35), height=max(1.2, scale[1]), sections=18)
                beak.apply_transform(trimesh.transformations.rotation_matrix(math.pi / 2, [1, 0, 0]))
                beak.apply_translation(pos)
                overlays.append(beak)
                for x in [-0.45, 0.45]:
                    overlays.append(self._rivet([x, pos[1] - 0.25, pos[2] + 0.25], radius=0.13))
            elif "mushroom" in text or "fungal" in text:
                cap = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
                cap.apply_scale([max(1.6, scale[0] * 0.55), max(1.1, scale[1] * 0.55), max(0.35, scale[2] * 0.35)])
                cap.apply_translation(pos)
                overlays.append(cap)
                for x in [-0.9, -0.3, 0.3, 0.9]:
                    overlays.append(self._box("mushroom_gill_line", [pos[0] + x, pos[1] - 0.9, pos[2] - 0.25], [0.08, 0.12, 0.7]))
            elif "crystal" in text or "ice" in text:
                for side in [-1.0, -0.45, 0.0, 0.45, 1.0]:
                    spike = trimesh.creation.cone(radius=0.22, height=max(1.6, scale[2] * (0.45 + abs(side) * 0.25)), sections=5)
                    spike.apply_translation([pos[0] + side * scale[0] * 0.35, pos[1], pos[2] + abs(side) * 0.25])
                    overlays.append(spike)
            elif "flame" in text or "fire" in text:
                for i, x in enumerate([-0.7, -0.25, 0.25, 0.7]):
                    flame = trimesh.creation.cone(radius=0.28, height=max(1.4, scale[2] * (0.45 + i * 0.08)), sections=12)
                    flame.apply_transform(trimesh.transformations.rotation_matrix(x * 0.35, [0, 0, 1]))
                    flame.apply_translation([pos[0] + x, pos[1] - 0.1, pos[2]])
                    overlays.append(flame)
            elif "banner" in text or "flag" in text:
                overlays.append(self._limb_between([pos[0], pos[1], pos[2] - scale[2] * 0.5], [pos[0], pos[1], pos[2] + scale[2] * 0.5], 0.12, sections=12))
                overlays.append(self._box("tattered_banner_flag", [pos[0] + scale[0] * 0.35, pos[1] - 0.12, pos[2] + scale[2] * 0.18], [scale[0] * 0.7, 0.10, scale[2] * 0.45]))
            elif "crest" in text or "plume" in text:
                for i, zoff in enumerate([0.0, 0.35, 0.7, 1.05]):
                    plume = self._box("helmet_crest_segment", [pos[0], pos[1] - 0.15, pos[2] + zoff], [scale[0], 0.12, 0.22])
                    plume.apply_transform(trimesh.transformations.rotation_matrix(0.08 * i, [0, 0, 1]))
                    overlays.append(plume)
            elif "coat" in text or "cape" in text or "cloak" in text or "fur" in text:
                overlays.append(self._box("large_back_cloak_or_coat", [pos[0], max(0.9, pos[1]), pos[2]], [scale[0], max(0.18, scale[1]), scale[2]]))
                for x in [-1.4, -0.7, 0.0, 0.7, 1.4]:
                    overlays.append(self._box("cloak_deep_vertical_fold", [pos[0] + x, pos[1] - 0.25, pos[2]], [0.12, 0.08, scale[2] * 0.85]))
            elif "mandible" in text:
                for side in [-1.0, 1.0]:
                    mandible = trimesh.creation.cone(radius=0.18, height=1.5, sections=14)
                    mandible.apply_transform(trimesh.transformations.rotation_matrix(side * 0.75, [0, 0, 1]))
                    mandible.apply_transform(trimesh.transformations.rotation_matrix(math.pi / 2, [1, 0, 0]))
                    mandible.apply_translation([pos[0] + side * 0.55, pos[1] - 0.3, pos[2]])
                    overlays.append(mandible)
            elif "shield" in text:
                shield = self._box("planned_kite_shield_body", pos, [scale[0], max(0.18, scale[1]), scale[2]])
                overlays.append(shield)
                overlays.append(self._rivet([pos[0], pos[1] - 0.26, pos[2] + scale[2] * 0.1], radius=max(0.22, scale[0] * 0.12)))
                for z in [-0.38, 0.0, 0.38]:
                    overlays.append(self._box("planned_shield_rim_segment", [pos[0], pos[1] - 0.32, pos[2] + z * scale[2]], [scale[0] * 0.9, 0.08, 0.08]))
            elif "hat" in text:
                hat = trimesh.creation.cone(radius=max(0.7, scale[0] * 0.55), height=max(2.2, scale[2]), sections=32)
                hat.apply_translation(pos)
                overlays.append(hat)
                brim = trimesh.creation.torus(major_radius=max(0.85, scale[0] * 0.6), minor_radius=0.08, major_sections=32, minor_sections=6)
                brim.apply_translation([pos[0], pos[1], pos[2] - scale[2] * 0.45])
                overlays.append(brim)
            elif "horn" in text:
                for side in [-1.0, 1.0]:
                    horn = trimesh.creation.cone(radius=0.22, height=max(1.4, scale[0] * 0.6), sections=18)
                    horn.apply_transform(trimesh.transformations.rotation_matrix(side * 0.65, [0, 1, 0]))
                    horn.apply_translation([pos[0] + side * 0.9, pos[1] - 0.12, pos[2] + 0.15])
                    overlays.append(horn)
            elif "wing" in text:
                for side in [-1.0, 1.0]:
                    wing = self._box("planned_bat_wing_membrane", [side * 3.2, pos[1], pos[2]], [scale[0] * 0.42, max(0.12, scale[1]), scale[2]])
                    wing.apply_transform(trimesh.transformations.rotation_matrix(side * 0.28, [0, 0, 1]))
                    overlays.append(wing)
                    for rib in [0.25, 0.55, 0.85]:
                        overlays.append(self._limb_between([side * 1.2, pos[1] - 0.05, pos[2] + 1.8], [side * scale[0] * rib, pos[1] - 0.18, pos[2] - 2.3], 0.08, sections=10))
            elif "rib" in text or "skull" in text:
                if "rib" in text:
                    for side in [-1.0, 1.0]:
                        for z in [12.35, 12.85, 13.35, 13.85, 14.35]:
                            rib = self._box("planned_exposed_rib", [side * 0.8, -2.06, z], [1.15, 0.08, 0.10])
                            rib.apply_transform(trimesh.transformations.rotation_matrix(side * 0.20, [0, 0, 1]))
                            overlays.append(rib)
                    overlays.append(self._box("planned_spine_column", [0, -2.02, 13.3], [0.18, 0.10, 2.7]))
                if "skull" in text:
                    overlays.append(self._box("planned_skull_brow", [0, -1.82, 18.55], [1.25, 0.14, 0.18]))
                    overlays.append(self._box("planned_skull_teeth", [0, -1.88, 17.85], [0.95, 0.10, 0.18]))
            elif "quiver" in text or "arrow" in text:
                overlays.append(self._box("planned_quiver_tube", pos, [scale[0], scale[1], scale[2]]))
                for x in [-0.24, 0.0, 0.24]:
                    overlays.append(self._limb_between([pos[0] + x, pos[1], pos[2] + 1.1], [pos[0] + x, pos[1], pos[2] + 2.4], 0.035, sections=8))
            elif "pauldron" in text or "shoulder" in text:
                for side in [-1.0, 1.0]:
                    overlays.append(self._shoulder_pad([side * max(3.4, scale[0] * 0.52), -0.05, pos[2]], mirror=side > 0, scale=max(1.0, scale[0] / 5.0)))
                    overlays.append(self._box("planned_pauldron_trim", [side * max(3.4, scale[0] * 0.52), -1.65, pos[2] - 0.15], [max(1.6, scale[0] * 0.32), 0.15, 0.16]))
            elif "backpack" in text or "power pack" in text or "reactor" in text:
                overlays.append(self._box("planned_power_backpack", [0, max(1.8, pos[1]), pos[2]], [scale[0], max(0.8, scale[1]), scale[2]]))
                for x in [-scale[0] * 0.28, scale[0] * 0.28]:
                    vent = trimesh.creation.cylinder(radius=max(0.22, scale[0] * 0.08), height=max(0.7, scale[1]), sections=18)
                    vent.apply_translation([x, max(2.25, pos[1] + scale[1] * 0.45), pos[2] + scale[2] * 0.25])
                    overlays.append(vent)
            elif "greave" in text or "boot" in text or "shin" in text:
                for side in [-1.0, 1.0]:
                    overlays.append(self._box("planned_heavy_greave", [side * 1.55, -1.25, 5.6], [max(1.0, scale[0] * 0.24), 0.22, max(1.7, scale[2] * 0.48)]))
                    overlays.append(self._box("planned_toe_cap", [side * 1.55, -1.55, 2.8], [max(1.2, scale[0] * 0.28), 0.28, 0.35]))
            elif "seal" in text or "scroll" in text or "parchment" in text:
                overlays += self._purity_seal(pos)
                overlays += self._purity_seal([pos[0] + 1.2, pos[1] - 0.05, pos[2] - 1.0])
            elif "armor" in text or "plate" in text or "panel" in text or "mechanical" in text or "piston" in text:
                overlays.append(self._box("planned_armor_plate", pos, [scale[0], max(0.10, scale[1]), scale[2]]))
                for z_offset in [-0.35, 0.0, 0.35]:
                    overlays.append(self._box("planned_panel_line", [pos[0], pos[1] - 0.12, pos[2] + z_offset * scale[2]], [scale[0] * 0.82, 0.06, 0.05]))
            elif "cloth" in text or "robe" in text or "loincloth" in text:
                for x in [-0.45, 0.0, 0.45]:
                    overlays.append(self._box("planned_cloth_fold", [pos[0] + x, pos[1] - 0.2, pos[2]], [0.18, 0.08, scale[2]]))
        return overlays

    def _base(self, plan: MiniaturePlan) -> trimesh.Trimesh:
        diameter = float(plan.base.get("diameter_mm", 32))
        height = float(plan.base.get("height_mm", 2.2))
        base = trimesh.creation.cylinder(radius=diameter / 2, height=height, sections=72)
        base.apply_translation([0, 0, height / 2])

        rocks = []
        for i in range(10):
            ang = i * math.tau / 10
            rock = trimesh.creation.icosphere(subdivisions=1, radius=0.55)
            rock.apply_scale([1.4, 0.8, 0.45])
            rock.apply_translation([
                math.cos(ang) * diameter * 0.32,
                math.sin(ang) * diameter * 0.32,
                height + 0.2,
            ])
            rocks.append(rock)

        debris = []
        for i in range(14):
            ang = (i + 0.35) * math.tau / 14
            radius = diameter * (0.18 + 0.23 * ((i * 37) % 11) / 10.0)
            chip = trimesh.creation.box(extents=[0.75, 0.18, 0.12])
            chip.apply_transform(trimesh.transformations.rotation_matrix(ang, [0, 0, 1]))
            chip.apply_translation([
                math.cos(ang) * radius,
                math.sin(ang) * radius,
                height + 0.12,
            ])
            debris.append(chip)

        return trimesh.util.concatenate([base, *rocks, *debris])

    def _build_power_armored_humanoid(self, plan: MiniaturePlan, archetype: dict[str, Any]) -> list[trimesh.Trimesh]:
        meshes = []
        shoulder_width = self._archetype_float(archetype, "shoulder_width", 1.0, 0.75, 1.8)
        torso_bulk = self._archetype_float(archetype, "torso_bulk", 1.0, 0.75, 1.7)
        leg_spread = self._archetype_float(archetype, "leg_spread", 1.0, 0.75, 1.55)
        armor_thickness = self._archetype_float(archetype, "armor_thickness", 1.0, 0.75, 1.7)
        helmet_scale = self._archetype_float(archetype, "helmet_scale", 1.0, 0.75, 1.45)
        backpack_scale = self._archetype_float(archetype, "backpack_scale", 1.0, 0.75, 1.7)
        weapon_scale = self._archetype_float(archetype, "weapon_scale", 1.0, 0.75, 1.8)
        pose_twist = self._archetype_float(archetype, "pose_twist", 1.0, 0.0, 1.8) - 1.0
        stance_forward = self._archetype_float(archetype, "stance_forward", 1.0, 0.75, 1.6)

        # boots / legs
        foot_x = 2.2 * leg_spread
        meshes.append(self._box("left_boot", [-foot_x, -0.12 * stance_forward, 3.1], [1.8 * armor_thickness, 2.3, 1.4]))
        meshes.append(self._box("right_boot", [foot_x, 0.18 * stance_forward, 3.1], [1.8 * armor_thickness, 2.3, 1.4]))
        meshes.append(self._capsule([-1.7 * leg_spread, -0.05 * stance_forward, 7.0], [1.25 * armor_thickness, 1.05, 4.8], rotate_y=0.08 * pose_twist))
        meshes.append(self._capsule([1.7 * leg_spread, 0.12 * stance_forward, 7.0], [1.25 * armor_thickness, 1.05, 4.8], rotate_y=-0.08 * pose_twist))

        # torso / chest armor
        torso = self._capsule([0, 0, 13.0], [3.4 * torso_bulk, 2.2 * torso_bulk, 5.2], rotate_z=0.06 * pose_twist)
        chest = self._box("chest_plate", [0, -1.35 * torso_bulk, 14.0], [4.2 * torso_bulk, 0.55 * armor_thickness, 3.8])
        abdomen = self._box("abdomen_plate", [0, -1.2 * torso_bulk, 10.9], [3.2 * torso_bulk, 0.45 * armor_thickness, 1.5])
        meshes += [torso, chest, abdomen]

        # helmet/head
        helmet = trimesh.creation.icosphere(subdivisions=2, radius=1.45 * helmet_scale)
        helmet.apply_scale([1.0, 0.9, 1.15])
        helmet.apply_translation([0, 0, 18.1])
        visor = self._box("visor", [0, -1.25 * helmet_scale, 18.25], [1.65 * helmet_scale, 0.25, 0.45])
        mouth_grill = self._box("mouth_grill", [0, -1.3 * helmet_scale, 17.55], [1.1 * helmet_scale, 0.22, 0.45])
        meshes += [helmet, visor, mouth_grill]

        # huge pauldrons
        meshes.append(self._shoulder_pad([-3.55 * shoulder_width, 0, 16.0], mirror=False, scale=shoulder_width * armor_thickness))
        meshes.append(self._shoulder_pad([3.55 * shoulder_width, 0, 16.0], mirror=True, scale=shoulder_width * armor_thickness))

        # arms
        meshes.append(self._capsule([-4.6 * shoulder_width, -0.15, 13.4], [0.9 * armor_thickness, 0.8, 3.7], rotate_y=0.55 + 0.18 * pose_twist))
        meshes.append(self._capsule([4.6 * shoulder_width, -0.15, 13.4], [0.9 * armor_thickness, 0.8, 3.7], rotate_y=-0.55 + 0.18 * pose_twist))
        meshes.append(self._capsule([-3.1 * shoulder_width, -1.15 * stance_forward, 12.0], [0.75 * armor_thickness, 0.7, 3.2], rotate_x=1.25))
        meshes.append(self._capsule([3.1 * shoulder_width, -1.15 * stance_forward, 12.0], [0.75 * armor_thickness, 0.7, 3.2], rotate_x=-1.25))

        # backpack
        backpack = self._box("power_pack", [0, 1.65 * backpack_scale, 15.2], [3.6 * backpack_scale, 1.0 * backpack_scale, 3.6 * backpack_scale])
        vent_l = trimesh.creation.cylinder(radius=0.45, height=1.3, sections=20)
        vent_l.apply_translation([-1.25 * backpack_scale, 2.25 * backpack_scale, 16.5])
        vent_r = vent_l.copy()
        vent_r.apply_translation([2.5 * backpack_scale, 0, 0])
        meshes += [backpack, vent_l, vent_r]

        # bolter rifle across chest
        meshes += self._rifle([0, -2.25 * stance_forward, 13.2], length=7.2 * weapon_scale, scale=weapon_scale)

        # armor panel details
        for x in [-1.7, 1.7]:
            meshes.append(self._box("knee_plate", [x * leg_spread, -0.9, 7.5], [1.45 * armor_thickness, 0.35, 1.05]))
            meshes.append(self._box("shin_plate", [x * leg_spread, -0.95, 5.4], [1.25 * armor_thickness, 0.35, 1.9]))

        meshes += self._power_armor_core_design_parts(
            shoulder_width=shoulder_width,
            torso_bulk=torso_bulk,
            leg_spread=leg_spread,
            armor_thickness=armor_thickness,
            helmet_scale=helmet_scale,
            backpack_scale=backpack_scale,
            weapon_scale=weapon_scale,
            stance_forward=stance_forward,
        )

        return meshes

    def _power_armor_core_design_parts(
        self,
        *,
        shoulder_width: float,
        torso_bulk: float,
        leg_spread: float,
        armor_thickness: float,
        helmet_scale: float,
        backpack_scale: float,
        weapon_scale: float,
        stance_forward: float,
    ) -> list[trimesh.Trimesh]:
        """Build recognizable miniature sculpt features, not just primitive body masses."""
        details: list[trimesh.Trimesh] = []

        # Helmet/face cluster: lenses, grille, cheek filters, neck gorget.
        for x in [-0.45, 0.45]:
            lens = self._box("helmet_deep_lens_socket", [x * helmet_scale, -1.72 * helmet_scale, 18.42], [0.36 * helmet_scale, 0.10, 0.16])
            lens.apply_transform(trimesh.transformations.rotation_matrix(0.08 if x < 0 else -0.08, [0, 0, 1]))
            details.append(lens)
        for x in [-0.48, 0.0, 0.48]:
            details.append(self._box("helmet_breath_grille_bar", [x * helmet_scale, -1.84 * helmet_scale, 17.70], [0.09, 0.08, 0.48]))
        for x in [-0.72, 0.72]:
            details.append(self._box("helmet_side_filter", [x * helmet_scale, -1.62 * helmet_scale, 17.82], [0.38, 0.18, 0.52]))
        gorget = trimesh.creation.torus(major_radius=1.45 * helmet_scale, minor_radius=0.11 * armor_thickness, major_sections=32, minor_sections=8)
        gorget.apply_scale([1.15, 0.75, 0.22])
        gorget.apply_translation([0, -0.15, 16.95])
        details.append(gorget)

        # Layered torso armor and abdominal plates that read as sculpted hard surface.
        for z, width in [(15.00, 4.25), (14.45, 4.55), (13.90, 4.15)]:
            plate = self._box("overlapping_chest_plate", [0, -1.96 * torso_bulk, z], [width * torso_bulk, 0.16 * armor_thickness, 0.16])
            details.append(plate)
        for row, z in enumerate([13.20, 12.65, 12.10, 11.55]):
            width = (2.95 - row * 0.18) * torso_bulk
            details.append(self._box("segmented_abdominal_plate", [0, -1.90 * torso_bulk, z], [width, 0.15 * armor_thickness, 0.28]))
            for x in [-width * 0.42, width * 0.42]:
                details.append(self._rivet([x, -2.03 * torso_bulk, z], radius=0.09 * armor_thickness))
        for side in [-1.0, 1.0]:
            strap = self._box("diagonal_torso_strap", [side * 1.05 * torso_bulk, -2.04 * torso_bulk, 13.25], [0.26, 0.12, 3.8])
            strap.apply_transform(trimesh.transformations.rotation_matrix(-side * 0.34, [0, 0, 1]))
            details.append(strap)

        # Shoulder trim rows and raised blank heraldry plates.
        for side in [-1.0, 1.0]:
            sx = side * 3.55 * shoulder_width
            for offset, width in [(0.42, 2.45), (0.0, 2.75), (-0.42, 2.35)]:
                details.append(self._box("layered_pauldron_armor_band", [sx, -1.74, 16.05 + offset], [width * shoulder_width, 0.14 * armor_thickness, 0.13]))
            badge = trimesh.creation.cylinder(radius=0.50 * shoulder_width, height=0.12 * armor_thickness, sections=24)
            badge.apply_transform(trimesh.transformations.rotation_matrix(math.pi / 2, [1, 0, 0]))
            badge.apply_translation([sx + side * 0.28, -1.94, 16.15])
            details.append(badge)

        # Arms: gauntlets, elbow pads, palm/finger blocks around weapon grip.
        for side in [-1.0, 1.0]:
            arm_x = side * 3.35 * shoulder_width
            details.append(self._box("large_elbow_cop", [side * 4.25 * shoulder_width, -0.95 * stance_forward, 12.65], [0.82 * armor_thickness, 0.34, 0.82]))
            details.append(self._box("blocky_power_gauntlet", [arm_x, -1.95 * stance_forward, 12.35], [0.95 * armor_thickness, 0.58, 0.78]))
            for finger in [-0.24, -0.08, 0.08, 0.24]:
                details.append(self._box("individual_armored_finger", [arm_x + side * finger, -2.32 * stance_forward, 12.00], [0.10, 0.12, 0.42]))
            wrist = trimesh.creation.torus(major_radius=0.48 * armor_thickness, minor_radius=0.055 * armor_thickness, major_sections=20, minor_sections=6)
            wrist.apply_scale([1.25, 0.75, 0.42])
            wrist.apply_translation([arm_x, -1.86 * stance_forward, 12.82])
            details.append(wrist)

        # Legs: separated thigh plates, knee trim, greave grooves and heavy toe caps.
        for side in [-1.0, 1.0]:
            leg_x = side * 1.55 * leg_spread
            details.append(self._box("upper_thigh_armor_front", [leg_x, -1.04, 8.35], [1.05 * armor_thickness, 0.20, 1.35]))
            details.append(self._box("rounded_knee_trim", [leg_x, -1.25, 7.25], [1.30 * armor_thickness, 0.16, 0.16]))
            for z in [6.45, 5.85, 5.25, 4.65]:
                details.append(self._box("deep_greave_panel_line", [leg_x, -1.34, z], [0.95 * armor_thickness, 0.09, 0.08]))
            details.append(self._box("split_boot_toe_cap", [leg_x, -1.42, 2.75], [1.35 * armor_thickness, 0.25, 0.30]))
            details.append(self._box("boot_center_toe_split", [leg_x, -1.58, 2.78], [0.08, 0.08, 0.38]))

        # Backpack: exhaust stacks, ribbed vents, hoses/cables to the torso.
        for x in [-1.20 * backpack_scale, 1.20 * backpack_scale]:
            stack = trimesh.creation.cylinder(radius=0.32 * backpack_scale, height=1.55 * backpack_scale, sections=22)
            stack.apply_translation([x, 2.78 * backpack_scale, 16.35])
            details.append(stack)
            for z in [15.70, 16.05, 16.40, 16.75]:
                details.append(self._box("backpack_vent_louver", [x, 3.05 * backpack_scale, z], [0.58 * backpack_scale, 0.08, 0.08]))
        for side in [-1.0, 1.0]:
            details.append(self._limb_between([side * 0.82, 2.18 * backpack_scale, 14.5], [side * 1.55, 0.65, 13.1], 0.075 * armor_thickness, sections=10))

        # Weapon additional shape language: muzzle brake, scope, vents, magazine lugs.
        details.append(self._box("weapon_scope_block", [1.0 * weapon_scale, -2.96 * stance_forward, 14.04], [1.45 * weapon_scale, 0.25, 0.24]))
        details.append(self._box("weapon_muzzle_brake_block", [4.65 * weapon_scale, -2.90 * stance_forward, 13.36], [0.46 * weapon_scale, 0.44, 0.44]))
        for x in [2.0, 2.45, 2.90, 3.35, 3.80]:
            details.append(self._box("weapon_barrel_side_vent", [x * weapon_scale, -3.12 * stance_forward, 13.50], [0.20 * weapon_scale, 0.08, 0.08]))
        details.append(self._box("oversized_box_magazine", [-0.20 * weapon_scale, -3.08 * stance_forward, 12.55], [0.78 * weapon_scale, 0.34, 1.15]))

        # Painter-friendly accessories: pouches, seals, tiny plaques.
        for x in [-1.70, -1.05, 1.05, 1.70]:
            details.append(self._box("belt_pouch_with_flap", [x * torso_bulk, -2.12 * torso_bulk, 10.10], [0.44, 0.22, 0.55]))
            details.append(self._box("belt_pouch_flap", [x * torso_bulk, -2.25 * torso_bulk, 10.35], [0.36, 0.07, 0.08]))
        details += self._purity_seal([1.55 * torso_bulk, -2.18 * torso_bulk, 14.15])
        details += self._purity_seal([-1.45 * torso_bulk, -2.18 * torso_bulk, 12.85])

        return details

    def _build_generic_humanoid(self, plan: MiniaturePlan, archetype: dict[str, Any]) -> list[trimesh.Trimesh]:
        all_text = f"{plan.subject} {plan.style} " + " ".join(p.name + " " + p.shape for p in plan.parts)
        all_text = all_text.lower()
        is_orc = any(k in all_text for k in ["orc", "ork", "brute", "warboss"])
        torso_bulk = self._archetype_float(archetype, "torso_bulk", 1.0, 0.65, 1.7)
        shoulder_width = self._archetype_float(archetype, "shoulder_width", 1.0, 0.65, 1.8)
        leg_spread = self._archetype_float(archetype, "leg_spread", 1.0, 0.65, 1.6)
        pose_twist = self._archetype_float(archetype, "pose_twist", 1.0, 0.0, 1.8) - 1.0
        if any(k in all_text for k in ["wizard", "robe", "mage", "sorcerer", "archer", "ranger", "thin agile", "slender"]):
            torso_bulk = min(torso_bulk, 0.82)
            shoulder_width = min(shoulder_width, 0.88)
        if any(k in all_text for k in ["robot", "mech", "war-machine", "wide squat", "bulky massing"]):
            torso_bulk = max(torso_bulk, 1.35)
            shoulder_width = max(shoulder_width, 1.35)
            leg_spread = max(leg_spread, 1.22)
        signature = sum((index + 1) * ord(ch) for index, ch in enumerate(all_text[:240]))
        variant_width = 0.78 + (signature % 31) / 100.0
        variant_height = 0.86 + ((signature // 17) % 29) / 100.0
        variant_pose = (((signature // 221) % 41) - 20) / 100.0
        leg_spread *= 0.85 + ((signature // 997) % 35) / 100.0
        shoulder_width *= 0.85 + ((signature // 1597) % 35) / 100.0
        torso_scale = [3.8, 2.35, 5.8] if is_orc else [3.0 * torso_bulk, 2.0 * torso_bulk, 5.4]
        leg_scale = [1.15, 1.0, 5.0] if is_orc else [1.0 * torso_bulk, 0.9, 5.0]
        arm_scale = [1.05, 0.95, 4.5] if is_orc else [0.85 * torso_bulk, 0.75, 4.2]
        head_radius = 1.65 if is_orc else 1.45 * (0.92 if "robot" in all_text or "mech" in all_text else 1.0)
        if not is_orc:
            torso_scale[0] *= variant_width
            torso_scale[2] *= variant_height
            leg_scale[0] *= variant_width
            arm_scale[0] *= variant_width
            head_radius *= 0.94 + ((signature // 2873) % 12) / 100.0
        meshes = [
            self._capsule([0, 0, 13], torso_scale, rotate_z=0.05 * pose_twist + variant_pose),
            self._capsule([-1.4 * leg_spread, -0.08 * pose_twist, 7], leg_scale, rotate_y=0.08 * pose_twist),
            self._capsule([1.4 * leg_spread, 0.12 * pose_twist, 7], leg_scale, rotate_y=-0.08 * pose_twist),
            self._capsule([-3.6 * shoulder_width, -0.15, 13.6], arm_scale, rotate_y=0.5 + 0.12 * pose_twist + variant_pose),
            self._capsule([3.6 * shoulder_width, -0.15, 13.6], arm_scale, rotate_y=-0.5 + 0.12 * pose_twist + variant_pose),
        ]

        head = trimesh.creation.icosphere(subdivisions=2, radius=head_radius)
        head.apply_translation([0, 0, 18.4])
        meshes.append(head)

        if is_orc:
            meshes.append(self._box("orc_lower_jaw_mass", [0, -1.15, 17.7], [2.1, 0.75, 0.8]))
            meshes.append(self._shoulder_pad([-3.35, -0.05, 16.0], mirror=False))
            meshes.append(self._shoulder_pad([3.35, -0.05, 16.0], mirror=True))
            meshes.append(self._box("left_heavy_boot", [-1.5, -0.15, 2.75], [1.9, 2.0, 1.1]))
            meshes.append(self._box("right_heavy_boot", [1.5, -0.15, 2.75], [1.9, 2.0, 1.1]))

        if any(k in all_text for k in ["wizard", "robe", "mage", "sorcerer", "archer", "ranger", "cloak"]):
            meshes.append(self._box("distinct_robed_lower_silhouette", [0, -0.25, 7.7], [4.2, 1.7, 6.2]))
            meshes.append(self._box("deep_robe_front_panel", [0, -1.25, 8.8], [2.1, 0.26, 5.4]))
            for x in [-1.35, -0.75, -0.2, 0.35, 0.95, 1.45]:
                fold = self._box("long_robed_silhouette_fold", [x, -1.58, 8.2], [0.16, 0.11, 5.8])
                fold.apply_transform(trimesh.transformations.rotation_matrix(x * 0.10, [0, 0, 1]))
                meshes.append(fold)

        if any(k in all_text for k in ["robot", "mech", "android", "cyborg", "war-machine"]):
            meshes.append(self._box("distinct_robot_block_torso_core", [0, -1.1, 13.4], [4.8, 1.1, 4.4]))
            meshes.append(self._box("robot_square_head_faceplate", [0, -0.8, 18.35], [2.1, 1.2, 1.8]))
            for x in [-1.4, 1.4]:
                meshes.append(self._limb_between([x, -0.6, 9.0], [x, -0.9, 4.0], 0.22, sections=14))
                meshes.append(self._limb_between([x + 0.38, -0.6, 9.0], [x + 0.38, -0.9, 4.0], 0.16, sections=12))

        if any(k in all_text for k in ["undead", "skeleton", "zombie", "necromancer"]):
            meshes.append(self._box("sunken_undead_chest_gap", [0, -1.75, 13.5], [2.0, 0.18, 2.8]))
            for x in [-0.8, -0.4, 0.0, 0.4, 0.8]:
                meshes.append(self._box("thin_bone_rib_silhouette", [x, -1.95, 13.5], [0.10, 0.10, 2.3]))

        if any(k in all_text for k in ["demon", "daemon", "horned", "wing"]):
            meshes.append(self._capsule([0, 0.35, 7.3], [0.45, 0.45, 4.2], rotate_y=math.pi / 2))

        for part in plan.parts:
            if part.kind in {"weapon", "mounted_weapon", "weapon_arm"}:
                meshes.append(self._weapon_mesh(part))

        return meshes

    def _build_orc_humanoid(self, plan: MiniaturePlan) -> list[trimesh.Trimesh]:
        """Build a visibly different fantasy orc silhouette instead of the generic mannequin."""
        meshes: list[trimesh.Trimesh] = []
        all_text = f"{plan.subject} {plan.style} " + " ".join(p.name + " " + p.shape for p in plan.parts)
        all_text = all_text.lower()
        has_axe = "axe" in all_text

        # Wide, hunched core.
        meshes.append(self._capsule([0, 0, 12.8], [4.4, 2.7, 5.9], rotate_x=-0.08))
        meshes.append(self._box("orc_pelvis", [0, 0.05, 8.8], [4.6, 2.6, 1.7]))
        meshes.append(self._capsule([0, -0.25, 15.6], [4.9, 2.35, 1.2], rotate_x=-0.12))

        # Braced, asymmetrical legs and heavy feet.
        meshes.append(self._limb_between([-2.0, 0.2, 8.3], [-2.8, -0.15, 3.3], 0.82, sections=22))
        meshes.append(self._limb_between([1.8, 0.1, 8.2], [2.7, 0.55, 3.2], 0.86, sections=22))
        meshes.append(self._box("left_orc_boot", [-3.0, -0.6, 2.85], [2.4, 2.1, 1.0]))
        meshes.append(self._box("right_orc_boot", [2.9, 0.2, 2.85], [2.4, 2.1, 1.0]))

        # Head pushed forward with jaw/tusks so the silhouette reads as orc.
        head = trimesh.creation.icosphere(subdivisions=2, radius=1.75)
        head.apply_scale([1.08, 0.92, 1.0])
        head.apply_translation([0, -0.55, 18.6])
        meshes.append(head)
        meshes.append(self._box("massive_orc_jaw", [0, -1.95, 17.9], [2.35, 0.88, 0.82]))
        meshes.append(self._box("orc_brow", [0, -1.85, 18.85], [2.2, 0.34, 0.32]))
        meshes.append(self._box("orc_flat_nose", [0, -2.05, 18.35], [0.5, 0.35, 0.58]))
        for x in [-0.52, 0.52]:
            tusk = trimesh.creation.cone(radius=0.20, height=1.25, sections=16)
            tusk.apply_transform(trimesh.transformations.rotation_matrix(math.pi / 2, [1, 0, 0]))
            tusk.apply_translation([x, -2.28, 18.0])
            meshes.append(tusk)
        for x, direction in [(-2.0, -1.0), (2.0, 1.0)]:
            ear = trimesh.creation.cone(radius=0.28, height=0.9, sections=16)
            ear.apply_transform(trimesh.transformations.rotation_matrix(direction * math.pi / 2, [0, 1, 0]))
            ear.apply_translation([x, -0.35, 18.55])
            meshes.append(ear)

        # Shoulder pads, armor plates, and fantasy cloth -- no sci-fi clutter.
        meshes.append(self._shoulder_pad([-4.0, -0.15, 15.8], mirror=False))
        meshes.append(self._shoulder_pad([4.0, -0.15, 15.8], mirror=True))
        meshes.append(self._box("orc_chest_plate", [0, -1.9, 13.9], [3.1, 0.42, 2.2]))
        meshes.append(self._box("orc_belt", [0, -1.75, 10.05], [5.0, 0.4, 0.55]))
        meshes.append(self._box("orc_lioncloth", [0, -1.95, 8.25], [1.35, 0.25, 2.8]))
        for x in [-1.7, 1.7]:
            meshes.append(self._box("orc_knee_plate", [x * 1.45, -0.75, 6.8], [1.35, 0.32, 0.9]))

        if has_axe:
            # Pose both arms toward a big axe on the right/front side.
            meshes.append(self._limb_between([-3.8, -0.1, 15.0], [-1.9, -1.35, 13.0], 0.68, sections=22))
            meshes.append(self._limb_between([-1.9, -1.35, 13.0], [1.8, -2.25, 13.3], 0.56, sections=22))
            meshes.append(self._limb_between([3.9, -0.1, 15.0], [4.6, -1.4, 13.4], 0.72, sections=22))
            meshes.append(self._limb_between([4.6, -1.4, 13.4], [4.3, -2.35, 11.2], 0.58, sections=22))
            for hand_pos in ([1.8, -2.25, 13.3], [4.3, -2.35, 11.2]):
                hand = trimesh.creation.icosphere(subdivisions=1, radius=0.62)
                hand.apply_scale([1.15, 0.8, 0.8])
                hand.apply_translation(hand_pos)
                meshes.append(hand)
        else:
            meshes.append(self._limb_between([-3.8, -0.1, 15.0], [-5.2, -0.5, 11.0], 0.72, sections=22))
            meshes.append(self._limb_between([-5.2, -0.5, 11.0], [-4.2, -0.9, 8.5], 0.58, sections=22))
            meshes.append(self._limb_between([3.8, -0.1, 15.0], [5.2, -0.5, 11.0], 0.72, sections=22))
            meshes.append(self._limb_between([5.2, -0.5, 11.0], [4.2, -0.9, 8.5], 0.58, sections=22))

        for part in plan.parts:
            if part.kind in {"weapon", "mounted_weapon", "weapon_arm"}:
                meshes.append(self._weapon_mesh(part))

        return meshes

    def _build_quadruped_beast(self, plan: MiniaturePlan, archetype: dict[str, Any]) -> list[trimesh.Trimesh]:
        meshes = [
            self._capsule([0, 0, 8], [6.5, 2.6, 2.4], rotate_y=math.pi / 2),
            self._capsule([-5.2, 0, 8.6], [2.3, 1.7, 1.8]),
            self._capsule([-3.5, -1.8, 4.5], [0.9, 0.8, 3.0]),
            self._capsule([-3.5, 1.8, 4.5], [0.9, 0.8, 3.0]),
            self._capsule([3.7, -1.7, 4.6], [0.9, 0.8, 2.8]),
            self._capsule([3.7, 1.7, 4.6], [0.9, 0.8, 2.8]),
            self._capsule([6.2, 0, 7.2], [3.2, 0.55, 0.55], rotate_y=math.pi / 2),
        ]

        # claws
        for x, y in [(-4.2, -2.3), (-4.2, 2.3), (4.4, -2.1), (4.4, 2.1)]:
            for i in range(3):
                claw = trimesh.creation.cone(radius=0.22, height=1.25, sections=16)
                claw.apply_transform(trimesh.transformations.rotation_matrix(math.pi / 2, [1, 0, 0]))
                claw.apply_translation([x + (i - 1) * 0.35, y, 3.0])
                meshes.append(claw)

        # back turret
        meshes += self._turret([0, 0, 11.0])

        return meshes

    def _surface_detail_parts(self, plan: MiniaturePlan, all_text: str) -> list[trimesh.Trimesh]:
        """Add deterministic miniature-scale readable details over the scaffold."""
        details: list[trimesh.Trimesh] = []
        wants_high_detail = any(
            k in all_text
            for k in [
                "detailed", "high detail", "highly detailed", "intricate", "ornate", "8k", "display quality",
                "wargaming", "tabletop", "miniature", "heroic scale", "resin-printable", "resin printable",
            ]
        )

        if any(k in all_text for k in ["beast", "quadruped", "lizard", "bear", "dragon"]):
            details += self._beast_surface_details()
        elif any(k in all_text for k in ["orc", "ork", "brute", "warboss"]):
            details += self._orc_surface_details(all_text)
            details += self._orc_intricate_details(all_text)
        else:
            details += self._humanoid_surface_details(all_text)

        if any(k in all_text for k in ["clockwork", "gear", "gears", "watch"]):
            details += self._clockwork_surface_details()

        if any(k in all_text for k in ["grimdark", "warhammer", "40k", "space marine", "power armor", "power armour", "bolter"]):
            details += self._grimdark_power_armor_details()

        if any(k in all_text for k in ["wargaming", "tabletop", "miniature", "heroic scale", "resin-printable", "resin printable"]):
            details += self._wargaming_miniature_parts(all_text)

        if wants_high_detail:
            details += self._high_detail_surface_parts(all_text)

        return details

    def _humanoid_surface_details(self, all_text: str) -> list[trimesh.Trimesh]:
        details: list[trimesh.Trimesh] = []

        # Face/helmet readability.
        details.append(self._box("brow_or_visor", [0, -1.38, 18.65], [1.65, 0.18, 0.22]))
        details.append(self._box("nose_ridge", [0, -1.46, 18.2], [0.28, 0.16, 0.55]))
        details.append(self._box("mouth_or_grill", [0, -1.42, 17.75], [1.05, 0.16, 0.20]))

        # Chest, belt, straps, and readable armor plates.
        details.append(self._box("raised_chest_emblem", [0, -1.72, 14.25], [1.05, 0.16, 1.05]))
        details.append(self._box("belt", [0, -1.45, 10.15], [4.2, 0.28, 0.42]))
        for x in [-1.65, 0.0, 1.65]:
            details.append(self._box("belt_pouch", [x, -1.72, 9.55], [0.75, 0.42, 0.75]))

        for x in [-1.25, 1.25]:
            strap = self._box("torso_strap", [x, -1.58, 13.75], [0.34, 0.22, 4.7])
            strap.apply_transform(trimesh.transformations.rotation_matrix(0.34 if x < 0 else -0.34, [0, 0, 1]))
            details.append(strap)

        # Knee/shin/forearm plates add intentional hard-surface structure.
        for x in [-1.4, 1.4]:
            details.append(self._box("knee_cap", [x, -0.98, 7.25], [1.15, 0.24, 0.9]))
            details.append(self._box("shin_ridge", [x, -1.02, 5.2], [0.45, 0.24, 2.0]))

        for x, angle in [(-3.35, 0.45), (3.35, -0.45)]:
            forearm = self._box("forearm_bracer", [x, -1.05, 12.05], [1.25, 0.30, 1.8])
            forearm.apply_transform(trimesh.transformations.rotation_matrix(angle, [0, 0, 1]))
            details.append(forearm)

        # Rivets read well on printed miniatures and make simple armor less random.
        for x in [-1.65, -0.55, 0.55, 1.65]:
            for z in [12.7, 15.25]:
                details.append(self._rivet([x, -1.82, z], radius=0.14))

        for x in [-2.55, -1.55, 1.55, 2.55]:
            for z in [16.15, 16.75]:
                details.append(self._rivet([x, -1.2, z], radius=0.12))

        if any(k in all_text for k in ["marine", "soldier", "sci-fi", "power armor", "rifle", "gun", "bolter"]):
            details += self._chain_links(start=[-2.25, -1.78, 10.4], step=[0.45, 0.0, -0.06], count=11, radius=0.22)
            details += self._purity_seal([1.75, -1.9, 13.6])
            details += self._purity_seal([-1.85, -1.9, 15.0])
            details += self._ammo_or_pouch_row()

        # Thin raised seams behave like visible panel lines after STL export.
        for z in [11.4, 12.35, 13.3, 14.25, 15.2]:
            details.append(self._box("horizontal_armor_seam", [0, -1.86, z], [3.45, 0.10, 0.09]))
        for x in [-2.25, 2.25]:
            details.append(self._box("side_armor_seam", [x, -1.5, 13.6], [0.10, 0.16, 4.2]))

        if any(k in all_text for k in ["cloak", "robe", "wizard", "knight", "fantasy"]):
            details += self._cloth_folds()

        if any(k in all_text for k in ["marine", "soldier", "armor", "sci-fi", "power armor"]):
            details += self._power_armor_panel_details()

        return details

    def _power_armor_panel_details(self) -> list[trimesh.Trimesh]:
        details: list[trimesh.Trimesh] = []
        for x in [-1.1, 0.0, 1.1]:
            details.append(self._box("helmet_vent", [x, -1.48, 17.55], [0.18, 0.16, 0.48]))
        for x in [-2.1, 2.1]:
            details.append(self._box("pauldron_trim", [x, -1.15, 16.45], [1.8, 0.22, 0.22]))
            details.append(self._box("pauldron_lower_trim", [x, -1.25, 15.55], [1.45, 0.18, 0.18]))
        for x in [-1.35, 1.35]:
            details.append(self._box("boot_toe_plate", [x, -1.05, 2.85], [1.25, 0.36, 0.42]))
            details.append(self._box("greave_panel_line", [x, -1.18, 5.9], [1.05, 0.12, 0.10]))

        for x in [-1.35, -0.45, 0.45, 1.35]:
            details.append(self._box("chest_vent_slit", [x, -1.93, 12.9], [0.46, 0.10, 0.08]))
        return details

    def _grimdark_power_armor_details(self) -> list[trimesh.Trimesh]:
        """Heroic sci-fi wargaming armor language without copying specific protected designs."""
        details: list[trimesh.Trimesh] = []

        # Oversized shoulder rims and blank heraldry plates are the strongest tabletop silhouette cue.
        for side in [-1.0, 1.0]:
            shoulder_x = side * 3.72
            for z, width in [(16.55, 2.55), (16.05, 2.75), (15.55, 2.35)]:
                trim = self._box("heroic_pauldron_layered_trim", [shoulder_x, -1.72, z], [width, 0.18, 0.16])
                trim.apply_transform(trimesh.transformations.rotation_matrix(side * 0.08, [0, 0, 1]))
                details.append(trim)
            badge = trimesh.creation.cylinder(radius=0.46, height=0.12, sections=24)
            badge.apply_transform(trimesh.transformations.rotation_matrix(math.pi / 2, [1, 0, 0]))
            badge.apply_translation([shoulder_x + side * 0.28, -1.92, 16.22])
            details.append(badge)
            for z in [15.45, 15.85, 16.25, 16.65]:
                details.append(self._rivet([shoulder_x - side * 0.95, -1.95, z], radius=0.11))
                details.append(self._rivet([shoulder_x + side * 0.95, -1.95, z], radius=0.11))

        # Helmet readability: brow, cheek filters, mouth grille and top crest.
        details.append(self._box("deep_helmet_brow", [0.0, -1.64, 18.54], [1.85, 0.16, 0.18]))
        details.append(self._box("vertical_helmet_crest", [0.0, -1.58, 19.18], [0.26, 0.16, 0.82]))
        for x in [-0.52, 0.52]:
            details.append(self._box("helmet_cheek_filter", [x, -1.70, 17.86], [0.34, 0.15, 0.42]))
        for x in [-0.42, -0.14, 0.14, 0.42]:
            details.append(self._box("helmet_grill_slit", [x, -1.82, 17.58], [0.08, 0.08, 0.38]))

        # Chest aquila-like abstract wing panels, armor lamellae, vents, cables, seals.
        for side in [-1.0, 1.0]:
            for i, z in enumerate([14.65, 14.35, 14.05, 13.75]):
                feather = self._box("abstract_winged_chest_plate", [side * (0.45 + i * 0.32), -2.16, z], [0.55, 0.10, 0.11])
                feather.apply_transform(trimesh.transformations.rotation_matrix(-side * 0.28, [0, 0, 1]))
                details.append(feather)
        for z in [12.05, 12.55, 13.05, 13.55]:
            details.append(self._box("stacked_ab_armor_lamella", [0.0, -2.12, z], [2.8, 0.11, 0.10]))
        details += self._purity_seal([-1.55, -2.20, 14.25])
        details += self._purity_seal([1.45, -2.20, 12.85])
        details += self._chain_links(start=[-2.0, -2.22, 10.55], step=[0.42, 0.0, 0.02], count=10, radius=0.15)

        # Power-pack vents/cables on the back and leg armor that reads from tabletop distance.
        for x in [-1.25, 1.25]:
            vent = trimesh.creation.cylinder(radius=0.36, height=1.35, sections=20)
            vent.apply_translation([x, 2.65, 16.45])
            details.append(vent)
            for z in [15.95, 16.28, 16.61]:
                details.append(self._box("power_pack_vent_slit", [x, 3.34, z], [0.46, 0.08, 0.08]))
        for x in [-1.45, 1.45]:
            details.append(self._box("oversized_knee_plate_trim", [x, -1.55, 7.28], [1.25, 0.12, 0.12]))
            details.append(self._box("heavy_boot_toe_cap", [x, -1.58, 2.78], [1.45, 0.20, 0.26]))
            for z in [4.75, 5.35, 5.95, 6.55]:
                details.append(self._box("greave_panel_groove", [x, -1.62, z], [0.92, 0.08, 0.08]))

        # Chunkier rifle silhouette with scope, muzzle brake, magazine, and barrel vents.
        details.append(self._box("chunky_sci_fi_rifle_receiver", [0.8, -2.95, 13.25], [4.9, 0.38, 0.86]))
        details.append(self._box("rifle_box_magazine", [0.1, -3.03, 12.55], [0.78, 0.32, 1.05]))
        details.append(self._box("rifle_top_scope", [0.95, -3.08, 13.98], [1.5, 0.26, 0.24]))
        for x in [2.2, 2.65, 3.1, 3.55, 4.0]:
            details.append(self._box("rifle_barrel_vent_cut", [x, -3.18, 13.45], [0.22, 0.08, 0.08]))
        details.append(self._box("rifle_muzzle_brake", [4.72, -3.0, 13.42], [0.36, 0.46, 0.46]))

        return details

    def _high_detail_surface_parts(self, all_text: str) -> list[trimesh.Trimesh]:
        """Add extra named details when the prompt explicitly asks for a detailed miniature."""
        details: list[trimesh.Trimesh] = []

        # Dense but printable face details: eyes, cheek plates/scars, teeth/grill slits.
        for x in [-0.42, 0.42]:
            details.append(self._rivet([x, -1.66, 18.42], radius=0.075))
            details.append(self._box("tiny_cheek_mark", [x * 1.55, -1.68, 18.05], [0.32, 0.07, 0.055]))
        for x in [-0.45, -0.15, 0.15, 0.45]:
            details.append(self._box("tiny_mouth_tooth_or_grill", [x, -1.72, 17.62], [0.08, 0.07, 0.28]))

        # Layered torso trim, small pouches, stitches, and rows of readable rivets.
        for z in [11.55, 12.15, 12.75, 13.35, 13.95, 14.55, 15.15]:
            details.append(self._box("fine_torso_panel_line", [0, -2.05, z], [3.75, 0.075, 0.055]))
        for x in [-1.85, -1.35, -0.85, -0.35, 0.35, 0.85, 1.35, 1.85]:
            details.append(self._rivet([x, -2.12, 15.05], radius=0.085))
            details.append(self._rivet([x, -2.10, 12.0], radius=0.085))
        for x in [-2.1, -1.45, -0.8, 0.8, 1.45, 2.1]:
            pouch = self._box("fine_belt_pouch", [x, -2.08, 9.82], [0.32, 0.18, 0.42])
            details.append(pouch)
            details.append(self._box("fine_pouch_flap", [x, -2.20, 10.05], [0.26, 0.06, 0.07]))

        # Shoulder/knee/boot edging gives hard-surface models silhouette-level detail.
        for side in [-1.0, 1.0]:
            for z in [15.65, 16.05, 16.45, 16.85]:
                details.append(self._rivet([side * 3.75, -1.82, z], radius=0.095))
                details.append(self._rivet([side * 4.35, -1.78, z], radius=0.095))
            details.append(self._box("fine_pauldron_edge_cut", [side * 3.95, -1.92, 15.25], [1.55, 0.075, 0.06]))
        for x in [-1.45, 1.45]:
            for z in [4.65, 5.25, 5.85, 6.45, 7.05]:
                details.append(self._box("fine_greave_strap", [x, -1.54, z], [0.95, 0.08, 0.055]))
            details.append(self._box("fine_boot_toe_split", [x, -1.62, 2.55], [0.08, 0.08, 0.48]))

        if any(k in all_text for k in ["cloak", "robe", "cloth", "wizard", "fantasy", "orc", "ork"]):
            for x in [-1.05, -0.65, -0.25, 0.25, 0.65, 1.05]:
                fold = self._box("fine_cloth_fold", [x, -2.16, 7.75], [0.10, 0.07, 1.35])
                fold.apply_transform(trimesh.transformations.rotation_matrix(x * 0.18, [0, 0, 1]))
                details.append(fold)

        if any(k in all_text for k in ["gun", "rifle", "bolter", "cannon", "turret"]):
            for x in [-1.8, -1.25, -0.7, 0.7, 1.25, 1.8]:
                details.append(self._box("fine_weapon_casing_screw", [x, -2.64, 13.55], [0.14, 0.08, 0.14]))
            for x in [2.5, 3.1, 3.7, 4.3]:
                details.append(self._box("fine_barrel_vent", [x, -2.66, 13.38], [0.22, 0.07, 0.08]))

        if any(k in all_text for k in ["base", "miniature", "display", "diorama"]):
            for i in range(14):
                angle = i * math.tau / 14.0
                radius = 7.2 + 1.2 * (i % 3)
                details.append(self._rivet([math.cos(angle) * radius, math.sin(angle) * radius, 2.42], radius=0.11))

        return details

    def _wargaming_miniature_parts(self, all_text: str) -> list[trimesh.Trimesh]:
        """Recognizable tabletop-miniature greebles that make the model read as a wargaming sculpt."""
        details: list[trimesh.Trimesh] = []

        # Face and head silhouette details.
        details.append(self._box("deep_eye_shadow_left", [-0.44, -1.72, 18.48], [0.34, 0.10, 0.10]))
        details.append(self._box("deep_eye_shadow_right", [0.44, -1.72, 18.48], [0.34, 0.10, 0.10]))
        details.append(self._box("nose_bridge_or_helmet_ridge", [0.0, -1.76, 18.16], [0.18, 0.10, 0.62]))
        for x in [-0.52, 0.52]:
            cheek = self._box("cheek_plate_or_scar", [x, -1.79, 18.02], [0.42, 0.08, 0.08])
            cheek.apply_transform(trimesh.transformations.rotation_matrix(-0.22 if x < 0 else 0.22, [0, 0, 1]))
            details.append(cheek)

        # Heroic-scale armor plates: raised trim reads better than texture-only detail in STL.
        for x in [-1.35, 0.0, 1.35]:
            details.append(self._box("raised_abdominal_armor_plate", [x, -2.03, 11.55], [0.86, 0.11, 0.54]))
        for z in [12.45, 13.05, 13.65, 14.25, 14.85]:
            details.append(self._box("layered_breastplate_lamella", [0.0, -2.08, z], [3.65, 0.10, 0.10]))
        for side in [-1.0, 1.0]:
            details.append(self._box("large_pauldron_front_rim", [side * 3.95, -1.86, 16.18], [1.85, 0.13, 0.15]))
            details.append(self._box("large_pauldron_lower_rim", [side * 3.95, -1.78, 15.42], [1.55, 0.12, 0.13]))
            for z in [15.62, 16.0, 16.38, 16.76]:
                details.append(self._rivet([side * 3.35, -1.93, z], radius=0.105))
                details.append(self._rivet([side * 4.55, -1.93, z], radius=0.105))

        # Backpack, vents, and cables for sci-fi/armored soldiers; harmless extra mass for generic warriors.
        if any(k in all_text for k in ["marine", "soldier", "armor", "armored", "sci-fi", "rifle", "gun", "bolter", "power"]):
            details.append(self._box("miniature_power_pack_core", [0.0, 2.05, 15.0], [2.7, 0.72, 2.9]))
            for x in [-0.95, 0.95]:
                vent = trimesh.creation.cylinder(radius=0.28, height=1.0, sections=18)
                vent.apply_translation([x, 2.52, 16.25])
                details.append(vent)
            details += self._chain_links(start=[-1.9, -2.12, 10.2], step=[0.43, 0.0, 0.035], count=10, radius=0.15)
            details += self._purity_seal([1.65, -2.16, 14.15])

        # Fantasy/wargaming trophies and readable cloth/straps.
        if any(k in all_text for k in ["orc", "ork", "warboss", "fantasy", "knight", "chaos", "demon", "wizard"]):
            details += self._chain_links(start=[-2.25, -2.08, 10.45], step=[0.42, 0.0, -0.04], count=12, radius=0.16)
            details += self._trophy_skulls()
            for x in [-1.2, -0.6, 0.0, 0.6, 1.2]:
                strip = self._box("ragged_cloth_or_leather_strip", [x, -2.18, 8.0], [0.22, 0.09, 1.55])
                strip.apply_transform(trimesh.transformations.rotation_matrix(x * 0.16, [0, 0, 1]))
                details.append(strip)

        # Hands and fingers around the front/weapon area keep arms from reading as tubes.
        for hand_x, hand_z in [(-3.4, 12.0), (3.4, 12.0), (1.9, 13.25), (4.25, 11.25)]:
            palm = trimesh.creation.icosphere(subdivisions=1, radius=0.36)
            palm.apply_scale([1.15, 0.7, 0.85])
            palm.apply_translation([hand_x, -1.9, hand_z])
            details.append(palm)
            for finger in [-0.18, 0.0, 0.18]:
                details.append(self._box("blocky_miniature_finger", [hand_x + finger, -2.18, hand_z - 0.25], [0.10, 0.12, 0.42]))

        # Scenic base: rubble, cracked tiles, shell casings/teeth shapes.
        for i in range(18):
            angle = i * math.tau / 18.0
            radius = 5.2 + (i % 5) * 1.45
            x = math.cos(angle) * radius
            y = math.sin(angle) * radius
            if i % 3 == 0:
                rock = trimesh.creation.icosphere(subdivisions=1, radius=0.28 + 0.05 * (i % 4))
                rock.apply_scale([1.45, 0.9, 0.48])
                rock.apply_translation([x, y, 2.42])
                details.append(rock)
            else:
                tile = self._box("cracked_base_tile_or_debris", [x, y, 2.35], [0.72, 0.16, 0.09])
                tile.apply_transform(trimesh.transformations.rotation_matrix(angle, [0, 0, 1]))
                details.append(tile)

        return details

    def _trophy_skulls(self) -> list[trimesh.Trimesh]:
        trophies: list[trimesh.Trimesh] = []
        for x, z in [(-1.7, 9.45), (1.7, 9.45)]:
            skull = trimesh.creation.icosphere(subdivisions=1, radius=0.32)
            skull.apply_scale([0.9, 0.7, 1.05])
            skull.apply_translation([x, -2.16, z])
            jaw = self._box("tiny_trophy_skull_jaw", [x, -2.24, z - 0.32], [0.42, 0.08, 0.20])
            trophies.extend([skull, jaw])
            for eye_x in [-0.10, 0.10]:
                trophies.append(self._box("tiny_trophy_skull_eye_socket", [x + eye_x, -2.40, z + 0.04], [0.07, 0.06, 0.07]))
        return trophies

    def _cloth_folds(self) -> list[trimesh.Trimesh]:
        folds: list[trimesh.Trimesh] = []
        for x in [-1.2, -0.4, 0.4, 1.2]:
            fold = self._box("cloth_fold", [x, 1.25, 9.2], [0.25, 0.32, 5.2])
            fold.apply_transform(trimesh.transformations.rotation_matrix(x * 0.12, [0, 0, 1]))
            folds.append(fold)
        return folds

    def _clockwork_surface_details(self) -> list[trimesh.Trimesh]:
        details: list[trimesh.Trimesh] = []
        for x, z, radius in [(-1.1, 14.3, 0.55), (1.1, 14.3, 0.55), (0.0, 15.25, 0.42)]:
            gear = trimesh.creation.cylinder(radius=radius, height=0.16, sections=20)
            gear.apply_transform(trimesh.transformations.rotation_matrix(math.pi / 2, [1, 0, 0]))
            gear.apply_translation([x, -1.92, z])
            details.append(gear)
            for tooth in range(8):
                angle = tooth * math.tau / 8
                tooth_box = self._box("gear_tooth", [x + math.cos(angle) * radius * 1.1, -1.98, z + math.sin(angle) * radius * 1.1], [0.16, 0.18, 0.22])
                details.append(tooth_box)
        details += self._chain_links(start=[-1.8, -2.04, 15.8], step=[0.42, 0.0, 0.0], count=9, radius=0.18)
        return details

    def _orc_surface_details(self, all_text: str) -> list[trimesh.Trimesh]:
        details: list[trimesh.Trimesh] = []
        details.append(self._box("heavy_brow_ridge", [0, -1.63, 18.65], [1.9, 0.24, 0.28]))
        details.append(self._box("flat_orc_nose", [0, -1.75, 18.15], [0.42, 0.26, 0.58]))
        details.append(self._box("square_lower_jaw", [0, -1.82, 17.65], [1.8, 0.34, 0.38]))
        for x in [-0.45, 0.45]:
            tusk = trimesh.creation.cone(radius=0.18, height=1.0, sections=14)
            tusk.apply_transform(trimesh.transformations.rotation_matrix(math.pi / 2, [1, 0, 0]))
            tusk.apply_translation([x, -1.95, 17.95])
            details.append(tusk)
        for x in [-0.72, 0.72]:
            ear = trimesh.creation.cone(radius=0.22, height=0.8, sections=14)
            ear.apply_transform(trimesh.transformations.rotation_matrix(math.pi / 2, [0, 1, 0]))
            ear.apply_translation([x * 2.0, -0.15, 18.45])
            details.append(ear)
        details.append(self._box("leather_belt", [0, -1.72, 10.0], [4.7, 0.36, 0.45]))
        details.append(self._box("ragged_lioncloth_center", [0, -1.85, 8.35], [1.15, 0.22, 2.6]))
        for x in [-0.85, 0.85]:
            flap = self._box("ragged_lioncloth_side", [x, -1.82, 8.05], [0.65, 0.20, 2.0])
            flap.apply_transform(trimesh.transformations.rotation_matrix(0.12 if x < 0 else -0.12, [0, 0, 1]))
            details.append(flap)
        for x in [-1.65, 1.65]:
            details.append(self._box("orc_knee_plate", [x, -1.05, 7.25], [1.3, 0.28, 0.75]))
            details.append(self._box("orc_boot_toe", [x, -1.12, 2.75], [1.35, 0.42, 0.42]))
        for x, angle in [(-3.65, 0.45), (3.65, -0.45)]:
            details.append(self._box("orc_wrist_wrap", [x, -1.12, 12.3], [1.35, 0.24, 0.45]))
            bracer = self._box("orc_bracer", [x, -1.05, 11.55], [1.2, 0.30, 1.45])
            bracer.apply_transform(trimesh.transformations.rotation_matrix(angle, [0, 0, 1]))
            details.append(bracer)
        for x in [-2.5, 2.5]:
            for z in [13.2, 14.4, 15.6]:
                spike = trimesh.creation.cone(radius=0.22, height=0.9, sections=14)
                spike.apply_transform(trimesh.transformations.rotation_matrix(math.pi / 2, [1, 0, 0]))
                spike.apply_translation([x, -1.35, z])
                details.append(spike)
        for x in [-1.8, -0.9, 0.0, 0.9, 1.8]:
            details.append(self._box("jagged_armor_patch", [x, -1.96, 12.15], [0.7, 0.14, 0.35]))
        for x in [-1.4, -0.55, 0.55, 1.4]:
            details.append(self._rivet([x, -2.02, 10.05], radius=0.16))
        if "axe" in all_text:
            details += self._chain_links(start=[-1.45, -2.0, 13.35], step=[0.38, 0.0, 0.05], count=8, radius=0.16)
        return details

    def _orc_intricate_details(self, all_text: str) -> list[trimesh.Trimesh]:
        """Prompt-specific high-detail fantasy orc geometry.

        These are intentionally recognizable sculpt details (stitches, rivets,
        scars, trophies, trim) rather than random noise or unrelated objects.
        """
        details: list[trimesh.Trimesh] = []

        # Layered armor trim and rivets on chest plate.
        details.append(self._box("orc_chest_top_trim", [0, -2.14, 14.95], [3.45, 0.16, 0.16]))
        details.append(self._box("orc_chest_bottom_trim", [0, -2.14, 12.85], [3.25, 0.16, 0.16]))
        for x in [-1.45, -0.72, 0.0, 0.72, 1.45]:
            details.append(self._rivet([x, -2.24, 14.7], radius=0.13))
            details.append(self._rivet([x, -2.24, 13.1], radius=0.13))

        # Belt stitching and hanging teeth trophies.
        for x in [-2.15, -1.65, -1.15, -0.65, -0.15, 0.35, 0.85, 1.35, 1.85, 2.35]:
            stitch = self._box("belt_stitch", [x, -2.08, 10.18], [0.12, 0.10, 0.42])
            stitch.apply_transform(trimesh.transformations.rotation_matrix(0.18, [0, 0, 1]))
            details.append(stitch)
        for x in [-1.3, 0.0, 1.3]:
            tooth = trimesh.creation.cone(radius=0.13, height=0.72, sections=12)
            tooth.apply_transform(trimesh.transformations.rotation_matrix(math.pi, [1, 0, 0]))
            tooth.apply_translation([x, -2.10, 9.35])
            details.append(tooth)
            details.append(self._rivet([x, -2.08, 9.85], radius=0.12))

        # Ragged cloth hem with individual torn strips.
        for x, z, length in [(-0.95, 7.15, 0.85), (-0.45, 6.95, 1.15), (0.15, 7.1, 0.95), (0.72, 7.0, 1.05)]:
            strip = self._box("ragged_cloth_torn_strip", [x, -2.08, z], [0.28, 0.10, length])
            strip.apply_transform(trimesh.transformations.rotation_matrix(x * 0.12, [0, 0, 1]))
            details.append(strip)

        # Readable skin scars on chest and arms.
        for start, angle in [((-1.0, -2.18, 14.2), -0.42), ((0.85, -2.18, 13.75), 0.38), ((-3.65, -1.55, 12.35), 0.52), ((3.65, -1.55, 12.0), -0.52)]:
            x, y, z = start
            scar = self._box("raised_scar", [x, y, z], [1.05, 0.08, 0.09])
            scar.apply_transform(trimesh.transformations.rotation_matrix(angle, [0, 0, 1]))
            details.append(scar)
            for offset in [-0.32, 0.0, 0.32]:
                nick = self._box("scar_cross_cut", [x + offset, y - 0.03, z + offset * 0.15], [0.08, 0.07, 0.36])
                nick.apply_transform(trimesh.transformations.rotation_matrix(angle + math.pi / 2, [0, 0, 1]))
                details.append(nick)

        # Shoulder pad rims and spike rivet rows.
        for side in [-1.0, 1.0]:
            details.append(self._box("shoulder_front_trim", [side * 3.95, -1.62, 16.1], [1.55, 0.16, 0.16]))
            details.append(self._box("shoulder_lower_trim", [side * 3.95, -1.55, 15.35], [1.30, 0.14, 0.14]))
            for z in [15.55, 16.05, 16.55]:
                details.append(self._rivet([side * 3.55, -1.72, z], radius=0.12))
                details.append(self._rivet([side * 4.35, -1.72, z], radius=0.12))

        # Boot straps and knee-plate edge chips.
        for x in [-2.45, 2.45]:
            details.append(self._box("boot_ankle_strap", [x, -1.70, 3.15], [1.75, 0.14, 0.18]))
            details.append(self._box("boot_toe_seam", [x, -1.75, 2.65], [1.35, 0.13, 0.10]))
            for dx in [-0.45, 0.0, 0.45]:
                details.append(self._rivet([x + dx, -1.86, 3.18], radius=0.10))
        for x in [-2.45, 2.45]:
            details.append(self._box("knee_plate_top_trim", [x, -1.10, 7.25], [1.25, 0.12, 0.12]))
            details.append(self._box("knee_plate_gouge", [x + 0.25, -1.18, 6.75], [0.45, 0.10, 0.10]))

        if "axe" in all_text:
            details += self._axe_grip_and_blade_details()

        return details

    def _axe_grip_and_blade_details(self) -> list[trimesh.Trimesh]:
        details: list[trimesh.Trimesh] = []
        # Match the brutal axe coordinates from _weapon_mesh.
        x = 4.5
        y = -2.8
        for z in [9.4, 10.0, 10.6, 11.2, 11.8, 12.4, 13.0, 13.6]:
            ring = trimesh.creation.torus(major_radius=0.43, minor_radius=0.045, major_sections=18, minor_sections=6)
            ring.apply_translation([x, y, z])
            details.append(ring)
        for z in [17.7, 18.25, 18.8]:
            details.append(self._box("axe_socket_rivet_left", [x - 0.42, y - 0.36, z], [0.18, 0.12, 0.18]))
            details.append(self._box("axe_socket_rivet_right", [x + 0.42, y - 0.36, z], [0.18, 0.12, 0.18]))
        for bx, bz, angle in [(2.65, 18.9, -0.22), (6.35, 18.15, 0.22), (2.85, 17.35, 0.18), (6.15, 19.35, -0.18)]:
            notch = self._box("axe_edge_chip", [bx, y - 0.55, bz], [0.55, 0.12, 0.16])
            notch.apply_transform(trimesh.transformations.rotation_matrix(angle, [0, 0, 1]))
            details.append(notch)
        for bx in [3.65, 4.5, 5.35]:
            rune = self._box("axe_blade_rune", [bx, y - 0.58, 18.6], [0.12, 0.10, 0.55])
            rune.apply_transform(trimesh.transformations.rotation_matrix(0.35 if bx < 4.5 else -0.35, [0, 0, 1]))
            details.append(rune)
        return details

    def _beast_surface_details(self) -> list[trimesh.Trimesh]:
        details: list[trimesh.Trimesh] = []
        # Spine plates and tail spikes give the animal a readable silhouette.
        for x, z in [(-3.5, 10.2), (-2.0, 10.8), (-0.5, 11.0), (1.0, 10.7), (2.5, 10.1), (4.5, 8.8), (6.0, 7.9)]:
            spike = trimesh.creation.cone(radius=0.34, height=1.15, sections=16)
            spike.apply_translation([x, 0, z])
            details.append(spike)

        # Scale/skin plates on each side.
        for x in [-3.0, -1.5, 0.0, 1.5, 3.0]:
            for y in [-1.55, 1.55]:
                scale = self._box("side_scale", [x, y, 8.6], [0.85, 0.20, 0.42])
                details.append(scale)

        # Eyes and snout ridge.
        for y in [-0.55, 0.55]:
            details.append(self._rivet([-6.55, y, 9.2], radius=0.16))
        details.append(self._box("snout_ridge", [-6.4, 0, 8.65], [0.45, 1.2, 0.24]))

        for x in [-5.7, -5.2, -4.7]:
            for y in [-0.55, 0.55]:
                tooth = trimesh.creation.cone(radius=0.11, height=0.55, sections=10)
                tooth.apply_transform(trimesh.transformations.rotation_matrix(math.pi / 2, [0, 1, 0]))
                tooth.apply_translation([x, y, 8.05])
                details.append(tooth)
        return details

    def _ammo_or_pouch_row(self) -> list[trimesh.Trimesh]:
        details: list[trimesh.Trimesh] = []
        for i, x in enumerate([-1.6, -1.05, -0.5, 0.5, 1.05, 1.6]):
            pouch = self._box("tiny_pouch_or_ammo", [x, -1.88, 10.65 + (0.08 if i % 2 else 0.0)], [0.38, 0.28, 0.55])
            details.append(pouch)
        return details

    def _purity_seal(self, pos: list[float]) -> list[trimesh.Trimesh]:
        x, y, z = pos
        wax = trimesh.creation.cylinder(radius=0.28, height=0.12, sections=18)
        wax.apply_transform(trimesh.transformations.rotation_matrix(math.pi / 2, [1, 0, 0]))
        wax.apply_translation([x, y, z])

        ribbon_l = self._box("seal_ribbon_l", [x - 0.13, y - 0.02, z - 0.55], [0.18, 0.08, 0.9])
        ribbon_l.apply_transform(trimesh.transformations.rotation_matrix(0.12, [0, 0, 1]))
        ribbon_r = self._box("seal_ribbon_r", [x + 0.13, y - 0.02, z - 0.55], [0.18, 0.08, 0.9])
        ribbon_r.apply_transform(trimesh.transformations.rotation_matrix(-0.12, [0, 0, 1]))

        line_1 = self._box("seal_script_line", [x, y - 0.08, z - 0.45], [0.32, 0.06, 0.035])
        line_2 = self._box("seal_script_line", [x, y - 0.08, z - 0.68], [0.28, 0.06, 0.035])
        return [wax, ribbon_l, ribbon_r, line_1, line_2]

    def _chain_links(self, start: list[float], step: list[float], count: int, radius: float = 0.2) -> list[trimesh.Trimesh]:
        links: list[trimesh.Trimesh] = []
        for i in range(count):
            link = trimesh.creation.torus(major_radius=radius, minor_radius=radius * 0.22, major_sections=18, minor_sections=6)
            link.apply_scale([1.35, 0.72, 0.55])
            link.apply_transform(trimesh.transformations.rotation_matrix(math.pi / 2, [1, 0, 0]))
            if i % 2:
                link.apply_transform(trimesh.transformations.rotation_matrix(math.pi / 2, [0, 0, 1]))
            link.apply_translation([
                start[0] + step[0] * i,
                start[1] + step[1] * i,
                start[2] + step[2] * i,
            ])
            links.append(link)
        return links

    def _subdivide_and_sculpt_surface(self, mesh: trimesh.Trimesh, all_text: str) -> trimesh.Trimesh:
        """Add shallow geometric relief so the exported STL has real surface detail."""
        try:
            detailed = mesh.copy()
            wants_high_detail = any(
                k in all_text
                for k in [
                    "detailed", "high detail", "highly detailed", "intricate", "ornate", "8k", "display quality",
                    "wargaming", "tabletop", "miniature", "heroic scale", "resin-printable", "resin printable",
                ]
            )
            wants_8k_definition = any(
                k in all_text
                for k in ["8k", "8 k", "ultra detail", "ultra detailed", "high definition", "maximum detail", "max detail"]
            )
            target_faces = 500000 if wants_8k_definition else 120000 if wants_high_detail else 42000
            face_ceiling = 900000 if wants_8k_definition else 220000 if wants_high_detail else 90000
            while len(detailed.faces) < target_faces:
                vertices, faces = trimesh.remesh.subdivide(detailed.vertices, detailed.faces)
                detailed = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
                if len(detailed.faces) >= target_faces or len(detailed.faces) > face_ceiling:
                    break

            detailed.fix_normals()
            normals = np.asarray(detailed.vertex_normals, dtype=np.float64)
            vertices = np.asarray(detailed.vertices, dtype=np.float64)
            x = vertices[:, 0]
            y = vertices[:, 1]
            z = vertices[:, 2]

            character_mask = z > 2.45
            base_mask = z <= 2.75
            front_mask = y < 0.25

            # Coherent miniature relief: only add shallow, category-specific grooves.
            # Avoid generic sine noise, which reads like random shapes on small minis.
            relief = np.zeros(len(vertices), dtype=np.float64)

            if any(k in all_text for k in ["beast", "lizard", "bear", "dragon", "organic"]):
                scales = (np.sin(x * 5.8) > 0.45) & (np.sin(z * 4.9 + y * 2.0) > 0.20)
                relief += np.where(character_mask & scales, 0.040, 0.0)
                if wants_high_detail:
                    hide_wrinkles = (np.abs(np.sin(x * 9.5 + z * 1.4)) > 0.965) & (np.abs(y) > 0.6)
                    relief += np.where(character_mask & hide_wrinkles, -0.030, 0.0)
            elif any(k in all_text for k in ["orc", "ork", "brute", "warboss"]):
                torso_zone = character_mask & (z > 10.5) & (z < 16.0) & front_mask & (np.abs(x) < 3.8)
                muscle_grooves = np.abs(np.sin((x + 0.25) * 2.4))
                relief += np.where(torso_zone & (muscle_grooves > 0.94), -0.032, 0.0)
                scar_zone = character_mask & (z > 12.0) & (z < 15.5) & front_mask
                scars = np.abs(np.sin((x * 1.7 + z * 0.55)))
                relief += np.where(scar_zone & (scars > 0.982), 0.025, 0.0)
                if wants_high_detail:
                    stitch_zone = character_mask & (z > 7.5) & (z < 15.8) & front_mask
                    stitches = np.abs(np.sin(x * 7.0 + z * 1.3))
                    relief += np.where(stitch_zone & (stitches > 0.975), 0.026, 0.0)
            else:
                # Armor/cloth gets panel-like bands and shallow engraved grooves.
                torso_zone = character_mask & (z > 8.8) & (z < 17.2) & front_mask
                band = np.abs(np.sin(z * 6.4))
                relief += np.where(torso_zone & (band > 0.93), -0.038, 0.0)

                vertical_panel = np.abs(np.sin((x + 0.15) * 5.2))
                relief += np.where(torso_zone & (vertical_panel > 0.965), -0.030, 0.0)

                limb_zone = character_mask & (z > 3.2) & (z < 13.0) & (np.abs(x) > 0.8)
                relief += np.where(limb_zone & (np.abs(np.sin(z * 7.2)) > 0.94), -0.026, 0.0)
                if wants_high_detail:
                    micro_panel = np.abs(np.sin(x * 8.3 + z * 2.1))
                    relief += np.where(torso_zone & (micro_panel > 0.973), -0.024, 0.0)

            # Base dirt is different from body detail and keeps the stand from looking plain.
            relief += np.where(base_mask, 0.018 * np.sin(x * 6.0 + y * 2.0) * np.cos(y * 5.2), 0.0)
            if wants_high_detail:
                relief += np.where(base_mask, 0.012 * np.sin(x * 13.0 - y * 7.0), 0.0)

            relief_limit = 0.070 if wants_high_detail else 0.050
            relief = np.where(character_mask | base_mask, np.clip(relief, -relief_limit, relief_limit), 0.0)
            detailed.vertices = vertices + normals * relief[:, None]

            try:
                detailed.remove_degenerate_faces()
                detailed.remove_unreferenced_vertices()
                detailed.fix_normals()
            except Exception:
                pass

            return detailed
        except Exception as exc:
            print(f"Surface sculpt detail warning: {exc}")
            return mesh

    def _weapon_mesh(self, part: PartSpec) -> trimesh.Trimesh:
        name = part.name.lower()
        shape_text = f"{part.name} {part.shape} {' '.join(part.detail_tags)}".lower()

        if "staff" in name or "staff" in shape_text:
            base = np.asarray(part.position, dtype=float)
            shaft = trimesh.creation.cylinder(radius=0.16, sections=18, segment=np.array([base + [-0.4, 0, -4.8], base + [0.4, 0, 4.8]], dtype=float))
            orb = trimesh.creation.icosphere(subdivisions=2, radius=0.55)
            orb.apply_translation(base + np.array([0.55, -0.05, 5.1]))
            rings = []
            for z in [-2.2, -0.6, 1.0, 3.7]:
                ring = trimesh.creation.torus(major_radius=0.23, minor_radius=0.035, major_sections=18, minor_sections=6)
                ring.apply_translation(base + np.array([0.08, 0.0, z]))
                rings.append(ring)
            return trimesh.util.concatenate([shaft, orb, *rings])

        if "sword" in name or "sword" in shape_text:
            base = np.asarray(part.position, dtype=float)
            blade = self._box("sword_blade", base.tolist(), [0.34, 0.16, 5.4])
            guard = self._box("sword_crossguard", (base + np.array([0, -0.05, -2.3])).tolist(), [1.55, 0.22, 0.22])
            grip = self._limb_between((base + np.array([0, 0, -3.9])).tolist(), (base + np.array([0, 0, -2.45])).tolist(), 0.16, sections=14)
            pommel = self._rivet((base + np.array([0, 0, -4.25])).tolist(), radius=0.22)
            return trimesh.util.concatenate([blade, guard, grip, pommel])

        if "bow" in name or "bow" in shape_text:
            base = np.asarray(part.position, dtype=float)
            upper = self._limb_between((base + np.array([0.0, 0.0, 0.0])).tolist(), (base + np.array([0.55, 0.0, 3.5])).tolist(), 0.11, sections=12)
            lower = self._limb_between((base + np.array([0.0, 0.0, 0.0])).tolist(), (base + np.array([0.55, 0.0, -3.5])).tolist(), 0.11, sections=12)
            string = self._limb_between((base + np.array([0.58, -0.18, 3.35])).tolist(), (base + np.array([0.58, -0.18, -3.35])).tolist(), 0.035, sections=8)
            grip = self._box("bow_grip", base.tolist(), [0.35, 0.18, 0.8])
            return trimesh.util.concatenate([upper, lower, string, grip])

        if "cannon" in name or "arm_cannon" in name:
            base = np.asarray(part.position, dtype=float)
            casing = self._box("arm_cannon_casing", base.tolist(), [3.2, 0.9, 1.25])
            barrel = trimesh.creation.cylinder(radius=0.34, height=2.5, sections=18)
            barrel.apply_transform(trimesh.transformations.rotation_matrix(math.pi / 2, [0, 1, 0]))
            barrel.apply_translation(base + np.array([1.85, -0.02, 0.08]))
            vents = [self._box("arm_cannon_heat_vent", (base + np.array([x, -0.55, 0.48])).tolist(), [0.24, 0.08, 0.10]) for x in [-0.8, -0.35, 0.1, 0.55]]
            return trimesh.util.concatenate([casing, barrel, *vents])

        if "axe" in name:
            if any(k in shape_text for k in ["gear", "clockwork", "circular"]):
                shaft = trimesh.creation.cylinder(radius=0.35, height=8.0, sections=24)
                shaft.apply_transform(trimesh.transformations.rotation_matrix(math.pi / 2, [1, 0, 0]))

                head = trimesh.creation.cylinder(radius=2.1, height=0.45, sections=32)
                head.apply_translation([0, -4.1, 0])

                teeth = []
                for i in range(12):
                    angle = i * math.tau / 12
                    tooth = self._box("axe_gear_tooth", [math.cos(angle) * 2.25, -4.1, math.sin(angle) * 2.25], [0.28, 0.5, 0.45])
                    teeth.append(tooth)

                weapon = trimesh.util.concatenate([shaft, head, *teeth])
                weapon.apply_translation(part.position)
                return weapon

            base = np.asarray(part.position, dtype=float)
            bottom = base + np.array([4.5, 0.0, -6.0])
            top = base + np.array([4.5, 0.0, 5.6])
            shaft = trimesh.creation.cylinder(radius=0.34, sections=24, segment=np.array([bottom, top], dtype=float))
            grip_parts = []
            for offset in [-3.7, -2.8, -1.9, -1.0, -0.1]:
                wrap = trimesh.creation.torus(major_radius=0.38, minor_radius=0.055, major_sections=18, minor_sections=6)
                wrap.apply_translation(base + np.array([4.5, 0.0, offset]))
                grip_parts.append(wrap)
            blade_center = top + np.array([0.0, -0.08, -0.4])
            blade_l = self._box("left_crescent_axe_blade", (blade_center + np.array([-1.05, -0.12, 0.0])).tolist(), [2.2, 0.42, 2.8])
            blade_r = self._box("right_crescent_axe_blade", (blade_center + np.array([1.05, -0.12, 0.0])).tolist(), [2.2, 0.42, 2.8])
            socket = self._box("axe_head_socket", blade_center.tolist(), [1.0, 0.55, 1.2])
            nicks = [
                self._box("axe_blade_chip", (blade_center + np.array([-2.05, -0.34, 0.85])).tolist(), [0.38, 0.18, 0.42]),
                self._box("axe_blade_chip", (blade_center + np.array([2.05, -0.34, -0.65])).tolist(), [0.38, 0.18, 0.42]),
            ]
            weapon = trimesh.util.concatenate([shaft, blade_l, blade_r, socket, *grip_parts, *nicks])
            return weapon

        if "club" in name or "mace" in name:
            shaft = trimesh.creation.cylinder(radius=0.45, height=8.0, sections=24)
            head = trimesh.creation.cylinder(radius=1.25, height=3.0, sections=24)
            head.apply_translation([0, 0, 4.0])
            spikes = []
            for i in range(8):
                ang = i * math.tau / 8
                spike = trimesh.creation.cone(radius=0.18, height=0.8, sections=12)
                spike.apply_transform(trimesh.transformations.rotation_matrix(math.pi / 2, [0, 1, 0]))
                spike.apply_translation([math.cos(ang) * 1.25, math.sin(ang) * 1.25, 4.35])
                spikes.append(spike)
            weapon = trimesh.util.concatenate([shaft, head, *spikes])
            weapon.apply_translation(part.position)
            return weapon

        if "turret" in name or "gun" in name or "cannon" in name:
            return trimesh.util.concatenate(self._turret(part.position))

        return self._box("planned_weapon_or_part", part.position, part.scale)

    def _rifle(self, pos: list[float], length: float = 7.0, scale: float = 1.0) -> list[trimesh.Trimesh]:
        scale = float(np.clip(scale, 0.7, 1.8))
        body = self._box("rifle_body", pos, [4.0 * scale, 0.65 * scale, 0.9 * scale])

        barrel = trimesh.creation.cylinder(radius=0.28 * scale, height=length, sections=20)
        barrel.apply_transform(trimesh.transformations.rotation_matrix(math.pi / 2, [0, 1, 0]))
        barrel.apply_translation([pos[0] + 1.8 * scale, pos[1], pos[2] + 0.1 * scale])

        grip = self._box("rifle_grip", [pos[0] - 0.6 * scale, pos[1], pos[2] - 0.9 * scale], [0.45 * scale, 0.55 * scale, 1.2 * scale])
        magazine = self._box("rifle_mag", [pos[0] + 0.55 * scale, pos[1], pos[2] - 0.85 * scale], [0.7 * scale, 0.6 * scale, 1.2 * scale])

        return [body, barrel, grip, magazine]

    def _turret(self, pos: list[float]) -> list[trimesh.Trimesh]:
        x, y, z = pos
        mount = self._box("turret_mount", [x, y, z], [1.7, 1.4, 1.0])
        barrel = trimesh.creation.cylinder(radius=0.35, height=3.4, sections=24)
        barrel.apply_transform(trimesh.transformations.rotation_matrix(math.pi / 2, [0, 1, 0]))
        barrel.apply_translation([x + 2.2, y, z + 0.2])
        drum = trimesh.creation.cylinder(radius=0.7, height=0.8, sections=24)
        drum.apply_translation([x - 0.95, y, z])
        return [mount, barrel, drum]

    def _rivet(self, pos: list[float], radius: float = 0.16) -> trimesh.Trimesh:
        rivet = trimesh.creation.icosphere(subdivisions=1, radius=radius)
        rivet.apply_scale([1.0, 0.55, 1.0])
        rivet.apply_translation(pos)
        return rivet

    def _shoulder_pad(self, pos: list[float], mirror: bool = False, scale: float = 1.0) -> trimesh.Trimesh:
        scale = float(np.clip(scale, 0.75, 2.0))
        pad = trimesh.creation.icosphere(subdivisions=2, radius=1.65 * scale)
        pad.apply_scale([1.35, 1.0, 0.75])
        pad.apply_translation(pos)

        rim = trimesh.creation.torus(major_radius=1.35 * scale, minor_radius=0.12 * scale, major_sections=36, minor_sections=8)
        rim.apply_scale([1.25, 0.75, 0.35])
        rim.apply_translation([pos[0], pos[1] - 0.15, pos[2] - 0.25])

        return trimesh.util.concatenate([pad, rim])

    def _capsule(
        self,
        pos: list[float],
        scale: list[float],
        rotate_x: float = 0,
        rotate_y: float = 0,
        rotate_z: float = 0,
    ) -> trimesh.Trimesh:
        mesh = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
        mesh.apply_scale(scale)

        if rotate_x:
            mesh.apply_transform(trimesh.transformations.rotation_matrix(rotate_x, [1, 0, 0]))
        if rotate_y:
            mesh.apply_transform(trimesh.transformations.rotation_matrix(rotate_y, [0, 1, 0]))
        if rotate_z:
            mesh.apply_transform(trimesh.transformations.rotation_matrix(rotate_z, [0, 0, 1]))

        mesh.apply_translation(pos)
        return mesh

    def _limb_between(self, start: list[float], end: list[float], radius: float, sections: int = 18) -> trimesh.Trimesh:
        return trimesh.creation.cylinder(radius=radius, sections=sections, segment=np.array([start, end], dtype=float))

    def _box(self, name: str, pos: list[float], extents: list[float]) -> trimesh.Trimesh:
        mesh = self._rounded_box(extents)
        mesh.apply_translation(pos)
        return mesh

    def _rounded_box(self, extents: list[float]) -> trimesh.Trimesh:
        """Create a printable rounded cuboid instead of a raw rectangular block.

        Wargaming minis still need hard-surface rectangular design language, but
        raw cube primitives make armor/details look blocky and undefined. A
        low-exponent super-ellipsoid gives broad flat-ish faces with softened
        corners, and the later voxel remesh fuses these into a more sculpted STL.
        """
        if os.environ.get("MESHMEND_USE_RAW_BOX_PRIMITIVES", "").strip().lower() in {"1", "true", "yes"}:
            return trimesh.creation.box(extents=extents)

        ex = np.asarray(extents, dtype=float)
        if ex.shape[0] < 3 or np.any(ex <= 0):
            return trimesh.creation.box(extents=extents)

        # Extremely flat engraved lines need to stay crisp and lightweight.
        if float(ex.min()) < 0.055:
            return trimesh.creation.box(extents=extents)

        try:
            rings = 9
            segments = 16
            exponent = 0.30
            half = ex[:3] / 2.0

            def spow(values: np.ndarray, power: float) -> np.ndarray:
                return np.sign(values) * (np.abs(values) ** power)

            vertices: list[list[float]] = [[0.0, 0.0, -float(half[2])]]
            for i in range(1, rings):
                eta = -math.pi / 2.0 + math.pi * i / rings
                ce = math.cos(eta)
                se = math.sin(eta)
                for j in range(segments):
                    omega = -math.pi + math.tau * j / segments
                    co = math.cos(omega)
                    so = math.sin(omega)
                    vertices.append([
                        float(half[0] * spow(np.array([ce]), exponent)[0] * spow(np.array([co]), exponent)[0]),
                        float(half[1] * spow(np.array([ce]), exponent)[0] * spow(np.array([so]), exponent)[0]),
                        float(half[2] * spow(np.array([se]), exponent)[0]),
                    ])
            top_index = len(vertices)
            vertices.append([0.0, 0.0, float(half[2])])

            faces: list[list[int]] = []
            for j in range(segments):
                a = 1 + j
                b = 1 + ((j + 1) % segments)
                faces.append([0, b, a])

            ring_count = rings - 1
            for ring in range(ring_count - 1):
                for j in range(segments):
                    a = 1 + ring * segments + j
                    b = 1 + ring * segments + ((j + 1) % segments)
                    c = 1 + (ring + 1) * segments + ((j + 1) % segments)
                    d = 1 + (ring + 1) * segments + j
                    faces.append([a, b, c])
                    faces.append([a, c, d])

            last_ring_start = 1 + (ring_count - 1) * segments
            for j in range(segments):
                a = last_ring_start + j
                b = last_ring_start + ((j + 1) % segments)
                faces.append([a, b, top_index])

            mesh = trimesh.Trimesh(vertices=np.asarray(vertices), faces=np.asarray(faces), process=False)
            mesh.merge_vertices()
            mesh.remove_unreferenced_vertices()
            try:
                mesh.remove_degenerate_faces()
            except Exception:
                pass
            try:
                mesh.fix_normals()
            except Exception:
                pass
            return mesh
        except Exception:
            return trimesh.creation.box(extents=extents)
