# app/stl_quality_reviewer.py

import numpy as np
import trimesh
import re


class STLQualityReviewer:
    def review(self, mesh: trimesh.Trimesh, prompt: str = "", capability_tier: str = "procedural_draft") -> dict:
        prompt_l = prompt.lower()
        intent = self._generation_intent(prompt_l)
        issues = []
        warnings = []
        score = 100
        production_requested = any(
            term in prompt_l
            for term in [
                "production quality", "production-ready", "production ready", "studio quality", "studio-quality",
                "display quality", "sellable", "commercial quality", "professional miniature",
            ]
        )
        miniature_prompt = any(
            term in prompt_l
            for term in ["miniature", "wargaming", "tabletop", "warrior", "orc", "ork", "knight", "soldier", "marine"]
        ) and intent == "character_miniature"

        if mesh is None or len(mesh.vertices) < 500:
            issues.append("Mesh has too few vertices for a detailed miniature.")
            score -= 35

        if len(mesh.faces) < 1000:
            issues.append("Mesh has too few faces for miniature detail.")
            score -= 30
        elif miniature_prompt and len(mesh.faces) < 50000:
            issues.append("Wargaming miniature output is too low-density for detailed tabletop use.")
            score -= 25

        if production_requested and len(mesh.faces) < 250000:
            issues.append("Production/studio quality request needs much denser final sculpt geometry.")
            score -= 25

        extents = np.asarray(mesh.extents, dtype=float)
        if np.max(extents) <= 0:
            issues.append("Mesh has invalid dimensions.")
            score -= 50
        else:
            height = extents[2]
            width = extents[0]
            depth = extents[1]
            bbox_vol = float(np.prod(np.maximum(extents, 1e-6)))
            fullness = float(abs(mesh.volume)) / (bbox_vol + 1e-8)

            if intent == "character_miniature" and height > 0 and width / height < 0.18:
                issues.append("Model is too column-like; likely failed humanoid generation.")
                score -= 25

            if intent not in {"prop_object", "terrain_object"} and depth / max(height, 1e-6) < 0.12:
                issues.append("Model is too flat or relief-like.")
                score -= 25

            if intent == "character_miniature" and height / max(width, depth, 1e-6) < 0.32 and fullness > 0.18:
                issues.append("Model looks like a base/rock cylinder instead of a miniature subject.")
                score -= 35

            fullness_limit = 0.86 if intent in {"prop_object", "terrain_object", "vehicle_object"} else 0.72
            if fullness > fullness_limit:
                issues.append("Model is too blob-like and lacks a distinct miniature silhouette.")
                score -= 25

        if not bool(mesh.is_watertight):
            warnings.append("Mesh is not watertight; slicer repair may be required before production printing.")
            score -= 10

        components = mesh.split(only_watertight=False)
        component_count = len(components)
        cohesive_sculpt = capability_tier == "structured_cohesive_sculpt"
        if component_count < 4 and not cohesive_sculpt and any(w in prompt_l for w in ["marine", "warrior", "orc", "knight", "soldier"]):
            issues.append("Humanoid prompt expected multiple visible body/gear components.")
            score -= 20
        if component_count > 300:
            warnings.append("Mesh contains many separate shells; production use should boolean-union/remesh the sculpt.")
            score -= 10

        if any(w in prompt_l for w in ["rifle", "bolter", "gun"]) and not self._has_long_weapon_shape(mesh):
            if cohesive_sculpt:
                warnings.append("Prompt requested a firearm, but the cohesive fuse softened the long weapon silhouette.")
                score -= 5
            else:
                issues.append("Prompt requested a firearm, but no clear long weapon silhouette was detected.")
                score -= 20

        if any(w in prompt_l for w in ["wing", "wings", "angel", "demon", "daemon"]) and not self._has_wide_feature(mesh):
            issues.append("Prompt requested wings or a winged creature, but no wide wing-like silhouette was detected.")
            score -= 20

        if any(w in prompt_l for w in ["banner", "flag", "standard bearer"]) and not self._has_tall_thin_feature(mesh):
            issues.append("Prompt requested a banner/flag, but no tall banner-like silhouette was detected.")
            score -= 15

        if any(w in prompt_l for w in ["shield", "viking", "knight"]) and not self._has_wide_feature(mesh):
            warnings.append("Prompt requested a shield/large armor feature; silhouette check could not confirm it clearly.")
            score -= 5

        if any(w in prompt_l for w in ["armor", "marine", "soldier"]) and len(mesh.faces) < 5000:
            issues.append("Armored miniature needs more surface complexity.")
            score -= 15

        production_ready = False
        if production_requested:
            issues.append(
                "Production/studio-quality generation is not guaranteed by the current local procedural pipeline; "
                "treat this as a draft sculpt requiring artist review/remesh/detail pass."
            )
        elif capability_tier in {"trained_exemplar_kitbash", "neural_experimental"} and score >= 90 and bool(mesh.is_watertight):
            production_ready = True

        passed = score >= 70 and not any("Production/studio-quality" in issue for issue in issues)

        return {
            "passed": passed,
            "production_requested": production_requested,
            "production_ready": production_ready,
            "capability_tier": capability_tier,
            "score": max(0, score),
            "issues": issues,
            "warnings": warnings,
            "vertex_count": len(mesh.vertices),
            "face_count": len(mesh.faces),
            "component_count": component_count,
            "watertight": bool(mesh.is_watertight),
            "extents": extents.tolist(),
        }

    @staticmethod
    def _generation_intent(prompt_l: str) -> str:
        tokens = set(re.findall(r"[a-z0-9']+", prompt_l or ""))
        if any(term in prompt_l for term in ("full body", "full-body", "whole character", "entire character")):
            return "character_miniature"
        if tokens & {"mask", "helmet", "helm", "headpiece", "faceplate"}:
            return "wearable_object"
        if tokens & {"rifle", "gun", "pistol", "sword", "axe", "hammer", "shield", "banner", "weapon", "prop", "accessory"}:
            return "prop_object"
        if tokens & {"bust", "portrait"} or "head bust" in prompt_l:
            return "bust"
        if tokens & {"terrain", "scenery", "building", "ruin", "dungeon", "objective"}:
            return "terrain_object"
        if tokens & {"vehicle", "tank", "ship", "walker", "turret"}:
            return "vehicle_object"
        if tokens & {
            "miniature", "figure", "character", "warrior", "soldier", "knight", "wizard", "mage",
            "orc", "ork", "elf", "dwarf", "demon", "daemon", "undead", "skeleton", "zombie",
            "ranger", "archer", "marine", "humanoid", "creature", "monster", "beast",
        }:
            return "character_miniature"
        return "printable_subject"

    def _has_long_weapon_shape(self, mesh: trimesh.Trimesh) -> bool:
        try:
            components = mesh.split(only_watertight=False)
            for part in components:
                e = np.asarray(part.extents, dtype=float)
                if np.max(e) > 3.0 * max(np.min(e), 1e-6):
                    return True
            return False
        except Exception:
            return False

    def _has_wide_feature(self, mesh: trimesh.Trimesh) -> bool:
        try:
            for part in mesh.split(only_watertight=False):
                e = np.asarray(part.extents, dtype=float)
                if e[0] > 2.5 and e[2] > 1.5 and e[1] < max(e[0], e[2]) * 0.65:
                    return True
            return False
        except Exception:
            return False

    def _has_tall_thin_feature(self, mesh: trimesh.Trimesh) -> bool:
        try:
            for part in mesh.split(only_watertight=False):
                e = np.asarray(part.extents, dtype=float)
                if e[2] > 4.0 and max(e[0], e[1]) < e[2] * 0.45:
                    return True
            return False
        except Exception:
            return False
