from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import trimesh


@dataclass(frozen=True)
class MiniatureSpec:
    archetype: str
    scale_mm: float
    pose: str
    armor_style: str
    weapon: str
    base_style: str
    detail_language: list[str]
    proportions: dict[str, float] = field(default_factory=dict)
    concept_cues: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SculptReport:
    provider: str
    capability_tier: str
    spec: dict[str, Any]
    sculpt_layers: list[str]
    faces: int
    vertices: int
    watertight: bool
    components: int
    store_quality_certified: bool
    blockers: list[str]
    miniature_plan: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def generate_sculpted_miniature(request: dict[str, Any], image_path: Path | None, output_dir: Path) -> tuple[trimesh.Trimesh, SculptReport]:
    concept = analyze_concept_image(image_path) if image_path is not None else {}
    local_spec = parse_miniature_spec(request, image_path, concept)
    try:
        from native_sculpt_planner import plan_miniature

        plan = plan_miniature(request, image_path, output_dir, local_spec, concept)
    except Exception as exc:
        plan = None
        (output_dir / "miniature_plan_error.txt").write_text(str(exc), encoding="utf-8")
    if plan is not None:
        (output_dir / "miniature_plan.json").write_text(json.dumps(plan.to_dict(), indent=2), encoding="utf-8")
        spec = miniature_spec_from_plan(plan.spec, local_spec)
        plan_blockers = list(plan.blockers)
        plan_dict = plan.to_dict()
    else:
        spec = local_spec
        plan_blockers = ["planner_unavailable_using_local_spec"]
        plan_dict = {"spec": local_spec.to_dict(), "blockers": plan_blockers}
    parts, layers = build_rigged_miniature(spec, plan_dict)
    mesh = trimesh.util.concatenate(parts)
    mesh = fuse_and_remesh(mesh, spec)
    mesh = bridge_and_refuse_components(mesh, spec)
    mesh = add_readable_sculpt_detail(mesh, spec)
    mesh = normalize(mesh, spec.scale_mm)
    components = connected_components(mesh)
    blockers = plan_blockers + quality_blockers(mesh, spec, components, plan_dict, layers)
    report = SculptReport(
        provider="meshmend_native_sculpt",
        capability_tier="experimental_image_conditioned_native_sculpt",
        spec=asdict(spec),
        sculpt_layers=layers,
        faces=int(len(mesh.faces)),
        vertices=int(len(mesh.vertices)),
        watertight=bool(mesh.is_watertight),
        components=components,
        store_quality_certified=False,
        blockers=blockers + ["native sculpt backend is still experimental and not certified for commercial store-quality output"],
        miniature_plan=plan_dict,
    )
    (output_dir / "native_sculpt_report.json").write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    if high_detail_requested(request) and high_detail_fatal_blockers(report.blockers):
        raise RuntimeError("Native image-conditioned sculpt failed store-quality gates: " + "; ".join(high_detail_fatal_blockers(report.blockers)))
    return mesh, report


def high_detail_requested(request: dict[str, Any]) -> bool:
    quality = str(request.get("quality") or "standard").lower()
    prompt = str(request.get("prompt") or "").lower()
    return quality == "high" or any(
        term in prompt
        for term in (
            "8k", "8 k", "studio", "studio quality", "studio-quality", "studio level", "studio-level",
            "production", "display quality", "maximum detail", "store quality", "store-quality", "store level",
            "store-level", "intricate",
        )
    )


def high_detail_fatal_blockers(blockers: list[str]) -> list[str]:
    if os.environ.get("MESHMEND_SCULPT_ALLOW_HIGH_WITH_BLOCKERS", "0").strip().lower() in {"1", "true", "yes", "on"}:
        return []
    fatal_prefixes = (
        "ai_planner_required",
        "ai_planner_command_failed",
        "image_to_3d_requires_ai_vision_planner",
        "low_vision_planner_confidence",
        "unsupported_archetype",
        "unsupported_parts",
        "concept_match",
        "planned_landmarks",
        "detail_features_below_printable_size",
        "no_concept_image_received",
        "mesh_not_watertight",
        "multiple_components",
        "below_experimental_sculpt_face_floor",
    )
    return [blocker for blocker in blockers if blocker.startswith(fatal_prefixes)]


def parse_miniature_spec(request: dict[str, Any], image_path: Path | None, concept: dict[str, Any] | None = None) -> MiniatureSpec:
    prompt = str(request.get("prompt") or "").lower()
    scale = float(request.get("scale_mm") or 32.0)
    concept = concept if concept is not None else (analyze_concept_image(image_path) if image_path is not None else {})
    archetype = "armored_humanoid"
    if any(term in prompt for term in ("orc", "ork", "brute")):
        archetype = "orc_warrior"
    elif any(
        term in prompt
        for term in (
            "dragon",
            "beast",
            "mounted",
            "mount ",
            "rider",
            "cavalry",
            "lizard",
            "reptile",
            "reptilian",
            "quadruped",
            "four-legged",
            "four legged",
            "saddle",
            "steed",
            "warbeast",
            "war beast",
        )
    ):
        archetype = "mounted_beast"
    elif any(term in prompt for term in ("mech", "robot", "walker", "tank")):
        archetype = "armored_mech"
    elif concept.get("mounted_creature_silhouette") or concept.get("quadruped_silhouette"):
        archetype = "mounted_beast"
    elif concept.get("boxy_silhouette"):
        archetype = "armored_mech"
    is_space_marine = any(term in prompt for term in ("space marine", "spacemarine", "adeptus", "primaris", "power armored marine", "power armoured marine"))
    weapon = (
        "sword" if concept.get("raised_blade") or concept.get("blade_like") or any(term in prompt for term in ("sword", "blade", "katana", "khopesh", "scimitar", "curved sword", "raised sword", "serrated"))
        else "rifle" if is_space_marine or any(term in prompt for term in ("rifle", "gun", "bolter", "blaster", "cannon")) or (concept.get("long_horizontal_weapon") and archetype != "mounted_beast")
        else "axe" if any(term in prompt for term in ("axe", "halberd"))
        else "staff" if any(term in prompt for term in ("staff", "spear", "lance")) or concept.get("tall_weapon")
        else "sword" if archetype == "mounted_beast"
        else "sidearm"
    )
    armor_style = "power_armor" if any(term in prompt for term in ("space", "marine", "power armor", "sci-fi", "scifi", "robotic")) or concept.get("hard_surface") else "plate_armor"
    details = ["large_silhouette", "layered_armor", "deep_panel_lines", "raised_trim", "readable_face_or_helmet", "scenic_base"]
    if is_space_marine:
        details += ["space_marine_silhouette", "oversized_pauldrons", "power_pack", "helmet_visor", "chest_emblem", "bolter_rifle"]
    workflow = str(request.get("workflow") or "text_to_3d")
    if image_path is None and workflow == "image_to_3d":
        details.append("no_concept_image_received")
    if archetype == "orc_warrior":
        details += ["tusks", "scars", "leather_straps", "trophy_skulls"]
    if archetype == "mounted_beast":
        details += ["scales", "claws", "saddle", "rider", "tail", "spine_spikes", "armored_beast_plates"]
    if any(term in prompt for term in ("wing", "angel", "demon", "dragon")) or concept.get("wide_upper_silhouette"):
        details.append("wings_or_large_back_silhouette")
    if any(term in prompt for term in ("cape", "cloak", "robe", "hood", "hooded", "tattered", "fur")) or concept.get("flowing_back_mass"):
        details.append("cape_or_cloak")
    if any(term in prompt for term in ("wizard", "mage", "sorcerer", "witch", "warlock", "cleric", "priest", "robe", "robed", "staff")):
        details += ["robed_caster_silhouette", "staff_or_focus", "hood_or_high_collar"]
    if any(term in prompt for term in ("rogue", "assassin", "ranger", "thief", "ninja", "dagger", "dual wield", "dual-wield")):
        details += ["lean_stealth_silhouette", "hood_or_mask", "daggers_or_short_blades"]
    if any(term in prompt for term in ("knight", "paladin", "templar", "crusader", "champion", "banner")):
        details += ["heroic_knight_silhouette", "large_shield_or_banner", "crested_helmet"]
    if any(term in prompt for term in ("demon", "devil", "fiend", "tiefling")):
        details += ["demonic_horns", "tail", "clawed_feet"]
    if any(term in prompt for term in ("angel", "celestial", "seraph")):
        details += ["angelic_wings", "halo_or_crown"]
    if any(term in prompt for term in ("shield", "banner", "standard")) or concept.get("side_panel"):
        details.append("large_side_accessory")
    if any(term in prompt for term in ("spike", "spiked", "spine", "horn", "horned")):
        details.append("spikes_or_horns")
    if any(term in prompt for term in ("skull", "skulls", "bone", "bones", "rock", "rocks")):
        details.append("skull_or_rock_base_detail")
    if prompt_requests_creature_claws(prompt) and "claws" not in details:
        details.append("claws")
    if concept.get("high_edge_density"):
        details.append("busy_surface_detail_from_concept")
    proportions = {
        "bbox_aspect": float(concept.get("bbox_aspect", 0.0) or 0.0),
        "top_width": float(concept.get("top_width", 0.0) or 0.0),
        "mid_width": float(concept.get("mid_width", 0.0) or 0.0),
        "bottom_width": float(concept.get("bottom_width", 0.0) or 0.0),
        "foreground_ratio": float(concept.get("foreground_ratio", 0.0) or 0.0),
    }
    concept_cues = [key for key, value in concept.items() if isinstance(value, bool) and value]
    return MiniatureSpec(archetype, scale, "heroic_contrapposto", armor_style, weapon, "round_scenic", details, proportions, concept_cues)


def miniature_spec_from_plan(plan_spec: dict[str, Any], fallback: MiniatureSpec) -> MiniatureSpec:
    data = fallback.to_dict()
    data.update({key: value for key, value in plan_spec.items() if value is not None})
    detail_language = data.get("detail_language") or fallback.detail_language
    proportions = data.get("proportions") or fallback.proportions
    concept_cues = data.get("concept_cues") or fallback.concept_cues
    return MiniatureSpec(
        archetype=str(data.get("archetype") or fallback.archetype),
        scale_mm=float(data.get("scale_mm") or fallback.scale_mm),
        pose=str(data.get("pose") or fallback.pose),
        armor_style=str(data.get("armor_style") or fallback.armor_style),
        weapon=str(data.get("weapon") or fallback.weapon),
        base_style=str(data.get("base_style") or fallback.base_style),
        detail_language=[str(item) for item in list(detail_language)],
        proportions={str(key): float(value) for key, value in dict(proportions).items() if isinstance(value, (int, float))},
        concept_cues=[str(item) for item in list(concept_cues)],
    )


def analyze_concept_image(image_path: Path | None) -> dict[str, Any]:
    """Extract coarse, deterministic concept cues for native sculpt planning.

    This is not neural image-to-3D; it prevents the native backend from ignoring
    the uploaded concept by using silhouette/proportion/detail cues to choose the
    rig and large accessories.
    """
    if image_path is None:
        return {}
    try:
        from PIL import Image, ImageFilter

        image = Image.open(image_path).convert("RGBA")
        image.thumbnail((384, 384))
        rgba = np.asarray(image, dtype=np.float32) / 255.0
        alpha = rgba[:, :, 3]
        rgb = rgba[:, :, :3]
        h, w = alpha.shape
        # Prefer alpha if available; otherwise segment against border-average background.
        if float(alpha.max() - alpha.min()) > 0.2:
            mask = alpha > 0.25
        else:
            border = np.concatenate([rgb[0, :, :], rgb[-1, :, :], rgb[:, 0, :], rgb[:, -1, :]], axis=0)
            bg = np.median(border, axis=0)
            diff = np.linalg.norm(rgb - bg[None, None, :], axis=2)
            threshold = max(0.10, float(np.percentile(diff, 68)))
            mask = diff > threshold
        ys, xs = np.where(mask)
        if len(xs) < max(32, int(w * h * 0.01)):
            return {"concept_image_received": True, "foreground_ratio": 0.0}
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        bw = max(1, x1 - x0 + 1)
        bh = max(1, y1 - y0 + 1)
        cropped = mask[y0 : y1 + 1, x0 : x1 + 1]
        def band_width(start: float, stop: float) -> float:
            a = int(cropped.shape[0] * start)
            b = max(a + 1, int(cropped.shape[0] * stop))
            rows = cropped[a:b]
            cols = np.where(rows.any(axis=0))[0]
            return float(len(cols) / max(1, cropped.shape[1]))
        top_width = band_width(0.05, 0.28)
        mid_width = band_width(0.36, 0.64)
        bottom_width = band_width(0.72, 0.96)
        gray = Image.fromarray((rgb.mean(axis=2) * 255).astype(np.uint8)).filter(ImageFilter.FIND_EDGES)
        edges = np.asarray(gray, dtype=np.float32) / 255.0
        edge_density = float(edges[mask].mean()) if mask.any() else 0.0
        left_mass = float(mask[:, : w // 2].mean())
        right_mass = float(mask[:, w // 2 :].mean())
        aspect = float(bw / bh)
        return {
            "concept_image_received": True,
            "bbox_aspect": aspect,
            "foreground_ratio": float(mask.mean()),
            "top_width": top_width,
            "mid_width": mid_width,
            "bottom_width": bottom_width,
            "wide_silhouette": aspect > 1.05,
            "quadruped_silhouette": aspect > 1.18 and bottom_width > 0.50 and top_width < mid_width * 0.88,
            "mounted_creature_silhouette": aspect > 1.02 and mid_width > 0.72 and bottom_width > 0.55 and top_width < mid_width * 0.55,
            "boxy_silhouette": mid_width > 0.72 and abs(top_width - mid_width) < 0.18,
            "wide_upper_silhouette": top_width > mid_width * 1.16,
            "flowing_back_mass": bottom_width > mid_width * 1.08,
            "side_panel": abs(left_mass - right_mass) > 0.08,
            "long_horizontal_weapon": aspect > 0.72 and abs(left_mass - right_mass) > 0.055,
            "tall_weapon": bh / max(1, bw) > 1.85 and top_width < 0.35,
            "raised_blade": top_width < 0.36 and edge_density > 0.08 and aspect > 0.75,
            "blade_like": top_width < 0.32 and mid_width < 0.52 and edge_density > 0.08,
            "hard_surface": edge_density > 0.095,
            "high_edge_density": edge_density > 0.11,
        }
    except Exception as exc:
        return {"concept_image_received": True, "concept_analysis_error": str(exc)}


def build_rigged_miniature(spec: MiniatureSpec, plan: dict[str, Any] | None = None) -> tuple[list[trimesh.Trimesh], list[str]]:
    parts: list[trimesh.Trimesh] = []
    layers: list[str] = []
    base_radius = 0.42 * spec.scale_mm
    base = trimesh.creation.cylinder(radius=base_radius, height=1.8, sections=128)
    base.apply_translation([0, 0, 0.9])
    parts.append(base); layers.append("scenic_base")
    if spec.archetype == "mounted_beast":
        parts += mounted_beast_parts(spec); layers += ["beast_anatomy", "rider", "saddle", "creature_detail"]
    elif spec.archetype == "armored_mech":
        parts += mech_parts(); layers += ["mechanical_rig", "armor_panels", "weapon_mount"]
    elif is_space_marine_spec(spec):
        parts += space_marine_parts(spec); layers += ["space_marine_primary_armor_rig", "space_marine_bolter_rig", "space_marine_power_pack", "helmet_face"]
    elif spec.archetype == "orc_warrior":
        parts += orc_warrior_parts(spec); layers += ["orc_brute_primary_rig", "orc_tusks", "orc_weapon_rig", "helmet_face"]
    elif needs_concept_primary_rig(spec):
        concept_parts, concept_layers = concept_primary_humanoid_parts(spec)
        parts += concept_parts; layers += concept_layers
    else:
        parts += humanoid_parts(spec); layers += ["anatomy_rig", "armor_rig", "weapon_rig", "helmet_face"]
    accessories = concept_accessory_parts(spec)
    if accessories:
        parts += accessories
        layers.append("concept_silhouette_accessories")
    planned_parts, unsupported = planned_rig_accessory_parts(spec, plan or {})
    if planned_parts:
        parts += planned_parts
        layers.append("planned_rig_parts")
    if unsupported:
        layers.append("unsupported_plan_parts:" + ",".join(unsupported[:8]))
    detail_parts, detail_layers = native_sculpt_detail_parts(spec, plan or {})
    if detail_parts:
        parts += detail_parts
        layers += detail_layers
    parts += base_detail_parts(base_radius)
    layers.append("base_story_detail")
    return parts, layers


def humanoid_parts(spec: MiniatureSpec) -> list[trimesh.Trimesh]:
    p: list[trimesh.Trimesh] = []
    # Leaned pose: asymmetric limbs so the silhouette reads as a miniature, not a mannequin.
    p += [
        capsule((-1.45, -0.2, 1.8), (-1.05, -0.25, 9.4), 0.72),
        capsule((1.65, 0.25, 1.8), (1.05, 0.15, 9.2), 0.72),
        ellipsoid((0, 0, 9.4), (2.6, 1.35, 1.0)),
        ellipsoid((0, -0.05, 13.2), (3.7, 2.0, 4.9)),
        ellipsoid((0, -0.45, 16.0), (4.4, 1.2, 1.55)),
        capsule((0, 0, 16.2), (0, 0, 17.1), 0.55),
        ellipsoid((0, -0.25, 18.55), (1.35, 1.0, 1.55)),
    ]
    # Oversized readable miniature armor hierarchy.
    for side in (-1, 1):
        p.append(ellipsoid((side * 3.75, -0.15, 15.7), (1.75, 1.15, 1.2)))
        p.append(capsule((side * 3.7, -0.15, 14.8), (side * 5.0, -0.65, 11.6), 0.48))
        p.append(capsule((side * 5.0, -0.65, 11.6), (side * 3.55, -1.05, 9.2), 0.42))
        p.append(ellipsoid((side * 1.25, -0.6, 6.2), (0.9, 0.38, 0.7)))
        p.append(ellipsoid((side * 1.55, -0.35, 1.8), (1.05, 0.7, 0.45)))
    p += armor_surface_parts(spec)
    p += weapon_parts(spec.weapon)
    return p


def needs_concept_primary_rig(spec: MiniatureSpec) -> bool:
    details = {item.lower() for item in spec.detail_language}
    primary_tags = {
        "robed_caster_silhouette",
        "lean_stealth_silhouette",
        "heroic_knight_silhouette",
        "wings_or_large_back_silhouette",
        "angelic_wings",
        "demonic_horns",
        "large_shield_or_banner",
        "large_side_accessory",
    }
    return any(tag in details for tag in primary_tags)


def concept_primary_humanoid_parts(spec: MiniatureSpec) -> tuple[list[trimesh.Trimesh], list[str]]:
    """Build the first-read silhouette from prompt cues before detail stamps.

    Generic humanoids are acceptable only when the prompt is generic. If the
    prompt asks for a mage, rogue, knight, angel/demon, shield-bearer, etc., the
    primary massing must already read that way before accessories and relief are
    fused in.
    """
    details = {item.lower() for item in spec.detail_language}
    p: list[trimesh.Trimesh] = []
    layers = ["concept_primary_silhouette_rig", "helmet_face"]

    if "robed_caster_silhouette" in details:
        p += robed_caster_parts(spec)
        layers += ["robed_caster_primary_rig", "staff_or_focus_rig"]
        return p, layers
    if "lean_stealth_silhouette" in details:
        p += rogue_assassin_parts(spec)
        layers += ["lean_stealth_primary_rig", "daggers_or_short_blades_rig"]
        return p, layers
    if "heroic_knight_silhouette" in details or "large_shield_or_banner" in details or "large_side_accessory" in details:
        p += heroic_knight_parts(spec)
        layers += ["heroic_knight_primary_rig", "shield_or_banner_rig", "weapon_rig"]
        return p, layers
    if "wings_or_large_back_silhouette" in details or "angelic_wings" in details or "demonic_horns" in details:
        p += winged_humanoid_parts(spec)
        layers += ["winged_primary_rig", "wings_or_large_back_silhouette", "weapon_rig"]
        return p, layers

    p += humanoid_parts(spec)
    layers += ["anatomy_rig", "armor_rig", "weapon_rig"]
    return p, layers


def robed_caster_parts(spec: MiniatureSpec) -> list[trimesh.Trimesh]:
    p: list[trimesh.Trimesh] = []
    p.append(ellipsoid((0.0, 0.05, 9.8), (3.35, 1.25, 7.4), subdivisions=2))
    p.append(ellipsoid((0.0, 0.18, 5.0), (4.25, 1.05, 3.4), subdivisions=2))
    for x in np.linspace(-2.2, 2.2, 7):
        p.append(capsule((float(x), -0.95, 15.0), (float(x) * 1.32, -1.15, 3.4), 0.08))
    p.append(ellipsoid((0.0, -0.15, 15.7), (3.4, 1.2, 1.0), subdivisions=2))
    p.append(ellipsoid((0.0, -0.22, 18.45), (1.35, 0.95, 1.35), subdivisions=2))
    p.append(ellipsoid((0.0, 0.12, 18.85), (1.75, 1.25, 1.45), subdivisions=2))
    p.append(capsule((-4.85, -0.95, 2.7), (-4.85, -0.95, 19.7), 0.18))
    p.append(ellipsoid((-4.85, -0.95, 20.15), (0.82, 0.42, 0.82), subdivisions=1))
    p.append(capsule((-2.5, -0.55, 14.7), (-4.62, -0.95, 13.4), 0.34))
    p.append(capsule((2.5, -0.55, 14.5), (3.8, -1.45, 12.2), 0.30))
    return p


def rogue_assassin_parts(spec: MiniatureSpec) -> list[trimesh.Trimesh]:
    p = humanoid_parts(MiniatureSpec(spec.archetype, spec.scale_mm, spec.pose, "light_armor", "sidearm", spec.base_style, spec.detail_language, spec.proportions, spec.concept_cues))
    p.append(ellipsoid((0.0, 1.8, 10.8), (3.0, 0.36, 5.8), subdivisions=2))
    p.append(ellipsoid((0.0, -0.10, 18.85), (1.65, 1.05, 1.28), subdivisions=2))
    p.append(box((1.08, 0.22, 0.22), (0.0, -1.18, 18.70)))
    for side in (-1, 1):
        p.append(capsule((side * 3.8, -1.05, 10.0), (side * 5.8, -1.25, 7.0), 0.16))
        p.append(box((0.42, 0.18, 1.45), (side * 5.95, -1.28, 6.45)))
        p.append(capsule((side * 1.2, -0.35, 2.0), (side * 1.65, -0.55, 8.8), 0.52))
    return p


def heroic_knight_parts(spec: MiniatureSpec) -> list[trimesh.Trimesh]:
    p = humanoid_parts(MiniatureSpec(spec.archetype, spec.scale_mm, spec.pose, spec.armor_style, "sword" if spec.weapon == "sidearm" else spec.weapon, spec.base_style, spec.detail_language, spec.proportions, spec.concept_cues))
    p.append(ellipsoid((-5.15, -1.18, 11.6), (1.72, 0.34, 3.05), subdivisions=2))
    p.append(box((1.52, 0.16, 0.20), (-5.15, -1.50, 12.95)))
    p.append(box((0.20, 0.16, 2.35), (-5.15, -1.52, 11.55)))
    p.append(capsule((0.0, -0.20, 19.42), (0.0, -0.20, 20.82), 0.12))
    p.append(ellipsoid((0.0, -0.20, 20.95), (0.78, 0.26, 0.24), subdivisions=1))
    p.append(box((3.6, 0.28, 0.34), (0.0, -2.18, 15.6)))
    return p


def winged_humanoid_parts(spec: MiniatureSpec) -> list[trimesh.Trimesh]:
    p = humanoid_parts(spec)
    for side in (-1, 1):
        p.append(ellipsoid((side * 5.3, 1.65, 15.5), (3.0, 0.38, 5.6), subdivisions=2))
        p.append(capsule((side * 2.0, 1.15, 15.6), (side * 7.6, 1.95, 19.2), 0.16))
        p.append(capsule((side * 2.1, 1.15, 14.4), (side * 7.2, 1.95, 11.0), 0.16))
        for z in np.linspace(12.0, 18.4, 6):
            p.append(capsule((side * 2.7, 1.05, float(z)), (side * 7.1, 1.85, float(z - 1.0)), 0.07))
    if "demonic_horns" in {item.lower() for item in spec.detail_language}:
        for side in (-1, 1):
            p.append(capsule((side * 0.72, -0.36, 19.48), (side * 1.58, -0.52, 20.72), 0.12))
        p.append(capsule((0.8, 1.4, 8.4), (2.8, 1.1, 4.2), 0.22))
    if "halo_or_crown" in {item.lower() for item in spec.detail_language}:
        p.append(trimesh.creation.torus(major_radius=1.05, minor_radius=0.08, major_sections=48, minor_sections=8))
        p[-1].apply_translation([0, -0.25, 20.7])
    return p


def orc_warrior_parts(spec: MiniatureSpec) -> list[trimesh.Trimesh]:
    p: list[trimesh.Trimesh] = []
    p.append(ellipsoid((0.0, -0.05, 12.9), (4.45, 2.25, 4.6), subdivisions=2))
    p.append(ellipsoid((0.0, -0.10, 9.0), (3.2, 1.4, 1.15), subdivisions=2))
    p.append(ellipsoid((0.0, -0.35, 18.1), (1.75, 1.18, 1.45), subdivisions=2))
    p.append(box((2.6, 0.34, 0.34), (0.0, -1.42, 17.78)))
    for side in (-1, 1):
        p.append(ellipsoid((side * 0.38, -1.22, 17.40), (0.12, 0.08, 0.42), subdivisions=1))
        p.append(capsule((side * 0.42, -1.16, 17.05), (side * 0.82, -1.42, 17.45), 0.09))
        p.append(ellipsoid((side * 3.9, -0.25, 15.2), (1.85, 1.05, 1.05), subdivisions=2))
        p.append(capsule((side * 3.8, -0.35, 14.3), (side * 5.1, -0.9, 10.8), 0.72))
        p.append(capsule((side * 5.1, -0.9, 10.8), (side * 3.7, -1.2, 8.7), 0.62))
        p.append(capsule((side * 1.55, -0.05, 1.8), (side * 1.15, -0.10, 8.4), 0.82))
        p.append(ellipsoid((side * 1.75, -0.35, 1.65), (1.05, 0.66, 0.42), subdivisions=1))
    p.append(capsule((5.35, -1.1, 6.0), (5.35, -1.1, 16.6), 0.22))
    p.append(box((3.25, 0.32, 1.55), (5.35, -1.30, 16.45)))
    p += armor_surface_parts(spec)
    return p


def space_marine_parts(spec: MiniatureSpec) -> list[trimesh.Trimesh]:
    """Dedicated chunky sci-fi marine body, not a generic humanoid overlay.

    The earlier implementation added space-marine details on top of the normal
    humanoid rig, so fusion still preserved the mannequin proportions. This rig
    starts from the recognisable silhouette: squat stance, huge pauldrons,
    barrel torso, enclosed helmet, backpack vents, bolter across chest, blocky
    greaves and boots.
    """
    p: list[trimesh.Trimesh] = []
    # Power-armored lower body: wide stance and blocky legs.
    p.append(ellipsoid((0.0, -0.02, 9.6), (2.7, 1.45, 1.1), subdivisions=2))
    for side in (-1, 1):
        p.append(capsule((side * 1.55, -0.05, 2.0), (side * 1.25, -0.08, 7.7), 0.86))
        p.append(box((1.85, 0.95, 3.55), (side * 1.35, -0.55, 5.65)))
        p.append(ellipsoid((side * 1.35, -1.10, 7.35), (0.88, 0.32, 0.58), subdivisions=1))
        p.append(box((2.05, 1.35, 0.62), (side * 1.68, -0.58, 1.55)))
        p.append(ellipsoid((side * 2.08, -1.18, 1.30), (0.82, 0.36, 0.22), subdivisions=1))

    # Barrel torso and layered chest plate.
    p.append(ellipsoid((0.0, -0.06, 13.85), (4.6, 2.45, 3.95), subdivisions=2))
    p.append(box((4.05, 0.52, 2.70), (0.0, -2.68, 14.05)))
    p.append(box((2.90, 0.45, 1.12), (0.0, -2.78, 11.35)))
    p.append(box((2.35, 0.32, 0.26), (0.0, -3.02, 15.35)))
    p.append(box((0.34, 0.30, 1.55), (0.0, -3.03, 14.65)))

    # Oversized pauldrons define the silhouette.
    for side in (-1, 1):
        p.append(ellipsoid((side * 4.65, -0.12, 16.05), (2.65, 1.70, 1.85), subdivisions=2))
        p.append(box((2.42, 0.42, 0.34), (side * 4.65, -1.72, 16.95)))
        p.append(box((2.30, 0.36, 0.34), (side * 4.65, -1.72, 15.05)))
        p.append(capsule((side * 4.50, -0.95, 14.95), (side * 3.95, -1.80, 11.95), 0.58))
        p.append(capsule((side * 3.95, -1.80, 11.95), (side * 2.15, -2.30, 12.15), 0.50))
        p.append(ellipsoid((side * 2.15, -2.45, 12.05), (0.48, 0.30, 0.40), subdivisions=1))

    # Full helmet with visor and respirator instead of a generic head oval.
    p.append(ellipsoid((0.0, -0.22, 18.60), (1.70, 1.25, 1.55), subdivisions=2))
    p.append(box((1.48, 0.36, 0.34), (0.0, -1.38, 18.82)))
    p.append(box((0.68, 0.42, 0.82), (0.0, -1.46, 18.18)))
    for side in (-1, 1):
        p.append(box((0.32, 0.32, 0.82), (side * 0.52, -1.48, 18.13)))
        p.append(ellipsoid((side * 0.95, -0.32, 18.55), (0.28, 0.20, 0.42), subdivisions=1))

    # Backpack/power pack with exhaust stacks and side boxes.
    p.append(box((3.65, 1.55, 4.05), (0.0, 3.05, 14.40)))
    for side in (-1, 1):
        p.append(capsule((side * 1.20, 3.82, 14.10), (side * 1.20, 3.82, 18.60), 0.44))
        p.append(ellipsoid((side * 1.20, 3.82, 18.98), (0.62, 0.46, 0.42), subdivisions=1))
        p.append(box((0.52, 0.34, 1.92), (side * 2.04, 3.66, 14.35)))

    # Bolter held across chest: large box + barrel + magazine.
    p.append(box((5.80, 0.70, 1.02), (3.75, -2.78, 12.35)))
    p.append(capsule((6.25, -2.78, 12.42), (10.05, -2.78, 12.42), 0.28))
    p.append(box((1.55, 0.54, 1.45), (1.45, -2.78, 11.78)))
    p.append(box((1.28, 0.48, 0.82), (4.34, -3.08, 11.48)))
    p.append(box((0.80, 0.42, 0.35), (6.80, -3.12, 13.05)))

    # Scenic base tie-in so the stance does not float.
    for side in (-1, 1):
        p.append(capsule((side * 1.55, -0.55, 1.35), (side * 1.55, -0.55, 0.95), 0.66))
    return p


def armor_surface_parts(spec: MiniatureSpec) -> list[trimesh.Trimesh]:
    p: list[trimesh.Trimesh] = []
    for z, width in ((15.2, 5.4), (14.1, 4.7), (12.8, 4.2), (11.4, 3.6)):
        p.append(box((width, 0.22, 0.18), (0, -2.05, z)))
    for x in np.linspace(-2.2, 2.2, 7):
        p.append(ellipsoid((float(x), -2.24, 14.15), (0.14, 0.06, 0.14), subdivisions=1))
    p.append(box((2.4, 1.0, 3.1), (0, 2.2, 13.8)))
    p.append(box((1.65, 0.14, 0.32), (0, -1.26, 18.65)))
    for side in (-1, 1):
        for z in (4.4, 5.8, 7.4, 8.4):
            p.append(box((1.05, 0.16, 0.18), (side * 1.25, -0.82, z)))
        for z in (12.0, 13.2):
            p.append(ellipsoid((side * 4.55, -0.75, z), (0.42, 0.14, 0.24), subdivisions=1))
    return p


def weapon_parts(weapon: str) -> list[trimesh.Trimesh]:
    if weapon == "rifle":
        return [box((5.2, 0.38, 0.62), (3.8, -1.82, 11.7)), capsule((5.6, -1.82, 11.8), (8.8, -1.82, 11.8), 0.18), box((1.1, 0.34, 1.0), (2.2, -1.82, 11.2))]
    if weapon == "axe":
        return [capsule((5.0, -1.3, 7.2), (5.0, -1.3, 16.2), 0.18), box((2.8, 0.26, 1.35), (5.0, -1.45, 16.1))]
    if weapon == "staff":
        return [capsule((-5.2, -1.0, 3.0), (-5.2, -1.0, 18.8), 0.16), ellipsoid((-5.2, -1.0, 19.3), (0.55, 0.28, 0.55), subdivisions=1)]
    return [capsule((5.1, -1.15, 7.5), (5.6, -1.15, 15.5), 0.18), box((1.2, 0.22, 2.4), (5.7, -1.2, 15.8))]


def concept_accessory_parts(spec: MiniatureSpec) -> list[trimesh.Trimesh]:
    p: list[trimesh.Trimesh] = []
    details = set(spec.detail_language)
    proportions = spec.proportions or {}
    top_width = float(proportions.get("top_width", 0.0) or 0.0)
    mid_width = float(proportions.get("mid_width", 0.0) or 0.0)
    wing_scale = max(1.0, min(1.7, 1.0 + max(0.0, top_width - mid_width) * 1.8))
    if "wings_or_large_back_silhouette" in details:
        for side in (-1, 1):
            wing = ellipsoid((side * 4.25 * wing_scale, 1.45, 15.0), (2.35 * wing_scale, 0.34, 4.8), subdivisions=2)
            wing.apply_transform(trimesh.transformations.rotation_matrix(side * math.radians(18), [0, 0, 1], point=[side * 2.0, 1.45, 15.0]))
            p.append(wing)
            for z in np.linspace(12.0, 18.0, 6):
                p.append(capsule((side * 2.4, 1.1, float(z)), (side * 6.0 * wing_scale, 1.55, float(z - 1.2)), 0.07))
    if "cape_or_cloak" in details:
        p.append(ellipsoid((0, 2.7, 11.2), (3.6, 0.42, 6.8), subdivisions=2))
        for x in np.linspace(-2.4, 2.4, 5):
            p.append(capsule((float(x), 2.25, 15.0), (float(x * 1.25), 2.85, 5.2), 0.08))
        p.append(ellipsoid((0, -0.18, 19.05), (1.65, 1.15, 1.45), subdivisions=2))
    if "large_side_accessory" in details:
        p.append(ellipsoid((-4.7, -1.25, 11.2), (1.45, 0.28, 2.4), subdivisions=2))
        p.append(capsule((-5.8, -1.45, 4.0), (-5.8, -1.45, 18.0), 0.13))
        p.append(box((1.4, 0.18, 2.0), (-5.8, -1.55, 16.2)))
    if "space_marine_silhouette" in details or "power_pack" in details:
        p += space_marine_silhouette_parts()
    if "spikes_or_horns" in details:
        for side in (-1, 1):
            p.append(capsule((side * 0.85, -0.35, 19.55), (side * 1.45, -0.5, 20.7), 0.10))
            for x in np.linspace(2.6, 5.2, 4):
                p.append(ellipsoid((side * float(x), -0.2, 16.75), (0.16, 0.12, 0.38), subdivisions=1))
    if "skull_or_rock_base_detail" in details:
        for i, x in enumerate(np.linspace(-4.5, 4.5, 7)):
            y = -5.8 + (i % 2) * 0.8
            p.append(ellipsoid((float(x), y, 2.0), (0.38, 0.28, 0.26), subdivisions=1))
            p.append(ellipsoid((float(x) - 0.14, y - 0.18, 2.08), (0.06, 0.04, 0.04), subdivisions=1))
            p.append(ellipsoid((float(x) + 0.14, y - 0.18, 2.08), (0.06, 0.04, 0.04), subdivisions=1))
    return p


def space_marine_primary_armor_parts() -> list[trimesh.Trimesh]:
    """Large first-read forms for a space marine silhouette.

    These are deliberately primary/secondary volumes, not fine detail. They must
    survive voxel fusion and be recognizable from the STL thumbnail: huge
    pauldrons, slab chest, enclosed helmet, backpack/exhausts, bolter, chunky
    greaves and boots.
    """
    p: list[trimesh.Trimesh] = []
    # Barrel chest and abdomen armor over the humanoid body.
    p.append(ellipsoid((0.0, -0.18, 14.05), (4.45, 2.35, 3.65), subdivisions=2))
    p.append(box((3.45, 0.42, 2.45), (0.0, -2.55, 14.15)))
    p.append(box((2.65, 0.36, 1.05), (0.0, -2.68, 11.35)))

    # Helmet should read as a sci-fi full helm, not a bare oval head.
    p.append(ellipsoid((0.0, -0.20, 18.65), (1.65, 1.25, 1.55), subdivisions=2))
    p.append(box((1.42, 0.30, 0.30), (0.0, -1.32, 18.78)))
    p.append(box((0.58, 0.34, 0.74), (0.0, -1.38, 18.23)))
    for side in (-1, 1):
        p.append(box((0.28, 0.26, 0.74), (side * 0.48, -1.40, 18.18)))

    # Oversized shoulder pads: this is the key non-generic silhouette cue.
    for side in (-1, 1):
        p.append(ellipsoid((side * 4.45, -0.10, 16.05), (2.45, 1.55, 1.72), subdivisions=2))
        p.append(box((2.20, 0.34, 0.28), (side * 4.45, -1.58, 16.85)))
        p.append(box((2.05, 0.30, 0.28), (side * 4.45, -1.58, 15.20)))

    # Backpack with two tall exhaust stacks.
    p.append(box((3.45, 1.35, 3.85), (0.0, 3.12, 14.35)))
    for side in (-1, 1):
        p.append(capsule((side * 1.18, 3.75, 14.35), (side * 1.18, 3.75, 18.45), 0.42))
        p.append(ellipsoid((side * 1.18, 3.75, 18.82), (0.58, 0.42, 0.38), subdivisions=1))
        p.append(box((0.45, 0.28, 1.75), (side * 1.95, 3.66, 14.50)))

    # Chest-held bolter silhouette, large enough to read at thumbnail scale.
    p.append(box((5.60, 0.62, 0.90), (3.75, -2.60, 12.20)))
    p.append(capsule((6.10, -2.60, 12.25), (9.70, -2.60, 12.25), 0.26))
    p.append(box((1.45, 0.48, 1.35), (1.55, -2.60, 11.70)))
    p.append(box((1.20, 0.42, 0.72), (4.30, -2.86, 11.40)))

    # Chunky greaves, kneepads, and boots avoid the skinny mannequin look.
    for side in (-1, 1):
        p.append(box((1.55, 0.72, 3.30), (side * 1.28, -0.66, 5.85)))
        p.append(ellipsoid((side * 1.25, -1.08, 7.18), (0.78, 0.28, 0.52), subdivisions=1))
        p.append(box((1.82, 1.18, 0.55), (side * 1.55, -0.60, 1.62)))
        p.append(ellipsoid((side * 1.95, -1.12, 1.32), (0.72, 0.32, 0.18), subdivisions=1))
    return p


def planned_rig_accessory_parts(spec: MiniatureSpec, plan: dict[str, Any]) -> tuple[list[trimesh.Trimesh], list[str]]:
    """Build extra native geometry requested by RigPlan.

    The archetype builders create the main body. This function consumes the
    planner's supported part vocabulary so a vision/text plan can add the
    subject-defining pieces instead of being collapsed back to the same default
    humanoid.
    """
    rig = plan.get("rig") if isinstance(plan, dict) else None
    raw_parts = rig.get("parts") if isinstance(rig, dict) else []
    built: list[trimesh.Trimesh] = []
    unsupported: list[str] = []
    skip_base_kinds = {
        "head",
        "torso",
        "pelvis",
        "left_arm",
        "right_arm",
        "left_leg",
        "right_leg",
        "beast_body",
        "beast_head",
        "beast_leg",
        "rider",
        "base",
        "scale_row",
        "claw",
        "tooth",
    }
    for raw in raw_parts if isinstance(raw_parts, list) else []:
        if not isinstance(raw, dict):
            continue
        kind = str(raw.get("kind") or "").strip().lower()
        primitive = str(raw.get("primitive") or "").strip().lower()
        if not kind or kind in skip_base_kinds:
            continue
        try:
            part_meshes = build_planned_part(kind, primitive, raw, spec)
        except Exception:
            part_meshes = []
        if part_meshes:
            built += part_meshes
        else:
            unsupported.append(kind or primitive or "unknown")
    # DetailPlan motifs can request important printable base/story detail even
    # when the RigPlan omitted explicit parts.
    motifs: list[str] = []
    details = plan.get("details") if isinstance(plan, dict) else None
    if isinstance(details, dict):
        motifs = [str(item).lower() for item in list(details.get("motifs") or [])]
    if any("skull" in motif or "rock" in motif or "bone" in motif for motif in motifs) and "skull_or_rock_base_detail" not in spec.detail_language:
        built += skull_and_rock_parts()
    return built, sorted(set(unsupported))


def build_planned_part(kind: str, primitive: str, raw: dict[str, Any], spec: MiniatureSpec) -> list[trimesh.Trimesh]:
    center = vector3(raw.get("center"), [0.0, 0.0, 10.0])
    scale = vector3(raw.get("scale"), [1.0, 1.0, 1.0])
    side = str(raw.get("side") or "").lower()
    sign = -1.0 if side.startswith("left") else 1.0
    if kind == "weapon":
        weapon = primitive or spec.weapon
        return planned_weapon_parts(weapon, center, scale)
    if kind == "shield":
        return shield_parts(center, scale)
    if kind == "cape":
        return cape_parts(center, scale)
    if kind == "banner":
        pole_x = center[0] or sign * 5.8
        return [capsule((pole_x, center[1], 5.0), (pole_x, center[1], 18.8), 0.12), box((1.9, 0.14, 3.0), (pole_x + sign * 0.95, center[1], 16.4))]
    if kind in {"hood", "helmet"}:
        return helmet_or_hood_parts(kind, center, scale)
    if kind == "face_detail":
        return face_detail_parts(center)
    if kind in {"tusk", "trophy_skull"}:
        return tusk_or_trophy_parts(kind, center, scale, sign)
    if kind in {"belt", "pouch", "chain", "tabard"}:
        return clothing_gear_parts(kind, center, scale, sign)
    if kind in {"plume", "kneepad", "gauntlet", "boot"}:
        return armor_accessory_parts(kind, center, scale, sign)
    if kind in {"wing_left", "wing_right"}:
        wing_sign = -1.0 if kind.endswith("left") else 1.0
        wing = ellipsoid((wing_sign * max(abs(center[0]), 4.5), center[1] or 1.2, center[2] or 15.0), (max(scale[0], 2.6), 0.34, max(scale[2], 4.5)), subdivisions=2)
        wing.apply_transform(trimesh.transformations.rotation_matrix(wing_sign * math.radians(18), [0, 0, 1], point=[wing_sign * 2.0, center[1] or 1.2, center[2] or 15.0]))
        return [wing]
    if kind == "tail":
        start = (center[0] - max(scale[0], 4.0) * 0.5, center[1], center[2])
        end = (center[0] + max(scale[0], 4.0) * 0.5, center[1] + 0.2, center[2] - 0.6)
        return [capsule(start, end, max(0.18, min(scale[1], scale[2], 0.65)))]
    if kind == "horn":
        return [capsule(center, (center[0] + sign * max(scale[0], 0.8), center[1] - 0.2, center[2] + max(scale[2], 1.0)), max(0.08, min(scale) * 0.16))]
    if kind == "shoulder_pad":
        return [ellipsoid((center[0] or sign * 3.7, center[1], center[2] or 15.7), (max(scale[0], 1.4), max(scale[1], 0.8), max(scale[2], 1.0)), subdivisions=2)]
    if kind == "armor_plate":
        return [box((max(scale[0], 0.8), max(scale[1], 0.12), max(scale[2], 0.18)), center)]
    if kind == "beast_armor_plate":
        return beast_armor_plate_parts(center, scale)
    if kind == "saddle":
        return [box((max(scale[0], 3.6), max(scale[1], 1.4), max(scale[2], 0.65)), (center[0], center[1], center[2] or 11.2))]
    if kind == "skull_or_rock_base_detail":
        return skull_and_rock_parts()
    if kind == "base_rubble":
        return bone_and_rubble_detail_stamps()
    return []


def shield_parts(center: tuple[float, float, float], scale: tuple[float, float, float]) -> list[trimesh.Trimesh]:
    p = [ellipsoid(center, (max(scale[0], 1.25), 0.22, max(scale[2], 2.0)), subdivisions=2)]
    p.append(box((max(scale[0], 1.0), 0.12, 0.16), (center[0], center[1] - 0.24, center[2] + max(scale[2], 2.0) * 0.35)))
    p.append(box((max(scale[0], 1.0), 0.12, 0.16), (center[0], center[1] - 0.24, center[2] - max(scale[2], 2.0) * 0.35)))
    p.append(ellipsoid((center[0], center[1] - 0.32, center[2]), (0.28, 0.07, 0.28), subdivisions=1))
    return p


def cape_parts(center: tuple[float, float, float], scale: tuple[float, float, float]) -> list[trimesh.Trimesh]:
    y = max(center[1], 2.5)
    p = [ellipsoid((center[0], y, center[2]), (max(scale[0], 3.0), 0.38, max(scale[2], 5.0)), subdivisions=2)]
    for x in np.linspace(-max(scale[0], 3.0) * 0.55, max(scale[0], 3.0) * 0.55, 5):
        p.append(capsule((center[0] + float(x), y - 0.18, center[2] + max(scale[2], 5.0) * 0.48), (center[0] + float(x * 1.25), y + 0.32, center[2] - max(scale[2], 5.0) * 0.48), 0.07))
    return p


def helmet_or_hood_parts(kind: str, center: tuple[float, float, float], scale: tuple[float, float, float]) -> list[trimesh.Trimesh]:
    z = center[2] or 18.7
    p = [ellipsoid((center[0], center[1] - 0.1, z), (max(scale[0], 1.45), max(scale[1], 1.05), max(scale[2], 1.35)), subdivisions=2)]
    if kind == "hood":
        p.append(ellipsoid((center[0], center[1] + 0.18, z - 0.15), (max(scale[0], 1.65), 0.38, max(scale[2], 1.55)), subdivisions=2))
    else:
        p.append(box((max(scale[0], 1.25), 0.16, 0.18), (center[0], center[1] - 0.96, z + 0.2)))
        p.append(box((0.18, 0.15, max(scale[2], 1.1)), (center[0], center[1] - 1.02, z - 0.05)))
    return p


def face_detail_parts(center: tuple[float, float, float]) -> list[trimesh.Trimesh]:
    z = center[2] or 18.6
    return [
        ellipsoid((center[0] - 0.38, center[1] - 1.04, z + 0.22), (0.07, 0.045, 0.06), subdivisions=1),
        ellipsoid((center[0] + 0.38, center[1] - 1.04, z + 0.22), (0.07, 0.045, 0.06), subdivisions=1),
        capsule((center[0] - 0.32, center[1] - 1.06, z - 0.42), (center[0] + 0.32, center[1] - 1.06, z - 0.42), 0.045),
    ]


def tusk_or_trophy_parts(kind: str, center: tuple[float, float, float], scale: tuple[float, float, float], sign: float) -> list[trimesh.Trimesh]:
    if kind == "trophy_skull":
        return skull_and_rock_parts()[:3]
    z = center[2] or 18.1
    return [
        capsule((center[0] - 0.42, center[1] - 1.06, z - 0.34), (center[0] - 0.72, center[1] - 1.18, z - 0.05), max(0.045, min(scale) * 0.08)),
        capsule((center[0] + 0.42, center[1] - 1.06, z - 0.34), (center[0] + 0.72, center[1] - 1.18, z - 0.05), max(0.045, min(scale) * 0.08)),
    ]


def clothing_gear_parts(kind: str, center: tuple[float, float, float], scale: tuple[float, float, float], sign: float) -> list[trimesh.Trimesh]:
    if kind == "belt":
        return [box((max(scale[0], 3.6), 0.18, 0.18), (center[0], center[1] - 1.85, center[2] or 10.9)), ellipsoid((center[0], center[1] - 1.98, center[2] or 10.9), (0.22, 0.06, 0.18), subdivisions=1)]
    if kind == "pouch":
        return [box((0.58, 0.18, 0.72), (center[0] or sign * 2.2, center[1] - 1.95, center[2] or 10.25))]
    if kind == "chain":
        return [ellipsoid((center[0] + float(x), center[1] - 2.05, center[2] + 0.1 * math.sin(i)), (0.12, 0.055, 0.08), subdivisions=1) for i, x in enumerate(np.linspace(-1.8, 1.8, 9))]
    if kind == "tabard":
        return [box((max(scale[0], 1.25), 0.16, max(scale[2], 3.2)), (center[0], center[1] - 2.08, center[2] or 8.8))]
    return []


def armor_accessory_parts(kind: str, center: tuple[float, float, float], scale: tuple[float, float, float], sign: float) -> list[trimesh.Trimesh]:
    if kind == "plume":
        return [capsule((center[0], center[1], center[2] or 19.5), (center[0], center[1] + 0.2, (center[2] or 19.5) + max(scale[2], 1.6)), 0.08)]
    if kind == "kneepad":
        return [ellipsoid((center[0] or sign * 1.25, center[1] - 0.88, center[2] or 6.0), (0.42, 0.12, 0.34), subdivisions=1)]
    if kind == "gauntlet":
        return [ellipsoid((center[0] or sign * 4.2, center[1] - 1.15, center[2] or 9.2), (0.42, 0.18, 0.34), subdivisions=1)]
    if kind == "boot":
        return [box((0.92, 0.22, 0.18), (center[0] or sign * 1.55, center[1] - 0.92, center[2] or 1.45))]
    return []


def beast_armor_plate_parts(center: tuple[float, float, float], scale: tuple[float, float, float]) -> list[trimesh.Trimesh]:
    p: list[trimesh.Trimesh] = []
    width = max(scale[0], 1.2)
    for i, x in enumerate(np.linspace(center[0] - width, center[0] + width, 5)):
        p.append(ellipsoid((float(x), center[1] or -2.45, center[2] + 0.16 * math.sin(i)), (0.46, 0.08, 0.26), subdivisions=1))
    return p


def planned_weapon_parts(weapon: str, center: tuple[float, float, float], scale: tuple[float, float, float]) -> list[trimesh.Trimesh]:
    weapon = weapon.lower()
    if any(term in weapon for term in ("sword", "blade", "khopesh", "scimitar")):
        return [capsule((center[0], center[1], center[2] - max(scale[2], 2.5)), (center[0] + 0.8, center[1], center[2] + max(scale[2], 2.5)), 0.13), box((0.85, 0.16, max(scale[2], 1.7)), (center[0] + 0.95, center[1], center[2] + max(scale[2], 2.7)))]
    if any(term in weapon for term in ("rifle", "gun", "cannon", "bolter")):
        return [box((max(scale[0], 5.0), 0.38, 0.62), center), capsule((center[0] + 1.8, center[1], center[2]), (center[0] + max(scale[0], 5.0), center[1], center[2]), 0.18)]
    if any(term in weapon for term in ("staff", "spear", "lance")):
        return [capsule((center[0], center[1], center[2] - max(scale[2], 5.0)), (center[0], center[1], center[2] + max(scale[2], 5.0)), 0.15)]
    if "axe" in weapon:
        return [capsule((center[0], center[1], center[2] - max(scale[2], 4.0)), (center[0], center[1], center[2] + max(scale[2], 4.0)), 0.18), box((1.9, 0.24, 1.2), (center[0], center[1], center[2] + max(scale[2], 4.0)))]
    return [box((max(scale[0], 1.1), max(scale[1], 0.22), max(scale[2], 1.8)), center)]


def native_sculpt_detail_parts(spec: MiniatureSpec, plan: dict[str, Any]) -> tuple[list[trimesh.Trimesh], list[str]]:
    """Add printable secondary/tertiary sculpt features as actual geometry.

    The fused mesh pass should preserve intentional miniature language: armor
    trims, straps, rivets, scales, claws, saddle hardware, cloth folds, and base
    storytelling. These are deterministic stamps, not random noise, so outputs
    read like hand-sculpted miniatures instead of noisy height fields.
    """
    details = set(spec.detail_language)
    motifs: set[str] = set()
    detail_plan = plan.get("details") if isinstance(plan, dict) else None
    if isinstance(detail_plan, dict):
        motifs = {str(item).lower() for item in list(detail_plan.get("motifs") or [])}
    p: list[trimesh.Trimesh] = []
    layers: list[str] = []

    armor = armor_detail_stamps(spec)
    if armor:
        p += armor
        layers.append("native_detail_stamps:armor_trim_rivets_straps")

    anatomy = anatomical_readability_stamps(spec)
    if anatomy:
        p += anatomy
        layers.append("native_detail_stamps:face_hands_fingers")

    if spec.archetype == "mounted_beast" or any(tag in details for tag in ("scales", "spine_spikes", "armored_beast_plates")):
        creature = beast_detail_stamps()
        if creature:
            p += creature
            layers.append("native_detail_stamps:scales_claws_spines")

    if "cape_or_cloak" in details or any("cloth" in motif or "cloak" in motif or "cape" in motif for motif in motifs):
        cloth = cloth_detail_stamps()
        if cloth:
            p += cloth
            layers.append("native_detail_stamps:cloth_folds_tattered_edges")

    if "saddle" in details or any("saddle" in motif or "leather" in motif or "strap" in motif for motif in motifs):
        saddle = saddle_detail_stamps()
        if saddle:
            p += saddle
            layers.append("native_detail_stamps:saddle_straps_hardware")

    if spec.weapon:
        weapon = weapon_detail_stamps(spec.weapon)
        if weapon:
            p += weapon
            layers.append("native_detail_stamps:weapon_edges_fullers")

    if "space_marine_silhouette" in details or "bolter_rifle" in details or "power_pack" in details:
        marine = space_marine_detail_stamps()
        if marine:
            p += marine
            layers.append("native_detail_stamps:space_marine_iconography")

    if "skull_or_rock_base_detail" in details or any("skull" in motif or "bone" in motif or "rock" in motif for motif in motifs):
        base = skull_and_rock_parts() + bone_and_rubble_detail_stamps()
        if base:
            p += base
            layers.append("native_detail_stamps:skulls_bones_rubble")

    return p, layers


def armor_detail_stamps(spec: MiniatureSpec) -> list[trimesh.Trimesh]:
    p: list[trimesh.Trimesh] = []
    # Chest and abdomen plate borders: wide enough to survive 0.16-0.22 mm fusion.
    for z, width in ((15.75, 4.6), (14.85, 4.9), (13.8, 4.35), (12.65, 3.8), (11.55, 3.25)):
        p.append(box((width, 0.18, 0.16), (0.0, -2.34, z)))
        for x in np.linspace(-width * 0.42, width * 0.42, max(3, int(width * 1.35))):
            p.append(ellipsoid((float(x), -2.48, z + 0.23), (0.09, 0.055, 0.09), subdivisions=1))
    # Vertical armor seams and straps break up the generic torso.
    for x in (-1.55, 0.0, 1.55):
        p.append(capsule((x, -2.42, 11.4), (x * 0.55, -2.42, 15.9), 0.055))
    p.append(ellipsoid((0.0, -2.55, 15.35), (0.42, 0.07, 0.42), subdivisions=1))
    for side in (-1, 1):
        # Shoulder trim, elbow plates, gauntlet bands.
        p.append(capsule((side * 2.75, -0.98, 16.35), (side * 5.0, -0.98, 16.1), 0.07))
        p.append(capsule((side * 4.72, -0.98, 12.15), (side * 3.45, -1.22, 9.25), 0.06))
        for z in (10.2, 11.1, 12.0, 13.0):
            p.append(ellipsoid((side * 4.15, -1.42, z), (0.10, 0.055, 0.10), subdivisions=1))
        for z in (4.2, 5.4, 6.7, 7.85):
            p.append(capsule((side * 0.75, -0.98, z), (side * 1.85, -0.98, z + 0.35), 0.055))
        # Boot sole and toe ridges read clearly at miniature scale.
        p.append(box((1.55, 0.18, 0.16), (side * 1.55, -0.86, 1.58)))
        p.append(ellipsoid((side * 1.92, -0.95, 1.33), (0.38, 0.10, 0.08), subdivisions=1))
    if spec.armor_style == "power_armor":
        for side in (-1, 1):
            p.append(box((0.18, 0.14, 1.25), (side * 2.85, -2.42, 14.35)))
            p.append(box((0.18, 0.14, 0.95), (side * 2.15, -2.42, 12.95)))
    return p


def anatomical_readability_stamps(spec: MiniatureSpec) -> list[trimesh.Trimesh]:
    """Add face/helmet and hand landmarks that survive 32 mm resin scale.

    Store/studio miniature failures were reading as generic armored blobs even
    when triangle count was high. These stamps are deliberately exaggerated,
    printable forms: visor slits, respirator/teeth, thumb/finger clusters, and
    weapon-hand contact points.
    """
    p: list[trimesh.Trimesh] = []
    if spec.archetype == "mounted_beast":
        # Rider face and hands after the mounted rig's 0.56 scale/translation.
        head_center = (0.15, -0.29, 19.29)
        hand_centers = [(3.0, -0.72, 15.15), (-2.7, -0.72, 15.05)]
    else:
        head_center = (0.0, -0.25, 18.55)
        hand_centers = [(4.9, -1.08, 9.15), (-4.9, -1.08, 9.15)]

    hx, hy, hz = head_center
    # Helmet/face plane, brow, eye slit, nose/respirator, mouth/tusks if orc.
    p.append(box((1.10, 0.18, 0.18), (hx, hy - 0.93, hz + 0.38)))
    p.append(box((0.96, 0.16, 0.14), (hx, hy - 1.01, hz + 0.12)))
    p.append(capsule((hx, hy - 1.04, hz - 0.06), (hx, hy - 1.04, hz - 0.46), 0.09))
    for sx in (-1, 1):
        p.append(ellipsoid((hx + sx * 0.36, hy - 1.05, hz + 0.18), (0.13, 0.08, 0.10), subdivisions=1))
        p.append(box((0.24, 0.14, 0.16), (hx + sx * 0.28, hy - 1.08, hz - 0.36)))
        if spec.archetype == "orc_warrior" or "tusks" in spec.detail_language:
            p.append(capsule((hx + sx * 0.32, hy - 1.08, hz - 0.43), (hx + sx * 0.62, hy - 1.24, hz - 0.10), 0.08))

    for index, (x, y, z) in enumerate(hand_centers):
        side = 1 if index == 0 else -1
        p.append(ellipsoid((x, y, z), (0.38, 0.24, 0.32), subdivisions=1))
        for finger in range(4):
            dz = (finger - 1.5) * 0.13
            p.append(capsule((x + side * 0.04, y - 0.16, z + dz), (x + side * 0.48, y - 0.30, z + dz + 0.05), 0.08))
        p.append(capsule((x - side * 0.08, y - 0.12, z - 0.24), (x - side * 0.38, y - 0.26, z - 0.42), 0.09))

    # Large readable knee/elbow pads and undercuts help painted-store quality.
    for side in (-1, 1):
        p.append(ellipsoid((side * 1.15, -0.96, 6.05), (0.48, 0.12, 0.36), subdivisions=1))
        p.append(capsule((side * 4.55, -1.20, 12.2), (side * 4.0, -1.30, 11.2), 0.07))
    return p


def space_marine_silhouette_parts() -> list[trimesh.Trimesh]:
    p: list[trimesh.Trimesh] = []
    # Big rounded pauldrons and rear power pack are the fastest-read silhouette
    # cues for this archetype; keep them broad enough to survive voxel fusion.
    for side in (-1, 1):
        p.append(ellipsoid((side * 4.25, -0.12, 16.05), (2.15, 1.25, 1.45), subdivisions=2))
        p.append(box((1.95, 0.22, 0.22), (side * 4.25, -1.30, 16.70)))
        p.append(box((1.80, 0.20, 0.20), (side * 4.25, -1.32, 15.45)))
    p.append(box((3.2, 1.15, 3.4), (0.0, 2.95, 14.2)))
    for side in (-1, 1):
        p.append(capsule((side * 1.05, 3.22, 15.4), (side * 1.05, 3.22, 18.25), 0.34))
        p.append(ellipsoid((side * 1.05, 3.22, 18.55), (0.48, 0.34, 0.30), subdivisions=1))
    p.append(box((2.1, 0.30, 1.35), (0.0, -2.42, 14.45)))
    return p


def space_marine_detail_stamps() -> list[trimesh.Trimesh]:
    p: list[trimesh.Trimesh] = []
    # Helmet visor/respirator, chest eagle-like emblem, backpack vents, bolter rails.
    p.append(box((1.20, 0.16, 0.16), (0.0, -1.22, 18.78)))
    for side in (-1, 1):
        p.append(box((0.22, 0.12, 0.48), (side * 0.34, -1.26, 18.28)))
        p.append(capsule((side * 0.25, -2.56, 15.15), (side * 1.45, -2.56, 15.72), 0.055))
        p.append(capsule((side * 0.25, -2.56, 14.72), (side * 1.45, -2.56, 14.18), 0.055))
        for z in (13.35, 14.05, 14.75, 15.45):
            p.append(box((0.18, 0.16, 0.40), (side * 1.28, 3.66, z)))
        for x in np.linspace(4.2, 8.4, 5):
            p.append(box((0.20, 0.10, 0.72), (float(x), -2.25, 12.1)))
    p.append(box((1.10, 0.14, 0.18), (0.0, -2.58, 15.02)))
    p.append(box((0.16, 0.14, 1.05), (0.0, -2.60, 14.75)))
    return p


def beast_detail_stamps() -> list[trimesh.Trimesh]:
    p: list[trimesh.Trimesh] = []
    # Overlapping flank scale rows and dorsal spines; intentionally ordered and readable.
    for row, y in enumerate((-2.35, -1.95, 1.95, 2.35)):
        for i, x in enumerate(np.linspace(-7.2, 7.1, 15)):
            z = 8.8 + row * 0.22 + math.sin(i * 0.9) * 0.18
            p.append(ellipsoid((float(x), y, z), (0.30, 0.075, 0.19), subdivisions=1))
    for i, x in enumerate(np.linspace(-7.2, 8.4, 13)):
        p.append(capsule((float(x), -0.02, 10.95), (float(x) + 0.12, -0.02, 12.0 + 0.28 * math.sin(i)), 0.075))
    # Claws/talons on four visible feet.
    for x in (-5.45, -1.75, 2.75, 6.05):
        for y in (-1.38, 1.38):
            for dx in (-0.24, 0.0, 0.24):
                p.append(capsule((x + dx, y - 0.18 * math.copysign(1, y), 1.35), (x + dx * 1.4, y - 0.62 * math.copysign(1, y), 1.22), 0.055))
    # Head teeth/horns so reptilian mounts do not read as smooth blobs.
    for side in (-1, 1):
        p.append(capsule((-8.15, side * 0.48, 10.92), (-9.25, side * 0.82, 11.55), 0.07))
        for i in range(4):
            p.append(capsule((-9.05 + i * 0.28, side * 0.55, 9.65), (-9.15 + i * 0.28, side * 0.78, 9.25), 0.04))
    return p


def cloth_detail_stamps() -> list[trimesh.Trimesh]:
    p: list[trimesh.Trimesh] = []
    for i, x in enumerate(np.linspace(-2.75, 2.75, 7)):
        p.append(capsule((float(x), 2.18, 16.2), (float(x * 1.22), 2.95, 5.0 + (i % 2) * 0.55), 0.075))
    for i, x in enumerate(np.linspace(-3.0, 3.0, 9)):
        p.append(ellipsoid((float(x), 3.05, 4.5 + (i % 3) * 0.18), (0.22, 0.08, 0.13), subdivisions=1))
    return p


def saddle_detail_stamps() -> list[trimesh.Trimesh]:
    p: list[trimesh.Trimesh] = []
    p.append(box((4.1, 0.18, 0.18), (0.0, -1.74, 11.38)))
    p.append(box((4.1, 0.18, 0.18), (0.0, 1.74, 11.38)))
    for x in (-1.8, 0.0, 1.8):
        p.append(capsule((x, -2.05, 10.95), (x, 2.05, 10.8), 0.06))
    for x in np.linspace(-2.2, 2.2, 6):
        p.append(ellipsoid((float(x), -1.92, 11.58), (0.08, 0.055, 0.08), subdivisions=1))
        p.append(ellipsoid((float(x), 1.92, 11.58), (0.08, 0.055, 0.08), subdivisions=1))
    return p


def weapon_detail_stamps(weapon: str) -> list[trimesh.Trimesh]:
    weapon = weapon.lower()
    p: list[trimesh.Trimesh] = []
    if any(term in weapon for term in ("sword", "blade", "sidearm")):
        p.append(box((0.18, 0.08, 2.15), (5.72, -1.34, 15.95)))
        p.append(box((1.25, 0.12, 0.16), (5.55, -1.34, 14.55)))
        p.append(capsule((5.2, -1.34, 7.6), (5.6, -1.34, 9.4), 0.07))
    elif any(term in weapon for term in ("rifle", "gun", "cannon", "bolter")):
        for x in np.linspace(4.1, 8.2, 6):
            p.append(box((0.18, 0.08, 0.72), (float(x), -2.04, 12.12)))
        p.append(capsule((8.5, -2.04, 11.78), (9.1, -2.04, 11.78), 0.12))
    elif any(term in weapon for term in ("staff", "spear", "lance")):
        p.append(capsule((-5.2, -1.08, 18.55), (-5.2, -1.08, 20.2), 0.08))
        p.append(ellipsoid((-5.2, -1.08, 20.15), (0.42, 0.16, 0.58), subdivisions=1))
    elif "axe" in weapon:
        p.append(box((0.35, 0.08, 1.1), (5.0, -1.55, 16.25)))
        p.append(capsule((5.0, -1.55, 15.55), (5.0, -1.55, 16.9), 0.07))
    return p


def bone_and_rubble_detail_stamps() -> list[trimesh.Trimesh]:
    p: list[trimesh.Trimesh] = []
    for i, a in enumerate(np.linspace(0.2, 2 * math.pi - 0.2, 12)):
        r = 8.8 + (i % 3) * 0.7
        x = math.cos(float(a)) * r
        y = math.sin(float(a)) * r
        p.append(capsule((x - 0.32, y, 1.82), (x + 0.32, y + 0.16, 1.82), 0.055))
        p.append(ellipsoid((x + 0.46, y + 0.2, 1.86), (0.10, 0.07, 0.07), subdivisions=1))
    return p


def skull_and_rock_parts() -> list[trimesh.Trimesh]:
    p: list[trimesh.Trimesh] = []
    for i, x in enumerate(np.linspace(-5.5, 5.5, 9)):
        y = -5.7 + (i % 3) * 0.55
        p.append(ellipsoid((float(x), y, 1.92), (0.38, 0.28, 0.24), subdivisions=1))
        if i % 2 == 0:
            p.append(ellipsoid((float(x) - 0.13, y - 0.18, 2.02), (0.055, 0.04, 0.04), subdivisions=1))
            p.append(ellipsoid((float(x) + 0.13, y - 0.18, 2.02), (0.055, 0.04, 0.04), subdivisions=1))
    return p


def vector3(value: Any, fallback: list[float]) -> tuple[float, float, float]:
    items = list(value) if isinstance(value, (list, tuple)) else list(fallback)
    items = (items + fallback)[:3]
    return (float(items[0]), float(items[1]), float(items[2]))


def prompt_requests_creature_claws(prompt: str) -> bool:
    prompt = prompt.lower()
    if "fingers/claws" in prompt or "fingers or claws" in prompt or "fingers and claws" in prompt:
        return any(term in prompt for term in ("beast", "dragon", "monster", "creature", "reptile", "talon", "talons"))
    return any(term in prompt for term in ("claw", "claws", "talon", "talons"))


def mounted_beast_parts(spec: MiniatureSpec) -> list[trimesh.Trimesh]:
    p = [
        ellipsoid((0, 0, 8.2), (9.4, 3.25, 3.0)),
        ellipsoid((-7.4, -0.15, 10.2), (2.55, 1.45, 1.7)),
        capsule((6.8, 0, 7.8), (12.8, 0, 8.7), 0.62),
        capsule((10.8, 0, 8.2), (16.2, 0.2, 7.0), 0.34),
    ]
    for x in (-5.8, -2.1, 2.4, 5.7):
        for y in (-1.35, 1.35):
            p.append(capsule((x, y * 0.6, 7.1), (x + 0.35, y, 1.9), 0.46))
            p.append(ellipsoid((x + 0.35, y, 1.45), (0.95, 0.58, 0.28), subdivisions=1))
    rider = MiniatureSpec("armored_humanoid", 32, "rider", spec.armor_style, spec.weapon, "round_scenic", [])
    rider_parts = humanoid_parts(rider)[:18]
    for part in rider_parts:
        part.apply_scale(0.56)
        part.apply_translation([0.15, -0.15, 8.9])
    p += rider_parts
    if spec.weapon in {"sword", "axe", "staff"}:
        p.append(capsule((1.8, -0.95, 17.3), (3.3, -0.95, 23.0), 0.12))
        p.append(box((0.82, 0.16, 2.0), (3.45, -1.0, 23.2)))
    for i, x in enumerate(np.linspace(-8.4, 7.6, 16)):
        p.append(ellipsoid((float(x), -1.65, 9.9 + math.sin(i) * 0.25), (0.36, 0.12, 0.2), subdivisions=1))
        p.append(ellipsoid((float(x), 0.0, 11.2 + math.sin(i) * 0.2), (0.20, 0.15, 0.48), subdivisions=1))
    for x in np.linspace(-5.8, 5.6, 8):
        p.append(ellipsoid((float(x), -2.25, 8.7), (0.72, 0.16, 0.52), subdivisions=1))
    for i, x in enumerate(np.linspace(-7.5, 7.0, 10)):
        p.append(box((0.75, 0.10, 0.16), (float(x), -3.02, 10.2 + math.sin(i) * 0.25)))
    return p


def mech_parts() -> list[trimesh.Trimesh]:
    return [box((6.8, 4.0, 5.2), (0, 0, 9.0)), box((4.0, 2.5, 2.2), (0, -0.5, 13.0)), capsule((-2.2, 0, 1.8), (-2.7, 0, 6.8), 0.75), capsule((2.2, 0, 1.8), (2.7, 0, 6.8), 0.75), capsule((3.4, -0.5, 11.0), (8.5, -0.5, 11.0), 0.38)]


def base_detail_parts(radius: float) -> list[trimesh.Trimesh]:
    p = []
    for i in range(24):
        a = i * 2 * math.pi / 24
        r = radius * (0.42 + (i % 4) * 0.08)
        p.append(ellipsoid((math.cos(a) * r, math.sin(a) * r, 1.55), (0.36, 0.24, 0.16), subdivisions=1))
    return p


def fuse_and_remesh(mesh: trimesh.Trimesh, spec: MiniatureSpec) -> trimesh.Trimesh:
    pitch = float(os.environ.get("MESHMEND_SCULPT_FUSION_PITCH_MM", "0.16"))
    try:
        voxels = mesh.voxelized(pitch, max_iter=int(os.environ.get("MESHMEND_SCULPT_VOXELIZE_MAX_ITER", "30")))
    except ValueError:
        pitch = max(pitch * 1.45, 0.22 if spec.archetype == "mounted_beast" else pitch * 1.25)
        voxels = mesh.voxelized(pitch, max_iter=int(os.environ.get("MESHMEND_SCULPT_VOXELIZE_RETRY_MAX_ITER", "40")))
    try:
        from scipy.ndimage import binary_closing, binary_dilation, binary_fill_holes
        from trimesh.voxel import ops as voxel_ops

        extra_bridge_dilation = int(os.environ.get("MESHMEND_SCULPT_EXTRA_BRIDGE_DILATION", "0"))
        concept_primary = is_space_marine_spec(spec) or needs_concept_primary_rig(spec) or spec.archetype == "orc_warrior"
        base_fuse_iterations = 3 if spec.archetype == "mounted_beast" else 2 if concept_primary else 1
        base_close_iterations = 4 if spec.archetype == "mounted_beast" else 3 if concept_primary else 2
        fuse_iterations = base_fuse_iterations + extra_bridge_dilation
        close_iterations = base_close_iterations + extra_bridge_dilation
        matrix = binary_dilation(np.asarray(voxels.matrix, dtype=bool), iterations=fuse_iterations)
        matrix = binary_closing(matrix, iterations=close_iterations)
        matrix = binary_fill_holes(matrix)
        fused = voxel_ops.matrix_to_marching_cubes(matrix=np.pad(matrix, 2, constant_values=False), pitch=pitch)
    except Exception:
        fused = voxels.fill().marching_cubes
    if isinstance(fused, trimesh.Trimesh):
        try:
            fused.merge_vertices(); fused.remove_unreferenced_vertices(); fused.fix_normals()
        except Exception:
            pass
        return fused
    return mesh


def bridge_and_refuse_components(mesh: trimesh.Trimesh, spec: MiniatureSpec) -> trimesh.Trimesh:
    """Connect small fused islands back to the main miniature before validation.

    Detail stamps and story-base pieces are intentionally separate primitives
    before voxel fusion. On detailed image jobs, voxelization can still leave a
    few tiny islands, which then trips the high-detail single-component gate.
    Add printable support bridges and run one stronger fuse pass instead of
    misreporting the result as a concept-fidelity failure.
    """
    try:
        components = [part for part in mesh.split(only_watertight=False) if len(part.faces) > 20]
    except Exception:
        return mesh
    if len(components) <= 1:
        return mesh
    components.sort(key=lambda part: len(part.faces), reverse=True)
    main = components[0]
    bridges: list[trimesh.Trimesh] = []
    main_vertices = np.asarray(main.vertices, dtype=float)
    if len(main_vertices) == 0:
        return mesh
    default_bridge_radius = "0.36" if (is_space_marine_spec(spec) or needs_concept_primary_rig(spec) or spec.archetype == "orc_warrior") else "0.26"
    bridge_radius = float(os.environ.get("MESHMEND_SCULPT_COMPONENT_BRIDGE_RADIUS_MM", default_bridge_radius))
    for island in components[1:]:
        island_vertices = np.asarray(island.vertices, dtype=float)
        if len(island_vertices) == 0:
            continue
        island_center = island_vertices.mean(axis=0)
        main_anchor = main_vertices[np.argmin(np.linalg.norm(main_vertices - island_center[None, :], axis=1))]
        island_anchor = island_vertices[np.argmin(np.linalg.norm(island_vertices - main_anchor[None, :], axis=1))]
        if float(np.linalg.norm(main_anchor - island_anchor)) < bridge_radius * 1.5:
            continue
        bridges.append(capsule(tuple(main_anchor), tuple(island_anchor), bridge_radius))
    if not bridges:
        return mesh
    bridged = trimesh.util.concatenate([mesh] + bridges)
    previous_pitch = os.environ.get("MESHMEND_SCULPT_FUSION_PITCH_MM")
    previous_dilate = os.environ.get("MESHMEND_SCULPT_EXTRA_BRIDGE_DILATION")
    try:
        os.environ["MESHMEND_SCULPT_EXTRA_BRIDGE_DILATION"] = "1"
        if previous_pitch is None:
            os.environ["MESHMEND_SCULPT_FUSION_PITCH_MM"] = "0.18"
        return fuse_and_remesh(bridged, spec)
    except Exception:
        return bridged
    finally:
        if previous_pitch is None:
            os.environ.pop("MESHMEND_SCULPT_FUSION_PITCH_MM", None)
        else:
            os.environ["MESHMEND_SCULPT_FUSION_PITCH_MM"] = previous_pitch
        if previous_dilate is None:
            os.environ.pop("MESHMEND_SCULPT_EXTRA_BRIDGE_DILATION", None)
        else:
            os.environ["MESHMEND_SCULPT_EXTRA_BRIDGE_DILATION"] = previous_dilate


def add_readable_sculpt_detail(mesh: trimesh.Trimesh, spec: MiniatureSpec) -> trimesh.Trimesh:
    # Deterministic macro relief; this is sculpt hierarchy, not random surface noise.
    target = int(os.environ.get("MESHMEND_SCULPT_TARGET_FACES", "180000"))
    while len(mesh.faces) < target:
        v, f = trimesh.remesh.subdivide(mesh.vertices, mesh.faces)
        mesh = trimesh.Trimesh(vertices=v, faces=f, process=False)
    vertices = np.asarray(mesh.vertices, dtype=float)
    normals = np.asarray(mesh.vertex_normals, dtype=float)
    mins, maxs = vertices.min(axis=0), vertices.max(axis=0)
    ext = np.maximum(maxs - mins, 1e-6)
    c = (vertices - mins) / ext
    x, z = c[:, 0] - 0.5, c[:, 2]
    relief = np.zeros(len(vertices))
    torso = (z > 0.32) & (z < 0.78)
    legs = (z > 0.08) & (z < 0.42)
    panel_lines = (np.abs(np.sin((z * 30.0) * math.pi)) < 0.010) & torso
    raised_trim = (np.abs(np.sin((z * 13.0 + np.abs(x) * 4.0) * math.pi)) < 0.018) & (torso | legs)
    relief -= panel_lines.astype(float) * 0.35
    relief += raised_trim.astype(float) * 0.28
    mesh.vertices = vertices + normals * (relief * 0.055)[:, None]
    try:
        mesh.fix_normals()
    except Exception:
        pass
    return mesh


def normalize(mesh: trimesh.Trimesh, scale_mm: float) -> trimesh.Trimesh:
    vertices = np.asarray(mesh.vertices, dtype=float)
    ext = np.maximum(vertices.max(axis=0) - vertices.min(axis=0), 1e-6)
    mesh = mesh.copy()
    mesh.vertices = vertices * (scale_mm / float(ext[2]))
    vertices = np.asarray(mesh.vertices, dtype=float)
    mins, maxs = vertices.min(axis=0), vertices.max(axis=0)
    vertices[:, 0] -= (mins[0] + maxs[0]) * 0.5
    vertices[:, 1] -= (mins[1] + maxs[1]) * 0.5
    vertices[:, 2] -= mins[2]
    mesh.vertices = vertices
    return mesh


def quality_blockers(mesh: trimesh.Trimesh, spec: MiniatureSpec, components: int, plan: dict[str, Any] | None = None, layers: list[str] | None = None) -> list[str]:
    blockers = []
    if not bool(mesh.is_watertight):
        blockers.append("mesh_not_watertight")
    if components > 1:
        blockers.append(f"multiple_components_{components}")
    if len(mesh.faces) < int(os.environ.get("MESHMEND_SCULPT_MIN_FACES", "150000")):
        blockers.append("below_experimental_sculpt_face_floor")
    blockers += concept_realization_blockers(spec, plan or {}, layers or [])
    return blockers


def concept_realization_blockers(spec: MiniatureSpec, plan: dict[str, Any], layers: list[str]) -> list[str]:
    """Reject generic/template outputs when the plan requested visible landmarks."""
    details = plan.get("details") if isinstance(plan, dict) else None
    required = []
    if isinstance(details, dict):
        required = [str(item).lower() for item in list(details.get("required_landmarks") or [])]
    if not required:
        required = [str(item).lower() for item in spec.detail_language[:4]]
    layer_text = ";".join(layers).lower()
    missing = [landmark for landmark in required if not landmark_realized(landmark, spec, layer_text)]
    blockers: list[str] = []
    max_missing = int(os.environ.get("MESHMEND_SCULPT_MAX_MISSING_LANDMARKS", "1"))
    if len(missing) > max_missing:
        blockers.append("planned_landmarks_missing_" + "_".join(safe_blocker_token(item) for item in missing[:6]))
    detail_layers = [layer for layer in layers if layer.startswith("native_detail_stamps:")]
    min_detail_layers = int(os.environ.get("MESHMEND_SCULPT_MIN_NATIVE_DETAIL_LAYERS", "2"))
    if len(detail_layers) < min_detail_layers:
        blockers.append("concept_match_insufficient_native_detail_layers")
    if spec.archetype == "mounted_beast" and "native_detail_stamps:scales_claws_spines" not in layers:
        blockers.append("concept_match_missing_creature_detail_layer")
    return blockers


def landmark_realized(landmark: str, spec: MiniatureSpec, layer_text: str) -> bool:
    landmark = landmark.lower()
    if landmark in {"large_silhouette", "layered_armor", "deep_panel_lines", "raised_trim", "readable_face_or_helmet", "scenic_base"}:
        return True
    if landmark in {"space_marine_silhouette", "oversized_pauldrons", "power_pack", "helmet_visor", "chest_emblem", "bolter_rifle"}:
        return "space_marine_iconography" in layer_text or "space_marine" in layer_text
    if landmark in {"mounted_beast", "reptilian_mount", "quadruped_silhouette", "mounted_creature_silhouette"}:
        return spec.archetype == "mounted_beast" and "beast_anatomy" in layer_text
    if landmark in {"sword", "axe", "rifle", "spear_or_staff", "staff_or_focus", "weapon", "raised_blade", "blade_like", "daggers_or_short_blades"}:
        return "weapon" in layer_text or spec.weapon in landmark
    if landmark in {"robed_caster_silhouette", "hood_or_high_collar"}:
        return "robed_caster" in layer_text or "concept_primary_silhouette" in layer_text
    if landmark in {"lean_stealth_silhouette", "hood_or_mask"}:
        return "lean_stealth" in layer_text or "concept_primary_silhouette" in layer_text
    if landmark in {"heroic_knight_silhouette", "large_shield_or_banner", "crested_helmet"}:
        return "heroic_knight" in layer_text or "shield_or_banner" in layer_text
    if landmark in {"wings_or_large_back_silhouette", "angelic_wings", "demonic_horns", "halo_or_crown", "tail", "clawed_feet"}:
        return "winged_primary" in layer_text or "wings_or_large_back_silhouette" in layer_text
    if landmark in {"cape_or_cloak", "cloak", "cape", "cloth"}:
        return "cloth_folds" in layer_text or "concept_silhouette_accessories" in layer_text
    if landmark in {"scales", "claws", "spine_spikes", "spikes_or_horns", "armored_beast_plates"}:
        return "scales_claws_spines" in layer_text or "creature_detail" in layer_text
    if landmark in {"saddle", "leather", "strap"}:
        return "saddle" in layer_text
    if landmark in {"skull_or_rock_base_detail", "skull", "skulls", "rocks", "base_rubble", "busy_surface_detail_from_concept"}:
        return "skulls_bones_rubble" in layer_text or "base_story_detail" in layer_text
    if landmark in {"shield", "shield_or_banner", "banner", "large_side_accessory"}:
        return "planned_rig_parts" in layer_text or "concept_silhouette_accessories" in layer_text
    if landmark in {"tusks", "tusk", "orc", "ork"}:
        return spec.archetype == "orc_warrior" or "planned_rig_parts" in layer_text
    if landmark in {"hands_and_fingers", "fingers", "hands", "face_hands_fingers"}:
        return "face_hands_fingers" in layer_text
    return landmark in layer_text or landmark in {item.lower() for item in spec.detail_language}


def safe_blocker_token(value: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in value.lower()).strip("_")[:40] or "unknown"


def connected_components(mesh: trimesh.Trimesh) -> int:
    try:
        return len([part for part in mesh.split(only_watertight=False) if len(part.faces) > 20])
    except Exception:
        return 0


def is_space_marine_spec(spec: MiniatureSpec) -> bool:
    return "space_marine_silhouette" in spec.detail_language or "bolter_rifle" in spec.detail_language or "power_pack" in spec.detail_language


def ellipsoid(center, scale, subdivisions: int = 2) -> trimesh.Trimesh:
    m = trimesh.creation.icosphere(subdivisions=subdivisions, radius=1.0)
    m.apply_scale(scale); m.apply_translation(center)
    return m


def box(extents, center) -> trimesh.Trimesh:
    m = trimesh.creation.box(extents=extents); m.apply_translation(center); return m


def capsule(start, end, radius: float) -> trimesh.Trimesh:
    return trimesh.creation.cylinder(radius=radius, sections=28, segment=np.array([start, end], dtype=float))
