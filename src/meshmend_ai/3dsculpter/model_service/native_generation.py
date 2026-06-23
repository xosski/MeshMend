from __future__ import annotations

import math
import os
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import trimesh


ARCHETYPE_KNOWLEDGE: dict[str, dict[str, Any]] = {
    "high_elf_warrior": {
        "terms": ("high elf", "high-elf", "elven warrior", "elf warrior", "elven knight", "aelf", "aelves"),
        "silhouette": "tall slender fantasy warrior; pointed ears; elegant crested helm; refined armor",
        "landmarks": ("pointed ears", "helmet crest", "leaf/rune trim", "kite shield", "spear or long blade", "cape/tabard"),
    },
    "dwarf_warrior": {
        "terms": ("dwarf", "dwarven", "duardin"),
        "silhouette": "short broad stocky warrior; heavy beard; runic armor; axe or hammer",
        "landmarks": ("wide torso", "braided beard", "round shield", "axe/hammer", "rune plates"),
    },
    "orc_brute": {
        "terms": ("orc", "ork", "goblin brute", "greenskin"),
        "silhouette": "hunched muscular brute; tusks; crude spiked armor; cleaver/axe",
        "landmarks": ("tusks", "heavy jaw", "spikes", "scrap plates", "oversized weapon"),
    },
    "undead_warrior": {
        "terms": ("undead", "skeleton", "wight", "lich", "death knight", "zombie"),
        "silhouette": "skeletal or deathly warrior; skull face; ribs/bone plates; tattered cloth",
        "landmarks": ("skull head", "rib bones", "tattered cape", "bone shield", "grave base"),
    },
    "samurai_warrior": {
        "terms": ("samurai", "ronin", "ashigaru", "katana"),
        "silhouette": "lamellar armored warrior; kabuto-style crest; katana; skirt plates",
        "landmarks": ("helmet crest", "lamellar plates", "katana", "sode shoulder guards", "waist skirt plates"),
    },
    "viking_raider": {
        "terms": ("viking", "norse", "raider", "berserker"),
        "silhouette": "fur-cloaked raider; beard; round shield; axe; rugged armor",
        "landmarks": ("beard", "round shield", "axe", "fur cloak", "rune stones"),
    },
    "pirate_captain": {
        "terms": ("pirate", "privateer", "corsair", "buccaneer"),
        "silhouette": "swashbuckling coat; tricorn hat; cutlass; pistol; boots",
        "landmarks": ("tricorn hat", "long coat", "cutlass", "pistol", "sash"),
    },
    "robot_mech": {
        "terms": ("robot", "android", "mech", "cyborg", "automaton"),
        "silhouette": "mechanical figure; boxy armor panels; antenna/sensors; cable joints",
        "landmarks": ("square plates", "sensor visor", "antenna", "cables", "mechanical joints"),
    },
    "ranger_archer": {
        "terms": ("ranger", "archer", "bowman", "hunter", "scout"),
        "silhouette": "hooded light-armored scout; bow; quiver; cloak; lean stance",
        "landmarks": ("hood", "bow", "quiver", "cloak", "belt pouches"),
    },
    "lizardfolk_warrior": {
        "terms": ("lizardfolk", "lizardman", "saurus", "dragonborn", "reptilian warrior"),
        "silhouette": "reptilian humanoid; long snout; tail; scales; claws; primitive weapon",
        "landmarks": ("snout", "tail", "scales", "claws", "crest spines"),
    },
}


@dataclass(frozen=True)
class NativeGenerationReport:
    provider: str
    capability_tier: str
    subject_type: str
    generated_parts: list[str]
    definition_feature_count: int
    fused_solid: bool
    watertight: bool
    components: int
    faces: int
    vertices: int
    production_ready: bool
    issues: list[str]
    semantic_plan: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def generate_native_miniature(request: dict[str, Any], image_path: Path | None, output_dir: Path) -> tuple[trimesh.Trimesh, NativeGenerationReport]:
    """Compatibility wrapper for the staged archetype generator.

    The old implementation ended in ``build_armored_humanoid`` for most prompts,
    producing the cube/cylinder/sphere mannequin. Keep this public function from
    ever returning that primitive fallback path.
    """
    prompt = str(request.get("prompt") or "")
    scale_mm = requested_scale_mm(request)
    target_faces = int(float(request.get("target_polycount") or request.get("target_faces") or 100_000))
    try:
        from meshmend.studio import MiniatureSculptQualityGate, StagedMiniaturePipeline, StudioMiniatureSpec
    except Exception as exc:
        raise RuntimeError(f"GENERATION FAILED: archetype generator failed. Failing function name: generate_native_miniature. {exc}") from exc

    spec = StudioMiniatureSpec.from_prompt(prompt, scale_mm=scale_mm, target_faces=target_faces)
    try:
        result = StagedMiniaturePipeline(quality_gate=MiniatureSculptQualityGate()).generate(spec, candidates_per_category=1)
    except Exception as exc:
        raise RuntimeError(f"GENERATION FAILED: archetype generator failed. Failing function name: generate_native_miniature. {exc}") from exc
    mesh = result.mesh
    mesh.metadata["meshmend_native_generation"] = False
    mesh.metadata["meshmend_native_subject_type"] = str((mesh.metadata.get("meshmend_generation_trace") or {}).get("parsed_archetype") or spec.archetype)
    mesh.metadata["units"] = "mm"
    parts = list(mesh.metadata.get("studio_components", []))
    definition_feature_count = len(parts)
    components = connected_component_count(mesh)
    issues: list[str] = [] if result.quality_report.passed else list(result.quality_report.issues)
    report = NativeGenerationReport(
        provider="meshmend_staged_archetype_pipeline",
        capability_tier="staged_archetype_native_sculpt",
        subject_type=str(mesh.metadata["meshmend_native_subject_type"]),
        generated_parts=parts,
        definition_feature_count=definition_feature_count,
        fused_solid=bool(mesh.metadata.get("studio_solid_fused", False)),
        watertight=bool(mesh.is_watertight),
        components=components,
        faces=int(len(mesh.faces)),
        vertices=int(len(mesh.vertices)),
        production_ready=not issues,
        issues=issues,
        semantic_plan={"generation_trace": dict(mesh.metadata.get("meshmend_generation_trace") or {})},
    )
    (output_dir / "native_generation_report.json").write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    return mesh, report


def infer_subject_type(prompt: str, image_path: Path | None) -> str:
    if any(term in prompt for term in ("high elf", "high-elf", "elven warrior", "elf warrior", "elven knight", "aelf", "aelves")):
        return "high_elf_warrior"
    if any(term in prompt for term in ("lizardfolk", "lizardman", "saurus", "dragonborn", "reptilian warrior")):
        return "armored_humanoid"
    if any(term in prompt for term in ("mounted", "rider", "cavalry", "mount", "dragon", "beast", "lizard", "dinosaur")):
        return "mounted_creature"
    if any(term in prompt for term in ("vehicle", "tank", "bike", "chariot", "walker")):
        return "vehicle"
    if image_path is not None:
        try:
            from PIL import Image

            image = Image.open(image_path)
            width, height = image.size
            if height > 0 and width / height >= float(os.environ.get("MESHMEND_NATIVE_MOUNTED_IMAGE_ASPECT", "1.12")):
                return "mounted_creature"
        except Exception:
            pass
    return "armored_humanoid"


def lookup_semantic_archetype(prompt: str) -> dict[str, Any]:
    """Local prompt knowledge lookup for offline generation.

    This deliberately does not call the web or a paid API. It is a small bundled
    knowledge base that can later be replaced/augmented by an offline RAG/wiki or
    plugin provider.
    """
    matches: list[dict[str, Any]] = []
    for name, profile in ARCHETYPE_KNOWLEDGE.items():
        terms = tuple(profile.get("terms") or ())
        hits = [term for term in terms if term in prompt]
        if hits:
            matches.append(
                {
                    "archetype": name,
                    "matched_terms": hits,
                    "silhouette": profile.get("silhouette", ""),
                    "landmarks": list(profile.get("landmarks") or ()),
                }
            )
    if not matches:
        return {
            "archetype": "generic_prompt_driven_humanoid",
            "matched_terms": [],
            "silhouette": "generic humanoid refined by prompt landmarks",
            "landmarks": [],
            "lookup_source": "bundled_offline_archetype_knowledge",
        }
    primary = dict(matches[0])
    primary["all_matches"] = [dict(match) for match in matches]
    primary["lookup_source"] = "bundled_offline_archetype_knowledge"
    return primary


def build_mounted_creature(scale_mm: float) -> tuple[trimesh.Trimesh, list[str]]:
    parts: list[trimesh.Trimesh] = []
    names: list[str] = []

    base = trimesh.creation.cylinder(radius=1.0, height=1.2, sections=128)
    base.apply_scale([0.58 * scale_mm, 0.34 * scale_mm, 1.0])
    base.apply_translation([0.0, 0.0, 0.6])
    parts.append(base); names.append("oval_scenic_base")

    body = ellipsoid((0, 0, 8.2), (10.4, 4.2, 4.2), subdivisions=3)
    chest = ellipsoid((-4.8, -0.1, 9.0), (4.6, 4.1, 4.7), subdivisions=3)
    hips = ellipsoid((4.8, 0.0, 8.0), (4.8, 3.8, 3.9), subdivisions=3)
    neck = cylinder_between((-8.2, 0.0, 9.8), (-11.4, 0.0, 11.2), 1.25, sections=32)
    head = ellipsoid((-13.3, -0.15, 11.1), (3.9, 2.3, 2.7), subdivisions=3)
    jaw = ellipsoid((-15.0, -0.2, 10.4), (2.5, 1.4, 1.0), subdivisions=2)
    tail = tapered_chain([(8.2, 0, 8.3), (12.5, 0.1, 8.6), (16.0, 0.0, 9.0)], [0.95, 0.62], sections=28)
    parts.extend([body, chest, hips, neck, head, jaw, tail]); names.extend(["beast_body", "beast_chest", "beast_hips", "neck", "head", "jaw", "tail"])

    for x in (-5.2, -1.8, 3.2, 6.6):
        z0 = 1.25
        z1 = 6.8 + (0.8 if x < 0 else 0.0)
        y = -1.65 if x in (-5.2, 3.2) else 1.55
        upper = cylinder_between((x, y * 0.7, z1), (x + 0.7, y, 3.9), 0.72, sections=28)
        lower = cylinder_between((x + 0.7, y, 3.9), (x + 0.35, y * 1.08, z0), 0.58, sections=28)
        foot = ellipsoid((x - 0.25, y * 1.12, 1.55), (1.7, 0.95, 0.55), subdivisions=2)
        parts.extend([upper, lower, foot]); names.extend(["beast_leg", "beast_lower_leg", "clawed_foot"])

    # Rider and saddle.
    saddle = ellipsoid((-0.8, -0.05, 11.0), (4.2, 2.6, 0.7), subdivisions=2)
    rider_torso = ellipsoid((-1.0, -0.1, 15.0), (2.5, 1.55, 3.4), subdivisions=3)
    rider_head = ellipsoid((-1.3, -0.25, 18.5), (1.25, 0.95, 1.35), subdivisions=2)
    rider_helmet = trimesh.creation.cone(radius=0.95, height=1.25, sections=28)
    rider_helmet.apply_translation([-1.3, -0.25, 19.65])
    arm_l = cylinder_between((-2.2, -0.25, 16.2), (-4.2, -0.85, 14.0), 0.32, sections=18)
    arm_r = cylinder_between((0.0, -0.25, 16.2), (2.0, -1.0, 18.2), 0.32, sections=18)
    blade = build_curved_blade((2.7, -1.1, 19.2), scale=1.0)
    parts.extend([saddle, rider_torso, rider_head, rider_helmet, arm_l, arm_r, blade]); names.extend(["saddle", "rider_torso", "rider_head", "rider_helmet", "rider_left_arm", "rider_right_arm", "raised_blade"])

    # Explicit controlled detail: spikes, armor plates, rivets, straps, terrain.
    for x in np.linspace(-12.5, 10.0, 12):
        spike = trimesh.creation.cone(radius=0.28, height=1.35, sections=18)
        spike.apply_transform(trimesh.transformations.rotation_matrix(-math.pi / 2, [0, 1, 0]))
        spike.apply_translation([float(x), 0.0, 11.4 + math.sin(float(x)) * 0.45])
        parts.append(spike); names.append("back_spike")
    for x in np.linspace(-6.5, 5.8, 7):
        plate = box((1.55, 0.18, 0.72), (float(x), -2.28, 10.2 + math.sin(float(x) * 0.7) * 0.45))
        parts.append(plate); names.append("armor_plate")
        for dx in (-0.45, 0.45):
            rivet = ellipsoid((float(x) + dx, -2.41, 10.55), (0.16, 0.08, 0.16), subdivisions=1)
            parts.append(rivet); names.append("rivet")
    for x in np.linspace(-8.0, 8.0, 9):
        rock = ellipsoid((float(x), 3.0 * math.sin(float(x)), 1.35), (0.55, 0.42, 0.24), subdivisions=1)
        parts.append(rock); names.append("base_rock")

    for x in np.linspace(-11.8, 6.8, 11):
        for y in (-1.72, 1.72):
            scale = ellipsoid((float(x), y, 10.15 + 0.35 * math.sin(float(x))), (0.42, 0.16, 0.26), subdivisions=1)
            parts.append(scale); names.append("raised_creature_scale")
    for x in (-14.6, -14.2, -13.8):
        tooth = trimesh.creation.cone(radius=0.12, height=0.72, sections=12)
        tooth.apply_translation([x, -0.95, 9.65])
        parts.append(tooth); names.append("visible_tooth")
    for y in (-0.62, 0.62):
        horn = cylinder_between((-13.6, y, 12.65), (-15.8, y * 1.15, 13.7), 0.18, sections=14)
        parts.append(horn); names.append("head_horn")
    for x in (-5.2, -1.8, 3.2, 6.6):
        for y in (-1.95, 1.95):
            claw = trimesh.creation.cone(radius=0.16, height=0.72, sections=12)
            claw.apply_transform(trimesh.transformations.rotation_matrix(math.pi / 2, [0, 1, 0]))
            claw.apply_translation([x - 1.0, y, 1.35])
            parts.append(claw); names.append("toe_claw")
    for x in (-4.3, -0.7, 2.9):
        strap = box((0.34, 5.0, 0.32), (x, -0.02, 11.35))
        parts.append(strap); names.append("saddle_strap")

    return trimesh.util.concatenate(parts), names


def build_armored_humanoid(scale_mm: float, prompt: str = "") -> tuple[trimesh.Trimesh, list[str]]:
    raise RuntimeError(
        "GENERATION FAILED: archetype generator failed. "
        "Failing function name: build_armored_humanoid. Primitive humanoid mannequin fallback is disabled."
    )
    parts: list[trimesh.Trimesh] = []
    names: list[str] = []
    base = trimesh.creation.cylinder(radius=8.2, height=1.6, sections=96)
    base.apply_translation([0, 0, 0.8])
    parts.append(base); names.append("round_base")
    parts.extend([
        ellipsoid((0, 0, 12.5), (4.4, 2.5, 6.0), 3),
        ellipsoid((0, -0.15, 17.2), (1.7, 1.25, 1.9), 2),
        box((6.5, 0.55, 1.1), (0, -1.65, 14.8)),
        cylinder_between((-1.6, 0, 1.6), (-1.2, 0, 9.0), 0.62),
        cylinder_between((1.6, 0, 1.6), (1.2, 0, 9.0), 0.62),
        cylinder_between((-3.4, 0, 14.5), (-5.4, -0.4, 9.0), 0.48),
        cylinder_between((3.4, 0, 14.5), (5.4, -0.4, 9.0), 0.48),
        box((5.5, 0.45, 0.45), (0, -2.2, 11.0)),
    ])
    names.extend(["torso", "helmet", "chest_trim", "left_leg", "right_leg", "left_arm", "right_arm", "weapon"])
    for side in (-1.0, 1.0):
        shoulder = ellipsoid((side * 3.55, -0.05, 15.2), (1.55, 1.25, 1.15), 2)
        kneepad = ellipsoid((side * 1.35, -0.45, 6.0), (0.85, 0.48, 0.75), 1)
        boot = ellipsoid((side * 1.55, -0.3, 1.8), (0.95, 0.6, 0.45), 1)
        parts.extend([shoulder, kneepad, boot])
        names.extend(["large_shoulder_pad", "knee_armor", "boot"])
    backpack = box((2.3, 1.0, 3.2), (0, 1.95, 13.6))
    visor = box((1.55, 0.16, 0.28), (0, -1.28, 17.45))
    belt = box((4.5, 0.42, 0.45), (0, -1.55, 10.2))
    parts.extend([backpack, visor, belt])
    names.extend(["backpack", "helmet_visor", "utility_belt"])
    for z in (11.2, 12.8, 14.0):
        trim = box((4.9, 0.22, 0.16), (0, -1.72, z))
        parts.append(trim); names.append("recessed_readability_trim")
    for x in np.linspace(-1.8, 1.8, 5):
        rivet = ellipsoid((float(x), -1.92, 10.55), (0.16, 0.08, 0.16), subdivisions=1)
        parts.append(rivet); names.append("belt_rivet")
    for x in (-2.8, 2.8):
        for z in (12.0, 13.25, 14.5):
            plate = box((0.72, 0.20, 0.62), (x, -1.78, z))
            parts.append(plate); names.append("torso_armor_plate")
    for side in (-1.0, 1.0):
        for z in (3.9, 5.0, 7.2, 8.3):
            band = box((0.95, 0.18, 0.22), (side * 1.35, -0.78, z))
            parts.append(band); names.append("leg_armor_trim")
    for side in (-1.0, 1.0):
        for z in (11.2, 13.0):
            arm_trim = ellipsoid((side * 4.55, -0.45, z), (0.42, 0.18, 0.24), subdivisions=1)
            parts.append(arm_trim); names.append("arm_armor_trim")
    for x in np.linspace(-2.2, 2.2, 7):
        chest_rivet = ellipsoid((float(x), -1.93, 13.8), (0.13, 0.07, 0.13), subdivisions=1)
        parts.append(chest_rivet); names.append("chest_rivet")
    for x in (-0.65, 0.65):
        exhaust = cylinder_between((x, 2.55, 14.2), (x, 2.55, 16.2), 0.26, sections=18)
        parts.append(exhaust); names.append("backpack_exhaust")
    for angle in np.linspace(0.0, 2.0 * math.pi, 18, endpoint=False):
        rubble = ellipsoid((6.4 * math.cos(float(angle)), 6.4 * math.sin(float(angle)), 1.65), (0.36, 0.24, 0.18), subdivisions=1)
        parts.append(rubble); names.append("scenic_base_rubble")
    detail_parts, detail_names = build_printable_detail_stamps("generic_armored_humanoid", prompt)
    parts.extend(detail_parts)
    names.extend(detail_names)
    prompt_parts, prompt_names = build_humanoid_prompt_features(prompt)
    parts.extend(prompt_parts)
    names.extend(prompt_names)
    semantic_parts, semantic_names = build_semantic_archetype_features(prompt)
    parts.extend(semantic_parts)
    names.extend(semantic_names)
    return trimesh.util.concatenate(parts), names


def build_high_elf_warrior(scale_mm: float, prompt: str = "") -> tuple[trimesh.Trimesh, list[str]]:
    """Build a legally distinct high-elf fantasy warrior silhouette.

    The generic armored humanoid reads like a bulky sci-fi soldier. High elves
    need a different first-read shape: tall/slender proportions, pointed ears,
    crested helm, elegant tabard/cape, kite shield, spear or long blade, and
    leaf/rune armor trim that survives STL export.
    """
    parts: list[trimesh.Trimesh] = []
    names: list[str] = []

    def add(mesh: trimesh.Trimesh, name: str) -> None:
        parts.append(mesh)
        names.append(name)

    base = trimesh.creation.cylinder(radius=7.4, height=1.35, sections=96)
    base.apply_translation([0, 0, 0.68])
    add(base, "round_display_base")

    # Slender heroic anatomy, intentionally narrower than power armor.
    add(ellipsoid((0.0, 0.0, 12.4), (2.75, 1.62, 5.65), 3), "slender_elven_torso")
    add(ellipsoid((0.0, -0.08, 17.25), (1.18, 0.92, 1.45), 2), "narrow_elven_head")
    add(ellipsoid((0.0, -0.25, 14.85), (3.55, 1.05, 1.05), 2), "elegant_breastplate")
    add(box((3.9, 0.26, 0.36), (0.0, -1.26, 10.45)), "thin_belt")
    add(box((2.0, 0.32, 4.9), (0.0, -1.38, 8.6)), "long_front_tabard")
    add(box((3.25, 0.34, 0.32), (0.0, -1.62, 15.25)), "bold_elven_breastplate_top_lip")
    add(box((2.55, 0.36, 0.28), (0.0, -1.66, 14.25)), "bold_elven_breastplate_lower_lip")
    for side in (-1.0, 1.0):
        diagonal = box((2.25, 0.34, 0.24), (side * 0.72, -1.72, 13.6))
        diagonal.apply_transform(trimesh.transformations.rotation_matrix(side * 0.48, [0, 1, 0], point=(side * 0.72, -1.72, 13.6)))
        add(diagonal, "bold_leaf_breastplate_diagonal_panel")
    for z in (8.15, 9.25, 10.35):
        add(box((2.55, 0.34, 0.18), (0.0, -1.72, z)), "bold_tabard_horizontal_trim")

    for side in (-1.0, 1.0):
        add(cylinder_between((side * 1.05, 0.0, 1.45), (side * 1.0, 0.0, 8.75), 0.42, sections=24), "slender_armored_leg")
        add(ellipsoid((side * 1.05, -0.22, 5.35), (0.64, 0.36, 0.95), 1), "pointed_knee_guard")
        add(ellipsoid((side * 1.22, -0.25, 1.45), (0.78, 0.48, 0.36), 1), "narrow_pointed_boot")
        add(cylinder_between((side * 2.55, 0.0, 14.6), (side * 4.25, -0.5, 10.35), 0.34, sections=20), "slender_arm")
        add(ellipsoid((side * 2.68, -0.05, 15.45), (0.95, 0.72, 0.68), 2), "leaf_shaped_pauldron")
        for z in (4.15, 5.45, 6.75):
            add(box((0.96, 0.28, 0.28), (side * 1.05, -0.92, z)), "bold_greave_plate_layer")
        for index, z in enumerate((15.35, 15.95, 16.55)):
            fin = box((1.7 - index * 0.22, 0.28, 0.22), (side * 3.03, -0.98, z))
            fin.apply_transform(trimesh.transformations.rotation_matrix(side * 0.28, [0, 0, 1], point=(side * 3.03, -0.98, z)))
            add(fin, "stacked_leaf_pauldron_plate")

    # Pointed ears and tall helm/crest are mandatory first-read landmarks.
    for side in (-1.0, 1.0):
        ear = trimesh.creation.cone(radius=0.24, height=1.35, sections=18)
        ear.apply_transform(trimesh.transformations.rotation_matrix(side * math.pi / 2, [0, 1, 0]))
        ear.apply_translation([side * 1.32, -0.08, 17.55])
        add(ear, "long_pointed_elf_ear")
        add(box((0.58, 0.16, 0.18), (side * 0.52, -1.03, 17.55)), "almond_eye_slit")
    add(trimesh.creation.cone(radius=0.92, height=2.65, sections=36), "tall_conical_elven_helm")
    parts[-1].apply_translation([0.0, -0.05, 19.2])
    add(box((1.95, 0.26, 0.22), (0.0, -1.02, 18.28)), "bold_helmet_brow_ridge")
    for side in (-1.0, 1.0):
        add(box((0.36, 0.22, 1.05), (side * 0.78, -1.04, 17.68)), "bold_helmet_cheek_guard")
    crest = box((0.28, 0.22, 3.25), (0.0, -0.12, 20.2))
    crest.apply_transform(trimesh.transformations.rotation_matrix(0.12, [1, 0, 0], point=(0.0, -0.12, 20.2)))
    add(crest, "high_helmet_crest")
    for z in (19.2, 20.05, 20.9):
        add(box((1.25, 0.18, 0.18), (0.0, -0.42, z)), "segmented_helmet_crest_plate")

    # Elegant fantasy weapons: spear for spear/lance prompts, otherwise long sword.
    wants_spear = any(term in prompt for term in ("spear", "lance", "halberd", "glaive"))
    if wants_spear:
        add(cylinder_between((4.95, -1.0, 4.0), (4.95, -1.0, 22.55), 0.16, sections=18), "long_elven_spear_shaft")
        add(cylinder_between((4.12, -0.46, 10.55), (4.95, -1.0, 11.65), 0.28, sections=18), "right_hand_gripping_spear")
        add(cylinder_between((3.72, -0.42, 13.1), (4.95, -1.0, 13.85), 0.22, sections=18), "upper_hand_spear_contact")
        for z in (10.8, 11.35, 13.2, 13.75):
            add(cylinder_between((4.55, -1.02, z), (5.35, -1.02, z), 0.065, sections=10), "visible_spear_grip_wrap")
        spearhead = trimesh.creation.cone(radius=0.56, height=1.8, sections=28)
        spearhead.apply_translation([4.95, -1.0, 22.25])
        add(spearhead, "leaf_spear_head")
        add(box((0.92, 0.18, 0.20), (4.95, -1.18, 21.28)), "spearhead_cross_guard")
    else:
        add(cylinder_between((5.0, -1.08, 7.6), (5.0, -1.08, 18.6), 0.13, sections=18), "long_elven_sword_grip")
        add(cylinder_between((4.12, -0.46, 10.55), (5.0, -1.08, 10.95), 0.28, sections=18), "right_hand_gripping_sword")
        blade = box((0.38, 0.16, 7.2), (5.0, -1.08, 18.2))
        add(blade, "long_straight_elven_blade")
        add(box((0.72, 0.20, 6.65), (5.0, -1.28, 18.2)), "raised_sword_center_ridge")
        pommel = ellipsoid((5.0, -1.08, 7.1), (0.38, 0.22, 0.32), 1)
        add(pommel, "gem_pommel")

    shield = ellipsoid((-4.85, -1.28, 11.95), (1.18, 0.28, 2.9), 2)
    add(shield, "tall_kite_shield")
    add(box((0.24, 0.12, 2.55), (-4.85, -1.62, 11.95)), "shield_center_ridge")
    add(box((1.55, 0.20, 0.22), (-4.85, -1.86, 14.15)), "bold_shield_top_trim")
    add(box((1.05, 0.22, 0.22), (-4.85, -1.88, 9.9)), "bold_shield_lower_trim")
    add(ellipsoid((-4.85, -1.94, 12.1), (0.44, 0.16, 0.44), 1), "large_shield_center_gem")

    # Cape and cloth folds for fantasy silhouette.
    add(box((4.8, 0.34, 7.6), (0.0, 1.86, 10.9)), "flowing_back_cape")
    for x in np.linspace(-1.8, 1.8, 5):
        add(cylinder_between((float(x), 1.98, 7.2), (float(x) * 0.45, 2.02, 15.0), 0.14, sections=10), "cape_vertical_fold")

    # Leaf/rune trim as raised geometry, not texture.
    for side in (-1.0, 1.0):
        for z in (11.35, 12.25, 13.15, 14.05, 14.95):
            leaf = ellipsoid((side * 1.58, -1.76, z), (0.42, 0.16, 0.24), 1)
            add(leaf, "raised_leaf_armor_motif")
        for z in (3.25, 4.15, 5.05, 6.15, 7.25, 8.15):
            add(box((0.82, 0.20, 0.20), (side * 1.02, -0.78, z)), "greave_rune_trim")
        for z in (10.9, 12.4, 13.9):
            add(cylinder_between((side * 1.95, -1.84, z - 0.42), (side * 1.95, -1.84, z + 0.42), 0.10, sections=10), "raised_elven_vine_scroll")
    for x in np.linspace(-1.35, 1.35, 7):
        add(ellipsoid((float(x), -1.66, 15.05), (0.18, 0.10, 0.18), 1), "breastplate_gem_rivet")
    for x in np.linspace(-1.15, 1.15, 5):
        add(box((0.28, 0.18, 0.95), (float(x), -1.74, 13.15)), "raised_breastplate_engraved_bar")
    for z in (9.0, 10.0):
        add(box((1.55, 0.20, 0.16), (0.0, -1.72, z)), "tabard_raised_border")
    for z in (8.25, 9.55, 10.85):
        add(ellipsoid((0.0, -1.78, z), (0.24, 0.12, 0.24), 1), "tabard_center_gem")
    for z in (11.0, 12.2, 13.4):
        add(box((0.72, 0.18, 0.18), (-4.85, -1.9, z)), "kite_shield_raised_chevron")
        add(box((0.72, 0.18, 0.18), (-4.85, -1.9, z + 0.48)), "kite_shield_raised_chevron")
    for z in np.linspace(7.2, 15.0, 8):
        add(cylinder_between((-2.0, 1.98, float(z)), (-1.25, 2.02, float(z) + 0.7), 0.16, sections=10), "cape_deep_fold_ridge")
        add(cylinder_between((2.0, 1.98, float(z)), (1.25, 2.02, float(z) + 0.7), 0.16, sections=10), "cape_deep_fold_ridge")

    # Scenic but non-branded base elements.
    for angle in np.linspace(0.0, 2.0 * math.pi, 14, endpoint=False):
        add(ellipsoid((5.7 * math.cos(float(angle)), 5.7 * math.sin(float(angle)), 1.45), (0.22, 0.16, 0.1), 1), "small_base_stone")
    detail_parts, detail_names = build_printable_detail_stamps("high_elf_warrior", prompt)
    for mesh, name in zip(detail_parts, detail_names):
        add(mesh, name)

    prompt_parts, prompt_names = build_humanoid_prompt_features(prompt)
    # Only add non-bulky prompt details; avoid power-armor contamination.
    excluded_prompt_fragments = (
        "power",
        "bolter",
        "pauldron",
        "staff",
        "shield",
        "cape",
        "sword",
        "axe",
    )
    filtered = [
        (mesh, name)
        for mesh, name in zip(prompt_parts, prompt_names)
        if not any(fragment in name for fragment in excluded_prompt_fragments)
    ]
    for mesh, name in filtered:
        add(mesh, name)
    return trimesh.util.concatenate(parts), names


def build_humanoid_prompt_features(prompt: str) -> tuple[list[trimesh.Trimesh], list[str]]:
    """Add first-read prompt landmarks so default native output is not generic.

    The base native humanoid is intentionally printable and stable, but by itself
    every text prompt reads as the same armored soldier. These additions are
    coarse, fused geometry landmarks rather than fine texture: they survive STL
    export and make mage/rogue/knight/demon/angel/shield/banner prompts visibly
    different before any downstream smoothing or voxel fusion.
    """
    prompt = prompt.lower()
    parts: list[trimesh.Trimesh] = []
    names: list[str] = []

    def add(mesh: trimesh.Trimesh, name: str) -> None:
        parts.append(mesh)
        names.append(name)

    wants_alien_bioform = any(
        term in prompt
        for term in (
            "termagant",
            "termagaunt",
            "tyranid",
            "hormagaunt",
            "gaunt alien",
            "insectoid alien",
            "chitin alien",
            "bioform",
            "fleshborer",
        )
    )
    wants_power_armor = any(
        term in prompt
        for term in (
            "space marine",
            "spacemarine",
            "spcace marine",
            "sapce marine",
            "spcae marine",
            "power armor",
            "power armour",
            "adeptus",
            "primaris",
            "sci fi armored soldier",
            "sci-fi armored soldier",
        )
    )
    wants_robes = any(term in prompt for term in ("wizard", "mage", "sorcerer", "witch", "warlock", "cleric", "priest", "robe", "robed"))
    wants_hood = wants_robes or any(term in prompt for term in ("hood", "hooded", "ranger", "rogue", "assassin"))
    wants_cape = wants_robes or any(term in prompt for term in ("cape", "cloak", "tattered", "fur cloak"))
    wants_rogue = any(term in prompt for term in ("rogue", "assassin", "ranger", "thief", "ninja", "dagger", "dual wield", "dual-wield"))
    wants_knight = any(term in prompt for term in ("knight", "paladin", "templar", "crusader", "champion"))
    wants_demon = any(term in prompt for term in ("demon", "devil", "fiend", "tiefling"))
    wants_angel = any(term in prompt for term in ("angel", "celestial", "seraph", "wing", "wings"))
    wants_shield = any(term in prompt for term in ("shield", "buckler"))
    wants_banner = any(term in prompt for term in ("banner", "standard", "flag"))
    wants_plague_doctor = any(term in prompt for term in ("plague doctor", "plague mask", "beaked mask", "bird mask", "doctor mask"))
    wants_staff = wants_robes or any(term in prompt for term in ("staff", "spear", "lance"))
    wants_axe = any(term in prompt for term in ("axe", "halberd"))
    wants_sword = not wants_alien_bioform and (wants_knight or any(term in prompt for term in ("sword", "blade", "katana", "khopesh", "scimitar")))

    if wants_alien_bioform:
        add(ellipsoid((0.0, -1.35, 17.45), (1.35, 1.15, 0.72), subdivisions=2), "prompt_alien_sloped_chitin_head")
        add(tapered_chain([(0.0, 2.25, 9.8), (0.0, 4.6, 7.2), (0.0, 6.4, 4.3)], [0.32, 0.14], sections=18), "prompt_alien_tapering_tail")
        for side in (-1.0, 1.0):
            add(cylinder_between((side * 2.6, -1.7, 13.7), (side * 4.9, -2.8, 8.4), 0.18, sections=14), "prompt_alien_scything_forelimb")
            add(cylinder_between((side * 1.25, -1.2, 8.2), (side * 2.55, -2.15, 3.2), 0.2, sections=14), "prompt_alien_front_clawed_leg")
            add(cylinder_between((side * 1.1, 1.0, 8.0), (side * 2.45, 1.85, 3.0), 0.2, sections=14), "prompt_alien_rear_clawed_leg")
        for z in (10.6, 12.0, 13.4):
            add(box((3.4, 0.22, 0.22), (0.0, -2.02, z)), "prompt_alien_ribbed_carapace_plate")
        add(box((2.1, 0.34, 0.42), (2.95, -2.45, 11.4)), "prompt_alien_fleshborer_bioweapon")

    if wants_plague_doctor:
        add(ellipsoid((0.0, -1.45, 18.35), (0.95, 1.45, 0.52), subdivisions=2), "prompt_long_plague_beak_mask")
        add(cylinder_between((0.0, -1.95, 18.25), (0.0, -3.45, 18.05), 0.32, sections=18), "prompt_projecting_beak_tip")
        for side in (-1.0, 1.0):
            add(ellipsoid((side * 0.48, -1.52, 18.62), (0.18, 0.08, 0.18), subdivisions=1), "prompt_round_goggle_eye")
        brim = trimesh.creation.cylinder(radius=1.72, height=0.18, sections=48)
        brim.apply_translation([0.0, -0.08, 19.52])
        add(brim, "prompt_wide_brim_hat")
        crown = trimesh.creation.cone(radius=0.92, height=1.25, sections=40)
        crown.apply_translation([0.0, -0.08, 20.2])
        add(crown, "prompt_tall_hat_crown")
        add(box((4.9, 0.36, 8.6), (0.0, 2.32, 10.9)), "prompt_long_doctor_coat")
        for side in (-1.0, 1.0):
            add(box((0.32, 0.18, 6.2), (side * 1.75, -1.85, 9.4)), "prompt_coat_front_fold")
        add(cylinder_between((5.1, -0.9, 7.2), (5.1, -0.9, 18.5), 0.14, sections=16), "prompt_doctor_cane")

    if wants_power_armor:
        for side in (-1.0, 1.0):
            add(ellipsoid((side * 4.25, -0.1, 15.25), (2.2, 1.45, 1.55), subdivisions=2), "prompt_massive_round_pauldron")
            add(ellipsoid((side * 1.45, -0.3, 4.65), (1.05, 0.62, 1.65), subdivisions=2), "prompt_chunky_greave")
            add(ellipsoid((side * 1.6, -0.45, 1.65), (1.28, 0.82, 0.5), subdivisions=1), "prompt_oversized_boot")
        add(box((2.7, 1.25, 4.0), (0.0, 2.35, 13.8)), "prompt_rear_power_backpack")
        for x in (-0.82, 0.82):
            add(cylinder_between((x, 3.05, 14.5), (x, 3.05, 17.4), 0.32, sections=18), "prompt_power_pack_exhaust")
        add(box((4.2, 0.52, 0.62), (0.0, -2.08, 13.7)), "prompt_raised_chest_emblem")
        add(box((7.2, 0.46, 0.58), (0.8, -2.52, 12.6)), "prompt_bolter_rifle_across_chest")
        add(cylinder_between((4.2, -2.52, 12.6), (7.3, -2.52, 12.6), 0.28, sections=18), "prompt_bolter_barrel")
        add(box((1.1, 0.5, 1.45), (2.65, -2.58, 11.75)), "prompt_bolter_magazine")

    if wants_robes:
        add(ellipsoid((0.0, -0.05, 9.2), (4.6, 2.35, 5.9), subdivisions=3), "prompt_robed_lower_silhouette")
        for side in (-1.0, 1.0):
            add(box((0.32, 0.16, 5.2), (side * 2.25, -1.88, 10.2)), "prompt_vertical_robe_fold")
        for z in (7.5, 9.4, 11.3):
            add(box((4.5, 0.18, 0.16), (0.0, -1.95, z)), "prompt_robe_trim_band")

    if wants_hood:
        hood = trimesh.creation.cone(radius=1.65, height=2.25, sections=32)
        hood.apply_translation([0.0, -0.2, 19.35])
        add(hood, "prompt_pointed_hood")

    if wants_cape:
        add(box((5.8, 0.42, 8.2), (0.0, 2.42, 11.6)), "prompt_back_cape_mass")
        for x in np.linspace(-2.1, 2.1, 5):
            add(cylinder_between((float(x), 2.74, 7.8), (float(x) * 0.55, 2.86, 15.5), 0.08, sections=10), "prompt_cape_fold_ridge")

    if wants_rogue:
        add(cylinder_between((-4.8, -1.25, 13.0), (-6.7, -1.7, 8.4), 0.14, sections=14), "prompt_left_dagger")
        add(cylinder_between((4.8, -1.25, 13.0), (6.7, -1.7, 8.4), 0.14, sections=14), "prompt_right_dagger")
        add(box((1.15, 0.32, 0.42), (-2.55, -1.95, 10.3)), "prompt_belt_pouch")
        add(box((1.15, 0.32, 0.42), (2.55, -1.95, 10.3)), "prompt_belt_pouch")

    if wants_knight:
        plume = trimesh.creation.cone(radius=0.38, height=2.6, sections=18)
        plume.apply_translation([0.0, -0.12, 20.7])
        add(plume, "prompt_crested_helmet_plume")
        add(box((2.2, 0.22, 3.5), (0.0, -2.08, 12.9)), "prompt_front_tabard")

    if wants_shield:
        shield = ellipsoid((-5.9, -1.65, 12.0), (1.45, 0.28, 2.45), subdivisions=2)
        add(shield, "prompt_large_side_shield")
        add(box((1.0, 0.12, 0.16), (-5.9, -1.98, 12.0)), "prompt_shield_boss_bar")

    if wants_banner:
        add(cylinder_between((4.7, 0.55, 9.2), (4.7, 0.55, 20.7), 0.13, sections=14), "prompt_banner_pole")
        add(box((2.8, 0.16, 3.7), (3.25, 0.52, 18.1)), "prompt_hanging_banner")

    if wants_staff:
        add(cylinder_between((5.6, -1.1, 7.0), (5.6, -1.1, 21.0), 0.13, sections=16), "prompt_tall_staff")
        add(ellipsoid((5.6, -1.1, 21.35), (0.55, 0.42, 0.55), subdivisions=2), "prompt_staff_orb")
    elif wants_axe:
        add(cylinder_between((5.6, -1.1, 8.6), (5.6, -1.1, 19.5), 0.16, sections=16), "prompt_axe_handle")
        add(box((1.9, 0.22, 1.35), (6.25, -1.1, 18.75)), "prompt_axe_head")
    elif wants_sword:
        sword = build_curved_blade((5.8, -1.1, 14.9), scale=0.95)
        add(sword, "prompt_raised_sword")

    if wants_demon:
        for side in (-1.0, 1.0):
            add(cylinder_between((side * 0.72, -0.1, 19.35), (side * 1.55, -0.08, 20.95), 0.15, sections=14), "prompt_demonic_horn")
        add(tapered_chain([(1.8, 1.7, 8.5), (3.6, 2.5, 6.4), (5.3, 2.1, 5.2)], [0.22, 0.14], sections=14), "prompt_demonic_tail")

    if wants_angel:
        left_wing = box((0.38, 0.34, 7.2), (-3.45, 2.34, 14.7))
        left_wing.apply_transform(trimesh.transformations.rotation_matrix(0.42, [0, 1, 0], point=(-3.45, 2.34, 14.7)))
        right_wing = box((0.38, 0.34, 7.2), (3.45, 2.34, 14.7))
        right_wing.apply_transform(trimesh.transformations.rotation_matrix(-0.42, [0, 1, 0], point=(3.45, 2.34, 14.7)))
        add(left_wing, "prompt_left_wing_silhouette")
        add(right_wing, "prompt_right_wing_silhouette")
        for side in (-1.0, 1.0):
            for z in (12.2, 14.0, 15.8):
                add(cylinder_between((side * 3.2, 2.68, z), (side * 5.3, 2.78, z - 1.0), 0.07, sections=10), "prompt_wing_feather_ridge")

    if wants_power_armor or (not wants_alien_bioform and any(term in prompt for term in ("skull", "skulls", "bone", "bones"))):
        for x in (-3.8, 3.8):
            add(ellipsoid((x, -5.7, 1.85), (0.48, 0.36, 0.34), subdivisions=1), "prompt_base_skull")
            add(ellipsoid((x - 0.16, -6.02, 1.9), (0.06, 0.04, 0.06), subdivisions=1), "prompt_skull_eye_socket")
            add(ellipsoid((x + 0.16, -6.02, 1.9), (0.06, 0.04, 0.06), subdivisions=1), "prompt_skull_eye_socket")

    return parts, names


def build_semantic_archetype_features(prompt: str) -> tuple[list[trimesh.Trimesh], list[str]]:
    """Add local-knowledge archetype landmarks for many model families."""
    plan = lookup_semantic_archetype(prompt)
    archetype = str(plan.get("archetype") or "")
    parts: list[trimesh.Trimesh] = []
    names: list[str] = []

    def add(mesh: trimesh.Trimesh, name: str) -> None:
        parts.append(mesh)
        names.append(name)

    if archetype == "high_elf_warrior":
        add(ellipsoid((0, -0.08, 13.5), (3.9, 1.45, 4.8), 3), "semantic_high_elf_tall_slender_torso")
        add(trimesh.creation.cone(radius=0.42, height=2.1, sections=18), "semantic_high_elf_helmet_crest")
        parts[-1].apply_translation([0, -0.12, 20.15])
        for side in (-1.0, 1.0):
            add(trimesh.creation.cone(radius=0.22, height=1.15, sections=12), "semantic_high_elf_pointed_ear")
            parts[-1].apply_transform(trimesh.transformations.rotation_matrix(side * math.pi / 2, [0, 1, 0]))
            parts[-1].apply_translation([side * 1.28, -0.2, 18.25])
            add(ellipsoid((side * 2.25, -1.95, 13.2), (0.32, 0.12, 0.22), 1), "semantic_high_elf_leaf_armor_relief")
            add(cylinder_between((side * 2.48, -1.98, 11.2), (side * 2.48, -1.98, 15.1), 0.08, 10), "semantic_high_elf_vine_trim")
        add(ellipsoid((-5.65, -1.45, 12.0), (1.25, 0.24, 2.65), 2), "semantic_high_elf_kite_shield")
        for z in (11.0, 12.2, 13.4):
            add(box((1.0, 0.13, 0.16), (-5.65, -1.78, z)), "semantic_high_elf_shield_leaf_bar")
        add(cylinder_between((5.25, -1.05, 7.2), (5.25, -1.05, 21.4), 0.12, 16), "semantic_high_elf_long_spear")
        spear_tip = trimesh.creation.cone(radius=0.42, height=1.1, sections=18)
        spear_tip.apply_translation([5.25, -1.05, 21.95])
        add(spear_tip, "semantic_high_elf_spear_leaf_tip")
        add(box((4.7, 0.32, 7.2), (0, 2.05, 11.0)), "semantic_high_elf_flowing_cape")

    elif archetype == "dwarf_warrior":
        add(ellipsoid((0, -0.05, 11.8), (4.9, 2.2, 4.9), 3), "semantic_dwarf_broad_stocky_torso")
        add(ellipsoid((0, -1.25, 15.6), (1.45, 0.52, 2.0), 2), "semantic_dwarf_braided_beard")
        for side in (-1.0, 1.0):
            add(cylinder_between((side * 0.75, -1.2, 14.1), (side * 1.15, -1.25, 11.3), 0.16, 12), "semantic_dwarf_beard_braid")
        add(ellipsoid((-5.6, -1.55, 11.7), (1.65, 0.32, 1.65), 2), "semantic_dwarf_round_shield")
        add(cylinder_between((5.2, -1.1, 8.4), (5.2, -1.1, 17.6), 0.18, 16), "semantic_dwarf_axe_handle")
        add(box((2.0, 0.24, 1.25), (5.85, -1.1, 17.2)), "semantic_dwarf_axe_head")
        for x in np.linspace(-1.8, 1.8, 5):
            add(box((0.42, 0.13, 0.18), (float(x), -1.92, 13.1)), "semantic_dwarf_rune_plate")

    elif archetype == "orc_brute":
        add(ellipsoid((0, -0.05, 12.4), (5.4, 2.45, 5.0), 3), "semantic_orc_hunched_muscular_torso")
        add(ellipsoid((0, -0.35, 16.6), (2.05, 1.12, 1.45), 2), "semantic_orc_heavy_jaw_head")
        for side in (-1.0, 1.0):
            add(trimesh.creation.cone(radius=0.15, height=0.9, sections=12), "semantic_orc_tusk")
            parts[-1].apply_transform(trimesh.transformations.rotation_matrix(side * math.pi / 2, [0, 1, 0]))
            parts[-1].apply_translation([side * 0.75, -1.45, 16.25])
            add(trimesh.creation.cone(radius=0.28, height=1.25, sections=14), "semantic_orc_armor_spike")
            parts[-1].apply_translation([side * 3.1, -0.1, 16.8])
        add(cylinder_between((5.1, -1.0, 7.7), (5.1, -1.0, 17.8), 0.2, 16), "semantic_orc_cleaver_handle")
        add(box((2.1, 0.3, 2.2), (5.8, -1.0, 17.0)), "semantic_orc_cleaver_head")

    elif archetype == "undead_warrior":
        add(ellipsoid((0, -0.22, 17.1), (1.35, 0.92, 1.25), 2), "semantic_undead_skull_head")
        for side in (-1.0, 1.0):
            add(ellipsoid((side * 0.42, -1.03, 17.25), (0.16, 0.06, 0.16), 1), "semantic_undead_eye_socket")
        for z in (11.5, 12.4, 13.3, 14.2):
            add(box((3.2, 0.16, 0.18), (0, -1.82, z)), "semantic_undead_visible_rib")
        add(box((5.3, 0.34, 7.1), (0, 2.1, 10.8)), "semantic_undead_tattered_cloak")
        add(ellipsoid((-5.6, -1.4, 11.5), (1.35, 0.28, 2.1), 2), "semantic_undead_bone_shield")

    elif archetype == "samurai_warrior":
        add(trimesh.creation.cone(radius=1.25, height=1.25, sections=36), "semantic_samurai_kabuto_helmet")
        parts[-1].apply_translation([0, -0.1, 19.1])
        add(box((0.28, 0.18, 2.2), (0, -0.2, 20.1)), "semantic_samurai_helmet_crest")
        for z in (10.8, 11.7, 12.6, 13.5, 14.4):
            add(box((4.9, 0.18, 0.16), (0, -1.86, z)), "semantic_samurai_lamellar_plate")
        add(cylinder_between((5.0, -1.05, 7.6), (5.0, -1.05, 18.4), 0.13, 18), "semantic_samurai_katana")
        add(box((4.6, 0.22, 2.8), (0, -1.55, 8.8)), "semantic_samurai_skirt_plates")

    elif archetype == "viking_raider":
        add(ellipsoid((0, -1.22, 15.5), (1.35, 0.48, 1.75), 2), "semantic_viking_beard")
        add(ellipsoid((-5.5, -1.48, 11.8), (1.75, 0.32, 1.75), 2), "semantic_viking_round_shield")
        add(cylinder_between((5.2, -1.0, 8.2), (5.2, -1.0, 17.6), 0.18, 16), "semantic_viking_axe_handle")
        add(box((1.8, 0.24, 1.25), (5.9, -1.0, 17.1)), "semantic_viking_axe_head")
        add(box((5.6, 0.42, 6.5), (0, 2.18, 11.2)), "semantic_viking_fur_cloak")
        for x in np.linspace(-4.5, 4.5, 5):
            add(ellipsoid((float(x), 5.6, 1.45), (0.32, 0.22, 0.12), 1), "semantic_viking_rune_stone")

    elif archetype == "pirate_captain":
        brim = trimesh.creation.cylinder(radius=1.7, height=0.16, sections=40)
        brim.apply_scale([1.35, 0.55, 1.0]); brim.apply_translation([0, -0.1, 18.75]); add(brim, "semantic_pirate_tricorn_brim")
        add(box((5.0, 0.32, 6.8), (0, 1.95, 10.6)), "semantic_pirate_long_coat")
        add(cylinder_between((5.2, -1.05, 8.0), (5.2, -1.05, 15.9), 0.13, 16), "semantic_pirate_cutlass")
        add(box((1.65, 0.36, 0.55), (-4.7, -1.4, 12.2)), "semantic_pirate_flintlock_pistol")
        add(box((4.8, 0.3, 0.38), (0, -1.75, 10.5)), "semantic_pirate_sash")

    elif archetype == "robot_mech":
        for x in (-1.5, 1.5):
            add(box((1.05, 0.36, 4.8), (x, -0.7, 6.0)), "semantic_robot_rectangular_leg_armor")
        add(box((5.2, 1.4, 5.8), (0, -0.1, 13.1)), "semantic_robot_boxy_chest_panel")
        add(box((1.8, 0.4, 0.38), (0, -1.55, 17.35)), "semantic_robot_sensor_visor")
        add(cylinder_between((0.7, 0.5, 18.2), (1.4, 0.85, 20.2), 0.08, 10), "semantic_robot_antenna")
        for side in (-1.0, 1.0):
            add(cylinder_between((side * 2.3, 0.85, 11.2), (side * 3.6, 0.9, 7.5), 0.08, 10), "semantic_robot_external_cable")

    elif archetype == "ranger_archer":
        hood = trimesh.creation.cone(radius=1.45, height=2.0, sections=32); hood.apply_translation([0, -0.15, 19.0]); add(hood, "semantic_ranger_hood")
        add(box((5.1, 0.34, 7.2), (0, 2.1, 10.7)), "semantic_ranger_cloak")
        add(cylinder_between((4.9, -1.2, 8.8), (4.9, -1.2, 17.4), 0.09, 14), "semantic_ranger_bow")
        add(box((1.0, 0.42, 3.4), (-2.6, 2.2, 13.0)), "semantic_ranger_quiver")
        for x in (-2.6, 2.6):
            add(box((0.9, 0.28, 0.36), (x, -1.75, 10.4)), "semantic_ranger_belt_pouch")

    elif archetype == "lizardfolk_warrior":
        add(ellipsoid((0, -1.2, 17.2), (1.65, 1.15, 0.72), 2), "semantic_lizardfolk_long_snout")
        add(tapered_chain([(0, 2.2, 9.8), (0, 4.6, 7.0), (0, 6.2, 4.2)], [0.28, 0.14], 18), "semantic_lizardfolk_tail")
        for z in (10.8, 12.0, 13.2, 14.4):
            add(box((3.4, 0.18, 0.18), (0, -1.92, z)), "semantic_lizardfolk_scale_row")
        for side in (-1.0, 1.0):
            add(trimesh.creation.cone(radius=0.12, height=0.6, sections=10), "semantic_lizardfolk_claw")
            parts[-1].apply_translation([side * 4.95, -0.9, 9.1])
            add(trimesh.creation.cone(radius=0.14, height=0.9, sections=12), "semantic_lizardfolk_head_spine")
            parts[-1].apply_translation([side * 0.5, 0.1, 18.5])

    return parts, names


def build_vehicle(scale_mm: float) -> tuple[trimesh.Trimesh, list[str]]:
    base = trimesh.creation.cylinder(radius=scale_mm * 0.48, height=1.4, sections=96)
    base.apply_scale([1.35, 0.75, 1.0])
    base.apply_translation([0, 0, 0.7])
    hull = box((15, 8, 5), (0, 0, 5.2))
    turret = box((5, 4, 2.4), (0, -0.4, 9.1))
    cannon = cylinder_between((2.4, -0.4, 9.1), (10.5, -0.4, 9.1), 0.35)
    return trimesh.util.concatenate([base, hull, turret, cannon]), ["oval_base", "hull", "turret", "cannon"]


def normalize_native_mesh(mesh: trimesh.Trimesh, scale_mm: float, *, subject_type: str) -> trimesh.Trimesh:
    vertices = np.asarray(mesh.vertices, dtype=float)
    ext = np.maximum(vertices.max(axis=0) - vertices.min(axis=0), 1e-6)
    if subject_type in {"armored_humanoid", "high_elf_warrior"}:
        current = float(ext[2])
    else:
        current = float(np.max(ext))
    mesh = mesh.copy()
    mesh.vertices = vertices * (scale_mm / current)
    vertices = np.asarray(mesh.vertices, dtype=float)
    mins = vertices.min(axis=0)
    maxs = vertices.max(axis=0)
    vertices[:, 0] -= (mins[0] + maxs[0]) * 0.5
    vertices[:, 1] -= (mins[1] + maxs[1]) * 0.5
    vertices[:, 2] -= mins[2]
    mesh.vertices = vertices
    try:
        mesh.remove_unreferenced_vertices()
        mesh.merge_vertices()
        mesh.fix_normals()
    except Exception:
        pass
    return mesh


def fuse_native_parts(mesh: trimesh.Trimesh, request: dict[str, Any]) -> tuple[trimesh.Trimesh, bool]:
    """Union native kit/sculpt parts into one printable shell.

    The native builder intentionally creates many readable parts. Exporting that
    raw concatenation leaves intersecting/internal shells that can display as STL
    artifacts or slicer noise. Voxel fusion turns the kitbash into a single solid
    miniature while preserving millimeter-scale raised detail.
    """
    if os.environ.get("MESHMEND_NATIVE_FUSE_PARTS", "1").strip().lower() in {"0", "false", "no"}:
        return mesh, False
    try:
        vertices = np.asarray(mesh.vertices, dtype=float)
        if len(vertices) == 0 or len(mesh.faces) == 0:
            return mesh, False
        ext = np.maximum(vertices.max(axis=0) - vertices.min(axis=0), 1e-6)
        max_extent = float(np.max(ext))
        pitch = float(os.environ.get("MESHMEND_NATIVE_FUSION_PITCH_MM", "0.08"))
        pitch = max(max_extent / 420.0, min(pitch, max_extent / 90.0))
        voxels = mesh.voxelized(pitch)
        try:
            from scipy.ndimage import binary_closing, binary_dilation, binary_fill_holes
            from trimesh.voxel import ops as voxel_ops

            matrix = np.asarray(voxels.matrix, dtype=bool)
            close_iterations = int(os.environ.get("MESHMEND_NATIVE_FUSION_CLOSE_ITERATIONS", "1"))
            dilate_iterations = int(os.environ.get("MESHMEND_NATIVE_FUSION_DILATE_ITERATIONS", "1"))
            if dilate_iterations > 0:
                matrix = binary_dilation(matrix, iterations=dilate_iterations)
            if close_iterations > 0:
                matrix = binary_closing(matrix, iterations=close_iterations)
            matrix = binary_fill_holes(matrix)
            solid = voxel_ops.matrix_to_marching_cubes(matrix=np.pad(matrix, 2, constant_values=False), pitch=pitch)
        except Exception:
            solid = voxels.fill().marching_cubes
        if not isinstance(solid, trimesh.Trimesh) or len(solid.faces) < 1000:
            return mesh, False
        solid.metadata.update(mesh.metadata)
        solid.metadata["meshmend_native_fused_solid"] = True
        solid.metadata["meshmend_native_fusion_pitch_mm"] = float(pitch)
        try:
            solid.remove_unreferenced_vertices()
            solid.merge_vertices()
            solid.fix_normals()
        except Exception:
            pass
        return solid, True
    except Exception:
        return mesh, False


def native_quality_issues(mesh: trimesh.Trimesh, definition_feature_count: int, fused_solid: bool, components: int) -> list[str]:
    issues: list[str] = []
    if len(mesh.faces) < int(os.environ.get("MESHMEND_NATIVE_MIN_FACES", "50000")):
        issues.append("native_low_face_count")
    ext = np.maximum(np.asarray(mesh.extents, dtype=float), 1e-6)
    if float(ext.min() / ext.max()) < 0.10:
        issues.append("native_too_flat")
    if definition_feature_count < int(os.environ.get("MESHMEND_NATIVE_MIN_DEFINITION_FEATURES", "30")):
        issues.append(f"native_under_defined_features_{definition_feature_count}")
    if os.environ.get("MESHMEND_NATIVE_REQUIRE_FUSED_SOLID", "1").strip().lower() not in {"0", "false", "no"}:
        if not fused_solid:
            issues.append("native_parts_not_fused")
        if not bool(mesh.is_watertight):
            issues.append("native_mesh_not_watertight")
        if components > int(os.environ.get("MESHMEND_NATIVE_MAX_COMPONENTS", "1")):
            issues.append(f"native_too_many_components_{components}")
    return issues


def connected_component_count(mesh: trimesh.Trimesh) -> int:
    try:
        return int(len([part for part in mesh.split(only_watertight=False) if len(part.faces) > 20]))
    except Exception:
        return 0


def count_definition_features(parts: list[str]) -> int:
    definition_terms = (
        "scale",
        "tooth",
        "horn",
        "claw",
        "strap",
        "rivet",
        "plate",
        "trim",
        "visor",
        "belt",
        "shoulder",
        "knee",
        "backpack",
        "spike",
        "rock",
        "robe",
        "hood",
        "cape",
        "dagger",
        "shield",
        "banner",
        "staff",
        "sword",
        "axe",
        "wing",
        "skull",
        "tabard",
        "plume",
        "alien",
        "chitin",
        "carapace",
        "bio",
        "tail",
        "forelimb",
        "elf",
        "elven",
        "ear",
        "helm",
        "crest",
        "leaf",
        "rune",
        "gem",
        "spear",
        "boot",
        "breastplate",
        "detail",
        "panel",
        "seam",
        "fold",
        "engraved",
        "embossed",
        "motif",
        "scroll",
        "pouch",
    )
    return sum(1 for part in parts if any(term in part for term in definition_terms))


def refine_native_mesh(mesh: trimesh.Trimesh, request: dict[str, Any]) -> trimesh.Trimesh:
    quality = str(request.get("quality") or "standard").lower()
    prompt = str(request.get("prompt") or "").lower()
    requested_faces = int(float(request.get("target_polycount") or 0) or 0)
    high_detail = quality == "high" or requested_faces >= 250000 or any(
        term in prompt
        for term in (
            "store quality", "store-quality", "store level", "store-level", "studio", "studio quality",
            "studio-quality", "studio level", "studio-level", "8k", "8 k", "high detail",
        )
    )
    target_faces = int(os.environ.get("MESHMEND_NATIVE_TARGET_FACES", "65000" if high_detail else "18000"))
    refined = mesh.copy()
    while len(refined.faces) < target_faces:
        try:
            vertices, faces = trimesh.remesh.subdivide(refined.vertices, refined.faces)
            refined = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        except Exception:
            break
        if len(refined.faces) > target_faces * 4:
            break
    try:
        refined.fix_normals()
    except Exception:
        pass
    return refined


def ellipsoid(center: tuple[float, float, float], scale: tuple[float, float, float], subdivisions: int = 2) -> trimesh.Trimesh:
    mesh = trimesh.creation.icosphere(subdivisions=subdivisions, radius=1.0)
    mesh.apply_scale(scale)
    mesh.apply_translation(center)
    return mesh


def box(extents: tuple[float, float, float], center: tuple[float, float, float]) -> trimesh.Trimesh:
    mesh = trimesh.creation.box(extents=extents)
    mesh.apply_translation(center)
    return mesh


def cylinder_between(start: tuple[float, float, float], end: tuple[float, float, float], radius: float, sections: int = 24) -> trimesh.Trimesh:
    start_v = np.asarray(start, dtype=float)
    end_v = np.asarray(end, dtype=float)
    vector = end_v - start_v
    height = float(np.linalg.norm(vector))
    if height <= 1e-8:
        return ellipsoid(tuple(start_v), (radius, radius, radius), subdivisions=1)
    mesh = trimesh.creation.cylinder(radius=radius, height=height, sections=sections)
    direction = vector / height
    transform = trimesh.geometry.align_vectors([0, 0, 1], direction)
    mesh.apply_transform(transform)
    mesh.apply_translation((start_v + end_v) * 0.5)
    return mesh


def tapered_chain(points: list[tuple[float, float, float]], radii: list[float], sections: int = 24) -> trimesh.Trimesh:
    parts = []
    for index in range(len(points) - 1):
        radius = radii[min(index, len(radii) - 1)]
        parts.append(cylinder_between(points[index], points[index + 1], radius, sections=sections))
    return trimesh.util.concatenate(parts)


def build_curved_blade(center: tuple[float, float, float], scale: float = 1.0) -> trimesh.Trimesh:
    shaft = cylinder_between((center[0], center[1], center[2] - 4.0 * scale), (center[0], center[1], center[2] + 3.4 * scale), 0.12 * scale, sections=18)
    blade = box((0.45 * scale, 0.16 * scale, 3.8 * scale), (center[0] + 0.55 * scale, center[1], center[2] + 3.0 * scale))
    blade.apply_transform(trimesh.transformations.rotation_matrix(-0.35, [0, 1, 0], point=center))
    return trimesh.util.concatenate([shaft, blade])


def build_printable_detail_stamps(archetype: str, prompt: str = "") -> tuple[list[trimesh.Trimesh], list[str]]:
    """Raised, printable detail stamps that survive voxel fusion and STL export.

    These are intentionally not texture-only details. They are chunky 0.16-0.32mm
    raised panels/ridges/gems at the native 32mm scaffold scale, so they remain
    visible after local solidification and resin-slicer repair.
    """
    parts: list[trimesh.Trimesh] = []
    names: list[str] = []

    def add(mesh: trimesh.Trimesh, name: str) -> None:
        parts.append(mesh)
        names.append(name)

    # Universal armor read: panel seams, rivets, belt pouches, base texture.
    for x in np.linspace(-2.4, 2.4, 7):
        add(ellipsoid((float(x), -2.05, 14.35), (0.16, 0.10, 0.16), 1), "printable_chest_rivet_detail")
    for z in (11.15, 12.25, 13.35, 14.45):
        add(box((4.65, 0.20, 0.16), (0.0, -2.02, z)), "printable_layered_armor_panel_seam")
    for side in (-1.0, 1.0):
        for z in (4.0, 5.2, 6.4, 7.6):
            add(box((0.85, 0.18, 0.18), (side * 1.22, -0.92, z)), "printable_greave_detail_band")
        add(box((0.82, 0.30, 0.52), (side * 2.45, -1.92, 10.25)), "printable_belt_pouch_detail")
    for angle in np.linspace(0.0, 2.0 * math.pi, 28, endpoint=False):
        radius = 6.65 + 0.28 * math.sin(float(angle) * 3.0)
        add(ellipsoid((radius * math.cos(float(angle)), radius * math.sin(float(angle)), 1.42), (0.16, 0.12, 0.08), 1), "printable_base_gravel_detail")

    if archetype == "high_elf_warrior" or "elf" in prompt:
        for side in (-1.0, 1.0):
            for z in (11.6, 12.8, 14.0):
                add(ellipsoid((side * 2.1, -1.98, z), (0.26, 0.12, 0.16), 1), "printable_elven_leaf_detail")
                add(cylinder_between((side * 2.28, -2.02, z - 0.42), (side * 2.28, -2.02, z + 0.42), 0.075, sections=10), "printable_elven_vine_detail")
        for z in (11.0, 12.1, 13.2):
            add(box((1.18, 0.20, 0.16), (-4.85, -2.05, z)), "printable_shield_embossed_leaf_bar")
        for z in np.linspace(8.0, 14.5, 6):
            add(cylinder_between((0.0, 1.98, float(z)), (0.0, 2.02, float(z) + 0.85), 0.16, sections=10), "printable_cape_center_fold_detail")

    if any(term in prompt for term in ("dwarf", "dwarven")):
        for x in np.linspace(-1.8, 1.8, 5):
            add(box((0.36, 0.18, 0.22), (float(x), -2.08, 13.0)), "printable_dwarf_rune_detail")
        for side in (-1.0, 1.0):
            add(cylinder_between((side * 0.55, -1.72, 15.0), (side * 1.1, -1.78, 12.2), 0.11, 10), "printable_beard_braid_detail")

    if any(term in prompt for term in ("orc", "ork", "brute")):
        for side in (-1.0, 1.0):
            for z in (12.0, 13.4, 15.0):
                add(trimesh.creation.cone(radius=0.18, height=0.78, sections=12), "printable_orc_spike_detail")
                parts[-1].apply_translation([side * 2.9, -0.9, z])
        add(box((3.2, 0.24, 0.22), (0.0, -2.08, 12.65)), "printable_scrap_armor_jagged_plate")

    if any(term in prompt for term in ("samurai", "ronin", "katana")):
        for z in np.linspace(10.7, 14.6, 7):
            add(box((4.7, 0.20, 0.13), (0.0, -2.10, float(z))), "printable_samurai_lamellar_lace_detail")

    if any(term in prompt for term in ("lizardfolk", "lizardman", "dragonborn", "reptilian")):
        for x in np.linspace(-1.8, 1.8, 5):
            for z in (11.1, 12.2, 13.3, 14.4):
                add(ellipsoid((float(x), -2.08, z), (0.18, 0.10, 0.14), 1), "printable_reptile_scale_detail")

    return parts, names


def requested_scale_mm(request: dict[str, Any]) -> float:
    try:
        return float(request.get("scale_mm") or 32.0)
    except Exception:
        return 32.0
