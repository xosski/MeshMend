from __future__ import annotations

import math
import os
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import trimesh


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

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def generate_native_miniature(request: dict[str, Any], image_path: Path | None, output_dir: Path) -> tuple[trimesh.Trimesh, NativeGenerationReport]:
    """Generate MeshMend-owned polygon geometry instead of reconstructing an image.

    This is intentionally scaffold/part based. It does not try to hallucinate a
    full mesh from pixels; it builds printable miniature structure controlled by
    MeshMend: base, creature/body/limbs/rider/weapon, and explicit raised/recessed
    detail geometry. The image is used only to choose broad subject layout.
    """
    prompt = str(request.get("prompt") or "").lower()
    scale_mm = requested_scale_mm(request)
    subject_type = infer_subject_type(prompt, image_path)
    if subject_type == "mounted_creature":
        mesh, parts = build_mounted_creature(scale_mm)
    elif subject_type == "vehicle":
        mesh, parts = build_vehicle(scale_mm)
    else:
        mesh, parts = build_armored_humanoid(scale_mm, prompt)
    mesh, fused_solid = fuse_native_parts(mesh, request)
    mesh = refine_native_mesh(mesh, request)
    mesh = normalize_native_mesh(mesh, scale_mm, subject_type=subject_type)
    mesh.metadata["meshmend_native_generation"] = True
    mesh.metadata["meshmend_native_subject_type"] = subject_type
    mesh.metadata["units"] = "mm"
    definition_feature_count = count_definition_features(parts)
    components = connected_component_count(mesh)
    issues = native_quality_issues(mesh, definition_feature_count, fused_solid, components)
    report = NativeGenerationReport(
        provider="meshmend_native",
        capability_tier="procedural_printable_draft",
        subject_type=subject_type,
        generated_parts=parts,
        definition_feature_count=definition_feature_count,
        fused_solid=fused_solid,
        watertight=bool(mesh.is_watertight),
        components=components,
        faces=int(len(mesh.faces)),
        vertices=int(len(mesh.vertices)),
        production_ready=not issues,
        issues=issues,
    )
    (output_dir / "native_generation_report.json").write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    return mesh, report


def infer_subject_type(prompt: str, image_path: Path | None) -> str:
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
    prompt_parts, prompt_names = build_humanoid_prompt_features(prompt)
    parts.extend(prompt_parts)
    names.extend(prompt_names)
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

    wants_robes = any(term in prompt for term in ("wizard", "mage", "sorcerer", "witch", "warlock", "cleric", "priest", "robe", "robed"))
    wants_hood = wants_robes or any(term in prompt for term in ("hood", "hooded", "ranger", "rogue", "assassin"))
    wants_cape = wants_robes or any(term in prompt for term in ("cape", "cloak", "tattered", "fur cloak"))
    wants_rogue = any(term in prompt for term in ("rogue", "assassin", "ranger", "thief", "ninja", "dagger", "dual wield", "dual-wield"))
    wants_knight = any(term in prompt for term in ("knight", "paladin", "templar", "crusader", "champion"))
    wants_demon = any(term in prompt for term in ("demon", "devil", "fiend", "tiefling"))
    wants_angel = any(term in prompt for term in ("angel", "celestial", "seraph", "wing", "wings"))
    wants_shield = any(term in prompt for term in ("shield", "buckler"))
    wants_banner = any(term in prompt for term in ("banner", "standard", "flag"))
    wants_staff = wants_robes or any(term in prompt for term in ("staff", "spear", "lance"))
    wants_axe = any(term in prompt for term in ("axe", "halberd"))
    wants_sword = wants_knight or any(term in prompt for term in ("sword", "blade", "katana", "khopesh", "scimitar"))

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

    if any(term in prompt for term in ("skull", "skulls", "bone", "bones")):
        for x in (-3.8, 3.8):
            add(ellipsoid((x, -5.7, 1.85), (0.48, 0.36, 0.34), subdivisions=1), "prompt_base_skull")
            add(ellipsoid((x - 0.16, -6.02, 1.9), (0.06, 0.04, 0.06), subdivisions=1), "prompt_skull_eye_socket")
            add(ellipsoid((x + 0.16, -6.02, 1.9), (0.06, 0.04, 0.06), subdivisions=1), "prompt_skull_eye_socket")

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
    if subject_type == "armored_humanoid":
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
        pitch = float(os.environ.get("MESHMEND_NATIVE_FUSION_PITCH_MM", "0.12"))
        pitch = max(max_extent / 420.0, min(pitch, max_extent / 90.0))
        voxels = mesh.voxelized(pitch)
        try:
            from scipy.ndimage import binary_closing, binary_dilation, binary_fill_holes
            from trimesh.voxel import ops as voxel_ops

            matrix = np.asarray(voxels.matrix, dtype=bool)
            close_iterations = int(os.environ.get("MESHMEND_NATIVE_FUSION_CLOSE_ITERATIONS", "2"))
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


def requested_scale_mm(request: dict[str, Any]) -> float:
    try:
        return float(request.get("scale_mm") or 32.0)
    except Exception:
        return 32.0
