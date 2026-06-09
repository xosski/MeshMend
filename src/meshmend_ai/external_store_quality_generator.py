from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


SUPPORTED_MODEL_SUFFIXES = {".stl", ".glb", ".obj", ".ply", ".3mf", ".fbx", ".usdz"}


def main() -> int:
    parser = argparse.ArgumentParser(description="MeshMend external store-quality model generator scaffold")
    parser.add_argument("--input", required=True, help="MeshMend request JSON path")
    parser.add_argument("--prompt", required=True, help="Prompt text file path")
    parser.add_argument("--image", default="", help="Optional decoded input image path for image-to-3D")
    parser.add_argument("--output-dir", required=True, help="Directory where model files and result.json are written")
    parser.add_argument("--quality", default="high")
    parser.add_argument("--target-polycount", default="2000000")
    parser.add_argument(
        "--backend-command",
        default=os.environ.get("MESHMEND_EXTERNAL_GENERATOR_COMMAND", ""),
        help=(
            "Real external generator command. Placeholders: {input_json}, {prompt_path}, {enhanced_prompt_path}, "
            "{spec_path}, {image_path}, {output_dir}, {quality}, {target_polycount}."
        ),
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_progress(output_dir, 5, "external_prepare", "Preparing certified external generation request")

    request = read_json(Path(args.input))
    prompt = Path(args.prompt).read_text(encoding="utf-8") if Path(args.prompt).exists() else str(request.get("prompt") or "")
    target_polycount = int(float(args.target_polycount or request.get("target_polycount") or 2_000_000))
    image_path = Path(args.image) if args.image else None
    spec = build_miniature_spec(request, prompt, image_path, target_polycount)
    enhanced_prompt = build_enhanced_prompt(prompt, spec)

    spec_path = output_dir / "external_miniature_spec.json"
    enhanced_prompt_path = output_dir / "external_enhanced_prompt.txt"
    spec_path.write_text(json.dumps(spec, indent=2), encoding="utf-8")
    enhanced_prompt_path.write_text(enhanced_prompt, encoding="utf-8")

    backend_command = str(args.backend_command or "").strip()
    if not backend_command:
        return fail(
            output_dir,
            "No real external generator is configured. Set MESHMEND_EXTERNAL_GENERATOR_COMMAND to your certified text/image-to-3D command.",
            exit_code=2,
            spec=spec,
        )

    try:
        command = backend_command.format(
            input_json=str(Path(args.input)),
            prompt_path=str(Path(args.prompt)),
            enhanced_prompt_path=str(enhanced_prompt_path),
            spec_path=str(spec_path),
            image_path=str(image_path) if image_path else "none",
            output_dir=str(output_dir),
            quality=str(args.quality),
            target_polycount=str(target_polycount),
        )
    except Exception as exc:
        return fail(output_dir, f"External generator command has invalid placeholders or quoting: {exc}", spec=spec)
    write_progress(output_dir, 18, "external_backend", "Running external model generator")
    try:
        completed = subprocess.run(
            command_args(command),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=env_float("MESHMEND_EXTERNAL_GENERATOR_TIMEOUT_SECONDS", "7200"),
        )
    except subprocess.TimeoutExpired as exc:
        return fail(output_dir, f"External generator timed out after {exc.timeout:g}s", exit_code=124, spec=spec)
    except Exception as exc:
        return fail(output_dir, f"External generator could not be launched: {exc}", spec=spec)
    (output_dir / "external_generator_stdout.txt").write_text(completed.stdout or "", encoding="utf-8")
    (output_dir / "external_generator_stderr.txt").write_text(completed.stderr or "", encoding="utf-8")
    if completed.returncode != 0:
        return fail(output_dir, completed.stderr.strip() or completed.stdout.strip() or f"external generator exited {completed.returncode}", spec=spec)

    write_progress(output_dir, 82, "external_validate", "Validating external model artifact")
    try:
        result = load_backend_result(output_dir, completed.stdout)
        model_path = resolve_model_path(output_dir, result)
        mesh_info = inspect_mesh(model_path)
    except Exception as exc:
        return fail(output_dir, f"External generator did not produce a valid model artifact: {exc}", spec=spec)

    if not bool(result.get("store_quality_certified")) and os.environ.get("MESHMEND_EXTERNAL_TRUST_UNCERTIFIED_OUTPUT", "0").strip().lower() not in {"1", "true", "yes", "on"}:
        return fail(
            output_dir,
            "Underlying external generator did not return store_quality_certified=true",
            spec=spec,
            mesh_info=mesh_info,
        )
    score_issues = required_quality_score_issues(result)
    if score_issues:
        return fail(
            output_dir,
            "Underlying external generator did not meet store-quality score contract: " + "; ".join(score_issues),
            spec=spec,
            mesh_info=mesh_info,
        )

    min_faces = int(target_polycount * env_float("MESHMEND_CERTIFIED_MIN_FACE_RATIO", "0.75", fallback_name="MESHMEND_EXTERNAL_MIN_FACE_RATIO"))
    issues = quality_issues(mesh_info, min_faces)
    if issues:
        return fail(output_dir, "External generator output failed store-quality scaffold checks: " + "; ".join(issues), spec=spec, mesh_info=mesh_info)

    final = {
        "model_file": model_path.name,
        "model_format": model_path.suffix.lower().lstrip("."),
        "provider": result.get("provider") or "meshmend_external_store_quality_generator",
        "capability_tier": "certified_store_quality_external",
        "geometry_source": result.get("geometry_source") or "external_certified_sculpt_backend",
        "store_quality_certified": True,
        "workflow": str(request.get("workflow") or ("image_to_3d" if image_path else "text_to_3d")),
        "source_image": image_path.name if image_path else None,
        "miniature_spec": spec,
        "store_quality_scores": store_quality_scores(result),
        "mesh_info": {**dict(result.get("mesh_info") or {}), **mesh_info, "detail_source": "external_certified_sculpt_geometry"},
        "consumed_credits": int(result.get("consumed_credits") or 0),
    }
    (output_dir / "result.json").write_text(json.dumps(final, indent=2), encoding="utf-8")
    write_progress(output_dir, 96, "external_complete", "External store-quality generator completed")
    print(json.dumps(final))
    return 0


def build_miniature_spec(request: dict[str, Any], prompt: str, image_path: Path | None, target_polycount: int) -> dict[str, Any]:
    text = prompt.lower()
    subject_text = prompt_subject_text(prompt).lower()
    scale_mm = float(request.get("scale_mm") or 32.0)
    image_cues = reference_image_subject_cues(image_path) if image_path is not None else {}
    semantic_plan = local_semantic_plan(subject_text)
    semantic_archetype = str(semantic_plan.get("archetype") or "generic_humanoid")
    is_alien_bioform = any(
        term in subject_text
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
    prompt_has_subject = any(
        term in subject_text
        for term in (
            "space marine", "spacemarine", "adeptus", "primaris", "mounted", "rider", "dragon", "beast",
            "cavalry", "wizard", "mage", "sorcerer", "witch", "warlock", "cleric", "priest", "rogue",
            "assassin", "ranger", "knight", "paladin", "orc", "ork", "brute", "creature", "monster",
            "termagant", "termagaunt", "tyranid", "hormagaunt", "insectoid alien", "bioform",
            "high elf", "high-elf", "elf warrior", "elven warrior", "dwarf", "dwarven", "samurai",
            "ronin", "viking", "pirate", "robot", "android", "mech", "lizardfolk", "dragonborn",
        )
    )
    is_space_marine = any(term in subject_text for term in ("space marine", "spacemarine", "adeptus", "primaris", "power armored marine", "power armoured marine"))
    archetype = (
        "alien_chitin_bioform" if is_alien_bioform
        else "power_armored_space_marine" if is_space_marine
        else semantic_archetype if semantic_archetype != "generic_humanoid"
        else "mounted_beast" if any(term in subject_text for term in ("mounted", "rider", "dragon", "beast", "cavalry"))
        else "robed_caster" if any(term in subject_text for term in ("wizard", "mage", "sorcerer", "witch", "warlock", "cleric", "priest", "robe", "robed"))
        else "stealth_rogue" if any(term in subject_text for term in ("rogue", "assassin", "ranger", "thief", "ninja", "dagger"))
        else "heroic_knight" if any(term in subject_text for term in ("knight", "paladin", "templar", "crusader", "champion"))
        else "orc_warrior" if any(term in subject_text for term in ("orc", "ork", "brute"))
        else "mounted_beast" if image_cues.get("wide_reference") and not prompt_has_subject
        else "image_reference_subject" if image_path is not None and not prompt_has_subject
        else "armored_humanoid"
    )
    weapon = "fleshborer_bioweapon" if is_alien_bioform else "rifle" if is_space_marine or any(term in subject_text for term in ("rifle", "gun", "bolter", "blaster")) else "axe" if "axe" in subject_text else "staff" if any(term in subject_text for term in ("staff", "spear", "lance")) else "visible_reference_weapon" if image_path is not None and not prompt_has_subject else "sword"
    if is_alien_bioform:
        landmarks = [
            "sloped_chitin_head_with_mandibles",
            "ribbed_organic_carapace",
            "four_clawed_legs",
            "two_scything_forelimbs",
            "fleshborer_bioweapon",
            "long_tapering_tail",
            "small_scenic_base",
            "do_not_substitute_generic_humanoid",
        ]
    else:
        landmarks = ["readable_face_or_helmet", "hands_and_fingers", "weapon_bevels", "armor_trim", "deep_panel_lines", "scenic_base"]
    for landmark in semantic_plan.get("landmarks") or []:
        value = str(landmark).replace(" ", "_")
        if value not in landmarks:
            landmarks.append(value)
    if is_space_marine:
        landmarks += [
            "power_armored_barrel_torso",
            "oversized_pauldrons",
            "helmet_visor_and_respirator",
            "rear_power_pack_with_exhausts",
            "chest_emblem",
            "bolter_rifle_across_chest",
            "chunky_greaves_and_boots",
        ]
    if archetype == "robed_caster":
        landmarks += ["robe_mass_and_cloth_folds", "staff_or_focus", "hood_or_high_collar"]
    if archetype == "stealth_rogue":
        landmarks += ["lean_hooded_silhouette", "mask_or_shadowed_face", "daggers_or_short_blades"]
    if archetype == "heroic_knight":
        landmarks += ["heroic_plate_armor", "large_shield_or_banner", "crested_helmet"]
    if archetype == "orc_warrior":
        landmarks += ["brute_proportions", "tusks", "heavy_weapon", "trophy_straps"]
    if archetype == "high_elf_warrior":
        landmarks += ["tall_slender_proportions", "pointed_ears", "crested_helmet", "leaf_rune_trim", "kite_shield", "cape_or_tabard"]
    if archetype == "dwarf_warrior":
        landmarks += ["short_stocky_proportions", "braided_beard", "runic_armor", "round_shield", "axe_or_hammer"]
    if archetype == "samurai_warrior":
        landmarks += ["kabuto_helmet", "lamellar_plate_rows", "sode_shoulders", "katana", "skirt_plates"]
    if archetype == "robot_mech":
        landmarks += ["boxy_panels", "sensor_visor", "cables", "mechanical_joints"]
    if archetype == "lizardfolk_warrior":
        landmarks += ["long_snout", "visible_tail", "scale_rows", "claws", "crest_spines"]
    if archetype == "mounted_beast" and image_path is not None and not prompt_has_subject:
        landmarks += [
            "preserve_uploaded_reference_silhouette",
            "mounted_rider_if_visible",
            "beast_body_and_tail_if_visible",
            "horns_spikes_or_crests_if_visible",
            "clawed_feet_or_mount_legs",
            "saddle_reins_and_barding_if_visible",
            "oval_or_scenic_base_from_reference",
        ]
    elif archetype == "image_reference_subject":
        landmarks += [
            "preserve_uploaded_reference_silhouette",
            "do_not_replace_with_generic_humanoid",
            "visible_weapon_or_accessory_from_reference",
            "distinctive_head_helmet_or_face_from_reference",
            "base_shape_from_reference",
        ]
    for term, landmark in (
        ("cape", "cloth_folds"),
        ("cloak", "cloth_folds"),
        ("shield", "shield"),
        ("banner", "banner"),
        ("skull", "skulls_and_rubble"),
        ("rock", "skulls_and_rubble"),
        ("scales", "scale_rows"),
        ("claws", "claws"),
        ("horn", "horns_or_spikes"),
        ("spike", "horns_or_spikes"),
    ):
        if term not in subject_text:
            continue
        if is_alien_bioform and landmark in {"cloth_folds", "shield", "banner", "skulls_and_rubble"}:
            continue
        if term in text and landmark not in landmarks:
            landmarks.append(landmark)
    landmarks = list(dict.fromkeys(landmarks))
    return {
        "scale_mm": scale_mm,
        "target_polycount": target_polycount,
        "archetype": archetype,
        "weapon": weapon,
        "quality_bar": "commercial tabletop miniature STL, 360-degree sculpt, resin-printable, not a blockout",
        "generation_pipeline": {
            "concept_generation_separate_from_mesh_generation": True,
            "modular_parts": ["head", "torso", "legs", "left_arm", "right_arm", "weapons", "accessories"],
            "assembly_before_sculpting": True,
            "sculpt_passes": [
                "primary_large_forms",
                "secondary_armor_details_trims_vents_pouches",
                "tertiary_micro_detail_engravings_insignia_texture",
            ],
            "artifact_rejection": ["planes", "sheets", "floating_geometry", "disconnected_shells", "non_manifold_topology", "open_surfaces"],
            "quality_critic_minimum_score": 85,
        },
        "input_image": image_path.name if image_path else None,
        "required_landmarks": landmarks,
        "printability": {
            "watertight": True,
            "max_components": int(os.environ.get("MESHMEND_EXTERNAL_MAX_COMPONENTS", "3")),
            "minimum_feature_mm": 0.16,
            "preferred_scale_mm": scale_mm,
        },
    }


def prompt_subject_text(prompt: str) -> str:
    text = " ".join((prompt or "").split())
    for marker in (
        "Create a production/studio-quality",
        "Create a final store/studio-quality",
        "Create a production",
        "Create a final",
    ):
        index = text.lower().find(marker.lower())
        if index > 0:
            return text[:index].strip()
    return text


def local_semantic_plan(subject_text: str) -> dict[str, Any]:
    """Return bundled offline archetype cues used to avoid generic outputs."""
    try:
        service_dir = Path(__file__).resolve().parent / "3dsculpter" / "model_service"
        if str(service_dir) not in sys.path:
            sys.path.insert(0, str(service_dir))
        from native_generation import lookup_semantic_archetype

        plan = lookup_semantic_archetype(subject_text)
        return dict(plan) if isinstance(plan, dict) else {}
    except Exception:
        return {"archetype": "generic_humanoid", "landmarks": []}


def reference_image_subject_cues(image_path: Path | None) -> dict[str, Any]:
    if image_path is None:
        return {}
    try:
        from PIL import Image

        image = Image.open(image_path).convert("RGBA")
        width, height = image.size
        aspect = float(width / max(height, 1))
        return {
            "image_aspect_ratio": aspect,
            "wide_reference": aspect >= float(os.environ.get("MESHMEND_REFERENCE_WIDE_SUBJECT_ASPECT", "1.12")),
        }
    except Exception:
        return {}


def build_enhanced_prompt(prompt: str, spec: dict[str, Any]) -> str:
    landmarks = ", ".join(spec["required_landmarks"])
    reference_sentence = ""
    if spec.get("input_image"):
        reference_sentence = (
            "Use the uploaded reference image as the primary subject source: preserve its silhouette, mount/creature/vehicle/body plan, "
            "pose, visible weapons, tail/wings/spikes/horns/armor plates/base shape, and distinctive accessories. Do not substitute a generic humanoid. "
        )
    return (
        f"{prompt.strip()}\n\n"
        "Create a final store/studio-quality resin-printable tabletop miniature, not concept art and not a primitive blockout. "
        f"Scale: {spec['scale_mm']}mm. Archetype: {spec['archetype']}. Weapon: {spec['weapon']}. "
        f"Required visible landmarks: {landmarks}. "
        f"{reference_sentence}"
        "Pipeline requirement: first create a complete concept/specification, then generate separate modular meshes for head, torso, legs, left arm, right arm, weapons, and accessories; assemble these modules into a complete miniature before sculpting. "
        "Run a primary sculpt pass for large anatomical and silhouette forms, a secondary sculpt pass for armor details/trims/vents/pouches, and a tertiary sculpt pass for micro-detail, engravings, insignia, and surface texture. "
        "Use clean 360-degree sculpt anatomy, readable face/helmet, hands/fingers or claws, bevelled weapons, layered armor/cloth/leather/metal details, "
        "deep recesses and raised trim that survive miniature painting, a fused scenic base, watertight topology, and commercial STL-ready geometry. "
        "Avoid and reject smooth blobs, flat bas-relief, planes, sheets, card-like depth, generic rocks, floating geometry, disconnected shells, non-manifold topology, open surfaces, and low-poly decimated surfaces. "
        "The final Miniature Quality Critic score must be at least 85/100 for silhouette quality, anatomical quality, armor design quality, detail density, printability, and professional resin miniature similarity."
    )


def load_backend_result(output_dir: Path, stdout: str) -> dict[str, Any]:
    result_path = output_dir / "result.json"
    if result_path.exists():
        return read_json(result_path)
    stripped = (stdout or "").strip()
    if stripped.startswith("{"):
        value = json.loads(stripped)
        return dict(value) if isinstance(value, dict) else {}
    for path in sorted(output_dir.iterdir(), key=lambda item: item.stat().st_mtime, reverse=True):
        if path.suffix.lower() in SUPPORTED_MODEL_SUFFIXES:
            return {"model_file": path.name}
    return {}


def resolve_model_path(output_dir: Path, result: dict[str, Any]) -> Path:
    model_file = str(result.get("model_file") or "").strip()
    if model_file:
        path = output_dir / Path(model_file).name
        if path.exists() and path.suffix.lower() in SUPPORTED_MODEL_SUFFIXES:
            return path
        raise RuntimeError(f"reported model_file is missing or unsupported: {model_file}")
    candidates = [path for path in output_dir.iterdir() if path.suffix.lower() in SUPPORTED_MODEL_SUFFIXES]
    if not candidates:
        raise RuntimeError("no supported model file found")
    return max(candidates, key=lambda item: item.stat().st_mtime)


def inspect_mesh(model_path: Path) -> dict[str, Any]:
    import numpy as np
    import trimesh

    mesh = trimesh.load(model_path, force="mesh", process=True)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
        raise RuntimeError("model is empty or not a Trimesh")
    components = len([part for part in mesh.split(only_watertight=False) if len(part.faces) > 20])
    edge_counts = np.bincount(mesh.edges_unique_inverse) if len(mesh.faces) else np.array([], dtype=int)
    boundary_edges = int((edge_counts == 1).sum()) if len(edge_counts) else 0
    non_manifold_edges = int((edge_counts > 2).sum()) if len(edge_counts) else 0
    extents = np.maximum(np.asarray(mesh.extents, dtype=float), 1e-6)
    smooth_surface_area_ratio = mesh_smooth_surface_area_ratio(mesh)
    sheet_card_artifact = False
    background_slab_artifact = False
    low_relief_sheet = False
    try:
        service_dir = Path(__file__).resolve().parent / "3dsculpter" / "model_service"
        if str(service_dir) not in sys.path:
            sys.path.insert(0, str(service_dir))
        from postprocess_backend import likely_background_slab, likely_horizontal_sheet_card, likely_image_low_relief_sheet

        sheet_card_artifact = bool(likely_horizontal_sheet_card(mesh))
        background_slab_artifact = bool(likely_background_slab(mesh, {"workflow": "image_to_3d"}))
        low_relief_sheet = bool(likely_image_low_relief_sheet(mesh))
    except Exception:
        pass
    return {
        "faces": int(len(mesh.faces)),
        "vertices": int(len(mesh.vertices)),
        "watertight": bool(mesh.is_watertight),
        "components": int(components),
        "boundary_edges": boundary_edges,
        "non_manifold_edges": non_manifold_edges,
        "extents_mm": [float(value) for value in extents],
        "depth_ratio": float(extents.min() / extents.max()),
        "sheet_card_artifact": sheet_card_artifact,
        "background_slab_artifact": background_slab_artifact,
        "low_relief_sheet": low_relief_sheet,
        "smooth_surface_area_ratio": smooth_surface_area_ratio,
        "validated_by_external_scaffold": True,
    }


def quality_issues(mesh_info: dict[str, Any], min_faces: int) -> list[str]:
    issues: list[str] = []
    face_tolerance = env_float("MESHMEND_CERTIFIED_FACE_TARGET_TOLERANCE", "0.005")
    effective_min_faces = int(min_faces * max(0.0, 1.0 - face_tolerance))
    if int(mesh_info.get("faces") or 0) < effective_min_faces:
        issues.append(f"faces_below_target:{mesh_info.get('faces')}<{effective_min_faces}")
    if not bool(mesh_info.get("watertight")):
        issues.append("mesh_not_watertight")
    if int(mesh_info.get("boundary_edges") or 0) > 0:
        issues.append(f"open_surfaces:{mesh_info.get('boundary_edges')}")
    if int(mesh_info.get("non_manifold_edges") or 0) > 0:
        issues.append(f"non_manifold_topology:{mesh_info.get('non_manifold_edges')}")
    max_components = env_int("MESHMEND_CERTIFIED_MAX_COMPONENTS", "3", fallback_name="MESHMEND_EXTERNAL_MAX_COMPONENTS")
    if int(mesh_info.get("components") or 0) > max_components:
        issues.append(f"too_many_components:{mesh_info.get('components')}>{max_components}")
    min_depth_ratio = env_float("MESHMEND_CERTIFIED_MIN_DEPTH_RATIO", "0.18", fallback_name="MESHMEND_EXTERNAL_MIN_DEPTH_RATIO")
    if float(mesh_info.get("depth_ratio") or 0.0) < min_depth_ratio:
        issues.append(f"too_flat:{mesh_info.get('depth_ratio')}<{min_depth_ratio}")
    if bool(mesh_info.get("sheet_card_artifact")):
        issues.append("sheet_card_artifact")
    if bool(mesh_info.get("background_slab_artifact")):
        issues.append("background_slab_artifact")
    if bool(mesh_info.get("low_relief_sheet")):
        issues.append("low_relief_sheet")
    smooth_limit = env_float("MESHMEND_CERTIFIED_MAX_SMOOTH_SURFACE_AREA_RATIO", "0.68")
    smooth_ratio = float(mesh_info.get("smooth_surface_area_ratio") or 0.0)
    if smooth_ratio > smooth_limit:
        issues.append(f"large_smooth_primitive_surfaces_dominate:{smooth_ratio:.2f}>{smooth_limit:.2f}")
    return issues


def mesh_smooth_surface_area_ratio(mesh: Any) -> float:
    import numpy as np

    if len(mesh.faces) == 0 or len(mesh.face_adjacency) == 0:
        return 1.0
    areas = np.asarray(mesh.area_faces, dtype=float)
    total_area = max(float(areas.sum()), 1e-8)
    adjacency = np.asarray(mesh.face_adjacency, dtype=np.int64)
    angles = np.asarray(mesh.face_adjacency_angles, dtype=float)
    smooth_faces = np.zeros(len(mesh.faces), dtype=bool)
    smooth_pairs = adjacency[angles < np.radians(7.5)]
    if len(smooth_pairs):
        smooth_faces[np.unique(smooth_pairs)] = True
    return float(areas[smooth_faces].sum() / total_area)


def required_quality_score_issues(result: dict[str, Any]) -> list[str]:
    if os.environ.get("MESHMEND_REQUIRE_EXTERNAL_QUALITY_SCORES", "1").strip().lower() in {"0", "false", "no", "off"}:
        return []
    scores = store_quality_scores(result)
    required = (
        "silhouette_quality",
        "anatomical_quality",
        "armor_design_quality",
        "detail_density_score",
        "printability_score",
        "professional_resin_similarity",
    )
    min_score = env_float("MESHMEND_CERTIFIED_MIN_QUALITY_SCORE", "85")
    issues: list[str] = []
    for key in required:
        if key not in scores:
            issues.append(f"missing_{key}")
            continue
        try:
            value = float(scores[key])
        except (TypeError, ValueError):
            issues.append(f"invalid_{key}:{scores[key]!r}")
            continue
        normalized = value * 100.0 if value <= 1.0 and min_score > 1.0 else value
        if normalized < min_score:
            issues.append(f"{key}_below_min:{normalized:.2f}<{min_score:.2f}")
    overall = scores.get("overall") or scores.get("critic_score")
    if overall is not None:
        try:
            overall_value = float(overall)
            overall_normalized = overall_value * 100.0 if overall_value <= 1.0 and min_score > 1.0 else overall_value
            if overall_normalized < min_score:
                issues.append(f"critic_overall_below_min:{overall_normalized:.2f}<{min_score:.2f}")
        except (TypeError, ValueError):
            issues.append(f"invalid_critic_overall:{overall!r}")
    certifier = str(scores.get("certifier") or result.get("certifier") or "").strip()
    if not certifier:
        issues.append("missing_certifier")
    return issues


def store_quality_scores(result: dict[str, Any]) -> dict[str, Any]:
    scores = result.get("store_quality_scores") or result.get("quality_scores") or {}
    if not isinstance(scores, dict):
        scores = {}
    mesh_info = result.get("mesh_info") or {}
    if isinstance(mesh_info, dict):
        for key in (
            "semantic_fidelity_score",
            "anatomy_score",
            "silhouette_quality",
            "anatomical_quality",
            "armor_design_quality",
            "detail_density_score",
            "surface_finish_score",
            "printability_score",
            "professional_resin_similarity",
            "overall",
            "critic_score",
            "certifier",
        ):
            if key not in scores and key in mesh_info:
                scores[key] = mesh_info[key]
    if "anatomical_quality" not in scores and "anatomy_score" in scores:
        scores["anatomical_quality"] = scores["anatomy_score"]
    if "detail_density_score" not in scores and "detail_density" in scores:
        scores["detail_density_score"] = scores["detail_density"]
    if "printability_score" not in scores and "printability" in scores:
        scores["printability_score"] = scores["printability"]
    if "silhouette_quality" not in scores and "semantic_fidelity_score" in scores:
        scores["silhouette_quality"] = scores["semantic_fidelity_score"]
    if "armor_design_quality" not in scores and "surface_finish_score" in scores:
        scores["armor_design_quality"] = scores["surface_finish_score"]
    if "professional_resin_similarity" not in scores and "surface_finish_score" in scores:
        scores["professional_resin_similarity"] = scores["surface_finish_score"]
    return dict(scores)


def env_float(name: str, default: str, *, fallback_name: str | None = None) -> float:
    raw = os.environ.get(name) or (os.environ.get(fallback_name) if fallback_name else None) or default
    return float(raw)


def env_int(name: str, default: str, *, fallback_name: str | None = None) -> int:
    raw = os.environ.get(name) or (os.environ.get(fallback_name) if fallback_name else None) or default
    return int(float(raw))


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    return dict(value) if isinstance(value, dict) else {}


def write_progress(output_dir: Path, progress: int, stage: str, message: str) -> None:
    payload = {"progress": progress, "stage": stage, "message": message, "updated_at": time.time()}
    (output_dir / "progress.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def fail(output_dir: Path, message: str, *, exit_code: int = 1, spec: dict[str, Any] | None = None, mesh_info: dict[str, Any] | None = None) -> int:
    payload = {
        "error": message,
        "provider": "meshmend_external_store_quality_generator",
        "capability_tier": "external_scaffold_unconfigured_or_failed",
        "store_quality_certified": False,
        "miniature_spec": spec or {},
        "mesh_info": mesh_info or {},
    }
    (output_dir / "result.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_progress(output_dir, 100, "external_failed", message)
    print(json.dumps(payload), file=sys.stderr)
    return exit_code


def command_args(command: str) -> list[str]:
    if os.name == "nt":
        return [part[1:-1] if len(part) >= 2 and part[0] == part[-1] == '"' else part for part in shlex.split(command, posix=False)]
    return shlex.split(command, posix=True)


if __name__ == "__main__":
    raise SystemExit(main())
