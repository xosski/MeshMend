from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class VisionEvidence:
    provider: str
    confidence: float
    subject: str
    archetype: str
    pose: str
    style_tags: list[str]
    detected_parts: list[dict[str, Any]]
    silhouette: dict[str, float]
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class RigPart:
    id: str
    kind: str
    primitive: str
    anchor: str
    side: str | None
    center: list[float]
    scale: list[float]
    rotation_deg: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    required: bool = True
    confidence: float = 1.0


@dataclass(frozen=True)
class RigPlan:
    archetype: str
    parts: list[RigPart]
    pose_tags: list[str]
    unsupported_parts: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class DetailPlan:
    style: str
    materials: list[str]
    zones: list[dict[str, Any]]
    motifs: list[str]
    min_feature_mm: float = 0.18
    surface_language: list[str] = field(default_factory=list)
    material_zones: list[dict[str, Any]] = field(default_factory=list)
    required_landmarks: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class MiniaturePlan:
    spec: dict[str, Any]
    evidence: VisionEvidence
    rig: RigPlan
    details: DetailPlan
    blockers: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


SUPPORTED_ARCHETYPES = {"armored_humanoid", "orc_warrior", "mounted_beast", "armored_mech"}
SUPPORTED_PARTS = {
    "head",
    "torso",
    "pelvis",
    "left_arm",
    "right_arm",
    "left_leg",
    "right_leg",
    "weapon",
    "shield",
    "cape",
    "wing_left",
    "wing_right",
    "tail",
    "horn",
    "shoulder_pad",
    "armor_plate",
    "beast_body",
    "beast_head",
    "beast_leg",
    "saddle",
    "rider",
    "base",
    "skull_or_rock_base_detail",
    "banner",
    "hood",
    "helmet",
    "face_detail",
    "tusk",
    "trophy_skull",
    "belt",
    "pouch",
    "chain",
    "tabard",
    "plume",
    "kneepad",
    "gauntlet",
    "boot",
    "beast_armor_plate",
    "scale_row",
    "claw",
    "tooth",
    "base_rubble",
}


def plan_miniature(
    request: dict[str, Any],
    image_path: Path | None,
    output_dir: Path,
    local_spec: Any,
    local_concept: dict[str, Any],
) -> MiniaturePlan:
    provider = os.environ.get("MESHMEND_SCULPT_PLANNER_PROVIDER", "local_heuristic").strip().lower() or "local_heuristic"
    workflow = str(request.get("workflow") or "text_to_3d")
    require_ai = os.environ.get("MESHMEND_REQUIRE_AI_SCULPT_PLANNER", "0").strip().lower() in {"1", "true", "yes", "on"}
    require_ai_for_text = os.environ.get("MESHMEND_REQUIRE_AI_SCULPT_PLANNER_FOR_TEXT", "0").strip().lower() in {"1", "true", "yes", "on"}
    ai_required = require_ai and (workflow == "image_to_3d" or image_path is not None or require_ai_for_text)
    local_plan = local_heuristic_plan(local_spec, local_concept, image_path, request)
    if provider == "command" or (ai_required and bundled_ollama_planner_path() is not None):
        try:
            command_plan = command_planner_plan(request, image_path, output_dir, local_plan)
            blockers = validate_plan(command_plan)
            return MiniaturePlan(command_plan.spec, command_plan.evidence, command_plan.rig, command_plan.details, command_plan.blockers + blockers)
        except Exception as exc:
            blocker = f"ai_planner_optional_command_failed:{exc}"
            if ai_required:
                blocker = f"ai_planner_required_but_failed:{exc}"
            return MiniaturePlan(
                local_plan.spec,
                local_plan.evidence,
                local_plan.rig,
                local_plan.details,
                local_plan.blockers + [blocker],
            )
    if ai_required:
        return MiniaturePlan(
            local_plan.spec,
            local_plan.evidence,
            local_plan.rig,
            local_plan.details,
            local_plan.blockers + ["ai_planner_required_but_not_configured"],
        )
    return MiniaturePlan(local_plan.spec, local_plan.evidence, local_plan.rig, local_plan.details, local_plan.blockers + validate_plan(local_plan))


def local_heuristic_plan(local_spec: Any, local_concept: dict[str, Any], image_path: Path | None, request: dict[str, Any] | None = None) -> MiniaturePlan:
    spec = local_spec.to_dict() if hasattr(local_spec, "to_dict") else dict(getattr(local_spec, "__dict__", {}))
    archetype = str(spec.get("archetype") or "armored_humanoid")
    concept_cues = list(spec.get("concept_cues") or [])
    detected_parts = [{"kind": cue, "label": cue.replace("_", " "), "confidence": 0.55} for cue in concept_cues]
    request = request or {}
    workflow = str(request.get("workflow") or "text_to_3d")
    prompt = str(request.get("prompt") or "").strip()
    prompt_terms = [term for term in prompt.replace("\n", " ").split(" ") if len(term.strip()) > 2]
    confidence = 0.62 if image_path is not None else (0.58 if len(prompt_terms) >= 8 else 0.5 if len(prompt_terms) >= 3 else 0.38)
    warnings = []
    blockers = []
    prompt_landmarks = prompt_required_landmarks(prompt)
    concept_landmarks = concept_required_landmarks(local_concept, spec)
    if image_path is None and workflow == "image_to_3d":
        warnings.append("no_concept_image_received")
        blockers.append("no_concept_image_received")
    elif image_path is not None and not concept_landmarks:
        warnings.append("weak_local_concept_landmarks")
    if image_path is not None and workflow == "image_to_3d" and high_detail_requested(request) and not allow_local_image_heuristic_high_detail():
        warnings.append("local_heuristic_image_planner_not_concept_fidelity_safe")
        blockers.append("image_to_3d_requires_ai_vision_planner_for_concept_fidelity")
    evidence = VisionEvidence(
        provider="local_heuristic",
        confidence=confidence,
        subject=subject_from_spec(spec),
        archetype=archetype,
        pose=str(spec.get("pose") or "heroic_contrapposto"),
        style_tags=list(spec.get("detail_language") or []),
        detected_parts=detected_parts,
        silhouette={key: float(value) for key, value in dict(spec.get("proportions") or {}).items()},
        warnings=warnings,
    )
    rig = default_rig_plan(archetype, spec)
    details = default_detail_plan(archetype, spec, sorted(set(prompt_landmarks + concept_landmarks)))
    return MiniaturePlan(spec, evidence, rig, details, blockers)


def command_planner_plan(request: dict[str, Any], image_path: Path | None, output_dir: Path, local_plan: MiniaturePlan) -> MiniaturePlan:
    command_template = os.environ.get("MESHMEND_SCULPT_PLANNER_COMMAND", "").strip() or bundled_ollama_planner_command()
    if not command_template:
        raise RuntimeError("MESHMEND_SCULPT_PLANNER_COMMAND is not configured")
    prompt_path = output_dir / "planner_prompt.txt"
    schema_path = output_dir / "planner_schema.json"
    plan_path = output_dir / "ai_miniature_plan.json"
    prompt_path.write_text(str(request.get("prompt") or ""), encoding="utf-8")
    schema_path.write_text(json.dumps(planner_schema(), indent=2), encoding="utf-8")
    command = command_template.format(
        image_path=str(image_path or ""),
        prompt_path=str(prompt_path),
        schema_path=str(schema_path),
        plan_path=str(plan_path),
        output_dir=str(output_dir),
        local_plan_path=str(write_local_plan(output_dir, local_plan)),
    )
    command_args = planner_command_args(command, image_path)
    completed = subprocess.run(
        command_args,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=float(os.environ.get("MESHMEND_SCULPT_PLANNER_TIMEOUT_SECONDS", "120")),
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or completed.stdout.strip() or f"planner command exited {completed.returncode}")
    raw = json.loads(plan_path.read_text(encoding="utf-8") if plan_path.exists() else completed.stdout)
    return miniature_plan_from_dict(raw, fallback=local_plan)


def bundled_ollama_planner_command() -> str:
    """Return the bundled Ollama planner command when available.

    Several local setups configured MESHMEND_SCULPT_PLANNER_COMMAND to point at
    the model_service directory, but the bundled template lives at the project
    root. Falling back here lets high-detail image jobs use vision planning
    instead of silently collapsing to generic local heuristics.
    """
    script = bundled_ollama_planner_path()
    if script is None:
        return ""
    return f'"{sys.executable}" "{script}" --image "{{image_path}}" --prompt "{{prompt_path}}" --schema "{{schema_path}}" --out "{{plan_path}}"'


def bundled_ollama_planner_path() -> Path | None:
    candidates = [
        Path(__file__).resolve().parent / "sculpt_vision_planner_ollama.template.py",
        Path(__file__).resolve().parents[2] / "sculpt_vision_planner_ollama.template.py",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def resolve_planner_command_args(args: list[str]) -> list[str]:
    bundled = bundled_ollama_planner_path()
    if bundled is None:
        return args
    resolved: list[str] = []
    for arg in args:
        if Path(arg).name == "sculpt_vision_planner_ollama.template.py" and not Path(arg).exists():
            resolved.append(str(bundled))
        else:
            resolved.append(arg)
    return resolved


def planner_command_args(command: str, image_path: Path | None) -> list[str]:
    args = resolve_planner_command_args(shlex.split(command, posix=os.name != "nt"))
    if image_path is not None:
        return args
    cleaned: list[str] = []
    skip_next = False
    for index, arg in enumerate(args):
        if skip_next:
            skip_next = False
            continue
        if arg == "--image":
            next_arg = args[index + 1] if index + 1 < len(args) else ""
            if not next_arg or next_arg.lower() in {"none", "null"} or next_arg.startswith("--"):
                skip_next = bool(next_arg) and not next_arg.startswith("--")
                continue
        cleaned.append(arg)
    return cleaned


def write_local_plan(output_dir: Path, local_plan: MiniaturePlan) -> Path:
    path = output_dir / "local_heuristic_plan.json"
    path.write_text(json.dumps(local_plan.to_dict(), indent=2), encoding="utf-8")
    return path


def miniature_plan_from_dict(raw: dict[str, Any], fallback: MiniaturePlan) -> MiniaturePlan:
    spec = dict(fallback.spec)
    spec.update(dict(raw.get("spec") or {}))
    evidence_raw = dict(raw.get("evidence") or {})
    evidence = VisionEvidence(
        provider=str(evidence_raw.get("provider") or "command"),
        confidence=float(evidence_raw.get("confidence") or 0.0),
        subject=str(evidence_raw.get("subject") or fallback.evidence.subject),
        archetype=str(evidence_raw.get("archetype") or spec.get("archetype") or fallback.evidence.archetype),
        pose=str(evidence_raw.get("pose") or spec.get("pose") or fallback.evidence.pose),
        style_tags=list(evidence_raw.get("style_tags") or fallback.evidence.style_tags),
        detected_parts=list(evidence_raw.get("detected_parts") or fallback.evidence.detected_parts),
        silhouette=dict(evidence_raw.get("silhouette") or fallback.evidence.silhouette),
        warnings=list(evidence_raw.get("warnings") or []),
    )
    rig_raw = dict(raw.get("rig") or {})
    parts = [rig_part_from_dict(part) for part in list(rig_raw.get("parts") or []) if isinstance(part, dict)]
    rig = RigPlan(
        archetype=str(rig_raw.get("archetype") or spec.get("archetype") or fallback.rig.archetype),
        parts=parts or fallback.rig.parts,
        pose_tags=list(rig_raw.get("pose_tags") or fallback.rig.pose_tags),
        unsupported_parts=list(rig_raw.get("unsupported_parts") or []),
    )
    details_raw = dict(raw.get("details") or {})
    details = DetailPlan(
        style=str(details_raw.get("style") or fallback.details.style),
        materials=list(details_raw.get("materials") or fallback.details.materials),
        zones=list(details_raw.get("zones") or fallback.details.zones),
        motifs=list(details_raw.get("motifs") or fallback.details.motifs),
        min_feature_mm=float(details_raw.get("min_feature_mm") or fallback.details.min_feature_mm),
        surface_language=list(details_raw.get("surface_language") or fallback.details.surface_language),
        material_zones=list(details_raw.get("material_zones") or fallback.details.material_zones),
        required_landmarks=list(details_raw.get("required_landmarks") or fallback.details.required_landmarks),
    )
    return MiniaturePlan(spec, evidence, rig, details, list(raw.get("blockers") or []))


def rig_part_from_dict(raw: dict[str, Any]) -> RigPart:
    return RigPart(
        id=str(raw.get("id") or raw.get("kind") or "part"),
        kind=str(raw.get("kind") or "unsupported"),
        primitive=str(raw.get("primitive") or "ellipsoid"),
        anchor=str(raw.get("anchor") or "base_center"),
        side=None if raw.get("side") is None else str(raw.get("side")),
        center=[float(v) for v in list(raw.get("center") or [0.0, 0.0, 0.0])[:3]],
        scale=[float(v) for v in list(raw.get("scale") or [1.0, 1.0, 1.0])[:3]],
        rotation_deg=[float(v) for v in list(raw.get("rotation_deg") or [0.0, 0.0, 0.0])[:3]],
        required=bool(raw.get("required", True)),
        confidence=float(raw.get("confidence") or 1.0),
    )


def validate_plan(plan: MiniaturePlan) -> list[str]:
    blockers: list[str] = []
    archetype = str(plan.spec.get("archetype") or plan.rig.archetype)
    if archetype not in SUPPORTED_ARCHETYPES:
        blockers.append(f"unsupported_archetype_{archetype}")
    if plan.evidence.confidence < float(os.environ.get("MESHMEND_SCULPT_MIN_PLANNER_CONFIDENCE", "0.45")):
        blockers.append("low_vision_planner_confidence")
    unsupported = set(plan.rig.unsupported_parts)
    unsupported.update(part.kind for part in plan.rig.parts if part.kind not in SUPPORTED_PARTS)
    if unsupported:
        blockers.append("unsupported_parts_" + "_".join(sorted(unsupported)[:5]))
    if plan.details.min_feature_mm < 0.12:
        blockers.append("detail_features_below_printable_size")
    required_landmarks = [str(item) for item in list(plan.details.required_landmarks or [])]
    min_required_landmarks = int(os.environ.get("MESHMEND_SCULPT_MIN_REQUIRED_LANDMARKS", "2"))
    if plan.evidence.provider == "local_heuristic" and plan.evidence.warnings and "weak_local_concept_landmarks" in plan.evidence.warnings and len(required_landmarks) < min_required_landmarks:
        blockers.append("concept_match_insufficient_local_landmarks")
    if len(required_landmarks) < min_required_landmarks and plan.evidence.confidence >= 0.55:
        blockers.append("concept_match_too_few_required_landmarks")
    return blockers


def default_rig_plan(archetype: str, spec: dict[str, Any]) -> RigPlan:
    parts: list[RigPart]
    if archetype == "mounted_beast":
        parts = [
            RigPart("beast_body", "beast_body", "ellipsoid", "base_center", None, [0, 0, 8.2], [9.4, 3.25, 3.0]),
            RigPart("beast_head", "beast_head", "ellipsoid", "beast_body", None, [-7.4, -0.15, 10.2], [2.55, 1.45, 1.7]),
            RigPart("tail", "tail", "capsule", "beast_body", None, [12.8, 0, 7.6], [5.4, 0.34, 0.34]),
            RigPart("rider", "rider", "humanoid_rig", "saddle", None, [0.15, -0.15, 13.0], [0.56, 0.56, 0.56]),
            RigPart("weapon", "weapon", str(spec.get("weapon") or "sword"), "rider_right_hand", None, [3.3, -0.95, 20.0], [0.82, 0.16, 2.0]),
        ]
    else:
        parts = [
            RigPart("torso", "torso", "ellipsoid", "base_center", None, [0, 0, 13.2], [3.7, 2.0, 4.9]),
            RigPart("head", "head", "ellipsoid", "torso", None, [0, -0.25, 18.55], [1.35, 1.0, 1.55]),
            RigPart("weapon", "weapon", str(spec.get("weapon") or "sidearm"), "right_hand", None, [5.7, -1.2, 15.8], [1.2, 0.22, 2.4]),
        ]
    return RigPlan(archetype, parts, [str(spec.get("pose") or "heroic_contrapposto")])


def default_detail_plan(archetype: str, spec: dict[str, Any], required_landmarks: list[str] | None = None) -> DetailPlan:
    details = list(spec.get("detail_language") or [])
    zones = [{"zone": "global", "motifs": details[:8], "density": 0.55}]
    surface_language = ["clean_secondary_forms", "deep_panel_lines", "raised_trim", "printable_tertiary_stamps"]
    material_zones = [
        {"zone": "armor", "material": "hard_surface_plate", "motifs": ["panel_lines", "rivets", "raised_trim"]},
        {"zone": "base", "material": "stone_bone", "motifs": ["rocks", "skulls", "rubble"]},
    ]
    if archetype == "mounted_beast":
        zones = [
            {"zone": "beast_back", "motifs": ["spine_spikes", "scale_rows", "armor_plates"], "density": 0.8},
            {"zone": "rider", "motifs": ["layered_armor", "raised_trim", "helmet_or_hood"], "density": 0.7},
            {"zone": "base", "motifs": ["rocks", "skulls"], "density": 0.5},
        ]
        surface_language += ["overlapping_scale_rows", "large_dorsal_spines", "leather_saddle_straps"]
        material_zones += [
            {"zone": "beast", "material": "scaled_creature_hide", "motifs": ["scales", "claws", "teeth", "spines"]},
            {"zone": "saddle", "material": "leather_hardware", "motifs": ["straps", "buckles", "stitching"]},
        ]
    if "cape_or_cloak" in details:
        surface_language.append("cloth_fold_ridges")
        material_zones.append({"zone": "cloak", "material": "cloth", "motifs": ["folds", "tattered_edges"]})
    return DetailPlan(";".join(details[:5]) or archetype, details, zones, details, 0.18, surface_language, material_zones, required_landmarks or [])


def prompt_required_landmarks(prompt: str) -> list[str]:
    prompt = prompt.lower()
    landmarks: list[str] = []
    terms = {
        "space_marine_silhouette": ("space marine", "spacemarine", "adeptus", "primaris"),
        "oversized_pauldrons": ("space marine", "spacemarine", "pauldron", "pauldrons"),
        "power_pack": ("space marine", "spacemarine", "power pack", "backpack"),
        "helmet_visor": ("space marine", "spacemarine", "helmet", "visor"),
        "chest_emblem": ("space marine", "spacemarine", "aquila", "chest emblem"),
        "bolter_rifle": ("space marine", "spacemarine", "bolter"),
        "mounted_beast": ("mounted", "mount ", "rider", "cavalry", "steed", "war beast", "warbeast"),
        "reptilian_mount": ("reptile", "reptilian", "lizard", "dragon"),
        "sword": ("sword", "blade", "khopesh", "scimitar"),
        "axe": ("axe", "halberd"),
        "rifle": ("rifle", "gun", "bolter", "cannon"),
        "spear_or_staff": ("spear", "staff", "lance"),
        "shield": ("shield",),
        "robed_caster_silhouette": ("wizard", "mage", "sorcerer", "witch", "warlock", "cleric", "priest", "robe", "robed"),
        "lean_stealth_silhouette": ("rogue", "assassin", "ranger", "thief", "ninja", "dagger"),
        "heroic_knight_silhouette": ("knight", "paladin", "templar", "crusader", "champion"),
        "cape_or_cloak": ("cape", "cloak", "robe", "hood", "hooded"),
        "spikes_or_horns": ("spike", "spiked", "horn", "horned", "spine"),
        "demonic_horns": ("demon", "devil", "fiend", "tiefling"),
        "angelic_wings": ("angel", "celestial", "seraph"),
        "scales": ("scales", "scaled", "scale rows", "scale armor"),
        "saddle": ("saddle",),
        "skull_or_rock_base_detail": ("skull", "skulls", "bone", "bones", "rock", "rocks", "rubble"),
        "banner": ("banner", "standard"),
        "tusks": ("tusk", "tusks", "orc", "ork"),
    }
    for landmark, needles in terms.items():
        if any(needle in prompt for needle in needles):
            landmarks.append(landmark)
    if prompt_requests_creature_claws(prompt):
        landmarks.append("claws")
    return landmarks


def prompt_requests_creature_claws(prompt: str) -> bool:
    prompt = prompt.lower()
    if "fingers/claws" in prompt or "fingers or claws" in prompt or "fingers and claws" in prompt:
        return any(term in prompt for term in ("beast", "dragon", "monster", "creature", "reptile", "talon", "talons"))
    return any(term in prompt for term in ("claw", "claws", "talon", "talons"))


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


def allow_local_image_heuristic_high_detail() -> bool:
    return os.environ.get("MESHMEND_ALLOW_LOCAL_IMAGE_HEURISTIC_HIGH_DETAIL", "0").strip().lower() in {"1", "true", "yes", "on"}


def concept_required_landmarks(local_concept: dict[str, Any], spec: dict[str, Any]) -> list[str]:
    landmarks: list[str] = []
    cue_map = {
        "mounted_creature_silhouette": "mounted_beast",
        "quadruped_silhouette": "mounted_beast",
        "wide_upper_silhouette": "wings_or_large_back_silhouette",
        "flowing_back_mass": "cape_or_cloak",
        "side_panel": "shield_or_banner",
        "long_horizontal_weapon": "rifle",
        "tall_weapon": "spear_or_staff",
        "raised_blade": "sword",
        "blade_like": "sword",
        "hard_surface": "layered_armor",
        "high_edge_density": "busy_surface_detail_from_concept",
    }
    for cue, landmark in cue_map.items():
        if local_concept.get(cue):
            landmarks.append(landmark)
    return sorted(set(landmarks))


def subject_from_spec(spec: dict[str, Any]) -> str:
    archetype = str(spec.get("archetype") or "miniature")
    weapon = str(spec.get("weapon") or "")
    if archetype == "mounted_beast":
        return f"rider on armored beast with {weapon}".strip()
    return f"{archetype.replace('_', ' ')} with {weapon}".strip()


def planner_schema() -> dict[str, Any]:
    return {
        "required_top_level_keys": ["spec", "evidence", "rig", "details"],
        "supported_archetypes": sorted(SUPPORTED_ARCHETYPES),
        "supported_part_kinds": sorted(SUPPORTED_PARTS),
        "required_detail_contract": {
            "details.required_landmarks": "Subject-defining landmarks that must be visible in the generated miniature, e.g. sword, shield, cape_or_cloak, scales, claws, saddle, skull_or_rock_base_detail.",
            "details.surface_language": "Readable sculpt surface vocabulary: armor_panel_lines, rivets, cloth_folds, scale_rows, leather_straps, weapon_fullers, skulls, rubble.",
            "details.material_zones": "Region-to-material mapping. Each zone should include a zone name, material, and motifs list.",
        },
        "notes": "Return JSON only. Do not return mesh data or polygons. MeshMend generates STL geometry natively. For image_to_3d, required_landmarks must come from visible image evidence, not generic defaults.",
    }
