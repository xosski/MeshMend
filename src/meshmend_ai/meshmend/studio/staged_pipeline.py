from __future__ import annotations

from dataclasses import asdict, dataclass, field
import base64
import json
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any
import urllib.request

import numpy as np
import trimesh

from meshmend.core.io import save_mesh
from meshmend.core.mesh_ops import auto_scale_to_height, remesh_subdivide
from meshmend.export import export_slicer_ready
from meshmend.sculpt import SculptEngine
from meshmend.studio.assets import (
    AnchorPoint,
    ConnectionSocket,
    ModularAssetProvider,
    ModularMiniaturePart,
    PartCategory,
    default_anchors,
    default_sockets,
    validate_part,
)
from meshmend.studio.pipeline import StudioMiniatureSpec
from meshmend.studio.quality import MiniatureQualityCritic, MiniatureSculptQualityGate, StudioQualityGate, StudioQualityReport


STAGED_CATEGORIES: tuple[PartCategory, ...] = (
    PartCategory.HEAD,
    PartCategory.TORSO,
    PartCategory.LEGS,
    PartCategory.LEFT_ARM,
    PartCategory.RIGHT_ARM,
    PartCategory.WEAPONS,
    PartCategory.ACCESSORIES,
    PartCategory.BASE,
)

LEGACY_CATEGORY_ALIASES: dict[PartCategory, PartCategory] = {
    PartCategory.HELMET: PartCategory.HEAD,
    PartCategory.HEAD_HELMET: PartCategory.HEAD,
    PartCategory.CHEST_ARMOR: PartCategory.TORSO,
    PartCategory.TORSO_BODY: PartCategory.TORSO,
    PartCategory.SHOULDER_PADS: PartCategory.TORSO,
    PartCategory.ARMS: PartCategory.LEFT_ARM,
    PartCategory.WEAPON: PartCategory.WEAPONS,
    PartCategory.BACKPACK: PartCategory.ACCESSORIES,
    PartCategory.BACKPACK_ACCESSORIES: PartCategory.ACCESSORIES,
}


@dataclass(slots=True)
class StageResult:
    name: str
    passed: bool
    issues: list[str] = field(default_factory=list)
    artifacts: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class GenerationFailed(RuntimeError):
    """Explicit stop: never replace a failed archetype with a mannequin."""

    def __init__(self, stage: str, detail: str, trace: dict[str, Any] | None = None) -> None:
        self.stage = stage
        self.detail = detail
        self.trace = trace or {}
        super().__init__(f"GENERATION FAILED: failing_stage={stage}: {detail}")


@dataclass(slots=True)
class MiniatureAssemblyResult:
    mesh: trimesh.Trimesh
    selected_parts: dict[PartCategory, ModularMiniaturePart]
    stage_results: list[StageResult]
    quality_report: StudioQualityReport


@dataclass(slots=True)
class MiniatureConceptDesign:
    faction: str
    role: str
    armor_class: str
    equipment: list[str]
    pose: str
    silhouette: str
    head_type: str
    armor_type: str
    backpack_type: str
    weapon_type: str
    pose_type: str
    faction_style: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


class ConceptGenerator:
    """Prompt -> miniature design brief, separate from mesh generation."""

    def generate(self, spec: StudioMiniatureSpec) -> MiniatureConceptDesign:
        return CharacterArchetypeGenerator().generate(spec)


@dataclass(slots=True)
class ArchetypeCandidate:
    kind: str
    name: str
    tags: list[str]
    silhouette_weight: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class CharacterArchetypeGenerator:
    """Design engine: faction, role, equipment, pose, and silhouette before mesh.

    This intentionally runs before any geometry exists. The output is not a
    humanoid rig; it is a miniature design brief a director can assemble from
    modular parts.
    """

    def generate(self, spec: StudioMiniatureSpec) -> MiniatureConceptDesign:
        text = spec.prompt.lower()
        archetype_text = str(getattr(spec, "archetype", "") or "").lower().replace("_", " ")
        searchable = f"{archetype_text} {text.replace('_', ' ')}"
        space_terminator = any(
            term in searchable
            for term in (
                "space terminator",
                "terminator",
                "tactical dreadnought",
                "space marine",
                "power armored space marine",
                "power armoured space marine",
                "power armored",
                "power armoured",
                "armored star knight",
                "armoured star knight",
                "star knight",
            )
        )
        if space_terminator:
            weapon = "heavy_storm_rifle" if any(term in searchable for term in ("rifle", "bolter", "gun")) else "power_fist_cannon"
            return MiniatureConceptDesign(
                faction="void_terminator_order",
                role="space_terminator",
                armor_class="massive_exo_plate_armor",
                equipment=["helmet_lenses", weapon, "reactor_backpack", "huge_pauldrons"],
                pose="slow_braced_advance",
                silhouette="towering_bulky_exo_armored_terminator_with_huge_pauldrons_and_heavy_weapon",
                head_type="recessed_exo_helmet",
                armor_type="massive_exo_plate_armor",
                backpack_type="reactor_backpack",
                weapon_type=weapon,
                pose_type="slow_braced_advance",
                faction_style="void_terminator_order",
            )
        astra_shock = any(term in text for term in ("astra shock", "shock trooper", "astra", "guardsman", "military resin", "heavy infantry", "line trooper"))
        if astra_shock:
            weapon = "las_rifle" if any(term in text for term in ("rifle", "las", "gun", "carbine")) else "field_carbine"
            return MiniatureConceptDesign(
                faction="astra_regiment",
                role="astra_shock_trooper",
                armor_class="flak_plate_and_fatigues",
                equipment=["helmet_lenses", weapon, "field_pack", "ammo_pouches"],
                pose="braced_firing_advance",
                silhouette="military_resin_shock_trooper_with_helmet_flak_armor_rifle_and_field_pack",
                head_type="field_helmet_rebreather",
                armor_type="flak_plate_and_fatigues",
                backpack_type="field_pack",
                weapon_type=weapon,
                pose_type="braced_firing_advance",
                faction_style="astra_regiment",
            )
        high_elf = any(term in text for term in ("high elf", "high-elf", "elf warrior", "elven warrior", "aelven"))
        if high_elf:
            weapon = "glaive" if "glaive" in text else "spear" if "spear" in text else "glaive"
            return MiniatureConceptDesign(
                faction="high_elf_host",
                role="high_elf_warrior",
                armor_class="layered_fantasy_armor",
                equipment=["pointed_ears_or_elf_helm", weapon, "cape_or_tabard", "leaf_plate_edges"],
                pose="heroic_posture",
                silhouette="tall_slender_elven_warrior_with_cape_and_pole_weapon",
                head_type="elf_helm",
                armor_type="layered_fantasy_armor",
                backpack_type="cape_or_tabard",
                weapon_type=weapon,
                pose_type="heroic_posture",
                faction_style="high_elf_host",
            )
        dwarf = any(term in text for term in ("dwarf", "dwarven", "duardin"))
        if dwarf:
            weapon = "hammer" if "hammer" in text else "axe" if "axe" in text else "runic_axe"
            return MiniatureConceptDesign(
                faction="dwarven_hold",
                role="dwarf_warrior",
                armor_class="runic_heavy_armor",
                equipment=["braided_beard", weapon, "round_shield", "rune_plate_edges"],
                pose="braced_guard",
                silhouette="short_broad_stocky_dwarf_with_beard_shield_and_axe",
                head_type="dwarf_helm_and_beard",
                armor_type="runic_heavy_armor",
                backpack_type="round_shield",
                weapon_type=weapon,
                pose_type="braced_guard",
                faction_style="dwarven_hold",
            )
        orc = any(term in text for term in ("orc", "ork", "brute", "greenskin"))
        if orc:
            weapon = "cleaver" if "cleaver" in text else "axe" if "axe" in text else "massive_choppa"
            return MiniatureConceptDesign(
                faction="orc_warband",
                role="orc_brute",
                armor_class="crude_scrap_armor",
                equipment=["tusks", "heavy_jaw", weapon, "spiked_scrap_plate"],
                pose="hunched_charge",
                silhouette="hunched_muscular_orc_brute_with_tusks_and_oversized_weapon",
                head_type="orc_tusk_head",
                armor_type="crude_scrap_armor",
                backpack_type="spike_trophy_rack",
                weapon_type=weapon,
                pose_type="hunched_charge",
                faction_style="orc_warband",
            )
        human_knight = any(term in text for term in ("human knight", "knight", "paladin", "templar", "crusader"))
        if human_knight:
            weapon = "sword" if "sword" in text else "mace" if "mace" in text else "longsword"
            return MiniatureConceptDesign(
                faction="human_kingdom",
                role="human_knight",
                armor_class="plate_armor",
                equipment=["crested_helm", weapon, "kite_shield", "surcoat_tabard"],
                pose="upright_guard",
                silhouette="upright_human_knight_with_crest_shield_sword_and_tabard",
                head_type="crested_knight_helm",
                armor_type="plate_armor",
                backpack_type="surcoat_tabard",
                weapon_type=weapon,
                pose_type="upright_guard",
                faction_style="human_kingdom",
            )
        faction = "astra_cultists" if any(term in text for term in ("astra", "cultist", "cult")) else "grimdark_expeditionary" if any(term in text for term in ("trench", "gas mask")) else f"original_{spec.style}_warband"
        role = "plasma_vanguard" if "plasma" in text else "rifle_infantry" if any(term in text for term in ("rifle", "gun", "carbine")) else "line_trooper"
        armor_class = "heavy_trench_armor" if any(term in text for term in ("trench", "cultist", "coat")) else "heavy_power_armor" if spec.style == "sci_fi" else "layered_plate_armor"
        equipment = ["weapon", "backpack", "pouches"]
        if any(term in text for term in ("gas mask", "cultist", "trench", "toxic")):
            equipment.append("gas_mask")
        if "plasma" in text:
            equipment.append("plasma_coils")
        if any(term in text for term in ("seal", "purity", "insignia")):
            equipment.append("purity_seals")
        if any(term in text for term in ("chain", "skull", "trophy")):
            equipment.append("trophies")
        pose = "advancing" if any(term in text for term in ("advancing", "running", "charge", "charging")) else "braced_firing" if any(term in text for term in ("rifle", "gun", "plasma")) else spec.pose
        silhouette = "masked_trench_vanguard" if "gas_mask" in equipment or armor_class == "heavy_trench_armor" else "broad_armored_rifleman" if role == "rifle_infantry" else "heroic_armored_specialist"
        head = "gas_mask" if any(term in text for term in ("gas mask", "cultist", "trench", "toxic")) else "helmet" if spec.helmet else "bare_head"
        armor = armor_class
        backpack = "reactor_vent_pack" if any(term in text for term in ("plasma", "vent", "reactor")) else "field_pack" if any(term in text for term in ("trench", "cultist", "pouch")) else "power_backpack"
        weapon = "plasma_carbine" if "plasma" in text else "rifle" if any(term in text for term in ("rifle", "gun", "carbine")) else spec.weapon
        return MiniatureConceptDesign(faction, role, armor_class, equipment, pose, silhouette, head, armor, backpack, weapon, pose, faction)

    def candidates(self, spec: StudioMiniatureSpec) -> dict[str, list[ArchetypeCandidate]]:
        design = self.generate(spec)
        return {
            "heads": self._head_candidates(design),
            "bodies": self._body_candidates(design),
            "weapons": self._weapon_candidates(design),
            "equipment": self._equipment_candidates(design),
            "poses": self._pose_candidates(design),
            "factions": self._faction_candidates(design),
        }

    def _head_candidates(self, design: MiniatureConceptDesign) -> list[ArchetypeCandidate]:
        names = [design.head_type, "rebreather_helmet", "hooded_mask", "sensor_visor", "bare_scarred_head"]
        return [ArchetypeCandidate("head", name, ["head", name, "faction_head_identity"], 0.9 if name == design.head_type else 0.55) for name in names]

    def _body_candidates(self, design: MiniatureConceptDesign) -> list[ArchetypeCandidate]:
        names = [design.armor_class, "long_coat_carapace", "segmented_plate", "hazard_suit", "breacher_armor"]
        return [ArchetypeCandidate("body", name, ["body", "torso", name, "wide_shoulders"], 0.92 if name == design.armor_class else 0.60) for name in names]

    def _weapon_candidates(self, design: MiniatureConceptDesign) -> list[ArchetypeCandidate]:
        names = [design.weapon_type, "rifle", "shotgun", "long_las", "blade_pistol"]
        return [ArchetypeCandidate("weapon", name, ["weapon", "weapon_barrel", name], 0.90 if name == design.weapon_type else 0.50) for name in names]

    def _equipment_candidates(self, design: MiniatureConceptDesign) -> list[ArchetypeCandidate]:
        names = list(dict.fromkeys([*design.equipment, "backpack", "pouches", "cables", "insignia", "trophies"]))
        return [ArchetypeCandidate("equipment", name, ["accessory", name], 0.75) for name in names[:20]]

    def _pose_candidates(self, design: MiniatureConceptDesign) -> list[ArchetypeCandidate]:
        names = [design.pose, "advancing", "braced_firing", "commanding", "reloading"]
        return [ArchetypeCandidate("pose", name, ["pose", name], 0.88 if name == design.pose else 0.50) for name in dict.fromkeys(names)]

    def _faction_candidates(self, design: MiniatureConceptDesign) -> list[ArchetypeCandidate]:
        names = [design.faction, "grimdark_expeditionary", "ash_waste_militia", "void_trenchers", "industrial_cult"]
        return [ArchetypeCandidate("faction", name, ["faction", name], 0.95 if name == design.faction else 0.55) for name in dict.fromkeys(names)]


class MiniatureDirector:
    """Selects archetype candidates before modular mesh creation."""

    def assemble_brief(self, candidates: dict[str, list[ArchetypeCandidate]]) -> dict[str, Any]:
        selected = {kind: max(options, key=lambda item: item.silhouette_weight) for kind, options in candidates.items() if options}
        return {
            "selected": {kind: candidate.to_dict() for kind, candidate in selected.items()},
            "silhouette_tags": sorted({tag for candidate in selected.values() for tag in candidate.tags}),
            "candidate_counts": {kind: len(options) for kind, options in candidates.items()},
        }


@dataclass(slots=True)
class ShapeLanguageProfile:
    archetype: str
    silhouette: str
    anatomy: list[str]
    armor: list[str]
    equipment: list[str]
    pose: list[str]
    required_silhouette_tags: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class MiniatureBlueprintGenerator:
    """Create the full non-geometry miniature blueprint before part meshes exist."""

    def generate(self, design: MiniatureConceptDesign, shape_language: ShapeLanguageProfile, ai_shape_plan: dict[str, Any]) -> dict[str, Any]:
        directives = dict(ai_shape_plan.get("part_directives") or {})
        return {
            "blueprint_type": "pre_geometry_miniature_blueprint",
            "archetype": shape_language.archetype,
            "silhouette": shape_language.silhouette,
            "race_faction": design.faction,
            "role": design.role,
            "required_traits": list(shape_language.required_silhouette_tags),
            "components": {
                "head": {"generator": "head_generator", "asset_intent": design.head_type, "shape_directive": directives.get("head", {})},
                "helmet": {"generator": "helmet_generator", "asset_intent": design.head_type, "shape_directive": directives.get("head", {})},
                "torso": {"generator": "torso_generator", "asset_intent": "narrow heroic torso" if design.role == "high_elf_warrior" else design.armor_type, "shape_directive": directives.get("torso", {})},
                "arms": {"generator": "arm_generator", "asset_intent": "long elegant arms" if design.role == "high_elf_warrior" else "archetype arms", "shape_directive": {"left": directives.get("left_arm", {}), "right": directives.get("right_arm", {})}},
                "legs": {"generator": "leg_generator", "asset_intent": "long elegant legs" if design.role == "high_elf_warrior" else "archetype legs", "shape_directive": directives.get("legs", {})},
                "hands": {"generator": "hand_generator", "asset_intent": "weapon grip hands", "shape_directive": {"left": directives.get("left_arm", {}), "right": directives.get("right_arm", {})}},
                "armor": {"generator": "armor_generator", "asset_intent": design.armor_type, "required": list(shape_language.armor)},
                "weapon": {"generator": "weapon_generator", "asset_intent": design.weapon_type, "shape_directive": directives.get("weapons", {})},
                "cloak": {"generator": "cloak_generator", "asset_intent": design.backpack_type, "shape_directive": directives.get("accessories", {})},
                "accessories": {"generator": "accessory_generator", "asset_intent": list(design.equipment), "shape_directive": directives.get("accessories", {})},
            },
            "no_geometry": True,
        }


class ShapeLanguageEngine:
    """Archetype-specific shape language before sculpt/detail generation."""

    def generate(self, spec: StudioMiniatureSpec, design: MiniatureConceptDesign) -> ShapeLanguageProfile:
        text = spec.prompt.lower()
        high_elf = any(term in text for term in ("high elf", "high-elf", "elf warrior", "elven warrior", "aelven"))
        if high_elf:
            return ShapeLanguageProfile(
                archetype="high_elf_warrior",
                silhouette="tall_slender_elven_warrior_with_cape_and_pole_weapon",
                anatomy=["elongated_limbs", "narrow_waist", "heroic_posture", "long_neck"],
                armor=["layered_fantasy_armor", "crested_elf_helm", "slender_pauldrons", "leaf_plate_edges"],
                equipment=["pointed_ears_or_elf_helm", "sword_spear_or_glaive", "cape_or_tabard"],
                pose=["upright_heroic_stance", "weapon_forward_readable"],
                required_silhouette_tags=[
                    "high_elf_warrior_shape",
                    "elongated_limbs",
                    "narrow_waist",
                    "heroic_posture",
                    "pointed_ears_or_elf_helm",
                    "layered_fantasy_armor",
                    "sword_spear_or_glaive",
                    "cape_or_tabard",
                ],
            )
        if design.role == "astra_shock_trooper":
            return ShapeLanguageProfile(
                archetype="astra_shock_trooper",
                silhouette="military_resin_shock_trooper_with_helmet_flak_armor_rifle_and_field_pack",
                anatomy=["human_military_proportions", "braced_firing_advance", "readable_hands"],
                armor=["flak_plate_and_fatigues", "helmet_lenses", "chest_flak_plate", "knee_greaves"],
                equipment=["las_rifle", "field_pack", "ammo_pouches"],
                pose=["braced_firing_advance", "rifle_across_body_readable"],
                required_silhouette_tags=[
                    "astra_shock_trooper_shape",
                    "human_military_proportions",
                    "braced_firing_advance",
                    "field_helmet_rebreather",
                    "flak_plate_and_fatigues",
                    "las_rifle",
                    "field_pack",
                    "hands",
                ],
            )
        if design.role == "space_terminator":
            return ShapeLanguageProfile(
                archetype="space_terminator",
                silhouette="towering_bulky_exo_armored_terminator_with_huge_pauldrons_and_heavy_weapon",
                anatomy=["bulky_exo_proportions", "massive_shoulders", "slow_braced_advance", "thick_power_limbs"],
                armor=["massive_exo_plate_armor", "huge_pauldrons", "recessed_exo_helmet", "reactor_backpack"],
                equipment=["heavy_storm_rifle", "power_fist_cannon", "helmet_lenses"],
                pose=["slow_braced_advance", "heavy_weapon_forward_readable"],
                required_silhouette_tags=[
                    "space_terminator_shape",
                    "bulky_exo_proportions",
                    "massive_shoulders",
                    "huge_pauldrons",
                    "recessed_exo_helmet",
                    "massive_exo_plate_armor",
                    "heavy_storm_rifle",
                    "reactor_backpack",
                    "hands",
                ],
            )
        if design.role == "dwarf_warrior":
            return ShapeLanguageProfile(
                archetype="dwarf_warrior",
                silhouette="short_broad_stocky_dwarf_with_beard_shield_and_axe",
                anatomy=["short_stocky_proportions", "broad_torso", "thick_limbs", "low_center_of_gravity"],
                armor=["runic_heavy_armor", "wide_shoulder_plates", "heavy_boots"],
                equipment=["braided_beard", "axe_or_hammer", "round_shield"],
                pose=["braced_guard", "grounded_stance"],
                required_silhouette_tags=[
                    "dwarf_warrior_shape",
                    "short_stocky_proportions",
                    "broad_torso",
                    "braided_beard",
                    "axe_or_hammer",
                    "round_shield",
                    "runic_heavy_armor",
                ],
            )
        if design.role == "orc_brute":
            return ShapeLanguageProfile(
                archetype="orc_brute",
                silhouette="hunched_muscular_orc_brute_with_tusks_and_oversized_weapon",
                anatomy=["hunched_posture", "massive_shoulders", "long_powerful_arms", "thick_legs"],
                armor=["crude_scrap_armor", "spiked_scrap_plate"],
                equipment=["tusks", "heavy_jaw", "oversized_choppa", "spike_trophy_rack"],
                pose=["hunched_charge", "weapon_overweight_readable"],
                required_silhouette_tags=[
                    "orc_brute_shape",
                    "hunched_posture",
                    "massive_shoulders",
                    "long_powerful_arms",
                    "tusks",
                    "heavy_jaw",
                    "oversized_choppa",
                    "crude_scrap_armor",
                ],
            )
        if design.role == "human_knight":
            return ShapeLanguageProfile(
                archetype="human_knight",
                silhouette="upright_human_knight_with_crest_shield_sword_and_tabard",
                anatomy=["human_heroic_proportions", "upright_guard", "balanced_limb_length"],
                armor=["plate_armor", "crested_helm", "clean_shoulder_plate"],
                equipment=["sword_or_longsword", "kite_shield", "surcoat_tabard"],
                pose=["upright_guard", "shield_forward_readable"],
                required_silhouette_tags=[
                    "human_knight_shape",
                    "human_heroic_proportions",
                    "crested_helm",
                    "plate_armor",
                    "kite_shield",
                    "sword_or_longsword",
                    "surcoat_tabard",
                ],
            )
        return ShapeLanguageProfile(
            archetype=design.role,
            silhouette=design.silhouette,
            anatomy=["heroic_scale", "readable_head_torso_limbs"],
            armor=[design.armor_class],
            equipment=list(design.equipment),
            pose=[design.pose],
            required_silhouette_tags=["head", "torso", "legs", "weapon", "backpack"],
        )


class StudioShapeAIPlanner:
    """AI director that turns a prompt into geometry-driving shape directives.

    The staged pipeline must not stop at keyword labels. This planner produces a
    structured design contract before any mesh exists: race/role proportions,
    per-part scale/offset directives, required silhouette features, and an audit
    trail showing whether an external AI planner or the bundled local semantic
    planner supplied the plan.
    """

    def generate(
        self,
        spec: StudioMiniatureSpec,
        design: MiniatureConceptDesign,
        shape_language: ShapeLanguageProfile,
    ) -> dict[str, Any]:
        command_plan = self._command_plan(spec, design, shape_language)
        if command_plan is not None:
            return self._normalize_plan(command_plan, spec, design, shape_language, source="external_ai_shape_planner")
        return self._local_semantic_plan(spec, design, shape_language)

    def _command_plan(
        self,
        spec: StudioMiniatureSpec,
        design: MiniatureConceptDesign,
        shape_language: ShapeLanguageProfile,
    ) -> dict[str, Any] | None:
        command_template = os.environ.get("MESHMEND_STUDIO_SHAPE_AI_COMMAND", "").strip()
        if not command_template:
            return None
        schema = self._planner_schema()
        with tempfile.TemporaryDirectory(prefix="meshmend_studio_shape_ai_") as directory:
            root = Path(directory)
            prompt_path = root / "prompt.txt"
            schema_path = root / "shape_plan_schema.json"
            plan_path = root / "shape_plan.json"
            prompt_path.write_text(spec.prompt, encoding="utf-8")
            schema_path.write_text(json.dumps(schema, indent=2), encoding="utf-8")
            command = command_template.format(
                prompt_path=str(prompt_path),
                schema_path=str(schema_path),
                plan_path=str(plan_path),
                design_json=json.dumps(design.to_dict()),
                shape_language_json=json.dumps(shape_language.to_dict()),
            )
            completed = subprocess.run(
                command,
                shell=True,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=float(os.environ.get("MESHMEND_STUDIO_SHAPE_AI_TIMEOUT_SECONDS", "120")),
            )
            if completed.returncode != 0:
                if os.environ.get("MESHMEND_REQUIRE_STUDIO_SHAPE_AI", "0").strip().lower() in {"1", "true", "yes", "on"}:
                    raise RuntimeError("studio shape AI command failed: " + (completed.stderr.strip() or completed.stdout.strip()))
                return None
            raw = plan_path.read_text(encoding="utf-8") if plan_path.exists() else completed.stdout
            return json.loads(raw)

    def _local_semantic_plan(
        self,
        spec: StudioMiniatureSpec,
        design: MiniatureConceptDesign,
        shape_language: ShapeLanguageProfile,
    ) -> dict[str, Any]:
        planner_subject = design.role
        planner_parts: list[dict[str, Any]] = []
        try:
            from llm_part_planner import LLMPartPlanner

            plan = LLMPartPlanner().create_plan(spec.prompt, style=design.role, scale_mm=spec.scale_mm)
            planner_subject = plan.subject
            planner_parts = [asdict(part) for part in plan.parts]
        except Exception:
            planner_parts = []
        role = design.role
        part_directives: dict[str, dict[str, Any]]
        if role == "high_elf_warrior":
            part_directives = {
                "head": {"scale": [0.92, 0.96, 1.08], "intent": "slender elven helm and ears"},
                "torso": {"scale": [0.88, 0.96, 1.06], "intent": "narrow waist and tall chest"},
                "legs": {"scale": [0.88, 0.95, 1.12], "intent": "elongated graceful legs"},
                "left_arm": {"scale": [0.88, 0.95, 1.08], "intent": "long slender arms"},
                "right_arm": {"scale": [0.88, 0.95, 1.08], "intent": "long slender arms"},
                "weapons": {"scale": [1.06, 1.06, 1.08], "intent": "long readable spear/glaive with printable thickness"},
                "accessories": {"scale": [0.90, 0.92, 1.08], "intent": "flowing cape/tabard silhouette"},
            }
        elif role == "astra_shock_trooper":
            part_directives = {
                "head": {"scale": [0.96, 0.98, 1.0], "intent": "field helmet with rebreather and lenses"},
                "torso": {"scale": [1.02, 0.98, 1.0], "intent": "military flak vest over fatigues"},
                "legs": {"scale": [0.98, 1.0, 1.02], "intent": "advancing human infantry legs"},
                "left_arm": {"scale": [0.98, 1.0, 1.0], "intent": "support hand under rifle"},
                "right_arm": {"scale": [0.98, 1.0, 1.0], "intent": "trigger hand holding rifle"},
                "weapons": {"scale": [1.16, 1.0, 0.98], "intent": "long las rifle across body"},
                "accessories": {"scale": [1.0, 1.02, 1.0], "intent": "field pack and ammo pouches"},
            }
        elif role == "space_terminator":
            part_directives = {
                "head": {"scale": [1.08, 1.02, 0.98], "intent": "small recessed helmet inside exo armor"},
                "torso": {"scale": [1.34, 1.18, 1.08], "intent": "huge exo torso and pauldrons"},
                "legs": {"scale": [1.20, 1.08, 0.96], "intent": "thick armored legs"},
                "left_arm": {"scale": [1.26, 1.10, 1.02], "intent": "oversized power fist arm"},
                "right_arm": {"scale": [1.22, 1.08, 1.02], "intent": "heavy weapon arm"},
                "weapons": {"scale": [1.28, 1.12, 1.0], "intent": "blocky heavy storm rifle with printable silhouette"},
                "accessories": {"scale": [1.18, 1.12, 1.04], "intent": "large reactor backpack"},
            }
        elif role == "dwarf_warrior":
            part_directives = {
                "head": {"scale": [1.08, 1.0, 0.92], "intent": "wide helm and beard mass"},
                "torso": {"scale": [1.16, 1.05, 0.88], "intent": "short broad armored torso"},
                "legs": {"scale": [1.10, 1.02, 0.78], "intent": "short thick legs"},
                "left_arm": {"scale": [1.10, 1.02, 0.88], "intent": "compact powerful arm"},
                "right_arm": {"scale": [1.10, 1.02, 0.88], "intent": "compact powerful arm"},
                "weapons": {"scale": [1.05, 1.0, 0.92], "intent": "stout axe or hammer"},
                "accessories": {"scale": [1.12, 1.0, 0.92], "intent": "round shield and runic mass"},
            }
        elif role == "orc_brute":
            part_directives = {
                "head": {"scale": [1.18, 1.08, 0.95], "intent": "heavy jaw tusk head"},
                "torso": {"scale": [1.18, 1.08, 0.96], "offset": [0.0, 0.0, -0.25], "intent": "hunched massive shoulders"},
                "legs": {"scale": [1.05, 1.02, 0.92], "intent": "thick squat charging legs"},
                "left_arm": {"scale": [1.22, 1.06, 1.12], "intent": "long powerful brute arm"},
                "right_arm": {"scale": [1.22, 1.06, 1.12], "intent": "long powerful brute arm"},
                "weapons": {"scale": [1.24, 1.08, 1.05], "intent": "oversized crude choppa"},
                "accessories": {"scale": [1.12, 1.0, 1.05], "intent": "spikes and trophy rack"},
            }
        elif role == "human_knight":
            part_directives = {
                "head": {"scale": [1.0, 1.0, 1.05], "intent": "crested knight helm"},
                "torso": {"scale": [1.0, 1.0, 1.02], "intent": "upright balanced plate armor"},
                "legs": {"scale": [0.98, 1.0, 1.02], "intent": "balanced human legs"},
                "left_arm": {"scale": [1.0, 1.0, 1.0], "intent": "shield arm"},
                "right_arm": {"scale": [1.0, 1.0, 1.0], "intent": "sword arm"},
                "weapons": {"scale": [1.0, 1.0, 1.08], "intent": "longsword readable in silhouette"},
                "accessories": {"scale": [1.04, 1.0, 1.05], "intent": "kite shield and tabard"},
            }
        else:
            part_directives = {
                "head": {"scale": [1.0, 1.0, 1.0], "intent": "readable head"},
                "torso": {"scale": [1.0, 1.0, 1.0], "intent": "readable torso"},
                "legs": {"scale": [1.0, 1.0, 1.0], "intent": "readable legs"},
                "left_arm": {"scale": [1.0, 1.0, 1.0], "intent": "readable left arm"},
                "right_arm": {"scale": [1.0, 1.0, 1.0], "intent": "readable right arm"},
                "weapons": {"scale": [1.0, 1.0, 1.0], "intent": "readable weapon"},
                "accessories": {"scale": [1.0, 1.0, 1.0], "intent": "readable accessories"},
            }
        return self._normalize_plan(
            {
                "source": "local_semantic_ai_shape_planner",
                "confidence": 0.82 if role != "line_trooper" else 0.62,
                "archetype": role,
                "subject": planner_subject,
                "part_directives": part_directives,
                "required_silhouette_features": list(shape_language.required_silhouette_tags),
                "planner_parts": planner_parts,
            },
            spec,
            design,
            shape_language,
            source="local_semantic_ai_shape_planner",
        )

    def _normalize_plan(
        self,
        plan: dict[str, Any],
        spec: StudioMiniatureSpec,
        design: MiniatureConceptDesign,
        shape_language: ShapeLanguageProfile,
        *,
        source: str,
    ) -> dict[str, Any]:
        directives = dict(plan.get("part_directives") or {})
        normalized: dict[str, Any] = {
            "source": str(plan.get("source") or source),
            "confidence": float(plan.get("confidence") or 0.5),
            "archetype": str(plan.get("archetype") or design.role),
            "subject": str(plan.get("subject") or design.role),
            "scale_mm": float(spec.scale_mm),
            "part_directives": {},
            "required_silhouette_features": list(plan.get("required_silhouette_features") or shape_language.required_silhouette_tags),
            "planner_parts": list(plan.get("planner_parts") or []),
        }
        for category in STAGED_CATEGORIES:
            raw = dict(directives.get(category.value) or {})
            scale = list(raw.get("scale") or [1.0, 1.0, 1.0])[:3]
            offset = list(raw.get("offset") or [0.0, 0.0, 0.0])[:3]
            normalized["part_directives"][category.value] = {
                "scale": [float(value) for value in [*scale, 1.0, 1.0, 1.0][:3]],
                "offset": [float(value) for value in [*offset, 0.0, 0.0, 0.0][:3]],
                "intent": str(raw.get("intent") or category.value),
            }
        return normalized

    def _planner_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "required": ["archetype", "part_directives", "required_silhouette_features"],
            "properties": {
                "archetype": {"type": "string"},
                "subject": {"type": "string"},
                "confidence": {"type": "number"},
                "required_silhouette_features": {"type": "array", "items": {"type": "string"}},
                "part_directives": {
                    "type": "object",
                    "additionalProperties": {
                        "type": "object",
                        "properties": {
                            "scale": {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3},
                            "offset": {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3},
                            "intent": {"type": "string"},
                        },
                    },
                },
            },
        }


@dataclass(slots=True)
class SilhouetteCritic:
    """Reject blockouts before detail if the unit is not recognizable in silhouette."""

    minimum_score: float = 0.78

    def evaluate(self, mesh: trimesh.Trimesh, concept: MiniatureConceptDesign, shape_language: ShapeLanguageProfile | None = None) -> tuple[bool, list[str], dict[str, Any]]:
        components = set(str(item) for item in mesh.metadata.get("studio_components", []))
        bounds = np.asarray(mesh.bounds, dtype=float)
        extents = np.maximum(bounds[1] - bounds[0], 1e-6)
        width_height = float(extents[0] / max(extents[2], 1e-6))
        depth_height = float(extents[1] / max(extents[2], 1e-6))
        identity_tags = {
            "gas_mask_filter",
            "trench_coat_hem",
            "oversized_backpack",
            "plasma_coil",
            "weapon_barrel",
            "wide_shoulders",
            "advancing_stride",
            "dwarf_warrior_shape",
            "short_stocky_proportions",
            "broad_torso",
            "braided_beard",
            "axe_or_hammer",
            "round_shield",
            "orc_brute_shape",
            "hunched_posture",
            "massive_shoulders",
            "long_powerful_arms",
            "tusks",
            "oversized_choppa",
            "human_knight_shape",
            "human_heroic_proportions",
            "crested_helm",
            "kite_shield",
            "sword_or_longsword",
            "surcoat_tabard",
            "astra_shock_trooper_shape",
            "human_military_proportions",
            "braced_firing_advance",
            "field_helmet_rebreather",
            "flak_plate_and_fatigues",
            "las_rifle",
            "field_pack",
            "space_terminator_shape",
            "bulky_exo_proportions",
            "huge_pauldrons",
            "recessed_exo_helmet",
            "massive_exo_plate_armor",
            "heavy_storm_rifle",
            "reactor_backpack",
        }
        present_identity = sorted(components & identity_tags)
        score = 0.35 + min(len(present_identity) / 6.0, 1.0) * 0.45
        if 0.35 <= width_height <= 1.35 and 0.18 <= depth_height <= 1.25:
            score += 0.10
        issues: list[str] = []
        if concept.head_type != "bare_head" and any(tag in components for tag in ("helmet", "gas_mask_filter", "helmet_lenses")):
            score += 0.05
        if concept.weapon_type and any(tag in components for tag in ("weapon", "weapon_barrel", "plasma_coil")):
            score += 0.05
        if any(tag in components for tag in ("primitive_sphere_body", "primitive_capsule_limb", "primitive_cylinder_core", "mannequin_rig", "mannequin_core")):
            issues.append("primitive_or_mannequin_base_body_rejected")
        if shape_language is not None:
            missing_shape_tags = [tag for tag in shape_language.required_silhouette_tags if tag not in components]
            if missing_shape_tags:
                issues.append("missing_shape_language_silhouette_tags:" + ",".join(missing_shape_tags))
            if shape_language.archetype == "high_elf_warrior" and len(missing_shape_tags) == 0:
                score = max(score, 0.95)
            if shape_language.archetype in {"dwarf_warrior", "orc_brute", "human_knight", "astra_shock_trooper", "space_terminator"} and len(missing_shape_tags) == 0:
                score = max(score, 0.93)
        if score < self.minimum_score:
            issues.append(f"silhouette_not_identifiable:{score:.2f}<{self.minimum_score:.2f}")
        if shape_language is None or shape_language.archetype != "high_elf_warrior":
            if len(present_identity) < 4:
                issues.append("insufficient_unit_identity_silhouette_tags:" + ",".join(present_identity))
        elif score < self.minimum_score:
            issues.append("insufficient_unit_identity_silhouette_tags:" + ",".join(present_identity))
        metrics = {
            "score": round(score, 3),
            "present_identity_tags": present_identity,
            "width_height_ratio": round(width_height, 3),
            "depth_height_ratio": round(depth_height, 3),
            "concept": concept.to_dict(),
            "shape_language": shape_language.to_dict() if shape_language else None,
        }
        return not issues, issues, metrics


@dataclass(slots=True)
class PreSculptRecognizabilityGate:
    """Brutal pre-sculpt gate: reject abstract/mannequin assemblies immediately."""

    min_thumbnail_occupancy: float = 0.055
    min_thumbnail_span: float = 0.55

    def evaluate(self, mesh: trimesh.Trimesh, concept: MiniatureConceptDesign, shape_language: ShapeLanguageProfile | None = None) -> tuple[bool, list[str], dict[str, Any]]:
        components = set(str(item) for item in mesh.metadata.get("studio_components", []))
        checks = {
            "head_recognizable": bool(components & {"head", "helmet", "gas_mask_filter", "helmet_lenses"}),
            "torso_recognizable": bool(components & {"torso", "body", "chest_armor", "heavy_trench_armor", "trench_coat_hem"}),
            "arms_recognizable": {"left_arm", "right_arm"}.issubset(components) or "arms" in components,
            "legs_recognizable": bool(components & {"legs", "left_leg", "right_leg", "advancing_stride", "greaves"}),
            "weapon_recognizable": bool(components & {"weapon", "weapon_barrel", "plasma_coil", "plasma_carbine", "muzzle_detail"}),
        }
        if shape_language is not None and shape_language.archetype == "high_elf_warrior":
            checks.update(
                {
                    "high_elf_silhouette_recognizable": "high_elf_warrior_shape" in components,
                    "elongated_limbs_recognizable": "elongated_limbs" in components,
                    "narrow_waist_recognizable": "narrow_waist" in components,
                    "elf_head_recognizable": "pointed_ears_or_elf_helm" in components,
                    "fantasy_weapon_recognizable": "sword_spear_or_glaive" in components,
                    "cape_or_tabard_recognizable": "cape_or_tabard" in components,
                    "hands_recognizable": "hands" in components,
                    "authored_shape_library_used": "authored_shape_library" in components,
                }
            )
        elif shape_language is not None and shape_language.archetype == "dwarf_warrior":
            checks.update(
                {
                    "dwarf_silhouette_recognizable": "dwarf_warrior_shape" in components,
                    "short_stocky_proportions_recognizable": "short_stocky_proportions" in components,
                    "broad_torso_recognizable": "broad_torso" in components,
                    "beard_recognizable": "braided_beard" in components,
                    "dwarf_weapon_recognizable": "axe_or_hammer" in components,
                    "round_shield_recognizable": "round_shield" in components,
                }
            )
        elif shape_language is not None and shape_language.archetype == "orc_brute":
            checks.update(
                {
                    "orc_silhouette_recognizable": "orc_brute_shape" in components,
                    "hunched_posture_recognizable": "hunched_posture" in components,
                    "massive_shoulders_recognizable": "massive_shoulders" in components,
                    "long_powerful_arms_recognizable": "long_powerful_arms" in components,
                    "tusks_recognizable": "tusks" in components,
                    "oversized_weapon_recognizable": "oversized_choppa" in components,
                }
            )
        elif shape_language is not None and shape_language.archetype == "human_knight":
            checks.update(
                {
                    "human_knight_silhouette_recognizable": "human_knight_shape" in components,
                    "human_proportions_recognizable": "human_heroic_proportions" in components,
                    "crested_helm_recognizable": "crested_helm" in components,
                    "kite_shield_recognizable": "kite_shield" in components,
                    "sword_recognizable": "sword_or_longsword" in components,
                    "tabard_recognizable": "surcoat_tabard" in components,
                }
            )
        elif shape_language is not None and shape_language.archetype == "astra_shock_trooper":
            checks.update(
                {
                    "astra_silhouette_recognizable": "astra_shock_trooper_shape" in components,
                    "military_proportions_recognizable": "human_military_proportions" in components,
                    "helmet_rebreather_recognizable": "field_helmet_rebreather" in components,
                    "flak_armor_recognizable": "flak_plate_and_fatigues" in components,
                    "rifle_recognizable": "las_rifle" in components,
                    "field_pack_recognizable": "field_pack" in components,
                    "hands_recognizable": "hands" in components,
                }
            )
        elif shape_language is not None and shape_language.archetype == "space_terminator":
            checks.update(
                {
                    "terminator_silhouette_recognizable": "space_terminator_shape" in components,
                    "bulky_exo_proportions_recognizable": "bulky_exo_proportions" in components,
                    "huge_pauldrons_recognizable": "huge_pauldrons" in components,
                    "exo_helmet_recognizable": "recessed_exo_helmet" in components,
                    "massive_armor_recognizable": "massive_exo_plate_armor" in components,
                    "heavy_weapon_recognizable": "heavy_storm_rifle" in components,
                    "reactor_backpack_recognizable": "reactor_backpack" in components,
                    "hands_recognizable": "hands" in components,
                }
            )
        thumbnail = _silhouette_thumbnail_metrics(mesh, resolution=64)
        issues = [name for name, passed in checks.items() if not passed]
        if thumbnail["occupancy"] < self.min_thumbnail_occupancy:
            issues.append(f"silhouette_64_thumbnail_too_sparse:{thumbnail['occupancy']:.3f}<{self.min_thumbnail_occupancy:.3f}")
        if thumbnail["height_span"] < self.min_thumbnail_span or thumbnail["width_span"] < self.min_thumbnail_span * 0.35:
            issues.append("silhouette_64_thumbnail_not_readable")
        if any(tag in components for tag in ("primitive_sphere_body", "primitive_capsule_limb", "primitive_cylinder_core", "mannequin_rig", "mannequin_core")):
            issues.append("primitive_or_mannequin_base_body_rejected")
        if shape_language is not None and shape_language.archetype == "high_elf_warrior":
            primitive_leaks = sorted(components & {"cube_head", "cylinder_arm", "cylinder_leg", "box_torso", "primitive_weapon_extrusion"})
            if primitive_leaks:
                issues.append("primitive_shapes_visible_in_high_elf_assembly:" + ",".join(primitive_leaks))
            if "authored_shape_library" not in components:
                issues.append("high_elf_authored_shape_library_missing")
        metrics = {
            "part_checks": checks,
            "silhouette_64_thumbnail": thumbnail,
            "black_silhouette_previews": render_black_silhouette_previews(mesh),
            "concept": concept.to_dict(),
            "shape_language": shape_language.to_dict() if shape_language else None,
        }
        return not issues, issues, metrics


@dataclass(slots=True)
class MannequinDetector:
    """Hard pre-sculpt rejector for mannequin/capsule/cube/voxel outputs."""

    max_mannequin_score: float = 0.10

    def evaluate(self, mesh: trimesh.Trimesh, concept: MiniatureConceptDesign, shape_language: ShapeLanguageProfile | None = None) -> tuple[bool, list[str], dict[str, Any]]:
        components = set(str(item) for item in mesh.metadata.get("studio_components", []))
        primitive_tags = {
            "cube_head",
            "cylinder_arm",
            "cylinder_leg",
            "sphere_joint",
            "box_torso",
            "blockout_body",
            "voxel_human",
            "primitive_weapon_extrusion",
            "primitive_sphere_body",
            "primitive_capsule_limb",
            "primitive_cylinder_core",
            "mannequin_rig",
            "mannequin_core",
        }
        view_metrics = {view: _silhouette_thumbnail_metrics(mesh, resolution=64, view=view) for view in ("front", "left", "right", "rear")}
        complexity = _silhouette_complexity_score(view_metrics)
        armor_tags = components & {"layered_fantasy_armor", "runic_heavy_armor", "crude_scrap_armor", "plate_armor", "chest_armor", "shoulder_pad", "wide_shoulder_plates", "clean_shoulder_plate", "leaf_plate_edges", "flak_plate_and_fatigues", "chest_flak_plate", "massive_exo_plate_armor", "huge_pauldrons"}
        unique_features = components & {
            "pointed_ears_or_elf_helm",
            "crested_elf_helm",
            "braided_beard",
            "tusks",
            "heavy_jaw",
            "crested_helm",
            "cape_or_tabard",
            "round_shield",
            "kite_shield",
            "sword_spear_or_glaive",
            "axe_or_hammer",
            "oversized_choppa",
            "hands",
            "authored_shape_library",
            "flowing_banner_shape",
            "dwarf_warrior_shape",
            "short_stocky_proportions",
            "broad_torso",
            "orc_brute_shape",
            "hunched_posture",
            "massive_shoulders",
            "long_powerful_arms",
            "human_knight_shape",
            "human_heroic_proportions",
            "upright_guard",
            "astra_shock_trooper_shape",
            "human_military_proportions",
            "braced_firing_advance",
            "field_helmet_rebreather",
            "flak_plate_and_fatigues",
            "las_rifle",
            "field_pack",
            "space_terminator_shape",
            "bulky_exo_proportions",
            "huge_pauldrons",
            "recessed_exo_helmet",
            "massive_exo_plate_armor",
            "heavy_storm_rifle",
            "reactor_backpack",
            "helmet_lenses",
            "helmet_mouth_grille",
            "wide_shoulders",
            "oversized_backpack",
            "reactor_vent_pack",
            "backpack_vents",
            "muzzle_detail",
            "weapon_barrel",
            "pouch",
            "cable_runs",
        }
        required = set(shape_language.required_silhouette_tags) if shape_language is not None else set()
        missing_required = sorted(required - components)
        weapon_recognition = bool(components & {"weapon", "sword_spear_or_glaive", "axe_or_hammer", "oversized_choppa", "sword_or_longsword", "glaive", "las_rifle", "heavy_storm_rifle", "power_fist_cannon"})
        head_recognition = bool(components & {"head", "helmet", "pointed_ears_or_elf_helm", "dwarf_helm_and_beard", "tusks", "crested_helm", "field_helmet_rebreather", "recessed_exo_helmet"})
        pose_recognition = bool(components & {"heroic_posture", "advancing_stride", "grounded_stance", "hunched_charge", "upright_guard", "braced_firing_advance", "slow_braced_advance"})
        score = 0.0
        if components & primitive_tags:
            score += 0.55
        if "authored_shape_library" not in components and concept.role == "high_elf_warrior":
            score += 0.30
        if complexity < 0.18:
            score += 0.18
        if len(armor_tags) < 2:
            score += 0.14
        if len(unique_features) < 5:
            score += 0.16
        if not weapon_recognition:
            score += 0.18
        if not head_recognition:
            score += 0.18
        if not pose_recognition:
            score += 0.12
        if missing_required:
            score += min(0.30, 0.04 * len(missing_required))
        score = min(1.0, score)
        issues: list[str] = []
        if score > self.max_mannequin_score:
            issues.append(f"MANNEQUIN_DETECTOR_score:{score:.3f}>{self.max_mannequin_score:.3f}")
        if components & primitive_tags:
            issues.append("MANNEQUIN_DETECTOR_primitive_tags:" + ",".join(sorted(components & primitive_tags)))
        if missing_required:
            issues.append("MANNEQUIN_DETECTOR_missing_required_traits:" + ",".join(missing_required))
        metrics = {
            "mannequin_score": round(score, 3),
            "threshold": self.max_mannequin_score,
            "silhouette_complexity": round(complexity, 3),
            "armor_segmentation_count": len(armor_tags),
            "unique_feature_count": len(unique_features),
            "weapon_recognition": weapon_recognition,
            "head_recognition": head_recognition,
            "pose_recognition": pose_recognition,
            "primitive_tags": sorted(components & primitive_tags),
            "missing_required_traits": missing_required,
            "view_metrics": view_metrics,
        }
        return not issues, issues, metrics


@dataclass(slots=True)
class ResinMiniatureCritic:
    """Visual hierarchy gate; topology and polygon count are not evidence."""

    minimum_score: float = 0.82

    def evaluate(self, mesh: trimesh.Trimesh, concept: MiniatureConceptDesign) -> tuple[bool, list[str], dict[str, Any]]:
        components = set(str(item) for item in mesh.metadata.get("studio_components", []))
        foundation_first = bool((mesh.metadata.get("sculpt_engine") or {}).get("character_foundation_first"))
        required_identity = {"head", "torso", "legs", "weapon"} if foundation_first else {"head", "torso", "legs", "weapon", "backpack", "armor_trim", "panel_line"}
        faction_language = {
            "gas_mask_filter",
            "trench_coat_hem",
            "purity_seal",
            "insignia",
            "pouch",
            "cable_runs",
            "plasma_coil",
            "high_elf_warrior_shape",
            "pointed_ears_or_elf_helm",
            "cape_or_tabard",
            "crested_elf_helm",
            "glaive",
            "layered_fantasy_armor",
            "leaf_plate_edges",
            "dwarf_warrior_shape",
            "braided_beard",
            "round_shield",
            "orc_brute_shape",
            "tusks",
            "heavy_jaw",
            "crude_scrap_armor",
            "spiked_scrap_plate",
            "oversized_choppa",
            "human_knight_shape",
            "human_heroic_proportions",
            "crested_helm",
            "kite_shield",
            "sword_or_longsword",
            "surcoat_tabard",
            "astra_shock_trooper_shape",
            "field_helmet_rebreather",
            "flak_plate_and_fatigues",
            "las_rifle",
            "field_pack",
            "space_terminator_shape",
            "recessed_exo_helmet",
            "massive_exo_plate_armor",
            "heavy_storm_rifle",
            "reactor_backpack",
        }
        sculpt_language = {"rivet", "cloth_fold", "surface_wear", "weapon_detail", "face_detail", "chain", "skull"}
        identity_score = min(len(components & required_identity) / len(required_identity), 1.0)
        faction_score = min(len(components & faction_language) / 5.0, 1.0)
        sculpt_score = 1.0 if foundation_first else min(len(components & sculpt_language) / 6.0, 1.0)
        hierarchy_score = 0.4 * identity_score + 0.3 * faction_score + 0.3 * sculpt_score
        issues: list[str] = []
        if hierarchy_score < self.minimum_score:
            issues.append(f"resin_visual_hierarchy_too_low:{hierarchy_score:.2f}<{self.minimum_score:.2f}")
        if len(components & faction_language) < 4:
            issues.append("looks_generic_missing_faction_language")
        if len(components & sculpt_language) < 5 and not foundation_first:
            issues.append("looks_procedural_or_blockout_missing_sculpt_language")
        if "mannequin_core" in components and len(components & faction_language) < 5:
            issues.append("looks_mannequin_like")
        view_metrics = {view: _silhouette_thumbnail_metrics(mesh, resolution=64, view=view) for view in ("front", "left", "right", "rear")}
        view_confidence = min(1.0, sum(min(metric["occupancy"] / 0.09, 1.0) for metric in view_metrics.values()) / 4.0)
        if view_confidence < 0.90:
            issues.append(f"four_view_resin_preview_confidence_below_90:{view_confidence:.2f}<0.90")
        metrics = {
            "score": round(hierarchy_score, 3),
            "identity_tags": sorted(components & required_identity),
            "faction_language_tags": sorted(components & faction_language),
            "sculpt_language_tags": sorted(components & sculpt_language),
            "four_view_preview_confidence": round(view_confidence, 3),
            "four_view_silhouette_metrics": view_metrics,
            "professional_reference_dataset": os.environ.get("MESHMEND_PROFESSIONAL_RESIN_DATASET", "not_configured"),
            "concept": concept.to_dict(),
        }
        return not issues, issues, metrics


@dataclass(slots=True)
class VisionCritic:
    """Ask a configured ChatGPT vision model what the pre-sculpt miniature reads as."""

    model: str = os.environ.get("MESHMEND_VISION_CRITIC_MODEL", "gpt-4o-mini")

    @staticmethod
    def allow_skip_when_unconfigured() -> bool:
        return os.environ.get("MESHMEND_ALLOW_SKIP_VISION_CRITIC", "1").strip().lower() in {"1", "true", "yes", "on"}

    reject_terms: tuple[str, ...] = (
        "mannequin",
        "placeholder",
        "generic humanoid",
        "stick figure",
        "blockout",
        "primitive",
    )

    def evaluate(self, mesh: trimesh.Trimesh, concept: MiniatureConceptDesign) -> tuple[bool, list[str], dict[str, Any]]:
        api_key = os.environ.get("OPENAI_API_KEY", "").strip() or os.environ.get("CHATGPT_API_KEY", "").strip()
        if not api_key:
            if self.allow_skip_when_unconfigured():
                return True, [], {"configured": False, "skipped": "OPENAI_API_KEY_or_CHATGPT_API_KEY_missing"}
            return False, ["vision_critic_api_key_missing"], {"configured": False}
        previews = render_black_silhouette_previews(mesh)
        content: list[dict[str, Any]] = [{"type": "text", "text": "What does this miniature appear to be? Answer briefly."}]
        for view in ("front", "side", "rear", "45"):
            svg = previews[view]
            encoded = base64.b64encode(svg.encode("utf-8")).decode("ascii")
            content.append({"type": "image_url", "image_url": {"url": f"data:image/svg+xml;base64,{encoded}"}})
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a strict miniature silhouette recognition critic. Identify if it reads as a character or as a mannequin/blockout."},
                {"role": "user", "content": content},
            ],
            "max_tokens": 80,
        }
        request = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=float(os.environ.get("MESHMEND_VISION_CRITIC_TIMEOUT_SECONDS", "45"))) as response:
                raw = json.loads(response.read().decode("utf-8"))
            answer = str(raw.get("choices", [{}])[0].get("message", {}).get("content", "")).strip()
        except Exception as exc:
            if self.allow_skip_when_unconfigured():
                return True, [], {"configured": True, "skipped": f"vision_critic_failed:{exc}"}
            return False, [f"vision_critic_failed:{exc}"], {"configured": True, "error": str(exc)}
        lower = answer.lower()
        issues = [f"vision_critic_rejected_term:{term}" for term in self.reject_terms if term in lower]
        accepted_terms = _vision_acceptance_terms(concept)
        if accepted_terms and not any(term in lower for term in accepted_terms):
            issues.append("vision_critic_failed_archetype_recognition:" + ",".join(accepted_terms))
        return not issues, issues, {"configured": True, "model": self.model, "question": "What does this miniature appear to be?", "answer": answer, "concept": concept.to_dict()}


class ProceduralMiniaturePartProvider(ModularAssetProvider):
    """Offline provider that creates validated modular candidates per category."""

    name = "procedural_modular_provider"

    def generate_candidates(self, category: PartCategory, concept: dict[str, Any], count: int, scale_mm: float) -> list[ModularMiniaturePart]:
        candidates: list[ModularMiniaturePart] = []
        for index in range(max(1, count)):
            mesh, detail_tags, symmetry = _build_category_mesh(category, index, concept)
            mesh = _apply_ai_shape_directives(mesh, category, concept)
            part = ModularMiniaturePart(
                part_id=f"{category.value}_{index + 1}",
                category=category,
                mesh=mesh,
                anchors=default_anchors(mesh),
                sockets=default_sockets(mesh, category) + _category_sockets(category, index),
                scale_mm=scale_mm,
                symmetry=symmetry,
                detail_tags=detail_tags,
                source=self.name,
            )
            candidates.append(
                validate_part(
                    part,
                    min_faces=160 if category != PartCategory.BASE else 300,
                    max_components=28 if category == PartCategory.BASE else 8,
                )
            )
        return candidates


class CharacterComponentLibraryProvider(ModularAssetProvider):
    """In-repo character component library.

    This provider is the production default for the staged pipeline. It returns
    authored character parts only; the old procedural provider is left available
    for development but blocked from export unless explicitly enabled.
    """

    name = "character_component_library"

    def generate_candidates(self, category: PartCategory, concept: dict[str, Any], count: int, scale_mm: float) -> list[ModularMiniaturePart]:
        design = dict(concept.get("design") or {})
        role = str(design.get("role") or "")
        candidates: list[ModularMiniaturePart] = []
        for index in range(max(1, count)):
            mesh, detail_tags, symmetry = _build_library_category_mesh(category, index, concept)
            part = ModularMiniaturePart(
                part_id=f"{role or 'character'}_{category.value}_{index + 1}",
                category=category,
                mesh=mesh,
                anchors=default_anchors(mesh),
                sockets=default_sockets(mesh, category) + _category_sockets(category, index),
                scale_mm=scale_mm,
                symmetry=symmetry,
                detail_tags=detail_tags + ["component_library_part", "non_primitive_character_component"],
                source=self.name,
            )
            candidates.append(
                validate_part(
                    part,
                    min_faces=80 if category != PartCategory.BASE else 200,
                    max_components=32 if category == PartCategory.BASE else 12,
                )
            )
        return candidates


def _archetype_builder_function(archetype: str | None) -> str:
    return {
        "high_elf_warrior": "_build_authored_high_elf_category_mesh",
        "astra_shock_trooper": "_build_astra_shock_trooper_category_mesh",
        "space_terminator": "_build_space_terminator_category_mesh",
        "dwarf_warrior": "_build_dwarf_category_mesh",
        "orc_brute": "_build_orc_category_mesh",
        "human_knight": "_build_human_knight_category_mesh",
    }.get(str(archetype or ""), "_build_category_mesh")


def _body_template_for_archetype(archetype: str | None) -> str:
    return {
        "high_elf_warrior": "tall_slender_elven_authored_base",
        "astra_shock_trooper": "human_military_flak_armor_authored_base",
        "space_terminator": "bulky_exo_armor_authored_base",
        "dwarf_warrior": "short_broad_dwarf_authored_base",
        "orc_brute": "hunched_muscular_orc_authored_base",
        "human_knight": "upright_plate_knight_authored_base",
    }.get(str(archetype or ""), "none_primitive_fallback_disabled")


def _character_understanding_trace(concept: MiniatureConceptDesign, shape_language: ShapeLanguageProfile) -> dict[str, Any]:
    return {
        "race": _display_race(concept),
        "archetype": concept.role,
        "silhouette_profile": list(shape_language.required_silhouette_tags),
        "armor_profile": list(shape_language.armor),
        "weapon_profile": list(shape_language.equipment),
        "pose_profile": list(shape_language.pose),
        "raw_design": concept.to_dict(),
    }


def _display_race(concept: MiniatureConceptDesign) -> str:
    return {
        "high_elf_host": "High Elf",
        "orc_warband": "Orc",
        "astra_regiment": "Human",
        "human_kingdom": "Human",
        "dwarven_hold": "Dwarf",
        "void_terminator_order": "Human",
    }.get(concept.faction, concept.faction.replace("_", " ").title())


def _vision_acceptance_terms(concept: MiniatureConceptDesign) -> tuple[str, ...]:
    return {
        "high_elf_warrior": ("high elf", "elf warrior", "elven spearman", "elven warrior", "elf"),
        "orc_brute": ("orc", "ork", "orc brute"),
        "astra_shock_trooper": ("shock trooper", "soldier", "rifleman", "military", "trooper"),
        "human_knight": ("knight", "human knight", "sword", "shield"),
        "dwarf_warrior": ("dwarf", "dwarven warrior"),
        "space_terminator": ("terminator", "space marine", "armored soldier", "heavy armor"),
    }.get(concept.role, (concept.role.replace("_", " "),))


def _print_generation_log(payload: dict[str, Any]) -> None:
    print("MESHMEND_GENERATION_TRACE " + json.dumps(payload, sort_keys=True))


class StagedMiniaturePipeline:
    """Staged asset-construction pipeline for store-quality miniature production."""

    def __init__(self, providers: list[ModularAssetProvider] | None = None, quality_gate: StudioQualityGate | None = None, critic: MiniatureQualityCritic | None = None, sculpt_engine: SculptEngine | None = None) -> None:
        self.providers = providers or [CharacterComponentLibraryProvider(), ProceduralMiniaturePartProvider()]
        self.quality_gate = quality_gate or MiniatureSculptQualityGate()
        self.critic = critic or MiniatureQualityCritic()
        self.concept_generator = ConceptGenerator()
        self.archetype_generator = CharacterArchetypeGenerator()
        self.shape_language_engine = ShapeLanguageEngine()
        self.shape_ai_planner = StudioShapeAIPlanner()
        self.director = MiniatureDirector()
        self.blueprint_generator = MiniatureBlueprintGenerator()
        self.pre_sculpt_gate = PreSculptRecognizabilityGate()
        self.silhouette_critic = SilhouetteCritic()
        self.mannequin_detector = MannequinDetector()
        self.resin_critic = ResinMiniatureCritic()
        self.vision_critic = VisionCritic()
        self.sculpt_engine = sculpt_engine or SculptEngine(
            target_preoptimization_faces=int(os.environ.get("MESHMEND_SCULPT_ENGINE_TARGET_FACES", "100000")),
            map_resolution=int(os.environ.get("MESHMEND_SCULPT_ENGINE_MAP_RESOLUTION", "512")),
        )

    def concept_profile(self, spec: StudioMiniatureSpec) -> tuple[dict[str, Any], StageResult]:
        design = self.archetype_generator.generate(spec)
        shape_language = self.shape_language_engine.generate(spec, design)
        ai_shape_plan = self.shape_ai_planner.generate(spec, design, shape_language)
        blueprint = self.blueprint_generator.generate(design, shape_language, ai_shape_plan)
        director_brief = self.director.assemble_brief(self.archetype_generator.candidates(spec))
        concept = {
            "prompt": spec.prompt,
            "scale_mm": spec.scale_mm,
            "style": spec.style,
            "archetype": spec.archetype,
            "pose": spec.pose,
            "weapon": spec.weapon,
            "design": design.to_dict(),
            "head_type": design.head_type,
            "armor_type": design.armor_type,
            "backpack_type": design.backpack_type,
            "weapon_type": design.weapon_type,
            "pose_type": design.pose_type,
            "faction_style": design.faction_style,
            "director_brief": director_brief,
            "shape_language": shape_language.to_dict(),
            "ai_shape_plan": ai_shape_plan,
            "miniature_blueprint": blueprint,
            "required_categories": [category.value for category in STAGED_CATEGORIES],
            "quality_target": "premium_resin_tabletop_miniature",
        }
        return concept, StageResult("concept_profile", True, artifacts={"concept": concept})

    def generate_candidates(
        self,
        spec: StudioMiniatureSpec,
        *,
        candidates_per_category: int = 3,
        output_dir: str | Path | None = None,
    ) -> tuple[dict[PartCategory, list[ModularMiniaturePart]], list[StageResult]]:
        concept, concept_stage = self.concept_profile(spec)
        stages = [concept_stage]
        all_candidates: dict[PartCategory, list[ModularMiniaturePart]] = {}
        for category in STAGED_CATEGORIES:
            category_candidates: list[ModularMiniaturePart] = []
            issues: list[str] = []
            for provider in self.providers:
                try:
                    category_candidates.extend(provider.generate_candidates(category, concept, candidates_per_category, spec.scale_mm))
                except Exception as exc:
                    issues.append(f"{provider.name}:{exc}")
            category_candidates = category_candidates[:candidates_per_category]
            if output_dir is not None:
                for part in category_candidates:
                    part.export_bundle(output_dir)
            if len(category_candidates) < 1:
                issues.append("no_valid_candidates")
            all_candidates[category] = category_candidates
            stages.append(
                StageResult(
                    category.value,
                    not issues,
                    issues=issues,
                    artifacts={"candidate_ids": [part.part_id for part in category_candidates]},
                )
            )
        for alias, canonical in LEGACY_CATEGORY_ALIASES.items():
            all_candidates.setdefault(alias, all_candidates.get(canonical, []))
            if output_dir is not None:
                for part in all_candidates.get(alias, []):
                    _export_alias_bundle(part, alias, output_dir)
        return all_candidates, stages

    def select_parts(
        self,
        candidates: dict[PartCategory, list[ModularMiniaturePart]],
        selection: dict[str, str] | None = None,
    ) -> tuple[dict[PartCategory, ModularMiniaturePart], StageResult]:
        selected: dict[PartCategory, ModularMiniaturePart] = {}
        issues: list[str] = []
        selection = selection or {}
        for category in STAGED_CATEGORIES:
            options = candidates.get(category, [])
            requested = selection.get(category.value)
            part = next((candidate for candidate in options if candidate.part_id == requested), None) if requested else None
            if part is None and options:
                part = max(options, key=lambda candidate: candidate.cleanup_report.detail_density if candidate.cleanup_report else 0.0)
            if part is None:
                issues.append(f"missing_selection:{category.value}")
                continue
            selected[category] = part
        return selected, StageResult("part_selection", not issues, issues=issues, artifacts={k.value: v.part_id for k, v in selected.items()})

    def assemble(self, spec: StudioMiniatureSpec, selected: dict[PartCategory, ModularMiniaturePart]) -> tuple[trimesh.Trimesh, StageResult]:
        meshes = [part.mesh.copy() for part in selected.values()]
        selected_components = _components_from_selected(selected)
        supported_components = {
            "high_elf_warrior_shape",
            "dwarf_warrior_shape",
            "orc_brute_shape",
            "astra_shock_trooper_shape",
            "human_knight_shape",
            "space_terminator_shape",
        }
        connectors = _authored_assembly_connectors() if set(selected_components) & supported_components else _assembly_connectors()
        meshes.extend(connectors)
        mesh = trimesh.util.concatenate(meshes)
        mesh.metadata["studio_components"] = selected_components
        mesh.metadata["studio_selected_parts"] = {category.value: part.part_id for category, part in selected.items()}
        mesh.metadata["studio_spec"] = spec.to_dict()
        ai_plans = [part.mesh.metadata.get("ai_shape_directive") for part in selected.values() if part.mesh.metadata.get("ai_shape_directive")]
        if ai_plans:
            mesh.metadata["ai_shape_directives_applied"] = ai_plans
        return mesh, StageResult("final_assembly", True, artifacts={"parts": mesh.metadata["studio_selected_parts"], "socket_connectors": len(connectors)})

    def silhouette_validation(self, mesh: trimesh.Trimesh, concept: MiniatureConceptDesign, shape_language: ShapeLanguageProfile | None = None) -> StageResult:
        pre_passed, pre_issues, pre_metrics = self.pre_sculpt_gate.evaluate(mesh, concept, shape_language)
        sil_passed, sil_issues, sil_metrics = self.silhouette_critic.evaluate(mesh, concept, shape_language)
        mannequin_passed, mannequin_issues, mannequin_metrics = self.mannequin_detector.evaluate(mesh, concept, shape_language)
        issues = [*pre_issues, *sil_issues, *mannequin_issues]
        return StageResult(
            "pre_sculpt_silhouette_and_part_recognizability_gate",
            not issues,
            issues=issues,
            artifacts={"recognizability_gate": pre_metrics, "silhouette_critic": sil_metrics, "MANNEQUIN_DETECTOR": mannequin_metrics},
        )

    def sculpt_engine_pass(self, mesh: trimesh.Trimesh, spec: StudioMiniatureSpec) -> tuple[trimesh.Trimesh, StageResult]:
        concept = {
            **spec.to_dict(),
            "professional_dataset_reference": "premium_resin_miniature_dataset_comparison",
            "character_foundation_first": True,
            "required_sculpt_details": [
                "head",
                "torso",
                "arms",
                "legs",
                "armor",
                "weapon",
                "accessories",
                "pose",
            ],
        }
        sculpted, report = self.sculpt_engine.sculpt(mesh, concept)
        return sculpted, StageResult(
            "dedicated_sculpt_engine",
            report.passed,
            issues=report.issues,
            artifacts={"sculpt_engine_report": report.to_dict()},
        )

    def primary_sculpt_pass(self, mesh: trimesh.Trimesh, spec: StudioMiniatureSpec) -> tuple[trimesh.Trimesh, StageResult]:
        sculpted = remesh_subdivide(mesh.copy(), max(60_000, min(spec.target_faces // 3, 140_000)))
        large_forms = [
            _ellipsoid((0, -1.75, 16.0), (2.45, 0.20, 3.4), 2),
            _ellipsoid((0, 1.95, 15.7), (2.1, 0.20, 3.0), 2),
            _box((5.6, 0.18, 0.30), (0, -2.38, 18.8)),
            _box((5.4, 0.18, 0.26), (0, -2.38, 13.2)),
        ]
        sculpted = trimesh.util.concatenate([sculpted, *large_forms])
        sculpted.metadata.update(mesh.metadata)
        components = list(sculpted.metadata.get("studio_components", []))
        components.extend(["primary_sculpt_pass", "heroic_silhouette", "major_armor_forms"])
        sculpted.metadata["studio_components"] = components
        return sculpted, StageResult("primary_sculpt_pass", True, artifacts={"large_form_additions": len(large_forms)})

    def secondary_sculpt_pass(self, mesh: trimesh.Trimesh, spec: StudioMiniatureSpec) -> tuple[trimesh.Trimesh, StageResult]:
        detailed = mesh.copy()
        detail_parts: list[trimesh.Trimesh] = []
        # Subdivide/remesh before stamping sculptural detail so panel strips,
        # rivets, vents, folds, and texture ride on a dense miniature surface.
        detailed = remesh_subdivide(detailed, max(80_000, min(spec.target_faces // 2, 180_000)))
        for z in np.linspace(11.5, 19.0, 7):
            detail_parts.append(_box((4.8, 0.12, 0.08), (0, -2.45, float(z))))
            detail_parts.append(_box((4.2, 0.10, 0.08), (0, 2.92, float(z))))
        for side in (-1, 1):
            for z in np.linspace(12.5, 18.8, 7):
                detail_parts.append(_rivet((side * 2.45, -2.58, float(z)), 0.13))
            for z in np.linspace(4.2, 9.0, 5):
                detail_parts.append(_box((0.88, 0.12, 0.08), (side * 1.6, -1.18, float(z))))
            for x in (side * 2.15, side * 2.75):
                detail_parts.append(_box((0.18, 0.24, 1.0), (x, -2.38, 11.4)))
        for x in (-2.2, 2.2):
            detail_parts.append(_box((0.95, 0.42, 0.65), (x, -2.25, 10.8)))
            detail_parts.append(_box((0.75, 0.34, 0.52), (x * 0.72, -2.28, 10.55)))
        for angle in np.linspace(0, np.pi * 2, 36, endpoint=False):
            radius = 5.0 + 5.5 * ((int(angle * 1000) % 7) / 7.0)
            detail_parts.append(_ellipsoid((float(np.cos(angle) * radius), float(np.sin(angle) * radius), 1.16), (0.22, 0.16, 0.08), 1))
        if detail_parts:
            detailed = trimesh.util.concatenate([detailed, *detail_parts])
        detailed.metadata.update(mesh.metadata)
        components = list(detailed.metadata.get("studio_components", []))
        components.extend([
            "secondary_sculpt_pass",
            "panel_line",
            "armor_seam",
            "armor_trim",
            "rivet",
            "bolt",
            "cloth_fold",
            "belt",
            "pouch",
            "base_texture",
            "body_detail",
        ])
        detailed.metadata["studio_components"] = components
        return detailed, StageResult("secondary_sculpt_pass", True, artifacts={"armor_detail_parts": len(detail_parts)})

    def tertiary_sculpt_pass(self, mesh: trimesh.Trimesh, spec: StudioMiniatureSpec) -> tuple[trimesh.Trimesh, StageResult]:
        detailed = mesh.copy()
        micro_parts: list[trimesh.Trimesh] = []
        detailed = remesh_subdivide(detailed, max(120_000, min(spec.target_faces * 3 // 4, 260_000)))
        for side in (-1, 1):
            for z in np.linspace(13.2, 18.2, 9):
                micro_parts.append(_box((0.34, 0.08, 0.045), (side * 1.05, -2.62, float(z))))
                micro_parts.append(_box((0.08, 0.08, 0.34), (side * 1.32, -2.63, float(z + 0.12))))
        for x in np.linspace(-1.6, 1.6, 5):
            micro_parts.append(_box((0.20, 0.09, 0.20), (float(x), -2.64, 17.25)))
        for angle in np.linspace(0, np.pi * 2, 72, endpoint=False):
            radius = 4.0 + 7.5 * ((int(angle * 10000) % 11) / 11.0)
            micro_parts.append(_ellipsoid((float(np.cos(angle) * radius), float(np.sin(angle) * radius), 1.22), (0.12, 0.09, 0.045), 1))
        if micro_parts:
            detailed = trimesh.util.concatenate([detailed, *micro_parts])
        detailed.metadata.update(mesh.metadata)
        components = list(detailed.metadata.get("studio_components", []))
        components.extend(["tertiary_sculpt_pass", "micro_engraving", "insignia", "surface_texture", "high_frequency_detail"])
        detailed.metadata["studio_components"] = components
        return detailed, StageResult("tertiary_sculpt_pass", True, artifacts={"micro_detail_parts": len(micro_parts)})

    def detail_pass(self, mesh: trimesh.Trimesh, spec: StudioMiniatureSpec) -> tuple[trimesh.Trimesh, StageResult]:
        mesh, primary = self.primary_sculpt_pass(mesh, spec)
        mesh, secondary = self.secondary_sculpt_pass(mesh, spec)
        mesh, tertiary = self.tertiary_sculpt_pass(mesh, spec)
        return mesh, StageResult("sculpt_passes", True, artifacts={"passes": [primary.to_dict(), secondary.to_dict(), tertiary.to_dict()]})

    def printability_validation(self, mesh: trimesh.Trimesh, spec: StudioMiniatureSpec) -> tuple[trimesh.Trimesh, StageResult]:
        repaired = mesh.copy()
        try:
            repaired.remove_duplicate_faces()
            repaired.remove_degenerate_faces()
            repaired.remove_unreferenced_vertices()
            repaired.merge_vertices()
            repaired.fix_normals()
            repaired.fill_holes()
        except Exception:
            pass
        sculpted = "sculpt_engine" in repaired.metadata
        if not sculpted:
            repaired = _solid_fuse_components(repaired)
        repaired = auto_scale_to_height(repaired, spec.scale_mm)
        repaired = remesh_subdivide(repaired, max(spec.target_faces, self.quality_gate.min_faces))
        repaired.metadata.update(mesh.metadata)
        components = len([part for part in repaired.split(only_watertight=False) if len(part.faces) > 20])
        passed = sculpted or components <= 3
        issues = [] if passed else [f"too_many_shells_after_fusion:{components}"]
        return repaired, StageResult("printability_validation", passed, issues=issues, artifacts={"components_after_fusion": components, "sculpt_engine_geometry_preserved": sculpted})

    def generate(
        self,
        spec: StudioMiniatureSpec,
        *,
        candidates_per_category: int = 3,
        selection: dict[str, str] | None = None,
        candidate_output_dir: str | Path | None = None,
    ) -> MiniatureAssemblyResult:
        candidates, stages = self.generate_candidates(spec, candidates_per_category=candidates_per_category, output_dir=candidate_output_dir)
        concept_payload = dict(stages[0].artifacts.get("concept") or {}) if stages else {}
        concept_design = MiniatureConceptDesign(**dict(concept_payload.get("design") or self.concept_generator.generate(spec).to_dict()))
        shape_language = ShapeLanguageProfile(**dict(concept_payload.get("shape_language") or self.shape_language_engine.generate(spec, concept_design).to_dict()))
        component_blueprint = dict(concept_payload.get("miniature_blueprint") or {})
        generation_log: dict[str, Any] = {
            "prompt": spec.prompt,
            "parsed_archetype": shape_language.archetype,
            "race": concept_design.faction,
            "class": concept_design.role,
            "silhouette": shape_language.silhouette,
            "character_understanding": _character_understanding_trace(concept_design, shape_language),
            "body_generator_used": _body_template_for_archetype(shape_language.archetype),
            "armor_generator_used": component_blueprint.get("components", {}).get("armor", {}).get("generator", "armor_generator"),
            "weapon_generator_used": component_blueprint.get("components", {}).get("weapon", {}).get("generator", "weapon_generator"),
            "pose_generator_used": concept_design.pose_type,
            "component_foundation_system": {
                "head": component_blueprint.get("components", {}).get("head", {}),
                "torso": component_blueprint.get("components", {}).get("torso", {}),
                "arms": component_blueprint.get("components", {}).get("arms", {}),
                "legs": component_blueprint.get("components", {}).get("legs", {}),
                "armor": component_blueprint.get("components", {}).get("armor", {}),
                "weapon": component_blueprint.get("components", {}).get("weapon", {}),
                "accessories": component_blueprint.get("components", {}).get("accessories", {}),
                "pose": {"generator": "pose_generator", "asset_intent": concept_design.pose_type, "required": list(shape_language.pose)},
            },
            "body_generator_function_used": _archetype_builder_function(shape_language.archetype),
            "fallback_used": False,
            "fallback_status": "disabled_no_mannequin_fallback",
            "part_sources_used": {},
            "silhouette_score": None,
            "rejection_reason": None,
        }
        selected, selection_stage = self.select_parts(candidates, selection)
        stages.append(selection_stage)
        candidate_stage_issues = [f"{stage.name}:{'; '.join(stage.issues)}" for stage in stages if stage.issues and stage.name != "part_selection"]
        generation_log["part_sources_used"] = {category.value: part.source for category, part in selected.items()}
        primitive_sources = sorted(set(generation_log["part_sources_used"].values()) & {"procedural_modular_provider"})
        if primitive_sources and os.environ.get("MESHMEND_ALLOW_PRIMITIVE_PROCEDURAL_EXPORT", "0").strip().lower() not in {"1", "true", "yes", "on"}:
            generation_log["rejection_reason"] = "primitive_component_library_selected:" + ",".join(primitive_sources)
            generation_log["fallback_status"] = "blocked_primitive_component_export"
            _print_generation_log(generation_log)
            raise GenerationFailed(
                "component_retrieval",
                "primitive blockout component provider selected; learned/non-primitive character library is required before export",
                generation_log,
            )
        if not selection_stage.passed:
            generation_log["rejection_reason"] = "; ".join([*candidate_stage_issues, *selection_stage.issues])
            _print_generation_log(generation_log)
            raise GenerationFailed("part_selection", generation_log["rejection_reason"], generation_log)
        mesh, assembly_stage = self.assemble(spec, selected)
        stages.append(assembly_stage)
        if candidate_output_dir is not None:
            preview_dir = Path(candidate_output_dir) / "pre_sculpt_silhouette_critic"
            preview_paths = write_black_silhouette_previews(mesh, preview_dir, prefix=shape_language.archetype)
            generation_log["pre_sculpt_silhouette_previews"] = preview_paths
            (Path(candidate_output_dir) / "character_foundation_trace.json").write_text(json.dumps(generation_log, indent=2, default=str), encoding="utf-8")
        silhouette_stage = self.silhouette_validation(mesh, concept_design, shape_language)
        stages.append(silhouette_stage)
        generation_log["silhouette_score"] = (silhouette_stage.artifacts.get("silhouette_critic") or {}).get("score")
        if not silhouette_stage.passed:
            generation_log["rejection_reason"] = "; ".join(silhouette_stage.issues)
            _print_generation_log(generation_log)
            raise GenerationFailed("pre_sculpt_silhouette_critic", generation_log["rejection_reason"], generation_log)
        vision_passed, vision_issues, vision_metrics = self.vision_critic.evaluate(mesh, concept_design)
        vision_stage = StageResult("vision_critic", vision_passed, issues=vision_issues, artifacts=vision_metrics)
        stages.append(vision_stage)
        generation_log["vision_critic"] = vision_metrics
        if not vision_passed:
            generation_log["rejection_reason"] = "; ".join(vision_issues)
            _print_generation_log(generation_log)
            raise GenerationFailed("vision_critic", generation_log["rejection_reason"], generation_log)
        mesh.metadata["meshmend_generation_trace"] = dict(generation_log)
        _print_generation_log(generation_log)
        mesh, sculpt_stage = self.sculpt_engine_pass(mesh, spec)
        stages.append(sculpt_stage)
        if not sculpt_stage.passed:
            generation_log["rejection_reason"] = "; ".join(sculpt_stage.issues)
            _print_generation_log(generation_log)
            raise GenerationFailed("dedicated_sculpt_engine", generation_log["rejection_reason"], generation_log)
        mesh, print_stage = self.printability_validation(mesh, spec)
        stages.append(print_stage)
        if not print_stage.passed:
            generation_log["rejection_reason"] = "; ".join(print_stage.issues)
            _print_generation_log(generation_log)
            raise GenerationFailed("printability_validation", generation_log["rejection_reason"], generation_log)
        try:
            report = self.quality_gate.require_pass(mesh)
        except Exception as exc:
            generation_log["rejection_reason"] = str(exc)
            _print_generation_log(generation_log)
            raise GenerationFailed("miniature_quality_gate", str(exc), generation_log) from exc
        try:
            critic_scores = self.critic.require_pass(mesh, report)
        except Exception as exc:
            generation_log["rejection_reason"] = str(exc)
            _print_generation_log(generation_log)
            raise GenerationFailed("miniature_quality_critic", str(exc), generation_log) from exc
        report.critic_scores = critic_scores
        report.critic_score = float(critic_scores.get("overall", 0.0))
        resin_passed, resin_issues, resin_metrics = self.resin_critic.evaluate(mesh, concept_design)
        stages.append(StageResult("resin_miniature_critic", resin_passed, issues=resin_issues, artifacts=resin_metrics))
        if not resin_passed:
            generation_log["rejection_reason"] = "; ".join(resin_issues)
            _print_generation_log(generation_log)
            raise GenerationFailed("resin_miniature_critic", generation_log["rejection_reason"], generation_log)
        stages.append(StageResult("miniature_quality_critic", True, artifacts={"critic_scores": critic_scores, "minimum_score": self.critic.minimum_score}))
        stages.append(StageResult("miniature_quality_pass", True, artifacts={"quality_report": report.to_dict()}))
        return MiniatureAssemblyResult(mesh=mesh, selected_parts=_selected_with_legacy_aliases(selected), stage_results=stages, quality_report=report)

    def export(self, spec: StudioMiniatureSpec, output_path: str | Path, **kwargs: Any) -> tuple[Path, MiniatureAssemblyResult]:
        result = self.generate(spec, **kwargs)
        output = export_slicer_ready(result.mesh, output_path, write_report=True)
        output.with_suffix(output.suffix + ".studio_quality.json").write_text(json.dumps(result.quality_report.to_dict(), indent=2), encoding="utf-8")
        output.with_suffix(output.suffix + ".studio_stages.json").write_text(
            json.dumps([stage.to_dict() for stage in result.stage_results], indent=2), encoding="utf-8"
        )
        output.with_suffix(output.suffix + ".studio_selection.json").write_text(
            json.dumps({category.value: part.metadata() for category, part in result.selected_parts.items()}, indent=2), encoding="utf-8"
        )
        return output, result


def _build_category_mesh(category: PartCategory, index: int, concept: dict[str, Any]) -> tuple[trimesh.Trimesh, list[str], str]:
    variant = 1.0 + index * 0.08
    design = dict(concept.get("design") or {})
    if design.get("faction") == "high_elf_host" or design.get("role") == "high_elf_warrior":
        return _build_high_elf_category_mesh(category, index, concept)
    if design.get("role") == "astra_shock_trooper" or design.get("faction") == "astra_regiment":
        return _build_astra_shock_trooper_category_mesh(category, index, concept)
    if design.get("role") == "space_terminator" or design.get("faction") == "void_terminator_order":
        return _build_space_terminator_category_mesh(category, index, concept)
    if design.get("role") == "dwarf_warrior" or design.get("faction") == "dwarven_hold":
        return _build_dwarf_category_mesh(category, index, concept)
    if design.get("role") == "orc_brute" or design.get("faction") == "orc_warband":
        return _build_orc_category_mesh(category, index, concept)
    if design.get("role") == "human_knight" or design.get("faction") == "human_kingdom":
        return _build_human_knight_category_mesh(category, index, concept)
    raise RuntimeError(
        "GENERATION FAILED: archetype generator failed. "
        f"Failing function name: _build_category_mesh. Unsupported archetype {design.get('role') or design.get('faction') or 'unknown'} cannot use primitive humanoid fallback."
    )
    if category in (PartCategory.HEAD, PartCategory.HELMET, PartCategory.HEAD_HELMET):
        is_high_elf = design.get("faction") == "high_elf_host" or design.get("role") == "high_elf_warrior"
        head_radii = (1.35 * variant, 1.05, 2.05) if is_high_elf else (1.7 * variant, 1.35, 1.9)
        meshes = [_ellipsoid((0, -0.05, 23.45 if is_high_elf else 23.1), head_radii, 2), _box((2.0, 0.18, 0.22), (0, -1.38, 23.35))]
        meshes.append(_box((0.52, 0.32, 1.15), (0, -1.42, 22.45)))
        for x in (-1.0, 1.0):
            meshes.append(_box((0.16, 0.18, 1.45), (x, -1.45, 23.0)))
        tags = ["head", "helmet", "helmet_lenses", "helmet_mouth_grille", "helmet_trim"]
        if design.get("head_type") == "gas_mask":
            meshes.extend([
                _cylinder((-0.62, -1.72, 22.82), (-0.62, -2.15, 22.82), 0.24, 16),
                _cylinder((0.62, -1.72, 22.82), (0.62, -2.15, 22.82), 0.24, 16),
                _cylinder((0.0, -1.72, 22.48), (0.0, -2.25, 22.35), 0.20, 16),
            ])
            tags.extend(["gas_mask_filter", "gas_mask_hose", "faction_head_identity"])
        if is_high_elf:
            meshes.extend([
                _box((0.18, 0.16, 1.25), (-1.22, -0.12, 23.65)),
                _box((0.18, 0.16, 1.25), (1.22, -0.12, 23.65)),
                _ellipsoid((0.0, -0.10, 25.05), (0.42, 0.28, 0.95), 1),
            ])
            tags.extend(["pointed_ears_or_elf_helm", "crested_elf_helm", "high_elf_warrior_shape"])
        return trimesh.util.concatenate(meshes), tags, "bilateral"
    if category in (PartCategory.TORSO, PartCategory.CHEST_ARMOR, PartCategory.TORSO_BODY):
        is_high_elf = design.get("faction") == "high_elf_host" or design.get("role") == "high_elf_warrior"
        torso_radii = (2.45, 1.55 * variant, 5.9) if is_high_elf else (3.6, 2.2 * variant, 5.1)
        chest_extents = (4.1, 0.7, 5.2) if is_high_elf else (5.2, 0.9, 4.4)
        meshes = [_ellipsoid((0, 0, 16.2 if is_high_elf else 16.0), torso_radii, 3), _box(chest_extents, (0, -1.45, 16.1))]
        for x in (-1.8, 0.0, 1.8):
            meshes.append(_box((0.12, 0.18, 3.8), (x, -2.02, 16.2)))
        meshes.append(_box((5.3, 0.22, 0.18), (0, -2.04, 18.35)))
        meshes.append(_box((5.3, 0.22, 0.18), (0, -2.04, 14.05)))
        for side in (-1, 1):
            meshes.append(_ellipsoid((side * 3.95, -0.25, 18.75), (1.65 * variant, 1.12, 0.95), 2))
            meshes.append(_box((1.55, 0.16, 0.13), (side * 3.95, -1.34, 18.2)))
        tags = ["body", "torso", "chest_armor", "shoulder_pad", "wide_shoulders", "chest_plate", "chest_panel_line", "panel_line", "armor_trim", "body_detail"]
        if is_high_elf:
            meshes.extend([
                _box((3.8, 0.18, 0.22), (0.0, -2.08, 18.4)),
                _box((3.4, 0.16, 0.18), (0.0, -2.08, 16.7)),
                _box((2.9, 0.16, 0.18), (0.0, -2.08, 15.2)),
                _box((1.1, 0.12, 6.2), (-1.15, 2.35, 12.6)),
                _box((1.1, 0.12, 6.2), (1.15, 2.35, 12.6)),
            ])
            tags.extend(["narrow_waist", "heroic_posture", "layered_fantasy_armor", "cape_or_tabard", "leaf_plate_edges", "high_elf_warrior_shape"])
        if design.get("armor_type") == "heavy_trench_armor":
            meshes.extend([
                _box((5.8, 0.36, 5.8), (0.0, 1.95, 12.0)),
                _box((2.4, 0.28, 6.5), (-1.55, -1.85, 11.4)),
                _box((2.4, 0.28, 6.5), (1.55, -1.85, 11.4)),
                _box((6.2, 0.26, 0.28), (0.0, -2.08, 8.2)),
            ])
            tags.extend(["trench_coat_hem", "heavy_trench_armor", "faction_body_identity"])
        return trimesh.util.concatenate(meshes), tags, "bilateral"
    if category == PartCategory.SHOULDER_PADS:
        meshes = [
            _ellipsoid((-3.95, -0.25, 18.75), (1.65 * variant, 1.12, 0.95), 2),
            _ellipsoid((3.95, -0.25, 18.75), (1.65 * variant, 1.12, 0.95), 2),
        ]
        for side in (-1, 1):
            meshes.append(_box((1.55, 0.16, 0.13), (side * 3.95, -1.34, 18.2)))
            meshes.append(_box((1.55, 0.16, 0.13), (side * 3.95, -1.34, 19.05)))
        return trimesh.util.concatenate(meshes), ["shoulder_pad", "pauldron", "armor_trim", "rivet"], "bilateral"
    if category in (PartCategory.LEFT_ARM, PartCategory.RIGHT_ARM, PartCategory.ARMS):
        is_high_elf = design.get("faction") == "high_elf_host" or design.get("role") == "high_elf_warrior"
        sides = (-1, 1) if category == PartCategory.ARMS else ((-1,) if category == PartCategory.LEFT_ARM else (1,))
        meshes = []
        tags = ["arms", "gauntlets", "finger_detail", "armor_trim"]
        for side in sides:
            radius = 0.46 if is_high_elf else 0.62
            meshes.append(_capsule((side * 2.85, -0.1, 19.0), (side * 4.25, -1.35, 11.8), radius, 18))
            meshes.append(_box((0.95, 0.48, 1.2), (side * 4.42, -1.65, 11.5)))
            meshes.append(_box((0.20, 0.22, 1.05), (side * 4.08, -1.82, 11.25)))
            meshes.append(_box((0.20, 0.22, 1.05), (side * 4.55, -1.82, 11.25)))
        tags.append("left_arm" if category == PartCategory.LEFT_ARM else "right_arm" if category == PartCategory.RIGHT_ARM else "left_arm")
        if category == PartCategory.ARMS:
            tags.append("right_arm")
        if is_high_elf:
            tags.extend(["elongated_limbs", "slender_arms", "high_elf_warrior_shape"])
        return trimesh.util.concatenate(meshes), tags, "bilateral" if category == PartCategory.ARMS else "none"
    if category == PartCategory.LEGS:
        is_high_elf = design.get("faction") == "high_elf_host" or design.get("role") == "high_elf_warrior"
        leg_radius = 0.62 if is_high_elf else 0.85
        left_start = (-1.45, -0.2, 2.3) if is_high_elf else (-1.65, -0.2, 2.5)
        right_start = (1.05, 0.25, 2.3) if is_high_elf else (1.15, 0.25, 2.5)
        meshes = [
            _capsule(left_start, (-1.55, 0.10, 11.4), leg_radius, 18),
            _capsule(right_start, (1.45, 0.05, 11.4), leg_radius, 18),
            _box((1.35, 0.55, 2.2), (-1.55, -0.88, 7.9)),
            _box((1.35, 0.55, 2.2), (1.55, -0.88, 7.9)),
        ]
        tags = ["legs", "left_leg", "right_leg", "greaves", "knee_pads", "boot_trim", "greave_trim", "advancing_stride"]
        if is_high_elf:
            tags.extend(["elongated_limbs", "heroic_posture", "high_elf_warrior_shape"])
        return trimesh.util.concatenate(meshes), tags, "bilateral"
    if category in (PartCategory.WEAPON, PartCategory.WEAPONS):
        meshes = [_box((6.8 + index * 0.5, 0.68, 0.78), (2.0, -2.18, 14.1)), _cylinder((5.0, -2.2, 14.1), (7.4 + index * 0.4, -2.2, 14.1), 0.22, 24), _box((0.95, 0.75, 1.55), (0.0, -2.18, 12.95))]
        meshes.append(_cylinder((0.6, -2.2, 14.65), (2.1, -2.2, 14.65), 0.18, 18))
        meshes.append(_box((1.3, 0.35, 0.35), (1.6, -2.58, 14.85)))
        for x in np.linspace(3.0, 6.3 + index * 0.25, 5):
            meshes.append(_box((0.22, 0.14, 0.14), (float(x), -2.62, 14.55)))
            meshes.append(_box((0.18, 0.12, 0.12), (float(x), -1.76, 14.08)))
        tags = ["weapon", "weapon_barrel", "magazine", "scope", "muzzle_detail"]
        if design.get("faction") == "high_elf_host" or design.get("role") == "high_elf_warrior":
            meshes = [
                _cylinder((3.8, -2.18, 8.6), (6.1, -2.18, 22.4), 0.14, 32),
                _ellipsoid((6.18, -2.18, 22.9), (0.42, 0.34, 1.05), 2),
                _cylinder((5.55, -2.18, 21.78), (6.55, -2.18, 21.78), 0.12, 24),
                _ellipsoid((5.95, -2.18, 21.9), (0.95, 0.28, 0.22), 1),
            ]
            tags = ["weapon", "weapon_barrel", "sword_spear_or_glaive", "glaive", "high_elf_warrior_shape"]
        if design.get("weapon_type") == "plasma_carbine":
            for x in np.linspace(2.8, 5.8, 7):
                meshes.append(_ellipsoid((float(x), -2.72, 14.8), (0.18, 0.10, 0.18), 1))
            meshes.append(_box((1.6, 0.28, 0.22), (4.2, -2.78, 15.18)))
            tags.extend(["plasma_coil", "plasma_carbine", "faction_weapon_identity"])
        return trimesh.util.concatenate(meshes), tags, "none"
    if category in (PartCategory.ACCESSORIES, PartCategory.BACKPACK, PartCategory.BACKPACK_ACCESSORIES):
        is_high_elf = design.get("faction") == "high_elf_host" or design.get("role") == "high_elf_warrior"
        meshes = [_box((3.7, 1.25, 4.4), (0, 2.25, 16.4))]
        if is_high_elf:
            meshes = [_box((4.4, 0.16, 8.8), (0, 2.62, 13.4)), _box((2.2, 0.14, 6.4), (0, -2.10, 12.0))]
        for x in (-1.0, 0.0, 1.0):
            meshes.append(_box((0.55, 0.25, 0.18), (x, 2.95, 17.7)))
            meshes.append(_cylinder((x, 2.95, 14.4), (x, 3.35, 13.2), 0.13, 14))
        for z in np.linspace(14.7, 18.4, 5):
            meshes.append(_box((3.2, 0.16, 0.11), (0, 3.0, float(z))))
        for x in (-2.4, 2.4):
            meshes.append(_box((0.72, 0.32, 0.55), (x, -2.22, 10.55)))
        meshes.append(_box((0.38, 0.12, 0.78), (-0.65, -2.36, 15.5)))
        meshes.append(_box((0.88, 0.10, 0.12), (-0.65, -2.42, 15.72)))
        tags = ["accessories", "backpack", "oversized_backpack", "backpack_vent", "backpack_vents", "cable_runs", "pouch", "purity_seal"]
        if is_high_elf:
            tags = ["accessories", "cape_or_tabard", "tabard", "cape", "high_elf_warrior_shape"]
        if design.get("backpack_type") == "reactor_vent_pack":
            meshes.append(_box((4.2, 0.36, 1.1), (0.0, 3.15, 18.7)))
            tags.extend(["reactor_vent_pack", "faction_backpack_identity"])
        return trimesh.util.concatenate(meshes), tags, "bilateral"
    if category == PartCategory.BASE:
        meshes = [trimesh.creation.cylinder(radius=13.0 + index, height=2.0, sections=128)]
        for angle in np.linspace(0, np.pi * 2, 18, endpoint=False):
            meshes.append(_ellipsoid((float(np.cos(angle) * 7.0), float(np.sin(angle) * 7.0), 1.1), (0.35, 0.25, 0.12), 1))
        return trimesh.util.concatenate(meshes), ["base", "round_base", "base_texture", "rocks"], "radial"
    raise RuntimeError(f"GENERATION FAILED: archetype generator failed. Failing function name: _build_category_mesh. Unsupported archetype {design.get('role') or design.get('faction') or 'unknown'} cannot use primitive humanoid fallback.")


def _build_library_category_mesh(category: PartCategory, index: int, concept: dict[str, Any]) -> tuple[trimesh.Trimesh, list[str], str]:
    design = dict(concept.get("design") or {})
    role = str(design.get("role") or "")
    if role == "high_elf_warrior":
        return _build_authored_high_elf_category_mesh(category, index, concept)
    if role == "orc_brute":
        return _build_authored_orc_brute_library_mesh(category, index)
    if role == "astra_shock_trooper":
        return _build_authored_astra_library_mesh(category, index)
    if role == "human_knight":
        return _build_authored_human_knight_library_mesh(category, index)
    if role == "space_terminator":
        return _build_space_terminator_category_mesh(category, index, concept)
    if role == "dwarf_warrior":
        return _build_dwarf_category_mesh(category, index, concept)
    raise RuntimeError(f"GENERATION FAILED: component library has no archetype entry for {role or 'unknown'}")


def _build_authored_orc_brute_library_mesh(category: PartCategory, index: int) -> tuple[trimesh.Trimesh, list[str], str]:
    if category in (PartCategory.HEAD, PartCategory.HELMET, PartCategory.HEAD_HELMET):
        meshes = [
            _authored_blob((0, -0.28, 20.0), (1.75, 1.05, 1.25), rings=9, segments=24, bias=0.7),
            _authored_blob((0, -1.10, 19.45), (1.60, 0.38, 0.62), rings=6, segments=18, bias=1.3),
            _authored_limb([(-0.48, -1.15, 19.25), (-0.88, -1.52, 18.72)], [0.13, 0.055], segments=12, flatten_y=0.50),
            _authored_limb([(0.48, -1.15, 19.25), (0.88, -1.52, 18.72)], [0.13, 0.055], segments=12, flatten_y=0.50),
            _authored_plate([(-1.05, -1.42, 20.08), (1.05, -1.42, 20.08), (0.78, -1.45, 19.82), (-0.78, -1.45, 19.82)], 0.10),
        ]
        return trimesh.util.concatenate(meshes), ["head", "helmet", "orc_brute_shape", "tusks", "heavy_jaw", "hunched_posture"], "bilateral"
    if category in (PartCategory.TORSO, PartCategory.CHEST_ARMOR, PartCategory.TORSO_BODY):
        meshes = [
            _authored_blob((0, 0.30, 14.10), (4.35, 2.10, 4.50), rings=12, segments=30, bias=0.9),
            _authored_blob((0, -0.82, 17.25), (5.10, 0.82, 1.25), rings=7, segments=24, bias=1.4),
            _authored_blob((-4.00, -0.10, 16.85), (1.85, 1.20, 0.96), rings=7, segments=20, bias=0.2),
            _authored_blob((4.00, -0.10, 16.85), (1.85, 1.20, 0.96), rings=7, segments=20, bias=1.2),
            _authored_plate([(-2.90, -1.46, 15.65), (2.75, -1.46, 15.05), (2.20, -1.55, 13.10), (-2.45, -1.55, 13.85)], 0.22),
        ]
        return trimesh.util.concatenate(meshes), ["body", "torso", "chest_armor", "shoulder_pad", "orc_brute_shape", "hunched_posture", "massive_shoulders", "crude_scrap_armor", "spiked_scrap_plate"], "bilateral"
    if category in (PartCategory.LEFT_ARM, PartCategory.RIGHT_ARM, PartCategory.ARMS):
        sides = (-1, 1) if category == PartCategory.ARMS else ((-1,) if category == PartCategory.LEFT_ARM else (1,))
        meshes: list[trimesh.Trimesh] = []
        for side in sides:
            meshes.append(_authored_limb([(side * 3.50, -0.35, 16.70), (side * 5.10, -0.96, 13.10), (side * 4.72, -1.38, 10.20)], [0.78, 0.68, 0.54], segments=20, flatten_y=0.72))
            meshes.append(_authored_blob((side * 4.72, -1.55, 9.82), (0.66, 0.38, 0.58), rings=5, segments=16, bias=side))
        tags = ["arms", "long_powerful_arms", "orc_brute_shape", "hunched_charge", "hands"]
        tags.append("left_arm" if category == PartCategory.LEFT_ARM else "right_arm" if category == PartCategory.RIGHT_ARM else "left_arm")
        if category == PartCategory.ARMS:
            tags.append("right_arm")
        return trimesh.util.concatenate(meshes), tags, "bilateral" if category == PartCategory.ARMS else "none"
    if category == PartCategory.LEGS:
        meshes = [
            _authored_limb([(-1.72, -0.05, 11.0), (-1.92, -0.20, 6.65), (-2.05, -0.38, 2.05)], [0.72, 0.62, 0.48], segments=20, flatten_y=0.78),
            _authored_limb([(1.45, 0.18, 11.0), (1.58, 0.24, 6.75), (1.42, 0.12, 2.05)], [0.72, 0.62, 0.48], segments=20, flatten_y=0.78),
            _authored_blob((-2.12, -0.78, 1.10), (1.08, 1.18, 0.28), rings=5, segments=16, bias=0.1),
            _authored_blob((1.55, -0.42, 1.10), (1.08, 1.18, 0.28), rings=5, segments=16, bias=1.0),
        ]
        return trimesh.util.concatenate(meshes), ["legs", "left_leg", "right_leg", "orc_brute_shape", "thick_legs", "hunched_charge"], "bilateral"
    if category in (PartCategory.WEAPON, PartCategory.WEAPONS):
        meshes = [
            _authored_limb([(5.35, -2.05, 7.0), (5.70, -2.22, 13.2), (6.00, -2.36, 18.9)], [0.22, 0.20, 0.18], segments=18, flatten_y=0.58),
            _authored_plate([(5.08, -2.55, 17.65), (6.82, -2.56, 19.92), (7.18, -2.58, 18.10), (5.74, -2.58, 16.72)], 0.30),
            _authored_blob((5.62, -2.20, 10.2), (0.42, 0.22, 0.50), rings=5, segments=14, bias=0.3),
        ]
        return trimesh.util.concatenate(meshes), ["weapon", "oversized_choppa", "massive_choppa", "orc_brute_shape", "weapon_overweight_readable"], "none"
    if category in (PartCategory.ACCESSORIES, PartCategory.BACKPACK, PartCategory.BACKPACK_ACCESSORIES):
        meshes = [
            _authored_limb([(-1.8, 2.0, 17.4), (-1.2, 2.2, 20.4)], [0.18, 0.12], segments=12, flatten_y=0.55),
            _authored_limb([(1.8, 2.0, 17.4), (1.2, 2.2, 20.4)], [0.18, 0.12], segments=12, flatten_y=0.55),
            _authored_blob((-1.18, 2.24, 20.55), (0.28, 0.16, 0.30), rings=4, segments=12, bias=0.2),
            _authored_blob((1.18, 2.24, 20.55), (0.28, 0.16, 0.30), rings=4, segments=12, bias=1.2),
        ]
        return trimesh.util.concatenate(meshes), ["accessories", "backpack", "spike_trophy_rack", "orc_brute_shape", "spiked_scrap_plate"], "bilateral"
    if category == PartCategory.BASE:
        return _authored_round_base(9.8 + index, height=2.0, segments=96), ["base", "round_base", "base_texture"], "radial"
    raise ValueError(f"Unsupported Orc library category: {category}")


def _build_authored_astra_library_mesh(category: PartCategory, index: int) -> tuple[trimesh.Trimesh, list[str], str]:
    if category in (PartCategory.HEAD, PartCategory.HELMET, PartCategory.HEAD_HELMET):
        meshes = [_authored_blob((0, -0.05, 21.78), (1.10, 0.82, 1.18), rings=9, segments=22, bias=0.4), _authored_blob((0, -0.92, 21.48), (0.78, 0.25, 0.42), rings=5, segments=16, bias=1.1), _authored_limb([(0, -1.08, 21.18), (0, -1.48, 20.68)], [0.14, 0.10], segments=12, flatten_y=0.60)]
        return trimesh.util.concatenate(meshes), ["head", "helmet", "helmet_lenses", "field_helmet_rebreather", "astra_shock_trooper_shape", "human_military_proportions"], "bilateral"
    if category in (PartCategory.TORSO, PartCategory.CHEST_ARMOR, PartCategory.TORSO_BODY):
        meshes = [_authored_blob((0, 0.02, 15.25), (2.55, 1.22, 4.35), rings=11, segments=26, bias=0.6), _authored_plate([(-2.20, -1.18, 17.45), (-0.70, -1.38, 18.38), (0.70, -1.38, 18.38), (2.20, -1.18, 17.45), (1.62, -1.24, 14.70), (0, -1.32, 14.05), (-1.62, -1.24, 14.70)], 0.18), _authored_blob((-2.45, -0.20, 17.80), (0.72, 0.48, 0.42), rings=6, segments=16, bias=0.3), _authored_blob((2.45, -0.20, 17.80), (0.72, 0.48, 0.42), rings=6, segments=16, bias=1.2)]
        return trimesh.util.concatenate(meshes), ["body", "torso", "chest_armor", "shoulder_pad", "chest_flak_plate", "flak_plate_and_fatigues", "human_military_proportions", "braced_firing_advance", "astra_shock_trooper_shape"], "bilateral"
    if category in (PartCategory.LEFT_ARM, PartCategory.RIGHT_ARM, PartCategory.ARMS):
        sides = (-1, 1) if category == PartCategory.ARMS else ((-1,) if category == PartCategory.LEFT_ARM else (1,))
        meshes = []
        for side in sides:
            points = [(-2.34, -0.28, 17.25), (-3.10, -1.06, 15.40), (-2.58, -1.72, 13.70)] if side < 0 else [(2.34, -0.28, 17.20), (2.92, -1.12, 15.16), (3.44, -1.78, 13.54)]
            meshes.append(_authored_limb(points, [0.38, 0.32, 0.26], segments=16, flatten_y=0.72))
            meshes.append(_authored_blob((points[-1][0], points[-1][1] - 0.12, points[-1][2] - 0.10), (0.32, 0.20, 0.28), rings=5, segments=14, bias=side))
        tags = ["arms", "hands", "astra_shock_trooper_shape", "human_military_proportions", "braced_firing_advance"]
        tags.append("left_arm" if category == PartCategory.LEFT_ARM else "right_arm" if category == PartCategory.RIGHT_ARM else "left_arm")
        if category == PartCategory.ARMS:
            tags.append("right_arm")
        return trimesh.util.concatenate(meshes), tags, "bilateral" if category == PartCategory.ARMS else "none"
    if category == PartCategory.LEGS:
        meshes = [_authored_limb([(-1.00, -0.05, 11.0), (-1.36, -0.20, 7.10), (-1.52, -0.36, 3.20)], [0.48, 0.40, 0.30], segments=16, flatten_y=0.76), _authored_limb([(1.00, 0.16, 11.0), (1.22, 0.28, 7.20), (1.12, 0.20, 3.20)], [0.48, 0.40, 0.30], segments=16, flatten_y=0.76), _authored_blob((-1.62, -0.80, 1.18), (0.70, 1.00, 0.22), rings=5, segments=14, bias=0.2), _authored_blob((1.18, -0.42, 1.18), (0.70, 1.00, 0.22), rings=5, segments=14, bias=1.0)]
        return trimesh.util.concatenate(meshes), ["legs", "left_leg", "right_leg", "knee_greaves", "human_military_proportions", "braced_firing_advance", "astra_shock_trooper_shape"], "bilateral"
    if category in (PartCategory.WEAPON, PartCategory.WEAPONS):
        meshes = [_authored_limb([(-4.90, -2.05, 14.75), (-0.20, -2.18, 14.25), (6.70, -2.26, 13.75)], [0.24, 0.21, 0.14], segments=20, flatten_y=0.62), _authored_blob((0.75, -2.30, 13.46), (0.76, 0.20, 0.78), rings=5, segments=14, bias=0.8), _authored_blob((4.55, -2.30, 14.44), (1.08, 0.16, 0.24), rings=5, segments=14, bias=1.4), _authored_limb([(6.64, -2.26, 13.75), (7.70, -2.30, 13.62)], [0.14, 0.09], segments=14, flatten_y=0.62)]
        return trimesh.util.concatenate(meshes), ["weapon", "weapon_barrel", "las_rifle", "rifle_across_body_readable", "astra_shock_trooper_shape"], "none"
    if category in (PartCategory.ACCESSORIES, PartCategory.BACKPACK, PartCategory.BACKPACK_ACCESSORIES):
        meshes = [_authored_blob((0, 3.05, 15.75), (2.20, 1.35, 3.15), rings=7, segments=18, bias=0.5), _authored_blob((-2.42, -1.72, 10.75), (0.48, 0.22, 0.72), rings=5, segments=12, bias=0.2), _authored_blob((2.42, -1.72, 10.75), (0.48, 0.22, 0.72), rings=5, segments=12, bias=1.2), _authored_limb([(-1.65, 3.82, 18.2), (-1.65, 4.30, 13.0)], [0.26, 0.18], segments=14, flatten_y=0.75), _authored_limb([(1.65, 3.82, 18.2), (1.65, 4.30, 13.0)], [0.26, 0.18], segments=14, flatten_y=0.75)]
        return trimesh.util.concatenate(meshes), ["accessories", "backpack", "field_pack", "ammo_pouches", "astra_shock_trooper_shape"], "bilateral"
    if category == PartCategory.BASE:
        return _authored_round_base(8.6 + index, height=2.0, segments=96), ["base", "round_base", "base_texture"], "radial"
    raise ValueError(f"Unsupported Astra library category: {category}")


def _build_authored_human_knight_library_mesh(category: PartCategory, index: int) -> tuple[trimesh.Trimesh, list[str], str]:
    if category in (PartCategory.HEAD, PartCategory.HELMET, PartCategory.HEAD_HELMET):
        meshes = [_authored_blob((0, -0.05, 22.25), (1.28, 0.98, 1.45), rings=9, segments=22, bias=0.2), _authored_blob((0, -0.95, 22.42), (1.10, 0.18, 0.38), rings=5, segments=16, bias=0.8), _authored_limb([(0, -0.10, 23.18), (0, -0.10, 24.55)], [0.16, 0.08], segments=12, flatten_y=0.55)]
        return trimesh.util.concatenate(meshes), ["head", "helmet", "human_knight_shape", "crested_helm", "human_heroic_proportions"], "bilateral"
    if category in (PartCategory.TORSO, PartCategory.CHEST_ARMOR, PartCategory.TORSO_BODY):
        meshes = [_authored_blob((0, 0.0, 15.20), (2.95, 1.65, 4.85), rings=12, segments=28, bias=0.4), _authored_blob((0, -1.28, 16.82), (2.75, 0.30, 1.15), rings=6, segments=18, bias=1.0), _authored_blob((-2.75, -0.18, 17.82), (1.12, 0.72, 0.58), rings=6, segments=16, bias=0.2), _authored_blob((2.75, -0.18, 17.82), (1.12, 0.72, 0.58), rings=6, segments=16, bias=1.1), _authored_plate([(-1.55, -1.55, 14.1), (1.55, -1.55, 14.1), (0.82, -1.65, 7.8), (0, -1.72, 6.7), (-0.82, -1.65, 7.8)], 0.12)]
        return trimesh.util.concatenate(meshes), ["body", "torso", "chest_armor", "human_knight_shape", "human_heroic_proportions", "plate_armor", "clean_shoulder_plate", "surcoat_tabard"], "bilateral"
    if category in (PartCategory.LEFT_ARM, PartCategory.RIGHT_ARM, PartCategory.ARMS):
        sides = (-1, 1) if category == PartCategory.ARMS else ((-1,) if category == PartCategory.LEFT_ARM else (1,))
        meshes = []
        for side in sides:
            points = [(side * 2.42, -0.20, 17.55), (side * 3.25, -0.82, 14.8), (side * (4.35 if side > 0 else 4.05), -1.35, 11.2)]
            meshes.append(_authored_limb(points, [0.44, 0.36, 0.28], segments=18, flatten_y=0.72))
            meshes.append(_authored_blob((points[-1][0], points[-1][1] - 0.12, points[-1][2] - 0.10), (0.34, 0.20, 0.30), rings=5, segments=14, bias=side))
        tags = ["arms", "human_knight_shape", "human_heroic_proportions", "balanced_limb_length", "hands"]
        tags.append("left_arm" if category == PartCategory.LEFT_ARM else "right_arm" if category == PartCategory.RIGHT_ARM else "left_arm")
        if category == PartCategory.ARMS:
            tags.append("right_arm")
        return trimesh.util.concatenate(meshes), tags, "bilateral" if category == PartCategory.ARMS else "none"
    if category == PartCategory.LEGS:
        meshes = [_authored_limb([(-1.08, -0.08, 11.0), (-1.28, -0.22, 7.2), (-1.48, -0.38, 2.2)], [0.50, 0.42, 0.32], segments=18, flatten_y=0.76), _authored_limb([(1.08, 0.12, 11.0), (1.22, 0.22, 7.2), (1.30, 0.10, 2.2)], [0.50, 0.42, 0.32], segments=18, flatten_y=0.76), _authored_blob((-1.62, -0.70, 1.05), (0.78, 1.02, 0.24), rings=5, segments=14, bias=0.2), _authored_blob((1.45, -0.44, 1.05), (0.78, 1.02, 0.24), rings=5, segments=14, bias=1.0)]
        return trimesh.util.concatenate(meshes), ["legs", "left_leg", "right_leg", "human_knight_shape", "human_heroic_proportions", "upright_guard"], "bilateral"
    if category in (PartCategory.WEAPON, PartCategory.WEAPONS):
        meshes = [
            _authored_limb([(4.35, -1.62, 9.0), (4.70, -1.66, 14.8), (5.05, -1.70, 20.4)], [0.24, 0.20, 0.16], segments=18, flatten_y=0.88),
            _authored_limb([(4.58, -1.84, 19.25), (5.12, -1.84, 21.20), (5.66, -1.84, 19.25)], [0.26, 0.18, 0.26], segments=16, flatten_y=0.82),
            _authored_blob((4.62, -1.68, 14.1), (0.48, 0.32, 0.48), rings=5, segments=14, bias=0.6),
        ]
        return trimesh.util.concatenate(meshes), ["weapon", "sword_or_longsword", "human_knight_shape", "weapon_forward_readable"], "none"
    if category in (PartCategory.ACCESSORIES, PartCategory.BACKPACK, PartCategory.BACKPACK_ACCESSORIES):
        meshes = [_authored_blob((-5.95, -1.70, 12.55), (1.85, 0.24, 3.55), rings=8, segments=18, bias=0.5), _authored_plate([(-1.65, -1.92, 16.8), (1.65, -1.92, 16.8), (0.78, -2.04, 6.8), (0, -2.10, 5.7), (-0.78, -2.04, 6.8)], 0.12), _authored_blob((0, 1.82, 12.20), (1.55, 0.15, 4.20), rings=7, segments=18, bias=0.9)]
        return trimesh.util.concatenate(meshes), ["accessories", "kite_shield", "surcoat_tabard", "human_knight_shape"], "none"
    if category == PartCategory.BASE:
        return _authored_round_base(8.8 + index, height=2.0, segments=96), ["base", "round_base", "base_texture"], "radial"
    raise ValueError(f"Unsupported Human Knight library category: {category}")


def _apply_ai_shape_directives(mesh: trimesh.Trimesh, category: PartCategory, concept: dict[str, Any]) -> trimesh.Trimesh:
    plan = dict(concept.get("ai_shape_plan") or {})
    directives = dict(plan.get("part_directives") or {})
    directive = dict(directives.get(category.value) or {})
    if not directive:
        return mesh
    scale = np.asarray(directive.get("scale") or [1.0, 1.0, 1.0], dtype=float)
    offset = np.asarray(directive.get("offset") or [0.0, 0.0, 0.0], dtype=float)
    if scale.shape[0] < 3:
        scale = np.pad(scale, (0, 3 - scale.shape[0]), constant_values=1.0)
    if offset.shape[0] < 3:
        offset = np.pad(offset, (0, 3 - offset.shape[0]), constant_values=0.0)
    scale = np.clip(scale[:3], 0.45, 1.85)
    offset = np.clip(offset[:3], -3.5, 3.5)
    if np.allclose(scale, [1.0, 1.0, 1.0], atol=1e-6) and np.allclose(offset, [0.0, 0.0, 0.0], atol=1e-6):
        mesh.metadata["ai_shape_directive"] = {
            "category": category.value,
            "source": plan.get("source"),
            "intent": directive.get("intent", category.value),
            "scale": [float(value) for value in scale],
            "offset": [float(value) for value in offset],
            "applied": False,
        }
        return mesh
    transformed = mesh.copy()
    bounds = np.asarray(transformed.bounds, dtype=float)
    pivot = np.asarray([(bounds[0, 0] + bounds[1, 0]) * 0.5, (bounds[0, 1] + bounds[1, 1]) * 0.5, bounds[0, 2]], dtype=float)
    vertices = np.asarray(transformed.vertices, dtype=float)
    transformed.vertices = (vertices - pivot) * scale + pivot + offset
    transformed.metadata.update(mesh.metadata)
    transformed.metadata["ai_shape_directive"] = {
        "category": category.value,
        "source": plan.get("source"),
        "intent": directive.get("intent", category.value),
        "scale": [float(value) for value in scale],
        "offset": [float(value) for value in offset],
        "applied": True,
    }
    return transformed


def _category_sockets(category: PartCategory, index: int) -> list[ConnectionSocket]:
    if category in (PartCategory.WEAPON, PartCategory.WEAPONS):
        return [ConnectionSocket("right_hand_grip", "hand", (2.0, -2.0, 14.0), 0.35, (0, 1, 0))]
    if category in (PartCategory.BACKPACK, PartCategory.BACKPACK_ACCESSORIES):
        return [ConnectionSocket("torso_back", "backpack", (0, 1.55, 16.0), 0.55, (0, -1, 0))]
    if category == PartCategory.BASE:
        return [ConnectionSocket("feet", "base", (0, 0, 1.0), 1.2, (0, 0, 1))]
    return []


def _build_high_elf_category_mesh(category: PartCategory, index: int, concept: dict[str, Any]) -> tuple[trimesh.Trimesh, list[str], str]:
    """High Elf Warrior component library.

    This branch intentionally does not reuse the generic humanoid blockout. It
    builds the recognizable archetype first: tall/lean anatomy, narrow waist,
    elf helm/ears, layered fantasy armor, glaive, and cape/tabard silhouette.
    """
    return _build_authored_high_elf_category_mesh(category, index, concept)


def _build_authored_high_elf_category_mesh(category: PartCategory, index: int, concept: dict[str, Any]) -> tuple[trimesh.Trimesh, list[str], str]:
    """Authored High Elf library; no cube/sphere/cylinder/capsule final parts."""
    if category in (PartCategory.HEAD, PartCategory.HELMET, PartCategory.HEAD_HELMET):
        meshes = [
            _authored_blob((0, -0.05, 23.55), (1.02, 0.78, 1.58), rings=11, segments=26, bias=0.2),
            _authored_blob((0, -0.08, 24.70), (0.42, 0.28, 1.25), rings=8, segments=22, bias=1.1),
            _authored_limb([(-0.96, -0.05, 23.66), (-1.38, -0.06, 24.05), (-1.92, -0.08, 24.25)], [0.16, 0.11, 0.045], segments=14, flatten_y=0.55),
            _authored_limb([(0.96, -0.05, 23.66), (1.38, -0.06, 24.05), (1.92, -0.08, 24.25)], [0.16, 0.11, 0.045], segments=14, flatten_y=0.55),
            _authored_plate([(-0.62, -0.94, 23.22), (-0.22, -1.02, 23.58), (0.24, -1.02, 23.58), (0.66, -0.94, 23.22), (0.18, -0.98, 23.05), (-0.18, -0.98, 23.05)], 0.08),
        ]
        return trimesh.util.concatenate(meshes), [
            "head",
            "helmet",
            "pointed_ears_or_elf_helm",
            "crested_elf_helm",
            "elf_face",
            "high_elf_warrior_shape",
            "authored_shape_library",
            "anatomical_component_library",
        ], "bilateral"
    if category in (PartCategory.TORSO, PartCategory.CHEST_ARMOR, PartCategory.TORSO_BODY):
        meshes = [
            _authored_blob((0, 0.0, 16.55), (2.05, 1.25, 5.65), rings=12, segments=28, bias=0.5),
            _authored_plate([(-2.05, -1.20, 18.85), (0, -1.42, 19.45), (2.05, -1.20, 18.85), (1.70, -1.22, 17.50), (0, -1.32, 16.92), (-1.70, -1.22, 17.50)], 0.18),
            _authored_plate([(-1.92, -1.24, 16.90), (1.92, -1.24, 16.90), (1.36, -1.28, 15.78), (0, -1.34, 15.22), (-1.36, -1.28, 15.78)], 0.16),
            _authored_plate([(-1.20, -1.28, 15.15), (1.20, -1.28, 15.15), (0.72, -1.30, 14.40), (0, -1.34, 14.10), (-0.72, -1.30, 14.40)], 0.14),
            _authored_blob((-2.10, -0.25, 19.02), (0.98, 0.64, 0.46), rings=7, segments=18, bias=0.3),
            _authored_blob((2.10, -0.25, 19.02), (0.98, 0.64, 0.46), rings=7, segments=18, bias=1.3),
            _authored_plate([(-2.20, 1.78, 18.2), (2.20, 1.78, 18.2), (1.62, 1.96, 7.30), (0, 2.10, 6.0), (-1.62, 1.96, 7.30)], 0.14),
            _authored_plate([(-1.05, -1.72, 14.0), (1.05, -1.72, 14.0), (0.52, -1.86, 6.6), (0, -1.92, 5.8), (-0.52, -1.86, 6.6)], 0.12),
        ]
        return trimesh.util.concatenate(meshes), [
            "body",
            "torso",
            "chest_armor",
            "shoulder_pad",
            "narrow_waist",
            "heroic_posture",
            "layered_fantasy_armor",
            "cape_or_tabard",
            "leaf_plate_edges",
            "high_elf_warrior_shape",
            "authored_shape_library",
            "armor_component_library",
        ], "bilateral"
    if category in (PartCategory.LEFT_ARM, PartCategory.RIGHT_ARM, PartCategory.ARMS):
        sides = (-1, 1) if category == PartCategory.ARMS else ((-1,) if category == PartCategory.LEFT_ARM else (1,))
        meshes: list[trimesh.Trimesh] = []
        tags = ["arms", "hands", "elongated_limbs", "slender_arms", "high_elf_warrior_shape", "authored_shape_library", "anatomical_component_library"]
        for side in sides:
            meshes.append(_authored_limb([(side * 2.42, -0.12, 18.7), (side * 3.12, -0.50, 16.6), (side * 3.62, -0.82, 14.9)], [0.42, 0.34, 0.28], segments=18, flatten_y=0.72))
            meshes.append(_authored_limb([(side * 3.62, -0.82, 14.9), (side * 4.02, -1.14, 13.1), (side * 4.34, -1.42, 11.85)], [0.29, 0.24, 0.20], segments=18, flatten_y=0.72))
            meshes.append(_authored_blob((side * 4.45, -1.54, 11.55), (0.34, 0.20, 0.42), rings=5, segments=14, bias=side))
        tags.append("left_arm" if category == PartCategory.LEFT_ARM else "right_arm" if category == PartCategory.RIGHT_ARM else "left_arm")
        if category == PartCategory.ARMS:
            tags.append("right_arm")
        return trimesh.util.concatenate(meshes), tags, "bilateral" if category == PartCategory.ARMS else "none"
    if category == PartCategory.LEGS:
        meshes = [
            _authored_limb([(-1.05, -0.05, 10.8), (-1.24, -0.12, 8.3), (-1.35, -0.20, 5.9)], [0.50, 0.43, 0.34], segments=18, flatten_y=0.78),
            _authored_limb([(1.05, 0.12, 10.8), (1.14, 0.26, 8.3), (1.20, 0.34, 5.9)], [0.50, 0.43, 0.34], segments=18, flatten_y=0.78),
            _authored_limb([(-1.35, -0.20, 5.9), (-1.48, -0.32, 3.6), (-1.55, -0.42, 1.65)], [0.34, 0.29, 0.24], segments=18, flatten_y=0.78),
            _authored_limb([(1.20, 0.34, 5.9), (1.32, 0.28, 3.6), (1.38, 0.18, 1.65)], [0.34, 0.29, 0.24], segments=18, flatten_y=0.78),
            _authored_blob((-1.62, -0.78, 1.05), (0.70, 1.05, 0.22), rings=5, segments=14, bias=0.4),
            _authored_blob((1.45, -0.34, 1.05), (0.70, 1.05, 0.22), rings=5, segments=14, bias=1.2),
        ]
        return trimesh.util.concatenate(meshes), [
            "legs",
            "left_leg",
            "right_leg",
            "elongated_limbs",
            "heroic_posture",
            "advancing_stride",
            "high_elf_warrior_shape",
            "authored_shape_library",
            "anatomical_component_library",
        ], "bilateral"
    if category in (PartCategory.WEAPON, PartCategory.WEAPONS):
        meshes = [
            _authored_limb([(1.30, -1.95, 6.6), (1.52, -2.05, 15.8), (1.72, -2.15, 27.2)], [0.28, 0.23, 0.18], segments=24, flatten_y=1.0),
            _authored_plate([(1.10, -2.24, 26.35), (1.72, -2.25, 28.50), (2.30, -2.24, 26.35), (1.72, -2.26, 26.70)], 0.82),
            _authored_plate([(0.70, -2.24, 26.18), (1.62, -2.25, 26.62), (2.74, -2.24, 26.18), (1.62, -2.26, 25.82)], 0.78),
            _authored_blob((1.44, -2.12, 13.20), (0.42, 0.46, 0.78), rings=7, segments=18, bias=0.8),
            _authored_blob((1.72, -2.12, 26.18), (0.64, 0.52, 0.48), rings=6, segments=18, bias=1.4),
        ]
        return trimesh.util.concatenate(meshes), [
            "weapon",
            "weapon_barrel",
            "sword_spear_or_glaive",
            "glaive",
            "weapon_forward_readable",
            "high_elf_warrior_shape",
            "authored_shape_library",
            "weapon_component_library",
        ], "none"
    if category in (PartCategory.ACCESSORIES, PartCategory.BACKPACK, PartCategory.BACKPACK_ACCESSORIES):
        meshes = [
            _authored_plate([(-2.35, 2.28, 19.0), (2.35, 2.28, 19.0), (1.65, 2.48, 7.2), (0, 2.62, 5.6), (-1.65, 2.48, 7.2)], 0.14),
            _authored_plate([(-1.05, -1.90, 17.0), (1.05, -1.90, 17.0), (0.55, -2.02, 6.7), (0, -2.08, 5.7), (-0.55, -2.02, 6.7)], 0.10),
            _authored_limb([(-1.55, 1.88, 18.2), (-1.62, 1.94, 13.2), (-1.48, 2.02, 8.0)], [0.34, 0.24, 0.16], segments=18, flatten_y=0.45),
            _authored_limb([(1.55, 1.88, 18.2), (1.62, 1.94, 13.2), (1.48, 2.02, 8.0)], [0.34, 0.24, 0.16], segments=18, flatten_y=0.45),
        ]
        return trimesh.util.concatenate(meshes), [
            "accessories",
            "backpack",
            "cape_or_tabard",
            "cape",
            "tabard",
            "flowing_banner_shape",
            "high_elf_warrior_shape",
            "authored_shape_library",
            "cloth_component_library",
        ], "bilateral"
    if category == PartCategory.BASE:
        meshes = [_authored_round_base(9.0 + index, height=2.0, segments=96)]
        for angle in np.linspace(0, np.pi * 2, 18, endpoint=False):
            meshes.append(_authored_blob((float(np.cos(angle) * 4.8), float(np.sin(angle) * 4.8), 1.1), (0.30, 0.22, 0.10), rings=4, segments=10, bias=float(angle)))
        return trimesh.util.concatenate(meshes), ["base", "round_base", "base_texture"], "radial"
    raise ValueError(f"Unsupported High Elf category: {category}")


def _build_astra_shock_trooper_category_mesh(category: PartCategory, index: int, concept: dict[str, Any]) -> tuple[trimesh.Trimesh, list[str], str]:
    """Astra Shock Trooper base form: human military miniature, not mannequin."""
    if category in (PartCategory.HEAD, PartCategory.HELMET, PartCategory.HEAD_HELMET):
        meshes = [
            _authored_blob((0, -0.05, 21.78), (1.12, 0.82, 1.22), rings=9, segments=22, bias=0.4),
            _authored_blob((0, -0.92, 21.48), (0.78, 0.26, 0.42), rings=5, segments=16, bias=1.1),
            _authored_blob((-0.42, -1.12, 21.86), (0.22, 0.12, 0.18), rings=4, segments=12, bias=0.2),
            _authored_blob((0.42, -1.12, 21.86), (0.22, 0.12, 0.18), rings=4, segments=12, bias=1.2),
            _authored_limb([(0.0, -1.10, 21.18), (0.0, -1.46, 20.68)], [0.16, 0.11], segments=12, flatten_y=0.60),
        ]
        return trimesh.util.concatenate(meshes), [
            "head",
            "helmet",
            "helmet_lenses",
            "field_helmet_rebreather",
            "astra_shock_trooper_shape",
            "human_military_proportions",
        ], "bilateral"
    if category in (PartCategory.TORSO, PartCategory.CHEST_ARMOR, PartCategory.TORSO_BODY):
        meshes = [
            _authored_blob((0, 0.02, 15.25), (2.55, 1.22, 4.35), rings=11, segments=26, bias=0.6),
            _authored_plate([(-2.20, -1.18, 17.45), (-0.70, -1.38, 18.38), (0.70, -1.38, 18.38), (2.20, -1.18, 17.45), (1.62, -1.24, 14.70), (0, -1.32, 14.05), (-1.62, -1.24, 14.70)], 0.18),
            _authored_blob((-2.45, -0.20, 17.80), (0.72, 0.48, 0.42), rings=6, segments=16, bias=0.3),
            _authored_blob((2.45, -0.20, 17.80), (0.72, 0.48, 0.42), rings=6, segments=16, bias=1.2),
            _authored_plate([(-1.75, -1.32, 13.65), (1.75, -1.32, 13.65), (1.24, -1.42, 10.15), (0, -1.50, 9.20), (-1.24, -1.42, 10.15)], 0.12),
        ]
        return trimesh.util.concatenate(meshes), [
            "body",
            "torso",
            "chest_armor",
            "shoulder_pad",
            "chest_flak_plate",
            "flak_plate_and_fatigues",
            "human_military_proportions",
            "braced_firing_advance",
            "astra_shock_trooper_shape",
        ], "bilateral"
    if category in (PartCategory.LEFT_ARM, PartCategory.RIGHT_ARM, PartCategory.ARMS):
        sides = (-1, 1) if category == PartCategory.ARMS else ((-1,) if category == PartCategory.LEFT_ARM else (1,))
        meshes: list[trimesh.Trimesh] = []
        tags = ["arms", "hands", "astra_shock_trooper_shape", "human_military_proportions", "braced_firing_advance"]
        for side in sides:
            if side < 0:
                points = [(-2.34, -0.28, 17.25), (-3.10, -1.06, 15.40), (-2.58, -1.72, 13.70)]
            else:
                points = [(2.34, -0.28, 17.20), (2.92, -1.12, 15.16), (3.44, -1.78, 13.54)]
            meshes.append(_authored_limb(points, [0.38, 0.32, 0.26], segments=16, flatten_y=0.72))
            meshes.append(_authored_blob((points[-1][0], points[-1][1] - 0.12, points[-1][2] - 0.10), (0.32, 0.20, 0.28), rings=5, segments=14, bias=side))
        tags.append("left_arm" if category == PartCategory.LEFT_ARM else "right_arm" if category == PartCategory.RIGHT_ARM else "left_arm")
        if category == PartCategory.ARMS:
            tags.append("right_arm")
        return trimesh.util.concatenate(meshes), tags, "bilateral" if category == PartCategory.ARMS else "none"
    if category == PartCategory.LEGS:
        meshes = [
            _authored_limb([(-1.00, -0.05, 11.0), (-1.36, -0.20, 7.10), (-1.52, -0.36, 3.20)], [0.48, 0.40, 0.30], segments=16, flatten_y=0.76),
            _authored_limb([(1.00, 0.16, 11.0), (1.22, 0.28, 7.20), (1.12, 0.20, 3.20)], [0.48, 0.40, 0.30], segments=16, flatten_y=0.76),
            _authored_blob((-1.62, -0.80, 1.18), (0.70, 1.00, 0.22), rings=5, segments=14, bias=0.2),
            _authored_blob((1.18, -0.42, 1.18), (0.70, 1.00, 0.22), rings=5, segments=14, bias=1.0),
        ]
        return trimesh.util.concatenate(meshes), ["legs", "left_leg", "right_leg", "knee_greaves", "human_military_proportions", "braced_firing_advance", "astra_shock_trooper_shape"], "bilateral"
    if category in (PartCategory.WEAPON, PartCategory.WEAPONS):
        meshes = [
            _authored_limb([(-2.75, -2.05, 14.35), (0.60, -2.18, 14.22), (4.90, -2.24, 14.12)], [0.22, 0.20, 0.16], segments=20, flatten_y=0.62),
            _authored_blob((1.05, -2.30, 13.46), (0.58, 0.18, 0.68), rings=5, segments=14, bias=0.8),
            _authored_blob((3.35, -2.30, 14.62), (0.82, 0.16, 0.22), rings=5, segments=14, bias=1.4),
            _authored_limb([(4.86, -2.24, 14.12), (5.72, -2.26, 14.12)], [0.12, 0.09], segments=14, flatten_y=0.62),
        ]
        return trimesh.util.concatenate(meshes), ["weapon", "weapon_barrel", "las_rifle", "rifle_across_body_readable", "astra_shock_trooper_shape"], "none"
    if category in (PartCategory.ACCESSORIES, PartCategory.BACKPACK, PartCategory.BACKPACK_ACCESSORIES):
        meshes = [
            _authored_blob((0, 1.82, 15.55), (1.72, 0.46, 2.35), rings=7, segments=18, bias=0.5),
            _authored_blob((-2.08, -1.66, 10.75), (0.42, 0.22, 0.62), rings=5, segments=12, bias=0.2),
            _authored_blob((2.08, -1.66, 10.75), (0.42, 0.22, 0.62), rings=5, segments=12, bias=1.2),
        ]
        return trimesh.util.concatenate(meshes), ["accessories", "backpack", "field_pack", "ammo_pouches", "astra_shock_trooper_shape"], "bilateral"
    if category == PartCategory.BASE:
        return _race_base(8.6 + index), ["base", "round_base", "base_texture"], "radial"
    raise ValueError(f"Unsupported Astra Shock Trooper category: {category}")


def _build_space_terminator_category_mesh(category: PartCategory, index: int, concept: dict[str, Any]) -> tuple[trimesh.Trimesh, list[str], str]:
    """Space Terminator base form: squat massive exo armor with heavy weapon."""
    if category in (PartCategory.HEAD, PartCategory.HELMET, PartCategory.HEAD_HELMET):
        meshes = [
            _authored_blob((0, -0.18, 20.75), (1.18, 0.74, 1.05), rings=8, segments=22, bias=0.4),
            _authored_blob((0, -1.08, 20.55), (0.95, 0.22, 0.36), rings=5, segments=16, bias=1.0),
            _authored_blob((-0.42, -1.24, 20.88), (0.24, 0.12, 0.16), rings=4, segments=12, bias=0.2),
            _authored_blob((0.42, -1.24, 20.88), (0.24, 0.12, 0.16), rings=4, segments=12, bias=1.2),
        ]
        return trimesh.util.concatenate(meshes), ["head", "helmet", "helmet_lenses", "recessed_exo_helmet", "space_terminator_shape", "bulky_exo_proportions"], "bilateral"
    if category in (PartCategory.TORSO, PartCategory.CHEST_ARMOR, PartCategory.TORSO_BODY):
        meshes = [
            _authored_blob((0, 0.10, 15.05), (3.95, 2.00, 4.75), rings=12, segments=30, bias=0.7),
            _authored_blob((-4.10, -0.10, 17.95), (1.70, 1.08, 1.08), rings=7, segments=20, bias=0.3),
            _authored_blob((4.10, -0.10, 17.95), (1.70, 1.08, 1.08), rings=7, segments=20, bias=1.3),
            _authored_plate([(-2.75, -1.70, 17.50), (0, -1.95, 18.45), (2.75, -1.70, 17.50), (2.30, -1.72, 14.30), (0, -1.88, 13.15), (-2.30, -1.72, 14.30)], 0.24),
            _authored_blob((0, -1.78, 12.25), (2.05, 0.34, 0.70), rings=5, segments=16, bias=0.9),
        ]
        return trimesh.util.concatenate(meshes), ["body", "torso", "chest_armor", "shoulder_pad", "massive_exo_plate_armor", "huge_pauldrons", "massive_shoulders", "space_terminator_shape", "bulky_exo_proportions"], "bilateral"
    if category in (PartCategory.LEFT_ARM, PartCategory.RIGHT_ARM, PartCategory.ARMS):
        sides = (-1, 1) if category == PartCategory.ARMS else ((-1,) if category == PartCategory.LEFT_ARM else (1,))
        meshes: list[trimesh.Trimesh] = []
        tags = ["arms", "hands", "thick_power_limbs", "space_terminator_shape", "bulky_exo_proportions", "slow_braced_advance"]
        for side in sides:
            meshes.append(_authored_limb([(side * 3.70, -0.10, 17.40), (side * 4.82, -0.72, 14.05), (side * 4.45, -1.26, 10.90)], [0.82, 0.72, 0.56], segments=20, flatten_y=0.78))
            hand_scale = (0.74, 0.44, 0.70) if side < 0 else (0.58, 0.34, 0.48)
            meshes.append(_authored_blob((side * 4.42, -1.42, 10.55), hand_scale, rings=5, segments=16, bias=side))
        tags.append("left_arm" if category == PartCategory.LEFT_ARM else "right_arm" if category == PartCategory.RIGHT_ARM else "left_arm")
        if category == PartCategory.ARMS:
            tags.append("right_arm")
        return trimesh.util.concatenate(meshes), tags, "bilateral" if category == PartCategory.ARMS else "none"
    if category == PartCategory.LEGS:
        meshes = [
            _authored_limb([(-1.50, -0.05, 11.25), (-1.75, -0.16, 6.95), (-1.88, -0.30, 2.95)], [0.86, 0.74, 0.58], segments=20, flatten_y=0.82),
            _authored_limb([(1.50, 0.12, 11.25), (1.66, 0.20, 6.95), (1.58, 0.10, 2.95)], [0.86, 0.74, 0.58], segments=20, flatten_y=0.82),
            _authored_blob((-2.00, -0.78, 1.12), (1.08, 1.25, 0.30), rings=5, segments=16, bias=0.3),
            _authored_blob((1.66, -0.42, 1.12), (1.08, 1.25, 0.30), rings=5, segments=16, bias=1.0),
        ]
        return trimesh.util.concatenate(meshes), ["legs", "left_leg", "right_leg", "thick_power_limbs", "slow_braced_advance", "space_terminator_shape", "bulky_exo_proportions"], "bilateral"
    if category in (PartCategory.WEAPON, PartCategory.WEAPONS):
        meshes = [
            _authored_blob((4.80, -2.02, 13.95), (1.58, 0.42, 0.78), rings=6, segments=18, bias=0.5),
            _authored_limb([(3.50, -2.10, 14.05), (6.45, -2.18, 14.05)], [0.28, 0.22], segments=18, flatten_y=0.62),
            _authored_blob((6.78, -2.20, 14.05), (0.52, 0.26, 0.38), rings=5, segments=14, bias=1.1),
            _authored_blob((4.02, -2.22, 13.10), (0.62, 0.24, 0.70), rings=5, segments=14, bias=0.7),
        ]
        return trimesh.util.concatenate(meshes), ["weapon", "weapon_barrel", "heavy_storm_rifle", "power_fist_cannon", "heavy_weapon_forward_readable", "space_terminator_shape"], "none"
    if category in (PartCategory.ACCESSORIES, PartCategory.BACKPACK, PartCategory.BACKPACK_ACCESSORIES):
        meshes = [
            _authored_blob((0, 2.25, 16.35), (2.55, 0.70, 2.25), rings=7, segments=20, bias=0.5),
            _authored_blob((-1.10, 2.70, 18.75), (0.58, 0.32, 0.78), rings=5, segments=14, bias=0.1),
            _authored_blob((1.10, 2.70, 18.75), (0.58, 0.32, 0.78), rings=5, segments=14, bias=1.1),
        ]
        return trimesh.util.concatenate(meshes), ["accessories", "backpack", "reactor_backpack", "space_terminator_shape", "bulky_exo_proportions"], "bilateral"
    if category == PartCategory.BASE:
        return _race_base(10.6 + index), ["base", "round_base", "base_texture"], "radial"
    raise ValueError(f"Unsupported Space Terminator category: {category}")


def _build_dwarf_category_mesh(category: PartCategory, index: int, concept: dict[str, Any]) -> tuple[trimesh.Trimesh, list[str], str]:
    """Dwarf Warrior silhouette library: short, broad, bearded, shielded."""
    if category in (PartCategory.HEAD, PartCategory.HELMET, PartCategory.HEAD_HELMET):
        meshes = [
            _ellipsoid((0, -0.10, 17.85), (1.25, 0.95, 1.20), 2),
            _ellipsoid((0, -1.10, 16.90), (1.15, 0.28, 1.30), 2),
            _oriented_cone((-0.78, -0.92, 16.75), (-1.25, -1.15, 15.45), 0.18, 16),
            _oriented_cone((0.78, -0.92, 16.75), (1.25, -1.15, 15.45), 0.18, 16),
            _ellipsoid((0, -0.08, 18.95), (1.50, 0.86, 0.44), 1),
            _oriented_cone((-0.72, -0.04, 19.12), (-1.62, -0.04, 19.22), 0.18, 14),
            _oriented_cone((0.72, -0.04, 19.12), (1.62, -0.04, 19.22), 0.18, 14),
        ]
        return trimesh.util.concatenate(meshes), ["head", "helmet", "dwarf_warrior_shape", "braided_beard", "dwarf_helm_and_beard", "short_stocky_proportions"], "bilateral"
    if category in (PartCategory.TORSO, PartCategory.CHEST_ARMOR, PartCategory.TORSO_BODY):
        meshes = [
            _ellipsoid((0, -0.05, 12.55), (3.75, 1.85, 3.85), 3),
            _ellipsoid((0, -0.25, 9.80), (3.25, 1.55, 1.30), 2),
            _ellipsoid((-3.35, -0.15, 14.30), (1.35, 0.92, 0.78), 2),
            _ellipsoid((3.35, -0.15, 14.30), (1.35, 0.92, 0.78), 2),
            _ellipsoid((0, -1.42, 13.55), (2.95, 0.22, 0.72), 2),
            _ellipsoid((0, -1.44, 11.95), (3.20, 0.22, 0.45), 1),
        ]
        return trimesh.util.concatenate(meshes), ["body", "torso", "chest_armor", "shoulder_pad", "dwarf_warrior_shape", "short_stocky_proportions", "broad_torso", "runic_heavy_armor", "wide_shoulder_plates"], "bilateral"
    if category in (PartCategory.LEFT_ARM, PartCategory.RIGHT_ARM, PartCategory.ARMS):
        sides = (-1, 1) if category == PartCategory.ARMS else ((-1,) if category == PartCategory.LEFT_ARM else (1,))
        meshes: list[trimesh.Trimesh] = []
        tags = ["arms", "thick_limbs", "dwarf_warrior_shape", "short_stocky_proportions"]
        for side in sides:
            meshes.append(_tapered_limb((side * 3.10, -0.16, 13.95), (side * 4.15, -0.70, 10.75), 0.62, 0.54))
            meshes.append(_tapered_limb((side * 4.15, -0.70, 10.75), (side * 3.25, -1.32, 8.10), 0.54, 0.46))
            meshes.append(_ellipsoid((side * 3.18, -1.46, 7.82), (0.55, 0.30, 0.48), 1))
        tags.append("left_arm" if category == PartCategory.LEFT_ARM else "right_arm" if category == PartCategory.RIGHT_ARM else "left_arm")
        if category == PartCategory.ARMS:
            tags.append("right_arm")
        return trimesh.util.concatenate(meshes), tags, "bilateral" if category == PartCategory.ARMS else "none"
    if category == PartCategory.LEGS:
        meshes = [
            _tapered_limb((-1.18, -0.04, 8.60), (-1.42, -0.12, 4.05), 0.68, 0.58),
            _tapered_limb((1.18, 0.08, 8.60), (1.42, 0.12, 4.05), 0.68, 0.58),
            _ellipsoid((-1.58, -0.62, 2.45), (1.10, 1.18, 0.50), 1),
            _ellipsoid((1.58, -0.62, 2.45), (1.10, 1.18, 0.50), 1),
        ]
        return trimesh.util.concatenate(meshes), ["legs", "left_leg", "right_leg", "dwarf_warrior_shape", "short_stocky_proportions", "thick_limbs", "heavy_boots", "grounded_stance"], "bilateral"
    if category in (PartCategory.WEAPON, PartCategory.WEAPONS):
        meshes = [
            _tapered_limb((4.10, -1.85, 7.10), (4.70, -1.95, 14.90), 0.16, 0.13),
            _ellipsoid((4.70, -2.02, 15.35), (1.30, 0.30, 0.70), 1),
            _ellipsoid((5.52, -2.02, 15.32), (0.48, 0.24, 0.42), 1),
            _ellipsoid((3.85, -2.02, 15.32), (0.48, 0.24, 0.42), 1),
        ]
        return trimesh.util.concatenate(meshes), ["weapon", "axe_or_hammer", "dwarf_warrior_shape", "weapon_forward_readable"], "none"
    if category in (PartCategory.ACCESSORIES, PartCategory.BACKPACK, PartCategory.BACKPACK_ACCESSORIES):
        meshes = [
            _ellipsoid((-4.70, -1.72, 11.10), (1.18, 0.24, 2.05), 2),
            _ellipsoid((-4.70, -1.88, 11.10), (1.42, 0.14, 1.42), 1),
            _ellipsoid((0, 1.82, 12.0), (2.30, 0.20, 2.20), 1),
        ]
        return trimesh.util.concatenate(meshes), ["accessories", "backpack", "round_shield", "dwarf_warrior_shape", "rune_plate_edges"], "none"
    if category == PartCategory.BASE:
        return _race_base(8.4 + index), ["base", "round_base", "base_texture"], "radial"
    raise ValueError(f"Unsupported Dwarf category: {category}")


def _build_orc_category_mesh(category: PartCategory, index: int, concept: dict[str, Any]) -> tuple[trimesh.Trimesh, list[str], str]:
    """Orc Brute silhouette library: hunched, huge shoulders, tusks, oversized weapon."""
    if category in (PartCategory.HEAD, PartCategory.HELMET, PartCategory.HEAD_HELMET):
        meshes = [
            _ellipsoid((0, -0.28, 20.05), (1.75, 1.05, 1.25), 2),
            _ellipsoid((0, -1.12, 19.55), (1.55, 0.34, 0.62), 1),
            _oriented_cone((-0.48, -1.36, 19.34), (-0.82, -1.74, 18.88), 0.12, 12),
            _oriented_cone((0.48, -1.36, 19.34), (0.82, -1.74, 18.88), 0.12, 12),
            _ellipsoid((0, 0.38, 19.82), (1.62, 0.44, 0.68), 1),
        ]
        return trimesh.util.concatenate(meshes), ["head", "helmet", "orc_brute_shape", "tusks", "heavy_jaw", "hunched_posture"], "bilateral"
    if category in (PartCategory.TORSO, PartCategory.CHEST_ARMOR, PartCategory.TORSO_BODY):
        meshes = [
            _ellipsoid((0, 0.28, 14.10), (4.35, 2.15, 4.50), 3),
            _ellipsoid((0, -0.85, 17.28), (5.10, 0.78, 1.25), 2),
            _ellipsoid((-3.98, -0.10, 16.85), (1.85, 1.25, 0.95), 2),
            _ellipsoid((3.98, -0.10, 16.85), (1.85, 1.25, 0.95), 2),
            _ellipsoid((0, -1.58, 14.38), (3.20, 0.26, 1.00), 1),
        ]
        return trimesh.util.concatenate(meshes), ["body", "torso", "chest_armor", "shoulder_pad", "orc_brute_shape", "hunched_posture", "massive_shoulders", "crude_scrap_armor", "spiked_scrap_plate"], "bilateral"
    if category in (PartCategory.LEFT_ARM, PartCategory.RIGHT_ARM, PartCategory.ARMS):
        sides = (-1, 1) if category == PartCategory.ARMS else ((-1,) if category == PartCategory.LEFT_ARM else (1,))
        meshes: list[trimesh.Trimesh] = []
        tags = ["arms", "orc_brute_shape", "long_powerful_arms", "hunched_posture"]
        for side in sides:
            meshes.append(_tapered_limb((side * 3.80, -0.42, 16.25), (side * 5.40, -1.30, 11.20), 0.82, 0.66))
            meshes.append(_tapered_limb((side * 5.40, -1.30, 11.20), (side * 3.65, -1.88, 6.80), 0.66, 0.54))
            meshes.append(_ellipsoid((side * 3.52, -2.02, 6.52), (0.70, 0.34, 0.56), 1))
        tags.append("left_arm" if category == PartCategory.LEFT_ARM else "right_arm" if category == PartCategory.RIGHT_ARM else "left_arm")
        if category == PartCategory.ARMS:
            tags.append("right_arm")
        return trimesh.util.concatenate(meshes), tags, "bilateral" if category == PartCategory.ARMS else "none"
    if category == PartCategory.LEGS:
        meshes = [
            _tapered_limb((-1.45, 0.05, 10.35), (-1.82, -0.22, 5.15), 0.82, 0.64),
            _tapered_limb((1.45, 0.32, 10.35), (1.72, 0.22, 5.15), 0.82, 0.64),
            _ellipsoid((-1.92, -0.68, 1.88), (1.20, 1.22, 0.55), 1),
            _ellipsoid((1.75, -0.28, 1.88), (1.20, 1.22, 0.55), 1),
        ]
        return trimesh.util.concatenate(meshes), ["legs", "left_leg", "right_leg", "orc_brute_shape", "thick_legs", "hunched_charge"], "bilateral"
    if category in (PartCategory.WEAPON, PartCategory.WEAPONS):
        meshes = [
            _tapered_limb((4.00, -2.12, 6.80), (5.85, -2.30, 18.40), 0.22, 0.18),
            _ellipsoid((6.00, -2.36, 18.95), (1.70, 0.34, 1.10), 1),
            _ellipsoid((5.48, -2.42, 17.75), (1.20, 0.26, 0.58), 1),
        ]
        return trimesh.util.concatenate(meshes), ["weapon", "oversized_choppa", "massive_choppa", "orc_brute_shape", "weapon_overweight_readable"], "none"
    if category in (PartCategory.ACCESSORIES, PartCategory.BACKPACK, PartCategory.BACKPACK_ACCESSORIES):
        meshes = [_ellipsoid((0, 2.18, 15.40), (2.50, 0.28, 2.35), 2)]
        for side in (-1, 1):
            meshes.append(_oriented_cone((side * 0.80, 2.28, 17.0), (side * 1.60, 2.44, 19.6), 0.18, 14))
        return trimesh.util.concatenate(meshes), ["accessories", "backpack", "spike_trophy_rack", "orc_brute_shape", "spiked_scrap_plate"], "bilateral"
    if category == PartCategory.BASE:
        return _race_base(9.8 + index), ["base", "round_base", "base_texture"], "radial"
    raise ValueError(f"Unsupported Orc category: {category}")


def _build_human_knight_category_mesh(category: PartCategory, index: int, concept: dict[str, Any]) -> tuple[trimesh.Trimesh, list[str], str]:
    """Human Knight silhouette library: upright, shield/sword, crested helm, tabard."""
    if category in (PartCategory.HEAD, PartCategory.HELMET, PartCategory.HEAD_HELMET):
        meshes = [
            _ellipsoid((0, -0.05, 22.25), (1.28, 0.98, 1.45), 2),
            _ellipsoid((0, -0.95, 22.42), (1.10, 0.18, 0.38), 1),
            _oriented_cone((0, -0.04, 23.25), (0, -0.04, 25.20), 0.22, 18),
        ]
        return trimesh.util.concatenate(meshes), ["head", "helmet", "human_knight_shape", "crested_helm", "human_heroic_proportions"], "bilateral"
    if category in (PartCategory.TORSO, PartCategory.CHEST_ARMOR, PartCategory.TORSO_BODY):
        meshes = [
            _ellipsoid((0, 0.0, 15.20), (2.95, 1.65, 4.85), 3),
            _ellipsoid((0, -1.28, 16.82), (2.75, 0.30, 1.15), 2),
            _ellipsoid((-2.75, -0.18, 17.82), (1.12, 0.72, 0.58), 2),
            _ellipsoid((2.75, -0.18, 17.82), (1.12, 0.72, 0.58), 2),
            _ellipsoid((0, -1.55, 11.10), (1.55, 0.18, 3.15), 1),
        ]
        return trimesh.util.concatenate(meshes), ["body", "torso", "chest_armor", "human_knight_shape", "human_heroic_proportions", "plate_armor", "clean_shoulder_plate", "surcoat_tabard"], "bilateral"
    if category in (PartCategory.LEFT_ARM, PartCategory.RIGHT_ARM, PartCategory.ARMS):
        sides = (-1, 1) if category == PartCategory.ARMS else ((-1,) if category == PartCategory.LEFT_ARM else (1,))
        meshes: list[trimesh.Trimesh] = []
        tags = ["arms", "human_knight_shape", "human_heroic_proportions", "balanced_limb_length"]
        for side in sides:
            end_x = side * (4.75 if side < 0 else 4.05)
            meshes.append(_tapered_limb((side * 2.65, -0.10, 17.25), (side * 3.72, -0.72, 13.55), 0.52, 0.42))
            meshes.append(_tapered_limb((side * 3.72, -0.72, 13.55), (end_x, -1.36, 10.15), 0.42, 0.34))
            meshes.append(_ellipsoid((end_x, -1.46, 9.82), (0.42, 0.24, 0.44), 1))
        tags.append("left_arm" if category == PartCategory.LEFT_ARM else "right_arm" if category == PartCategory.RIGHT_ARM else "left_arm")
        if category == PartCategory.ARMS:
            tags.append("right_arm")
        return trimesh.util.concatenate(meshes), tags, "bilateral" if category == PartCategory.ARMS else "none"
    if category == PartCategory.LEGS:
        meshes = [
            _tapered_limb((-1.08, -0.04, 10.45), (-1.38, -0.16, 5.50), 0.56, 0.42),
            _tapered_limb((1.08, 0.12, 10.45), (1.30, 0.18, 5.50), 0.56, 0.42),
            _tapered_limb((-1.38, -0.16, 5.50), (-1.55, -0.38, 1.55), 0.42, 0.32),
            _tapered_limb((1.30, 0.18, 5.50), (1.38, 0.04, 1.55), 0.42, 0.32),
            _ellipsoid((-1.62, -0.70, 1.05), (0.78, 1.02, 0.24), 1),
            _ellipsoid((1.45, -0.44, 1.05), (0.78, 1.02, 0.24), 1),
        ]
        return trimesh.util.concatenate(meshes), ["legs", "left_leg", "right_leg", "human_knight_shape", "human_heroic_proportions", "upright_guard"], "bilateral"
    if category in (PartCategory.WEAPON, PartCategory.WEAPONS):
        meshes = [
            _tapered_limb((4.95, -1.62, 7.20), (4.95, -1.70, 18.65), 0.14, 0.10),
            _oriented_cone((4.95, -1.70, 18.65), (4.95, -1.70, 21.25), 0.34, 20),
            _ellipsoid((4.95, -1.70, 14.20), (0.90, 0.18, 0.16), 1),
        ]
        return trimesh.util.concatenate(meshes), ["weapon", "sword_or_longsword", "human_knight_shape", "weapon_forward_readable"], "none"
    if category in (PartCategory.ACCESSORIES, PartCategory.BACKPACK, PartCategory.BACKPACK_ACCESSORIES):
        meshes = [
            _ellipsoid((-4.75, -1.70, 12.55), (1.24, 0.22, 2.95), 2),
            _oriented_cone((-4.75, -1.70, 9.15), (-4.75, -1.70, 7.40), 1.12, 24),
            _ellipsoid((0, 1.82, 12.20), (1.65, 0.15, 4.60), 1),
        ]
        return trimesh.util.concatenate(meshes), ["accessories", "kite_shield", "surcoat_tabard", "human_knight_shape"], "none"
    if category == PartCategory.BASE:
        return _race_base(8.8 + index), ["base", "round_base", "base_texture"], "radial"
    raise ValueError(f"Unsupported Human Knight category: {category}")


def _race_base(radius: float) -> trimesh.Trimesh:
    meshes = [trimesh.creation.cylinder(radius=radius, height=2.0, sections=128)]
    for angle in np.linspace(0, np.pi * 2, 10, endpoint=False):
        meshes.append(_ellipsoid((float(np.cos(angle) * radius * 0.52), float(np.sin(angle) * radius * 0.52), 1.1), (0.28, 0.20, 0.10), 1))
    return trimesh.util.concatenate(meshes)


def _authored_blob(center: tuple[float, float, float], radii: tuple[float, float, float], *, rings: int = 10, segments: int = 24, bias: float = 0.0) -> trimesh.Trimesh:
    """Authored organic/armor volume; avoids box/sphere/cylinder/capsule primitives."""
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    for i in range(rings + 1):
        v = i / rings
        theta = -np.pi / 2 + v * np.pi
        ring_radius = np.cos(theta)
        z = np.sin(theta)
        for j in range(segments):
            u = j / segments
            phi = u * np.pi * 2
            ripple = 1.0 + 0.055 * np.sin(3 * phi + bias) * np.sin(np.pi * v) + 0.035 * np.cos(5 * phi - bias) * np.sin(np.pi * v) ** 2
            x = np.cos(phi) * ring_radius * ripple
            y = np.sin(phi) * ring_radius * (1.0 + 0.04 * np.cos(2 * phi + bias))
            vertices.append([center[0] + x * radii[0], center[1] + y * radii[1], center[2] + z * radii[2]])
    for i in range(rings):
        for j in range(segments):
            a = i * segments + j
            b = i * segments + (j + 1) % segments
            c = (i + 1) * segments + (j + 1) % segments
            d = (i + 1) * segments + j
            faces.append([a, b, c])
            faces.append([a, c, d])
    mesh = trimesh.Trimesh(vertices=np.asarray(vertices, dtype=float), faces=np.asarray(faces, dtype=int), process=True)
    return _closed_authored_mesh(mesh)


def _authored_limb(points: list[tuple[float, float, float]], radii: list[float], *, segments: int = 18, flatten_y: float = 0.82) -> trimesh.Trimesh:
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    pts = [np.asarray(point, dtype=float) for point in points]
    for idx, point in enumerate(pts):
        tangent = pts[min(idx + 1, len(pts) - 1)] - pts[max(idx - 1, 0)]
        if float(np.linalg.norm(tangent)) < 1e-6:
            tangent = np.asarray([0.0, 0.0, 1.0])
        tangent = tangent / max(float(np.linalg.norm(tangent)), 1e-6)
        normal = np.cross(tangent, np.asarray([0.0, 1.0, 0.0]))
        if float(np.linalg.norm(normal)) < 1e-6:
            normal = np.asarray([1.0, 0.0, 0.0])
        normal = normal / max(float(np.linalg.norm(normal)), 1e-6)
        binormal = np.cross(tangent, normal)
        radius = float(radii[min(idx, len(radii) - 1)])
        for j in range(segments):
            phi = j / segments * np.pi * 2
            wobble = 1.0 + 0.07 * np.sin(phi * 3 + idx)
            pos = point + normal * np.cos(phi) * radius * wobble + binormal * np.sin(phi) * radius * flatten_y
            vertices.append([float(pos[0]), float(pos[1]), float(pos[2])])
    for idx in range(len(pts) - 1):
        for j in range(segments):
            a = idx * segments + j
            b = idx * segments + (j + 1) % segments
            c = (idx + 1) * segments + (j + 1) % segments
            d = (idx + 1) * segments + j
            faces.append([a, b, c])
            faces.append([a, c, d])
    return _closed_authored_mesh(trimesh.Trimesh(vertices=np.asarray(vertices, dtype=float), faces=np.asarray(faces, dtype=int), process=True))


def _authored_plate(points: list[tuple[float, float, float]], thickness: float = 0.16) -> trimesh.Trimesh:
    front = np.asarray(points, dtype=float)
    back = front.copy()
    back[:, 1] += thickness
    vertices = np.vstack([front, back])
    n = len(points)
    faces: list[list[int]] = []
    for i in range(1, n - 1):
        faces.append([0, i, i + 1])
        faces.append([n, n + i + 1, n + i])
    for i in range(n):
        j = (i + 1) % n
        faces.append([i, j, n + j])
        faces.append([i, n + j, n + i])
    return _closed_authored_mesh(trimesh.Trimesh(vertices=vertices, faces=np.asarray(faces, dtype=int), process=True))


def _authored_round_base(radius: float, height: float, *, segments: int = 96) -> trimesh.Trimesh:
    vertices: list[list[float]] = []
    top_center = [0.0, 0.0, height]
    bottom_center = [0.0, 0.0, 0.0]
    vertices.extend([top_center, bottom_center])
    for z in (height, 0.0):
        for i in range(segments):
            angle = i / segments * np.pi * 2
            edge_radius = radius * (1.0 + 0.018 * np.sin(angle * 9.0))
            vertices.append([float(np.cos(angle) * edge_radius), float(np.sin(angle) * edge_radius), float(z)])
    faces: list[list[int]] = []
    top_offset = 2
    bottom_offset = 2 + segments
    for i in range(segments):
        j = (i + 1) % segments
        faces.append([0, top_offset + i, top_offset + j])
        faces.append([1, bottom_offset + j, bottom_offset + i])
        faces.append([top_offset + i, bottom_offset + i, bottom_offset + j])
        faces.append([top_offset + i, bottom_offset + j, top_offset + j])
    return _closed_authored_mesh(trimesh.Trimesh(vertices=np.asarray(vertices, dtype=float), faces=np.asarray(faces, dtype=int), process=True))


def _closed_authored_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    try:
        closed = mesh.convex_hull
        if isinstance(closed, trimesh.Trimesh) and len(closed.faces) > 0:
            return closed
    except Exception:
        pass
    mesh.process(validate=True)
    return mesh


def _components_from_selected(selected: dict[PartCategory, ModularMiniaturePart]) -> list[str]:
    components = ["body", "head", "left_arm", "right_arm", "left_leg", "right_leg", "weapon"]
    for part in selected.values():
        components.extend(part.detail_tags)
        components.append(part.category.value)
    return components


def _selected_with_legacy_aliases(selected: dict[PartCategory, ModularMiniaturePart]) -> dict[PartCategory, ModularMiniaturePart]:
    result = dict(selected)
    for alias, canonical in LEGACY_CATEGORY_ALIASES.items():
        if canonical in selected:
            result.setdefault(alias, selected[canonical])
    return result


def _export_alias_bundle(part: ModularMiniaturePart, alias: PartCategory, output_dir: str | Path) -> None:
    directory = Path(output_dir) / alias.value / part.part_id
    directory.mkdir(parents=True, exist_ok=True)
    mesh_path = directory / f"{part.part_id}.stl"
    save_mesh(part.mesh, mesh_path)
    metadata = part.metadata() | {"category_alias": alias.value, "canonical_category": part.category.value}
    (directory / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (directory / "cleanup_report.json").write_text(json.dumps((part.cleanup_report.to_dict() if part.cleanup_report else {}), indent=2), encoding="utf-8")
    from meshmend.studio.assets import render_preview_svg

    (directory / "preview.svg").write_text(render_preview_svg(part.mesh, title=f"{alias.value}:{part.part_id}"), encoding="utf-8")


def _assembly_connectors() -> list[trimesh.Trimesh]:
    """Physical socket connectors that make selected modules one printable body."""
    return [
        _capsule((0.0, 0.0, 20.25), (0.0, -0.04, 22.15), 0.62, 18),
        _capsule((-2.55, -0.05, 18.0), (-3.55, -0.14, 18.25), 0.55, 18),
        _capsule((2.55, -0.05, 18.0), (3.55, -0.14, 18.25), 0.55, 18),
        _capsule((-1.35, 0.0, 10.0), (-0.9, 0.0, 12.0), 0.62, 18),
        _capsule((1.35, 0.0, 10.0), (0.9, 0.0, 12.0), 0.62, 18),
        _capsule((-1.35, 0.0, 1.0), (-1.35, 0.0, 2.8), 0.58, 18),
        _capsule((1.35, 0.0, 1.0), (1.35, 0.0, 2.8), 0.58, 18),
        _capsule((0.0, 1.35, 16.0), (0.0, 2.25, 16.4), 0.52, 18),
        _capsule((4.2, -1.3, 12.3), (2.0, -2.15, 14.1), 0.28, 14),
    ]


def _authored_assembly_connectors() -> list[trimesh.Trimesh]:
    """Non-primitive printable connectors for archetype-authored base forms."""
    return [
        _authored_limb([(0.0, 0.0, 20.25), (0.0, -0.04, 22.15)], [0.46, 0.38], segments=16, flatten_y=0.72),
        _authored_limb([(-2.20, -0.05, 18.0), (-3.25, -0.14, 18.25)], [0.36, 0.30], segments=16, flatten_y=0.72),
        _authored_limb([(2.20, -0.05, 18.0), (3.25, -0.14, 18.25)], [0.36, 0.30], segments=16, flatten_y=0.72),
        _authored_limb([(-1.08, 0.0, 10.0), (-0.90, 0.0, 12.0)], [0.40, 0.34], segments=16, flatten_y=0.72),
        _authored_limb([(1.08, 0.0, 10.0), (0.90, 0.0, 12.0)], [0.40, 0.34], segments=16, flatten_y=0.72),
        _authored_limb([(-1.35, 0.0, 1.0), (-1.35, 0.0, 2.8)], [0.34, 0.30], segments=16, flatten_y=0.72),
        _authored_limb([(1.35, 0.0, 1.0), (1.35, 0.0, 2.8)], [0.34, 0.30], segments=16, flatten_y=0.72),
        _authored_limb([(0.0, 1.35, 16.0), (0.0, 2.25, 16.4)], [0.36, 0.30], segments=16, flatten_y=0.72),
        _authored_limb([(4.2, -1.3, 12.3), (3.2, -1.7, 13.2), (2.0, -2.15, 14.1)], [0.24, 0.22, 0.20], segments=14, flatten_y=0.70),
    ]


def _ellipsoid(center: tuple[float, float, float], radii: tuple[float, float, float], subdivisions: int) -> trimesh.Trimesh:
    mesh = trimesh.creation.icosphere(subdivisions=max(1, subdivisions), radius=1.0)
    mesh.apply_scale(radii)
    mesh.apply_translation(center)
    return mesh


def _box(extents: tuple[float, float, float], center: tuple[float, float, float]) -> trimesh.Trimesh:
    mesh = trimesh.creation.box(extents=extents)
    mesh.apply_translation(center)
    return mesh


def _rivet(center: tuple[float, float, float], radius: float) -> trimesh.Trimesh:
    mesh = trimesh.creation.uv_sphere(radius=radius, count=[12, 6])
    mesh.apply_scale([1.0, 0.55, 1.0])
    mesh.apply_translation(center)
    return mesh


def _capsule(start: tuple[float, float, float], end: tuple[float, float, float], radius: float, sections: int) -> trimesh.Trimesh:
    start_arr = np.asarray(start, dtype=float)
    end_arr = np.asarray(end, dtype=float)
    cylinder = _cylinder(start, end, radius, sections)
    a = trimesh.creation.uv_sphere(radius=radius, count=[sections, max(8, sections // 2)])
    b = a.copy()
    a.apply_translation(start_arr)
    b.apply_translation(end_arr)
    return trimesh.util.concatenate([cylinder, a, b])


def _cylinder(start: tuple[float, float, float], end: tuple[float, float, float], radius: float, sections: int) -> trimesh.Trimesh:
    return trimesh.creation.cylinder(radius=radius, sections=sections, segment=np.vstack([np.asarray(start, dtype=float), np.asarray(end, dtype=float)]))


def _tapered_limb(start: tuple[float, float, float], end: tuple[float, float, float], start_radius: float, end_radius: float) -> trimesh.Trimesh:
    start_arr = np.asarray(start, dtype=float)
    end_arr = np.asarray(end, dtype=float)
    mid = (start_arr + end_arr) * 0.5
    length = max(float(np.linalg.norm(end_arr - start_arr)), 1e-6)
    limb = trimesh.creation.capsule(radius=max(float(start_radius), float(end_radius)) * 0.72, height=length, count=[24, 12])
    limb.apply_scale([max(float(start_radius), 1e-3) / max(float(end_radius), 1e-3), 1.0, 1.0])
    direction = end_arr - start_arr
    transform = trimesh.geometry.align_vectors([0, 0, 1], direction)
    limb.apply_transform(transform)
    limb.apply_translation(mid)
    return limb


def _oriented_cone(start: tuple[float, float, float], end: tuple[float, float, float], radius: float, sections: int) -> trimesh.Trimesh:
    start_arr = np.asarray(start, dtype=float)
    end_arr = np.asarray(end, dtype=float)
    direction = end_arr - start_arr
    length = max(float(np.linalg.norm(direction)), 1e-6)
    cone = trimesh.creation.cone(radius=float(radius), height=length, sections=sections)
    cone.apply_transform(trimesh.geometry.align_vectors([0, 0, 1], direction))
    cone.apply_translation((start_arr + end_arr) * 0.5)
    return cone


def _solid_fuse_components(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Fuse overlapping kitbash shells into one printable solid.

    Boolean backends are not guaranteed in a local/offline install, so use a
    conservative voxel union. This is intentionally in the printability stage,
    after part selection/detailing, so random sheets and bad parts have already
    been rejected and only validated miniature modules are fused.
    """
    components = [part for part in mesh.split(only_watertight=False) if len(part.faces) > 20]
    if len(components) <= 3:
        return mesh
    try:
        extents = np.maximum(np.asarray(mesh.extents, dtype=float), 1e-6)
        pitch = max(0.20, min(0.38, float(extents.max()) / 96.0))
        voxels = mesh.voxelized(pitch).fill()
        fused = voxels.marching_cubes
        try:
            from scipy.ndimage import binary_closing, binary_fill_holes
            from trimesh.voxel import ops as voxel_ops

            matrix = np.asarray(voxels.matrix, dtype=bool)
            matrix = binary_closing(matrix, iterations=2)
            matrix = binary_fill_holes(matrix)
            closed = voxel_ops.matrix_to_marching_cubes(matrix=np.pad(matrix, 2, constant_values=False), pitch=pitch)
            if isinstance(closed, trimesh.Trimesh) and len(closed.faces) >= len(fused.faces) * 0.45:
                fused = closed
        except Exception:
            pass
        if not isinstance(fused, trimesh.Trimesh) or len(fused.faces) < 1000:
            return mesh
        fused.metadata.update(mesh.metadata)
        try:
            fused.remove_duplicate_faces()
            fused.remove_degenerate_faces()
            fused.remove_unreferenced_vertices()
            fused.merge_vertices()
            fused.fix_normals()
            fused.fill_holes()
        except Exception:
            pass
        fused.metadata["studio_solid_fused"] = True
        fused.metadata["studio_pre_fusion_components"] = len(components)
        fused.metadata["studio_fusion_pitch_mm"] = pitch
        return fused
    except Exception:
        return mesh


def _silhouette_thumbnail_metrics(mesh: trimesh.Trimesh, *, resolution: int = 64, view: str = "front") -> dict[str, Any]:
    """Project mesh vertices into a tiny thumbnail and report readability metrics."""
    vertices = np.asarray(mesh.vertices, dtype=float)
    if len(vertices) == 0:
        return {"view": view, "resolution": resolution, "occupancy": 0.0, "width_span": 0.0, "height_span": 0.0, "edge_transitions": 0}
    if view == "front":
        coords = vertices[:, [0, 2]]
    elif view == "rear":
        coords = np.column_stack([-vertices[:, 0], vertices[:, 2]])
    elif view == "left":
        coords = vertices[:, [1, 2]]
    elif view == "right":
        coords = np.column_stack([-vertices[:, 1], vertices[:, 2]])
    elif view in {"45", "three_quarter", "three_quarter_45"}:
        angle = np.radians(45.0)
        coords = np.column_stack([vertices[:, 0] * np.cos(angle) - vertices[:, 1] * np.sin(angle), vertices[:, 2]])
    else:
        coords = vertices[:, [0, 2]]
    mins = coords.min(axis=0)
    span = np.maximum(np.ptp(coords, axis=0), 1e-6)
    normalized = np.clip((coords - mins) / span, 0.0, 1.0)
    pixels = np.clip((normalized * (resolution - 1)).astype(int), 0, resolution - 1)
    grid = np.zeros((resolution, resolution), dtype=bool)
    grid[pixels[:, 1], pixels[:, 0]] = True
    rows = np.where(grid.any(axis=1))[0]
    cols = np.where(grid.any(axis=0))[0]
    edge_transitions = int(np.abs(np.diff(grid.astype(np.int8), axis=0)).sum() + np.abs(np.diff(grid.astype(np.int8), axis=1)).sum())
    return {
        "view": view,
        "resolution": int(resolution),
        "occupancy": round(float(grid.mean()), 4),
        "width_span": round(float((cols.max() - cols.min() + 1) / resolution), 4) if len(cols) else 0.0,
        "height_span": round(float((rows.max() - rows.min() + 1) / resolution), 4) if len(rows) else 0.0,
        "edge_transitions": edge_transitions,
    }


def render_black_silhouette_previews(mesh: trimesh.Trimesh) -> dict[str, str]:
    """Return front/side/rear/45-degree black SVG silhouette previews before sculpting."""
    return {view: _black_silhouette_svg(mesh, view=view, title=f"pre-sculpt {view} silhouette") for view in ("front", "side", "rear", "45")}


def write_black_silhouette_previews(mesh: trimesh.Trimesh, output_dir: str | Path, *, prefix: str) -> dict[str, str]:
    """Write pre-sculpt silhouette SVGs and return the paths by view."""
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}
    for view, svg in render_black_silhouette_previews(mesh).items():
        path = root / f"{prefix}_{view}.svg"
        path.write_text(svg, encoding="utf-8")
        written[view] = str(path)
    return written


def silhouette_similarity_signature(mesh: trimesh.Trimesh) -> tuple[float, ...]:
    """Compact signature used by regression tests to catch identical mannequin bodies."""
    metrics = [_silhouette_thumbnail_metrics(mesh, view=view) for view in ("front", "left", "rear", "45")]
    values: list[float] = []
    for metric in metrics:
        values.extend([float(metric["occupancy"]), float(metric["width_span"]), float(metric["height_span"]), float(metric["edge_transitions"]) / 1000.0])
    extents = np.maximum(np.asarray(mesh.extents, dtype=float), 1e-6)
    values.extend([float(extents[0] / extents[2]), float(extents[1] / extents[2])])
    return tuple(round(value, 4) for value in values)


def silhouette_similarity_ratio(first: trimesh.Trimesh, second: trimesh.Trimesh) -> float:
    """Return 0..1 silhouette similarity; archetype separation fails above 0.40."""
    similarities: list[float] = []
    for view in ("front", "side", "rear", "45"):
        a = _silhouette_occupancy_mask(first, view=view, resolution=96, ignore_display_base=True)
        b = _silhouette_occupancy_mask(second, view=view, resolution=96, ignore_display_base=True)
        intersection = int(np.logical_and(a, b).sum())
        union = int(np.logical_or(a, b).sum())
        iou = float(intersection / max(union, 1))
        metrics_a = _mask_metrics(a)
        metrics_b = _mask_metrics(b)
        occupancy_gap = abs(float(metrics_a["occupancy"]) - float(metrics_b["occupancy"]))
        edge_gap = abs(float(metrics_a["edge_transitions"]) - float(metrics_b["edge_transitions"])) / 1000.0
        aspect_gap = abs(_projected_character_aspect(first, view) - _projected_character_aspect(second, view))
        similarities.append(max(0.0, iou - occupancy_gap * 3.0 - edge_gap * 0.8 - aspect_gap * 2.00))
    return round(float(sum(similarities) / max(len(similarities), 1)), 4)


def _mask_metrics(mask: np.ndarray) -> dict[str, float]:
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    edge_transitions = int(np.abs(np.diff(mask.astype(np.int8), axis=0)).sum() + np.abs(np.diff(mask.astype(np.int8), axis=1)).sum())
    resolution = max(int(mask.shape[0]), 1)
    return {
        "occupancy": float(mask.mean()),
        "width_span": float((cols.max() - cols.min() + 1) / resolution) if len(cols) else 0.0,
        "height_span": float((rows.max() - rows.min() + 1) / resolution) if len(rows) else 0.0,
        "edge_transitions": float(edge_transitions),
    }


def _projected_character_aspect(mesh: trimesh.Trimesh, view: str) -> float:
    vertices = np.asarray(mesh.vertices, dtype=float)
    if len(vertices) == 0:
        return 0.0
    z_min = float(vertices[:, 2].min())
    vertices = vertices[vertices[:, 2] > z_min + 2.15]
    if len(vertices) == 0:
        return 0.0
    coords = _project_silhouette_vertices(vertices, view)
    span = np.maximum(np.ptp(coords, axis=0), 1e-6)
    return float(span[0] / span[1])


def _silhouette_occupancy_mask(mesh: trimesh.Trimesh, *, view: str, resolution: int = 96, ignore_display_base: bool = False) -> np.ndarray:
    vertices = np.asarray(mesh.vertices, dtype=float)
    mask = np.zeros((resolution, resolution), dtype=bool)
    if len(vertices) == 0:
        return mask
    if ignore_display_base:
        z_min = float(vertices[:, 2].min())
        vertices = vertices[vertices[:, 2] > z_min + 2.15]
        if len(vertices) == 0:
            return mask
    coords = _project_silhouette_vertices(vertices, view)
    center = (coords.min(axis=0) + coords.max(axis=0)) * 0.5
    scale = max(float(np.ptp(coords[:, 0])), float(np.ptp(coords[:, 1])), 1e-6)
    normalized = np.clip(((coords - center) / scale) + 0.5, 0.0, 1.0)
    indices = np.clip((normalized * (resolution - 1)).astype(int), 0, resolution - 1)
    mask[indices[:, 1], indices[:, 0]] = True
    for _ in range(1):
        grown = mask.copy()
        grown[:-1, :] |= mask[1:, :]
        grown[1:, :] |= mask[:-1, :]
        grown[:, :-1] |= mask[:, 1:]
        grown[:, 1:] |= mask[:, :-1]
        grown[:-1, :-1] |= mask[1:, 1:]
        grown[1:, 1:] |= mask[:-1, :-1]
        grown[:-1, 1:] |= mask[1:, :-1]
        grown[1:, :-1] |= mask[:-1, 1:]
        mask = grown
    return mask


def _black_silhouette_svg(mesh: trimesh.Trimesh, *, view: str, title: str) -> str:
    vertices = np.asarray(mesh.vertices, dtype=float)
    if len(vertices) == 0:
        return "<svg xmlns='http://www.w3.org/2000/svg' width='320' height='320'></svg>"
    coords = _project_silhouette_vertices(vertices, view)
    mins = coords.min(axis=0)
    span = np.maximum(np.ptp(coords, axis=0), 1e-6)
    points = (coords - mins) / span
    points[:, 0] = points[:, 0] * 260 + 30
    points[:, 1] = 290 - points[:, 1] * 260
    hull = _convex_hull_2d([(float(x), float(y)) for x, y in points[:: max(1, len(points) // 1200)]])
    polygon = " ".join(f"{x:.1f},{y:.1f}" for x, y in hull)
    return (
        "<svg xmlns='http://www.w3.org/2000/svg' width='320' height='320' viewBox='0 0 320 320'>"
        "<rect width='320' height='320' fill='white'/>"
        f"<text x='12' y='22' font-size='13' font-family='monospace' fill='#222'>{title}</text>"
        f"<polygon points='{polygon}' fill='black' stroke='black' stroke-width='1'/></svg>"
    )


def _project_silhouette_vertices(vertices: np.ndarray, view: str) -> np.ndarray:
    if view == "front":
        return vertices[:, [0, 2]]
    if view == "rear":
        return np.column_stack([-vertices[:, 0], vertices[:, 2]])
    if view in {"side", "left"}:
        return vertices[:, [1, 2]]
    if view == "right":
        return np.column_stack([-vertices[:, 1], vertices[:, 2]])
    if view in {"45", "three_quarter", "three_quarter_45"}:
        angle = np.radians(45.0)
        return np.column_stack([vertices[:, 0] * np.cos(angle) - vertices[:, 1] * np.sin(angle), vertices[:, 2]])
    return vertices[:, [0, 2]]


def _convex_hull_2d(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    unique = sorted(set(points))
    if len(unique) <= 1:
        return unique

    def cross(origin: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
        return (a[0] - origin[0]) * (b[1] - origin[1]) - (a[1] - origin[1]) * (b[0] - origin[0])

    lower: list[tuple[float, float]] = []
    for point in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper: list[tuple[float, float]] = []
    for point in reversed(unique):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return lower[:-1] + upper[:-1]


def _silhouette_complexity_score(view_metrics: dict[str, dict[str, Any]]) -> float:
    if not view_metrics:
        return 0.0
    scores: list[float] = []
    for metric in view_metrics.values():
        occupancy = float(metric.get("occupancy") or 0.0)
        width_span = float(metric.get("width_span") or 0.0)
        height_span = float(metric.get("height_span") or 0.0)
        transitions = float(metric.get("edge_transitions") or 0.0)
        normalized_edges = min(1.0, transitions / 420.0)
        readable_span = min(width_span, 1.0) * min(height_span, 1.0)
        occupancy_balance = 1.0 - min(1.0, abs(occupancy - 0.11) / 0.11)
        scores.append(max(0.0, 0.50 * normalized_edges + 0.30 * readable_span + 0.20 * occupancy_balance))
    return float(sum(scores) / max(1, len(scores)))
