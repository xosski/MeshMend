"""
LLM-style miniature part planner.

This does not require an online LLM yet.
It creates a structured sculpt plan from prompt + optional image tags.
Later you can swap the rule planner with Ollama, LM Studio, OpenAI, etc.
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Any
import json
import re


@dataclass
class PartSpec:
    name: str
    kind: str
    shape: str
    size: str
    position: list[float]
    scale: list[float]
    detail_tags: list[str] = field(default_factory=list)


@dataclass
class MiniaturePlan:
    subject: str
    style: str
    scale_mm: float
    parts: list[PartSpec]
    base: dict[str, Any]
    print_rules: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


class LLMPartPlanner:
    """
    First-stage 'LLM brain' for converting vague prompts into concrete model parts.
    """

    def create_plan(
        self,
        prompt: str,
        style: str = "Generic Humanoid",
        scale_mm: float = 28.0,
        image_tags: list[str] | None = None,
    ) -> MiniaturePlan:
        original_prompt = self._extract_original_prompt(prompt)
        prompt_l = prompt.lower()
        subject_l = original_prompt.lower()
        scale_mm = self._resolve_scale_mm(prompt_l, scale_mm)
        image_tags = image_tags or []
        intent = self._detect_generation_intent(subject_l)

        if intent != "character_miniature":
            return self._object_plan(
                original_prompt=original_prompt,
                prompt_l=prompt_l,
                subject_l=subject_l,
                intent=intent,
                style=style,
                scale_mm=scale_mm,
            )

        if any(w in prompt_l for w in ["detailed", "high detail", "highly detailed", "intricate", "ornate", "8k", "display quality"]):
            style = f"{style} highly detailed ornate display quality"
        if any(w in prompt_l for w in ["wargaming", "tabletop", "miniature", "heroic scale", "resin-printable", "resin printable"]):
            style = f"{style} wargaming tabletop miniature heroic scale resin-printable"
        if any(w in prompt_l for w in ["warhammer", "40k", "grimdark", "space marine", "power armor", "power armour", "bolter"]):
            style = f"{style} grimdark heroic power-armored wargaming warrior"
        if any(w in prompt_l for w in ["heavy", "bulky", "juggernaut", "brutal", "terminator"]):
            style = f"{style} heavy bulky juggernaut stance"
        if any(w in prompt_l for w in ["agile", "scout", "sniper", "slender", "recon"]):
            style = f"{style} agile scout slender recon stance"
        if any(w in prompt_l for w in ["dynamic", "charging", "running", "lunging", "action pose"]):
            style = f"{style} dynamic action pose"
        if any(w in prompt_l for w in ["wizard", "mage", "sorcerer", "robe", "staff"]):
            style = f"{style} robed fantasy spellcaster"
        if any(w in prompt_l for w in ["knight", "paladin", "shield", "sword"]):
            style = f"{style} armored fantasy knight"
        if any(w in prompt_l for w in ["robot", "mech", "android", "cyborg", "war machine", "wide squat"]):
            style = f"{style} mechanical war-machine"
        if any(w in prompt_l for w in ["undead", "skeleton", "zombie", "necromancer", "skull"]):
            style = f"{style} undead skeletal horror"
        if any(w in prompt_l for w in ["demon", "daemon", "wing", "wings", "horned"]):
            style = f"{style} horned winged demon"
        if any(w in prompt_l for w in ["archer", "ranger", "bow", "crossbow"]):
            style = f"{style} ranged archer ranger"

        subject = self._detect_subject(prompt_l, style)
        parts = self._base_humanoid_parts(prompt_l)

        if any(w in prompt_l for w in ["orc", "ork", "brute", "warboss"]):
            parts += self._orc_parts(prompt_l)

        if "axe" in prompt_l:
            gear_axe = any(w in prompt_l for w in ["gear axe", "gear", "gears", "clockwork", "watch"])
            parts.append(
                PartSpec(
                    name="two_handed_gear_axe" if gear_axe else "two_handed_brutal_axe",
                    kind="weapon",
                    shape="long shaft with circular gear axe head" if gear_axe else "long haft with chipped crescent axe blade",
                    size="large",
                    position=[0, -2.8, 15],
                    scale=[2.2, 0.6, 9.0],
                    detail_tags=["gear teeth", "clockwork rings", "chipped blade edge"] if gear_axe else ["chipped blade edge", "leather haft wrap", "iron rivets"],
                )
            )

        if any(w in prompt_l for w in ["club", "great club", "mace"]):
            parts.append(
                PartSpec(
                    name="chain_wrapped_great_club",
                    kind="weapon",
                    shape="heavy cylinder club",
                    size="large",
                    position=[-5, 0, 14],
                    scale=[1.5, 1.5, 9],
                    detail_tags=["wrapped chains", "spikes", "rusted iron"],
                )
            )

        if any(w in prompt_l for w in ["claw", "bear trap", "trap"]):
            parts.append(
                PartSpec(
                    name="bear_trap_claw",
                    kind="weapon_arm",
                    shape="crescent mechanical jaw",
                    size="large",
                    position=[5.5, 0, 12],
                    scale=[2.8, 1.2, 2.8],
                    detail_tags=["spring hinge", "jagged trap teeth", "bolts"],
                )
            )

        if any(w in prompt_l for w in ["turret", "auto turret", "gun", "cannon"]):
            parts.append(
                PartSpec(
                    name="shoulder_auto_turret",
                    kind="mounted_weapon",
                    shape="short rotating barrel on mechanical arm",
                    size="medium",
                    position=[2.5, 0, 23],
                    scale=[2.2, 1.3, 2.0],
                    detail_tags=["barrel holes", "ammo drum", "clockwork mount"],
                )
            )

        if any(w in prompt_l for w in ["warhammer", "40k", "grimdark", "space marine", "power armor", "power armour", "bolter"]):
            parts += self._grimdark_power_armor_parts(prompt_l)

        if any(w in prompt_l for w in ["wizard", "mage", "sorcerer", "robe", "staff"]):
            parts += self._wizard_parts(prompt_l)

        if any(w in prompt_l for w in ["knight", "paladin", "shield", "sword"]):
            parts += self._knight_parts(prompt_l)

        if any(w in prompt_l for w in ["robot", "mech", "android", "cyborg", "war machine", "wide squat"]):
            parts += self._robot_parts(prompt_l)

        if any(w in prompt_l for w in ["undead", "skeleton", "zombie", "necromancer", "skull"]):
            parts += self._undead_parts(prompt_l)

        if any(w in prompt_l for w in ["demon", "daemon", "wing", "wings", "horned"]):
            parts += self._demon_parts(prompt_l)

        if any(w in prompt_l for w in ["archer", "ranger", "bow", "crossbow"]):
            parts += self._archer_parts(prompt_l)

        parts += self._signature_prompt_parts(prompt_l)

        if any(w in prompt_l for w in ["clockwork", "watch", "gear", "gears"]):
            parts += self._clockwork_armor_parts()

        if any(w in prompt_l for w in ["beast", "lizard", "bear", "quadruped"]):
            parts = self._beast_parts(prompt_l)

        return MiniaturePlan(
            subject=subject,
            style=style,
            scale_mm=scale_mm,
            parts=parts,
            base={
                "enabled": True,
                "shape": "round",
                "diameter_mm": max(32, scale_mm * 1.25),
                "height_mm": 2.2,
                "details": ["rocks", "dirt", "broken gears", "small skulls"],
            },
            print_rules={
                "generation_intent": "character_miniature",
                "minimum_thickness_mm": 0.8,
                "merge_into_single_piece": True,
                "avoid_thin_floating_parts": True,
                "exaggerate_weapon_thickness": True,
                "heroic_scale": True,
                "supported_scales_mm": [15, 20, 25, 28, 32, 35, 40, 48, 54, 75],
            },
        )

    @staticmethod
    def _extract_original_prompt(prompt: str) -> str:
        text = (prompt or "").strip()
        patterns = [
            r"Original user prompt, preserve exact subject and unique requested traits:\s*(.*?)(?:\.\s*Enhanced sculpt cues:|$)",
            r"Original user prompt, preserve exact requested subject and traits:\s*(.*?)(?:\.\s*Enhanced cues:|$)",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.I | re.S)
            if match:
                candidate = match.group(1).strip()
                if candidate:
                    return candidate
        return text

    @staticmethod
    def _detect_generation_intent(prompt: str) -> str:
        lower = (prompt or "").lower()
        tokens = set(re.findall(r"[a-z0-9']+", lower))
        if any(term in lower for term in ("full body", "full-body", "whole character", "entire character")):
            return "character_miniature"
        if tokens & {"mask", "helmet", "helm", "headpiece", "faceplate"}:
            return "wearable_object"
        if tokens & {"rifle", "gun", "pistol", "sword", "axe", "hammer", "shield", "banner", "weapon", "prop", "accessory"}:
            return "prop_object"
        if tokens & {"bust", "portrait"} or "head bust" in lower:
            return "bust"
        if tokens & {"terrain", "scenery", "building", "ruin", "dungeon", "objective"}:
            return "terrain_object"
        if tokens & {"vehicle", "tank", "ship", "walker", "turret"}:
            return "vehicle_object"
        character_terms = {
            "miniature", "figure", "character", "warrior", "soldier", "knight", "wizard", "mage",
            "orc", "ork", "elf", "dwarf", "demon", "daemon", "undead", "skeleton", "zombie",
            "ranger", "archer", "marine", "humanoid", "creature", "monster", "beast",
        }
        if tokens & character_terms:
            return "character_miniature"
        return "printable_subject"

    def _object_plan(
        self,
        *,
        original_prompt: str,
        prompt_l: str,
        subject_l: str,
        intent: str,
        style: str,
        scale_mm: float,
    ) -> MiniaturePlan:
        subject = original_prompt.strip() or "printable subject"
        parts: list[PartSpec] = []
        base_enabled = False

        if intent == "wearable_object":
            subject = subject if any(w in subject_l for w in ["mask", "helmet", "helm"]) else f"standalone mask/helmet inspired by {subject}"
            parts = [
                PartSpec("mask_shell", "object_core", "curved face shell", "large", [0, 0, 8], [5.0, 1.4, 7.0], ["raised brow", "cheek planes", "rim lip"]),
                PartSpec("eye_lens_pair", "detail", "deep round eye lenses", "medium", [0, -1.25, 9.2], [3.2, 0.35, 0.9], ["recessed lenses", "thick rims"]),
                PartSpec("side_straps", "detail", "side leather or metal straps", "medium", [0, 0.55, 7.8], [5.8, 0.25, 4.8], ["buckles", "rivets"]),
                PartSpec("vent_filter_cluster", "detail", "front vent filter cluster", "medium", [0, -1.55, 6.5], [2.4, 0.45, 1.3], ["grille slots", "small bolts"]),
            ]
            if any(w in subject_l for w in ["plague", "doctor", "beak", "bird", "raven"]):
                parts.append(PartSpec("long_plague_beak", "detail", "long tapered bird beak mask nose", "large", [0, -3.0, 7.7], [1.25, 4.8, 1.35], ["center seam", "rivet rows", "hooked tip"]))
            if any(w in subject_l for w in ["dragon", "demon", "horn", "horned"]):
                parts.append(PartSpec("helmet_horns", "detail", "swept horns", "large", [0, -0.1, 11.4], [4.2, 0.75, 2.1], ["horn ridges", "sharp tips"]))
        elif intent == "prop_object":
            subject = f"standalone {subject} prop"
            if any(w in subject_l for w in ["rifle", "gun", "pistol", "cannon", "bolter"]):
                parts = [
                    PartSpec("receiver_body", "object_core", "chunky weapon receiver", "large", [0, 0, 5], [7.0, 1.4, 1.8], ["panel seams", "screws"]),
                    PartSpec("long_barrel", "detail", "long vented barrel", "large", [5.4, 0, 5.15], [5.6, 0.48, 0.48], ["muzzle brake", "barrel vents"]),
                    PartSpec("stock", "detail", "angular shoulder stock", "medium", [-5.0, 0, 4.9], [3.2, 1.0, 1.6], ["butt plate", "ribs"]),
                    PartSpec("magazine", "detail", "box magazine", "medium", [0.2, -0.2, 3.5], [1.2, 0.9, 2.0], ["feed lips", "cartridge ridges"]),
                    PartSpec("scope", "detail", "top optic scope", "medium", [1.0, 0, 6.35], [2.4, 0.55, 0.55], ["lens rims", "mount rings"]),
                ]
            elif any(w in subject_l for w in ["sword", "blade", "katana"]):
                parts = [
                    PartSpec("blade", "object_core", "long tapered blade", "large", [0, 0, 6], [1.0, 0.22, 10.0], ["central ridge", "sharpened bevels"]),
                    PartSpec("crossguard", "detail", "ornate crossguard", "medium", [0, 0, 1.2], [4.2, 0.35, 0.45], ["quillons", "engraved bands"]),
                    PartSpec("grip", "detail", "wrapped grip", "medium", [0, 0, -1.3], [0.75, 0.55, 2.6], ["wrap spiral", "pommel"]),
                ]
            else:
                parts = [
                    PartSpec("prop_core", "object_core", "standalone detailed prop silhouette", "large", [0, 0, 5], [6.0, 1.2, 3.2], ["raised trim", "panel seams", "rivet rows"]),
                    PartSpec("prop_detail_cluster", "detail", "secondary raised details", "medium", [0, -0.8, 5.6], [4.5, 0.35, 1.4], ["greebles", "straps", "bolts"]),
                ]
        elif intent == "bust":
            parts = [
                PartSpec("bust_head", "object_core", "large head and neck bust", "large", [0, 0, 9], [3.2, 2.6, 4.2], ["face planes", "deep eye sockets"]),
                PartSpec("bust_shoulders", "detail", "truncated shoulder bust", "large", [0, 0, 5], [6.0, 2.4, 3.0], ["collar", "cloth folds", "armor rim"]),
            ]
            base_enabled = True
        else:
            parts = [
                PartSpec("subject_core", "object_core", "standalone printable subject", "large", [0, 0, 5], [5.5, 2.2, 4.0], ["silhouette preserving mass", "raised surface details"]),
                PartSpec("subject_detail_layer", "detail", "prompt-specific surface details", "medium", [0, -1.15, 5.7], [4.6, 0.35, 2.0], ["panel seams", "texture ridges", "rivet rows"]),
            ]

        if any(w in prompt_l for w in ["detailed", "high detail", "highly detailed", "intricate", "ornate", "8k", "display quality"]):
            style = f"{style} high-detail standalone printable object"

        return MiniaturePlan(
            subject=subject,
            style=f"{style} {intent} preserve-user-requested-topology no-forced-humanoid",
            scale_mm=scale_mm,
            parts=parts,
            base={
                "enabled": base_enabled,
                "shape": "round",
                "diameter_mm": max(24, scale_mm * 0.95),
                "height_mm": 2.0,
                "details": ["subtle rim", "label plaque"],
            },
            print_rules={
                "generation_intent": intent,
                "minimum_thickness_mm": 0.8,
                "merge_into_single_piece": True,
                "avoid_thin_floating_parts": True,
                "preserve_requested_topology": True,
                "heroic_scale": False,
                "supported_scales_mm": [15, 20, 25, 28, 32, 35, 40, 48, 54, 75],
            },
        )

    def _resolve_scale_mm(self, prompt: str, requested_scale_mm: float) -> float:
        match = re.search(r"\b(15|20|25|28|30|32|35|40|48|54|75)\s*mm\b", prompt)
        if match:
            return float(match.group(1))
        return float(requested_scale_mm)

    def _detect_subject(self, prompt: str, style: str) -> str:
        if "beast" in prompt or "lizard" in prompt:
            return "mutated quadruped warbeast"
        if any(w in prompt for w in ["warhammer", "40k", "grimdark", "space marine", "power armor", "power armour", "bolter"]):
            return "grimdark power-armored wargaming warrior"
        if any(w in prompt for w in ["robot", "mech", "android", "cyborg", "war machine", "wide squat"]):
            return "mechanical wargaming war-machine"
        if any(w in prompt for w in ["wizard", "mage", "sorcerer", "robe", "staff"]):
            return "robed fantasy spellcaster"
        if any(w in prompt for w in ["knight", "paladin", "shield", "sword"]):
            return "armored fantasy knight"
        if any(w in prompt for w in ["undead", "skeleton", "zombie", "necromancer", "skull"]):
            return "undead skeletal warrior"
        if any(w in prompt for w in ["demon", "daemon", "wing", "wings", "horned"]):
            return "horned winged demon miniature"
        if any(w in prompt for w in ["archer", "ranger", "bow", "crossbow"]):
            return "ranged archer ranger miniature"
        if "warboss" in prompt:
            return "clockwork orc warboss"
        if "orc" in prompt or "ork" in prompt:
            return "orc brute"
        return style.lower()

    def _base_humanoid_parts(self, prompt: str) -> list[PartSpec]:
        return [
            PartSpec("torso", "body", "barrel chest", "large", [0, 0, 13], [4.5, 3.0, 6.5]),
            PartSpec("head", "body", "snarling head", "medium", [0, 0, 19], [2.5, 2.0, 2.8]),
            PartSpec("left_leg", "limb", "bent armored leg", "large", [-1.8, 0, 7], [1.6, 1.4, 6]),
            PartSpec("right_leg", "limb", "braced armored leg", "large", [1.8, 0, 7], [1.6, 1.4, 6]),
            PartSpec("left_arm", "limb", "muscular weapon arm", "large", [-4.2, 0, 14], [1.4, 1.2, 6]),
            PartSpec("right_arm", "limb", "muscular weapon arm", "large", [4.2, 0, 14], [1.4, 1.2, 6]),
        ]

    def _orc_parts(self, prompt: str) -> list[PartSpec]:
        return [
            PartSpec("large_tusks", "detail", "curved tusks", "small", [0, -1.8, 18.5], [0.4, 0.4, 1.4]),
            PartSpec("jaw", "detail", "oversized lower jaw", "medium", [0, -1.2, 18], [1.8, 0.9, 0.8]),
            PartSpec("ragged_cloth", "cloth", "torn loincloth", "medium", [0, -0.5, 9], [2.0, 0.35, 4.0]),
        ]

    def _clockwork_armor_parts(self) -> list[PartSpec]:
        return [
            PartSpec("chest_clock", "armor", "round watch face", "medium", [0, -2.2, 14], [2.4, 0.4, 2.4],
                     ["roman numerals", "clock hands", "brass rim"]),
            PartSpec("left_shoulder_watch_plate", "armor", "round clock shoulder pad", "medium", [-3.4, 0, 18], [2.2, 1.0, 2.2],
                     ["brass gears", "rivets"]),
            PartSpec("right_shoulder_gear_plate", "armor", "layered gear shoulder pad", "medium", [3.4, 0, 18], [2.2, 1.0, 2.2],
                     ["interlocking gears", "rivets"]),
            PartSpec("knee_gears", "armor", "gear knee plates", "small", [0, -1.5, 6], [1.2, 0.4, 1.2],
                     ["small cog teeth"]),
        ]

    def _grimdark_power_armor_parts(self, prompt: str) -> list[PartSpec]:
        return [
            PartSpec("oversized_gothic_pauldrons", "armor", "massive rounded shoulder armor with raised trim", "large", [0, 0, 17], [7.8, 2.4, 2.2],
                     ["heroic silhouette", "rivet rows", "chapter-badge-like blank emblem"]),
            PartSpec("ribbed_power_pack", "armor", "large rear power backpack with twin vents", "large", [0, 2.3, 15.5], [4.2, 1.4, 4.2],
                     ["exhaust vents", "cables", "service studs"]),
            PartSpec("heavy_plate_greaves", "armor", "oversized armored boots and shin plates", "large", [0, -0.9, 5.5], [5.0, 0.8, 4.0],
                     ["knee seals", "toe plates", "hard panel lines"]),
            PartSpec("boxy_bolt_rifle", "weapon", "chunky sci-fi rifle with box magazine and vented barrel", "large", [0, -2.8, 13.2], [7.5, 0.8, 1.4],
                     ["barrel vents", "box magazine", "scope", "muzzle brake"]),
            PartSpec("purity_scrolls_and_seals", "detail", "wax seals and hanging parchment strips", "small", [0, -2.1, 13.8], [2.8, 0.25, 2.0],
                     ["scroll text lines", "wax medallions", "deep recesses"]),
        ]

    def _wizard_parts(self, prompt: str) -> list[PartSpec]:
        return [
            PartSpec("flowing_robe_mass", "cloth", "long layered robe with deep folds", "large", [0, -0.25, 9.0], [4.5, 0.5, 7.2],
                     ["vertical robe folds", "torn hem", "belt cord"]),
            PartSpec("tall_wizard_hat", "detail", "tall pointed hat", "medium", [0, 0, 21.0], [1.7, 1.7, 3.8],
                     ["hat brim", "creased cone"]),
            PartSpec("spell_staff", "weapon", "long staff with crystal orb", "large", [-4.8, -0.8, 13.0], [0.5, 0.5, 10.0],
                     ["orb", "rings", "wrapped grip"]),
        ]

    def _knight_parts(self, prompt: str) -> list[PartSpec]:
        return [
            PartSpec("kite_shield", "armor", "large kite shield with raised rim", "large", [-4.7, -1.0, 12.2], [2.7, 0.45, 4.4],
                     ["shield boss", "blank heraldry", "rim rivets"]),
            PartSpec("long_sword", "weapon", "straight sword with crossguard", "large", [4.8, -1.0, 13.0], [0.45, 0.35, 8.0],
                     ["crossguard", "pommel", "central blade ridge"]),
            PartSpec("helmet_crest", "detail", "helmet crest plume", "medium", [0, -0.1, 20.0], [0.55, 0.35, 2.2],
                     ["plume ridges", "visor slit"]),
        ]

    def _robot_parts(self, prompt: str) -> list[PartSpec]:
        return [
            PartSpec("mechanical_chest_core", "armor", "angular robot torso with glowing core", "large", [0, -1.6, 13.6], [4.8, 0.55, 4.2],
                     ["core ring", "vents", "panel seams"]),
            PartSpec("piston_legs", "detail", "exposed hydraulic pistons", "medium", [0, -1.2, 6.5], [4.2, 0.3, 4.8],
                     ["cylinders", "cables", "knee actuators"]),
            PartSpec("arm_cannon", "weapon", "boxy forearm cannon", "large", [4.8, -1.8, 12.5], [3.6, 0.9, 1.4],
                     ["muzzle ports", "heat vents", "ammo feed"]),
        ]

    def _undead_parts(self, prompt: str) -> list[PartSpec]:
        return [
            PartSpec("exposed_ribcage", "detail", "visible rib cage bones", "medium", [0, -1.8, 13.4], [3.0, 0.18, 3.0],
                     ["ribs", "spine", "hollow chest"]),
            PartSpec("skull_face", "detail", "skull face with deep eye sockets", "small", [0, -1.55, 18.5], [1.4, 0.3, 1.2],
                     ["eye sockets", "teeth", "cheek bones"]),
            PartSpec("tattered_banner_cloth", "cloth", "tattered hanging burial cloth", "medium", [0, -1.8, 8.2], [3.2, 0.22, 3.8],
                     ["ragged strips", "holes", "frayed edges"]),
        ]

    def _demon_parts(self, prompt: str) -> list[PartSpec]:
        return [
            PartSpec("curved_horns", "detail", "large curved horns", "medium", [0, -0.5, 19.6], [3.2, 0.8, 1.6],
                     ["horn ridges", "sharp tips"]),
            PartSpec("bat_wings", "detail", "spread bat wings", "large", [0, 1.4, 14.5], [8.5, 0.35, 6.2],
                     ["wing membrane", "finger bones", "torn edges"]),
            PartSpec("spiked_tail", "detail", "long spiked tail", "medium", [0, 1.6, 8.0], [5.5, 0.45, 0.45],
                     ["barbed tail tip", "spines"]),
        ]

    def _archer_parts(self, prompt: str) -> list[PartSpec]:
        return [
            PartSpec("curved_bow", "weapon", "curved bow with string", "large", [-4.6, -1.0, 13.2], [0.45, 0.35, 7.2],
                     ["bow string", "wrapped grip", "limb tips"]),
            PartSpec("back_quiver", "detail", "quiver of arrows", "medium", [1.5, 1.7, 14.0], [1.0, 0.75, 3.6],
                     ["arrow fletching", "strap", "tube rim"]),
            PartSpec("hooded_cloak", "cloth", "hooded ranger cloak", "large", [0, 0.7, 12.0], [4.0, 0.35, 7.0],
                     ["hood", "cloth folds", "ragged hem"]),
        ]

    def _signature_prompt_parts(self, prompt: str) -> list[PartSpec]:
        """Large silhouette features for prompts that would otherwise collapse to a mannequin."""
        parts: list[PartSpec] = []

        def add(
            name: str,
            kind: str,
            shape: str,
            position: list[float],
            scale: list[float],
            tags: list[str] | None = None,
            size: str = "medium",
        ) -> None:
            parts.append(PartSpec(name, kind, shape, size, position, scale, tags or []))

        if any(w in prompt for w in ["samurai", "ronin", "katana", "ninja"]):
            add("samurai_helmet_crest", "detail", "tall helmet crest plume", [0, -0.2, 20.7], [0.7, 0.35, 2.4], ["layered neck guard", "crest ridges"])
            add("katana_scabbard", "weapon", "curved sword and scabbard", [3.9, -1.1, 11.8], [0.35, 0.25, 7.2], ["wrapped grip", "curved blade"])

        if any(w in prompt for w in ["pirate", "corsair", "buccaneer"]):
            add("tricorne_hat", "detail", "wide tricorne pirate hat", [0, -0.1, 20.1], [3.2, 1.2, 0.7], ["hat brim", "skull charm"])
            add("cutlass", "weapon", "curved cutlass sword", [4.6, -1.0, 12.6], [0.35, 0.25, 6.6], ["basket guard"])

        if any(w in prompt for w in ["plague doctor", "doctor", "beaked mask"]):
            add("plague_beak_mask", "detail", "long beaked mask", [0, -2.0, 18.3], [0.8, 1.6, 0.55], ["round eye lenses", "mask straps"])
            add("long_coat", "cloth", "long heavy coat with tails", [0, 0.4, 9.2], [4.2, 0.45, 7.2], ["coat folds", "buttons"])

        if any(w in prompt for w in ["mushroom", "fungal", "fungus", "myconid"]):
            add("mushroom_cap_head", "detail", "wide mushroom cap head", [0, 0, 20.0], [4.0, 3.0, 1.0], ["gill lines", "spots"])
            add("fungal_growths", "detail", "cluster of small mushrooms on shoulders", [0, -1.4, 15.5], [3.8, 0.35, 1.6], ["tiny caps", "stalks"])

        if any(w in prompt for w in ["ice", "frost", "crystal", "crystalline"]):
            add("crystal_back_spikes", "detail", "jagged crystal spikes on back", [0, 1.5, 15.0], [5.0, 0.6, 4.5], ["faceted crystal", "sharp silhouette"])
            add("ice_crown", "detail", "spiky ice crown", [0, -0.1, 20.2], [2.5, 0.5, 1.4], ["crystal points"])

        if any(w in prompt for w in ["fire", "flame", "burning", "inferno"]):
            add("flame_plume", "detail", "rising flame plume", [0, 0.2, 20.5], [2.4, 0.6, 2.6], ["licking flames"])
            add("burning_back_banner", "detail", "tattered flame banner", [0, 1.8, 15.5], [3.2, 0.3, 5.2], ["torn banner", "flame tongues"])

        if any(w in prompt for w in ["insect", "bug", "mantis", "beetle", "chitin"]):
            add("chitin_back_carapace", "armor", "large beetle-like back carapace", [0, 1.5, 14.0], [4.8, 1.2, 5.5], ["segmented plates"])
            add("mandibles", "detail", "large curved insect mandibles", [0, -1.9, 18.1], [2.3, 0.45, 1.2], ["sharp tips"])

        if any(w in prompt for w in ["viking", "barbarian", "raider"]):
            add("round_viking_shield", "armor", "large round shield with boss", [-4.7, -1.0, 12.0], [2.8, 0.45, 2.8], ["wood planks", "rim rivets"])
            add("fur_cloak", "cloth", "heavy fur cloak", [0, 1.0, 12.0], [5.2, 0.5, 6.5], ["fur tufts", "ragged hem"])

        if any(w in prompt for w in ["banner", "standard bearer", "flag"]):
            add("back_banner", "detail", "tall back banner pole and flag", [0, 1.8, 17.0], [3.2, 0.25, 7.0], ["tattered flag", "pole rings"], size="large")

        return parts

    def _beast_parts(self, prompt: str) -> list[PartSpec]:
        return [
            PartSpec("beast_body", "body", "low reptile-bear torso", "large", [0, 0, 8], [8, 3.2, 3.5],
                     ["bald wrinkled skin", "patches of coarse fur"]),
            PartSpec("beast_head", "body", "snarling bear-lizard head", "large", [-5, 0, 9], [3, 2.2, 2.4],
                     ["tusks", "flat snout", "small ears"]),
            PartSpec("front_left_bear_claw", "limb", "massive claw paw", "large", [-4, -2, 4], [1.4, 1.2, 3],
                     ["long hooked claws"]),
            PartSpec("front_right_bear_claw", "limb", "massive claw paw", "large", [-4, 2, 4], [1.4, 1.2, 3],
                     ["long hooked claws"]),
            PartSpec("rear_left_leg", "limb", "reptile hind leg", "medium", [4, -2, 4], [1.2, 1.1, 3]),
            PartSpec("rear_right_leg", "limb", "reptile hind leg", "medium", [4, 2, 4], [1.2, 1.1, 3]),
            PartSpec("tail", "body", "thick tapering reptile tail", "medium", [6, 0, 6], [5, 0.8, 0.8],
                     ["tail spikes"]),
            PartSpec("back_slug_turret", "mounted_weapon", "crude slug cannon turret", "medium", [0, 0, 12], [3.2, 1.4, 1.6],
                     ["ammo drum", "barrel holes", "bolted saddle"]),
        ]
