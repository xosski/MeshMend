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
        prompt_l = prompt.lower()
        scale_mm = self._resolve_scale_mm(prompt_l, scale_mm)
        image_tags = image_tags or []

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
                "minimum_thickness_mm": 0.8,
                "merge_into_single_piece": True,
                "avoid_thin_floating_parts": True,
                "exaggerate_weapon_thickness": True,
                "heroic_scale": True,
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
