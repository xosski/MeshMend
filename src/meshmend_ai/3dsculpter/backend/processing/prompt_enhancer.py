"""Enhance prompts for miniature/printable 3D generation.

The older enhancer always rewrote every request into a full-body tabletop
miniature. That made object/reference prompts such as "plague doctor mask" or
"ornate sci-fi rifle" collapse into the same humanoid mannequin. Keep the
resin-print/detail guidance, but preserve the user's requested topology.
"""

import re

class MiniaturePromptEnhancer:
    """Enhance user prompts with miniature-specific constraints"""
    
    # Anatomy guidelines for standard proportions
    ANATOMY_HINTS = {
        "human": "anatomically correct human proportions, head 1/7 of body height",
        "dwarf": "stocky build, short legs, muscular frame, head 1/6 of height",
        "elf": "tall slender form, graceful posture, fine features, head 1/8 of height",
        "orc": "broad shoulders, powerful build, protruding jaw, head 1/7 of height",
        "dragon": "quadrupedal, muscular frame, proportional wings, sharp features",
        "humanoid": "human-like proportions, anatomically correct limbs",
    }
    
    # Detail emphasis for tabletop painting
    DETAIL_HINTS = {
        "armor": "prominent armor plating, defined edges, deep recesses for washing",
        "weapon": "clear weapon definition, sharp details, good grip ergonomics",
        "face": "expressive facial features, clear eye sockets, defined mouth",
        "robes": "flowing fabric folds, layered appearance, texture variation",
        "metal": "metallic sheen, surface details, rivets and seams",
        "organic": "natural texture, muscle definition, realistic proportions",
        "marine": "power armor with panel lines, oversized pauldrons, chest emblem relief, helmet vents",
        "space marine": "grimdark power armor, segmented plating, purity seals, gothic mechanical details",
    }
    
    # Scale anchors
    SCALE_HINTS = {
        "15mm": "15mm mass-battle wargaming miniature scale, simplified but readable oversized details",
        "20mm": "20mm wargaming miniature scale, printable exaggerated details",
        "25mm": "25mm classic tabletop miniature scale, clear heroic proportions",
        "28mm": "28mm tabletop miniature scale, hero proportions",
        "32mm": "32mm heroic scale miniature, dramatic proportions",
        "35mm": "35mm large heroic miniature scale, crisp painter-friendly detail",
        "40mm": "40mm skirmish/display miniature scale, strong silhouette and readable gear",
        "48mm": "48mm large wargaming/display miniature scale, highly readable sculpted details",
        "54mm": "54mm display miniature scale, high detail and dramatic proportions",
        "75mm": "75mm large display miniature scale, showcase-level surface detail",
        "tabletop": "tabletop miniature scale 15-75mm, suitable for 3D printing",
    }

    CHARACTER_TERMS = {
        "miniature", "figure", "character", "warrior", "soldier", "knight", "wizard", "mage",
        "orc", "ork", "elf", "dwarf", "demon", "daemon", "undead", "skeleton", "zombie",
        "ranger", "archer", "marine", "humanoid", "creature", "monster", "beast",
    }
    OBJECT_TERMS = {
        "mask", "helmet", "helm", "headpiece", "faceplate", "weapon", "rifle", "gun", "pistol",
        "sword", "axe", "hammer", "shield", "banner", "prop", "accessory", "terrain", "building",
        "vehicle", "tank", "ship", "turret", "bust", "statue", "idol", "totem",
    }

    @staticmethod
    def _generation_intent(prompt: str) -> str:
        lower = (prompt or "").lower()
        tokens = set(re.findall(r"[a-z0-9']+", lower))
        if any(term in lower for term in ("full body", "full-body", "whole character", "entire character")):
            return "character_miniature"
        if tokens & {"bust", "portrait"} or "head bust" in lower:
            return "bust"
        if tokens & {"mask", "helmet", "helm", "headpiece", "faceplate"}:
            return "wearable_object"
        if tokens & {"rifle", "gun", "pistol", "sword", "axe", "hammer", "shield", "banner", "weapon", "prop", "accessory"}:
            return "prop_object"
        if tokens & {"terrain", "scenery", "building", "ruin", "dungeon", "base", "objective"}:
            return "terrain_object"
        if tokens & {"vehicle", "tank", "ship", "walker", "turret"}:
            return "vehicle_object"
        if tokens & MiniaturePromptEnhancer.CHARACTER_TERMS:
            return "character_miniature"
        return "printable_subject"
    
    @staticmethod
    def enhance_prompt(prompt: str, scale: str = "28mm") -> str:
        """
        Enhance user prompt with miniature-specific guidance
        
        Args:
            prompt: User's original prompt
            scale: Target scale (28mm, 32mm, tabletop)
            
        Returns:
            Enhanced prompt with anatomical and technical constraints
        """
        if not prompt or not isinstance(prompt, str):
            return (
                "a tabletop miniature STL, small resin-printable character model, "
                "anatomically correct, detailed sculpt, sharp focus, not a rock, not a cylinder, not base-only"
            )
        
        prompt_lower = prompt.lower()
        enhanced = prompt
        
        # Add anatomical hints based on subject
        for key, hint in MiniaturePromptEnhancer.ANATOMY_HINTS.items():
            if key in prompt_lower:
                enhanced += f", {hint}"
                break
        
        # Add detail hints
        details_added = []
        for key, hint in MiniaturePromptEnhancer.DETAIL_HINTS.items():
            if key in prompt_lower and hint not in details_added:
                enhanced += f", {hint}"
                details_added.append(hint)
        
        # Add scale anchor
        scale_hint = MiniaturePromptEnhancer.SCALE_HINTS.get(scale, MiniaturePromptEnhancer.SCALE_HINTS["28mm"])
        if "tabletop" not in enhanced.lower() and "miniature" not in enhanced.lower():
            enhanced += f", {scale_hint}"
        
        intent = MiniaturePromptEnhancer._generation_intent(prompt)

        # Add universal printable-sculpt improvements without changing topology.
        if "3d printable" not in enhanced.lower():
            enhanced += ", 3D printable sculpt, crisp hard-surface details, sharp edges, deep recesses, high micro-detail, optimized for resin"

        if intent == "character_miniature":
            enhanced += (
                ", full-body tabletop miniature STL, preserve the requested character silhouette and distinctive gear, "
                "clear pose, printable thickness, watertight geometry, not a smooth blob, not a rock, "
                "not a plain cylinder, not a generic pedestal, not base-only"
            )
        elif intent == "wearable_object":
            enhanced += (
                ", standalone mask/helmet/accessory STL, preserve the requested object silhouette, openings, straps, lenses, "
                "vents, rims, panel seams, raised trim, printable wall thickness, watertight geometry, no forced humanoid body"
            )
        elif intent == "prop_object":
            enhanced += (
                ", standalone weapon or prop STL, preserve the requested item shape, grip, blade/barrel/head profile, "
                "mechanical greebles, panel seams, readable silhouette, printable thickness, watertight geometry, no forced humanoid body"
            )
        elif intent == "terrain_object":
            enhanced += (
                ", standalone terrain/scenery STL, preserve layout, walls, rubble, ground texture, readable architectural shapes, "
                "printable thickness, watertight geometry, no forced character body"
            )
        elif intent == "vehicle_object":
            enhanced += (
                ", standalone vehicle/mechanical STL, preserve hull, cockpit, treads/wheels/legs/turrets, armor plates, vents, "
                "panel seams, printable thickness, watertight geometry, no forced humanoid body"
            )
        else:
            enhanced += (
                ", preserve the requested subject topology and silhouette, printable thickness, watertight geometry, "
                "not a smooth blob, not a generic pedestal"
            )
        
        return enhanced
    
    @staticmethod
    def validate_prompt(prompt: str) -> tuple[bool, str]:
        """
        Validate prompt for miniature generation feasibility
        
        Args:
            prompt: User's prompt
            
        Returns:
            (is_valid, message)
        """
        if not prompt or len(prompt) < 5:
            return False, "Prompt too short. Please describe at least 5 words."
        
        # Problematic keywords
        problematic = ["microscopic", "1mm", "2mm", "100mm", "enormous", "infinite", "impossible"]
        for word in problematic:
            if word in prompt.lower():
                return False, f"Prompt contains '{word}' which may be incompatible with 28-32mm miniature scale."
        
        # Too complex
        if len(prompt) > 500:
            return False, "Prompt too long. Keep it under 500 characters for best results."
        
        # All good
        return True, "Prompt valid for miniature generation"
    
    @staticmethod
    def get_prompt_suggestions(base_prompt: str) -> list[str]:
        """Get variations of the prompt for trying different styles"""
        variations = [
            base_prompt,
            f"{base_prompt}, heroic pose, dramatic lighting",
            f"{base_prompt}, standing ready, combat stance",
            f"{base_prompt}, detailed armor, intricate design",
            f"{base_prompt}, epic scale, impressive details",
        ]
        return variations
