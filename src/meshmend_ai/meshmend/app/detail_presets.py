from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DetailPreset:
    name: str
    panel_lines: bool = True
    rivets: bool = True
    battle_damage: bool = True
    surface_texture: str = "fine_metal"
    mechanical_grooves: bool = False
    cloth_grain: bool = False
    terrain_roughness: bool = False
    organic_texture: bool = False
    default_strength: float = 0.35


DETAIL_PRESETS: dict[str, DetailPreset] = {
    "sci-fi armor": DetailPreset("sci-fi armor", mechanical_grooves=True, surface_texture="brushed_metal", default_strength=0.38),
    "fantasy armor": DetailPreset("fantasy armor", surface_texture="hammered_metal", default_strength=0.34),
    "stone ruins": DetailPreset(
        "stone ruins",
        panel_lines=True,
        rivets=False,
        surface_texture="stone",
        terrain_roughness=True,
        default_strength=0.50,
    ),
    "rubble terrain": DetailPreset(
        "rubble terrain",
        panel_lines=False,
        rivets=False,
        surface_texture="rubble",
        terrain_roughness=True,
        default_strength=0.55,
    ),
    "mechanical vents": DetailPreset("mechanical vents", mechanical_grooves=True, surface_texture="machined", default_strength=0.44),
    "alien organic": DetailPreset(
        "alien organic",
        panel_lines=False,
        rivets=False,
        battle_damage=False,
        surface_texture="organic",
        organic_texture=True,
        default_strength=0.40,
    ),
    "cloth folds": DetailPreset(
        "cloth folds",
        panel_lines=False,
        rivets=False,
        battle_damage=False,
        surface_texture="woven_cloth",
        cloth_grain=True,
        default_strength=0.32,
    ),
    "skull/bone detail": DetailPreset(
        "skull/bone detail",
        panel_lines=False,
        rivets=False,
        battle_damage=True,
        surface_texture="bone",
        organic_texture=True,
        default_strength=0.36,
    ),
    "Sci-fi armor": DetailPreset("Sci-fi armor", mechanical_grooves=True, surface_texture="brushed_metal", default_strength=0.38),
    "Grimdark miniature": DetailPreset("Grimdark miniature", mechanical_grooves=True, surface_texture="worn_metal", default_strength=0.46),
    "Fantasy armor": DetailPreset("Fantasy armor", surface_texture="hammered_metal", default_strength=0.34),
    "Stone terrain": DetailPreset(
        "Stone terrain",
        panel_lines=False,
        rivets=False,
        surface_texture="stone",
        terrain_roughness=True,
        default_strength=0.48,
    ),
    "Ruined concrete": DetailPreset(
        "Ruined concrete",
        panel_lines=True,
        rivets=False,
        surface_texture="concrete",
        terrain_roughness=True,
        default_strength=0.52,
    ),
    "Alien organic": DetailPreset(
        "Alien organic",
        panel_lines=False,
        rivets=False,
        battle_damage=False,
        surface_texture="organic",
        organic_texture=True,
        default_strength=0.40,
    ),
    "Cloth folds": DetailPreset(
        "Cloth folds",
        panel_lines=False,
        rivets=False,
        battle_damage=False,
        surface_texture="woven_cloth",
        cloth_grain=True,
        default_strength=0.32,
    ),
    "Mechanical paneling": DetailPreset("Mechanical paneling", mechanical_grooves=True, surface_texture="machined", default_strength=0.42),
    "Battle damage": DetailPreset("Battle damage", panel_lines=False, rivets=False, surface_texture="scarred", default_strength=0.50),
}


def preset_names() -> list[str]:
    return list(DETAIL_PRESETS)


def get_preset(name: str) -> DetailPreset:
    return DETAIL_PRESETS.get(name, DETAIL_PRESETS["Sci-fi armor"])
