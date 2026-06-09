from __future__ import annotations

from dataclasses import dataclass



@dataclass(frozen=True, slots=True)
class KitbashAsset:
    name: str
    category: str
    description: str


def list_builtin_assets() -> list[KitbashAsset]:
    """MVP metadata for legally distinct kitbash categories."""
    return [
        KitbashAsset("round_base_25mm", "base", "25mm circular gaming base"),
        KitbashAsset("round_base_32mm", "base", "32mm circular gaming base"),
        KitbashAsset("generic_power_sword", "weapon", "non-branded sci-fantasy blade"),
        KitbashAsset("generic_rifle", "weapon", "non-branded blockout rifle"),
        KitbashAsset("plain_shoulder_armor", "armor", "blank shoulder armor suitable for custom symbols"),
    ]
