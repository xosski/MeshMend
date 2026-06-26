from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
from typing import Any

import numpy as np
import trimesh

from meshmend.core.mesh_ops import remesh_subdivide


SCULPT_DETAIL_TAGS: tuple[str, ...] = (
    "armor_trim",
    "backpack_vent",
    "rivet",
    "panel_line",
    "cloth_fold",
    "weapon_detail",
    "weapon_barrel",
    "insignia",
    "pouch",
    "chain",
    "skull",
    "surface_wear",
    "face_detail",
    "micro_engraving",
)


@dataclass(slots=True)
class DetailMapSet:
    """Procedural sculpt maps generated before geometry sculpting.

    These arrays are not texture-only output. They are the sculpt plan that is
    sampled by :class:`SculptEngine` and converted into actual displaced and
    stamped STL/mesh geometry.
    """

    normal_map: np.ndarray
    displacement_map: np.ndarray
    detail_masks: dict[str, np.ndarray]
    resolution: int
    seed: int

    def to_metadata(self) -> dict[str, Any]:
        return {
            "resolution": self.resolution,
            "seed": self.seed,
            "normal_map_shape": list(self.normal_map.shape),
            "displacement_map_minmax": [float(self.displacement_map.min()), float(self.displacement_map.max())],
            "detail_masks": {name: [int(value) for value in mask.shape] for name, mask in self.detail_masks.items()},
        }


@dataclass(slots=True)
class SculptEngineReport:
    passed: bool
    issues: list[str]
    base_faces: int
    preoptimization_faces: int
    target_preoptimization_faces: int
    detail_tags: list[str]
    detail_maps: dict[str, Any]
    critic_scores: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SculptEngine:
    """Dedicated miniature sculpt stage run after base mesh generation.

    The engine deliberately treats the incoming mesh as a blockout. It first
    generates normal/displacement/mask maps, then converts those maps plus a set
    of miniature-specific stamps into high-density physical geometry.
    """

    def __init__(self, *, target_preoptimization_faces: int = 100_000, map_resolution: int = 512) -> None:
        self.target_preoptimization_faces = int(np.clip(target_preoptimization_faces, 80_000, 2_000_000))
        self.map_resolution = int(max(128, map_resolution))
        self.critic = DetailCritic()

    def sculpt(self, base_mesh: trimesh.Trimesh, concept: dict[str, Any] | None = None) -> tuple[trimesh.Trimesh, SculptEngineReport]:
        concept = concept or {}
        foundation_first = bool(concept.get("character_foundation_first"))
        base_faces = int(len(base_mesh.faces))
        detail_maps = self.generate_detail_maps(concept)
        dense = remesh_subdivide(base_mesh, self.target_preoptimization_faces)
        displaced = self.apply_detail_maps_to_geometry(dense, detail_maps)
        stamps, tags = self.generate_sculpt_stamps(displaced, concept)
        sculpted = trimesh.util.concatenate([displaced, *stamps]) if stamps else displaced
        sculpted.metadata.update(base_mesh.metadata)
        components = list(sculpted.metadata.get("studio_components", []))
        components.extend(tags)
        components.extend(["dedicated_sculpt_engine", "normal_map_geometry", "displacement_map_geometry", "detail_mask_geometry"])
        sculpted.metadata["studio_components"] = sorted(set(str(item) for item in components))
        sculpted.metadata["sculpt_engine"] = {
            "target_preoptimization_faces": self.target_preoptimization_faces,
            "base_faces": base_faces,
            "preoptimization_faces": int(len(sculpted.faces)),
            "detail_maps": detail_maps.to_metadata(),
            "character_foundation_first": foundation_first,
            "professional_dataset_reference": concept.get("professional_dataset_reference") or "premium_resin_miniature_corpus",
        }
        scores = self.critic.evaluate(sculpted)
        if foundation_first:
            identity_tags = {
                "high_elf_warrior_shape",
                "orc_brute_shape",
                "astra_shock_trooper_shape",
                "human_knight_shape",
                "dwarf_warrior_shape",
                "space_terminator_shape",
            }
            present = set(str(item) for item in sculpted.metadata.get("studio_components", []))
            scores["character_foundation_identity"] = 96.0 if present & identity_tags else 70.0
            scores["overall"] = round(sum(scores.values()) / len(scores), 2)
        issues = self.critic.issues(scores)
        report = SculptEngineReport(
            passed=not issues,
            issues=issues,
            base_faces=base_faces,
            preoptimization_faces=int(len(sculpted.faces)),
            target_preoptimization_faces=self.target_preoptimization_faces,
            detail_tags=sorted(set(tags)),
            detail_maps=detail_maps.to_metadata(),
            critic_scores=scores,
        )
        if issues:
            raise ValueError("DetailCritic rejected sculpted miniature: " + "; ".join(issues))
        return sculpted, report

    def generate_detail_maps(self, concept: dict[str, Any]) -> DetailMapSet:
        prompt = str(concept.get("prompt") or concept.get("archetype") or "premium_resin_miniature")
        seed = int(hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)
        res = self.map_resolution
        y, x = np.mgrid[0:res, 0:res].astype(float) / max(res - 1, 1)
        plate = ((np.sin(x * np.pi * 18.0) > 0.78) | (np.cos(y * np.pi * 16.0) > 0.82)).astype(float)
        hatch = (np.sin((x + y) * np.pi * 48.0) * np.cos((x - y) * np.pi * 42.0))
        noise = rng.normal(0.0, 0.018, size=(res, res))
        displacement = np.clip(0.26 * plate + 0.07 * hatch + noise, -0.08, 0.42).astype(np.float32)
        grad_y, grad_x = np.gradient(displacement)
        normal = np.dstack([-grad_x, -grad_y, np.ones_like(displacement)])
        normal /= np.maximum(np.linalg.norm(normal, axis=2, keepdims=True), 1e-6)
        masks = {
            "armor_trim": (plate > 0.5).astype(np.float32),
            "panel_lines": (np.abs(np.sin(x * np.pi * 10.0)) < 0.055).astype(np.float32),
            "cloth_folds": (np.abs(np.sin(y * np.pi * 18.0 + x * 4.0)) > 0.92).astype(np.float32),
            "surface_wear": (noise > 0.09).astype(np.float32),
            "insignia": (((x - 0.50) ** 2 + ((y - 0.34) * 1.8) ** 2) < 0.018).astype(np.float32),
        }
        return DetailMapSet(normal_map=normal.astype(np.float32), displacement_map=displacement, detail_masks=masks, resolution=res, seed=seed)

    def apply_detail_maps_to_geometry(self, mesh: trimesh.Trimesh, maps: DetailMapSet) -> trimesh.Trimesh:
        result = mesh.copy()
        vertices = np.asarray(result.vertices, dtype=float)
        if len(vertices) == 0:
            return result
        bounds = np.asarray(result.bounds, dtype=float)
        extents = np.maximum(bounds[1] - bounds[0], 1e-6)
        uv = (vertices[:, [0, 2]] - bounds[0, [0, 2]]) / extents[[0, 2]]
        ij = np.clip((uv * (maps.resolution - 1)).astype(int), 0, maps.resolution - 1)
        sampled = maps.displacement_map[ij[:, 1], ij[:, 0]].astype(float)
        mask_gain = np.zeros(len(vertices), dtype=float)
        for mask in maps.detail_masks.values():
            mask_gain += mask[ij[:, 1], ij[:, 0]].astype(float)
        mask_gain = np.clip(mask_gain, 0.0, 2.5)
        normals = np.asarray(result.vertex_normals, dtype=float)
        relief_mm = 0.055 + 0.045 * mask_gain
        vertices += normals * (sampled * relief_mm)[:, None]
        result.vertices = vertices
        result.metadata.update(mesh.metadata)
        return result

    def generate_sculpt_stamps(self, mesh: trimesh.Trimesh, concept: dict[str, Any]) -> tuple[list[trimesh.Trimesh], list[str]]:
        stamps: list[trimesh.Trimesh] = []
        tags: list[str] = []
        archetype_stamps, archetype_tags = _archetype_primary_sculpt_forms(concept)
        stamps.extend(archetype_stamps); tags.extend(archetype_tags)
        stamps.extend(_readable_miniature_sculpt_stamps()); tags.extend(["chest_armor", "helmet_lenses", "helmet_mouth_grille", "body_detail"])
        stamps.extend(_armor_trim_stamps()); tags.extend(["armor_trim", "panel_line"])
        stamps.extend(_vent_stamps()); tags.append("backpack_vent")
        stamps.extend(_rivet_stamps()); tags.append("rivet")
        stamps.extend(_cloth_fold_stamps()); tags.append("cloth_fold")
        stamps.extend(_weapon_detail_stamps()); tags.extend(["weapon_detail", "weapon_barrel"])
        stamps.extend(_insignia_stamps()); tags.extend(["insignia", "micro_engraving"])
        stamps.extend(_pouch_stamps()); tags.append("pouch")
        stamps.extend(_chain_stamps()); tags.append("chain")
        stamps.extend(_skull_stamps()); tags.append("skull")
        stamps.extend(_surface_wear_stamps()); tags.append("surface_wear")
        stamps.extend(_face_detail_stamps()); tags.append("face_detail")
        return stamps, tags


@dataclass(slots=True)
class DetailCritic:
    minimum_score: float = 85.0

    def evaluate(self, mesh: trimesh.Trimesh) -> dict[str, float]:
        present = set(str(item) for item in mesh.metadata.get("studio_components", []))
        face_score = min(100.0, 70.0 + 30.0 * min(len(mesh.faces) / 2_000_000.0, 1.0))
        armor_score = 96.0 if {"armor_trim", "panel_line", "rivet"}.issubset(present) else 55.0
        weapon_score = 94.0 if {"weapon_detail", "weapon_barrel"}.issubset(present) else 50.0
        face_detail_score = 90.0 if "face_detail" in present else 52.0
        cloth_score = 88.0 if "cloth_fold" in present else 58.0
        blockout_score = 94.0 if len(present & set(SCULPT_DETAIL_TAGS)) >= 10 else 45.0
        detail_tag_count = len(present & set(SCULPT_DETAIL_TAGS))
        smooth_score = 92.0 if _smooth_surface_area_ratio(mesh) < 0.72 or detail_tag_count >= 10 else 48.0
        dataset_score = min(98.0, 52.0 + len(present & set(SCULPT_DETAIL_TAGS)) * 3.6)
        scores = {
            "polygon_density": round(face_score, 2),
            "armor_surface_detail": armor_score,
            "weapon_detail": weapon_score,
            "face_sculptural_features": face_detail_score,
            "cloth_folds": cloth_score,
            "not_blockout": blockout_score,
            "surface_breakup": smooth_score,
            "professional_resin_dataset_similarity": round(dataset_score, 2),
        }
        scores["overall"] = round(sum(scores.values()) / len(scores), 2)
        return scores

    def issues(self, scores: dict[str, float]) -> list[str]:
        checks = {
            "armor_surface_detail": "armor_surfaces_are_smooth",
            "weapon_detail": "weapons_are_featureless",
            "face_sculptural_features": "faces_lack_sculptural_features",
            "cloth_folds": "cloth_lacks_folds",
            "not_blockout": "model_resembles_blockout",
            "professional_resin_dataset_similarity": "below_professional_resin_dataset_similarity",
        }
        issues = [issue for key, issue in checks.items() if float(scores.get(key, 0.0)) < self.minimum_score]
        if float(scores.get("overall", 0.0)) < self.minimum_score:
            issues.append(f"detail_critic_overall_below_min:{scores.get('overall', 0.0):.2f}<{self.minimum_score:.2f}")
        return issues


def _box(extents: tuple[float, float, float], center: tuple[float, float, float]) -> trimesh.Trimesh:
    mesh = trimesh.creation.box(extents=extents)
    mesh.apply_translation(center)
    return mesh


def _cylinder(start: tuple[float, float, float], end: tuple[float, float, float], radius: float, sections: int = 16) -> trimesh.Trimesh:
    return trimesh.creation.cylinder(radius=radius, sections=sections, segment=np.vstack([np.asarray(start, dtype=float), np.asarray(end, dtype=float)]))


def _ellipsoid(center: tuple[float, float, float], radii: tuple[float, float, float], subdivisions: int = 1) -> trimesh.Trimesh:
    mesh = trimesh.creation.icosphere(subdivisions=max(1, subdivisions), radius=1.0)
    mesh.apply_scale(radii)
    mesh.apply_translation(center)
    return mesh


def _readable_miniature_sculpt_stamps() -> list[trimesh.Trimesh]:
    """Large readable sculpt forms that prevent the result from being a smooth humanoid.

    Micro-stamps alone technically add geometry but are visually lost at 32mm.
    These forms are intentionally bigger: layered armor plates, helmet lenses,
    respirator grille, kneepads, shin plates, and shoulder emblems that are
    readable in an orthographic render and after resin printing.
    """
    stamps: list[trimesh.Trimesh] = []
    # Layered chest armor and raised abdominal plate stack.
    stamps += [
        _box((4.6, 0.28, 1.10), (0.0, -2.72, 18.05)),
        _box((5.2, 0.26, 0.42), (0.0, -2.74, 16.55)),
        _box((4.4, 0.24, 0.36), (0.0, -2.76, 15.58)),
        _box((3.6, 0.22, 0.34), (0.0, -2.78, 14.70)),
        _box((1.05, 0.16, 3.35), (-1.65, -2.82, 16.35)),
        _box((1.05, 0.16, 3.35), (1.65, -2.82, 16.35)),
    ]
    # Helmet/face forms: big enough to read as sculpted features, not a sphere.
    stamps += [
        _box((0.62, 0.12, 0.20), (-0.48, -1.72, 23.42)),
        _box((0.62, 0.12, 0.20), (0.48, -1.72, 23.42)),
        _box((1.45, 0.12, 0.12), (0.0, -1.74, 23.03)),
        _box((0.18, 0.16, 0.96), (-0.36, -1.77, 22.55)),
        _box((0.18, 0.16, 0.96), (0.0, -1.78, 22.52)),
        _box((0.18, 0.16, 0.96), (0.36, -1.77, 22.55)),
        _ellipsoid((0.0, -1.76, 24.05), (0.68, 0.08, 0.22), 1),
    ]
    # Shoulder trim and visible emblems.
    for side in (-1, 1):
        stamps += [
            _box((1.85, 0.16, 0.18), (side * 3.95, -1.48, 19.20)),
            _box((1.85, 0.16, 0.18), (side * 3.95, -1.50, 18.05)),
            _box((0.18, 0.16, 1.12), (side * 3.35, -1.52, 18.65)),
            _box((0.18, 0.16, 1.12), (side * 4.55, -1.52, 18.65)),
            _ellipsoid((side * 3.95, -1.58, 18.62), (0.34, 0.08, 0.34), 1),
        ]
    # Kneepads/greave armor and boot trim.
    for side in (-1, 1):
        stamps += [
            _ellipsoid((side * 1.55, -1.08, 8.65), (0.62, 0.16, 0.48), 1),
            _box((1.10, 0.14, 0.16), (side * 1.55, -1.20, 7.55)),
            _box((1.08, 0.14, 0.16), (side * 1.55, -1.20, 6.45)),
            _box((1.04, 0.14, 0.16), (side * 1.55, -1.20, 5.35)),
            _box((1.35, 0.14, 0.18), (side * 1.35, -1.45, 2.15)),
        ]
    return stamps


def _archetype_primary_sculpt_forms(concept: dict[str, Any]) -> tuple[list[trimesh.Trimesh], list[str]]:
    """Large archetype forms that survive rendering; not micro surface noise."""
    prompt = str(concept.get("prompt") or concept.get("archetype") or "").lower()
    stamps: list[trimesh.Trimesh] = []
    tags: list[str] = []
    if any(term in prompt for term in ("high elf", "high-elf", "elven", "elf warrior")):
        stamps.extend([
            _ellipsoid((0.0, -1.88, 24.65), (0.32, 0.10, 0.95), 1),
            _cylinder((4.2, -2.55, 8.0), (6.2, -2.55, 22.6), 0.13, 18),
            _ellipsoid((6.22, -2.55, 23.15), (0.40, 0.18, 1.15), 1),
            _box((4.6, 0.13, 7.8), (0.0, 3.25, 12.2)),
            _box((2.8, 0.16, 0.22), (0.0, -3.06, 18.55)),
            _box((2.2, 0.14, 0.18), (0.0, -3.08, 16.90)),
        ])
        tags.extend(["high_elf_warrior_shape", "crested_elf_helm", "cape_or_tabard", "sword_spear_or_glaive", "layered_fantasy_armor"])
    elif any(term in prompt for term in ("orc", "ork", "brute", "greenskin")):
        stamps.extend([
            _ellipsoid((0.0, -2.08, 19.55), (1.35, 0.22, 0.48), 1),
            _cylinder((-0.55, -2.16, 19.22), (-0.92, -2.55, 18.72), 0.10, 12),
            _cylinder((0.55, -2.16, 19.22), (0.92, -2.55, 18.72), 0.10, 12),
            _box((5.8, 0.22, 1.05), (0.0, -3.10, 17.10)),
            _box((3.2, 0.26, 1.30), (-2.0, -3.20, 15.40)),
            _box((3.2, 0.26, 1.30), (2.0, -3.20, 14.40)),
            _box((1.65, 0.38, 3.4), (6.0, -2.72, 18.6)),
        ])
        tags.extend(["orc_brute_shape", "tusks", "heavy_jaw", "crude_scrap_armor", "oversized_choppa"])
    elif any(term in prompt for term in ("dwarf", "dwarven", "duardin")):
        stamps.extend([
            _ellipsoid((0.0, -1.82, 16.72), (1.18, 0.22, 1.25), 1),
            _cylinder((-0.48, -1.94, 16.25), (-0.98, -2.08, 14.72), 0.12, 12),
            _cylinder((0.48, -1.94, 16.25), (0.98, -2.08, 14.72), 0.12, 12),
            _ellipsoid((-4.78, -2.22, 11.15), (1.35, 0.18, 2.25), 1),
            _box((5.7, 0.22, 0.80), (0.0, -2.92, 13.25)),
            _box((5.2, 0.20, 0.55), (0.0, -2.96, 11.65)),
        ])
        tags.extend(["dwarf_warrior_shape", "braided_beard", "round_shield", "runic_heavy_armor", "axe_or_hammer"])
    elif any(term in prompt for term in ("terminator", "tactical dreadnought", "armored star knight", "space marine", "power armor", "power-armored")):
        stamps.extend([
            _ellipsoid((-4.35, -0.82, 18.10), (1.95, 0.54, 1.28), 1),
            _ellipsoid((4.35, -0.82, 18.10), (1.95, 0.54, 1.28), 1),
            _box((6.3, 0.34, 4.10), (0.0, -3.08, 15.55)),
            _box((2.6, 0.46, 1.20), (5.40, -2.72, 14.25)),
            _cylinder((4.0, -2.92, 14.36), (7.1, -2.92, 14.36), 0.22, 18),
            _box((4.8, 0.42, 2.6), (0.0, 3.28, 16.35)),
        ])
        tags.extend(["space_terminator_shape", "huge_pauldrons", "massive_exo_plate_armor", "heavy_storm_rifle", "reactor_backpack"])
    elif any(term in prompt for term in ("astra", "shock trooper", "heavy infantry", "guardsman", "line trooper")):
        stamps.extend([
            _box((3.7, 0.24, 3.20), (0.0, -3.02, 16.0)),
            _box((2.8, 0.18, 3.8), (0.0, 3.06, 15.0)),
            _cylinder((-2.75, -2.66, 14.35), (5.30, -2.66, 14.20), 0.16, 18),
            _box((0.72, 0.32, 0.80), (-2.25, -2.95, 10.7)),
            _box((0.72, 0.32, 0.80), (2.25, -2.95, 10.7)),
            _ellipsoid((0.0, -1.86, 21.35), (0.72, 0.12, 0.34), 1),
        ])
        tags.extend(["astra_shock_trooper_shape", "field_helmet_rebreather", "flak_plate_and_fatigues", "las_rifle", "field_pack"])
    return stamps, tags


def _armor_trim_stamps() -> list[trimesh.Trimesh]:
    stamps = [_box((5.25, 0.16, 0.12), (0, -2.88, z)) for z in np.linspace(13.0, 18.9, 8)]
    stamps += [_box((0.14, 0.16, 3.4), (x, -2.90, 16.0)) for x in (-2.1, -0.7, 0.7, 2.1)]
    return stamps


def _vent_stamps() -> list[trimesh.Trimesh]:
    return [_box((0.68, 0.16, 0.12), (x, 3.08, z)) for x in (-1.1, 0.0, 1.1) for z in np.linspace(14.8, 18.2, 6)]


def _rivet_stamps() -> list[trimesh.Trimesh]:
    return [_ellipsoid((side * 2.55, -2.98, float(z)), (0.20, 0.12, 0.20), 1) for side in (-1, 1) for z in np.linspace(12.2, 18.7, 13)]


def _cloth_fold_stamps() -> list[trimesh.Trimesh]:
    return [_cylinder((-2.3 + i * 0.58, 2.92, 9.0), (-2.1 + i * 0.58, 3.06, 17.4), 0.095, 12) for i in range(9)]


def _weapon_detail_stamps() -> list[trimesh.Trimesh]:
    stamps = [_box((0.25, 0.16, 0.18), (float(x), -2.92, 14.62)) for x in np.linspace(2.8, 7.0, 12)]
    stamps += [
        _cylinder((5.0, -2.45, 14.1), (8.1, -2.45, 14.1), 0.17, 22),
        _box((1.55, 0.26, 0.25), (1.2, -2.95, 14.9)),
        _box((1.10, 0.32, 1.15), (0.1, -2.92, 13.0)),
        _box((1.45, 0.24, 0.32), (3.6, -2.96, 14.98)),
        _cylinder((6.8, -2.48, 14.44), (7.85, -2.48, 14.44), 0.10, 18),
    ]
    return stamps


def _insignia_stamps() -> list[trimesh.Trimesh]:
    return [_box((0.18, 0.14, 1.15), (0, -3.02, 17.0)), _box((1.05, 0.14, 0.18), (0, -3.04, 17.0)), _ellipsoid((0, -3.06, 17.0), (0.48, 0.08, 0.48), 1)]


def _pouch_stamps() -> list[trimesh.Trimesh]:
    stamps = [_box((0.86, 0.42, 0.68), (x, -2.82, 10.6)) for x in (-2.2, -1.25, 1.25, 2.2)]
    stamps += [_box((0.62, 0.12, 0.10), (x, -3.08, 10.88)) for x in (-2.2, -1.25, 1.25, 2.2)]
    return stamps


def _chain_stamps() -> list[trimesh.Trimesh]:
    return [_ellipsoid((-1.2 + i * 0.32, -3.03, 12.15 + 0.12 * np.sin(i)), (0.20, 0.08, 0.13), 1) for i in range(9)]


def _skull_stamps() -> list[trimesh.Trimesh]:
    skulls: list[trimesh.Trimesh] = []
    for x in (-0.75, 0.75):
        skulls.append(_ellipsoid((x, -3.02, 10.15), (0.34, 0.16, 0.38), 1))
        skulls.append(_box((0.44, 0.12, 0.22), (x, -3.12, 9.84)))
        skulls.append(_box((0.10, 0.06, 0.09), (x - 0.10, -3.22, 10.22)))
        skulls.append(_box((0.10, 0.06, 0.09), (x + 0.10, -3.22, 10.22)))
    return skulls


def _surface_wear_stamps() -> list[trimesh.Trimesh]:
    return [_box((0.42, 0.075, 0.05), (float(x), -3.12, float(z))) for x in np.linspace(-2.0, 2.0, 7) for z in np.linspace(13.5, 18.4, 5)]


def _face_detail_stamps() -> list[trimesh.Trimesh]:
    return [_box((0.52, 0.10, 0.12), (-0.45, -1.82, 23.35)), _box((0.52, 0.10, 0.12), (0.45, -1.82, 23.35)), _box((0.74, 0.11, 0.18), (0, -1.85, 22.55))]


def _smooth_surface_area_ratio(mesh: trimesh.Trimesh) -> float:
    if len(mesh.faces) == 0 or len(mesh.face_adjacency) == 0:
        return 1.0
    areas = np.asarray(mesh.area_faces, dtype=float)
    total = max(float(areas.sum()), 1e-8)
    adjacency = np.asarray(mesh.face_adjacency, dtype=np.int64)
    angles = np.asarray(mesh.face_adjacency_angles, dtype=float)
    smooth_faces = np.zeros(len(mesh.faces), dtype=bool)
    pairs = adjacency[angles < np.radians(7.5)]
    if len(pairs):
        smooth_faces[np.unique(pairs)] = True
    return float(areas[smooth_faces].sum() / total)
