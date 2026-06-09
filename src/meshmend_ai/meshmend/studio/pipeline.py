from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path

import numpy as np
import trimesh

from meshmend.compliance.filters import sanitize_prompt
from meshmend.core.io import save_mesh
from meshmend.core.mesh_ops import auto_scale_to_height, remesh_subdivide
from meshmend.export import export_slicer_ready
from meshmend.studio.quality import MiniatureSculptQualityGate, StudioQualityGate, StudioQualityReport


@dataclass(slots=True)
class StudioMiniatureSpec:
    prompt: str
    scale_mm: float = 32.0
    archetype: str = "heavy_infantry"
    style: str = "sci_fi"
    pose: str = "heroic_standing"
    weapon: str = "rifle"
    helmet: bool = True
    backpack: bool = True
    cape: bool = False
    robe: bool = False
    claws: bool = False
    base: str = "round_textured"
    target_faces: int = 250_000
    details: list[str] = field(default_factory=list)

    @classmethod
    def from_prompt(cls, prompt: str, *, scale_mm: float = 32.0, target_faces: int = 250_000) -> "StudioMiniatureSpec":
        clean_prompt, _warnings = sanitize_prompt(prompt)
        text = clean_prompt.lower()
        style = "fantasy" if any(word in text for word in ("fantasy", "knight", "orc", "elf", "robe", "wizard", "claw")) else "sci_fi"
        weapon = "sword" if any(word in text for word in ("sword", "blade", "axe", "mace")) else "rifle"
        if "claw" in text:
            weapon = "claws"
        details = ["panel_lines", "rivets", "armor_seams", "base_texture"]
        if style == "sci_fi":
            details += ["helmet_lenses", "backpack_vents", "weapon_barrel", "cable_runs"]
        if style == "fantasy":
            details += ["cloth_folds", "trim", "scrolls"]
        return cls(
            prompt=clean_prompt,
            scale_mm=float(scale_mm),
            style=style,
            weapon=weapon,
            helmet=not any(word in text for word in ("bare head", "unhelmeted")),
            backpack=style == "sci_fi" or "backpack" in text,
            cape="cape" in text or "cloak" in text,
            robe="robe" in text or "wizard" in text,
            claws="claw" in text,
            target_faces=int(target_faces),
            details=details,
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "StudioMiniatureSpec":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**payload)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class StudioMiniaturePipeline:
    """Offline staged miniature generator.

    This deliberately replaces image-to-3D guessing for store-mode output. It is
    a procedural/kitbash sculpt pipeline: anatomy blockout, armor/equipment
    modules, readable miniature-scale detail, repair/cleanup, and a hard quality
    gate before export. Local AI model hooks can later supply pose/spec data, but
    the exportable geometry is deterministic and auditable.
    """

    def __init__(self, quality_gate: StudioQualityGate | None = None) -> None:
        self.quality_gate = quality_gate or MiniatureSculptQualityGate()

    def generate(self, spec: StudioMiniatureSpec) -> tuple[trimesh.Trimesh, StudioQualityReport]:
        parts: list[trimesh.Trimesh] = []
        names: list[str] = []
        self._add_anatomy(parts, names, spec)
        self._add_armor(parts, names, spec)
        self._add_equipment(parts, names, spec)
        self._add_detail(parts, names, spec)
        if spec.base:
            self._add_base(parts, names, spec)
        mesh = trimesh.util.concatenate(parts)
        mesh.metadata["studio_components"] = names
        mesh.metadata["studio_spec"] = spec.to_dict()
        mesh.metadata["meshmend_generator"] = "studio_procedural_miniature_pipeline"
        mesh = self.cleanup(mesh, spec)
        report = self.quality_gate.require_pass(mesh)
        return mesh, report

    def export(self, spec: StudioMiniatureSpec, output_path: str | Path) -> tuple[Path, StudioQualityReport]:
        mesh, report = self.generate(spec)
        output = export_slicer_ready(mesh, output_path, write_report=True)
        quality_path = output.with_suffix(output.suffix + ".studio_quality.json")
        quality_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
        spec_path = output.with_suffix(output.suffix + ".studio_spec.json")
        spec_path.write_text(json.dumps(spec.to_dict(), indent=2), encoding="utf-8")
        return output, report

    def cleanup(self, mesh: trimesh.Trimesh, spec: StudioMiniatureSpec) -> trimesh.Trimesh:
        cleaned = mesh.copy()
        try:
            cleaned.remove_duplicate_faces()
            cleaned.remove_degenerate_faces()
            cleaned.remove_infinite_values()
            cleaned.remove_unreferenced_vertices()
            cleaned.merge_vertices()
            cleaned.fix_normals()
            cleaned.fill_holes()
        except Exception:
            pass
        cleaned = _remove_artifact_shells(cleaned)
        cleaned = auto_scale_to_height(cleaned, spec.scale_mm)
        cleaned = remesh_subdivide(cleaned, max(spec.target_faces, self.quality_gate.min_faces))
        # A tiny Taubin pass removes jagged generated seams while preserving
        # hard-surface detail better than Laplacian smoothing.
        try:
            trimesh.smoothing.filter_taubin(cleaned, lamb=0.08, nu=-0.085, iterations=1)
        except Exception:
            pass
        cleaned.metadata.update(mesh.metadata)
        cleaned.metadata["units"] = "mm"
        return cleaned

    def _add_anatomy(self, parts: list[trimesh.Trimesh], names: list[str], spec: StudioMiniatureSpec) -> None:
        additions = [
            ("body", _ellipsoid((0, 0, 16.0), (3.6, 2.2, 5.2), 3)),
            ("pelvis", _ellipsoid((0, 0, 10.8), (3.1, 1.9, 1.8), 2)),
            ("head", _ellipsoid((0, -0.08, 23.0), (1.55, 1.25, 1.75), 2)),
            ("left_leg", _capsule_between((-1.35, 0, 2.6), (-1.75, 0.05, 10.2), 0.82, 24)),
            ("right_leg", _capsule_between((1.35, 0, 2.6), (1.75, 0.05, 10.2), 0.82, 24)),
            ("left_arm", _capsule_between((-3.25, -0.15, 18.5), (-4.45, -1.35, 11.8), 0.62, 20)),
            ("right_arm", _capsule_between((3.25, -0.15, 18.5), (4.45, -1.35, 12.2), 0.62, 20)),
        ]
        for name, mesh in additions:
            parts.append(mesh); names.append(name)
        names.extend(["arms", "legs"])
        for x in (-1.35, 1.35):
            foot = _ellipsoid((x, -0.45, 1.1), (1.25, 2.0, 0.42), 1)
            parts.append(foot); names.append("boot")

    def _add_armor(self, parts: list[trimesh.Trimesh], names: list[str], spec: StudioMiniatureSpec) -> None:
        armor_parts = [
            ("chest_armor", _box((5.2, 1.0, 4.9), (0, -1.5, 16.2))),
            ("back_armor", _box((4.8, 0.65, 4.4), (0, 1.45, 16.1))),
            ("belt", _box((5.6, 0.8, 0.55), (0, -0.65, 11.5))),
            ("left_pauldron", _ellipsoid((-3.9, -0.25, 18.8), (1.7, 1.15, 1.0), 2)),
            ("right_pauldron", _ellipsoid((3.9, -0.25, 18.8), (1.7, 1.15, 1.0), 2)),
        ]
        for name, mesh in armor_parts:
            parts.append(mesh); names.append(name)
        names.append("shoulder_pad")
        if spec.helmet:
            parts.append(_ellipsoid((0, -0.08, 23.0), (1.85, 1.45, 1.95), 2)); names.append("helmet")
            parts.append(_box((2.15, 0.16, 0.24), (0, -1.48, 23.25))); names.append("helmet_lenses")
            parts.append(_box((0.55, 0.42, 1.25), (0, -1.46, 22.35))); names.append("helmet_mouth_grille")
        for x in (-1.55, 1.55):
            parts.append(_box((1.35, 0.55, 2.2), (x, -0.88, 8.0))); names.append("thigh_armor")
            parts.append(_box((1.25, 0.45, 2.4), (x, -0.82, 4.5))); names.append("greave_armor")

    def _add_equipment(self, parts: list[trimesh.Trimesh], names: list[str], spec: StudioMiniatureSpec) -> None:
        if spec.backpack:
            parts.append(_box((3.8, 1.3, 4.6), (0, 2.25, 16.4))); names.append("backpack")
            for x in (-1.05, 0, 1.05):
                parts.append(_box((0.58, 0.28, 0.18), (x, 2.98, 17.8))); names.append("backpack_vent")
                parts.append(_cylinder_between((x, 2.85, 14.3), (x, 3.25, 12.9), 0.16, 12)); names.append("backpack_cable")
        if spec.cape or spec.robe:
            for z in np.linspace(9.0, 17.5, 7):
                width = 4.6 + (17.5 - z) * 0.22
                parts.append(_box((width, 0.22, 0.32), (0, 2.55, float(z)))); names.append("cloth_fold")
        if spec.weapon == "sword":
            parts.append(_box((0.48, 0.22, 8.2), (4.95, -1.55, 13.4))); names.append("weapon")
            parts.append(_box((2.1, 0.38, 0.35), (4.95, -1.55, 9.4))); names.append("weapon_guard")
        elif spec.weapon == "claws" or spec.claws:
            for side in (-1, 1):
                for offset in (-0.28, 0, 0.28):
                    claw = trimesh.creation.cone(radius=0.11, height=1.35, sections=12)
                    claw.apply_transform(trimesh.transformations.rotation_matrix(np.radians(90), [1, 0, 0]))
                    claw.apply_translation([side * 4.45 + offset, -2.05, 11.3])
                    parts.append(claw); names.append("weapon")
        else:
            parts.append(_box((6.8, 0.68, 0.78), (2.0, -2.18, 14.1))); names.append("weapon")
            parts.append(_cylinder_between((5.0, -2.2, 14.1), (7.3, -2.2, 14.1), 0.22, 18)); names.append("weapon_barrel")
            parts.append(_box((0.95, 0.75, 1.55), (0.0, -2.18, 12.95))); names.append("magazine")
            parts.append(_box((1.55, 0.38, 0.42), (1.6, -2.58, 14.8))); names.append("scope")

    def _add_detail(self, parts: list[trimesh.Trimesh], names: list[str], spec: StudioMiniatureSpec) -> None:
        # Armor seams, trim, and panel lines are physical raised/recess-defining
        # geometry, not texture. These survive STL/OBJ/GLB export.
        detail_preremesh = trimesh.util.concatenate(parts)
        dense_preview = remesh_subdivide(detail_preremesh, max(60_000, min(spec.target_faces // 2, 160_000)))
        parts.clear()
        parts.append(dense_preview)
        names.append("pre_detail_subdivided_body")
        for z in (14.4, 16.2, 18.0):
            parts.append(_box((4.6, 0.18, 0.12), (0, -2.04, z))); names.append("armor_seam")
        for x in (-2.1, 2.1):
            parts.append(_box((0.14, 0.2, 3.6), (x, -2.05, 16.3))); names.append("panel_line")
        for side in (-1, 1):
            for z in np.linspace(12.0, 18.2, 6):
                parts.append(_rivet((side * 2.35, -2.14, float(z)), 0.16)); names.append("rivet")
        for x in (-1.7, 1.7):
            for z in (5.0, 6.2, 7.4, 8.6):
                parts.append(_box((0.9, 0.13, 0.1), (x, -1.15, z))); names.append("greave_trim")
        for side in (-1, 1):
            parts.append(_box((0.22, 0.22, 1.25), (side * 4.2, -1.72, 12.1))); names.append("finger_detail")
            parts.append(_box((0.22, 0.22, 1.25), (side * 4.55, -1.68, 12.1))); names.append("finger_detail")
        for x in (-2.2, -1.2, 1.2, 2.2):
            parts.append(_box((0.65, 0.36, 0.48), (x, -2.12, 10.75))); names.append("pouch")
        names.extend(["armor_trim", "bolt", "body_detail"])
        if spec.style == "fantasy":
            for z in (15.2, 16.7, 18.1):
                parts.append(_box((4.4, 0.16, 0.1), (0, -2.18, z))); names.append("fantasy_trim")

    def _add_base(self, parts: list[trimesh.Trimesh], names: list[str], spec: StudioMiniatureSpec) -> None:
        base = trimesh.creation.cylinder(radius=13.0, height=2.0, sections=128)
        base.apply_translation([0, 0, 0.0])
        parts.append(base); names.append("base")
        for i, angle in enumerate(np.linspace(0, np.pi * 2, 24, endpoint=False)):
            radius = 4.0 + (i % 5) * 1.55
            x, y = np.cos(angle) * radius, np.sin(angle) * radius
            pebble = _ellipsoid((float(x), float(y), 1.15), (0.38, 0.28, 0.12), 1)
            parts.append(pebble); names.append("base_texture")


def generate_studio_miniature(prompt: str, *, scale_mm: float = 32.0, target_faces: int = 90_000) -> tuple[trimesh.Trimesh, StudioQualityReport]:
    spec = StudioMiniatureSpec.from_prompt(prompt, scale_mm=scale_mm, target_faces=target_faces)
    return StudioMiniaturePipeline().generate(spec)


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


def _capsule_between(start: tuple[float, float, float], end: tuple[float, float, float], radius: float, sections: int) -> trimesh.Trimesh:
    start_arr = np.asarray(start, dtype=float)
    end_arr = np.asarray(end, dtype=float)
    cylinder = _cylinder_between(start, end, radius, sections)
    a = trimesh.creation.uv_sphere(radius=radius, count=[sections, max(8, sections // 2)])
    b = a.copy()
    a.apply_translation(start_arr)
    b.apply_translation(end_arr)
    return trimesh.util.concatenate([cylinder, a, b])


def _cylinder_between(start: tuple[float, float, float], end: tuple[float, float, float], radius: float, sections: int) -> trimesh.Trimesh:
    start_arr = np.asarray(start, dtype=float)
    end_arr = np.asarray(end, dtype=float)
    return trimesh.creation.cylinder(radius=radius, sections=sections, segment=np.vstack([start_arr, end_arr]))


def _remove_artifact_shells(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    components = [part for part in mesh.split(only_watertight=False) if len(part.faces) > 12]
    if not components:
        return mesh
    main = max(components, key=lambda part: float(part.area))
    main_center = main.bounds.mean(axis=0)
    main_radius = max(float(np.linalg.norm(main.extents)), 1e-6)
    kept = []
    for part in components:
        area_ratio = float(part.area) / max(float(main.area), 1e-6)
        distance = float(np.linalg.norm(part.bounds.mean(axis=0) - main_center))
        if area_ratio >= 0.0008 and distance <= main_radius * 0.95:
            kept.append(part)
    if not kept:
        kept = [main]
    result = trimesh.util.concatenate(kept)
    result.metadata.update(mesh.metadata)
    result.metadata["studio_removed_artifact_shells"] = len(components) - len(kept)
    return result
