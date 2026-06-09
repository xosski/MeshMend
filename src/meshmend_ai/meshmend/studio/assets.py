from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
import json
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

from meshmend.core.io import save_mesh


class PartCategory(StrEnum):
    CONCEPT_PROFILE = "concept_profile"
    HEAD = "head"
    TORSO = "torso"
    LEGS = "legs"
    LEFT_ARM = "left_arm"
    RIGHT_ARM = "right_arm"
    WEAPONS = "weapons"
    ACCESSORIES = "accessories"
    HELMET = "helmet"
    CHEST_ARMOR = "chest_armor"
    SHOULDER_PADS = "shoulder_pads"
    BACKPACK = "backpack"
    WEAPON = "weapon"
    HEAD_HELMET = "head_helmet"
    TORSO_BODY = "torso_body"
    ARMS = "arms"
    BACKPACK_ACCESSORIES = "backpack_accessories"
    BASE = "base"


@dataclass(slots=True)
class AnchorPoint:
    name: str
    position: tuple[float, float, float]
    normal: tuple[float, float, float] = (0.0, 0.0, 1.0)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ConnectionSocket:
    name: str
    kind: str
    position: tuple[float, float, float]
    radius_mm: float
    direction: tuple[float, float, float] = (0.0, 0.0, 1.0)

    def usable(self) -> bool:
        return self.radius_mm >= 0.25 and float(np.linalg.norm(np.asarray(self.direction, dtype=float))) > 1e-6

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class PartCleanupReport:
    faces: int
    vertices: int
    watertight: bool
    boundary_edges: int
    non_manifold_edges: int
    components: int
    sheet_artifacts: int
    primitive_score: float
    detail_density: float
    issues: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.issues

    def to_dict(self) -> dict[str, Any]:
        return asdict(self) | {"passed": self.passed}


@dataclass(slots=True)
class ModularMiniaturePart:
    part_id: str
    category: PartCategory
    mesh: trimesh.Trimesh
    anchors: list[AnchorPoint]
    sockets: list[ConnectionSocket]
    scale_mm: float
    symmetry: str = "none"
    detail_tags: list[str] = field(default_factory=list)
    source: str = "procedural"
    cleanup_report: PartCleanupReport | None = None

    def metadata(self) -> dict[str, Any]:
        return {
            "part_id": self.part_id,
            "category": self.category.value,
            "scale_mm": self.scale_mm,
            "symmetry": self.symmetry,
            "source": self.source,
            "detail_tags": list(self.detail_tags),
            "anchors": [anchor.to_dict() for anchor in self.anchors],
            "connection_sockets": [socket.to_dict() for socket in self.sockets],
            "cleanup_report": self.cleanup_report.to_dict() if self.cleanup_report else None,
        }

    def export_bundle(self, output_dir: str | Path, *, mesh_format: str = "stl") -> dict[str, str]:
        """Write mesh, preview render, metadata, and cleanup report for selection UI.

        The preview is an offline SVG orthographic thumbnail so this works without
        GPU/OpenGL/paid services. A GUI can replace it with a richer render later.
        """
        directory = Path(output_dir) / self.category.value / self.part_id
        directory.mkdir(parents=True, exist_ok=True)
        mesh_path = directory / f"{self.part_id}.{mesh_format.lower().lstrip('.')}"
        save_mesh(self.mesh, mesh_path)
        metadata_path = directory / "metadata.json"
        metadata_path.write_text(json.dumps(self.metadata(), indent=2), encoding="utf-8")
        cleanup_path = directory / "cleanup_report.json"
        cleanup_path.write_text(json.dumps((self.cleanup_report.to_dict() if self.cleanup_report else {}), indent=2), encoding="utf-8")
        preview_path = directory / "preview.svg"
        preview_path.write_text(render_preview_svg(self.mesh, title=self.part_id), encoding="utf-8")
        return {
            "mesh_file": str(mesh_path),
            "preview_render": str(preview_path),
            "metadata_file": str(metadata_path),
            "cleanup_report": str(cleanup_path),
        }


class ModularAssetProvider:
    """Shared interface for AI-generated and hand-authored kitbash parts."""

    name = "base_provider"

    def generate_candidates(self, category: PartCategory, concept: dict[str, Any], count: int, scale_mm: float) -> list[ModularMiniaturePart]:
        raise NotImplementedError


class DirectoryKitbashProvider(ModularAssetProvider):
    """Load hand-authored kitbash parts from a directory.

    Expected layout: `<root>/<category>/<part_id>.stl|obj|glb` with optional
    `<part_id>.metadata.json`. Missing anchors/sockets are inferred from bounds.
    """

    name = "directory_kitbash"

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def generate_candidates(self, category: PartCategory, concept: dict[str, Any], count: int, scale_mm: float) -> list[ModularMiniaturePart]:
        category_dir = self.root / category.value
        if not category_dir.exists():
            return []
        candidates: list[ModularMiniaturePart] = []
        for mesh_path in sorted(category_dir.iterdir()):
            if mesh_path.suffix.lower() not in {".stl", ".obj", ".glb", ".ply"}:
                continue
            loaded = trimesh.load(mesh_path, force="mesh", process=False)
            if isinstance(loaded, trimesh.Scene):
                loaded = trimesh.util.concatenate(tuple(loaded.geometry.values()))
            if not isinstance(loaded, trimesh.Trimesh) or len(loaded.faces) == 0:
                continue
            meta_path = mesh_path.with_suffix(mesh_path.suffix + ".metadata.json")
            metadata = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
            part = ModularMiniaturePart(
                part_id=mesh_path.stem,
                category=category,
                mesh=loaded,
                anchors=_anchors_from_metadata(metadata) or default_anchors(loaded),
                sockets=_sockets_from_metadata(metadata) or default_sockets(loaded, category),
                scale_mm=float(metadata.get("scale_mm") or scale_mm),
                symmetry=str(metadata.get("symmetry") or "none"),
                detail_tags=list(metadata.get("detail_tags") or ["kitbash_detail"]),
                source=f"kitbash:{mesh_path}",
            )
            candidates.append(part)
            if len(candidates) >= count:
                break
        return candidates


def default_anchors(mesh: trimesh.Trimesh) -> list[AnchorPoint]:
    bounds = np.asarray(mesh.bounds, dtype=float)
    center = bounds.mean(axis=0)
    return [
        AnchorPoint("center", tuple(float(v) for v in center)),
        AnchorPoint("bottom", (float(center[0]), float(center[1]), float(bounds[0, 2])), (0.0, 0.0, -1.0)),
        AnchorPoint("top", (float(center[0]), float(center[1]), float(bounds[1, 2])), (0.0, 0.0, 1.0)),
    ]


def default_sockets(mesh: trimesh.Trimesh, category: PartCategory) -> list[ConnectionSocket]:
    bounds = np.asarray(mesh.bounds, dtype=float)
    center = bounds.mean(axis=0)
    radius = max(float(np.min(np.maximum(bounds[1] - bounds[0], 1e-6))) * 0.12, 0.3)
    sockets = [ConnectionSocket("root", "root", tuple(float(v) for v in center), radius)]
    if category in (PartCategory.TORSO, PartCategory.TORSO_BODY):
        sockets += [
            ConnectionSocket("head", "neck", (float(center[0]), float(center[1]), float(bounds[1, 2])), radius),
            ConnectionSocket("left_arm", "shoulder", (float(bounds[0, 0]), float(center[1]), float(center[2])), radius),
            ConnectionSocket("right_arm", "shoulder", (float(bounds[1, 0]), float(center[1]), float(center[2])), radius),
        ]
    return sockets


def evaluate_part_cleanup(mesh: trimesh.Trimesh, *, detail_tags: list[str], min_faces: int = 200, max_components: int = 8) -> PartCleanupReport:
    diagnostic = mesh.copy()
    try:
        diagnostic.remove_duplicate_faces()
        diagnostic.remove_degenerate_faces()
        diagnostic.remove_unreferenced_vertices()
        diagnostic.merge_vertices()
        diagnostic.fix_normals()
    except Exception:
        pass
    faces = int(len(diagnostic.faces))
    vertices = int(len(diagnostic.vertices))
    edge_counts = np.bincount(diagnostic.edges_unique_inverse) if faces else np.array([], dtype=int)
    boundary_edges = int((edge_counts == 1).sum()) if len(edge_counts) else 0
    non_manifold_edges = int((edge_counts > 2).sum()) if len(edge_counts) else 0
    components = len([part for part in diagnostic.split(only_watertight=False) if len(part.faces) > 20])
    sheet_artifacts = _part_sheet_artifacts(diagnostic)
    primitive_score = _primitive_score(diagnostic)
    detail_density = _detail_density(diagnostic, detail_tags)
    issues: list[str] = []
    if faces < min_faces:
        issues.append(f"part_faces_below_minimum:{faces}<{min_faces}")
    if not bool(diagnostic.is_watertight):
        issues.append("part_not_watertight")
    if boundary_edges:
        issues.append(f"part_boundary_edges:{boundary_edges}")
    if non_manifold_edges:
        issues.append(f"part_non_manifold_edges:{non_manifold_edges}")
    if components > max_components:
        issues.append(f"part_disconnected_shells:{components}")
    if sheet_artifacts:
        issues.append("part_sheet_or_plane_artifact")
    if primitive_score > 0.86 and detail_density < 0.30:
        issues.append("part_primitive_only_geometry")
    if detail_density < 0.16:
        issues.append(f"part_detail_density_too_low:{detail_density:.2f}")
    return PartCleanupReport(
        faces=faces,
        vertices=vertices,
        watertight=bool(diagnostic.is_watertight),
        boundary_edges=boundary_edges,
        non_manifold_edges=non_manifold_edges,
        components=components,
        sheet_artifacts=sheet_artifacts,
        primitive_score=primitive_score,
        detail_density=detail_density,
        issues=issues,
    )


def validate_part(part: ModularMiniaturePart, *, min_faces: int = 200, max_components: int = 8) -> ModularMiniaturePart:
    report = evaluate_part_cleanup(part.mesh, detail_tags=part.detail_tags, min_faces=min_faces, max_components=max_components)
    socket_issues = [socket.name for socket in part.sockets if not socket.usable()]
    if socket_issues:
        report.issues.append("unusable_sockets:" + ",".join(socket_issues))
    part.cleanup_report = report
    if report.issues:
        raise ValueError(f"Part {part.part_id} failed quality gate: " + "; ".join(report.issues))
    return part


def render_preview_svg(mesh: trimesh.Trimesh, *, title: str) -> str:
    vertices = np.asarray(mesh.vertices, dtype=float)
    if len(vertices) == 0:
        return "<svg xmlns='http://www.w3.org/2000/svg' width='320' height='320'></svg>"
    xy = vertices[:, [0, 2]]
    mins = xy.min(axis=0)
    ext = np.maximum(np.ptp(xy, axis=0), 1e-6)
    points = (xy - mins) / ext
    points[:, 0] = points[:, 0] * 260 + 30
    points[:, 1] = 290 - points[:, 1] * 260
    sample = points[:: max(1, len(points) // 700)]
    circles = "\n".join(f"<circle cx='{x:.1f}' cy='{y:.1f}' r='1.2' fill='#2f6fed'/>" for x, y in sample)
    return (
        "<svg xmlns='http://www.w3.org/2000/svg' width='320' height='320' viewBox='0 0 320 320'>"
        "<rect width='320' height='320' fill='#f8f8f8'/>"
        f"<text x='12' y='22' font-size='13' font-family='monospace'>{title}</text>"
        f"{circles}</svg>"
    )


def _anchors_from_metadata(metadata: dict[str, Any]) -> list[AnchorPoint]:
    return [AnchorPoint(**item) for item in metadata.get("anchors", [])]


def _sockets_from_metadata(metadata: dict[str, Any]) -> list[ConnectionSocket]:
    return [ConnectionSocket(**item) for item in metadata.get("connection_sockets", [])]


def _detail_density(mesh: trimesh.Trimesh, detail_tags: list[str]) -> float:
    extents = np.maximum(np.asarray(mesh.extents, dtype=float), 1e-6)
    surface_scale = max(float(np.linalg.norm(extents[:2]) * max(extents[2], 1.0)), 1.0)
    geometry_density = min(float(len(mesh.faces)) / (surface_scale * 120.0), 0.75)
    tag_density = min(len(detail_tags) / 8.0, 0.35)
    return min(1.0, geometry_density + tag_density)


def _primitive_score(mesh: trimesh.Trimesh) -> float:
    if len(mesh.faces) == 0:
        return 1.0
    extents = np.maximum(np.asarray(mesh.extents, dtype=float), 1e-6)
    normals = np.asarray(mesh.face_normals, dtype=float)
    areas = np.asarray(mesh.area_faces, dtype=float)
    axis_ratio = float(areas[np.max(np.abs(normals), axis=1) > 0.985].sum() / max(float(areas.sum()), 1e-8))
    compactness = float(extents.min() / extents.max())
    low_face_penalty = 1.0 if len(mesh.faces) < 500 else 0.0
    return min(1.0, axis_ratio * 0.65 + compactness * 0.25 + low_face_penalty * 0.35)


def _part_sheet_artifacts(mesh: trimesh.Trimesh) -> int:
    extents = np.maximum(np.asarray(mesh.extents, dtype=float), 1e-6)
    if float(extents.min() / extents.max()) < 0.035 and len(mesh.faces) > 20:
        return 1
    return 0
