from __future__ import annotations

from dataclasses import dataclass
import os

import numpy as np
import trimesh


# "8K" here means dense display/printing mesh quality, not merely 8,000 triangles.
# A 28-32mm miniature needs hundreds of thousands to millions of triangles before
# sub-millimeter edge pitch checks become meaningful.
TARGET_8K_TRIANGLES = int(os.environ.get("MESHMEND_TARGET_8K_TRIANGLES", "1000000"))
TARGET_HIGH_RESOLUTION_PITCH_MM = 0.10
MIN_HIGH_RESOLUTION_PITCH_MM = 0.025
MAX_DETAIL_TRIANGLES = int(os.environ.get("MESHMEND_MAX_DETAIL_TRIANGLES", "8000000"))


@dataclass(frozen=True, slots=True)
class Detail8KReport:
    passed: bool
    faces: int
    vertices: int
    target_faces: int
    target_pitch_mm: float
    median_edge_mm: float
    p95_edge_mm: float
    extents: list[float]
    issues: list[str]

    def summary(self) -> str:
        status = "passed" if self.passed else "needs attention"
        issue_text = "; ".join(self.issues) if self.issues else "no issues"
        pitch_um = self.target_pitch_mm * 1000.0
        p95_um = self.p95_edge_mm * 1000.0
        return (
            f"Mesh-density check {status}: {self.faces} faces, {self.vertices} vertices, "
            f"target <={pitch_um:.0f} um, p95 edge {p95_um:.0f} um; {issue_text}"
        )


def assess_8k_detail(
    mesh: trimesh.Trimesh,
    target_faces: int = TARGET_8K_TRIANGLES,
    target_pitch_mm: float = TARGET_HIGH_RESOLUTION_PITCH_MM,
) -> Detail8KReport:
    issues: list[str] = []
    faces = int(len(mesh.faces))
    vertices = int(len(mesh.vertices))
    extents = np.asarray(mesh.extents, dtype=float)
    max_extent = float(np.max(extents)) if len(extents) else 0.0
    min_extent = float(np.min(extents)) if len(extents) else 0.0
    median_edge_mm, p95_edge_mm = _edge_length_stats(mesh)

    if faces < target_faces:
        issues.append(f"below_{target_faces}_triangle_target")
    if vertices < target_faces // 3:
        issues.append("low_vertex_density")
    if p95_edge_mm > target_pitch_mm:
        issues.append(f"edge_pitch_above_{target_pitch_mm * 1000:.0f}um_target")
    if max_extent <= 1e-9:
        issues.append("zero_size_mesh")
    elif min_extent / max_extent < 0.035:
        issues.append("mesh_is_too_flat_for_miniature")
    if not bool(mesh.is_watertight):
        issues.append("not_watertight")

    return Detail8KReport(
        passed=not issues,
        faces=faces,
        vertices=vertices,
        target_faces=target_faces,
        target_pitch_mm=target_pitch_mm,
        median_edge_mm=median_edge_mm,
        p95_edge_mm=p95_edge_mm,
        extents=extents.round(4).tolist(),
        issues=issues,
    )


def ensure_8k_detail(
    mesh: trimesh.Trimesh,
    target_faces: int = TARGET_8K_TRIANGLES,
    max_faces: int = MAX_DETAIL_TRIANGLES,
    target_pitch_mm: float = TARGET_HIGH_RESOLUTION_PITCH_MM,
) -> tuple[trimesh.Trimesh, Detail8KReport]:
    return ensure_high_resolution_detail(mesh, target_faces, max_faces, target_pitch_mm)


def ensure_high_resolution_detail(
    mesh: trimesh.Trimesh,
    target_faces: int = TARGET_8K_TRIANGLES,
    max_faces: int = MAX_DETAIL_TRIANGLES,
    target_pitch_mm: float = TARGET_HIGH_RESOLUTION_PITCH_MM,
) -> tuple[trimesh.Trimesh, Detail8KReport]:
    target_pitch_mm = max(MIN_HIGH_RESOLUTION_PITCH_MM, min(float(target_pitch_mm), TARGET_HIGH_RESOLUTION_PITCH_MM))
    detailed = mesh.copy()
    detailed.merge_vertices()
    detailed.remove_unreferenced_vertices()

    while _needs_more_detail(detailed, target_faces, target_pitch_mm) and 0 < len(detailed.faces) * 4 <= max_faces:
        vertices, faces = trimesh.remesh.subdivide(detailed.vertices, detailed.faces)
        detailed = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        detailed.merge_vertices()
        detailed.remove_unreferenced_vertices()

    return detailed, assess_8k_detail(detailed, target_faces, target_pitch_mm)


def _needs_more_detail(mesh: trimesh.Trimesh, target_faces: int, target_pitch_mm: float) -> bool:
    if len(mesh.faces) < target_faces:
        return True
    _median_edge_mm, p95_edge_mm = _edge_length_stats(mesh)
    return p95_edge_mm > target_pitch_mm


def _edge_length_stats(mesh: trimesh.Trimesh) -> tuple[float, float]:
    if len(mesh.edges_unique) == 0:
        return 0.0, 0.0
    vertices = np.asarray(mesh.vertices, dtype=float)
    edges = np.asarray(mesh.edges_unique, dtype=np.int64)
    lengths = np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1)
    if len(lengths) == 0:
        return 0.0, 0.0
    return float(np.median(lengths)), float(np.percentile(lengths, 95))
