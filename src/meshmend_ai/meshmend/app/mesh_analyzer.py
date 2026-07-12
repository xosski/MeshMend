from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import trimesh


@dataclass(slots=True)
class MeshIssue:
    kind: str
    severity: str
    message: str


@dataclass(slots=True)
class MeshAnalysis:
    vertices: int
    faces: int
    watertight: bool
    manifold: bool
    boundary_edges: int
    non_manifold_edges: int
    duplicate_vertex_estimate: int
    degenerate_faces: int
    flipped_normals_risk: bool
    shells: int
    floating_shells: int
    thin_part_warnings: int
    dimensions_mm: list[float]
    surface_area_mm2: float
    low_detail_regions: int
    damaged_regions: int
    sharp_edges: int
    high_curvature_faces: int
    high_normal_variance_vertices: int
    protected_detail_faces: int
    protected_detail_vertices: int
    local_repair_vertices: int
    issues: list[MeshIssue] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["issues"] = [asdict(issue) for issue in self.issues]
        return payload


def analyze_mesh(mesh: trimesh.Trimesh, *, min_thickness_mm: float = 0.8) -> MeshAnalysis:
    """Inspect miniature printability and detail risks without altering geometry."""
    diagnostic = mesh.copy()
    try:
        diagnostic.merge_vertices()
        diagnostic.remove_unreferenced_vertices()
    except Exception:
        pass
    edge_counts = np.bincount(diagnostic.edges_unique_inverse) if len(diagnostic.faces) else np.array([], dtype=int)
    boundary_edges = int(np.count_nonzero(edge_counts == 1)) if len(edge_counts) else 0
    non_manifold_edges = int(np.count_nonzero(edge_counts > 2)) if len(edge_counts) else 0
    degenerate_faces = int(len(diagnostic.faces) - int(np.count_nonzero(diagnostic.nondegenerate_faces()))) if len(diagnostic.faces) else 0
    duplicate_vertex_estimate = _duplicate_vertex_estimate(mesh)
    shells = [part for part in diagnostic.split(only_watertight=False) if len(part.faces) > 8]
    floating_shells = max(0, len(shells) - 1)
    thin_part_warnings = _estimate_thin_parts(shells, min_thickness_mm)
    low_detail_regions = _estimate_low_detail_regions(diagnostic)
    damaged_regions = int(bool(boundary_edges)) + int(bool(non_manifold_edges)) + int(bool(degenerate_faces))
    flipped_normals_risk = bool(len(diagnostic.faces) and not diagnostic.is_winding_consistent)
    protection = detail_protection_zones(diagnostic)
    local_repair_vertices = len(local_repair_vertex_neighborhood(diagnostic))

    issues: list[MeshIssue] = []
    if boundary_edges:
        issues.append(MeshIssue("open_boundaries", "high", f"{boundary_edges} boundary edges / likely holes detected."))
    if non_manifold_edges:
        issues.append(MeshIssue("non_manifold_edges", "high", f"{non_manifold_edges} non-manifold edges detected."))
    if degenerate_faces:
        issues.append(MeshIssue("degenerate_faces", "medium", f"{degenerate_faces} degenerate faces can confuse slicers."))
    if duplicate_vertex_estimate:
        issues.append(MeshIssue("duplicate_vertices", "medium", f"Approximately {duplicate_vertex_estimate} duplicate vertices detected."))
    if flipped_normals_risk:
        issues.append(MeshIssue("normals", "medium", "Winding is inconsistent; normals may be flipped in some regions."))
    if floating_shells:
        issues.append(MeshIssue("floating_shells", "medium", f"{floating_shells} secondary shells/islands detected."))
    if thin_part_warnings:
        issues.append(MeshIssue("thin_parts", "medium", f"{thin_part_warnings} shells have an axis thinner than {min_thickness_mm:.2f} mm."))
    if low_detail_regions:
        issues.append(MeshIssue("low_detail_regions", "low", f"{low_detail_regions} broad/low-density face clusters may be low-detail areas."))
    if protection["protected_detail_faces"]:
        issues.append(
            MeshIssue(
                "protected_detail_zones",
                "info",
                f"{protection['protected_detail_faces']} high-curvature/high-normal-variance faces protected from smoothing or simplification.",
            )
        )

    return MeshAnalysis(
        vertices=int(len(mesh.vertices)),
        faces=int(len(mesh.faces)),
        watertight=bool(diagnostic.is_watertight),
        manifold=non_manifold_edges == 0,
        boundary_edges=boundary_edges,
        non_manifold_edges=non_manifold_edges,
        duplicate_vertex_estimate=duplicate_vertex_estimate,
        degenerate_faces=degenerate_faces,
        flipped_normals_risk=flipped_normals_risk,
        shells=len(shells),
        floating_shells=floating_shells,
        thin_part_warnings=thin_part_warnings,
        dimensions_mm=[float(value) for value in np.asarray(mesh.extents, dtype=float)],
        surface_area_mm2=float(mesh.area),
        low_detail_regions=low_detail_regions,
        damaged_regions=damaged_regions,
        sharp_edges=int(protection["sharp_edges"]),
        high_curvature_faces=int(protection["high_curvature_faces"]),
        high_normal_variance_vertices=int(protection["high_normal_variance_vertices"]),
        protected_detail_faces=int(protection["protected_detail_faces"]),
        protected_detail_vertices=int(protection["protected_detail_vertices"]),
        local_repair_vertices=int(local_repair_vertices),
        issues=issues,
    )


def detail_protection_zones(
    mesh: trimesh.Trimesh,
    *,
    sharp_angle_degrees: float = 35.0,
    normal_variance_degrees: float = 25.0,
) -> dict[str, int | np.ndarray]:
    """Find high-detail zones that must not be smoothed or simplified.

    High curvature and high normal variance are common signals for armor trim,
    panel lines, cloth folds, rivets, chains, teeth, and sculpted damage. These
    zones are treated as protected even when they are near a structural defect.
    """
    face_count = len(mesh.faces)
    vertex_count = len(mesh.vertices)
    protected_faces = np.zeros(face_count, dtype=bool)
    protected_vertices = np.zeros(vertex_count, dtype=bool)
    high_curvature_faces = np.zeros(face_count, dtype=bool)
    high_normal_variance_vertices = np.zeros(vertex_count, dtype=bool)

    if face_count and len(mesh.face_adjacency):
        sharp_threshold = np.deg2rad(float(sharp_angle_degrees))
        angles = np.asarray(mesh.face_adjacency_angles, dtype=float)
        sharp_pairs = np.asarray(mesh.face_adjacency)[angles >= sharp_threshold]
        if len(sharp_pairs):
            high_curvature_faces[np.unique(sharp_pairs.reshape(-1))] = True

    if face_count and vertex_count:
        normal_threshold = np.deg2rad(float(normal_variance_degrees))
        face_normals = np.asarray(mesh.face_normals, dtype=float)
        for vertex_index, face_indices in enumerate(mesh.vertex_faces):
            valid_faces = face_indices[face_indices >= 0]
            if len(valid_faces) < 2:
                continue
            normals = face_normals[valid_faces]
            mean = normals.mean(axis=0)
            norm = float(np.linalg.norm(mean))
            if norm <= 1e-12:
                high_normal_variance_vertices[vertex_index] = True
                continue
            mean /= norm
            max_angle = float(np.max(np.arccos(np.clip(normals @ mean, -1.0, 1.0))))
            if max_angle >= normal_threshold:
                high_normal_variance_vertices[vertex_index] = True

    protected_faces |= high_curvature_faces
    if np.any(high_normal_variance_vertices) and face_count:
        protected_faces |= np.any(high_normal_variance_vertices[np.asarray(mesh.faces)], axis=1)
    if np.any(protected_faces):
        protected_vertices[np.unique(np.asarray(mesh.faces)[protected_faces].reshape(-1))] = True
    protected_vertices |= high_normal_variance_vertices

    return {
        "sharp_edges": int(np.count_nonzero(high_curvature_faces)),
        "high_curvature_faces": int(np.count_nonzero(high_curvature_faces)),
        "high_normal_variance_vertices": int(np.count_nonzero(high_normal_variance_vertices)),
        "protected_detail_faces": int(np.count_nonzero(protected_faces)),
        "protected_detail_vertices": int(np.count_nonzero(protected_vertices)),
        "protected_face_mask": protected_faces,
        "protected_vertex_mask": protected_vertices,
    }


def local_repair_vertex_neighborhood(mesh: trimesh.Trimesh, *, expansion_rings: int = 1) -> set[int]:
    """Return vertices near structural defects for local-only repair planning."""
    vertices: set[int] = set()
    if len(mesh.faces) == 0:
        return vertices
    edge_counts = np.bincount(mesh.edges_unique_inverse)
    defect_edges = mesh.edges_unique[(edge_counts == 1) | (edge_counts > 2)] if len(edge_counts) else np.empty((0, 2), dtype=int)
    vertices.update(int(index) for index in defect_edges.reshape(-1))
    if len(mesh.faces):
        degenerate_mask = ~mesh.nondegenerate_faces()
        if np.any(degenerate_mask):
            vertices.update(int(index) for index in np.asarray(mesh.faces)[degenerate_mask].reshape(-1))
    adjacency = mesh.vertex_neighbors
    frontier = set(vertices)
    for _ in range(max(0, int(expansion_rings))):
        next_frontier: set[int] = set()
        for vertex in frontier:
            next_frontier.update(int(neighbor) for neighbor in adjacency[vertex])
        vertices.update(next_frontier)
        frontier = next_frontier
    return vertices


def _duplicate_vertex_estimate(mesh: trimesh.Trimesh) -> int:
    if len(mesh.vertices) == 0:
        return 0
    rounded = np.round(np.asarray(mesh.vertices, dtype=float), decimals=8)
    unique = np.unique(rounded, axis=0)
    return max(0, int(len(mesh.vertices) - len(unique)))


def _estimate_thin_parts(shells: list[trimesh.Trimesh], min_thickness_mm: float) -> int:
    warnings = 0
    for shell in shells:
        if len(shell.faces) < 20:
            continue
        if float(np.min(shell.extents)) < min_thickness_mm:
            warnings += 1
    return warnings


def _estimate_low_detail_regions(mesh: trimesh.Trimesh) -> int:
    """Cheap detector for broad low-density areas, not a smoothing instruction."""
    if len(mesh.faces) < 32:
        return int(len(mesh.faces) > 0)
    face_areas = np.asarray(mesh.area_faces, dtype=float)
    if len(face_areas) == 0:
        return 0
    median_area = float(np.median(face_areas))
    if median_area <= 1e-12:
        return 0
    broad_faces = face_areas > median_area * 12.0
    return int(np.count_nonzero(broad_faces) // 16)
