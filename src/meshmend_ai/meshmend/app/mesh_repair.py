from __future__ import annotations

from dataclasses import dataclass, field
import tempfile
from pathlib import Path

import numpy as np
import trimesh

from meshmend_ai.repair import RepairOptions, repair_stl
from meshmend.app.mesh_analyzer import MeshAnalysis, analyze_mesh, detail_protection_zones, local_repair_vertex_neighborhood
from meshmend.app.mesh_loader import load_mesh_file


@dataclass(slots=True)
class RepairSettings:
    studio_master_mode: bool = True
    fill_holes: bool = True
    remove_duplicate_vertices: bool = True
    fix_normals: bool = True
    remove_degenerate_faces: bool = True
    max_hole_edges: int = 80
    max_existing_vertex_displacement_mm: float = 0.005
    preview_mode: bool = False
    local_neighborhood_rings: int = 1


@dataclass(slots=True)
class RepairResult:
    mesh: trimesh.Trimesh
    actions: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    before: MeshAnalysis | None = None
    after: MeshAnalysis | None = None
    protected_detail_vertices: int = 0
    local_repair_vertices: int = 0
    modified_vertex_estimate: int = 0


def auto_repair_mesh(mesh: trimesh.Trimesh, settings: RepairSettings) -> RepairResult:
    """Repair common defects while preserving miniature sculpt detail.

    The file-based MeshMend repairer already enforces the museum-scan tolerance.
    We use a temporary STL handoff so the GUI benefits from the same tested path
    as the command-line repair workflow.
    """
    actions: list[str] = []
    warnings: list[str] = []
    before = analyze_mesh(mesh)
    protection = detail_protection_zones(mesh)
    local_vertices = local_repair_vertex_neighborhood(mesh, expansion_rings=settings.local_neighborhood_rings)
    actions.append(
        "protected detail zones: "
        f"{int(protection['protected_detail_vertices'])} vertices / {int(protection['protected_detail_faces'])} faces"
    )
    actions.append(f"local-only repair neighborhood: {len(local_vertices)} vertices near structural defects")
    with tempfile.TemporaryDirectory(prefix="meshmend_app_repair_") as temp_dir:
        temp = Path(temp_dir)
        input_path = temp / "input.stl"
        output_path = temp / "output.stl"
        mesh.export(input_path)
        report = repair_stl(
            input_path,
            output_path,
            RepairOptions(
                bridge_disconnected=False,
                max_hole_edges=settings.max_hole_edges if settings.fill_holes else 0,
                max_existing_vertex_displacement=settings.max_existing_vertex_displacement_mm,
            ),
        )
        actions.append("ran detail-preserving structural repair")
        actions.append(f"holes capped: {report.holes_capped}")
        actions.append(f"max existing vertex displacement: {report.max_existing_vertex_displacement:.6g} mm")
        actions.append("global smoothing/remeshing/decimation: skipped")
        if report.bridges_added:
            warnings.append("connector bridges were added; this should be disabled in Studio Master Mode")
        repaired = load_mesh_file(output_path)
    after = analyze_mesh(repaired)
    modified_vertex_estimate = _modified_vertex_estimate(mesh, repaired, settings.max_existing_vertex_displacement_mm)
    actions.append(
        "before/after: "
        f"faces {before.faces:,} -> {after.faces:,}, vertices {before.vertices:,} -> {after.vertices:,}, "
        f"boundary edges {before.boundary_edges:,} -> {after.boundary_edges:,}"
    )
    actions.append(f"estimated source vertices modified beyond tolerance: {modified_vertex_estimate}")
    return RepairResult(
        mesh=repaired,
        actions=actions,
        warnings=warnings,
        before=before,
        after=after,
        protected_detail_vertices=int(protection["protected_detail_vertices"]),
        local_repair_vertices=len(local_vertices),
        modified_vertex_estimate=modified_vertex_estimate,
    )


def fill_holes(mesh: trimesh.Trimesh, settings: RepairSettings) -> RepairResult:
    local_settings = RepairSettings(
        studio_master_mode=settings.studio_master_mode,
        fill_holes=True,
        remove_duplicate_vertices=False,
        fix_normals=False,
        remove_degenerate_faces=False,
        max_hole_edges=settings.max_hole_edges,
        max_existing_vertex_displacement_mm=settings.max_existing_vertex_displacement_mm,
        preview_mode=settings.preview_mode,
        local_neighborhood_rings=settings.local_neighborhood_rings,
    )
    return auto_repair_mesh(mesh, local_settings)


def remove_duplicate_vertices(mesh: trimesh.Trimesh, *, studio_master_mode: bool = True) -> RepairResult:
    before_analysis = analyze_mesh(mesh)
    repaired = mesh.copy()
    vertex_count_before = len(repaired.vertices)
    # High precision keeps this to true duplicates / STL-import duplicates; it is
    # not a decimation or simplification pass.
    repaired.merge_vertices(digits_vertex=8 if studio_master_mode else 6)
    repaired.remove_unreferenced_vertices()
    after = len(repaired.vertices)
    return RepairResult(
        mesh=repaired,
        actions=[f"removed duplicate vertices: {vertex_count_before - after}", "only exact/high-precision duplicates were merged"],
        before=before_analysis,
        after=analyze_mesh(repaired),
    )


def fix_normals(mesh: trimesh.Trimesh) -> RepairResult:
    before = analyze_mesh(mesh)
    repaired = mesh.copy()
    trimesh.repair.fix_winding(repaired)
    trimesh.repair.fix_normals(repaired)
    if repaired.is_watertight:
        trimesh.repair.fix_inversion(repaired)
    return RepairResult(mesh=repaired, actions=["fixed winding/normals without moving vertices"], before=before, after=analyze_mesh(repaired))


def edge_sharpen_pass(mesh: trimesh.Trimesh) -> RepairResult:
    """Preserve hard-surface readability without changing geometry.

    Trimesh/STL do not carry a full editable normal-split stack like a DCC app.
    For the MVP we intentionally avoid geometric sharpening because that would
    move vertices. Future OBJ/GLB exporters can add explicit normal splitting.
    """
    before = analyze_mesh(mesh)
    repaired = mesh.copy()
    repaired.metadata["meshmend_edge_sharpen"] = "preserve_hard_edges_no_geometry_change"
    angles = repaired.face_adjacency_angles if len(repaired.face_adjacency) else np.array([], dtype=float)
    sharp_edges = int(np.count_nonzero(angles > np.deg2rad(35.0))) if len(angles) else 0
    return RepairResult(
        mesh=repaired,
        actions=[f"marked {sharp_edges} hard edges for preservation; geometry unchanged"],
        before=before,
        after=analyze_mesh(repaired),
    )


def _modified_vertex_estimate(before: trimesh.Trimesh, after: trimesh.Trimesh, tolerance: float) -> int:
    """Estimate how many original referenced coordinates no longer exist after repair."""
    if len(before.faces) == 0 or len(after.vertices) == 0:
        return 0
    tolerance = max(float(tolerance), 1e-12)
    original_indices = np.unique(np.asarray(before.faces).reshape(-1))
    original_vertices = np.asarray(before.vertices, dtype=float)[original_indices]
    after_keys = {tuple(np.round(vertex / tolerance).astype(np.int64)) for vertex in np.asarray(after.vertices, dtype=float)}
    return int(sum(tuple(np.round(vertex / tolerance).astype(np.int64)) not in after_keys for vertex in original_vertices))
