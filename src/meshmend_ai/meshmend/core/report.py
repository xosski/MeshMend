from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import trimesh


@dataclass(slots=True)
class PrintabilityReport:
    manifold: bool
    watertight: bool
    holes: int
    non_manifold_edges: int
    thin_parts: int
    floating_shells: int
    dimensions_mm: list[float]
    polygon_count: int
    vertex_count: int
    warnings: list[str]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def build_printability_report(mesh: trimesh.Trimesh, min_thickness_mm: float = 0.8) -> PrintabilityReport:
    """Compute resin-print oriented mesh diagnostics without cloud services."""
    diagnostic = mesh.copy()
    try:
        diagnostic.merge_vertices()
        diagnostic.remove_unreferenced_vertices()
    except Exception:
        pass
    edge_counts = np.bincount(diagnostic.edges_unique_inverse) if len(diagnostic.faces) else np.array([], dtype=int)
    boundary_edges = int((edge_counts == 1).sum()) if len(edge_counts) else 0
    non_manifold_edges = int((edge_counts > 2).sum()) if len(edge_counts) else 0
    shells = [part for part in diagnostic.split(only_watertight=False) if len(part.faces) > 8]
    floating_shells = _count_floating_shells(shells)
    extents = [float(v) for v in np.asarray(diagnostic.extents, dtype=float)]
    thin_parts = _estimate_thin_parts(diagnostic, min_thickness_mm=min_thickness_mm)
    warnings: list[str] = []
    if not diagnostic.is_watertight:
        warnings.append("mesh is not watertight")
    if boundary_edges:
        warnings.append(f"{boundary_edges} boundary edges / likely holes")
    if non_manifold_edges:
        warnings.append(f"{non_manifold_edges} non-manifold edges")
    if floating_shells:
        warnings.append(f"{floating_shells} floating shells/islands")
    if thin_parts:
        warnings.append(f"{thin_parts} sampled areas may be thinner than {min_thickness_mm:.2f}mm")
    return PrintabilityReport(
        manifold=non_manifold_edges == 0,
        watertight=bool(diagnostic.is_watertight),
        holes=boundary_edges,
        non_manifold_edges=non_manifold_edges,
        thin_parts=thin_parts,
        floating_shells=floating_shells,
        dimensions_mm=extents,
        polygon_count=int(len(diagnostic.faces)),
        vertex_count=int(len(diagnostic.vertices)),
        warnings=warnings,
    )


def _estimate_thin_parts(mesh: trimesh.Trimesh, min_thickness_mm: float) -> int:
    """Cheap local thin-wall estimate using small component extents.

    Full wall-thickness analysis needs ray/voxel backends; this MVP reports a
    conservative warning when connected shells have a suspiciously tiny axis.
    """
    count = 0
    for part in mesh.split(only_watertight=False):
        if len(part.faces) < 20:
            continue
        if float(np.min(part.extents)) < float(min_thickness_mm):
            count += 1
    return count


def _count_floating_shells(shells: list[trimesh.Trimesh], tolerance_mm: float = 1.0) -> int:
    """Count shells that are spatially separate from the primary printable body.

    Many kitbash/procedural workflows intentionally use overlapping closed parts
    before a final boolean union. Those are not floating islands for a slicer, so
    only warn when a component's bounding box is separated from the main shell.
    """
    if len(shells) <= 1:
        return 0
    bounds_list = [shell.bounds.astype(float) for shell in sorted(shells, key=lambda part: float(part.area), reverse=True)]
    main_bounds = bounds_list[0].copy()
    remaining = bounds_list[1:]
    changed = True
    while changed:
        changed = False
        next_remaining = []
        for bounds in remaining:
            separated = any(
                bounds[1, axis] < main_bounds[0, axis] - tolerance_mm or bounds[0, axis] > main_bounds[1, axis] + tolerance_mm
                for axis in range(3)
            )
            if separated:
                next_remaining.append(bounds)
            else:
                main_bounds[0] = np.minimum(main_bounds[0], bounds[0])
                main_bounds[1] = np.maximum(main_bounds[1], bounds[1])
                changed = True
        remaining = next_remaining
    return len(remaining)
