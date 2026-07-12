from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import trimesh


@dataclass(slots=True)
class RepairOptions:
    bridge_disconnected: bool = False
    connector_radius: float = 0.75
    connector_sections: int = 16
    max_bridge_distance: float | None = None
    merge_digits: int = 6
    max_hole_edges: int = 80
    max_existing_vertex_displacement: float = 0.005


@dataclass(slots=True)
class RepairReport:
    input_path: str
    output_path: str
    vertices_before: int
    faces_before: int
    vertices_after: int
    faces_after: int
    components_before: int
    components_after: int
    watertight_before: bool
    watertight_after: bool
    boundary_edges_before: int
    boundary_edges_after: int
    holes_capped: int
    bridges_added: int
    max_existing_vertex_displacement: float
    detail_preservation: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def repair_stl(
    input_path: str | Path,
    output_path: str | Path,
    options: RepairOptions | None = None,
) -> RepairReport:
    """Repair an STL/mesh file and export the result.

    The repair path intentionally favors predictable geometry operations over a
    generative model. AI can choose settings and explain tradeoffs, but the STL
    changes themselves should be deterministic and repeatable.

    Museum-scan rule: do not smooth, decimate, subdivide, remesh, relax, inflate,
    shrink, or average the source surface. Existing vertex coordinates are kept
    fixed; repair adds/removes only structural faces/vertices needed for duplicate
    cleanup, degenerate removal, normal orientation, and small boundary closure.
    """

    options = options or RepairOptions()
    input_path = Path(input_path)
    output_path = Path(output_path)

    original = _load_mesh(input_path)
    components_before = _component_count(original)
    boundary_edges_before = _boundary_edge_count(original)

    repaired = original.copy()
    _clean_mesh(repaired, options.merge_digits)
    holes_capped = _cap_boundary_holes(repaired, options.max_hole_edges)
    _repair_orientation(repaired)
    _clean_mesh(repaired, options.merge_digits)

    bridges_added = 0
    if options.bridge_disconnected:
        repaired, bridges_added = _bridge_components(repaired, options)
        _clean_mesh(repaired, options.merge_digits)
        holes_capped += _cap_boundary_holes(repaired, options.max_hole_edges)
        _repair_orientation(repaired)
        _clean_mesh(repaired, options.merge_digits)

    max_existing_vertex_displacement = _max_existing_vertex_displacement(original, repaired, options.max_existing_vertex_displacement)
    if max_existing_vertex_displacement > options.max_existing_vertex_displacement:
        raise ValueError(
            "Repair would move or remove original sculpt vertices beyond the museum-scan tolerance "
            f"({max_existing_vertex_displacement:.6g} > {options.max_existing_vertex_displacement:.6g} model units)."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    repaired.export(output_path)

    return RepairReport(
        input_path=str(input_path),
        output_path=str(output_path),
        vertices_before=len(original.vertices),
        faces_before=len(original.faces),
        vertices_after=len(repaired.vertices),
        faces_after=len(repaired.faces),
        components_before=components_before,
        components_after=_component_count(repaired),
        watertight_before=bool(original.is_watertight),
        watertight_after=bool(repaired.is_watertight),
        boundary_edges_before=boundary_edges_before,
        boundary_edges_after=_boundary_edge_count(repaired),
        holes_capped=holes_capped,
        bridges_added=bridges_added,
        max_existing_vertex_displacement=max_existing_vertex_displacement,
        detail_preservation=(
            "museum_scan: preserved source vertex coordinates; no smoothing, subdivision, decimation, remeshing, "
            "inflation, shrink, or topology optimization outside structural defects"
        ),
    )


def _load_mesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(path, force="mesh", process=False)
    if isinstance(loaded, trimesh.Scene):
        meshes = [geometry for geometry in loaded.geometry.values() if isinstance(geometry, trimesh.Trimesh)]
        if not meshes:
            raise ValueError(f"No mesh geometry found in {path}")
        loaded = trimesh.util.concatenate(meshes)
    if not isinstance(loaded, trimesh.Trimesh):
        raise ValueError(f"Unsupported mesh input: {path}")
    if len(loaded.vertices) == 0 or len(loaded.faces) == 0:
        raise ValueError(f"Mesh has no vertices or faces: {path}")
    return loaded


def _clean_mesh(mesh: trimesh.Trimesh, merge_digits: int) -> None:
    mesh.update_faces(mesh.unique_faces())
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.remove_unreferenced_vertices()
    mesh.merge_vertices(digits_vertex=merge_digits)
    mesh.remove_unreferenced_vertices()


def _repair_orientation(mesh: trimesh.Trimesh) -> None:
    trimesh.repair.fix_winding(mesh)
    trimesh.repair.fix_normals(mesh)
    trimesh.repair.fill_holes(mesh)
    if mesh.is_watertight:
        trimesh.repair.fix_inversion(mesh)


def _component_count(mesh: trimesh.Trimesh) -> int:
    if len(mesh.faces) == 0:
        return 0

    counted = mesh.copy()
    counted.merge_vertices()
    counted.remove_unreferenced_vertices()
    if len(counted.faces) == 0:
        return 0

    parent = list(range(len(counted.vertices)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        root_left = find(left)
        root_right = find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    referenced: set[int] = set()
    for a, b, c in counted.faces:
        a = int(a)
        b = int(b)
        c = int(c)
        referenced.update((a, b, c))
        union(a, b)
        union(a, c)

    return len({find(index) for index in referenced})


def _boundary_edge_count(mesh: trimesh.Trimesh) -> int:
    if len(mesh.faces) == 0:
        return 0
    counts = np.bincount(mesh.edges_unique_inverse)
    return int(np.count_nonzero(counts == 1))


def _max_existing_vertex_displacement(
    original: trimesh.Trimesh,
    repaired: trimesh.Trimesh,
    tolerance: float,
) -> float:
    """Verify original sculpt vertices still exist within the allowed tolerance.

    The repair pipeline should not move source vertices at all. Duplicate welding
    may collapse repeated coordinates, and degenerate-face cleanup may remove
    unused references, but every original coordinate must still be represented by
    a repaired vertex within the miniature-scale tolerance.
    """

    if len(original.vertices) == 0 or len(original.faces) == 0 or len(repaired.vertices) == 0:
        return 0.0
    original_referenced = np.unique(np.asarray(original.faces).reshape(-1))
    original_vertices = np.asarray(original.vertices)[original_referenced]
    tolerance = max(float(tolerance), 0.0)
    if tolerance <= 0.0:
        return 0.0 if _all_vertices_exactly_represented(original_vertices, repaired.vertices) else float("inf")

    repaired_keys = {tuple(np.round(vertex / tolerance).astype(np.int64)) for vertex in np.asarray(repaired.vertices)}
    missing = []
    for vertex in original_vertices:
        key = tuple(np.round(vertex / tolerance).astype(np.int64))
        if key not in repaired_keys:
            missing.append(vertex)
            if len(missing) >= 64:
                break
    if not missing:
        return 0.0

    # Slow path only runs for unexpected drift/removal. Keep it chunked so very
    # large miniature scans fail with a useful displacement estimate instead of
    # trying to allocate an NxM distance matrix.
    repaired_vertices = np.asarray(repaired.vertices, dtype=float)
    worst = 0.0
    for vertex in missing:
        best_squared = float("inf")
        for start in range(0, len(repaired_vertices), 50_000):
            chunk = repaired_vertices[start : start + 50_000]
            deltas = chunk - vertex
            distances = np.einsum("ij,ij->i", deltas, deltas)
            best_squared = min(best_squared, float(np.min(distances)))
        worst = max(worst, float(np.sqrt(best_squared)))
    return worst


def _all_vertices_exactly_represented(original_vertices: np.ndarray, repaired_vertices: np.ndarray) -> bool:
    repaired_keys = {tuple(np.asarray(vertex, dtype=float)) for vertex in repaired_vertices}
    return all(tuple(np.asarray(vertex, dtype=float)) in repaired_keys for vertex in original_vertices)


def _cap_boundary_holes(mesh: trimesh.Trimesh, max_hole_edges: int) -> int:
    loops = _boundary_loops(mesh)
    new_faces: list[list[int]] = []
    new_vertices: list[np.ndarray] = []

    for loop in loops:
        if len(loop) < 3 or len(loop) > max_hole_edges:
            continue
        points = mesh.vertices[np.array(loop)]
        center_index = len(mesh.vertices) + len(new_vertices)
        new_vertices.append(points.mean(axis=0))
        for index, vertex in enumerate(loop):
            next_vertex = loop[(index + 1) % len(loop)]
            new_faces.append([vertex, next_vertex, center_index])

    if not new_faces:
        return 0

    mesh.vertices = np.vstack([mesh.vertices, np.array(new_vertices)])
    mesh.faces = np.vstack([mesh.faces, np.array(new_faces, dtype=np.int64)])
    mesh.remove_unreferenced_vertices()
    return len(new_vertices)


def _boundary_loops(mesh: trimesh.Trimesh) -> list[list[int]]:
    if len(mesh.faces) == 0:
        return []

    counts = np.bincount(mesh.edges_unique_inverse)
    boundary_edges = mesh.edges_unique[counts == 1]
    adjacency: dict[int, list[int]] = {}
    for a, b in boundary_edges:
        adjacency.setdefault(int(a), []).append(int(b))
        adjacency.setdefault(int(b), []).append(int(a))

    loops: list[list[int]] = []
    seen_vertices: set[int] = set()
    for start in adjacency:
        if start in seen_vertices or len(adjacency[start]) != 2:
            continue

        loop = [start]
        previous: int | None = None
        current = start
        closed = False

        while True:
            neighbors = adjacency[current]
            if len(neighbors) != 2:
                break
            candidates = [neighbor for neighbor in neighbors if neighbor != previous]
            if not candidates:
                break
            next_vertex = candidates[0]
            if next_vertex == start:
                closed = len(loop) >= 3
                break
            if next_vertex in loop:
                break
            loop.append(next_vertex)
            previous, current = current, next_vertex

        seen_vertices.update(loop)
        if closed:
            loops.append(loop)

    return loops


def _bridge_components(
    mesh: trimesh.Trimesh,
    options: RepairOptions,
) -> tuple[trimesh.Trimesh, int]:
    components = list(mesh.split(only_watertight=False))
    if len(components) <= 1:
        return mesh, 0

    components.sort(key=lambda component: component.area, reverse=True)
    combined = components[0]
    bridges: list[trimesh.Trimesh] = []

    for component in components[1:]:
        start_index, end_index, distance = _closest_vertex_indices(combined.vertices, component.vertices)
        start = combined.vertices[start_index]
        end = component.vertices[end_index]
        if options.max_bridge_distance is not None and distance > options.max_bridge_distance:
            continue
        if distance <= 1e-9:
            combined = trimesh.util.concatenate([combined, component])
            continue

        bridge = _create_anchored_bridge(
            start=start,
            end=end,
            start_neighbor=_anchor_neighbor(combined, start_index),
            end_neighbor=_anchor_neighbor(component, end_index),
            radius=options.connector_radius,
            sections=options.connector_sections,
        )
        bridges.append(bridge)
        combined = trimesh.util.concatenate([combined, bridge, component])

    if not bridges:
        return combined, 0
    return combined, len(bridges)


def _create_anchored_bridge(
    start: np.ndarray,
    end: np.ndarray,
    start_neighbor: np.ndarray | None,
    end_neighbor: np.ndarray | None,
    radius: float,
    sections: int,
) -> trimesh.Trimesh:
    """Create a tapered connector whose anchors weld to existing component edges.

    A plain overlapping cylinder is printable, but it remains a separate mesh
    island topologically. This bridge includes exact endpoint and neighboring
    vertices from both components, then adds one face that shares an existing
    mesh edge on each side. After ``merge_vertices`` runs, the detached piece
    becomes part of the same connected component.
    """

    direction = end - start
    distance = float(np.linalg.norm(direction))
    if distance <= 1e-9:
        return trimesh.Trimesh(vertices=np.array([start]), faces=np.empty((0, 3), dtype=np.int64), process=False)

    sections = max(6, int(sections))
    radius = max(float(radius), 1e-9)
    unit = direction / distance
    basis_u, basis_v = _orthonormal_basis(unit)
    taper_length = min(distance / 3.0, radius * 2.0)
    start_ring_center = start + unit * taper_length
    end_ring_center = end - unit * taper_length

    vertices = [
        np.array(start, dtype=float),
        np.array(end, dtype=float),
        np.array(start_neighbor if start_neighbor is not None else start, dtype=float),
        np.array(end_neighbor if end_neighbor is not None else end, dtype=float),
    ]
    for index in range(sections):
        angle = (2.0 * np.pi * index) / sections
        offset = radius * (np.cos(angle) * basis_u + np.sin(angle) * basis_v)
        vertices.append(start_ring_center + offset)
    for index in range(sections):
        angle = (2.0 * np.pi * index) / sections
        offset = radius * (np.cos(angle) * basis_u + np.sin(angle) * basis_v)
        vertices.append(end_ring_center + offset)

    start_ring_offset = 4
    end_ring_offset = 4 + sections
    faces: list[list[int]] = []
    for index in range(sections):
        next_index = (index + 1) % sections
        start_current = start_ring_offset + index
        start_next = start_ring_offset + next_index
        end_current = end_ring_offset + index
        end_next = end_ring_offset + next_index

        faces.append([0, start_next, start_current])
        faces.append([start_current, start_next, end_next])
        faces.append([start_current, end_next, end_current])
        faces.append([1, end_current, end_next])

    if start_neighbor is not None:
        faces.append([0, 2, start_ring_offset])
    if end_neighbor is not None:
        faces.append([1, end_ring_offset, 3])

    return trimesh.Trimesh(vertices=np.array(vertices), faces=np.array(faces, dtype=np.int64), process=False)


def _orthonormal_basis(unit: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    reference = np.array([1.0, 0.0, 0.0])
    if abs(float(np.dot(unit, reference))) > 0.9:
        reference = np.array([0.0, 1.0, 0.0])
    basis_u = np.cross(unit, reference)
    basis_u /= np.linalg.norm(basis_u)
    basis_v = np.cross(unit, basis_u)
    basis_v /= np.linalg.norm(basis_v)
    return basis_u, basis_v


def _anchor_neighbor(mesh: trimesh.Trimesh, vertex_index: int) -> np.ndarray | None:
    face_hits = np.where(mesh.faces == vertex_index)[0]
    if len(face_hits) == 0:
        return None

    face = mesh.faces[int(face_hits[0])]
    for neighbor_index in face:
        if int(neighbor_index) != vertex_index:
            return mesh.vertices[int(neighbor_index)]
    return None


def _closest_vertex_indices(
    vertices_a: np.ndarray,
    vertices_b: np.ndarray,
    chunk_size: int = 25_000,
) -> tuple[int, int, float]:
    best_a = 0
    best_b = 0
    best_distance_squared = float("inf")

    for start in range(0, len(vertices_a), chunk_size):
        chunk = vertices_a[start : start + chunk_size]
        deltas = chunk[:, None, :] - vertices_b[None, :, :]
        distances = np.einsum("ijk,ijk->ij", deltas, deltas)
        flat_index = int(np.argmin(distances))
        distance_squared = float(distances.flat[flat_index])
        if distance_squared < best_distance_squared:
            local_a, local_b = np.unravel_index(flat_index, distances.shape)
            best_a = start + int(local_a)
            best_b = int(local_b)
            best_distance_squared = distance_squared

    return best_a, best_b, float(np.sqrt(best_distance_squared))
