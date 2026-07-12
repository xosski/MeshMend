from __future__ import annotations

import numpy as np
import trimesh


def add_crisp_edge_bevels(
    mesh: trimesh.Trimesh,
    *,
    bevel_width: float = 0.08,
    sharp_angle_degrees: float = 35.0,
    max_edges: int = 96,
) -> tuple[trimesh.Trimesh, int]:
    """Add tiny raised bevel facets along hard edges.

    This is not global remeshing. The first implementation creates additive
    bevel-highlight strips on sharp edges so blockouts read as miniature armor
    plates with crisp chamfers after export.
    """
    if len(mesh.face_adjacency) == 0:
        return mesh.copy(), 0
    angles = np.asarray(mesh.face_adjacency_angles, dtype=float)
    sharp = np.flatnonzero(angles >= np.deg2rad(sharp_angle_degrees))
    if len(sharp) == 0:
        return mesh.copy(), 0
    strips = []
    edges = np.asarray(mesh.face_adjacency_edges, dtype=int)[sharp[:max_edges]]
    face_pairs = np.asarray(mesh.face_adjacency, dtype=int)[sharp[:max_edges]]
    for edge, pair in zip(edges, face_pairs):
        a, b = np.asarray(mesh.vertices[edge], dtype=float)
        direction = b - a
        length = float(np.linalg.norm(direction))
        if length <= bevel_width * 2.0:
            continue
        direction /= length
        normal = np.asarray(mesh.face_normals[pair[0]] + mesh.face_normals[pair[1]], dtype=float)
        norm = float(np.linalg.norm(normal))
        if norm <= 1e-12:
            normal = np.asarray(mesh.face_normals[pair[0]], dtype=float)
            norm = max(float(np.linalg.norm(normal)), 1e-12)
        normal /= norm
        side = np.cross(normal, direction)
        side /= max(float(np.linalg.norm(side)), 1e-12)
        strips.append(_bevel_strip(a, b, normal, side, bevel_width))
    if not strips:
        return mesh.copy(), 0
    combined = trimesh.util.concatenate([mesh.copy(), *strips])
    combined.remove_unreferenced_vertices()
    combined.metadata["meshmend_edge_bevels"] = len(strips)
    return combined, len(strips)


def _bevel_strip(a: np.ndarray, b: np.ndarray, normal: np.ndarray, side: np.ndarray, width: float) -> trimesh.Trimesh:
    inset = width * 0.35
    height = width * 0.45
    vertices = np.array(
        [
            a + side * inset + normal * height,
            b + side * inset + normal * height,
            b - side * inset + normal * height,
            a - side * inset + normal * height,
        ],
        dtype=float,
    )
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
