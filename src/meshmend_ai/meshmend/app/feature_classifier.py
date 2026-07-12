from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np
import trimesh

from meshmend.app.mesh_analyzer import detail_protection_zones


FEATURE_CLASSES = (
    "armor plate",
    "cloth",
    "leather",
    "organic",
    "weapon",
    "terrain",
    "mechanical",
    "smooth protected area",
    "damaged area",
)


@dataclass(slots=True)
class ClassifiedRegions:
    face_labels: np.ndarray
    counts: dict[str, int]
    blank_faces: np.ndarray
    protected_faces: np.ndarray


def classify_regions(mesh: trimesh.Trimesh) -> ClassifiedRegions:
    """Heuristically classify miniature surface regions for procedural detailing.

    This first implementation is geometry-only. It uses face area, normal
    direction, curvature/normal variance, and proximity to boundaries to choose
    conservative labels that can be refined later with semantic tools.
    """
    face_count = len(mesh.faces)
    labels = np.full(face_count, "armor plate", dtype=object)
    if face_count == 0:
        return ClassifiedRegions(labels, {}, np.zeros(0, dtype=bool), np.zeros(0, dtype=bool))

    protection = detail_protection_zones(mesh)
    protected_faces = np.asarray(protection["protected_face_mask"], dtype=bool)
    face_areas = np.asarray(mesh.area_faces, dtype=float)
    median_area = float(np.median(face_areas)) if len(face_areas) else 0.0
    normals = np.asarray(mesh.face_normals, dtype=float)
    centroids = np.asarray(mesh.triangles_center, dtype=float)
    bounds = np.asarray(mesh.bounds, dtype=float)
    z_span = max(float(bounds[1, 2] - bounds[0, 2]), 1e-9)
    z_norm = (centroids[:, 2] - bounds[0, 2]) / z_span

    labels[protected_faces] = "smooth protected area"
    upward = normals[:, 2] > 0.65
    bottom_third = z_norm < 0.28
    if median_area > 0.0:
        broad = face_areas > median_area * 8.0
    else:
        broad = np.zeros(face_count, dtype=bool)
    labels[upward & bottom_third & broad] = "terrain"
    labels[(np.abs(normals[:, 2]) < 0.25) & broad & ~protected_faces] = "mechanical"
    labels[(face_areas < median_area * 0.7) & ~protected_faces] = "organic"

    edge_counts = np.bincount(mesh.edges_unique_inverse) if len(mesh.faces) else np.array([], dtype=int)
    if len(edge_counts):
        boundary_edges = mesh.edges_unique[edge_counts == 1]
        if len(boundary_edges):
            boundary_vertices = set(int(index) for index in boundary_edges.reshape(-1))
            damaged = np.array([any(int(vertex) in boundary_vertices for vertex in face) for face in mesh.faces], dtype=bool)
            labels[damaged] = "damaged area"

    blank_faces = broad & ~protected_faces
    counts = dict(Counter(str(label) for label in labels))
    return ClassifiedRegions(labels, counts, blank_faces, protected_faces)
