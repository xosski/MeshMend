from __future__ import annotations

import math

import numpy as np
import trimesh


def auto_scale_to_height(mesh: trimesh.Trimesh, height_mm: float) -> trimesh.Trimesh:
    """Scale a miniature so its Z height matches the chosen heroic scale."""
    result = mesh.copy()
    extents = np.asarray(result.extents, dtype=float)
    current_height = float(extents[2])
    if current_height <= 1e-8:
        raise ValueError("Cannot scale mesh with zero height")
    result.apply_scale(float(height_mm) / current_height)
    result.metadata["units"] = "mm"
    result.metadata["target_height_mm"] = float(height_mm)
    return result


def add_circular_base(mesh: trimesh.Trimesh, radius_mm: float | None = None, height_mm: float = 2.0) -> trimesh.Trimesh:
    """Add a printable circular tabletop base under the model."""
    result = mesh.copy()
    extents = np.asarray(result.extents, dtype=float)
    radius = float(radius_mm) if radius_mm is not None else max(12.5, float(max(extents[0], extents[1])) * 0.62)
    base = trimesh.creation.cylinder(radius=radius, height=float(height_mm), sections=96)
    model_min_z = float(result.bounds[0][2])
    base.apply_translation([0.0, 0.0, model_min_z - float(height_mm) * 0.5])
    combined = trimesh.util.concatenate([result, base])
    combined.metadata.update(result.metadata)
    combined.metadata["meshmend_base"] = {"shape": "circular", "radius_mm": radius, "height_mm": float(height_mm)}
    return combined


def decimate_mesh(mesh: trimesh.Trimesh, target_faces: int) -> trimesh.Trimesh:
    """Reduce polygon count using the best local simplifier available."""
    target_faces = max(4, int(target_faces))
    if len(mesh.faces) <= target_faces:
        return mesh.copy()
    # trimesh delegates to optional backends when installed. If unavailable, keep
    # the original mesh rather than silently damaging topology with naive sampling.
    for method_name in ("simplify_quadric_decimation", "simplify_quadratic_decimation"):
        method = getattr(mesh, method_name, None)
        if method is None:
            continue
        try:
            simplified = method(target_faces)
            if isinstance(simplified, trimesh.Trimesh) and len(simplified.faces) > 0:
                simplified.metadata.update(mesh.metadata)
                return simplified
        except Exception:
            continue
    return mesh.copy()


def remesh_subdivide(mesh: trimesh.Trimesh, target_faces: int) -> trimesh.Trimesh:
    """Local remesh/detail densification via midpoint subdivision."""
    target_faces = max(int(target_faces), len(mesh.faces))
    result = mesh.copy()
    while len(result.faces) < target_faces:
        next_face_count = len(result.faces) * 4
        if next_face_count > target_faces * 4.0:
            break
        vertices, faces = trimesh.remesh.subdivide(result.vertices, result.faces)
        result = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    result.metadata.update(mesh.metadata)
    return result


def transform_mesh(mesh: trimesh.Trimesh, translate=(0.0, 0.0, 0.0), rotate_z_degrees: float = 0.0, scale: float = 1.0) -> trimesh.Trimesh:
    result = mesh.copy()
    if scale != 1.0:
        result.apply_scale(float(scale))
    if rotate_z_degrees:
        result.apply_transform(trimesh.transformations.rotation_matrix(math.radians(rotate_z_degrees), [0, 0, 1]))
    result.apply_translation(np.asarray(translate, dtype=float))
    return result
