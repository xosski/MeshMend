from __future__ import annotations

from enum import StrEnum

import numpy as np
import trimesh


class SculptTool(StrEnum):
    SMOOTH = "smooth"
    INFLATE = "inflate"
    PINCH = "pinch"
    FLATTEN = "flatten"
    GRAB = "grab"
    CREASE = "crease"
    DETAIL_STAMP = "detail_stamp"


def apply_sculpt_tool(mesh: trimesh.Trimesh, tool: SculptTool, center: tuple[float, float, float], radius: float, strength: float) -> trimesh.Trimesh:
    """Apply an MVP vertex brush operation.

    The desktop UI currently wires mesh-level actions; this function is the local
    sculpting core for future viewport picking and tablet input.
    """
    result = mesh.copy()
    vertices = np.asarray(result.vertices, dtype=float)
    center_arr = np.asarray(center, dtype=float)
    distances = np.linalg.norm(vertices - center_arr, axis=1)
    mask = distances < float(radius)
    if not np.any(mask):
        return result
    falloff = (1.0 - distances[mask] / max(float(radius), 1e-6)) ** 2
    normals = np.asarray(result.vertex_normals, dtype=float)

    if tool == SculptTool.INFLATE:
        vertices[mask] += normals[mask] * (float(strength) * falloff)[:, None]
    elif tool == SculptTool.SMOOTH:
        centroid = vertices[mask].mean(axis=0)
        vertices[mask] += (centroid - vertices[mask]) * (float(strength) * 0.15 * falloff)[:, None]
    elif tool == SculptTool.FLATTEN:
        avg_z = float(vertices[mask, 2].mean())
        vertices[mask, 2] += (avg_z - vertices[mask, 2]) * (float(strength) * falloff)
    elif tool == SculptTool.PINCH:
        vertices[mask] += (center_arr - vertices[mask]) * (float(strength) * 0.1 * falloff)[:, None]
    elif tool == SculptTool.GRAB:
        vertices[mask] += np.array([float(strength), 0.0, 0.0]) * falloff[:, None]
    elif tool in {SculptTool.CREASE, SculptTool.DETAIL_STAMP}:
        vertices[mask] -= normals[mask] * (float(strength) * 0.5 * falloff)[:, None]

    result.vertices = vertices
    return result
