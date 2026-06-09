from __future__ import annotations

from pathlib import Path
from typing import Iterable

import trimesh


SUPPORTED_FORMATS = {".stl", ".obj", ".glb", ".ply"}


def load_mesh(path: str | Path) -> trimesh.Trimesh:
    """Load STL/OBJ/GLB/PLY as one editable trimesh mesh."""
    mesh_path = Path(path)
    if mesh_path.suffix.lower() not in SUPPORTED_FORMATS:
        raise ValueError(f"Unsupported mesh format: {mesh_path.suffix}. Supported: {sorted(SUPPORTED_FORMATS)}")
    loaded = trimesh.load(mesh_path, force="mesh", process=False)
    if isinstance(loaded, trimesh.Scene):
        geometries: Iterable[trimesh.Trimesh] = [g for g in loaded.geometry.values() if hasattr(g, "faces") and len(g.faces) > 0]
        loaded = trimesh.util.concatenate(tuple(geometries))
    if not isinstance(loaded, trimesh.Trimesh) or len(loaded.faces) == 0:
        raise ValueError(f"{mesh_path} did not contain mesh geometry")
    return loaded


def save_mesh(mesh: trimesh.Trimesh, path: str | Path) -> Path:
    """Export a mesh to STL/OBJ/GLB/PLY."""
    output_path = Path(path)
    if output_path.suffix.lower() not in SUPPORTED_FORMATS:
        raise ValueError(f"Unsupported export format: {output_path.suffix}. Supported: {sorted(SUPPORTED_FORMATS)}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(output_path)
    return output_path
