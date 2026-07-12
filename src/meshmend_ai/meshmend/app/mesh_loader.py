from __future__ import annotations

from pathlib import Path

import trimesh


SUPPORTED_IMPORT_FORMATS = {".stl", ".obj", ".glb", ".ply"}


def load_mesh_file(path: str | Path) -> trimesh.Trimesh:
    """Load an STL/OBJ/GLB/PLY file without destructive preprocessing.

    ``process=False`` is intentional: MeshMend's desktop app is a miniature
    restoration tool, not a generic optimizer. We preserve input vertex/face data
    and only run explicit repair operations chosen by the user.
    """
    mesh_path = Path(path)
    suffix = mesh_path.suffix.lower()
    if suffix not in SUPPORTED_IMPORT_FORMATS:
        raise ValueError(f"Unsupported mesh format '{suffix}'. Supported: {', '.join(sorted(SUPPORTED_IMPORT_FORMATS))}")
    loaded = trimesh.load(mesh_path, force="mesh", process=False)
    if isinstance(loaded, trimesh.Scene):
        meshes = [geometry for geometry in loaded.geometry.values() if isinstance(geometry, trimesh.Trimesh) and len(geometry.faces) > 0]
        if not meshes:
            raise ValueError(f"No mesh geometry found in {mesh_path}")
        loaded = trimesh.util.concatenate(meshes)
    if not isinstance(loaded, trimesh.Trimesh) or len(loaded.faces) == 0:
        raise ValueError(f"No mesh faces found in {mesh_path}")
    return loaded
