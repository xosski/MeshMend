from __future__ import annotations

import json
from pathlib import Path

import trimesh

from meshmend.app.mesh_analyzer import analyze_mesh


SUPPORTED_EXPORT_FORMATS = {".stl", ".obj"}


def export_mesh_file(mesh: trimesh.Trimesh, path: str | Path, *, write_analysis: bool = True) -> Path:
    """Export a repaired miniature mesh as STL or OBJ.

    The exporter writes geometry as-is. It does not decimate, smooth, scale, or
    reprocess the model during export.
    """
    output_path = Path(path)
    suffix = output_path.suffix.lower()
    if suffix not in SUPPORTED_EXPORT_FORMATS:
        raise ValueError("MeshMend MVP exports STL and OBJ. Choose a .stl or .obj output path.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(output_path)
    if write_analysis:
        report_path = output_path.with_suffix(output_path.suffix + ".meshmend_analysis.json")
        report_path.write_text(json.dumps(analyze_mesh(mesh).to_dict(), indent=2), encoding="utf-8")
    return output_path
