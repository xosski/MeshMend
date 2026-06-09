from __future__ import annotations

import json
from pathlib import Path

import trimesh

from meshmend.core.io import save_mesh
from meshmend.core.report import build_printability_report


def export_slicer_ready(mesh: trimesh.Trimesh, output_path: str | Path, *, write_report: bool = True) -> Path:
    """Export a local slicer-ready mesh plus an adjacent printability report."""
    output = save_mesh(mesh, output_path)
    if write_report:
        report_path = output.with_suffix(output.suffix + ".printability.json")
        report_path.write_text(json.dumps(build_printability_report(mesh).to_dict(), indent=2), encoding="utf-8")
    return output
