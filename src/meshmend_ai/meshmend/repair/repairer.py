from __future__ import annotations

from dataclasses import dataclass
import tempfile
from pathlib import Path

import trimesh

from meshmend.core.io import load_mesh
from meshmend.core.report import PrintabilityReport, build_printability_report


@dataclass(slots=True)
class RepairResult:
    mesh: trimesh.Trimesh
    before: PrintabilityReport
    after: PrintabilityReport
    actions: list[str]


def repair_mesh(mesh: trimesh.Trimesh, *, keep_largest_shell: bool = False) -> RepairResult:
    """Run MeshMend's existing deterministic repair engine in-memory.

    The legacy project already has a stronger file-based repair pipeline in
    `repair.py` with boundary-loop capping, normals repair, and optional
    component bridges. This wrapper keeps the new desktop MVP API mesh-native
    while reusing that implementation instead of duplicating it.
    """
    before = build_printability_report(mesh)
    actions: list[str] = ["reused existing MeshMend repair.repair_stl pipeline"]
    repaired = _repair_with_existing_engine(mesh, bridge_disconnected=not keep_largest_shell)
    if keep_largest_shell:
        shells = [part for part in repaired.split(only_watertight=False) if len(part.faces) >= 20]
        if len(shells) > 1:
            shells.sort(key=lambda part: float(part.area), reverse=True)
            repaired = shells[0]
            actions.append("kept largest shell")
    after = build_printability_report(repaired)
    return RepairResult(mesh=repaired, before=before, after=after, actions=actions)


def _repair_with_existing_engine(mesh: trimesh.Trimesh, *, bridge_disconnected: bool) -> trimesh.Trimesh:
    try:
        from repair import RepairOptions, repair_stl
    except Exception:
        # Package import path when MeshMend is installed as meshmend_ai.
        from meshmend_ai.repair import RepairOptions, repair_stl  # type: ignore

    with tempfile.TemporaryDirectory(prefix="meshmend_repair_") as temp_dir:
        temp = Path(temp_dir)
        input_path = temp / "input.stl"
        output_path = temp / "output.stl"
        mesh.export(input_path)
        repair_stl(
            input_path,
            output_path,
            RepairOptions(bridge_disconnected=bridge_disconnected),
        )
        return load_mesh(output_path)
