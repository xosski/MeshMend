from __future__ import annotations

import trimesh

from meshmend.app.detail_engine import StudioDetailEngine, StudioDetailResult
from meshmend.app.mesh_repair import RepairSettings, auto_repair_mesh
from meshmend.app.procedural_detail import DetailParameters


def run_studio_sculpt_pass(
    mesh: trimesh.Trimesh,
    *,
    preset_name: str,
    parameters: DetailParameters,
    final_cleanup: bool = True,
) -> StudioDetailResult:
    """Run the first procedural studio sculpt pass.

    The pass detects broad blank surfaces, protects high-curvature/normal-variant
    sculpted features, then adds resin-scale panel lines, rivets, vents, wear,
    and surface texture. It does not smooth or globally remesh the model.
    """
    result = StudioDetailEngine().apply_studio_detail(mesh, preset_name=preset_name, parameters=parameters)
    if not final_cleanup:
        return result
    cleanup = auto_repair_mesh(
        result.mesh,
        RepairSettings(studio_master_mode=True, max_hole_edges=5000, max_existing_vertex_displacement_mm=0.005),
    )
    result.report.actions.extend("final cleanup: " + action for action in cleanup.actions)
    result.report.warnings.extend(cleanup.warnings)
    if cleanup.after is not None:
        result.report.after = cleanup.after
        result.report.added_vertices = max(0, cleanup.after.vertices - result.report.before.vertices)
        result.report.added_faces = max(0, cleanup.after.faces - result.report.before.faces)
    return StudioDetailResult(mesh=cleanup.mesh, report=result.report)
