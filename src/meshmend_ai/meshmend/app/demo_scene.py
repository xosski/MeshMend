from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import json

import numpy as np
import trimesh

from meshmend.app.boolean_detail import StampPlacement, stamp_details
from meshmend.app.mesh_exporter import export_mesh_file
from meshmend.app.mesh_repair import RepairSettings, auto_repair_mesh
from meshmend.app.procedural_detail import DetailParameters
from meshmend.app.sculpt_pass import run_studio_sculpt_pass


@dataclass(slots=True)
class DemoSceneResult:
    mesh: trimesh.Trimesh
    output_path: Path | None
    summary: dict[str, object]


def build_studio_detail_demo(output_path: str | Path | None = None) -> DemoSceneResult:
    """Build a working miniature-detail demo scene from a full miniature blockout.

    The input starts as a readable tabletop miniature silhouette: base, boots,
    legs, torso armor, helmet/head, arms, rifle, shoulder pads, and backpack.
    The output then gets sculpt-pass panel lines, rivets, bevels, scratches/wear,
    vents, skull, seal, bolts, cracks, cables, and armor plates before export.
    """
    base = _miniature_blockout()
    sculpted = run_studio_sculpt_pass(
        base,
        preset_name="sci-fi armor",
        parameters=DetailParameters(
            detail_strength=0.62,
            rivet_density=0.78,
            panel_line_depth=0.09,
            battle_damage_amount=0.42,
            surface_texture_strength=0.055,
            edge_sharpness=0.85,
            minimum_printable_detail_size=0.05,
        ),
    )
    stamped = stamp_details(sculpted.mesh, _demo_stamps())
    final_repair = auto_repair_mesh(
        stamped,
        RepairSettings(studio_master_mode=True, max_hole_edges=5000, max_existing_vertex_displacement_mm=0.005),
    )
    final_mesh = final_repair.mesh
    output = Path(output_path) if output_path is not None else None
    if output is not None:
        export_mesh_file(final_mesh, output)
        summary_path = output.with_suffix(output.suffix + ".studio_demo.json")
    else:
        summary_path = None
    summary = {
        "input": "full sci-fi infantry miniature blockout",
        "faces_before": sculpted.report.before.faces,
        "faces_after_sculpt_pass": sculpted.report.after.faces,
        "faces_after_stamps": int(len(stamped.faces)),
        "faces_after_final_cleanup": int(len(final_mesh.faces)),
        "panel_lines": sculpted.report.panel_lines,
        "rivets": sculpted.report.rivets,
        "vents": sculpted.report.vents,
        "cracks": sculpted.report.cracks,
        "bevels": sculpted.report.bevels,
        "stamps": [placement.kind for placement in _demo_stamps()],
        "final_cleanup_actions": final_repair.actions,
        "note": "Demo uses procedural additive detail; no smoothing/remeshing/decimation dependency.",
    }
    if summary_path is not None:
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return DemoSceneResult(mesh=final_mesh, output_path=output, summary=summary)


def build_studio_armor_plate_demo(output_path: str | Path | None = None) -> DemoSceneResult:
    """Legacy focused material/detail test for one armor plate."""
    base = _armor_plate_blockout()
    sculpted = run_studio_sculpt_pass(
        base,
        preset_name="sci-fi armor",
        parameters=DetailParameters(
            detail_strength=0.62,
            rivet_density=0.78,
            panel_line_depth=0.09,
            battle_damage_amount=0.42,
            surface_texture_strength=0.055,
            edge_sharpness=0.85,
            minimum_printable_detail_size=0.05,
        ),
    )
    stamped = stamp_details(sculpted.mesh, _demo_stamps())
    output = Path(output_path) if output_path is not None else None
    summary = {
        "input": "plain armor plate/cube blockout",
        "faces_before": sculpted.report.before.faces,
        "faces_after_sculpt_pass": sculpted.report.after.faces,
        "faces_after_stamps": int(len(stamped.faces)),
        "panel_lines": sculpted.report.panel_lines,
        "rivets": sculpted.report.rivets,
        "vents": sculpted.report.vents,
        "cracks": sculpted.report.cracks,
        "bevels": sculpted.report.bevels,
        "stamps": [placement.kind for placement in _demo_stamps()],
        "note": "Armor plate detail material test; use build_studio_detail_demo for a full miniature silhouette.",
    }
    if output is not None:
        export_mesh_file(stamped, output)
        output.with_suffix(output.suffix + ".studio_demo.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return DemoSceneResult(mesh=stamped, output_path=output, summary=summary)


def _miniature_blockout() -> trimesh.Trimesh:
    """Create a non-blocky 32mm sci-fi infantry miniature base silhouette.

    This is still procedural and local, but it provides an actual miniature form
    before surface detailing: separate limbs, armor volumes, helmet, backpack,
    rifle, shoulder pads, scenic base, and readable stance.
    """
    parts: list[trimesh.Trimesh] = []
    base = trimesh.creation.cylinder(radius=13.0, height=2.2, sections=96)
    base.apply_translation((0.0, 0.0, 1.1))
    parts.append(base)

    # Scenic rubble breaks the flat base silhouette.
    for index, (x, y, sx, sy, sz) in enumerate(((-6, -4, 3.2, 2.0, 0.9), (5, 4, 2.8, 1.7, 0.7), (2, -6, 2.0, 1.2, 0.6))):
        rock = trimesh.creation.icosphere(subdivisions=1, radius=1.0)
        rock.apply_scale((sx, sy, sz))
        rock.apply_translation((x, y, 2.2 + index * 0.12))
        parts.append(rock)

    # Boots, legs, pelvis.
    parts.extend([
        _box((2.7, 2.1, 1.5), (-2.0, -0.4, 3.0)),
        _box((2.7, 2.1, 1.5), (2.0, 0.5, 3.0)),
        _capsule_between((-2.0, -0.3, 3.5), (-1.7, -0.2, 10.0), 0.95),
        _capsule_between((2.0, 0.4, 3.5), (1.7, 0.2, 10.0), 0.95),
        _box((4.8, 2.5, 2.0), (0.0, 0.0, 10.2)),
    ])

    torso = trimesh.creation.icosphere(subdivisions=2, radius=3.5)
    torso.apply_scale((1.05, 0.72, 1.35))
    torso.apply_translation((0.0, 0.0, 15.0))
    parts.append(torso)
    chest_plate = _box((5.7, 0.65, 4.1), (0.0, -2.45, 15.2))
    parts.append(chest_plate)

    # Helmet/head, visor, mouth grille.
    helmet = trimesh.creation.icosphere(subdivisions=2, radius=1.9)
    helmet.apply_scale((0.9, 0.78, 1.05))
    helmet.apply_translation((0.0, -0.15, 20.5))
    parts.append(helmet)
    parts.append(_box((1.8, 0.25, 0.35), (0.0, -1.55, 20.8)))
    for x in (-0.42, 0.0, 0.42):
        parts.append(_box((0.14, 0.22, 0.65), (x, -1.62, 20.0)))

    # Shoulders, arms, rifle across the body.
    for side in (-1.0, 1.0):
        shoulder = trimesh.creation.uv_sphere(radius=1.65, count=[24, 12])
        shoulder.apply_scale((1.35, 0.85, 0.72))
        shoulder.apply_translation((side * 4.0, -0.1, 17.0))
        parts.append(shoulder)
        parts.append(_capsule_between((side * 4.7, -0.3, 16.0), (side * 5.3, -1.5, 12.2), 0.58))
        parts.append(_capsule_between((side * 5.3, -1.5, 12.2), (side * 2.9, -2.5, 12.0), 0.48))
        hand = trimesh.creation.uv_sphere(radius=0.55, count=[12, 6])
        hand.apply_scale((0.8, 0.55, 0.65))
        hand.apply_translation((side * 2.8, -2.7, 12.0))
        parts.append(hand)

    parts.append(_box((7.8, 0.65, 0.75), (1.8, -3.05, 12.5)))
    parts.append(_capsule_between((5.4, -3.05, 12.55), (9.8, -3.05, 12.55), 0.28))
    parts.append(_box((1.1, 0.55, 2.0), (-0.8, -3.1, 11.5)))

    # Backpack/reactor and vents.
    parts.append(_box((3.8, 1.8, 5.0), (0.0, 2.5, 15.8)))
    for x in (-0.9, 0.9):
        parts.append(_capsule_between((x, 3.2, 13.2), (x, 3.2, 18.2), 0.32))

    mesh = trimesh.util.concatenate(parts)
    mesh.metadata["units"] = "mm"
    mesh.metadata["meshmend_demo_input"] = "full miniature blockout"
    return mesh


def _armor_plate_blockout() -> trimesh.Trimesh:
    body = trimesh.creation.box(extents=(18.0, 10.0, 3.2))
    raised_plate = trimesh.creation.box(extents=(12.8, 6.6, 0.45))
    raised_plate.apply_translation((0.0, 0.0, 1.82))
    side_plate_l = trimesh.creation.box(extents=(1.2, 7.4, 0.55))
    side_plate_l.apply_translation((-7.4, 0.0, 1.92))
    side_plate_r = side_plate_l.copy()
    side_plate_r.apply_translation((14.8, 0.0, 0.0))
    mesh = trimesh.util.concatenate([body, raised_plate, side_plate_l, side_plate_r])
    mesh.metadata["units"] = "mm"
    mesh.metadata["meshmend_demo_input"] = "broad armor plate blockout"
    return mesh


def _box(extents: tuple[float, float, float], center: tuple[float, float, float]) -> trimesh.Trimesh:
    mesh = trimesh.creation.box(extents=extents)
    mesh.apply_translation(center)
    return mesh


def _capsule_between(start: tuple[float, float, float], end: tuple[float, float, float], radius: float) -> trimesh.Trimesh:
    start_array = np.asarray(start, dtype=float)
    end_array = np.asarray(end, dtype=float)
    segment = end_array - start_array
    length = float(np.linalg.norm(segment))
    if length <= 1e-9:
        sphere = trimesh.creation.uv_sphere(radius=radius, count=[16, 8])
        sphere.apply_translation(start_array)
        return sphere
    capsule = trimesh.creation.capsule(radius=radius, height=length, count=[16, 8])
    direction = segment / length
    z_axis = np.array([0.0, 0.0, 1.0])
    axis = np.cross(z_axis, direction)
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm > 1e-12:
        axis /= axis_norm
        angle = float(np.arccos(np.clip(np.dot(z_axis, direction), -1.0, 1.0)))
        capsule.apply_transform(trimesh.transformations.rotation_matrix(angle, axis))
    capsule.apply_translation((start_array + end_array) * 0.5)
    return capsule


def _demo_stamps() -> list[StampPlacement]:
    up = (0.0, 0.0, 1.0)
    front = (0.0, -1.0, 0.0)
    back = (0.0, 1.0, 0.0)
    placements = [
        StampPlacement("skull", (0.0, -2.85, 15.9), front, 1.35),
        StampPlacement("seal", (-2.9, -2.85, 13.8), front, 1.15, -8.0),
        StampPlacement("plate", (2.7, -2.82, 14.2), front, 1.25, 10.0),
        StampPlacement("crack", (1.4, -2.88, 16.7), front, 1.5, 24.0),
        StampPlacement("vent", (-0.9, 3.55, 16.9), back, 1.05),
        StampPlacement("vent", (0.9, 3.55, 16.9), back, 1.05),
        StampPlacement("cable", (-1.8, -3.22, 12.9), front, 1.45, 0.0),
        StampPlacement("crack", (-5.0, -0.95, 17.1), (-1.0, -0.15, 0.0), 1.1, -18.0),
        StampPlacement("crack", (5.0, -0.95, 17.1), (1.0, -0.15, 0.0), 1.1, 18.0),
    ]
    for x in (-2.4, -1.2, 1.2, 2.4):
        placements.append(StampPlacement("bolt", (x, -2.95, 17.6), front, 0.72))
        placements.append(StampPlacement("bolt", (x, -2.95, 12.8), front, 0.72))
    for x in (-4.4, 4.4):
        for z in (16.6, 17.4):
            placements.append(StampPlacement("bolt", (x, -1.15, z), front, 0.7))
    for x in (-8.6, -6.2, 6.2, 8.6):
        placements.append(StampPlacement("bolt", (x, -7.8, 2.25), up, 0.8))
    return placements
