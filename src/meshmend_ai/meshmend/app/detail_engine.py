from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import trimesh

from meshmend.app.detail_presets import DetailPreset, get_preset
from meshmend.app.edge_bevel import add_crisp_edge_bevels
from meshmend.app.feature_classifier import ClassifiedRegions, classify_regions
from meshmend.app.mesh_analyzer import MeshAnalysis, analyze_mesh, detail_protection_zones
from meshmend.app.procedural_detail import (
    DetailParameters,
    GeneratedDetail,
    apply_micro_displacement,
    generate_battle_damage,
    generate_mechanical_vents,
    generate_panel_lines,
    generate_rivets,
)


@dataclass(slots=True)
class StudioDetailReport:
    preset: str
    before: MeshAnalysis
    after: MeshAnalysis
    classified_counts: dict[str, int]
    blank_faces_detected: int
    protected_faces: int
    texture_vertices: int
    panel_lines: int
    rivets: int
    vents: int
    cracks: int
    bevels: int
    added_vertices: int
    added_faces: int
    actions: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass(slots=True)
class StudioDetailResult:
    mesh: trimesh.Trimesh
    report: StudioDetailReport


class StudioDetailEngine:
    """Procedural miniature detail engine.

    This engine is intentionally not a smoother or remesher. It identifies broad,
    low-detail, unprotected surfaces and adds small resin-printable geometry:
    panel-line lips, rivets, vents, cracks, controlled wear, and low-amplitude
    material texture. Existing high-curvature/high-normal-variance details are
    protected from displacement.
    """

    def apply_studio_detail(
        self,
        mesh: trimesh.Trimesh,
        *,
        preset_name: str = "Sci-fi armor",
        parameters: DetailParameters | None = None,
    ) -> StudioDetailResult:
        params = parameters or DetailParameters()
        preset = get_preset(preset_name)
        before = analyze_mesh(mesh)
        classified = classify_regions(mesh)
        protection = detail_protection_zones(mesh)
        protected_vertices = np.asarray(protection["protected_vertex_mask"], dtype=bool)
        target_faces = _target_detail_faces(mesh, classified, preset)

        actions: list[str] = [
            f"preset: {preset.name}",
            f"blank/low-detail faces detected: {int(np.count_nonzero(classified.blank_faces))}",
            f"protected detail faces skipped: {int(np.count_nonzero(classified.protected_faces))}",
            "global smoothing/remeshing/decimation: skipped",
        ]
        warnings: list[str] = []
        if len(target_faces) == 0:
            warnings.append("No safe low-detail target faces found; protected sculpt detail was left unchanged.")

        textured, texture_vertices = apply_micro_displacement(mesh, target_faces, protected_vertices, params, preset)
        generated = [
            generate_panel_lines(textured, target_faces, params, preset),
            generate_rivets(textured, target_faces, params, preset),
            generate_mechanical_vents(textured, target_faces, params, preset),
            generate_battle_damage(textured, target_faces, params, preset),
        ]
        detail_meshes = [item.mesh for item in generated if len(item.mesh.faces) > 0]
        if detail_meshes:
            detailed = trimesh.util.concatenate([textured, *detail_meshes])
            detailed.remove_unreferenced_vertices()
        else:
            detailed = textured
        detailed, bevels = add_crisp_edge_bevels(
            detailed,
            bevel_width=max(params.minimum_printable_detail_size, 0.04 + params.edge_sharpness * 0.08),
            max_edges=int(24 + params.edge_sharpness * 120),
        )

        panel_lines = sum(item.panel_lines for item in generated)
        rivets = sum(item.rivets for item in generated)
        vents = sum(item.vents for item in generated)
        cracks = sum(item.cracks for item in generated)
        actions.extend(
            [
                f"surface texture vertices displaced: {texture_vertices}",
                f"panel line segments added: {panel_lines}",
                f"rivets/studs added: {rivets}",
                f"mechanical grooves/vents added: {vents}",
                f"cracks/battle damage marks added: {cracks}",
                f"crisp edge bevel strips added: {bevels}",
            ]
        )
        after = analyze_mesh(detailed)
        report = StudioDetailReport(
            preset=preset.name,
            before=before,
            after=after,
            classified_counts=classified.counts,
            blank_faces_detected=int(np.count_nonzero(classified.blank_faces)),
            protected_faces=int(np.count_nonzero(classified.protected_faces)),
            texture_vertices=texture_vertices,
            panel_lines=panel_lines,
            rivets=rivets,
            vents=vents,
            cracks=cracks,
            bevels=bevels,
            added_vertices=max(0, after.vertices - before.vertices),
            added_faces=max(0, after.faces - before.faces),
            actions=actions,
            warnings=warnings,
        )
        detailed.metadata["meshmend_studio_detail"] = {
            "preset": preset.name,
            "panel_lines": panel_lines,
            "rivets": rivets,
            "vents": vents,
            "cracks": cracks,
            "bevels": bevels,
        }
        return StudioDetailResult(mesh=detailed, report=report)


def _target_detail_faces(mesh: trimesh.Trimesh, classified: ClassifiedRegions, preset: DetailPreset) -> np.ndarray:
    if len(mesh.faces) == 0:
        return np.array([], dtype=np.int64)
    labels = classified.face_labels
    if preset.terrain_roughness:
        semantic = labels == "terrain"
    elif preset.organic_texture:
        semantic = labels == "organic"
    elif preset.mechanical_grooves:
        semantic = (labels == "mechanical") | (labels == "armor plate")
    elif preset.cloth_grain:
        semantic = (labels == "cloth") | (labels == "leather") | classified.blank_faces
    else:
        semantic = (labels == "armor plate") | classified.blank_faces
    # Broad blank faces may be bordered by sharp/protected edges (armor plates).
    # Keep the edges/vertices protected from displacement, but still allow
    # additive panel/rivet detail on the blank face interior.
    target = classified.blank_faces | (semantic & ~classified.protected_faces)
    indices = np.flatnonzero(target)
    if len(indices) == 0:
        indices = np.flatnonzero(~classified.protected_faces)
    if len(indices) == 0:
        # Low-poly/general shapes often classify every face as edge-adjacent.
        # For additive detail overlays, choose the largest faces rather than
        # returning a blob-like undecorated primitive. Existing vertices remain
        # protected from displacement by the procedural texture pass.
        indices = np.arange(len(mesh.faces), dtype=np.int64)
    if len(indices) == 0:
        return indices.astype(np.int64)
    areas = np.asarray(mesh.area_faces, dtype=float)[indices]
    order = np.argsort(areas)[::-1]
    return indices[order].astype(np.int64)
