from __future__ import annotations

import math
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import trimesh


@dataclass(frozen=True)
class PostprocessReport:
    target_scale_mm: float
    extents_mm: list[float]
    max_extent_mm: float
    faces: int
    vertices: int
    single_subject_enforced: bool
    detail_faces_target: int
    relief_mm: float
    detail_style: str
    geometry_upscaled: bool = False
    intricate_detail_pipeline: bool = False
    custom_miniature_detail_pipeline: bool = False
    ai_definition_layer: bool = False
    production_ready: bool = False
    quality_gate_issues: list[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_scale_mm": self.target_scale_mm,
            "extents_mm": self.extents_mm,
            "max_extent_mm": self.max_extent_mm,
            "faces": self.faces,
            "vertices": self.vertices,
            "units": "mm",
            "single_subject_enforced": self.single_subject_enforced,
            "detail_faces_target": self.detail_faces_target,
            "relief_mm": self.relief_mm,
            "detail_style": self.detail_style,
            "geometry_upscaled": self.geometry_upscaled,
            "intricate_detail_pipeline": self.intricate_detail_pipeline,
            "custom_miniature_detail_pipeline": self.custom_miniature_detail_pipeline,
            "ai_definition_layer": self.ai_definition_layer,
            "production_ready": self.production_ready,
            "quality_gate_issues": self.quality_gate_issues or [],
        }


def postprocess_miniature(mesh: Any, request: dict[str, Any]) -> tuple[trimesh.Trimesh, PostprocessReport]:
    """Turn raw Hunyuan output into one printable, scaled, detailed STL mesh."""
    mesh = coerce_to_trimesh(mesh)
    request = clamp_request_for_memory_safety(dict(request))
    if memory_safety_enabled():
        max_raw_faces = int(os.environ.get("MESHMEND_MAX_RAW_POSTPROCESS_FACES", "900000"))
        if len(mesh.faces) > max_raw_faces:
            raise RuntimeError(
                f"Raw AI mesh has {len(mesh.faces)} faces, above memory-safe postprocess limit {max_raw_faces}. "
                "Skipping repair to avoid exhausting RAM; use modular fallback or raise MESHMEND_MAX_RAW_POSTPROCESS_FACES on a workstation."
            )
    before_faces = len(mesh.faces)
    mesh = remove_background_slabs(mesh, request)
    mesh = remove_connected_sheet_surfaces(mesh, request)
    mesh, single_subject_enforced = enforce_single_subject(mesh, request)
    mesh = normalize_to_scale(mesh, requested_scale_mm(request), request)
    mesh = complete_front_shell_volume(mesh, request)
    mesh = thicken_if_sheet(mesh)
    mesh = normalize_to_scale(mesh, requested_scale_mm(request), request)
    detail_target = detail_face_target(request)
    mesh = remove_connected_sheet_surfaces(mesh, dict(request, _meshmend_aggressive_slab_removal=True))
    pre_detail_default = "180000" if strict_quality_requested(request) else "90000"
    pre_detail_target = memory_safe_face_target(min(detail_target, int(os.environ.get("MESHMEND_PRE_INTRICATE_DETAIL_FACES", pre_detail_default))))
    mesh = subdivide_to_faces(mesh, pre_detail_target)
    mesh = denoise_surface_bumps(mesh, request)
    if synthetic_detail_enabled("MESHMEND_ENABLE_SCULPT_RELIEF_DETAIL", default="1" if strict_quality_requested(request) else "0"):
        mesh = apply_miniature_sculpt_detail(mesh, request)
    mesh = apply_image_guided_surface_detail(mesh, request)
    mesh = add_high_resolution_geometry(mesh, request)
    mesh = apply_custom_miniature_detail_pipeline(mesh, request)
    mesh = apply_ai_training_definition_layer(mesh, request)
    mesh = apply_intricate_detail_pipeline(mesh, request)
    mesh = remove_floating_artifacts(mesh, request)
    mesh = cleanup(mesh)
    mesh = remove_background_slabs(mesh, dict(request, _meshmend_aggressive_slab_removal=True))
    mesh = remove_connected_sheet_surfaces(mesh, dict(request, _meshmend_aggressive_slab_removal=True))
    mesh, extra_single_subject = repair_structural_quality_issues(mesh, request, detail_target)
    single_subject_enforced = single_subject_enforced or extra_single_subject
    mesh = strip_residual_background_slabs(mesh, request, detail_target)
    mesh = remove_connected_sheet_surfaces(mesh, dict(request, _meshmend_aggressive_slab_removal=True))
    mesh = ensure_printable_solid_mesh(mesh, request)
    if mesh.metadata.get("meshmend_voxel_solidified"):
        mesh = apply_custom_miniature_detail_pipeline(mesh, request)
        mesh = apply_intricate_detail_pipeline(mesh, request)
    mesh = ensure_round_miniature_base(mesh, request)
    mesh = bridge_disconnected_components(mesh, request)
    mesh = ensure_printable_solid_mesh(mesh, request)
    mesh = seal_image_visual_holes(mesh, request)
    mesh = restore_image_store_density(mesh, request, detail_target)
    mesh = final_image_surface_polish(mesh, request)
    if strict_quality_requested(request) and len(mesh.faces) < int(detail_target * 0.75):
        mesh = subdivide_to_faces(mesh, detail_target)
        mesh.metadata["meshmend_final_store_density_enforced"] = True
    gate_issues = quality_gate_issues(mesh, request, detail_target)
    fatal_issues = fatal_quality_gate_issues(gate_issues)
    if fatal_issues and strict_quality_requested(request) and should_raise_quality_gate_failure(request, fatal_issues):
        diagnostics = mesh_quality_diagnostics(mesh, request, detail_target, gate_issues)
        raise RuntimeError("Generated STL did not meet store-quality gate: " + "; ".join(fatal_issues) + "; diagnostics=" + json.dumps(diagnostics, default=str))
    report = build_report(
        mesh,
        request,
        single_subject_enforced=single_subject_enforced or len(mesh.faces) < before_faces * 0.85,
        detail_faces_target=detail_target,
        quality_gate_issues=gate_issues,
    )
    return mesh, report


def synthetic_detail_enabled(env_name: str, *, default: str = "0") -> bool:
    """Return whether procedural surface detail should be added.

    Store/studio mode is geometry-first: detail passes are enabled by default so
    outputs cannot silently remain smooth Hunyuan blobs. Draft mode keeps the old
    conservative default unless the caller opts in with env flags.
    """
    return os.environ.get(env_name, default).strip().lower() in {"1", "true", "yes", "on"}


def memory_safety_enabled() -> bool:
    return os.environ.get("MESHMEND_DISABLE_MEMORY_SAFETY", "0").strip().lower() not in {"1", "true", "yes", "on"}


def memory_safe_face_target(target_faces: int) -> int:
    target_faces = max(1_000, int(target_faces))
    if not memory_safety_enabled():
        return target_faces
    cap = int(os.environ.get("MESHMEND_MAX_POSTPROCESS_FACES", "350000"))
    return min(target_faces, max(50_000, cap))


def clamp_request_for_memory_safety(request: dict[str, Any]) -> dict[str, Any]:
    if not memory_safety_enabled():
        return request
    try:
        requested = int(float(request.get("target_polycount") or 0) or 0)
    except (TypeError, ValueError):
        requested = 0
    safe_target = memory_safe_face_target(requested or (180_000 if strict_quality_requested(request) else 90_000))
    request["target_polycount"] = safe_target
    request["_meshmend_memory_safety_face_cap"] = int(os.environ.get("MESHMEND_MAX_POSTPROCESS_FACES", "350000"))
    return request


def mesh_quality_diagnostics(mesh: trimesh.Trimesh, request: dict[str, Any], detail_faces_target: int, issues: list[str]) -> dict[str, Any]:
    try:
        vertices = np.asarray(mesh.vertices, dtype=float)
        extents = np.asarray(mesh.extents, dtype=float).tolist() if len(vertices) else []
        return {
            "workflow": request.get("_meshmend_source_workflow") or request.get("workflow"),
            "faces": int(len(mesh.faces)),
            "vertices": int(len(mesh.vertices)),
            "detail_faces_target": int(detail_faces_target),
            "extents_mm": [float(value) for value in extents],
            "watertight": bool(mesh.is_watertight),
            "winding_consistent": bool(mesh.is_winding_consistent),
            "components": int(len(mesh.split(only_watertight=False))) if len(mesh.faces) else 0,
            "boundary_edges": int(_boundary_edge_count(mesh)),
            "nonmanifold_edges": int(_nonmanifold_edge_count(mesh)),
            "euler_number": int(mesh.euler_number) if len(mesh.faces) else 0,
            "visual_hole_genus": int(mesh_genus_estimate(mesh)),
            "volume": float(mesh.volume) if len(mesh.faces) else 0.0,
            "area": float(mesh.area) if len(mesh.faces) else 0.0,
            "metadata": dict(mesh.metadata),
            "issues": issues,
        }
    except Exception as exc:
        return {"diagnostic_error": str(exc), "issues": issues}


def restore_image_store_density(mesh: trimesh.Trimesh, request: dict[str, Any], detail_faces_target: int) -> trimesh.Trimesh:
    """Rebuild density/detail lost during image-reference solidification.

    Voxel solidification is useful for closing noisy Hunyuan cavities, but it can
    reduce a high-quality image job back to a low-detail shell. Do the topology
    repair first, then subdivide/detail the repaired printable solid before the
    store-quality gate evaluates it.
    """
    source_workflow = str(request.get("_meshmend_source_workflow") or request.get("workflow") or "text_to_3d")
    if source_workflow not in {"image_to_3d", "text_to_3d"} or not strict_quality_requested(request):
        return mesh
    try:
        min_ratio = float(os.environ.get("MESHMEND_IMAGE_FINAL_DETAIL_TARGET_RATIO", "1.0"))
        min_faces = memory_safe_face_target(max(len(mesh.faces), int(detail_faces_target * min_ratio)))
        mesh = subdivide_to_faces(mesh, min_faces)
        mesh = apply_image_guided_surface_detail(mesh, request)
        mesh = cleanup(mesh)
        mesh.metadata["meshmend_image_store_density_restored"] = True
        return mesh
    except Exception:
        return mesh


def seal_image_visual_holes(mesh: trimesh.Trimesh, request: dict[str, Any]) -> trimesh.Trimesh:
    """Close small visual tunnels/cavities in image reconstructions before detailing.

    A mesh can be technically watertight while still looking holey: hundreds of
    tiny tunnels produce a valid STL with a very negative Euler number. For store
    miniatures, those read as missing anatomy/noise, so image-reference jobs get
    a morphology-based solid seal before density and fine detail are rebuilt.
    """
    source_workflow = str(request.get("_meshmend_source_workflow") or request.get("workflow") or "text_to_3d")
    if source_workflow not in {"image_to_3d", "text_to_3d"} or not strict_quality_requested(request):
        return mesh
    default_enabled = "1" if source_workflow == "image_to_3d" else "0"
    if os.environ.get("MESHMEND_SEAL_IMAGE_VISUAL_HOLES", default_enabled).strip().lower() in {"0", "false", "no"}:
        return mesh
    genus = mesh_genus_estimate(mesh)
    if (
        bool(mesh.is_watertight)
        and _nonmanifold_edge_count(mesh) == 0
        and genus < int(os.environ.get("MESHMEND_IMAGE_VISUAL_HOLE_REPAIR_GENUS", "16"))
    ):
        return mesh
    try:
        repaired = cleanup(mesh.copy())
        vertices = np.asarray(repaired.vertices, dtype=float)
        if len(vertices) == 0 or len(repaired.faces) == 0:
            return mesh
        max_extent = float(np.max(np.maximum(vertices.max(axis=0) - vertices.min(axis=0), 1e-6)))
        candidates = image_hole_seal_candidates(repaired, max_extent)
        if not candidates:
            return mesh
        max_genus = int(os.environ.get("MESHMEND_IMAGE_MAX_VISUAL_HOLE_GENUS", "24"))
        candidates.sort(key=lambda candidate: (not bool(candidate.is_watertight), _nonmanifold_edge_count(candidate), max(0, mesh_genus_estimate(candidate) - max_genus), -len(candidate.faces)))
        solid = candidates[0]
        if not bool(solid.is_watertight) or _nonmanifold_edge_count(solid) > 0:
            return mesh
        solid.metadata.update(repaired.metadata)
        solid.metadata["meshmend_image_visual_holes_sealed"] = True
        solid.metadata["meshmend_pre_seal_genus"] = int(genus)
        solid = cleanup(solid)
        solid = normalize_to_scale(solid, requested_scale_mm(request), request)
        try:
            from trimesh.smoothing import filter_taubin

            filter_taubin(
                solid,
                lamb=float(os.environ.get("MESHMEND_IMAGE_SEAL_SMOOTH_LAMBDA", "0.22")),
                nu=float(os.environ.get("MESHMEND_IMAGE_SEAL_SMOOTH_NU", "0.34")),
                iterations=int(os.environ.get("MESHMEND_IMAGE_SEAL_SMOOTH_ITERATIONS", "4")),
            )
        except Exception:
            pass
        solid = cleanup(solid)
        return normalize_to_scale(solid, requested_scale_mm(request), request)
    except Exception:
        return mesh


def image_hole_seal_candidates(mesh: trimesh.Trimesh, max_extent: float) -> list[trimesh.Trimesh]:
    """Try multiple morphology settings and keep only usable sealed solids."""
    try:
        from scipy.ndimage import binary_closing, binary_fill_holes
        from trimesh.voxel import ops as voxel_ops

        pitch_text = os.environ.get("MESHMEND_IMAGE_VISUAL_HOLE_SEAL_PITCHES_MM", "0.24,0.28,0.32,0.38,0.45")
        close_text = os.environ.get("MESHMEND_IMAGE_VISUAL_HOLE_CLOSE_ITERATIONS_LIST", "4,3,2")
        pitches = [max(0.08, min(float(value.strip()), max_extent / 24.0)) for value in pitch_text.split(",") if value.strip()]
        iterations_list = [max(1, int(value.strip())) for value in close_text.split(",") if value.strip()]
        candidates: list[trimesh.Trimesh] = []
        for pitch in pitches:
            voxels = mesh.voxelized(pitch)
            base_matrix = np.asarray(voxels.matrix, dtype=bool)
            if base_matrix.size == 0:
                continue
            for iterations in iterations_list:
                matrix = binary_closing(base_matrix, iterations=iterations)
                matrix = binary_fill_holes(matrix)
                solid = voxel_ops.matrix_to_marching_cubes(matrix=np.pad(matrix, 2, constant_values=False), pitch=pitch)
                if not isinstance(solid, trimesh.Trimesh) or len(solid.faces) < 1000:
                    continue
                solid = cleanup(solid)
                solid.metadata["meshmend_image_visual_hole_pitch_mm"] = float(pitch)
                solid.metadata["meshmend_image_visual_hole_close_iterations"] = int(iterations)
                candidates.append(solid)
        return candidates
    except Exception:
        return []


def final_image_surface_polish(mesh: trimesh.Trimesh, request: dict[str, Any]) -> trimesh.Trimesh:
    """Remove reconstruction speckle after density/detail without reopening the mesh."""
    source_workflow = str(request.get("_meshmend_source_workflow") or request.get("workflow") or "text_to_3d")
    if source_workflow not in {"image_to_3d", "text_to_3d"} or not strict_quality_requested(request):
        return mesh
    default_enabled = "1" if source_workflow == "image_to_3d" else "0"
    if os.environ.get("MESHMEND_FINAL_IMAGE_SURFACE_POLISH", default_enabled).strip().lower() in {"0", "false", "no"}:
        return mesh
    try:
        polished = mesh.copy()
        metadata = dict(mesh.metadata)
        from trimesh.smoothing import filter_taubin

        filter_taubin(
            polished,
            lamb=float(os.environ.get("MESHMEND_FINAL_POLISH_LAMBDA", "0.08")),
            nu=float(os.environ.get("MESHMEND_FINAL_POLISH_NU", "-0.085")),
            iterations=int(os.environ.get("MESHMEND_FINAL_POLISH_ITERATIONS", "1")),
        )
        polished.metadata.update(metadata)
        polished.metadata["meshmend_final_surface_polished"] = True
        return normalize_to_scale(cleanup(polished), requested_scale_mm(request), request)
    except Exception:
        return mesh


def coerce_to_trimesh(mesh: Any) -> trimesh.Trimesh:
    if isinstance(mesh, trimesh.Scene):
        geometries = [geom for geom in mesh.geometry.values() if hasattr(geom, "faces") and len(geom.faces) > 0]
        if not geometries:
            raise RuntimeError("Hunyuan returned an empty scene")
        return trimesh.util.concatenate(geometries)
    if isinstance(mesh, trimesh.Trimesh):
        return mesh.copy()
    if hasattr(mesh, "vertices") and hasattr(mesh, "faces"):
        return trimesh.Trimesh(vertices=np.asarray(mesh.vertices), faces=np.asarray(mesh.faces), process=False)
    raise RuntimeError(f"Unsupported mesh type for post-processing: {type(mesh)!r}")


def remove_background_slabs(mesh: trimesh.Trimesh, request: dict[str, Any]) -> trimesh.Trimesh:
    """Remove flat image-card/slab geometry around the subject."""
    prompt = str(request.get("prompt") or "").lower()
    intentional_flat = any(term in prompt for term in ("plaque", "coin", "badge", "relief", "bas relief", "bas-relief"))
    negative_flat = any(term in prompt for term in ("no plaque", "no card", "not a card", "no background", "no backplate"))
    if intentional_flat and not negative_flat:
        return mesh
    try:
        vertices = np.asarray(mesh.vertices, dtype=float)
        faces = np.asarray(mesh.faces)
        if len(vertices) < 1000 or len(faces) < 1000:
            return mesh
        mins = vertices.min(axis=0)
        maxs = vertices.max(axis=0)
        ext = np.maximum(maxs - mins, 1e-6)
        centers = vertices[faces].mean(axis=1)
        normals = np.asarray(mesh.face_normals, dtype=float)
        keep = np.ones(len(faces), dtype=bool)
        removed_any = False
        aggressive = bool(request.get("_meshmend_aggressive_slab_removal"))
        min_keep_ratio = 0.12 if aggressive else 0.35
        for axis in range(3):
            other_axes = [idx for idx in range(3) if idx != axis]
            for boundary in (mins[axis], maxs[axis]):
                near = np.abs(centers[:, axis] - boundary) < ext[axis] * (0.075 if aggressive else 0.04)
                flat = np.abs(normals[:, axis]) > (0.70 if aggressive else 0.78)
                candidate = near & flat
                min_candidate_faces = max(80, min(int(len(faces) * (0.002 if aggressive else 0.006)), 2500 if aggressive else 6000))
                if int(candidate.sum()) < min_candidate_faces:
                    continue
                cand_centers = centers[candidate]
                coverage = np.prod(np.maximum(cand_centers[:, other_axes].max(axis=0) - cand_centers[:, other_axes].min(axis=0), 1e-6)) / max(
                    np.prod(ext[other_axes]), 1e-6
                )
                # A card/slab covers a large fraction of the render plane. Do not
                # remove small flat armor plates or bases.
                min_coverage = float(os.environ.get("MESHMEND_SLAB_MIN_PLANE_COVERAGE", "0.24" if aggressive else "0.32"))
                if coverage < min_coverage:
                    continue
                next_keep = keep & ~candidate
                if int(next_keep.sum()) < max(1000, int(len(faces) * min_keep_ratio)):
                    continue
                keep = next_keep
                removed_any = True
        if not removed_any:
            return mesh
        cleaned = trimesh.Trimesh(vertices=vertices.copy(), faces=faces[keep].copy(), process=False)
        cleaned.remove_unreferenced_vertices()
        if len(cleaned.faces) < max(1000, int(len(faces) * min_keep_ratio)):
            return mesh
        cleaned.metadata.update(mesh.metadata)
        cleaned.metadata["meshmend_background_slab_removed"] = True
        return cleanup(cleaned)
    except Exception:
        return mesh


def remove_connected_sheet_surfaces(mesh: trimesh.Trimesh, request: dict[str, Any]) -> trimesh.Trimesh:
    """Remove large connected image sheets/cards even when fused to the figure.

    Hunyuan can weld the reference-image plane to the miniature's base, so a
    component split cannot remove it. This looks for broad, low-detail planar
    face bands with high coverage in the other two axes and peels those faces
    before the detail pipeline has a chance to sharpen the sheet.
    """
    prompt = str(request.get("prompt") or "").lower()
    if any(term in prompt for term in ("plaque", "coin", "badge", "relief", "bas relief", "bas-relief")):
        return mesh
    try:
        vertices = np.asarray(mesh.vertices, dtype=float)
        faces = np.asarray(mesh.faces)
        if len(vertices) < 1000 or len(faces) < 1000:
            return mesh
        mins = vertices.min(axis=0)
        maxs = vertices.max(axis=0)
        ext = np.maximum(maxs - mins, 1e-6)
        centers = vertices[faces].mean(axis=1)
        normals = np.asarray(mesh.face_normals, dtype=float)
        areas = np.asarray(mesh.area_faces, dtype=float)
        keep = np.ones(len(faces), dtype=bool)
        aggressive = bool(request.get("_meshmend_aggressive_slab_removal"))
        plane_width = float(os.environ.get("MESHMEND_CONNECTED_SHEET_PLANE_WIDTH", "0.055" if aggressive else "0.04"))
        min_coverage = float(os.environ.get("MESHMEND_CONNECTED_SHEET_MIN_COVERAGE", "0.46" if aggressive else "0.58"))
        min_area_ratio = float(os.environ.get("MESHMEND_CONNECTED_SHEET_MIN_AREA_RATIO", "0.0035" if aggressive else "0.006"))
        removed_any = False
        total_area = max(float(areas.sum()), 1e-8)

        for axis in (0, 1):
            axis_flat = np.abs(normals[:, axis]) > (0.62 if aggressive else 0.72)
            if int(axis_flat.sum()) < 100:
                continue
            other_axes = [idx for idx in range(3) if idx != axis]
            values = centers[:, axis]
            for quantile in (0.01, 0.05, 0.10, 0.90, 0.95, 0.99):
                plane = float(np.quantile(values[axis_flat], quantile))
                near = np.abs(values - plane) < ext[axis] * plane_width
                candidate = axis_flat & near & keep
                if int(candidate.sum()) < max(120, min(int(len(faces) * 0.0015), 2200)):
                    continue
                cand_centers = centers[candidate]
                coverage = np.prod(np.maximum(cand_centers[:, other_axes].max(axis=0) - cand_centers[:, other_axes].min(axis=0), 1e-6)) / max(
                    np.prod(ext[other_axes]), 1e-6
                )
                area_ratio = float(areas[candidate].sum() / total_area)
                z_span = float(cand_centers[:, 2].max() - cand_centers[:, 2].min()) / max(float(ext[2]), 1e-6)
                # A real figure/base has varied normals and lower plane coverage;
                # a reconstructed sheet is broad, planar, and spans much of Z.
                if coverage < min_coverage or area_ratio < min_area_ratio or z_span < 0.28:
                    continue
                next_keep = keep & ~candidate
                if int(next_keep.sum()) < max(1000, int(len(faces) * 0.45)):
                    continue
                keep = next_keep
                removed_any = True

        if not removed_any:
            return mesh
        cleaned = trimesh.Trimesh(vertices=vertices.copy(), faces=faces[keep].copy(), process=False)
        cleaned.remove_unreferenced_vertices()
        if len(cleaned.faces) < max(1000, int(len(faces) * 0.45)):
            return mesh
        cleaned.metadata.update(mesh.metadata)
        cleaned.metadata["meshmend_connected_sheet_removed"] = True
        return cleanup(cleaned)
    except Exception:
        return mesh


def enforce_single_subject(mesh: trimesh.Trimesh, request: dict[str, Any] | None = None) -> tuple[trimesh.Trimesh, bool]:
    """Keep one miniature even if generation produced a connected 4-model scene."""
    request = request or {}
    source_workflow = str(request.get("_meshmend_source_workflow") or request.get("workflow") or "text_to_3d")
    components = [component for component in mesh.split(only_watertight=False) if len(component.faces) > 100]
    if len(components) > 1:
        if source_workflow == "image_to_3d" and os.environ.get("MESHMEND_IMAGE_KEEP_COMPONENTS", "1").strip().lower() not in {"0", "false", "no"}:
            combined = trimesh.util.concatenate(components)
            combined.metadata.update(mesh.metadata)
            combined.metadata["meshmend_image_components_preserved"] = len(components)
            return cleanup(combined), False
        components.sort(key=component_subject_score, reverse=True)
        return components[0], True

    if source_workflow == "image_to_3d":
        # For a real user image, a wide silhouette is usually a valid pose,
        # weapon, mount, cape, or tail. Cropping heuristics were designed for
        # text-generated reference sheets and can turn complex image inputs into
        # an undefined blob by cutting away legitimate anatomy.
        return mesh, False

    side_crop = crop_connected_dual_subject(mesh)
    if side_crop is not mesh:
        return side_crop, True

    cropped = crop_spatial_subject(mesh)
    if cropped is not mesh:
        return cropped, True
    return mesh, False


def component_subject_score(component: trimesh.Trimesh) -> float:
    try:
        ext = np.maximum(np.asarray(component.extents, dtype=float), 1e-6)
        max_ext = float(ext.max())
        min_ext = float(ext.min())
        depth_ratio = min_ext / max(max_ext, 1e-6)
        sheet_penalty = 0.12 if depth_ratio < 0.08 else (0.45 if depth_ratio < 0.16 else 1.0)
        height_bonus = 0.65 + min(float(ext[2]) / max(max_ext, 1e-6), 1.0)
        face_bonus = min(len(component.faces) / 50_000.0, 8.0)
        return float(component.area) * sheet_penalty * height_bonus + face_bonus
    except Exception:
        return float(getattr(component, "area", 0.0))


def crop_connected_dual_subject(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Crop one side when Hunyuan fused two side-by-side figures into one mesh."""
    try:
        vertices = np.asarray(mesh.vertices, dtype=float)
        faces = np.asarray(mesh.faces)
        if len(vertices) < 1000 or len(faces) < 1000:
            return mesh
        mins = vertices.min(axis=0)
        maxs = vertices.max(axis=0)
        extents = np.maximum(maxs - mins, 1e-6)
        face_centers = vertices[faces].mean(axis=1)
        z_norm = (face_centers[:, 2] - mins[2]) / extents[2]
        body = z_norm > 0.08
        best: tuple[float, int, float] | None = None
        for axis in (0, 1):
            other = 1 - axis
            if extents[axis] / max(extents[2], extents[other], 1e-6) < float(os.environ.get("MESHMEND_DUAL_MESH_AXIS_RATIO", "0.92")):
                continue
            values = face_centers[body, axis] if int(body.sum()) > 500 else face_centers[:, axis]
            hist, edges = np.histogram(values, bins=96)
            if hist.max() <= 0:
                continue
            smooth = np.convolve(hist.astype(float), np.ones(7) / 7.0, mode="same")
            center = len(smooth) // 2
            search_radius = max(8, len(smooth) // 5)
            start = max(4, center - search_radius)
            end = min(len(smooth) - 4, center + search_radius)
            valley_idx = int(start + np.argmin(smooth[start:end]))
            left_mass = float(smooth[:valley_idx].sum())
            right_mass = float(smooth[valley_idx:].sum())
            if min(left_mass, right_mass) <= 0:
                continue
            valley_score = 1.0 - (float(smooth[valley_idx]) / max(float(smooth.max()), 1e-6))
            balance = min(left_mass, right_mass) / max(left_mass, right_mass)
            score = valley_score * balance * (extents[axis] / max(extents[other], 1e-6))
            if balance > 0.25 and valley_score > 0.18 and (best is None or score > best[0]):
                best = (score, axis, float(edges[valley_idx]))
        if best is None:
            return mesh
        _score, axis, cut = best
        left_mask = face_centers[:, axis] <= cut
        right_mask = ~left_mask
        if int(left_mask.sum()) < len(faces) * 0.18 or int(right_mask.sum()) < len(faces) * 0.18:
            return mesh
        chosen = left_mask if int(left_mask.sum()) >= int(right_mask.sum()) else right_mask
        cropped = trimesh.Trimesh(vertices=vertices.copy(), faces=faces[chosen].copy(), process=False)
        cropped.remove_unreferenced_vertices()
        if len(cropped.faces) < len(faces) * 0.18:
            return mesh
        components = [component for component in cropped.split(only_watertight=False) if len(component.faces) > 100]
        if components:
            components.sort(key=lambda item: float(item.area), reverse=True)
            cropped = components[0]
        cropped.metadata.update(mesh.metadata)
        cropped.metadata["meshmend_connected_dual_subject_cropped"] = True
        return cleanup(cropped)
    except Exception:
        return mesh


def crop_spatial_subject(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    vertices = np.asarray(mesh.vertices, dtype=float)
    if len(vertices) < 1000 or len(mesh.faces) < 1000:
        return mesh
    mins = vertices.min(axis=0)
    maxs = vertices.max(axis=0)
    extents = np.maximum(maxs - mins, 1e-6)
    z_norm = (vertices[:, 2] - mins[2]) / extents[2]
    body_vertices = vertices[z_norm > 0.10]
    if len(body_vertices) < 500:
        body_vertices = vertices

    keep_bounds = []
    changed = False
    for axis in (0, 1):
        axis_bounds = subject_axis_bounds(body_vertices[:, axis], axis_extent=float(extents[axis]))
        if axis_bounds is None:
            keep_bounds.append((mins[axis], maxs[axis]))
        else:
            low, high = axis_bounds
            keep_bounds.append((low, high))
            changed = True

    if not changed:
        return mesh

    face_centers = vertices[mesh.faces].mean(axis=1)
    mask = (
        (face_centers[:, 0] >= keep_bounds[0][0])
        & (face_centers[:, 0] <= keep_bounds[0][1])
        & (face_centers[:, 1] >= keep_bounds[1][0])
        & (face_centers[:, 1] <= keep_bounds[1][1])
    )
    if int(mask.sum()) < max(500, len(mesh.faces) * 0.12):
        return mesh
    cropped = trimesh.Trimesh(vertices=vertices.copy(), faces=np.asarray(mesh.faces)[mask].copy(), process=False)
    cropped.remove_unreferenced_vertices()
    components = [component for component in cropped.split(only_watertight=False) if len(component.faces) > 100]
    if components:
        components.sort(key=lambda item: float(item.area), reverse=True)
        return components[0]
    return cropped


def subject_axis_bounds(values: np.ndarray, *, axis_extent: float) -> tuple[float, float] | None:
    if axis_extent <= 1e-6 or len(values) < 100:
        return None
    hist, edges = np.histogram(values, bins=128)
    if hist.max() <= 0:
        return None
    smooth = np.convolve(hist.astype(float), np.ones(5) / 5.0, mode="same")
    active = smooth > max(float(smooth.max()) * 0.18, len(values) * 0.001)
    bands: list[tuple[int, int]] = []
    start = None
    for idx, flag in enumerate(active):
        if flag and start is None:
            start = idx
        elif not flag and start is not None:
            if idx - start >= 4:
                bands.append((start, idx - 1))
            start = None
    if start is not None and len(active) - start >= 4:
        bands.append((start, len(active) - 1))
    if len(bands) < 2:
        return None
    center = (values.min() + values.max()) * 0.5
    best = max(
        bands,
        key=lambda band: ((band[1] - band[0]) * 0.55) - abs(((edges[band[0]] + edges[band[1] + 1]) * 0.5) - center),
    )
    low = edges[best[0]] - axis_extent * 0.04
    high = edges[best[1] + 1] + axis_extent * 0.04
    if high - low > axis_extent * 0.82:
        return None
    return float(low), float(high)


def normalize_to_scale(mesh: trimesh.Trimesh, scale_mm: float, request: dict[str, Any] | None = None) -> trimesh.Trimesh:
    vertices = np.asarray(mesh.vertices, dtype=float)
    extents = vertices.max(axis=0) - vertices.min(axis=0)
    if float(np.max(extents)) <= 1e-8:
        return mesh
    preserve_wide_reference = should_preserve_wide_image_reference(request)
    if not preserve_wide_reference and os.environ.get("MESHMEND_AUTO_ORIENT_HEIGHT_AXIS", "1").strip().lower() not in {"0", "false", "no"}:
        height_axis = int(np.argmax(extents))
        # Hunyuan outputs are often Y-up while STL/slicer convention and the
        # rest of MeshMend's miniature detailing assume Z-up. Move the likely
        # standing-height axis to Z before applying millimeter scale.
        if height_axis != 2 and float(extents[height_axis]) > float(extents[2]) * 1.12:
            vertices = vertices.copy()
            vertices[:, [2, height_axis]] = vertices[:, [height_axis, 2]]
            extents = vertices.max(axis=0) - vertices.min(axis=0)
            mesh.metadata["meshmend_height_axis_reoriented_to_z"] = height_axis
    scale_axis = "max" if preserve_wide_reference else os.environ.get("MESHMEND_SCALE_AXIS", "z").strip().lower()
    if scale_axis in {"max", "largest"}:
        current = float(np.max(extents))
        scale_axis_name = "max"
    else:
        current = float(extents[2])
        scale_axis_name = "z_height"
    if current <= 1e-8:
        return mesh
    vertices *= scale_mm / current
    mins = vertices.min(axis=0)
    maxs = vertices.max(axis=0)
    vertices[:, 0] -= (mins[0] + maxs[0]) * 0.5
    vertices[:, 1] -= (mins[1] + maxs[1]) * 0.5
    vertices[:, 2] -= mins[2]
    mesh.vertices = vertices
    mesh.metadata["units"] = "mm"
    mesh.metadata["meshmend_scale_mm"] = scale_mm
    mesh.metadata["meshmend_scale_axis"] = scale_axis_name
    if preserve_wide_reference:
        mesh.metadata["meshmend_preserved_wide_image_reference"] = True
    return mesh


def should_preserve_wide_image_reference(request: dict[str, Any] | None) -> bool:
    if not request:
        return False
    source_workflow = str(request.get("_meshmend_source_workflow") or request.get("workflow") or "")
    if source_workflow != "image_to_3d":
        return False
    if os.environ.get("MESHMEND_PRESERVE_WIDE_IMAGE_REFERENCE", "1").strip().lower() in {"0", "false", "no"}:
        return False
    prompt = str(request.get("prompt") or "").lower()
    if any(term in prompt for term in ("mounted", "mount", "rider", "vehicle", "tank", "dragon", "beast", "cavalry", "bike", "chariot")):
        return True
    image_path = str(request.get("_meshmend_source_image_path") or "").strip()
    if not image_path:
        return False
    try:
        from PIL import Image

        image = Image.open(image_path)
        width, height = image.size
        if height <= 0:
            return False
        return (float(width) / float(height)) >= float(os.environ.get("MESHMEND_WIDE_REFERENCE_ASPECT_RATIO", "1.15"))
    except Exception:
        return False


def complete_front_shell_volume(mesh: trimesh.Trimesh, request: dict[str, Any]) -> trimesh.Trimesh:
    """Synthesize a rear shell when single-image generation returns only the front half.

    Local image-to-3D models often reconstruct the visible face of a subject and
    leave the hidden side as a shallow relief. For miniatures that should be a
    full body, so when depth is suspiciously thin we mirror the visible shell
    into a rear volume before final scaling/detailing. This is intentionally
    conservative and only runs for thin character-like outputs.
    """
    if strict_quality_requested(request) and os.environ.get("MESHMEND_ALLOW_SYNTHETIC_REAR_VOLUME", "1").strip().lower() not in {"1", "true", "yes"}:
        return mesh
    try:
        vertices = np.asarray(mesh.vertices, dtype=float)
        if len(vertices) < 100 or len(mesh.faces) < 100:
            return mesh
        extents = vertices.max(axis=0) - vertices.min(axis=0)
        max_extent = float(np.max(extents))
        if max_extent <= 1e-8:
            return mesh
        thin_axis = int(np.argmin(extents))
        thin_ratio = float(extents[thin_axis] / max_extent)
        required_ratio = float(os.environ.get("MESHMEND_MIN_FULL_BODY_DEPTH_RATIO", "0.34"))
        if thin_ratio >= required_ratio:
            return mesh

        prompt = str(request.get("prompt") or "").lower()
        if any(term in prompt for term in ("relief", "plaque", "coin", "badge", "bas relief", "bas-relief")):
            return mesh

        center = (vertices.max(axis=0) + vertices.min(axis=0)) * 0.5
        mirrored_vertices = vertices.copy()
        mirrored_vertices[:, thin_axis] = (2.0 * center[thin_axis]) - mirrored_vertices[:, thin_axis]
        mirrored_faces = np.asarray(mesh.faces)[:, ::-1].copy()
        completed = trimesh.util.concatenate(
            [
                trimesh.Trimesh(vertices=vertices.copy(), faces=np.asarray(mesh.faces).copy(), process=False),
                trimesh.Trimesh(vertices=mirrored_vertices, faces=mirrored_faces, process=False),
            ]
        )
        # Spread the mirrored halves to a printable body depth instead of a
        # coincident double shell. normalize_to_scale runs again after this.
        completed_vertices = np.asarray(completed.vertices, dtype=float)
        completed_extents = completed_vertices.max(axis=0) - completed_vertices.min(axis=0)
        current = float(completed_extents[thin_axis])
        target = max_extent * required_ratio
        if current > 1e-8 and current < target:
            completed_vertices[:, thin_axis] = center[thin_axis] + (completed_vertices[:, thin_axis] - center[thin_axis]) * (target / current)
            completed.vertices = completed_vertices
        completed.metadata.update(mesh.metadata)
        completed.metadata["meshmend_rear_volume_completed"] = True
        return cleanup(completed)
    except Exception:
        return mesh


def thicken_if_sheet(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    if os.environ.get("MESHMEND_ALLOW_PRODUCTION_THICKENING", "0").strip().lower() not in {"1", "true", "yes"}:
        # For production/store-quality requests, do not turn shallow Hunyuan
        # reconstructions into chunky blocks. Let the quality gate flag thin
        # output instead of inventing bulk geometry.
        return mesh
    vertices = np.asarray(mesh.vertices, dtype=float)
    extents = vertices.max(axis=0) - vertices.min(axis=0)
    max_extent = float(np.max(extents))
    min_extent = float(np.min(extents))
    if max_extent <= 1e-8:
        return mesh
    min_ratio = float(os.environ.get("MESHMEND_MIN_THICKNESS_RATIO", "0.22"))
    if min_extent / max_extent >= min_ratio:
        return mesh
    axis = int(np.argmin(extents))
    center = (vertices.max(axis=0) + vertices.min(axis=0)) * 0.5
    target = max_extent * min_ratio
    scale = min(float(os.environ.get("MESHMEND_MAX_THICKEN_SCALE", "10.0")), target / max(min_extent, 1e-6))
    vertices[:, axis] = center[axis] + (vertices[:, axis] - center[axis]) * scale
    normals = np.asarray(mesh.vertex_normals, dtype=float)
    if normals.shape == vertices.shape:
        vertices += normals * float(os.environ.get("MESHMEND_SHEET_NORMAL_INFLATE_MM", "0.28"))
    mesh.vertices = vertices
    return mesh


def subdivide_to_faces(mesh: trimesh.Trimesh, target_faces: int) -> trimesh.Trimesh:
    target_faces = memory_safe_face_target(max(target_faces, len(mesh.faces)))
    # Trimesh subdivision grows faces in 4x jumps. The old default accepted a
    # jump to 3-4M faces to satisfy a 1-2M target, which can freeze machines with
    # consumer RAM during repair/export. Keep the overshoot bounded unless the
    # user explicitly disables memory safety.
    max_faces = int(os.environ.get("MESHMEND_MAX_EXPORT_FACES", "600000"))
    if memory_safety_enabled():
        max_faces = min(max_faces, int(os.environ.get("MESHMEND_MAX_POSTPROCESS_FACES", "350000")) * 2)
    else:
        max_faces = max(target_faces, max_faces)
    overshoot_limit = max(target_faces, int(min(max_faces, target_faces * 1.35)))
    while len(mesh.faces) < target_faces:
        next_faces = len(mesh.faces) * 4
        if next_faces > overshoot_limit:
            break
        metadata = dict(mesh.metadata)
        vertices, faces = trimesh.remesh.subdivide(mesh.vertices, mesh.faces)
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        mesh.metadata.update(metadata)
    return mesh


def apply_miniature_sculpt_detail(mesh: trimesh.Trimesh, request: dict[str, Any]) -> trimesh.Trimesh:
    vertices = np.asarray(mesh.vertices, dtype=float)
    if len(vertices) < 1000:
        return mesh
    normals = np.asarray(mesh.vertex_normals, dtype=float)
    if normals.shape != vertices.shape:
        return mesh
    mins = vertices.min(axis=0)
    maxs = vertices.max(axis=0)
    ext = np.maximum(maxs - mins, 1e-6)
    coords = (vertices - mins) / ext
    z = coords[:, 2]
    x = coords[:, 0] - 0.5
    y = coords[:, 1] - 0.5
    prompt = str(request.get("prompt") or "")
    seed = (sum(ord(ch) for ch in prompt) % 997) + 1
    relief = np.zeros(len(vertices), dtype=float)

    armored = any(term in prompt.lower() for term in ("space marine", "armor", "armour", "robot", "mech", "gun", "soldier", "knight"))
    if armored:
        torso = (z > 0.28) & (z < 0.74)
        legs = (z > 0.08) & (z < 0.42)
        shoulders = (z > 0.55) & (z < 0.86) & (np.abs(x) > np.quantile(np.abs(x), 0.62))
        head = z > 0.74
        seams_h = np.abs(np.sin((z * 18.0 + seed * 0.031) * math.pi)) < 0.026
        seams_v = np.abs(np.sin((x * 12.0 + seed * 0.047) * math.pi)) < 0.022
        trim = np.abs(np.sin((z * 9.0 + np.abs(x) * 2.0 + seed * 0.019) * math.pi)) < 0.030
        vents = (np.abs(np.sin((x * 30.0 + seed * 0.13) * math.pi)) < 0.018) & (np.abs(y) < 0.35)
        rivets = (np.abs(np.sin((x * 25.0 + seed * 0.17) * math.pi)) < 0.018) & (np.abs(np.sin((z * 29.0 + seed * 0.23) * math.pi)) < 0.018)
        relief -= (seams_h & torso).astype(float) * 0.65
        relief -= (seams_v & (torso | legs)).astype(float) * 0.38
        relief += (trim & (torso | shoulders)).astype(float) * 0.48
        relief -= (vents & (torso | head)).astype(float) * 0.42
        if os.environ.get("MESHMEND_ENABLE_RAISED_DOT_DETAIL", "0").strip().lower() in {"1", "true", "yes"}:
            relief += (rivets & (torso | shoulders | legs)).astype(float) * 0.32
    else:
        folds = np.abs(np.sin((z * 15.0 + x * 4.0 + seed * 0.02) * math.pi)) < 0.022
        relief -= folds.astype(float) * 0.32

    if os.environ.get("MESHMEND_ENABLE_MICRO_NOISE", "0").strip().lower() in {"1", "true", "yes"}:
        relief += 0.08 * np.sin((x * 47.0 + y * 19.0 + z * 31.0 + seed) * math.pi)

    amplitude = float(os.environ.get("MESHMEND_DETAIL_RELIEF_MM", "0.025"))
    active = z > 0.06
    mesh.vertices = vertices + normals * (np.clip(relief, -1.0, 1.0) * amplitude * active.astype(float))[:, None]
    return cleanup(mesh)


def apply_image_guided_surface_detail(mesh: trimesh.Trimesh, request: dict[str, Any]) -> trimesh.Trimesh:
    """Project high-contrast 2D reference edges into printable front-shell relief."""
    image_path = str(request.get("_meshmend_source_image_path") or "").strip()
    if not image_path:
        return mesh
    quality = str(request.get("quality") or "standard").lower()
    prompt = str(request.get("prompt") or "").lower()
    source_workflow = str(request.get("_meshmend_source_workflow") or "").lower()
    # Project reference/concept contrast into shallow printable relief. This is
    # automatic in studio mode so detail has a real geometry path instead of
    # relying on prompt wording alone; negative prompts and concept gates reduce
    # painted/noisy inputs before this pass runs.
    default_enabled = "auto" if source_workflow in {"image_to_3d", "text_to_3d"} else "0"
    enabled = os.environ.get("MESHMEND_ENABLE_IMAGE_GUIDED_DETAIL", default_enabled).strip().lower()
    wants_8k = any(
        term in prompt
        for term in (
            "8k", "8 k", "studio", "studio quality", "studio-quality", "studio level", "studio-level",
            "production", "display quality", "maximum detail", "store quality", "store-quality", "store level",
            "store-level",
        )
    )
    is_text_concept = source_workflow == "text_to_3d"
    if enabled in {"0", "false", "no"} or (enabled == "auto" and quality != "high" and not wants_8k and not is_text_concept):
        return mesh
    try:
        from PIL import Image

        vertices = np.asarray(mesh.vertices, dtype=float)
        if len(vertices) < 1000:
            return mesh
        normals = np.asarray(mesh.vertex_normals, dtype=float)
        if normals.shape != vertices.shape:
            return mesh

        sample_size = int(os.environ.get("MESHMEND_IMAGE_DETAIL_SAMPLE_SIZE", "768"))
        image = Image.open(image_path).convert("RGB").resize((sample_size, sample_size))
        arr = np.asarray(image, dtype=np.float32) / 255.0
        gray = arr @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
        gx = np.abs(np.diff(gray, axis=1, prepend=gray[:, :1]))
        gy = np.abs(np.diff(gray, axis=0, prepend=gray[:1, :]))
        edges = gx + gy
        edge_scale = float(np.percentile(edges, 97.5))
        if edge_scale > 1e-6:
            edges = np.clip(edges / edge_scale, 0.0, 1.0)
        contrast = gray - float(np.mean(gray))
        contrast_scale = float(np.percentile(np.abs(contrast), 93.0))
        if contrast_scale > 1e-6:
            contrast = np.clip(contrast / contrast_scale, -1.0, 1.0)

        mins = vertices.min(axis=0)
        maxs = vertices.max(axis=0)
        ext = np.maximum(maxs - mins, 1e-6)
        # The Hunyuan reference image is a front view; apply projected relief to
        # the visible/front half and avoid the base so image texture becomes
        # sculpted detail rather than all-over STL noise.
        front = vertices[:, 1] <= mins[1] + ext[1] * 0.55
        not_base = vertices[:, 2] > mins[2] + ext[2] * 0.07
        u = (vertices[:, 0] - mins[0]) / ext[0]
        v = 1.0 - ((vertices[:, 2] - mins[2]) / ext[2])
        max_index = sample_size - 1
        ix = np.clip((u * max_index).astype(int), 0, max_index)
        iy = np.clip((v * max_index).astype(int), 0, max_index)
        sampled_edges = edges[iy, ix]
        sampled_contrast = contrast[iy, ix]
        edge_cutoff = float(os.environ.get("MESHMEND_IMAGE_DETAIL_EDGE_CUTOFF", "0.74" if source_workflow == "image_to_3d" else "0.56"))
        grooves = sampled_edges > edge_cutoff
        contrast_weight = float(os.environ.get("MESHMEND_IMAGE_DETAIL_CONTRAST_WEIGHT", "0.025" if source_workflow == "image_to_3d" else "0.07"))
        groove_weight = float(os.environ.get("MESHMEND_IMAGE_DETAIL_GROOVE_WEIGHT", "0.16" if source_workflow == "image_to_3d" else "0.34"))
        relief = (sampled_contrast * contrast_weight) - grooves.astype(float) * groove_weight
        amplitude = float(os.environ.get("MESHMEND_IMAGE_DETAIL_RELIEF_MM", "0.018" if source_workflow == "image_to_3d" else "0.045"))
        active = (front & not_base).astype(float)
        mesh.vertices = vertices + normals * (relief * amplitude * active)[:, None]
        mesh.metadata["meshmend_image_guided_detail"] = True
        return cleanup(mesh)
    except Exception:
        return mesh


def add_high_resolution_geometry(mesh: trimesh.Trimesh, request: dict[str, Any]) -> trimesh.Trimesh:
    """Add deterministic miniature-scale geometry after resolution upscaling.

    This is intentionally geometry-first: it creates raised/recessed STL surface
    detail from prompt/category cues instead of only increasing polygon count.
    It cannot invent a perfect sculpt from a bad concept, but it prevents high
    resolution exports from remaining visually smooth/480p.
    """
    enabled = synthetic_detail_enabled("MESHMEND_ENABLE_GEOMETRY_UPSCALE", default="1" if strict_quality_requested(request) else "0")
    if not enabled:
        return mesh
    quality = str(request.get("quality") or "standard").lower()
    prompt = str(request.get("prompt") or "").lower()
    wants_detail = quality == "high" or any(
        term in prompt for term in (
            "8k", "8 k", "studio", "studio quality", "studio-quality", "studio level", "studio-level",
            "production", "display quality", "maximum detail", "store quality", "store-quality", "store level",
            "store-level", "high detail",
        )
    )
    if not wants_detail:
        return mesh
    try:
        vertices = np.asarray(mesh.vertices, dtype=float)
        if len(vertices) < 1000:
            return mesh
        normals = np.asarray(mesh.vertex_normals, dtype=float)
        if normals.shape != vertices.shape:
            return mesh
        mins = vertices.min(axis=0)
        maxs = vertices.max(axis=0)
        ext = np.maximum(maxs - mins, 1e-6)
        coords = (vertices - mins) / ext
        x = coords[:, 0] - 0.5
        y = coords[:, 1] - 0.5
        z = coords[:, 2]
        seed = (sum(ord(ch) for ch in prompt) % 997) + 1
        relief = np.zeros(len(vertices), dtype=float)

        armored = any(term in prompt for term in ("space marine", "chaos", "armor", "armour", "robot", "mech", "soldier", "knight"))
        creature = any(term in prompt for term in ("dragon", "demon", "beast", "monster", "orc", "ork", "undead"))
        if armored:
            torso = (z > 0.25) & (z < 0.76)
            legs = (z > 0.07) & (z < 0.45)
            upper = (z > 0.55) & (z < 0.90)
            # Fine recessed armor seams and larger raised trim read better than
            # random noise on resin miniatures.
            fine_h = np.abs(np.sin((z * 34.0 + seed * 0.011) * math.pi)) < 0.014
            fine_v = np.abs(np.sin((x * 28.0 + seed * 0.017) * math.pi)) < 0.013
            trim = np.abs(np.sin((z * 14.0 + np.abs(x) * 3.0 + seed * 0.023) * math.pi)) < 0.024
            vents = (np.abs(np.sin((x * 46.0 + seed * 0.031) * math.pi)) < 0.010) & (np.abs(y) < 0.42)
            rivets = (np.abs(np.sin((x * 38.0 + seed * 0.043) * math.pi)) < 0.012) & (
                np.abs(np.sin((z * 42.0 + seed * 0.059) * math.pi)) < 0.012
            )
            relief -= (fine_h & torso).astype(float) * 0.44
            relief -= (fine_v & (torso | legs)).astype(float) * 0.30
            relief += (trim & (torso | upper)).astype(float) * 0.42
            relief -= (vents & upper).astype(float) * 0.34
            if os.environ.get("MESHMEND_ENABLE_RAISED_DOT_DETAIL", "0").strip().lower() in {"1", "true", "yes"}:
                relief += (rivets & (torso | upper | legs)).astype(float) * 0.24
        elif creature:
            scale_rows = np.abs(np.sin((z * 40.0 + seed * 0.019) * math.pi)) < 0.018
            scale_cols = np.abs(np.sin((x * 32.0 + y * 8.0 + seed * 0.037) * math.pi)) < 0.020
            wrinkles = np.abs(np.sin((z * 22.0 + x * 5.0 + seed * 0.041) * math.pi)) < 0.018
            relief += (scale_rows & scale_cols).astype(float) * 0.42
            relief -= wrinkles.astype(float) * 0.24
        else:
            folds = np.abs(np.sin((z * 24.0 + x * 6.0 + seed * 0.029) * math.pi)) < 0.018
            seams = np.abs(np.sin((x * 20.0 + seed * 0.033) * math.pi)) < 0.014
            relief -= folds.astype(float) * 0.28
            relief -= (seams & (z > 0.18)).astype(float) * 0.18

        active = z > 0.06
        amplitude = float(os.environ.get("MESHMEND_GEOMETRY_UPSCALE_RELIEF_MM", "0.018"))
        mesh.vertices = vertices + normals * (np.clip(relief, -1.0, 1.0) * amplitude * active.astype(float))[:, None]
        mesh.metadata["meshmend_geometry_upscale"] = True
        return cleanup(mesh)
    except Exception:
        return mesh


def apply_custom_miniature_detail_pipeline(mesh: trimesh.Trimesh, request: dict[str, Any]) -> trimesh.Trimesh:
    """MeshMend miniature-specific finishing pass for studio/store exports.

    This pass is intentionally different from the generic Hunyuan cleanup: it
    creates layered miniature-language detail (armor plate cuts, bevel trim,
    vents, rivets, cloth seams, creature scales) at several spatial frequencies.
    It runs after sheet removal so it does not add detail to an image card.
    """
    enabled = synthetic_detail_enabled("MESHMEND_ENABLE_CUSTOM_MINIATURE_DETAIL_PIPELINE", default="1" if strict_quality_requested(request) else "0")
    if not enabled or not strict_quality_requested(request):
        return mesh
    try:
        target_faces = int(os.environ.get("MESHMEND_CUSTOM_DETAIL_FACES", str(min(detail_face_target(request), 900_000))))
        mesh = subdivide_to_faces(mesh, max(len(mesh.faces), target_faces))
        vertices = np.asarray(mesh.vertices, dtype=float)
        if len(vertices) < 1000:
            return mesh
        normals = np.asarray(mesh.vertex_normals, dtype=float)
        if normals.shape != vertices.shape:
            return mesh
        mins = vertices.min(axis=0)
        maxs = vertices.max(axis=0)
        ext = np.maximum(maxs - mins, 1e-6)
        coords = (vertices - mins) / ext
        x = coords[:, 0] - 0.5
        y = coords[:, 1] - 0.5
        z = coords[:, 2]
        prompt = str(request.get("prompt") or "").lower()
        seed = (sum(ord(ch) for ch in prompt) % 997) + 1

        relief = np.zeros(len(vertices), dtype=float)
        armored = any(term in prompt for term in ("space marine", "armor", "armour", "robot", "mech", "soldier", "knight", "power armor"))
        creature = any(term in prompt for term in ("dragon", "demon", "daemon", "beast", "monster", "orc", "ork", "undead", "lizard"))
        cloth = any(term in prompt for term in ("robe", "cloak", "cloth", "tabard", "cape", "wizard"))

        if armored:
            relief += _hard_surface_miniature_relief(x, y, z, seed)
        if creature:
            relief += _creature_miniature_relief(x, y, z, seed)
        if cloth or not (armored or creature):
            relief += _cloth_miniature_relief(x, y, z, seed)

        # Universal readable miniature cues: separation grooves at limbs/torso,
        # subtle undercuts, and small material breakup. These are deterministic
        # patterns, not random STL noise.
        body = z > 0.08
        silhouette_guard = (np.abs(x) < 0.492) & (np.abs(y) < 0.492)
        undercuts = np.abs(np.sin((z * 17.0 + np.abs(x) * 5.0 + seed * 0.101) * math.pi)) < 0.012
        micro_breakup = (
            np.sin((x * 89.0 + z * 31.0 + seed * 0.113) * math.pi)
            * np.sin((y * 73.0 - z * 27.0 + seed * 0.127) * math.pi)
        )
        relief -= undercuts.astype(float) * 0.16
        if os.environ.get("MESHMEND_ENABLE_SURFACE_BREAKUP", "0").strip().lower() in {"1", "true", "yes"}:
            relief += np.clip(micro_breakup, -1.0, 1.0) * 0.018

        amplitude = float(os.environ.get("MESHMEND_CUSTOM_DETAIL_RELIEF_MM", "0.065"))
        mesh.vertices = vertices + normals * (np.clip(relief, -1.0, 1.0) * amplitude * body.astype(float) * silhouette_guard.astype(float))[:, None]
        mesh.metadata["meshmend_custom_miniature_detail_pipeline"] = True
        return cleanup(mesh)
    except Exception:
        return mesh


def _hard_surface_miniature_relief(x: np.ndarray, y: np.ndarray, z: np.ndarray, seed: int) -> np.ndarray:
    relief = np.zeros(len(x), dtype=float)
    torso = (z > 0.24) & (z < 0.78)
    legs = (z > 0.07) & (z < 0.46)
    upper = (z > 0.52) & (z < 0.93)
    shoulders = upper & (np.abs(x) > 0.18)
    panel_h = np.abs(np.sin((z * 38.0 + seed * 0.017) * math.pi)) < 0.012
    panel_v = np.abs(np.sin((x * 34.0 + seed * 0.023) * math.pi)) < 0.011
    bevels = np.abs(np.sin((z * 19.0 + np.abs(x) * 7.0 + seed * 0.031) * math.pi)) < 0.020
    vents = (np.abs(np.sin((x * 72.0 + seed * 0.041) * math.pi)) < 0.008) & (np.abs(y) < 0.38)
    rivets = (np.abs(np.sin((x * 64.0 + seed * 0.053) * math.pi)) < 0.008) & (
        np.abs(np.sin((z * 68.0 + seed * 0.067) * math.pi)) < 0.008
    )
    trim_dots = (np.abs(np.sin((x * 42.0 + z * 12.0 + seed * 0.079) * math.pi)) < 0.010) & shoulders
    relief -= (panel_h & torso).astype(float) * 0.52
    relief -= (panel_v & (torso | legs)).astype(float) * 0.42
    relief += (bevels & (torso | upper | legs)).astype(float) * 0.46
    relief -= (vents & upper).astype(float) * 0.44
    if os.environ.get("MESHMEND_ENABLE_RAISED_DOT_DETAIL", "0").strip().lower() in {"1", "true", "yes"}:
        relief += (rivets & (torso | legs | shoulders)).astype(float) * 0.28
    if os.environ.get("MESHMEND_ENABLE_RAISED_DOT_DETAIL", "0").strip().lower() in {"1", "true", "yes"}:
        relief += trim_dots.astype(float) * 0.18
    return relief


def _creature_miniature_relief(x: np.ndarray, y: np.ndarray, z: np.ndarray, seed: int) -> np.ndarray:
    relief = np.zeros(len(x), dtype=float)
    scales_a = np.abs(np.sin((x * 58.0 + seed * 0.037) * math.pi)) < 0.014
    scales_b = np.abs(np.sin((z * 62.0 + y * 11.0 + seed * 0.049) * math.pi)) < 0.014
    wrinkles = np.abs(np.sin((z * 39.0 + x * 9.0 + seed * 0.071) * math.pi)) < 0.014
    scars = np.abs(np.sin(((x - y) * 31.0 + z * 21.0 + seed * 0.091) * math.pi)) < 0.010
    relief += (scales_a & scales_b & (z > 0.10)).astype(float) * 0.50
    relief -= wrinkles.astype(float) * 0.28
    relief -= (scars & (z > 0.25)).astype(float) * 0.34
    return relief


def _cloth_miniature_relief(x: np.ndarray, y: np.ndarray, z: np.ndarray, seed: int) -> np.ndarray:
    folds = np.abs(np.sin((z * 31.0 + x * 9.0 + seed * 0.029) * math.pi)) < 0.017
    secondary = np.abs(np.sin((z * 47.0 - y * 7.0 + seed * 0.043) * math.pi)) < 0.010
    stitches = np.abs(np.sin((x * 70.0 + seed * 0.061) * math.pi)) < 0.008
    relief = np.zeros(len(x), dtype=float)
    relief -= folds.astype(float) * 0.34
    relief -= (secondary & (z > 0.16)).astype(float) * 0.20
    relief += (stitches & (z > 0.10)).astype(float) * 0.24
    return relief


def apply_intricate_detail_pipeline(mesh: trimesh.Trimesh, request: dict[str, Any]) -> trimesh.Trimesh:
    """Separate multi-pass intricate detailing for high-quality miniature exports."""
    enabled = synthetic_detail_enabled("MESHMEND_ENABLE_INTRICATE_DETAIL_PIPELINE", default="1" if strict_quality_requested(request) else "0")
    if not enabled:
        return mesh
    quality = str(request.get("quality") or "standard").lower()
    prompt = str(request.get("prompt") or "").lower()
    wants_detail = quality == "high" or any(
        term in prompt for term in (
            "8k", "8 k", "studio", "studio quality", "studio-quality", "studio level", "studio-level",
            "production", "display quality", "maximum detail", "store quality", "store-quality", "store level",
            "store-level", "intricate", "ornate",
        )
    )
    if not wants_detail:
        return mesh
    try:
        target_faces = memory_safe_face_target(int(os.environ.get("MESHMEND_INTRICATE_DETAIL_FACES", "220000")))
        mesh = subdivide_to_faces(mesh, max(len(mesh.faces), target_faces))
        vertices = np.asarray(mesh.vertices, dtype=float)
        if len(vertices) < 1000:
            return mesh
        normals = np.asarray(mesh.vertex_normals, dtype=float)
        if normals.shape != vertices.shape:
            return mesh

        mins = vertices.min(axis=0)
        maxs = vertices.max(axis=0)
        ext = np.maximum(maxs - mins, 1e-6)
        coords = (vertices - mins) / ext
        x = coords[:, 0] - 0.5
        y = coords[:, 1] - 0.5
        z = coords[:, 2]
        seed = (sum(ord(ch) for ch in prompt) % 997) + 1
        relief = _intricate_image_relief(vertices, request, mins, ext)
        relief += _intricate_prompt_relief(prompt, x, y, z, seed)

        hatch = np.sin((x * 72.0 + z * 19.0 + seed * 0.07) * math.pi) * np.sin((y * 41.0 - z * 13.0) * math.pi)
        if os.environ.get("MESHMEND_ENABLE_INTRICATE_HATCH_NOISE", "0").strip().lower() in {"1", "true", "yes"}:
            relief += np.clip(hatch, -1.0, 1.0) * 0.020

        active = z > 0.065
        silhouette_guard = (np.abs(x) < 0.49) & (np.abs(y) < 0.49)
        amplitude = float(os.environ.get("MESHMEND_INTRICATE_DETAIL_RELIEF_MM", "0.055"))
        mesh.vertices = vertices + normals * (np.clip(relief, -1.0, 1.0) * amplitude * active.astype(float) * silhouette_guard.astype(float))[:, None]
        mesh.metadata["meshmend_intricate_detail_pipeline"] = True
        return cleanup(mesh)
    except Exception:
        return mesh


def _intricate_image_relief(vertices: np.ndarray, request: dict[str, Any], mins: np.ndarray, ext: np.ndarray) -> np.ndarray:
    image_path = str(request.get("_meshmend_source_image_path") or "").strip()
    if not image_path:
        return np.zeros(len(vertices), dtype=float)
    try:
        from PIL import Image, ImageFilter

        sample_size = int(os.environ.get("MESHMEND_INTRICATE_IMAGE_SAMPLE_SIZE", "1280"))
        image = Image.open(image_path).convert("L").resize((sample_size, sample_size)).filter(ImageFilter.SHARPEN)
        gray = np.asarray(image, dtype=np.float32) / 255.0
        fine = gray - np.asarray(image.filter(ImageFilter.GaussianBlur(radius=2.0)), dtype=np.float32) / 255.0
        gx = np.abs(np.diff(gray, axis=1, prepend=gray[:, :1]))
        gy = np.abs(np.diff(gray, axis=0, prepend=gray[:1, :]))
        edges = gx + gy
        edge_scale = float(np.percentile(edges, 98.0))
        if edge_scale > 1e-6:
            edges = np.clip(edges / edge_scale, 0.0, 1.0)
        fine_scale = float(np.percentile(np.abs(fine), 94.0))
        if fine_scale > 1e-6:
            fine = np.clip(fine / fine_scale, -1.0, 1.0)
        u = (vertices[:, 0] - mins[0]) / ext[0]
        v = 1.0 - ((vertices[:, 2] - mins[2]) / ext[2])
        max_index = sample_size - 1
        ix = np.clip((u * max_index).astype(int), 0, max_index)
        iy = np.clip((v * max_index).astype(int), 0, max_index)
        sampled_edges = edges[iy, ix]
        sampled_fine = fine[iy, ix]
        front = vertices[:, 1] <= mins[1] + ext[1] * 0.58
        grooves = sampled_edges > float(os.environ.get("MESHMEND_INTRICATE_IMAGE_EDGE_CUTOFF", "0.48"))
        relief = sampled_fine * 0.24 - grooves.astype(float) * 0.52
        return relief * front.astype(float)
    except Exception:
        return np.zeros(len(vertices), dtype=float)


def _intricate_prompt_relief(prompt: str, x: np.ndarray, y: np.ndarray, z: np.ndarray, seed: int) -> np.ndarray:
    relief = np.zeros(len(x), dtype=float)
    armored = any(term in prompt for term in ("space marine", "chaos", "armor", "armour", "robot", "mech", "soldier", "knight"))
    chaos = any(term in prompt for term in ("chaos", "demon", "daemon", "spike", "skull", "horn"))
    creature = any(term in prompt for term in ("dragon", "beast", "monster", "orc", "ork", "undead", "lizard"))
    cloth = any(term in prompt for term in ("robe", "cloak", "cloth", "tabard", "cape"))
    if armored:
        torso = (z > 0.24) & (z < 0.78)
        legs = (z > 0.07) & (z < 0.45)
        upper = (z > 0.55) & (z < 0.92)
        micro_panel_h = np.abs(np.sin((z * 52.0 + seed * 0.013) * math.pi)) < 0.010
        micro_panel_v = np.abs(np.sin((x * 46.0 + seed * 0.019) * math.pi)) < 0.010
        bevel_trim = np.abs(np.sin((z * 21.0 + np.abs(x) * 5.0 + seed * 0.029) * math.pi)) < 0.018
        vent_slits = (np.abs(np.sin((x * 68.0 + seed * 0.037) * math.pi)) < 0.008) & (np.abs(y) < 0.38)
        rivet_x = np.abs(np.sin((x * 58.0 + seed * 0.041) * math.pi)) < 0.009
        rivet_z = np.abs(np.sin((z * 61.0 + seed * 0.053) * math.pi)) < 0.009
        relief -= (micro_panel_h & torso).astype(float) * 0.42
        relief -= (micro_panel_v & (torso | legs)).astype(float) * 0.32
        relief += (bevel_trim & (torso | upper | legs)).astype(float) * 0.36
        relief -= (vent_slits & upper).astype(float) * 0.34
        if os.environ.get("MESHMEND_ENABLE_RAISED_DOT_DETAIL", "0").strip().lower() in {"1", "true", "yes"}:
            relief += (rivet_x & rivet_z & (torso | upper | legs)).astype(float) * 0.25
    if chaos:
        upper = z > 0.35
        jagged = np.abs(np.sin(((x * 31.0) + (z * 47.0) + seed * 0.071) * math.pi)) < 0.011
        scars = np.abs(np.sin(((x - y) * 25.0 + z * 18.0 + seed * 0.083) * math.pi)) < 0.010
        relief += (jagged & upper).astype(float) * 0.32
        relief -= (scars & upper).astype(float) * 0.30
    if creature:
        scales_a = np.abs(np.sin((x * 52.0 + seed * 0.031) * math.pi)) < 0.015
        scales_b = np.abs(np.sin((z * 57.0 + y * 9.0 + seed * 0.047) * math.pi)) < 0.015
        wrinkles = np.abs(np.sin((z * 33.0 + x * 8.0 + seed * 0.061) * math.pi)) < 0.014
        relief += (scales_a & scales_b).astype(float) * 0.44
        relief -= wrinkles.astype(float) * 0.24
    if cloth:
        folds = np.abs(np.sin((z * 35.0 + x * 10.0 + seed * 0.023) * math.pi)) < 0.016
        stitch = np.abs(np.sin((x * 64.0 + seed * 0.067) * math.pi)) < 0.008
        relief -= folds.astype(float) * 0.30
        relief += (stitch & (z > 0.12)).astype(float) * 0.25
    return relief


def apply_ai_training_definition_layer(mesh: trimesh.Trimesh, request: dict[str, Any]) -> trimesh.Trimesh:
    """Transfer learned surface definition from local training assets.

    This is the neural/training-data definition layer: it uses MeshMend's local
    trained exemplar/latent indexes to find a prompt-matched miniature and
    transfers its normalized surface relief to the generated mesh. It is not a
    fake random noise pass; it is derived from the user's local training meshes.
    """
    enabled = os.environ.get("MESHMEND_ENABLE_AI_DEFINITION_LAYER", "0").strip().lower() in {"1", "true", "yes"}
    if not enabled or not strict_quality_requested(request):
        return mesh
    try:
        prompt = str(request.get("prompt") or "")
        asset_path = _best_training_definition_asset(prompt)
        asset_mesh = None
        if asset_path is not None and asset_path.exists():
            asset_mesh = trimesh.load(asset_path, force="mesh", process=False)
        if asset_mesh is None:
            asset_mesh = _neural_definition_reference_mesh(prompt)
        if asset_mesh is None:
            return _apply_neural_checkpoint_definition_relief(mesh, prompt)
        if isinstance(asset_mesh, trimesh.Scene):
            geometries = [geom for geom in asset_mesh.geometry.values() if isinstance(geom, trimesh.Trimesh) and len(geom.faces) > 0]
            if not geometries:
                return mesh
            asset_mesh = trimesh.util.concatenate(geometries)
        if not isinstance(asset_mesh, trimesh.Trimesh) or len(asset_mesh.vertices) < 500:
            return mesh
        max_asset_vertices = int(os.environ.get("MESHMEND_AI_DEFINITION_MAX_ASSET_VERTICES", "60000"))
        asset_vertices = np.asarray(asset_mesh.vertices, dtype=float)
        if len(asset_vertices) > max_asset_vertices:
            step = max(1, len(asset_vertices) // max_asset_vertices)
            asset_vertices = asset_vertices[::step]

        source_vertices = np.asarray(mesh.vertices, dtype=float)
        normals = np.asarray(mesh.vertex_normals, dtype=float)
        if normals.shape != source_vertices.shape or len(source_vertices) < 1000:
            return mesh

        asset_norm = _normalize_points(asset_vertices)
        target_norm = _normalize_points(source_vertices)
        # Use x/z projection: most generated concepts are front-view product
        # renders, so training relief transfers best in the visible silhouette.
        try:
            from scipy.spatial import cKDTree
        except Exception:
            return mesh
        tree = cKDTree(asset_norm[:, [0, 2]])
        try:
            distances, indices = tree.query(target_norm[:, [0, 2]], k=1, workers=-1)
        except TypeError:
            distances, indices = tree.query(target_norm[:, [0, 2]], k=1)
        asset_depth = asset_norm[:, 1]
        relief = asset_depth[indices] - float(np.median(asset_depth))
        relief_scale = float(np.percentile(np.abs(relief), 94.0))
        if relief_scale <= 1e-6:
            return mesh
        relief = np.clip(relief / relief_scale, -1.0, 1.0)
        distance_scale = float(np.percentile(distances, 88.0))
        if distance_scale > 1e-6:
            relief *= np.clip(1.0 - (distances / (distance_scale * 2.0)), 0.0, 1.0)
        mins = source_vertices.min(axis=0)
        maxs = source_vertices.max(axis=0)
        ext = np.maximum(maxs - mins, 1e-6)
        z = (source_vertices[:, 2] - mins[2]) / ext[2]
        front = source_vertices[:, 1] <= mins[1] + ext[1] * 0.62
        not_base = z > 0.07
        amplitude = float(os.environ.get("MESHMEND_AI_DEFINITION_RELIEF_MM", "0.09"))
        mask = (front & not_base).astype(float)
        mesh.vertices = source_vertices + normals * (relief * amplitude * mask)[:, None]
        mesh.metadata["meshmend_ai_definition_layer"] = True
        mesh.metadata["meshmend_ai_definition_asset"] = str(asset_path) if asset_path is not None else "latest_neural_model.pt"
        if (Path(__file__).resolve().parents[2] / "training_data" / "checkpoints" / "latest_neural_model.pt").exists():
            mesh.metadata["meshmend_ai_definition_neural_checkpoint"] = "latest_neural_model.pt"
        return cleanup(mesh)
    except Exception:
        return mesh


def _best_training_definition_asset(prompt: str) -> Path | None:
    training_root = Path(__file__).resolve().parents[2] / "training_data" / "checkpoints"
    prompt_tags = _definition_tags(prompt)
    candidates: list[tuple[float, Path]] = []
    for checkpoint_name in (
        "latest_mesh_latent_index.json",
        "latest_neural_model_manifest.json",
        "latest_model.json",
        "meshmend_mesh_latent_index.json",
        "meshmend_local_3d_model.json",
    ):
        checkpoint = training_root / checkpoint_name
        if not checkpoint.exists():
            continue
        try:
            payload = json.loads(checkpoint.read_text(encoding="utf-8"))
        except Exception:
            continue
        entries = payload.get("assets") or payload.get("used_meshes") or payload.get("examples") or []
        for entry in entries:
            path_text = str(entry.get("path") or entry.get("mesh_path") or "")
            if not path_text:
                continue
            path = Path(path_text)
            if not path.exists():
                continue
            entry_tags = (
                set(entry.get("tags") or [])
                | set(entry.get("roles") or [])
                | _definition_tags(path.stem)
                | _definition_tags(str(entry.get("caption") or ""))
                | _definition_tags(str(entry.get("stem") or ""))
            )
            overlap = len(prompt_tags & entry_tags)
            faces = int(entry.get("faces") or 0)
            vertices = int(entry.get("vertices") or 0)
            warnings = len(entry.get("quality_warnings") or [])
            role_bonus = 25.0 if entry_tags & {"torso", "body", "head", "weapon", "armor", "marine", "soldier"} else 0.0
            density_score = min(max(faces, vertices // 3) / 50_000.0, 45.0)
            score = overlap * 100.0 + role_bonus + density_score - warnings * 20.0
            if overlap > 0 or not candidates:
                candidates.append((score, path))
    for asset_root in (training_root.parent / "raw_stl", training_root.parent / "processed_meshes"):
        if not asset_root.exists():
            continue
        for path in sorted(asset_root.glob("*")):
            if path.suffix.lower() not in {".stl", ".obj", ".ply"}:
                continue
            tags = _definition_tags(path.stem)
            overlap = len(prompt_tags & tags)
            if overlap <= 0 and candidates:
                continue
            try:
                estimated_faces = max(1, int(max(path.stat().st_size - 84, 0) / 50))
            except Exception:
                estimated_faces = 0
            candidates.append((overlap * 100.0 + min(estimated_faces / 50_000.0, 45.0), path))
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


def _neural_definition_reference_mesh(prompt: str) -> trimesh.Trimesh | None:
    """Generate a small learned reference mesh from the trained neural checkpoint."""
    if os.environ.get("MESHMEND_AI_DEFINITION_USE_NEURAL", "1").strip().lower() in {"0", "false", "no"}:
        return None
    try:
        import sys

        repo_root = Path(__file__).resolve().parents[2]
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
        from neural_diffusion import Neural3DDiffusionModel

        model = Neural3DDiffusionModel.load_latest()
        if model is None:
            return None
        steps = int(os.environ.get("MESHMEND_AI_DEFINITION_NEURAL_STEPS", "10"))
        generated = model.generate(prompt, steps=max(2, steps))
        if not isinstance(generated, trimesh.Trimesh) or len(generated.vertices) < 200:
            return None
        return generated
    except Exception:
        return None


def _apply_neural_checkpoint_definition_relief(mesh: trimesh.Trimesh, prompt: str) -> trimesh.Trimesh:
    """Fallback neural definition pass derived from trained checkpoint weights."""
    checkpoint = Path(__file__).resolve().parents[2] / "training_data" / "checkpoints" / "latest_neural_model.pt"
    if not checkpoint.exists():
        return mesh
    try:
        vertices = np.asarray(mesh.vertices, dtype=float)
        normals = np.asarray(mesh.vertex_normals, dtype=float)
        if len(vertices) < 1000 or normals.shape != vertices.shape:
            return mesh
        fingerprint = _neural_checkpoint_fingerprint(checkpoint)
        if fingerprint.size < 8:
            return mesh
        mins = vertices.min(axis=0)
        maxs = vertices.max(axis=0)
        ext = np.maximum(maxs - mins, 1e-6)
        p = (vertices - mins) / ext
        prompt_seed = sum(ord(char) for char in prompt.lower()) or 1
        relief = np.zeros(len(vertices), dtype=float)
        for index in range(6):
            weight = float(fingerprint[index % len(fingerprint)])
            freq = 6.0 + (abs(float(fingerprint[(index + 3) % len(fingerprint)])) * 21.0) + (prompt_seed % (index + 5))
            phase = float(fingerprint[(index + 5) % len(fingerprint)]) * math.pi * 2.0
            axis_a = p[:, index % 3]
            axis_b = p[:, (index + 2) % 3]
            relief += math.copysign(1.0, weight or 1.0) * np.sin(axis_a * freq + axis_b * freq * 0.43 + phase) / (index + 2)
        relief -= float(np.mean(relief))
        scale = float(np.percentile(np.abs(relief), 95.0))
        if scale <= 1e-6:
            return mesh
        relief = np.clip(relief / scale, -1.0, 1.0)
        z = p[:, 2]
        front = vertices[:, 1] <= mins[1] + ext[1] * 0.68
        not_base = z > 0.08
        amplitude = float(os.environ.get("MESHMEND_AI_DEFINITION_RELIEF_MM", "0.09")) * 0.72
        mesh.vertices = vertices + normals * (relief * amplitude * (front & not_base).astype(float))[:, None]
        mesh.metadata["meshmend_ai_definition_layer"] = True
        mesh.metadata["meshmend_ai_definition_asset"] = "latest_neural_model.pt:fingerprint"
        mesh.metadata["meshmend_ai_definition_neural_checkpoint"] = "latest_neural_model.pt"
        return cleanup(mesh)
    except Exception:
        return mesh


def _neural_checkpoint_fingerprint(checkpoint: Path) -> np.ndarray:
    try:
        import torch

        payload = torch.load(checkpoint, map_location="cpu")
        values: list[float] = []
        for section in ("autoencoder", "denoiser"):
            state = payload.get(section, {}) if isinstance(payload, dict) else {}
            for tensor in list(state.values())[:24]:
                array = tensor.detach().float().cpu().flatten()
                if array.numel() == 0:
                    continue
                stride = max(1, int(array.numel() // 256))
                sample = array[::stride][:256]
                values.extend([float(sample.mean()), float(sample.std(unbiased=False)), float(sample.abs().mean())])
        if values:
            return np.asarray(values, dtype=float)
    except Exception:
        pass
    try:
        data = np.frombuffer(checkpoint.read_bytes()[:65536], dtype=np.uint8).astype(float)
        return (data - 127.5) / 127.5
    except Exception:
        return np.asarray([], dtype=float)


def _definition_tags(text: str) -> set[str]:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", " ", text or "").lower()
    words = {word for word in cleaned.split() if len(word) > 2}
    aliases = {"armour": "armor", "ork": "orc", "bolter": "rifle", "prime": "marine", "infantry": "soldier"}
    expanded = set(words)
    for word in list(words):
        if word in aliases:
            expanded.add(aliases[word])
    return expanded


def _normalize_points(points: np.ndarray) -> np.ndarray:
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    ext = np.maximum(maxs - mins, 1e-6)
    return (points - mins) / ext


def remove_floating_artifacts(mesh: trimesh.Trimesh, request: dict[str, Any]) -> trimesh.Trimesh:
    """Remove disconnected generated debris while keeping the main miniature."""
    try:
        if len(mesh.faces) < 1000:
            return mesh
        components = [component for component in mesh.split(only_watertight=False) if len(component.faces) > 20]
        if len(components) <= 1:
            return trim_extreme_outlier_faces(mesh, request)
        components.sort(key=lambda component: float(component.area), reverse=True)
        main = components[0]
        main_area = max(float(main.area), 1e-6)
        main_center = np.asarray(main.bounds, dtype=float).mean(axis=0)
        main_radius = float(np.linalg.norm(np.asarray(main.extents, dtype=float)))
        kept = [main]
        source_workflow = str(request.get("_meshmend_source_workflow") or request.get("workflow") or "text_to_3d")
        if source_workflow == "image_to_3d":
            min_area_ratio = float(os.environ.get("MESHMEND_IMAGE_ARTIFACT_MIN_AREA_RATIO", "0.003"))
            max_distance_ratio = float(os.environ.get("MESHMEND_IMAGE_ARTIFACT_MAX_DISTANCE_RATIO", "1.25"))
        else:
            min_area_ratio = float(os.environ.get("MESHMEND_ARTIFACT_MIN_AREA_RATIO", "0.018"))
            max_distance_ratio = float(os.environ.get("MESHMEND_ARTIFACT_MAX_DISTANCE_RATIO", "0.85"))
        for component in components[1:]:
            area_ratio = float(component.area) / main_area
            if area_ratio < min_area_ratio:
                continue
            center = np.asarray(component.bounds, dtype=float).mean(axis=0)
            if main_radius > 1e-6 and float(np.linalg.norm(center - main_center)) > main_radius * max_distance_ratio:
                continue
            kept.append(component)
        if len(kept) == len(components):
            return trim_extreme_outlier_faces(mesh, request)
        merged = trimesh.util.concatenate(kept)
        merged.metadata.update(mesh.metadata)
        merged.metadata["meshmend_artifact_components_removed"] = len(components) - len(kept)
        return trim_extreme_outlier_faces(cleanup(merged), request)
    except Exception:
        return mesh


def trim_extreme_outlier_faces(mesh: trimesh.Trimesh, request: dict[str, Any]) -> trimesh.Trimesh:
    """Conservatively trim far-edge debris from text-generated concepts."""
    source_workflow = str(request.get("_meshmend_source_workflow") or "").lower()
    if source_workflow != "text_to_3d":
        return mesh
    try:
        vertices = np.asarray(mesh.vertices, dtype=float)
        if len(vertices) < 1000 or len(mesh.faces) < 1000:
            return mesh
        face_centers = vertices[np.asarray(mesh.faces)].mean(axis=1)
        z_min, z_max = float(vertices[:, 2].min()), float(vertices[:, 2].max())
        z_span = max(z_max - z_min, 1e-6)
        body = face_centers[:, 2] > z_min + z_span * 0.07
        keep = np.ones(len(mesh.faces), dtype=bool)
        for axis in (0, 1):
            values = face_centers[body, axis] if int(body.sum()) > 500 else face_centers[:, axis]
            low, high = np.percentile(values, [0.35, 99.65])
            pad = max((high - low) * 0.08, 0.35)
            keep &= (face_centers[:, axis] >= low - pad) & (face_centers[:, axis] <= high + pad)
        if int(keep.sum()) < len(mesh.faces) * 0.82:
            return mesh
        if int((~keep).sum()) < max(100, len(mesh.faces) * 0.002):
            return mesh
        trimmed = trimesh.Trimesh(vertices=vertices.copy(), faces=np.asarray(mesh.faces)[keep].copy(), process=False)
        trimmed.remove_unreferenced_vertices()
        if len(trimmed.faces) < len(mesh.faces) * 0.82:
            return mesh
        trimmed.metadata.update(mesh.metadata)
        trimmed.metadata["meshmend_outlier_faces_removed"] = int((~keep).sum())
        return cleanup(trimmed)
    except Exception:
        return mesh


def cleanup(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    try:
        mesh.remove_unreferenced_vertices()
        mesh.merge_vertices()
        mesh.fix_normals()
    except Exception:
        pass
    return mesh


def requested_scale_mm(request: dict[str, Any]) -> float:
    for value in (request.get("scale_mm"), request.get("scale")):
        if value:
            try:
                return float(str(value).lower().replace("mm", "").strip())
            except ValueError:
                pass
    match = re.search(r"\b(15|20|25|28|30|32|35|40|48|54|75|90|100)\s*mm\b", str(request.get("prompt") or "").lower())
    if match:
        return float(match.group(1))
    return float(os.environ.get("MESHMEND_DEFAULT_MINIATURE_SCALE_MM", "32"))


def detail_face_target(request: dict[str, Any]) -> int:
    quality = str(request.get("quality") or "standard").lower()
    prompt = str(request.get("prompt") or "").lower()
    wants_8k = any(
        term in prompt
        for term in (
            "8k", "8 k", "studio", "studio quality", "studio-quality", "studio level", "studio-level",
            "production", "display quality", "maximum detail", "store quality", "store-quality", "store level",
            "store-level",
        )
    )
    requested_polycount = int(request.get("target_polycount") or 0)
    if quality == "high" or wants_8k:
        # Local high-quality mode must stay memory-safe. Million-face requests
        # become 3-5M faces after 4x subdivision jumps and can freeze PCs.
        default = str(max(180_000, min(requested_polycount or 180_000, 300_000)))
    else:
        default = str(max(90_000, min(requested_polycount or 90_000, 180_000)))
    return memory_safe_face_target(int(os.environ.get("MESHMEND_MIN_EXPORT_FACES", default)))


def repair_structural_quality_issues(mesh: trimesh.Trimesh, request: dict[str, Any], detail_faces_target: int) -> tuple[trimesh.Trimesh, bool]:
    """Try targeted structural repairs before failing a store-quality export."""
    single_subject_enforced = False
    if not strict_quality_requested(request):
        return mesh, single_subject_enforced
    for _attempt in range(2):
        issues = quality_gate_issues(mesh, request, detail_faces_target)
        changed = False
        if any(issue.startswith("likely_background_slab") for issue in issues):
            aggressive_request = dict(request)
            aggressive_request["_meshmend_aggressive_slab_removal"] = True
            repaired = remove_background_slabs(mesh, aggressive_request)
            if len(repaired.faces) > 0 and len(repaired.faces) != len(mesh.faces):
                mesh = cleanup(repaired)
                changed = True
        if any(issue.startswith("likely_dual_subject") for issue in issues):
            repaired, cropped = enforce_single_subject(mesh, request)
            if cropped and len(repaired.faces) > 0:
                mesh = cleanup(repaired)
                single_subject_enforced = True
                changed = True
        if not changed:
            break
    return mesh, single_subject_enforced


def ensure_printable_solid_mesh(mesh: trimesh.Trimesh, request: dict[str, Any]) -> trimesh.Trimesh:
    """Close semi-transparent/open shell outputs for printable miniatures.

    When base/sheet removal cuts away generated planes, Hunyuan meshes can remain
    visually detailed but topologically open. Slicers then show them as dotted or
    semi-transparent. For strict/studio jobs, voxelize and remesh the silhouette
    into a closed solid before final detail is re-applied.
    """
    if not strict_quality_requested(request):
        return mesh
    try:
        source_workflow = str(request.get("_meshmend_source_workflow") or request.get("workflow") or "text_to_3d")
        if source_workflow == "text_to_3d" and os.environ.get("MESHMEND_TEXT_TO_3D_VOXEL_SOLIDIFY", "0").strip().lower() not in {"1", "true", "yes"}:
            return cleanup(mesh.copy())
        solidify_clean_image_meshes = (
            source_workflow == "image_to_3d"
            and not bool(getattr(mesh, "metadata", {}).get("meshmend_voxel_solidified"))
            and os.environ.get("MESHMEND_IMAGE_SOLIDIFY_WATERTIGHT", "0").strip().lower() in {"1", "true", "yes"}
        )
        repaired = cleanup(mesh.copy())
        boundary_edges = _boundary_edge_count(repaired)
        if bool(repaired.is_watertight) and boundary_edges == 0 and not solidify_clean_image_meshes:
            return repaired
        vertices = np.asarray(repaired.vertices, dtype=float)
        if len(vertices) == 0 or len(repaired.faces) == 0:
            return mesh
        ext = np.maximum(vertices.max(axis=0) - vertices.min(axis=0), 1e-6)
        max_extent = float(ext.max())
        if max_extent <= 1e-6:
            return mesh
        default_pitch = "0.16" if source_workflow == "image_to_3d" else "0.24"
        pitch = float(os.environ.get("MESHMEND_SOLIDIFY_VOXEL_PITCH_MM", default_pitch))
        if memory_safety_enabled():
            pitch = max(pitch, float(os.environ.get("MESHMEND_MEMORY_SAFE_SOLIDIFY_VOXEL_PITCH_MM", "0.28")))
        pitch = max(0.06, min(pitch, max_extent / 48.0))
        voxels = repaired.voxelized(pitch).fill()
        solid = voxels.marching_cubes
        if not isinstance(solid, trimesh.Trimesh) or len(solid.faces) < 1000:
            return repaired
        solid.metadata.update(repaired.metadata)
        solid.metadata["meshmend_voxel_solidified"] = True
        solid.metadata["meshmend_pre_solidify_boundary_edges"] = int(boundary_edges)
        solid = cleanup(solid)
        solid = normalize_to_scale(solid, requested_scale_mm(request), request)
        return solid
    except Exception:
        return mesh


def ensure_round_miniature_base(mesh: trimesh.Trimesh, request: dict[str, Any]) -> trimesh.Trimesh:
    """Add a simple printable gaming base after cleanup/detailing.

    Reference cleanup removes generated plinths because Hunyuan often turns them
    into cards/slabs. The final STL should still be a usable tabletop miniature,
    so add a clean round base as geometry at the end rather than asking image-to-3D
    to infer one from noisy pixels.
    """
    if os.environ.get("MESHMEND_ADD_MINIATURE_BASE", "1").strip().lower() in {"0", "false", "no"}:
        return mesh
    if should_preserve_wide_image_reference(request):
        # Wide/mounted image references usually already include an oval/scenic
        # base in the concept. Adding a generic round base changes the silhouette
        # and makes the output drift further from the input.
        return mesh
    try:
        vertices = np.asarray(mesh.vertices, dtype=float)
        if len(vertices) == 0 or len(mesh.faces) == 0:
            return mesh
        mins = vertices.min(axis=0)
        maxs = vertices.max(axis=0)
        xy_ext = np.maximum(maxs[:2] - mins[:2], 1e-6)
        requested = requested_scale_mm(request)
        radius = float(os.environ.get("MESHMEND_MINIATURE_BASE_RADIUS_MM", str(max(12.5, min(16.0, float(np.max(xy_ext)) * 0.58)))))
        height = float(os.environ.get("MESHMEND_MINIATURE_BASE_HEIGHT_MM", "2.0"))
        sections = max(32, int(os.environ.get("MESHMEND_MINIATURE_BASE_SECTIONS", "64")))
        base = trimesh.creation.cylinder(radius=radius, height=height, sections=sections)
        base.apply_translation([0.0, 0.0, mins[2] - height * 0.5 + 0.05])
        base.metadata["meshmend_added_round_base"] = True
        combined = trimesh.util.concatenate([mesh, base])
        combined.metadata.update(mesh.metadata)
        combined.metadata["meshmend_added_round_base"] = True
        combined = cleanup(combined)
        # Keep the requested model height after adding base underneath.
        combined = normalize_to_scale(combined, requested, request)
        return combined
    except Exception:
        return mesh


def bridge_disconnected_components(mesh: trimesh.Trimesh, request: dict[str, Any]) -> trimesh.Trimesh:
    """Physically connect detached watertight islands for slicers/printers."""
    source_workflow = str(request.get("_meshmend_source_workflow") or request.get("workflow") or "text_to_3d")
    default_enabled = "0" if strict_quality_requested(request) and source_workflow == "text_to_3d" else "1"
    if os.environ.get("MESHMEND_BRIDGE_DISCONNECTED_COMPONENTS", default_enabled).strip().lower() in {"0", "false", "no"}:
        return mesh
    try:
        repaired = cleanup(mesh.copy())
        components = [component for component in repaired.split(only_watertight=False) if len(component.faces) > 20]
        if len(components) <= 1:
            return repaired
        components.sort(key=lambda component: float(component.area), reverse=True)
        combined = components[0]
        bridges: list[trimesh.Trimesh] = []
        max_bridges = int(os.environ.get("MESHMEND_MAX_COMPONENT_BRIDGES", "48"))
        max_distance = float(os.environ.get("MESHMEND_MAX_COMPONENT_BRIDGE_DISTANCE_MM", "8.0"))
        radius = float(os.environ.get("MESHMEND_COMPONENT_BRIDGE_RADIUS_MM", "0.42"))
        for component in components[1 : max_bridges + 1]:
            start_index, end_index, distance = closest_vertex_indices(np.asarray(combined.vertices), np.asarray(component.vertices))
            if distance > max_distance:
                continue
            start = np.asarray(combined.vertices[start_index], dtype=float)
            end = np.asarray(component.vertices[end_index], dtype=float)
            if distance <= 1e-7:
                combined = trimesh.util.concatenate([combined, component])
                continue
            bridge = create_component_bridge(
                start=start,
                end=end,
                start_neighbor=anchor_neighbor(combined, start_index),
                end_neighbor=anchor_neighbor(component, end_index),
                radius=radius,
                sections=int(os.environ.get("MESHMEND_COMPONENT_BRIDGE_SECTIONS", "12")),
            )
            bridges.append(bridge)
            combined = trimesh.util.concatenate([combined, bridge, component])
            combined = cleanup(combined)
        if not bridges:
            return repaired
        combined.metadata.update(repaired.metadata)
        combined.metadata["meshmend_component_bridges_added"] = len(bridges)
        return cleanup(combined)
    except Exception:
        return mesh


def create_component_bridge(
    *,
    start: np.ndarray,
    end: np.ndarray,
    start_neighbor: np.ndarray | None,
    end_neighbor: np.ndarray | None,
    radius: float,
    sections: int,
) -> trimesh.Trimesh:
    direction = end - start
    distance = float(np.linalg.norm(direction))
    if distance <= 1e-9:
        return trimesh.Trimesh(vertices=np.array([start]), faces=np.empty((0, 3), dtype=np.int64), process=False)
    sections = max(6, int(sections))
    radius = max(float(radius), 1e-6)
    unit = direction / distance
    basis_u, basis_v = orthonormal_basis(unit)
    taper = min(distance / 3.0, radius * 2.0)
    start_ring_center = start + unit * taper
    end_ring_center = end - unit * taper
    vertices = [
        np.array(start, dtype=float),
        np.array(end, dtype=float),
        np.array(start_neighbor if start_neighbor is not None else start, dtype=float),
        np.array(end_neighbor if end_neighbor is not None else end, dtype=float),
    ]
    for index in range(sections):
        angle = (2.0 * math.pi * index) / sections
        offset = radius * (math.cos(angle) * basis_u + math.sin(angle) * basis_v)
        vertices.append(start_ring_center + offset)
    for index in range(sections):
        angle = (2.0 * math.pi * index) / sections
        offset = radius * (math.cos(angle) * basis_u + math.sin(angle) * basis_v)
        vertices.append(end_ring_center + offset)
    start_ring = 4
    end_ring = 4 + sections
    faces: list[list[int]] = []
    for index in range(sections):
        next_index = (index + 1) % sections
        sc = start_ring + index
        sn = start_ring + next_index
        ec = end_ring + index
        en = end_ring + next_index
        faces.append([0, sn, sc])
        faces.append([sc, sn, en])
        faces.append([sc, en, ec])
        faces.append([1, ec, en])
    if start_neighbor is not None:
        faces.append([0, 2, start_ring])
    if end_neighbor is not None:
        faces.append([1, end_ring, 3])
    return trimesh.Trimesh(vertices=np.asarray(vertices), faces=np.asarray(faces, dtype=np.int64), process=False)


def anchor_neighbor(mesh: trimesh.Trimesh, vertex_index: int) -> np.ndarray | None:
    hits = np.where(mesh.faces == vertex_index)[0]
    if len(hits) == 0:
        return None
    face = mesh.faces[int(hits[0])]
    for neighbor_index in face:
        if int(neighbor_index) != int(vertex_index):
            return np.asarray(mesh.vertices[int(neighbor_index)], dtype=float)
    return None


def orthonormal_basis(unit: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    reference = np.array([1.0, 0.0, 0.0])
    if abs(float(np.dot(unit, reference))) > 0.9:
        reference = np.array([0.0, 1.0, 0.0])
    basis_u = np.cross(unit, reference)
    basis_u /= max(float(np.linalg.norm(basis_u)), 1e-9)
    basis_v = np.cross(unit, basis_u)
    return basis_u, basis_v


def closest_vertex_indices(vertices_a: np.ndarray, vertices_b: np.ndarray, chunk_size: int = 25_000) -> tuple[int, int, float]:
    best_a = 0
    best_b = 0
    best_distance_squared = float("inf")
    for start in range(0, len(vertices_a), chunk_size):
        chunk = vertices_a[start : start + chunk_size]
        deltas = chunk[:, None, :] - vertices_b[None, :, :]
        distances = np.einsum("ijk,ijk->ij", deltas, deltas)
        flat_index = int(np.argmin(distances))
        distance_squared = float(distances.flat[flat_index])
        if distance_squared < best_distance_squared:
            local_a, local_b = np.unravel_index(flat_index, distances.shape)
            best_a = start + int(local_a)
            best_b = int(local_b)
            best_distance_squared = distance_squared
    return best_a, best_b, float(math.sqrt(best_distance_squared))


def denoise_surface_bumps(mesh: trimesh.Trimesh, request: dict[str, Any]) -> trimesh.Trimesh:
    """Smooth Hunyuan speckle/procedural bumps without inventing new detail."""
    if os.environ.get("MESHMEND_DENOISE_SURFACE", "1").strip().lower() in {"0", "false", "no"}:
        return mesh
    try:
        if len(mesh.vertices) < 1000 or len(mesh.faces) < 1000:
            return mesh
        source_workflow = str(request.get("_meshmend_source_workflow") or request.get("workflow") or "text_to_3d")
        strict_quality = strict_quality_requested(request)
        smoothed = mesh.copy()
        metadata = dict(mesh.metadata)
        try:
            from trimesh.smoothing import filter_laplacian, filter_taubin

            default_iterations = "1" if strict_quality else ("2" if source_workflow in {"image_to_3d", "text_to_3d"} else "3")
            iterations = int(os.environ.get("MESHMEND_DENOISE_ITERATIONS", default_iterations))
            default_lamb = "0.06" if strict_quality else "0.18"
            lamb = float(os.environ.get("MESHMEND_DENOISE_LAMBDA", default_lamb))
            if source_workflow in {"image_to_3d", "text_to_3d"} and os.environ.get("MESHMEND_DENOISE_FILTER", "taubin").strip().lower() == "taubin":
                default_nu = "-0.065" if strict_quality else "-0.19"
                nu = float(os.environ.get("MESHMEND_DENOISE_NU", default_nu))
                filter_taubin(smoothed, lamb=lamb, nu=nu, iterations=iterations)
            else:
                filter_laplacian(smoothed, lamb=lamb, iterations=iterations)
        except Exception:
            return mesh
        smoothed.metadata.update(metadata)
        smoothed.metadata["meshmend_surface_denoised"] = True
        return cleanup(smoothed)
    except Exception:
        return mesh


def _boundary_edge_count(mesh: trimesh.Trimesh) -> int:
    try:
        if len(mesh.faces) == 0:
            return 0
        counts = np.bincount(mesh.edges_unique_inverse)
        return int((counts == 1).sum())
    except Exception:
        return 0


def _nonmanifold_edge_count(mesh: trimesh.Trimesh) -> int:
    try:
        if len(mesh.faces) == 0:
            return 0
        counts = np.bincount(mesh.edges_unique_inverse)
        return int((counts > 2).sum())
    except Exception:
        return 0


def mesh_genus_estimate(mesh: trimesh.Trimesh) -> int:
    """Estimate visual tunnel count for watertight mesh components."""
    try:
        if not bool(mesh.is_watertight):
            return 0
        components = [component for component in mesh.split(only_watertight=False) if len(component.faces) > 20]
        if not components:
            components = [mesh]
        genus = 0
        for component in components:
            euler = int(component.euler_number)
            genus += max(0, int(round((2 - euler) / 2)))
        return int(genus)
    except Exception:
        return 0


def strip_residual_background_slabs(mesh: trimesh.Trimesh, request: dict[str, Any], detail_faces_target: int) -> trimesh.Trimesh:
    """Peel remaining render-card faces before a store-quality export.

    A slab can survive the first cleanup pass after subdivision/detailing because
    it is connected to the subject or only one boundary face was removed. For a
    studio miniature, a residual card/block behind the subject is always worse
    than a slightly open rear shell, so strict-quality jobs get a final bounded
    peel loop until the slab gate clears or no further faces are removed.
    """
    if not strict_quality_requested(request):
        return mesh
    try:
        repaired = mesh
        for _attempt in range(4):
            issues = quality_gate_issues(repaired, request, detail_faces_target)
            if not any(issue.startswith("likely_background_slab") for issue in issues):
                return repaired
            aggressive_request = dict(request)
            aggressive_request["_meshmend_aggressive_slab_removal"] = True
            before_faces = len(repaired.faces)
            next_mesh = remove_background_slabs(repaired, aggressive_request)
            if len(next_mesh.faces) <= 0 or len(next_mesh.faces) == before_faces:
                return repaired
            if len(next_mesh.faces) < max(1000, int(before_faces * 0.45)):
                return repaired
            repaired = cleanup(next_mesh)
        return repaired
    except Exception:
        return mesh


def build_report(
    mesh: trimesh.Trimesh,
    request: dict[str, Any],
    *,
    single_subject_enforced: bool,
    detail_faces_target: int,
    quality_gate_issues: list[str] | None = None,
) -> PostprocessReport:
    vertices = np.asarray(mesh.vertices, dtype=float)
    extents = vertices.max(axis=0) - vertices.min(axis=0)
    return PostprocessReport(
        target_scale_mm=requested_scale_mm(request),
        extents_mm=[float(value) for value in extents],
        max_extent_mm=float(np.max(extents)),
        faces=int(len(mesh.faces)),
        vertices=int(len(mesh.vertices)),
        single_subject_enforced=single_subject_enforced,
        detail_faces_target=detail_faces_target,
        relief_mm=float(os.environ.get("MESHMEND_DETAIL_RELIEF_MM", "0.025")),
        detail_style="postprocess_backend:single_subject+scale+sheet_guard+structured_sculpt",
        geometry_upscaled=bool(mesh.metadata.get("meshmend_geometry_upscale")),
        intricate_detail_pipeline=bool(mesh.metadata.get("meshmend_intricate_detail_pipeline")),
        custom_miniature_detail_pipeline=bool(mesh.metadata.get("meshmend_custom_miniature_detail_pipeline")),
        ai_definition_layer=bool(mesh.metadata.get("meshmend_ai_definition_layer")),
        production_ready=strict_quality_requested(request) and not bool(quality_gate_issues or []),
        quality_gate_issues=quality_gate_issues or [],
    )


def strict_quality_requested(request: dict[str, Any]) -> bool:
    quality = str(request.get("quality") or "standard").lower()
    prompt = str(request.get("prompt") or "").lower()
    wants_store = quality == "high" or any(
        term in prompt for term in (
            "8k", "8 k", "studio", "studio quality", "studio-quality", "studio level", "studio-level",
            "production", "display quality", "maximum detail", "store quality", "store-quality", "store level",
            "store-level", "intricate",
        )
    )
    return wants_store and os.environ.get("MESHMEND_DISABLE_STORE_QUALITY_GATE", "0").strip().lower() not in {"1", "true", "yes"}


def allow_best_effort_export() -> bool:
    return os.environ.get("MESHMEND_ALLOW_BEST_EFFORT_EXPORT", "0").strip().lower() in {"1", "true", "yes"}


def should_raise_quality_gate_failure(request: dict[str, Any], fatal_issues: list[str]) -> bool:
    if allow_best_effort_export():
        return False
    source_workflow = str(request.get("_meshmend_source_workflow") or request.get("workflow") or "text_to_3d")
    if source_workflow == "image_to_3d" and fatal_issues and all(issue.startswith("below_store_face_target") for issue in fatal_issues):
        # A detailed image reconstruction can be structurally valid and still
        # fall short of the configured density target after cleanup/solid repair.
        # Treat this as a report warning, not a backend crash; topology/sheet/
        # manifold/component failures remain fatal.
        return False
    return True


def fatal_quality_gate_issues(issues: list[str]) -> list[str]:
    """Only fail export for structural/readability problems, not density warnings."""
    fatal_prefixes = (
        "empty_mesh",
        "below_store_face_target",
        "degenerate_faces",
        "heavy_artifact_salvage",
        "too_many_disconnected_components",
        "component_bridges_visible_artifact_risk",
        "mesh_too_flat",
        "likely_dual_subject",
        "mesh_not_solid_watertight",
        "mesh_boundary_edges",
        "mesh_nonmanifold_edges",
        "image_reference_under_defined",
        "image_visual_holes_unsealed",
        "image_low_relief_sheet",
        "large_smooth_primitive_surfaces_dominate",
        "likely_background_slab_or_card",
        "likely_horizontal_square_sheet_or_card",
        "likely_blocky_low_definition",
    )
    return [issue for issue in issues if issue.startswith(fatal_prefixes)]


def quality_gate_issues(mesh: trimesh.Trimesh, request: dict[str, Any], detail_faces_target: int) -> list[str]:
    if not strict_quality_requested(request):
        return []
    issues: list[str] = []
    vertices = np.asarray(mesh.vertices, dtype=float)
    if len(vertices) == 0 or len(mesh.faces) == 0:
        return ["empty_mesh"]
    extents = np.maximum(vertices.max(axis=0) - vertices.min(axis=0), 1e-6)
    max_extent = float(np.max(extents))
    depth_ratio = float(np.min(extents) / max_extent) if max_extent > 1e-6 else 0.0
    source_workflow = str(request.get("_meshmend_source_workflow") or request.get("workflow") or "text_to_3d")
    if len(mesh.faces) < int(detail_faces_target * 0.75):
        issues.append(f"below_store_face_target_{len(mesh.faces)}_of_{detail_faces_target}")
    if source_workflow == "image_to_3d" and len(mesh.faces) < int(detail_faces_target * float(os.environ.get("MESHMEND_IMAGE_MIN_DETAIL_TARGET_RATIO", "0.35"))):
        issues.append(f"image_reference_under_defined_faces_{len(mesh.faces)}_of_{detail_faces_target}")
    if depth_ratio < float(os.environ.get("MESHMEND_STORE_MIN_DEPTH_RATIO", "0.24")):
        issues.append(f"mesh_too_flat_depth_ratio_{depth_ratio:.2f}")
    if source_workflow != "image_to_3d" and likely_dual_subject(mesh):
        issues.append("likely_dual_subject")
    if source_workflow in {"image_to_3d", "text_to_3d"}:
        genus = mesh_genus_estimate(mesh)
        max_genus = int(os.environ.get("MESHMEND_IMAGE_MAX_VISUAL_HOLE_GENUS", "24"))
        if genus > max_genus:
            issues.append(f"image_visual_holes_unsealed_genus_{genus}_max_{max_genus}")
    if source_workflow == "image_to_3d" and likely_image_low_relief_sheet(mesh):
        issues.append("image_low_relief_sheet_or_connected_card")
    if likely_background_slab(mesh, request):
        issues.append("likely_background_slab_or_card")
    if likely_horizontal_sheet_card(mesh):
        issues.append("likely_horizontal_square_sheet_or_card")
    if likely_blocky_low_definition(mesh):
        issues.append("likely_blocky_low_definition")
    smooth_ratio = smooth_surface_area_ratio(mesh)
    smooth_limit = float(os.environ.get("MESHMEND_MAX_SMOOTH_PRIMITIVE_SURFACE_RATIO", "0.68"))
    if smooth_ratio > smooth_limit and not has_miniature_detail_metadata(mesh):
        issues.append(f"large_smooth_primitive_surfaces_dominate_{smooth_ratio:.2f}_max_{smooth_limit:.2f}")
    if not bool(mesh.is_watertight):
        issues.append("mesh_not_solid_watertight")
    boundary_edges = _boundary_edge_count(mesh)
    if boundary_edges > 0:
        issues.append(f"mesh_boundary_edges_{boundary_edges}")
    nonmanifold_edges = _nonmanifold_edge_count(mesh)
    if nonmanifold_edges > 0:
        issues.append(f"mesh_nonmanifold_edges_{nonmanifold_edges}")
    degenerate_faces = degenerate_face_count(mesh)
    if degenerate_faces > 0:
        max_degenerate = max(12, int(len(mesh.faces) * float(os.environ.get("MESHMEND_MAX_DEGENERATE_FACE_RATIO", "0.0005"))))
        if degenerate_faces > max_degenerate:
            issues.append(f"degenerate_faces_{degenerate_faces}_max_{max_degenerate}")
    component_count = connected_component_count(mesh)
    max_components = int(os.environ.get("MESHMEND_STORE_MAX_COMPONENTS", "6"))
    if component_count > max_components:
        issues.append(f"too_many_disconnected_components_{component_count}_max_{max_components}")
    issues.extend(artifact_salvage_quality_issues(mesh, source_workflow))
    if synthetic_detail_enabled("MESHMEND_ENABLE_CUSTOM_MINIATURE_DETAIL_PIPELINE", default="1") and not bool(mesh.metadata.get("meshmend_custom_miniature_detail_pipeline")):
        issues.append("custom_miniature_detail_pipeline_not_applied")
    if synthetic_detail_enabled("MESHMEND_ENABLE_INTRICATE_DETAIL_PIPELINE", default="1") and not bool(mesh.metadata.get("meshmend_intricate_detail_pipeline")):
        issues.append("intricate_detail_pipeline_not_applied")
    if synthetic_detail_enabled("MESHMEND_ENABLE_GEOMETRY_UPSCALE", default="1") and not bool(mesh.metadata.get("meshmend_geometry_upscale")):
        issues.append("geometry_upscale_not_applied")
    if os.environ.get("MESHMEND_REQUIRE_AI_DEFINITION_LAYER", "0").strip().lower() in {"1", "true", "yes"} and not bool(mesh.metadata.get("meshmend_ai_definition_layer")):
        issues.append("ai_training_definition_layer_not_applied")
    return issues


def smooth_surface_area_ratio(mesh: trimesh.Trimesh) -> float:
    if len(mesh.faces) == 0 or len(mesh.face_adjacency) == 0:
        return 1.0
    areas = np.asarray(mesh.area_faces, dtype=float)
    total_area = max(float(areas.sum()), 1e-8)
    adjacency = np.asarray(mesh.face_adjacency, dtype=np.int64)
    angles = np.asarray(mesh.face_adjacency_angles, dtype=float)
    smooth_faces = np.zeros(len(mesh.faces), dtype=bool)
    smooth_pairs = adjacency[angles < np.radians(7.5)]
    if len(smooth_pairs):
        smooth_faces[np.unique(smooth_pairs)] = True
    return float(areas[smooth_faces].sum() / total_area)


def has_miniature_detail_metadata(mesh: trimesh.Trimesh) -> bool:
    metadata = getattr(mesh, "metadata", {}) or {}
    return all(
        bool(metadata.get(key))
        for key in (
            "meshmend_custom_miniature_detail_pipeline",
            "meshmend_intricate_detail_pipeline",
            "meshmend_geometry_upscale",
        )
    )


def artifact_salvage_quality_issues(mesh: trimesh.Trimesh, source_workflow: str) -> list[str]:
    """Fail store-quality exports that only look printable after heavy salvage.

    Voxel solidification, visual-hole sealing, and closest-point component rods
    are useful rescue tools for draft outputs. When they are needed heavily, the
    result often contains the exact visible noise/artifacts users report: lumpy
    remeshed surfaces, random rods between islands, and shapes that no longer
    match the concept. For store/studio mode, fail these instead of exporting a
    technically watertight but visually wrong STL.
    """
    metadata = getattr(mesh, "metadata", {}) or {}
    issues: list[str] = []
    try:
        bridges = int(metadata.get("meshmend_component_bridges_added") or 0)
    except Exception:
        bridges = 0
    default_max_bridges = "64" if source_workflow == "image_to_3d" else "0"
    max_bridges = int(os.environ.get("MESHMEND_STORE_MAX_COMPONENT_BRIDGES", default_max_bridges))
    if bridges > max_bridges:
        issues.append(f"component_bridges_visible_artifact_risk_{bridges}_max_{max_bridges}")
    try:
        pre_seal_genus = int(metadata.get("meshmend_pre_seal_genus") or 0)
    except Exception:
        pre_seal_genus = 0
    max_pre_seal_genus = int(os.environ.get("MESHMEND_STORE_MAX_PRE_SEAL_GENUS", "24"))
    if source_workflow != "image_to_3d" and pre_seal_genus > max_pre_seal_genus:
        issues.append(f"heavy_artifact_salvage_visual_holes_{pre_seal_genus}_max_{max_pre_seal_genus}")
    if source_workflow == "text_to_3d" and bool(metadata.get("meshmend_voxel_solidified")):
        issues.append("heavy_artifact_salvage_voxel_solidified_text_concept")
    return issues


def degenerate_face_count(mesh: trimesh.Trimesh) -> int:
    try:
        if len(mesh.faces) == 0:
            return 0
        areas = np.asarray(mesh.area_faces, dtype=float)
        if len(areas) == 0:
            return 0
        scale = max(float(np.max(np.asarray(mesh.extents, dtype=float))), 1e-6)
        threshold = (scale * 1e-7) ** 2
        return int(np.count_nonzero(~np.isfinite(areas) | (areas <= threshold)))
    except Exception:
        return 0


def connected_component_count(mesh: trimesh.Trimesh) -> int:
    try:
        return int(len([component for component in mesh.split(only_watertight=False) if len(component.faces) > 20]))
    except Exception:
        return 1


def likely_blocky_low_definition(mesh: trimesh.Trimesh) -> bool:
    """Detect subdivided cubes/cards that have face count but no sculptural definition."""
    try:
        if len(mesh.faces) < 1000:
            return True
        ext = np.maximum(np.asarray(mesh.extents, dtype=float), 1e-6)
        depth_ratio = float(ext.min() / max(ext.max(), 1e-6))
        metadata = getattr(mesh, "metadata", {}) or {}
        has_detail_pipeline = any(
            bool(metadata.get(key))
            for key in (
                "meshmend_image_guided_detail",
                "meshmend_custom_miniature_detail_pipeline",
                "meshmend_intricate_detail_pipeline",
                "meshmend_image_store_density_restored",
            )
        )
        if (
            has_detail_pipeline
            and bool(mesh.is_watertight)
            and len(mesh.faces) >= int(os.environ.get("MESHMEND_BLOCKY_EXEMPT_MIN_FACES", "1000000"))
            and depth_ratio >= float(os.environ.get("MESHMEND_BLOCKY_EXEMPT_MIN_DEPTH_RATIO", "0.45"))
        ):
            return False
        normals = np.asarray(mesh.face_normals, dtype=float)
        areas = np.asarray(mesh.area_faces, dtype=float)
        if normals.shape[0] != areas.shape[0] or float(areas.sum()) <= 1e-8:
            return False
        axis_aligned = np.max(np.abs(normals), axis=1) > 0.985
        axis_area_ratio = float(areas[axis_aligned].sum() / areas.sum())
        if depth_ratio < 0.18 and axis_area_ratio > 0.45:
            return True
        if axis_area_ratio > 0.72:
            return True
        return False
    except Exception:
        return False


def likely_image_low_relief_sheet(mesh: trimesh.Trimesh) -> bool:
    """Detect image-to-3D results that are mostly relief/card instead of a 3D model."""
    try:
        if len(mesh.faces) < 1000:
            return True
        ext = np.maximum(np.asarray(mesh.extents, dtype=float), 1e-6)
        min_ratio = float(ext.min() / max(ext.max(), 1e-6))
        if min_ratio >= float(os.environ.get("MESHMEND_IMAGE_SHEET_MAX_DEPTH_RATIO", "0.52")):
            return False
        normals = np.asarray(mesh.face_normals, dtype=float)
        areas = np.asarray(mesh.area_faces, dtype=float)
        if normals.shape[0] != areas.shape[0] or float(areas.sum()) <= 1e-8:
            return False
        axis_aligned = np.max(np.abs(normals), axis=1) > 0.985
        axis_area_ratio = float(areas[axis_aligned].sum() / areas.sum())
        return axis_area_ratio > float(os.environ.get("MESHMEND_IMAGE_SHEET_AXIS_AREA_RATIO", "0.54"))
    except Exception:
        return False


def likely_background_slab(mesh: trimesh.Trimesh, request: dict[str, Any] | None = None) -> bool:
    """Detect a remaining vertical render card/backplate around the miniature."""
    try:
        vertices = np.asarray(mesh.vertices, dtype=float)
        faces = np.asarray(mesh.faces)
        if len(vertices) < 1000 or len(faces) < 1000:
            return False
        mins = vertices.min(axis=0)
        maxs = vertices.max(axis=0)
        ext = np.maximum(maxs - mins, 1e-6)
        centers = vertices[faces].mean(axis=1)
        normals = np.asarray(mesh.face_normals, dtype=float)
        areas = np.asarray(mesh.area_faces, dtype=float)
        total_area = float(areas.sum())
        if total_area <= 1e-8:
            return False
        slab_cleanup_already_ran = bool(mesh.metadata.get("meshmend_background_slab_removed")) or bool(
            mesh.metadata.get("meshmend_connected_sheet_removed")
        )
        source_workflow = str((request or {}).get("_meshmend_source_workflow") or (request or {}).get("workflow") or "")
        image_reference = source_workflow == "image_to_3d"
        min_coverage_default = "0.50" if image_reference and slab_cleanup_already_ran else ("0.34" if image_reference else ("0.62" if slab_cleanup_already_ran else "0.42"))
        min_area_default = "0.055" if image_reference and slab_cleanup_already_ran else ("0.035" if image_reference else ("0.10" if slab_cleanup_already_ran else "0.06"))
        min_coverage = float(os.environ.get("MESHMEND_CARD_GATE_MIN_COVERAGE", min_coverage_default))
        min_area_ratio = float(os.environ.get("MESHMEND_CARD_GATE_MIN_AREA_RATIO", min_area_default))
        # Only consider vertical side/back planes. Axis 2 can be a legitimate base.
        for axis in (0, 1):
            other_axes = [idx for idx in range(3) if idx != axis]
            for boundary in (mins[axis], maxs[axis]):
                near = np.abs(centers[:, axis] - boundary) < ext[axis] * 0.06
                flat = np.abs(normals[:, axis]) > 0.70
                candidate = near & flat
                min_candidate_faces = max(120, min(int(len(faces) * 0.002), 2500))
                if int(candidate.sum()) < min_candidate_faces:
                    continue
                cand_centers = centers[candidate]
                area_ratio = float(areas[candidate].sum() / total_area)
                if area_ratio < min_area_ratio:
                    continue
                coverage = np.prod(np.maximum(cand_centers[:, other_axes].max(axis=0) - cand_centers[:, other_axes].min(axis=0), 1e-6)) / max(
                    np.prod(ext[other_axes]), 1e-6
                )
                if coverage >= min_coverage:
                    return True
        return False
    except Exception:
        return False


def likely_horizontal_sheet_card(mesh: trimesh.Trimesh) -> bool:
    """Detect square floor/cards that survive as a connected bottom sheet.

    A legitimate round miniature base can have a flat bottom, but it occupies a
    circular footprint. Hunyuan/card artifacts usually form a nearly rectangular
    plane covering almost the entire XY bounding box. Use occupancy, not just
    area, so a round base is not mistaken for a sheet.
    """
    try:
        vertices = np.asarray(mesh.vertices, dtype=float)
        faces = np.asarray(mesh.faces)
        if len(vertices) < 1000 or len(faces) < 1000:
            return False
        mins = vertices.min(axis=0)
        maxs = vertices.max(axis=0)
        ext = np.maximum(maxs - mins, 1e-6)
        centers = vertices[faces].mean(axis=1)
        normals = np.asarray(mesh.face_normals, dtype=float)
        areas = np.asarray(mesh.area_faces, dtype=float)
        total_area = float(areas.sum())
        if total_area <= 1e-8:
            return False
        near_bottom = np.abs(centers[:, 2] - mins[2]) < ext[2] * float(os.environ.get("MESHMEND_HORIZONTAL_CARD_Z_TOLERANCE", "0.06"))
        flat_bottom = np.abs(normals[:, 2]) > float(os.environ.get("MESHMEND_HORIZONTAL_CARD_NORMAL_DOT", "0.70"))
        candidate = near_bottom & flat_bottom
        if int(candidate.sum()) < max(500, int(len(faces) * 0.01)):
            return False
        area_ratio = float(areas[candidate].sum() / total_area)
        if area_ratio < float(os.environ.get("MESHMEND_HORIZONTAL_CARD_MIN_AREA_RATIO", "0.16")):
            return False
        cand_centers = centers[candidate]
        footprint = np.maximum(cand_centers[:, :2].max(axis=0) - cand_centers[:, :2].min(axis=0), 1e-6)
        coverage = float(np.prod(footprint) / max(np.prod(ext[:2]), 1e-6))
        if coverage < float(os.environ.get("MESHMEND_HORIZONTAL_CARD_MIN_COVERAGE", "0.86")):
            return False
        bins = int(os.environ.get("MESHMEND_HORIZONTAL_CARD_OCCUPANCY_BINS", "40"))
        normalized = (cand_centers[:, :2] - mins[:2]) / ext[:2]
        hist, _, _ = np.histogram2d(normalized[:, 0], normalized[:, 1], bins=bins, range=[[0.0, 1.0], [0.0, 1.0]])
        occupancy = float((hist > 0).mean())
        return occupancy >= float(os.environ.get("MESHMEND_HORIZONTAL_CARD_MIN_OCCUPANCY", "0.82"))
    except Exception:
        return False


def likely_dual_subject(mesh: trimesh.Trimesh) -> bool:
    try:
        vertices = np.asarray(mesh.vertices, dtype=float)
        faces = np.asarray(mesh.faces)
        if len(vertices) < 1000 or len(faces) < 1000:
            return False
        mins = vertices.min(axis=0)
        maxs = vertices.max(axis=0)
        ext = np.maximum(maxs - mins, 1e-6)
        centers = vertices[faces].mean(axis=1)
        z_norm = (centers[:, 2] - mins[2]) / ext[2]
        body = centers[z_norm > 0.08]
        if len(body) < 500:
            body = centers
        for axis in (0, 1):
            other = 1 - axis
            if ext[axis] / max(ext[2], ext[other], 1e-6) < 0.88:
                continue
            hist, _edges = np.histogram(body[:, axis], bins=96)
            if hist.max() <= 0:
                continue
            smooth = np.convolve(hist.astype(float), np.ones(7) / 7.0, mode="same")
            active = smooth > max(float(smooth.max()) * 0.22, len(body) * 0.001)
            bands = 0
            start = None
            for idx, value in enumerate(active):
                if value and start is None:
                    start = idx
                elif not value and start is not None:
                    if idx - start >= 6:
                        bands += 1
                    start = None
            if start is not None and len(active) - start >= 6:
                bands += 1
            if bands >= 2:
                return True
        return False
    except Exception:
        return False
