from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass
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
        }


def postprocess_miniature(mesh: Any, request: dict[str, Any]) -> tuple[trimesh.Trimesh, PostprocessReport]:
    """Turn raw Hunyuan output into one printable, scaled, detailed STL mesh."""
    mesh = coerce_to_trimesh(mesh)
    before_faces = len(mesh.faces)
    mesh, single_subject_enforced = enforce_single_subject(mesh)
    mesh = normalize_to_scale(mesh, requested_scale_mm(request))
    mesh = complete_front_shell_volume(mesh, request)
    mesh = thicken_if_sheet(mesh)
    mesh = normalize_to_scale(mesh, requested_scale_mm(request))
    detail_target = detail_face_target(request)
    mesh = subdivide_to_faces(mesh, detail_target)
    mesh = apply_miniature_sculpt_detail(mesh, request)
    mesh = cleanup(mesh)
    report = build_report(
        mesh,
        request,
        single_subject_enforced=single_subject_enforced or len(mesh.faces) < before_faces * 0.85,
        detail_faces_target=detail_target,
    )
    return mesh, report


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


def enforce_single_subject(mesh: trimesh.Trimesh) -> tuple[trimesh.Trimesh, bool]:
    """Keep one miniature even if generation produced a connected 4-model scene."""
    components = [component for component in mesh.split(only_watertight=False) if len(component.faces) > 100]
    if len(components) > 1:
        components.sort(key=lambda item: float(item.area), reverse=True)
        return components[0], True

    cropped = crop_spatial_subject(mesh)
    if cropped is not mesh:
        return cropped, True
    return mesh, False


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


def normalize_to_scale(mesh: trimesh.Trimesh, scale_mm: float) -> trimesh.Trimesh:
    vertices = np.asarray(mesh.vertices, dtype=float)
    extents = vertices.max(axis=0) - vertices.min(axis=0)
    max_extent = float(np.max(extents))
    if max_extent <= 1e-8:
        return mesh
    vertices *= scale_mm / max_extent
    mins = vertices.min(axis=0)
    maxs = vertices.max(axis=0)
    vertices[:, 0] -= (mins[0] + maxs[0]) * 0.5
    vertices[:, 1] -= (mins[1] + maxs[1]) * 0.5
    vertices[:, 2] -= mins[2]
    mesh.vertices = vertices
    mesh.metadata["units"] = "mm"
    mesh.metadata["meshmend_scale_mm"] = scale_mm
    return mesh


def complete_front_shell_volume(mesh: trimesh.Trimesh, request: dict[str, Any]) -> trimesh.Trimesh:
    """Synthesize a rear shell when single-image generation returns only the front half.

    Local image-to-3D models often reconstruct the visible face of a subject and
    leave the hidden side as a shallow relief. For miniatures that should be a
    full body, so when depth is suspiciously thin we mirror the visible shell
    into a rear volume before final scaling/detailing. This is intentionally
    conservative and only runs for thin character-like outputs.
    """
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
    target_faces = max(target_faces, len(mesh.faces))
    while len(mesh.faces) < target_faces:
        if len(mesh.faces) * 4 > target_faces * 1.8:
            break
        vertices, faces = trimesh.remesh.subdivide(mesh.vertices, mesh.faces)
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
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
        relief += (rivets & (torso | shoulders | legs)).astype(float) * 0.70
    else:
        folds = np.abs(np.sin((z * 15.0 + x * 4.0 + seed * 0.02) * math.pi)) < 0.022
        relief -= folds.astype(float) * 0.32

    if os.environ.get("MESHMEND_ENABLE_MICRO_NOISE", "0").strip().lower() in {"1", "true", "yes"}:
        relief += 0.08 * np.sin((x * 47.0 + y * 19.0 + z * 31.0 + seed) * math.pi)

    amplitude = float(os.environ.get("MESHMEND_DETAIL_RELIEF_MM", "0.08"))
    active = z > 0.06
    mesh.vertices = vertices + normals * (np.clip(relief, -1.0, 1.0) * amplitude * active.astype(float))[:, None]
    return cleanup(mesh)


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
    wants_8k = any(term in prompt for term in ("8k", "8 k", "studio", "production", "display quality", "maximum detail"))
    default = "1200000" if quality == "high" or wants_8k else "450000"
    return int(os.environ.get("MESHMEND_MIN_EXPORT_FACES", default))


def build_report(
    mesh: trimesh.Trimesh,
    request: dict[str, Any],
    *,
    single_subject_enforced: bool,
    detail_faces_target: int,
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
        relief_mm=float(os.environ.get("MESHMEND_DETAIL_RELIEF_MM", "0.08")),
        detail_style="postprocess_backend:single_subject+scale+sheet_guard+structured_sculpt",
    )
