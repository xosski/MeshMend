from __future__ import annotations

from dataclasses import dataclass

import hashlib

import numpy as np
import trimesh

from meshmend.app.detail_presets import DetailPreset


@dataclass(slots=True)
class DetailParameters:
    detail_strength: float = 0.35
    rivet_density: float = 0.35
    panel_line_depth: float = 0.08
    battle_damage_amount: float = 0.20
    surface_texture_strength: float = 0.04
    edge_sharpness: float = 0.50
    minimum_printable_detail_size: float = 0.05


@dataclass(slots=True)
class GeneratedDetail:
    mesh: trimesh.Trimesh
    rivets: int = 0
    panel_lines: int = 0
    vents: int = 0
    cracks: int = 0
    texture_vertices: int = 0


def apply_micro_displacement(
    mesh: trimesh.Trimesh,
    eligible_faces: np.ndarray,
    protected_vertices: np.ndarray,
    params: DetailParameters,
    preset: DetailPreset,
) -> tuple[trimesh.Trimesh, int]:
    """Add subtle procedural surface texture to low-detail, unprotected vertices."""
    textured = mesh.copy()
    if len(textured.vertices) == 0 or len(textured.faces) == 0:
        return textured, 0
    eligible_vertices = np.zeros(len(textured.vertices), dtype=bool)
    if len(eligible_faces):
        eligible_vertices[np.unique(np.asarray(textured.faces)[eligible_faces].reshape(-1))] = True
    eligible_vertices &= ~protected_vertices
    if not np.any(eligible_vertices):
        return textured, 0

    normals = np.asarray(textured.vertex_normals, dtype=float)
    vertices = np.asarray(textured.vertices, dtype=float).copy()
    amplitude = max(params.minimum_printable_detail_size, params.surface_texture_strength) * max(0.0, params.detail_strength)
    # Keep first implementation resin-safe and not blobby: displacement remains shallow.
    amplitude = min(amplitude, 0.12)
    seed = _stable_seed(preset.name)
    phase = (seed % 1024) / 1024.0
    coords = vertices * (0.37 + params.detail_strength)
    noise = (
        np.sin(coords[:, 0] * 3.1 + phase)
        + np.sin(coords[:, 1] * 4.7 + phase * 2.0)
        + np.sin(coords[:, 2] * 5.3 + phase * 3.0)
    ) / 3.0
    if preset.cloth_grain:
        noise += 0.35 * np.sin(coords[:, 0] * 14.0)
    elif preset.terrain_roughness:
        noise += 0.45 * np.sin((coords[:, 0] + coords[:, 1]) * 8.0)
    elif preset.organic_texture:
        noise += 0.35 * np.sin(np.linalg.norm(coords[:, :2], axis=1) * 10.0)
    vertices[eligible_vertices] += normals[eligible_vertices] * (noise[eligible_vertices, None] * amplitude)
    textured.vertices = vertices
    return textured, int(np.count_nonzero(eligible_vertices))


def generate_panel_lines(mesh: trimesh.Trimesh, face_indices: np.ndarray, params: DetailParameters, preset: DetailPreset) -> GeneratedDetail:
    if not preset.panel_lines or len(face_indices) == 0:
        return GeneratedDetail(mesh=trimesh.Trimesh(vertices=np.empty((0, 3)), faces=np.empty((0, 3), dtype=np.int64), process=False))
    details: list[trimesh.Trimesh] = []
    max_lines = max(1, int(96 * params.detail_strength))
    for face_index in face_indices[:max_lines]:
        tri = np.asarray(mesh.triangles[int(face_index)], dtype=float)
        normal = np.asarray(mesh.face_normals[int(face_index)], dtype=float)
        center = tri.mean(axis=0)
        edges = [(tri[1] - tri[0]), (tri[2] - tri[1]), (tri[0] - tri[2])]
        tangent = max(edges, key=lambda edge: float(np.linalg.norm(edge)))
        length = float(np.linalg.norm(tangent)) * 0.62
        if length <= params.minimum_printable_detail_size * 3:
            continue
        tangent = tangent / np.linalg.norm(tangent)
        width = max(params.minimum_printable_detail_size, params.panel_line_depth * 0.55)
        depth = max(params.minimum_printable_detail_size * 0.6, params.panel_line_depth)
        # Two tiny raised lips frame a shallow central recess; this reads as an
        # engraved panel line in resin without requiring fragile booleans.
        bitangent = np.cross(normal, tangent)
        bitangent /= max(float(np.linalg.norm(bitangent)), 1e-12)
        details.append(_oriented_box(center - bitangent * width, tangent, bitangent, normal, (length, width * 0.28, depth * 0.45)))
        details.append(_oriented_box(center + bitangent * width, tangent, bitangent, normal, (length, width * 0.28, depth * 0.45)))
    return _combine_details(details, panel_lines=len(details) // 2)


def generate_rivets(mesh: trimesh.Trimesh, face_indices: np.ndarray, params: DetailParameters, preset: DetailPreset) -> GeneratedDetail:
    if not preset.rivets or len(face_indices) == 0:
        return GeneratedDetail(mesh=trimesh.Trimesh(vertices=np.empty((0, 3)), faces=np.empty((0, 3), dtype=np.int64), process=False))
    details: list[trimesh.Trimesh] = []
    step = max(1, int(round(1.0 / max(params.rivet_density, 0.05))))
    max_rivets = int(240 * max(0.05, params.rivet_density) * max(0.25, params.detail_strength))
    for face_index in face_indices[::step][:max_rivets]:
        tri = np.asarray(mesh.triangles[int(face_index)], dtype=float)
        normal = np.asarray(mesh.face_normals[int(face_index)], dtype=float)
        center = tri.mean(axis=0)
        radius = max(params.minimum_printable_detail_size * 0.65, 0.045 + 0.07 * params.detail_strength)
        height = max(params.minimum_printable_detail_size * 0.55, radius * 0.55)
        rivet = trimesh.creation.uv_sphere(radius=radius, count=[12, 6])
        rivet.apply_scale([1.0, 1.0, 0.45])
        _align_local_z(rivet, normal)
        rivet.apply_translation(center + normal * (height + params.minimum_printable_detail_size * 0.3))
        details.append(rivet)
    return _combine_details(details, rivets=len(details))


def generate_battle_damage(mesh: trimesh.Trimesh, face_indices: np.ndarray, params: DetailParameters, preset: DetailPreset) -> GeneratedDetail:
    if not preset.battle_damage or params.battle_damage_amount <= 0 or len(face_indices) == 0:
        return GeneratedDetail(mesh=trimesh.Trimesh(vertices=np.empty((0, 3)), faces=np.empty((0, 3), dtype=np.int64), process=False))
    details: list[trimesh.Trimesh] = []
    max_cracks = int(64 * params.battle_damage_amount * max(0.25, params.detail_strength))
    for offset, face_index in enumerate(face_indices[:max_cracks]):
        tri = np.asarray(mesh.triangles[int(face_index)], dtype=float)
        normal = np.asarray(mesh.face_normals[int(face_index)], dtype=float)
        center = tri.mean(axis=0)
        tangent = tri[(offset + 1) % 3] - tri[offset % 3]
        length_norm = float(np.linalg.norm(tangent))
        if length_norm <= 1e-9:
            continue
        tangent /= length_norm
        bitangent = np.cross(normal, tangent)
        bitangent /= max(float(np.linalg.norm(bitangent)), 1e-12)
        length = max(params.minimum_printable_detail_size * 3.0, length_norm * 0.35)
        width = max(params.minimum_printable_detail_size * 0.45, 0.035 + params.battle_damage_amount * 0.05)
        crack = _oriented_box(center, tangent, bitangent, normal, (length, width, max(width, params.panel_line_depth * 0.45)))
        details.append(crack)
    return _combine_details(details, cracks=len(details))


def generate_mechanical_vents(mesh: trimesh.Trimesh, face_indices: np.ndarray, params: DetailParameters, preset: DetailPreset) -> GeneratedDetail:
    if not preset.mechanical_grooves or len(face_indices) == 0:
        return GeneratedDetail(mesh=trimesh.Trimesh(vertices=np.empty((0, 3)), faces=np.empty((0, 3), dtype=np.int64), process=False))
    details: list[trimesh.Trimesh] = []
    for face_index in face_indices[: max(1, int(36 * params.detail_strength))]:
        tri = np.asarray(mesh.triangles[int(face_index)], dtype=float)
        normal = np.asarray(mesh.face_normals[int(face_index)], dtype=float)
        center = tri.mean(axis=0)
        tangent = tri[1] - tri[0]
        if np.linalg.norm(tangent) <= 1e-9:
            continue
        tangent /= np.linalg.norm(tangent)
        bitangent = np.cross(normal, tangent)
        bitangent /= max(float(np.linalg.norm(bitangent)), 1e-12)
        for groove in range(3):
            details.append(
                _oriented_box(
                    center + bitangent * ((groove - 1) * params.minimum_printable_detail_size * 2.0),
                    tangent,
                    bitangent,
                    normal,
                    (params.minimum_printable_detail_size * 5.0, params.minimum_printable_detail_size * 0.8, params.minimum_printable_detail_size),
                )
            )
    return _combine_details(details, vents=len(details))


def _combine_details(details: list[trimesh.Trimesh], **counts: int) -> GeneratedDetail:
    if not details:
        mesh = trimesh.Trimesh(vertices=np.empty((0, 3)), faces=np.empty((0, 3), dtype=np.int64), process=False)
    else:
        mesh = trimesh.util.concatenate(details)
        mesh.remove_unreferenced_vertices()
    return GeneratedDetail(mesh=mesh, **counts)


def _oriented_box(center: np.ndarray, tangent: np.ndarray, bitangent: np.ndarray, normal: np.ndarray, extents: tuple[float, float, float]) -> trimesh.Trimesh:
    box = trimesh.creation.box(extents=extents)
    transform = np.eye(4)
    transform[:3, 0] = tangent / max(float(np.linalg.norm(tangent)), 1e-12)
    transform[:3, 1] = bitangent / max(float(np.linalg.norm(bitangent)), 1e-12)
    transform[:3, 2] = normal / max(float(np.linalg.norm(normal)), 1e-12)
    transform[:3, 3] = center + transform[:3, 2] * (extents[2] * 0.52)
    box.apply_transform(transform)
    return box


def _align_local_z(mesh: trimesh.Trimesh, normal: np.ndarray) -> None:
    normal = normal / max(float(np.linalg.norm(normal)), 1e-12)
    z_axis = np.array([0.0, 0.0, 1.0])
    axis = np.cross(z_axis, normal)
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm <= 1e-12:
        return
    axis /= axis_norm
    angle = float(np.arccos(np.clip(np.dot(z_axis, normal), -1.0, 1.0)))
    mesh.apply_transform(trimesh.transformations.rotation_matrix(angle, axis))


def _stable_seed(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()[:8], 16)
