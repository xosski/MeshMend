from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh


@dataclass(frozen=True, slots=True)
class StampPlacement:
    kind: str
    center: tuple[float, float, float]
    normal: tuple[float, float, float] = (0.0, 0.0, 1.0)
    scale: float = 1.0
    rotation_degrees: float = 0.0


def stamp_details(base: trimesh.Trimesh, placements: list[StampPlacement]) -> trimesh.Trimesh:
    """Add miniature detail stamps to a model.

    The first implementation uses additive geometry stamps. This is deliberately
    more robust than destructive booleans for STL/scan inputs while still giving
    visible rivets, vents, skulls, seals, plates, cracks, bolts, and cables.
    """
    stamps = [_make_stamp(placement) for placement in placements]
    stamps = [stamp for stamp in stamps if len(stamp.faces) > 0]
    if not stamps:
        return base.copy()
    combined = trimesh.util.concatenate([base.copy(), *stamps])
    combined.remove_unreferenced_vertices()
    combined.metadata["meshmend_boolean_detail_stamps"] = [placement.kind for placement in placements]
    return combined


def _make_stamp(placement: StampPlacement) -> trimesh.Trimesh:
    kind = placement.kind.lower().strip()
    scale = max(float(placement.scale), 1e-6)
    if kind in {"rivet", "stud", "bolt"}:
        mesh = trimesh.creation.uv_sphere(radius=0.09 * scale, count=[14, 8])
        mesh.apply_scale([1.0, 1.0, 0.45])
    elif kind == "vent":
        mesh = _vent_stamp(scale)
    elif kind == "skull":
        mesh = _skull_stamp(scale)
    elif kind in {"seal", "purity seal"}:
        mesh = _seal_stamp(scale)
    elif kind == "plate":
        mesh = trimesh.creation.box(extents=(1.0 * scale, 0.08 * scale, 0.55 * scale))
    elif kind == "crack":
        mesh = _crack_stamp(scale)
    elif kind == "cable":
        mesh = _cable_stamp(scale)
    else:
        mesh = trimesh.creation.uv_sphere(radius=0.08 * scale, count=[10, 6])
        mesh.apply_scale([1.0, 1.0, 0.35])
    _orient_and_place(mesh, placement)
    return mesh


def _vent_stamp(scale: float) -> trimesh.Trimesh:
    parts = []
    for index in range(4):
        slat = trimesh.creation.box(extents=(0.62 * scale, 0.035 * scale, 0.07 * scale))
        slat.apply_translation((0.0, (index - 1.5) * 0.12 * scale, 0.0))
        parts.append(slat)
    frame = trimesh.creation.box(extents=(0.78 * scale, 0.52 * scale, 0.035 * scale))
    frame.apply_translation((0.0, 0.0, -0.035 * scale))
    parts.append(frame)
    return trimesh.util.concatenate(parts)


def _skull_stamp(scale: float) -> trimesh.Trimesh:
    skull = trimesh.creation.uv_sphere(radius=0.20 * scale, count=[16, 10])
    skull.apply_scale([0.88, 0.65, 1.0])
    jaw = trimesh.creation.box(extents=(0.24 * scale, 0.12 * scale, 0.16 * scale))
    jaw.apply_translation((0.0, -0.02 * scale, -0.22 * scale))
    eyes = []
    for x in (-0.07, 0.07):
        eye = trimesh.creation.uv_sphere(radius=0.045 * scale, count=[8, 4])
        eye.apply_translation((x * scale, -0.13 * scale, 0.03 * scale))
        eyes.append(eye)
    return trimesh.util.concatenate([skull, jaw, *eyes])


def _seal_stamp(scale: float) -> trimesh.Trimesh:
    wax = trimesh.creation.cylinder(radius=0.16 * scale, height=0.055 * scale, sections=18)
    ribbon_a = trimesh.creation.box(extents=(0.08 * scale, 0.04 * scale, 0.55 * scale))
    ribbon_b = ribbon_a.copy()
    ribbon_a.apply_translation((-0.055 * scale, 0.0, -0.32 * scale))
    ribbon_b.apply_translation((0.055 * scale, 0.0, -0.32 * scale))
    return trimesh.util.concatenate([wax, ribbon_a, ribbon_b])


def _crack_stamp(scale: float) -> trimesh.Trimesh:
    parts = []
    for index, (length, angle) in enumerate(((0.55, 0.0), (0.32, 35.0), (0.26, -42.0))):
        shard = trimesh.creation.box(extents=(length * scale, 0.025 * scale, 0.05 * scale))
        shard.apply_transform(trimesh.transformations.rotation_matrix(np.deg2rad(angle), [0, 0, 1]))
        shard.apply_translation(((index - 1) * 0.10 * scale, 0.0, 0.0))
        parts.append(shard)
    return trimesh.util.concatenate(parts)


def _cable_stamp(scale: float) -> trimesh.Trimesh:
    points = []
    for i in range(16):
        t = (i / 15.0) - 0.5
        points.append((t * 1.0 * scale, np.sin(t * np.pi * 2.0) * 0.08 * scale, 0.0))
    return trimesh.creation.capsule(radius=0.035 * scale, height=1.0 * scale, count=[8, 8]).apply_translation((0, 0, 0)) or _polyline_tubes(points, 0.035 * scale)


def _polyline_tubes(points: list[tuple[float, float, float]], radius: float) -> trimesh.Trimesh:
    segments = []
    for a, b in zip(points, points[1:]):
        segments.append(trimesh.creation.cylinder(radius=radius, segment=np.array([a, b], dtype=float), sections=8))
    return trimesh.util.concatenate(segments) if segments else trimesh.Trimesh()


def _orient_and_place(mesh: trimesh.Trimesh, placement: StampPlacement) -> None:
    normal = np.asarray(placement.normal, dtype=float)
    normal /= max(float(np.linalg.norm(normal)), 1e-12)
    z_axis = np.array([0.0, 0.0, 1.0])
    axis = np.cross(z_axis, normal)
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm > 1e-12:
        axis /= axis_norm
        angle = float(np.arccos(np.clip(np.dot(z_axis, normal), -1.0, 1.0)))
        mesh.apply_transform(trimesh.transformations.rotation_matrix(angle, axis))
    if placement.rotation_degrees:
        mesh.apply_transform(trimesh.transformations.rotation_matrix(np.deg2rad(placement.rotation_degrees), normal))
    mesh.apply_translation(np.asarray(placement.center, dtype=float) + normal * (0.035 * placement.scale))
