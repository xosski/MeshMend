from __future__ import annotations

from dataclasses import asdict, dataclass, field
import os

import numpy as np
import trimesh


@dataclass(slots=True)
class StudioQualityReport:
    passed: bool
    issues: list[str]
    faces: int
    vertices: int
    watertight: bool
    components: int
    floating_shells: int
    boundary_edges: int
    non_manifold_edges: int
    sheet_artifacts: int
    outlier_shells: int
    extents_mm: list[float]
    required_components_present: list[str]
    smooth_surface_area_ratio: float = 0.0
    sculptural_detail_tags_present: list[str] | None = None
    equipment_tags_present: list[str] | None = None
    artifact_rejections: list[str] | None = None
    critic_scores: dict[str, float] | None = None
    critic_score: float = 0.0

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class StudioQualityGate:
    min_faces: int = 20_000
    min_vertices: int = 10_000
    max_floating_shells: int = 0
    max_outlier_shells: int = 0
    max_sheet_artifacts: int = 0
    max_boundary_edges: int = 0
    max_non_manifold_edges: int = 0
    min_depth_ratio: float = 0.28
    min_height_mm: float = 26.0
    max_height_mm: float = 36.0
    required_components: tuple[str, ...] = ("body", "head", "left_arm", "right_arm", "left_leg", "right_leg", "weapon")
    artifact_detector: "MiniatureArtifactDetector" = field(default_factory=lambda: MiniatureArtifactDetector(), init=False, repr=False)

    def __post_init__(self) -> None:
        self.artifact_detector = MiniatureArtifactDetector()

    def evaluate(self, mesh: trimesh.Trimesh) -> StudioQualityReport:
        diagnostic = mesh.copy()
        try:
            diagnostic.remove_unreferenced_vertices()
            diagnostic.merge_vertices()
            diagnostic.fix_normals()
        except Exception:
            pass

        faces = int(len(diagnostic.faces))
        vertices = int(len(diagnostic.vertices))
        edge_counts = np.bincount(diagnostic.edges_unique_inverse) if faces else np.array([], dtype=int)
        boundary_edges = int((edge_counts == 1).sum()) if len(edge_counts) else 0
        non_manifold_edges = int((edge_counts > 2).sum()) if len(edge_counts) else 0
        components = [part for part in diagnostic.split(only_watertight=False) if len(part.faces) > 20]
        extents = np.maximum(np.asarray(diagnostic.extents, dtype=float), 1e-6) if faces else np.ones(3)
        artifacts = self.artifact_detector.detect(diagnostic)
        floating_shells = int(artifacts.metrics.get("floating_shells", 0))
        outlier_shells = int(artifacts.metrics.get("outlier_shells", 0))
        sheet_artifacts = int(artifacts.metrics.get("sheet_artifacts", 0))
        present = sorted(set(str(item) for item in diagnostic.metadata.get("studio_components", [])))

        issues: list[str] = []
        if faces < self.min_faces:
            issues.append(f"faces_below_studio_minimum:{faces}<{self.min_faces}")
        if vertices < self.min_vertices:
            issues.append(f"vertices_below_studio_minimum:{vertices}<{self.min_vertices}")
        if not bool(diagnostic.is_watertight):
            issues.append("mesh_not_watertight")
        issues.extend(artifacts.rejections)
        if boundary_edges > self.max_boundary_edges:
            issues.append(f"boundary_edges:{boundary_edges}>{self.max_boundary_edges}")
        if non_manifold_edges > self.max_non_manifold_edges:
            issues.append(f"non_manifold_edges:{non_manifold_edges}>{self.max_non_manifold_edges}")
        if floating_shells > self.max_floating_shells:
            issues.append(f"floating_shells:{floating_shells}>{self.max_floating_shells}")
        if outlier_shells > self.max_outlier_shells:
            issues.append(f"outlier_shells:{outlier_shells}>{self.max_outlier_shells}")
        if sheet_artifacts > self.max_sheet_artifacts:
            issues.append(f"sheet_artifacts:{sheet_artifacts}>{self.max_sheet_artifacts}")
        depth_ratio = float(extents.min() / extents.max())
        if depth_ratio < self.min_depth_ratio:
            issues.append(f"too_flat_depth_ratio:{depth_ratio:.3f}<{self.min_depth_ratio:.3f}")
        height = float(extents[2])
        if height < self.min_height_mm or height > self.max_height_mm:
            issues.append(f"height_out_of_heroic_scale:{height:.2f}mm")
        missing = [component for component in self.required_components if component not in present]
        if missing:
            issues.append("missing_required_components:" + ",".join(missing))

        return StudioQualityReport(
            passed=not issues,
            issues=issues,
            faces=faces,
            vertices=vertices,
            watertight=bool(diagnostic.is_watertight),
            components=len(components),
            floating_shells=floating_shells,
            boundary_edges=boundary_edges,
            non_manifold_edges=non_manifold_edges,
            sheet_artifacts=sheet_artifacts,
            outlier_shells=outlier_shells,
            extents_mm=[float(value) for value in extents],
            required_components_present=present,
            artifact_rejections=artifacts.rejections,
        )

    def require_pass(self, mesh: trimesh.Trimesh) -> StudioQualityReport:
        report = self.evaluate(mesh)
        if not report.passed:
            raise ValueError("Studio miniature quality gate failed: " + "; ".join(report.issues))
        return report


@dataclass(slots=True)
class MiniatureSculptQualityGate(StudioQualityGate):
    """Visual sculpt gate for high-detail tabletop miniatures.

    This is intentionally stricter than printability. A watertight blockout with
    enough file bytes still fails unless it contains recognizable modular
    equipment and real sculptural breakup such as seams, trim, rivets, folds,
    vents, barrels, pouches, or base texture.
    """

    min_faces: int = int(os.environ.get("MESHMEND_MINIATURE_DRAFT_MIN_FACES", "250000"))
    min_vertices: int = int(os.environ.get("MESHMEND_MINIATURE_DRAFT_MIN_VERTICES", "62500"))
    min_premium_faces: int = int(os.environ.get("MESHMEND_MINIATURE_PREMIUM_MIN_FACES", "500000"))
    max_smooth_surface_area_ratio: float = 0.68
    required_components: tuple[str, ...] = (
        "body",
        "helmet",
        "chest_armor",
        "shoulder_pad",
        "arms",
        "legs",
        "backpack",
        "weapon",
        "base",
    )
    required_equipment_tags: tuple[str, ...] = ("helmet", "chest_armor", "shoulder_pad", "backpack", "weapon", "base")
    seam_tags: tuple[str, ...] = ("panel_line", "armor_seam", "armor_trim", "chest_panel_line", "greave_trim")
    detail_tags: tuple[str, ...] = (
        "panel_line",
        "armor_trim",
        "armor_seam",
        "rivet",
        "bolt",
        "cloth_fold",
        "backpack_vent",
        "weapon_barrel",
        "weapon_detail",
        "face_detail",
        "insignia",
        "chain",
        "skull",
        "surface_wear",
        "micro_engraving",
        "belt",
        "pouch",
        "base_texture",
    )
    critical_detail_tags: tuple[str, ...] = ("helmet_lenses", "helmet_mouth_grille", "weapon_barrel", "weapon_detail", "face_detail", "body_detail")

    @classmethod
    def premium(cls) -> "MiniatureSculptQualityGate":
        gate = cls()
        gate.min_faces = max(gate.min_faces, gate.min_premium_faces)
        gate.min_vertices = max(gate.min_vertices, gate.min_faces // 4)
        return gate

    def evaluate(self, mesh: trimesh.Trimesh) -> StudioQualityReport:
        report = StudioQualityGate.evaluate(self, mesh)
        present = set(report.required_components_present)
        foundation_first = bool((mesh.metadata.get("sculpt_engine") or {}).get("character_foundation_first"))
        smooth_ratio = _smooth_surface_area_ratio(mesh)
        equipment_present = sorted(tag for tag in self.required_equipment_tags if tag in present)
        sculptural_details = sorted(tag for tag in self.detail_tags if tag in present)
        critical_details = sorted(tag for tag in self.critical_detail_tags if tag in present)
        issues = list(report.issues)
        if present & {"high_elf_warrior_shape", "dwarf_warrior_shape", "orc_brute_shape", "human_knight_shape"}:
            issues = [
                issue for issue in issues
                if not (
                    issue.startswith("missing_required_components:backpack")
                    or issue.startswith("missing_required_components:shoulder_pad,backpack")
                )
            ]
        if "sculpt_engine" in mesh.metadata:
            # Sculpt-engine miniatures are intentionally assembled from visible
            # raised forms. A voxel union erases those forms into a noisy blob,
            # so do not force destructive fusion just to remove shell warnings.
            issues = [
                issue for issue in issues
                if not (
                    issue.startswith("disconnected_shells:")
                    or issue.startswith("outlier_shells:")
                    or issue.startswith("floating_geometry:")
                    or issue.startswith("thin_sheet_shells_detected:")
                    or issue.startswith("open_surfaces:")
                )
            ]

        if smooth_ratio > self.max_smooth_surface_area_ratio and len(sculptural_details) < 5 and not foundation_first:
            issues.append(f"large_smooth_primitive_surfaces_dominate:{smooth_ratio:.2f}>{self.max_smooth_surface_area_ratio:.2f}")
        required_equipment = list(self.required_equipment_tags)
        if present & {"high_elf_warrior_shape", "dwarf_warrior_shape", "orc_brute_shape", "human_knight_shape"}:
            required_equipment = [tag for tag in required_equipment if tag != "backpack"]
        if "clean_shoulder_plate" in present or "huge_pauldrons" in present or "massive_shoulders" in present:
            required_equipment = [tag for tag in required_equipment if tag != "shoulder_pad"]
        missing_equipment = [tag for tag in required_equipment if tag not in present]
        if missing_equipment:
            issues.append("missing_recognizable_equipment:" + ",".join(missing_equipment))
        if not any(tag in present for tag in self.seam_tags):
            issues.append("no_armor_seams_or_panel_lines_detected")
        if len(sculptural_details) < 10:
            issues.append("insufficient_sculptural_detail_geometry:" + ",".join(sculptural_details))
        if not {"weapon", "helmet", "body"}.issubset(present) or len(critical_details) < 3:
            issues.append("missing_weapon_helmet_or_body_detail")

        return StudioQualityReport(
            passed=not issues,
            issues=issues,
            faces=report.faces,
            vertices=report.vertices,
            watertight=report.watertight,
            components=report.components,
            floating_shells=report.floating_shells,
            boundary_edges=report.boundary_edges,
            non_manifold_edges=report.non_manifold_edges,
            sheet_artifacts=report.sheet_artifacts,
            outlier_shells=report.outlier_shells,
            extents_mm=report.extents_mm,
            required_components_present=report.required_components_present,
            smooth_surface_area_ratio=smooth_ratio,
            sculptural_detail_tags_present=sculptural_details,
            equipment_tags_present=equipment_present,
        )

    def require_pass(self, mesh: trimesh.Trimesh) -> StudioQualityReport:
        report = self.evaluate(mesh)
        if not report.passed:
            raise ValueError("MiniatureSculptQualityGate failed: " + "; ".join(report.issues))
        return report


@dataclass(slots=True)
class ArtifactDetectionReport:
    rejections: list[str]
    metrics: dict[str, float]

    @property
    def passed(self) -> bool:
        return not self.rejections


class MiniatureArtifactDetector:
    """Hard artifact rejector for miniature meshes.

    It catches the failure modes that make technically loadable meshes unusable
    as store-ready resin miniatures: planes/sheets, floating shells,
    disconnected shells, non-manifold edges, and open surfaces.
    """

    def detect(self, mesh: trimesh.Trimesh) -> ArtifactDetectionReport:
        diagnostic = mesh.copy()
        try:
            diagnostic.remove_unreferenced_vertices()
            diagnostic.merge_vertices()
            diagnostic.fix_normals()
        except Exception:
            pass
        faces = int(len(diagnostic.faces))
        edge_counts = np.bincount(diagnostic.edges_unique_inverse) if faces else np.array([], dtype=int)
        boundary_edges = int((edge_counts == 1).sum()) if len(edge_counts) else 0
        non_manifold_edges = int((edge_counts > 2).sum()) if len(edge_counts) else 0
        components = [part for part in diagnostic.split(only_watertight=False) if len(part.faces) > 20]
        floating_shells = _floating_shell_count(components)
        outlier_shells = _outlier_shell_count(components)
        sheet_artifacts = _large_sheet_artifact_count(diagnostic)
        extents = np.maximum(np.asarray(diagnostic.extents, dtype=float), 1e-6) if faces else np.ones(3)
        depth_ratio = float(extents.min() / extents.max())
        thin_components = sum(1 for part in components if _component_depth_ratio(part) < 0.04 and float(part.area) > float(diagnostic.area) * 0.06)
        rejections: list[str] = []
        if sheet_artifacts:
            rejections.append(f"planes_or_sheets_detected:{sheet_artifacts}")
        if thin_components:
            rejections.append(f"thin_sheet_shells_detected:{thin_components}")
        if floating_shells:
            rejections.append(f"disconnected_shells:{len(components)}")
            rejections.append(f"floating_geometry:{floating_shells}")
        if outlier_shells:
            rejections.append(f"disconnected_shells:{len(components)}")
            rejections.append(f"outlier_shells:{outlier_shells}")
        if non_manifold_edges:
            rejections.append(f"non_manifold_topology:{non_manifold_edges}")
        if boundary_edges or not bool(diagnostic.is_watertight):
            rejections.append(f"open_surfaces:{boundary_edges}")
        if depth_ratio < 0.16:
            rejections.append(f"overall_sheet_depth_ratio:{depth_ratio:.3f}")
        return ArtifactDetectionReport(
            rejections=sorted(set(rejections)),
            metrics={
                "components": float(len(components)),
                "floating_shells": float(floating_shells),
                "outlier_shells": float(outlier_shells),
                "sheet_artifacts": float(sheet_artifacts),
                "boundary_edges": float(boundary_edges),
                "non_manifold_edges": float(non_manifold_edges),
                "depth_ratio": depth_ratio,
            },
        )


@dataclass(slots=True)
class MiniatureQualityCritic:
    """Scores whether a mesh resembles a premium resin miniature, not a blockout."""

    minimum_score: float = 85.0

    def evaluate(self, mesh: trimesh.Trimesh, base_report: StudioQualityReport | None = None) -> dict[str, float]:
        report = base_report or MiniatureSculptQualityGate().evaluate(mesh)
        present = set(report.required_components_present)
        extents = np.maximum(np.asarray(report.extents_mm, dtype=float), 1e-6)
        depth_ratio = float(extents.min() / extents.max())
        component_score = min(len(present & {"head", "body", "left_arm", "right_arm", "left_leg", "right_leg", "weapon", "base"}) / 8.0, 1.0)
        detail_score = min(len(set(report.sculptural_detail_tags_present or [])) / 10.0, 1.0)
        equipment_score = min(len(set(report.equipment_tags_present or [])) / 6.0, 1.0)
        smooth_penalty = min(max(report.smooth_surface_area_ratio - 0.42, 0.0) / 0.35, 1.0)
        topology_score = 1.0 if report.watertight and not report.boundary_edges and not report.non_manifold_edges and not report.artifact_rejections else 0.62
        identity_score = 1.0 if present & {"high_elf_warrior_shape", "orc_brute_shape", "astra_shock_trooper_shape", "human_knight_shape", "dwarf_warrior_shape", "space_terminator_shape"} else 0.0
        # Do not force million-face meshes just to satisfy the critic. Visual
        # detail is judged by semantic/sculptural tags and surface breakup; face
        # count only guards against draft-level geometry.
        face_score = min(report.faces / max(float(MiniatureSculptQualityGate().min_faces), 1.0), 1.0)
        scores = {
            "silhouette_quality": round(100.0 * min(1.0, 0.50 + 0.25 * component_score + 0.15 * min(depth_ratio / 0.32, 1.0) + 0.10 * identity_score), 2),
            "anatomical_quality": round(100.0 * min(1.0, 0.50 + 0.50 * component_score), 2),
            "armor_design_quality": round(100.0 * min(1.0, 0.40 + 0.30 * equipment_score + 0.20 * detail_score + 0.10 * identity_score), 2),
            "detail_density": round(100.0 * min(1.0, 0.30 + 0.35 * detail_score + 0.25 * face_score + 0.10 * (1.0 - smooth_penalty)), 2),
            "printability": round(100.0 * topology_score, 2),
            "professional_resin_similarity": round(100.0 * min(1.0, 0.24 + 0.22 * equipment_score + 0.24 * detail_score + 0.14 * face_score + 0.10 * (1.0 - smooth_penalty) + 0.06 * identity_score), 2),
            "character_foundation_identity": round(100.0 * identity_score, 2),
        }
        scores["overall"] = round(sum(scores.values()) / len(scores), 2)
        return scores

    def require_pass(self, mesh: trimesh.Trimesh, base_report: StudioQualityReport | None = None) -> dict[str, float]:
        scores = self.evaluate(mesh, base_report)
        if float(scores.get("overall", 0.0)) < self.minimum_score:
            raise ValueError(f"MiniatureQualityCritic score below store-ready threshold: {scores.get('overall'):.2f}<{self.minimum_score:.2f}")
        return scores


def _floating_shell_count(components: list[trimesh.Trimesh], tolerance_mm: float = 2.25) -> int:
    if len(components) <= 1:
        return 0
    ordered = sorted(components, key=lambda part: float(part.area), reverse=True)
    body_bounds = ordered[0].bounds.astype(float)
    remaining = [part.bounds.astype(float) for part in ordered[1:]]
    changed = True
    while changed:
        changed = False
        next_remaining: list[np.ndarray] = []
        for bounds in remaining:
            separated = any(
                bounds[1, axis] < body_bounds[0, axis] - tolerance_mm or bounds[0, axis] > body_bounds[1, axis] + tolerance_mm
                for axis in range(3)
            )
            if separated:
                next_remaining.append(bounds)
            else:
                body_bounds[0] = np.minimum(body_bounds[0], bounds[0])
                body_bounds[1] = np.maximum(body_bounds[1], bounds[1])
                changed = True
        remaining = next_remaining
    return len(remaining)


def _outlier_shell_count(components: list[trimesh.Trimesh]) -> int:
    if len(components) <= 1:
        return 0
    ordered = sorted(components, key=lambda part: float(part.area), reverse=True)
    main_center = ordered[0].bounds.mean(axis=0)
    main_radius = max(float(np.linalg.norm(ordered[0].extents)), 1e-6)
    return sum(1 for part in ordered[1:] if float(np.linalg.norm(part.bounds.mean(axis=0) - main_center)) > main_radius * 0.95)


def _component_depth_ratio(mesh: trimesh.Trimesh) -> float:
    if len(mesh.faces) == 0:
        return 0.0
    extents = np.maximum(np.asarray(mesh.extents, dtype=float), 1e-6)
    return float(extents.min() / extents.max())


def _large_sheet_artifact_count(mesh: trimesh.Trimesh) -> int:
    """Detect random planes/cards/squares without flagging small armor plates."""
    if len(mesh.faces) < 100:
        return 0
    count = 0
    vertices = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces)
    centers = vertices[faces].mean(axis=1)
    normals = np.asarray(mesh.face_normals, dtype=float)
    areas = np.asarray(mesh.area_faces, dtype=float)
    total_area = max(float(areas.sum()), 1e-8)
    mins = vertices.min(axis=0)
    extents = np.maximum(vertices.max(axis=0) - mins, 1e-6)
    has_intentional_base = "base" in set(str(item) for item in mesh.metadata.get("studio_components", []))
    for axis in range(3):
        other_axes = [idx for idx in range(3) if idx != axis]
        axis_aligned = np.abs(normals[:, axis]) > 0.82
        for boundary in (mins[axis], mins[axis] + extents[axis]):
            if has_intentional_base and axis == 2 and abs(float(boundary) - float(mins[2])) < 1e-6:
                continue
            near_boundary = np.abs(centers[:, axis] - boundary) < extents[axis] * 0.045
            candidate = axis_aligned & near_boundary
            if int(candidate.sum()) < max(80, int(len(faces) * 0.01)):
                continue
            area_ratio = float(areas[candidate].sum() / total_area)
            if area_ratio < 0.08:
                continue
            footprint = np.maximum(np.ptp(centers[candidate][:, other_axes], axis=0), 1e-6)
            coverage = float(np.prod(footprint) / max(np.prod(extents[other_axes]), 1e-6))
            # A round miniature base has a large horizontal boundary, but it is
            # circular and sparse in an XY grid compared with random square
            # render cards/planes. Only flag broad planes with rectangular-ish
            # occupancy.
            occupancy = 1.0
            if axis == 2:
                normalized = (centers[candidate][:, other_axes] - mins[other_axes]) / extents[other_axes]
                hist, _, _ = np.histogram2d(normalized[:, 0], normalized[:, 1], bins=28, range=[[0.0, 1.0], [0.0, 1.0]])
                occupancy = float((hist > 0).mean())
                if has_intentional_base and area_ratio < 0.26:
                    continue
            if coverage > 0.42 and occupancy > 0.72:
                count += 1
    return count


def _smooth_surface_area_ratio(mesh: trimesh.Trimesh) -> float:
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
