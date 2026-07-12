from __future__ import annotations

from dataclasses import asdict, dataclass, field
from contextlib import contextmanager
import importlib.util
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
import trimesh

from .repair import RepairOptions, RepairReport, _component_count, repair_stl


@dataclass(slots=True)
class AssistantPlan:
    """A deterministic repair plan chosen by the MeshMend assistant."""

    bridge_disconnected: bool
    connector_radius: float
    connector_sections: int
    max_bridge_distance: float | None
    merge_digits: int
    max_hole_edges: int
    max_existing_vertex_displacement: float
    components_detected: int
    boundary_edges_detected: int
    reasons: list[str] = field(default_factory=list)

    def to_repair_options(self) -> RepairOptions:
        return RepairOptions(
            bridge_disconnected=self.bridge_disconnected,
            connector_radius=self.connector_radius,
            connector_sections=self.connector_sections,
            max_bridge_distance=self.max_bridge_distance,
            merge_digits=self.merge_digits,
            max_hole_edges=self.max_hole_edges,
            max_existing_vertex_displacement=self.max_existing_vertex_displacement,
        )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class AssistantResult:
    plan: AssistantPlan
    report: RepairReport
    explanation: str
    perseus: dict[str, object] | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "plan": self.plan.to_dict(),
            "report": self.report.to_dict(),
            "explanation": self.explanation,
            "perseus": self.perseus,
        }


class MeshMendAssistant:
    """AI-style assistant for repairing detached STL islands and print issues.

    Mesh edits remain deterministic and inspectable. The assistant layer does the
    AI-style work: inspect the model, choose safe repair settings, explain what
    changed, and optionally pass the explanation through the local Perseus memory
    orchestrator when those integration files are present.
    """

    def __init__(self, enable_perseus: bool = False, allow_ollama_fallback: bool = False):
        self.enable_perseus = bool(enable_perseus)
        self.allow_ollama_fallback = bool(allow_ollama_fallback)
        self._perseus_orchestrator: Any | None = None

    def build_plan(
        self,
        input_path: str | Path,
        *,
        connector_radius: float | None = None,
        connector_sections: int = 16,
        max_bridge_distance: float | None = None,
        merge_digits: int = 6,
        max_hole_edges: int = 80,
        max_existing_vertex_displacement: float = 0.005,
        force_bridge: bool = False,
    ) -> AssistantPlan:
        mesh = _load_mesh_for_planning(Path(input_path))
        component_count = _component_count(mesh)
        boundary_edges = _boundary_edge_count(mesh)
        radius = connector_radius if connector_radius is not None else _suggest_connector_radius(mesh)

        reasons: list[str] = []
        if component_count > 1 and force_bridge:
            reasons.append(
                f"Detected {component_count} disconnected mesh components; detached islands will be anchored to the main body."
            )
        elif component_count > 1:
            reasons.append(
                f"Detected {component_count} disconnected mesh components; museum-scan mode will not add generic connector geometry unless bridging is explicitly requested."
            )
        elif force_bridge:
            reasons.append("Detached-piece repair was requested, but only one component was detected.")
        else:
            reasons.append("Only one connected component was detected, so no connector bridge is needed.")

        if boundary_edges:
            reasons.append(f"Detected {boundary_edges} boundary edges; eligible small holes will be capped.")
        if connector_radius is None:
            reasons.append(f"Selected connector radius {radius:.4g} from the model size.")
        reasons.append(
            "Museum-scan preservation enabled: no smoothing, remeshing, decimation, subdivision, inflation, shrink, or global topology optimization."
        )
        reasons.append(f"Existing sculpt vertices must remain within {max_existing_vertex_displacement:g} model units.")

        return AssistantPlan(
            bridge_disconnected=force_bridge,
            connector_radius=radius,
            connector_sections=max(6, int(connector_sections)),
            max_bridge_distance=max_bridge_distance,
            merge_digits=merge_digits,
            max_hole_edges=max_hole_edges,
            max_existing_vertex_displacement=max_existing_vertex_displacement,
            components_detected=component_count,
            boundary_edges_detected=boundary_edges,
            reasons=reasons,
        )

    def repair(
        self,
        input_path: str | Path,
        output_path: str | Path,
        *,
        connector_radius: float | None = None,
        connector_sections: int = 16,
        max_bridge_distance: float | None = None,
        merge_digits: int = 6,
        max_hole_edges: int = 80,
        max_existing_vertex_displacement: float = 0.005,
        force_bridge: bool = False,
    ) -> AssistantResult:
        plan = self.build_plan(
            input_path,
            connector_radius=connector_radius,
            connector_sections=connector_sections,
            max_bridge_distance=max_bridge_distance,
            merge_digits=merge_digits,
            max_hole_edges=max_hole_edges,
            max_existing_vertex_displacement=max_existing_vertex_displacement,
            force_bridge=force_bridge,
        )
        report = repair_stl(input_path, output_path, plan.to_repair_options())
        explanation = self.explain(plan, report)
        perseus_result = self._run_perseus(explanation) if self.enable_perseus else None
        if perseus_result and isinstance(perseus_result.get("response"), str):
            explanation = str(perseus_result["response"]).strip() or explanation
        return AssistantResult(plan=plan, report=report, explanation=explanation, perseus=perseus_result)

    @staticmethod
    def explain(plan: AssistantPlan, report: RepairReport) -> str:
        lines = [
            "MeshMend Assistant repair complete.",
            f"Components: {report.components_before} -> {report.components_after}.",
            f"Boundary edges: {report.boundary_edges_before} -> {report.boundary_edges_after}.",
            f"Holes capped: {report.holes_capped}; anchored bridges added: {report.bridges_added}.",
            f"Watertight: {report.watertight_before} -> {report.watertight_after}.",
            f"Max existing vertex displacement: {report.max_existing_vertex_displacement:.6g}.",
            f"Detail preservation: {report.detail_preservation}.",
        ]
        if plan.reasons:
            lines.append("Plan: " + " ".join(plan.reasons))
        return "\n".join(lines)

    def _run_perseus(self, prompt: str) -> dict[str, object] | None:
        orchestrator_class = _load_perseus_orchestrator_class()
        if orchestrator_class is None:
            return {
                "response": prompt,
                "provider": "meshmend-native",
                "warning": "Perseus memory orchestrator was not available; used MeshMend native explanation.",
            }

        if self._perseus_orchestrator is None:
            with _perseus_working_directory():
                self._perseus_orchestrator = orchestrator_class(
                    local_answer_fn=lambda user_prompt, hidden_context: prompt,
                    allow_ollama_fallback=self.allow_ollama_fallback,
                )
        with _perseus_working_directory():
            return self._perseus_orchestrator.handle(prompt)

    def learning_status(self) -> dict[str, object]:
        """Exercise the Perseus learning loop enough to verify it is wired."""

        orchestrator_class = _load_perseus_orchestrator_class()
        if orchestrator_class is None:
            return {"available": False, "status": "Perseus memory orchestrator could not be loaded"}

        try:
            with _perseus_working_directory():
                orchestrator = orchestrator_class(
                    local_answer_fn=lambda user_prompt, hidden_context: "MeshMend learning diagnostic response.",
                    allow_ollama_fallback=False,
                )
                result = orchestrator.handle("MeshMend learning diagnostic: remember successful STL repair patterns.")
            return {
                "available": True,
                "provider": result.get("provider"),
                "quality": result.get("quality"),
                "training_decision": result.get("training_decision"),
                "brain_state": result.get("brain_state"),
            }
        except Exception as exc:
            return {"available": False, "status": f"Learning diagnostic failed: {exc}"}


def _load_mesh_for_planning(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(path, force="mesh", process=False)
    if isinstance(loaded, trimesh.Scene):
        meshes = [geometry for geometry in loaded.geometry.values() if isinstance(geometry, trimesh.Trimesh)]
        if not meshes:
            raise ValueError(f"No mesh geometry found in {path}")
        loaded = trimesh.util.concatenate(meshes)
    if not isinstance(loaded, trimesh.Trimesh):
        raise ValueError(f"Unsupported mesh input: {path}")
    if len(loaded.vertices) == 0 or len(loaded.faces) == 0:
        raise ValueError(f"Mesh has no vertices or faces: {path}")
    return loaded


def _boundary_edge_count(mesh: trimesh.Trimesh) -> int:
    if len(mesh.faces) == 0:
        return 0
    counts = np.bincount(mesh.edges_unique_inverse)
    return int(np.count_nonzero(counts == 1))


def _suggest_connector_radius(mesh: trimesh.Trimesh) -> float:
    extents = np.asarray(mesh.bounding_box.extents, dtype=float)
    diagonal = float(np.linalg.norm(extents))
    if diagonal <= 1e-9:
        return 0.75
    return max(0.25, min(2.0, diagonal * 0.0075))


def _load_perseus_orchestrator_class():
    integration_path = Path(__file__).resolve().parent / "Perseus Integration" / "Perseus_Memory_Orchestrator.py"
    if not integration_path.exists():
        return None

    module_key = "meshmend_ai_perseus_memory_orchestrator"
    existing = sys.modules.get(module_key)
    if existing is not None and hasattr(existing, "PerseusMemoryOrchestrator"):
        return getattr(existing, "PerseusMemoryOrchestrator")

    spec = importlib.util.spec_from_file_location(module_key, integration_path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_key] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_key, None)
        return None
    return getattr(module, "PerseusMemoryOrchestrator", None)


@contextmanager
def _perseus_working_directory():
    memory_dir = Path(__file__).resolve().parent / "learning_memory"
    memory_dir.mkdir(parents=True, exist_ok=True)
    previous = Path.cwd()
    os.chdir(memory_dir)
    try:
        yield memory_dir
    finally:
        os.chdir(previous)
