from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tempfile
from typing import Protocol

import numpy as np
import trimesh

from meshmend.compliance.filters import sanitize_prompt
from meshmend.core.mesh_ops import add_circular_base, auto_scale_to_height, remesh_subdivide


@dataclass(slots=True)
class GenerationRequest:
    prompt: str
    height_mm: float = 32.0
    target_faces: int = 180_000
    add_base: bool = True
    output_path: Path | None = None


class LocalGenerationAdapter(Protocol):
    name: str

    def generate(self, request: GenerationRequest) -> trimesh.Trimesh:
        """Generate a mesh using only local code/models."""


class PlaceholderMiniatureAdapter:
    """Free local procedural adapter used until open-weight models are installed.

    This is intentionally labeled placeholder: it proves the local adapter and UI
    path without faking a cloud generation API or claiming studio-quality AI.
    """

    name = "placeholder_procedural_local"

    def generate(self, request: GenerationRequest) -> trimesh.Trimesh:
        clean_prompt, _warnings = sanitize_prompt(request.prompt)
        lower = clean_prompt.lower()
        parts: list[trimesh.Trimesh] = []

        body = trimesh.creation.capsule(radius=3.2, height=13.5, count=[24, 24])
        body.apply_translation([0, 0, 12.0])
        parts.append(body)

        head = trimesh.creation.uv_sphere(radius=2.2, count=[24, 12])
        head.apply_translation([0, 0, 21.5])
        parts.append(head)

        for x in (-4.0, 4.0):
            arm = trimesh.creation.capsule(radius=0.9, height=9.5, count=[16, 12])
            arm.apply_transform(trimesh.transformations.rotation_matrix(np.radians(20 if x < 0 else -20), [0, 1, 0]))
            arm.apply_translation([x, 0, 13.0])
            parts.append(arm)
        for x in (-1.4, 1.4):
            leg = trimesh.creation.capsule(radius=1.05, height=10.0, count=[16, 12])
            leg.apply_translation([x, 0, 5.0])
            parts.append(leg)

        if any(term in lower for term in ("armor", "armour", "knight", "marine", "soldier")):
            chest = trimesh.creation.box(extents=[6.8, 2.4, 6.0])
            chest.apply_translation([0, -0.2, 14.5])
            parts.append(chest)
            for x in (-4.2, 4.2):
                shoulder = trimesh.creation.uv_sphere(radius=2.0, count=[16, 8])
                shoulder.apply_scale([1.25, 0.9, 0.75])
                shoulder.apply_translation([x, 0, 17.5])
                parts.append(shoulder)

        if any(term in lower for term in ("sword", "blade", "weapon")):
            blade = trimesh.creation.box(extents=[0.45, 0.2, 11.0])
            blade.apply_translation([5.5, -0.1, 13.0])
            hilt = trimesh.creation.box(extents=[2.2, 0.4, 0.5])
            hilt.apply_translation([5.5, -0.1, 8.0])
            parts.extend([blade, hilt])
        elif any(term in lower for term in ("rifle", "gun", "blaster")):
            gun = trimesh.creation.box(extents=[7.0, 0.8, 1.0])
            gun.apply_translation([3.5, -2.4, 14.0])
            parts.append(gun)

        mesh = trimesh.util.concatenate(parts)
        mesh = auto_scale_to_height(mesh, request.height_mm)
        mesh = remesh_subdivide(mesh, request.target_faces)
        if request.add_base:
            mesh = add_circular_base(mesh)
        mesh.metadata["meshmend_generator"] = self.name
        mesh.metadata["meshmend_prompt"] = clean_prompt
        return mesh


class ExistingMeshMendLocalSculptorAdapter:
    """Adapter over the repository's existing local native miniature generator.

    This uses `3dsculpter/model_service/native_generation.py`, which was already
    present in MeshMend and builds MeshMend-owned printable miniature geometry.
    It avoids the slower service/Hunyuan path for the desktop MVP smoke while
    still reusing the project's existing generation code instead of a new toy
    generator.
    """

    name = "existing_meshmend_local_sculptor"

    def generate(self, request: GenerationRequest) -> trimesh.Trimesh:
        clean_prompt, _warnings = sanitize_prompt(request.prompt)
        from meshmend.studio import MiniatureSculptQualityGate, StagedMiniaturePipeline, StudioMiniatureSpec

        studio_spec = StudioMiniatureSpec.from_prompt(clean_prompt, scale_mm=request.height_mm, target_faces=int(request.target_faces))
        with tempfile.TemporaryDirectory(prefix="meshmend_native_gen_") as temp_dir:
            output_path = Path(temp_dir) / "meshmend_staged_archetype.stl"
            _output, assembly = StagedMiniaturePipeline(quality_gate=MiniatureSculptQualityGate()).export(studio_spec, output_path, candidates_per_category=1)
            mesh = assembly.mesh
        components = set(str(item) for item in mesh.metadata.get("studio_components", []))
        if request.add_base and "base" not in components and not bool(mesh.metadata.get("meshmend_added_round_base")):
            mesh = add_circular_base(mesh)
        mesh.metadata["meshmend_generator"] = self.name
        mesh.metadata["meshmend_native_report"] = {
            "provider": "meshmend_staged_archetype_pipeline",
            "subject_type": assembly.quality_report.required_components_present,
            "stage_results": [stage.to_dict() for stage in assembly.stage_results],
            "quality_report": assembly.quality_report.to_dict(),
        }
        mesh.metadata["meshmend_prompt"] = clean_prompt
        return mesh


def get_adapter(name: str = "existing") -> LocalGenerationAdapter:
    if name in {"existing", "existing_local", "existing_meshmend_local_sculptor"}:
        return ExistingMeshMendLocalSculptorAdapter()
    if name in {"placeholder", "placeholder_procedural_local"}:
        return PlaceholderMiniatureAdapter()
    raise ValueError(f"Unknown local generation adapter: {name}")
