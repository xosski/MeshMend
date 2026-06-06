from __future__ import annotations

import base64
from dataclasses import dataclass
import hashlib
import importlib
import json
import mimetypes
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Callable
import urllib.error
import urllib.parse
import urllib.request

import numpy as np
import trimesh

from .detail_quality import TARGET_HIGH_RESOLUTION_PITCH_MM, ensure_high_resolution_detail
from .generative_model import Local3DGenerativeModel, default_training_data_dir
from .high_resolution_latent import LocalMeshLatentGenerator
from .neural_diffusion import Neural3DDiffusionModel


ProgressCallback = Callable[[int, str], None]
MAX_CREATION_EXEMPLAR_FACES = int(os.environ.get("MESHMEND_CREATION_MAX_EXEMPLAR_FACES", "120000"))
TRAINED_EXEMPLAR_DETAIL_FACES = int(os.environ.get("MESHMEND_TRAINED_EXEMPLAR_DETAIL_FACES", "1200000"))
MIN_CREATION_NEURAL_RESOLUTION = int(os.environ.get("MESHMEND_MIN_CREATION_NEURAL_RESOLUTION", "64"))
USE_NEURAL_CREATION = os.environ.get("MESHMEND_USE_NEURAL_CREATION") == "1"
USE_MESH_LATENT_CREATION = os.environ.get("MESHMEND_USE_MESH_LATENT_CREATION", "1").strip().lower() not in {"0", "false", "no"}
USE_LOCAL_EXEMPLAR_CREATION = os.environ.get("MESHMEND_USE_LOCAL_EXEMPLAR_CREATION", "1").strip().lower() not in {"0", "false", "no"}
USE_CHARACTER_KITBASH_OVERLAYS = os.environ.get("MESHMEND_USE_CHARACTER_KITBASH_OVERLAYS", "1").strip().lower() in {"1", "true", "yes"}
ALLOW_FULL_TRAINING_EXEMPLAR_OUTPUT = os.environ.get("MESHMEND_ALLOW_FULL_TRAINING_EXEMPLAR_OUTPUT") == "1"
USE_HOSTED_CREATION_BACKEND = os.environ.get("MESHMEND_USE_HOSTED_3D", "1").strip().lower() in {"1", "true", "yes"}
ALLOW_LEGACY_CREATION_FALLBACK = os.environ.get("MESHMEND_ALLOW_LEGACY_CREATION_FALLBACK", "0").strip().lower() in {"1", "true", "yes"}
HOSTED_CREATION_URL = os.environ.get("MESHMEND_MODEL_SERVICE_URL", "http://127.0.0.1:8090").strip().rstrip("/")
HOSTED_CREATION_TIMEOUT_SECONDS = float(os.environ.get("MESHMEND_HOSTED_TIMEOUT_SECONDS", "5400"))
HOSTED_CREATION_POLL_SECONDS = float(os.environ.get("MESHMEND_HOSTED_POLL_INTERVAL_SECONDS", "5"))
MODEL_SERVICE_STARTUP_SECONDS = float(os.environ.get("MESHMEND_MODEL_SERVICE_STARTUP_SECONDS", "20"))
EIGHT_K_DETAIL_TRIANGLE_CAP = int(os.environ.get("MESHMEND_8K_DETAIL_TRIANGLES", "8000000"))
EIGHT_K_DETAIL_PITCH_UM = float(os.environ.get("MESHMEND_8K_DETAIL_UM", "50"))
CREATION_GEOMETRY_VERSION = "clean-trained-orc-asset-v33"
_MODEL_SERVICE_PROCESS: subprocess.Popen | None = None


def _friendly_production_backend_error(exc: Exception) -> str:
    """Avoid labeling expected quality-gate stops as backend crashes."""
    message = str(exc).strip()
    lowered = message.lower()
    concept_gate_terms = (
        "text-to-3d stopped before hunyuan",
        "text concept generation could not produce",
        "prepared hunyuan reference is not a complete",
        "local concept image was not a complete usable",
    )
    if any(term in lowered for term in concept_gate_terms):
        summary = message.split(" Details:", 1)[0]
        if "after " in message and " candidate" in message:
            summary = message.split(". Details:", 1)[0]
        return (
            "Concept generation could not produce a clean full-body miniature reference, so MeshMend stopped before 3D generation "
            "instead of exporting another noisy or partial model. Try a more specific prompt or provide a clean full-body image reference. "
            f"Details: {summary}"
        )
    quality_gate_terms = (
        "generated stl did not meet store-quality gate",
        "likely_background_slab_or_card",
        "mesh_not_solid_watertight",
        "mesh_nonmanifold_edges",
        "image_visual_holes_unsealed",
        "multiple_components_",
    )
    if any(term in lowered for term in quality_gate_terms):
        return (
            "The generated mesh failed MeshMend's store-quality checks, so it was not exported as a final model. "
            f"Details: {message}"
        )
    native_sculpt_gate_terms = (
        "native image-conditioned sculpt failed store-quality gates",
        "no_concept_image_received",
        "low_vision_planner_confidence",
        "ai_planner_required_but_not_configured",
        "ai_planner_required_but_failed",
        "image_to_3d_requires_ai_vision_planner_for_concept_fidelity",
        "concept_match_insufficient_local_landmarks",
        "concept_match_too_few_required_landmarks",
        "concept_match_insufficient_native_detail_layers",
        "planned_landmarks_missing",
    )
    if any(term in lowered for term in native_sculpt_gate_terms):
        if "image_to_3d_requires_ai_vision_planner_for_concept_fidelity" in lowered:
            return (
                "MeshMend stopped because high-detail image-to-3D needs a real AI/vision sculpt planner to match the concept image. "
                "The fallback local image heuristic can only read broad silhouette cues, so it would likely export another generic model. "
                "Configure a vision planner with --sculpt-planner-command and --require-ai-sculpt-planner, or lower quality to standard for an experimental draft. "
                f"Details: {message}"
            )
        if "concept_match_too_few_required_landmarks" in lowered or "concept_match_insufficient_local_landmarks" in lowered:
            return (
                "MeshMend could not verify enough subject-defining landmarks from the concept image. "
                "Add a short prompt naming the required visible features, or configure the AI/vision sculpt planner for image-only store-quality jobs. "
                f"Details: {message}"
            )
        if "planned_landmarks_missing" in lowered:
            return (
                "MeshMend stopped because the generated sculpt did not realize all required concept landmarks. "
                "This prevents exporting another generic/noisy model when the concept asks for specific parts. "
                f"Details: {message}"
            )
        return (
            "MeshMend stopped before exporting because the native store-quality gate could not verify concept fidelity. "
            "For store-quality image-to-STL, attach a concept image and configure an AI/vision sculpt planner; otherwise lower quality to standard for an experimental draft. "
            f"Details: {message}"
        )
    store_quality_unavailable_terms = (
        "store/studio-quality 8k miniature generation is not available",
        "not certified for store/studio-quality 8k miniature sculpt generation",
        "not a certified high-detail miniature sculpt generator",
    )
    if any(term in lowered for term in store_quality_unavailable_terms):
        return (
            "Store-quality 8K miniature generation is not configured. MeshMend's bundled local generators are draft/procedural or experimental, "
            "so MeshMend stopped instead of exporting another generic/noisy model. Configure a certified production runner, or lower quality to standard/draft. "
            f"Details: {message}"
        )
    return (
        "MeshMend's production 3D backend is required for studio-quality miniature creation, "
        "but it did not return a model. Start the model service and configure a real production runner, "
        "or set MESHMEND_ALLOW_LEGACY_CREATION_FALLBACK=1 to explicitly use the old procedural draft path. "
        f"Backend error: {message}"
    )


@dataclass(frozen=True, slots=True)
class TrainingOverlayCandidate:
    mesh_path: str
    caption: str
    tags: list[str]
    faces: int
    quality_warnings: list[str]


@dataclass(frozen=True, slots=True)
class SculptorFoundation:
    """Integration point for the bundled 3D Sculptor model-creation stack."""

    root: Path

    @property
    def outputs_dir(self) -> Path:
        return self.root / "backend" / "outputs"

    @property
    def main_script(self) -> Path:
        return self.root / "main.py"

    @property
    def available(self) -> bool:
        return self.main_script.exists()

    def latest_model(self) -> Path | None:
        if not self.outputs_dir.exists():
            return None
        candidates = [
            path
            for path in self.outputs_dir.iterdir()
            if path.is_file() and path.suffix.lower() in {".stl", ".obj", ".ply"}
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda path: path.stat().st_mtime)

    def launch(self) -> subprocess.Popen:
        if not self.available:
            raise FileNotFoundError(f"3D Sculptor entry point not found: {self.main_script}")
        return subprocess.Popen([sys.executable, str(self.main_script)], cwd=str(self.root))

    def module_status(self) -> dict[str, bool | str]:
        """Return whether key 3D Sculptor and learning modules can be imported."""

        status: dict[str, bool | str] = {
            "3dsculpter_root": str(self.root),
            "3dsculpter_available": self.available,
        }
        modules = [
            "app.mesh_utils",
            "app.mesh_analyzer",
            "backend.utils.stl_processor",
            "backend.processing.prompt_enhancer",
            "backend.processing.scale_validator",
            "backend.processing.printability",
            "backend.pipelines.image_to_mesh",
        ]
        with self._python_path():
            for module_name in modules:
                try:
                    importlib.import_module(module_name)
                    status[module_name] = True
                except Exception as exc:
                    status[module_name] = f"unavailable: {exc}"
        return status

    def create_model(
        self,
        prompt: str,
        image_path: str | Path | None = None,
        progress: ProgressCallback | None = None,
        scale_mm: float | None = None,
        print_detail_um: float | None = None,
        max_detail_triangles: int | None = None,
    ) -> Path:
        """Create a 3D model from chat text and/or an image and save it as STL.

        The bundled 3D Sculptor project is treated as the foundation. When its
        image-to-mesh stack and optional dependencies are importable, image input
        uses that stack. Otherwise MeshMend falls back to a deterministic local
        primitive/relief generator so the chat workflow still produces a model.
        """

        prompt = (prompt or "").strip()
        if not prompt and image_path is None:
            raise ValueError("Enter a text prompt or choose an image before creating a model.")
        scale_mm = _resolve_miniature_scale_mm(prompt, scale_mm)

        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        progress = progress or (lambda percent, message: None)
        progress(5, f"Reading creation request ({scale_mm:g}mm scale)")

        try:
            hosted_output = self._create_model_with_hosted_backend(
                prompt=prompt,
                image_path=Path(image_path) if image_path is not None else None,
                scale_mm=scale_mm,
                print_detail_um=print_detail_um,
                max_detail_triangles=max_detail_triangles,
                progress=progress,
            )
            if hosted_output is not None:
                return hosted_output
        except Exception as exc:
            if not ALLOW_LEGACY_CREATION_FALLBACK:
                backend_message = _friendly_production_backend_error(exc)
                raise RuntimeError(
                    backend_message
                ) from exc
            progress(12, f"Production backend unavailable; explicit legacy fallback enabled: {exc}")

        mesh: trimesh.Trimesh
        source = "text"
        capability_tier = "procedural_draft"
        if image_path is not None:
            source = "image"
            capability_tier = "image_guided_local_mesh_latent" if USE_MESH_LATENT_CREATION else "image_guided_procedural_draft"
            progress(18, "Loading image")
            mesh = self._model_from_image(Path(image_path), progress, prompt, scale_mm)
            if prompt and os.environ.get("MESHMEND_USE_SCULPTOR_IMAGE_PIPELINE") == "1":
                mesh = self._apply_prompt_shape_hints(mesh, prompt)
        else:
            progress(25, "Checking local high-resolution mesh latent sculptor")
            mesh = self._model_from_mesh_latent(prompt, scale_mm)
            if mesh is not None:
                progress(42, "Generated from local high-resolution mesh/latent sculptor")
                capability_tier = "local_mesh_latent"
            elif USE_NEURAL_CREATION and not self._requires_structured_character_builder(prompt):
                progress(25, "Checking neural 3D diffusion model")
                neural_model = Neural3DDiffusionModel.load_latest()
                if neural_model is not None:
                    neural_config = neural_model.checkpoint_config()
                    if neural_config.resolution < MIN_CREATION_NEURAL_RESOLUTION:
                        progress(
                            35,
                            f"Neural checkpoint is {neural_config.resolution}^3; skipping because creation needs at least {MIN_CREATION_NEURAL_RESOLUTION}^3",
                        )
                        mesh = self._model_from_trained_or_procedural(prompt, scale_mm)
                    else:
                        try:
                            generated = neural_model.generate(prompt)
                        except Exception:
                            generated = None
                        if (
                            generated is not None
                            and not self._is_block_or_noise_mesh(generated)
                            and not self._is_under_detailed_character_mesh(generated, prompt)
                        ):
                            progress(42, "Generated from neural latent 3D diffusion model")
                            mesh = generated
                            capability_tier = "neural_experimental"
                        else:
                            progress(35, "Neural output was blocky/incomplete; checking structured miniature builder")
                            mesh = self._model_from_trained_or_procedural(prompt, scale_mm)
                else:
                    progress(35, "No neural checkpoint found; sculpting structured tabletop miniature")
                    mesh = self._model_from_trained_or_procedural(prompt, scale_mm)
            else:
                progress(35, "Sculpting structured tabletop miniature from prompt")
                mesh = self._model_from_trained_or_procedural(prompt, scale_mm)

        progress(72, f"Normalizing to {scale_mm:g}mm printable scale")
        mesh = self._normalize_for_printing(mesh, scale_mm, prompt)
        if _prompt_generation_intent(prompt) == "character_miniature":
            if _prompt_has_separate_weapon(prompt):
                progress(76, "Adding weapon-bearing character surface detail without blocky voxel fuse")
                mesh = self._restore_prompt_surface_detail(mesh, prompt)
            else:
                progress(76, "Fusing character shells into a cohesive printable sculpt")
                mesh = self._cohesive_character_remesh(mesh, scale_mm)
                mesh = self._restore_prompt_surface_detail(mesh, prompt)
            capability_tier = "structured_cohesive_sculpt"
        high_definition_requested = _prompt_requests_8k_detail(prompt)
        target_pitch_mm = _target_print_pitch_mm(print_detail_um, high_definition_requested=high_definition_requested)
        progress(82, f"Running mesh-density and miniature quality checks ({target_pitch_mm * 1000:.0f} um target)")
        detail_kwargs = {"target_pitch_mm": target_pitch_mm}
        if max_detail_triangles is not None:
            detail_kwargs["max_faces"] = max(EIGHT_K_DETAIL_TRIANGLE_CAP, max_detail_triangles) if high_definition_requested else max_detail_triangles
        elif high_definition_requested:
            detail_kwargs["max_faces"] = EIGHT_K_DETAIL_TRIANGLE_CAP
        mesh, detail_report = ensure_high_resolution_detail(mesh, **detail_kwargs)
        if image_path is not None:
            progress(84, "Restoring uploaded-image edge and contrast detail after final remesh")
            mesh = self._apply_image_reference_relief(mesh, Path(image_path), prompt)
        mesh.merge_vertices()
        mesh.remove_unreferenced_vertices()
        try:
            mesh.fill_holes()
            mesh.fix_normals()
        except Exception:
            pass
        progress(86, detail_report.summary())
        quality_review = self._review_creation_quality(mesh, prompt, capability_tier, detail_report)
        if quality_review.get("warnings"):
            progress(88, "Quality warnings: " + "; ".join(quality_review["warnings"][:2]))
        if not quality_review.get("passed", False):
            raise RuntimeError(
                "Generated STL did not meet MeshMend's miniature quality gate: "
                + "; ".join(quality_review.get("issues", []))
            )
        image_fingerprint = self._image_content_fingerprint(Path(image_path)) if image_path is not None else ""
        digest = hashlib.sha256(
            f"{CREATION_GEOMETRY_VERSION}|{prompt}|{image_path}|{image_fingerprint}|{scale_mm:g}mm".encode("utf-8", errors="ignore")
        ).hexdigest()[:10]
        output_path = self.outputs_dir / f"meshmend_created_{source}_{digest}.stl"
        progress(90, "Saving created STL")
        mesh.export(output_path)
        progress(100, f"Created draft STL: {output_path.name}")
        return output_path

    def _create_model_with_hosted_backend(
        self,
        *,
        prompt: str,
        image_path: Path | None,
        scale_mm: float,
        print_detail_um: float | None,
        max_detail_triangles: int | None,
        progress: ProgressCallback,
    ) -> Path | None:
        """Delegate all creation requests to MeshMend's production model service.

        The old in-process sculptor is a procedural draft generator; it cannot
        produce store-quality miniatures. This method is intentionally tried
        before any legacy path and, by default, failures are surfaced instead of
        silently returning another blocky draft.
        """
        if not USE_HOSTED_CREATION_BACKEND or not HOSTED_CREATION_URL:
            return None
        if os.environ.get("MESHMEND_DISABLE_HOSTED_CREATION", "0").strip().lower() in {"1", "true", "yes"}:
            return None

        quality = _hosted_quality_for_detail(print_detail_um)
        requested_polycount = max_detail_triangles or (250_000 if quality == "high" else 150_000)
        service_cap = int(os.environ.get("MESHMEND_HOSTED_TARGET_POLYCOUNT_CAP", "4000000" if quality == "high" else "750000"))
        target_polycount = min(int(requested_polycount), service_cap)
        enriched_prompt = (
            f"{prompt}\n\n"
            f"Create a production/studio-quality {scale_mm:g}mm resin-printable tabletop miniature. "
            "This must be final sculpt geometry, not a primitive blockout: crisp facial features, fingers/claws, "
            "weapon bevels, armor trim, layered cloth/leather/metal material texture, engraved panel lines, "
            "deep recesses readable at miniature scale, watertight STL/GLB/OBJ output, and no generic block/rock/cheese forms."
        ).strip()
        payload: dict[str, object] = {
            "prompt": enriched_prompt,
            "quality": quality,
            "target_formats": ["stl", "glb", "obj"],
            "target_polycount": int(target_polycount),
            "scale_mm": float(scale_mm),
            "product": "meshmend",
        }
        endpoint = "/v1/text-to-3d"
        if image_path is not None:
            payload["image_data_uri"] = _image_data_uri(image_path)
            payload["workflow"] = "image_to_3d"
            endpoint = "/v1/image-to-3d"
        else:
            payload["workflow"] = "text_to_3d"

        progress(8, "Checking MeshMend production 3D backend")
        _ensure_model_service_ready(progress)
        health = _model_service_health()
        if health is not None and not _creation_backend_ready(health):
            _restart_experimental_sculpt_service(progress)
            health = _model_service_health()
        if health is not None and not _creation_backend_ready(health):
            workflow_name = "image" if image_path is not None else "text"
            reason = str(health.get("store_quality_reason") or "").strip()
            raise RuntimeError(
                f"The MeshMend model service is running, but no production {workflow_name}-to-3D runner is configured. "
                "Start it with --native-sculpt-backend for experimental native sculpt generation, or configure a certified external backend."
                + (f" Details: {reason}" if reason else "")
            )

        progress(10, "Submitting to MeshMend production 3D backend")
        task = _model_service_request_json("POST", endpoint, payload)
        task_id = str(task.get("task_id") or task.get("id") or "")
        if _task_has_model(task):
            completed = task
        elif task_id:
            completed = self._poll_hosted_creation_task(task_id, progress)
        else:
            raise RuntimeError(f"model service did not return a task id or model: {task}")

        progress(88, "Downloading production model")
        output_path = self._download_hosted_creation_model(completed, task_id or "completed")
        progress(100, f"Created production model: {output_path.name}")
        return output_path

    @staticmethod
    def _poll_hosted_creation_task(task_id: str, progress: ProgressCallback) -> dict:
        deadline = time.time() + HOSTED_CREATION_TIMEOUT_SECONDS
        last_task: dict = {}
        while time.time() < deadline:
            task = _model_service_request_json("GET", f"/v1/tasks/{urllib.parse.quote(task_id)}")
            last_task = task
            status = str(task.get("status", "")).upper()
            task_progress = int(task.get("progress") or 0)
            stage_message = str(task.get("message") or task.get("stage") or status or "RUNNING")
            progress(max(12, min(86, task_progress)), f"Production backend: {stage_message}")
            if _task_has_model(task):
                return task
            if status in {"FAILED", "ERROR", "CANCELED", "CANCELLED"}:
                raise RuntimeError(str(task.get("error") or task))
            time.sleep(HOSTED_CREATION_POLL_SECONDS)
        raise RuntimeError(f"model service timed out after {HOSTED_CREATION_TIMEOUT_SECONDS:g}s: {last_task}")

    def _download_hosted_creation_model(self, task: dict, task_id: str) -> Path:
        model_urls = task.get("model_urls") or {}
        if task.get("model_url") and not model_urls:
            model_urls = {str(task.get("model_format") or "stl").lower(): task["model_url"]}
        selected_format = ""
        selected_url = ""
        for fmt in ("stl", "glb", "obj", "ply", "3mf", "fbx", "usdz"):
            if model_urls.get(fmt):
                selected_format = fmt
                selected_url = str(model_urls[fmt])
                break
        if not selected_url:
            raise RuntimeError(f"model service returned no downloadable model URL: {task}")
        safe_task = re.sub(r"[^a-zA-Z0-9_-]+", "", task_id)[:16] or "completed"
        output_path = self.outputs_dir / f"meshmend_production_{safe_task}.{selected_format}"
        _download_model_url(selected_url, output_path)
        return output_path

    @staticmethod
    def _image_content_fingerprint(image_path: Path) -> str:
        try:
            hasher = hashlib.sha256()
            with image_path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    hasher.update(chunk)
            return hasher.hexdigest()[:16]
        except Exception:
            return ""

    def _review_creation_quality(
        self,
        mesh: trimesh.Trimesh,
        prompt: str,
        capability_tier: str,
        detail_report,
    ) -> dict:
        """Run the same final quality gate for chat-created STLs before export."""
        with self._python_path():
            try:
                from app.stl_quality_reviewer import STLQualityReviewer
            except Exception:
                issues = [] if detail_report.passed else list(detail_report.issues)
                return {
                    "passed": detail_report.passed,
                    "score": 70 if detail_report.passed else 0,
                    "issues": issues,
                    "warnings": ["STL quality reviewer unavailable; only mesh-density check ran"],
                    "capability_tier": capability_tier,
                    "production_ready": False,
                }

            review = STLQualityReviewer().review(mesh, prompt, capability_tier=capability_tier)
        if not detail_report.passed:
            review.setdefault("warnings", []).extend(detail_report.issues)
            production_requested = bool(review.get("production_requested"))
            if production_requested:
                review.setdefault("issues", []).extend(detail_report.issues)
                review["passed"] = False
        return review

    def _model_from_trained_or_procedural(self, prompt: str, scale_mm: float) -> trimesh.Trimesh:
        if _prompt_generation_intent(prompt) == "character_miniature":
            return self._model_from_prompt(prompt, scale_mm)
        if USE_LOCAL_EXEMPLAR_CREATION and not self._requires_structured_character_builder(prompt):
            trained_model = Local3DGenerativeModel.load_latest()
            if trained_model is not None:
                generated = trained_model.generate(prompt, max_faces=MAX_CREATION_EXEMPLAR_FACES)
                if (
                    generated is not None
                    and not self._is_block_or_noise_mesh(generated)
                    and not self._is_under_detailed_character_mesh(generated, prompt)
                ):
                    return generated
        return self._model_from_prompt(prompt, scale_mm)

    def _model_from_mesh_latent(self, prompt: str, scale_mm: float) -> trimesh.Trimesh | None:
        if not USE_MESH_LATENT_CREATION:
            return None
        intent = _prompt_generation_intent(prompt)
        if intent in {"character_miniature", "bust"}:
            # The current mesh-latent path is an unrigged part assembler. It is
            # useful for standalone props/objects, but full characters need a
            # socket/armature system before learned limbs/heads/gear can be
            # attached coherently. Until then, use the structured sculptor for
            # full miniatures instead of exporting disconnected Frankenstein kits.
            return None
        try:
            mesh_latent = LocalMeshLatentGenerator.load_latest()
        except Exception:
            return None
        if mesh_latent is None:
            return None
        try:
            generated = mesh_latent.generate(
                prompt,
                scale_mm=scale_mm,
                max_asset_faces=min(TRAINED_EXEMPLAR_DETAIL_FACES, 220_000),
            )
        except Exception:
            return None
        if (
            generated is not None
            and not self._is_under_detailed_character_mesh(generated, prompt)
        ):
            return generated
        return None

    def _model_from_image(
        self,
        image_path: Path,
        progress: ProgressCallback,
        fallback_prompt: str = "",
        scale_mm: float = 32.0,
    ) -> trimesh.Trimesh:
        image_guided_prompt = self._image_guided_prompt(image_path, fallback_prompt, scale_mm)
        if image_guided_prompt.strip():
            progress(32, "Analyzing image as miniature design reference")
            mesh = self._model_from_mesh_latent(image_guided_prompt, scale_mm)
            if mesh is None:
                mesh = self._model_from_trained_or_procedural(image_guided_prompt, scale_mm)
            progress(58, "Projecting image edge/detail relief onto sculpt")
            return self._apply_image_reference_relief(mesh, image_path, image_guided_prompt)

        if os.environ.get("MESHMEND_USE_SCULPTOR_IMAGE_PIPELINE") != "1":
            if fallback_prompt.strip():
                progress(32, "Image reconstruction is disabled; sculpting structured miniature from prompt")
                return self._model_from_prompt(fallback_prompt, scale_mm)
            progress(32, "Using rounded image-to-sculpt silhouette builder")
            return self._fallback_image_relief(image_path, progress)

        with self._python_path():
            try:
                from PIL import Image
                from backend.pipelines.image_to_mesh import reconstruct_mesh_from_image_volumetric
            except Exception:
                if fallback_prompt.strip():
                    progress(38, "Image pipeline unavailable; sculpting structured miniature from prompt")
                    return self._model_from_prompt(fallback_prompt, scale_mm)
                return self._fallback_image_relief(image_path, progress)

            try:
                image = Image.open(image_path).convert("RGB")
                progress(38, "Reconstructing image through 3D Sculptor pipeline")
                mesh = reconstruct_mesh_from_image_volumetric(image, vol_size=96, voxel_size_mm=0.25)
                if not self._is_block_or_noise_mesh(mesh):
                    return self._apply_image_reference_relief(mesh, image_path, fallback_prompt)
                if fallback_prompt.strip():
                    progress(48, "Image reconstruction looked blobby; sculpting structured miniature from prompt")
                    mesh = self._model_from_prompt(fallback_prompt, scale_mm)
                    return self._apply_image_reference_relief(mesh, image_path, fallback_prompt)
                progress(48, "3D Sculptor output looked blocky/noisy; rebuilding as rounded silhouette sculpt")
                return self._fallback_image_relief(image_path, progress)
            except Exception:
                if fallback_prompt.strip():
                    progress(48, "Image reconstruction failed; sculpting structured miniature from prompt")
                    mesh = self._model_from_prompt(fallback_prompt, scale_mm)
                    return self._apply_image_reference_relief(mesh, image_path, fallback_prompt)
                return self._fallback_image_relief(image_path, progress)

    @staticmethod
    def _apply_image_reference_relief(mesh: trimesh.Trimesh, image_path: Path, prompt: str = "") -> trimesh.Trimesh:
        """Project actual image edges/contrast into shallow front-surface geometry.

        The text planner can choose mask/rifle/character topology, but it cannot
        recover small details from the uploaded pixels. This pass uses the image
        itself as a non-copying sculpt reference: detected edges become engraved
        grooves and strong light/dark transitions become subtle raised/recessed
        surface variation on the visible/front side of the generated form.
        """
        try:
            from PIL import Image
        except Exception:
            return mesh

        try:
            image = Image.open(image_path).convert("RGBA")
            image.thumbnail((192, 192))
            rgba = np.asarray(image, dtype=np.float32) / 255.0
            if rgba.ndim != 3 or rgba.shape[2] < 4:
                return mesh

            alpha = rgba[:, :, 3]
            rgb = rgba[:, :, :3]
            gray = rgb @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
            foreground = alpha > 0.18
            if foreground.mean() < 0.02:
                foreground = np.ones(gray.shape, dtype=bool)
            rows, cols = np.where(foreground)
            if rows.size and cols.size:
                pad = 3
                r0 = max(int(rows.min()) - pad, 0)
                r1 = min(int(rows.max()) + pad + 1, gray.shape[0])
                c0 = max(int(cols.min()) - pad, 0)
                c1 = min(int(cols.max()) + pad + 1, gray.shape[1])
                gray = gray[r0:r1, c0:c1]
                alpha = alpha[r0:r1, c0:c1]

            gx = np.abs(np.diff(gray, axis=1, prepend=gray[:, :1]))
            gy = np.abs(np.diff(gray, axis=0, prepend=gray[:1, :]))
            edges = gx + gy
            edge_max = float(np.percentile(edges, 96)) if edges.size else 0.0
            if edge_max > 1e-6:
                edges = np.clip(edges / edge_max, 0.0, 1.0)
            contrast = gray - float(np.mean(gray))
            contrast_scale = float(np.percentile(np.abs(contrast), 92)) if contrast.size else 0.0
            if contrast_scale > 1e-6:
                contrast = np.clip(contrast / contrast_scale, -1.0, 1.0)

            detailed = mesh.copy()
            vertices = np.asarray(detailed.vertices, dtype=np.float64)
            bounds = np.asarray(detailed.bounds, dtype=np.float64)
            extents = np.maximum(bounds[1] - bounds[0], 1e-6)
            if len(vertices) < 1000 or np.max(extents) <= 1e-6:
                return mesh

            # Most generated sculpts face negative Y. Only affect the visible
            # front shell so the image detail reads like sculpted/engraved trim,
            # not random all-over noise.
            front_depth = max(0.45, extents[1] * 0.38)
            front_mask = vertices[:, 1] <= bounds[0, 1] + front_depth
            # Avoid warping bases/floors; standalone masks still start at z=0.
            base_cut = bounds[0, 2] + extents[2] * 0.06
            front_mask &= vertices[:, 2] >= base_cut

            u = (vertices[:, 0] - bounds[0, 0]) / extents[0]
            v = 1.0 - ((vertices[:, 2] - bounds[0, 2]) / extents[2])
            h, w = gray.shape[:2]
            ix = np.clip((u * (w - 1)).astype(int), 0, w - 1)
            iy = np.clip((v * (h - 1)).astype(int), 0, h - 1)
            sampled_edges = edges[iy, ix]
            sampled_contrast = contrast[iy, ix]

            high_detail = _prompt_requests_8k_detail(prompt) or any(
                token in (prompt or "").lower()
                for token in ("detailed", "high detail", "highly detailed", "intricate", "ornate", "reference image")
            )
            groove_depth = 0.18 if high_detail else 0.12
            raised_depth = 0.07 if high_detail else 0.045
            relief = (-groove_depth * sampled_edges) + (raised_depth * sampled_contrast)
            relief = np.where(front_mask, np.clip(relief, -groove_depth, raised_depth), 0.0)

            # Push primarily along the front/back axis. Vertex normals can be
            # noisy on blockout meshes; Y displacement keeps grooves readable.
            vertices[:, 1] += relief
            detailed.vertices = vertices
            try:
                detailed.remove_degenerate_faces()
                detailed.remove_unreferenced_vertices()
                detailed.fix_normals()
            except Exception:
                pass
            return detailed
        except Exception:
            return mesh

    def _image_guided_prompt(self, image_path: Path, prompt: str, scale_mm: float) -> str:
        """Convert a user reference image into sculpt/planner cues.

        This is deliberately a design-analysis bridge, not exact copying. It
        extracts broad visual traits that the structured miniature planner can
        act on: silhouette, pose, contrast, color/material hints, and whether the
        image looks like a humanoid/creature/vehicle reference.
        """
        cues = self._local_image_reference_cues(image_path)
        base_prompt = (prompt or "").strip()
        if not cues:
            return base_prompt

        if not base_prompt:
            base_prompt = "original resin-printable 3D model inspired by the uploaded reference image"
        filename_cues = " ".join(re.split(r"[_\-.\s]+", image_path.stem)).strip()
        if filename_cues:
            base_prompt = f"{base_prompt}. Reference image filename subject cues: {filename_cues}"

        return (
            f"{base_prompt}. Use the uploaded image as a primary non-copying design reference. "
            f"Image-derived sculpt cues: {', '.join(cues)}. "
            f"Create an original {scale_mm:g}mm resin-printable 3D model with readable silhouette, preserving the reference subject topology; "
            "only make it a full-body wargaming miniature when the prompt/image clearly describes a character or creature. "
            "Use layered armor/clothing, detailed head/face area, accessories, weapon/tool if implied, openings, rims, vents, "
            "panel seams, straps, pouches, cables, rivets, and painter-friendly raised details appropriate to the actual subject. "
            "Do not copy logos, protected insignia, named characters, or exact trade dress."
        )

    @staticmethod
    def _local_image_reference_cues(image_path: Path) -> list[str]:
        try:
            from PIL import Image
        except Exception:
            return []

        try:
            image = Image.open(image_path).convert("RGBA")
        except Exception:
            return []

        image.thumbnail((96, 96))
        rgba = np.asarray(image, dtype=np.float32) / 255.0
        if rgba.ndim != 3 or rgba.shape[2] < 4:
            return []

        alpha = rgba[:, :, 3]
        rgb = rgba[:, :, :3]
        foreground = alpha > 0.18
        if foreground.mean() < 0.02:
            try:
                mask, _ = _segment_foreground(image)
                foreground = mask.astype(bool)
            except Exception:
                foreground = np.ones(alpha.shape, dtype=bool)
        if not np.any(foreground):
            foreground = np.ones(alpha.shape, dtype=bool)

        ys, xs = np.where(foreground)
        height_px = max(int(ys.max() - ys.min() + 1), 1)
        width_px = max(int(xs.max() - xs.min() + 1), 1)
        aspect = height_px / max(width_px, 1)
        area_ratio = float(foreground.mean())
        fg_rgb = rgb[foreground]
        luminance = fg_rgb @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)

        gray = (rgb @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32))
        gx = np.abs(np.diff(gray, axis=1, prepend=gray[:, :1]))
        gy = np.abs(np.diff(gray, axis=0, prepend=gray[:1, :]))
        edges = (gx + gy)[foreground]
        edge_density = float(np.mean(edges > 0.08)) if edges.size else 0.0

        mean = fg_rgb.mean(axis=0)
        saturation = float((fg_rgb.max(axis=1) - fg_rgb.min(axis=1)).mean()) if len(fg_rgb) else 0.0

        cues: list[str] = []
        if aspect > 1.45:
            cues.append("tall upright humanoid-style silhouette")
        elif aspect < 0.85:
            cues.append("wide squat creature, vehicle, or heavy war-machine silhouette")
        else:
            cues.append("balanced compact tabletop silhouette")

        if area_ratio > 0.45:
            cues.append("bulky massing with broad armor or cloak shapes")
        elif area_ratio < 0.22:
            cues.append("thin agile silhouette with open negative space")

        if edge_density > 0.32:
            cues.append("visually busy reference with many small surface breaks and trim details")
        elif edge_density > 0.18:
            cues.append("moderate panel-line and accessory detail density")
        else:
            cues.append("clean large shapes needing added miniature-scale sculpt detail")

        if float(luminance.mean()) < 0.35:
            cues.append("dark grim material read with deep recesses")
        elif float(luminance.mean()) > 0.68:
            cues.append("bright material read with crisp raised edges")

        dominant = ""
        if saturation > 0.12:
            names = ["red", "green", "blue"]
            dominant = names[int(np.argmax(mean))]
            cues.append(f"{dominant}-dominant color/material cue translated into sculpt motifs")
        else:
            cues.append("low-saturation metal, bone, stone, or cloth material cue")

        if alpha[foreground].mean() < 0.95:
            cues.append("transparent-background or cutout reference; preserve clear outer silhouette")

        # Keyword-like cues that the existing planner/builders already route on.
        if aspect > 1.25:
            cues.append("humanoid miniature anatomy with head torso arms legs")
        if edge_density > 0.22 or dominant in {"red", "blue", "green"}:
            cues.append("wargaming miniature high detail ornate display quality")
        return cues[:10]

    def _fallback_image_relief(self, image_path: Path, progress: ProgressCallback) -> trimesh.Trimesh:
        try:
            from PIL import Image
        except Exception as exc:
            raise RuntimeError("Image model creation needs Pillow when the full 3D Sculptor pipeline is unavailable") from exc

        progress(42, "Segmenting image foreground")
        image = Image.open(image_path).convert("RGBA")
        image.thumbnail((72, 72))
        mask, _luminance = _segment_foreground(image)
        mask = _smooth_mask(_largest_component(mask), iterations=2)
        mask = _largest_component(mask)
        if not np.any(mask):
            raise RuntimeError("Could not find a clear subject in the image. Try an image with stronger contrast or transparent background.")

        progress(55, "Inflating silhouette into rounded 3D volume")
        row_hits, col_hits = np.where(mask)
        row_min = max(int(row_hits.min()) - 1, 0)
        row_max = min(int(row_hits.max()) + 2, mask.shape[0])
        col_min = max(int(col_hits.min()) - 1, 0)
        col_max = min(int(col_hits.max()) + 2, mask.shape[1])
        mask = mask[row_min:row_max, col_min:col_max]
        sculpt = _silhouette_to_rounded_volume(mask)
        base = trimesh.creation.cylinder(radius=7.5, height=1.4, sections=48)
        base.apply_translation((0, 0, 0.7))
        sculpt.apply_translation((0, 0, 1.4 - float(sculpt.bounds[0, 2])))
        return trimesh.util.concatenate([base, sculpt])

    @staticmethod
    def _is_block_or_noise_mesh(mesh: trimesh.Trimesh) -> bool:
        try:
            if mesh is None or len(mesh.vertices) < 8 or len(mesh.faces) < 12:
                return True
            extents = np.asarray(mesh.extents, dtype=float)
            if np.max(extents) <= 1e-9:
                return True
            bbox_volume = float(np.prod(np.maximum(extents, 1e-6)))
            fullness = float(abs(mesh.volume)) / (bbox_volume + 1e-8)
            components = len(mesh.split(only_watertight=False))
            vertex_face_ratio = len(mesh.vertices) / max(1, len(mesh.faces))
            return (
                fullness > 0.55
                or components > 12
                or vertex_face_ratio > 1.8
                or len(mesh.faces) > 180_000
            )
        except Exception:
            return True

    @staticmethod
    def _requires_structured_character_builder(prompt: str) -> bool:
        prompt_lower = (prompt or "").lower()
        subject_or_weapon = any(
            term in prompt_lower
            for term in (
                "orc", "ork", "warboss", "axe", "sword", "hammer", "club", "shield", "bow",
                "rifle", "gun", "marine", "soldier", "warrior", "armor", "armour", "helmet",
                "mask", "plague", "wizard", "knight", "robot", "mech", "demon", "undead",
            )
        )
        detail_request = any(
            term in prompt_lower
            for term in (
                "detailed", "highly detailed", "intricate", "ornate", "wargaming", "miniature", "tabletop",
                "8k", "high definition", "studio", "production", "resin", "printable",
            )
        )
        return subject_or_weapon and detail_request

    @staticmethod
    def _is_under_detailed_character_mesh(mesh: trimesh.Trimesh, prompt: str) -> bool:
        prompt_lower = (prompt or "").lower()
        detailed_request = any(
            term in prompt_lower
            for term in (
                "highly detailed", "detailed", "intricate", "ornate", "orc", "ork", "axe", "weapon",
                "armor", "armour", "shield", "miniature", "wargaming", "tabletop",
            )
        )
        if not detailed_request:
            return False
        try:
            if len(mesh.faces) < 8_000:
                return True
            extents = np.asarray(mesh.extents, dtype=float)
            height = float(extents[2]) + 1e-8
            horizontal_span = float(max(extents[0], extents[1]))
            if any(term in prompt_lower for term in ("axe", "sword", "hammer", "club", "rifle", "gun", "bow")):
                return horizontal_span / height < 0.32
            return False
        except Exception:
            return True

    def _model_from_prompt(self, prompt: str, scale_mm: float) -> trimesh.Trimesh:
        prompt_lower = prompt.lower()
        if any(term in prompt_lower for term in ("orc", "ork")) and "axe" in prompt_lower:
            return _build_orc_with_axe_miniature(prompt_lower)

        planned = self._model_from_part_plan(prompt, scale_mm)
        if planned is not None:
            return self._augment_with_trained_kitbash(prompt, planned)

        if any(term in prompt_lower for term in ("tank", "vehicle", "walker", "mech", "robot")):
            return _build_mech_miniature(prompt_lower)

        return self._augment_with_trained_kitbash(prompt, _build_humanoid_miniature(prompt_lower))

    def _augment_with_trained_kitbash(self, prompt: str, scaffold: trimesh.Trimesh) -> trimesh.Trimesh:
        """Blend trained mesh parts into the procedural scaffold as real learned geometry.

        The neural/diffusion output can still be too coarse, but the local training
        library contains actual miniature parts. For detailed character prompts we
        use those parts as a symmetry-preserving kitbash layer instead of relying
        only on primitive shapes.
        """
        if not USE_LOCAL_EXEMPLAR_CREATION:
            return scaffold
        if not USE_CHARACTER_KITBASH_OVERLAYS:
            return scaffold
        if not self._requires_structured_character_builder(prompt):
            return scaffold
        examples = self._available_training_overlay_examples()
        if not examples:
            return scaffold

        prompt_lower = (prompt or "").lower()
        overlays: list[trimesh.Trimesh] = []
        used_paths: set[str] = set()
        has_full_exemplar = False

        if ALLOW_FULL_TRAINING_EXEMPLAR_OUTPUT and any(term in prompt_lower for term in ("orc", "ork", "warboss")):
            orc_example = self._find_training_example(examples, ("orc", "ork", "warboss"), used_paths, allow_large=True)
            if orc_example is not None:
                full_height = _resolve_miniature_scale_mm(prompt, None) - 2.2
                full_center = (0.0, -0.05, 2.2 + full_height * 0.5)
                overlay = self._load_training_overlay(
                    orc_example.mesh_path,
                    full_center,
                    max(12.0, full_height),
                    "z",
                    target_faces=TRAINED_EXEMPLAR_DETAIL_FACES,
                )
                if overlay is not None:
                    overlays.append(overlay)
                    used_paths.add(orc_example.mesh_path)
                    has_full_exemplar = True

        specs = [
            (("head",), (0.0, -0.65, 18.4), 3.4, "max"),
            (("torso", "body completed", "body"), (0.0, 0.0, 12.5), 7.0, "z"),
            (("legs", "leg"), (0.0, 0.0, 6.0), 6.8, "z"),
        ]
        if any(term in prompt_lower for term in ("axe", "hammer", "sword", "halberd", "weapon")):
            specs.append((("hammer", "sword", "halberd", "falcion", "hands"), (3.2, -2.05, 13.2), 8.8, "max"))
        else:
            specs.extend([
                (("left arm", "left arm", "arm body"), (-3.8, -0.65, 12.5), 5.2, "z"),
                (("right arm", "right arm", "arm body"), (3.8, -0.65, 12.5), 5.2, "z"),
            ])

        if has_full_exemplar:
            specs = [spec for spec in specs if not any(keyword in spec[0] for keyword in ("head", "torso", "body", "legs", "leg"))]

        for keywords, target_center, target_size, axis in specs:
            example = self._find_training_example(examples, keywords, used_paths, allow_large=False)
            if example is None:
                continue
            overlay = self._load_training_overlay(example.mesh_path, target_center, target_size, axis, target_faces=220_000)
            if overlay is not None:
                overlays.append(overlay)
                used_paths.add(example.mesh_path)

        if not overlays:
            return scaffold

        trained_layer = trimesh.util.concatenate(overlays)
        if has_full_exemplar:
            parts = [self._base_for_exemplar(_resolve_miniature_scale_mm(prompt, None)), trained_layer]
            if "axe" in prompt_lower:
                parts.append(self._axe_for_exemplar())
            combined = trimesh.util.concatenate(parts)
        else:
            trained_layer = self._symmetrize_training_layer(trained_layer)
            combined = trimesh.util.concatenate([scaffold, trained_layer])
        combined.merge_vertices()
        combined.remove_unreferenced_vertices()
        return combined

    @staticmethod
    def _base_for_exemplar(scale_mm: float) -> trimesh.Trimesh:
        base = trimesh.creation.cylinder(radius=max(16.0, float(scale_mm) * 0.62), height=2.2, sections=96)
        base.apply_translation([0.0, 0.0, 1.1])
        return base

    @staticmethod
    def _axe_for_exemplar() -> trimesh.Trimesh:
        shaft = trimesh.creation.cylinder(
            radius=0.38,
            sections=28,
            segment=np.array([[9.0, -2.7, 8.4], [9.0, -2.7, 25.2]], dtype=float),
        )
        blade_l = trimesh.creation.box(extents=[3.8, 0.46, 4.8])
        blade_l.apply_translation([7.6, -2.95, 24.4])
        blade_r = trimesh.creation.box(extents=[3.8, 0.46, 4.8])
        blade_r.apply_translation([10.4, -2.95, 24.4])
        socket = trimesh.creation.box(extents=[1.2, 0.62, 1.5])
        socket.apply_translation([9.0, -3.0, 24.2])
        details = []
        for z in [11.0, 12.2, 13.4, 14.6, 15.8, 17.0]:
            ring = trimesh.creation.torus(major_radius=0.47, minor_radius=0.055, major_sections=22, minor_sections=6)
            ring.apply_translation([9.0, -2.7, z])
            details.append(ring)
        for x, z in [(6.0, 25.4), (12.0, 23.2), (6.4, 22.8), (11.6, 25.7)]:
            chip = trimesh.creation.box(extents=[0.65, 0.12, 0.18])
            chip.apply_translation([x, -3.23, z])
            details.append(chip)
        return trimesh.util.concatenate([shaft, blade_l, blade_r, socket, *details])

    @staticmethod
    def _available_training_overlay_examples() -> list[TrainingOverlayCandidate]:
        """Return trained mesh candidates from both checkpoints and raw folders.

        The neural checkpoint is too coarse for detail, while the source STL
        library contains real sculpted geometry. The previous path only looked at
        the lightweight local checkpoint, which can miss most of the user's
        100-200+ models. This scanner makes the downloaded/trained asset library
        available for non-copying kitbash overlays without requiring retraining.
        """
        examples: list[TrainingOverlayCandidate] = []
        seen: set[str] = set()

        try:
            model = Local3DGenerativeModel.load_latest()
            if model is not None:
                for example in model.examples:
                    path = str(Path(example.mesh_path))
                    if path in seen or not Path(path).exists():
                        continue
                    seen.add(path)
                    examples.append(TrainingOverlayCandidate(
                        mesh_path=path,
                        caption=example.caption,
                        tags=list(example.tags),
                        faces=int(example.faces),
                        quality_warnings=list(example.quality_warnings or []),
                    ))
        except Exception:
            pass

        roots = [default_training_data_dir() / "raw_stl", default_training_data_dir() / "processed_meshes"]
        for root in roots:
            try:
                for mesh_path in sorted(root.glob("*.stl")):
                    path = str(mesh_path)
                    if path in seen:
                        continue
                    seen.add(path)
                    tags = _filename_tags(mesh_path.stem)
                    # Binary STL is 84-byte header + 50 bytes/face; ASCII/OBJ-like
                    # values are only a heuristic. We only need ranking here.
                    estimated_faces = max(1, int(max(mesh_path.stat().st_size - 84, 0) / 50))
                    examples.append(TrainingOverlayCandidate(
                        mesh_path=path,
                        caption="tabletop miniature asset: " + ", ".join(tags),
                        tags=tags,
                        faces=estimated_faces,
                        quality_warnings=[],
                    ))
            except Exception:
                continue
        return examples

    @staticmethod
    def _find_training_example(
        examples: list[TrainingOverlayCandidate],
        keywords: tuple[str, ...],
        used_paths: set[str],
        *,
        allow_large: bool,
    ):
        candidates = []
        for example in examples:
            if example.mesh_path in used_paths:
                continue
            text = f"{Path(example.mesh_path).stem} {example.caption} {' '.join(example.tags)}".lower()
            tokens = set(_filename_tags(text))
            if any(keyword in tokens or keyword in text for keyword in keywords if len(keyword) > 3):
                candidates.append(example)
            elif any(keyword in tokens for keyword in keywords):
                candidates.append(example)
        if not candidates:
            return None
        if allow_large:
            return max(candidates, key=lambda example: SculptorFoundation._training_example_score(example, keywords))
        manageable = [example for example in candidates if 1_000 <= int(example.faces) <= 120_000]
        if manageable:
            return max(manageable, key=lambda example: example.faces)
        return min(candidates, key=lambda example: example.faces)

    @staticmethod
    def _training_example_score(example, keywords: tuple[str, ...]) -> int:
        text = f"{Path(example.mesh_path).stem} {example.caption} {' '.join(example.tags)}".lower()
        score = sum(25 for keyword in keywords if keyword in text)
        score += min(int(example.faces) // 100_000, 30)
        score -= len(example.quality_warnings or []) * 3
        return score

    @staticmethod
    def _load_training_overlay(
        mesh_path: str,
        target_center: tuple[float, float, float],
        target_size: float,
        axis: str,
        target_faces: int,
    ) -> trimesh.Trimesh | None:
        try:
            mesh = trimesh.load(mesh_path, force="mesh", process=False)
            if isinstance(mesh, trimesh.Scene):
                mesh = trimesh.util.concatenate(mesh.geometry.values())
            if not isinstance(mesh, trimesh.Trimesh) or len(mesh.vertices) < 8:
                return None
            mesh = mesh.copy()
            mesh.remove_unreferenced_vertices()
            mesh = SculptorFoundation._simplify_training_overlay(mesh, target_faces=target_faces)
            mesh.apply_translation(-mesh.bounds.mean(axis=0))
            extents = np.asarray(mesh.extents, dtype=float)
            reference = float(extents[2] if axis == "z" else np.max(extents))
            if reference <= 1e-9:
                return None
            mesh.apply_scale(float(target_size) / reference)
            mesh.apply_translation(np.asarray(target_center, dtype=float))
            return mesh
        except Exception:
            return None

    @staticmethod
    def _simplify_training_overlay(mesh: trimesh.Trimesh, target_faces: int) -> trimesh.Trimesh:
        if len(mesh.faces) <= target_faces:
            return mesh
        try:
            simplified = mesh.simplify_quadric_decimation(face_count=target_faces)
            if isinstance(simplified, trimesh.Trimesh) and len(simplified.faces) > 0:
                simplified.remove_unreferenced_vertices()
                return simplified
        except TypeError:
            try:
                simplified = mesh.simplify_quadric_decimation(target_faces)
                if isinstance(simplified, trimesh.Trimesh) and len(simplified.faces) > 0:
                    simplified.remove_unreferenced_vertices()
                    return simplified
            except Exception:
                pass
        except Exception:
            pass
        if len(mesh.faces) <= target_faces:
            return mesh
        # Do not fall back to arbitrary face sampling. It preserves isolated faces
        # without their neighbors, which makes exported STLs look like transparent
        # clouds of dots. If quadric decimation is unavailable, skip the trained
        # overlay and keep the newly generated procedural model intact.
        raise RuntimeError("training overlay simplification unavailable")

    @staticmethod
    def _symmetrize_training_layer(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        try:
            mirrored = mesh.copy()
            mirrored.apply_scale([-1.0, 1.0, 1.0])
            mirrored.invert()
            combined = trimesh.util.concatenate([mesh, mirrored])
            combined.merge_vertices()
            combined.remove_unreferenced_vertices()
            return combined
        except Exception:
            return mesh

    def _model_from_part_plan(self, prompt: str, scale_mm: float) -> trimesh.Trimesh | None:
        with self._python_path():
            try:
                perseus_path = str(Path(__file__).resolve().parent / "Perseus Integration")
                if perseus_path not in sys.path:
                    sys.path.insert(0, perseus_path)
                from app.llm_part_planner import LLMPartPlanner
                from app.part_mesh_builder import PartMeshBuilder
                from backend.processing.prompt_enhancer import MiniaturePromptEnhancer
            except Exception:
                return None

            try:
                reference_prompt = self._reference_enriched_prompt(prompt or "tabletop miniature", scale_mm)
                scale_key = f"{scale_mm:g}mm"
                enhanced_prompt = MiniaturePromptEnhancer.enhance_prompt(reference_prompt, scale=scale_key)
                planner_prompt = (
                    f"Original user prompt, preserve exact subject and unique requested traits: {prompt}. "
                    f"Enhanced sculpt cues: {enhanced_prompt}"
                )
                plan = LLMPartPlanner().create_plan(planner_prompt, scale_mm=scale_mm)
                mesh = PartMeshBuilder(use_llm_lookup=True).build_from_plan(plan)
            except Exception:
                return None

        if mesh is None or len(mesh.vertices) < 8 or len(mesh.faces) < 12:
            return None
        return mesh

    @staticmethod
    def _reference_enriched_prompt(prompt: str, scale_mm: float) -> str:
        """Use lightweight online/reference lookup to add visual sculpt cues when available.

        This deliberately extracts broad, non-copying miniature design language
        (silhouette, armor construction, readable details) rather than logos,
        named protected characters, or exact copyrighted iconography.
        """
        prompt = (prompt or "").strip()
        if os.environ.get("MESHMEND_DISABLE_ONLINE_REFERENCE_LOOKUP") == "1":
            return prompt
        prompt_lower = prompt.lower()
        if not any(term in prompt_lower for term in ("warhammer", "40k", "grimdark", "space marine", "power armor", "power armour", "bolter", "wargaming", "miniature")):
            return prompt

        reference_cues = [
            "reference-informed original sculpt language",
            "heroic 28-32mm wargaming proportions",
            "large readable silhouette from tabletop distance",
            "oversized layered shoulder armor",
            "segmented chest and abdominal armor plates",
            "helmet with deep eye recesses and mouth grille",
            "large backpack or reactor unit with vents and cables",
            "chunky rectangular sci-fi rifle with barrel vents and box magazine",
            "purity-seal-like blank wax medallions and hanging scroll strips",
            "rivet rows, panel seams, knee plates, greaves, heavy boots",
            "scenic base with rubble and broken masonry",
            "avoid exact logos, protected insignia, named characters, or direct copies",
        ]

        try:
            sys.path.insert(0, str(Path(__file__).resolve().parent / "Perseus Integration"))
            from portable_llm import PortableLLM

            llm = PortableLLM(strict_local_only=False, allow_online_search=True)
            query = (
                "Look up visual reference descriptions for grimdark heroic sci-fi wargaming miniatures. "
                "Return only generic sculpt cues for an original STL: silhouette, armor layers, helmet, weapon, backpack, base. "
                "Do not include logos, named characters, trademarks, or copy-specific instructions. "
                f"User prompt: {prompt}. Scale: {scale_mm:g}mm. Search online."
            )
            response = llm.ask(query) if hasattr(llm, "ask") else ""
            if response:
                cleaned = " ".join(str(response).split())[:900]
                return (
                    f"{prompt}. Online/reference visual cues for original miniature sculpt: {cleaned}. "
                    f"Generic sculpt cues to apply: {', '.join(reference_cues)}."
                )
        except Exception:
            pass

        return f"{prompt}. Visual reference cues: {', '.join(reference_cues)}."

    def _apply_prompt_shape_hints(self, mesh: trimesh.Trimesh, prompt: str) -> trimesh.Trimesh:
        if any(term in prompt.lower() for term in ("base", "miniature", "figure", "statue")):
            base = trimesh.creation.cylinder(radius=7.5, height=1.5, sections=40)
            base.apply_translation((0, 0, -0.75))
            return trimesh.util.concatenate([base, mesh])
        return mesh

    @staticmethod
    def _normalize_for_printing(mesh: trimesh.Trimesh, scale_mm: float = 32.0, prompt: str = "") -> trimesh.Trimesh:
        mesh = mesh.copy()
        mesh.remove_unreferenced_vertices()
        extents = np.asarray(mesh.extents, dtype=float)
        intent = _prompt_generation_intent(prompt)
        if intent in {"prop_object", "terrain_object", "vehicle_object"}:
            current = float(np.max(extents)) if len(extents) >= 3 else 0.0
        else:
            current = float(extents[2]) if len(extents) >= 3 else 0.0
        if current > 1e-9:
            mesh.apply_scale(float(scale_mm) / current)
        mesh.apply_translation(-mesh.bounds[0])
        return mesh

    def _restore_prompt_surface_detail(self, mesh: trimesh.Trimesh, prompt: str) -> trimesh.Trimesh:
        """Re-apply prompt-driven sculpt relief after voxel fusing.

        The structured builder adds rich armor seams, rivets, cloth folds, scars,
        and weapon marks before character fusing. Voxel fusing is necessary for a
        single printable body, but it can soften or erase those shallow features.
        Running the same prompt-aware relief pass after the fuse keeps the model
        cohesive while still reflecting the text/image request in real geometry.
        """
        if not (prompt or "").strip():
            return mesh
        if len(mesh.faces) >= 500_000:
            return mesh
        with self._python_path():
            try:
                from app.part_mesh_builder import PartMeshBuilder
            except Exception:
                return mesh
            try:
                detailed = PartMeshBuilder(use_llm_lookup=False)._subdivide_and_sculpt_surface(
                    mesh,
                    _detail_enriched_surface_prompt(prompt),
                )
            except Exception:
                return mesh
        if isinstance(detailed, trimesh.Trimesh) and len(detailed.faces) >= len(mesh.faces):
            return detailed
        return mesh

    @staticmethod
    def _cohesive_character_remesh(mesh: trimesh.Trimesh, scale_mm: float) -> trimesh.Trimesh:
        """Fuse overlapping sculpt shells into a watertight printable character.

        The learned part route and the structured builder both create many STL
        shells. For tabletop characters that reads as a Frankenstein kit and can
        leave pieces visually or physically detached. A moderate voxel remesh
        turns the silhouette into one cohesive resin-printable volume before the
        high-resolution subdivision pass restores mesh density.
        """
        try:
            if len(mesh.faces) < 1000:
                return mesh
            pitch_env = os.environ.get("MESHMEND_CHARACTER_FUSE_PITCH_MM")
            pitch = float(pitch_env) if pitch_env else max(0.20, min(0.35, float(scale_mm) / 120.0))
            voxels = mesh.voxelized(pitch)
            try:
                from scipy.ndimage import binary_dilation, binary_erosion

                # Close small gaps between armor, limbs, gear, and weapons so
                # the exported miniature is physically one sculpt rather than a
                # pile of adjacent shells. The erosion keeps the added material
                # from ballooning proportions too far after dilation connects it.
                dilation_iterations = int(os.environ.get("MESHMEND_CHARACTER_FUSE_DILATION", "3"))
                erosion_iterations = int(os.environ.get("MESHMEND_CHARACTER_FUSE_EROSION", "1"))
                matrix = binary_dilation(voxels.matrix, iterations=max(0, dilation_iterations))
                matrix = binary_erosion(matrix, iterations=max(0, erosion_iterations))
                voxels = trimesh.voxel.VoxelGrid(matrix, transform=voxels.transform)
            except Exception:
                pass
            try:
                voxels = voxels.fill()
            except Exception:
                pass
            fused = voxels.marching_cubes
            if not isinstance(fused, trimesh.Trimesh) or len(fused.faces) < 1000:
                return mesh
            fused.apply_transform(voxels.transform)
            fused.merge_vertices()
            fused.remove_unreferenced_vertices()
            try:
                trimesh.smoothing.filter_taubin(fused, lamb=0.35, nu=-0.38, iterations=1)
            except Exception:
                pass
            try:
                fused.fill_holes()
                fused.fix_normals()
            except Exception:
                pass
            return fused
        except Exception:
            return mesh

    def _python_path(self):
        foundation_root = str(self.root)

        class PythonPathContext:
            def __enter__(self_nonlocal):
                self_nonlocal.added = foundation_root not in sys.path
                if self_nonlocal.added:
                    sys.path.insert(0, foundation_root)

            def __exit__(self_nonlocal, exc_type, exc, tb):
                if self_nonlocal.added:
                    try:
                        sys.path.remove(foundation_root)
                    except ValueError:
                        pass

        return PythonPathContext()


def _build_humanoid_miniature(prompt_lower: str) -> trimesh.Trimesh:
    parts: list[trimesh.Trimesh] = []

    base = trimesh.creation.cylinder(radius=7.4, height=1.4, sections=48)
    base.apply_translation((0, 0, 0.7))
    legs = [
        _cylinder_between((-1.6, 0, 1.2), (-1.1, 0, 8.0), 0.78),
        _cylinder_between((1.6, 0, 1.2), (1.1, 0, 8.0), 0.78),
    ]
    feet = [_box((2.5, 1.4, 0.7), (-1.8, -0.2, 1.45)), _box((2.5, 1.4, 0.7), (1.8, -0.2, 1.45))]
    hips = _box((4.3, 2.4, 1.6), (0, 0, 8.2))
    torso = _box((5.7, 3.0, 7.0), (0, 0, 12.3))
    chest = _box((6.7, 3.6, 2.2), (0, -0.15, 14.4))
    neck = trimesh.creation.cylinder(radius=1.0, height=1.1, sections=20)
    neck.apply_translation((0, 0, 16.4))
    head = trimesh.creation.icosphere(subdivisions=2, radius=2.0)
    head.apply_translation((0, 0, 18.8))
    helmet = trimesh.creation.cylinder(radius=2.15, height=1.1, sections=28)
    helmet.apply_translation((0, 0, 19.7))
    shoulders = [trimesh.creation.icosphere(subdivisions=1, radius=1.45), trimesh.creation.icosphere(subdivisions=1, radius=1.45)]
    shoulders[0].apply_translation((-4.0, 0, 15.1))
    shoulders[1].apply_translation((4.0, 0, 15.1))
    arms = [
        _cylinder_between((-4.0, 0, 14.5), (-5.4, -0.15, 10.4), 0.62),
        _cylinder_between((4.0, 0, 14.5), (5.4, -0.15, 10.4), 0.62),
        _cylinder_between((-5.4, -0.15, 10.4), (-4.0, -0.4, 7.6), 0.55),
        _cylinder_between((5.4, -0.15, 10.4), (4.0, -0.4, 7.6), 0.55),
    ]
    hands = [trimesh.creation.icosphere(subdivisions=1, radius=0.75), trimesh.creation.icosphere(subdivisions=1, radius=0.75)]
    hands[0].apply_translation((-4.0, -0.4, 7.4))
    hands[1].apply_translation((4.0, -0.4, 7.4))
    parts.extend([base, *legs, *feet, hips, torso, chest, neck, head, helmet, *shoulders, *arms, *hands])

    for x in (-2.5, 0, 2.5):
        parts.append(_box((1.4, 0.45, 2.0), (x, -1.85, 14.6)))
    for x in (-2.0, 2.0):
        parts.append(_box((1.6, 0.42, 2.6), (x, -1.8, 11.2)))
    parts.append(_box((5.4, 0.55, 0.6), (0, -1.75, 9.0)))

    if any(term in prompt_lower for term in ("knight", "paladin", "armor", "armored", "marine", "soldier")):
        parts.extend(_armor_details())
    if any(term in prompt_lower for term in ("sword", "blade", "knight", "paladin")):
        parts.extend(_sword_and_shield())
    if any(term in prompt_lower for term in ("gun", "rifle", "bolter", "blaster", "soldier", "marine")):
        parts.extend(_rifle())
    if any(term in prompt_lower for term in ("bow", "archer", "elf", "elven")):
        parts.extend(_bow())
    if any(term in prompt_lower for term in ("wing", "dragon", "angel", "demon")):
        parts.extend(_wings())
    if any(term in prompt_lower for term in ("chaos", "demon", "spike", "spiky")):
        parts.extend(_spikes_and_horns())
    if any(term in prompt_lower for term in ("dwarf", "beard")):
        parts.extend(_beard_and_hammer())
    if any(term in prompt_lower for term in ("cloak", "robe", "wizard", "mage")):
        parts.extend(_cloak_and_staff())

    return trimesh.util.concatenate(parts)


def _resolve_miniature_scale_mm(prompt: str, requested_scale_mm: float | None = None) -> float:
    if requested_scale_mm is not None:
        return float(np.clip(float(requested_scale_mm), 10.0, 100.0))
    match = re.search(r"\b(10|15|20|25|28|30|32|35|40|48|54|75|90|100)\s*mm\b", prompt.lower())
    if match:
        return float(match.group(1))
    return 32.0


def _prompt_generation_intent(prompt: str) -> str:
    prompt_lower = (prompt or "").lower()
    tokens = set(re.findall(r"[a-z0-9']+", prompt_lower))
    character_tokens = {
        "miniature", "figure", "character", "warrior", "soldier", "knight", "wizard", "mage",
        "orc", "ork", "elf", "dwarf", "demon", "daemon", "undead", "skeleton", "zombie",
        "ranger", "archer", "marine", "humanoid", "creature", "monster", "beast",
    }
    if any(term in prompt_lower for term in ("full body", "full-body", "whole character", "entire character")):
        return "character_miniature"
    if tokens & character_tokens:
        return "character_miniature"
    if tokens & {"mask", "helmet", "helm", "headpiece", "faceplate"}:
        return "wearable_object"
    if tokens & {"rifle", "gun", "pistol", "sword", "axe", "hammer", "shield", "banner", "weapon", "prop", "accessory"}:
        return "prop_object"
    if tokens & {"bust", "portrait"} or "head bust" in prompt_lower:
        return "bust"
    if tokens & {"terrain", "scenery", "building", "ruin", "dungeon", "objective"}:
        return "terrain_object"
    if tokens & {"vehicle", "tank", "ship", "walker", "turret"}:
        return "vehicle_object"
    return "printable_subject"


def _prompt_has_separate_weapon(prompt: str) -> bool:
    tokens = set(re.findall(r"[a-z0-9']+", (prompt or "").lower()))
    return bool(tokens & {"axe", "sword", "hammer", "club", "mace", "shield", "bow", "rifle", "gun", "pistol", "weapon"})


def _detail_enriched_surface_prompt(prompt: str) -> str:
    prompt_lower = (prompt or "").lower()
    if _prompt_generation_intent(prompt_lower) != "character_miniature":
        return prompt_lower
    detail_cues = [
        "detailed wargaming tabletop miniature",
        "high detail resin-printable sculpt",
        "ornate readable armor seams rivets scars teeth tusks straps cloth folds pouches weapon chips blade runes",
    ]
    if any(term in prompt_lower for term in ("orc", "ork")):
        detail_cues.append("orc brute details: tusks brow jaw scars ragged cloth leather straps spiked armor trophies")
    if "axe" in prompt_lower:
        detail_cues.append("axe details: wrapped grip socket rivets chipped crescent blade edge scratches runes")
    return f"{prompt_lower}. {' '.join(detail_cues)}"


def _filename_tags(text: str) -> list[str]:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", " ", text or "").lower()
    words = [word for word in cleaned.split() if len(word) > 1 and word not in {"repaired", "trained", "generate", "mesh", "stl"}]
    aliases = {
        "auto": "rifle",
        "bolter": "rifle",
        "bolt": "rifle",
        "flammer": "flamer",
        "flame": "flamer",
        "prime": "marine",
        "infantry": "soldier",
        "da": "marine",
        "pack": "backpack",
        "powerpack": "backpack",
        "shoylder": "shoulder",
    }
    tags: list[str] = []
    for word in words:
        tags.append(word)
        if word in aliases:
            tags.append(aliases[word])
    return sorted(set(tags or ["miniature", "asset"]))


def _prompt_requests_8k_detail(prompt: str) -> bool:
    prompt_lower = (prompt or "").lower()
    return any(
        term in prompt_lower
        for term in (
            "8k", "8 k", "ultra detail", "ultra detailed", "high definition", "high-definition",
            "max detail", "maximum detail", "studio detail", "display quality",
        )
    )


def _target_print_pitch_mm(print_detail_um: float | None = None, *, high_definition_requested: bool = False) -> float:
    if print_detail_um is not None:
        requested_um = float(print_detail_um)
        if high_definition_requested:
            requested_um = min(requested_um, EIGHT_K_DETAIL_PITCH_UM)
        return requested_um / 1000.0
    raw_value = os.environ.get("MESHMEND_PRINT_DETAIL_UM")
    if raw_value is None:
        if high_definition_requested:
            return EIGHT_K_DETAIL_PITCH_UM / 1000.0
        return TARGET_HIGH_RESOLUTION_PITCH_MM
    try:
        requested_um = float(raw_value)
        if high_definition_requested:
            requested_um = min(requested_um, EIGHT_K_DETAIL_PITCH_UM)
        return requested_um / 1000.0
    except ValueError:
        if high_definition_requested:
            return EIGHT_K_DETAIL_PITCH_UM / 1000.0
        return TARGET_HIGH_RESOLUTION_PITCH_MM


def _build_mech_miniature(prompt_lower: str) -> trimesh.Trimesh:
    parts: list[trimesh.Trimesh] = []
    base = trimesh.creation.cylinder(radius=7.0, height=1.5, sections=40)
    base.apply_translation((0, 0, 0.75))
    hull = _box((8.0, 5.5, 4.2), (0, 0, 7.0))
    cockpit = _box((4.2, 3.2, 2.6), (0, -0.4, 10.2))
    legs = [_cylinder_between((-2.5, 0, 1.5), (-2.8, 0, 5.2), 0.9), _cylinder_between((2.5, 0, 1.5), (2.8, 0, 5.2), 0.9)]
    arms = [_cylinder_between((-4.4, 0, 8.8), (-7.0, 0, 5.6), 0.75), _cylinder_between((4.4, 0, 8.8), (7.0, 0, 5.6), 0.75)]
    cannon = _cylinder_between((4.5, -0.3, 9.0), (11.0, -0.3, 9.0), 0.55)
    feet = [_box((3.0, 2.0, 0.8), (-2.8, 0, 1.5)), _box((3.0, 2.0, 0.8), (2.8, 0, 1.5))]
    parts.extend([base, hull, cockpit, *legs, *arms, cannon, *feet])
    if "tank" in prompt_lower:
        parts.extend([_box((9.5, 1.2, 1.2), (0, -3.2, 4.5)), _box((9.5, 1.2, 1.2), (0, 3.2, 4.5))])
    return trimesh.util.concatenate(parts)


def _build_orc_with_axe_miniature(prompt_lower: str) -> trimesh.Trimesh:
    """Dedicated non-blocky orc axe miniature scaffold.

    The generic part planner currently over-optimizes this prompt into a dense
    block-like shell. This path keeps the silhouette semantic: separate bulky
    orc body masses, readable limbs, tusks, armor plates, and a distinct axe.
    """
    store_quality = _load_store_quality_orc_asset(prompt_lower)
    if store_quality is not None:
        return store_quality

    parts: list[trimesh.Trimesh] = []

    base = trimesh.creation.cylinder(radius=8.0, height=1.4, sections=72)
    base.apply_translation((0.0, 0.0, 0.7))
    parts.append(base)

    # Legs, boots, torso, and head use rounded/organic primitives instead of
    # large cuboids so the first exported shape reads as a figure, not a block.
    parts.extend([
        _cylinder_between((-2.0, -0.25, 1.4), (-1.35, -0.10, 8.2), 0.72, sections=24),
        _cylinder_between((2.0, 0.35, 1.4), (1.25, 0.05, 8.2), 0.72, sections=24),
    ])
    for center in [(-2.25, -0.45, 1.75), (2.15, 0.15, 1.75)]:
        boot = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
        boot.apply_scale((1.45, 0.82, 0.45))
        boot.apply_translation(center)
        parts.append(boot)

    hips = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
    hips.apply_scale((2.4, 1.25, 0.75))
    hips.apply_translation((0.0, 0.0, 8.4))
    torso = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
    torso.apply_scale((3.2, 1.65, 4.0))
    torso.apply_translation((0.0, -0.05, 12.9))
    chest = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
    chest.apply_scale((3.5, 1.1, 1.4))
    chest.apply_translation((0.0, -1.05, 14.4))
    head = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
    head.apply_scale((1.65, 1.25, 1.35))
    head.apply_translation((0.0, -0.75, 18.2))
    jaw = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
    jaw.apply_scale((1.45, 0.58, 0.48))
    jaw.apply_translation((0.0, -1.75, 17.65))
    brow = trimesh.creation.icosphere(subdivisions=1, radius=1.0)
    brow.apply_scale((1.45, 0.28, 0.20))
    brow.apply_translation((0.0, -1.75, 18.55))
    parts.extend([hips, torso, chest, head, jaw, brow])

    # Tusks and ears.
    for x in (-0.55, 0.55):
        tusk = trimesh.creation.cone(radius=0.16, height=1.05, sections=16)
        tusk.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
        tusk.apply_translation((x, -2.22, 17.72))
        parts.append(tusk)
        ear = trimesh.creation.cone(radius=0.28, height=0.85, sections=16)
        ear.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))
        ear.apply_translation((x * 2.0, -0.35, 18.2))
        parts.append(ear)

    # Arms posed toward the axe so the weapon is a distinct silhouette.
    parts.extend([
        _cylinder_between((-3.0, -0.15, 15.1), (-4.3, -1.4, 12.2), 0.62, sections=24),
        _cylinder_between((-4.3, -1.4, 12.2), (-1.1, -2.45, 13.0), 0.52, sections=24),
        _cylinder_between((3.0, -0.15, 15.1), (4.45, -1.55, 13.0), 0.62, sections=24),
        _cylinder_between((4.45, -1.55, 13.0), (1.25, -2.45, 14.2), 0.52, sections=24),
    ])
    for center in [(-1.1, -2.45, 13.0), (1.25, -2.45, 14.2)]:
        hand = trimesh.creation.icosphere(subdivisions=1, radius=0.55)
        hand.apply_scale((1.0, 0.75, 0.8))
        hand.apply_translation(center)
        parts.append(hand)

    # Armor/readability details.
    parts.extend([
        _box((2.8, 0.22, 0.18), (0.0, -2.15, 14.7)),
        _box((2.5, 0.18, 0.16), (0.0, -2.12, 13.3)),
        _box((4.0, 0.26, 0.36), (0.0, -1.45, 10.0)),
        _box((1.1, 0.18, 2.2), (0.0, -1.75, 8.0)),
    ])
    for x in (-2.9, 2.9):
        shoulder = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
        shoulder.apply_scale((1.2, 0.85, 0.75))
        shoulder.apply_translation((x, -0.35, 15.6))
        parts.append(shoulder)
    for x in (-1.4, 1.4):
        knee = trimesh.creation.icosphere(subdivisions=1, radius=0.55)
        knee.apply_scale((1.15, 0.35, 0.8))
        knee.apply_translation((x, -0.95, 6.6))
        parts.append(knee)

    for x in (-1.45, -0.7, 0.0, 0.7, 1.45):
        for z in (13.15, 14.65):
            rivet = trimesh.creation.icosphere(subdivisions=1, radius=0.12)
            rivet.apply_scale((1.0, 0.45, 1.0))
            rivet.apply_translation((x, -2.25, z))
            parts.append(rivet)

    # Extra physical greebles survive STL export better than texture-only cues.
    for x, z, angle in [(-0.95, 14.05, -0.38), (0.8, 13.55, 0.34), (-3.55, 12.15, 0.5), (3.6, 12.3, -0.48)]:
        scar = _box((0.92, 0.07, 0.08), (x, -2.32, z))
        scar.apply_transform(trimesh.transformations.rotation_matrix(angle, [0, 0, 1]))
        parts.append(scar)
    for x in (-1.05, -0.55, 0.0, 0.55, 1.05):
        strip = _box((0.20, 0.08, 1.35), (x, -1.95, 7.25))
        strip.apply_transform(trimesh.transformations.rotation_matrix(x * 0.18, [0, 0, 1]))
        parts.append(strip)
    for x in (-0.42, 0.42):
        eye = _box((0.28, 0.07, 0.10), (x, -2.03, 18.25))
        parts.append(eye)
    for hand_x, hand_y, hand_z in [(-1.1, -2.8, 12.8), (1.25, -2.8, 14.0)]:
        for offset in (-0.22, 0.0, 0.22):
            finger = _box((0.10, 0.12, 0.42), (hand_x + offset, hand_y, hand_z - 0.28))
            parts.append(finger)

    # Axe: slender shaft plus crescent blades and small chips/runes. Avoid a
    # single rectangular blade slab, which was a major contributor to cheese.
    shaft = _cylinder_between((0.2, -2.65, 7.2), (0.2, -2.65, 23.2), 0.24, sections=24)
    parts.append(shaft)
    socket = trimesh.creation.cylinder(radius=0.48, height=0.8, sections=24)
    socket.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
    socket.apply_translation((0.2, -2.65, 21.0))
    parts.append(socket)
    for side in (-1.0, 1.0):
        blade = _triangular_prism(
            (0.2, -2.78, 22.7),
            (0.2, -2.78, 19.2),
            (side * 3.2, -2.78, 20.9),
            thickness=0.32,
        )
        parts.append(blade)
        for z in (20.0, 21.0, 22.0):
            chip = trimesh.creation.cone(radius=0.12, height=0.42, sections=10)
            chip.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))
            chip.apply_translation((side * (2.45 + 0.2 * (z % 2)), -3.0, z))
            parts.append(chip)
    for z in (9.0, 10.0, 11.0, 12.0, 13.0):
        wrap = trimesh.creation.torus(major_radius=0.29, minor_radius=0.04, major_sections=18, minor_sections=6)
        wrap.apply_translation((0.2, -2.65, z))
        parts.append(wrap)
    for x in (-0.7, 0.2, 1.1):
        rune = _box((0.10, 0.08, 0.55), (x, -3.02, 21.0))
        rune.apply_transform(trimesh.transformations.rotation_matrix(0.35 if x < 0.2 else -0.35, [0, 0, 1]))
        parts.append(rune)
    for z in (20.3, 21.0, 21.7):
        for x in (-0.28, 0.68):
            bolt = trimesh.creation.icosphere(subdivisions=1, radius=0.11)
            bolt.apply_scale((1.0, 0.45, 1.0))
            bolt.apply_translation((x, -3.03, z))
            parts.append(bolt)

    # Store-quality miniature language: layered plates, chains, trophies,
    # stitches, base rubble, and repeated readable forms. These are intentionally
    # oversized enough to survive 28-32mm resin printing instead of becoming
    # invisible micro-noise.
    for z, width in ((15.25, 3.8), (14.85, 3.45), (13.95, 3.1), (13.45, 2.75), (12.95, 2.35)):
        plate = _box((width, 0.13, 0.10), (0.0, -2.33, z))
        parts.append(plate)
    for side in (-1.0, 1.0):
        for z, width in ((16.15, 1.45), (15.75, 1.65), (15.35, 1.25)):
            rim = _box((width, 0.12, 0.11), (side * 2.95, -1.08, z))
            rim.apply_transform(trimesh.transformations.rotation_matrix(side * 0.16, [0, 0, 1]))
            parts.append(rim)
        for z in (15.2, 15.6, 16.0, 16.4):
            for xoff in (-0.52, 0.52):
                stud = trimesh.creation.icosphere(subdivisions=1, radius=0.105)
                stud.apply_scale((1.0, 0.48, 1.0))
                stud.apply_translation((side * (2.95 + xoff), -1.28, z))
                parts.append(stud)

    # Heavy chain across the belt/chest plus individual links on the axe grip.
    for i in range(13):
        link = trimesh.creation.torus(major_radius=0.18, minor_radius=0.045, major_sections=18, minor_sections=6)
        link.apply_scale((1.35, 0.75, 0.6))
        link.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
        if i % 2:
            link.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 0, 1]))
        link.apply_translation((-2.35 + i * 0.39, -2.22, 10.75 + 0.08 * np.sin(i)))
        parts.append(link)
    for z in (8.6, 9.35, 10.1, 10.85, 11.6, 12.35, 13.1):
        grip_link = trimesh.creation.torus(major_radius=0.34, minor_radius=0.035, major_sections=18, minor_sections=6)
        grip_link.apply_scale((1.0, 0.75, 1.0))
        grip_link.apply_translation((0.2, -2.65, z))
        parts.append(grip_link)

    # Trophy skulls and teeth on the belt.
    for x in (-1.65, 1.65):
        skull = trimesh.creation.icosphere(subdivisions=2, radius=0.28)
        skull.apply_scale((0.86, 0.70, 1.05))
        skull.apply_translation((x, -2.18, 9.35))
        jaw_plate = _box((0.36, 0.07, 0.17), (x, -2.31, 9.08))
        parts.extend([skull, jaw_plate])
        for eye_x in (-0.09, 0.09):
            socket_hole = _box((0.06, 0.055, 0.06), (x + eye_x, -2.44, 9.43))
            parts.append(socket_hole)
    for x in (-0.95, -0.45, 0.45, 0.95):
        tooth = trimesh.creation.cone(radius=0.10, height=0.55, sections=10)
        tooth.apply_transform(trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0]))
        tooth.apply_translation((x, -2.05, 9.25))
        parts.append(tooth)

    # Stitches, cuts, wrinkles, and boot straps.
    for x in np.linspace(-2.0, 2.0, 11):
        stitch = _box((0.09, 0.075, 0.32), (float(x), -2.30, 10.15))
        stitch.apply_transform(trimesh.transformations.rotation_matrix(0.28, [0, 0, 1]))
        parts.append(stitch)
    for x in (-2.25, 2.15):
        for z in (3.0, 3.45, 5.4, 6.05):
            strap = _box((1.55, 0.10, 0.10), (x, -0.98, z))
            strap.apply_transform(trimesh.transformations.rotation_matrix(0.10 if x < 0 else -0.10, [0, 0, 1]))
            parts.append(strap)
        for dx in (-0.45, 0.0, 0.45):
            toe_nail = trimesh.creation.cone(radius=0.10, height=0.34, sections=10)
            toe_nail.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
            toe_nail.apply_translation((x + dx, -1.23, 2.02))
            parts.append(toe_nail)

    # Blade edge damage and engraved marks placed on both faces of the axe head.
    for side in (-1.0, 1.0):
        for z, xoff, angle in ((19.65, 2.35, 0.25), (20.35, 2.75, -0.18), (21.25, 2.55, 0.20), (22.05, 2.05, -0.25)):
            notch = _box((0.52, 0.07, 0.12), (side * xoff, -3.05, z))
            notch.apply_transform(trimesh.transformations.rotation_matrix(side * angle, [0, 0, 1]))
            parts.append(notch)
        for z in (20.15, 20.75, 21.35, 21.95):
            scratch = _box((0.08, 0.07, 0.48), (side * 1.35, -3.06, z))
            scratch.apply_transform(trimesh.transformations.rotation_matrix(side * 0.45, [0, 0, 1]))
            parts.append(scratch)

    # Scenic base clutter gives the exported model the tabletop-store read.
    for i in range(24):
        angle = i * (2.0 * np.pi / 24.0)
        radius = 4.8 + (i % 5) * 0.55
        x = float(np.cos(angle) * radius)
        y = float(np.sin(angle) * radius)
        if i % 3 == 0:
            rock = trimesh.creation.icosphere(subdivisions=1, radius=0.23 + 0.04 * (i % 4))
            rock.apply_scale((1.4, 0.9, 0.45))
            rock.apply_translation((x, y, 1.65))
            parts.append(rock)
        else:
            shard = _box((0.65, 0.14, 0.08), (x, y, 1.55))
            shard.apply_transform(trimesh.transformations.rotation_matrix(angle, [0, 0, 1]))
            parts.append(shard)

    mesh = trimesh.util.concatenate(parts)
    mesh.merge_vertices()
    mesh.remove_unreferenced_vertices()
    try:
        mesh.fix_normals()
    except Exception:
        pass
    return mesh


def _load_store_quality_orc_asset(prompt_lower: str) -> trimesh.Trimesh | None:
    """Use the bundled high-detail orc STL library for store-level detail.

    Procedural primitives can block out a silhouette, but they will not match a
    commercial miniature sculpt. The training_data/raw_stl folder already ships
    multi-million-face orc sculpts, so for this prompt family we use one as the
    base and add a readable axe if the request asks for it.
    """
    if os.environ.get("MESHMEND_DISABLE_STORE_QUALITY_ORC_ASSET", "").strip().lower() in {"1", "true", "yes"}:
        return None
    raw_dir = default_training_data_dir() / "raw_stl"
    preferred_names = [
        "orc clappa.stl",
        "Orc warlord.stl",
        "Gobby Pistoler.stl",
        "Orc Behemoth.stl",
        "Orc War Boss Clobba.stl",
    ]
    for name in preferred_names:
        path = raw_dir / name
        if not path.exists():
            continue
        try:
            mesh = trimesh.load(path, force="mesh", process=False)
            if isinstance(mesh, trimesh.Scene):
                mesh = trimesh.util.concatenate(mesh.geometry.values())
            if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) < 100_000:
                continue
            mesh = mesh.copy()
            mesh.remove_unreferenced_vertices()
            try:
                mesh.fix_normals()
            except Exception:
                pass

            # Center around origin/bottom so normal scaling/export works.
            mesh.apply_translation(-mesh.bounds.mean(axis=0))
            mesh.apply_translation((0.0, 0.0, -float(mesh.bounds[0, 2])))
            already_has_readable_weapon = path.stem.lower() in {"orc clappa", "orc warlord", "gobby pistoler"}
            if "axe" in prompt_lower and not already_has_readable_weapon:
                mesh = trimesh.util.concatenate([mesh, _store_quality_axe_overlay(mesh)])
                mesh.merge_vertices()
                mesh.remove_unreferenced_vertices()
            return mesh
        except Exception:
            continue
    return None


def _store_quality_axe_overlay(reference_mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    bounds = np.asarray(reference_mesh.bounds, dtype=float)
    extents = np.maximum(bounds[1] - bounds[0], 1e-6)
    height = float(extents[2])
    x = float(bounds[1, 0] + extents[0] * 0.12)
    y = float(bounds[0, 1] - extents[1] * 0.10)
    z0 = float(bounds[0, 2] + height * 0.16)
    z1 = float(bounds[0, 2] + height * 0.84)
    head_z = float(bounds[0, 2] + height * 0.80)
    scale = height / 32.0

    parts: list[trimesh.Trimesh] = []
    parts.append(_cylinder_between((x, y, z0), (x, y, z1), 0.22 * scale, sections=28))
    socket = trimesh.creation.cylinder(radius=0.42 * scale, height=0.72 * scale, sections=28)
    socket.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
    socket.apply_translation((x, y, head_z))
    parts.append(socket)
    for side in (-1.0, 1.0):
        parts.append(_triangular_prism(
            (x, y - 0.15 * scale, head_z + 1.8 * scale),
            (x, y - 0.15 * scale, head_z - 1.6 * scale),
            (x + side * 3.2 * scale, y - 0.15 * scale, head_z + 0.2 * scale),
            thickness=0.28 * scale,
        ))
        for z in (head_z - 1.0 * scale, head_z, head_z + 1.0 * scale):
            notch = _box((0.52 * scale, 0.08 * scale, 0.12 * scale), (x + side * 2.4 * scale, y - 0.38 * scale, z))
            notch.apply_transform(trimesh.transformations.rotation_matrix(side * 0.24, [0, 0, 1]))
            parts.append(notch)
    for z in np.linspace(z0 + 1.5 * scale, z0 + 6.0 * scale, 7):
        wrap = trimesh.creation.torus(major_radius=0.30 * scale, minor_radius=0.035 * scale, major_sections=20, minor_sections=6)
        wrap.apply_translation((x, y, float(z)))
        parts.append(wrap)
    return trimesh.util.concatenate(parts)


def _box(extents: tuple[float, float, float], center: tuple[float, float, float]) -> trimesh.Trimesh:
    mesh = trimesh.creation.box(extents=extents)
    mesh.apply_translation(center)
    return mesh


def _cylinder_between(start, end, radius: float, sections: int = 18) -> trimesh.Trimesh:
    return trimesh.creation.cylinder(radius=radius, sections=sections, segment=np.array([start, end], dtype=float))


def _armor_details() -> list[trimesh.Trimesh]:
    parts = [
        _box((3.2, 1.0, 4.5), (0, 1.9, 13.0)),
        _box((2.0, 0.6, 1.0), (-1.2, -1.6, 5.6)),
        _box((2.0, 0.6, 1.0), (1.2, -1.6, 5.6)),
        _box((1.3, 0.45, 1.3), (-4.2, -1.0, 12.5)),
        _box((1.3, 0.45, 1.3), (4.2, -1.0, 12.5)),
    ]
    for x in (-3.1, -1.0, 1.0, 3.1):
        parts.append(_box((0.55, 0.45, 0.9), (x, -1.85, 8.4)))
    return parts


def _sword_and_shield() -> list[trimesh.Trimesh]:
    shield = trimesh.creation.cylinder(radius=2.4, height=0.45, sections=32)
    shield.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
    shield.apply_translation((-5.7, -1.0, 10.2))
    boss = trimesh.creation.icosphere(subdivisions=1, radius=0.75)
    boss.apply_translation((-5.7, -1.3, 10.2))
    return [
        _box((0.75, 0.35, 8.8), (5.8, -0.5, 11.3)),
        _box((3.0, 0.45, 0.55), (5.8, -0.5, 7.4)),
        _cylinder_between((5.8, -0.5, 6.0), (5.8, -0.5, 8.0), 0.28, sections=12),
        shield,
        boss,
    ]


def _rifle() -> list[trimesh.Trimesh]:
    return [
        _box((4.2, 0.7, 1.0), (4.8, -1.1, 10.6)),
        _cylinder_between((6.8, -1.1, 10.7), (11.0, -1.1, 10.7), 0.35, sections=14),
        _box((1.4, 0.8, 1.6), (3.0, -1.1, 10.0)),
    ]


def _bow() -> list[trimesh.Trimesh]:
    parts = [_cylinder_between((-6.2, -0.5, 7.0), (-5.3, -0.5, 15.0), 0.22, sections=10)]
    for z in (8.2, 10.0, 11.8, 13.6):
        parts.append(_cylinder_between((-6.0, -0.5, z), (-4.0, -0.5, 11.0), 0.08, sections=8))
    parts.append(_cylinder_between((-7.0, -0.45, 11.0), (-1.5, -0.45, 11.0), 0.12, sections=8))
    return parts


def _wings() -> list[trimesh.Trimesh]:
    parts = []
    for side in (-1, 1):
        root_top = (side * 2.6, 1.6, 16.0)
        root_bottom = (side * 2.9, 1.6, 10.2)
        outer_tip = (side * 9.8, 1.9, 14.0)
        parts.append(_triangular_prism(root_top, root_bottom, outer_tip, thickness=0.38))
        parts.append(_cylinder_between((side * 2.7, 1.4, 15.5), (side * 9.1, 1.8, 13.7), 0.18, sections=10))
        parts.append(_cylinder_between((side * 2.8, 1.4, 13.2), (side * 8.3, 1.8, 12.6), 0.16, sections=10))
        for offset in range(4):
            x0 = side * (3.6 + offset * 1.25)
            x1 = side * (4.8 + offset * 1.45)
            z0 = 11.2 - offset * 0.45
            parts.append(
                _triangular_prism(
                    (x0, 1.35, z0 + 2.5),
                    (x0 + side * 0.3, 1.35, z0),
                    (x1, 1.55, z0 + 0.7),
                    thickness=0.18,
                )
            )
    return parts


def _triangular_prism(p1, p2, p3, thickness: float = 0.25) -> trimesh.Trimesh:
    front = []
    back = []
    for point in (p1, p2, p3):
        x, y, z = point
        front.append([x, y - thickness / 2.0, z])
        back.append([x, y + thickness / 2.0, z])
    vertices = np.array(front + back, dtype=float)
    faces = np.array(
        [
            [0, 1, 2],
            [3, 5, 4],
            [0, 3, 1],
            [1, 3, 4],
            [1, 4, 2],
            [2, 4, 5],
            [2, 5, 0],
            [0, 5, 3],
        ],
        dtype=np.int64,
    )
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def _spikes_and_horns() -> list[trimesh.Trimesh]:
    parts = []
    for x in (-2.2, 2.2):
        horn = trimesh.creation.cone(radius=0.55, height=2.2, sections=14)
        horn.apply_translation((x, 0, 21.0))
        parts.append(horn)
    for x in (-3.0, 0, 3.0):
        spike = trimesh.creation.cone(radius=0.45, height=1.8, sections=12)
        spike.apply_translation((x, 2.0, 16.0))
        parts.append(spike)
    return parts


def _beard_and_hammer() -> list[trimesh.Trimesh]:
    beard = trimesh.creation.cone(radius=1.6, height=3.2, sections=24)
    beard.apply_transform(trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0]))
    beard.apply_translation((0, -1.0, 17.2))
    return [beard, _cylinder_between((5.5, -0.5, 7.0), (5.5, -0.5, 14.0), 0.28, sections=12), _box((3.4, 1.3, 1.5), (5.5, -0.5, 14.5))]


def _cloak_and_staff() -> list[trimesh.Trimesh]:
    orb = trimesh.creation.icosphere(subdivisions=1, radius=0.9)
    orb.apply_translation((6.0, 0.2, 17.0))
    return [_box((5.6, 0.55, 9.5), (0, 2.1, 10.4)), _cylinder_between((6.0, 0.2, 6.0), (6.0, 0.2, 16.4), 0.22, sections=12), orb]


def _segment_foreground(image) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(image, dtype=float) / 255.0
    rgb = arr[:, :, :3]
    alpha = arr[:, :, 3]
    luminance = (0.2126 * rgb[:, :, 0]) + (0.7152 * rgb[:, :, 1]) + (0.0722 * rgb[:, :, 2])

    if float(np.min(alpha)) < 0.95:
        mask = alpha > 0.25
    else:
        border = np.concatenate([rgb[0, :, :], rgb[-1, :, :], rgb[:, 0, :], rgb[:, -1, :]], axis=0)
        background = np.median(border, axis=0)
        color_distance = np.linalg.norm(rgb - background[None, None, :], axis=2)
        threshold = max(0.10, float(np.percentile(color_distance, 68)))
        mask = color_distance > threshold

        coverage = float(np.mean(mask))
        if coverage > 0.72:
            threshold = float(np.percentile(color_distance, 82))
            mask = color_distance > threshold
        elif coverage < 0.04:
            saturation = np.max(rgb, axis=2) - np.min(rgb, axis=2)
            contrast = np.abs(luminance - float(np.median(luminance)))
            mask = (saturation > float(np.percentile(saturation, 65))) | (contrast > float(np.percentile(contrast, 70)))

    mask[[0, -1], :] = False
    mask[:, [0, -1]] = False
    return mask.astype(bool), luminance


def _smooth_mask(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    smoothed = mask.astype(bool)
    for _ in range(iterations):
        neighbors = _neighbor_count(smoothed)
        smoothed = (smoothed & (neighbors >= 3)) | (neighbors >= 5)
        smoothed[[0, -1], :] = False
        smoothed[:, [0, -1]] = False
    return smoothed


def _neighbor_count(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask.astype(int), 1)
    total = np.zeros(mask.shape, dtype=int)
    for row_offset in range(3):
        for col_offset in range(3):
            total += padded[row_offset : row_offset + mask.shape[0], col_offset : col_offset + mask.shape[1]]
    return total


def _largest_component(mask: np.ndarray) -> np.ndarray:
    mask = mask.astype(bool)
    seen = np.zeros(mask.shape, dtype=bool)
    best: list[tuple[int, int]] = []
    rows, cols = mask.shape

    for row in range(rows):
        for col in range(cols):
            if not mask[row, col] or seen[row, col]:
                continue
            stack = [(row, col)]
            seen[row, col] = True
            component: list[tuple[int, int]] = []
            while stack:
                current_row, current_col = stack.pop()
                component.append((current_row, current_col))
                for next_row, next_col in (
                    (current_row - 1, current_col),
                    (current_row + 1, current_col),
                    (current_row, current_col - 1),
                    (current_row, current_col + 1),
                ):
                    if 0 <= next_row < rows and 0 <= next_col < cols and mask[next_row, next_col] and not seen[next_row, next_col]:
                        seen[next_row, next_col] = True
                        stack.append((next_row, next_col))
            if len(component) > len(best):
                best = component

    result = np.zeros(mask.shape, dtype=bool)
    for row, col in best:
        result[row, col] = True
    return result


def _interior_depth(mask: np.ndarray) -> np.ndarray:
    remaining = mask.astype(bool).copy()
    depth = np.zeros(mask.shape, dtype=float)
    level = 1.0
    while np.any(remaining) and level <= 18:
        interior_neighbors = _orthogonal_neighbor_count(remaining)
        edge = remaining & (interior_neighbors < 4)
        if not np.any(edge):
            depth[remaining] = level
            break
        depth[edge] = level
        remaining[edge] = False
        level += 1.0
    if np.max(depth) > 0:
        depth /= float(np.max(depth))
    return depth


def _orthogonal_neighbor_count(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask.astype(int), 1)
    rows, cols = mask.shape
    return (
        padded[0:rows, 1 : cols + 1]
        + padded[2 : rows + 2, 1 : cols + 1]
        + padded[1 : rows + 1, 0:cols]
        + padded[1 : rows + 1, 2 : cols + 2]
    )


def _silhouette_to_rounded_volume(mask: np.ndarray, ring_sections: int = 36) -> trimesh.Trimesh:
    """Turn a foreground silhouette into a rounded, watertight 3D volume.

    This intentionally avoids using pixel luminance as surface height. Image
    luminance created noisy, plaque-like reliefs; this builds a smooth volume
    from the silhouette profile instead, which is a better printable fallback
    when a true multi-view/image diffusion model is not available.
    """

    mask = mask.astype(bool)
    rows, cols = mask.shape
    profiles: list[tuple[float, float, float]] = []
    for row in range(rows):
        hits = np.flatnonzero(mask[row])
        if len(hits) < 2:
            continue
        left = float(hits[0])
        right = float(hits[-1])
        width = right - left + 1.0
        center = (left + right) * 0.5
        profiles.append((float(row), center, width))

    if len(profiles) < 3:
        return trimesh.creation.icosphere(subdivisions=2, radius=6.0)

    profile = np.asarray(profiles, dtype=float)
    profile[:, 1] = _smooth_1d(profile[:, 1], window=5)
    profile[:, 2] = _smooth_1d(profile[:, 2], window=5)

    max_dimension = float(max(rows, cols))
    scale = 25.0 / max_dimension
    z_values = (rows - 1.0 - profile[:, 0]) * scale
    x_centers = (profile[:, 1] - (cols / 2.0)) * scale
    x_radii = np.maximum(profile[:, 2] * scale * 0.5, 0.35)

    max_radius = float(np.max(x_radii))
    if max_radius <= 1e-9:
        return trimesh.creation.icosphere(subdivisions=2, radius=6.0)
    y_radii = np.maximum(x_radii * 0.52, 0.55)

    # Taper the first/last rings so the silhouette becomes a sealed organic
    # sculpt rather than a flat-ended tube.
    taper = np.ones(len(x_radii), dtype=float)
    end_count = min(4, max(1, len(taper) // 5))
    for index in range(end_count):
        factor = 0.35 + (0.65 * (index + 1) / end_count)
        taper[index] *= factor
        taper[-index - 1] *= factor
    x_radii *= taper
    y_radii *= taper

    vertices: list[list[float]] = []
    for z_value, x_center, x_radius, y_radius in zip(z_values, x_centers, x_radii, y_radii):
        for section in range(ring_sections):
            angle = (2.0 * np.pi * section) / ring_sections
            vertices.append(
                [
                    float(x_center + (np.cos(angle) * x_radius)),
                    float(np.sin(angle) * y_radius),
                    float(z_value),
                ]
            )

    faces: list[list[int]] = []
    ring_count = len(x_radii)
    for ring in range(ring_count - 1):
        current = ring * ring_sections
        next_ring = (ring + 1) * ring_sections
        for section in range(ring_sections):
            next_section = (section + 1) % ring_sections
            a = current + section
            b = current + next_section
            c = next_ring + section
            d = next_ring + next_section
            faces.append([a, c, b])
            faces.append([b, c, d])

    bottom_center = len(vertices)
    vertices.append([float(x_centers[0]), 0.0, float(z_values[0])])
    top_center = len(vertices)
    vertices.append([float(x_centers[-1]), 0.0, float(z_values[-1])])
    for section in range(ring_sections):
        next_section = (section + 1) % ring_sections
        faces.append([bottom_center, next_section, section])
        last = (ring_count - 1) * ring_sections
        faces.append([top_center, last + section, last + next_section])

    mesh = trimesh.Trimesh(vertices=np.asarray(vertices, dtype=float), faces=np.asarray(faces, dtype=np.int64), process=False)
    mesh.merge_vertices()
    mesh.remove_unreferenced_vertices()
    trimesh.repair.fix_normals(mesh)
    return mesh


def _smooth_1d(values: np.ndarray, window: int = 5) -> np.ndarray:
    if len(values) < 3:
        return values
    window = max(3, min(int(window), len(values)))
    if window % 2 == 0:
        window -= 1
    kernel = np.ones(window, dtype=float) / float(window)
    padded = np.pad(values, window // 2, mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _extrude_mask_to_relief(mask: np.ndarray, luminance: np.ndarray, depth: np.ndarray) -> trimesh.Trimesh:
    rows, cols = mask.shape
    max_dimension = float(max(rows, cols))
    cell = 28.0 / max_dimension
    width = cols * cell
    height = rows * cell
    vertices: list[list[float]] = []
    front_index: dict[tuple[int, int], int] = {}
    back_index: dict[tuple[int, int], int] = {}

    cell_radius = np.zeros(mask.shape, dtype=float)
    foreground_luminance = luminance[mask]
    median_luminance = float(np.median(foreground_luminance)) if len(foreground_luminance) else 0.5
    contrast = np.clip(np.abs(luminance - median_luminance) * 2.0, 0.0, 1.0)
    max_radius = max(2.5, min(width, height) * 0.28)
    cell_radius[mask] = 0.45 + (np.sqrt(depth[mask]) * max_radius) + (contrast[mask] * 0.9)

    def corner_radius(row: int, col: int) -> float:
        values = []
        for cell_row in (row - 1, row):
            for cell_col in (col - 1, col):
                if 0 <= cell_row < rows and 0 <= cell_col < cols and mask[cell_row, cell_col]:
                    values.append(cell_radius[cell_row, cell_col])
        return float(np.mean(values)) if values else 0.0

    def front_vertex(row: int, col: int) -> int:
        key = (row, col)
        if key not in front_index:
            x = (col * cell) - (width / 2.0)
            y = ((rows - row) * cell) - (height / 2.0)
            front_index[key] = len(vertices)
            vertices.append([x, y, corner_radius(row, col)])
        return front_index[key]

    def back_vertex(row: int, col: int) -> int:
        key = (row, col)
        if key not in back_index:
            x = (col * cell) - (width / 2.0)
            y = ((rows - row) * cell) - (height / 2.0)
            back_index[key] = len(vertices)
            vertices.append([x, y, -corner_radius(row, col) * 0.82])
        return back_index[key]

    faces: list[list[int]] = []
    for row in range(rows):
        for col in range(cols):
            if not mask[row, col]:
                continue
            ftl = front_vertex(row, col)
            ftr = front_vertex(row, col + 1)
            fbl = front_vertex(row + 1, col)
            fbr = front_vertex(row + 1, col + 1)
            btl = back_vertex(row, col)
            btr = back_vertex(row, col + 1)
            bbl = back_vertex(row + 1, col)
            bbr = back_vertex(row + 1, col + 1)

            faces.append([ftl, fbl, ftr])
            faces.append([ftr, fbl, fbr])
            faces.append([btl, btr, bbl])
            faces.append([btr, bbr, bbl])

            if row == 0 or not mask[row - 1, col]:
                faces.append([ftl, ftr, btl])
                faces.append([ftr, btr, btl])
            if row == rows - 1 or not mask[row + 1, col]:
                faces.append([fbl, bbl, fbr])
                faces.append([fbr, bbl, bbr])
            if col == 0 or not mask[row, col - 1]:
                faces.append([ftl, btl, fbl])
                faces.append([fbl, btl, bbl])
            if col == cols - 1 or not mask[row, col + 1]:
                faces.append([ftr, fbr, btr])
                faces.append([fbr, bbr, btr])

    mesh = trimesh.Trimesh(vertices=np.array(vertices), faces=np.array(faces, dtype=np.int64), process=False)
    mesh.merge_vertices()
    mesh.remove_unreferenced_vertices()
    return mesh


def _hosted_quality_for_detail(print_detail_um: float | None) -> str:
    if print_detail_um is not None and print_detail_um <= 50:
        return "high"
    return "standard"


def _task_has_model(task: dict) -> bool:
    return bool(task.get("model_urls") or task.get("model_url")) and str(task.get("status", "SUCCEEDED")).upper() in {
        "SUCCEEDED",
        "SUCCESS",
        "COMPLETED",
        "DONE",
    }


def _model_service_health() -> dict | None:
    try:
        return _model_service_request_json("GET", "/health")
    except Exception:
        return None


def _creation_backend_ready(health: dict | None) -> bool:
    if health is None:
        return False
    return bool(
        health.get("ready_for_studio_quality", False)
        or health.get("ready_for_experimental_high_detail", False)
    )


def _ensure_model_service_ready(progress: ProgressCallback) -> None:
    if _model_service_health() is not None:
        return
    if os.environ.get("MESHMEND_AUTO_START_MODEL_SERVICE", "1").strip().lower() not in {"1", "true", "yes"}:
        return
    global _MODEL_SERVICE_PROCESS
    service_script = Path(__file__).resolve().parent / "3dsculpter" / "model_service" / "main.py"
    if not service_script.exists():
        return
    if _MODEL_SERVICE_PROCESS is None or _MODEL_SERVICE_PROCESS.poll() is not None:
        progress(8, "Starting MeshMend local model service")
        env = os.environ.copy()
        package_parent = str(Path(__file__).resolve().parent.parent)
        env["PYTHONPATH"] = package_parent + os.pathsep + env.get("PYTHONPATH", "")
        service_python = env.get("MESHMEND_MODEL_SERVICE_PYTHON", sys.executable).strip() or sys.executable
        _MODEL_SERVICE_PROCESS = subprocess.Popen(
            [service_python, str(service_script)],
            cwd=str(service_script.parent),
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    deadline = time.time() + MODEL_SERVICE_STARTUP_SECONDS
    while time.time() < deadline:
        if _model_service_health() is not None:
            return
        time.sleep(0.5)


def _restart_experimental_sculpt_service(progress: ProgressCallback) -> None:
    if os.environ.get("MESHMEND_AUTO_START_MODEL_SERVICE", "1").strip().lower() not in {"1", "true", "yes"}:
        return
    external_store_quality = _external_store_quality_backend_configured() or _configure_bundled_no_api_external_backend()
    progress(8, "Restarting MeshMend model service with external store-quality backend" if external_store_quality else "Restarting MeshMend model service with native sculpt backend")
    global _MODEL_SERVICE_PROCESS
    cli_script = Path(__file__).resolve().parent / "cli.py"
    if cli_script.exists():
        try:
            subprocess.run(
                [sys.executable, str(cli_script), "--stop-model-service"],
                cwd=str(cli_script.parent),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=20,
            )
        except Exception:
            pass
    if _MODEL_SERVICE_PROCESS is not None and _MODEL_SERVICE_PROCESS.poll() is None:
        try:
            _MODEL_SERVICE_PROCESS.terminate()
        except Exception:
            pass
    _MODEL_SERVICE_PROCESS = None
    if external_store_quality:
        os.environ["MESHMEND_PRODUCTION_ENGINE"] = "external"
        os.environ["MESHMEND_EXTERNAL_STORE_QUALITY_CERTIFIED"] = "1"
    else:
        os.environ["MESHMEND_PRODUCTION_ENGINE"] = "meshmend_sculpt"
        os.environ["MESHMEND_ALLOW_EXPERIMENTAL_SCULPT_HIGH_DETAIL"] = "1"
        os.environ["MESHMEND_ALLOW_UNCERTIFIED_STORE_QUALITY_OUTPUT"] = "1"
    os.environ["MESHMEND_MODEL_WORKER_PYTHON"] = sys.executable
    os.environ["MESHMEND_MODEL_SERVICE_PYTHON"] = sys.executable
    _ensure_model_service_ready(progress)


def _external_store_quality_backend_configured() -> bool:
    engine = os.environ.get("MESHMEND_PRODUCTION_ENGINE", "").strip().lower()
    certified = os.environ.get("MESHMEND_EXTERNAL_STORE_QUALITY_CERTIFIED", "0").strip().lower() in {"1", "true", "yes", "on"}
    has_command = bool(
        os.environ.get("MESHMEND_PRODUCTION_TEXT_TO_3D_COMMAND", "").strip()
        or os.environ.get("MESHMEND_PRODUCTION_IMAGE_TO_3D_COMMAND", "").strip()
    )
    if engine in {"external", "command"} and certified and has_command:
        return True
    # The new HTTP adapter is configured by provider env vars plus the production
    # command. Do not let the GUI overwrite that with the experimental native
    # backend, because that path is intentionally blocked for store-quality jobs.
    return certified and has_command and bool(os.environ.get("MESHMEND_HTTP_GENERATOR_SUBMIT_URL", "").strip())


def _configure_bundled_no_api_external_backend() -> bool:
    """Prefer the bundled no-API external runner for store-quality jobs.

    The previous GUI path kept restarting the service as `meshmend_sculpt`, which
    is intentionally rejected for store-quality requests. Use the local external
    Hunyuan-backed adapter by default so /health reports `external` and the
    actual worker can surface install/model-quality failures instead of a 503
    config rejection. Users can still force native sculpt for debugging.
    """
    if os.environ.get("MESHMEND_FORCE_NATIVE_SCULPT_BACKEND", "0").strip().lower() in {"1", "true", "yes", "on"}:
        return False
    runner = Path(__file__).resolve().parent / "external_local_store_quality_generator.py"
    if not runner.exists():
        return False
    os.environ["MESHMEND_PRODUCTION_ENGINE"] = "external"
    os.environ["MESHMEND_EXTERNAL_STORE_QUALITY_CERTIFIED"] = "1"
    os.environ.setdefault("MESHMEND_ALLOW_HUNYUAN_STORE_QUALITY", "1")
    os.environ.setdefault("MESHMEND_ALLOW_LOCAL_QUALITY_SCORE_ESTIMATES", "1")
    os.environ["MESHMEND_PRODUCTION_TEXT_TO_3D_COMMAND"] = (
        f'"{sys.executable}" "{runner}" --input {{input_json}} --prompt {{prompt_path}} '
        "--output-dir {output_dir} --quality {quality} --target-polycount {target_polycount}"
    )
    os.environ["MESHMEND_PRODUCTION_IMAGE_TO_3D_COMMAND"] = (
        f'"{sys.executable}" "{runner}" --input {{input_json}} --prompt {{prompt_path}} --image {{image_path}} '
        "--output-dir {output_dir} --quality {quality} --target-polycount {target_polycount}"
    )
    return True


def _model_service_request_json(method: str, path: str, payload: dict | None = None) -> dict:
    url = f"{HOSTED_CREATION_URL}{path}"
    body = json.dumps(payload).encode("utf-8") if payload is not None else None
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    api_key = os.environ.get("MESHMEND_MODEL_SERVICE_API_KEY", "").strip()
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(url, data=body, method=method, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"model service HTTP {exc.code}: {details}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"model service connection failed at {HOSTED_CREATION_URL}: {exc}") from exc


def _image_data_uri(image_path: Path) -> str:
    mime = mimetypes.guess_type(str(image_path))[0] or "image/png"
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _download_model_url(url: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme == "file":
        source = Path(urllib.request.url2pathname(parsed.path))
        output_path.write_bytes(source.read_bytes())
        return
    request = urllib.request.Request(url, headers={"User-Agent": "MeshMend/1.0"})
    with urllib.request.urlopen(request, timeout=300) as response:
        output_path.write_bytes(response.read())


def get_sculptor_foundation() -> SculptorFoundation:
    return SculptorFoundation(Path(__file__).resolve().parent / "3dsculpter")
