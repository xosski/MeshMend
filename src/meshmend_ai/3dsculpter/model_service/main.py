from __future__ import annotations

import json
import logging
import os
import platform
import shutil
import shlex
import socket
import subprocess
import sys
import threading
import time
import traceback
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from starlette.background import BackgroundTask
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel


SERVICE_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = SERVICE_ROOT.parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))
SERVICE_BUILD_ID = "meshmend-model-service-base-form-v4"
TASKS_DIR = Path(os.environ.get("MESHMEND_MODEL_SERVICE_TASKS_DIR", SERVICE_ROOT / "tasks"))
OUTPUTS_DIR = Path(os.environ.get("MESHMEND_MODEL_SERVICE_OUTPUTS_DIR", SERVICE_ROOT / "outputs"))
LOGS_DIR = Path(os.environ.get("MESHMEND_BACKEND_LOG_DIR", WORKSPACE_ROOT / "logs"))
BACKEND_LOG_FILE = LOGS_DIR / "meshmend_backend.log"
SERVICE_PORT = os.environ.get("MESHMEND_MODEL_SERVICE_PORT", "8090").strip() or "8090"
PUBLIC_BASE_URL = os.environ.get("MESHMEND_MODEL_SERVICE_PUBLIC_URL", f"http://127.0.0.1:{SERVICE_PORT}").rstrip("/")
MODEL_WORKER_PYTHON = os.environ.get("MESHMEND_MODEL_WORKER_PYTHON", sys.executable).strip() or sys.executable
DEFAULT_WORKER_COMMAND = f'"{MODEL_WORKER_PYTHON}" "{SERVICE_ROOT / "production_worker.py"}" --input "{{input_json}}" --output-dir "{{output_dir}}"'
TEXT_TO_3D_COMMAND = os.environ.get("MESHMEND_TEXT_TO_3D_COMMAND", DEFAULT_WORKER_COMMAND).strip()
IMAGE_TO_3D_COMMAND = os.environ.get("MESHMEND_IMAGE_TO_3D_COMMAND", DEFAULT_WORKER_COMMAND).strip()

TASKS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)


def _configure_logging() -> logging.Logger:
    logger = logging.getLogger("meshmend_backend")
    logger.setLevel(logging.INFO)
    if not any(isinstance(handler, logging.FileHandler) and getattr(handler, "baseFilename", "") == str(BACKEND_LOG_FILE) for handler in logger.handlers):
        handler = logging.FileHandler(BACKEND_LOG_FILE, encoding="utf-8")
        handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(name)s %(message)s'))
        logger.addHandler(handler)
    return logger


LOGGER = _configure_logging()

app = FastAPI(title="MeshMend Independent 3D Model Service", version="0.1.0")


class TextTo3DRequest(BaseModel):
    prompt: str
    quality: str = "high"
    studio_mode: bool = True
    debug_mode: bool = False
    target_formats: list[str] = ["stl", "glb", "obj"]
    # Keep the service default memory-safe. Local repair/detail passes can grow
    # faces in 4x jumps, so million-face requests can become 3-5M face meshes and
    # freeze consumer PCs. Certified external render farms can still request more
    # via MESHMEND_HOSTED_TARGET_POLYCOUNT or direct payload overrides.
    target_polycount: int = 180_000
    scale_mm: float | None = None
    workflow: str = "text_to_3d"
    product: str = "meshmend"


class ImageTo3DRequest(TextTo3DRequest):
    image_data_uri: str
    workflow: str = "image_to_3d"


class GeneratePartRequest(BaseModel):
    category: str
    prompt: str = "sci-fi heavy infantry"
    count: int = 3
    scale_mm: float = 32.0
    target_faces: int = 40_000


class AssembleMiniatureRequest(BaseModel):
    prompt: str = "sci-fi heavy infantry with rifle and backpack"
    scale_mm: float = 32.0
    target_faces: int = 90_000
    output_format: str = "stl"
    candidates_per_category: int = 3


@dataclass
class TaskRecord:
    id: str
    workflow: str
    status: str = "PENDING"
    progress: int = 0
    stage: str = "queued"
    message: str = "Queued"
    last_progress_at: float = field(default_factory=time.time)
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    prompt: str = ""
    model_urls: dict[str, str] = field(default_factory=dict)
    thumbnail_url: str | None = None
    error: str | None = None
    consumed_credits: int = 0


_tasks: dict[str, TaskRecord] = {}
_served_files: dict[str, Path] = {}
_lock = threading.Lock()
_hunyuan_import_cache: dict[str, Any] | None = None
_hunyuan_import_cache_at = 0.0


@app.get("/health")
async def health():
    production_text_command = os.environ.get("MESHMEND_PRODUCTION_TEXT_TO_3D_COMMAND", "").strip()
    production_image_command = os.environ.get("MESHMEND_PRODUCTION_IMAGE_TO_3D_COMMAND", "").strip()
    production_engine = os.environ.get("MESHMEND_PRODUCTION_ENGINE", "meshmend_native")
    external_generator_configured = bool(os.environ.get("MESHMEND_EXTERNAL_GENERATOR_COMMAND", "").strip())
    hunyuan_mode = production_engine.strip().lower() in {"free_local", "free_local_hunyuan", "hunyuan", "hunyuan3d"}
    hunyuan_import = _hunyuan_import_status() if hunyuan_mode else None
    runner_ready = _production_runner_ready("text_to_3d") or _production_runner_ready("image_to_3d")
    studio_ready = _studio_quality_runner_ready("text_to_3d") or _studio_quality_runner_ready("image_to_3d")
    experimental_high_detail_ready = _experimental_high_detail_runner_ready("text_to_3d") or _experimental_high_detail_runner_ready("image_to_3d")
    dependency_summary = _dependency_health_summary()
    return {
        "status": "healthy",
        "backend_running": True,
        "model_quality_acceptable": bool(studio_ready),
        "model_quality_note": "backend health is separate from miniature quality; studio output is accepted only when the worker returns a passing strict quality report",
        "memory_safety": _memory_safety_status(),
        "service_build_id": SERVICE_BUILD_ID,
        "service_root": str(SERVICE_ROOT),
        "workspace_root": str(WORKSPACE_ROOT),
        "service_python": sys.executable,
        "log_file": str(BACKEND_LOG_FILE),
        "text_to_3d_configured": bool(TEXT_TO_3D_COMMAND),
        "image_to_3d_configured": bool(IMAGE_TO_3D_COMMAND),
        "production_text_to_3d_configured": bool(production_text_command),
        "production_image_to_3d_configured": bool(production_image_command),
        "production_engine": production_engine,
        "external_generator_configured": external_generator_configured,
        "ready_for_generation": runner_ready,
        "ready_for_studio_quality": studio_ready,
        "ready_for_experimental_high_detail": experimental_high_detail_ready,
        "capability_tier": _capability_tier(production_engine),
        "store_quality_reason": _store_quality_reason(production_engine) if not studio_ready else "strict studio-quality gate configured; outputs must return studio_quality_certified=true",
        "model_worker_python": MODEL_WORKER_PYTHON,
        "hunyuan3d_path": os.environ.get("MESHMEND_HUNYUAN3D_PATH", ""),
        "hunyuan3d_model": os.environ.get("MESHMEND_HUNYUAN3D_MODEL", ""),
        "hunyuan3d_subfolder": os.environ.get("MESHMEND_HUNYUAN3D_SUBFOLDER", ""),
        "hunyuan_import": hunyuan_import,
        "dependency_summary": dependency_summary,
    }


@app.get("/diagnostics")
async def diagnostics():
    """Deep backend diagnostics: dependencies, paths, ports, permissions, and recent failures."""
    report = _build_backend_diagnostics()
    LOGGER.info(json.dumps({"event": "diagnostics_requested", "status": report.get("status")}, default=str))
    return report


@app.get("/test-mesh")
async def test_mesh(format: str = "stl"):
    """Generate a known simple watertight mesh to prove backend/export plumbing works."""
    fmt = _safe_export_format(format)
    task_id = "testmesh_" + uuid.uuid4().hex[:12]
    output_dir = OUTPUTS_DIR / task_id
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        mesh = _create_known_test_mesh()
        output_path = output_dir / f"known_simple_mesh.{fmt}"
        _export_mesh(mesh, output_path)
        info = _mesh_summary(mesh)
        LOGGER.info(json.dumps({"event": "test_mesh_generated", "task_id": task_id, "mesh": info}, default=str))
        return {
            "status": "ok",
            "purpose": "backend/export smoke test only; not a miniature quality test",
            "model_urls": {fmt: f"{PUBLIC_BASE_URL}/v1/files/{task_file_name(output_path)}"},
            "mesh_info": info,
        }
    except Exception as exc:
        LOGGER.exception("test_mesh_failed")
        raise HTTPException(status_code=500, detail={"error": str(exc), "traceback": traceback.format_exc()})


@app.post("/generate-part")
async def generate_part(request: GeneratePartRequest):
    """Generate validated modular candidates for one part category using the fallback provider."""
    try:
        from meshmend.studio.assets import PartCategory
        from meshmend.studio.pipeline import StudioMiniatureSpec
        from meshmend.studio.staged_pipeline import ProceduralMiniaturePartProvider

        category = PartCategory(str(request.category))
        spec = StudioMiniatureSpec.from_prompt(request.prompt, scale_mm=request.scale_mm, target_faces=request.target_faces)
        output_dir = OUTPUTS_DIR / ("part_" + uuid.uuid4().hex[:12])
        output_dir.mkdir(parents=True, exist_ok=True)
        provider = ProceduralMiniaturePartProvider()
        concept = {"prompt": spec.prompt, "scale_mm": spec.scale_mm, "style": spec.style, "weapon": spec.weapon}
        parts = provider.generate_candidates(category, concept, max(1, min(int(request.count), 6)), spec.scale_mm)
        response_parts = []
        for part in parts:
            bundle = part.export_bundle(output_dir)
            mesh_path = Path(bundle["mesh_file"])
            response_parts.append(
                {
                    "part_id": part.part_id,
                    "category": category.value,
                    "mesh_url": f"{PUBLIC_BASE_URL}/v1/files/{task_file_name(mesh_path)}" if mesh_path.exists() else None,
                    "bundle_dir": str(bundle),
                    "anchors": [anchor.to_dict() for anchor in part.anchors],
                    "sockets": [socket_.to_dict() for socket_ in part.sockets],
                    "scale_mm": part.scale_mm,
                    "symmetry": part.symmetry,
                    "cleanup_report": part.cleanup_report.to_dict() if part.cleanup_report else None,
                }
            )
        LOGGER.info(json.dumps({"event": "generate_part", "category": category.value, "count": len(response_parts)}, default=str))
        return {"status": "ok", "quality_tier": "procedural_modular_part", "studio_quality_certified": False, "parts": response_parts}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        LOGGER.exception("generate_part_failed")
        raise HTTPException(status_code=500, detail={"error": str(exc), "traceback": traceback.format_exc()})


@app.post("/assemble-miniature")
async def assemble_miniature(request: AssembleMiniatureRequest):
    """Assemble a valid fallback miniature from procedural kitbash modules."""
    try:
        result = _generate_modular_fallback(
            prompt=request.prompt,
            scale_mm=request.scale_mm,
            target_faces=request.target_faces,
            output_format=request.output_format,
            candidates_per_category=request.candidates_per_category,
            reason="explicit_assemble_miniature_endpoint",
        )
        LOGGER.info(json.dumps({"event": "assemble_miniature", "result": result.get("mesh_info")}, default=str))
        return result
    except Exception as exc:
        LOGGER.exception("assemble_miniature_failed")
        raise HTTPException(status_code=500, detail={"error": str(exc), "traceback": traceback.format_exc()})


@app.post("/v1/text-to-3d")
async def text_to_3d(request: TextTo3DRequest):
    payload = _request_dict(request)
    if not TEXT_TO_3D_COMMAND:
        if _modular_fallback_enabled():
            payload["_meshmend_force_modular_fallback_reason"] = "MESHMEND_TEXT_TO_3D_COMMAND is not configured"
            return _enqueue("text_to_3d", payload)
        raise HTTPException(
            status_code=503,
            detail="MESHMEND_TEXT_TO_3D_COMMAND is not configured on the independent model service.",
        )
    if not _production_runner_ready("text_to_3d"):
        if _modular_fallback_enabled():
            payload["_meshmend_force_modular_fallback_reason"] = "No production text-to-3D runner is configured"
            return _enqueue("text_to_3d", payload)
        raise HTTPException(
            status_code=503,
            detail="No production text-to-3D runner is configured. Set MESHMEND_PRODUCTION_TEXT_TO_3D_COMMAND, or explicitly set MESHMEND_PRODUCTION_ENGINE=legacy_sculptor for draft-only fallback.",
        )
    if _store_quality_requested(payload) and not _can_accept_high_detail_request("text_to_3d"):
        if _modular_fallback_enabled():
            payload["_meshmend_force_modular_fallback_reason"] = _store_quality_reason(os.environ.get("MESHMEND_PRODUCTION_ENGINE", "meshmend_native"))
            return _enqueue("text_to_3d", payload)
        raise HTTPException(status_code=503, detail=_store_quality_reason(os.environ.get("MESHMEND_PRODUCTION_ENGINE", "meshmend_native")))
    return _enqueue("text_to_3d", payload)


@app.post("/v1/image-to-3d")
async def image_to_3d(request: ImageTo3DRequest):
    payload = _request_dict(request)
    if not IMAGE_TO_3D_COMMAND:
        if _modular_fallback_enabled():
            payload["_meshmend_force_modular_fallback_reason"] = "MESHMEND_IMAGE_TO_3D_COMMAND is not configured"
            return _enqueue("image_to_3d", payload)
        raise HTTPException(
            status_code=503,
            detail="MESHMEND_IMAGE_TO_3D_COMMAND is not configured on the independent model service.",
        )
    if not _production_runner_ready("image_to_3d"):
        if _modular_fallback_enabled():
            payload["_meshmend_force_modular_fallback_reason"] = "No production image-to-3D runner is configured"
            return _enqueue("image_to_3d", payload)
        raise HTTPException(
            status_code=503,
            detail="No production image-to-3D runner is configured. Set MESHMEND_PRODUCTION_IMAGE_TO_3D_COMMAND, or explicitly set MESHMEND_PRODUCTION_ENGINE=legacy_sculptor for draft-only fallback.",
        )
    if _store_quality_requested(payload) and not _can_accept_high_detail_request("image_to_3d"):
        if _modular_fallback_enabled():
            payload["_meshmend_force_modular_fallback_reason"] = _store_quality_reason(os.environ.get("MESHMEND_PRODUCTION_ENGINE", "meshmend_native"))
            return _enqueue("image_to_3d", payload)
        raise HTTPException(status_code=503, detail=_store_quality_reason(os.environ.get("MESHMEND_PRODUCTION_ENGINE", "meshmend_native")))
    return _enqueue("image_to_3d", payload)


@app.get("/v1/tasks/{task_id}")
async def get_task(task_id: str):
    with _lock:
        task = _tasks.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    return asdict(task)


@app.get("/v1/files/{filename}")
async def get_file(filename: str):
    safe_name = Path(filename).name
    path = OUTPUTS_DIR / safe_name
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    background = None
    if _cleanup_outputs_after_download_enabled():
        background = BackgroundTask(_cleanup_downloaded_file_artifacts, safe_name)
    return FileResponse(path, background=background)


@app.post("/v1/tasks/{task_id}/cleanup")
async def cleanup_task_outputs(task_id: str):
    """Remove model-service task/output artifacts after the client has saved the result."""
    safe_task_id = Path(task_id).name
    if not safe_task_id or safe_task_id != task_id:
        raise HTTPException(status_code=400, detail="Invalid task id")
    removed = _cleanup_task_artifacts(safe_task_id, remove_flattened=True)
    LOGGER.info(json.dumps({"event": "task_artifacts_cleanup_requested", "task_id": safe_task_id, "removed": removed}, default=str))
    return {"status": "ok", "task_id": safe_task_id, "removed": removed}


def _enqueue(workflow: str, payload: dict[str, Any]) -> dict[str, str]:
    task_id = uuid.uuid4().hex
    task = TaskRecord(id=task_id, workflow=workflow, prompt=str(payload.get("prompt") or ""))
    with _lock:
        _tasks[task_id] = task
    thread = threading.Thread(target=_run_task, args=(task_id, payload), daemon=True)
    thread.start()
    return {"task_id": task_id, "status": "PENDING"}


def _production_runner_ready(workflow: str) -> bool:
    engine = os.environ.get("MESHMEND_PRODUCTION_ENGINE", "meshmend_native").strip().lower()
    if engine in {"meshmend_native", "native", "embedded_native"}:
        try:
            from meshmend.studio import StagedMiniaturePipeline, StudioMiniatureSpec  # noqa: F401

            return True
        except Exception:
            return False
    if engine in {"meshmend_sculpt", "native_sculpt", "sculpt"}:
        try:
            import native_sculpt_backend  # noqa: F401

            return True
        except Exception:
            return False
    if engine in {"free_local", "free_local_hunyuan", "hunyuan", "hunyuan3d"}:
        return bool(_hunyuan_import_status().get("ok", False))
    if engine in {"legacy_sculptor", "embedded"}:
        return False
    if engine not in {"external", "command"}:
        return False
    command_var = "MESHMEND_PRODUCTION_IMAGE_TO_3D_COMMAND" if workflow == "image_to_3d" else "MESHMEND_PRODUCTION_TEXT_TO_3D_COMMAND"
    return bool(os.environ.get(command_var, "").strip())


def _studio_quality_runner_ready(workflow: str) -> bool:
    engine = os.environ.get("MESHMEND_PRODUCTION_ENGINE", "meshmend_native").strip().lower()
    if engine in {"free_local", "free_local_hunyuan", "hunyuan", "hunyuan3d"}:
        if not _production_runner_ready(workflow):
            return False
        return _strict_studio_gate_enforced()
    if engine not in {"external", "command"}:
        return False
    if os.environ.get("MESHMEND_EXTERNAL_STORE_QUALITY_CERTIFIED", "0").strip().lower() not in {"1", "true", "yes", "on"}:
        return False
    command_var = "MESHMEND_PRODUCTION_IMAGE_TO_3D_COMMAND" if workflow == "image_to_3d" else "MESHMEND_PRODUCTION_TEXT_TO_3D_COMMAND"
    command = os.environ.get(command_var, "").strip()
    if not command:
        return False
    if "external_store_quality_generator.py" in command.replace("\\", "/"):
        return bool(os.environ.get("MESHMEND_EXTERNAL_GENERATOR_COMMAND", "").strip())
    return True


def _experimental_high_detail_runner_ready(workflow: str) -> bool:
    engine = os.environ.get("MESHMEND_PRODUCTION_ENGINE", "meshmend_native").strip().lower()
    if engine not in {"meshmend_sculpt", "native_sculpt", "sculpt"}:
        return False
    if os.environ.get("MESHMEND_ALLOW_EXPERIMENTAL_SCULPT_HIGH_DETAIL", "0").strip().lower() not in {"1", "true", "yes", "on"}:
        return False
    return _production_runner_ready(workflow)


def _can_accept_high_detail_request(workflow: str) -> bool:
    if _studio_quality_runner_ready(workflow):
        return True
    # Experimental native sculpt may be useful for internal A/B testing, but it
    # must not satisfy store/studio-quality jobs by default; otherwise users get
    # the same uncertified/generic native mesh when they explicitly asked for a
    # production miniature. Keep an intentionally loud escape hatch for local
    # debugging only.
    allow_uncertified = os.environ.get("MESHMEND_ALLOW_UNCERTIFIED_STORE_QUALITY_OUTPUT", "0").strip().lower() in {"1", "true", "yes", "on"}
    return allow_uncertified and _experimental_high_detail_runner_ready(workflow)


def _strict_studio_gate_enforced() -> bool:
    """Return true only when studio requests fail closed instead of exporting best effort meshes."""
    if os.environ.get("MESHMEND_DISABLE_STORE_QUALITY_GATE", "0").strip().lower() in {"1", "true", "yes", "on"}:
        return False
    if os.environ.get("MESHMEND_ALLOW_BEST_EFFORT_EXPORT", "0").strip().lower() in {"1", "true", "yes", "on"}:
        return False
    return True


def _store_quality_requested(payload: dict[str, Any]) -> bool:
    quality = str(payload.get("quality") or "standard").lower()
    prompt = str(payload.get("prompt") or "").lower()
    return quality == "high" or any(
        term in prompt
        for term in (
            "8k", "8 k", "studio", "studio quality", "studio-quality", "studio level", "studio-level",
            "production", "display quality", "maximum detail", "store quality", "store-quality", "store level",
            "store-level", "intricate",
        )
    )


def _modular_fallback_enabled() -> bool:
    # Do not let an old shell/service environment silently resurrect the
    # procedural placeholder miniature path. It was the source of repeated
    # generic mannequin outputs. Keep any future fallback behind a deliberately
    # named debug-only double opt-in so production/UI requests fail loudly.
    return (
        os.environ.get("MESHMEND_ENABLE_MODULAR_FALLBACK", "0").strip().lower() in {"1", "true", "yes", "on"}
        and os.environ.get("MESHMEND_ALLOW_PROCEDURAL_PLACEHOLDER_FALLBACK_FOR_DEBUG", "0").strip().lower() in {"1", "true", "yes", "on"}
    )


def _capability_tier(engine: str) -> str:
    engine = engine.strip().lower()
    if engine in {"external", "command"} and os.environ.get("MESHMEND_EXTERNAL_STORE_QUALITY_CERTIFIED", "0").strip().lower() in {"1", "true", "yes", "on"}:
        return "certified_store_quality_external"
    if engine in {"meshmend_native", "native", "embedded_native"}:
        return "procedural_printable_draft"
    if engine in {"meshmend_sculpt", "native_sculpt", "sculpt"}:
        return "experimental_image_conditioned_native_sculpt"
    if engine in {"free_local", "free_local_hunyuan", "hunyuan", "hunyuan3d"}:
        return "strict_gated_local_hunyuan" if _strict_studio_gate_enforced() else "experimental_image_reconstruction"
    if engine in {"external", "command"}:
        return "external_uncertified"
    return "unconfigured"


def _store_quality_reason(engine: str) -> str:
    if engine.strip().lower() in {"free_local", "free_local_hunyuan", "hunyuan", "hunyuan3d"}:
        if not _strict_studio_gate_enforced():
            return (
                "Local Hunyuan is configured, but strict store/studio-quality enforcement is disabled. "
                "Unset MESHMEND_DISABLE_STORE_QUALITY_GATE and MESHMEND_ALLOW_BEST_EFFORT_EXPORT so studio requests fail unless the postprocess report is production_ready."
            )
        return "Local Hunyuan is not ready; verify Hunyuan3D imports successfully in the configured worker Python."
    if engine.strip().lower() in {"meshmend_sculpt", "native_sculpt", "sculpt"}:
        return (
            "Configured engine 'meshmend_sculpt' is an experimental native MiniatureSpec -> rig -> sculpt backend. "
            "It is not certified store/studio-quality output and should not be returned for production miniature requests. "
            "Configure a certified external production runner and set MESHMEND_PRODUCTION_ENGINE=external plus "
            "MESHMEND_EXTERNAL_STORE_QUALITY_CERTIFIED=1. For local debugging only, set both "
            "MESHMEND_ALLOW_EXPERIMENTAL_SCULPT_HIGH_DETAIL=1 and MESHMEND_ALLOW_UNCERTIFIED_STORE_QUALITY_OUTPUT=1."
        )
    return (
        f"Configured engine '{engine or 'meshmend_native'}' is {_capability_tier(engine)} and is not certified for store/studio-quality 8K miniature sculpt generation. "
        "Configure an external production runner and set MESHMEND_EXTERNAL_STORE_QUALITY_CERTIFIED=1, or submit a standard/draft-quality request."
    )


def _hunyuan_import_status() -> dict[str, Any]:
    global _hunyuan_import_cache, _hunyuan_import_cache_at
    cache_ttl = float(os.environ.get("MESHMEND_HUNYUAN_IMPORT_CHECK_CACHE_SECONDS", "300"))
    now = time.time()
    if _hunyuan_import_cache is not None and now - _hunyuan_import_cache_at < cache_ttl:
        cached = dict(_hunyuan_import_cache)
        cached["cached"] = True
        return cached
    script = (
        "import sys\n"
        "try:\n"
        "    from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline\n"
        "    print('ok')\n"
        "except Exception as exc:\n"
        "    print(repr(exc))\n"
        "    raise SystemExit(1)\n"
    )
    env = os.environ.copy()
    hunyuan_path = env.get("MESHMEND_HUNYUAN3D_PATH", "").strip()
    if hunyuan_path:
        env["PYTHONPATH"] = hunyuan_path + os.pathsep + env.get("PYTHONPATH", "")
    try:
        completed = subprocess.run(
            [MODEL_WORKER_PYTHON, "-c", script],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            timeout=float(os.environ.get("MESHMEND_HUNYUAN_IMPORT_CHECK_TIMEOUT_SECONDS", "60")),
        )
    except Exception as exc:
        result = {
            "ok": False,
            "worker_python": MODEL_WORKER_PYTHON,
            "error": str(exc),
        }
        _hunyuan_import_cache = result
        _hunyuan_import_cache_at = now - max(0.0, cache_ttl - float(os.environ.get("MESHMEND_HUNYUAN_FAILED_IMPORT_CACHE_SECONDS", "15")))
        return result
    output = (completed.stdout or completed.stderr or "").strip()
    result = {
        "ok": completed.returncode == 0,
        "worker_python": MODEL_WORKER_PYTHON,
        "output": output,
    }
    _hunyuan_import_cache = result
    _hunyuan_import_cache_at = now
    return result


def _run_task(task_id: str, payload: dict[str, Any]) -> None:
    workflow = str(payload.get("workflow") or "text_to_3d")
    command_template = IMAGE_TO_3D_COMMAND if workflow == "image_to_3d" else TEXT_TO_3D_COMMAND
    task_dir = TASKS_DIR / task_id
    output_dir = OUTPUTS_DIR / task_id
    task_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    input_json = task_dir / "input.json"
    result_json = output_dir / "result.json"
    input_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    LOGGER.info(json.dumps({"event": "task_started", "task_id": task_id, "workflow": workflow, "prompt_preview": str(payload.get("prompt") or "")[:160]}, default=str))
    _update(task_id, status="IN_PROGRESS", progress=5, stage="starting", message="Starting production worker", started_at=time.time())
    forced_fallback_reason = str(payload.get("_meshmend_force_modular_fallback_reason") or "").strip()
    if forced_fallback_reason:
        error = (
            "GENERATION FAILED: archetype generator failed. "
            "Failing function name: _generate_modular_fallback. "
            "Procedural mannequin/placeholder fallback is disabled. Original reason: " + forced_fallback_reason
        )
        result_json.write_text(json.dumps({"error": error, "fallback_used": False}, indent=2), encoding="utf-8")
        _update(task_id, status="FAILED", progress=100, stage="failed", message=error, finished_at=time.time(), error=error)
        LOGGER.error(json.dumps({"event": "task_forced_fallback_blocked", "task_id": task_id, "reason": forced_fallback_reason}, default=str))
        return
    try:
        command = command_template.format(input_json=str(input_json), output_dir=str(output_dir), task_id=task_id)
        command_args = _worker_command_args(command, input_json=input_json, output_dir=output_dir)
        (output_dir / "worker_command.json").write_text(
            json.dumps(
                {
                    "command_template": command_template,
                    "formatted_command": command,
                    "command_args": command_args,
                    "model_worker_python": MODEL_WORKER_PYTHON,
                    "worker_python_exists": Path(MODEL_WORKER_PYTHON).exists(),
                    "service_python": sys.executable,
                    "hunyuan3d_path": os.environ.get("MESHMEND_HUNYUAN3D_PATH", ""),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        completed = _run_worker_with_progress(task_id, command_args, output_dir)
        if completed.returncode != 0:
            worker_error = _worker_error_message(result_json, completed.stdout, completed.stderr)
            raise RuntimeError(worker_error or f"model command exited {completed.returncode}")

        result = _load_worker_result(result_json, completed.stdout, output_dir)

        validated_base_form = _result_validated_base_form(result, output_dir)
        if _store_quality_requested(payload) and not _result_studio_certified(result) and not validated_base_form:
            certification = _result_studio_certification_summary(result)
            raise RuntimeError(
                "Studio-quality request completed without certification. "
                "The mesh was not released because strict studio requests must return studio_quality_certified=true "
                f"or mesh_info.production_ready=true. Certification summary: {json.dumps(certification, default=str)}"
            )

        progress_result = _read_worker_progress(output_dir)
        if progress_result:
            _apply_worker_progress(task_id, progress_result)
        model_urls = _result_model_urls(result, output_dir)
        if not model_urls:
            raise RuntimeError("model worker completed but did not produce a supported model file or model_urls")
        _update(
            task_id,
            status="SUCCEEDED",
            progress=100,
            stage="complete",
            message="Validated base form ready; decorative detail is disabled" if validated_base_form and not _result_studio_certified(result) else "Production model ready",
            finished_at=time.time(),
            model_urls=model_urls,
            thumbnail_url=result.get("thumbnail_url"),
            consumed_credits=int(result.get("consumed_credits") or (30 if workflow == "image_to_3d" else 20)),
        )
        LOGGER.info(json.dumps({"event": "task_succeeded", "task_id": task_id, "model_urls": model_urls}, default=str))
    except Exception as exc:
        LOGGER.exception("task_failed_before_optional_fallback")
        if _modular_fallback_enabled():
            try:
                _update(task_id, status="IN_PROGRESS", progress=85, stage="modular_fallback", message="AI/backend generation failed; assembling validated procedural fallback")
                fallback_result = _generate_modular_fallback(
                    prompt=str(payload.get("prompt") or "fallback miniature"),
                    scale_mm=float(payload.get("scale_mm") or 32.0),
                    target_faces=int(payload.get("target_polycount") or payload.get("target_faces") or 90_000),
                    output_format="stl",
                    candidates_per_category=3,
                    output_dir=output_dir,
                    reason=str(exc),
                )
                result_json.write_text(json.dumps(fallback_result, indent=2, default=str), encoding="utf-8")
                model_urls = {str(k): str(v) for k, v in fallback_result.get("model_urls", {}).items()}
                if model_urls:
                    _update(
                        task_id,
                        status="SUCCEEDED",
                        progress=100,
                        stage="fallback_complete",
                        message="Valid procedural fallback miniature ready; not studio-quality certified",
                        finished_at=time.time(),
                        model_urls=model_urls,
                        thumbnail_url=None,
                        consumed_credits=0,
                    )
                    LOGGER.info(json.dumps({"event": "task_fallback_succeeded", "task_id": task_id, "reason": str(exc), "model_urls": model_urls}, default=str))
                    return
            except Exception:
                LOGGER.exception("task_modular_fallback_failed")
        try:
            (output_dir / "worker_error.txt").write_text(str(exc), encoding="utf-8")
            (output_dir / "service_diagnostics.json").write_text(
                json.dumps(build_service_diagnostics(exc, output_dir, result_json), indent=2, default=str),
                encoding="utf-8",
            )
            (output_dir / "service_traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
        except Exception:
            pass
        _update(task_id, status="FAILED", progress=100, stage="failed", message="Generation failed", finished_at=time.time(), error=str(exc))
        LOGGER.error(json.dumps({"event": "task_failed", "task_id": task_id, "error": str(exc), "traceback": traceback.format_exc()}, default=str))


def build_service_diagnostics(exc: Exception, output_dir: Path, result_json: Path) -> dict[str, Any]:
    files = []
    try:
        for path in sorted(output_dir.iterdir(), key=lambda item: item.stat().st_mtime):
            if path.is_file():
                files.append({"name": path.name, "size": path.stat().st_size, "modified": path.stat().st_mtime})
    except Exception:
        pass
    return {
        "error": str(exc),
        "error_type": type(exc).__name__,
        "traceback": traceback.format_exc(),
        "progress": _read_worker_progress(output_dir),
        "worker_result": _safe_read_json(result_json),
        "output_files": files,
        "service_python": sys.executable,
        "model_worker_python": MODEL_WORKER_PYTHON,
        "hunyuan3d_path": os.environ.get("MESHMEND_HUNYUAN3D_PATH", ""),
        "production_engine": os.environ.get("MESHMEND_PRODUCTION_ENGINE", ""),
        "backend_diagnostics": _build_backend_diagnostics(include_recent_failures=False),
    }


def _build_backend_diagnostics(*, include_recent_failures: bool = True) -> dict[str, Any]:
    imports = {name: _check_import(name) for name in ("fastapi", "pydantic", "numpy", "trimesh", "scipy", "PIL", "torch", "diffusers", "open3d", "hy3dgen")}
    permissions = {name: _permission_check(path) for name, path in {"tasks_dir": TASKS_DIR, "outputs_dir": OUTPUTS_DIR, "logs_dir": LOGS_DIR, "service_root": SERVICE_ROOT}.items()}
    worker_python = _python_status(MODEL_WORKER_PYTHON)
    gpu = _gpu_status()
    port = _port_status("127.0.0.1", int(SERVICE_PORT))
    configured_paths = {
        "service_root": _path_status(SERVICE_ROOT),
        "workspace_root": _path_status(WORKSPACE_ROOT),
        "model_worker_python": _path_status(Path(MODEL_WORKER_PYTHON)),
        "hunyuan3d_path": _path_status(Path(os.environ.get("MESHMEND_HUNYUAN3D_PATH", ""))) if os.environ.get("MESHMEND_HUNYUAN3D_PATH", "").strip() else {"configured": False},
        "external_generator_command": {"configured": bool(os.environ.get("MESHMEND_EXTERNAL_GENERATOR_COMMAND", "").strip())},
    }
    failed_checks = []
    failed_checks.extend(f"import:{name}" for name, result in imports.items() if not result.get("ok") and name in {"fastapi", "pydantic", "numpy", "trimesh"})
    failed_checks.extend(f"permission:{name}" for name, result in permissions.items() if not result.get("writable"))
    if not worker_python.get("ok"):
        failed_checks.append("worker_python")
    recent = _recent_failure_reports() if include_recent_failures else []
    return {
        "status": "ok" if not failed_checks else "degraded",
        "backend_running": True,
        "model_quality_acceptable": bool(_studio_quality_runner_ready("text_to_3d") or _studio_quality_runner_ready("image_to_3d")),
        "model_quality_note": "This diagnostic checks backend stability, not visual miniature quality.",
        "failed_checks": failed_checks,
        "service": {
            "build_id": SERVICE_BUILD_ID,
            "python": sys.executable,
            "platform": platform.platform(),
            "cwd": os.getcwd(),
            "pid": os.getpid(),
            "port": SERVICE_PORT,
            "public_base_url": PUBLIC_BASE_URL,
            "log_file": str(BACKEND_LOG_FILE),
        },
        "commands": {
            "text_to_3d_command": TEXT_TO_3D_COMMAND,
            "image_to_3d_command": IMAGE_TO_3D_COMMAND,
            "model_worker_python": MODEL_WORKER_PYTHON,
            "production_engine": os.environ.get("MESHMEND_PRODUCTION_ENGINE", "meshmend_native"),
        },
        "imports": imports,
        "gpu": gpu,
        "memory": _memory_safety_status(),
        "paths": configured_paths,
        "permissions": permissions,
        "port": port,
        "worker_python": worker_python,
        "hunyuan_import": _hunyuan_import_status() if os.environ.get("MESHMEND_PRODUCTION_ENGINE", "").strip().lower() in {"free_local", "free_local_hunyuan", "hunyuan", "hunyuan3d"} else None,
        "recent_failures": recent,
        "active_tasks": {task_id: asdict(task) for task_id, task in list(_tasks.items())[-20:]},
    }


def _dependency_health_summary() -> dict[str, Any]:
    required = {name: _check_import(name) for name in ("fastapi", "pydantic", "numpy", "trimesh")}
    optional = {name: _check_import(name) for name in ("torch", "diffusers", "open3d", "hy3dgen")}
    return {
        "required_ok": all(result.get("ok") for result in required.values()),
        "required": required,
        "optional": optional,
    }


def _check_import(module_name: str) -> dict[str, Any]:
    try:
        import importlib

        module = importlib.import_module(module_name)
        return {"ok": True, "version": str(getattr(module, "__version__", "")), "file": str(getattr(module, "__file__", ""))}
    except Exception as exc:
        return {"ok": False, "error": str(exc), "traceback": traceback.format_exc()}


def _gpu_status() -> dict[str, Any]:
    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
        return {
            "torch_import_ok": True,
            "cuda_available": cuda_available,
            "cuda_device_count": int(torch.cuda.device_count()) if cuda_available else 0,
            "cuda_devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if cuda_available else [],
        }
    except Exception as exc:
        return {"torch_import_ok": False, "cuda_available": False, "error": str(exc)}


def _memory_safety_status() -> dict[str, Any]:
    try:
        import psutil

        vm = psutil.virtual_memory()
        return {
            "enabled": _memory_safety_enabled(),
            "max_postprocess_faces": int(os.environ.get("MESHMEND_MAX_POSTPROCESS_FACES", "350000")),
            "max_export_faces": int(os.environ.get("MESHMEND_MAX_EXPORT_FACES", "600000")),
            "available_gb": round(float(vm.available) / (1024**3), 2),
            "total_gb": round(float(vm.total) / (1024**3), 2),
            "percent_used": float(vm.percent),
        }
    except Exception as exc:
        return {
            "enabled": _memory_safety_enabled(),
            "max_postprocess_faces": int(os.environ.get("MESHMEND_MAX_POSTPROCESS_FACES", "350000")),
            "max_export_faces": int(os.environ.get("MESHMEND_MAX_EXPORT_FACES", "600000")),
            "psutil_error": str(exc),
        }


def _memory_safety_enabled() -> bool:
    return os.environ.get("MESHMEND_DISABLE_MEMORY_SAFETY", "0").strip().lower() not in {"1", "true", "yes", "on"}


def _path_status(path: Path) -> dict[str, Any]:
    try:
        return {"path": str(path), "exists": path.exists(), "is_file": path.is_file(), "is_dir": path.is_dir()}
    except Exception as exc:
        return {"path": str(path), "error": str(exc)}


def _permission_check(path: Path) -> dict[str, Any]:
    try:
        path.mkdir(parents=True, exist_ok=True) if path.suffix == "" else path.parent.mkdir(parents=True, exist_ok=True)
        probe_dir = path if path.is_dir() or path.suffix == "" else path.parent
        probe = probe_dir / f".meshmend_write_probe_{uuid.uuid4().hex}"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
        return {"path": str(path), "writable": True}
    except Exception as exc:
        return {"path": str(path), "writable": False, "error": str(exc)}


def _python_status(python_path: str) -> dict[str, Any]:
    try:
        completed = subprocess.run([python_path, "-c", "import sys,platform; print(sys.executable); print(platform.platform())"], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=15)
        return {"ok": completed.returncode == 0, "stdout": completed.stdout.strip(), "stderr": completed.stderr.strip(), "returncode": completed.returncode}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def _port_status(host: str, port: int) -> dict[str, Any]:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.5)
            connected = sock.connect_ex((host, port)) == 0
        return {"host": host, "port": port, "accepting_connections": connected}
    except Exception as exc:
        return {"host": host, "port": port, "error": str(exc)}


def _recent_failure_reports(limit: int = 8) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    try:
        task_dirs = [path for path in OUTPUTS_DIR.iterdir() if path.is_dir()]
        for output_dir in sorted(task_dirs, key=lambda item: item.stat().st_mtime, reverse=True):
            error_path = output_dir / "worker_error.txt"
            traceback_path = output_dir / "service_traceback.txt"
            if not error_path.exists() and not traceback_path.exists():
                continue
            reports.append(
                {
                    "task_id": output_dir.name,
                    "modified": output_dir.stat().st_mtime,
                    "error": error_path.read_text(encoding="utf-8", errors="replace")[-4000:] if error_path.exists() else "",
                    "traceback": traceback_path.read_text(encoding="utf-8", errors="replace")[-8000:] if traceback_path.exists() else "",
                }
            )
            if len(reports) >= limit:
                break
    except Exception as exc:
        reports.append({"error": str(exc)})
    return reports


def _create_known_test_mesh() -> Any:
    import trimesh

    base = trimesh.creation.cylinder(radius=8.0, height=2.0, sections=64)
    body = trimesh.creation.icosphere(subdivisions=3, radius=3.0)
    body.apply_scale([0.75, 0.65, 1.45])
    body.apply_translation([0, 0, 6.0])
    head = trimesh.creation.icosphere(subdivisions=2, radius=1.15)
    head.apply_translation([0, 0, 10.3])
    mesh = trimesh.util.concatenate([base, body, head])
    mesh.metadata["units"] = "mm"
    mesh.metadata["purpose"] = "backend_test_mesh"
    return mesh


def _generate_modular_fallback(
    *,
    prompt: str,
    scale_mm: float,
    target_faces: int,
    output_format: str,
    candidates_per_category: int,
    reason: str,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    if not _modular_fallback_enabled():
        raise RuntimeError(
            "GENERATION FAILED: archetype generator failed. "
            "Failing function name: _generate_modular_fallback. "
            "Procedural mannequin/placeholder fallback is disabled."
        )
    from meshmend.studio.pipeline import StudioMiniatureSpec
    from meshmend.studio.staged_pipeline import StagedMiniaturePipeline

    fmt = _safe_export_format(output_format)
    output_dir = output_dir or (OUTPUTS_DIR / ("fallback_" + uuid.uuid4().hex[:12]))
    output_dir.mkdir(parents=True, exist_ok=True)
    spec = StudioMiniatureSpec.from_prompt(prompt, scale_mm=scale_mm, target_faces=max(24_000, min(int(target_faces), 150_000)))
    pipeline = StagedMiniaturePipeline()
    result = pipeline.generate(spec, candidates_per_category=max(1, min(int(candidates_per_category), 6)), candidate_output_dir=output_dir / "candidates")
    mesh = result.mesh
    output_path = output_dir / f"modular_fallback_miniature.{fmt}"
    _export_mesh(mesh, output_path)
    quality_report = result.quality_report.to_dict()
    payload = {
        "status": "ok",
        "source": "procedural_modular_fallback",
        "quality_tier": "valid_procedural_placeholder_miniature",
        "studio_quality_certified": False,
        "studio_quality_note": "Fallback guarantees valid modular geometry; it is not claimed as store/studio-quality visual sculpt output.",
        "fallback_reason": reason,
        "model_file": output_path.name,
        "model_format": fmt,
        "model_urls": {fmt: f"{PUBLIC_BASE_URL}/v1/files/{task_file_name(output_path)}"},
        "mesh_info": _mesh_summary(mesh),
        "quality_report": quality_report,
        "stage_results": [stage.to_dict() for stage in result.stage_results],
        "selected_parts": {category.value: part.part_id for category, part in result.selected_parts.items()},
        "consumed_credits": 0,
    }
    (output_dir / "fallback_report.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return payload


def _safe_export_format(format_name: str) -> str:
    fmt = str(format_name or "stl").lower().lstrip(".")
    if fmt not in {"stl", "obj", "glb", "ply"}:
        raise HTTPException(status_code=400, detail="format must be one of: stl, obj, glb, ply")
    return fmt


def _export_mesh(mesh: Any, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(output_path))


def _mesh_summary(mesh: Any) -> dict[str, Any]:
    try:
        components = [part for part in mesh.split(only_watertight=False) if len(part.faces) > 20]
    except Exception:
        components = []
    return {
        "faces": int(len(mesh.faces)),
        "vertices": int(len(mesh.vertices)),
        "watertight": bool(getattr(mesh, "is_watertight", False)),
        "components": len(components),
        "extents_mm": [float(value) for value in getattr(mesh, "extents", [])],
        "units": str(getattr(mesh, "metadata", {}).get("units", "mm")),
    }


def _safe_read_json(path: Path) -> Any:
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"_read_error": str(exc)}
    return None


def _run_worker_with_progress(task_id: str, command_args: list[str], output_dir: Path) -> subprocess.CompletedProcess[str]:
    process = subprocess.Popen(
            command_args,
            cwd=str(SERVICE_ROOT),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
    )
    stdout_parts: list[str] = []
    stderr_parts: list[str] = []
    deadline = time.time() + float(os.environ.get("MESHMEND_MODEL_COMMAND_TIMEOUT_SECONDS", "10800"))
    last_progress_mtime = 0.0
    last_heartbeat = 0.0
    last_progress_seen_at = time.time()
    last_output_count = 0
    stalled_timeout = float(os.environ.get("MESHMEND_MODEL_STALLED_TIMEOUT_SECONDS", "3600"))
    stderr_thread = threading.Thread(target=_collect_pipe, args=(process.stderr, stderr_parts), daemon=True)
    stdout_thread = threading.Thread(target=_collect_pipe, args=(process.stdout, stdout_parts), daemon=True)
    stderr_thread.start()
    stdout_thread.start()
    while process.poll() is None:
        if time.time() > deadline:
            process.kill()
            raise RuntimeError(f"model command timed out after {os.environ.get('MESHMEND_MODEL_COMMAND_TIMEOUT_SECONDS', '10800')}s")
        progress_path = output_dir / "progress.json"
        try:
            if progress_path.exists():
                mtime = progress_path.stat().st_mtime
                if mtime > last_progress_mtime:
                    last_progress_mtime = mtime
                    progress = _read_worker_progress(output_dir)
                    if progress:
                        _apply_worker_progress(task_id, progress)
                        last_progress_seen_at = time.time()
        except Exception:
            pass
        output_count = len(stdout_parts) + len(stderr_parts)
        if output_count > last_output_count:
            last_output_count = output_count
            last_progress_seen_at = time.time()
        if stalled_timeout > 0 and time.time() - last_progress_seen_at > stalled_timeout:
            process.kill()
            raise RuntimeError(f"model command made no progress for {stalled_timeout:g}s")
        if time.time() - last_heartbeat >= float(os.environ.get("MESHMEND_PROGRESS_HEARTBEAT_SECONDS", "10")):
            last_heartbeat = time.time()
            _heartbeat_progress(task_id)
        time.sleep(float(os.environ.get("MESHMEND_PROGRESS_POLL_SECONDS", "1.0")))
    stdout_thread.join(timeout=2)
    stderr_thread.join(timeout=2)
    progress = _read_worker_progress(output_dir)
    if progress:
        _apply_worker_progress(task_id, progress)
    return subprocess.CompletedProcess(command_args, process.returncode or 0, "".join(stdout_parts), "".join(stderr_parts))


def _collect_pipe(pipe: Any, sink: list[str]) -> None:
    if pipe is None:
        return
    try:
        for line in pipe:
            sink.append(line)
    except Exception:
        pass


def _read_worker_progress(output_dir: Path) -> dict[str, Any] | None:
    try:
        path = output_dir / "progress.json"
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _apply_worker_progress(task_id: str, progress: dict[str, Any]) -> None:
    percent = int(progress.get("progress") or progress.get("percent") or 0)
    _update(
        task_id,
        progress=max(5, min(99, percent)),
        stage=str(progress.get("stage") or "running"),
        message=str(progress.get("message") or progress.get("stage") or "Running"),
        last_progress_at=float(progress.get("updated_at") or time.time()),
    )


def _heartbeat_progress(task_id: str) -> None:
    with _lock:
        task = _tasks.get(task_id)
        if task is None or task.status != "IN_PROGRESS":
            return
        elapsed = int(time.time() - (task.last_progress_at or task.started_at or task.created_at))
        stage = task.stage
        current = int(task.progress or 0)
        message = str(task.message).split(" — still running", 1)[0].split(" (", 1)[0]
    synthetic = current
    if stage == "hunyuan_shape_generation":
        synthetic = min(68, current + 1)
    elif stage == "postprocessing":
        synthetic = min(93, current + 1)
    elif stage in {"hunyuan_loading", "concept_generating"}:
        synthetic = min(35, current + 1)
    heartbeat_message = f"{message} — still running ({elapsed}s since last stage update)"
    _update(task_id, progress=max(current, synthetic), message=heartbeat_message)


def _worker_command_args(command: str, *, input_json: Path, output_dir: Path) -> list[str]:
    """Return subprocess argv for model worker execution.

    The default worker command is entirely internal, so build it as a list. This
    avoids Windows quoting/path edge cases such as quoted Python paths with
    spaces being treated as a literal executable name.
    """
    default_args = [
        MODEL_WORKER_PYTHON,
        str(SERVICE_ROOT / "production_worker.py"),
        "--input",
        str(input_json),
        "--output-dir",
        str(output_dir),
    ]
    if command == DEFAULT_WORKER_COMMAND.format(input_json=str(input_json), output_dir=str(output_dir), task_id=""):
        return default_args
    default_no_task = DEFAULT_WORKER_COMMAND.format(input_json=str(input_json), output_dir=str(output_dir))
    if command == default_no_task:
        return default_args
    if os.name == "nt":
        return [part[1:-1] if len(part) >= 2 and part[0] == part[-1] == '"' else part for part in shlex.split(command, posix=False)]
    return shlex.split(command, posix=True)


def _load_worker_result(result_json: Path, stdout: str, output_dir: Path) -> dict[str, Any]:
    if result_json.exists():
        return json.loads(result_json.read_text(encoding="utf-8"))
    stripped = (stdout or "").strip()
    if stripped.startswith("{"):
        return json.loads(stripped)
    for suffix in (".stl", ".glb", ".obj", ".ply", ".3mf"):
        matches = list(output_dir.glob(f"*{suffix}"))
        if matches:
            return {"model_file": matches[0].name, "model_format": suffix.lstrip(".")}
    return {}


def _result_studio_certified(result: dict[str, Any]) -> bool:
    if bool(result.get("studio_quality_certified")):
        return True
    if bool(result.get("store_quality_certified")):
        return True
    mesh_info = result.get("mesh_info")
    if isinstance(mesh_info, dict) and bool(mesh_info.get("production_ready")):
        return True
    if isinstance(mesh_info, dict) and bool(mesh_info.get("store_quality_certified")):
        return True
    quality_report = result.get("quality_report")
    if isinstance(quality_report, dict) and bool(quality_report.get("production_ready")):
        return True
    return False


def _result_validated_base_form(result: dict[str, Any], output_dir: Path) -> bool:
    """Accept an explicitly validated body without falsely studio-certifying it."""
    if not bool(result.get("base_form_only")) or not bool(result.get("base_form_validated")):
        return False
    if str(result.get("release_tier") or "") != "validated_base_form":
        return False
    model_file = Path(str(result.get("model_file") or "")).name
    if not model_file or not (output_dir / model_file).is_file():
        return False
    mesh_info = result.get("mesh_info") if isinstance(result.get("mesh_info"), dict) else {}
    structural_prefixes = (
        "empty_mesh", "degenerate_faces", "mesh_too_flat", "likely_dual_subject",
        "likely_background_slab", "likely_horizontal_square_sheet", "likely_blocky_low_definition",
        "mesh_not_solid_watertight", "mesh_boundary_edges", "mesh_nonmanifold_edges",
        "too_many_disconnected_components", "zero_or_tiny_volume", "collapsed_bounding_box",
        "image_visual_holes_unsealed", "image_low_relief_sheet", "heavy_artifact_salvage",
        "component_bridges_visible_artifact_risk", "excessive_bilateral_asymmetry",
    )
    issues = [str(issue) for issue in mesh_info.get("quality_gate_issues") or []]
    return not any(issue.startswith(structural_prefixes) for issue in issues)


def _result_studio_certification_summary(result: dict[str, Any]) -> dict[str, Any]:
    mesh_info = result.get("mesh_info") if isinstance(result.get("mesh_info"), dict) else {}
    quality_report = result.get("quality_report") if isinstance(result.get("quality_report"), dict) else {}
    return {
        "provider": result.get("provider"),
        "studio_quality_certified": bool(result.get("studio_quality_certified")),
        "store_quality_certified": bool(result.get("store_quality_certified")),
        "mesh_info_production_ready": bool(mesh_info.get("production_ready")),
        "mesh_info_store_quality_certified": bool(mesh_info.get("store_quality_certified")),
        "quality_report_production_ready": bool(quality_report.get("production_ready")),
        "quality_gate_issues": mesh_info.get("quality_gate_issues") or quality_report.get("quality_gate_issues") or [],
        "studio_quality_note": result.get("studio_quality_note"),
    }


def _worker_error_message(result_json: Path, stdout: str, stderr: str) -> str:
    """Return a concise worker error instead of progress bars and model logs."""
    try:
        if result_json.exists():
            result = json.loads(result_json.read_text(encoding="utf-8"))
            error = str(result.get("error") or "").strip()
            if error:
                return error
    except Exception:
        pass
    for text in (stderr, stdout):
        stripped = (text or "").strip()
        if not stripped:
            continue
        last_line = stripped.splitlines()[-1]
        try:
            parsed = json.loads(last_line)
            error = str(parsed.get("error") or "").strip()
            if error:
                return error
        except Exception:
            return stripped[-2000:]
    return ""


def _request_dict(request: BaseModel) -> dict[str, Any]:
    if hasattr(request, "model_dump"):
        return request.model_dump()
    return request.dict()


def _result_model_urls(result: dict[str, Any], output_dir: Path) -> dict[str, str]:
    model_file = result.get("model_file")
    if model_file:
        path = output_dir / Path(str(model_file)).name
        if path.exists():
            fmt = str(result.get("model_format") or path.suffix.lstrip(".")).lower()
            return {fmt: f"{PUBLIC_BASE_URL}/v1/files/{task_file_name(path)}"}
    if isinstance(result.get("model_urls"), dict):
        return {str(k): str(v) for k, v in result["model_urls"].items()}
    urls: dict[str, str] = {}
    for path in output_dir.iterdir():
        if path.suffix.lower() in {".stl", ".glb", ".obj", ".ply", ".3mf", ".fbx", ".usdz"}:
            urls[path.suffix.lower().lstrip(".")] = f"{PUBLIC_BASE_URL}/v1/files/{task_file_name(path)}"
    return urls


def task_file_name(path: Path) -> str:
    # Flatten task output paths into a single static file namespace.
    flattened = f"{path.parent.name}_{path.name}"
    target = OUTPUTS_DIR / flattened
    if not target.exists():
        shutil.copy2(path, target)
    with _lock:
        _served_files[flattened] = path
    return flattened


def _cleanup_outputs_after_download_enabled() -> bool:
    return os.environ.get("MESHMEND_CLEAN_MODEL_SERVICE_OUTPUTS_AFTER_DOWNLOAD", "1").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _cleanup_downloaded_file_artifacts(filename: str) -> None:
    """Best-effort cleanup after FastAPI has finished streaming a generated model."""
    safe_name = Path(filename).name
    if not safe_name:
        return
    removed: dict[str, list[str]] = {"files": [], "dirs": []}
    try:
        source_path = None
        with _lock:
            source_path = _served_files.pop(safe_name, None)
        flattened_path = OUTPUTS_DIR / safe_name
        if flattened_path.exists() and flattened_path.is_file():
            flattened_path.unlink()
            removed["files"].append(str(flattened_path))
        if source_path is not None:
            source_path = source_path.resolve()
            outputs_root = OUTPUTS_DIR.resolve()
            try:
                source_path.relative_to(outputs_root)
            except ValueError:
                source_path = None
        if source_path is not None:
            removed = _merge_cleanup_removed(removed, _cleanup_task_artifacts(source_path.parent.name, remove_flattened=False))
        LOGGER.info(json.dumps({"event": "download_cleanup_complete", "filename": safe_name, "removed": removed}, default=str))
    except Exception:
        LOGGER.exception("download_cleanup_failed")


def _cleanup_task_artifacts(task_id: str, *, remove_flattened: bool) -> dict[str, list[str]]:
    removed: dict[str, list[str]] = {"files": [], "dirs": []}
    task_output_dir = OUTPUTS_DIR / task_id
    task_input_dir = TASKS_DIR / task_id

    if remove_flattened:
        with _lock:
            flattened_names = [name for name, source in _served_files.items() if source.parent.name == task_id]
            for name in flattened_names:
                _served_files.pop(name, None)
        for name in flattened_names:
            path = OUTPUTS_DIR / Path(name).name
            if path.exists() and path.is_file():
                try:
                    path.unlink()
                    removed["files"].append(str(path))
                except Exception:
                    LOGGER.exception("flattened_output_cleanup_failed")
        for path in OUTPUTS_DIR.glob(f"{task_id}_*"):
            if path.is_file():
                try:
                    path.unlink()
                    removed["files"].append(str(path))
                except Exception:
                    LOGGER.exception("flattened_output_cleanup_failed")

    for directory in (task_output_dir, task_input_dir):
        if directory.exists() and directory.is_dir():
            try:
                shutil.rmtree(directory)
                removed["dirs"].append(str(directory))
            except Exception:
                LOGGER.exception("task_directory_cleanup_failed")
    return removed


def _merge_cleanup_removed(left: dict[str, list[str]], right: dict[str, list[str]]) -> dict[str, list[str]]:
    for key in ("files", "dirs"):
        left.setdefault(key, []).extend(right.get(key, []))
    return left


def _update(task_id: str, **changes: Any) -> None:
    with _lock:
        task = _tasks.get(task_id)
        if task is None:
            return
        for key, value in changes.items():
            setattr(task, key, value)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("MESHMEND_MODEL_SERVICE_PORT", "8090")))
