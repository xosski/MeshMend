from __future__ import annotations

import json
import os
import shutil
import shlex
import subprocess
import sys
import threading
import time
import traceback
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel


SERVICE_ROOT = Path(__file__).resolve().parent
SERVICE_BUILD_ID = "meshmend-model-service-progress-v3"
TASKS_DIR = Path(os.environ.get("MESHMEND_MODEL_SERVICE_TASKS_DIR", SERVICE_ROOT / "tasks"))
OUTPUTS_DIR = Path(os.environ.get("MESHMEND_MODEL_SERVICE_OUTPUTS_DIR", SERVICE_ROOT / "outputs"))
SERVICE_PORT = os.environ.get("MESHMEND_MODEL_SERVICE_PORT", "8090").strip() or "8090"
PUBLIC_BASE_URL = os.environ.get("MESHMEND_MODEL_SERVICE_PUBLIC_URL", f"http://127.0.0.1:{SERVICE_PORT}").rstrip("/")
MODEL_WORKER_PYTHON = os.environ.get("MESHMEND_MODEL_WORKER_PYTHON", sys.executable).strip() or sys.executable
DEFAULT_WORKER_COMMAND = f'"{MODEL_WORKER_PYTHON}" "{SERVICE_ROOT / "production_worker.py"}" --input "{{input_json}}" --output-dir "{{output_dir}}"'
TEXT_TO_3D_COMMAND = os.environ.get("MESHMEND_TEXT_TO_3D_COMMAND", DEFAULT_WORKER_COMMAND).strip()
IMAGE_TO_3D_COMMAND = os.environ.get("MESHMEND_IMAGE_TO_3D_COMMAND", DEFAULT_WORKER_COMMAND).strip()

TASKS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="MeshMend Independent 3D Model Service", version="0.1.0")


class TextTo3DRequest(BaseModel):
    prompt: str
    quality: str = "high"
    target_formats: list[str] = ["stl", "glb", "obj"]
    target_polycount: int = 2_000_000
    scale_mm: float | None = None
    workflow: str = "text_to_3d"
    product: str = "meshmend"


class ImageTo3DRequest(TextTo3DRequest):
    image_data_uri: str
    workflow: str = "image_to_3d"


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
    return {
        "status": "healthy",
        "service_build_id": SERVICE_BUILD_ID,
        "service_root": str(SERVICE_ROOT),
        "service_python": sys.executable,
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
        "store_quality_reason": _store_quality_reason(production_engine) if not studio_ready else "certified production backend configured",
        "model_worker_python": MODEL_WORKER_PYTHON,
        "hunyuan3d_path": os.environ.get("MESHMEND_HUNYUAN3D_PATH", ""),
        "hunyuan3d_model": os.environ.get("MESHMEND_HUNYUAN3D_MODEL", ""),
        "hunyuan3d_subfolder": os.environ.get("MESHMEND_HUNYUAN3D_SUBFOLDER", ""),
        "hunyuan_import": hunyuan_import,
    }


@app.post("/v1/text-to-3d")
async def text_to_3d(request: TextTo3DRequest):
    if not TEXT_TO_3D_COMMAND:
        raise HTTPException(
            status_code=503,
            detail="MESHMEND_TEXT_TO_3D_COMMAND is not configured on the independent model service.",
        )
    if not _production_runner_ready("text_to_3d"):
        raise HTTPException(
            status_code=503,
            detail="No production text-to-3D runner is configured. Set MESHMEND_PRODUCTION_TEXT_TO_3D_COMMAND, or explicitly set MESHMEND_PRODUCTION_ENGINE=legacy_sculptor for draft-only fallback.",
        )
    if _store_quality_requested(_request_dict(request)) and not _can_accept_high_detail_request("text_to_3d"):
        raise HTTPException(status_code=503, detail=_store_quality_reason(os.environ.get("MESHMEND_PRODUCTION_ENGINE", "meshmend_native")))
    return _enqueue("text_to_3d", _request_dict(request))


@app.post("/v1/image-to-3d")
async def image_to_3d(request: ImageTo3DRequest):
    if not IMAGE_TO_3D_COMMAND:
        raise HTTPException(
            status_code=503,
            detail="MESHMEND_IMAGE_TO_3D_COMMAND is not configured on the independent model service.",
        )
    if not _production_runner_ready("image_to_3d"):
        raise HTTPException(
            status_code=503,
            detail="No production image-to-3D runner is configured. Set MESHMEND_PRODUCTION_IMAGE_TO_3D_COMMAND, or explicitly set MESHMEND_PRODUCTION_ENGINE=legacy_sculptor for draft-only fallback.",
        )
    if _store_quality_requested(_request_dict(request)) and not _can_accept_high_detail_request("image_to_3d"):
        raise HTTPException(status_code=503, detail=_store_quality_reason(os.environ.get("MESHMEND_PRODUCTION_ENGINE", "meshmend_native")))
    return _enqueue("image_to_3d", _request_dict(request))


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
    return FileResponse(path)


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
            import native_generation  # noqa: F401

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


def _capability_tier(engine: str) -> str:
    engine = engine.strip().lower()
    if engine in {"external", "command"} and os.environ.get("MESHMEND_EXTERNAL_STORE_QUALITY_CERTIFIED", "0").strip().lower() in {"1", "true", "yes", "on"}:
        return "certified_store_quality_external"
    if engine in {"meshmend_native", "native", "embedded_native"}:
        return "procedural_printable_draft"
    if engine in {"meshmend_sculpt", "native_sculpt", "sculpt"}:
        return "experimental_image_conditioned_native_sculpt"
    if engine in {"free_local", "free_local_hunyuan", "hunyuan", "hunyuan3d"}:
        return "experimental_image_reconstruction"
    if engine in {"external", "command"}:
        return "external_uncertified"
    return "unconfigured"


def _store_quality_reason(engine: str) -> str:
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

    _update(task_id, status="IN_PROGRESS", progress=5, stage="starting", message="Starting production worker", started_at=time.time())
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
            message="Production model ready",
            finished_at=time.time(),
            model_urls=model_urls,
            thumbnail_url=result.get("thumbnail_url"),
            consumed_credits=int(result.get("consumed_credits") or (30 if workflow == "image_to_3d" else 20)),
        )
    except Exception as exc:
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
    deadline = time.time() + float(os.environ.get("MESHMEND_MODEL_COMMAND_TIMEOUT_SECONDS", "3600"))
    last_progress_mtime = 0.0
    last_heartbeat = 0.0
    last_progress_seen_at = time.time()
    last_output_count = 0
    stalled_timeout = float(os.environ.get("MESHMEND_MODEL_STALLED_TIMEOUT_SECONDS", "900"))
    stderr_thread = threading.Thread(target=_collect_pipe, args=(process.stderr, stderr_parts), daemon=True)
    stdout_thread = threading.Thread(target=_collect_pipe, args=(process.stdout, stdout_parts), daemon=True)
    stderr_thread.start()
    stdout_thread.start()
    while process.poll() is None:
        if time.time() > deadline:
            process.kill()
            raise RuntimeError(f"model command timed out after {os.environ.get('MESHMEND_MODEL_COMMAND_TIMEOUT_SECONDS', '3600')}s")
        progress_path = output_dir / "progress.json"
        if progress_path.exists():
            try:
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
    return flattened


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
