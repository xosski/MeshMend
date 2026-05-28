from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel


SERVICE_ROOT = Path(__file__).resolve().parent
SERVICE_BUILD_ID = "meshmend-model-service-debug-worker-command-v2"
TASKS_DIR = Path(os.environ.get("MESHMEND_MODEL_SERVICE_TASKS_DIR", SERVICE_ROOT / "tasks"))
OUTPUTS_DIR = Path(os.environ.get("MESHMEND_MODEL_SERVICE_OUTPUTS_DIR", SERVICE_ROOT / "outputs"))
PUBLIC_BASE_URL = os.environ.get("MESHMEND_MODEL_SERVICE_PUBLIC_URL", "http://127.0.0.1:8090").rstrip("/")
MODEL_WORKER_PYTHON = os.environ.get("MESHMEND_MODEL_WORKER_PYTHON", sys.executable).strip() or sys.executable
DEFAULT_WORKER_COMMAND = f'"{MODEL_WORKER_PYTHON}" "{SERVICE_ROOT / "production_worker.py"}" --input "{{input_json}}" --output-dir "{{output_dir}}"'
TEXT_TO_3D_COMMAND = os.environ.get("MESHMEND_TEXT_TO_3D_COMMAND", DEFAULT_WORKER_COMMAND).strip()
IMAGE_TO_3D_COMMAND = os.environ.get("MESHMEND_IMAGE_TO_3D_COMMAND", DEFAULT_WORKER_COMMAND).strip()

TASKS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="MeshMend Independent 3D Model Service", version="0.1.0")


class TextTo3DRequest(BaseModel):
    prompt: str
    quality: str = "standard"
    target_formats: list[str] = ["stl", "glb", "obj"]
    target_polycount: int = 100000
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


@app.get("/health")
async def health():
    production_text_command = os.environ.get("MESHMEND_PRODUCTION_TEXT_TO_3D_COMMAND", "").strip()
    production_image_command = os.environ.get("MESHMEND_PRODUCTION_IMAGE_TO_3D_COMMAND", "").strip()
    production_engine = os.environ.get("MESHMEND_PRODUCTION_ENGINE", "external")
    runner_ready = _production_runner_ready("text_to_3d") or _production_runner_ready("image_to_3d")
    hunyuan_import = _hunyuan_import_status() if production_engine.strip().lower() in {"free_local", "free_local_hunyuan", "hunyuan", "hunyuan3d"} else None
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
        "ready_for_studio_quality": runner_ready,
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
    engine = os.environ.get("MESHMEND_PRODUCTION_ENGINE", "external").strip().lower()
    if engine in {"free_local", "free_local_hunyuan", "hunyuan", "hunyuan3d"}:
        return True
    if engine in {"legacy_sculptor", "embedded"}:
        return True
    if engine not in {"external", "command"}:
        return False
    command_var = "MESHMEND_PRODUCTION_IMAGE_TO_3D_COMMAND" if workflow == "image_to_3d" else "MESHMEND_PRODUCTION_TEXT_TO_3D_COMMAND"
    return bool(os.environ.get(command_var, "").strip())


def _hunyuan_import_status() -> dict[str, Any]:
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
            timeout=float(os.environ.get("MESHMEND_HUNYUAN_IMPORT_CHECK_TIMEOUT_SECONDS", "20")),
        )
    except Exception as exc:
        return {
            "ok": False,
            "worker_python": MODEL_WORKER_PYTHON,
            "error": str(exc),
        }
    output = (completed.stdout or completed.stderr or "").strip()
    return {
        "ok": completed.returncode == 0,
        "worker_python": MODEL_WORKER_PYTHON,
        "output": output,
    }


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

    _update(task_id, status="IN_PROGRESS", progress=5, started_at=time.time())
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
        completed = subprocess.run(
            command_args,
            cwd=str(SERVICE_ROOT),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=float(os.environ.get("MESHMEND_MODEL_COMMAND_TIMEOUT_SECONDS", "3600")),
        )
        if completed.returncode != 0:
            raise RuntimeError(completed.stderr.strip() or completed.stdout.strip() or f"model command exited {completed.returncode}")

        result = _load_worker_result(result_json, completed.stdout, output_dir)
        model_urls = _result_model_urls(result, output_dir)
        if not model_urls:
            raise RuntimeError("model worker completed but did not produce a supported model file or model_urls")
        _update(
            task_id,
            status="SUCCEEDED",
            progress=100,
            finished_at=time.time(),
            model_urls=model_urls,
            thumbnail_url=result.get("thumbnail_url"),
            consumed_credits=int(result.get("consumed_credits") or (30 if workflow == "image_to_3d" else 20)),
        )
    except Exception as exc:
        try:
            (output_dir / "worker_error.txt").write_text(str(exc), encoding="utf-8")
        except Exception:
            pass
        _update(task_id, status="FAILED", progress=100, finished_at=time.time(), error=str(exc))


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
    return shlex.split(command, posix=os.name != "nt")


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


def _request_dict(request: BaseModel) -> dict[str, Any]:
    if hasattr(request, "model_dump"):
        return request.model_dump()
    return request.dict()


def _result_model_urls(result: dict[str, Any], output_dir: Path) -> dict[str, str]:
    if isinstance(result.get("model_urls"), dict):
        return {str(k): str(v) for k, v in result["model_urls"].items()}
    model_file = result.get("model_file")
    if model_file:
        path = output_dir / Path(str(model_file)).name
        if path.exists():
            fmt = str(result.get("model_format") or path.suffix.lstrip(".")).lower()
            return {fmt: f"{PUBLIC_BASE_URL}/v1/files/{task_file_name(path)}"}
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
        target.write_bytes(path.read_bytes())
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
