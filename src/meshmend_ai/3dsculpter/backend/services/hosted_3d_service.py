from __future__ import annotations

import base64
import json
import mimetypes
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from utils.config import OUTPUT_DIR


REMOTE_3D_ENABLED = os.environ.get("MESHMEND_USE_HOSTED_3D", "1").strip().lower() in {"1", "true", "yes"}
HOSTED_PROVIDER = os.environ.get("MESHMEND_HOSTED_3D_PROVIDER", "meshmend").strip().lower()
SELF_HOSTED_MODEL_SERVICE_URL = os.environ.get("MESHMEND_MODEL_SERVICE_URL", "http://127.0.0.1:8090").strip().rstrip("/")
SELF_HOSTED_MODEL_SERVICE_API_KEY = os.environ.get("MESHMEND_MODEL_SERVICE_API_KEY", "").strip()
HOSTED_POLL_INTERVAL_SECONDS = float(os.environ.get("MESHMEND_HOSTED_POLL_INTERVAL_SECONDS", "5"))
HOSTED_TIMEOUT_SECONDS = float(os.environ.get("MESHMEND_HOSTED_TIMEOUT_SECONDS", "1800"))
HOSTED_TARGET_FORMATS = [
    item.strip().lower()
    for item in os.environ.get("MESHMEND_HOSTED_TARGET_FORMATS", "stl,glb,obj").split(",")
    if item.strip()
]


@dataclass(frozen=True)
class HostedGenerationResult:
    provider: str
    task_id: str
    status: str
    model_file: str
    model_format: str
    thumbnail_url: str | None
    raw_task: dict[str, Any]
    consumed_credits: int


class Hosted3DError(RuntimeError):
    pass


class EmbeddedProvider:
    """Legacy in-program procedural generator.

    This path is intentionally no longer the default. It exists only as an
    explicit fallback/developer mode because it cannot produce production-level
    miniatures consistently.
    """

    def generate_text_to_3d(self, prompt: str, *, quality: str = "standard") -> HostedGenerationResult:
        output_path = self._create_model(prompt, image_path=None, quality=quality)
        return self._result(output_path, workflow="embedded_text_to_3d", credits=20)

    def generate_image_to_3d(self, image_path: Path, prompt: str = "", *, quality: str = "high") -> HostedGenerationResult:
        output_path = self._create_model(prompt, image_path=image_path, quality=quality)
        return self._result(output_path, workflow="embedded_image_to_3d", credits=30)

    @staticmethod
    def _create_model(prompt: str, image_path: Path | None, quality: str) -> Path:
        meshmend_src = Path(__file__).resolve().parents[4]
        if str(meshmend_src) not in sys.path:
            sys.path.insert(0, str(meshmend_src))
        from meshmend_ai.sculptor import get_sculptor_foundation

        q = (quality or "standard").lower()
        print_detail_um = 35 if q == "high" else 50 if q == "standard" else 100
        max_triangles = 2_500_000 if q == "high" else 900_000 if q == "standard" else 350_000
        enriched_prompt = (
            f"{prompt}. Premium embedded generation: prioritize distinct subject topology, dense sculpted detail, "
            "sharp miniature-scale edges, engraved panel lines, cloth/skin/material texture, non-generic silhouette, "
            "and avoid primitive blockout shapes."
        )
        return get_sculptor_foundation().create_model(
            enriched_prompt,
            image_path=image_path,
            scale_mm=32.0,
            print_detail_um=print_detail_um,
            max_detail_triangles=max_triangles,
        )

    @staticmethod
    def _result(output_path: Path, workflow: str, credits: int) -> HostedGenerationResult:
        output_path = Path(output_path)
        return HostedGenerationResult(
            provider="embedded",
            task_id=output_path.stem,
            status="SUCCEEDED",
            model_file=output_path.name,
            model_format=output_path.suffix.lower().lstrip(".") or "stl",
            thumbnail_url=None,
            raw_task={"workflow": workflow, "model_file": output_path.name},
            consumed_credits=credits,
        )


class SelfHostedProvider:
    """Client for the local MeshMend production model service.

    Expected worker contract:
      POST /v1/text-to-3d  JSON -> {task_id|id|result} or completed task object
      POST /v1/image-to-3d JSON with image_data_uri -> {task_id|id|result} or task
      GET  /v1/tasks/{task_id} -> task status object

    A completed task should include `model_urls` or `model_url`; signed/local URLs
    are downloaded into this backend's OUTPUT_DIR and returned to the app.
    """

    def __init__(self, base_url: str | None = None, api_key: str | None = None):
        self.base_url = (base_url or SELF_HOSTED_MODEL_SERVICE_URL).strip().rstrip("/")
        self.api_key = api_key if api_key is not None else SELF_HOSTED_MODEL_SERVICE_API_KEY
        if not self.base_url:
            raise Hosted3DError("MESHMEND_MODEL_SERVICE_URL is not configured for the independent model service.")

    def generate_text_to_3d(self, prompt: str, *, quality: str = "standard") -> HostedGenerationResult:
        payload = {
            "prompt": prompt,
            "quality": quality,
            "target_formats": HOSTED_TARGET_FORMATS,
            "target_polycount": self._target_polycount(quality),
            "scale_mm": self._scale_mm_from_prompt(prompt),
            "workflow": "text_to_3d",
            "product": "meshmend",
        }
        task = self._submit_and_wait("/v1/text-to-3d", payload)
        return self._download_result(task, preferred_stem="selfhosted_text")

    def generate_image_to_3d(self, image_path: Path, prompt: str = "", *, quality: str = "high") -> HostedGenerationResult:
        payload = {
            "prompt": prompt,
            "image_data_uri": self._image_data_uri(image_path),
            "quality": quality,
            "target_formats": HOSTED_TARGET_FORMATS,
            "target_polycount": self._target_polycount(quality),
            "scale_mm": self._scale_mm_from_prompt(prompt),
            "workflow": "image_to_3d",
            "product": "meshmend",
        }
        task = self._submit_and_wait("/v1/image-to-3d", payload)
        return self._download_result(task, preferred_stem="selfhosted_image")

    def _submit_and_wait(self, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        response = self._request_json("POST", endpoint, payload)
        if self._is_succeeded(response):
            return response
        task_id = response.get("task_id") or response.get("id") or response.get("result")
        if not task_id:
            raise Hosted3DError(f"Independent model service did not return a task id or completed task: {response}")
        return self._poll_task(str(task_id))

    def _poll_task(self, task_id: str) -> dict[str, Any]:
        deadline = time.time() + HOSTED_TIMEOUT_SECONDS
        last_task: dict[str, Any] = {}
        while time.time() < deadline:
            task = self._request_json("GET", f"/v1/tasks/{urllib.parse.quote(task_id)}")
            last_task = task
            if self._is_succeeded(task):
                return task
            status = str(task.get("status", "")).upper()
            if status in {"FAILED", "ERROR", "CANCELED", "CANCELLED"}:
                error = task.get("error") or task.get("task_error") or task
                raise Hosted3DError(f"Independent model service task failed: {error}")
            time.sleep(HOSTED_POLL_INTERVAL_SECONDS)
        raise Hosted3DError(f"Independent model service task timed out after {HOSTED_TIMEOUT_SECONDS:g}s: {last_task}")

    def _download_result(self, task: dict[str, Any], *, preferred_stem: str) -> HostedGenerationResult:
        model_urls = task.get("model_urls") or {}
        if task.get("model_url") and not model_urls:
            model_urls = {str(task.get("model_format") or "glb").lower(): task["model_url"]}
        selected_format = ""
        selected_url = ""
        for fmt in HOSTED_TARGET_FORMATS + ["stl", "glb", "obj", "ply", "3mf", "fbx", "usdz"]:
            url = model_urls.get(fmt)
            if url:
                selected_format = fmt
                selected_url = url
                break
        if not selected_url:
            raise Hosted3DError(f"Independent model service returned no downloadable model URL: {task}")
        task_id = str(task.get("task_id") or task.get("id") or "task")
        output_name = f"{preferred_stem}_{task_id.replace('-', '')[:16]}.{selected_format}"
        output_path = OUTPUT_DIR / output_name
        self._download_file(selected_url, output_path)
        return HostedGenerationResult(
            provider="self_hosted",
            task_id=task_id,
            status=str(task.get("status", "SUCCEEDED")),
            model_file=output_name,
            model_format=selected_format,
            thumbnail_url=task.get("thumbnail_url"),
            raw_task=task,
            consumed_credits=int(task.get("consumed_credits") or task.get("credits") or 0),
        )

    def _request_json(self, method: str, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        url = self.base_url + path
        body = json.dumps(payload).encode("utf-8") if payload is not None else None
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(url, data=body, method=method, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace")
            raise Hosted3DError(f"Independent model service error {exc.code}: {details}") from exc
        except urllib.error.URLError as exc:
            raise Hosted3DError(f"Independent model service connection failed: {exc}") from exc

    @staticmethod
    def _is_succeeded(task: dict[str, Any]) -> bool:
        status = str(task.get("status", "")).upper()
        return status in {"SUCCEEDED", "SUCCESS", "COMPLETED", "DONE"} and bool(task.get("model_urls") or task.get("model_url"))

    @staticmethod
    def _download_file(url: str, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if url.startswith("file://"):
            source = Path(urllib.parse.urlparse(url).path)
            output_path.write_bytes(source.read_bytes())
            return
        request = urllib.request.Request(url, headers={"User-Agent": "MeshMend/1.0"})
        with urllib.request.urlopen(request, timeout=120) as response:
            output_path.write_bytes(response.read())

    @staticmethod
    def _image_data_uri(image_path: Path) -> str:
        mime = mimetypes.guess_type(str(image_path))[0] or "image/png"
        encoded = base64.b64encode(Path(image_path).read_bytes()).decode("ascii")
        return f"data:{mime};base64,{encoded}"

    @staticmethod
    def _target_polycount(quality: str) -> int:
        override = os.environ.get("MESHMEND_HOSTED_TARGET_POLYCOUNT")
        if override:
            return int(override)
        q = (quality or "standard").lower()
        if q == "high":
            return 180000
        if q == "low":
            return 50000
        return 100000

    @staticmethod
    def _scale_mm_from_prompt(prompt: str) -> float:
        match = __import__("re").search(r"\b(15|20|25|28|30|32|35|40|48|54|75|90|100)\s*mm\b", (prompt or "").lower())
        return float(match.group(1)) if match else 32.0


class MeshyProvider:
    base_url = "https://api.meshy.ai"

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.environ.get("MESHY_API_KEY") or os.environ.get("MESHMEND_MESHY_API_KEY")
        if not self.api_key:
            raise Hosted3DError("MESHY_API_KEY is not configured.")

    def generate_text_to_3d(self, prompt: str, *, quality: str = "standard") -> HostedGenerationResult:
        task_id = self._create_text_preview(prompt, quality=quality)
        task = self._poll_task("/openapi/v2/text-to-3d", task_id)
        return self._download_result(task, preferred_stem="meshy_text")

    def generate_image_to_3d(self, image_path: Path, prompt: str = "", *, quality: str = "high") -> HostedGenerationResult:
        image_uri = self._image_data_uri(image_path)
        task_id = self._create_image_task(image_uri, prompt=prompt, quality=quality)
        task = self._poll_task("/openapi/v1/image-to-3d", task_id)
        return self._download_result(task, preferred_stem="meshy_image")

    def _create_text_preview(self, prompt: str, *, quality: str) -> str:
        payload: dict[str, Any] = {
            "mode": "preview",
            "prompt": prompt,
            "should_remesh": True,
            "target_formats": HOSTED_TARGET_FORMATS,
            "moderation": True,
        }
        target_polycount = self._target_polycount(quality)
        if target_polycount:
            payload["target_polycount"] = target_polycount
        if "miniature" in prompt.lower() or "character" in prompt.lower() or "figure" in prompt.lower():
            payload["pose_mode"] = "a-pose"
        data = self._request_json("POST", "/openapi/v2/text-to-3d", payload)
        task_id = data.get("result")
        if not task_id:
            raise Hosted3DError(f"Meshy did not return a task id: {data}")
        return str(task_id)

    def _create_image_task(self, image_uri: str, *, prompt: str, quality: str) -> str:
        payload: dict[str, Any] = {
            "image_url": image_uri,
            "should_remesh": True,
            "should_texture": True,
            "enable_pbr": True,
            "target_formats": HOSTED_TARGET_FORMATS,
            "moderation": True,
        }
        if prompt:
            payload["texture_prompt"] = prompt[:600]
        target_polycount = self._target_polycount(quality)
        if target_polycount:
            payload["target_polycount"] = target_polycount
        data = self._request_json("POST", "/openapi/v1/image-to-3d", payload)
        task_id = data.get("result")
        if not task_id:
            raise Hosted3DError(f"Meshy did not return a task id: {data}")
        return str(task_id)

    def _poll_task(self, endpoint: str, task_id: str) -> dict[str, Any]:
        deadline = time.time() + HOSTED_TIMEOUT_SECONDS
        last_task: dict[str, Any] = {}
        while time.time() < deadline:
            task = self._request_json("GET", f"{endpoint}/{urllib.parse.quote(task_id)}")
            last_task = task
            status = str(task.get("status", "")).upper()
            if status == "SUCCEEDED":
                return task
            if status in {"FAILED", "CANCELED"}:
                error = task.get("task_error") or {}
                raise Hosted3DError(f"Hosted 3D task {status.lower()}: {error.get('message') or task}")
            time.sleep(HOSTED_POLL_INTERVAL_SECONDS)
        raise Hosted3DError(f"Hosted 3D task timed out after {HOSTED_TIMEOUT_SECONDS:g}s: {last_task}")

    def _download_result(self, task: dict[str, Any], *, preferred_stem: str) -> HostedGenerationResult:
        model_urls = task.get("model_urls") or {}
        selected_format = ""
        selected_url = ""
        for fmt in HOSTED_TARGET_FORMATS + ["stl", "glb", "obj", "fbx", "usdz"]:
            url = model_urls.get(fmt)
            if url:
                selected_format = fmt
                selected_url = url
                break
        if not selected_url:
            raise Hosted3DError(f"Hosted task succeeded but returned no downloadable model URL: {task}")
        task_id = str(task.get("id") or "task")
        output_name = f"{preferred_stem}_{task_id.replace('-', '')[:16]}.{selected_format}"
        output_path = OUTPUT_DIR / output_name
        self._download_file(selected_url, output_path)
        return HostedGenerationResult(
            provider="meshy",
            task_id=task_id,
            status=str(task.get("status", "SUCCEEDED")),
            model_file=output_name,
            model_format=selected_format,
            thumbnail_url=task.get("thumbnail_url"),
            raw_task=task,
            consumed_credits=int(task.get("consumed_credits") or 0),
        )

    def _request_json(self, method: str, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        url = self.base_url + path
        body = json.dumps(payload).encode("utf-8") if payload is not None else None
        request = urllib.request.Request(
            url,
            data=body,
            method=method,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace")
            raise Hosted3DError(f"Meshy API error {exc.code}: {details}") from exc
        except urllib.error.URLError as exc:
            raise Hosted3DError(f"Meshy API connection failed: {exc}") from exc

    @staticmethod
    def _download_file(url: str, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        request = urllib.request.Request(url, headers={"User-Agent": "MeshMend/1.0"})
        with urllib.request.urlopen(request, timeout=120) as response:
            output_path.write_bytes(response.read())

    @staticmethod
    def _image_data_uri(image_path: Path) -> str:
        mime = mimetypes.guess_type(str(image_path))[0] or "image/png"
        encoded = base64.b64encode(Path(image_path).read_bytes()).decode("ascii")
        return f"data:{mime};base64,{encoded}"

    @staticmethod
    def _target_polycount(quality: str) -> int:
        override = os.environ.get("MESHMEND_HOSTED_TARGET_POLYCOUNT")
        if override:
            return int(override)
        q = (quality or "standard").lower()
        if q == "high":
            return 150000
        if q == "low":
            return 50000
        return 100000


class Hosted3DService:
    def __init__(self):
        self.enabled = REMOTE_3D_ENABLED
        self.provider_name = HOSTED_PROVIDER

    def available(self) -> bool:
        if not self.enabled:
            return False
        if self.provider_name in {"embedded", "local", "in_program"}:
            return True
        if self.provider_name in {"self_hosted", "independent", "meshmend"}:
            return bool(SELF_HOSTED_MODEL_SERVICE_URL)
        if self.provider_name == "meshy":
            return bool(os.environ.get("MESHY_API_KEY") or os.environ.get("MESHMEND_MESHY_API_KEY"))
        return False

    def provider(self) -> EmbeddedProvider | SelfHostedProvider | MeshyProvider:
        if self.provider_name in {"embedded", "local", "in_program"}:
            return EmbeddedProvider()
        if self.provider_name in {"self_hosted", "independent", "meshmend"}:
            return SelfHostedProvider()
        if self.provider_name != "meshy":
            raise Hosted3DError(f"Unsupported hosted 3D provider: {self.provider_name}")
        return MeshyProvider()

    def generate_text(self, prompt: str, *, quality: str = "standard") -> HostedGenerationResult:
        return self.provider().generate_text_to_3d(prompt, quality=quality)

    def generate_image(self, image_path: Path, prompt: str = "", *, quality: str = "high") -> HostedGenerationResult:
        return self.provider().generate_image_to_3d(image_path, prompt, quality=quality)


hosted_3d_service = Hosted3DService()
