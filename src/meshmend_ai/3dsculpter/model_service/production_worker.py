from __future__ import annotations

import argparse
import base64
import inspect
import json
import mimetypes
import os
import re
import shlex
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any


SUPPORTED_MODEL_SUFFIXES = {".stl", ".glb", ".obj", ".ply", ".3mf", ".fbx", ".usdz"}


def production_quality_requested(request: dict[str, Any]) -> bool:
    quality = str(request.get("quality") or "standard").lower()
    prompt = str(request.get("prompt") or "").lower()
    return quality == "high" or any(
        term in prompt
        for term in (
            "8k",
            "8 k",
            "studio",
            "studio quality",
            "studio-quality",
            "studio level",
            "studio-level",
            "production",
            "display quality",
            "maximum detail",
            "store quality",
            "store-quality",
            "store level",
            "store-level",
            "intricate",
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="MeshMend production 3D model worker")
    parser.add_argument("--input", required=True, help="Path to request JSON")
    parser.add_argument("--output-dir", required=True, help="Directory where model outputs should be written")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        write_progress(output_dir, 6, "loading_request", "Loading generation request")
        request = json.loads(input_path.read_text(encoding="utf-8"))
        workflow = str(request.get("workflow") or "text_to_3d")
        engine = effective_production_engine(request)
        write_progress(output_dir, 8, "selecting_engine", f"Using {engine} backend")
        if engine in {"external", "command"}:
            result = run_external_engine(request, input_path, output_dir, workflow)
        elif engine in {"meshmend_sculpt", "native_sculpt", "sculpt"}:
            result = run_meshmend_native_sculpt(request, output_dir, workflow)
        elif engine in {"meshmend_native", "native", "embedded_native"}:
            result = run_meshmend_native(request, output_dir, workflow)
        elif engine in {"free_local", "free_local_hunyuan", "hunyuan", "hunyuan3d"}:
            result = run_free_local_hunyuan(request, input_path, output_dir, workflow)
        elif engine in {"legacy_sculptor", "embedded"}:
            if production_quality_requested(request) and os.environ.get("MESHMEND_ALLOW_LEGACY_STORE_QUALITY", "0").strip().lower() not in {"1", "true", "yes"}:
                raise RuntimeError("Legacy sculptor is disabled for store-quality requests because it produces procedural/blocky drafts.")
            result = run_legacy_sculptor(request, output_dir)
        else:
            raise RuntimeError(f"Unsupported MESHMEND_PRODUCTION_ENGINE: {engine}")
        (output_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        write_progress(output_dir, 99, "complete", "Worker completed model generation")
        print(json.dumps(result))
        return 0
    except Exception as exc:
        diagnostics = build_worker_diagnostics(output_dir, input_path, exc)
        try:
            (output_dir / "worker_diagnostics.json").write_text(json.dumps(diagnostics, indent=2, default=str), encoding="utf-8")
            (output_dir / "worker_traceback.txt").write_text(diagnostics.get("traceback", ""), encoding="utf-8")
        except Exception:
            pass
        error = {
            "error": str(exc),
            "error_type": type(exc).__name__,
            "diagnostics_file": "worker_diagnostics.json",
            "traceback_file": "worker_traceback.txt",
            "hint": production_setup_hint(),
        }
        (output_dir / "result.json").write_text(json.dumps(error, indent=2), encoding="utf-8")
        write_progress(output_dir, 100, "failed", str(exc))
        print(json.dumps(error), file=sys.stderr)
        return 1


def effective_production_engine(request: dict[str, Any]) -> str:
    """Choose the actual worker engine, protecting store-quality jobs from Hunyuan drift.

    Native/Hunyuan paths can produce draft/procedural or experimental meshes,
    but they are not certified high-detail miniature sculpt generators. Store-
    quality requests must use an explicitly certified production backend instead
    of silently falling back to generic native scaffolds or Hunyuan repair.
    """
    engine = os.environ.get("MESHMEND_PRODUCTION_ENGINE", "meshmend_native").strip().lower()
    if production_quality_requested(request) and not (certified_store_quality_engine(engine) or experimental_high_detail_engine(engine)):
        raise RuntimeError(store_quality_unavailable_message(engine))
    hunyuan_engines = {"free_local", "free_local_hunyuan", "hunyuan", "hunyuan3d"}
    force_hunyuan = os.environ.get("MESHMEND_FORCE_HUNYUAN_PRIMARY", "0").strip().lower() in {"1", "true", "yes", "on"}
    allow_hunyuan_store = os.environ.get("MESHMEND_ALLOW_HUNYUAN_STORE_QUALITY", "0").strip().lower() in {"1", "true", "yes", "on"}
    if engine in hunyuan_engines and production_quality_requested(request) and not (force_hunyuan or allow_hunyuan_store):
        return "meshmend_native"
    return engine


def certified_store_quality_engine(engine: str) -> bool:
    engine = engine.strip().lower()
    if engine in {"external", "command"}:
        return os.environ.get("MESHMEND_EXTERNAL_STORE_QUALITY_CERTIFIED", "0").strip().lower() in {"1", "true", "yes", "on"}
    if engine in {"free_local", "free_local_hunyuan", "hunyuan", "hunyuan3d"}:
        return os.environ.get("MESHMEND_ALLOW_HUNYUAN_STORE_QUALITY", "0").strip().lower() in {"1", "true", "yes", "on"}
    if engine in {"meshmend_native", "native", "embedded_native"}:
        return os.environ.get("MESHMEND_ALLOW_NATIVE_STORE_QUALITY", "0").strip().lower() in {"1", "true", "yes", "on"}
    return False


def experimental_high_detail_engine(engine: str) -> bool:
    engine = engine.strip().lower()
    if engine not in {"meshmend_sculpt", "native_sculpt", "sculpt"}:
        return False
    allow_experimental = os.environ.get("MESHMEND_ALLOW_EXPERIMENTAL_SCULPT_HIGH_DETAIL", "0").strip().lower() in {"1", "true", "yes", "on"}
    allow_uncertified_store = os.environ.get("MESHMEND_ALLOW_UNCERTIFIED_STORE_QUALITY_OUTPUT", "0").strip().lower() in {"1", "true", "yes", "on"}
    return allow_experimental and allow_uncertified_store


def store_quality_unavailable_message(engine: str) -> str:
    if engine.strip().lower() in {"meshmend_sculpt", "native_sculpt", "sculpt"}:
        return (
            "MeshMend native sculpt is experimental and is not certified for store/studio-quality output yet. "
            "Configure a certified external generator with MESHMEND_PRODUCTION_ENGINE=external and "
            "MESHMEND_EXTERNAL_STORE_QUALITY_CERTIFIED=1. For local debugging only, set both "
            "MESHMEND_ALLOW_EXPERIMENTAL_SCULPT_HIGH_DETAIL=1 and MESHMEND_ALLOW_UNCERTIFIED_STORE_QUALITY_OUTPUT=1; "
            "those results remain marked store_quality_certified=false."
        )
    return (
        "Store/studio-quality 8K miniature generation is not available with the configured backend. "
        f"Configured engine '{engine or 'meshmend_native'}' can produce draft/procedural or experimental meshes, "
        "but it is not a certified high-detail miniature sculpt generator. Configure an external certified production "
        "runner and set MESHMEND_PRODUCTION_ENGINE=external plus MESHMEND_EXTERNAL_STORE_QUALITY_CERTIFIED=1, "
        "or request standard/draft quality."
    )


def build_worker_diagnostics(output_dir: Path, input_path: Path, exc: Exception) -> dict[str, Any]:
    """Capture enough failure context to debug without rerunning blindly."""
    request: dict[str, Any] = {}
    try:
        request = json.loads(input_path.read_text(encoding="utf-8"))
    except Exception:
        request = {}
    progress = read_json_if_exists(output_dir / "progress.json")
    files = []
    try:
        for path in sorted(output_dir.iterdir(), key=lambda item: item.stat().st_mtime):
            if path.is_file():
                files.append(
                    {
                        "name": path.name,
                        "size": path.stat().st_size,
                        "modified": path.stat().st_mtime,
                    }
                )
    except Exception:
        files = []
    interesting_json = {}
    for pattern in (
        "concept_validation_metrics*.json",
        "hunyuan_generation_kwargs*.json",
        "progress.json",
        "result.json",
    ):
        try:
            for path in output_dir.glob(pattern):
                interesting_json[path.name] = read_json_if_exists(path)
        except Exception:
            pass
    return {
        "error": str(exc),
        "error_type": type(exc).__name__,
        "traceback": traceback.format_exc(),
        "workflow": request.get("workflow"),
        "quality": request.get("quality"),
        "target_polycount": request.get("target_polycount"),
        "has_image_data_uri": bool(request.get("image_data_uri")),
        "prompt_preview": str(request.get("prompt") or "")[:500],
        "progress": progress,
        "output_files": files,
        "artifacts": interesting_json,
        "environment": {
            "MESHMEND_PRODUCTION_ENGINE": os.environ.get("MESHMEND_PRODUCTION_ENGINE", ""),
            "MESHMEND_HUNYUAN3D_PATH": os.environ.get("MESHMEND_HUNYUAN3D_PATH", ""),
            "MESHMEND_HUNYUAN3D_MODEL": os.environ.get("MESHMEND_HUNYUAN3D_MODEL", ""),
            "MESHMEND_HUNYUAN3D_SUBFOLDER": os.environ.get("MESHMEND_HUNYUAN3D_SUBFOLDER", ""),
            "MESHMEND_USE_SDXL_CONCEPT": os.environ.get("MESHMEND_USE_SDXL_CONCEPT", ""),
            "MESHMEND_FREE_LOCAL_IMAGE_MODEL": os.environ.get("MESHMEND_FREE_LOCAL_IMAGE_MODEL", ""),
            "MESHMEND_FREE_LOCAL_IMAGE_SIZE": os.environ.get("MESHMEND_FREE_LOCAL_IMAGE_SIZE", ""),
            "MESHMEND_FREE_LOCAL_IMAGE_STEPS": os.environ.get("MESHMEND_FREE_LOCAL_IMAGE_STEPS", ""),
            "MESHMEND_CONCEPT_CANDIDATES": os.environ.get("MESHMEND_CONCEPT_CANDIDATES", ""),
            "MESHMEND_MAX_CONCEPT_CANDIDATES": os.environ.get("MESHMEND_MAX_CONCEPT_CANDIDATES", ""),
            "MESHMEND_CONCEPT_MIN_QUALITY_SCORE": os.environ.get("MESHMEND_CONCEPT_MIN_QUALITY_SCORE", ""),
            "MESHMEND_HUNYUAN3D_QUALITY_ATTEMPTS": os.environ.get("MESHMEND_HUNYUAN3D_QUALITY_ATTEMPTS", ""),
        },
    }


def read_json_if_exists(path: Path) -> Any:
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"_read_error": str(exc)}
    return None


def write_progress(output_dir: Path, progress: int, stage: str, message: str, **extra: Any) -> None:
    """Write progress for the model service poller and UI progress bar."""
    try:
        payload = {
            "progress": int(max(0, min(100, progress))),
            "stage": stage,
            "message": message,
            "updated_at": time.time(),
        }
        payload.update(extra)
        tmp = output_dir / "progress.tmp.json"
        final = output_dir / "progress.json"
        tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        tmp.replace(final)
    except Exception:
        pass


def run_external_engine(request: dict[str, Any], input_path: Path, output_dir: Path, workflow: str) -> dict[str, Any]:
    command_template = os.environ.get(
        "MESHMEND_PRODUCTION_IMAGE_TO_3D_COMMAND" if workflow == "image_to_3d" else "MESHMEND_PRODUCTION_TEXT_TO_3D_COMMAND",
        "",
    ).strip()
    if not command_template:
        raise RuntimeError(
            "No production model runner is configured. Set MESHMEND_PRODUCTION_TEXT_TO_3D_COMMAND "
            "and/or MESHMEND_PRODUCTION_IMAGE_TO_3D_COMMAND to a local generative 3D pipeline command."
        )

    image_path = ""
    if workflow == "image_to_3d" and request.get("image_data_uri"):
        image_path = str(write_image_data_uri(str(request["image_data_uri"]), output_dir / "input_image.png"))

    prompt_path = output_dir / "prompt.txt"
    prompt_path.write_text(str(request.get("prompt") or ""), encoding="utf-8")
    command = command_template.format(
        input_json=str(input_path),
        output_dir=str(output_dir),
        prompt=shlex.quote(str(request.get("prompt") or "")),
        prompt_path=str(prompt_path),
        image_path=image_path,
        quality=str(request.get("quality") or "standard"),
        target_polycount=str(request.get("target_polycount") or ""),
    )
    completed = subprocess.run(
        command_args(command),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=float(os.environ.get("MESHMEND_PRODUCTION_COMMAND_TIMEOUT_SECONDS", "7200")),
    )
    if completed.returncode != 0:
        failed_result = load_result_from_output(output_dir, completed.stdout)
        if failed_result.get("error"):
            raise RuntimeError(str(failed_result.get("error")))
        raise RuntimeError(compact_command_error(completed.stderr, completed.stdout, completed.returncode))
    result = load_result_from_output(output_dir, completed.stdout)
    if not result.get("model_file") and not result.get("model_urls"):
        raise RuntimeError("production command completed but did not produce a supported model file or result.json")
    if production_quality_requested(request):
        validate_store_quality_external_result(result, output_dir, request)
    return result


def validate_store_quality_external_result(result: dict[str, Any], output_dir: Path, request: dict[str, Any]) -> None:
    """Validate the certified external backend contract before accepting output."""
    if not bool(result.get("store_quality_certified")):
        raise RuntimeError("Certified production runner did not return store_quality_certified=true")
    score_issues = certified_quality_score_issues(result)
    if score_issues:
        raise RuntimeError("Certified production runner did not meet store-quality score contract: " + "; ".join(score_issues))
    model_file = str(result.get("model_file") or "").strip()
    if not model_file:
        raise RuntimeError("Certified production runner must return a local model_file for validation")
    model_path = output_dir / Path(model_file).name
    if not model_path.exists():
        raise RuntimeError(f"Certified production runner reported missing model_file: {model_file}")
    if model_path.suffix.lower() not in SUPPORTED_MODEL_SUFFIXES:
        raise RuntimeError(f"Certified production runner produced unsupported model format: {model_path.suffix}")
    try:
        import numpy as np
        import trimesh

        mesh = trimesh.load(model_path, force="mesh", process=False)
        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
        if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
            raise RuntimeError("model is empty or not a mesh")
        if not bool(mesh.is_watertight):
            cleaned = trimesh.load(model_path, force="mesh", process=True)
            if isinstance(cleaned, trimesh.Scene):
                cleaned = trimesh.util.concatenate(tuple(cleaned.geometry.values()))
            if isinstance(cleaned, trimesh.Trimesh) and len(cleaned.faces) > 0:
                try:
                    cleaned.merge_vertices()
                    cleaned.remove_degenerate_faces()
                    cleaned.remove_duplicate_faces()
                    cleaned.remove_unreferenced_vertices()
                    cleaned.fix_normals()
                    cleaned.fill_holes()
                except Exception:
                    pass
                if bool(cleaned.is_watertight):
                    mesh = cleaned
                    export_mesh(mesh, model_path)
        target_faces = certified_validation_face_target(result, request)
        min_faces = int(float(os.environ.get("MESHMEND_CERTIFIED_MIN_FACE_RATIO", "0.75")) * target_faces)
        face_tolerance = float(os.environ.get("MESHMEND_CERTIFIED_FACE_TARGET_TOLERANCE", "0.005"))
        effective_min_faces = int(min_faces * max(0.0, 1.0 - face_tolerance))
        if len(mesh.faces) < effective_min_faces:
            raise RuntimeError(f"certified mesh below face target: {len(mesh.faces)} < {effective_min_faces}")
        if not bool(mesh.is_watertight):
            raise RuntimeError("certified mesh is not watertight")
        components = len([part for part in mesh.split(only_watertight=False) if len(part.faces) > 20])
        max_components = int(os.environ.get("MESHMEND_CERTIFIED_MAX_COMPONENTS", "3"))
        if components > max_components:
            raise RuntimeError(f"certified mesh has too many components: {components} > {max_components}")
        extents = np.maximum(np.asarray(mesh.extents, dtype=float), 1e-6)
        if float(extents.min() / extents.max()) < float(os.environ.get("MESHMEND_CERTIFIED_MIN_DEPTH_RATIO", "0.18")):
            raise RuntimeError("certified mesh is too flat for miniature output")
        scale_mm = certified_requested_scale_mm(request)
        max_extent = float(extents.max())
        min_allowed_extent = scale_mm * float(os.environ.get("MESHMEND_CERTIFIED_MIN_SCALE_RATIO", "0.35"))
        max_allowed_extent = scale_mm * float(os.environ.get("MESHMEND_CERTIFIED_MAX_SCALE_RATIO", "4.0"))
        if max_extent < min_allowed_extent or max_extent > max_allowed_extent:
            raise RuntimeError(
                f"certified mesh has implausible miniature scale: max extent {max_extent:.2f}mm for requested scale {scale_mm:.2f}mm"
            )
        result.setdefault("capability_tier", "certified_store_quality_external")
        result.setdefault("geometry_source", "certified_external_3d_generator")
        result.setdefault("mesh_info", {})
        if isinstance(result["mesh_info"], dict):
            result.setdefault("store_quality_scores", certified_quality_scores(result))
            result["mesh_info"].update(
                {
                    "store_quality_certified": True,
                    "validated_by_meshmend": True,
                    "faces": int(len(mesh.faces)),
                    "vertices": int(len(mesh.vertices)),
                    "components": int(components),
                    "watertight": bool(mesh.is_watertight),
                    "extents_mm": [float(value) for value in extents],
                }
            )
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError(f"Could not validate certified production mesh: {exc}") from exc


def certified_validation_face_target(result: dict[str, Any], request: dict[str, Any]) -> int:
    target_faces = int(request.get("target_polycount") or 2_000_000)
    provider = str(result.get("provider") or "").lower()
    geometry_source = str(result.get("geometry_source") or "").lower()
    local_no_api = "no_api" in provider or "local_no_api" in geometry_source
    if not local_no_api:
        return target_faces
    mesh_info = result.get("mesh_info") or {}
    if isinstance(mesh_info, dict):
        try:
            detail_target = int(float(mesh_info.get("detail_faces_target") or 0))
        except (TypeError, ValueError):
            detail_target = 0
        if detail_target > 0:
            cap = int(os.environ.get("MESHMEND_LOCAL_EXTERNAL_MAX_CERT_FACE_TARGET", "1500000"))
            return max(1_200_000, min(target_faces, detail_target, cap))
    return target_faces


def compact_command_error(stderr: str, stdout: str, returncode: int) -> str:
    for stream in (stderr, stdout):
        text = (stream or "").strip()
        if not text:
            continue
        json_start = text.rfind('{"error"')
        if json_start >= 0:
            try:
                payload = json.loads(text[json_start:])
                if isinstance(payload, dict) and payload.get("error"):
                    return str(payload["error"])
            except Exception:
                pass
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        signal_lines = [line for line in lines if "error" in line.lower() or "failed" in line.lower()]
        if signal_lines:
            return signal_lines[-1][-2000:]
        if lines:
            return lines[-1][-2000:]
    return f"production command exited {returncode}"


def certified_quality_score_issues(result: dict[str, Any]) -> list[str]:
    if os.environ.get("MESHMEND_REQUIRE_EXTERNAL_QUALITY_SCORES", "1").strip().lower() in {"0", "false", "no", "off"}:
        return []
    scores = certified_quality_scores(result)
    min_score = float(os.environ.get("MESHMEND_CERTIFIED_MIN_QUALITY_SCORE", "0.80"))
    required = (
        "semantic_fidelity_score",
        "anatomy_score",
        "detail_density_score",
        "surface_finish_score",
        "printability_score",
    )
    issues: list[str] = []
    for key in required:
        if key not in scores:
            issues.append(f"missing_{key}")
            continue
        try:
            value = float(scores[key])
        except (TypeError, ValueError):
            issues.append(f"invalid_{key}:{scores[key]!r}")
            continue
        if value < min_score:
            issues.append(f"{key}_below_min:{value:.2f}<{min_score:.2f}")
    certifier = str(scores.get("certifier") or result.get("certifier") or "").strip()
    if not certifier:
        issues.append("missing_certifier")
    return issues


def certified_quality_scores(result: dict[str, Any]) -> dict[str, Any]:
    scores = result.get("store_quality_scores") or result.get("quality_scores") or {}
    if not isinstance(scores, dict):
        scores = {}
    mesh_info = result.get("mesh_info") or {}
    if isinstance(mesh_info, dict):
        for key in (
            "semantic_fidelity_score",
            "anatomy_score",
            "detail_density_score",
            "surface_finish_score",
            "printability_score",
            "certifier",
        ):
            if key not in scores and key in mesh_info:
                scores[key] = mesh_info[key]
    return dict(scores)


def certified_requested_scale_mm(request: dict[str, Any]) -> float:
    for value in (request.get("scale_mm"), request.get("scale")):
        if value:
            try:
                return float(str(value).lower().replace("mm", "").strip())
            except ValueError:
                pass
    match = re.search(r"\b(15|20|25|28|30|32|35|40|48|54|75|90|100)\s*mm\b", str(request.get("prompt") or "").lower())
    if match:
        return float(match.group(1))
    return float(os.environ.get("MESHMEND_DEFAULT_MINIATURE_SCALE_MM", "32"))


def run_meshmend_native(request: dict[str, Any], output_dir: Path, workflow: str) -> dict[str, Any]:
    """Generate MeshMend-controlled production geometry without Hunyuan as source."""
    write_progress(output_dir, 10, "native_start", "Generating MeshMend-owned miniature scaffold")
    image_path = None
    if workflow == "image_to_3d" and request.get("image_data_uri"):
        image_path = write_image_data_uri(str(request["image_data_uri"]), output_dir / "input_image.png")
    try:
        from native_generation import generate_native_miniature
    except Exception as exc:
        raise RuntimeError(f"MeshMend native generator is unavailable: {exc}") from exc

    write_progress(output_dir, 28, "native_polygon_generation", "Building controlled 3D body, limbs, base, and sculpt details")
    mesh, native_report = generate_native_miniature(request, image_path, output_dir)
    if production_quality_requested(request) and not native_report.production_ready:
        raise RuntimeError("MeshMend native generation did not meet store-quality definition gate: " + "; ".join(native_report.issues))
    output_format = os.environ.get("MESHMEND_NATIVE_OUTPUT_FORMAT", "stl").strip().lower().lstrip(".") or "stl"
    if f".{output_format}" not in SUPPORTED_MODEL_SUFFIXES:
        output_format = "stl"
    model_file = output_dir / f"meshmend_native.{output_format}"
    write_progress(output_dir, 92, "native_exporting", f"Exporting MeshMend-native {output_format.upper()} model")
    export_mesh(mesh, model_file)
    if not model_file.exists():
        raise RuntimeError("MeshMend native generator completed but no mesh file was exported")
    mesh_info = mesh_export_info(mesh, request)
    return {
        "model_file": model_file.name,
        "model_format": output_format,
        "provider": "meshmend_native",
        "geometry_source": "meshmend_backend_controlled_polygons",
        "capability_tier": "procedural_printable_draft",
        "store_quality_certified": False,
        "store_quality_blockers": [
            "procedural archetype geometry",
            "no certified prompt-specific high-detail sculpt generator configured",
            "native topology checks prove printability, not commercial miniature artistry",
        ],
        "source_image": image_path.name if image_path is not None else None,
        "workflow": workflow,
        "mesh_info": native_report.to_dict() | {"printable_topology_ready": native_report.production_ready, "store_quality_certified": False, "export_info": mesh_info},
        "consumed_credits": 0,
    }


def run_meshmend_native_sculpt(request: dict[str, Any], output_dir: Path, workflow: str) -> dict[str, Any]:
    """Run the experimental MiniatureSpec -> rig -> sculpt native pipeline."""
    write_progress(output_dir, 10, "sculpt_spec", "Parsing miniature spec and sculpt plan")
    image_path = None
    if workflow == "image_to_3d" and request.get("image_data_uri"):
        image_path = write_image_data_uri(str(request["image_data_uri"]), output_dir / "input_image.png")
    try:
        from native_sculpt_backend import generate_sculpted_miniature
    except Exception as exc:
        raise RuntimeError(f"MeshMend native sculpt backend is unavailable: {exc}") from exc
    write_progress(output_dir, 35, "sculpt_rig", "Building anatomy, armor, gear, and detail hierarchy")
    mesh, report = generate_sculpted_miniature(request, image_path, output_dir)
    output_format = os.environ.get("MESHMEND_NATIVE_SCULPT_OUTPUT_FORMAT", "stl").strip().lower().lstrip(".") or "stl"
    if f".{output_format}" not in SUPPORTED_MODEL_SUFFIXES:
        output_format = "stl"
    model_file = output_dir / f"meshmend_native_sculpt.{output_format}"
    write_progress(output_dir, 92, "sculpt_export", f"Exporting experimental native sculpt {output_format.upper()} model")
    export_mesh(mesh, model_file)
    mesh_info = mesh_export_info(mesh, request)
    return {
        "model_file": model_file.name,
        "model_format": output_format,
        "provider": "meshmend_native_sculpt",
        "geometry_source": "meshmend_miniature_spec_rig_sculpt_pipeline",
        "capability_tier": report.capability_tier,
        "store_quality_certified": False,
        "store_quality_blockers": report.blockers,
        "source_image": image_path.name if image_path is not None else None,
        "workflow": workflow,
        "mesh_info": report.to_dict() | {"export_info": mesh_info},
        "consumed_credits": 0,
    }


def run_free_local_hunyuan(request: dict[str, Any], input_path: Path, output_dir: Path, workflow: str) -> dict[str, Any]:
    """Run a no-API local Hunyuan3D backend.

    Hunyuan3D-2/2.1 is primarily image-to-3D. For text prompts, MeshMend first
    creates a local concept image, then sends that image through Hunyuan3D.
    Everything runs on the user's machine; no hosted API key is used.
    """
    write_progress(output_dir, 10, "preparing_reference", "Preparing image/reference for local Hunyuan3D")
    image_path = None
    if workflow == "image_to_3d" and request.get("image_data_uri"):
        image_path = write_image_data_uri(str(request["image_data_uri"]), output_dir / "input_image.png")
        prepare_image_reference = os.environ.get("MESHMEND_PREPARE_IMAGE_TO_3D_REFERENCE", "1").strip().lower() not in {"0", "false", "no"}
        crop_image_reference = os.environ.get("MESHMEND_CROP_IMAGE_TO_3D_REFERENCE", "0").strip().lower() in {"1", "true", "yes"}
        if prepare_image_reference or crop_image_reference:
            isolated = prepare_hunyuan_reference_image(image_path, output_dir / "input_subject.png", use_rembg=prepare_image_reference)
            if isolated is not None:
                image_path = isolated
        else:
            resized = resize_reference_image_for_hunyuan(image_path, output_dir / "input_reference.png")
            if resized is not None:
                image_path = resized
    if image_path is None:
        write_progress(output_dir, 12, "concept_start", "Generating text concept images")
        image_path = generate_local_concept_image(request, input_path, output_dir)
        request = dict(request)
        request["_meshmend_generated_text_concept"] = True
        source_concept_metrics = read_json_if_exists(output_dir / "concept_validation_metrics.json") or {}
        write_progress(output_dir, 28, "concept_ready", "Concept image selected; isolating subject")
        isolated = prepare_hunyuan_reference_image(image_path, output_dir / "concept_subject.png")
        if isolated is not None:
            image_path = isolated
        if reference_image_likely_card(image_path):
            raise RuntimeError(
                "Prepared Hunyuan reference still contains a large rectangular/card background. "
                "Refusing to generate a square STL; retry with a cleaner concept or provide an image reference."
            )
        reference_metrics = concept_validation_metrics(image_path)
        (output_dir / "concept_subject_validation_metrics.json").write_text(json.dumps(reference_metrics, indent=2), encoding="utf-8")
        source_concept_passed = bool(source_concept_metrics.get("passed", False))
        prepared_has_hard_failure = any(
            bool(reference_metrics.get(flag, False))
            for flag in ("likely_partial_body", "likely_base_band", "likely_background_panel", "likely_clipped_subject")
        )
        if strict_quality_requested_like(request) and not reference_metrics.get("passed", False) and not (
            source_concept_passed and not prepared_has_hard_failure
        ):
            raise RuntimeError(
                "Prepared Hunyuan reference is not a complete full-body miniature concept. "
                "Refusing to run image-to-3D because this would produce another noisy/partial STL. "
                "Use a full-body image reference or simplify the prompt. Details: " + json.dumps(reference_metrics)
            )
    return run_hunyuan_image_to_3d(image_path, request, output_dir)


def prepare_hunyuan_reference_image(input_path: Path, output_path: Path, *, use_rembg: bool = True) -> Path | None:
    """Prepare a tight single-subject RGB reference for Hunyuan.

    Hunyuan often reconstructs image padding as a flat square/card. This keeps
    only the subject bounding box, gives it a small margin, and writes an RGB
    square where the subject fills most of the frame.
    """
    try:
        from PIL import Image, ImageFilter
        import numpy as np
        from scipy.ndimage import binary_closing, binary_fill_holes, binary_opening, label

        quality = os.environ.get("MESHMEND_REFERENCE_IMAGE_QUALITY", "high").strip().lower()
        max_size = int(os.environ.get("MESHMEND_REFERENCE_IMAGE_SIZE", "1280" if quality == "high" else "768"))
        image = Image.open(input_path).convert("RGBA")
        if use_rembg:
            image = remove_reference_background_for_hunyuan(image)
        image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        rgba = np.asarray(image, dtype=np.float32) / 255.0
        rgb = rgba[:, :, :3]
        alpha = rgba[:, :, 3]
        h, w = alpha.shape
        gray = rgb @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
        has_transparent_alpha = np.mean(alpha < 0.98) > 0.01
        if has_transparent_alpha:
            mask = alpha > 0.08
        else:
            # Product renders should be on a light background. Treat non-light,
            # high-contrast pixels as subject and ignore the square canvas.
            border = np.concatenate([rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]], axis=0)
            bg = np.median(border, axis=0)
            color_dist = np.linalg.norm(rgb - bg[None, None, :], axis=2)
            saturation = rgb.max(axis=2) - rgb.min(axis=2)
            gray_cutoff = min(0.72, float(np.quantile(gray, 0.58)))
            color_cutoff = max(0.16, float(np.quantile(color_dist, 0.84)))
            # Light gray rectangular render sheets differ from a white border by
            # color distance, but they are still background. Require either dark
            # subject pixels or strong color difference that is not also a light
            # neutral backdrop.
            mask = (gray < gray_cutoff) | ((color_dist > color_cutoff) & (saturation > 0.055) & (gray < 0.86))

        mask = binary_fill_holes(binary_closing(binary_opening(mask, iterations=1), iterations=2))
        labeled, count = label(mask)
        if count <= 0:
            return None
        yy, xx = np.indices((h, w))
        center_x, center_y = w * 0.5, h * 0.52
        best_label = max(
            range(1, count + 1),
            key=lambda idx: int((labeled == idx).sum())
            * (1.0 - min((((float(xx[labeled == idx].mean()) - center_x) / w) ** 2 + ((float(yy[labeled == idx].mean()) - center_y) / h) ** 2) ** 0.5, 0.8))
            if int((labeled == idx).sum()) > h * w * 0.004
            else 0,
        )
        subject_with_accessories = keep_subject_and_accessories(labeled, best_label)
        subject = remove_broad_base_from_subject_mask(subject_with_accessories)
        removed_broad_base = int(subject.sum()) < int(subject_with_accessories.sum()) * 0.98
        rows, cols = np.where(subject)
        if rows.size == 0 or cols.size == 0:
            return None
        pad_x = max(4, int((cols.max() - cols.min() + 1) * 0.085))
        pad_y = max(4, int((rows.max() - rows.min() + 1) * 0.085))
        # Keep extra bottom context even after rembg/alpha matting. Generated
        # tabletop concepts often have boots/base in a faint halo that gets
        # under-segmented; tight crops then feed Hunyuan clipped feet and create
        # flat, janky bottoms.
        bottom_pad_y = max(pad_y, int((rows.max() - rows.min() + 1) * 0.20))
        left = max(0, int(cols.min()) - pad_x)
        right = min(w, int(cols.max()) + pad_x + 1)
        top = max(0, int(rows.min()) - pad_y)
        bottom = min(h, int(rows.max()) + bottom_pad_y + 1)

        original_crop = image.crop((left, top, right, bottom)).convert("RGBA")
        cropped = original_crop.copy()
        alpha_crop = (subject[top:bottom, left:right].astype(np.uint8) * 255)
        alpha_image = Image.fromarray(alpha_crop, mode="L").filter(ImageFilter.GaussianBlur(radius=0.5))
        cropped.putalpha(alpha_image)
        keep_rectangular = os.environ.get("MESHMEND_KEEP_RECTANGULAR_HUNYUAN_REFERENCE", "1").strip().lower() not in {"0", "false", "no"}
        if keep_rectangular:
            # Do not place a tall humanoid inside a square card. Hunyuan accepts
            # image paths and can resize internally; feeding a tight rectangular
            # crop avoids reconstructing square padding as a plaque. Keep a
            # small light margin, though: edge-clipped subjects turn into dotty,
            # semi-open meshes because Hunyuan has to hallucinate the missing
            # silhouette.
            margin = max(8, int(max(cropped.size) * float(os.environ.get("MESHMEND_REFERENCE_SAFE_MARGIN", "0.10"))))
            # Hunyuan reconstructs opaque render backdrops as square cards/slabs.
            # Use the segmentation mask as alpha by default so the worker sees an
            # isolated miniature, not the generated studio background. Users can
            # still opt into opaque crops if a specific Hunyuan build mishandles
            # alpha, but concept fidelity is better with transparency here.
            preserve_opaque_crop = os.environ.get("MESHMEND_PRESERVE_OPAQUE_REFERENCE_CROP", "0").strip().lower() in {"1", "true", "yes"}
            paste_crop = original_crop if preserve_opaque_crop and not has_transparent_alpha and not removed_broad_base else cropped
            padded = Image.new("RGBA", (paste_crop.size[0] + margin * 2, paste_crop.size[1] + margin * 2), (245, 245, 245, 0))
            padded.paste(paste_crop, (margin, margin), None if paste_crop is original_crop else paste_crop)
            cropped = padded
            cropped.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            if os.environ.get("MESHMEND_SHARPEN_HUNYUAN_REFERENCE", "1").strip().lower() in {"1", "true", "yes"}:
                cropped = cropped.filter(ImageFilter.SHARPEN)
            # Feed Hunyuan a tight white RGB product reference by default. Some
            # Hunyuan/PIL paths treat transparent pixels as black, which makes
            # the reconstruction drift away from the concept and lose readable
            # miniature detail. The crop/card guards above keep the white canvas
            # small enough to avoid the old square-card failure mode; alpha is
            # still available as an explicit override.
            use_alpha = os.environ.get("MESHMEND_HUNYUAN_REFERENCE_ALPHA", "0").strip().lower() in {"1", "true", "yes"}
            denoise_reference = should_denoise_hunyuan_reference(input_path)
            if use_alpha:
                if denoise_reference:
                    alpha_channel = cropped.getchannel("A")
                    rgb_only = cropped.convert("RGB").filter(ImageFilter.MedianFilter(size=3)).filter(ImageFilter.SMOOTH_MORE)
                    cropped = rgb_only.convert("RGBA")
                    cropped.putalpha(alpha_channel)
                cropped.save(output_path)
            else:
                rgb_crop = Image.new("RGB", cropped.size, (245, 245, 245))
                rgb_crop.paste(cropped, mask=cropped.getchannel("A"))
                if denoise_reference:
                    rgb_crop = rgb_crop.filter(ImageFilter.MedianFilter(size=3)).filter(ImageFilter.SMOOTH_MORE)
                rgb_crop.save(output_path)
        else:
            subject_max = max(cropped.size)
            canvas_size = int(subject_max / float(os.environ.get("MESHMEND_REFERENCE_SUBJECT_FILL", "0.88")))
            canvas = Image.new("RGBA", (canvas_size, canvas_size), (245, 245, 245, 0))
            canvas.paste(cropped, ((canvas_size - cropped.size[0]) // 2, (canvas_size - cropped.size[1]) // 2), cropped)
            canvas = canvas.resize((max_size, max_size), Image.Resampling.LANCZOS)
            rgb_canvas = Image.new("RGB", canvas.size, (245, 245, 245))
            rgb_canvas.paste(canvas, mask=canvas.getchannel("A"))
            if should_denoise_hunyuan_reference(input_path):
                rgb_canvas = rgb_canvas.filter(ImageFilter.MedianFilter(size=3)).filter(ImageFilter.SMOOTH_MORE)
            rgb_canvas.save(output_path)
        return output_path
    except Exception:
        return preprocess_reference_image_for_3d(input_path, output_path)


def should_denoise_hunyuan_reference(image_path: Path) -> bool:
    """Denoise only genuinely speckled references, not every clean concept.

    The previous default median+SMOOTH_MORE pass made generated concepts look
    cleaner to humans but blurred the silhouette and panel lines that Hunyuan3D
    uses for shape conditioning. That is a bad tradeoff when users already like
    the selected concept: the STL can drift away from it and look like generic
    noise. Keep the reference faithful by default, while retaining an automatic
    escape hatch for visibly noisy concepts and an env override for local setups.
    """
    explicit = os.environ.get("MESHMEND_DENOISE_HUNYUAN_REFERENCE", "").strip().lower()
    if explicit:
        return explicit in {"1", "true", "yes", "on"}
    try:
        metrics = concept_validation_metrics(image_path)
        return bool(metrics.get("likely_noisy_reference", False))
    except Exception:
        return False


def resize_reference_image_for_hunyuan(input_path: Path, output_path: Path) -> Path | None:
    """Resize a real user image without cropping away weapons, capes, mounts, or accessories."""
    try:
        from PIL import Image, ImageFilter

        quality = os.environ.get("MESHMEND_REFERENCE_IMAGE_QUALITY", "high").strip().lower()
        max_size = int(os.environ.get("MESHMEND_REFERENCE_IMAGE_SIZE", "1280" if quality == "high" else "768"))
        image = Image.open(input_path).convert("RGBA")
        image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        image = image.filter(ImageFilter.SHARPEN)
        image.save(output_path)
        return output_path
    except Exception:
        return None


def remove_reference_background_for_hunyuan(image: Any) -> Any:
    """Use Hunyuan/rembg matting to avoid reconstructing concept backdrops.

    Diffusers concepts often include a gray studio card behind the miniature.
    Threshold-based segmentation mistakes that card for subject pixels. Hunyuan's
    repository ships a rembg wrapper; use it before our fallback crop/mask logic
    so the shape generator receives an isolated RGBA subject.
    """
    if os.environ.get("MESHMEND_USE_HUNYUAN_REMBG", "1").strip().lower() in {"0", "false", "no"}:
        return image
    try:
        from hy3dgen.rembg import BackgroundRemover

        remover = BackgroundRemover()
        removed = remover(image.convert("RGB"))
        return removed.convert("RGBA")
    except Exception:
        return image


def reference_image_likely_card(image_path: Path) -> bool:
    """Detect prepared references that will reconstruct as a square/card."""
    try:
        from PIL import Image
        import numpy as np

        image = Image.open(image_path).convert("RGBA")
        rgba = np.asarray(image, dtype=np.float32) / 255.0
        alpha = rgba[:, :, 3]
        opaque = alpha > 0.08
        if int(opaque.sum()) <= 0:
            return True
        rows, cols = np.where(opaque)
        h, w = opaque.shape
        bbox_w = int(cols.max() - cols.min() + 1)
        bbox_h = int(rows.max() - rows.min() + 1)
        bbox_fill = float(opaque[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1].mean())
        alpha_ratio = float(opaque.mean())
        # A humanoid miniature silhouette is sparse/irregular. If alpha fills a
        # broad rectangular bounding box, it is almost certainly a backdrop card.
        # Rembg mattes for humanoids can have a fairly dense bounding box due to
        # halos, weapons, and capes. Only treat alpha shape alone as a card when
        # it fills most of the whole image and most of its own bounding box.
        has_alpha_cutout = float((alpha < 0.98).mean()) > 0.01
        if (
            has_alpha_cutout
            and bbox_w > w * 0.65
            and bbox_h > h * 0.55
            and alpha_ratio > float(os.environ.get("MESHMEND_REFERENCE_CARD_MIN_ALPHA_RATIO", "0.52"))
            and bbox_fill > float(os.environ.get("MESHMEND_REFERENCE_CARD_MAX_ALPHA_FILL", "0.86"))
        ):
            return True

        rgb = rgba[:, :, :3]
        gray = rgb @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
        gx = np.abs(np.diff(gray, axis=1, prepend=gray[:, :1]))
        gy = np.abs(np.diff(gray, axis=0, prepend=gray[:1, :]))
        if concept_has_large_background_panel(rgb, gray, gx + gy) and alpha_ratio > 0.22:
            return True
        return False
    except Exception:
        return False


def keep_subject_and_accessories(labeled: Any, primary_label: int, min_area_ratio: float = 0.025, expansion: float = 0.30) -> Any:
    """Keep the main subject plus nearby weapons/capes/horns that are disconnected in the mask."""
    try:
        import numpy as np

        primary = labeled == primary_label
        rows, cols = np.where(primary)
        if rows.size == 0 or cols.size == 0:
            return primary
        main_area = float(primary.sum())
        top, bottom = int(rows.min()), int(rows.max())
        left, right = int(cols.min()), int(cols.max())
        height = max(1, bottom - top + 1)
        width = max(1, right - left + 1)
        ex_top = top - int(height * expansion)
        ex_bottom = bottom + int(height * expansion)
        ex_left = left - int(width * expansion)
        ex_right = right + int(width * expansion)
        keep = primary.copy()
        for label_id in range(1, int(labeled.max()) + 1):
            if label_id == primary_label:
                continue
            comp = labeled == label_id
            if float(comp.sum()) < main_area * min_area_ratio:
                continue
            comp_rows, comp_cols = np.where(comp)
            if comp_rows.size == 0 or comp_cols.size == 0:
                continue
            center_y = float(comp_rows.mean())
            center_x = float(comp_cols.mean())
            if ex_left <= center_x <= ex_right and ex_top <= center_y <= ex_bottom:
                keep |= comp
        return keep
    except Exception:
        return labeled == primary_label


def remove_broad_base_from_subject_mask(subject: Any) -> Any:
    """Trim generated plinth/terrain/base pixels from a concept subject mask.

    Stable Diffusion often adds a wide display base even when prompted not to.
    Hunyuan then turns that base/ground patch into the giant connected sheet the
    user is seeing. This keeps the character and nearby accessories while cutting
    broad lower bands that are much wider than the mid-body silhouette.
    """
    try:
        import numpy as np

        mask = np.asarray(subject, dtype=bool).copy()
        h, w = mask.shape
        rows, cols = np.where(mask)
        if rows.size == 0 or cols.size == 0:
            return mask
        top = int(rows.min())
        bottom = int(rows.max())
        height = max(1, bottom - top + 1)
        lower_start = top + int(height * 0.66)
        mid_start = top + int(height * 0.24)
        mid_end = top + int(height * 0.58)
        row_width = mask.sum(axis=1).astype(float)
        mid_width = float(np.percentile(row_width[mid_start:max(mid_start + 1, mid_end)], 82.0))
        if mid_width <= 1.0:
            return mask
        # Only cut genuinely broad plinth/terrain bands. Earlier defaults were
        # close to the width of normal boots/capes on humanoid minis, which could
        # amputate the lower legs and then make Hunyuan generate a model that no
        # longer matched the selected concept.
        broad_threshold = max(mid_width * float(os.environ.get("MESHMEND_REFERENCE_BASE_WIDTH_RATIO", "1.35")), w * 0.50)
        broad_rows = np.where(row_width[lower_start : bottom + 1] > broad_threshold)[0]
        if broad_rows.size == 0:
            bottom_band = mask[top + int(height * 0.78) : bottom + 1, :]
            bottom_width_ratio = float((bottom_band.sum(axis=0) > 0).mean()) if bottom_band.size else 0.0
            if bottom_width_ratio <= float(os.environ.get("MESHMEND_REFERENCE_MAX_BOTTOM_WIDTH", "0.55")):
                return mask
            cut_row = top + int(height * float(os.environ.get("MESHMEND_REFERENCE_BASE_FALLBACK_CUT", "0.88")))
        else:
            cut_row = lower_start + int(broad_rows[0])
        # Do not amputate most of the miniature. This is only for broad bottom
        # plinth/terrain bands that start low in the silhouette.
        if cut_row < top + height * 0.58:
            return mask
        trimmed = mask.copy()
        trimmed[cut_row:, :] = False
        if trimmed.sum() < mask.sum() * 0.45:
            return mask
        return trimmed
    except Exception:
        return subject


def preprocess_reference_image_for_3d(input_path: Path, output_path: Path) -> Path | None:
    """Remove flat backgrounds before image-to-3D.

    Hunyuan3D is single-image driven, so a flat image backdrop can become a thin
    rear plane or plaque in the final STL. This step keeps only the largest
    foreground subject, crops it tightly, and writes an alpha-masked square image
    so the 3D modeler sees a miniature subject instead of the photo background.
    """
    try:
        from PIL import Image, ImageFilter
        import numpy as np
        from scipy.ndimage import binary_closing, binary_fill_holes, binary_opening, label

        quality = os.environ.get("MESHMEND_REFERENCE_IMAGE_QUALITY", "high").strip().lower()
        max_size = int(os.environ.get("MESHMEND_REFERENCE_IMAGE_SIZE", "1536" if quality == "high" else "1024"))
        image = Image.open(input_path).convert("RGBA")
        image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        rgba = np.asarray(image, dtype=np.float32) / 255.0
        rgb = rgba[:, :, :3]
        alpha = rgba[:, :, 3]
        h, w = alpha.shape

        if np.mean(alpha < 0.98) > 0.01:
            mask = alpha > 0.08
        else:
            border = np.concatenate([rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]], axis=0)
            bg = np.median(border, axis=0)
            color_dist = np.linalg.norm(rgb - bg[None, None, :], axis=2)
            gray = rgb @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
            # Use both background-color distance and tonal contrast so white,
            # gray, and busy-but-flat backdrops are rejected more aggressively.
            mask = (color_dist > max(0.08, float(np.quantile(color_dist, 0.64)))) | (
                np.abs(gray - np.median(gray[[0, -1], :])) > max(0.10, float(np.std(gray) * 0.45))
            )

        mask = binary_fill_holes(binary_closing(binary_opening(mask, iterations=1), iterations=2))
        labeled, count = label(mask)
        if count <= 0:
            return None
        yy, xx = np.indices((h, w))
        cx, cy = w * 0.5, h * 0.52
        best_label = 0
        best_score = -1.0
        for idx in range(1, count + 1):
            comp = labeled == idx
            area = int(comp.sum())
            if area < h * w * 0.01:
                continue
            mean_x = float(xx[comp].mean())
            mean_y = float(yy[comp].mean())
            center_penalty = (((mean_x - cx) / w) ** 2 + ((mean_y - cy) / h) ** 2) ** 0.5
            score = area * (1.0 - min(center_penalty * 1.5, 0.80))
            if score > best_score:
                best_score = score
                best_label = idx
        if best_label <= 0:
            return None

        subject = labeled == best_label
        rows, cols = np.where(subject)
        pad_x = max(8, int((cols.max() - cols.min() + 1) * 0.10))
        pad_y = max(8, int((rows.max() - rows.min() + 1) * 0.10))
        left = max(0, int(cols.min()) - pad_x)
        right = min(w, int(cols.max()) + pad_x + 1)
        top = max(0, int(rows.min()) - pad_y)
        bottom = min(h, int(rows.max()) + pad_y + 1)

        cropped = image.crop((left, top, right, bottom)).convert("RGBA")
        cropped_mask = subject[top:bottom, left:right]
        alpha_crop = (cropped_mask.astype(np.uint8) * 255)
        alpha_image = Image.fromarray(alpha_crop, mode="L").filter(ImageFilter.GaussianBlur(radius=0.8))
        cropped.putalpha(alpha_image)

        canvas_size = max(cropped.size)
        canvas = Image.new("RGBA", (canvas_size, canvas_size), (255, 255, 255, 0))
        canvas.paste(cropped, ((canvas_size - cropped.size[0]) // 2, (canvas_size - cropped.size[1]) // 2), cropped)
        canvas = canvas.resize((max_size, max_size), Image.Resampling.LANCZOS).filter(ImageFilter.SHARPEN)
        # Hunyuan is more predictable with a stable RGB subject on a light
        # background than with transparent RGBA; keep alpha only for our own
        # segmentation and flatten before the model sees it.
        rgb_canvas = Image.new("RGB", canvas.size, (245, 245, 245))
        rgb_canvas.paste(canvas, mask=canvas.getchannel("A"))
        raw_output = output_path.with_name(f"{output_path.stem}_raw.png")
        rgb_canvas.save(raw_output)
        isolate_single_subject_concept(raw_output, output_path)
        if not output_path.exists():
            raw_output.replace(output_path)
        else:
            try:
                raw_output.unlink()
            except Exception:
                pass
        return output_path
    except Exception:
        return None


def generate_local_concept_image(request: dict[str, Any], input_path: Path, output_dir: Path) -> Path:
    command_template = os.environ.get("MESHMEND_FREE_LOCAL_TEXT_TO_IMAGE_COMMAND", "").strip()
    prompt_path = output_dir / "prompt.txt"
    prompt_path.write_text(str(request.get("prompt") or ""), encoding="utf-8")
    if command_template:
        command = command_template.format(
            input_json=str(input_path),
            output_dir=str(output_dir),
            prompt=shlex.quote(str(request.get("prompt") or "")),
            prompt_path=str(prompt_path),
            quality=str(request.get("quality") or "standard"),
        )
        completed = subprocess.run(
            command_args(command),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=float(os.environ.get("MESHMEND_TEXT_IMAGE_COMMAND_TIMEOUT_SECONDS", "1800")),
        )
        if completed.returncode != 0:
            raise RuntimeError(completed.stderr.strip() or completed.stdout.strip() or "local text-to-image command failed")
        concept = find_first_file(output_dir, {".png", ".jpg", ".jpeg", ".webp"})
        if concept is not None:
            return concept
        raise RuntimeError("local text-to-image command completed but did not write an image into the output directory")

    explicit_image = find_prompt_reference_image(output_dir)
    if explicit_image is not None:
        return explicit_image

    try:
        return generate_diffusers_concept_image(request, output_dir)
    except Exception as exc:
        details = str(exc)
        if "Concept image failed" in details or "Text concept generation could not produce" in details:
            raise RuntimeError(
                "Text-to-3D stopped before Hunyuan because the local concept image was not a complete usable single-subject miniature reference. "
                "This prevents another janky or square STL. Try a cleaner/more specific prompt, provide an image reference, "
                f"or set MESHMEND_CONCEPT_MIN_QUALITY_SCORE lower to override. Details: {details}"
            ) from exc
        raise RuntimeError(
            "Text prompts need a local text-to-image step before Hunyuan3D image-to-3D. "
            "Set MESHMEND_FREE_LOCAL_TEXT_TO_IMAGE_COMMAND to a no-API local image generator, or install/configure diffusers. "
            f"Diffusers fallback failed: {details}"
        ) from exc


def generate_diffusers_concept_image(request: dict[str, Any], output_dir: Path) -> Path:
    from diffusers import StableDiffusionPipeline
    import torch

    quality = str(request.get("quality") or "standard").lower()
    prompt = str(request.get("prompt") or "")
    prompt_lower = prompt.lower()
    wants_studio_detail = production_quality_requested(request)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_sdxl_env = os.environ.get("MESHMEND_USE_SDXL_CONCEPT", "").strip().lower()
    if use_sdxl_env:
        use_sdxl = use_sdxl_env in {"1", "true", "yes"}
    else:
        # SDXL concept generation has repeatedly saturated local systems and
        # appeared stuck around the concept stage. Default to the lighter SD 1.5
        # path unless the user explicitly opts into SDXL.
        use_sdxl = False
    model_id = os.environ.get(
        "MESHMEND_FREE_LOCAL_IMAGE_MODEL",
        "stabilityai/stable-diffusion-xl-base-1.0" if use_sdxl else "runwayml/stable-diffusion-v1-5",
    )
    dtype = torch.float16 if device == "cuda" else torch.float32
    # Remove sculptor's long store-quality backend boilerplate before expanding
    # archetype terms. Otherwise expansions appended by normalize_miniature_prompt
    # sit after the boilerplate and compact_concept_prompt immediately discards
    # them, leaving SD1.5 with only "a space marine" instead of useful anatomy
    # and armor cues.
    compact_prompt = compact_concept_prompt(normalize_miniature_prompt(prompt))
    landmark_prompt = concept_landmark_prompt(prompt)
    miniature_prompt = (
        f"{landmark_prompt}, {compact_prompt}, monochrome unpainted matte graphite clay resin full body tabletop miniature, "
        "all requested landmarks visible, readable weapon across body, rear backpack visible as side silhouette, chest icon relief, skull rubble round base, "
        "helmeted bulky armored soldier silhouette, oversized shoulder pauldrons, backpack power unit, chunky greaves heavy boots, "
        "crisp sculpted armor seams bevels vents hands face and panel lines, head to boots visible, small round base, "
        "centered subject fills frame, modest white border, frontal orthographic product render, pure white background, "
        "no color no paint no weathering, not smooth blank armor, not bust not cropped"
    )
    negative_prompt = (
        "painted armor, white armor, colored armor, color accents, weathering, battle damage, scratches, grunge, dirt, black speckles, stipple, noisy texture, posterized edges, heavy outline, "
        "multiple characters, duplicates, lineup, squad, group, two views, three views, multi view, multiple views, triptych, front and back, rear view, side view, four views, "
        "reference sheet, turnaround, collage, grid, cropped, close-up, zoomed in, cut off head, cut off feet, cropped weapon, "
        "melted weapon, fused hands, fused fingers, malformed hands, malformed weapon, blurry weapon, deformed legs, missing feet, "
        "text, letters, words, logo, watermark, sign, label, nameplate, placard, border, frame, scenery, wall, grey background, gray background, rectangular panel, backdrop, backing card, "
        "display plinth, terrain slab, slab, square card, glossy texture, color noise, speckle, low detail, smooth blob, blocky, holes, malformed"
    )
    pipe = load_text_to_image_pipeline(model_id, dtype)
    pipe = pipe.to(device)
    enable_diffusers_memory_optimizations(pipe, output_dir)
    miniature_prompt = clamp_diffusers_prompt_to_token_limit(pipe, miniature_prompt)
    negative_prompt = clamp_diffusers_prompt_to_token_limit(pipe, negative_prompt)

    steps_default = "36" if use_sdxl else ("32" if wants_studio_detail else "16")
    steps = int(os.environ.get("MESHMEND_FREE_LOCAL_IMAGE_STEPS", steps_default))
    guidance = float(os.environ.get("MESHMEND_FREE_LOCAL_IMAGE_GUIDANCE", "7.0" if "xl" in model_id.lower() else "8.0"))
    size_default = "768" if use_sdxl else ("768" if wants_studio_detail else "512")
    size = int(os.environ.get("MESHMEND_FREE_LOCAL_IMAGE_SIZE", size_default))
    candidates_default = "5" if wants_studio_detail and not use_sdxl else "1"
    candidates = max(1, int(os.environ.get("MESHMEND_CONCEPT_CANDIDATES", candidates_default)))
    # Extra text-to-image candidates can hang for a long time on local GPUs when
    # SDXL is enabled. With the default lightweight SD1.5 path, allow several
    # bounded quality retries for studio requests so bad bust crops do not become
    # an immediate backend failure.
    max_candidates_default = "8" if wants_studio_detail and not use_sdxl else str(candidates)
    max_candidates = max(
        candidates,
        int(os.environ.get("MESHMEND_MAX_CONCEPT_CANDIDATES", max_candidates_default)),
    )
    (output_dir / "concept_generation_settings.json").write_text(
        json.dumps(
            {
                "model_id": model_id,
                "device": device,
                "use_sdxl": use_sdxl,
                "steps": steps,
                "guidance": guidance,
                "size": size,
                "initial_candidates": candidates,
                "max_candidates": max_candidates,
                "prompt": miniature_prompt,
                "negative_prompt": negative_prompt,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    best_path = None
    best_metrics: dict[str, Any] | None = None
    best_score = float("-inf")
    index = 0
    while index < candidates:
        active_prompt = concept_retry_prompt(miniature_prompt, best_metrics) if index > 0 else miniature_prompt
        active_prompt = clamp_diffusers_prompt_to_token_limit(pipe, active_prompt)
        (output_dir / f"concept_prompt_{index + 1}.txt").write_text(active_prompt, encoding="utf-8")
        write_progress(output_dir, 14 + int((index / max(candidates, 1)) * 12), "concept_generating", f"Generating concept candidate {index + 1} of {candidates}")
        generator = None
        seed_text = os.environ.get("MESHMEND_FREE_LOCAL_IMAGE_SEED", "").strip()
        if seed_text:
            generator = torch.Generator(device=device).manual_seed(int(seed_text) + index)
        try:
            result = pipe(
                prompt=active_prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=steps,
                guidance_scale=guidance,
                height=size,
                width=size,
                generator=generator,
            )
        except Exception as exc:
            fallback_disabled = os.environ.get("MESHMEND_DISABLE_CONCEPT_FALLBACK", "0").strip().lower() in {"1", "true", "yes"}
            can_fallback = use_sdxl and not fallback_disabled and not os.environ.get("MESHMEND_FREE_LOCAL_IMAGE_MODEL", "").strip()
            if not can_fallback:
                raise
            (output_dir / "concept_fallback.txt").write_text(
                "SDXL concept decode failed; retried with lighter Stable Diffusion 1.5 settings.\n"
                f"Original error: {exc}\n",
                encoding="utf-8",
            )
            try:
                del pipe
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            use_sdxl = False
            model_id = "runwayml/stable-diffusion-v1-5"
            guidance = float(os.environ.get("MESHMEND_FREE_LOCAL_IMAGE_GUIDANCE", "8.0"))
            steps = min(steps, int(os.environ.get("MESHMEND_FALLBACK_IMAGE_STEPS", "30")))
            size = min(size, int(os.environ.get("MESHMEND_FALLBACK_IMAGE_SIZE", "640")))
            candidates = min(candidates, int(os.environ.get("MESHMEND_FALLBACK_CONCEPT_CANDIDATES", "1")))
            max_candidates = min(max_candidates, int(os.environ.get("MESHMEND_FALLBACK_MAX_CONCEPT_CANDIDATES", "2" if wants_studio_detail else str(candidates))))
            pipe = load_text_to_image_pipeline(model_id, dtype).to(device)
            enable_diffusers_memory_optimizations(pipe, output_dir)
            miniature_prompt = clamp_diffusers_prompt_to_token_limit(pipe, miniature_prompt)
            negative_prompt = clamp_diffusers_prompt_to_token_limit(pipe, negative_prompt)
            index = 0
            best_path = None
            best_metrics = None
            best_score = float("-inf")
            continue
        image = result.images[0]
        concept_path = output_dir / f"concept_{index + 1}.png"
        image.save(concept_path)
        isolated_path = output_dir / f"concept_single_subject_{index + 1}.png"
        isolate_single_subject_concept(concept_path, isolated_path)
        candidate_path = isolated_path if isolated_path.exists() else concept_path
        metrics = concept_validation_metrics(candidate_path)
        if strict_quality_requested_like(request) and not metrics.get("passed", False) and (
            metrics.get("likely_base_band") or metrics.get("likely_background_panel") or metrics.get("likely_clipped_subject")
        ):
            repaired_path = output_dir / f"concept_single_subject_basefixed_{index + 1}.png"
            repaired = prepare_hunyuan_reference_image(candidate_path, repaired_path)
            if repaired is not None and repaired.exists():
                repaired_metrics = concept_validation_metrics(repaired)
                (output_dir / f"concept_validation_metrics_basefixed_{index + 1}.json").write_text(
                    json.dumps(repaired_metrics, indent=2), encoding="utf-8"
                )
                repaired_score = float(repaired_metrics.get("score", 0.0) or 0.0)
                original_score = float(metrics.get("score", 0.0) or 0.0)
                if repaired_metrics.get("passed", False) or (
                    repaired_score >= original_score
                    and not repaired_metrics.get("likely_clipped_subject")
                    and not repaired_metrics.get("likely_background_panel")
                ):
                    candidate_path = repaired
                    metrics = repaired_metrics
        (output_dir / f"concept_validation_metrics_{index + 1}.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        score = concept_quality_score(candidate_path)
        if metrics.get("likely_clipped_subject") and not metrics.get("likely_base_band"):
            score -= 3.0
        foreground_ratio = float(metrics.get("foreground_ratio", 0.0) or 0.0)
        if foreground_ratio > 0.76:
            # Very high foreground fill is usually a zoomed/upper-body crop or a
            # reference-sheet panel. Prefer a slightly imperfect full-body figure
            # over a sharp but unusable bust crop.
            score -= (foreground_ratio - 0.76) * 18.0
        score += 1.0 if metrics.get("passed") else -1.0
        if score > best_score:
            best_score = score
            best_path = candidate_path
            best_metrics = metrics
        index += 1
        if (
            index >= candidates
            and index < max_candidates
            and strict_quality_requested_like(request)
            and (best_metrics is None or not best_metrics.get("passed", False))
        ):
            candidates += 1

    final_path = output_dir / "concept_single_subject.png"
    if best_path is not None:
        write_progress(output_dir, 26, "concept_selecting", "Selecting best single-subject concept")
        final_path.write_bytes(Path(best_path).read_bytes())
        final_metrics = concept_validation_metrics(final_path)
        if strict_quality_requested_like(request) and not final_metrics.get("passed", False):
            trimmed_path = output_dir / "concept_single_subject_trimmed.png"
            prepared = prepare_hunyuan_reference_image(final_path, trimmed_path)
            if prepared is not None and prepared.exists():
                trimmed_metrics = concept_validation_metrics(prepared)
                (output_dir / "concept_validation_metrics_trimmed.json").write_text(json.dumps(trimmed_metrics, indent=2), encoding="utf-8")
                if trimmed_metrics.get("passed", False):
                    final_path.write_bytes(prepared.read_bytes())
                    final_metrics = trimmed_metrics
        (output_dir / "concept_validation_metrics.json").write_text(json.dumps(final_metrics, indent=2), encoding="utf-8")
        if strict_quality_requested_like(request) and not final_metrics.get("passed", False):
            raise RuntimeError(
                "Text concept generation could not produce a complete centered single-subject reference after "
                f"{index} candidate(s). Hunyuan was not run because this would create another janky STL. "
                "Use a cleaner image reference or simplify the text prompt. Details: "
                + json.dumps(final_metrics)
            )
        return final_path
    raise RuntimeError("local text-to-image did not produce a concept image")


def concept_retry_prompt(prompt: str, metrics: dict[str, Any] | None) -> str:
    """Nudge each concept retry away from the specific gate failure.

    CLIP only keeps the first ~77 tokens for the SD1.5 concept fallback. These
    retry instructions must be prepended, not appended, or they get decoded away
    by ``clamp_diffusers_prompt_to_token_limit`` and every retry repeats the
    same bad crop/noisy render.
    """
    additions = [
        "zoomed out complete full body miniature, small centered subject, large empty white border",
        "entire head torso arms hands weapon legs feet boots and round base visible",
        "plain white product cutout, no glow, no rim light, no halo, no card, no backdrop",
    ]
    if metrics:
        if metrics.get("likely_background_panel"):
            additions.insert(1, "no backdrop panel no card no poster no grey rectangle")
        if metrics.get("likely_clipped_subject") or metrics.get("likely_partial_body"):
            additions.insert(0, "zoomed out complete head hands weapon feet visible")
            additions.insert(1, "lower legs knees shins boots and small round base fully visible")
        if metrics.get("likely_squat_or_wide_subject"):
            additions.insert(0, "tall standing full body miniature clear legs separated from torso")
        if metrics.get("likely_overwide_upper_body"):
            additions.insert(0, "arms fully inside frame with empty white margin around shoulders")
        if metrics.get("likely_noisy_reference"):
            additions.insert(0, "smooth clean untextured clay sculpt no speckles no jagged artifacts")
        if metrics.get("likely_under_detailed_reference"):
            additions.insert(0, "crisp sculpted armor seams bevels vents hands weapon and facial details, not smooth blank armor")
        if float(metrics.get("foreground_ratio", 0.0) or 0.0) > 0.74:
            additions.insert(0, "subject occupies only forty five percent of frame")
        if float(metrics.get("border_contact_ratio", 0.0) or 0.0) > 0.20:
            additions.insert(0, "extra wide empty white border around entire miniature")
    first_addition, *remaining_additions = additions
    return ", ".join([first_addition, prompt] + remaining_additions)


def compact_concept_prompt(prompt: str) -> str:
    """Keep concept prompt under CLIP limits so important single-subject instructions survive."""
    text = re.sub(r"\s+", " ", prompt or "").strip()
    # Drop backend boilerplate that sculptor already appends and SDXL then truncates.
    text = re.split(r"Create a production/studio-quality|This must be final sculpt geometry|Requirements:", text, flags=re.IGNORECASE)[0].strip()
    words = text.split()
    return " ".join(words[:28]) if words else "high detail original miniature"


def normalize_miniature_prompt(prompt: str) -> str:
    """Fix common typos and convert protected-style requests into generic sculpt cues."""
    text = re.sub(r"\s+", " ", prompt or "").strip()
    lowered = text.lower()
    if any(typo in lowered for typo in ("space marine", "spcace marine", "sapce marine", "spcae marine")):
        text = re.sub(
            r"\b(?:space|spcace|sapce|spcae)\s+marine\b",
            "original heroic sci-fi armored soldier",
            text,
            flags=re.IGNORECASE,
        )
        if "power armor" not in lowered and "power armour" not in lowered:
            text += ", bulky gothic sci-fi power armor, helmet, backpack power unit, oversized blank shoulder pauldrons, heavy boots, heroic proportions, readable rifle, no logos or faction insignia"
    protected_replacements = {
        r"\bwarhammer\s*40k\b": "grimdark original sci-fi wargaming",
        r"\bwarhammer\b": "grimdark original fantasy wargaming",
        r"\bultramarines?\b": "blank heraldry armored soldier",
        r"\bblood\s+angels?\b": "ornate blank heraldry armored soldier",
        r"\bdark\s+angels?\b": "hooded blank heraldry armored soldier",
        r"\bblack\s+templars?\b": "tabard-wearing blank heraldry armored soldier",
        r"\badeptus\s+astartes\b": "original heroic sci-fi armored soldier",
    }
    for pattern, replacement in protected_replacements.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text


def concept_landmark_prompt(prompt: str) -> str:
    """Put prompt-defining concept landmarks before SD/CLIP truncates the text.

    Stable Diffusion 1.5 only reliably attends to the first CLIP window. The
    generic miniature style terms are useful, but if they come first the concept
    image becomes a generic armored person and Hunyuan faithfully reconstructs
    that generic reference. Keep archetype-defining shapes first.
    """
    lowered = (prompt or "").lower()
    landmarks: list[str] = []
    if any(term in lowered for term in ("space marine", "spcace marine", "sapce marine", "spcae marine", "power armor", "power armour", "adeptus", "primaris")):
        landmarks.extend(
            [
                "original bulky sci fi power armored soldier",
                "massive round shoulder pauldrons wider than torso",
                "helmet visor respirator face mask",
                "large rear power backpack with twin exhaust vents visible",
                "boxy heavy rifle held across chest with barrel and magazine",
                "barrel chest armor with raised chest emblem relief",
                "chunky greaves oversized boots",
                "skulls and rubble on small round scenic base",
            ]
        )
    if any(term in lowered for term in ("wizard", "mage", "sorcerer", "witch", "warlock", "cleric", "priest")):
        landmarks.extend(["robed caster silhouette", "pointed hood", "tall staff with orb", "deep robe folds"])
    if any(term in lowered for term in ("rogue", "assassin", "ranger", "thief", "ninja")):
        landmarks.extend(["lean hooded stealth silhouette", "two visible daggers", "belt pouches", "narrow cloak"])
    if any(term in lowered for term in ("knight", "paladin", "templar", "crusader", "champion")):
        landmarks.extend(["crested knight helmet", "front tabard", "large shield", "long sword"])
    if any(term in lowered for term in ("shield", "buckler")) and "large shield" not in landmarks:
        landmarks.append("large shield clearly visible")
    if any(term in lowered for term in ("banner", "standard", "flag")):
        landmarks.append("tall banner pole with hanging flag")
    if any(term in lowered for term in ("angel", "celestial", "seraph", "wing", "wings")):
        landmarks.append("two large feathered wings behind shoulders")
    if any(term in lowered for term in ("demon", "devil", "fiend", "tiefling")):
        landmarks.extend(["curved horns", "visible tail"])
    if any(term in lowered for term in ("skull", "skulls", "bone", "bones", "rubble", "rock", "rocks")) and not any("skulls and rubble" in item for item in landmarks):
        landmarks.append("skulls and rubble on small round scenic base")
    if not landmarks:
        landmarks.append("distinct non generic silhouette matching the requested subject")
    return ", ".join(dict.fromkeys(landmarks))


def clamp_diffusers_prompt_to_token_limit(pipe: Any, prompt: str) -> str:
    """Clamp prompts before diffusers/CLIP can exceed positional embeddings.

    Some local diffusers/transformers combinations warn and then later fail with
    token-index/position-index errors when prompt token count exceeds CLIP's 77
    token window. Word-count trimming is unreliable for BPE tokenizers, so use
    the pipeline tokenizer itself and decode the truncated token IDs.
    """
    text = re.sub(r"\s+", " ", prompt or "").strip()
    tokenizer = getattr(pipe, "tokenizer", None)
    if tokenizer is None:
        return text
    try:
        if hasattr(tokenizer, "clean_up_tokenization_spaces"):
            tokenizer.clean_up_tokenization_spaces = False
        max_length = int(getattr(tokenizer, "model_max_length", 77) or 77)
        if max_length <= 0 or max_length > 512:
            max_length = 77
        encoded = tokenizer(text, truncation=True, max_length=max_length, return_tensors=None)
        input_ids = encoded.get("input_ids") if isinstance(encoded, dict) else getattr(encoded, "input_ids", None)
        if input_ids is None:
            return text
        ids = input_ids[0] if input_ids and isinstance(input_ids[0], list) else input_ids
        try:
            return tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False).strip() or text
        except TypeError:
            return tokenizer.decode(ids, skip_special_tokens=True).strip() or text
    except Exception:
        return text


def strict_quality_requested_like(request: dict[str, Any]) -> bool:
    return production_quality_requested(request) and os.environ.get("MESHMEND_DISABLE_STORE_QUALITY_GATE", "0").strip().lower() not in {
        "1",
        "true",
        "yes",
    }


def enable_diffusers_memory_optimizations(pipe: Any, output_dir: Path) -> None:
    """Enable optional diffusers memory optimizations without making them fatal."""
    warnings: list[str] = []
    optimizations = (
        ("enable_attention_slicing", "attention_slicing"),
    )
    for method_name, label in optimizations:
        if not hasattr(pipe, method_name):
            continue
        if label == "vae_slicing" and os.environ.get("MESHMEND_DISABLE_VAE_SLICING", "0").strip().lower() in {"1", "true", "yes"}:
            warnings.append("vae_slicing disabled by MESHMEND_DISABLE_VAE_SLICING")
            continue
        if label == "vae_tiling" and os.environ.get("MESHMEND_DISABLE_VAE_TILING", "0").strip().lower() in {"1", "true", "yes"}:
            warnings.append("vae_tiling disabled by MESHMEND_DISABLE_VAE_TILING")
            continue
        try:
            getattr(pipe, method_name)()
        except Exception as exc:
            warnings.append(f"{label} unavailable: {exc}")
    if warnings:
        try:
            warning_path = output_dir / "diffusers_memory_optimization_warnings.txt"
            existing = warning_path.read_text(encoding="utf-8") if warning_path.exists() else ""
            warning_path.write_text(existing + "\n".join(warnings) + "\n", encoding="utf-8")
        except Exception:
            pass


def load_text_to_image_pipeline(model_id: str, dtype: Any) -> Any:
    try:
        if "xl" in model_id.lower() or "sdxl" in model_id.lower():
            from diffusers import StableDiffusionXLPipeline

            return StableDiffusionXLPipeline.from_pretrained(model_id, torch_dtype=dtype, use_safetensors=True)
    except Exception:
        pass
    from diffusers import StableDiffusionPipeline

    return StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype, safety_checker=None)


def concept_quality_score(image_path: Path) -> float:
    """Prefer sharp, high-contrast, single-subject concept images."""
    try:
        from PIL import Image
        import numpy as np

        image = Image.open(image_path).convert("L").resize((384, 384))
        arr = np.asarray(image, dtype=np.float32) / 255.0
        gx = np.diff(arr, axis=1, prepend=arr[:, :1])
        gy = np.diff(arr, axis=0, prepend=arr[:1, :])
        edge = np.sqrt(gx * gx + gy * gy)
        sharpness = float(np.percentile(edge, 95)) + float(edge.var()) * 3.0
        contrast = float(arr.std())
        # Penalize very wide foregrounds, which usually indicate lineups.
        fg = arr < np.quantile(arr, 0.82)
        rows, cols = np.where(fg)
        width_penalty = 0.0
        edge_contact_penalty = 0.0
        if rows.size and cols.size:
            fg_w = (cols.max() - cols.min() + 1) / arr.shape[1]
            fg_h = (rows.max() - rows.min() + 1) / arr.shape[0]
            width_penalty = max(0.0, (fg_w / max(fg_h, 1e-6)) - 0.75) * 0.08
            margin = max(4, int(arr.shape[0] * 0.035))
            edge_contact = (
                float(fg[:margin, :].mean())
                + float(fg[-margin:, :].mean())
                + float(fg[:, :margin].mean())
                + float(fg[:, -margin:].mean())
            )
            edge_contact_penalty = edge_contact * 0.18
        border = np.concatenate([arr[:12, :].ravel(), arr[-12:, :].ravel(), arr[:, :12].ravel(), arr[:, -12:].ravel()])
        border_penalty = max(0.0, 0.92 - float(border.mean())) * 0.20 + float(border.std()) * 0.12
        validation = concept_validation_metrics(image_path)
        band_penalty = max(0, int(validation.get("active_column_bands", 1)) - 1) * 0.18
        border_contact_penalty = float(validation.get("border_contact_ratio", 0.0)) * 0.20
        return sharpness + contrast * 0.35 - width_penalty - edge_contact_penalty - border_penalty - band_penalty - border_contact_penalty
    except Exception:
        return 0.0


def concept_validation_metrics(image_path: Path) -> dict[str, Any]:
    """Return lightweight single-subject/detail metrics for concept selection."""
    try:
        from PIL import Image
        import numpy as np

        rgba_image = Image.open(image_path).convert("RGBA").resize((512, 512))
        rgba = np.asarray(rgba_image, dtype=np.float32) / 255.0
        rgb = rgba[:, :, :3]
        alpha = rgba[:, :, 3]
        arr = rgb @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
        has_alpha_cutout = float((alpha < 0.98).mean()) > 0.01
        if has_alpha_cutout:
            fg = alpha > 0.08
        else:
            border_rgb = np.concatenate([rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]], axis=0)
            bg = np.median(border_rgb, axis=0)
            color_dist = np.linalg.norm(rgb - bg[None, None, :], axis=2)
            saturation = rgb.max(axis=2) - rgb.min(axis=2)
            fg = (color_dist > max(0.10, float(np.quantile(color_dist, 0.72)))) | (
                (arr < min(0.92, max(0.62, float(np.quantile(arr, 0.82))))) & (saturation > 0.045)
            )
        h, w = fg.shape
        margin = max(4, int(w * 0.035))
        top_contact = float(fg[:margin, :].mean())
        bottom_contact = float(fg[-margin:, :].mean())
        left_contact = float(fg[:, :margin].mean())
        right_contact = float(fg[:, -margin:].mean())
        border_contact = float(top_contact + bottom_contact + left_contact + right_contact) / 4.0
        col_density = fg.sum(axis=0).astype(float) / max(h, 1)
        if col_density.max() <= 0:
            return {"score": 0.0, "active_column_bands": 0, "border_contact_ratio": border_contact, "foreground_ratio": 0.0}
        smooth = np.convolve(col_density, np.ones(15) / 15.0, mode="same")
        active = smooth > max(float(smooth.max()) * 0.34, 0.025)
        bands = 0
        start = None
        for idx, value in enumerate(active):
            if value and start is None:
                start = idx
            elif not value and start is not None:
                if idx - start >= w * 0.055:
                    bands += 1
                start = None
        if start is not None and len(active) - start >= w * 0.055:
            bands += 1
        gx = np.abs(np.diff(arr, axis=1, prepend=arr[:, :1]))
        gy = np.abs(np.diff(arr, axis=0, prepend=arr[:1, :]))
        edges = gx + gy
        edge_density = float((edges > np.percentile(edges, 91.0)).mean())
        absolute_edge_density = float((edges > float(os.environ.get("MESHMEND_CONCEPT_NOISE_EDGE_THRESHOLD", "0.12"))).mean())
        foreground_ratio = float(fg.mean())
        row_density = fg.sum(axis=1).astype(float) / max(w, 1)
        bottom_density = float(row_density[int(h * 0.82) :].mean())
        mid_density = float(row_density[int(h * 0.35) : int(h * 0.65)].mean())
        bottom_width_ratio = float((fg[int(h * 0.78) :, :].sum(axis=0) > 0).mean())
        completeness = concept_full_body_completeness_metrics(fg)
        likely_base_band = bottom_width_ratio > float(os.environ.get("MESHMEND_CONCEPT_MAX_BOTTOM_WIDTH", "0.52")) and bottom_density > max(0.18, mid_density * 0.92)
        likely_background_panel = False if has_alpha_cutout else concept_has_large_background_panel(rgb, arr, gx + gy)
        score = (
            edge_density * 2.0
            + foreground_ratio
            - max(0, bands - 1) * 0.6
            - border_contact * 0.25
            - (0.35 if likely_base_band else 0.0)
            - (0.75 if likely_background_panel else 0.0)
        )
        max_border_contact = float(os.environ.get("MESHMEND_CONCEPT_MAX_BORDER_CONTACT", "0.38"))
        max_foreground_ratio = float(os.environ.get("MESHMEND_CONCEPT_MAX_FOREGROUND_RATIO", "0.74"))
        max_side_contact = float(os.environ.get("MESHMEND_CONCEPT_MAX_SIDE_CONTACT", "0.34"))
        max_bottom_width = float(os.environ.get("MESHMEND_CONCEPT_MAX_CLIPPED_BOTTOM_WIDTH", "0.50"))
        clipped_bottom = bottom_width_ratio > max_bottom_width and bottom_contact > 0.12 and bottom_density > max(0.16, mid_density * 0.55)
        clipped = max(top_contact, left_contact, right_contact) > max_side_contact or clipped_bottom
        require_full_body = os.environ.get("MESHMEND_REQUIRE_FULL_BODY_CONCEPT", "1").strip().lower() not in {"0", "false", "no"}
        likely_partial_body = bool(completeness.get("likely_partial_body", False)) if require_full_body else False
        subject_aspect = float(completeness.get("subject_aspect_ratio", 0.0) or 0.0)
        subject_height_ratio = float(completeness.get("subject_height_ratio", 0.0) or 0.0)
        min_subject_aspect = float(os.environ.get("MESHMEND_CONCEPT_MIN_SUBJECT_ASPECT", "1.08"))
        upper_width_ratio = float(completeness.get("upper_body_width_ratio", 0.0) or 0.0)
        lower_density = float(completeness.get("lower_body_density", 0.0) or 0.0)
        max_upper_width = float(os.environ.get("MESHMEND_CONCEPT_MAX_UPPER_BODY_WIDTH", "0.98"))
        likely_overwide_upper_body = require_full_body and upper_width_ratio > max_upper_width and subject_aspect < 1.15
        likely_squat_or_wide_subject = require_full_body and subject_aspect < min_subject_aspect
        wide_armored_full_body = (
            require_full_body
            and subject_aspect >= float(os.environ.get("MESHMEND_CONCEPT_MIN_WIDE_ARMORED_ASPECT", "0.92"))
            and subject_height_ratio >= float(os.environ.get("MESHMEND_CONCEPT_MIN_WIDE_ARMORED_HEIGHT", "0.62"))
            and lower_density >= float(os.environ.get("MESHMEND_CONCEPT_MIN_WIDE_ARMORED_LOWER_DENSITY", "0.32"))
            and border_contact < max_border_contact
            and not likely_base_band
            and not likely_background_panel
            and not clipped
        )
        if wide_armored_full_body:
            likely_partial_body = False
            likely_squat_or_wide_subject = False
        max_absolute_edge_density = float(os.environ.get("MESHMEND_CONCEPT_MAX_ABSOLUTE_EDGE_DENSITY", "0.072"))
        min_absolute_edge_density = float(os.environ.get("MESHMEND_CONCEPT_MIN_ABSOLUTE_EDGE_DENSITY", "0.040"))
        likely_noisy_reference = absolute_edge_density > max_absolute_edge_density
        likely_under_detailed_reference = require_full_body and absolute_edge_density < min_absolute_edge_density
        min_score = float(os.environ.get("MESHMEND_CONCEPT_MIN_QUALITY_SCORE", "0.38"))
        passed = (
            bands <= 1
            and not likely_base_band
            and not likely_background_panel
            and not clipped
            and not likely_partial_body
            and not likely_squat_or_wide_subject
            and not likely_overwide_upper_body
            and not likely_noisy_reference
            and not likely_under_detailed_reference
            and border_contact < max_border_contact
            and 0.08 <= foreground_ratio <= max_foreground_ratio
            and score >= min_score
        )
        return {
            "score": score,
            "active_column_bands": bands,
            "border_contact_ratio": border_contact,
            "top_contact_ratio": top_contact,
            "bottom_contact_ratio": bottom_contact,
            "left_contact_ratio": left_contact,
            "right_contact_ratio": right_contact,
            "foreground_ratio": foreground_ratio,
            "bottom_width_ratio": bottom_width_ratio,
            "bottom_density": bottom_density,
            **completeness,
            "likely_base_band": likely_base_band,
            "likely_background_panel": likely_background_panel,
            "likely_clipped_subject": clipped,
            "likely_partial_body": likely_partial_body,
            "likely_squat_or_wide_subject": likely_squat_or_wide_subject,
            "wide_armored_full_body": wide_armored_full_body,
            "min_subject_aspect_ratio": min_subject_aspect,
            "likely_overwide_upper_body": likely_overwide_upper_body,
            "max_upper_body_width_ratio": max_upper_width,
            "likely_noisy_reference": likely_noisy_reference,
            "likely_under_detailed_reference": likely_under_detailed_reference,
            "absolute_edge_density": absolute_edge_density,
            "max_absolute_edge_density": max_absolute_edge_density,
            "min_absolute_edge_density": min_absolute_edge_density,
            "edge_density": edge_density,
            "min_quality_score": min_score,
            "passed": passed,
        }
    except Exception as exc:
        return {"score": 0.0, "active_column_bands": 0, "border_contact_ratio": 1.0, "foreground_ratio": 0.0, "passed": False, "error": str(exc)}


def concept_full_body_completeness_metrics(foreground_mask: Any) -> dict[str, Any]:
    """Detect bust/upper-body concepts before Hunyuan turns them into partial meshes."""
    try:
        import numpy as np

        fg = np.asarray(foreground_mask, dtype=bool)
        h, w = fg.shape
        rows, cols = np.where(fg)
        if rows.size == 0 or cols.size == 0:
            return {"subject_aspect_ratio": 0.0, "likely_partial_body": True, "partial_body_reason": "no_foreground"}
        top = int(rows.min())
        bottom = int(rows.max())
        left = int(cols.min())
        right = int(cols.max())
        subject_h = max(1, bottom - top + 1)
        subject_w = max(1, right - left + 1)
        bbox = fg[top : bottom + 1, left : right + 1]
        row_density = bbox.sum(axis=1).astype(float) / max(subject_w, 1)
        upper_width = float((bbox[int(subject_h * 0.10) : max(int(subject_h * 0.34), int(subject_h * 0.10) + 1), :].sum(axis=0) > 0).mean())
        torso_width = float((bbox[int(subject_h * 0.35) : max(int(subject_h * 0.62), int(subject_h * 0.35) + 1), :].sum(axis=0) > 0).mean())
        lower_width = float((bbox[int(subject_h * 0.72) :, :].sum(axis=0) > 0).mean())
        lower_density = float(row_density[int(subject_h * 0.72) :].mean())
        aspect = float(subject_h / subject_w)
        height_ratio = float(subject_h / max(h, 1))
        width_ratio = float(subject_w / max(w, 1))
        min_aspect = float(os.environ.get("MESHMEND_CONCEPT_MIN_FULL_BODY_ASPECT", "1.0"))
        max_partial_lower_width = float(os.environ.get("MESHMEND_CONCEPT_MAX_PARTIAL_LOWER_WIDTH", "0.58"))
        # A generated bust/cropped torso usually has a short, wide silhouette and
        # ends in a broad lower band (waist/thigh crop). A complete tabletop
        # figure is normally at least as tall as it is wide after the prep crop,
        # even with weapons, capes, and a small round base. Do not reject broad
        # bases/pauldrons by themselves; they are common on armored minis.
        likely_short_wide_crop = aspect < min_aspect and width_ratio > 0.54 and lower_width > max_partial_lower_width
        likely_waist_cut = aspect < min_aspect and lower_width >= max(torso_width * 0.92, max_partial_lower_width) and lower_density > 0.16
        reason = ""
        if likely_short_wide_crop:
            reason = "short_wide_upper_body_crop"
        elif likely_waist_cut:
            reason = "broad_lower_waist_or_thigh_crop"
        return {
            "subject_aspect_ratio": aspect,
            "subject_height_ratio": height_ratio,
            "subject_width_ratio": width_ratio,
            "upper_body_width_ratio": upper_width,
            "torso_width_ratio": torso_width,
            "lower_body_width_ratio": lower_width,
            "lower_body_density": lower_density,
            "likely_partial_body": bool(likely_short_wide_crop or likely_waist_cut),
            "partial_body_reason": reason,
        }
    except Exception as exc:
        return {"likely_partial_body": False, "partial_body_error": str(exc)}


def concept_has_large_background_panel(rgb: Any, gray: Any, edges: Any) -> bool:
    """Detect smooth gray backdrop/card panels that Hunyuan turns into squares."""
    try:
        import numpy as np
        from scipy.ndimage import binary_closing, label

        saturation = rgb.max(axis=2) - rgb.min(axis=2)
        smooth = edges < float(np.percentile(edges, 62.0))
        neutral = saturation < float(os.environ.get("MESHMEND_CONCEPT_PANEL_MAX_SATURATION", "0.075"))
        mid_gray = (gray > 0.28) & (gray < 0.86)
        panel = binary_closing(smooth & neutral & mid_gray, iterations=2)
        h, w = gray.shape
        labeled, count = label(panel)
        for idx in range(1, count + 1):
            comp = labeled == idx
            area = int(comp.sum())
            if area < h * w * float(os.environ.get("MESHMEND_CONCEPT_PANEL_MIN_AREA", "0.10")):
                continue
            rows, cols = np.where(comp)
            if rows.size == 0 or cols.size == 0:
                continue
            width = int(cols.max() - cols.min() + 1)
            height = int(rows.max() - rows.min() + 1)
            fill = area / max(width * height, 1)
            upper_overlap = float((rows < h * 0.70).mean())
            if width > w * 0.42 and height > h * 0.22 and fill > 0.45 and upper_overlap > 0.55:
                return True
        return False
    except Exception:
        return False


def isolate_single_subject_concept(input_path: Path, output_path: Path) -> None:
    """Crop generated concept art to one central/largest subject.

    Text-to-image models often produce four-view lineups. Hunyuan then converts
    the whole lineup. This keeps a single foreground component before 3D.
    """
    try:
        from PIL import Image, ImageFilter
        import numpy as np
        from scipy.ndimage import binary_closing, binary_fill_holes, label

        image = Image.open(input_path).convert("RGB")
        arr = np.asarray(image, dtype=np.float32) / 255.0
        h, w = arr.shape[:2]
        border = np.concatenate([arr[0], arr[-1], arr[:, 0], arr[:, -1]], axis=0)
        bg = np.median(border, axis=0)
        color_dist = np.linalg.norm(arr - bg[None, None, :], axis=2)
        gray = arr @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
        fg = (color_dist > max(0.10, float(np.quantile(color_dist, 0.72)))) | (gray < float(np.quantile(gray, 0.42)))
        fg = binary_fill_holes(binary_closing(fg, iterations=2))
        panel_crop = crop_single_panel_from_lineup(image, fg)
        if panel_crop is not None:
            panel_crop.save(output_path)
            return
        labeled, count = label(fg)
        if count <= 0:
            return

        yy, xx = np.indices((h, w))
        cx, cy = w * 0.5, h * 0.52
        best_label = 0
        best_score = -1.0
        for idx in range(1, count + 1):
            comp = labeled == idx
            area = int(comp.sum())
            if area < h * w * 0.005:
                continue
            mean_x = float(xx[comp].mean())
            mean_y = float(yy[comp].mean())
            center_penalty = (((mean_x - cx) / w) ** 2 + ((mean_y - cy) / h) ** 2) ** 0.5
            score = area * (1.0 - min(center_penalty * 1.8, 0.85))
            if score > best_score:
                best_score = score
                best_label = idx
        if best_label <= 0:
            return

        comp = remove_broad_base_from_subject_mask(keep_subject_and_accessories(labeled, best_label, min_area_ratio=0.018, expansion=0.24))
        rows, cols = np.where(comp)
        pad_x = int(w * 0.08)
        pad_y = int(h * 0.08)
        left = max(0, int(cols.min()) - pad_x)
        right = min(w, int(cols.max()) + pad_x)
        top = max(0, int(rows.min()) - pad_y)
        bottom = min(h, int(rows.max()) + pad_y)
        if (right - left + 1) / max(bottom - top + 1, 1) >= float(os.environ.get("MESHMEND_DUAL_CONCEPT_WIDTH_RATIO", "0.95")) and has_separated_subject_bands(comp, left, right):
            half_crop = crop_single_side_subject(image, comp, left, right, top, bottom)
            if half_crop is not None:
                half_crop.save(output_path)
                return
        crop = image.crop((left, top, right, bottom))
        # Put the isolated subject back on a square white canvas, centered.
        size = max(crop.size)
        canvas = Image.new("RGB", (size, size), "white")
        canvas.paste(crop, ((size - crop.size[0]) // 2, (size - crop.size[1]) // 2))
        canvas = canvas.resize(image.size, Image.Resampling.LANCZOS)
        if os.environ.get("MESHMEND_SHARPEN_ISOLATED_CONCEPT", "0").strip().lower() in {"1", "true", "yes"}:
            canvas = canvas.filter(ImageFilter.SHARPEN)
        canvas.save(output_path)
    except Exception:
        return


def crop_single_panel_from_lineup(image: Any, foreground_mask: Any) -> Any | None:
    """If the concept is a 4-model lineup, keep one panel before Hunyuan sees it."""
    try:
        from PIL import Image, ImageFilter
        import numpy as np
        from scipy.ndimage import binary_fill_holes, label

        mask = np.asarray(foreground_mask, dtype=bool)
        h, w = mask.shape
        rows, cols = np.where(mask)
        if rows.size == 0 or cols.size == 0:
            return None
        left, right = int(cols.min()), int(cols.max())
        top, bottom = int(rows.min()), int(rows.max())
        fg_width = max(1, right - left + 1)
        fg_height = max(1, bottom - top + 1)

        # Single centered subjects are usually tall/narrow. A wide foreground is
        # commonly a 3/4-view lineup or four generated samples.
        if fg_width / fg_height < 0.72:
            return None

        column_density = mask[:, left : right + 1].sum(axis=0).astype(float)
        if column_density.max() <= 0:
            return None
        # Smooth density and find separated subject bands.
        kernel = np.ones(max(7, fg_width // 80), dtype=float)
        kernel /= kernel.sum()
        density = np.convolve(column_density, kernel, mode="same")
        active = density > max(density.max() * 0.22, h * 0.015)
        bands: list[tuple[int, int]] = []
        start = None
        for idx, value in enumerate(active):
            if value and start is None:
                start = idx
            elif not value and start is not None:
                if idx - start > fg_width * 0.06:
                    bands.append((start, idx - 1))
                start = None
        if start is not None and len(active) - start > fg_width * 0.06:
            bands.append((start, len(active) - 1))

        if len(bands) < 2:
            return None

        half_crop = crop_single_side_subject(image, mask, left, right, top, bottom)
        if half_crop is not None:
            return half_crop

        center = fg_width * 0.5
        # Prefer a central subject if present; otherwise choose the largest band.
        best_band = max(
            bands,
            key=lambda band: ((band[1] - band[0]) * 0.65) - abs(((band[0] + band[1]) * 0.5) - center),
        )
        pad_x = int(fg_width * 0.08)
        crop_left = max(0, left + best_band[0] - pad_x)
        crop_right = min(w, left + best_band[1] + pad_x)
        band_mask = mask[:, crop_left:crop_right]
        band_rows, _ = np.where(binary_fill_holes(band_mask))
        if band_rows.size:
            crop_top = max(0, int(band_rows.min()) - int(h * 0.08))
            crop_bottom = min(h, int(band_rows.max()) + int(h * 0.08))
        else:
            crop_top, crop_bottom = top, bottom
        crop = image.crop((crop_left, crop_top, crop_right, crop_bottom))
        size = max(crop.size)
        canvas = Image.new("RGB", (size, size), "white")
        canvas.paste(crop, ((size - crop.size[0]) // 2, (size - crop.size[1]) // 2))
        canvas = canvas.resize(image.size, Image.Resampling.LANCZOS)
        if os.environ.get("MESHMEND_SHARPEN_ISOLATED_CONCEPT", "0").strip().lower() in {"1", "true", "yes"}:
            canvas = canvas.filter(ImageFilter.SHARPEN)
        return canvas
    except Exception:
        return None


def crop_single_side_subject(image: Any, mask: Any, left: int, right: int, top: int, bottom: int) -> Any | None:
    """Crop one figure when concept art contains two side-by-side miniatures."""
    try:
        from PIL import Image, ImageFilter
        import numpy as np
        from scipy.ndimage import binary_fill_holes, label

        h, w = mask.shape
        fg_width = max(1, right - left + 1)
        fg_height = max(1, bottom - top + 1)
        if fg_width / fg_height < float(os.environ.get("MESHMEND_DUAL_CONCEPT_WIDTH_RATIO", "0.95")):
            return None
        if not has_separated_subject_bands(mask, left, right):
            return None

        mid = left + fg_width // 2
        margin = max(8, int(fg_width * 0.04))
        halves = [(left, max(left + 1, mid - margin)), (min(right - 1, mid + margin), right)]
        candidates: list[tuple[float, tuple[int, int, int, int]]] = []
        for h_left, h_right in halves:
            region = mask[:, h_left:h_right]
            filled = binary_fill_holes(region)
            labeled, count = label(filled)
            if count > 1:
                best_label = max(range(1, count + 1), key=lambda item: int((labeled == item).sum()))
                filled = labeled == best_label
            rows, cols = np.where(filled)
            if rows.size < h * w * 0.01:
                continue
            crop_left = max(0, h_left + int(cols.min()) - int(fg_width * 0.025))
            crop_right = min(w, h_left + int(cols.max()) + int(fg_width * 0.025))
            crop_top = max(0, int(rows.min()) - int(h * 0.06))
            crop_bottom = min(h, int(rows.max()) + int(h * 0.06))
            crop_w = max(1, crop_right - crop_left)
            crop_h = max(1, crop_bottom - crop_top)
            # Prefer a tall single figure with complete height and enough area.
            area = float(rows.size)
            aspect_score = -abs((crop_w / crop_h) - 0.48)
            height_score = crop_h / h
            center_penalty = abs(((crop_left + crop_right) * 0.5) - (w * 0.5)) / w
            candidates.append((area * (1.0 + aspect_score + height_score - center_penalty * 0.35), (crop_left, crop_top, crop_right, crop_bottom)))
        if not candidates:
            return None
        _score, bounds = max(candidates, key=lambda item: item[0])
        crop = image.crop(bounds)
        size = max(crop.size)
        canvas = Image.new("RGB", (size, size), "white")
        canvas.paste(crop, ((size - crop.size[0]) // 2, (size - crop.size[1]) // 2))
        canvas = canvas.resize(image.size, Image.Resampling.LANCZOS)
        if os.environ.get("MESHMEND_SHARPEN_ISOLATED_CONCEPT", "0").strip().lower() in {"1", "true", "yes"}:
            canvas = canvas.filter(ImageFilter.SHARPEN)
        return canvas
    except Exception:
        return None


def has_separated_subject_bands(mask: Any, left: int, right: int) -> bool:
    """Return true only when the foreground has multiple separated x-axis subject bands."""
    try:
        import numpy as np

        mask = np.asarray(mask, dtype=bool)
        h, w = mask.shape
        fg_width = max(1, right - left + 1)
        lower = mask[int(h * 0.72) :, left : right + 1]
        lower_width_ratio = float((lower.sum(axis=0) > 0).mean()) if lower.size else 0.0
        if lower_width_ratio > float(os.environ.get("MESHMEND_DUAL_CONCEPT_MAX_LOWER_BRIDGE", "0.58")):
            return False
        density = mask[:, left : right + 1].sum(axis=0).astype(float)
        if density.max() <= 0:
            return False
        kernel = np.ones(max(9, fg_width // 70), dtype=float)
        kernel /= kernel.sum()
        smooth = np.convolve(density, kernel, mode="same")
        active_threshold = max(float(smooth.max()) * 0.28, h * 0.018)
        active = smooth > active_threshold
        bands: list[tuple[int, int]] = []
        gaps: list[tuple[int, int]] = []
        start = None
        for idx, value in enumerate(active):
            if value and start is None:
                start = idx
            elif not value and start is not None:
                if idx - start > fg_width * 0.075:
                    bands.append((start, idx - 1))
                start = None
        if start is not None and len(active) - start > fg_width * 0.075:
            bands.append((start, len(active) - 1))
        gap_start = None
        for idx, value in enumerate(active):
            if not value and gap_start is None:
                gap_start = idx
            elif value and gap_start is not None:
                if idx - gap_start > fg_width * 0.10:
                    gaps.append((gap_start, idx - 1))
                gap_start = None
        if gap_start is not None and len(active) - gap_start > fg_width * 0.10:
            gaps.append((gap_start, len(active) - 1))
        if len(bands) < 2 or not gaps:
            return False
        center = fg_width * 0.5
        has_center_gap = any(abs(((gap[0] + gap[1]) * 0.5) - center) < fg_width * 0.24 for gap in gaps)
        return has_center_gap
    except Exception:
        return False


def run_hunyuan_image_to_3d(image_path: Path, request: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    workflow = str(request.get("workflow") or "text_to_3d")
    write_progress(output_dir, 30, "hunyuan_loading", "Loading Hunyuan3D model")
    hunyuan_repo = os.environ.get("MESHMEND_HUNYUAN3D_PATH", "").strip()
    if hunyuan_repo:
        repo_path = Path(hunyuan_repo).expanduser().resolve()
        if str(repo_path) not in sys.path:
            sys.path.insert(0, str(repo_path))
    try:
        from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
        from postprocess_backend import postprocess_miniature
    except Exception as exc:
        raise RuntimeError(
            "Hunyuan3D is not installed. Clone https://github.com/Tencent-Hunyuan/Hunyuan3D-2 or Hunyuan3D-2.1, "
            "install its requirements with `pip install -e .`, then set MESHMEND_HUNYUAN3D_PATH to that folder if needed. "
            "Do not rely on `pip install hy3dgen` unless it provides `hy3dgen.shapegen.Hunyuan3DDiTFlowMatchingPipeline` "
            "in this exact worker Python. "
            f"Worker Python: {sys.executable}. MESHMEND_HUNYUAN3D_PATH={hunyuan_repo or '(not set)'}. Import error: {exc}"
        ) from exc

    model_path = os.environ.get("MESHMEND_HUNYUAN3D_MODEL", "tencent/Hunyuan3D-2").strip()
    subfolder = os.environ.get("MESHMEND_HUNYUAN3D_SUBFOLDER", "hunyuan3d-dit-v2-0").strip()
    pipeline_kwargs: dict[str, Any] = {}
    if subfolder:
        pipeline_kwargs["subfolder"] = subfolder
    device = os.environ.get("MESHMEND_HUNYUAN3D_DEVICE", "").strip()
    if device:
        pipeline_kwargs["device"] = device

    pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(model_path, **pipeline_kwargs)
    default_attempts = "2" if production_quality_requested(request) else "1"
    attempts = max(1, int(os.environ.get("MESHMEND_HUNYUAN3D_QUALITY_ATTEMPTS", default_attempts)))
    last_error: Exception | None = None
    mesh = None
    postprocess_report = None
    prompt_landmark_overlay: dict[str, Any] = {"applied": False}
    for attempt in range(attempts):
        attempt_request = dict(request)
        attempt_request["_meshmend_hunyuan_attempt"] = attempt + 1
        generator_kwargs = hunyuan_generation_kwargs(attempt_request)
        if attempt > 0:
            generator_kwargs["num_inference_steps"] = max(
                int(generator_kwargs.get("num_inference_steps") or 0), int(os.environ.get("MESHMEND_HUNYUAN3D_RETRY_STEPS", "64"))
            )
            generator_kwargs["octree_resolution"] = max(
                int(generator_kwargs.get("octree_resolution") or 0), int(os.environ.get("MESHMEND_HUNYUAN3D_RETRY_OCTREE_RESOLUTION", "512"))
            )
        (output_dir / f"hunyuan_generation_kwargs_attempt_{attempt + 1}.json").write_text(
            json.dumps(generator_kwargs, indent=2, default=str), encoding="utf-8"
        )
        (output_dir / "hunyuan_generation_kwargs.json").write_text(json.dumps(generator_kwargs, indent=2, default=str), encoding="utf-8")
        try:
            write_progress(output_dir, 36, "hunyuan_shape_generation", f"Running Hunyuan3D shape generation attempt {attempt + 1} of {attempts}")
            mesh_result = call_hunyuan_pipeline(
                pipeline,
                image_path,
                generator_kwargs,
                output_dir,
                strict_quality=production_quality_requested(attempt_request),
            )
            raw_mesh = mesh_result[0] if isinstance(mesh_result, (list, tuple)) else mesh_result
            postprocess_request = dict(attempt_request)
            postprocess_request["_meshmend_source_image_path"] = str(image_path)
            # Hunyuan's local text prompt path is implemented as text -> concept image -> image-to-3D.
            # Validate and repair that generated concept mesh with the image-reconstruction profile;
            # otherwise strict text-to-3D defaults skip watertight solidification/component repair and
            # reject otherwise salvageable local outputs with boundary-edge/component gate failures.
            postprocess_request["_meshmend_source_workflow"] = (
                "image_to_3d" if bool(request.get("_meshmend_generated_text_concept")) else str(request.get("workflow") or "text_to_3d")
            )
            write_progress(output_dir, 74, "postprocessing", "Postprocessing mesh: scale, detail, cleanup, quality gates")
            mesh, postprocess_report = postprocess_miniature(raw_mesh, postprocess_request)
            mesh, prompt_landmark_overlay = apply_prompt_landmark_overlay(mesh, postprocess_request, output_dir)
            break
        except RuntimeError as exc:
            last_error = exc
            if "store-quality gate" not in str(exc) or attempt + 1 >= attempts:
                raise
            (output_dir / f"hunyuan_attempt_{attempt + 1}_rejected.txt").write_text(str(exc), encoding="utf-8")
    if mesh is None or postprocess_report is None:
        raise last_error or RuntimeError("Hunyuan3D did not produce a postprocessed mesh")

    output_format = os.environ.get("MESHMEND_HUNYUAN3D_OUTPUT_FORMAT", "stl").strip().lower().lstrip(".") or "stl"
    if f".{output_format}" not in SUPPORTED_MODEL_SUFFIXES:
        output_format = "stl"
    model_file = output_dir / f"meshmend_hunyuan.{output_format}"
    write_progress(output_dir, 94, "exporting", f"Exporting {output_format.upper()} model")
    export_mesh(mesh, model_file)
    if not model_file.exists():
        raise RuntimeError("Hunyuan3D completed but no mesh file was exported")
    mesh_info = mesh_export_info(mesh, request)
    report_info = postprocess_report.to_dict()
    report_info["prompt_landmark_overlay"] = prompt_landmark_overlay
    return {
        "model_file": model_file.name,
        "model_format": output_format,
        "provider": "free_local_hunyuan3d",
        "source_image": image_path.name,
        "text_concept_reference": bool(request.get("_meshmend_generated_text_concept")),
        "model_path": model_path,
        "subfolder": subfolder,
        "mesh_info": report_info
        | {"export_info": mesh_info, "text_concept_reference": bool(request.get("_meshmend_generated_text_concept"))},
        "consumed_credits": 0,
    }


def apply_prompt_landmark_overlay(mesh: Any, request: dict[str, Any], output_dir: Path) -> tuple[Any, dict[str, Any]]:
    """Attach coarse prompt landmarks when Hunyuan returns a generic body.

    Hunyuan's image-to-3D step can average a good prompt/concept back into a
    generic humanoid. For text-generated concepts, add STL-visible landmark
    geometry derived from the prompt so required accessories survive export.
    This is intentionally coarse secondary-form geometry, not texture.
    """
    if os.environ.get("MESHMEND_DISABLE_PROMPT_LANDMARK_OVERLAY", "0").strip().lower() in {"1", "true", "yes", "on"}:
        return mesh, {"applied": False, "reason": "disabled"}
    prompt = str(request.get("prompt") or "")
    if not prompt.strip():
        return mesh, {"applied": False, "reason": "empty_prompt"}
    try:
        import numpy as np
        import trimesh
        from native_generation import build_humanoid_prompt_features

        overlay_parts, overlay_names = build_humanoid_prompt_features(prompt.lower())
        if not overlay_parts:
            return mesh, {"applied": False, "reason": "no_prompt_landmarks"}
        overlay = trimesh.util.concatenate(overlay_parts)
        mesh_vertices = np.asarray(mesh.vertices, dtype=float)
        overlay_vertices = np.asarray(overlay.vertices, dtype=float)
        if len(mesh_vertices) == 0 or len(overlay_vertices) == 0:
            return mesh, {"applied": False, "reason": "empty_mesh"}
        mesh_min = mesh_vertices.min(axis=0)
        mesh_max = mesh_vertices.max(axis=0)
        overlay_min = overlay_vertices.min(axis=0)
        overlay_max = overlay_vertices.max(axis=0)
        mesh_height = float(max(mesh_max[2] - mesh_min[2], 1e-6))
        overlay_height = float(max(overlay_max[2] - min(0.0, overlay_min[2]), 1e-6))
        scale = mesh_height / overlay_height
        overlay.apply_scale(scale)
        overlay_vertices = np.asarray(overlay.vertices, dtype=float)
        overlay_min = overlay_vertices.min(axis=0)
        overlay_max = overlay_vertices.max(axis=0)
        mesh_center = (mesh_min + mesh_max) * 0.5
        overlay_center = (overlay_min + overlay_max) * 0.5
        overlay.apply_translation([
            float(mesh_center[0] - overlay_center[0]),
            float(mesh_center[1] - overlay_center[1]),
            float(mesh_min[2] - overlay_min[2]),
        ])
        combined = trimesh.util.concatenate([mesh, overlay])
        combined.metadata.update(getattr(mesh, "metadata", {}) or {})
        combined.metadata["meshmend_prompt_landmark_overlay"] = True
        combined.metadata["meshmend_prompt_landmark_parts"] = overlay_names
        try:
            combined.remove_unreferenced_vertices()
            combined.fix_normals()
        except Exception:
            pass
        report = {
            "applied": True,
            "part_count": len(overlay_names),
            "parts": overlay_names,
            "scale": scale,
        }
        (output_dir / "prompt_landmark_overlay.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        return combined, report
    except Exception as exc:
        return mesh, {"applied": False, "reason": f"overlay_failed:{exc}"}


def call_hunyuan_pipeline(
    pipeline: Any,
    image_path: Path,
    kwargs: dict[str, Any],
    output_dir: Path | None = None,
    *,
    strict_quality: bool = False,
) -> Any:
    """Call Hunyuan while preserving supported quality knobs.

    Older Hunyuan builds expose different keyword arguments. The previous worker
    dropped *all* quality kwargs after a TypeError, which could silently fall
    back to low-resolution defaults. This filters unsupported kwargs instead.
    """
    accepted = dict(kwargs)
    try:
        signature = inspect.signature(pipeline.__call__)
        has_var_kwargs = any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values())
        if not has_var_kwargs:
            accepted = {key: value for key, value in kwargs.items() if key in signature.parameters}
    except Exception:
        accepted = dict(kwargs)
    dropped: list[str] = [key for key in kwargs if key not in accepted]
    if strict_quality:
        missing_quality_knobs = [key for key in ("num_inference_steps", "octree_resolution") if key in kwargs and key not in accepted]
        if missing_quality_knobs:
            raise RuntimeError(
                "Hunyuan pipeline does not accept required store-quality generation kwargs: "
                + ", ".join(missing_quality_knobs)
            )
    while True:
        if output_dir is not None:
            try:
                (output_dir / "hunyuan_effective_kwargs.json").write_text(
                    json.dumps({"accepted": accepted, "dropped": dropped}, indent=2, default=str), encoding="utf-8"
                )
            except Exception:
                pass
        try:
            return pipeline(image=str(image_path), **accepted)
        except TypeError as exc:
            message = str(exc)
            lower = message.lower()
            unsupported_kwarg = "unexpected keyword" in lower or "got an unexpected" in lower or "invalid keyword" in lower
            if not unsupported_kwarg:
                raise
            strict = strict_quality or os.environ.get("MESHMEND_HUNYUAN3D_STRICT_KWARGS", "0").strip().lower() in {"1", "true", "yes"}
            match = re.search(r"['\"]([A-Za-z_][A-Za-z0-9_]*)['\"]", message)
            key = match.group(1) if match else ""
            if key and key in accepted and not strict:
                dropped.append(key)
                accepted.pop(key, None)
                continue
            raise RuntimeError(f"Hunyuan rejected generation kwargs {sorted(accepted)}: {exc}") from exc


def hunyuan_generation_kwargs(request: dict[str, Any]) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    workflow = str(request.get("workflow") or "text_to_3d")
    wants_8k = production_quality_requested(request)
    steps = os.environ.get("MESHMEND_HUNYUAN3D_STEPS", "").strip()
    default_steps = 64 if wants_8k else (36 if workflow == "image_to_3d" else 20)
    kwargs["num_inference_steps"] = int(steps or default_steps)
    guidance = os.environ.get("MESHMEND_HUNYUAN3D_GUIDANCE", "").strip()
    if guidance:
        kwargs["guidance_scale"] = float(guidance)
    seed = os.environ.get("MESHMEND_HUNYUAN3D_SEED", "").strip()
    if seed:
        try:
            import torch

            attempt_offset = max(0, int(request.get("_meshmend_hunyuan_attempt") or 1) - 1)
            kwargs["generator"] = torch.Generator().manual_seed(int(seed) + attempt_offset)
        except Exception:
            pass
    octree_resolution = os.environ.get("MESHMEND_HUNYUAN3D_OCTREE_RESOLUTION", "").strip()
    default_octree = 512 if wants_8k else (384 if workflow == "image_to_3d" else 256)
    kwargs["octree_resolution"] = int(octree_resolution or default_octree)
    target_polycount = int(request.get("target_polycount") or 0)
    if target_polycount:
        # Hunyuan versions expose different names for this knob. Prefer the
        # common face-count style only when caller explicitly asks via env.
        arg_name = os.environ.get("MESHMEND_HUNYUAN3D_FACE_ARG", "").strip()
        if arg_name:
            kwargs[arg_name] = target_polycount
    return kwargs


def keep_single_primary_mesh(mesh: Any) -> Any:
    """Keep one generated subject instead of exporting a multi-figure scene.

    Text-to-image models sometimes produce a lineup/reference sheet despite the
    prompt. Hunyuan then turns each view/character into a separate connected
    component. For miniature creation the expected output is one printable
    subject, so keep the largest connected body while preserving the mesh type.
    """
    try:
        import trimesh

        if isinstance(mesh, trimesh.Scene):
            geometries = [geom for geom in mesh.geometry.values() if hasattr(geom, "faces") and len(geom.faces) > 0]
            if not geometries:
                return mesh
            mesh = trimesh.util.concatenate(geometries)

        if not hasattr(mesh, "split"):
            return mesh
        components = [component for component in mesh.split(only_watertight=False) if len(component.faces) > 50]
        if len(components) <= 1:
            return mesh
        components.sort(key=lambda item: float(abs(getattr(item, "volume", 0.0))) if getattr(item, "volume", 0.0) else float(getattr(item, "area", 0.0)), reverse=True)
        return components[0]
    except Exception:
        return mesh


def coerce_to_trimesh(mesh: Any) -> Any:
    """Convert Hunyuan output into a mutable trimesh before post-processing."""
    try:
        import trimesh

        if isinstance(mesh, trimesh.Scene):
            geometries = [geom for geom in mesh.geometry.values() if hasattr(geom, "faces") and len(geom.faces) > 0]
            if geometries:
                return trimesh.util.concatenate(geometries)
            return mesh
        if isinstance(mesh, trimesh.Trimesh):
            return mesh.copy()
        if hasattr(mesh, "vertices") and hasattr(mesh, "faces"):
            return trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, process=False)
        return mesh
    except Exception:
        return mesh


def keep_single_spatial_subject(mesh: Any) -> Any:
    """Crop connected multi-figure scenes to one spatial subject cluster."""
    try:
        import numpy as np
        import trimesh

        if isinstance(mesh, trimesh.Scene):
            geometries = [geom for geom in mesh.geometry.values() if hasattr(geom, "faces") and len(geom.faces) > 0]
            if not geometries:
                return mesh
            mesh = trimesh.util.concatenate(geometries)
        if not hasattr(mesh, "vertices") or not hasattr(mesh, "faces") or len(mesh.vertices) < 1000:
            return mesh
        vertices = np.asarray(mesh.vertices, dtype=float)
        mins = vertices.min(axis=0)
        maxs = vertices.max(axis=0)
        ext = np.maximum(maxs - mins, 1e-6)
        # If width is suspiciously large compared to height/depth, it is likely
        # multiple figures side-by-side, possibly connected by a thin base/sheet.
        x_ratio = ext[0] / max(ext[2], 1e-6)
        if x_ratio < float(os.environ.get("MESHMEND_MULTI_SUBJECT_WIDTH_RATIO", "0.85")):
            return mesh

        x = vertices[:, 0]
        hist, edges = np.histogram(x, bins=96)
        if hist.max() <= 0:
            return mesh
        smooth = np.convolve(hist.astype(float), np.ones(5) / 5.0, mode="same")
        active = smooth > max(float(smooth.max()) * 0.20, len(vertices) * 0.0015)
        bands: list[tuple[int, int]] = []
        start = None
        for idx, flag in enumerate(active):
            if flag and start is None:
                start = idx
            elif not flag and start is not None:
                if idx - start >= 5:
                    bands.append((start, idx - 1))
                start = None
        if start is not None and len(active) - start >= 5:
            bands.append((start, len(active) - 1))
        if len(bands) < 2:
            return mesh

        center_x = (mins[0] + maxs[0]) * 0.5
        best = max(
            bands,
            key=lambda band: ((band[1] - band[0]) * 0.7) - abs(((edges[band[0]] + edges[band[1] + 1]) * 0.5) - center_x),
        )
        pad = ext[0] * 0.04
        keep_min = edges[best[0]] - pad
        keep_max = edges[best[1] + 1] + pad
        face_vertices = mesh.faces
        face_centers_x = vertices[face_vertices].mean(axis=1)[:, 0]
        face_mask = (face_centers_x >= keep_min) & (face_centers_x <= keep_max)
        if face_mask.sum() < len(mesh.faces) * 0.15:
            return mesh
        cropped = trimesh.Trimesh(vertices=vertices.copy(), faces=mesh.faces[face_mask].copy(), process=False)
        cropped.remove_unreferenced_vertices()
        components = [component for component in cropped.split(only_watertight=False) if len(component.faces) > 50]
        if components:
            components.sort(key=lambda item: float(getattr(item, "area", 0.0)), reverse=True)
            return components[0]
        return cropped
    except Exception:
        return mesh


def normalize_mesh_to_requested_scale(mesh: Any, request: dict[str, Any]) -> Any:
    """Scale generated assets to the requested tabletop miniature height."""
    try:
        import numpy as np

        if not hasattr(mesh, "vertices") or len(mesh.vertices) == 0:
            return mesh
        scale_mm = requested_scale_mm(request)
        vertices = np.asarray(mesh.vertices, dtype=float)
        extents = vertices.max(axis=0) - vertices.min(axis=0)
        current_height = float(np.max(extents))
        if current_height <= 1e-8:
            return mesh
        mesh.vertices = vertices * (scale_mm / current_height)

        vertices = np.asarray(mesh.vertices, dtype=float)
        mins = vertices.min(axis=0)
        maxs = vertices.max(axis=0)
        center_xy = (mins[:2] + maxs[:2]) * 0.5
        vertices[:, 0] -= center_xy[0]
        vertices[:, 1] -= center_xy[1]
        vertices[:, 2] -= mins[2]
        mesh.vertices = vertices
        annotate_mesh_scale(mesh, scale_mm)
        return mesh
    except Exception:
        return mesh


def requested_scale_mm(request: dict[str, Any]) -> float:
    for value in (request.get("scale_mm"), request.get("scale")):
        if value:
            try:
                return float(str(value).lower().replace("mm", "").strip())
            except ValueError:
                pass
    prompt = str(request.get("prompt") or "")
    match = re.search(r"\b(15|20|25|28|30|32|35|40|48|54|75|90|100)\s*mm\b", prompt.lower())
    if match:
        return float(match.group(1))
    return float(os.environ.get("MESHMEND_DEFAULT_MINIATURE_SCALE_MM", "32"))


def thicken_flat_mesh(mesh: Any, request: dict[str, Any]) -> Any:
    """Prevent single-view Hunyuan outputs from remaining paper-thin sheets."""
    try:
        import numpy as np

        if not hasattr(mesh, "vertices") or len(mesh.vertices) == 0:
            return mesh
        vertices = np.asarray(mesh.vertices, dtype=float)
        extents = vertices.max(axis=0) - vertices.min(axis=0)
        max_extent = float(np.max(extents))
        min_extent = float(np.min(extents))
        if max_extent <= 1e-8:
            return mesh
        thickness_ratio = min_extent / max_extent
        min_ratio = float(os.environ.get("MESHMEND_MIN_THICKNESS_RATIO", "0.18"))
        if thickness_ratio >= min_ratio:
            return mesh

        thin_axis = int(np.argmin(extents))
        center = (vertices.max(axis=0) + vertices.min(axis=0)) * 0.5
        target_thickness = max_extent * min_ratio
        axis_scale = min(float(os.environ.get("MESHMEND_MAX_THICKEN_SCALE", "8.0")), target_thickness / max(min_extent, 1e-6))
        vertices[:, thin_axis] = center[thin_axis] + (vertices[:, thin_axis] - center[thin_axis]) * axis_scale

        # Add a very small normal offset to separate coincident front/back areas
        # common in sheet reconstructions.
        normals = np.asarray(getattr(mesh, "vertex_normals", np.zeros_like(vertices)), dtype=float)
        if normals.shape == vertices.shape:
            vertices = vertices + normals * float(os.environ.get("MESHMEND_SHEET_NORMAL_INFLATE_MM", "0.22"))
        mesh.vertices = vertices
        mesh = normalize_mesh_to_requested_scale(mesh, request)
        return mesh
    except Exception:
        return mesh


def add_image_guided_surface_detail(mesh: Any, image_path: Path, request: dict[str, Any]) -> Any:
    """Project concept-image edge detail into STL geometry on the front shell."""
    if os.environ.get("MESHMEND_ENABLE_IMAGE_GUIDED_DETAIL", "0").strip().lower() not in {"1", "true", "yes"}:
        return mesh
    try:
        from PIL import Image
        import numpy as np

        if not hasattr(mesh, "vertices") or not hasattr(mesh, "faces") or len(mesh.vertices) < 100:
            return mesh
        image = Image.open(image_path).convert("RGB").resize((256, 256))
        arr = np.asarray(image, dtype=np.float32) / 255.0
        gray = arr @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
        gx = np.abs(np.diff(gray, axis=1, prepend=gray[:, :1]))
        gy = np.abs(np.diff(gray, axis=0, prepend=gray[:1, :]))
        edges = gx + gy
        edge_scale = float(np.percentile(edges, 97))
        if edge_scale > 1e-6:
            edges = np.clip(edges / edge_scale, 0.0, 1.0)
        contrast = gray - float(np.mean(gray))
        contrast_scale = float(np.percentile(np.abs(contrast), 92))
        if contrast_scale > 1e-6:
            contrast = np.clip(contrast / contrast_scale, -1.0, 1.0)

        vertices = np.asarray(mesh.vertices, dtype=float)
        normals = np.asarray(getattr(mesh, "vertex_normals", np.zeros_like(vertices)), dtype=float)
        if normals.shape != vertices.shape:
            return mesh
        mins = vertices.min(axis=0)
        maxs = vertices.max(axis=0)
        ext = np.maximum(maxs - mins, 1e-6)
        # Treat negative Y/front half as the visible concept-facing shell.
        front = vertices[:, 1] <= mins[1] + ext[1] * 0.48
        not_base = vertices[:, 2] > mins[2] + ext[2] * 0.06
        u = (vertices[:, 0] - mins[0]) / ext[0]
        v = 1.0 - ((vertices[:, 2] - mins[2]) / ext[2])
        ix = np.clip((u * 255).astype(int), 0, 255)
        iy = np.clip((v * 255).astype(int), 0, 255)
        sampled_edges = edges[iy, ix]
        sampled_contrast = contrast[iy, ix]
        grooves = sampled_edges > 0.62
        relief = (sampled_contrast * 0.08) - grooves.astype(float) * 0.35
        amplitude = float(os.environ.get("MESHMEND_IMAGE_DETAIL_RELIEF_MM", "0.025"))
        mask = (front & not_base).astype(float)
        mesh.vertices = vertices + normals * (relief * amplitude * mask)[:, None]
        return mesh
    except Exception:
        return mesh


def add_printable_surface_detail(mesh: Any, request: dict[str, Any]) -> Any:
    """Add sparse structured relief without covering the model in random noise."""
    if os.environ.get("MESHMEND_DISABLE_GEOMETRIC_DETAIL", "0").strip().lower() in {"1", "true", "yes"}:
        return mesh
    try:
        import numpy as np

        if not hasattr(mesh, "vertices") or not hasattr(mesh, "faces") or len(mesh.faces) < 100:
            return mesh
        quality = str(request.get("quality") or "standard").lower()
        target_faces = int(os.environ.get("MESHMEND_DETAIL_TARGET_FACES", "420000" if quality == "high" else "180000"))
        max_faces = int(os.environ.get("MESHMEND_DETAIL_MAX_FACES", "750000"))
        mesh = subdivide_for_detail(mesh, min(target_faces, max_faces))

        vertices = np.asarray(mesh.vertices, dtype=float)
        extents = np.maximum(vertices.max(axis=0) - vertices.min(axis=0), 1e-6)
        model_height = float(np.max(extents))
        if model_height <= 1e-6:
            return mesh
        normals = np.asarray(getattr(mesh, "vertex_normals", np.zeros_like(vertices)), dtype=float)
        if normals.shape != vertices.shape:
            return mesh

        z_min = float(vertices[:, 2].min())
        z_max = float(vertices[:, 2].max())
        z_norm = (vertices[:, 2] - z_min) / max(z_max - z_min, 1e-6)
        active = z_norm > 0.06
        coords = vertices / model_height
        prompt = str(request.get("prompt") or "")
        seed = (sum(ord(ch) for ch in prompt) % 997) + 1
        relief = np.zeros(len(vertices), dtype=float)

        lower = prompt.lower()
        if any(term in lower for term in ("space marine", "armor", "armour", "robot", "mech", "gun", "soldier")):
            relief += armored_miniature_relief(coords, z_norm, seed)
        else:
            wrinkle = np.abs(np.sin((coords[:, 2] * 13.0 + coords[:, 0] * 4.0 + seed * 0.02) * np.pi)) < 0.026
            relief -= wrinkle.astype(float) * 0.35

        # Optional micro texture is intentionally off by default because it read
        # as noise in STL. Enable only if the user wants rough materials.
        if os.environ.get("MESHMEND_ENABLE_MICRO_NOISE", "0").strip().lower() in {"1", "true", "yes"}:
            relief += 0.15 * np.sin((coords[:, 0] * 43.0 + coords[:, 1] * 17.0 + seed) * np.pi)

        amplitude_mm = float(os.environ.get("MESHMEND_DETAIL_RELIEF_MM", "0.075" if quality == "high" else "0.045"))
        vertices = vertices + normals * (np.clip(relief, -1.0, 1.0) * amplitude_mm * active.astype(float))[:, None]
        mesh.vertices = vertices
        try:
            mesh.remove_unreferenced_vertices()
            mesh.merge_vertices()
            mesh.fix_normals()
        except Exception:
            pass
        return mesh
    except Exception:
        return mesh


def subdivide_for_detail(mesh: Any, target_faces: int) -> Any:
    try:
        import trimesh

        while hasattr(mesh, "faces") and len(mesh.faces) < target_faces:
            next_faces = len(mesh.faces) * 4
            if next_faces > target_faces * 1.6:
                break
            vertices, faces = trimesh.remesh.subdivide(mesh.vertices, mesh.faces)
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        return mesh
    except Exception:
        return mesh


def ensure_minimum_export_density(mesh: Any, request: dict[str, Any]) -> Any:
    """Guarantee exported STL is not a tiny low-poly shell."""
    try:
        if not hasattr(mesh, "faces"):
            return mesh
        quality = str(request.get("quality") or "standard").lower()
        min_faces = int(os.environ.get("MESHMEND_MIN_EXPORT_FACES", "180000" if quality == "high" else "90000"))
        if len(mesh.faces) >= min_faces:
            return mesh
        return subdivide_for_detail(mesh, min_faces)
    except Exception:
        return mesh


def armored_miniature_relief(coords: Any, z_norm: Any, seed: int) -> Any:
    """Structured miniature-scale armor seams, trims, vents, and rivets."""
    import numpy as np

    x = coords[:, 0]
    y = coords[:, 1]
    z = coords[:, 2]
    relief = np.zeros(len(x), dtype=float)

    torso = (z_norm > 0.28) & (z_norm < 0.74)
    legs = (z_norm > 0.08) & (z_norm <= 0.42)
    shoulders = (z_norm > 0.58) & (z_norm < 0.86) & (np.abs(x) > np.quantile(np.abs(x), 0.62))
    head = z_norm > 0.74

    # Recessed armor plate seams.
    horizontal = np.abs(np.sin((z * 18.0 + seed * 0.031) * np.pi)) < 0.026
    vertical = np.abs(np.sin((x * 13.0 + seed * 0.047) * np.pi)) < 0.022
    diagonal = np.abs(np.sin(((x + z) * 10.0 + seed * 0.071) * np.pi)) < 0.018
    relief -= (horizontal & torso).astype(float) * 0.70
    relief -= (vertical & torso).astype(float) * 0.42
    relief -= (diagonal & legs).astype(float) * 0.35

    # Raised trim bands around shoulder/torso areas.
    trim = np.abs(np.sin((z * 9.0 + np.abs(x) * 2.0 + seed * 0.019) * np.pi)) < 0.030
    relief += (trim & (torso | shoulders)).astype(float) * 0.55

    # Vent/gill slits on upper torso/head regions.
    vents = (np.abs(np.sin((x * 31.0 + seed * 0.13) * np.pi)) < 0.018) & (np.abs(y) < np.quantile(np.abs(y), 0.72))
    relief -= (vents & (torso | head)).astype(float) * 0.45

    # Rivets as sparse raised dots. This is coordinate-based, deterministic, and
    # only affects a small percentage of vertices so it reads as detail rather
    # than surface noise.
    grid_x = np.abs(np.sin((x * 24.0 + seed * 0.17) * np.pi)) < 0.020
    grid_z = np.abs(np.sin((z * 28.0 + seed * 0.23) * np.pi)) < 0.020
    rivets = grid_x & grid_z & (torso | shoulders | legs)
    relief += rivets.astype(float) * 0.80

    return relief


def annotate_mesh_scale(mesh: Any, scale_mm: float) -> None:
    try:
        metadata = getattr(mesh, "metadata", None)
        if isinstance(metadata, dict):
            metadata["meshmend_scale_mm"] = float(scale_mm)
            metadata["units"] = "mm"
    except Exception:
        pass


def mesh_export_info(mesh: Any, request: dict[str, Any]) -> dict[str, Any]:
    try:
        import numpy as np

        vertices = np.asarray(mesh.vertices, dtype=float)
        extents = vertices.max(axis=0) - vertices.min(axis=0)
        return {
            "target_scale_mm": requested_scale_mm(request),
            "extents_mm": [float(value) for value in extents],
            "max_extent_mm": float(np.max(extents)),
            "faces": int(len(mesh.faces)) if hasattr(mesh, "faces") else None,
            "vertices": int(len(mesh.vertices)) if hasattr(mesh, "vertices") else None,
            "units": "mm",
            "detail_style": "single_subject_crop+minimum_density+structured_grooves",
        }
    except Exception as exc:
        return {"error": str(exc)}


def export_mesh(mesh: Any, output_path: Path) -> None:
    if output_path.suffix.lower() == ".stl":
        mesh = coerce_to_trimesh(mesh)
    if hasattr(mesh, "export"):
        mesh.export(output_path)
        return
    if hasattr(mesh, "save"):
        mesh.save(output_path)
        return
    raise RuntimeError(f"Unsupported Hunyuan3D mesh result type: {type(mesh)!r}")


def find_first_file(directory: Path, suffixes: set[str]) -> Path | None:
    for path in directory.iterdir():
        if path.is_file() and path.suffix.lower() in suffixes:
            return path
    return None


def find_prompt_reference_image(output_dir: Path) -> Path | None:
    # Reserved for future UI-supplied concept images in the task output folder.
    return find_first_file(output_dir, {".png", ".jpg", ".jpeg", ".webp"})


def run_legacy_sculptor(request: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    """Explicit opt-in fallback for development; not the production default."""
    meshmend_src = Path(__file__).resolve().parents[3]
    if str(meshmend_src.parent) not in sys.path:
        sys.path.insert(0, str(meshmend_src.parent))
    from meshmend_ai.sculptor import get_sculptor_foundation

    previous_disable = os.environ.get("MESHMEND_DISABLE_HOSTED_CREATION")
    os.environ["MESHMEND_DISABLE_HOSTED_CREATION"] = "1"
    image_path = None
    try:
        if request.get("image_data_uri"):
            image_path = write_image_data_uri(str(request["image_data_uri"]), output_dir / "input_image.png")
        output_path = get_sculptor_foundation().create_model(
            str(request.get("prompt") or "production miniature"),
            image_path=image_path,
            print_detail_um=35 if str(request.get("quality") or "").lower() == "high" else 50,
            max_detail_triangles=int(request.get("target_polycount") or 500_000) * 4,
        )
    finally:
        if previous_disable is None:
            os.environ.pop("MESHMEND_DISABLE_HOSTED_CREATION", None)
        else:
            os.environ["MESHMEND_DISABLE_HOSTED_CREATION"] = previous_disable
    target = output_dir / output_path.name
    target.write_bytes(Path(output_path).read_bytes())
    return {
        "model_file": target.name,
        "model_format": target.suffix.lower().lstrip("."),
        "provider": "legacy_sculptor",
        "warning": "This is the legacy procedural fallback, not the production generative backend.",
    }


def load_result_from_output(output_dir: Path, stdout: str) -> dict[str, Any]:
    result_json = output_dir / "result.json"
    if result_json.exists():
        return json.loads(result_json.read_text(encoding="utf-8"))
    stripped = (stdout or "").strip()
    if stripped.startswith("{"):
        return json.loads(stripped)
    for path in output_dir.iterdir():
        if path.suffix.lower() in SUPPORTED_MODEL_SUFFIXES:
            return {"model_file": path.name, "model_format": path.suffix.lower().lstrip(".")}
    return {}


def write_image_data_uri(data_uri: str, fallback_path: Path) -> Path:
    header, _, encoded = data_uri.partition(",")
    if not encoded:
        raise RuntimeError("image_data_uri is invalid")
    mime = "image/png"
    if header.startswith("data:") and ";" in header:
        mime = header[5:].split(";", 1)[0]
    allowed_mimes = {"image/png", "image/jpeg", "image/webp"}
    if mime not in allowed_mimes:
        raise RuntimeError(f"unsupported image MIME type: {mime}")
    suffix = mimetypes.guess_extension(mime) or ".png"
    path = fallback_path.with_suffix(suffix)
    data = base64.b64decode(encoded, validate=True)
    max_bytes = int(os.environ.get("MESHMEND_MAX_IMAGE_BYTES", "15728640"))
    if len(data) > max_bytes:
        raise RuntimeError(f"image_data_uri exceeds size limit: {len(data)} > {max_bytes} bytes")
    path.write_bytes(data)
    return path


def production_setup_hint() -> str:
    return (
        "For no-API local generation set MESHMEND_PRODUCTION_ENGINE=free_local_hunyuan, install Hunyuan3D-2/2.1, "
        "and optionally set MESHMEND_HUNYUAN3D_PATH to its repo. For a custom local runner, set "
        "MESHMEND_PRODUCTION_TEXT_TO_3D_COMMAND and/or MESHMEND_PRODUCTION_IMAGE_TO_3D_COMMAND. Commands receive "
        "{prompt_path}, {image_path}, {output_dir}, {quality}, and {target_polycount} placeholders and must write a "
        "supported model file or result.json into {output_dir}."
    )


def command_args(command: str) -> list[str]:
    if os.name == "nt":
        # POSIX splitting eats backslashes in unquoted Windows placeholders like
        # D:\MeshMend\...; non-POSIX splitting preserves them but keeps wrapping
        # quotes. Strip only those wrapping quotes after splitting.
        return [part[1:-1] if len(part) >= 2 and part[0] == part[-1] == '"' else part for part in shlex.split(command, posix=False)]
    return shlex.split(command, posix=True)


if __name__ == "__main__":
    raise SystemExit(main())
