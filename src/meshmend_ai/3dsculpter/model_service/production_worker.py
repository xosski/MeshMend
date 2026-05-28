from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any


SUPPORTED_MODEL_SUFFIXES = {".stl", ".glb", ".obj", ".ply", ".3mf", ".fbx", ".usdz"}


def main() -> int:
    parser = argparse.ArgumentParser(description="MeshMend production 3D model worker")
    parser.add_argument("--input", required=True, help="Path to request JSON")
    parser.add_argument("--output-dir", required=True, help="Directory where model outputs should be written")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        request = json.loads(input_path.read_text(encoding="utf-8"))
        workflow = str(request.get("workflow") or "text_to_3d")
        engine = os.environ.get("MESHMEND_PRODUCTION_ENGINE", "external").strip().lower()
        if engine in {"external", "command"}:
            result = run_external_engine(request, input_path, output_dir, workflow)
        elif engine in {"free_local", "free_local_hunyuan", "hunyuan", "hunyuan3d"}:
            result = run_free_local_hunyuan(request, input_path, output_dir, workflow)
        elif engine in {"legacy_sculptor", "embedded"}:
            result = run_legacy_sculptor(request, output_dir)
        else:
            raise RuntimeError(f"Unsupported MESHMEND_PRODUCTION_ENGINE: {engine}")
        (output_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result))
        return 0
    except Exception as exc:
        error = {
            "error": str(exc),
            "hint": production_setup_hint(),
        }
        (output_dir / "result.json").write_text(json.dumps(error, indent=2), encoding="utf-8")
        print(json.dumps(error), file=sys.stderr)
        return 1


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
        shlex.split(command, posix=os.name != "nt"),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=float(os.environ.get("MESHMEND_PRODUCTION_COMMAND_TIMEOUT_SECONDS", "7200")),
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or completed.stdout.strip() or f"production command exited {completed.returncode}")
    result = load_result_from_output(output_dir, completed.stdout)
    if not result.get("model_file") and not result.get("model_urls"):
        raise RuntimeError("production command completed but did not produce a supported model file or result.json")
    return result


def run_free_local_hunyuan(request: dict[str, Any], input_path: Path, output_dir: Path, workflow: str) -> dict[str, Any]:
    """Run a no-API local Hunyuan3D backend.

    Hunyuan3D-2/2.1 is primarily image-to-3D. For text prompts, MeshMend first
    creates a local concept image, then sends that image through Hunyuan3D.
    Everything runs on the user's machine; no hosted API key is used.
    """
    image_path = None
    if workflow == "image_to_3d" and request.get("image_data_uri"):
        image_path = write_image_data_uri(str(request["image_data_uri"]), output_dir / "input_image.png")
    if image_path is None:
        image_path = generate_local_concept_image(request, input_path, output_dir)
    return run_hunyuan_image_to_3d(image_path, request, output_dir)


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
            shlex.split(command, posix=os.name != "nt"),
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
        raise RuntimeError(
            "Text prompts need a local text-to-image step before Hunyuan3D image-to-3D. "
            "Set MESHMEND_FREE_LOCAL_TEXT_TO_IMAGE_COMMAND to a no-API local image generator, or install/configure diffusers. "
            f"Diffusers fallback failed: {exc}"
        ) from exc


def generate_diffusers_concept_image(request: dict[str, Any], output_dir: Path) -> Path:
    from diffusers import StableDiffusionPipeline
    import torch

    quality = str(request.get("quality") or "standard").lower()
    model_id = os.environ.get(
        "MESHMEND_FREE_LOCAL_IMAGE_MODEL",
        "stabilityai/stable-diffusion-xl-base-1.0" if quality == "high" else "runwayml/stable-diffusion-v1-5",
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    prompt = str(request.get("prompt") or "")
    miniature_prompt = (
        f"solo single character: {prompt}, one single centered full-body subject only, exactly one miniature figure, "
        "one person, isolated single model, centered product render, no extra characters, "
        "no duplicate models, no lineup, no army, no multiple views, no turnaround sheet, no collage, "
        "production concept art for a resin tabletop miniature, orthographic three-quarter front view, full subject visible, "
        "crisp silhouette, highly detailed armor trim, weapon bevels, face details, pouches, straps, panel lines, "
        "engraved recesses, layered cloth leather metal material texture, sharp focus, ultra crisp details, "
        "studio product render, high contrast, plain white background"
    )
    negative_prompt = (
        "multiple characters, duplicate, duplicates, lineup, army, squad, group, crowd, four views, reference sheet, "
        "turnaround, collage, split screen, grid, extra bodies, extra heads, extra weapons, blurry, cropped, text, "
        "watermark, low detail, smooth blob, blocky, cube, cheese, holes, malformed, plain smooth armor"
    )
    pipe = load_text_to_image_pipeline(model_id, dtype)
    pipe = pipe.to(device)
    if hasattr(pipe, "enable_attention_slicing"):
        pipe.enable_attention_slicing()
    if hasattr(pipe, "enable_vae_slicing"):
        pipe.enable_vae_slicing()

    steps = int(os.environ.get("MESHMEND_FREE_LOCAL_IMAGE_STEPS", "42" if quality == "high" else "30"))
    guidance = float(os.environ.get("MESHMEND_FREE_LOCAL_IMAGE_GUIDANCE", "7.0" if "xl" in model_id.lower() else "8.0"))
    size = int(os.environ.get("MESHMEND_FREE_LOCAL_IMAGE_SIZE", "1024" if quality == "high" else "768"))
    candidates = max(1, int(os.environ.get("MESHMEND_CONCEPT_CANDIDATES", "3" if quality == "high" else "1")))
    best_path = None
    best_score = -1.0
    for index in range(candidates):
        generator = None
        seed_text = os.environ.get("MESHMEND_FREE_LOCAL_IMAGE_SEED", "").strip()
        if seed_text:
            generator = torch.Generator(device=device).manual_seed(int(seed_text) + index)
        result = pipe(
            prompt=miniature_prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=steps,
            guidance_scale=guidance,
            height=size,
            width=size,
            generator=generator,
        )
        image = result.images[0]
        concept_path = output_dir / f"concept_{index + 1}.png"
        image.save(concept_path)
        isolated_path = output_dir / f"concept_single_subject_{index + 1}.png"
        isolate_single_subject_concept(concept_path, isolated_path)
        candidate_path = isolated_path if isolated_path.exists() else concept_path
        score = concept_quality_score(candidate_path)
        if score > best_score:
            best_score = score
            best_path = candidate_path

    final_path = output_dir / "concept_single_subject.png"
    if best_path is not None:
        final_path.write_bytes(Path(best_path).read_bytes())
        return final_path
    raise RuntimeError("local text-to-image did not produce a concept image")


def load_text_to_image_pipeline(model_id: str, dtype: Any) -> Any:
    try:
        if "xl" in model_id.lower() or "sdxl" in model_id.lower():
            from diffusers import StableDiffusionXLPipeline

            return StableDiffusionXLPipeline.from_pretrained(model_id, torch_dtype=dtype, use_safetensors=True)
    except Exception:
        pass
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
        if rows.size and cols.size:
            fg_w = (cols.max() - cols.min() + 1) / arr.shape[1]
            fg_h = (rows.max() - rows.min() + 1) / arr.shape[0]
            width_penalty = max(0.0, (fg_w / max(fg_h, 1e-6)) - 0.75) * 0.08
        return sharpness + contrast * 0.35 - width_penalty
    except Exception:
        return 0.0


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

        comp = labeled == best_label
        rows, cols = np.where(comp)
        pad_x = int(w * 0.08)
        pad_y = int(h * 0.08)
        left = max(0, int(cols.min()) - pad_x)
        right = min(w, int(cols.max()) + pad_x)
        top = max(0, int(rows.min()) - pad_y)
        bottom = min(h, int(rows.max()) + pad_y)
        crop = image.crop((left, top, right, bottom))
        # Put the isolated subject back on a square white canvas, centered.
        size = max(crop.size)
        canvas = Image.new("RGB", (size, size), "white")
        canvas.paste(crop, ((size - crop.size[0]) // 2, (size - crop.size[1]) // 2))
        canvas = canvas.resize(image.size, Image.Resampling.LANCZOS).filter(ImageFilter.SHARPEN)
        canvas.save(output_path)
    except Exception:
        return


def crop_single_panel_from_lineup(image: Any, foreground_mask: Any) -> Any | None:
    """If the concept is a 4-model lineup, keep one panel before Hunyuan sees it."""
    try:
        from PIL import Image, ImageFilter
        import numpy as np
        from scipy.ndimage import binary_fill_holes

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
        return canvas.resize(image.size, Image.Resampling.LANCZOS).filter(ImageFilter.SHARPEN)
    except Exception:
        return None


def run_hunyuan_image_to_3d(image_path: Path, request: dict[str, Any], output_dir: Path) -> dict[str, Any]:
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
    generator_kwargs = hunyuan_generation_kwargs(request)
    try:
        mesh_result = pipeline(image=str(image_path), **generator_kwargs)
    except TypeError:
        # Hunyuan3D versions differ in exposed inference kwargs. If a version
        # rejects our quality knobs, still generate rather than failing.
        mesh_result = pipeline(image=str(image_path))
    mesh = mesh_result[0] if isinstance(mesh_result, (list, tuple)) else mesh_result
    mesh, postprocess_report = postprocess_miniature(mesh, request)

    output_format = os.environ.get("MESHMEND_HUNYUAN3D_OUTPUT_FORMAT", "stl").strip().lower().lstrip(".") or "stl"
    if f".{output_format}" not in SUPPORTED_MODEL_SUFFIXES:
        output_format = "stl"
    model_file = output_dir / f"meshmend_hunyuan.{output_format}"
    export_mesh(mesh, model_file)
    if not model_file.exists():
        raise RuntimeError("Hunyuan3D completed but no mesh file was exported")
    mesh_info = mesh_export_info(mesh, request)
    return {
        "model_file": model_file.name,
        "model_format": output_format,
        "provider": "free_local_hunyuan3d",
        "source_image": image_path.name,
        "model_path": model_path,
        "subfolder": subfolder,
        "mesh_info": postprocess_report.to_dict() | {"export_info": mesh_info},
        "consumed_credits": 0,
    }


def hunyuan_generation_kwargs(request: dict[str, Any]) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    quality = str(request.get("quality") or "standard").lower()
    steps = os.environ.get("MESHMEND_HUNYUAN3D_STEPS", "").strip()
    kwargs["num_inference_steps"] = int(steps or (48 if quality == "high" else 36))
    guidance = os.environ.get("MESHMEND_HUNYUAN3D_GUIDANCE", "").strip()
    if guidance:
        kwargs["guidance_scale"] = float(guidance)
    seed = os.environ.get("MESHMEND_HUNYUAN3D_SEED", "").strip()
    if seed:
        try:
            import torch

            kwargs["generator"] = torch.Generator().manual_seed(int(seed))
        except Exception:
            pass
    octree_resolution = os.environ.get("MESHMEND_HUNYUAN3D_OCTREE_RESOLUTION", "").strip()
    kwargs["octree_resolution"] = int(octree_resolution or (384 if quality == "high" else 256))
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
    suffix = mimetypes.guess_extension(mime) or ".png"
    path = fallback_path.with_suffix(suffix)
    path.write_bytes(base64.b64decode(encoded))
    return path


def production_setup_hint() -> str:
    return (
        "For no-API local generation set MESHMEND_PRODUCTION_ENGINE=free_local_hunyuan, install Hunyuan3D-2/2.1, "
        "and optionally set MESHMEND_HUNYUAN3D_PATH to its repo. For a custom local runner, set "
        "MESHMEND_PRODUCTION_TEXT_TO_3D_COMMAND and/or MESHMEND_PRODUCTION_IMAGE_TO_3D_COMMAND. Commands receive "
        "{prompt_path}, {image_path}, {output_dir}, {quality}, and {target_polycount} placeholders and must write a "
        "supported model file or result.json into {output_dir}."
    )


if __name__ == "__main__":
    raise SystemExit(main())
