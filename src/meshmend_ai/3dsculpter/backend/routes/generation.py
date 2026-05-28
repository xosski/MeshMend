from fastapi import APIRouter, UploadFile, File, Form, Header, HTTPException
from fastapi.responses import FileResponse
import io
import numpy as np
from PIL import Image
from pathlib import Path
import sys
import trimesh

from models.image_generator import ImageGenerator
from pipelines.image_to_mesh import reconstruct_mesh_from_image, reconstruct_mesh_from_image_volumetric
from utils.stl_processor import STLProcessor
from utils.config import OUTPUT_DIR
from processing.prompt_enhancer import MiniaturePromptEnhancer
from processing.scale_validator import ScaleValidator
from processing.mesh_simplifier import MeshSimplifier
from processing.printability import PrintabilityAnalyzer
from services.hosted_3d_service import Hosted3DError, hosted_3d_service
from services.subscription_service import subscription_service

router = APIRouter()

MESHMEND_PACKAGE_DIR = Path(__file__).resolve().parents[3]
MESHMEND_SRC_DIR = MESHMEND_PACKAGE_DIR.parent
for candidate in (MESHMEND_SRC_DIR, MESHMEND_PACKAGE_DIR):
    candidate_text = str(candidate)
    if candidate_text not in sys.path:
        sys.path.insert(0, candidate_text)

# Initialize models (lazy loading)
image_gen = None
neural_3d_model = None
local_3d_model = None


def _try_meshmend_sculptor_generation(prompt: str, image_path: Path | None, scale: str | float, quality: str = "standard") -> tuple[trimesh.Trimesh | None, str | None]:
    """Use MeshMend's prompt/image-aware miniature sculptor as primary backend."""
    try:
        from meshmend_ai.sculptor import get_sculptor_foundation

        scale_mm = _scale_to_mm(scale)
        output_path = get_sculptor_foundation().create_model(
            prompt,
            image_path=image_path,
            scale_mm=scale_mm,
            print_detail_um=50 if (quality or "standard").lower() == "high" else 100,
        )
        mesh = trimesh.load(output_path, force="mesh")
        if isinstance(mesh, trimesh.Trimesh) and len(mesh.faces) >= 1000:
            return mesh, "meshmend_image_text_sculptor"
    except Exception as exc:
        print(f"[GENERATION] MeshMend sculptor unavailable; falling back: {exc}")
    return None, None


def _hosted_text_generation_response(prompt: str, quality: str, api_key: str | None) -> dict | None:
    """Run the paid hosted text-to-3D backend when configured.

    If MESHMEND_USE_HOSTED_3D is off, callers fall back to local generation. If
    it is on and configured, hosted failure is surfaced instead of silently
    returning a procedural draft to a paying user.
    """
    if not hosted_3d_service.available():
        return None
    estimated_credits = 20
    account = subscription_service.authorize_generation(api_key, estimated_credits, "text_to_3d")
    result = hosted_3d_service.generate_text(prompt, quality=quality)
    credits = result.consumed_credits or estimated_credits
    subscription_service.consume_credits(
        account,
        credits,
        "text_to_3d",
        provider=result.provider,
        provider_task_id=result.task_id,
        metadata=f"quality={quality};format={result.model_format}",
    )
    return {
        "status": "success",
        "prompt": prompt,
        "enhanced_prompt": prompt,
        "generation_backend": f"hosted_{result.provider}_text_to_3d",
        "model_file": result.model_file,
        "model_format": result.model_format,
        "thumbnail_url": result.thumbnail_url,
        "provider_task_id": result.task_id,
        "provider_status": result.status,
        "consumed_credits": credits,
        "quality": quality,
        "mesh_info": {"format": result.model_format, "source": "hosted_model_service"},
        "printability": {
            "score": None,
            "difficulty": "requires slicer/import validation",
            "supports_needed": None,
            "estimated_time_minutes": None,
            "estimated_weight_grams": None,
        },
    }


def _hosted_image_generation_response(image_path: Path, prompt: str, quality: str, api_key: str | None) -> dict | None:
    if not hosted_3d_service.available():
        return None
    estimated_credits = 30
    account = subscription_service.authorize_generation(api_key, estimated_credits, "image_to_3d")
    result = hosted_3d_service.generate_image(image_path, prompt=prompt, quality=quality)
    credits = result.consumed_credits or estimated_credits
    subscription_service.consume_credits(
        account,
        credits,
        "image_to_3d",
        provider=result.provider,
        provider_task_id=result.task_id,
        metadata=f"quality={quality};format={result.model_format}",
    )
    return {
        "status": "success",
        "generation_backend": f"hosted_{result.provider}_image_to_3d",
        "model_file": result.model_file,
        "model_format": result.model_format,
        "thumbnail_url": result.thumbnail_url,
        "provider_task_id": result.task_id,
        "provider_status": result.status,
        "consumed_credits": credits,
        "mesh_info": {"format": result.model_format, "source": "hosted_model_service"},
    }


def _scale_to_mm(scale: str | float) -> float:
    if isinstance(scale, (int, float)):
        return float(scale)
    match = __import__("re").search(r"(\d+(?:\.\d+)?)", str(scale))
    return float(match.group(1)) if match else 32.0

def get_image_generator():
    global image_gen
    if image_gen is None:
        image_gen = ImageGenerator()
    return image_gen


def get_neural_3d_model():
    """Load the trained neural 3D diffusion checkpoint, if one exists."""
    global neural_3d_model
    if neural_3d_model is not None:
        return neural_3d_model
    try:
        try:
            from meshmend_ai.neural_diffusion import Neural3DDiffusionModel
        except ModuleNotFoundError:
            from neural_diffusion import Neural3DDiffusionModel

        neural_3d_model = Neural3DDiffusionModel.load_latest()
        if neural_3d_model is not None:
            print(f"[GENERATION] Loaded neural 3D diffusion checkpoint: {neural_3d_model.checkpoint_path}")
    except Exception as exc:
        print(f"[GENERATION] Neural 3D diffusion unavailable: {exc}")
        neural_3d_model = None
    return neural_3d_model


def get_local_3d_model():
    """Load the trained local 3D exemplar model, if one exists."""
    global local_3d_model
    if local_3d_model is not None:
        return local_3d_model
    try:
        try:
            from meshmend_ai.generative_model import Local3DGenerativeModel
        except ModuleNotFoundError:
            from generative_model import Local3DGenerativeModel

        local_3d_model = Local3DGenerativeModel.load_latest()
        if local_3d_model is not None:
            print(f"[GENERATION] Loaded local 3D exemplar checkpoint: {local_3d_model.checkpoint_path}")
    except Exception as exc:
        print(f"[GENERATION] Local 3D model unavailable: {exc}")
        local_3d_model = None
    return local_3d_model


def _try_trained_3d_generation(enhanced_prompt: str, quality: str) -> tuple[trimesh.Trimesh | None, str | None]:
    """Try trained 3D AI generators before falling back to image-to-mesh heuristics."""
    neural = get_neural_3d_model()
    if neural is not None:
        try:
            steps = 36 if (quality or "standard").lower() == "high" else None
            mesh = neural.generate(enhanced_prompt, steps=steps)
            if _is_usable_trained_mesh(mesh):
                return mesh, "neural_3d_diffusion"
            print("[GENERATION] Neural 3D output was too coarse/collapsed; falling back.")
        except Exception as exc:
            print(f"[GENERATION] Neural 3D generation failed; falling back: {exc}")

    local = get_local_3d_model()
    if local is not None:
        try:
            mesh = local.generate(enhanced_prompt)
            if _is_usable_trained_mesh(mesh):
                return mesh, "local_3d_exemplar"
            print("[GENERATION] Local 3D output was unusable; falling back.")
        except Exception as exc:
            print(f"[GENERATION] Local 3D generation failed; falling back: {exc}")

    return None, None


def _is_usable_trained_mesh(mesh) -> bool:
    """Gate trained 3D outputs so bad checkpoints don't replace better fallback paths."""
    if mesh is None:
        return False
    try:
        if len(mesh.vertices) < 1000 or len(mesh.faces) < 1800:
            return False
        if _is_collapsed_mesh(mesh) or _is_blob_like_mesh(mesh):
            return False
        return True
    except Exception:
        return False


def _is_blob_like_mesh(mesh) -> bool:
    """Detect collapsed/overfilled meshes that look like generic blobs."""
    try:
        if mesh is None or len(mesh.vertices) < 400 or len(mesh.faces) < 600:
            return True

        extents = mesh.extents
        max_extent = float(max(extents))
        min_extent = float(min(extents))
        if max_extent <= 1e-6:
            return True

        flat_ratio = min_extent / max_extent
        bbox_vol = float(extents[0] * extents[1] * extents[2])
        mesh_vol = float(abs(mesh.volume))
        fullness = mesh_vol / (bbox_vol + 1e-8)

        # Too flat or too filled are both common bad reconstructions.
        return flat_ratio < 0.08 or fullness > 0.72
    except Exception:
        return False


def _detail_preset(quality: str) -> dict:
    """Map quality mode to image synthesis and reconstruction fidelity."""
    q = (quality or "standard").lower().strip()
    if q == "high":
        return {
            "steps": 48,
            "guidance": 9.0,
            "size": 1024,
            "num_points": 52000,
            "retry_points": 70000,
        }
    if q == "low":
        return {
            "steps": 24,
            "guidance": 7.5,
            "size": 640,
            "num_points": 10000,
            "retry_points": 13000,
        }
    return {
        "steps": 36,
        "guidance": 8.0,
        "size": 768,
        "num_points": 26000,
        "retry_points": 36000,
    }


def _mesh_fullness_score(mesh) -> float:
    """Estimate how volumetrically filled a mesh is relative to its bounding box."""
    try:
        ext = np.asarray(mesh.extents, dtype=np.float32)
        bbox_vol = float(np.prod(np.maximum(ext, 1e-6)))
        mesh_vol = float(abs(mesh.volume))
        return mesh_vol / (bbox_vol + 1e-8)
    except Exception:
        return 0.0


def _is_underfleshed_mesh(mesh) -> bool:
    """Detect thin shell-like meshes that need volumetric thickening."""
    try:
        ext = np.asarray(mesh.extents, dtype=np.float32)
        max_ext = float(np.max(ext)) + 1e-8
        min_ext = float(np.min(ext))
        thickness_ratio = min_ext / max_ext
        fullness = _mesh_fullness_score(mesh)
        return thickness_ratio < 0.17 or fullness < 0.055
    except Exception:
        return True


def _flesh_out_mesh(mesh, quality: str = "standard"):
    """Thicken thin meshes while keeping silhouette/proportions intact."""
    try:
        q = (quality or "standard").lower().strip()
        if q == "high":
            min_ratio, iso_scale, axis_cap = 0.24, 1.08, 2.6
        elif q == "low":
            min_ratio, iso_scale, axis_cap = 0.18, 1.04, 2.0
        else:
            min_ratio, iso_scale, axis_cap = 0.21, 1.06, 2.3

        ext = np.asarray(mesh.extents, dtype=np.float32)
        max_ext = float(np.max(ext)) + 1e-8
        thin_axis = int(np.argmin(ext))
        min_target = max_ext * min_ratio
        axis_scale = max(1.0, min_target / (float(ext[thin_axis]) + 1e-8))

        center = np.asarray(mesh.center_mass, dtype=np.float32)
        verts = np.asarray(mesh.vertices, dtype=np.float32) - center[None, :]
        scale_vec = np.ones(3, dtype=np.float32) * iso_scale
        scale_vec[thin_axis] *= min(axis_scale, axis_cap)
        verts *= scale_vec[None, :]
        mesh.vertices = verts + center[None, :]

        mesh = MeshSimplifier.polish_mesh(mesh, quality=q)
        return mesh
    except Exception:
        return mesh


def _is_detail_poor_mesh(mesh) -> bool:
    """Detect meshes likely to look soft/blob-like due to low geometric complexity."""
    try:
        v_count = int(len(mesh.vertices))
        f_count = int(len(mesh.faces))
        if v_count < 1800 or f_count < 3000:
            return True

        ext = np.asarray(mesh.extents, dtype=np.float32)
        max_ext = float(np.max(ext)) + 1e-8
        diagonal = float(np.linalg.norm(ext)) + 1e-8
        density = f_count / (diagonal * max_ext + 1e-8)
        return density < 550.0
    except Exception:
        return True


def _is_collapsed_mesh(mesh) -> bool:
    """Strict failure gate: only true for unusable reconstructions."""
    try:
        if mesh is None or len(mesh.vertices) < 900 or len(mesh.faces) < 1400:
            return True

        ext = np.asarray(mesh.extents, dtype=np.float32)
        max_ext = float(np.max(ext)) + 1e-8
        min_ext = float(np.min(ext))
        flat_ratio = min_ext / max_ext
        if flat_ratio < 0.045:
            return True

        bbox_vol = float(np.prod(np.maximum(ext, 1e-6)))
        fullness = float(abs(mesh.volume)) / (bbox_vol + 1e-8)
        return fullness < 0.018 or fullness > 0.86
    except Exception:
        return True

@router.post("/from-text")
async def generate_from_text(
    prompt: str = Form(...),
    scale: str = Form("28mm"),
    quality: str = Form("standard"),
    material: str = Form("fdm"),
    x_api_key: str | None = Header(default=None),
):
    """Generate tabletop miniature from text description"""
    try:
        # Validate prompt
        is_valid, validation_msg = MiniaturePromptEnhancer.validate_prompt(prompt)
        if not is_valid:
            raise HTTPException(status_code=400, detail=f"Invalid prompt: {validation_msg}")
        
        # Enhance prompt for miniature generation
        print(f"\n[GENERATION] Prompt: {prompt}")
        enhanced_prompt = MiniaturePromptEnhancer.enhance_prompt(prompt, scale=scale)
        print(f"[GENERATION] Enhanced: {enhanced_prompt}")

        try:
            hosted_response = _hosted_text_generation_response(enhanced_prompt, quality, x_api_key)
            if hosted_response is not None:
                return hosted_response
        except PermissionError as exc:
            raise HTTPException(status_code=402, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=402, detail=str(exc)) from exc
        except Hosted3DError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

        preset = _detail_preset(quality)
        generation_backend = "meshmend_image_text_sculptor"
        image_filename = None
        image = None

        mesh, meshmend_backend = _try_meshmend_sculptor_generation(
            f"Original user prompt, preserve exact requested subject and traits: {prompt}. Enhanced cues: {enhanced_prompt}",
            image_path=None,
            scale=scale,
            quality=quality,
        )
        if meshmend_backend is not None:
            generation_backend = meshmend_backend
        else:
            generation_backend = "sdxl_image_to_mesh"

        # First try real trained 3D generators. If no checkpoint exists or the
        # checkpoint produces a coarse/collapsed mesh, fall back to SDXL +
        # image-to-mesh reconstruction below.
        if mesh is None:
            print("[GENERATION] Step 1: Checking trained 3D AI generators...")
            mesh, trained_backend = _try_trained_3d_generation(enhanced_prompt, quality)
            if trained_backend is not None:
                generation_backend = trained_backend
                print(f"[GENERATION] Using trained 3D backend: {generation_backend}")
            else:
                print("[GENERATION] No usable trained 3D output; using SDXL image-to-mesh pipeline.")
        
        file_hash = hash(prompt) % 100000

        if mesh is None:
            # Generate image with SDXL, then reconstruct a mesh from image/depth.
            print("[GENERATION] Step 2: Generating SDXL concept image...")
            image_gen = get_image_generator()
            image = image_gen.generate(
                enhanced_prompt,
                num_inference_steps=preset["steps"],
                guidance_scale=preset["guidance"],
                height=preset["size"],
                width=preset["size"],
            )

            # Save intermediate image
            image_filename = f"image_{file_hash}.png"
            image_path = OUTPUT_DIR / image_filename
            image.save(image_path)

            # Generate 3D mesh from image using point-cloud reconstruction
            print("[GENERATION] Step 3: Reconstructing 3D mesh from image...")
            mesh = reconstruct_mesh_from_image(image, num_points=preset["num_points"], method="poisson")
            if _is_blob_like_mesh(mesh):
                print("[GENERATION] Poisson mesh looked blob-like; retrying with ball-pivot reconstruction...")
                generation_backend = "sdxl_image_to_mesh_ball_pivot"
                mesh = reconstruct_mesh_from_image(image, num_points=preset["retry_points"], method="ball_pivot")
            if _is_collapsed_mesh(mesh):
                print("[GENERATION] Mesh collapsed; rebuilding via volumetric fallback...")
                generation_backend = "sdxl_image_to_mesh_volumetric"
                mesh = reconstruct_mesh_from_image_volumetric(image, vol_size=128 if quality == "high" else 112, voxel_size_mm=0.22)
        
        # Validate and normalize scale
        print(f"[GENERATION] Step 4: Validating scale (target: {scale})...")
        mesh, scale_report = ScaleValidator.validate_and_normalize(mesh, target_scale=scale)
        print(f"  Status: {scale_report['status']} - Height: {scale_report['final_height_mm']:.1f}mm")
        
        # Polish mesh surface/topology before optional simplification.
        print("[GENERATION] Step 5: Polishing surface...")
        mesh = MeshSimplifier.polish_mesh(mesh, quality=quality)
        
        # Flesh-out pass for thin/shell-like reconstructions.
        if _is_underfleshed_mesh(mesh):
            print("[GENERATION] Mesh is under-fleshed; thickening volume...")
            mesh = _flesh_out_mesh(mesh, quality=quality)

        # Simplify mesh for optimal printing
        print("[GENERATION] Step 6: Optimizing for printing...")
        mesh = MeshSimplifier.simplify_for_printing(mesh, quality=quality)

        # Analyze printability
        print("[GENERATION] Step 7: Analyzing printability...")
        print_analysis = PrintabilityAnalyzer.analyze(mesh, material=material)
        print(f"  Score: {print_analysis['score']}/100 - {print_analysis['print_difficulty']}")

        # Save STL
        print("[GENERATION] Step 8: Saving STL...")
        stl_filename = f"model_{file_hash}.stl"
        stl_path = OUTPUT_DIR / stl_filename
        STLProcessor.save_stl(mesh, str(stl_path))
        
        # Get mesh info
        mesh_info = STLProcessor.get_mesh_info(mesh)
        
        print("[GENERATION] [OK] Complete!")
        return {
            "status": "success",
            "prompt": prompt,
            "enhanced_prompt": enhanced_prompt,
            "generation_backend": generation_backend,
            "model_file": stl_filename,
            "image_file": image_filename,
            "scale": {
                "target": scale,
                "actual_height_mm": scale_report['final_height_mm'],
                "status": scale_report['status'],
            },
            "quality": quality,
            "material": material,
            "mesh_info": mesh_info,
            "printability": {
                "score": print_analysis['score'],
                "difficulty": print_analysis['print_difficulty'],
                "supports_needed": print_analysis['support_needed'],
                "estimated_time_minutes": print_analysis['estimated_print_time_minutes'],
                "estimated_weight_grams": print_analysis['estimated_weight_grams'],
            },
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/from-image")
async def generate_from_image(file: UploadFile = File(...), x_api_key: str | None = Header(default=None)):
    """Generate 3D model from uploaded image"""
    try:
        # Save uploaded image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        filename = Path(file.filename or "uploaded_reference").stem
        image_path = OUTPUT_DIR / f"{filename}_reference.png"
        image.convert("RGBA").save(image_path)

        hosted_prompt = (
            f"uploaded reference image subject; filename subject cues: {filename}; "
            "create a high quality printable 3D model preserving the image subject topology and detail"
        )
        try:
            hosted_response = _hosted_image_generation_response(image_path, hosted_prompt, "high", x_api_key)
            if hosted_response is not None:
                return hosted_response
        except PermissionError as exc:
            raise HTTPException(status_code=402, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=402, detail=str(exc)) from exc
        except Hosted3DError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

        mesh, backend = _try_meshmend_sculptor_generation(
            f"original resin-printable 3D model based on the uploaded reference image; filename subject cues: {filename}; preserve the image subject topology instead of forcing a humanoid miniature; analyze the image silhouette, pose, clothing, equipment, object shape, creature traits, texture, and material cues",
            image_path=image_path,
            scale="32mm",
            quality="high",
        )
        
        # Generate 3D mesh from uploaded image
        if mesh is None:
            backend = "uploaded_image_to_mesh_reconstruction"
            mesh = reconstruct_mesh_from_image(image, num_points=20000, method="poisson")
            if _is_blob_like_mesh(mesh):
                print("[GENERATION] Poisson mesh looked blob-like for uploaded image; retrying with ball-pivot...")
                mesh = reconstruct_mesh_from_image(image, num_points=26000, method="ball_pivot")
            if _is_collapsed_mesh(mesh):
                print("[GENERATION] Uploaded mesh collapsed; rebuilding via volumetric fallback...")
                mesh = reconstruct_mesh_from_image_volumetric(image, vol_size=128, voxel_size_mm=0.22)

        # Polish and lightly optimize before export.
        mesh = MeshSimplifier.polish_mesh(mesh, quality="high")
        if _is_underfleshed_mesh(mesh):
            print("[GENERATION] Uploaded-image mesh is under-fleshed; thickening volume...")
            mesh = _flesh_out_mesh(mesh, quality="high")
        mesh = MeshSimplifier.simplify_for_printing(mesh, quality="high")
        
        # Save STL
        stl_path = OUTPUT_DIR / f"{filename}_3d.stl"
        STLProcessor.save_stl(mesh, str(stl_path))
        
        # Save mesh info
        mesh_info = STLProcessor.get_mesh_info(mesh)
        
        return {
            "status": "success",
            "model_file": f"{filename}_3d.stl",
            "generation_backend": backend,
            "mesh_info": mesh_info,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/download/{model_file}")
async def download_model(model_file: str):
    """Download generated STL model"""
    try:
        file_path = OUTPUT_DIR / model_file
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="Model not found")
        
        return FileResponse(
            path=file_path,
            media_type="application/octet-stream",
            filename=model_file
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/list")
async def list_models():
    """List all generated models"""
    try:
        models = []
        for file in OUTPUT_DIR.glob("*.stl"):
            mesh = STLProcessor.load_stl(str(file))
            info = STLProcessor.get_mesh_info(mesh)
            models.append({
                "filename": file.name,
                "info": info,
            })
        return {"models": models}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
