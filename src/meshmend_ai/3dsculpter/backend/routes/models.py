from fastapi import APIRouter, HTTPException
from pathlib import Path
import sys
import torch
from services.hosted_3d_service import SELF_HOSTED_MODEL_SERVICE_URL, hosted_3d_service, HOSTED_PROVIDER, HOSTED_TARGET_FORMATS
from services.subscription_service import REQUIRE_SUBSCRIPTION

router = APIRouter()

MESHMEND_PACKAGE_DIR = Path(__file__).resolve().parents[3]
MESHMEND_SRC_DIR = MESHMEND_PACKAGE_DIR.parent
for candidate in (MESHMEND_SRC_DIR, MESHMEND_PACKAGE_DIR):
    candidate_text = str(candidate)
    if candidate_text not in sys.path:
        sys.path.insert(0, candidate_text)


def _checkpoint_status() -> dict:
    status = {
        "local_3d_exemplar_checkpoint": None,
        "local_3d_exemplar_available": False,
        "neural_3d_diffusion_checkpoint": None,
        "neural_3d_diffusion_available": False,
    }
    try:
        try:
            from meshmend_ai.generative_model import default_checkpoint_path
        except ModuleNotFoundError:
            from generative_model import default_checkpoint_path
        local_path = default_checkpoint_path().parent / "latest_model.json"
        status["local_3d_exemplar_checkpoint"] = str(local_path)
        status["local_3d_exemplar_available"] = local_path.exists()
    except Exception:
        pass
    try:
        try:
            from meshmend_ai.neural_diffusion import default_neural_checkpoint_path
        except ModuleNotFoundError:
            from neural_diffusion import default_neural_checkpoint_path
        neural_path = default_neural_checkpoint_path().parent / "latest_neural_model.pt"
        status["neural_3d_diffusion_checkpoint"] = str(neural_path)
        status["neural_3d_diffusion_available"] = neural_path.exists()
    except Exception:
        pass
    return status

@router.get("/info")
async def get_models_info():
    """Get information about loaded models"""
    try:
        return {
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "cuda_available": torch.cuda.is_available(),
            "torch_version": torch.__version__,
            "checkpoints": _checkpoint_status(),
            "hosted_3d": {
                "enabled": hosted_3d_service.enabled,
                "available": hosted_3d_service.available(),
                "provider": HOSTED_PROVIDER,
                "self_hosted_model_service_url_configured": bool(SELF_HOSTED_MODEL_SERVICE_URL),
                "target_formats": HOSTED_TARGET_FORMATS,
                "subscription_required": REQUIRE_SUBSCRIPTION,
            },
            "models": {
                "hosted_3d_primary": "Embedded in-program premium generator by default; optional self-hosted worker only if MESHMEND_HOSTED_3D_PROVIDER=self_hosted",
                "text_to_image": "Stable Diffusion XL",
                "trained_3d_generator_primary": "Neural3DDiffusionModel when latest_neural_model.pt exists and passes quality gate",
                "trained_3d_generator_secondary": "Local3DGenerativeModel exemplar checkpoint when latest_model.json exists",
                "image_to_mesh": "Depth-estimation point cloud + Open3D Poisson/Ball Pivot fallback + volumetric silhouette fallback",
                "mesh_postprocessing": "ScaleValidator + MeshSimplifier + PrintabilityAnalyzer",
            },
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/unload")
async def unload_models():
    """Unload models to free VRAM"""
    try:
        import routes.generation as generation
        
        if generation.image_gen:
            generation.image_gen.unload_model()
            generation.image_gen = None

        generation.local_3d_model = None
        generation.neural_3d_model = None

        torch.cuda.empty_cache()
        
        return {"status": "success", "message": "Models unloaded"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
