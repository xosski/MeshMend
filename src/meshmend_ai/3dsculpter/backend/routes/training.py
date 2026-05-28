from fastapi import APIRouter, Form, HTTPException
import json
from pathlib import Path
import sys
import torch
import numpy as np

from utils.stl_processor import STLProcessor
from utils.config import EXAMPLES_DIR, PROCESSED_DIR

router = APIRouter()

MESHMEND_PACKAGE_DIR = Path(__file__).resolve().parents[3]
MESHMEND_SRC_DIR = MESHMEND_PACKAGE_DIR.parent
for candidate in (MESHMEND_SRC_DIR, MESHMEND_PACKAGE_DIR):
    candidate_text = str(candidate)
    if candidate_text not in sys.path:
        sys.path.insert(0, candidate_text)


def _source_dir_or_default(source_dir: str | None) -> Path:
    if source_dir:
        return Path(source_dir).expanduser().resolve()
    return EXAMPLES_DIR.resolve()


def _reset_generation_model_cache() -> None:
    """Ensure generation routes reload fresh checkpoints after training."""
    try:
        import routes.generation as generation

        generation.local_3d_model = None
        generation.neural_3d_model = None
    except Exception:
        pass

@router.post("/scan-examples")
async def scan_example_files():
    """Scan for example STL files to train on"""
    try:
        examples = []
        
        # Scan the project root for STL files
        project_root = EXAMPLES_DIR.parent.parent
        for stl_file in project_root.glob("*.stl"):
            try:
                mesh = STLProcessor.load_stl(str(stl_file))
                info = STLProcessor.get_mesh_info(mesh)
                examples.append({
                    "filename": stl_file.name,
                    "path": str(stl_file),
                    "info": info,
                })
            except Exception as e:
                print(f"Error processing {stl_file}: {e}")
        
        return {
            "status": "success",
            "examples_found": len(examples),
            "examples": examples,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/process-examples")
async def process_example_files():
    """Process example STL files for training"""
    try:
        project_root = EXAMPLES_DIR.parent.parent
        processed_count = 0
        
        for stl_file in project_root.glob("*.stl"):
            try:
                # Load and process mesh
                mesh = STLProcessor.load_stl(str(stl_file))
                
                # Normalize
                normalized_mesh = STLProcessor.normalize_mesh(mesh)
                
                # Convert to point cloud
                points = STLProcessor.mesh_to_pointcloud(normalized_mesh)
                
                # Save processed data
                output_file = PROCESSED_DIR / f"{stl_file.stem}_points.npy"
                np.save(output_file, points)
                
                # Save metadata
                meta_file = PROCESSED_DIR / f"{stl_file.stem}_meta.json"
                metadata = {
                    "original_file": stl_file.name,
                    "vertices": int(len(mesh.vertices)),
                    "faces": int(len(mesh.faces)),
                    "surface_area": float(mesh.area),
                    "volume": float(mesh.volume),
                }
                with open(meta_file, 'w') as f:
                    json.dump(metadata, f)
                
                processed_count += 1
                print(f"Processed: {stl_file.name}")
                
            except Exception as e:
                print(f"Error processing {stl_file}: {e}")
        
        return {
            "status": "success",
            "processed_count": processed_count,
            "output_dir": str(PROCESSED_DIR),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/fine-tune")
async def fine_tune_model(
    source_dir: str | None = Form(None),
    train_neural: bool = Form(False),
    neural_resolution: int = Form(96),
    autoencoder_epochs: int = Form(50),
    diffusion_epochs: int = Form(90),
):
    """Train MeshMend's local 3D generator and optionally the neural 3D diffusion model."""
    try:
        training_dir = _source_dir_or_default(source_dir)
        if not training_dir.exists():
            raise HTTPException(status_code=400, detail=f"Training directory does not exist: {training_dir}")

        try:
            from meshmend_ai.generative_model import Local3DGenerativeModel
            from meshmend_ai.high_resolution_latent import LocalMeshLatentGenerator
        except ModuleNotFoundError:
            from generative_model import Local3DGenerativeModel
            from high_resolution_latent import LocalMeshLatentGenerator

        messages: list[str] = []
        local_progress: list[str] = []
        local_result = Local3DGenerativeModel.train_from_directory(
            training_dir,
            progress=lambda _percent, message: local_progress.append(message),
        )
        messages.append(local_result.message)
        mesh_latent_progress: list[str] = []
        mesh_latent_result = LocalMeshLatentGenerator.train_from_directory(
            training_dir,
            progress=lambda _percent, message: mesh_latent_progress.append(message),
        )
        messages.append(mesh_latent_result.message)

        neural_result_payload = None
        if train_neural:
            try:
                from meshmend_ai.neural_diffusion import Neural3DDiffusionModel, NeuralTrainingConfig
            except ModuleNotFoundError:
                from neural_diffusion import Neural3DDiffusionModel, NeuralTrainingConfig

            neural_progress: list[str] = []
            neural_result = Neural3DDiffusionModel.train_from_directory(
                training_dir,
                config=NeuralTrainingConfig(
                    resolution=neural_resolution,
                    autoencoder_epochs=autoencoder_epochs,
                    diffusion_epochs=diffusion_epochs,
                    device="auto",
                ),
                progress=lambda _percent, message: neural_progress.append(message),
            )
            neural_result_payload = {
                "checkpoint_path": neural_result.checkpoint_path,
                "manifest_path": neural_result.manifest_path,
                "examples": neural_result.examples,
                "resolution": neural_result.resolution,
                "checkpoint_size_bytes": neural_result.checkpoint_size_bytes,
                "data_signature": neural_result.data_signature,
                "progress_tail": neural_progress[-8:],
            }
            messages.append(neural_result.message)

        _reset_generation_model_cache()

        return {
            "status": "success",
            "message": " ".join(messages),
            "source_dir": str(training_dir),
            "local_3d_model": {
                "checkpoint_path": local_result.checkpoint_path,
                "examples": local_result.examples,
                "images": local_result.images,
                "progress_tail": local_progress[-8:],
            },
            "mesh_latent_model": {
                "checkpoint_path": mesh_latent_result.checkpoint_path,
                "assets": mesh_latent_result.assets,
                "progress_tail": mesh_latent_progress[-8:],
            },
            "neural_3d_diffusion": neural_result_payload,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status")
async def training_status():
    """Get training status"""
    try:
        processed_files = list(PROCESSED_DIR.glob("*_points.npy"))
        
        return {
            "status": "ready",
            "processed_examples": len(processed_files),
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "training_examples_dir": str(EXAMPLES_DIR),
            "fine_tune_wires_generation": True,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
