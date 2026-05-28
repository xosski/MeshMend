from pathlib import Path
import torch

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
BACKEND_DIR = PROJECT_ROOT / "backend"
UPLOAD_DIR = BACKEND_DIR / "uploads"
OUTPUT_DIR = BACKEND_DIR / "outputs"
TRAINING_DATA_DIR = PROJECT_ROOT / "training_data"
EXAMPLES_DIR = TRAINING_DATA_DIR / "examples"
PROCESSED_DIR = TRAINING_DATA_DIR / "processed"

# Model configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_HALF_PRECISION = torch.cuda.is_available()

# Stable Diffusion config - using XL for better quality
SD_MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
SD_PRECISION = torch.float16 if USE_HALF_PRECISION else torch.float32

# Point-E config
POINT_E_MODEL = "openai/point-e"

# Training config
TRAINING_CONFIG = {
    "batch_size": 4,
    "learning_rate": 1e-4,
    "epochs": 10,
    "device": DEVICE,
}

# Ensure directories exist
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
