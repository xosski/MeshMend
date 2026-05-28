from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from routes.generation import router as generation_router
from routes.training import router as training_router
from routes.models import router as models_router
from routes.billing import router as billing_router
from utils.config import UPLOAD_DIR, OUTPUT_DIR

app = FastAPI(title="3D Sculpting AI", version="1.0.0")

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create necessary directories
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Include routers
app.include_router(generation_router, prefix="/api/generate", tags=["generation"])
app.include_router(training_router, prefix="/api/train", tags=["training"])
app.include_router(models_router, prefix="/api/models", tags=["models"])
app.include_router(billing_router, prefix="/api/billing", tags=["billing"])

@app.get("/")
async def root():
    return {"message": "3D Sculpting AI API", "status": "running"}

@app.get("/health")
async def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
