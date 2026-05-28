import torch
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler
from PIL import Image
import os
from utils.config import SD_MODEL_ID, DEVICE, SD_PRECISION

class ImageGenerator:
    """Generate images from text prompts using Stable Diffusion XL"""
    
    def __init__(self):
        self.device = DEVICE
        self.pipeline = None
        self.load_model()
    
    def load_model(self):
        """Load Stable Diffusion XL model"""
        print(f"Loading Stable Diffusion XL on {self.device}...")
        try:
            self.pipeline = StableDiffusionXLPipeline.from_pretrained(
                SD_MODEL_ID,
                torch_dtype=SD_PRECISION,
                safety_checker=None,
                use_auth_token=False,
                variant="fp16" if SD_PRECISION == torch.float16 else None
            )
            self.pipeline = self.pipeline.to(self.device)
            
            # Use faster scheduler
            self.pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
                self.pipeline.scheduler.config
            )
            
            # Enable memory optimizations
            self.pipeline.enable_attention_slicing()
            if self.device == "cuda":
                self.pipeline.enable_model_cpu_offload()
            
            print("Stable Diffusion XL loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def generate(self, prompt: str, num_inference_steps: int = 15, guidance_scale: float = 7.0, height: int = 512, width: int = 512) -> Image.Image:
        """Generate image from text prompt - optimized for speed"""
        if not prompt or not isinstance(prompt, str):
            prompt = "a detailed 3D sculpture"
        
        prompt = prompt.strip()
        
        # Enhance prompt for miniature generation
        if "marine" in prompt.lower() or "warrior" in prompt.lower() or "soldier" in prompt.lower():
            prompt = f"{prompt}, tabletop miniature, 28mm scale, highly detailed, professional sculpt, sharp focus"

        # Enforce Meshy-like single-character framing to help downstream meshing.
        if "single character" not in prompt.lower():
            prompt += ", single character, full body, centered composition, plain studio backdrop, no scenery"

        negative_prompt = (
            "blurry, low quality, deformed, ugly, bad anatomy, oversaturated, "
            "multiple characters, crowd, animals, duck, creature blob, abstract shape, "
            "busy background, scenery, landscape, text, watermark"
        )
        
        print(f"Generating image for prompt: '{prompt}'")
        print(f"Steps: {num_inference_steps}, Guidance: {guidance_scale}, Size: {height}x{width}, Device: {self.device}")
        
        try:
            # SDXL with optimized parameters for faster inference
            result = self.pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                height=height,
                width=width,
                generator=torch.Generator(device=self.device).manual_seed(42)
            )
            
            image = result.images[0]
            print(f"Image generated successfully: {image.size}")
            return image
        except Exception as e:
            print(f"Error generating image: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def generate_batch(self, prompts: list, **kwargs) -> list:
        """Generate multiple images"""
        images = []
        for prompt in prompts:
            image = self.generate(prompt, **kwargs)
            images.append(image)
        return images
    
    def unload_model(self):
        """Free up VRAM"""
        if self.pipeline is not None:
            self.pipeline = None
            torch.cuda.empty_cache()
