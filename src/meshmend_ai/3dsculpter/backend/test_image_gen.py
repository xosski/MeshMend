#!/usr/bin/env python3
"""Test image generation standalone"""

import sys
import torch
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

from models.image_generator import ImageGenerator
from utils.config import DEVICE

def test_image_generation():
    print(f"Device: {DEVICE}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    
    print("\nInitializing ImageGenerator...")
    gen = ImageGenerator()
    
    prompts = [
        "A detailed Warhammer 40k Space Marine with bolter",
        "A sci-fi robot",
        "A fantasy dragon"
    ]
    
    for prompt in prompts:
        try:
            print(f"\nGenerating for: '{prompt}'")
            image = gen.generate(prompt, num_inference_steps=15)
            
            # Save test image
            test_file = Path(__file__).parent / "test_outputs" / f"test_{prompts.index(prompt)}.png"
            test_file.parent.mkdir(exist_ok=True)
            image.save(test_file)
            print(f"✓ Saved to: {test_file}")
            
        except Exception as e:
            print(f"✗ Error: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    test_image_generation()
