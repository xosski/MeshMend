#!/usr/bin/env python3
"""Test 3D generation pipeline"""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from models.image_generator import ImageGenerator
from models.point_e_generator import PointEGenerator
from models.mesh_reconstructor import MeshReconstructor
from utils.stl_processor import STLProcessor

def test_3d_pipeline():
    print("=" * 60)
    print("Testing 3D Generation Pipeline")
    print("=" * 60)
    
    # Step 1: Generate image
    print("\n[1/4] Generating image...")
    img_gen = ImageGenerator()
    image = img_gen.generate("A detailed robot head", num_inference_steps=10)
    print(f"[OK] Image generated: {image.size}")
    
    # Step 2: Generate point cloud
    print("\n[2/4] Generating point cloud...")
    point_gen = PointEGenerator()
    points = point_gen.generate(image)
    print(f"[OK] Point cloud generated: {points.shape}")
    print(f"  Min: {points.min(axis=0)}")
    print(f"  Max: {points.max(axis=0)}")
    print(f"  Range (Z): {points[:, 2].max() - points[:, 2].min():.4f}")
    print(f"  Unique Z values: {len(np.unique(np.round(points[:, 2], 4)))}")
    
    # Check if points are valid
    if np.all(points == 0):
        print("  [ERROR] All points are zero!")
    elif np.isnan(points).any():
        print("  [ERROR] Contains NaN values!")
    elif points[:, 2].max() - points[:, 2].min() < 0.01:
        print("  [ERROR] All points are nearly flat (2D)! Point-E not working.")
    else:
        print(f"  [OK] Points look valid")
    
    # Step 3: Convert to mesh
    print("\n[3/4] Converting to mesh...")
    try:
        mesh = MeshReconstructor.pointcloud_to_mesh(points, method="poisson")
        print(f"[OK] Mesh created:")
        print(f"  Vertices: {len(mesh.vertices)}")
        print(f"  Faces: {len(mesh.faces)}")
        print(f"  Volume: {mesh.volume:.4f}")
        print(f"  Surface area: {mesh.area:.4f}")
        
        if len(mesh.vertices) == 0:
            print("  [ERROR] Mesh has no vertices!")
        else:
            # Step 4: Save STL
            print("\n[4/4] Saving STL...")
            output_path = Path(__file__).parent / "test_outputs" / "test_model.stl"
            output_path.parent.mkdir(exist_ok=True)
            
            STLProcessor.save_stl(mesh, str(output_path))
            print(f"[OK] STL saved: {output_path}")
            print(f"  File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
            
    except Exception as e:
        print(f"[ERROR] Error creating mesh: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_3d_pipeline()
