import trimesh
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import torch

class STLProcessor:
    """Process STL files for training and analysis"""
    
    @staticmethod
    def load_stl(filepath: str) -> trimesh.Trimesh:
        """Load STL file"""
        try:
            mesh = trimesh.load(filepath)
            return mesh
        except Exception as e:
            raise Exception(f"Error loading STL: {e}")
    
    @staticmethod
    def get_mesh_info(mesh: trimesh.Trimesh) -> dict:
        """Extract features from mesh for training"""
        return {
            "vertices_count": len(mesh.vertices),
            "faces_count": len(mesh.faces),
            "surface_area": float(mesh.area),
            "volume": float(mesh.volume),
            "bounds": mesh.bounds.tolist(),
            "center_mass": mesh.center_mass.tolist(),
        }
    
    @staticmethod
    def normalize_mesh(mesh: trimesh.Trimesh, target_size: float = 1.0) -> trimesh.Trimesh:
        """Normalize mesh to unit size and center"""
        mesh.apply_translation(-mesh.center_mass)
        mesh.apply_scale(target_size / mesh.extents.max())
        return mesh
    
    @staticmethod
    def mesh_to_pointcloud(mesh: trimesh.Trimesh, sample_count: int = 10000) -> np.ndarray:
        """Convert mesh to point cloud"""
        points, _ = trimesh.sample.sample_surface(mesh, sample_count)
        return points
    
    @staticmethod
    def pointcloud_to_tensor(points: np.ndarray) -> torch.Tensor:
        """Convert point cloud to PyTorch tensor"""
        return torch.from_numpy(points).float()
    
    @staticmethod
    def save_stl(mesh: trimesh.Trimesh, filepath: str):
        """Save mesh to STL file"""
        mesh.export(filepath, file_type='stl')
    
    @staticmethod
    def simplify_mesh(mesh: trimesh.Trimesh, target_reduction: float = 0.5) -> trimesh.Trimesh:
        """Simplify mesh using quadric mesh simplification"""
        try:
            simplified = mesh.simplify_quadric_mesh_simplification(
                target_reduction=target_reduction
            )
            return simplified
        except:
            return mesh
