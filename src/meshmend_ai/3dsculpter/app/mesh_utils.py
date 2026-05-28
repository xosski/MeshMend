"""
Mesh utilities for loading, saving, and manipulating 3D models
"""

import trimesh
from pathlib import Path
import numpy as np

class MeshManager:
    """Manages mesh operations"""
    
    SUPPORTED_FORMATS = {
        '.stl': 'stl',
        '.obj': 'obj',
        '.ply': 'ply',
        '.gltf': 'gltf',
        '.glb': 'glb'
    }
    
    @staticmethod
    def load_mesh(file_path: str) -> trimesh.Trimesh:
        """Load a mesh from file"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if file_path.suffix.lower() not in MeshManager.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported format: {file_path.suffix}")
        
        mesh = trimesh.load(str(file_path), process=True)
        
        # Ensure it's a single mesh
        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate(mesh.geometry.values())
        
        return mesh
    
    @staticmethod
    def save_mesh(mesh: trimesh.Trimesh, file_path: str) -> None:
        """Save mesh to file"""
        file_path = Path(file_path)
        
        if file_path.suffix.lower() not in MeshManager.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported format: {file_path.suffix}")
        
        file_path.parent.mkdir(parents=True, exist_ok=True)
        mesh.export(str(file_path))
    
    @staticmethod
    def scale_mesh(mesh: trimesh.Trimesh, scale_factor: float) -> trimesh.Trimesh:
        """Scale mesh by factor"""
        mesh.apply_scale(scale_factor)
        return mesh
    
    @staticmethod
    def center_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        """Center mesh at origin"""
        mesh.vertices -= mesh.center_mass
        return mesh
    
    @staticmethod
    def smooth_mesh(mesh: trimesh.Trimesh, iterations: int = 1) -> trimesh.Trimesh:
        """Apply Laplacian smoothing"""
        mesh = mesh.smooth_laplacian(iterations=iterations)
        return mesh
    
    @staticmethod
    def subdivide_mesh(mesh: trimesh.Trimesh, iterations: int = 1) -> trimesh.Trimesh:
        """Subdivide mesh for more detail"""
        for _ in range(iterations):
            mesh = mesh.subdivide()
        return mesh
    
    @staticmethod
    def decimate_mesh(mesh: trimesh.Trimesh, target_count: int) -> trimesh.Trimesh:
        """Reduce polygon count"""
        mesh.simplify_quadratic_mesh(target_count=target_count)
        return mesh
    
    @staticmethod
    def merge_meshes(meshes: list) -> trimesh.Trimesh:
        """Merge multiple meshes into one"""
        return trimesh.util.concatenate(meshes)
    
    @staticmethod
    def get_mesh_info(mesh: trimesh.Trimesh) -> dict:
        """Get mesh statistics"""
        return {
            'vertices': len(mesh.vertices),
            'faces': len(mesh.faces),
            'bounds': mesh.bounds.tolist(),
            'volume': mesh.volume,
            'surface_area': mesh.area,
            'is_watertight': mesh.is_watertight,
            'center': mesh.center_mass.tolist(),
            'extents': mesh.extents.tolist()
        }
