"""
Analyze mesh quality and printability
"""

import numpy as np
import trimesh
from dataclasses import dataclass

@dataclass
class MeshQuality:
    """Mesh quality metrics"""
    vertex_count: int
    face_count: int
    is_watertight: bool
    has_holes: bool
    bounds: np.ndarray
    volume: float
    surface_area: float
    min_edge_length: float
    max_edge_length: float
    avg_edge_length: float
    
    def is_valid_for_printing(self) -> bool:
        """Check if mesh is suitable for 3D printing"""
        # At least some geometry
        if self.vertex_count < 100:
            return False
        
        # Reasonable size (not microscopic)
        sizes = self.bounds[1] - self.bounds[0]
        if np.max(sizes) < 0.5:  # Less than 0.5mm
            return False
        
        # Reasonable edge lengths
        if self.min_edge_length < 0.01:  # Too thin
            return False
        
        return True
    
    def report(self) -> str:
        """Generate quality report"""
        lines = [
            "=== Mesh Quality Report ===",
            f"Vertices: {self.vertex_count}",
            f"Faces: {self.face_count}",
            f"Watertight: {self.is_watertight}",
            f"Has holes: {self.has_holes}",
            f"Volume: {self.volume:.2f} mm³",
            f"Surface Area: {self.surface_area:.2f} mm²",
            f"Edge length range: {self.min_edge_length:.3f} - {self.max_edge_length:.3f} mm",
            f"Avg edge length: {self.avg_edge_length:.3f} mm",
            f"Bounds: X={self.bounds[1,0]-self.bounds[0,0]:.1f}mm, "
            f"Y={self.bounds[1,1]-self.bounds[0,1]:.1f}mm, "
            f"Z={self.bounds[1,2]-self.bounds[0,2]:.1f}mm",
        ]
        if self.is_valid_for_printing():
            lines.append("✓ Valid for printing")
        else:
            lines.append("✗ Not suitable for printing")
        
        return "\n".join(lines)


class MeshAnalyzer:
    """Analyze mesh properties and quality"""
    
    @staticmethod
    def analyze(mesh: trimesh.Trimesh) -> MeshQuality:
        """Analyze mesh quality"""
        try:
            # Edge lengths
            edges = mesh.edges_unique_length
            min_edge = np.min(edges) if len(edges) > 0 else 0
            max_edge = np.max(edges) if len(edges) > 0 else 0
            avg_edge = np.mean(edges) if len(edges) > 0 else 0
            
            quality = MeshQuality(
                vertex_count=len(mesh.vertices),
                face_count=len(mesh.faces),
                is_watertight=mesh.is_watertight,
                has_holes=not mesh.is_watertight,
                bounds=mesh.bounds,
                volume=mesh.volume,
                surface_area=mesh.area,
                min_edge_length=min_edge,
                max_edge_length=max_edge,
                avg_edge_length=avg_edge
            )
            
            return quality
            
        except Exception as e:
            print(f"Analysis error: {e}")
            return None
    
    @staticmethod
    def fix_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        """Attempt to fix common mesh issues"""
        try:
            # Fill small holes
            mesh.fill_holes()
            
            # Remove isolated components (keep largest)
            meshes = mesh.split()
            if len(meshes) > 1:
                largest = max(meshes, key=lambda m: m.area)
                mesh = largest
            
            # Remove duplicate/degenerate faces
            mesh.remove_duplicate_faces()
            mesh.remove_degenerate_faces()
            
            # Merge nearby vertices
            mesh.merge_vertices()
            
            return mesh
            
        except Exception as e:
            print(f"Mesh repair warning: {e}")
            return mesh
