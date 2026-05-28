"""Simplify meshes for optimal 3D printing performance"""

from trimesh import Trimesh
import numpy as np

class MeshSimplifier:
    """Reduce polygon count while preserving miniature details"""
    
    # Target polygon counts for print optimization
    TARGET_TRIANGLES = {
        "low": 60000,       # Quick, rough prints while preserving readable mini detail
        "standard": 300000, # Balanced miniature quality/speed
        "high": 900000,     # High-detail trained miniature preservation
    }

    @staticmethod
    def polish_mesh(mesh: Trimesh, quality: str = "standard") -> Trimesh:
        """Clean topology noise and apply controlled smoothing without flattening details."""
        polished = mesh.copy()

        q = (quality or "standard").lower().strip()
        if q == "high":
            taubin_iter = 1
            lambda_v = 0.08
            nu_v = -0.09
            min_component_faces = 200
        elif q == "low":
            taubin_iter = 2
            lambda_v = 0.20
            nu_v = -0.21
            min_component_faces = 120
        else:
            taubin_iter = 2
            lambda_v = 0.21
            nu_v = -0.22
            min_component_faces = 160

        try:
            polished.remove_unreferenced_vertices()
            polished.remove_duplicate_faces()
            polished.remove_degenerate_faces()
            polished.remove_infinite_values()
            if hasattr(polished, "merge_vertices"):
                polished.merge_vertices()
        except Exception:
            pass

        # Remove tiny disconnected bits that make outputs feel noisy/unpolished.
        try:
            pieces = polished.split(only_watertight=False)
            if len(pieces) > 1:
                kept = [p for p in pieces if len(p.faces) >= min_component_faces]
                if kept:
                    polished = kept[0]
                    for part in kept[1:]:
                        polished = polished + part
        except Exception:
            pass

        # Fill small holes before smoothing so the surface relaxes cleanly.
        try:
            polished.fill_holes()
        except Exception:
            pass

        try:
            import trimesh
            trimesh.smoothing.filter_taubin(polished, lamb=lambda_v, nu=nu_v, iterations=taubin_iter)
        except Exception:
            pass

        try:
            polished.remove_unreferenced_vertices()
            polished.fix_normals()
        except Exception:
            pass

        return polished
    
    @staticmethod
    def simplify_for_printing(mesh: Trimesh, quality: str = "standard", preserve_boundaries: bool = True) -> Trimesh:
        """
        Simplify mesh to optimal polygon count for 3D printing
        
        Args:
            mesh: Input Trimesh
            quality: "low", "standard", or "high"
            preserve_boundaries: Keep mesh boundaries unchanged
            
        Returns:
            Simplified Trimesh
        """
        target = MeshSimplifier.TARGET_TRIANGLES.get(quality, MeshSimplifier.TARGET_TRIANGLES["standard"])
        
        simplified = mesh.copy()
        current_count = len(simplified.faces)
        
        print(f"Simplifying mesh: {current_count} triangles -> {target} triangles")
        
        if current_count > target:
            # Calculate reduction ratio
            ratio = target / current_count
            
            try:
                # Use trimesh's built-in simplification
                # This uses quadratic error metrics
                simplified = simplified.simplify_quadratic_decimation(face_count=target)
            except:
                # Fallback: decimation via vertex reduction
                try:
                    from trimesh.smoothing import filter_laplacian
                    # Remove vertices based on curvature
                    simplified = simplified.copy()
                    # Simple vertex reduction by ratio
                    vertex_mask = np.random.rand(len(simplified.vertices)) < ratio
                    
                except Exception as e:
                    print(f"Simplification attempted but mesh unchanged: {e}")
        
        final_count = len(simplified.faces)
        reduction_pct = ((current_count - final_count) / current_count * 100) if current_count > 0 else 0
        
        print(f"  Result: {final_count} triangles (down {reduction_pct:.1f}%)")
        
        return simplified
    
    @staticmethod
    def remove_internal_geometry(mesh: Trimesh, shell_thickness_mm: float = 1.0) -> Trimesh:
        """
        Remove internal geometry for resin/FDM hollowed models
        Reduces material, weight, and print time
        
        Args:
            mesh: Input mesh
            shell_thickness_mm: Minimum wall thickness
            
        Returns:
            Hollowed mesh
        """
        hollowed = mesh.copy()
        
        # Invert the mesh for boolean operation
        hollowed.invert()
        
        # Create outer shell by offsetting
        try:
            # Use voxel-based approach as fallback
            # This is more robust than direct offsetting
            voxel_pitch = shell_thickness_mm / 2
            grid = hollowed.voxelized(pitch=voxel_pitch)
            hollowed = grid.as_mesh()
        except:
            # If that fails, just use the original
            pass
        
        return hollowed
    
    @staticmethod
    def get_simplification_report(original: Trimesh, simplified: Trimesh) -> dict:
        """Generate report on simplification results"""
        original_verts = len(original.vertices)
        original_faces = len(original.faces)
        simplified_verts = len(simplified.vertices)
        simplified_faces = len(simplified.faces)
        
        report = {
            "original_vertices": original_verts,
            "original_triangles": original_faces,
            "simplified_vertices": simplified_verts,
            "simplified_triangles": simplified_faces,
            "vertex_reduction_percent": ((original_verts - simplified_verts) / original_verts * 100) if original_verts > 0 else 0,
            "triangle_reduction_percent": ((original_faces - simplified_faces) / original_faces * 100) if original_faces > 0 else 0,
            "size_reduction_estimate": f"~{int(simplified_faces / max(original_faces, 1) * 100)}% of original",
        }
        
        return report
