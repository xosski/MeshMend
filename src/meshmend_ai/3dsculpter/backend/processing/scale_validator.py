"""Validate and normalize 3D models to tabletop miniature scale"""

import numpy as np
from trimesh import Trimesh

class ScaleValidator:
    """Ensures generated models are common tabletop/wargaming miniature scale"""
    
    # Standard tabletop scales
    STANDARD_SCALES = {
        "15mm": 15.0,      # Mass-battle / historical small scale
        "20mm": 20.0,      # 1/72-ish wargaming scale
        "25mm": 25.0,      # Historical miniatures
        "28mm": 28.5,      # Classic heroic tabletop scale
        "32mm": 32.0,      # Modern heroic tabletop scale
        "35mm": 35.0,      # Large heroic character scale
        "40mm": 40.0,      # Display/skirmish scale
        "48mm": 48.0,      # Large display/hero miniature scale
        "54mm": 54.0,      # Traditional display miniature scale
        "75mm": 75.0,      # Large display miniature scale
    }
    
    # Tolerance in millimeters
    TOLERANCE = 2.0  # ±2mm acceptable
    
    # Expected heights for reference
    REFERENCE_HEIGHTS = {
        "human": 28.5,      # Average human miniature
        "dwarf": 22.0,      # Shorter stature
        "elf": 30.0,        # Taller, slender
        "humanoid": 28.5,   # Generic humanoid
    }
    
    @staticmethod
    def validate_and_normalize(mesh: Trimesh, target_scale: str = "28mm", model_type: str = "human") -> tuple[Trimesh, dict]:
        """
        Validate mesh is proper miniature scale and normalize if needed
        
        Args:
            mesh: Trimesh object to validate
            target_scale: e.g. "28mm", "32mm", "48mm", or another STANDARD_SCALES key
            model_type: "human", "dwarf", "elf", "creature", "object"
            
        Returns:
            (normalized_mesh, validation_report)
        """
        target_height_mm = ScaleValidator.STANDARD_SCALES.get(target_scale, 28.5)
        reference_height = ScaleValidator.REFERENCE_HEIGHTS.get(model_type, 28.5)
        
        # Get current mesh dimensions
        bounds = mesh.bounds
        current_height = bounds[1][2] - bounds[0][2]  # Z-axis height
        
        # If height is very small, probably in wrong units
        if current_height < 0.1:
            # Assume it's normalized to [0, 1], scale up
            current_height *= 100
        
        # Calculate scale factor needed
        scale_factor = target_height_mm / current_height if current_height > 0 else 1.0
        
        # Apply scaling to achieve target height
        normalized_mesh = mesh.copy()
        normalized_mesh.apply_scale(scale_factor)
        
        # Re-check dimensions after scaling
        new_bounds = normalized_mesh.bounds
        new_height = new_bounds[1][2] - new_bounds[0][2]
        
        # Generate validation report
        report = {
            "original_height_mm": float(current_height),
            "target_height_mm": float(target_height_mm),
            "final_height_mm": float(new_height),
            "scale_factor": float(scale_factor),
            "tolerance_mm": float(ScaleValidator.TOLERANCE),
            "is_valid": False,
            "status": "UNKNOWN",
            "warnings": [],
        }
        
        # Check if within tolerance
        height_diff = abs(new_height - target_height_mm)
        if height_diff <= ScaleValidator.TOLERANCE:
            report["is_valid"] = True
            report["status"] = "VALID"
        elif height_diff <= ScaleValidator.TOLERANCE * 2:
            report["is_valid"] = True
            report["status"] = "ACCEPTABLE"
            report["warnings"].append(f"Height {new_height:.1f}mm is slightly outside tolerance (±{ScaleValidator.TOLERANCE}mm)")
        else:
            report["is_valid"] = False
            report["status"] = "OUT_OF_SCALE"
            report["warnings"].append(f"Height {new_height:.1f}mm is significantly different from target {target_height_mm}mm")
        
        # Additional checks
        min_dimension = min(new_bounds[1][0] - new_bounds[0][0], 
                           new_bounds[1][1] - new_bounds[0][1])
        
        if min_dimension < 3:
            report["warnings"].append(f"Model base very small ({min_dimension:.1f}mm) - ensure proper orientation")
        
        if min_dimension > 50:
            report["warnings"].append(f"Model base very large ({min_dimension:.1f}mm) - check scale")
        
        # Check for inverted geometry
        if normalized_mesh.volume < 0:
            report["warnings"].append("Mesh has inverted normals - may affect printing")
        
        return normalized_mesh, report
    
    @staticmethod
    def get_scale_report(mesh: Trimesh, target_scale: str = "28mm") -> dict:
        """Get a human-readable scale validation report"""
        normalized, report = ScaleValidator.validate_and_normalize(mesh, target_scale)
        
        status_colors = {
            "VALID": "[OK] GREEN",
            "ACCEPTABLE": "[WARN] YELLOW", 
            "OUT_OF_SCALE": "[ERROR] RED",
        }
        
        readable = {
            "status": f"{status_colors.get(report['status'], 'UNKNOWN')} - {report['status']}",
            "height": f"{report['final_height_mm']:.1f}mm (target: {report['target_height_mm']:.1f}mm)",
            "offset": f"{abs(report['final_height_mm'] - report['target_height_mm']):.1f}mm from target",
            "valid_for_printing": report["is_valid"],
            "issues": report["warnings"] if report["warnings"] else ["None"],
        }
        
        return readable
    
    @staticmethod
    def auto_orient_for_printing(mesh: Trimesh) -> Trimesh:
        """Orient mesh optimally for 3D printing (base-up)"""
        oriented = mesh.copy()
        
        # Move to origin
        oriented.vertices -= oriented.bounds[0]
        
        # Ensure base is on XY plane
        # This is typically: Z should be height axis
        
        return oriented
