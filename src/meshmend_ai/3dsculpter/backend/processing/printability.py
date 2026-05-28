"""Analyze and optimize meshes for 3D printability"""

import numpy as np
from trimesh import Trimesh

class PrintabilityAnalyzer:
    """Check for print-readiness issues"""
    
    # Minimum feature sizes (mm)
    MIN_FEATURE_SIZES = {
        "fdm_pla": 0.4,      # FDM nozzle size
        "fdm_petg": 0.5,     # PETG slightly thicker
        "resin_standard": 0.2,
        "resin_high_detail": 0.1,
    }
    
    # Minimum wall thickness (mm)
    MIN_WALL_THICKNESS = {
        "fdm": 1.5,
        "resin": 0.8,
        "metal": 2.0,
    }
    
    # Maximum overhang angle (degrees from vertical)
    MAX_OVERHANG_ANGLE = 45  # degrees
    
    @staticmethod
    def analyze(mesh: Trimesh, material: str = "fdm") -> dict:
        """
        Analyze mesh for printability issues
        
        Args:
            mesh: Trimesh to analyze
            material: "fdm", "resin", or "metal"
            
        Returns:
            Printability report with score and issues
        """
        report = {
            "material": material,
            "score": 100,  # Out of 100
            "issues": [],
            "warnings": [],
            "print_difficulty": "EASY",
            "support_needed": False,
            "estimated_print_time_minutes": 0,
            "estimated_weight_grams": 0,
        }
        
        # Check 1: Mesh validity
        try:
            if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
                report["issues"].append("Mesh has no geometry - may fail to print")
                report["score"] -= 20
        except:
            pass
        
        # Check 2: Closed mesh
        if not mesh.is_watertight:
            report["issues"].append("Mesh is not watertight - needs sealing before printing")
            report["score"] -= 30
        
        # Check 3: Thin walls
        wall_thickness = PrintabilityAnalyzer._estimate_wall_thickness(mesh)
        min_wall = PrintabilityAnalyzer.MIN_WALL_THICKNESS.get(material, 1.5)
        
        if wall_thickness > 0 and wall_thickness < min_wall:
            report["warnings"].append(
                f"Thin walls detected ({wall_thickness:.2f}mm < {min_wall}mm minimum) - increase model scale or add internal structure"
            )
            report["score"] -= 10
        
        # Check 4: Overhangs
        overhangs = PrintabilityAnalyzer._detect_overhangs(mesh)
        if overhangs["has_overhangs"]:
            report["support_needed"] = True
            report["warnings"].append(
                f"Overhanging features detected - supports will be auto-generated"
            )
            if overhangs["severe_overhangs"]:
                report["issues"].append("Severe overhangs may require manual adjustment")
                report["score"] -= 15
            else:
                report["score"] -= 5
        
        # Check 5: Size reasonableness
        bounds = mesh.bounds
        max_dim = max(
            bounds[1][0] - bounds[0][0],
            bounds[1][1] - bounds[0][1],
            bounds[1][2] - bounds[0][2]
        )
        
        if max_dim < 5:
            report["warnings"].append(f"Model very small ({max_dim:.1f}mm) - may be hard to handle post-print")
            report["score"] -= 5
        elif max_dim > 80:
            report["warnings"].append(f"Model very large ({max_dim:.1f}mm) - consider splitting into parts")
            report["score"] -= 10
        
        # Check 6: Volume/printability
        if mesh.volume > 0:
            # Estimate material weight (varies by material)
            density_g_cm3 = {
                "fdm": 1.24,  # PLA
                "resin": 1.20,
                "metal": 8.0,
            }
            density = density_g_cm3.get(material, 1.2)
            volume_cm3 = mesh.volume / 1000  # Convert mm³ to cm³
            report["estimated_weight_grams"] = volume_cm3 * density
        
        # Estimate print time (rough)
        try:
            if material == "fdm":
                # Rough: ~10mm² per minute at 0.2mm layer
                estimated_area = bounds[1][0] * bounds[1][1] / 100  # Rough layer area
                if np.isfinite(estimated_area):
                    report["estimated_print_time_minutes"] = int(max(1, estimated_area * 10))
                else:
                    report["estimated_print_time_minutes"] = 30
            elif material == "resin":
                # Much faster: ~30 seconds per layer
                height = bounds[1][2] - bounds[0][2]
                if np.isfinite(height):
                    layers = int(max(10, height / 0.05))  # 50 micron layers
                    report["estimated_print_time_minutes"] = int(max(1, layers * 0.5 / 60))
                else:
                    report["estimated_print_time_minutes"] = 10
        except:
            report["estimated_print_time_minutes"] = 30
        
        # Set difficulty
        if report["score"] >= 90:
            report["print_difficulty"] = "VERY EASY"
        elif report["score"] >= 75:
            report["print_difficulty"] = "EASY"
        elif report["score"] >= 60:
            report["print_difficulty"] = "MODERATE"
        elif report["score"] >= 40:
            report["print_difficulty"] = "DIFFICULT"
        else:
            report["print_difficulty"] = "VERY DIFFICULT"
        
        return report
    
    @staticmethod
    def _estimate_wall_thickness(mesh: Trimesh) -> float:
        """Estimate minimum wall thickness"""
        # Simplified: use mesh thickness estimation
        try:
            thickness = mesh.section_thickness()
            return float(thickness) if thickness else 0
        except:
            return 0
    
    @staticmethod
    def _detect_overhangs(mesh: Trimesh, max_angle: float = 45) -> dict:
        """Detect overhanging features"""
        result = {
            "has_overhangs": False,
            "severe_overhangs": False,
            "overhang_count": 0,
            "degrees_from_vertical": max_angle,
        }
        
        try:
            # Check face normals against Z-axis
            normals = mesh.face_normals
            vertical = np.array([0, 0, 1])
            
            # Calculate angles
            angles = np.arccos(np.clip(np.dot(normals, vertical), -1, 1))
            angles_deg = np.degrees(angles)
            
            # Check for overhangs (>45 degrees from vertical = <45 degrees from horizontal)
            overhang_faces = np.where(angles_deg > (90 - max_angle))[0]
            
            if len(overhang_faces) > 0:
                result["has_overhangs"] = True
                result["overhang_count"] = len(overhang_faces)
                
                # Severe if very steep
                severe = np.where(angles_deg > 85)[0]  # Nearly horizontal
                if len(severe) > len(overhang_faces) * 0.1:
                    result["severe_overhangs"] = True
        except:
            pass
        
        return result
    
    @staticmethod
    def get_printability_score(mesh: Trimesh, material: str = "fdm") -> tuple[int, str]:
        """Get quick printability score (0-100) and status string"""
        report = PrintabilityAnalyzer.analyze(mesh, material)
        score = max(0, min(100, report["score"]))
        
        if score >= 90:
            status = "[EXCELLENT] Print with confidence"
        elif score >= 75:
            status = "[GOOD] Should print fine"
        elif score >= 60:
            status = "[OK] Acceptable - may need supports"
        elif score >= 40:
            status = "[CAUTION] Risky - review carefully before printing"
        else:
            status = "[DIFFICULT] Significant adjustments needed"
        
        return score, status
