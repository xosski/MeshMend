"""
Convert 3D volumes to meshes using marching cubes
Supports both occupancy grids and signed distance fields
"""

from __future__ import annotations
import numpy as np
import trimesh

def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Sigmoid activation function with clipping"""
    x = np.clip(x, -60, 60)
    return 1 / (1 + np.exp(-x))

def volume_to_mesh_trimesh(
    volume: np.ndarray,
    *,
    kind: str = "occupancy",   # "occupancy" or "sdf"
    iso: float = 0.5,          # occupancy default
    voxel_size_mm: float = 0.2 # your resolution in mm
) -> trimesh.Trimesh:
    """
    Convert 3D volume to mesh using marching cubes
    
    Args:
        volume: (D,H,W) float array
            - occupancy: probabilities [0..1] OR logits (any range)
            - sdf: signed distance field (iso typically 0.0)
        kind: "occupancy" or "sdf"
        iso: iso level for marching cubes
        voxel_size_mm: size of each voxel in mm
    
    Returns:
        trimesh.Trimesh object
    """
    if volume.ndim != 3:
        raise ValueError(f"Expected (D,H,W), got {volume.shape}")

    vol = volume.astype(np.float32)

    if kind == "occupancy":
        # if it looks like logits, squash with sigmoid
        if vol.min() < -0.5 or vol.max() > 1.5:
            vol = _sigmoid(vol)
        vol = np.clip(vol, 0.0, 1.0)
        iso_level = iso
    elif kind == "sdf":
        iso_level = iso  # usually 0.0
    else:
        raise ValueError("kind must be 'occupancy' or 'sdf'")

    try:
        from skimage.measure import marching_cubes
    except ImportError:
        raise ImportError(
            "scikit-image not installed. Run: pip install scikit-image"
        )
    
    # Ensure iso level is within data range
    vol_min = float(vol.min())
    vol_max = float(vol.max())
    
    if iso_level < vol_min or iso_level > vol_max:
        # Clamp iso level to valid range, slightly inside to ensure we get geometry
        iso_level = vol_min + (vol_max - vol_min) * 0.4
        print(f"Adjusted iso level from original to {iso_level:.3f} (data range: {vol_min:.3f}-{vol_max:.3f})")
    
    verts, faces, normals, _ = marching_cubes(vol, level=float(iso_level))

    # scale voxels into mm
    verts = verts * float(voxel_size_mm)

    # Create mesh
    mesh = trimesh.Trimesh(
        vertices=verts,
        faces=faces,
        vertex_normals=normals,
        process=False
    )

    # Basic cleanup
    try:
        if hasattr(mesh, 'remove_degenerate_faces'):
            mesh.remove_degenerate_faces()
        if hasattr(mesh, 'remove_duplicate_faces'):
            mesh.remove_duplicate_faces()
        if hasattr(mesh, 'remove_infinite_values'):
            mesh.remove_infinite_values()
        if hasattr(mesh, 'merge_vertices'):
            mesh.merge_vertices()
    except Exception as e:
        print(f"Warning during mesh cleanup: {e}")

    return mesh
