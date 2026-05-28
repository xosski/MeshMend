import numpy as np
import trimesh
from PIL import Image

try:
    from models.point_e_generator import PointEGenerator
    from models.mesh_reconstructor import MeshReconstructor
except ModuleNotFoundError:
    from backend.models.point_e_generator import PointEGenerator
    from backend.models.mesh_reconstructor import MeshReconstructor

try:
    from app.mesh_from_volume import volume_to_mesh_trimesh
except ModuleNotFoundError:
    volume_to_mesh_trimesh = None

def reconstruct_mesh_from_image(
    image: Image.Image,
    num_points: int = 6000,
    method: str = "poisson",
    smooth_iterations: int = 0,
) -> trimesh.Trimesh:
    """
    High-level pipeline: image -> depth -> point cloud (+normals) -> mesh
    method: "poisson" (default), "ball_pivot", or "convex"
    """
    gen = PointEGenerator()
    points, normals = gen.generate_with_normals(image, num_points=num_points)

    # Combine into (N, 6) for downstream convenience
    pts_with_normals = np.hstack([points, normals])

    mesh = MeshReconstructor.pointcloud_to_mesh(pts_with_normals, method=method)

    # Post-process cleanup
    try:
        mesh.remove_unreferenced_vertices()
        if hasattr(mesh, "merge_vertices"):
            mesh.merge_vertices()

        # Keep dominant body to avoid "polygon clump" satellites.
        parts = mesh.split(only_watertight=False)
        if len(parts) > 1:
            scored = []
            for part in parts:
                if len(part.vertices) < 80 or len(part.faces) < 120:
                    continue
                score = float(abs(part.volume)) if abs(float(part.volume)) > 0 else float(part.area)
                scored.append((score, part))
            if scored:
                mesh = max(scored, key=lambda t: t[0])[1]

        if hasattr(mesh, "remove_degenerate_faces"):
            mesh.remove_degenerate_faces()
        if hasattr(mesh, "remove_duplicate_faces"):
            mesh.remove_duplicate_faces()
    except Exception:
        pass

    # Optional smoothing (disabled by default to preserve detail)
    if smooth_iterations > 0:
        try:
            trimesh.smoothing.filter_laplacian(mesh, iterations=smooth_iterations, implicit_time_steps=1)
        except Exception:
            pass

    # Center the mesh
    try:
        mesh.vertices -= mesh.center_mass
    except Exception:
        pass

    return mesh


def reconstruct_mesh_from_image_volumetric(
    image: Image.Image,
    vol_size: int = 112,
    voxel_size_mm: float = 0.24,
) -> trimesh.Trimesh:
    """Reconstruct a fuller character body from image silhouette/depth cues."""
    if volume_to_mesh_trimesh is None:
        raise RuntimeError("volumetric reconstruction unavailable: app.mesh_from_volume not importable")

    from scipy.ndimage import (
        binary_closing,
        binary_fill_holes,
        binary_opening,
        binary_erosion,
        distance_transform_edt,
        gaussian_filter,
        label,
        zoom,
    )

    img = image.convert("RGB")
    arr = np.array(img).astype(np.float32) / 255.0
    gray = np.mean(arr, axis=2)
    sat = np.max(arr, axis=2) - np.min(arr, axis=2)

    # Foreground extraction with center-prior + border background rejection.
    fg_grabcut = None
    try:
        import cv2

        arr_u8 = (arr * 255.0).astype(np.uint8)
        h_gc, w_gc = arr_u8.shape[:2]
        gc_mask = np.zeros((h_gc, w_gc), np.uint8)
        bg_model = np.zeros((1, 65), np.float64)
        fg_model = np.zeros((1, 65), np.float64)
        rect = (max(1, int(w_gc * 0.07)), max(1, int(h_gc * 0.07)), int(w_gc * 0.86), int(h_gc * 0.86))
        cv2.grabCut(arr_u8, gc_mask, rect, bg_model, fg_model, 5, cv2.GC_INIT_WITH_RECT)
        fg_grabcut = (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD)
        ratio = float(np.mean(fg_grabcut))
        if ratio < 0.01 or ratio > 0.78:
            fg_grabcut = None
    except Exception:
        fg_grabcut = None

    border = np.concatenate([arr[0, :, :], arr[-1, :, :], arr[:, 0, :], arr[:, -1, :]], axis=0)
    bg_color = np.median(border, axis=0)
    color_dist = np.linalg.norm(arr - bg_color[None, None, :], axis=2)

    edge_x = np.gradient(gray, axis=1)
    edge_y = np.gradient(gray, axis=0)
    edge_mag = np.sqrt(edge_x**2 + edge_y**2)

    fg = (
        (color_dist > np.quantile(color_dist, 0.72))
        | (sat > np.quantile(sat, 0.60))
        | (gray < np.quantile(gray, 0.74))
        | (edge_mag > np.quantile(edge_mag, 0.84))
    )
    if fg_grabcut is not None:
        fg = fg & fg_grabcut

    # Keep only non-border-connected foreground.
    h, w = fg.shape
    bg_connected = np.zeros_like(fg, dtype=bool)
    queue = [(0, x) for x in range(w)] + [(h - 1, x) for x in range(w)] + [(y, 0) for y in range(h)] + [(y, w - 1) for y in range(h)]
    while queue:
        y_idx, x_idx = queue.pop()
        if y_idx < 0 or y_idx >= h or x_idx < 0 or x_idx >= w:
            continue
        if bg_connected[y_idx, x_idx] or fg[y_idx, x_idx]:
            continue
        bg_connected[y_idx, x_idx] = True
        queue.extend([(y_idx - 1, x_idx), (y_idx + 1, x_idx), (y_idx, x_idx - 1), (y_idx, x_idx + 1)])
    fg = ~bg_connected

    labeled, num = label(fg)
    if num > 0:
        yy, xx = np.indices((h, w))
        cx, cy = w * 0.5, h * 0.5
        best_score = -1.0
        best_mask = fg
        for i in range(1, num + 1):
            comp = labeled == i
            area = float(np.sum(comp))
            if area < h * w * 0.002:
                continue
            mx = float(np.mean(xx[comp]))
            my = float(np.mean(yy[comp]))
            center_penalty = np.sqrt(((mx - cx) / w) ** 2 + ((my - cy) / h) ** 2)
            score = area * (1.0 - min(0.95, center_penalty))
            if score > best_score:
                best_score = score
                best_mask = comp
        fg = best_mask

    # Prevent canvas-wide masks from becoming the same squat rock/cylinder for
    # unrelated source images. If coverage is excessive, rebuild from strong
    # image-specific color/saturation/edge cues near the subject center.
    fg_ratio = float(np.mean(fg))
    if fg_ratio > 0.52:
        fg = (
            (color_dist > np.quantile(color_dist, 0.82))
            | (sat > np.quantile(sat, 0.72))
            | (edge_mag > np.quantile(edge_mag, 0.88))
        )

    fg_ratio = float(np.mean(fg))
    if fg_ratio < 0.012 or fg_ratio > 0.62:
        yy, xx = np.indices((h, w))
        center_prior = (((xx - w * 0.5) / max(w * 0.45, 1)) ** 2 + ((yy - h * 0.52) / max(h * 0.45, 1)) ** 2) < 1.0
        fg = center_prior & (
            (color_dist > np.quantile(color_dist, 0.78))
            | (sat > np.quantile(sat, 0.68))
            | (edge_mag > np.quantile(edge_mag, 0.82))
        )

    fg = binary_opening(fg, iterations=1)
    fg = binary_closing(fg, iterations=2)
    fg = binary_fill_holes(fg)
    fg = binary_erosion(fg, iterations=1)

    depth = 1.0 - gray
    depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)

    fg_resized = zoom(fg.astype(np.float32), (vol_size / h, vol_size / w), order=0) > 0.5
    depth_resized = zoom(depth.astype(np.float32), (vol_size / h, vol_size / w), order=1)
    depth_resized = np.clip(depth_resized, 0.0, 1.0)

    fg_resized = binary_fill_holes(binary_closing(fg_resized, iterations=2))
    fg_resized = binary_erosion(fg_resized, iterations=1)
    dist_inside = distance_transform_edt(fg_resized.astype(np.uint8)).astype(np.float32)
    dist_inside = dist_inside / (dist_inside.max() + 1e-8)

    volume = np.zeros((vol_size, vol_size, vol_size), dtype=np.float32)

    # Build cross-sectional solids to avoid thin shell outputs.
    min_half = int(vol_size * 0.07)
    max_half = int(vol_size * 0.29)
    center_z = int(vol_size * 0.50)

    for y_idx in range(vol_size):
        row_mask = fg_resized[y_idx]
        row_x = np.where(row_mask)[0]
        if len(row_x) < 3:
            continue

        x_left = int(row_x.min())
        x_right = int(row_x.max())
        x_center = 0.5 * (x_left + x_right)
        rx_row = max(1.5, 0.5 * (x_right - x_left))
        row_depth = float(np.mean(depth_resized[y_idx, row_mask]))

        for x_idx in row_x:
            dx_norm = (x_idx - x_center) / (rx_row + 1e-8)
            ellipse_scale = np.sqrt(max(0.0, 1.0 - dx_norm * dx_norm))
            if ellipse_scale <= 0.0:
                continue

            center_weight = float(dist_inside[y_idx, x_idx])
            local_depth = float(depth_resized[y_idx, x_idx])
            half_thickness = min_half + int(
                (max_half - min_half)
                * (0.30 + 0.70 * center_weight)
                * (0.40 + 0.60 * local_depth)
                * ellipse_scale
            )
            half_thickness = max(1, half_thickness)

            z_shift = int((local_depth - 0.5) * vol_size * 0.08)
            row_shift = int((row_depth - 0.5) * vol_size * 0.04)
            zc = int(np.clip(center_z + z_shift + row_shift, 0, vol_size - 1))

            z0 = max(0, zc - half_thickness)
            z1 = min(vol_size, zc + half_thickness)
            volume[z0:z1, y_idx, x_idx] = 1.0

    # Add edge-driven detail embossing so armor/folds survive better.
    edge_resized = zoom(edge_mag.astype(np.float32), (vol_size / h, vol_size / w), order=1)
    edge_resized = np.clip(edge_resized / (edge_resized.max() + 1e-8), 0.0, 1.0)
    edge_mask = (edge_resized > np.quantile(edge_resized, 0.80)) & fg_resized
    detail_depth = max(1, int(vol_size * 0.015))
    for y_idx, x_idx in np.argwhere(edge_mask):
        local_depth = float(depth_resized[y_idx, x_idx])
        zc = int(np.clip(center_z + (local_depth - 0.5) * vol_size * 0.08, 0, vol_size - 1))
        z0 = max(0, zc - detail_depth)
        z1 = min(vol_size, zc + detail_depth)
        volume[z0:z1, y_idx, x_idx] = 1.0

    for z_idx in range(vol_size):
        z_norm = abs((z_idx - center_z) / max(1, center_z))
        taper = max(0.0, 1.0 - z_norm**1.30)
        volume[z_idx] *= max(0.0, taper)

    volume = gaussian_filter(volume, sigma=(0.60, 0.32, 0.60))
    volume = (volume > 0.31).astype(np.float32)

    mesh = volume_to_mesh_trimesh(volume, kind="occupancy", iso=0.56, voxel_size_mm=voxel_size_mm)
    try:
        mesh.remove_unreferenced_vertices()
        if hasattr(mesh, "merge_vertices"):
            mesh.merge_vertices()
    except Exception:
        pass

    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        raise RuntimeError("volumetric reconstruction generated empty mesh")
    return mesh
