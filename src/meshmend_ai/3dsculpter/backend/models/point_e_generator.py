import numpy as np
from PIL import Image

try:
    from utils.config import DEVICE
except ModuleNotFoundError:
    from backend.utils.config import DEVICE

try:
    import cv2
    CV2_AVAILABLE = True
except Exception:
    CV2_AVAILABLE = False


class PointEGenerator:
    """Generate 3D point clouds from images using depth estimation."""

    def __init__(self):
        self.device = DEVICE
        self.use_fallback = True
        self._depth_estimator = None

    def generate(self, image: Image.Image, num_points: int = 4096) -> np.ndarray:
        """Generate a point cloud from an image."""
        return self.enhanced_depth_to_pointcloud(image, num_points)

    def generate_with_normals(self, image: Image.Image, num_points: int = 4096) -> tuple[np.ndarray, np.ndarray]:
        """Generate point cloud and per-point normals from an image."""
        print("Generating 3D point cloud with normals from image...")
        image = image.convert("RGB")
        img_array = np.array(image).astype(np.float32) / 255.0
        h, w = img_array.shape[:2]
        fg_mask = self._extract_foreground_mask(img_array)

        try:
            depth_map = self._estimate_depth_midas(img_array)
        except Exception as exc:
            print(f"  MiDaS unavailable ({exc}), using simple depth estimation...")
            depth_map = self._estimate_depth_simple(img_array)

        # Keep background shallow so reconstruction favors the subject silhouette.
        depth_map = np.where(fg_mask, depth_map, depth_map * 0.25)

        points, normals = self._sample_points_with_normals(depth_map, h, w, fg_mask=fg_mask)
        internal_points = self._create_internal_points(depth_map, h, w, num_points // 5, fg_mask=fg_mask)
        internal_normals = np.tile(np.array([[0.0, 0.0, 1.0]], dtype=np.float32), (len(internal_points), 1))

        points = np.vstack([points, internal_points])
        normals = np.vstack([normals, internal_normals])

        if len(points) > num_points:
            indices = np.random.choice(len(points), num_points, replace=False)
        else:
            indices = np.random.choice(len(points), num_points, replace=True)
        points = points[indices]
        normals = normals[indices]

        normals = normals / (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12)

        print(f"Generated {len(points)} points (with normals)")
        print(f"  Depth range: [{depth_map.min():.3f}, {depth_map.max():.3f}]")
        print(f"  Foreground coverage: {float(np.mean(fg_mask)) * 100:.1f}%")
        return points, normals

    def enhanced_depth_to_pointcloud(self, image: Image.Image, num_points: int = 4096) -> np.ndarray:
        """Create a point cloud from image-derived depth."""
        print("Generating 3D point cloud from image...")
        image = image.convert("RGB")
        img_array = np.array(image).astype(np.float32) / 255.0

        h, w = img_array.shape[:2]
        fg_mask = self._extract_foreground_mask(img_array)

        try:
            depth_map = self._estimate_depth_midas(img_array)
        except Exception:
            print("  MiDaS unavailable, using simple depth estimation...")
            depth_map = self._estimate_depth_simple(img_array)

        depth_map = np.where(fg_mask, depth_map, depth_map * 0.25)

        points = self._depth_to_points(depth_map, h, w, fg_mask=fg_mask)
        internal_points = self._create_internal_points(depth_map, h, w, num_points // 5, fg_mask=fg_mask)
        points = np.vstack([points, internal_points])

        if len(points) > num_points:
            indices = np.random.choice(len(points), num_points, replace=False)
        else:
            indices = np.random.choice(len(points), num_points, replace=True)
        points = points[indices]

        print(f"Generated {len(points)} points")
        print(f"  Depth range: [{depth_map.min():.3f}, {depth_map.max():.3f}]")
        print(f"  Foreground coverage: {float(np.mean(fg_mask)) * 100:.1f}%")
        return points

    def _extract_foreground_mask(self, img_array: np.ndarray) -> np.ndarray:
        """Estimate a single-subject foreground mask to avoid meshing the background."""
        h, w = img_array.shape[:2]

        if CV2_AVAILABLE:
            try:
                img_u8 = (img_array * 255).astype(np.uint8)
                mask = np.zeros((h, w), np.uint8)

                # Strong center prior: generated miniatures are expected to be centered.
                margin_x = max(6, int(w * 0.12))
                margin_y = max(6, int(h * 0.12))
                rect = (margin_x, margin_y, max(1, w - 2 * margin_x), max(1, h - 2 * margin_y))
                bgd = np.zeros((1, 65), np.float64)
                fgd = np.zeros((1, 65), np.float64)

                cv2.grabCut(img_u8, mask, rect, bgd, fgd, 4, cv2.GC_INIT_WITH_RECT)
                fg_mask = np.logical_or(mask == cv2.GC_FGD, mask == cv2.GC_PR_FGD)
            except Exception:
                fg_mask = self._threshold_mask(img_array)
        else:
            fg_mask = self._threshold_mask(img_array)

        from scipy.ndimage import binary_closing, binary_fill_holes, binary_opening, binary_dilation

        fg_mask = binary_opening(fg_mask, structure=np.ones((3, 3), dtype=bool))
        fg_mask = binary_closing(fg_mask, structure=np.ones((5, 5), dtype=bool))
        fg_mask = binary_fill_holes(fg_mask)
        fg_mask = binary_dilation(fg_mask, structure=np.ones((3, 3), dtype=bool))

        # Ensure mask is usable; otherwise build a detail-driven centered mask.
        # A plain centered rectangle makes unrelated images reconstruct into the
        # same cylinder/blob, so keep image-specific color/edge cues instead.
        coverage = float(np.mean(fg_mask))
        if coverage < 0.04 or coverage > 0.90:
            fallback = self._detail_driven_mask(img_array)
            fallback_coverage = float(np.mean(fallback))
            if 0.015 <= fallback_coverage <= 0.72:
                return fallback

            # Last resort: an oval center prior intersected with strongest local
            # image variation, not a filled rectangle/cylinder.
            yy, xx = np.indices((h, w))
            center_prior = (((xx - w * 0.5) / max(w * 0.40, 1)) ** 2 + ((yy - h * 0.52) / max(h * 0.42, 1)) ** 2) < 1.0
            return center_prior & fallback

        return fg_mask.astype(bool)

    def _detail_driven_mask(self, img_array: np.ndarray) -> np.ndarray:
        """Fallback foreground mask based on image-specific edges/color distance."""
        h, w = img_array.shape[:2]
        gray = np.mean(img_array, axis=2)
        sat = np.max(img_array, axis=2) - np.min(img_array, axis=2)

        border = np.concatenate(
            [img_array[0, :, :], img_array[-1, :, :], img_array[:, 0, :], img_array[:, -1, :]],
            axis=0,
        )
        bg_color = np.median(border, axis=0)
        color_dist = np.linalg.norm(img_array - bg_color[None, None, :], axis=2)

        grad_x = np.gradient(gray, axis=0)
        grad_y = np.gradient(gray, axis=1)
        edge_mag = np.sqrt(grad_x**2 + grad_y**2)

        yy, xx = np.indices((h, w))
        center_prior = (((xx - w * 0.5) / max(w * 0.47, 1)) ** 2 + ((yy - h * 0.52) / max(h * 0.47, 1)) ** 2) < 1.0
        mask = center_prior & (
            (color_dist > np.quantile(color_dist, 0.74))
            | (sat > np.quantile(sat, 0.66))
            | (edge_mag > np.quantile(edge_mag, 0.80))
        )

        from scipy.ndimage import binary_closing, binary_fill_holes, binary_opening

        mask = binary_opening(mask, structure=np.ones((3, 3), dtype=bool))
        mask = binary_closing(mask, structure=np.ones((5, 5), dtype=bool))
        return binary_fill_holes(mask).astype(bool)

    def _threshold_mask(self, img_array: np.ndarray) -> np.ndarray:
        """Fallback mask extraction based on gradients and saturation."""
        gray = np.mean(img_array, axis=2)
        sat = np.max(img_array, axis=2) - np.min(img_array, axis=2)

        grad_x = np.gradient(gray, axis=0)
        grad_y = np.gradient(gray, axis=1)
        grad = np.sqrt(grad_x**2 + grad_y**2)
        grad = grad / (np.max(grad) + 1e-8)

        score = 0.60 * grad + 0.40 * sat
        threshold = float(np.quantile(score, 0.62))
        return score > threshold

    def _sample_mask_coordinates(self, fg_mask: np.ndarray, count: int) -> np.ndarray:
        """Sample coordinates from mask with replacement as needed."""
        coords = np.argwhere(fg_mask)
        if len(coords) == 0:
            h, w = fg_mask.shape
            ys = np.random.randint(0, h, size=count)
            xs = np.random.randint(0, w, size=count)
            return np.stack([ys, xs], axis=1)

        replace = len(coords) < count
        idx = np.random.choice(len(coords), size=count, replace=replace)
        return coords[idx]

    def _estimate_depth_simple(self, img_array: np.ndarray) -> np.ndarray:
        """Estimate pseudo depth from image cues when MiDaS is unavailable."""
        gray = np.mean(img_array, axis=2)
        brightness_depth = gray.copy()

        if CV2_AVAILABLE:
            edges = cv2.Canny((gray * 255).astype(np.uint8), 50, 150).astype(np.float32) / 255.0
        else:
            sx = np.gradient(gray, axis=0)
            sy = np.gradient(gray, axis=1)
            mag = np.sqrt(sx**2 + sy**2)
            edges = mag / (mag.max() + 1e-8)

        hsv = rgb_to_hsv(img_array)
        saturation = hsv[:, :, 1]

        # Bias toward high-frequency cues so hard-surface details survive in reconstruction.
        depth = brightness_depth * 0.30 + edges * 0.45 + saturation * 0.25

        from scipy.ndimage import gaussian_filter

        depth = gaussian_filter(depth, sigma=0.65)
        depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        return depth

    def _estimate_depth_midas(self, img_array: np.ndarray) -> np.ndarray:
        """Estimate depth using MiDaS via transformers pipeline."""
        import cv2
        from transformers import pipeline

        if self._depth_estimator is None:
            device_index = 0 if str(self.device).startswith("cuda") else -1
            self._depth_estimator = pipeline(
                "depth-estimation",
                model="Intel/dpt-hybrid-midas",
                device=device_index,
            )

        img_pil = Image.fromarray((img_array * 255).astype(np.uint8))
        result = self._depth_estimator(img_pil)
        depth_map = np.array(result["depth"])

        # Enhance local contrast so panel lines/ridges produce stronger geometry cues.
        from scipy.ndimage import gaussian_filter

        base = depth_map.astype(np.float32)
        local_mean = gaussian_filter(base, sigma=1.0)
        detail = np.clip(base - local_mean, -0.25, 0.25)
        depth_map = base + detail * 0.75
        depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-8)
        if depth_map.shape != img_array.shape[:2]:
            depth_map = cv2.resize(depth_map, (img_array.shape[1], img_array.shape[0]))
        return depth_map

    def _depth_to_points(self, depth_map: np.ndarray, h: int, w: int, fg_mask: np.ndarray | None = None) -> np.ndarray:
        """Convert depth map to 3D points with feature-based sampling."""
        from scipy.ndimage import gaussian_filter, sobel

        gx = sobel(depth_map, axis=0)
        gy = sobel(depth_map, axis=1)
        edge_strength = np.sqrt(gx**2 + gy**2)
        edge_strength = gaussian_filter(edge_strength, sigma=1)
        edge_strength = edge_strength / (edge_strength.max() + 1e-8)

        points: list[list[float]] = []
        sample_rate = 1 if (h * w) <= 800 * 800 else 2

        if fg_mask is None:
            fg_mask = np.ones((h, w), dtype=bool)

        for y_idx in range(0, h, sample_rate):
            for x_idx in range(0, w, sample_rate):
                if not fg_mask[min(y_idx, h - 1), min(x_idx, w - 1)]:
                    continue
                num_samples = 8 if edge_strength[min(y_idx, h - 1), min(x_idx, w - 1)] > 0.28 else 2
                for _ in range(num_samples):
                    jy = int(np.clip(y_idx + np.random.randint(-1, 2), 0, h - 1))
                    jx = int(np.clip(x_idx + np.random.randint(-1, 2), 0, w - 1))
                    if not fg_mask[jy, jx]:
                        continue

                    x = (jx / w) * 4 - 2
                    y = (jy / h) * 4 - 2
                    depth = depth_map[jy, jx]
                    depth = min(depth + edge_strength[jy, jx] * 0.3, 1.0)
                    z = depth * 3.0 - 1.0

                    noise = np.random.randn(3) * 0.0028
                    points.append([x + noise[0], y + noise[1], z + noise[2]])

        return np.array(points, dtype=np.float32)

    def _sample_points_with_normals(
        self,
        depth_map: np.ndarray,
        h: int,
        w: int,
        fg_mask: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Sample points and estimate normals from local depth gradients."""
        from scipy.ndimage import gaussian_filter, sobel

        gx = sobel(depth_map, axis=0)
        gy = sobel(depth_map, axis=1)
        edge_strength = np.sqrt(gx**2 + gy**2)
        edge_strength = gaussian_filter(edge_strength, sigma=1)
        edge_strength = edge_strength / (edge_strength.max() + 1e-8)

        fx = float(max(w, 1))
        fy = float(max(h, 1))
        cx = w / 2.0
        cy = h / 2.0

        points: list[list[float]] = []
        normals: list[np.ndarray] = []
        sample_rate = 1 if (h * w) <= 800 * 800 else 2

        if fg_mask is None:
            fg_mask = np.ones((h, w), dtype=bool)

        for y_idx in range(0, h, sample_rate):
            for x_idx in range(0, w, sample_rate):
                if not fg_mask[min(y_idx, h - 1), min(x_idx, w - 1)]:
                    continue
                num_samples = 8 if edge_strength[min(y_idx, h - 1), min(x_idx, w - 1)] > 0.28 else 2
                for _ in range(num_samples):
                    jy = int(np.clip(y_idx + np.random.randint(-1, 2), 0, h - 1))
                    jx = int(np.clip(x_idx + np.random.randint(-1, 2), 0, w - 1))
                    if not fg_mask[jy, jx]:
                        continue

                    depth = float(depth_map[jy, jx])
                    X = ((jx - cx) / fx) * depth
                    Y = ((jy - cy) / fy) * depth
                    Z = depth

                    dzdx = float(depth_map[jy, min(jx + 1, w - 1)] - depth_map[jy, max(jx - 1, 0)]) * 0.5
                    dzdy = float(depth_map[min(jy + 1, h - 1), jx] - depth_map[max(jy - 1, 0), jx]) * 0.5

                    normal = np.array([-dzdx * fx, -dzdy * fy, 1.0], dtype=np.float32)
                    normal = normal / (np.linalg.norm(normal) + 1e-12)

                    noise = np.random.randn(3) * 0.0028
                    points.append([X + noise[0], Y + noise[1], Z + noise[2]])
                    normals.append(normal)

        if not points:
            fallback_points = self._depth_to_points(depth_map, h, w, fg_mask=fg_mask)
            fallback_normals = np.tile(np.array([[0.0, 0.0, 1.0]], dtype=np.float32), (len(fallback_points), 1))
            return fallback_points, fallback_normals

        return np.array(points, dtype=np.float32), np.array(normals, dtype=np.float32)

    def _create_internal_points(
        self,
        depth_map: np.ndarray,
        h: int,
        w: int,
        num_points: int,
        fg_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """Create internal points for better volumetric reconstruction."""
        from scipy.ndimage import sobel

        points: list[list[float]] = []

        gx = sobel(depth_map, axis=0)
        gy = sobel(depth_map, axis=1)
        edge_strength = np.sqrt(gx**2 + gy**2)
        edge_strength = edge_strength / (edge_strength.max() + 1e-8)

        if fg_mask is None:
            fg_mask = np.ones((h, w), dtype=bool)

        num_layers = 2
        points_per_layer = max(1, num_points // num_layers)
        mask_coords = self._sample_mask_coordinates(fg_mask, max(points_per_layer * num_layers * 2, 64))

        for layer in range(num_layers):
            layer_depth_factor = (layer + 1) / num_layers
            for _ in range(points_per_layer):
                if np.random.rand() < 0.8:
                    y_idx, x_idx = mask_coords[np.random.randint(0, len(mask_coords))]
                else:
                    y_idx = np.random.randint(0, h)
                    x_idx = np.random.randint(0, w)
                    if not fg_mask[y_idx, x_idx]:
                        y_idx, x_idx = mask_coords[np.random.randint(0, len(mask_coords))]

                base_depth = depth_map[y_idx, x_idx]
                depth = np.clip(base_depth + edge_strength[y_idx, x_idx] * 0.5, 0, 1)

                x = (x_idx / w) * 4 - 2
                y = (y_idx / h) * 4 - 2
                z = (depth * 2.1 - 0.65) * layer_depth_factor

                noise = np.random.randn(3) * 0.005
                points.append([x + noise[0], y + noise[1], z + noise[2]])

        return np.array(points, dtype=np.float32)

    def unload_model(self):
        """Release cached model handles."""
        self._depth_estimator = None


def rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    """Convert RGB in [0, 1] to HSV in [0, 1]."""
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]

    max_c = np.maximum(r, np.maximum(g, b))
    min_c = np.minimum(r, np.minimum(g, b))

    v = max_c
    delta = max_c - min_c

    s = np.zeros_like(v)
    mask = v != 0
    s[mask] = delta[mask] / v[mask]

    h = np.zeros_like(v)
    mask_r = max_c == r
    mask_g = max_c == g
    mask_b = max_c == b

    h[mask_r] = (g[mask_r] - b[mask_r]) / (delta[mask_r] + 1e-8)
    h[mask_g] = 2 + (b[mask_g] - r[mask_g]) / (delta[mask_g] + 1e-8)
    h[mask_b] = 4 + (r[mask_b] - g[mask_b]) / (delta[mask_b] + 1e-8)

    h = (h * 60) % 360
    h = h / 360.0

    return np.stack([h, s, v], axis=-1)
