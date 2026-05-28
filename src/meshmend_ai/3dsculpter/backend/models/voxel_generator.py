import numpy as np
import torch
from PIL import Image
import trimesh
from scipy.ndimage import gaussian_filter, sobel

class VoxelGenerator:
    """Generate 3D models from images using depth and feature analysis"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def generate(self, image: Image.Image, resolution: int = 32) -> trimesh.Trimesh:
        """Generate 3D model from image"""
        print(f"Generating 3D model...")
        
        # Convert image to array
        img_array = np.array(image.convert('RGB')).astype(np.float32) / 255.0
        h, w = img_array.shape[:2]
        
        # Generate dense point cloud from image
        points = self._generate_point_cloud(img_array)
        
        if len(points) < 4:
            print("Error: Not enough points generated, returning empty mesh")
            return trimesh.Trimesh(vertices=np.array([]), faces=np.array([]))
        
        # Create mesh from points
        mesh = self._create_mesh_from_points(points)
        
        if mesh is None or len(mesh.vertices) == 0:
            print("Error: Mesh creation failed")
            return trimesh.Trimesh(vertices=np.array([]), faces=np.array([]))
        
        # Clean up mesh
        try:
            mesh.remove_unreferenced_vertices()
        except:
            pass
        
        try:
            # Remove duplicate vertices
            mesh.merge_vertices()
        except:
            pass
        
        # Smooth mesh
        try:
            trimesh.smoothing.filter_laplacian(mesh, iterations=3, implicit_time_steps=1)
        except:
            pass
        
        return mesh
    
    def _generate_point_cloud(self, img_array: np.ndarray) -> np.ndarray:
        """Generate point cloud from image with feature weighting"""
        h, w = img_array.shape[:2]
        
        # Create depth map
        depth_map = self._compute_depth_map(img_array)
        
        # Create feature map for dense sampling
        feature_map = self._compute_feature_map(img_array)
        
        # Generate points
        points = []
        
        # Sample with higher density at features
        step = max(1, int(np.sqrt(w * h / 5000)))  # Target ~5000 samples
        
        for y in range(0, h, step):
            for x in range(0, w, step):
                # Get depth at this location
                depth = depth_map[y, x]
                feature = feature_map[y, x]
                
                # 3D position
                px = (x / w) * 3 - 1.5
                py = (y / h) * 3 - 1.5
                pz = depth * 2 - 0.5
                
                # Base point
                points.append([px, py, pz])
                
                # Add extra points in high-feature areas
                if feature > 0.5:
                    for _ in range(2):
                        # Add points with slight variations
                        jitter = np.random.randn(3) * 0.05
                        points.append([px + jitter[0], py + jitter[1], pz + jitter[2]])
        
        # Add internal points for better mesh
        for _ in range(len(points) // 2):
            y = np.random.randint(0, h)
            x = np.random.randint(0, w)
            
            depth = depth_map[y, x]
            feature = feature_map[y, x]
            
            px = (x / w) * 3 - 1.5
            py = (y / h) * 3 - 1.5
            pz = depth * 2 - 0.5
            
            # Internal point - closer to center
            internal_z = pz * (0.2 + 0.3 * feature)
            jitter = np.random.randn(3) * 0.03
            points.append([px + jitter[0], py + jitter[1], internal_z + jitter[2]])
        
        print(f"Generated {len(points)} points")
        return np.array(points)
    
    def _compute_depth_map(self, img_array: np.ndarray) -> np.ndarray:
        """Compute depth from brightness and edges"""
        h, w = img_array.shape[:2]
        
        # Brightness-based
        gray = np.mean(img_array, axis=2)
        brightness = gray
        
        # Edge-based
        gx = sobel(gray, axis=0)
        gy = sobel(gray, axis=1)
        edges = np.sqrt(gx**2 + gy**2)
        edges = gaussian_filter(edges, sigma=1)
        edges = edges / (edges.max() + 1e-8)
        
        # Color variation
        r, g, b = img_array[:,:,0], img_array[:,:,1], img_array[:,:,2]
        color_var = np.std([r, g, b], axis=0)
        color_var = color_var / (color_var.max() + 1e-8)
        
        # Combine with exponential boost for features
        depth = brightness * 0.3 + edges * 0.4 + color_var * 0.3
        depth = gaussian_filter(depth, sigma=2)
        
        # Normalize to [0, 1]
        if depth.max() > depth.min():
            depth = (depth - depth.min()) / (depth.max() - depth.min())
        else:
            depth = np.ones_like(depth) * 0.5
        
        # Boost with power law for better relief
        depth = np.power(depth, 0.6)
        
        return depth
    
    def _compute_feature_map(self, img_array: np.ndarray) -> np.ndarray:
        """Compute feature importance map"""
        gray = np.mean(img_array, axis=2)
        
        # Edge strength
        gx = sobel(gray, axis=0)
        gy = sobel(gray, axis=1)
        edges = np.sqrt(gx**2 + gy**2)
        
        # Local contrast
        from scipy.ndimage import uniform_filter
        local_mean = uniform_filter(gray, size=5)
        contrast = np.abs(gray - local_mean)
        
        # Color variation
        r, g, b = img_array[:,:,0], img_array[:,:,1], img_array[:,:,2]
        color_var = np.std([r, g, b], axis=0)
        
        # Combine
        features = edges * 0.4 + contrast * 0.35 + color_var * 0.25
        features = gaussian_filter(features, sigma=1)
        features = features / (features.max() + 1e-8)
        
        return features
    
    def _create_mesh_from_points(self, points: np.ndarray) -> trimesh.Trimesh:
        """Create mesh from point cloud"""
        try:
            from scipy.spatial import ConvexHull, Delaunay
            
            # Use convex hull as base for stable mesh
            hull = ConvexHull(points)
            mesh = trimesh.Trimesh(vertices=hull.points, faces=hull.simplices)
            
            # Validate mesh
            if len(mesh.vertices) > 0 and len(mesh.faces) > 0:
                # Check for invalid values
                if not np.any(np.isnan(mesh.vertices)) and not np.any(np.isinf(mesh.vertices)):
                    return mesh
            
            # Fallback: create simple mesh
            return trimesh.Trimesh(vertices=hull.points)
        
        except Exception as e:
            print(f"Mesh creation failed: {e}")
            # Last resort: just return points as vertices
            try:
                return trimesh.Trimesh(vertices=points)
            except:
                return None
