import numpy as np
import trimesh
from scipy.spatial import ConvexHull

try:
    import open3d as o3d
    O3D_AVAILABLE = True
except Exception:
    O3D_AVAILABLE = False


class MeshReconstructor:
    """Reconstruct 3D meshes from point clouds and depth maps."""

    @staticmethod
    def pointcloud_to_mesh(points: np.ndarray, method: str = "poisson") -> trimesh.Trimesh:
        """Convert a point cloud to a mesh.

        Supports `(N, 3)` points or `(N, 6)` where the last 3 columns are normals.
        """
        if points.ndim != 2 or points.shape[1] < 3:
            raise ValueError("points must have shape (N, 3) or (N, 6)")

        normals = None
        if points.shape[1] >= 6:
            normals = points[:, 3:6]
            points = points[:, :3]

        normalized_method = method.lower().strip()
        if normalized_method == "convex":
            return MeshReconstructor.convex_hull_mesh(points)
        if normalized_method == "poisson":
            if O3D_AVAILABLE:
                return MeshReconstructor._poisson_reconstruction_o3d(points, normals=normals, depth=12)
            return MeshReconstructor.poisson_reconstruction(points)
        if normalized_method in {"ball", "ball_pivot", "ball-pivot"}:
            if O3D_AVAILABLE:
                return MeshReconstructor._ball_pivot_mesh_o3d(points, normals=normals, radius=None)
            return MeshReconstructor.ball_pivot_mesh(points)
        raise ValueError(f"Unsupported reconstruction method: {method}")

    @staticmethod
    def convex_hull_mesh(points: np.ndarray) -> trimesh.Trimesh:
        """Create a watertight-ish mesh using a convex hull."""
        try:
            hull = ConvexHull(points)
            return trimesh.Trimesh(vertices=hull.points, faces=hull.simplices)
        except Exception as exc:
            print(f"Convex hull failed: {exc}, using point cloud directly")
            return trimesh.Trimesh(vertices=points)

    @staticmethod
    def ball_pivot_mesh(points: np.ndarray, radius: float = 0.1) -> trimesh.Trimesh:
        """Fallback ball-pivot implementation via voxelization."""
        try:
            mesh = trimesh.Trimesh(vertices=points)
            mesh_voxel = mesh.voxelized(pitch=max(1e-4, radius * 0.1))
            return mesh_voxel.as_mesh()
        except Exception as exc:
            print(f"Ball pivot failed: {exc}")
            return MeshReconstructor.convex_hull_mesh(points)
            
    @staticmethod
    def poisson_reconstruction(points: np.ndarray, depth: int = 10) -> trimesh.Trimesh:
        """Fallback reconstruction using Delaunay hull surfaces."""
        try:
            from scipy.spatial import Delaunay

            if len(points) > 5000:
                indices = np.random.choice(len(points), 5000, replace=False)
                points_sample = points[indices]
            else:
                points_sample = points

            delaunay = Delaunay(points_sample)
            mesh = trimesh.Trimesh(vertices=delaunay.points, faces=delaunay.convex_hull)
            if len(mesh.vertices) > 0 and len(mesh.faces) > 0:
                return mesh
            return MeshReconstructor.convex_hull_mesh(points)
        except Exception as exc:
            print(f"Delaunay triangulation failed: {exc}, using convex hull")
            return MeshReconstructor.convex_hull_mesh(points)

    @staticmethod
    def _poisson_reconstruction_o3d(points: np.ndarray, normals: np.ndarray | None = None, depth: int = 12) -> trimesh.Trimesh:
        """Open3D Poisson reconstruction with robust fallbacks."""
        try:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
            
            if normals is not None and normals.shape == points.shape:
                nrm = normals.astype(np.float64)
                nrm = nrm / (np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12)
                pcd.normals = o3d.utility.Vector3dVector(nrm)
            else:
                pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))
                pcd.orient_normals_consistent_tangent_plane(20)

            mesh_o3d, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth)
            mesh_o3d = mesh_o3d.crop(pcd.get_axis_aligned_bounding_box())

            density_values = np.asarray(densities)
            if density_values.size == len(mesh_o3d.vertices):
                # Keep more low-density detail instead of pruning too aggressively.
                threshold = np.quantile(density_values, 0.015)
                mesh_o3d.remove_vertices_by_mask(~(density_values > threshold))

            mesh_o3d.remove_duplicated_vertices()
            mesh_o3d.remove_degenerate_triangles()
            mesh_o3d.remove_non_manifold_edges()
            mesh_o3d.compute_vertex_normals()

            vertices = np.asarray(mesh_o3d.vertices, dtype=np.float64)
            faces = np.asarray(mesh_o3d.triangles, dtype=np.int64)
            return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        except Exception as exc:
            print(f"Open3D Poisson failed: {exc}, falling back to Delaunay")
            return MeshReconstructor.poisson_reconstruction(points)

    @staticmethod
    def _ball_pivot_mesh_o3d(points: np.ndarray, normals: np.ndarray | None = None, radius: float | None = None) -> trimesh.Trimesh:
        """Open3D ball-pivot reconstruction with adaptive radius."""
        try:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))

            if normals is not None and normals.shape == points.shape:
                nrm = normals.astype(np.float64)
                nrm = nrm / (np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12)
                pcd.normals = o3d.utility.Vector3dVector(nrm)
            else:
                pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.03, max_nn=30))
                pcd.orient_normals_consistent_tangent_plane(20)

            if radius is None:
                dists = pcd.compute_nearest_neighbor_distance()
                radius = max(1e-4, float(np.mean(dists)) * 1.0) if len(dists) > 0 else 0.008

            radii = o3d.utility.DoubleVector([radius, radius * 2.0])
            mesh_o3d = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, radii)

            mesh_o3d.remove_duplicated_vertices()
            mesh_o3d.remove_degenerate_triangles()
            mesh_o3d.remove_non_manifold_edges()
            mesh_o3d.compute_vertex_normals()

            vertices = np.asarray(mesh_o3d.vertices, dtype=np.float64)
            faces = np.asarray(mesh_o3d.triangles, dtype=np.int64)
            return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        except Exception as exc:
            print(f"Open3D Ball Pivoting failed: {exc}, falling back to convex hull")
            return MeshReconstructor.convex_hull_mesh(points)

    @staticmethod
    def voxel_to_mesh(voxel_grid: np.ndarray) -> trimesh.Trimesh:
        """Convert a binary voxel grid into a mesh."""
        voxel_grid = voxel_grid.astype(bool)
        return trimesh.voxel.Voxel(voxel_grid).as_mesh()

    @staticmethod
    def smooth_mesh(mesh: trimesh.Trimesh, iterations: int = 5) -> trimesh.Trimesh:
        """Apply Laplacian smoothing to a mesh in-place."""
        try:
            trimesh.smoothing.filter_laplacian(mesh, iterations=iterations, implicit_time_steps=1)
            return mesh
        except Exception:
            return mesh

    @staticmethod
    def mesh_from_depth_map(depth_map: np.ndarray, fx: float = 500, fy: float = 500) -> trimesh.Trimesh:
        """Create mesh from a depth map using pinhole camera assumptions."""
        h, w = depth_map.shape
        x = np.arange(w)
        y = np.arange(h)
        xx, yy = np.meshgrid(x, y)

        xx = (xx - w / 2) / fx
        yy = (yy - h / 2) / fy
        
        points = np.stack([xx * depth_map, yy * depth_map, depth_map], axis=-1).reshape(-1, 3)
        points = points[depth_map.flatten() > 0]
        return MeshReconstructor.pointcloud_to_mesh(points)
