"""
3D Viewer using Vispy for PyQt6
"""

from vispy import scene
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton
import numpy as np
from vispy.geometry import meshdata
import trimesh

class Viewer3D(QWidget):
    def __init__(self):
        super().__init__()
        
        # Create canvas
        self.canvas = scene.SceneCanvas(keys='interactive', show=False)
        self.canvas.measure_fps()
        
        # Create view
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = 'turntable'
        self.view.camera.distance = 100
        
        # Model
        self.mesh_visual = None
        self.wire_visual = None
        self.current_mesh = None
        self.detail_mode = True
        
        # Layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        controls = QHBoxLayout()
        self.detail_mode_btn = QPushButton("Detail view: flat + edges")
        self.detail_mode_btn.clicked.connect(self.toggle_detail_mode)
        controls.addWidget(self.detail_mode_btn)
        controls.addStretch()
        layout.addLayout(controls)
        layout.addWidget(self.canvas.native)
    
    def load_mesh(self, trimesh_obj):
        """Load a trimesh object and display it"""
        if self.mesh_visual:
            try:
                # Remove old mesh from scene
                self.mesh_visual.parent = None
            except:
                pass
        if self.wire_visual:
            try:
                self.wire_visual.parent = None
            except:
                pass
        
        self.current_mesh = trimesh_obj
        
        # Create mesh data
        vertices = trimesh_obj.vertices.astype(np.float32)
        faces = trimesh_obj.faces.astype(np.uint32)
        
        # Create vispy mesh
        mesh_data = meshdata.MeshData(
            vertices=vertices,
            faces=faces
        )
        
        # Create mesh visual with detail-friendly shading. Smooth shading looks
        # prettier, but it hides small STL relief and makes detailed minis look
        # blobby/undetailed in the preview.
        self.mesh_visual = scene.visuals.Mesh(
            meshdata=mesh_data,
            shading='flat' if self.detail_mode else 'smooth',
            color=(0.72, 0.72, 0.68, 1.0)
        )
        
        self.view.add(self.mesh_visual)
        if self.detail_mode and len(faces) <= 1_500_000:
            self.wire_visual = scene.visuals.Mesh(
                meshdata=mesh_data,
                mode='lines',
                color=(0.05, 0.05, 0.05, 0.20),
            )
            self.view.add(self.wire_visual)
        else:
            self.wire_visual = None
        self.reset_view()

    def toggle_detail_mode(self):
        """Toggle between preview-smooth and detail-inspection rendering."""
        self.detail_mode = not self.detail_mode
        self.detail_mode_btn.setText("Detail view: flat + edges" if self.detail_mode else "Preview view: smooth")
        if self.current_mesh is not None:
            self.load_mesh(self.current_mesh)
    
    def rotate_model(self, x, y, z):
        """Rotate model by given angles (in degrees)"""
        if self.mesh_visual:
            # Convert to radians
            x_rad = np.radians(x)
            y_rad = np.radians(y)
            z_rad = np.radians(z)
            
            # Apply rotations
            self.mesh_visual.transform.rotate(x_rad, (1, 0, 0))
            self.mesh_visual.transform.rotate(y_rad, (0, 1, 0))
            self.mesh_visual.transform.rotate(z_rad, (0, 0, 1))
    
    def scale_model(self, factor):
        """Scale model by given factor"""
        if self.mesh_visual:
            current_scale = self.mesh_visual.transform.scale
            self.mesh_visual.transform.scale = (
                current_scale[0] * factor,
                current_scale[1] * factor,
                current_scale[2] * factor
            )
    
    def reset_view(self):
        """Reset camera to default view"""
        if self.current_mesh:
            bounds = self.current_mesh.bounds
            center = self.current_mesh.center_mass
            extents = self.current_mesh.extents
            
            # Position camera
            max_extent = np.max(extents)
            self.view.camera.center = center
            self.view.camera.distance = max_extent * 2.5
    
    def get_canvas_native(self):
        """Get native widget for embedding in PyQt"""
        return self.canvas.native
