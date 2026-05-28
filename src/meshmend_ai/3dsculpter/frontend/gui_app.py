# NEW FILE
import sys
import os
import traceback
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
from PIL import Image

# Ensure project root is on sys.path so "backend.*" imports work when run directly
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QPixmap, QAction
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QPushButton, QLabel, QFileDialog,
    QVBoxLayout, QHBoxLayout, QMessageBox, QProgressBar, QSizePolicy
)

from backend.pipelines.image_to_mesh import reconstruct_mesh_from_image, reconstruct_mesh_from_image_volumetric
from backend.processing.mesh_simplifier import MeshSimplifier

# Optional 3D viewers
try:
    import open3d as o3d
    O3D_AVAILABLE = True
except Exception:
    O3D_AVAILABLE = False

try:
    import trimesh
    TRIMESH_AVAILABLE = True
except Exception:
    TRIMESH_AVAILABLE = False


class MeshWorker(QThread):
    """
    QThread worker to run mesh reconstruction in the background to keep the UI responsive.
    """
    finished = Signal(object, str)  # emits (mesh or None, error_message)

    def __init__(self, image_path: str, method: str = "poisson", num_points: int = 6000, parent=None):
        super().__init__(parent)
        self.image_path = image_path
        self.method = method
        self.num_points = num_points

    def run(self):
        try:
            mesh = self._try_meshmend_sculptor()
            if mesh is not None:
                self.finished.emit(mesh, "")
                return

            pil_img = Image.open(self.image_path).convert("RGB")
            mesh = reconstruct_mesh_from_image(pil_img, num_points=self.num_points, method=self.method)
            if _is_collapsed_mesh(mesh):
                mesh = reconstruct_mesh_from_image_volumetric(pil_img, vol_size=128, voxel_size_mm=0.22)
            mesh = MeshSimplifier.polish_mesh(mesh, quality="high")
            mesh = MeshSimplifier.simplify_for_printing(mesh, quality="high")
            self.finished.emit(mesh, "")
        except Exception as e:
            tb = traceback.format_exc()
            self.finished.emit(None, f"{e}\n{tb}")

    def _try_meshmend_sculptor(self):
        """Use the prompt/image-aware MeshMend sculptor before old silhouette reconstruction."""
        try:
            meshmend_src = Path(__file__).resolve().parents[3]
            if str(meshmend_src) not in sys.path:
                sys.path.insert(0, str(meshmend_src))
            from meshmend_ai.sculptor import get_sculptor_foundation

            image_path = Path(self.image_path)
            filename_cues = " ".join(image_path.stem.replace("-", "_").split("_"))
            output_path = get_sculptor_foundation().create_model(
                f"uploaded reference image subject; filename cues: {filename_cues}; preserve subject topology and do not force a humanoid miniature",
                image_path=image_path,
                scale_mm=32.0,
                print_detail_um=50,
            )
            if TRIMESH_AVAILABLE:
                mesh = trimesh.load(output_path, force="mesh")
                if mesh is not None and len(mesh.faces) >= 1000:
                    return mesh
        except Exception as exc:
            print(f"MeshMend sculptor unavailable in image GUI; falling back: {exc}")
        return None


def _mesh_fullness_score(mesh) -> float:
    try:
        ext = np.asarray(mesh.extents, dtype=np.float32)
        bbox_vol = float(np.prod(np.maximum(ext, 1e-6)))
        mesh_vol = float(abs(mesh.volume))
        return mesh_vol / (bbox_vol + 1e-8)
    except Exception:
        return 0.0


def _is_blob_like_mesh(mesh) -> bool:
    try:
        if mesh is None or len(mesh.vertices) < 400 or len(mesh.faces) < 600:
            return True
        ext = np.asarray(mesh.extents, dtype=np.float32)
        max_extent = float(np.max(ext))
        min_extent = float(np.min(ext))
        if max_extent <= 1e-6:
            return True
        flat_ratio = min_extent / max_extent
        fullness = _mesh_fullness_score(mesh)
        return flat_ratio < 0.08 or fullness > 0.72
    except Exception:
        return True


def _is_underfleshed_mesh(mesh) -> bool:
    try:
        ext = np.asarray(mesh.extents, dtype=np.float32)
        max_ext = float(np.max(ext)) + 1e-8
        min_ext = float(np.min(ext))
        thickness_ratio = min_ext / max_ext
        fullness = _mesh_fullness_score(mesh)
        return thickness_ratio < 0.17 or fullness < 0.055
    except Exception:
        return True


def _is_detail_poor_mesh(mesh) -> bool:
    try:
        if mesh is None:
            return True
        v_count = int(len(mesh.vertices))
        f_count = int(len(mesh.faces))
        if v_count < 1800 or f_count < 3000:
            return True
        ext = np.asarray(mesh.extents, dtype=np.float32)
        max_ext = float(np.max(ext)) + 1e-8
        diagonal = float(np.linalg.norm(ext)) + 1e-8
        density = f_count / (diagonal * max_ext + 1e-8)
        return density < 550.0
    except Exception:
        return True


def _is_collapsed_mesh(mesh) -> bool:
    try:
        if mesh is None or len(mesh.vertices) < 900 or len(mesh.faces) < 1400:
            return True
        ext = np.asarray(mesh.extents, dtype=np.float32)
        max_ext = float(np.max(ext)) + 1e-8
        min_ext = float(np.min(ext))
        flat_ratio = min_ext / max_ext
        if flat_ratio < 0.045:
            return True
        bbox_vol = float(np.prod(np.maximum(ext, 1e-6)))
        fullness = float(abs(mesh.volume)) / (bbox_vol + 1e-8)
        return fullness < 0.018 or fullness > 0.86
    except Exception:
        return True


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image → STL GUI")
        self.resize(980, 720)

        self.image_path: str | None = None
        self.mesh = None  # trimesh.Trimesh
        self.temp_stl_path: str | None = None
        self.worker: MeshWorker | None = None

        self._setup_ui()
        self._setup_menu()

    def _setup_ui(self):
        central = QWidget(self)
        self.setCentralWidget(central)

        # Image preview
        self.image_label = QLabel("No image loaded")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.image_label.setStyleSheet("border: 1px solid #444;")

        # Buttons
        self.btn_open = QPushButton("Open Image")
        self.btn_open.clicked.connect(self.open_image)

        self.btn_reconstruct = QPushButton("Reconstruct Mesh")
        self.btn_reconstruct.clicked.connect(self.reconstruct_mesh)
        self.btn_reconstruct.setEnabled(False)

        self.btn_view = QPushButton("View 3D")
        self.btn_view.clicked.connect(self.view_mesh)
        self.btn_view.setEnabled(False)

        self.btn_save = QPushButton("Save STL")
        self.btn_save.clicked.connect(self.save_stl)
        self.btn_save.setEnabled(False)

        # Progress bar
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)  # Busy indicator
        self.progress.setVisible(False)

        # Layouts
        btn_row = QHBoxLayout()
        btn_row.addWidget(self.btn_open)
        btn_row.addWidget(self.btn_reconstruct)
        btn_row.addWidget(self.btn_view)
        btn_row.addWidget(self.btn_save)

        layout = QVBoxLayout()
        layout.addWidget(self.image_label, stretch=1)
        layout.addLayout(btn_row)
        layout.addWidget(self.progress)

        central.setLayout(layout)

    def _setup_menu(self):
        # basic menu
        menubar = self.menuBar()
        file_menu = menubar.addMenu("&File")

        act_open = QAction("Open Image...", self)
        act_open.triggered.connect(self.open_image)
        file_menu.addAction(act_open)

        act_save = QAction("Save STL...", self)
        act_save.triggered.connect(self.save_stl)
        act_save.setEnabled(False)
        self.act_save = act_save
        file_menu.addAction(act_save)

        file_menu.addSeparator()
        act_quit = QAction("Quit", self)
        act_quit.triggered.connect(self.close)
        file_menu.addAction(act_quit)

        view_menu = menubar.addMenu("&View")
        act_view3d = QAction("View 3D", self)
        act_view3d.triggered.connect(self.view_mesh)
        act_view3d.setEnabled(False)
        self.act_view3d = act_view3d
        view_menu.addAction(act_view3d)

    def open_image(self):
        dlg = QFileDialog(self, "Open Image")
        dlg.setNameFilters(["Images (*.png *.jpg *.jpeg *.bmp *.webp)", "All files (*)"])
        dlg.setFileMode(QFileDialog.ExistingFile)
        if dlg.exec():
            files = dlg.selectedFiles()
            if files:
                self.load_image(files[0])

    def load_image(self, path: str):
        self.image_path = path
        self.mesh = None
        self._cleanup_temp_stl()

        pix = QPixmap(self.image_path)
        if pix.isNull():
            QMessageBox.warning(self, "Error", "Failed to load image.")
            self.image_label.setText("No image loaded")
            self.btn_reconstruct.setEnabled(False)
            self.btn_view.setEnabled(False)
            self.btn_save.setEnabled(False)
            self.act_view3d.setEnabled(False)
            self.act_save.setEnabled(False)
            return

        # Fit image into label while preserving aspect ratio
        self._update_image_preview(pix)

        self.btn_reconstruct.setEnabled(True)
        self.btn_view.setEnabled(False)
        self.btn_save.setEnabled(False)
        self.act_view3d.setEnabled(False)
        self.act_save.setEnabled(False)

    def _update_image_preview(self, pix: QPixmap):
        # Scale to current label size
        target = self.image_label.size()
        if target.width() > 0 and target.height() > 0:
            scaled = pix.scaled(target, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.image_label.setPixmap(scaled)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Rescale preview on window resize
        if self.image_path and self.image_label.pixmap():
            pix = QPixmap(self.image_path)
            if not pix.isNull():
                self._update_image_preview(pix)

    def reconstruct_mesh(self):
        if not self.image_path:
            QMessageBox.information(self, "No Image", "Please open an image first.")
            return

        # Disable UI and show busy progress
        self.progress.setVisible(True)
        self.btn_open.setEnabled(False)
        self.btn_reconstruct.setEnabled(False)
        self.btn_view.setEnabled(False)
        self.btn_save.setEnabled(False)
        self.act_view3d.setEnabled(False)
        self.act_save.setEnabled(False)

        # Start background worker
        self.worker = MeshWorker(image_path=self.image_path, method="poisson", num_points=22000)
        self.worker.finished.connect(self._on_mesh_built)
        self.worker.start()

    def _on_mesh_built(self, mesh_obj, err_msg: str):
        self.progress.setVisible(False)
        self.btn_open.setEnabled(True)

        if err_msg or mesh_obj is None:
            QMessageBox.critical(self, "Reconstruction Error", f"Failed to reconstruct mesh:\n{err_msg}")
            self.mesh = None
            self.btn_view.setEnabled(False)
            self.btn_save.setEnabled(False)
            self.act_view3d.setEnabled(False)
            self.act_save.setEnabled(False)
            return

        # Accept the mesh
        self.mesh = mesh_obj

        # Basic post-check
        if not hasattr(self.mesh, "vertices") or len(self.mesh.vertices) == 0:
            QMessageBox.warning(self, "Empty Mesh", "Reconstruction produced an empty mesh.")
            self.btn_view.setEnabled(False)
            self.btn_save.setEnabled(False)
            self.act_view3d.setEnabled(False)
            self.act_save.setEnabled(False)
            return

        self.btn_view.setEnabled(True)
        self.btn_save.setEnabled(True)
        self.act_view3d.setEnabled(True)
        self.act_save.setEnabled(True)

        # Save a temp STL for quick preview/use (overwritten each run)
        self._save_temp_stl()

        QMessageBox.information(self, "Success", "Mesh reconstructed successfully.")

    def view_mesh(self):
        if self.mesh is None:
            QMessageBox.information(self, "No Mesh", "No mesh to view. Reconstruct first.")
            return

        # Prefer Open3D viewer if available
        if O3D_AVAILABLE:
            try:
                mesh_o3d = self._to_o3d_mesh(self.mesh)
                mesh_o3d.compute_vertex_normals()
                o3d.visualization.draw_geometries([mesh_o3d], window_name="3D Viewer")
                return
            except Exception as e:
                print(f"Open3D viewer failed: {e}")

        # Fallback to trimesh viewer
        if TRIMESH_AVAILABLE:
            try:
                scene = trimesh.Scene(self.mesh)
                scene.show()
                return
            except Exception as e:
                print(f"trimesh viewer failed: {e}")

        QMessageBox.warning(self, "Viewer Unavailable", "No 3D viewer available. Install open3d or trimesh viewer dependencies.")

    def save_stl(self):
        if self.mesh is None:
            QMessageBox.information(self, "No Mesh", "No mesh to save. Reconstruct first.")
            return

        path, _ = QFileDialog.getSaveFileName(self, "Save STL", "model.stl", "STL Files (*.stl)")
        if not path:
            return
        try:
            # Ensure binary STL
            self.mesh.export(path, file_type="stl")
            QMessageBox.information(self, "Saved", f"Saved STL to:\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Failed to save STL:\n{e}")

    def _save_temp_stl(self):
        # Create/overwrite temp STL for quick access
        try:
            if self.temp_stl_path and os.path.exists(self.temp_stl_path):
                os.remove(self.temp_stl_path)
            with NamedTemporaryFile(delete=False, suffix=".stl", prefix="preview_") as tmp:
                self.temp_stl_path = tmp.name
            self.mesh.export(self.temp_stl_path, file_type="stl")
        except Exception as e:
            print(f"Temp STL save failed: {e}")

    def _cleanup_temp_stl(self):
        if self.temp_stl_path and os.path.exists(self.temp_stl_path):
            try:
                os.remove(self.temp_stl_path)
            except Exception:
                pass
        self.temp_stl_path = None

    @staticmethod
    def _to_o3d_mesh(mesh_trimesh):
        import numpy as np
        import open3d as o3d
        vertices = np.asarray(mesh_trimesh.vertices, dtype=np.float64)
        faces = np.asarray(mesh_trimesh.faces, dtype=np.int64)
        mesh_o3d = o3d.geometry.TriangleMesh()
        mesh_o3d.vertices = o3d.utility.Vector3dVector(vertices)
        mesh_o3d.triangles = o3d.utility.Vector3iVector(faces)
        return mesh_o3d

    def closeEvent(self, event):
        try:
            if self.worker and self.worker.isRunning():
                self.worker.quit()
                self.worker.wait(2000)
        except Exception:
            pass
        self._cleanup_temp_stl()
        super().closeEvent(event)


def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
