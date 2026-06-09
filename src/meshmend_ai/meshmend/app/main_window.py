from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import trimesh
from PySide6 import QtCore, QtWidgets

from meshmend.ai import GenerationRequest, get_adapter
from meshmend.app.main import SCALE_PRESETS
from meshmend.core import add_circular_base, auto_scale_to_height, build_printability_report, decimate_mesh, load_mesh, remesh_subdivide
from meshmend.export import export_slicer_ready
from meshmend.repair import repair_mesh


class MeshViewport(QtWidgets.QWidget):
    """Small OpenGL viewport wrapper with a text fallback."""

    def __init__(self) -> None:
        super().__init__()
        self.mesh: trimesh.Trimesh | None = None
        layout = QtWidgets.QVBoxLayout(self)
        self._gl_view = None
        self._label = QtWidgets.QLabel("Open or generate a mesh to view it.")
        self._label.setAlignment(QtCore.Qt.AlignCenter)
        try:
            import pyqtgraph.opengl as gl

            self._gl = gl
            self._gl_view = gl.GLViewWidget()
            self._gl_view.setCameraPosition(distance=70)
            layout.addWidget(self._gl_view)
        except Exception:
            self._gl = None
            layout.addWidget(self._label)

    def set_mesh(self, mesh: trimesh.Trimesh | None) -> None:
        self.mesh = mesh
        if mesh is None:
            self._label.setText("No mesh loaded.")
            return
        if self._gl_view is None or self._gl is None:
            self._label.setText(f"Mesh loaded\nFaces: {len(mesh.faces):,}\nVertices: {len(mesh.vertices):,}\nExtents: {mesh.extents}")
            return
        self._gl_view.clear()
        vertices = np.asarray(mesh.vertices, dtype=float)
        centered = vertices - vertices.mean(axis=0)
        item = self._gl.GLMeshItem(vertexes=centered, faces=np.asarray(mesh.faces), smooth=False, color=(0.72, 0.72, 0.78, 1.0), shader="shaded")
        self._gl_view.addItem(item)
        grid = self._gl.GLGridItem()
        grid.scale(5, 5, 1)
        self._gl_view.addItem(grid)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("MeshMend Local MVP — no paid API required")
        self.mesh: trimesh.Trimesh | None = None
        self.current_path: Path | None = None

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QHBoxLayout(central)

        self.viewport = MeshViewport()
        root.addWidget(self.viewport, stretch=4)

        panel = QtWidgets.QVBoxLayout()
        root.addLayout(panel, stretch=1)

        self.prompt = QtWidgets.QPlainTextEdit("original armored star knight with rifle, ornate panels, heroic scale")
        panel.addWidget(QtWidgets.QLabel("Structured prompt"))
        panel.addWidget(self.prompt)

        self.scale = QtWidgets.QComboBox()
        self.scale.addItems(SCALE_PRESETS.keys())
        self.scale.setCurrentText("32mm")
        panel.addWidget(QtWidgets.QLabel("Miniature scale"))
        panel.addWidget(self.scale)

        self.adapter = QtWidgets.QComboBox()
        self.adapter.addItem("Existing MeshMend local sculptor", "existing")
        self.adapter.addItem("Procedural placeholder", "placeholder")
        panel.addWidget(QtWidgets.QLabel("Local generation adapter"))
        panel.addWidget(self.adapter)

        self.face_count = QtWidgets.QSpinBox()
        self.face_count.setRange(100, 2_000_000)
        self.face_count.setValue(25_000)
        panel.addWidget(QtWidgets.QLabel("Target faces"))
        panel.addWidget(self.face_count)

        for label, handler in (
            ("Open STL/OBJ/GLB/PLY", self.open_mesh),
            ("Generate local placeholder miniature", self.generate_mesh),
            ("Repair mesh", self.repair_current),
            ("Auto-scale", self.scale_current),
            ("Add circular base", self.add_base_current),
            ("Decimate", self.decimate_current),
            ("Remesh/subdivide", self.remesh_current),
            ("Export slicer-ready STL", self.export_current),
        ):
            button = QtWidgets.QPushButton(label)
            button.clicked.connect(handler)
            panel.addWidget(button)

        self.report = QtWidgets.QPlainTextEdit()
        self.report.setReadOnly(True)
        panel.addWidget(QtWidgets.QLabel("Printability report"))
        panel.addWidget(self.report, stretch=1)

    def open_mesh(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open mesh", "", "Meshes (*.stl *.obj *.glb *.ply)")
        if not path:
            return
        self.mesh = load_mesh(path)
        self.current_path = Path(path)
        self.refresh()

    def generate_mesh(self) -> None:
        height = SCALE_PRESETS[self.scale.currentText()]
        request = GenerationRequest(prompt=self.prompt.toPlainText(), height_mm=height, target_faces=self.face_count.value())
        self.mesh = get_adapter(str(self.adapter.currentData())).generate(request)
        self.current_path = None
        self.refresh()

    def repair_current(self) -> None:
        if self.mesh is None:
            return
        self.mesh = repair_mesh(self.mesh).mesh
        self.refresh()

    def scale_current(self) -> None:
        if self.mesh is None:
            return
        self.mesh = auto_scale_to_height(self.mesh, SCALE_PRESETS[self.scale.currentText()])
        self.refresh()

    def add_base_current(self) -> None:
        if self.mesh is None:
            return
        self.mesh = add_circular_base(self.mesh)
        self.refresh()

    def decimate_current(self) -> None:
        if self.mesh is None:
            return
        self.mesh = decimate_mesh(self.mesh, self.face_count.value())
        self.refresh()

    def remesh_current(self) -> None:
        if self.mesh is None:
            return
        self.mesh = remesh_subdivide(self.mesh, self.face_count.value())
        self.refresh()

    def export_current(self) -> None:
        if self.mesh is None:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export mesh", "meshmend_repaired.stl", "Meshes (*.stl *.obj *.glb *.ply)")
        if path:
            export_slicer_ready(self.mesh, path)

    def refresh(self) -> None:
        self.viewport.set_mesh(self.mesh)
        if self.mesh is None:
            self.report.setPlainText("")
            return
        self.report.setPlainText(json.dumps(build_printability_report(self.mesh).to_dict(), indent=2))
