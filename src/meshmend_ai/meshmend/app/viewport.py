from __future__ import annotations

import numpy as np
import trimesh
from PySide6 import QtCore, QtWidgets


class MeshViewport(QtWidgets.QWidget):
    """3D viewport with a dependency-light text fallback.

    If ``pyvistaqt`` is installed, the center panel shows an interactive mesh.
    Without it, the MVP still works: import/analysis/repair/export remain usable
    and the center panel shows model statistics.
    """

    def __init__(self) -> None:
        super().__init__()
        self.mesh: trimesh.Trimesh | None = None
        self._plotter = None
        self._fallback_label = QtWidgets.QLabel("Import a mesh to view it.")
        self._fallback_label.setAlignment(QtCore.Qt.AlignCenter)
        self._fallback_label.setWordWrap(True)

        layout = QtWidgets.QVBoxLayout(self)
        try:
            from pyvistaqt import QtInteractor

            self._pv = __import__("pyvista")
            self._plotter = QtInteractor(self)
            layout.addWidget(self._plotter)
            self._plotter.set_background("#20242b")
        except Exception:
            self._pv = None
            layout.addWidget(self._fallback_label)

    def set_mesh(self, mesh: trimesh.Trimesh | None) -> None:
        self.mesh = mesh
        if mesh is None:
            self._fallback_label.setText("No mesh loaded.")
            if self._plotter is not None:
                self._plotter.clear()
            return
        if self._plotter is None or self._pv is None:
            self._fallback_label.setText(
                "Mesh loaded\n\n"
                f"Vertices: {len(mesh.vertices):,}\n"
                f"Faces: {len(mesh.faces):,}\n"
                f"Dimensions: {_format_extents(mesh)}\n\n"
                "Install pyvista + pyvistaqt for the interactive viewport."
            )
            return
        self._plotter.clear()
        vertices = np.asarray(mesh.vertices, dtype=float)
        faces = np.hstack([np.full((len(mesh.faces), 1), 3, dtype=np.int64), np.asarray(mesh.faces, dtype=np.int64)]).ravel()
        pv_mesh = self._pv.PolyData(vertices, faces)
        self._plotter.add_mesh(pv_mesh, color="#b8bcc8", smooth_shading=False, show_edges=False)
        self._plotter.add_axes()
        self._plotter.reset_camera()


def _format_extents(mesh: trimesh.Trimesh) -> str:
    return " × ".join(f"{float(value):.3f} mm" for value in np.asarray(mesh.extents, dtype=float))
