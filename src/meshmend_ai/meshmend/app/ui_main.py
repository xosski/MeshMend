from __future__ import annotations

import json
from pathlib import Path

import trimesh
from PySide6 import QtCore, QtWidgets

from meshmend.app.demo_scene import build_studio_detail_demo
from meshmend.app.detail_presets import preset_names
from meshmend.app.mesh_analyzer import MeshAnalysis, analyze_mesh
from meshmend.app.mesh_exporter import export_mesh_file
from meshmend.app.mesh_loader import load_mesh_file
from meshmend.app.mesh_repair import (
    RepairResult,
    RepairSettings,
    auto_repair_mesh,
    edge_sharpen_pass,
    fill_holes,
    fix_normals,
    remove_duplicate_vertices,
)
from meshmend.app.procedural_detail import DetailParameters
from meshmend.app.sculpt_pass import run_studio_sculpt_pass
from meshmend.app.viewport import MeshViewport


class MeshMendWindow(QtWidgets.QMainWindow):
    """Desktop MVP for studio-quality miniature repair.

    The UI is intentionally organized around detail-preserving restoration:
    import/analyze/repair/export are first-class, while destructive operations
    such as smoothing, remeshing, and decimation are absent from the default app.
    """

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("MeshMend — Studio Miniature Repair")
        self.mesh: trimesh.Trimesh | None = None
        self.current_path: Path | None = None
        self.current_analysis: MeshAnalysis | None = None

        root = QtWidgets.QWidget()
        self.setCentralWidget(root)
        root_layout = QtWidgets.QVBoxLayout(root)

        work_area = QtWidgets.QHBoxLayout()
        root_layout.addLayout(work_area, stretch=5)

        self.left_panel = self._build_left_panel()
        work_area.addWidget(self.left_panel, stretch=0)

        self.viewport = MeshViewport()
        work_area.addWidget(self.viewport, stretch=1)

        self.right_panel = self._build_right_panel()
        work_area.addWidget(self.right_panel, stretch=0)

        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(2000)
        self.log.setPlaceholderText("Operation log")
        root_layout.addWidget(self.log, stretch=1)

        self._apply_studio_styles()
        self._log("Ready. Import STL, OBJ, or GLB to begin.")

    def _build_left_panel(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QFrame()
        panel.setMinimumWidth(280)
        panel.setMaximumWidth(340)
        layout = QtWidgets.QVBoxLayout(panel)

        title = QtWidgets.QLabel("MeshMend Tools")
        title.setObjectName("PanelTitle")
        layout.addWidget(title)

        self.studio_master_mode = QtWidgets.QCheckBox("Studio Master Mode")
        self.studio_master_mode.setChecked(True)
        self.studio_master_mode.setToolTip(
            "Preserve sharp edges and miniature-scale sculpt detail. Avoid smoothing, decimation, and global remeshing."
        )
        layout.addWidget(self.studio_master_mode)

        self.repair_preview_mode = QtWidgets.QCheckBox("Repair Preview Mode")
        self.repair_preview_mode.setChecked(False)
        self.repair_preview_mode.setToolTip("Run repair and show before/after statistics without replacing the loaded mesh.")
        layout.addWidget(self.repair_preview_mode)

        self.max_hole_edges = QtWidgets.QSpinBox()
        self.max_hole_edges.setRange(3, 5000)
        self.max_hole_edges.setValue(80)
        layout.addWidget(_labeled_widget("Max hole boundary edges", self.max_hole_edges))

        self.detail_preset = QtWidgets.QComboBox()
        self.detail_preset.addItems(preset_names())
        layout.addWidget(_labeled_widget("Detail preset", self.detail_preset))

        self.detail_strength = _slider(35)
        self.rivet_density = _slider(35)
        self.panel_line_depth = _slider(8, 1, 30)
        self.battle_damage_amount = _slider(20)
        self.surface_texture_strength = _slider(4, 0, 30)
        self.edge_sharpness = _slider(50)
        self.minimum_printable_detail_size = _slider(5, 1, 20)
        for label, widget in (
            ("Detail strength", self.detail_strength),
            ("Rivet density", self.rivet_density),
            ("Panel line depth", self.panel_line_depth),
            ("Battle damage amount", self.battle_damage_amount),
            ("Surface texture strength", self.surface_texture_strength),
            ("Edge sharpness", self.edge_sharpness),
            ("Minimum printable detail size (0.05mm default)", self.minimum_printable_detail_size),
        ):
            layout.addWidget(_labeled_widget(label, widget))

        buttons = [
            ("Import Mesh", self.import_mesh),
            ("Generate Detail Demo Scene", self.generate_detail_demo_scene),
            ("Analyze Mesh", self.analyze_current_mesh),
            ("Auto Repair Mesh", self.auto_repair_current_mesh),
            ("Studio Sculpt Pass", self.studio_sculpt_pass_current_mesh),
            ("Fill Holes", self.fill_holes_current_mesh),
            ("Remove Duplicate Vertices", self.remove_duplicate_vertices_current_mesh),
            ("Fix Normals", self.fix_normals_current_mesh),
            ("Detect Thin Parts", self.detect_thin_parts_current_mesh),
            ("Detect Non-Manifold Regions", self.detect_non_manifold_current_mesh),
            ("Curvature / Sharp Edge Analysis", self.curvature_analysis_current_mesh),
            ("Show Protected Detail Zones", self.protected_detail_zones_current_mesh),
            ("Edge Sharpen Pass", self.edge_sharpen_current_mesh),
            ("Export Repaired Mesh", self.export_mesh),
        ]
        for label, handler in buttons:
            button = QtWidgets.QPushButton(label)
            button.clicked.connect(handler)
            layout.addWidget(button)

        note = QtWidgets.QLabel(
            "Detail Preservation Mode: existing sculpted features are treated as intentional. "
            "MeshMend repairs structural defects; it does not smooth or simplify the miniature."
        )
        note.setWordWrap(True)
        note.setObjectName("HintText")
        layout.addWidget(note)
        layout.addStretch(1)
        return panel

    def _build_right_panel(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QFrame()
        panel.setMinimumWidth(320)
        panel.setMaximumWidth(420)
        layout = QtWidgets.QVBoxLayout(panel)

        title = QtWidgets.QLabel("Mesh Statistics / Problems")
        title.setObjectName("PanelTitle")
        layout.addWidget(title)

        self.stats = QtWidgets.QPlainTextEdit()
        self.stats.setReadOnly(True)
        self.stats.setPlaceholderText("Run analysis to show mesh statistics.")
        layout.addWidget(self.stats, stretch=2)

        self.problem_list = QtWidgets.QListWidget()
        layout.addWidget(QtWidgets.QLabel("Detected Problems"))
        layout.addWidget(self.problem_list, stretch=3)
        return panel

    def import_mesh(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Import mesh",
            "",
            "3D meshes (*.stl *.obj *.glb *.ply);;STL (*.stl);;OBJ (*.obj);;GLB (*.glb);;All files (*.*)",
        )
        if not path:
            return
        try:
            self.mesh = load_mesh_file(path)
            self.current_path = Path(path)
            self.viewport.set_mesh(self.mesh)
            self._log(f"Imported {path}")
            self.analyze_current_mesh()
        except Exception as exc:
            self._error("Import failed", exc)

    def analyze_current_mesh(self) -> None:
        mesh = self._require_mesh()
        if mesh is None:
            return
        self.current_analysis = analyze_mesh(mesh)
        self._render_analysis(self.current_analysis)
        self._log("Analysis complete.")

    def generate_detail_demo_scene(self) -> None:
        try:
            result = build_studio_detail_demo()
            self.mesh = result.mesh
            self.current_path = None
            self.viewport.set_mesh(self.mesh)
            self._log("Generated Studio Detail Demo Scene.")
            self._log(f"  faces after stamps: {result.summary['faces_after_stamps']:,}")
            self._log(f"  stamps: {', '.join(result.summary['stamps'])}")
            self.analyze_current_mesh()
        except Exception as exc:
            self._error("Generate detail demo failed", exc)

    def auto_repair_current_mesh(self) -> None:
        self._run_repair("Auto repair", lambda mesh: auto_repair_mesh(mesh, self._repair_settings()))

    def studio_sculpt_pass_current_mesh(self) -> None:
        mesh = self._require_mesh()
        if mesh is None:
            return
        try:
            self._log("Studio Sculpt Pass started.")
            result = run_studio_sculpt_pass(
                mesh,
                preset_name=str(self.detail_preset.currentText()),
                parameters=self._detail_parameters(),
            )
            if self.repair_preview_mode.isChecked():
                self._log("  PREVIEW MODE: sculpted mesh was generated but original mesh left unchanged.")
            else:
                self.mesh = result.mesh
                self.viewport.set_mesh(self.mesh)
            for action in result.report.actions:
                self._log(f"  {action}")
            for warning in result.report.warnings:
                self._log(f"  WARNING: {warning}")
            self._log(
                "  studio detail delta: "
                f"faces {result.report.before.faces:,} -> {result.report.after.faces:,}; "
                f"vertices {result.report.before.vertices:,} -> {result.report.after.vertices:,}; "
                f"added faces {result.report.added_faces:,}; added vertices {result.report.added_vertices:,}"
            )
            if not self.repair_preview_mode.isChecked():
                self.analyze_current_mesh()
            self._log("Studio Sculpt Pass complete.")
        except Exception as exc:
            self._error("Studio Sculpt Pass failed", exc)

    def fill_holes_current_mesh(self) -> None:
        self._run_repair("Fill holes", lambda mesh: fill_holes(mesh, self._repair_settings()))

    def remove_duplicate_vertices_current_mesh(self) -> None:
        self._run_repair(
            "Remove duplicate vertices",
            lambda mesh: remove_duplicate_vertices(mesh, studio_master_mode=self.studio_master_mode.isChecked()),
        )

    def fix_normals_current_mesh(self) -> None:
        self._run_repair("Fix normals", fix_normals)

    def edge_sharpen_current_mesh(self) -> None:
        self._run_repair("Edge sharpen pass", edge_sharpen_pass)

    def detect_thin_parts_current_mesh(self) -> None:
        self.analyze_current_mesh()
        if self.current_analysis is not None:
            self._log(f"Thin-part warnings: {self.current_analysis.thin_part_warnings}")

    def detect_non_manifold_current_mesh(self) -> None:
        self.analyze_current_mesh()
        if self.current_analysis is not None:
            self._log(f"Non-manifold edges: {self.current_analysis.non_manifold_edges}")

    def curvature_analysis_current_mesh(self) -> None:
        self.analyze_current_mesh()
        if self.current_analysis is not None:
            self._log(
                "Curvature analysis: "
                f"sharp edges={self.current_analysis.sharp_edges}, "
                f"high-curvature faces={self.current_analysis.high_curvature_faces}, "
                f"high-normal-variance vertices={self.current_analysis.high_normal_variance_vertices}"
            )

    def protected_detail_zones_current_mesh(self) -> None:
        self.analyze_current_mesh()
        if self.current_analysis is not None:
            self._log(
                "Protected detail zones: "
                f"{self.current_analysis.protected_detail_faces} faces / "
                f"{self.current_analysis.protected_detail_vertices} vertices. "
                "These high-curvature or high-normal-variance areas are protected from smoothing/simplification."
            )

    def export_mesh(self) -> None:
        mesh = self._require_mesh()
        if mesh is None:
            return
        default_name = "meshmend_repaired.stl"
        if self.current_path is not None:
            default_name = f"{self.current_path.stem}_meshmend.stl"
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export repaired mesh",
            default_name,
            "STL (*.stl);;OBJ (*.obj)",
        )
        if not path:
            return
        try:
            output = export_mesh_file(mesh, path)
            self._log(f"Exported {output}")
        except Exception as exc:
            self._error("Export failed", exc)

    def _run_repair(self, label: str, operation) -> None:
        mesh = self._require_mesh()
        if mesh is None:
            return
        try:
            self._log(f"{label} started.")
            result: RepairResult = operation(mesh)
            if self.repair_preview_mode.isChecked() or self._repair_settings().preview_mode:
                self._log("  PREVIEW MODE: original mesh left unchanged.")
            else:
                self.mesh = result.mesh
                self.viewport.set_mesh(self.mesh)
            for action in result.actions:
                self._log(f"  {action}")
            for warning in result.warnings:
                self._log(f"  WARNING: {warning}")
            if result.before is not None and result.after is not None:
                self._log(
                    "  preview/stat delta: "
                    f"watertight {result.before.watertight} -> {result.after.watertight}; "
                    f"non-manifold {result.before.non_manifold_edges} -> {result.after.non_manifold_edges}; "
                    f"protected vertices {result.protected_detail_vertices}; "
                    f"local repair vertices {result.local_repair_vertices}; "
                    f"modified source vertices {result.modified_vertex_estimate}"
                )
            if not self.repair_preview_mode.isChecked():
                self.analyze_current_mesh()
            self._log(f"{label} complete.")
        except Exception as exc:
            self._error(f"{label} failed", exc)

    def _repair_settings(self) -> RepairSettings:
        return RepairSettings(
            studio_master_mode=self.studio_master_mode.isChecked(),
            max_hole_edges=int(self.max_hole_edges.value()),
            max_existing_vertex_displacement_mm=0.005 if self.studio_master_mode.isChecked() else 0.02,
            preview_mode=self.repair_preview_mode.isChecked(),
        )

    def _detail_parameters(self) -> DetailParameters:
        return DetailParameters(
            detail_strength=self.detail_strength.value() / 100.0,
            rivet_density=self.rivet_density.value() / 100.0,
            panel_line_depth=self.panel_line_depth.value() / 100.0,
            battle_damage_amount=self.battle_damage_amount.value() / 100.0,
            surface_texture_strength=self.surface_texture_strength.value() / 100.0,
            edge_sharpness=self.edge_sharpness.value() / 100.0,
            minimum_printable_detail_size=self.minimum_printable_detail_size.value() / 100.0,
        )

    def _render_analysis(self, analysis: MeshAnalysis) -> None:
        summary = analysis.to_dict().copy()
        summary.pop("issues", None)
        self.stats.setPlainText(json.dumps(summary, indent=2))
        self.problem_list.clear()
        if not analysis.issues:
            self.problem_list.addItem("No major structural problems detected.")
            return
        for issue in analysis.issues:
            item = QtWidgets.QListWidgetItem(f"[{issue.severity.upper()}] {issue.message}")
            item.setData(QtCore.Qt.UserRole, issue.kind)
            self.problem_list.addItem(item)

    def _require_mesh(self) -> trimesh.Trimesh | None:
        if self.mesh is None:
            QtWidgets.QMessageBox.information(self, "No mesh loaded", "Import a mesh first.")
            return None
        return self.mesh

    def _log(self, message: str) -> None:
        self.log.appendPlainText(message)

    def _error(self, title: str, exc: Exception) -> None:
        self._log(f"ERROR: {title}: {exc}")
        QtWidgets.QMessageBox.critical(self, title, str(exc))

    def _apply_studio_styles(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow { background: #171a20; }
            QFrame { background: #20242b; border: 1px solid #313743; border-radius: 6px; }
            QLabel, QCheckBox { color: #e7eaf0; }
            QLabel#PanelTitle { font-weight: 700; font-size: 15px; padding-bottom: 6px; }
            QLabel#HintText { color: #aeb6c4; }
            QPushButton { padding: 7px; background: #2d6cdf; color: white; border-radius: 4px; }
            QPushButton:hover { background: #3c7cf0; }
            QPlainTextEdit, QListWidget, QSpinBox { background: #11141a; color: #e7eaf0; border: 1px solid #3b4350; }
            """
        )


def _labeled_widget(label: str, widget: QtWidgets.QWidget) -> QtWidgets.QWidget:
    container = QtWidgets.QWidget()
    layout = QtWidgets.QVBoxLayout(container)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.addWidget(QtWidgets.QLabel(label))
    layout.addWidget(widget)
    return container


def _slider(value: int, minimum: int = 0, maximum: int = 100) -> QtWidgets.QSlider:
    slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
    slider.setRange(minimum, maximum)
    slider.setValue(value)
    slider.setTickPosition(QtWidgets.QSlider.TicksBelow)
    slider.setTickInterval(max(1, (maximum - minimum) // 5))
    return slider
