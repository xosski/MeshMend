"""
Main application window using PyQt6
"""

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSplitter, 
    QPushButton, QLabel, QLineEdit, QTextEdit, QSpinBox, QDoubleSpinBox,
    QComboBox, QProgressBar, QScrollArea, QFrame, QFileDialog, QMessageBox
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QSize
from PyQt6.QtGui import QIcon, QFont
import numpy as np
from pathlib import Path

from app.viewer_3d import Viewer3D
from app.mesh_utils import MeshManager
from app.mesh_analyzer import MeshAnalyzer

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("3D Sculptor - Warhammer Miniature Generator")
        self.setGeometry(100, 100, 1600, 900)
        
        # Initialize managers
        self.mesh_manager = MeshManager()
        self.current_mesh = None
        self.ai_worker = None
        self.worker_thread = None
        self.source_image_path = None
        
        # Create UI
        self.init_ui()
        
    def init_ui(self):
        """Initialize the user interface"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout(central_widget)
        
        # Left panel: 3D Viewer
        self.viewer_3d = Viewer3D()
        main_layout.addWidget(self.viewer_3d, 2)
        
        # Right panel: Controls
        right_panel = self.create_control_panel()
        main_layout.addWidget(right_panel, 1)
        
        # Set splitter for resizing
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self.viewer_3d)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 1)
        
        main_layout.addWidget(splitter)
        
    def create_control_panel(self):
        """Create the right-side control panel"""
        panel = QFrame()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        
        # Title
        title = QLabel("Generate Miniature")
        title_font = QFont()
        title_font.setPointSize(14)
        title_font.setBold(True)
        title.setFont(title_font)
        layout.addWidget(title)
        
        # Description input
        layout.addWidget(QLabel("Description:"))
        self.prompt_input = QTextEdit()
        self.prompt_input.setPlaceholderText(
            "E.g., 'Space marine warrior with red armor and bolter weapon'"
        )
        self.prompt_input.setMaximumHeight(80)
        layout.addWidget(self.prompt_input)

        self.source_image_label = QLabel("Source image: None")
        self.source_image_label.setStyleSheet("color: #666;")
        layout.addWidget(self.source_image_label)

        source_image_btn = QPushButton("Import Source Image")
        source_image_btn.clicked.connect(self.import_source_image)
        layout.addWidget(source_image_btn)
        
        # Style selection
        layout.addWidget(QLabel("Miniature Style:"))
        self.style_combo = QComboBox()
        self.style_combo.addItems([
            "Sci-Fi Soldier",
            "Fantasy Knight",
            "Chaos Warrior",
            "Elven Archer",
            "Dwarf Engineer",
            "Ork Brute",
            "Undead Warrior",
            "Generic Humanoid"
        ])
        layout.addWidget(self.style_combo)
        
        # Detail level
        layout.addWidget(QLabel("Detail Level:"))
        self.detail_spin = QSpinBox()
        self.detail_spin.setMinimum(1)
        self.detail_spin.setMaximum(5)
        self.detail_spin.setValue(4)
        layout.addWidget(self.detail_spin)
        
        # Scale
        layout.addWidget(QLabel("Scale (mm):"))
        self.scale_combo = QComboBox()
        self.scale_combo.setEditable(True)
        self.scale_combo.addItems(["15", "20", "25", "28", "32", "35", "40", "48", "54", "75"])
        self.scale_combo.setCurrentText("32")
        layout.addWidget(self.scale_combo)
        
        # Generate button
        self.generate_btn = QPushButton("Generate Model")
        self.generate_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                padding: 8px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        self.generate_btn.clicked.connect(self.generate_model)
        layout.addWidget(self.generate_btn)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        # Status label
        self.status_label = QLabel("Ready")
        layout.addWidget(self.status_label)
        
        # Separator
        layout.addSpacing(10)
        
        # Model controls
        controls_title = QLabel("Model Controls")
        controls_title.setFont(title_font)
        layout.addWidget(controls_title)
        
        # Rotation controls
        layout.addWidget(QLabel("Rotation:"))
        rot_layout = QHBoxLayout()
        
        rot_x_btn = QPushButton("↻ X")
        rot_y_btn = QPushButton("↻ Y")
        rot_z_btn = QPushButton("↻ Z")
        
        rot_x_btn.clicked.connect(lambda: self.viewer_3d.rotate_model(10, 0, 0))
        rot_y_btn.clicked.connect(lambda: self.viewer_3d.rotate_model(0, 10, 0))
        rot_z_btn.clicked.connect(lambda: self.viewer_3d.rotate_model(0, 0, 10))
        
        rot_layout.addWidget(rot_x_btn)
        rot_layout.addWidget(rot_y_btn)
        rot_layout.addWidget(rot_z_btn)
        layout.addLayout(rot_layout)
        
        # Scale controls
        layout.addWidget(QLabel("Model Scale:"))
        scale_layout = QHBoxLayout()
        
        scale_up = QPushButton("+ Zoom")
        scale_down = QPushButton("- Zoom")
        reset_view = QPushButton("Reset View")
        
        scale_up.clicked.connect(lambda: self.viewer_3d.scale_model(1.1))
        scale_down.clicked.connect(lambda: self.viewer_3d.scale_model(0.9))
        reset_view.clicked.connect(self.viewer_3d.reset_view)
        
        scale_layout.addWidget(scale_up)
        scale_layout.addWidget(scale_down)
        layout.addLayout(scale_layout)
        layout.addWidget(reset_view)
        
        # File operations
        layout.addSpacing(10)
        file_title = QLabel("File Operations")
        file_title.setFont(title_font)
        layout.addWidget(file_title)
        
        export_btn = QPushButton("Export as STL")
        export_btn.clicked.connect(self.export_model)
        layout.addWidget(export_btn)
        
        import_btn = QPushButton("Import Model")
        import_btn.clicked.connect(self.import_model)
        layout.addWidget(import_btn)
        
        # Stretch to fill remaining space
        layout.addStretch()
        
        return panel
    
    def generate_model(self):
        """Generate a new model using AI"""
        prompt = self.prompt_input.toPlainText().strip()
        if not prompt and not self.source_image_path:
            QMessageBox.warning(self, "Error", "Please enter a description or import a source image")
            return
        
        # Lazy import to keep app startup fast; AI stack imports are expensive.
        try:
            from app.ai_generator import AIGeneratorWorker
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load AI generator: {e}")
            return

        try:
            scale_mm = float(self.scale_combo.currentText().strip().lower().removesuffix("mm"))
            if scale_mm < 10 or scale_mm > 100:
                raise ValueError("Scale must be between 10mm and 100mm")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Invalid miniature scale: {e}")
            return

        self.generate_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.status_label.setText("Generating model...")
        
        # Create worker thread
        self.worker_thread = QThread()
        self.ai_worker = AIGeneratorWorker(
            prompt=prompt,
            style=self.style_combo.currentText(),
            detail_level=self.detail_spin.value(),
            scale=scale_mm,
            source_image_path=self.source_image_path,
        )
        self.ai_worker.moveToThread(self.worker_thread)
        
        # Connect signals
        self.ai_worker.progress.connect(self.update_progress)
        self.ai_worker.progress_message.connect(self.update_progress_message)
        self.ai_worker.finished.connect(self.on_generation_complete)
        self.ai_worker.error.connect(self.on_generation_error)
        self.worker_thread.started.connect(self.ai_worker.run)
        
        self.worker_thread.start()
    
    def update_progress(self, value):
        """Update progress bar"""
        self.progress_bar.setValue(value)

    def update_progress_message(self, value, message):
        """Update progress bar and status text."""
        self.progress_bar.setValue(value)
        self.status_label.setText(f"{value}% - {message}")
    
    def on_generation_complete(self, mesh_data):
        """Handle completed generation"""
        self.worker_thread.quit()
        self.worker_thread.wait()
        
        self.current_mesh = mesh_data
        
        # Analyze mesh quality
        analyzer = MeshAnalyzer()
        quality = analyzer.analyze(mesh_data)
        
        # Display mesh
        self.viewer_3d.load_mesh(mesh_data)
        
        self.progress_bar.setVisible(False)
        self.generate_btn.setEnabled(True)
        
        # Show quality report
        if quality:
            report = quality.report()
            self.status_label.setText(f"✓ Generated | {quality.vertex_count} verts, {quality.face_count} faces")
            print("\n" + report)
            
            if not quality.is_valid_for_printing():
                QMessageBox.warning(
                    self,
                    "Quality Warning",
                    "Generated mesh may have quality issues.\n\n"
                    "The 3D shape may be incomplete or too detailed.\n"
                    "Try adjusting the description or detail level."
                )
            else:
                QMessageBox.information(
                    self,
                    "Success",
                    f"Miniature generated successfully!\n\n{report}"
                )
        else:
            self.status_label.setText("Model generated")
            QMessageBox.information(self, "Success", "Miniature generated successfully!")
    
    def on_generation_error(self, error):
        """Handle generation error"""
        self.worker_thread.quit()
        self.worker_thread.wait()
        
        self.progress_bar.setVisible(False)
        self.generate_btn.setEnabled(True)
        self.status_label.setText("Generation failed")
        
        QMessageBox.critical(self, "Error", f"Generation failed: {error}")
    
    def export_model(self):
        """Export current model as STL"""
        if self.current_mesh is None:
            QMessageBox.warning(self, "Error", "No model to export")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Model",
            "",
            "STL Files (*.stl);;OBJ Files (*.obj);;PLY Files (*.ply)"
        )
        
        if file_path:
            try:
                self.mesh_manager.save_mesh(self.current_mesh, file_path)
                QMessageBox.information(self, "Success", f"Model exported to {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Export failed: {str(e)}")
    
    def import_model(self):
        """Import a model from file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Import Model",
            "",
            "3D Files (*.stl *.obj *.ply);;STL Files (*.stl);;OBJ Files (*.obj);;PLY Files (*.ply)"
        )
        
        if file_path:
            try:
                mesh = self.mesh_manager.load_mesh(file_path)
                self.current_mesh = mesh
                self.viewer_3d.load_mesh(mesh)
                self.status_label.setText(f"Loaded: {Path(file_path).name}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Import failed: {str(e)}")

    def import_source_image(self):
        """Import a source image used for image-to-3D generation"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Import Source Image",
            "",
            "Image Files (*.png *.jpg *.jpeg *.bmp *.webp)"
        )

        if file_path:
            self.source_image_path = file_path
            self.source_image_label.setText(f"Source image: {Path(file_path).name}")
