"""
3D Sculptor - AI-Powered Warhammer-Adjacent Miniature Generator
Main PyQt6 Application Entry Point
"""

import sys
from pathlib import Path
from PyQt6.QtWidgets import QApplication
from app.main_window import MainWindow

def main():
    app = QApplication(sys.argv)
    app.setApplicationName("3D Sculptor")
    app.setApplicationVersion("1.0.0")
    
    # Create main window
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
