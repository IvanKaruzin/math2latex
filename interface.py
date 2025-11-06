import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                              QPushButton, QLabel, QTextEdit, QFileDialog)
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt

from inference import recognize_formula

class MathOCRInterface(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Math OCR")
        self.setGeometry(100, 100, 800, 600)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout()
        
        self.select_button = QPushButton("Выбрать изображение")
        self.select_button.clicked.connect(self.select_image)
        layout.addWidget(self.select_button)
        
        self.image_label = QLabel("Изображение не выбрано")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumHeight(300)
        self.image_label.setStyleSheet("border: 1px solid gray")
        layout.addWidget(self.image_label)
        
        self.latex_output = QTextEdit()
        self.latex_output.setPlaceholderText("LaTeX последовательность...")
        self.latex_output.setReadOnly(True)
        self.latex_output.setMaximumHeight(100)
        layout.addWidget(self.latex_output)
        
        central_widget.setLayout(layout)
        
    def select_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            "Выбрать изображение", 
            "", 
            "Images (*.png *.jpg *.jpeg)"
        )
        
        if file_path:
            pixmap = QPixmap(file_path)
            scaled_pixmap = pixmap.scaled(
                self.image_label.width(), 
                self.image_label.height(), 
                Qt.KeepAspectRatio, 
                Qt.SmoothTransformation
            )
            self.image_label.setPixmap(scaled_pixmap)
            
            try:
                predicted_latex = recognize_formula(file_path)
                self.latex_output.setText(predicted_latex)
            except Exception as e:
                self.latex_output.setText(f"Ошибка: {str(e)}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MathOCRInterface()
    window.show()
    sys.exit(app.exec_())
