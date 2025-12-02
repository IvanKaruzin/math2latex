import sys
from io import BytesIO

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                              QPushButton, QLabel, QTextEdit, QFileDialog, QMessageBox)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt

import matplotlib

matplotlib.use("Agg")
from matplotlib import rcParams

rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "DejaVu Serif"],
})

import matplotlib.pyplot as plt

from inference import recognize_formula

class MathOCRInterface(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("math2latex")
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
        self.latex_output.setMaximumHeight(100)
        self.latex_output.textChanged.connect(self.on_latex_changed)
        layout.addWidget(self.latex_output)

        self.copy_button = QPushButton("Сохранить в буфер обмена")
        self.copy_button.clicked.connect(self.copy_to_clipboard)
        layout.addWidget(self.copy_button)

        self.show_sequence_button = QPushButton("Отобразить последовательность")
        self.show_sequence_button.setEnabled(False)
        self.show_sequence_button.clicked.connect(self.show_predicted_sequence)
        layout.addWidget(self.show_sequence_button)

        self.preview_label = QLabel("Предпросмотр формулы появится здесь")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setMinimumHeight(150)
        self.preview_label.setStyleSheet("border: 1px dashed gray")
        layout.addWidget(self.preview_label)
        
        central_widget.setLayout(layout)

        self.last_rendered_sequence = ""
        
    def select_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            "Выбрать изображение", 
            "", 
            "Images (*.png *.jpg *.jpeg)"
        )
        
        if file_path:
            pixmap = QPixmap(file_path)
            label_width = max(self.image_label.width(), 600)
            label_height = max(self.image_label.height(), 250)
            scaled_pixmap = pixmap.scaled(
                label_width, 
                label_height, 
                Qt.KeepAspectRatio, 
                Qt.SmoothTransformation
            )
            self.image_label.setPixmap(scaled_pixmap)
            
            try:
                predicted_latex = recognize_formula(file_path)
                self.set_latex_sequence(predicted_latex)
            except Exception as e:
                self.latex_output.setText(f"Ошибка: {str(e)}")

    def set_latex_sequence(self, sequence: str):
        self.latex_output.blockSignals(True)
        self.latex_output.setText(sequence)
        self.latex_output.blockSignals(False)
        self.update_show_button_state()

    def on_latex_changed(self):
        self.update_show_button_state()

    def copy_to_clipboard(self):
        latex_text = self.latex_output.toPlainText().strip()
        if latex_text:
            clipboard = QApplication.clipboard()
            clipboard.setText(latex_text)
            QMessageBox.information(self, "Успешно", "LaTeX последовательность скопирована в буфер обмена.")
        else:
            QMessageBox.warning(self, "Пустая последовательность", "Нет текста для копирования.")

    def update_show_button_state(self):
        current_text = self.latex_output.toPlainText().strip()
        has_text = bool(current_text)
        changed_since_render = current_text != self.last_rendered_sequence
        self.show_sequence_button.setEnabled(has_text and changed_since_render)

    def show_predicted_sequence(self):
        latex_sequence = self.latex_output.toPlainText().strip()
        if not latex_sequence:
            QMessageBox.warning(self, "Пустая последовательность", "Введите LaTeX последовательность.")
            return

        pixmap = self.render_latex_to_pixmap(latex_sequence)
        if pixmap:
            target_width = max(self.preview_label.width(), 600)
            target_height = max(self.preview_label.height(), 250)
            scaled_pixmap = pixmap.scaled(
                target_width,
                target_height,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.preview_label.setPixmap(scaled_pixmap)
            self.last_rendered_sequence = latex_sequence
            self.update_show_button_state()
        else:
            self.preview_label.setText("Не удалось отобразить последовательность.")

    def render_latex_to_pixmap(self, latex_sequence: str) -> QPixmap:
        try:
            figure = plt.figure(figsize=(12, 5), dpi=200)
            figure.patch.set_alpha(0.0)
            ax = figure.add_subplot(111)
            ax.axis("off")
            ax.text(
                0.5,
                0.5,
                rf"$\displaystyle {latex_sequence}$",
                fontsize=60,
                ha="center",
                va="center"
            )
            

            buffer = BytesIO()
            plt.savefig(
                buffer,
                format="png",
                transparent=True,
                bbox_inches="tight",
                pad_inches=0.3
            )
            plt.close(figure)
            buffer.seek(0)

            image = QImage()
            if image.loadFromData(buffer.read(), "PNG"):
                return QPixmap.fromImage(image)
        except Exception as exc:
            self.preview_label.setText(f"Ошибка компиляции LaTeX: {exc}")
        return None

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MathOCRInterface()
    window.show()
    sys.exit(app.exec_())
