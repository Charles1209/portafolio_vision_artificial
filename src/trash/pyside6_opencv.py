import sys
import cv2
import numpy as np
from PySide6.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget, QPushButton
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QImage, QPixmap

class CameraApp(QWidget):
    def __init__(self):
        super().__init__()

        # Inicializar el widget principal
        self.setWindowTitle("Cámara OpenCV con PySide6")
        self.setGeometry(100, 100, 800, 600)

        # Crear un QLabel para mostrar el video
        self.video_label = QLabel(self)
        self.video_label.setAlignment(Qt.AlignCenter)

        # Botón para salir de la aplicación
        self.quit_button = QPushButton("Salir", self)
        self.quit_button.clicked.connect(self.close)

        # Layout vertical
        layout = QVBoxLayout()
        layout.addWidget(self.video_label)
        layout.addWidget(self.quit_button)
        self.setLayout(layout)

        # Inicializar la cámara de OpenCV
        self.cap = cv2.VideoCapture(0)  # 0 es el ID de la cámara, si tienes más cámaras puedes cambiar el número.

        # Crear un temporizador para actualizar la imagen
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # Actualizar cada 30 ms (~33 FPS)

    def update_frame(self):
        # Leer un frame de la cámara
        ret, frame = self.cap.read()

        if ret:
            # Convertir la imagen de BGR (OpenCV) a RGB (para Qt)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Convertir la imagen a formato QImage
            height, width, channel = frame.shape
            bytes_per_line = 3 * width
            q_img = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)

            # Convertir QImage a QPixmap y mostrarlo en QLabel
            self.video_label.setPixmap(QPixmap.fromImage(q_img))

    def closeEvent(self, event):
        # Cerrar la cámara y liberar recursos cuando se cierre la ventana
        self.cap.release()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Crear y mostrar la ventana principal
    window = CameraApp()
    window.show()

    # Ejecutar la aplicación
    sys.exit(app.exec())