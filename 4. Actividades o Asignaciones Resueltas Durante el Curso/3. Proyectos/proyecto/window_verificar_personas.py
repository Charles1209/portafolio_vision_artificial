import sys
import cv2
import numpy as np
from PySide6.QtWidgets import QMainWindow, QApplication, QLabel, QVBoxLayout, QWidget, QPushButton
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QImage, QPixmap

class window_verificar_personas(QMainWindow):
	def __init__(self):
		super().__init__()

		self.layout_verificacion = QVBoxLayout()

		#self.setGeometry(100, 100, 800, 600)

		# Crear un QLabel para mostrar el video
		self.label_video = QLabel(self)
		self.label_video.setAlignment(Qt.AlignCenter)
		self.boton_regresar = QPushButton("Regresar")

		list_layout_verificar_personas = [
			QLabel(),
			self.label_video,
			QLabel(),
			self.boton_regresar,
			QLabel()
		]

		for i in list_layout_verificar_personas:
			self.layout_verificacion.addWidget(i)

		self.boton_regresar.clicked.connect(self.closeEvent)

		self.widget = QWidget()
		self.widget.setLayout(self.layout_verificacion)
		self.setCentralWidget(self.widget)

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
			self.label_video.setPixmap(QPixmap.fromImage(q_img))

	def closeEvent(self, event):
		# Cerrar la cámara y liberar recursos cuando se cierre la ventana
		self.cap.release()
		event.accept()
		self.close

if __name__ == "__main__":
	app = QApplication(sys.argv)

	# Crear y mostrar la ventana principal
	window = window_verificar_personas()
	window.show()

	# Ejecutar la aplicación
	sys.exit(app.exec())