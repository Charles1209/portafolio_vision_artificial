from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
	QMainWindow,
	QVBoxLayout,
	QHBoxLayout,
	QWidget,
	QPushButton,
	QApplication,
	QLabel
)

class window_menu (QMainWindow):
	def __init__(self):
		super().__init__()

		self.layout_menu = QVBoxLayout()

		self.label_name = QLabel("Sistema de Verificación de Personas de la UTP")
		self.label_name.setAlignment(Qt.AlignCenter)
		self.boton_verificar = QPushButton("Verificar Personas")
		self.boton_insertar = QPushButton("Insertar Persona")
		self.boton_ver_identificados = QPushButton("Ver Personas Identificadas")
		self.boton_salir = QPushButton("Salir")

		list_layout_menu = [
			QLabel(),
			self.label_name,
			QLabel(),
			self.boton_verificar,
			QLabel(),
			self.boton_insertar,
			QLabel(),
			self.boton_ver_identificados,
			QLabel(),
			self.boton_salir,
			QLabel()
		]

		for w in list_layout_menu:
			self.layout_menu.addWidget(w)

		self.widget = QWidget()
		self.widget.setLayout(self.layout_menu)
		self.setCentralWidget(self.widget)

if __name__ == "__main__":
	app = QApplication()
	window = window_menu()
	window.show()
	app.exec()