from PySide6 import QtCore
from PySide6.QtWidgets import (
	QApplication,
	QMainWindow,
	QStackedLayout,
	QFrame,
	QVBoxLayout,
	QLabel,
	QPushButton,
	QWidget,
	QTableWidgetItem,
	QTableWidget,
	QCompleter
)
import proyecto
from window_menu import window_menu
#from window_verificar_personas import

class window_main(QMainWindow):
	def __init__(self):
		super().__init__()

		self.setWindowTitle("Sistema de Verificación de Personas de la UTP")

		self.layout_main = QStackedLayout()

		self.layout_menu = window_menu()

		list_widgets = [
			self.layout_menu
		]

		for i in list_widgets:
			self.layout_main.addWidget(i)
		
		#self.layout_menu.boton_ver_identificados.clicked.connect(proyecto.ver_identificados(proyecto.path_cropped_faces))
		self.layout_menu.boton_ver_identificados.clicked.connect(self.ver_identificados)
		self.layout_menu.boton_salir.clicked.connect(self.close)
		
		self.widget_main = QWidget()
		self.widget_main.setLayout(self.layout_main)
		self.setCentralWidget(self.widget_main)

	####################################
	####	Cambios de Ventana		####
	####################################

	# lasdfasdfasdfasdñfljadslfj

	## Aquí va otra cosa
	def ver_identificados(self):
		proyecto.ver_identificados(proyecto.path_cropped_faces)

app = QApplication()
window = window_main()
window.show()
app.exec()