# No sirve

import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.animation import FuncAnimation
from PySide6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget

# Clase principal de la ventana de PySide6
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Cámara Matplotlib en PySide6")

        # Configuración del layout principal
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Crear una figura de Matplotlib y un canvas (que será embebido en PySide6)
        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        # Datos para animación
        self.x_data = np.linspace(0, 2 * np.pi, 100)
        self.y_data = np.sin(self.x_data)
        self.line, = self.ax.plot([], [], lw=2)

        # Iniciar la animación usando FuncAnimation
        self.anim = FuncAnimation(self.figure, self.update_plot, frames=len(self.x_data),
                                  init_func=self.init_plot, blit=True, interval=100)

    def init_plot(self):
        """Inicializa los límites y los datos de la gráfica."""
        self.ax.set_xlim(0, 2 * np.pi)
        self.ax.set_ylim(-1, 1)
        self.line.set_data([], [])
        return self.line,

    def update_plot(self, frame):
        """Actualiza los datos del gráfico en cada frame."""
        self.line.set_data(self.x_data[:frame], self.y_data[:frame])
        return self.line,

# Código principal para ejecutar la aplicación
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())