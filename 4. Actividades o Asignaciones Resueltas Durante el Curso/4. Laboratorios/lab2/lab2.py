import cv2
import matplotlib.pyplot as plt

image = cv2.imread("img/lab2.jpg")

################################################################

# Arreglo para guı́a de los colores
colors = ('b','g','r')
for i, c in enumerate(colors):
	hist = cv2.calcHist([image], [i], None, [256], [0, 256])
	plt.plot(hist, color = c)
	plt.xlim([0,256])

################################################################

# Configuración de los Ejes
plt.xlabel('intensidad de iluminación')
plt.ylabel('cantidad de píxeles')
# Guardar el plot en formato pdf utilizando MatPlotLib
plt.savefig("src/lab2/out/TowerHistograma.pdf", dpi=600, bbox_inches='tight')
# Mostrar el Histograma
plt.show()

################################################################
