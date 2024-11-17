import cv2
import matplotlib.pyplot as plt
import numpy as np

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
from funciones import *

img = abrir_imagen("img/lab4.webp")

""" 1. El código mostrado en la Sección 2.1 muestra solo una línea detectada en la imagen,
realice la modificación necesaria en este para que se muestren todas líneas detectadas.
Adicional pinte cada una de las líneas de un color elegido aleatoriamente entre 10
posibles colores. Para esto debe utilizar la imagen sudo.PNG. """

def detectar_lineas(img):
	gray1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	edges1 = cv2.Canny(gray1, 50, 150, apertureSize=3)

	colors = [
		(255, 0, 0), (0, 255, 0), (0, 0, 255), 
		(255, 255, 0), (255, 0, 255), (0, 255, 255), 
		(128, 0, 0), (0, 128, 0), (0, 0, 128), 
		(128, 128, 0)
	]

	lines1 = cv2.HoughLines(edges1, 1, np.pi/180, 200)
	if lines1 is not None:
		for line in lines1:
			rho, theta = line[0]
			a = np.cos(theta)
			b = np.sin(theta)
			x0 = a * rho
			y0 = b * rho
			x1 = int(x0 + 1000 * (-b))
			y1 = int(y0 + 1000 * (a))
			x2 = int(x0 - 1000 * (-b))
			y2 = int(y0 - 1000 * (a))

			color = colors[np.random.choice(len(colors))]

			cv2.line(img, (x1, y1), (x2, y2), color, 2)

	return img

img_dl = abrir_imagen("img/sudo.PNG")
img_dl = detectar_lineas(img_dl)
mostrar_imagen(img_dl, "")

""" Utilizando el archivo monedas.jpg aplique el algoritmo de Hough para detectar la
mayor cantidad de círculos en la imagen. Pinte cada círculo detectado con un contorno
azul e indique su centro en color amarillo. Recorte cada círculo detectado en la
imagen y muestre el resultado en una matriz, utilizando subplots de matplotlib, se
recomienda utilizar un número de columnas predefinido y un número de filas dinámico. """

# Ver codigo de bryan