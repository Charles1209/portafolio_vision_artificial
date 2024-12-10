import cv2
import matplotlib.pyplot as plt
import numpy as np

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
from funciones import *

img = abrir_imagen("img/lab4.webp")

""" 1. Seleccione una imagen y aplique los detectores de Harris y Shi-Tomasi. Aplique a cada
imagen, cada detector. """

def calcular_harris(img_harris, blockSize=2, ksize=3, k=0.08):
	gray = cv2.cvtColor(img_harris, cv2.COLOR_BGR2GRAY)
	gray = np.float32(gray)

	dst = cv2.cornerHarris(gray, blockSize, ksize, k)
	dst = cv2.dilate(dst, None)

	img_harris[dst>0.01*dst.max()]=[0,0,255]

	return img_harris

def calcular_shi_tomasi(img_shi, cantidad=65, qlt=0.08, euDist=10, radio=8, color=(0,255,0), thickness=2):
	gray = cv2.cvtColor(img_shi, cv2.COLOR_BGR2GRAY)

	corners = cv2.goodFeaturesToTrack(gray, cantidad, qlt, euDist)
	corners = np.intp(corners)

	for i in corners:
		x,y = i.ravel()
		cv2.circle(img_shi, (x,y), radio, color, thickness)

	return img_shi

img_harris = img.copy() # se agrega el .copy() porque sino sale error de destinatation is read-only
						# y también que si no sale el error los cambios se guardan en la imagen original
img_harris = calcular_harris(img_harris)
#mostrar_imagen(img_harris, "")

img_shi = img.copy()
img_shi = calcular_shi_tomasi(img_shi)
#mostrar_imagen(img_shi, "")

""" 2. Aplique un filtro bilateral a cada imagen original. Posteriormente, aplique los detectores
a la imagen filtrada, grafique los resultados. """

img_harris_bilateral = img.copy()
img_harris_bilateral = cv2.bilateralFilter(img_harris_bilateral, d=9, sigmaColor=75, sigmaSpace=75)
img_harris_bilateral = calcular_harris(img_harris_bilateral)
#mostrar_imagen(img_harris_bilateral, "")

img_shi_bilateral = img.copy()
img_shi_bilateral = cv2.bilateralFilter(img_shi_bilateral, d=9, sigmaColor=75, sigmaSpace=75)
img_shi_bilateral = calcular_shi_tomasi(img_shi_bilateral)
#mostrar_imagen(img_shi_bilateral, "")

""" 3. Presente los resultados a modo comparativo de la detección antes y después de aplicar
el filtro bilateral. """

comparar_imagenes(img_harris, "Harris Normal", img_harris_bilateral, "Harris Bilateral")
comparar_imagenes(img_shi, "Shi-Tomasi Normal", img_shi_bilateral, "Shi-Tomasi Bilateral")