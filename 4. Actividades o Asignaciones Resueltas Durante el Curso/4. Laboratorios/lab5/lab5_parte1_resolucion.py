""" Para esta sección en su informe debe utilizar imágenes de su propiedad. En estas se deben
realizar las siguientes operaciones:

Pruebe el algoritmo de emparejamiento de características utilizando un descriptor
diferente (Brisk o Kaze). Utilice 2 imágenes diferentes en las cuales este presente una
instancia del mismo objeto y muestre las correspondencias encontradas. Guarde la
imagen resultante de la ejecución de este código y agréguela a su informe. """

import sys
# import os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
# from funciones import *

import cv2
import matplotlib.pyplot as plt
import numpy as np

def abrir_imagen(ruta):
	img = cv2.imread(ruta) # Se abre con cv2 para que los puntos no salgan con los colores invertidos

	if img is None:
		sys.exit('Fallo al cargar la imagen')

	return img

img1 = abrir_imagen("img/lab5_1.jpg")
img2 = abrir_imagen("img/lab5_2.jpg")

# Detector Brisk

img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
detectorBrisk = cv2.BRISK_create()

keyPoints1, descriptor1 = detectorBrisk.detectAndCompute(img1_gray, None)
keyPoints2, descriptor2 = detectorBrisk.detectAndCompute(img2_gray, None)

# Comparación

bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

bf_matches = bf_matcher.match(descriptor1, descriptor2)

bf_matches = sorted(bf_matches, key=lambda x: x.distance)

result = cv2.drawMatches(img1, keyPoints1, img2, keyPoints2, bf_matches[:20], None, matchColor=(255, 255, 0), singlePointColor=(255, 0, 255), flags=0)

plt.imshow(result[:, :, ::-1])
plt.title("Descriptor BRISK")
plt.axis('off')
plt.savefig("src/lab5/out/lab5_parte1_resultado.png")
plt.show()