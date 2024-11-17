import cv2
import numpy as np
import matplotlib.pyplot as plt

def show_image(img, title):
	plt.imshow(img)
	plt.title(title)
	plt.show()

#img = cv2.imread("lab2/img/image.jpg")
img = plt.imread("img/lab2.jpg")
show_image(img, "Imagen Original")

""" 1. Aplique para una imagen de su elección (diferente a la utilizada en el ejemplo), un
cropping del elemento u objeto principal de dicha imagen. """

img_cropping = img[50:500, 50:500]
show_image(img_cropping, "Imagen con Cropping")

""" 2. Utilizando el recorte anterior aplique una rotación de 167 grados utilizando como
pivote un punto desplazado 45 píxeles del centro en el eje x y para el eje y utilice el
centro del eje. """

height, width = img_cropping.shape[:2]

# Matriz de Transformación
M = np.float32([[1, 0, 45], [0, 1, 0]])

# Esta matriz se pasa a la función cv2.warpAffine():
img_translation = cv2.warpAffine(img_cropping, M, (width, height))
# Mostrar el resultado
show_image(img_translation, 'Resultado de la Traslación')

# Obtener las dimensiones de la imagen
height, width = img_translation.shape[:2]

M = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), 167, 1)
dst_image = cv2.warpAffine(img_translation, M, (width, height))
# Mostrar el centro de rotación
cv2.circle(dst_image, (round(width / 2.0), round(height / 2.0)), 5, (255, 0, 0), -1)
show_image(dst_image, 'Imagen rotada 167 grados')

""" 3. En otra copia del recorte realice una traslación de -55 píxeles en x y 89 píxeles en
el eje y. """

height, width = img_cropping.shape[:2]
M = np.float32([[1, 0, -55], [0, 1, 89]])
img_translation = cv2.warpAffine(img_cropping, M, (width, height))
show_image(img_translation, 'Resultado de la Traslación')

""" 4. En este momento debe tener 4 archivos de imagen en memoria (original, crop, rotada y
trasladada), calcule el histograma de color para cada una de estas imágenes. """

def mostrar_histograma(img):
	colors = ('b','g','r')
	for i, c in enumerate(colors):
		hist = cv2.calcHist([img], [i], None, [256], [0, 256])
		plt.plot(hist, color = c)
		plt.xlim([0,256])

	plt.xlabel('intensidad de iluminación')
	plt.ylabel('cantidad de píxeles')
	plt.show()

mostrar_histograma(img)
mostrar_histograma(img_cropping)
mostrar_histograma(dst_image)
mostrar_histograma(img_translation)