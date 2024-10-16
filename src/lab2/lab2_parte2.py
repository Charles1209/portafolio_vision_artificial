import cv2
import numpy as np
import matplotlib.pyplot as plt

################################################################

def show_with_matplotlib(img, title):
	"""Mostrar las imágenes usando las capacidades de MatPlotLib"""
	# Convertir imagen BGR a RGB
	img_RGB = img[:, :, ::-1]
	# Mostrar la imagen con matplotlib:
	plt.imshow(img_RGB)
	plt.title(title)
	plt.show()

image = cv2.imread("img/lena_image.png")
show_with_matplotlib(image, "Original Image")

################################################################

# El parámetro interpolación especifica como se calcularan los valores
# de los pı́xeles en la nueva imagen
# Escalar 0.5 del valor de la imagen original
dst_image = cv2.resize(image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
# Obtener las dimensiones de la imagen
height, width = image.shape[:2]
# Aplicación de un nuevo escalado basados en la imagen original
dst_image_2 = cv2.resize(image, (width * 2, height * 2), interpolation=cv2.INTER_LINEAR)
# Mostrar las imagenes escaladas
show_with_matplotlib(dst_image, 'Resized image')
show_with_matplotlib(dst_image_2, 'Resized image 2')

################################################################

# Matriz de Transformación
M = np.float32([[1, 0, 200], [0, 1, 30]])

################################################################

# Esta matriz se pasa a la función cv2.warpAffine():
dst_image = cv2.warpAffine(image, M, (width, height))
# Mostrar el resultado
show_with_matplotlib(dst_image, 'Resultado de la Traslación (valores positivos)')

################################################################

# La matriz también puede contener valores negativos
M = np.float32([[1, 0, -200], [0, 1, -30]])
dst_image = cv2.warpAffine(image, M, (width, height))
# Mostrar el resultado
show_with_matplotlib(dst_image, 'Resultado de la Traslación (valores negativos)')

################################################################

# Creación de Matriz de Rotación para 180 grados
# getRotationMatrix2D(Point2f center, double angle, double scale)
M = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), 180, 1)
dst_image = cv2.warpAffine(image, M, (width, height))
# Mostrar el centro de rotación
cv2.circle(dst_image, (round(width / 2.0), round(height / 2.0)), 5, (255, 0, 0), -1)
show_with_matplotlib(dst_image, 'Imagen rotada 180 grados')

################################################################

# En este caso cambiamos el centro de rotación
M = cv2.getRotationMatrix2D((width / 1.5, height / 1.5), 30, 1)
dst_image = cv2.warpAffine(image, M, (width, height))
# Mostrar el centro de rotación y rotar 30 grados
cv2.circle(dst_image, (round(width / 1.5), round(height / 1.5)), 5, (255, 0, 0), -1)
show_with_matplotlib(dst_image, 'Imagen rotada 30 grados')

################################################################

# En una copia de la imagen se dibujan los puntos de referencia
image_points = image.copy()
cv2.circle(image_points, (135, 45), 5, (255, 0, 255), -1)
cv2.circle(image_points, (385, 45), 5, (255, 0, 255), -1)
cv2.circle(image_points, (135, 230), 5, (255, 0, 255), -1)
# Mostrar los puntos de referencia
show_with_matplotlib(image_points, 'Antes de la transformación')
# Se crean 2 arreglos con los puntos de referencia
# y su ubicación deseada.
pts_1 = np.float32([[135, 45], [385, 45], [135, 230]])
pts_2 = np.float32([[135, 45], [385, 45], [150, 230]])
# Usando getAffineTransform se obtiene la matriz de transformación
M = cv2.getAffineTransform(pts_1, pts_2)
dst_image = cv2.warpAffine(image_points, M, (width, height))
# Mostrar el resultado:
show_with_matplotlib(dst_image, 'Tranformaciones Afines ')

################################################################

# Se crea en una copia la imagen para mostrar los puntos
# que forman parte del proceso
image_points = image.copy()
# Mostrar puntos y las lı́neas que conectan dichos puntos
cv2.circle(image_points, (230, 80), 5, (0, 0, 255), -1)
cv2.circle(image_points, (330, 80), 5, (0, 0, 255), -1)
cv2.circle(image_points, (230, 200), 5, (0, 0, 255), -1)
cv2.circle(image_points, (330, 200), 5, (0, 0, 255), -1)
cv2.line(image_points, (230, 80), (330, 80), (0, 0, 255))
cv2.line(image_points, (230, 200), (330, 200), (0, 0, 255))
cv2.line(image_points, (230, 80), (230, 200), (0, 0, 255))
cv2.line(image_points, (330, 200), (330, 80), (0, 0, 255))
# Mostrar la imagen
show_with_matplotlib(image_points, 'Antes del cropping')
# Para el recorte e utiliza la función slicing de numpy:
dst_image = image[80:200, 230:330]
# Mostrar la imagen
show_with_matplotlib(dst_image, 'Cropping de la imagen')