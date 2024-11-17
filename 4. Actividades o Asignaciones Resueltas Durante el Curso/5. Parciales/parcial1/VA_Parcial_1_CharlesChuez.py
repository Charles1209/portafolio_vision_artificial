# Charles Chuez 8-960-2188 charles.chuez@utp.ac.pa 1IL141

# mostrar los resultados utilizando MatPlotLib.

import cv2
import matplotlib.pyplot as plt
import numpy as np

def mostrar_imagen_plt(img, title):
	plt.imshow(img)
	plt.title(title)
	plt.axis('off')
	plt.show()

def mostrar_imagen_cv2(img, title):
	cv2.imshow(title, img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

def poner_texto(imagen, texto):
	cv2.putText (
		img=imagen,
		text=texto,
		org=(15,100),
		fontFace=3,
		fontScale=1,
		color=(0,0,255),
		thickness=5
	)

def calcular_histograma(imagen):
	# Obtener el alto y el ancho de la imagen de entrada
	alto, ancho = imagen.shape[:2]

	# Calcular histogramas
	histogramas = []
	for i in range(3):
		hist = cv2.calcHist([imagen], [i], None, [256], [0, 256])
		histogramas.append(hist)

	# Normalizar los histogramas para ajustarse a la altura de la imagen
	for i in range(3):
		cv2.normalize(histogramas[i], histogramas[i], 0, alto, cv2.NORM_MINMAX)

	# Crear una imagen blanca para el histograma del mismo tamaño que la imagen de entrada
	hist_img = np.ones((alto, ancho, 3), dtype=np.uint8) * 255

	# Dibujar los histogramas
	colores = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]  # Rojo, Verde, Azul
	bin_width = int(ancho / 256)  # Ajustar el ancho de cada bin al ancho de la imagen

	for i, color in enumerate(colores):
		for j in range(255):
			# Extraer valor escalar de histogramas[i][j] accediendo al primer elemento
			pt1 = (j * bin_width, alto - int(histogramas[i][j][0]))  # El valor ya está escalado entre 0 y el alto
			pt2 = ((j + 1) * bin_width, alto - int(histogramas[i][j + 1][0]))
			cv2.line(hist_img, pt1, pt2, color, 1, cv2.LINE_AA)

	return hist_img

# def calcular_histograma(imagen):
# 	# Obtener el alto y el ancho de la imagen de entrada
# 	alto, ancho = imagen.shape[:2]
	
# 	# Crear una imagen blanca para el histograma del mismo tamaño que la imagen de entrada
# 	#hist_img = np.ones((alto, ancho, 3), dtype=np.uint8) * 255

# 	# Crear una figura y un eje en matplotlib
# 	fig, ax = plt.subplots(figsize=(ancho / 100, alto / 100), dpi=100)
	
# 	# Dibujar histogramas para cada canal de color
# 	colores = ['r', 'g', 'b']
# 	# for i, color in enumerate(colores):
# 	# 	hist, _ = np.histogram(imagen[:, :, i].ravel(), bins=256, range=[0, 256])
# 	# 	ax.plot(hist, color=color, lw=2)
# 	for i, c in enumerate(colores):
# 		hist = cv2.calcHist([imagen], [i], None, [256], [0, 256])
# 		plt.plot(hist, color = c)
# 		plt.xlim([0,256])
	
# 	# Configurar los límites y eliminar ejes
# 	ax.set_xlim([0, 255])
# 	ax.set_ylim([0, alto])
# 	#ax.axis('off')
	
# 	# Convertir la figura a una imagen de NumPy
# 	fig.canvas.draw()
# 	hist_data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
# 	hist_data = hist_data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

# 	# Redimensionar la imagen para que coincida con el tamaño de la imagen original
# 	hist_img = cv2.resize(hist_data, (ancho, alto))

# 	# Cerrar la figura para liberar memoria
# 	plt.close(fig)

# 	return hist_img

# Modificar tamaño de la imagen
dsize = (int(640), int(480))

################################
########	Problema 1	########
################################

# Abrir imagen con OpenCV
""" img = cv2.imread("src/parcial1/img/Problema1.jpg")
poner_texto(img)
cv2.imshow("Imagen", img)
cv2.waitKey(0)
cv2.destroyAllWindows() """

# Abrir imagen con MatPlotLib
img = plt.imread("img/parcial1.jpg")
#mostrar_imagen_plt(img, "Original")

# Ajustar la imagen
img_resized = cv2.resize(img, dsize)
img_resized2 = cv2.resize(img, dsize)

# Aplicar filtro Median Blur
img_blr = cv2.medianBlur(img_resized, 15)
#mostrar_imagen_plt(img_blr, "Median Blur")

# Aplicar filtro bilateral
img_bilateral = cv2.bilateralFilter(img_resized, 27, 75, 75)
#mostrar_imagen_plt(img_bilateral, "Bilateral")

# Aplicar filtro gaussiano
img_gaussian = cv2.GaussianBlur(img_resized, (29, 29), 0)
#mostrar_imagen_plt(img_gaussian, "Gaussiano")

# Añadiendo correo electronico
b, g, r = cv2.split(img_resized)
img_resized = cv2.merge([r, g, b])
cv2.putText (
	img=img_resized,
	text="charles.chuez@utp.ac.pa",
	org=(15,400),
	fontFace=3,
	fontScale=1.2,
	color=(0,0,255),
	thickness=3
)
r, g, b = cv2.split(img_resized)
img_resized = cv2.merge([b, g, r])
#mostrar_imagen_plt(img_resized, "Redimencionado")

# Juntando impagenes
parte1 = np.concatenate((img_bilateral, img_blr), 1)
parte2 = np.concatenate((img_resized, img_gaussian), 1)
img_concatenado = np.concatenate((parte1, parte2), 0)
img_concatenado = cv2.resize(img_concatenado, dsize)

# Mostrando cuadrantes concatenados
#mostrar_imagen_plt(img_concatenado, "")

# Histograma de la imagen original
alto = 480
ancho = 640
histo_original = calcular_histograma(img_resized2)
#histo_original = cv2.resize(histo_original, dsize)
histo_concat = calcular_histograma(img_concatenado)
#histo_concat = cv2.resize(histo_concat, dsize)

#mostrar_imagen_plt(histo_original, "")
#mostrar_imagen_cv2(histo_original, "")
#mostrar_imagen_plt(histo_concat, "")
#mostrar_imagen_cv2(histo_concat, "")

# Mostrando Imágenes
plt.subplot(221)
plt.imshow(img_resized2)
plt.axis("off")
plt.subplot(222)
plt.imshow(histo_original)
plt.axis("off")
plt.subplot(223)
plt.imshow(img_concatenado)
plt.axis("off")
plt.subplot(224)
plt.imshow(histo_concat)
plt.axis("off")
plt.savefig("src/parcial1/out/Charles_Chuez_P1.png", dpi=600, bbox_inches='tight')
plt.show()

################################
########	Problema 2	########
################################

def rotate_image(image, angle):
	(h, w) = image.shape[:2]
	center = (w // 2, h // 2)
	M = cv2.getRotationMatrix2D(center, angle, 1.0)
	rotated = cv2.warpAffine(image, M, (w, h))
	return rotated


# Captura desde la cámara web (0 es el índice de la primera cámara conectada)
cap = cv2.VideoCapture(0)

while True:
	ret, frame = cap.read()

	frame = cv2.resize(frame, (640, 480))

	angles = [45, 135, 225, 315]
	rotated_images = []
	for angle in angles:
		rotated_img = rotate_image(frame, angle)
		cv2.putText(rotated_img, f'{angle} degrees', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
		rotated_images.append(rotated_img)

	top_row = np.hstack(rotated_images[:2])
	bottom_row = np.hstack(rotated_images[2:])
	composition = np.vstack([top_row, bottom_row])
	
	cv2.imshow('Composition', composition)

	key = cv2.waitKey(1) & 0xFF
	
	# Salir del bucle si se presiona 'q'
	if key == ord('q'):
		break
	
	# Guardar la imagen si se presiona 's'
	if key == ord('s'):
		# Convertir la imagen concatenada a formato RGB para guardar
		cv2.imwrite("src/parcial1/out/Charles_Chuez_P2.png", composition)   
	
cap.release()
cv2.destroyAllWindows()

################################
########	Problema 3	########
################################

# Captura de video desde la cámara
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
	ret, frame = cap.read()

	# Redimensiona el video capturado a 640x480
	#frame = cv2.resize(frame, dsize)

	# Muestra el histograma
	hist_img = calcular_histograma(frame)

	# Concatenamos el histograma al lado del video capturado
	#frame_hist = cv2.hconcat([frame, hist_img])
	frame_hist = np.concatenate((frame, hist_img), 1)

	# Mostramos la ventana
	cv2.imshow('Camara y Histograma', frame_hist)

	# Espera por una tecla
	key = cv2.waitKey(1) & 0xFF
	
	# Salir del bucle si se presiona 'q'
	if key == ord('q'):
		break
	
	# Guardar la imagen si se presiona 's'
	if key == ord('s'):
		# Convertir la imagen concatenada a formato RGB para guardar
		cv2.imwrite("src/parcial1/out/Charles_Chuez_P3.png", frame_hist)

# Libera la captura y cierra todas las ventanas
cap.release()
cv2.destroyAllWindows()