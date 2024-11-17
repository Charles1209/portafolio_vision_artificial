import cv2
import numpy as np

def calcular_histograma(imagen):
	# Obtener el alto y el ancho de la imagen de entrada
	#ancho, alto = dsize
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

# Modificar tamaño de la imagen
dsize = (int(640), int(480))

# Captura de video desde la cámara y establece la resolución
cap = cv2.VideoCapture(0)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
	ret, frame = cap.read()
	if not ret:
		break

	# Redimensiona el video capturado a 640x480
	frame = cv2.resize(frame, dsize)

	# Muestra el histograma
	hist_img = calcular_histograma(frame)

	# Concatenamos el histograma al lado del video capturado
	#frame_hist = cv2.hconcat([frame, hist_img])
	frame_hist = np.concatenate((frame, hist_img), 1)

	# Mostramos la ventana
	cv2.imshow('Camara y Histograma', frame_hist)

	# Si presionamos la tecla 'q' salimos del bucle
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

# Libera la captura y cierra todas las ventanas
cap.release()
cv2.destroyAllWindows()