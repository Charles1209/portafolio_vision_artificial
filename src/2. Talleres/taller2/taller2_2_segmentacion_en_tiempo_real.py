import cv2
import numpy as np
import matplotlib.pyplot as plt

################################################################################################

image = cv2.imread('img/taller2.png')
print('Image type: ', type(image), 'Image Dimensions : ', image.shape)

escala = 0.80
h,w,c = image.shape
h_final = int(h*escala)
w_final = int(w*escala)

image_copy = np.copy(image)
dsize = (w_final, h_final)
# escalar imagen
image_copy = cv2.resize(image_copy, dsize)
plt.imshow(image_copy)

# Rangos de Color
lower_blue = np.array([100, 0, 0])
upper_blue = np.array([255, 100, 120])

# Mascara segun Rangos
mask = cv2.inRange(image_copy, lower_blue, upper_blue)
plt.imshow(mask, cmap='gray')

masked_image = np.copy(image_copy)
masked_image[mask != 0] = [0, 0, 0]
plt.imshow(masked_image[:,:,::-1])

plt.show()

################################################################################################

# Capturar desde la c´amara web
video_capture = cv2.VideoCapture(0)

# Al descomentar estas lineas se puede modificar
# la resolucion de la imagen de entrada
# Definir la resolucion para la imagen de entrada
#video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
#video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Se obtiene la resolucion de la imagen de entrada
vid_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
vid_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

vid_dsize = (vid_width, vid_height)

while True:
	# iniciar captura
	ret, frame = video_capture.read()
	background_image = np.copy(frame)

	# se usa el tama~no de la img de entrada
	masked_image = cv2.resize(masked_image, vid_dsize)
	mask = cv2.resize(mask, vid_dsize)

	# Aplicar mascara a la imagen de entrada
	background_image[mask == 0] = [0, 0, 0]

	# Union de fondo y montaje
	final_image = background_image + masked_image

	cv2.imshow('OpenCV Segmentacion de Color', final_image)
	# Presionar 'q' para salir
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

video_capture.release()
cv2.destroyAllWindows()