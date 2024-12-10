import cv2
from taller1 import cartoonize

# Capturar desde la cámara web
video_capture = cv2.VideoCapture(0)
frame_index = 0

while True:
	# iniciar captura
	ret, frame = video_capture.read()
	# TODA MODIFICACION A LA IMAGEN DE ENTRADA
	# SE DEBE COLOCAR EN ESTA PARTE DEL CODIGO
	#
	frame = cartoonize(frame)
	
	# Mostrar el frame en una ventana llamada 'frame'
	cv2.imshow('frame', frame)

	# Presionar C para guardar
	# if cv2.waitKey(20) & 0xFF == ord('c'):
	# 	frame_name = "taller1/img/camera_frame_{}.png".format(frame_index)
	# 	cv2.imwrite(frame_name, final_image)
	# 	frame_index += 1
	# cv2.imshow('OpenCV Segmentacion de Color', final_image)

	# Presionar 'q' para salir
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

# Liberar camara y limpiar ventanas
video_capture.release()
cv2.destroyAllWindows()
