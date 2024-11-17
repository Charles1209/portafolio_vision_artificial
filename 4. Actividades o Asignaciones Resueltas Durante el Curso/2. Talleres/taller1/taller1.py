import cv2
import numpy as np
import numpy as np

def cartoonize(rgb_image: np.ndarray, num_pyr_downs=2, num_bilaterals=7):

	# Verificar que la imagen esté en formato RGB
	if rgb_image.shape[2] != 3:
		raise ValueError("La imagen debe tener 3 canales (RGB).")

	# Minimizar el tamaño de la imagen
	downsampled_img = rgb_image
	for _ in range(num_pyr_downs):
		downsampled_img = cv2.pyrDown(downsampled_img)

	for _ in range(num_bilaterals):
		filterd_small_img = cv2.bilateralFilter(downsampled_img, 9, 9, 7)

	filtered_normal_img = filterd_small_img
	for _ in range(num_pyr_downs):
		filtered_normal_img = cv2.pyrUp(filtered_normal_img)
	
	if filtered_normal_img.shape != rgb_image.shape:
		filtered_normal_img = cv2.resize(filtered_normal_img, rgb_image.shape[:2])

	if filtered_normal_img.shape[:2] != rgb_image.shape[:2]:
		filtered_normal_img = cv2.resize(filtered_normal_img, (rgb_image.shape[1], rgb_image.shape[0]))

	cv2.imshow("Paso 1", filtered_normal_img)
	cv2.waitKey(0)
	# cv2.destroyAllWindows()

	#Paso 2
	img_gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)

	cv2.imshow("Paso 2", img_gray)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

	#Paso 3
	img_blur = cv2.medianBlur(img_gray, 7)

	cv2.imshow("Paso 3", img_blur)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

	#Paso 4
	gray_edges = cv2.adaptiveThreshold(img_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 2)

	cv2.imshow("Paso 4", filtered_normal_img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

	#Paso 5
	rgb_edges = cv2.cvtColor(gray_edges, cv2.COLOR_GRAY2RGB)

	cv2.imshow("Paso 5", filtered_normal_img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

	# Asegurarse de que las dimensiones coincidan antes de la operación bitwise
	if filtered_normal_img.shape != rgb_edges.shape:
		rgb_edges = cv2.resize(rgb_edges, (filtered_normal_img.shape[1], filtered_normal_img.shape[0]))

	return cv2.bitwise_and(filtered_normal_img, rgb_edges)

if __name__ == "__main__":
	img = cv2.imread("img/taller1.jpg")

	cv2.imshow('Image Original', img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

	img_new = cartoonize(img)

	img_c = np.concatenate((img, img_new), axis=1)

	cv2.imshow('Image con Filtro', img_c)
	cv2.waitKey(0)
	cv2.destroyAllWindows()