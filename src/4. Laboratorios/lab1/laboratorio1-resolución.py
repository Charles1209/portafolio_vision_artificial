import cv2

im = cv2.imread("img/lab1.jpg")
cv2.imshow("Imagen Original", im)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Técnica del píxel
""" im_p = im
for x in range(300, 500):
	for y in range (130, 230):
		im_p[y, x] = (255, 255, 0)
cv2.imshow("Modificación de Píxel", im_p)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("src/img/imagen_pixel.jpg", im_p) """

""" 1. 1 modificación de color mediante la técnica del parche, el área modificar debe tener
un forma rectangular y ubicarse en el centro de la imagen. """
im[150:200, 300:500] = (0,0,0)
cv2.imshow("Mostrando parte 1", im)
cv2.waitKey(0)
cv2.destroyAllWindows()

""" 2. 1 modificación de color mediante la técnica del parche de forma cuadrada ubicada en la esquina
inferior izquierda con el siguiente color RGB (100, 255, 51) . """
im[300:350, 0:50] = (255,255,255)
cv2.imshow("Mostrando parte 2", im)
cv2.waitKey(0)
cv2.destroyAllWindows()


""" 3. Crear un parche de una zona rectangular de la imagen y ubicarlo en la esquina inferior derecha
de la imagen. """
parche = im[200:250, 300:500]
im[300:350, 600:800] = parche
cv2.imshow("Mostrando parte 3", im)
cv2.waitKey(0)
cv2.destroyAllWindows()