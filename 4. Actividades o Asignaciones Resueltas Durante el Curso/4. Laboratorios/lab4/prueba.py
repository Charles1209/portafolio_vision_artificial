import cv2
import numpy as np
import matplotlib.pyplot as plt

# Cargar la imagen
image = cv2.imread('img/monedas.jpg', cv2.IMREAD_COLOR)

h,w,c = image.shape
dsize = (int(w*0.250), int(h*0.250))
image = cv2.resize(image, dsize)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.medianBlur(gray, 5)

# Detectar círculos usando el algoritmo de Hough
circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1, minDist=20, param1=50, param2=30, minRadius=0, maxRadius=0)

# Asegurarse de que al menos un círculo fue encontrado
if circles is not None:
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        # Dibujar el contorno del círculo en azul
        cv2.circle(image, (i[0], i[1]), i[2], (255, 0, 0), 2)
        # Dibujar el centro del círculo en amarillo
        cv2.circle(image, (i[0], i[1]), 2, (0, 255, 255), 3)

# Mostrar la imagen con los círculos detectados
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
