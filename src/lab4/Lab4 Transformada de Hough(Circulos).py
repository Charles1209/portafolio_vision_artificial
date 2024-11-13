import cv2
import matplotlib.pyplot as plt
import numpy as np

# Cargar la imagen
imgIn = cv2.imread('img/monedas.jpg')

h, w, c = imgIn.shape

# Redimensionar la imagen
dsize = (int(w * 0.250), int(h * 0.250))
imgCir = cv2.resize(imgIn, dsize)

# Aplicar un filtro para suavizar la imagen
src = cv2.medianBlur(imgCir, 5)
src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

# Detectar círculos con el algoritmo de Hough
circles = cv2.HoughCircles(src, cv2.HOUGH_GRADIENT, 1, 20,
                           param1=50, param2=30, minRadius=10, maxRadius=50)

# Verificar que se hayan detectado círculos
if circles is not None:
    circles = np.uint16(np.around(circles))

    # Crear una lista para almacenar los recortes de cada círculo
    circle_images = []

    for i in circles[0, :]:
        # Dibujar el contorno del círculo en azul
        cv2.circle(imgCir, (i[0], i[1]), i[2], (255, 0, 0), 2)
        # Dibujar el centro del círculo en amarillo
        cv2.circle(imgCir, (i[0], i[1]), 2, (0, 255, 255), 3)

        # Recortar el área del círculo detectado
        x, y, r = i[0], i[1], i[2]
        x1, y1 = max(0, x - r), max(0, y - r)
        x2, y2 = min(imgCir.shape[1], x + r), min(imgCir.shape[0], y + r)
        cropped_circle = imgCir[y1:y2, x1:x2]
        circle_images.append(cropped_circle)

    # Guardar la imagen con los círculos detectados
    cv2.imwrite("src/lab4/out/circulos_detectados.png", imgCir)

    # Definir el número de columnas y filas para los subplots
    cols = 5  # Número de columnas predefinido
    rows = (len(circle_images) + cols - 1) // cols  # Calcular las filas de forma dinámica

    # Crear subplots y mostrar los recortes de cada círculo
    fig, axs = plt.subplots(rows, cols, figsize=(15, rows * 3))

    # Recorrer cada recorte y asignarlo a un subplot
    for idx, circle_img in enumerate(circle_images):
        row, col = divmod(idx, cols)
        axs[row, col].imshow(circle_img[:, :, ::-1])
        axs[row, col].axis('off')

    # Ocultar los ejes en los subplots vacíos si sobran espacios
    for idx in range(len(circle_images), rows * cols):
        row, col = divmod(idx, cols)
        axs[row, col].axis('off')

    # Mostrar la imagen original con los círculos detectados
    plt.figure()
    plt.imshow(imgCir[:, :, ::-1])
    plt.axis('off')
    plt.title('Imagen Original con los Circulos Detectados')
    plt.show()
