import cv2
import numpy as np
import matplotlib.pyplot as plt

################################################################

# Cargar una iagen
img_OpenCV = cv2.imread('img/tower.jpg')
h,w,c = img_OpenCV.shape
dsize = (int(w*0.250), int(h*0.250))
img_OpenCV = cv2.resize(img_OpenCV, dsize)
# Dividir la imagen en sus 3 canales (b, g, r):
b, g, r = cv2.split(img_OpenCV)
h,w,c = img_OpenCV.shape
print("Dimensiones de la imagen - Alto: {}, Ancho: {}, Canales: {}".format(h, w, c))

################################################################

# Combinar los canales pero en orden RGB
img_matplotlib = cv2.merge([r, g, b])

################################################################

# Ver ambas imágenes(img_OpenCV y img_matplotlib) usando matplotlib
# Esta mostrará la imagen con los colores erróneos
plt.subplot(121)
plt.imshow(img_OpenCV)
plt.title('img OpenCV')
# Esto mostrará los colores verdaderos
plt.subplot(122)
plt.imshow(img_matplotlib)
plt.title('img matplotlib')
plt.show()

################################################################

# Ver ambas imágenes(img_OpenCV y img_matplotlib) usando cv2.imshow()
# Esto mostrará los colores verdaderos
cv2.imshow('bgr image', img_OpenCV)
# Esta mostrará la imagen con los colores erróneos
cv2.imshow('rgb image', img_matplotlib)
cv2.waitKey(0)
cv2.destroyAllWindows()

################################################################

# Concatenar Imagenes Horizontalmente
# (img_OpenCV a la izquierda de img_matplotlib):
img_concats = np.concatenate((img_OpenCV, img_matplotlib), axis=1)
# Mostrar la imagen concatenada
cv2.imshow('bgr image and rgb image', img_concats)
cv2.waitKey(0)
cv2.destroyAllWindows()

################################################################
################################################################
################################################################
################################################################

# Obtener los canales, similar a cv2.split()
B = img_OpenCV[:, :, 0]
G = img_OpenCV[:, :, 1]
R = img_OpenCV[:, :, 2]

################################################################

# Transformar la imagen BGR a RGB usando Numpy :
# ::-1 iterable que imprime los elementos en orden inverso
img_RGB = img_OpenCV[:, :, ::-1]
# Ahora se muestra la imagen, pero los colores serán erróneos
cv2.imshow('img RGB (wrong color)', img_RGB)
cv2.waitKey(0)
cv2.destroyAllWindows()

################################################################

# Creación de una imagen con dimensiones 1040x580 y 3 canales de color
bgr = np.zeros((1040, 580, 3), dtype=np.uint8)

################################################################

# Asignar un color a todos los pı́xeles de la imagen
bgr[:,:,:] = (0, 80, 100)
# Asignar un color a una sección de la imagen
bgr[50:75,100:200,:] = (55, 255, 100)
# Mostrar la imagen creada
cv2.imshow('BGR', bgr)
cv2.waitKey(0)
cv2.destroyAllWindows()

################################################################

# Concatenar Horizontalmente 2 imágenes
img_concats = np.concatenate((img_OpenCV, bgr), axis=1)
cv2.imshow('bgr image and rgb image', img_concats)
cv2.waitKey(0)
cv2.destroyAllWindows()