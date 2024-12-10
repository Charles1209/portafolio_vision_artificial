#######################
####	Detectores ####
#######################

import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys

############################################
####	Detector de Esquinas Harris		####
############################################

imgHarris = cv2.imread('img/cruci1.jpg')
#Convertir a escala de grises
gray = cv2.cvtColor(imgHarris,cv2.COLOR_BGR2GRAY)
gray = np.float32(gray)

################################################################################################

blockSize = 2
ksize = 3
k = 0.08

dst = cv2.cornerHarris(gray,blockSize,ksize,k)
#print(dst)
# Se dilata el resultado para marcar las esquinas
# se hacen más visibles al visualizar la imagen
dst = cv2.dilate(dst,None)

################################################################################################

# Umbral para un valor ´optimo, puede variar según la imagen
puntos = imgHarris[dst>0.01*dst.max()].sum()

print( "Puntos Totales Usando Harris : {}".format(puntos) )

imgHarris[dst>0.01*dst.max()]=[0,0,255]
cv2.imshow('dst',imgHarris)
cv2.waitKey(0)

# Para destruir todas las ventanas creadas
cv2.destroyAllWindows()
cv2.imwrite("src/lab4/out/Harris.png", imgHarris)
plt.imshow(imgHarris[:, :, ::-1] )
plt.axis('off')
plt.show()

############################################
####	Detector de Esquinas Shi-Tomasi	####
############################################

imgShi = cv2.imread('img/tower2.png')
gray = cv2.cvtColor(imgShi,cv2.COLOR_BGR2GRAY)

# Cantidad Deseada
cantidad = 65

# Calidad [0-1]
qlt = 0.08

# Distancia Euclidea mínima entre puntos
euDist = 10
corners = cv2.goodFeaturesToTrack(gray,cantidad,qlt,euDist)
corners = np.intp(corners)

print( "Keypoints Totales Usando Shi-imgTomasi : {}".format(len(corners)) )

################################################################################################

# Radio del circulo
radio = 8

# color en BGR
color = (0, 255, 0)

# grosor de la l´ınea
thickness = 2

for i in corners:
	x,y = i.ravel()
	cv2.circle(imgShi,(x,y),radio, color, thickness)

plt.imshow(imgShi[:, :, ::-1] )
plt.savefig("src/lab4/out/Tomasi.png", dpi=600, orientation='portrait', transparent=True)
plt.axis('off')
plt.show()

############################
####	Detector Canny	####
############################

img2 = cv2.imread('img/monedasParcial.jpg')

# Verificar la existencia de la imagen
if img2 is None:
	sys.exit('Fallo al cargar la imagen')

img_gris = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
img_blur = cv2.blur(img_gris, (3,3))

edges = cv2.Canny(img_blur,50,150,apertureSize = 3)
edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
cv2.imwrite('src/lab4/out/edgesCanny.jpg', edges)

plt.subplot(121)
plt.axis("off")
plt.imshow(img2[:,:, ::-1])
plt.title('Imagen Original')

plt.subplot(122)
plt.axis("off")
#plt.imshow(edges, cmap='gray')
plt.imshow(edges)
plt.title('Bordes Detectados')

plt.savefig("src/lab4/out/BordesDetectados.png", dpi=600)

plt.show()