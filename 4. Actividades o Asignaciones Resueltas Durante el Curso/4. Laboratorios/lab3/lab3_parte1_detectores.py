import cv2
import numpy as np
import matplotlib.pyplot as plt

####################################
####	Detectores de Keypoints	####
####################################

################################################################################################

# Detector MSER

################################################################################################

# Leer la imagen y cambiar el espacio de color
imgname = 'img/monedas.jpg'
imgMSER = cv2.imread(imgname)
grayMSER = cv2.cvtColor(imgMSER, cv2.COLOR_BGR2GRAY)
# Inicializar MSER y asignar parámetros
mser = cv2.MSER_create()
# Realizar la detección, obteniendo las coordenadas
# de las regiones y sus bounding boxes
coordinates, bboxes = mser.detectRegions(grayMSER)

################################################################################################

coords = []
for coord, bbox in zip(coordinates, bboxes):
	x,y,w,h = bbox
	if w< 10 or h < 10 or w/h > 5 or h/w > 5:
		continue
	coords.append(coord)

print( "Regiones Detectadas usando MSER Detector : {}".format(len(coordinates)))

# lista de colores para asignar a las regiones
colors =   [[43, 43, 200], [43, 75, 200], [43, 106, 200], [43, 137, 200],
			[43, 169, 200], [43, 200, 195], [43, 200, 163], [43, 200, 132],
			[43, 200, 101], [43, 200, 69], [54, 200, 43], [85, 200, 43],
			[116, 200, 43], [148, 200, 43], [179, 200, 43], [200, 184, 43],
			[200, 153, 43], [200, 122, 43], [200, 90, 43], [200, 59, 43],
			[200, 43, 64], [200, 43, 95], [200, 43, 127], [200, 43, 158],
			[200, 43, 190], [174, 43, 200], [142, 43, 200], [111, 43, 200],
			[80, 43, 200], [43, 43, 200]]

# Pintar las regiones con colores aleatorios
np.random.seed(0)
canvas1 = imgMSER.copy()
canvas2 = cv2.cvtColor(grayMSER, cv2.COLOR_GRAY2BGR)
canvas3 = np.zeros_like(imgMSER)

################################################################################################

for cnt in coords:
	xx = cnt[:,0]
	yy = cnt[:,1]
	color = colors[np.random.choice(len(colors))]
	canvas1[yy, xx] = color
	canvas2[yy, xx] = color
	canvas3[yy, xx] = color

cv2.imwrite("src/lab3/out/result1_mser.png", canvas1)
cv2.imwrite("src/lab3/out/result2_mser.png", canvas2)
cv2.imwrite("src/lab3/out/result3_mser.png", canvas3)

plt.subplot(131)
plt.imshow(canvas1[:, :, ::-1])
plt.title('Original')
plt.axis('off')

plt.subplot(132)
plt.imshow(canvas2[:, :, ::-1])
plt.title('Escala de Grises')
plt.axis('off')

plt.subplot(133)
plt.imshow(canvas3[:, :, ::-1])
plt.title('Máscara')
plt.axis('off')

plt.savefig("src/lab3/out/MSER_Result.png", dpi=600, transparent=True)

plt.show()

################################################################################################

# Detector FAST

################################################################################################

imgFast = cv2.imread('img/tower2.png',0)

# Inicializar FAST con los valores por defecto, salvo el threshold

# Threshold
thre = 15
fast = cv2.FastFeatureDetector_create(thre)

# encontrar y dibujar los keypoints
kp = fast.detect(imgFast,None)
imgFast2 = cv2.drawKeypoints(imgFast, kp, None, color=(255,0,0))

# Imprimir los par´ametros utilizados
print( "Threshold: {}".format(fast.getThreshold()) )
print( "nonmaxSuppression:{}".format(fast.getNonmaxSuppression()) )
print( "vecindario: {}".format(fast.getType()) )
print( "Keypoints Totales con nonmaxSuppression: {}".format(len(kp)) )
cv2.imwrite('src/lab3/out/fast_true.png',imgFast2)

# Desactivar nonmaxSuppression
fast.setNonmaxSuppression(0)
kp = fast.detect(imgFast,None)
print( "Keypoints Totales sin nonMaxSuppression: {}".format(len(kp)) )

imgFast3 = cv2.drawKeypoints(imgFast, kp, None, color=(255,0,0))
cv2.imwrite('src/lab3/out/fast_false.png',imgFast3)

plt.subplot(121)
plt.imshow(imgFast2[:, :, ::-1])
plt.title('Con nonMaxSuppression')
plt.axis('off')
plt.subplot(122)
plt.imshow(imgFast3[:, :, ::-1])
plt.title('Sin nonMaxSuppression')
plt.axis('off')

plt.show()

################################################################################################

# Detector ORB

################################################################################################

imgORB = cv2.imread('img/tower2.png',0)

# Inicializar el Detector ORB
orb = cv2.ORB_create(nfeatures = 500)

# Encontrar los Keypoints
kp = orb.detect(imgORB,None)

# Pintar los Keypoints en la imagen original
imgORB2 = cv2.drawKeypoints(imgORB, kp, None, color=(0,255,0), flags=0)

print( "Keypoints Totales Usando ORB Detector : {}".format(len(kp)) )

plt.imshow(imgORB2[:, :, ::-1] )
plt.axis('off')
plt.savefig("src/lab3/out/ORB_KP.png", dpi=600, orientation='portrait', transparent=True)
plt.show()