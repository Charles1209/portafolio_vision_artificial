import cv2
import numpy as np
import matplotlib.pyplot as plt

def mostrar_imagenes(img1, msg1, img2, msg2):
	plt.subplot(121)
	plt.imshow(img1)
	plt.title(msg1)
	plt.axis('off')

	plt.subplot(122)
	plt.imshow(img2)
	plt.title(msg2)
	plt.axis('off')

	plt.show()

ruta = "img/lab3.jpg"
img = plt.imread(ruta)

################################################################################################

""" 1. Seleccione una imagen para el detector MSER. Aplique el detector. Luego aplique un
filtro bilateral a la imagen original. Posteriormente, aplique el detector a la imagen
filtrada, grafique los resultados. """

def gray_mser(img_gray):
	mser = cv2.MSER_create()

	regiones, bboxes = mser.detectRegions(img_gray)

	coords = []
	for region, bbox in zip(regiones, bboxes):
		x,y,w,h = bbox
		if w< 10 or h < 10 or w/h > 5 or h/w > 5:
			continue
		coords.append(region)

	colores =   [[43, 43, 200], [43, 75, 200], [43, 106, 200], [43, 137, 200],
				[43, 169, 200], [43, 200, 195], [43, 200, 163], [43, 200, 132],
				[43, 200, 101], [43, 200, 69], [54, 200, 43], [85, 200, 43],
				[116, 200, 43], [148, 200, 43], [179, 200, 43], [200, 184, 43],
				[200, 153, 43], [200, 122, 43], [200, 90, 43], [200, 59, 43],
				[200, 43, 64], [200, 43, 95], [200, 43, 127], [200, 43, 158],
				[200, 43, 190], [174, 43, 200], [142, 43, 200], [111, 43, 200],
				[80, 43, 200], [43, 43, 200]]

	np.random.seed(0)
	img_gray_mser = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB) # Aunque se ponga a color seguirá en gris. Se pone a color para darle color despues
	img_mask = np.zeros_like(img_gray_mser)

	for cnt in coords:
		xx = cnt[:,0]
		yy = cnt[:,1]
		color = colores[np.random.choice(len(colores))]
		img_gray_mser[yy, xx] = color
		img_mask[yy, xx] = color

	return img_gray_mser, img_mask

img_mser = plt.imread(ruta)
#img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_gray = cv2.cvtColor(img_mser, cv2.COLOR_RGB2GRAY) # Se abrió con plt y no con cv2
#img_gray = img_mser.copy() # No se puede transformar a color por los canales, trabaja con gris solamente
img_gray_mser, img_mask = gray_mser(img_gray)

img_bilateral = cv2.bilateralFilter(img_mser, d=9, sigmaColor=75, sigmaSpace=75)
img_bilateral_gray = cv2.cvtColor(img_bilateral, cv2.COLOR_RGB2GRAY)
img_bilateral_gray_mser, img_bilateral_mask = gray_mser(img_bilateral_gray)

################################################################################################

""" 2. Para los detectores FAST y ORB, seleccione una imagen (debe ser la misma imagen
para ambos detectores). Aplique los detectores, ajustando los parámetros para que se
produzca una buena identificación en la imagen. Luego, aplique a la imagen original
un giro de 90 utilizando la función:

ImageRot= cv2.rotate(ImageOrg, cv2.ROTATE_90_COUNTERCLOCKWISE)

La cual esta disponible en OpenCV. Seguidamente, aplique los algoritmos detectores
utilizando los mismos parámetros que utilizó con las imágenes originales. """

####################
####	FAST	####
####################

thre = 15
fast = cv2.FastFeatureDetector_create(thre)

def calcular_fast(img_fast):
	kp = fast.detect(img_fast, None)
	non_max_suppression = cv2.drawKeypoints(img_fast, kp, None, color=(255,0,0))

	fast.setNonmaxSuppression(0)
	kp = fast.detect(img_fast,None)

	max_suppression = cv2.drawKeypoints(img_fast, kp, None, color=(255,0,0))

	return non_max_suppression, max_suppression

img_fast = plt.imread(ruta)
img_fast_nms, img_fast_ms = calcular_fast(img_fast)

img_fast_rot = cv2.rotate(img_fast, cv2.ROTATE_90_COUNTERCLOCKWISE)
img_fast_rot_nms, img_fast_rot_ms = calcular_fast(img_fast_rot)

####################
####	ORB		####
####################

orb = cv2.ORB_create(nfeatures = 500)

def calcular_orb(img_orb):
	kp = orb.detect(img_orb, None)
	img_orb_kp = cv2.drawKeypoints(img_orb, kp, None, color=(0,255,0), flags=0)

	return img_orb_kp

img_orb = plt.imread(ruta)
img_orb_kp = calcular_orb(img_orb)

img_orb_rot = cv2.rotate(img_orb, cv2.ROTATE_90_COUNTERCLOCKWISE)
img_orb_rot_kp = calcular_orb(img_orb_rot)

################################################################################################

""" 3. Presente gráficamente los resultados para cada detector señalando los puntos detectados
antes y después de aplicar la modificación de la imagen original, se recomienda
utilizar subplots de matplotlib. """

mostrar_imagenes(
	img, "Original",
	img_bilateral_gray_mser, "Derector MSER"
)

mostrar_imagenes(
	img, "Original",
	img_fast_rot_ms, "Detector FAST MaxSuppression"
)

mostrar_imagenes(
	img, "Original",
	img_orb_rot_kp, "Detector ORB"
)

""" Al final para esta sección debería presentar un total de 3 pares de imágenes de antes
y después en total, 1 para cada algoritmo detector presentado. """