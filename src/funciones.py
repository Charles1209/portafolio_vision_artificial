import cv2
import matplotlib.pyplot as plt
import numpy as np
import sys

def abrir_imagen(ruta):
	img = cv2.imread(ruta) # Se abre con cv2 para que los puntos no salgan con los colores invertidos

	if img is None:
		sys.exit('Fallo al cargar la imagen')

	return img

def invertir_colores(img):
	x, y, z = cv2.split(img)
	img = cv2.merge([z, y, x])

	#img = img[:, :, ::-1]

	#img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
	#img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

	return img

def mostrar_imagen(img, msg):
	img = invertir_colores(img) # Se invierten los colores porque se abrio con cv2 y no con plt

	plt.imshow(img)
	plt.title(msg)
	plt.axis('off')
	plt.show()

def comparar_imagenes(img1, msg1, img2, msg2):
	img1 = invertir_colores(img1)
	img2 = invertir_colores(img2)

	plt.subplot(121)
	plt.imshow(img1)
	plt.title(msg1)
	plt.axis('off')

	plt.subplot(122)
	plt.imshow(img2)
	plt.title(msg2)
	plt.axis('off')

	plt.show()

def color_aleatorio():
	colors =   [[43, 43, 200], [43, 75, 200], [43, 106, 200], [43, 137, 200],
				[43, 169, 200], [43, 200, 195], [43, 200, 163], [43, 200, 132],
				[43, 200, 101], [43, 200, 69], [54, 200, 43], [85, 200, 43],
				[116, 200, 43], [148, 200, 43], [179, 200, 43], [200, 184, 43],
				[200, 153, 43], [200, 122, 43], [200, 90, 43], [200, 59, 43],
				[200, 43, 64], [200, 43, 95], [200, 43, 127], [200, 43, 158],
				[200, 43, 190], [174, 43, 200], [142, 43, 200], [111, 43, 200],
				[80, 43, 200], [43, 43, 200]]

	color = colors[np.random.choice(len(colors))]

	return color

# Detectores

def detector_mser(img):
	mser = cv2.MSER_create()

	img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	regiones, bboxes = mser.detectRegions(img_gray)

	coords = []
	for region, bbox in zip(regiones, bboxes):
		x,y,w,h = bbox
		if w< 10 or h < 10 or w/h > 5 or h/w > 5:
			continue
		coords.append(region)

	np.random.seed(0)
	img_gray_mser = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB) # Aunque se ponga a color seguirá en gris. Se pone a color para darle color despues
	#img_mask = np.zeros_like(img_gray_mser)

	for cnt in coords:
		xx = cnt[:,0]
		yy = cnt[:,1]
		color = color_aleatorio()
		img_gray_mser[yy, xx] = color
		#img_mask[yy, xx] = color

	#return img_gray_mser, img_mask
	return img_gray_mser

def detector_fast(img):
	img = invertir_colores(img)

	thre = 15
	fast = cv2.FastFeatureDetector_create(thre)

	kp = fast.detect(img, None)
	non_max_suppression = cv2.drawKeypoints(img, kp, None, color=(255,0,0))

	fast.setNonmaxSuppression(0)
	kp = fast.detect(img,None)

	#max_suppression = cv2.drawKeypoints(img, kp, None, color=(255,0,0))

	#return non_max_suppression, max_suppression

	non_max_suppression = invertir_colores(non_max_suppression)

	return non_max_suppression

def detector_orb(img_orb):
	orb = cv2.ORB_create(nfeatures = 500)

	kp = orb.detect(img_orb, None)
	img_orb_kp = cv2.drawKeypoints(img_orb, kp, None, color=(0,255,0), flags=0)

	return img_orb_kp

def detector_harris(img_harris, blockSize=2, ksize=3, k=0.08):
	gray = cv2.cvtColor(img_harris, cv2.COLOR_BGR2GRAY)
	gray = np.float32(gray)

	dst = cv2.cornerHarris(gray, blockSize, ksize, k)
	dst = cv2.dilate(dst, None)

	img_harris[dst>0.01*dst.max()]=[0,0,255]

	return img_harris

def detector_shi_tomasi(img_shi, cantidad=65, qlt=0.08, euDist=10, radio=8, color=(0,255,0), thickness=2):
	gray = cv2.cvtColor(img_shi, cv2.COLOR_BGR2GRAY)

	corners = cv2.goodFeaturesToTrack(gray, cantidad, qlt, euDist)
	corners = np.intp(corners)

	for i in corners:
		x,y = i.ravel()
		cv2.circle(img_shi, (x,y), radio, color, thickness)

	return img_shi

def detector_canny(img):
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	blur = cv2.blur(gray, (3,3))

	edges = cv2.Canny(blur,50,150,apertureSize = 3)
	edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

	return edges

# Descriptores

def descriptor_orb(imgORB):
	orb = cv2.ORB_create(nfeatures = 2500, edgeThreshold = 73, nlevels=18)

	kpORB, desORB = orb.detectAndCompute(imgORB, None) # desORB es para el histograma

	imgORB2 = cv2.drawKeypoints(imgORB,kpORB,np.array([]), color=(0,5,255), flags=0)

	return imgORB2

def descriptor_brisk(imgBrisk):
	grayBrisk = cv2.cvtColor(imgBrisk, cv2.COLOR_BGR2GRAY)

	detectorBrisk = cv2.BRISK_create()
	keypointsBrisk, descriptorsBrisk = detectorBrisk.detectAndCompute(grayBrisk, None) # descriptorBrisk para el histograma

	imgBrisk2 = cv2.drawKeypoints(imgBrisk,keypointsBrisk,np.array([]), color=(0,5,255), flags=0)
	
	return imgBrisk2

def descriptor_kaze(imgKaze):
	grayKaze = cv2.cvtColor(imgKaze, cv2.COLOR_BGR2GRAY)

	detectorKaze = cv2.KAZE_create()
	keypointsKaze, descriptorsKaze = detectorKaze.detectAndCompute(grayKaze, None)

	imgKaze2 = cv2.drawKeypoints(imgKaze,keypointsKaze,np.array([]), color=(0,5,255), flags=0)

	return imgKaze2