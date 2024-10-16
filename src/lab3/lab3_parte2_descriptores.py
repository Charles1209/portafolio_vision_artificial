import cv2
import numpy as np
import matplotlib.pyplot as plt

################################################################################################

# Descriptor ORB

################################################################################################

def show_img_ply(bins, val, pos ):
	plt.subplot(3,4,pos)
	plt.bar(bins, val, width = 0.6, color='#0504aa',alpha=0.7)
	plt.xlim(min(bins), max(bins))

################################################################################################

imgORB = cv2.imread('img/tower2.png')

# Inicializar Detector ORB
orb = cv2.ORB_create(nfeatures = 2500, edgeThreshold = 73, nlevels=18)

# Calcular Keypoints y Descriptores con ORB
kpORB, desORB = orb.detectAndCompute(imgORB, None)

################################################################################################

#Preparar Salida
bins = list(range(desORB.shape[1]))
fig, axs = plt.subplots(3,4, figsize=(15,10), sharex='col', sharey='row')
fig.suptitle("ORB Detector -Descriptor.",fontsize=19, va='top')
for i in range(12):
	val= desORB[i]
	pos = i+1
	show_img_ply(bins, val, pos)
	if(i == 11):
		plt.savefig("src/lab3/out/ORB_Det_Desc_Histograma.png", dpi=350, bbox_inches='tight')
plt.show()

# Dibujar los Keypoints
imgORB2 = cv2.drawKeypoints(imgORB,kpORB,np.array([]), color=(0,5,255), flags=0)

cv2.imwrite("src/lab3/out/Tower_ORB.png", imgORB2)
plt.imshow(imgORB2[:, :, ::-1])
plt.show()

################################################################################################

# Descriptor BRISK

################################################################################################

imgBrisk = cv2.imread('img/tower2.png')

# Convertimos la imagen a escala de grises,
# puesto que es un prerrequisito para usar descriptores binarios.
grayBrisk = cv2.cvtColor(imgBrisk, cv2.COLOR_BGR2GRAY)

# Instanciamos BRISK, el cual nos da los puntos
# clave ası́ como los descriptores binarios.
detectorBrisk = cv2.BRISK_create()
keypointsBrisk, descriptorsBrisk = detectorBrisk.detectAndCompute(grayBrisk, None)

# Imprimimos el número de puntos clave hallados,
# ası́ como las dimensiones del vector de features.
print(f'Número de puntos clave detectados.: {len(keypointsBrisk)}')
print(f'Dimensiones del vector de features: {descriptorsBrisk.shape}')

################################################################################################

#Preparar Salida
bins = list(range(descriptorsBrisk.shape[1]))
fig, axs = plt.subplots(3,4, figsize=(15,10), sharex='col', sharey='row')
fig.suptitle("BRISK Detec BRISK Desc.",fontsize=19, va='top')
for i in range(12):
	val= descriptorsBrisk[i]
	pos = i+1
	show_img_ply(bins, val, pos )
	if(i == 11):
		plt.savefig("src/lab3/out/BRISK_Det_BRISK_Des_Histograma.png", dpi=350, bbox_inches='tight')
plt.show()

# Dibujar los Keypoints
imgBrisk2 = cv2.drawKeypoints(imgBrisk,keypointsBrisk,np.array([]), color=(0,5,255), flags=0)
cv2.imwrite("src/lab3/out/Tower_BRISK.png", imgBrisk2)
plt.imshow(imgBrisk2[:, :, ::-1])
plt.title("Descriptor BRISK")
plt.show()

################################################################################################

# Descriptor KAZE

################################################################################################

# Cargamos una imagen de prueba y la mostramos en pantalla.
imgKaze = cv2.imread('img/tower2.png')

# Convertimos la imagen a escala de grises,
# puesto que es un prerrequisito para usar descriptores binarios.
grayKaze = cv2.cvtColor(imgKaze, cv2.COLOR_BGR2GRAY)

# Instanciamos KAZE, el cual nos da los puntos
# clave ası́ como los descriptores binarios.
detectorKaze = cv2.KAZE_create()
keypointsKaze, descriptorsKaze = detectorKaze.detectAndCompute(grayKaze, None)

# Imprimimos el número de puntos clave hallados,
# ası́ como las dimensiones del vector de features.
print(f'Número de puntos clave detectados.: {len(keypointsKaze)}')
print(f'Dimensiones del vector de features: {descriptorsKaze.shape}')

################################################################################################

#Preparar Salida
bins = list(range(descriptorsKaze.shape[1]))
fig, axs = plt.subplots(3,4, figsize=(15,10), sharex='col', sharey='row')
fig.suptitle("KAZE Detec KAZE Desc.",fontsize=19, va='top')
for i in range(12):
	val= descriptorsKaze[i]
	pos = i+1
	show_img_ply(bins, val, pos )
	if(i == 11):
		plt.savefig("src/lab3/out/KAZE_Det_KAZE_Des_Histograma.png", dpi=350, bbox_inches='tight')
plt.show()

# Dibujar los Keypoints
imgKaze2 = cv2.drawKeypoints(imgKaze,keypointsKaze,np.array([]),
color=(0,5,255), flags=0)

cv2.imwrite("src/lab3/out/Tower_KAZE.png", imgKaze2)
plt.imshow(imgKaze2[:, :, ::-1])
plt.show()

################################################################################################

# Detector FAST
# Descriptor ORB

################################################################################################

imgFast = cv2.imread('img/tower2.png',0)

# Threshold
thre = 15
fast = cv2.FastFeatureDetector_create(thre)

# encontrar y dibujar los keypoints
kpFAST = fast.detect(imgFast,None)

# Inicializar Detector ORB
orb = cv2.ORB_create()

# Calcular Descriptores con ORB
kpFAST, desORB = orb.compute(imgFast, kpFAST)

################################################################################################

#Preparar Salida
bins = list(range(desORB.shape[1]))
fig, axs = plt.subplots(3,4, figsize=(15,10), sharex='col', sharey='row')
fig.suptitle("FAST Detec ORB Desc.",fontsize=19, va='top')
for i in range(12):
	val= desORB[i]
	pos = i+1
	show_img_ply(bins, val, pos )
	if(i == 11):
		plt.savefig("src/lab3/out/FAST_Det_ORB_Des_Histograma.png", dpi=350, bbox_inches='tight')
plt.show()

# Dibujar los Keypoints
imgFAST2 = cv2.drawKeypoints(imgFast,kpFAST,np.array([]), color=(0,5,255), flags=0)
cv2.imwrite("src/lab3/out/Tower_FAST.png", imgFAST2)
plt.imshow(imgFAST2[:, :, ::-1])
plt.show()