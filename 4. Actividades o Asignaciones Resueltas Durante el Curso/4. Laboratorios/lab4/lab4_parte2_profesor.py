####################################
####	Transformada de Hough	####
####################################

import cv2
import numpy as np
import matplotlib.pyplot as plt

################################
####	Detectar Líneas		####
################################

img1 = cv2.imread('img/sudo.PNG')

gray1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
edges1 = cv2.Canny(gray1,50,150,apertureSize = 3)

lines1 = cv2.HoughLines(edges1,1,np.pi/180,200)
for rho,theta in lines1[0]:
	a = np.cos(theta)
	b = np.sin(theta)
	x0 = a*rho
	y0 = b*rho
	x1 = int(x0 + 1000*(-b))
	y1 = int(y0 + 1000*(a))
	x2 = int(x0 - 1000*(-b))
	y2 = int(y0 - 1000*(a))

	cv2.line(img1,(x1,y1),(x2,y2),(0,0,255),2)

cv2.imwrite('src/lab4/out/houghlines1.jpg', img1)
plt.imshow(img1[:, :, ::-1])

plt.show()

####################################################
####	Transformada Probabilística de Hough	####
####################################################

img3 = cv2.imread('img/cruci1.jpg')
h,w,c = img3.shape
# Nuevo Tamaño
dsize = (int(w*0.250), int(h*0.250))
# resize image
img4 = cv2.resize(img3, dsize)

gray4 = cv2.cvtColor(img4, cv2.COLOR_BGR2GRAY)
edges4 = cv2.Canny(gray4, 50, 150, apertureSize = 3)

lines4 = cv2.HoughLinesP(edges4, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
print(lines4[0])

for line in lines4:
	x1, y1, x2, y2 = line[0]
	cv2.line(img4, (x1,y1), (x2,y2), (0,255,0), 1, cv2.LINE_AA)

cv2.imwrite("src/lab4/out/lineasHough4.png", img4)

plt.subplot(121)
plt.imshow(img4[:, :, ::-1])

plt.subplot(122)
plt.imshow(edges4,cmap = 'gray')

plt.savefig("src/lab4/out/LineasHough_Comparativa.png", dpi=600)

plt.show()

################################
####	Detectar Círculos	####
################################

imgIn = cv2.imread('img/coin.jpg')

h,w,c = imgIn.shape
# Nuevo Tama~no
dsize = (int(w*0.250), int(h*0.250))
# resize image
imgCir = cv2.resize(imgIn, dsize)

src = cv2.medianBlur(imgCir, 5)
src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

circles = cv2.HoughCircles(src, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=80, minRadius=0, maxRadius=0)

circles = np.uint16(np.around(circles))
for i in circles[0,:]:
	# dibujar circulo
	cv2.circle(imgCir, (i[0], i[1]), i[2], (0,255,0), 2)
	# dibujar centro
	cv2.circle(imgCir, (i[0], i[1]), 2, (0,0,255), 3)

cv2.imwrite("src/lab4/out/circulos.png", imgCir)

plt.imshow(imgCir[:, :, ::-1])

plt.show()