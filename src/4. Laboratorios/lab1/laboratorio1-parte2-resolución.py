import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread("img/tower.jpg")
dsize = (int(854), int(480))
img = cv2.resize(img, dsize)
cv2.imwrite("src/lab1/out/tower2.jpg", img)

""" 1. Leer un imagen de su computadora se recomienda que esta imagen sea en formato
horizontal o landscape. Convertir la imagen a formato de MatPlotLib (RGB). """

img_bgr = cv2.imread("img/tower2.png")
#b, g, r = cv2.split(img_bgr)
#img_rgb = cv2.merge([r, g, b])
img_rgb = img_bgr[:, :, ::-1]
plt.imshow(img_rgb)
plt.title("img plt")
plt.show()

""" 2. Crear una imagen utilizando NumPy y asignar al fondo el color RGB (51, 184, 255) ,
crear un recuadro en dicha imagen que debe ser del siguiente color RGB
(255, 51, 200) , esta imagen debe ser en formato horizontal o apaisada. """

img_np = np.zeros((480, 854, 3), dtype=np.uint8)
img_np[:,:,:] = (51, 184, 255)
img_np[50:75,100:200,:] = (255, 51, 200)
plt.imshow(img_np)
plt.title("img NumPy")
plt.show()

""" 3. Concatene verticalmente las 3 imágenes (RGB, NumPy, BGR) utilizando la función
np.concatenate() y adjunte la imagen generada a su informe. """

img_rgb = cv2.resize(img_rgb, dsize)
img_np = cv2.resize(img_np, dsize)
img_bgr = cv2.resize(img_bgr, dsize)

img_np_concatenate = np.concatenate((img_rgb, img_np, img_bgr), axis=0)
plt.imshow(img_np_concatenate)
plt.title("RGB, NumPy and BGR image")
plt.show()

""" 4. Concatene verticalmente las 3 imágenes (RGB, NumPy, BGR) utilizando la función
subplot y adjunte la imagen o una captura de esta generada a su informe. """

plt.subplot(311)
plt.imshow(img_rgb)
plt.title('img RGB')
plt.subplot(312)
plt.imshow(img_np)
plt.title('img NumPy')
plt.subplot(313)
plt.imshow(img_bgr)
plt.title('img BGR')
plt.show()