import cv2
import numpy as np
import matplotlib.pyplot as plt

################################################################################################

image = cv2.imread('img/ctree_bluescreen.jpg')
print('Image type: ', type(image), 'Image Dimensions : ', image.shape)

################################################################################################

image_copy = np.copy(image)

image_copy = cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB)
plt.imshow(image_copy)

################################################################################################

lower_blue = np.array([0, 0, 100]) ##[R value, G value, B value]
upper_blue = np.array([120, 100, 255])

################################################################################################

mask = cv2.inRange(image_copy, lower_blue, upper_blue)
plt.imshow(mask, cmap='gray')

################################################################################################

masked_image = np.copy(image_copy)
masked_image[mask != 0] = [0, 0, 0]
plt.imshow(masked_image)

################################################################################################

background_image = cv2.imread('img/playa.JPG')
background_image = cv2.cvtColor(background_image, cv2.COLOR_BGR2RGB)

# Nuevo Tama~no
dsize = (500, 281)
# escalar imagen
crop_background = cv2.resize(background_image, dsize)
crop_background[mask == 0] = [0, 0, 0]

plt.imshow(crop_background)

################################################################################################

final_image = crop_background + masked_image
plt.imshow(final_image)

################################################################################################

plt.show()