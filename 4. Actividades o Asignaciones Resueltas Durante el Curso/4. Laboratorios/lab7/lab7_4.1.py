import cv2
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

# Constants:
SIZE_IMAGE = 20
NUMBER_CLASSES = 10

################################################################################################

def load_digits_and_labels(big_image):
    """ Returns all the digits from the 'big' image and creates the corresponding labels for each image"""

    # Load the 'big' image containing all the digits:
    digits_img = cv2.imread(big_image, 0)

    # Get all the digit images from the 'big' image:
    number_rows = digits_img.shape[1] / SIZE_IMAGE
    rows = np.vsplit(digits_img, digits_img.shape[0] / SIZE_IMAGE)

    digits = []
    for row in rows:
        row_cells = np.hsplit(row, number_rows)
        for digit in row_cells:
            digits.append(digit)
    digits = np.array(digits)

    # Create the labels for each image:
    labels = np.repeat(np.arange(NUMBER_CLASSES), len(digits) / NUMBER_CLASSES)
    return digits, labels

################################################################################################

def deskew(img):
    """Pre-processing of the images"""

    m = cv2.moments(img)
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = m['mu11'] / m['mu02']
    M = np.float32([[1, skew, -0.5 * SIZE_IMAGE * skew], [0, 1, 0]])
    img = cv2.warpAffine(img, M, (SIZE_IMAGE, SIZE_IMAGE), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
    return img

################################################################################################

def get_hog():
    """Get hog descriptor"""

    # cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins, derivAperture, winSigma, histogramNormType,
    # L2HysThreshold, gammaCorrection, nlevels, signedGradient)
    hog = cv2.HOGDescriptor((SIZE_IMAGE, SIZE_IMAGE), (8, 8), (4, 4), (8, 8), 9, 1, -1, 0, 0.2, 1, 64, True)

    print("get descriptor size: {}".format(hog.getDescriptorSize()))

    return hog

################################################################################################

def raw_pixels(img):
    """Return raw pixels as feature from the image"""

    return img.flatten()

################################################################################################

# Load all the digits and the corresponding labels:
digits, labels = load_digits_and_labels('src/lab7/Lab7_ML/digits.png')

# Shuffle data
# Constructs a random number generator:
rand = np.random.RandomState(1234)
# Randomly permute the sequence:
shuffle = rand.permutation(len(digits))
digits, labels = digits[shuffle], labels[shuffle]

################################################################################################

# HoG feature descriptor:
hog = get_hog()

# Compute the descriptors for all the images.
# In this case, the HoG descriptor is calculated
hog_descriptors = []
for img in digits:
    hog_descriptors.append(hog.compute(deskew(img)))
hog_descriptors = np.squeeze(hog_descriptors)

################################################################################################

# At this point we split the data into training and testing (70% for training):
hog_descriptors_train, hog_descriptors_test, labels_train, labels_test = train_test_split(hog_descriptors,labels,random_state=1, test_size=0.3)

################################################################################################

# Modificación 1: Cambiar capas ocultas
clf_mod1 = MLPClassifier(hidden_layer_sizes=(512, 256), activation="relu", random_state=25)
clf_mod1.fit(hog_descriptors_train, labels_train)

# Accuracy de entrenamiento y prueba
accuracy_train_mod1 = clf_mod1.score(hog_descriptors_train, labels_train)
accuracy_test_mod1 = clf_mod1.score(hog_descriptors_test, labels_test)

# Matriz de confusión para el conjunto de entrenamiento
ConfusionMatrixDisplay.from_estimator(clf_mod1, hog_descriptors_train, labels_train, normalize='true', cmap='Blues')
plt.title("Matriz de Confusión para Entrenamiento - Modificación 1\nCapas Ocultas: (512, 128), Activación: relu\nAccuracy Entrenamiento: {:.2f}".format(accuracy_train_mod1))
plt.savefig("src/lab7/out/lab7_4.1.png")
plt.show()

# Matriz de confusión para el conjunto de prueba
ConfusionMatrixDisplay.from_estimator(clf_mod1, hog_descriptors_test, labels_test, normalize='true', cmap='Blues')
plt.title("Matriz de Confusión para Prueba - Modificación 1\nCapas Ocultas: (512, 128), Activación: relu\nAccuracy Prueba: {:.2f}".format(accuracy_test_mod1))
plt.savefig("src/lab7/out/lab7_4.2.png")
plt.show()

################################################################################################

# Modificación 2: Cambiar activación y tasa de aprendizaje
clf_mod2 = MLPClassifier(hidden_layer_sizes=(256, 128, 64, 32), activation="tanh", learning_rate_init=0.01, random_state=25)
clf_mod2.fit(hog_descriptors_train, labels_train)

# Accuracy de entrenamiento y prueba
accuracy_train_mod2 = clf_mod2.score(hog_descriptors_train, labels_train)
accuracy_test_mod2 = clf_mod2.score(hog_descriptors_test, labels_test)

# Matriz de confusión para el conjunto de entrenamiento
ConfusionMatrixDisplay.from_estimator(clf_mod2, hog_descriptors_train, labels_train, normalize='true', cmap='Purples')
plt.title("Matriz de Confusión para Entrenamiento - Modificación 2\nCapas Ocultas: (256, 128, 64, 32), Activación: tanh, LR: 0.01\nAccuracy Entrenamiento: {:.2f}".format(accuracy_train_mod2))
plt.savefig("src/lab7/out/lab7_4.3.png")
plt.show()

# Matriz de confusión para el conjunto de prueba
ConfusionMatrixDisplay.from_estimator(clf_mod2, hog_descriptors_test, labels_test, normalize='true', cmap='Purples')
plt.title("Matriz de Confusión para Prueba - Modificación 2\nCapas Ocultas: (256, 128, 64, 32), Activación: tanh, LR: 0.01\nAccuracy Prueba: {:.2f}".format(accuracy_test_mod2))
plt.savefig("src/lab7/out/lab7_4.4.png")
plt.show()