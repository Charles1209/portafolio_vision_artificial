# Import required packages:
import cv2
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def deskew(img):
	"""Pre-processing of the images"""
	m = cv2.moments(img)
	if abs(m['mu02']) < 1e-2:
		return img.copy()
	skew = m['mu11'] / m['mu02']
	M = np.float32([[1, skew, -0.5 * SIZE_IMAGE * skew], [0, 1, 0]])
	img = cv2.warpAffine(img, M, (SIZE_IMAGE, SIZE_IMAGE), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
	return img

def svm_init(C=12.5, gamma=0.50625):
	"""Creates empty model and assigns main parameters"""
	model = cv2.ml.SVM_create()
	model.setGamma(gamma)
	model.setC(C)
	model.setKernel(cv2.ml.SVM_RBF)
	model.setType(cv2.ml.SVM_C_SVC)
	model.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6))
	return model

def svm_train(model, samples, responses):
	"""Trains the model using the samples and the responses"""
	model.train(samples, cv2.ml.ROW_SAMPLE, responses)
	return model

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

def plot_confusion_matrices(model, X_train, y_train, X_test, y_test):
	"""Plot confusion matrices for both training and testing data"""
	
	# Get predictions for training and testing data
	y_train_pred = model.predict(X_train)[1].ravel()
	y_test_pred = model.predict(X_test)[1].ravel()
	
	# Create figure with two subplots
	fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
	
	# Plot training confusion matrix
	ConfusionMatrixDisplay.from_predictions(
		y_train, 
		y_train_pred,
		ax=ax1,
		cmap='Blues',
		normalize='true',
		display_labels=range(10)
	)
	ax1.set_title('Matriz de Confusión (Training)')
	
	# Plot testing confusion matrix
	ConfusionMatrixDisplay.from_predictions(
		y_test, 
		y_test_pred,
		ax=ax2,
		cmap='Blues',
		normalize='true',
		display_labels=range(10)
	)
	ax2.set_title('Matriz de Confusión (Testing)')
	
	plt.tight_layout()
	plt.savefig("src/lab7/out/lab7_1.1.pdf")
	plt.show()

# Constants:
SIZE_IMAGE = 20
NUMBER_CLASSES = 10

# Load and prepare data
digits, labels = load_digits_and_labels('src/lab7/Lab7_ML/digits.png')

# Shuffle data
rand = np.random.RandomState(1234)
shuffle = rand.permutation(len(digits))
digits, labels = digits[shuffle], labels[shuffle]

# HoG feature descriptor
hog = cv2.HOGDescriptor((SIZE_IMAGE, SIZE_IMAGE), (8, 8), (4, 4), (8, 8), 9, 1, -1, 0, 0.2, 1, 64, True)

# Compute descriptors
hog_descriptors = []
for img in digits:
	img_deskewed = deskew(img)
	hog_descriptors.append(hog.compute(img_deskewed))
hog_descriptors = np.squeeze(hog_descriptors)

# Split data
partition = int(0.5 * len(hog_descriptors))
hog_descriptors_train, hog_descriptors_test = np.split(hog_descriptors, [partition])
labels_train, labels_test = np.split(labels, [partition])

# Train model
model = svm_init(C=12.5, gamma=0.50625)
svm_train(model, hog_descriptors_train, labels_train)

# Generate and plot confusion matrices
plot_confusion_matrices(model, hog_descriptors_train, labels_train, hog_descriptors_test, labels_test)