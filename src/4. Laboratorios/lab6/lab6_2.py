#import OpenCV module
import cv2
#import os module for reading training data directories and paths
import os
#import numpy to convert python lists to numpy arrays as 
#it is needed by OpenCV face recognizers
import numpy as np

################################################################################################

#there is no label 0 in our training data so subject name for index/label 0 is empty
#subjects = ["", "Ruben Blades", "Elvis Presley", "Liam Payne"]
subjects = ["", "Ruben Blades", "Elvis Presley"]

################################################################################################

# El .xml del profe no sirve, hay que descargarse el de internet
face_cascade_path = "src/lab6/Lab6_C_Files/opencv-files/lbpcascade_frontalface.xml"

# face_cascade = cv2.CascadeClassifier(face_cascade_path)
# os.chmod(face_cascade_path, 0o666)
# fs = cv2.FileStorage(face_cascade_path, cv2.FILE_STORAGE_WRITE)

# if not os.path.exists(face_cascade_path):
# 	raise FileNotFoundError(f"The file {face_cascade_path} does not exist.")
# elif face_cascade.empty():
# 	raise IOError(f"Unable to load the face cascade classifier from {face_cascade_path}")
# elif not fs.isOpened():
# 	raise IOError(f"No se puede abrir el archivo {face_cascade_path} en modo lectura.")

#function to detect face using OpenCV
def detect_face(img):
	#convert the test image to gray image as opencv face detector expects gray images
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	
	#load OpenCV face detector, I am using LBP which is fast
	#there is also a more accurate but slow Haar classifier
	#face_cascade = cv2.CascadeClassifier('opencv-files/lbpcascade_frontalface.xml')
	face_cascade = cv2.CascadeClassifier(face_cascade_path)

	# if face_cascade.empty():
	# 	raise IOError(f"Unable to load the face cascade classifier from {face_cascade_path}")

	#let's detect multiscale (some images may be closer to camera than others) images
	#result is a list of faces
	faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
	
	#if no faces are detected then return original img
	if (len(faces) == 0):
		return None, None
	
	#under the assumption that there will be only one face,
	#extract the face area
	(x, y, w, h) = faces[0]
	
	#return only the face part of the image
	return gray[y:y+w, x:x+h], faces[0]

################################################################################################

#this function will read all persons' training images, detect face from each image
#and will return two lists of exactly same size, one list 
# of faces and another list of labels for each face
def prepare_training_data(data_folder_path):
	
	#------STEP-1--------
	#get the directories (one directory for each subject) in data folder
	dirs = os.listdir(data_folder_path)
	
	#list to hold all subject faces
	faces = []
	#list to hold labels for all subjects
	labels = []
	
	#let's go through each directory and read images within it
	for dir_name in dirs:
		
		#our subject directories start with letter 's' so
		#ignore any non-relevant directories if any
		if not dir_name.startswith("s"):
			continue
			
		#------STEP-2--------
		#extract label number of subject from dir_name
		#format of dir name = slabel
		#, so removing letter 's' from dir_name will give us label
		label = int(dir_name.replace("s", ""))
		
		#build path of directory containin images for current subject subject
		#sample path_dir_subject = "training_data/s1"
		path_dir_subject = data_folder_path + "/" + dir_name
		
		#get the images names that are inside the given subject directory
		subject_images_names = os.listdir(path_dir_subject)
		
		#------STEP-3--------
		#go through each image name, read image, 
		#detect face and add face to list of faces
		for image_name in subject_images_names:
			
			#ignore system files like .DS_Store
			if image_name.startswith("."):
				continue
			
			#build image path
			#sample image path = training_data/s1/1.pgm
			image_path = path_dir_subject + "/" + image_name

			#read image
			image = cv2.imread(image_path)

			#display an image window to show the image 
			if image is None:
				print(f"Error loading image: {image_path}")
			else:
				#print(f"Printing Image {image_path}")
				cv2.imshow("Training on image...", image)
				cv2.waitKey(100)
			
			#detect face
			face, rect = detect_face(image)
			
			#------STEP-4--------
			#for the purpose of this tutorial
			#we will ignore faces that are not detected
			if face is not None:
				#add face to list of faces
				faces.append(face)
				#add label for this face
				labels.append(label)
			
	cv2.destroyAllWindows()
	cv2.waitKey(1) 
	cv2.destroyAllWindows()
	
	return faces, labels

################################################################################################


training_data_path = "src/lab6/Lab6_C_Files/training_data"
# if not os.path.exists(training_data_path):
# 	raise FileNotFoundError(f"The directory {training_data_path} does not exist.")

# if not os.path.exists(file_path):
# 	raise FileNotFoundError(f"El archivo {file_path} no existe.")
# os.chmod(training_data_path, 0o666)

# fs = cv2.FileStorage(file_path, cv2.FILE_STORAGE_WRITE)
# if not fs.isOpened():
#     raise IOError(f"No se puede abrir el archivo {file_path} en modo escritura.")

#let's first prepare our training data
#data will be in two lists of same size
#one list will contain all the faces
#and other list will contain respective labels for each face
print("Preparing data...")
faces, labels = prepare_training_data(training_data_path)
print("Data prepared")

print(faces)
print(labels)

#print total faces and labels
print("Total faces: ", len(faces))
print("Total labels: ", len(labels))

################################################################################################

#create our LBPH face recognizer 
face_recognizer = cv2.face.LBPHFaceRecognizer_create()

#or use EigenFaceRecognizer by replacing above line with 
#face_recognizer = cv2.face.EigenFaceRecognizer_create()

#or use FisherFaceRecognizer by replacing above line with 
#face_recognizer = cv2.face.FisherFaceRecognizer_create()

################################################################################################

#train our face recognizer of our training faces
face_recognizer.train(faces, np.array(labels))

################################################################################################

#function to draw rectangle on image 
#according to given (x, y) coordinates and 
#given width and heigh
def draw_rectangle(img, rect):
	(x, y, w, h) = rect
	cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
	
#function to draw text on give image starting from
#passed (x, y) coordinates. 
def draw_text(img, text, x, y):
	cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)

################################################################################################

#this function recognizes the person in image passed
#and draws a rectangle around detected face with name of the 
#subject
def predict(test_img):
	#make a copy of the image as we don't want to chang original image
	img = test_img.copy()
	#detect face from the image
	face, rect = detect_face(img)

	#predict the image using our face recognizer 
	label= face_recognizer.predict(face)
	print(label[1])  #valor de confidence, es una distancia entre más pequeño más cerca por lo tanto mejor
	#get name of respective label returned by face recognizer
	label_text = subjects[label[0]]
	
	#draw a rectangle around face detected
	draw_rectangle(img, rect)
	#draw name of predicted person
	draw_text(img, label_text, rect[0], rect[1]-5)
	
	return img

################################################################################################

print("Predicting images...")

#load test images
test_img1 = cv2.imread("src/lab6/Lab6_C_Files/test-data/test0.jpg")
test_img2 = cv2.imread("src/lab6/Lab6_C_Files/test-data/test6.jpg")
test_img3 = cv2.imread("src/lab6/Lab6_C_Files/test-data/test9.jpg")

###Si no detecta caras en la imagen dará un error

#perform a prediction
predicted_img1 = predict(test_img1)
predicted_img2 = predict(test_img2)
#predicted_img3 = predict(test_img3)
print("Prediction complete")

#display both images
cv2.imshow(subjects[1], predicted_img1)
cv2.imshow(subjects[2], predicted_img2)
#cv2.imshow(subjects[3], predicted_img3)
cv2.waitKey(0)
cv2.destroyAllWindows()