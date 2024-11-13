# No sirve

import cv2
import os
import numpy as np
import sys
import subprocess
import platform

def detect_face(img):
	face_cascade_path = "src/lab6/Lab6_C_Files/opencv-files/lbpcascade_frontalface.xml"

	#convert the test image to gray image as opencv face detector expects gray images
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	
	#load OpenCV face detector, I am using LBP which is fast
	face_cascade = cv2.CascadeClassifier(face_cascade_path)

	#let's detect multiscale (some images may be closer to camera than others) images
	# the result is a list of faces
	faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
	
	#if no faces are detected then return original img
	if (len(faces) == 0):
		return None, None
	
	#under the assumption that there will be only one face, extract the face area
	(x, y, w, h) = faces[0]
	
	#return only the face part of the image
	return gray[y:y+w, x:x+h], faces[0]

# Esto va para cuando quiero guardar un rostro
def prepare_training_data(path_training_data, path_cropped_faces):
	#------STEP-1--------
	#get the directories (one directory for each subject) in data folder
	dirs = os.listdir(path_training_data)
	print(f"Variable dirs en prepare_training_data: {dirs}")
	
	#list to hold all subject faces
	faces = []
	#list to hold labels for all subjects
	labels = []
	
	#let's go through each directory and read images within it
	for dir_name in dirs:
		
		#our subject directories start with letter 's' so ignore any non-relevant directories if any
		if not dir_name.startswith("s"):
			continue
	
		#------STEP-2--------
		# extract label number of subject from dir_name format of dir name = slabel, so removing letter 's'
		# from dir_name will give us label
		label = int(dir_name.replace("s", ""))
		
		#build path of directory containin images for current subject sample path_dir_subject = "training_data/s1"
		path_dir_subject = path_training_data + "/" + dir_name
		
		#get the images names that are inside the given subject directory
		subject_images_names = os.listdir(path_dir_subject)

		# Estudiar esto
		path_subject_save = os.path.join(path_cropped_faces, dir_name)
		if not os.path.exists(path_subject_save):
			os.makedirs(path_subject_save)
		
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
			face, _ = detect_face(image)
			
			#------STEP-4--------
			#for the purpose of this tutorial we will ignore faces that are not detected
			if face is not None:
				#add face to list of faces
				faces.append(face)
				#add label for this face
				labels.append(label)

				# Estudiar esto
				save_path = os.path.join(path_subject_save, image_name)
				cv2.imwrite(save_path, face)
			
	#cv2.destroyAllWindows()
	#cv2.waitKey(1) 
	cv2.destroyAllWindows()
	
	return faces, labels

#function to draw rectangle on image according to given (x, y) coordinates and given width and heigh
def draw_rectangle(img, rect):
	(x, y, w, h) = rect
	cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
	
#function to draw text on give image starting from passed (x, y) coordinates. 
def draw_text(img, text, x, y):
	cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)

#this function recognizes the person in image passed
#and draws a rectangle around detected face with name of the subject
def predict(test_img, subjects):
	#make a copy of the image as we don't want to chang original image
	img = test_img.copy()

	#detect face from the image
	face, rect = detect_face(img)

	# mostrar imagen de prueba recortada
	cv2.imshow("Mostrando rostro de predicción", face)

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

def limpiar_consola():
	sistema = platform.system()
	
	if sistema == "Windows":
		os.system('cls')  # Comando para limpiar en Windows
	else:
		os.system('clear')  # Comando para limpiar en Linux y macOS

def verificar_persona():
	pass

def insertar_persona(subjects):
	camara = cv2.VideoCapture(0)

	while(True):
		ret, frame = camara.read()
		cv2.imshow('Login Facial',frame)         #Mostramos el video en pantalla
		if (cv2.waitKey(1) == 27) or (cv2.waitKey(1) & 0xFF == ord('q')): #Cuando oprimamos "Escape" o "q" rompe el video
			camara.release() #Cerramos
			cv2.destroyAllWindows()
			break

	rostro, _ = detect_face(frame)
	cv2.imshow("Rostro Detectado", rostro)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

	#subjects.append()

def ver_identificados(path_cropped_faces):
	# Detectar el sistema operativo
	sistema = platform.system()
	print(sistema)
	
	if sistema == "Windows":
		os.startfile(path_cropped_faces)  # Para Windows
	elif sistema == "Linux":
		subprocess.run(['xdg-open', path_cropped_faces])  # Para Linux
	elif sistema == "Darwin":  # macOS
		subprocess.run(['open', path_cropped_faces])      # Para macOS

def funcion_init():
	path_init = "src/proyecto/init"
	path_training_data = "src/lab6_bryan/Lab6_C_Files/training_data"
	path_cropped_faces = "src/lab6_bryan/Lab6_C_Files/cropped_faces"
	subjects = []

	if not os.path.exists(path_init):
		os.makedirs(path_init)
		subjects.append("")
	if not os.path.exists(path_cropped_faces):
		os.makedirs(path_cropped_faces)
	if not os.path.exists(path_training_data):
		os.makedirs(path_training_data)
	
	return path_training_data, path_cropped_faces, subjects

if __name__ == "__main__":
	path_training_data, path_cropped_faces, subjects = funcion_init()

################################################################################################

	#let's first prepare our training data data will be in two lists of same size
	#one list will contain all the faces and other list will contain respective labels for each face
	print("Preparing data...")
	faces, labels = prepare_training_data(path_training_data, path_cropped_faces)
	with open("src/proyecto/hola", "w") as file:
		for face in faces:
			file.write(str(face) + "\n")

	print("Data prepared")

	#print total faces and labels
	print("Total faces: ", len(faces))
	print("Total labels: ", len(labels))
	print(f"Faces: {faces}")
	print(f"Labels: {labels}")

	#create our LBPH face recognizer 
	face_recognizer = cv2.face.LBPHFaceRecognizer_create()

	#or use EigenFaceRecognizer by replacing above line with 
	#face_recognizer = cv2.face.EigenFaceRecognizer_create()

	#or use FisherFaceRecognizer by replacing above line with 
	#face_recognizer = cv2.face.FisherFaceRecognizer_create()

	#train our face recognizer of our training faces
	face_recognizer.train(faces, np.array(labels))

	print("Predicting images...")

	#there is no label 0 in our training data so subject name for index/label 0 is empty
	subjects = ["", "Ruben Blades", "Elvis Presley"]
	subjects = ["", "Elvis Presly", "Ruben Blades"]

	#load test images
	test_img1 = cv2.imread("src/lab6_bryan/Lab6_C_Files/test-data/test0.jpg")
	test_img2 = cv2.imread("src/lab6_bryan/Lab6_C_Files/test-data/test6.jpg")

	###Si no detecta caras en la imagen dará un error

	#perform a prediction
	predicted_img1 = predict(test_img1, subjects)
	predicted_img2 = predict(test_img2, subjects)
	print("Prediction complete")

	#display both images
	cv2.imshow(subjects[1], predicted_img1)
	cv2.imshow(subjects[2], predicted_img2)
	#cv2.imshow(subjects[3], predicted_img3)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

################################################################################################

	while True:
		#limpiar_consola()
		print("\nBienvenido")
		print("\n1. Verificar persona.")
		print("2. Insertar persona.")
		print("3. Ver personas identificadas.")
		print("0. Salir.")
		opcion = int(input("\nInserte un número del menú: "))

		match opcion:
			case 0: sys.exit("\nHasta Pronto.\n")
			case 1: verificar_persona()
			case 2: insertar_persona(subjects)
			case 3: ver_identificados(path_cropped_faces)
			case _: 
				while (opcion < 0 or opcion > 3):
					print("Por favor inserte una opción válida.")
					opcion = int(input("\nInserte un número del menú: "))