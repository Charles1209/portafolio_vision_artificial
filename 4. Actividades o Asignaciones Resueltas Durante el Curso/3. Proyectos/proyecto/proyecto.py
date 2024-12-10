import cv2
import os
import numpy as np
import subprocess
import platform

################################################################################################

sistema = platform.system()
	
path_face_cascade = str(os.getcwd()) + "/src/lbpcascade_frontalface.xml"
path_names_file = str(os.getcwd()) + "/src/nombres.txt"
path_faces_file = str(os.getcwd()) + "/src/rostros.npz"
path_cropped_faces = str(os.getcwd()) + "/src/rostros"
path_unknown_faces = str(os.getcwd()) + "/src/unknown"

################################################################################################

def detect_face(img):
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	
	face_cascade = cv2.CascadeClassifier(path_face_cascade)

	faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
	
	if (len(faces) == 0):
		return None, None
	
	(x, y, w, h) = faces[0]
	
	return gray[y:y+w, x:x+h], faces[0]

################################################################################################

def funcion_basura():
	def prepare_training_data(path_cropped_faces):
		dirs = os.listdir(path_cropped_faces)

		faces = []
		labels = []

		for dir_name in dirs:
			label = int(dir_name)

			path_dir_subject = path_cropped_faces + "/" + dir_name

			subject_images_names = os.listdir(path_dir_subject)

			for image_name in subject_images_names:
				image_path = path_dir_subject + "/" + image_name

				image = cv2.imread(image_path)

				face, rect = detect_face(image)

				if face is not None:
					faces.append(face)
					labels.append(label)

		return faces, labels

	faces, labels = prepare_training_data(path_cropped_faces)
	face_recognizer = cv2.face.LBPHFaceRecognizer_create()
	try:
		face_recognizer.train(faces, np.array(labels))
		return face_recognizer
	except:
		return None

################################################################################################

def draw_rectangle(img, rect):
	(x, y, w, h) = rect
	cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
	
def draw_text(img, text, x, y):
	cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)

def predict(test_img, subjects, face_recognizer):
	img = test_img.copy()
	face, rect = detect_face(img)

	# Ensure that a face was detected before proceeding
	if face is None:
		print("No face detected.")
		return img  # Return the original image without modifications

	try:
		label, distance = face_recognizer.predict(face)

		max_distance = 150  # Valor máximo esperado de distancia
		confidence = ((max_distance - distance) / max_distance) * 100
		confidence = max(0, min(100, confidence))  # Ensure it's between 0 and 100

		#distance_threshold = 61
		distance_threshold = 50

		draw_rectangle(img, rect)

		#if confidence < 59 and distance > distance_threshold:
		if confidence < 50 and distance > distance_threshold:
			label_text = "Desconocido"
			draw_text(img, f"{label_text}", rect[0], rect[1]-5)
		else:
			label_text = subjects[label]
			draw_text(img, f"{label_text} ({confidence:.1f}%)", rect[0], rect[1]-5)

	except cv2.error as e:
		print(f"OpenCV error during prediction: {e}")
		return img  # Return the original image without modifications

	return img

################################################################################################

def limpiar_consola():
	sistema = platform.system()
	
	if sistema == "Windows":
		os.system('cls')  # Comando para limpiar en Windows
	else:
		os.system('clear')  # Comando para limpiar en Linux y macOS

def verificar_persona(names, face_recognizer):
	camara = cv2.VideoCapture(0)

	while True:
		ret, frame = camara.read()

		imagen_predicha = predict(frame, names, face_recognizer)

		cv2.imshow('Verificando Usuario', imagen_predicha)

		key = cv2.waitKey(1) & 0xFF

		if key == ord('q') or key == 27:
			cv2.destroyAllWindows()
			camara.release()
			break
		elif key == ord("s"):
			cv2.imwrite(path_unknown_faces + "/desconocido.png", frame)

def insertar_persona(path_cropped_faces, labels, names):
	def guardar_rostros(path_cropped_faces, labels, names):
		camara = cv2.VideoCapture(0)
		nombre = None
		faces = []
	
		nombre = input("Ingrese un nombre para el nuevo usuario: ")

		while(True):
			_, frame = camara.read() # ret, frame = camara.read()
			
			rostro, _ = detect_face(frame)

			cv2.imshow("Detectando Rostro", frame)

			if rostro is not None:
				if nombre in names:
					posicion = names.index(nombre)

					if sistema == "Windows":
						path_nombre_carpeta = path_cropped_faces + "\\" + str(posicion)
					else:
						path_nombre_carpeta = path_cropped_faces + "/" + str(posicion)

					numero = os.listdir(path_nombre_carpeta)
					numero = len(numero)
					path_nombre_imagen = path_nombre_carpeta + "/" + str(numero+1) + ".png"
					labels.append(int(posicion))
				else:
					cantidad = os.listdir(path_cropped_faces)
					cantidad = len(cantidad)

					path_nombre_carpeta = path_cropped_faces + "/" + str(cantidad+1)
					os.makedirs(path_nombre_carpeta)
					path_nombre_imagen = path_nombre_carpeta + "/1.png"
					names.append(nombre)
					with open(path_names_file, "a") as file:
						file.write(f"\n{nombre}")
					labels.append(cantidad+1)

				try:
					with np.load(path_faces_file) as data:
						faces = [data[key] for key in data.files]
				except FileNotFoundError:
					pass

				faces.append(rostro)
				np.savez(path_faces_file, *faces)

				cv2.imwrite(path_nombre_imagen, rostro)
			else:
				pass
	
			if (cv2.waitKey(1) == 27) or (cv2.waitKey(1) & 0xFF == ord('q')): #Cuando oprimamos "Escape" o "q" rompe el video
				cv2.destroyAllWindows()
				camara.release() #Cerramos
				break

		return faces, labels, names
	
	def reconocer_rostros(faces, labels):
		face_recognizer = cv2.face.LBPHFaceRecognizer_create()

		face_recognizer.train(faces, np.array(labels, dtype=np.int32))

	faces, labels, names = guardar_rostros(path_cropped_faces, labels, names)

	reconocer_rostros(faces, labels)

def ver_identificados(path_cropped_faces):
	if sistema == "Windows":
		os.startfile(path_cropped_faces)  # Para Windows
	elif sistema == "Linux":
		subprocess.run(['xdg-open', path_cropped_faces])  # Para Linux
	elif sistema == "Darwin":  # macOS
		subprocess.run(['open', path_cropped_faces])      # Para macOS

def funcion_inicial():
	if not os.path.exists(path_cropped_faces):
		os.makedirs(path_cropped_faces)
	
	if not os.path.exists(path_unknown_faces):
		os.makedirs(path_unknown_faces)

	names = []
	if not os.path.isfile(path_names_file):
		names.append("")
		with open(path_names_file, "w") as file:
			file.write("")
	else:
		with open(path_names_file, "r") as file:
			words_from_file = [line.strip() for line in file.readlines()]
		names = words_from_file

	faces = [] # esto es tmp
	labels = []

	name_faces = os.listdir(path_cropped_faces)
	
	for number_face in name_faces:
		path_number_file = path_cropped_faces + "/" + number_face
		number_file = os.listdir(path_number_file)

		for file in number_file:
			if file.startswith("."):
				continue

			image_path = path_number_file + "/" + file

			image = cv2.imread(image_path)
			faces.append(image)
			labels.append(int(number_face))
	
	face_recognizer = funcion_basura()

	return path_cropped_faces, labels, names, face_recognizer

path_cropped_faces, labels, names, face_recognizer = funcion_inicial()

if __name__ == "__main__":
	while True:
		#limpiar_consola()
		print("\nBienvenido")
		print("\n1. Verificar persona.")
		print("2. Insertar persona.")
		print("3. Ver personas identificadas.")
		print("0. Salir.")
		opcion = int(input("\nInserte un número del menú: "))

		match opcion:
			case 0:
				print("\nHasta pronto.\n")
				break
			case 1:
				path_cropped_faces, labels, names, face_recognizer = funcion_inicial()

				if face_recognizer is None:
					limpiar_consola()
					print("\nSin rostros en la base de datos. No se reconocerá ningún usuario.")
				else:
					verificar_persona(names, face_recognizer)
			case 2: 
				insertar_persona(path_cropped_faces, labels, names)
			case 3:
				ver_identificados(path_cropped_faces)
			case _: 
				while (opcion < 0 or opcion > 3):
					print("Por favor inserte una opción válida.")
					opcion = int(input("\nInserte un número del menú: "))