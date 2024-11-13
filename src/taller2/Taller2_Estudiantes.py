import cv2 
import numpy as np
import matplotlib.pyplot as plt
from tkinter import *
from PIL import Image, ImageTk 

cap = cv2.VideoCapture(0) 

width, height = 640,480 #1280,720 
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width) 
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height) 

mostrarFiltro = True

def open_camera(): 
	# Capture the video frame by frame 
	_, frame = cap.read() 
	
	# Convert image from one color space to other 
	opencv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA) 

	# Capture the latest frame and transform to image 
	captured_image = Image.fromarray(opencv_image)
	
	# Convert captured image to photoimage 
	photo_image = ImageTk.PhotoImage(image=captured_image) 

	# Displaying photoimage in the label 
	label_widget.photo_image = photo_image 

	# Configure image in the label 
	label_widget.configure(image=photo_image) 

	# Repeat the same process after every 10 seconds 
	label_widget.after(10, open_camera) 

def cerrar_camara():
	cap.release()
	
def trabajar_imagen(frame):
	image = cv2.imread('img/taller2.png')
	
	escala = 0.80
	h,w,c = image.shape
	h_final = int(h*escala)
	w_final = int(w*escala)

	image_copy = np.copy(image)
	dsize = (w_final, h_final)

	# escalar imagen
	image_copy = cv2.resize(image_copy, dsize)

	# Rangos de Color
	lower_blue = np.array([100, 0, 0])
	upper_blue = np.array([255, 100, 120])

	mask = cv2.inRange(image_copy, lower_blue, upper_blue)

	masked_image = np.copy(image_copy)
	masked_image[mask != 0] = [0, 0, 0]
	plt.imshow(masked_image[:,:,::-1])

	# Se obtiene la resolucion de la imagen de entrada
	vid_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
	vid_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

	vid_dsize = (vid_width, vid_height)

	background_image = np.copy(frame)

	# se usa el tama~no de la img de entrada
	masked_image = cv2.resize(masked_image, vid_dsize)
	mask = cv2.resize(mask, vid_dsize)

	# Aplicar mascara a la imagen de entrada
	background_image[mask == 0] = [0, 0, 0]

	# Union de fondo y montaje
	final_image = background_image + masked_image

	return final_image
	
def aplicar_filtro():
	# iniciar captura
	_, frame = cap.read()

	frame = trabajar_imagen(frame)
	
	# Convert image from one color space to other 
	opencv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA) 

	# Capture the latest frame and transform to image 
	captured_image = Image.fromarray(opencv_image)
	
	# Convert captured image to photoimage 
	photo_image = ImageTk.PhotoImage(image=captured_image) 

	# Displaying photoimage in the label 
	label_widget.photo_image = photo_image 

	# Configure image in the label 
	label_widget.configure(image=photo_image) 

	# Repeat the same process after every 10 seconds 
	label_widget.after(10, open_camera) 

def quitar_filtro():
	global mostrarFiltro 
	mostrarFiltro = not mostrarFiltro

# Setup de App de TKinter
app = Tk() 
app.bind('<Escape>', lambda e: app.quit()) 
app.title("Aplicar Segmentacion por Color")

# Procesos para Centrar la ventana en la pantalla
window_width = 720
window_height = 560

# get the screen dimension  https://www.pythontutorial.net/tkinter/tkinter-window/
screen_width = app.winfo_screenwidth()
screen_height = app.winfo_screenheight()

# find the center point
center_x = int(screen_width/2 - window_width / 2)
center_y = int(screen_height/2 - window_height / 2)

# set the position of the window to the center of the screen
app.geometry(f'{window_width}x{window_height}+{center_x}+{center_y}')
app.resizable(False, False)

label_widget = Label(app) 
label_widget.grid(row=0, column=0, columnspan=3)

button1 = Button(app, text="Open Camera", command=open_camera) 
button1.grid(row=1, column=0)

button2 = Button(app, text="Close Camera", command=cerrar_camara) 
button2.grid(row=1, column=1)

button3 = Button(app, text="Aplicar Filtro", command=aplicar_filtro) 
button3.grid(row=1, column=3)

button4 = Button(app, text="Quitar Filtro", command=quitar_filtro) 
button4.grid(row=1, column=4)

# Create an infinite loop for displaying app on screen 
app.mainloop() 
