import cv2
from tkinter import *

# funciones que se llaman desde los botones

def open_camera():
	# Proceso para mostrar la imagen en pantalla
	video_capture = cv2.VideoCapture(0)

def cerrar_camara():
	cap.release()
	pass

# Setup de App de TKinter
app = Tk()
app.bind('<Escape>', lambda e: app.quit())
app.title("Aplicar Segmentacion por Color")

# Procesos para Centrar la ventana en la pantalla
window_width = 720
window_height = 560

# get the screen dimension
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

# Create an infinite loop for displaying app on screen
app.mainloop()