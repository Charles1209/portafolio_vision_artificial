# Este código emplea un modelo CNN ya entrenado para realizar inferencias fuera del código de entrenamiento

import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

path = "img/lab8_1.jpeg"
img=image.load_img(path, target_size=(150, 150))

x=image.img_to_array(img)
imageFile=np.expand_dims(x, axis=0)

import json
a_file = open("src/lab8/pets_indices.json", "r")
class_indices = json.load(a_file)
model = load_model('src/lab8/pets_categorical.h5')

classes = model.predict(imageFile)
print(classes)
clas =np.where(classes[0] == max(classes[0]))
categoria = clas[0][0]
print(categoria)
print(list(class_indices.keys())[list(class_indices.values()).index(categoria)])

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
plt.imshow(img)
plt.show()

def clasificar(imagePath):
	global model
	img=image.load_img(imagePath, target_size=(150, 150))
	x=image.img_to_array(img)
	imgFile=np.expand_dims(x, axis=0)
	classes = model.predict(imgFile)
	clas =np.where(classes[0] == max(classes[0]))
	categoria = clas[0][0]
	labelCategoria= (list(class_indices.keys())[list(class_indices.values()).index(categoria)])
	labelResultado = f'En la imagen aparece un {labelCategoria}.'
	return labelResultado

import gradio as gr

gr.Interface(
	fn = clasificar, # funcion
	inputs = gr.Image(type="filepath"), # tipo de entrada
	outputs = "textbox"
).launch()