import os
import cv2
import numpy as np

#Definición de los directorios del dataset
#base_dir = r'C:\Users\jockr\Desarrollo\petsDataset'
#base_dir = "/home/phoenix/git/vision_artificial_linux/src/lab7/PetsDataSet"
base_dir = str(os.getcwd()) + "/src/lab7/PetsDataSet"
train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'test')

# Directorio con las imagenes de training 
train_cats_dir = os.path.join(train_dir, 'cats')
train_dogs_dir = os.path.join(train_dir, 'dogs')
train_parrots_dir = os.path.join(train_dir, 'parrots')

# Directorio con las imagenes de test
test_cats_dir = os.path.join(test_dir, 'cats')
test_dogs_dir = os.path.join(test_dir, 'dogs')
test_parrots_dir = os.path.join(test_dir, 'parrots')

# Lista de Nombres de Archivos de Train
train_cat_fnames = os.listdir(train_cats_dir )
train_dog_fnames = os.listdir( train_dogs_dir )
train_parrot_fnames = os.listdir( train_parrots_dir )

# Lista de Nombres de Archivos de Test
test_cat_fnames = os.listdir( test_cats_dir )
test_dog_fnames = os.listdir( test_dogs_dir )
test_parrot_fnames = os.listdir( test_parrots_dir )

# Lista de Rutas de archivos de train
train_cat_fnames = list(map(lambda x: os.path.join(train_cats_dir, x) , train_cat_fnames)) 
train_dog_fnames = list(map(lambda x: os.path.join(train_dogs_dir, x) , train_dog_fnames)) 
train_parrot_fnames = list(map(lambda x: os.path.join(train_parrots_dir, x) , train_parrot_fnames)) 


# Lista de Rutas de archivos de test
test_cat_fnames = list(map(lambda x: os.path.join(test_cats_dir, x) , test_cat_fnames))
test_dog_fnames = list(map(lambda x: os.path.join(test_dogs_dir, x) , test_dog_fnames))
test_parrot_fnames = list(map(lambda x: os.path.join(test_parrots_dir, x) , test_parrot_fnames))


# Lista con todas las paths de archivos de train
train_fnames = train_cat_fnames + train_dog_fnames + train_parrot_fnames

# Lista con todas las paths de archivos de test
test_fnames = test_cat_fnames + test_dog_fnames + test_parrot_fnames


# Prepara las etiquetas de cada instancia de train
train_labels =[ 0 for _ in train_cat_fnames ]
train_labels.extend([ 1 for _ in train_dog_fnames ])
train_labels.extend([ 2 for _ in train_parrot_fnames ])


# Prepara las etiquetas de cada instancia de test
test_labels =[ 0 for _ in test_cat_fnames ]
test_labels.extend([ 1 for _ in test_dog_fnames ])
test_labels.extend([ 2 for _ in test_parrot_fnames ])

################################################################################################

from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def generarMC(model, X, y, categorias, label, ruta, normal=None):
	fig=ConfusionMatrixDisplay.from_estimator(model, X, y, display_labels=categorias, cmap="Greens", normalize=normal)  
	fig.figure_.suptitle(label)
	plt.savefig(ruta)
	plt.show()
	
def generarMCPredictions(y_real, y_pred, categorias ):
	fig=ConfusionMatrixDisplay.from_estimator(y_real, y_pred, display_labels=categorias, cmap="Greens")  
	fig.figure_.suptitle("Confusion Matrix")
	plt.savefig("src/lab7/out/lab7_3.2.png")
	plt.show()

################################################################################################

def get_hog(list_fnames):
	# Definir Resolucion de las imagenes a usar
	img_size =(380,380)

	# Definir los parámetros del descriptor HOG
	win_size = img_size     # Tamaño de la ventana de detección
	block_size = (40, 40)   # Tamaño del bloque
	block_stride = (20, 20) # Desplazamiento del bloque
	cell_size = (20, 20)    # Tamaño de la celda
	nbins = 9               # Número de bins del histograma

	# Crear un descriptor HOG con parámetros personalizados
	hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)

	hog_descriptor_set = []
	for imgs in list_fnames:
		# Cargar la imagen 
		orgimage = cv2.imread(  imgs  )
		image = cv2.resize(orgimage, img_size)

		# Convertir la imagen a escala de grises 
		gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	
		# Calcular los descriptores HOG de la imagen en escala de grises
		hog_descriptors = hog.compute(gray_image)
		hog_descriptor_set.append(hog_descriptors)
	
	# Remover los ejes/dimensiones de longitud 1 de la lista
	hog_descriptor_set = np.squeeze(hog_descriptor_set) 
	return hog_descriptor_set
	   
################################################################################################
 
train_hog = get_hog(train_fnames)
test_hog = get_hog(test_fnames)
 
################################################################################################

from sklearn.ensemble import RandomForestClassifier

#Definición del Modelo 
clfRF = RandomForestClassifier(n_jobs=6, random_state=0, n_estimators=1000)
#fit/train -> Se hace el entrenamiendo del modelo
clfRF.fit(train_hog, np.array(train_labels))

################################################################################################

# Precisión lograda por el clasificador
print(f'{clfRF.score(test_hog, np.array(test_labels)):.2%}')

################################################################################################

ruta = "src/lab7/out/lab7_3_matriz_de_confusión_no_normalizada.png"
label = "Matriz de confusión no normalizada"

# Matriz de confusión no normalizada
generarMC(clfRF, test_hog, test_labels, ['cat', 'dog','parrot'], label, ruta)

################################################################################################

ruta = "src/lab7/out/lab7_3_matriz_de_confusión_normalizada.png"
label = "Matriz de confusión normalizada"

# Matriz de confusión normalizada
generarMC(clfRF, test_hog, test_labels, ['cat', 'dog','parrot'], label, ruta, 'all')

################################################################################################

ruta = "src/lab7/out/lab7_3_matriz_de_confusión_train_normalizada.png"
label = "Matriz de confusión train normalizada"

generarMC(clfRF, train_hog, train_labels, ['cat', 'dog','parrot'], label, ruta, 'all')

################################################################################################

def clasificar(path):
	global clfRF
	hogdesc = get_hog([path])
	labels = ['cat', 'dog','parrot']
	res = clfRF.predict(hogdesc.reshape(1, -1))
	resprob = clfRF.predict_proba(hogdesc.reshape(1, -1))
	labelresultado = f'en la imagen aparece un {labels[res[0]]}, con una probalidad de { resprob[0,res[0]]:.2%}' 
	return labelresultado

################################################################################################

import gradio as gr

gr.Interface(
	fn = clasificar,    # funcion 
	inputs = gr.Image(type="filepath"), # tipo de entrada
	outputs = "textbox"  
).launch(share=True)