import os
import cv2
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import gradio as gr

def generarMC(model, X, y, categorias, label, ruta, normal=None):
    fig=ConfusionMatrixDisplay.from_estimator(model, X, y, display_labels=categorias, cmap="Greens", normalize=normal)  
    fig.figure_.suptitle(label)
    plt.savefig(ruta)
    plt.show()

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
        orgimage = cv2.imread(imgs)
        image = cv2.resize(orgimage, img_size)

        # Convertir la imagen a escala de grises 
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
        # Calcular los descriptores HOG de la imagen en escala de grises
        hog_descriptors = hog.compute(gray_image)
        hog_descriptor_set.append(hog_descriptors)
    
    # Remover los ejes/dimensiones de longitud 1 de la lista
    hog_descriptor_set = np.squeeze(hog_descriptor_set) 
    return hog_descriptor_set

# Configuración de directorios
base_dir = str(os.getcwd()) + "/src/lab7/PetsDataSet"
train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'test')

# Directorios de entrenamiento
train_cats_dir = os.path.join(train_dir, 'cats')
train_dogs_dir = os.path.join(train_dir, 'dogs')
train_parrots_dir = os.path.join(train_dir, 'parrots')

# Directorios de prueba
test_cats_dir = os.path.join(test_dir, 'cats')
test_dogs_dir = os.path.join(test_dir, 'dogs')
test_parrots_dir = os.path.join(test_dir, 'parrots')

# Obtener listas de archivos
train_cat_fnames = list(map(lambda x: os.path.join(train_cats_dir, x), os.listdir(train_cats_dir)))
train_dog_fnames = list(map(lambda x: os.path.join(train_dogs_dir, x), os.listdir(train_dogs_dir)))
train_parrot_fnames = list(map(lambda x: os.path.join(train_parrots_dir, x), os.listdir(train_parrots_dir)))

test_cat_fnames = list(map(lambda x: os.path.join(test_cats_dir, x), os.listdir(test_cats_dir)))
test_dog_fnames = list(map(lambda x: os.path.join(test_dogs_dir, x), os.listdir(test_dogs_dir)))
test_parrot_fnames = list(map(lambda x: os.path.join(test_parrots_dir, x), os.listdir(test_parrots_dir)))

# Combinar listas de archivos
train_fnames = train_cat_fnames + train_dog_fnames + train_parrot_fnames
test_fnames = test_cat_fnames + test_dog_fnames + test_parrot_fnames

# Preparar etiquetas
train_labels = [0] * len(train_cat_fnames) + [1] * len(train_dog_fnames) + [2] * len(train_parrot_fnames)
test_labels = [0] * len(test_cat_fnames) + [1] * len(test_dog_fnames) + [2] * len(test_parrot_fnames)

# Obtener características HOG
train_hog = get_hog(train_fnames)
test_hog = get_hog(test_fnames)

# Definir y entrenar el clasificador MLP
clf_mlp = MLPClassifier(
    hidden_layer_sizes=(100, 50),  # Dos capas ocultas
    activation='relu',             # Función de activación ReLU
    solver='adam',                 # Optimizador Adam
    max_iter=1000,                # Máximo número de iteraciones
    random_state=0,               # Semilla aleatoria para reproducibilidad
    verbose=True                  # Mostrar progreso del entrenamiento
)

# Entrenar el modelo
clf_mlp.fit(train_hog, np.array(train_labels))

# Evaluar el modelo
print(f'Precisión del modelo MLP: {clf_mlp.score(test_hog, np.array(test_labels)):.2%}')

# Generar matrices de confusión
# No normalizada
generarMC(
    clf_mlp, 
    test_hog, 
    test_labels, 
    ['cat', 'dog', 'parrot'], 
    "Matriz de confusión MLP no normalizada",
    "src/lab7/out/lab7_3_matriz_de_confusion_mlp_no_normalizada.png"
)

# Normalizada
generarMC(
    clf_mlp, 
    test_hog, 
    test_labels, 
    ['cat', 'dog', 'parrot'], 
    "Matriz de confusión MLP normalizada",
    "src/lab7/out/lab7_3_matriz_de_confusion_mlp_normalizada.png",
    'all'
)

# Matriz de confusión para datos de entrenamiento
generarMC(
    clf_mlp, 
    train_hog, 
    train_labels, 
    ['cat', 'dog', 'parrot'], 
    "Matriz de confusión MLP train normalizada",
    "src/lab7/out/lab7_3_matriz_de_confusion_mlp_train_normalizada.png",
    'all'
)

# Función de clasificación para Gradio
def clasificar(path):
    hogdesc = get_hog([path])
    labels = ['cat', 'dog', 'parrot']
    res = clf_mlp.predict(hogdesc.reshape(1, -1))
    resprob = clf_mlp.predict_proba(hogdesc.reshape(1, -1))
    labelresultado = f'en la imagen aparece un {labels[res[0]]}, con una probabilidad de {resprob[0,res[0]]:.2%}'
    return labelresultado

# Interfaz Gradio
gr.Interface(
    fn=clasificar,
    inputs=gr.Image(type="filepath"),
    outputs="textbox"
).launch(share=True)