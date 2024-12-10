# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 22:28:34 2024

@author: patri
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Función para mostrar los histogramas de los descriptores
def show_histogram(descriptors, title):
    bins = list(range(descriptors.shape[1]))
    fig, axs = plt.subplots(4, 4, figsize=(15, 10), sharex='col', sharey='row')
    fig.suptitle(title, fontsize=19)
    
    for i in range(min(16, descriptors.shape[0])):  # Asegurarse de que hay al menos 16 descriptores
        val = descriptors[i]
        pos = i + 1
        ax = axs[(pos-1) // 4, (pos-1) % 4]
        ax.bar(bins, val, width=0.6, color='#0504aa', alpha=0.7)
        ax.set_xlim(min(bins), max(bins))

    plt.tight_layout()
    plt.show()

# Cargar imagen 
img_path = 'img/maybeth.jpg'
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

# Verificar si la imagen se ha cargado correctamente
if img is None:
    print(f"Error: no se pudo cargar la imagen desde {img_path}. Verifica la ruta.")
else:
    # Usar STAR como detector y BRIEF como descriptor
    star = cv2.xfeatures2d.StarDetector_create()
    brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()

    # Detectar keypoints usando STAR
    keypoints_star = star.detect(img, None)
    
    # Describir keypoints usando BRIEF
    keypoints_star, descriptors_brief = brief.compute(img, keypoints_star)

    # Dibujar keypoints en la imagen original
    img_keypoints_star = cv2.drawKeypoints(img, keypoints_star, None, color=(0, 255, 0))

    # Mostrar la imagen con los keypoints detectados
    plt.imshow(img_keypoints_star, cmap='gray')
    plt.title('Keypoints detectados (STAR + BRIEF)')
    plt.axis('off')
    plt.show()

    # Mostrar los histogramas de los 16 primeros descriptores si existen
    if descriptors_brief is not None and descriptors_brief.shape[0] >= 16:
        show_histogram(descriptors_brief, "Histogramas de descriptores (STAR + BRIEF)")
    else:
        print("No hay suficientes descriptores para mostrar histogramas.")
