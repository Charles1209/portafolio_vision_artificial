import cv2
import numpy as np
import time

# Cargar nombres de las clases
rows = open('src/lab8/Restnet_Pesos/synset_words.txt').read().strip().split('\n')
classes = [r[r.find(' ') + 1:].split(',')[0] for r in rows]

# Cargar el modelo Caffe preentrenado y la arquitectura ResNet-50
net = cv2.dnn.readNetFromCaffe("src/lab8/Restnet_Pesos/ResNet-50-deploy.prototxt", 
                               "src/lab8/Restnet_Pesos/ResNet-50-model.caffemodel")

# Capturar video desde la cámara
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: No se puede abrir la cámara")
    exit()

while True:
    # Leer un cuadro de la cámara
    ret, frame = cap.read()
    if not ret:
        print("Error: No se puede recibir el cuadro. Saliendo...")
        break
    
    # Crear el blob de entrada con el tamaño (224,224) y la media (104, 117, 123)
    blob = cv2.dnn.blobFromImage(frame, 1, (224, 224), (104, 117, 123))
    net.setInput(blob)
    preds = net.forward()

    # Obtener la predicción con mayor probabilidad
    idx = np.argmax(preds[0])
    label = classes[idx]
    prob = preds[0][idx] * 100

    # Dibujar la etiqueta y la probabilidad en la imagen
    text = f"Label: {label}, Prob: {prob:.2f}%"
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    # Mostrar el cuadro en una ventana
    cv2.imshow('Clasificador en tiempo real', frame)

    # Capturar la tecla presionada
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('s'):  # Si se presiona 's', guardar una captura de pantalla
        timestamp = time.strftime("%Y%m%d-%H%M%S")  # Obtener un timestamp para el nombre del archivo
        filename = f"src/lab8/captura_{timestamp}.jpg"
        cv2.imwrite(filename, frame)
        print(f"Captura de pantalla guardada como {filename}")
    
    elif key == ord('q'):  # Si se presiona 'q', salir del bucle
        break

# Liberar los recursos
cap.release()
cv2.destroyAllWindows()