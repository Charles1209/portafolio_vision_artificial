# Importación de librerías
from PIL import Image
from PIL import ImageTk
import cv2
import imutils
import customtkinter
import numpy as np
import math
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions

# Variables globales
cap = None
det = False

# Funciones de la aplicación
def iniciar():
    global cap
    cap = cv2.VideoCapture(0)
    visualizar()

def visualizar():
    global cap
    global det
    if cap is not None:
        ret, frame = cap.read()
        if ret:
            if det:
                frame = modificar(frame)
            frame = imutils.resize(frame, width=640)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            im = Image.fromarray(frame)
            img = customtkinter.CTkImage(light_image=im, size=(frame.shape[1], frame.shape[0]))
            lblVideo.configure(image=img)
            lblVideo.image = img
            lblVideo.after(10, visualizar)
        else:
            lblVideo.image = ""
            cap.release()

def detectar():
    global det
    det = not det

def finalizar():
    global cap
    global det
    det = False
    cap.release()  # Libera la cámara
    root.quit()    # Detiene el bucle de eventos de Tkinter
    root.destroy() # Cierra la ventana de la interfaz

def modificar(frame):
    proc_frame = detectorFaceAndHand(frame)
    return proc_frame

# Funciones de detección
from mediapipe.framework.formats import landmark_pb2

# Configuración de visualización
MARGIN = 10  # pixels
ROW_SIZE = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
TEXT_COLOR = (255, 0, 0)

def detectorFaceAndHand(img):
    global detector, hands
    height, width, _ = img.shape
    annotated_image = img.copy()

    # Detección de rostro
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)
    detection_result = detector.detect(image)
    annotated_image = visualize(annotated_image, detection_result)

    # Detección de manos
    results_hands = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if results_hands.multi_hand_landmarks:
        for hand_landmarks in results_hands.multi_hand_landmarks:
            # Dibujar los puntos de las manos
            mp.solutions.drawing_utils.draw_landmarks(
                annotated_image, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS,
                mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                mp.solutions.drawing_styles.get_default_hand_connections_style()
            )

            # Obtener las coordenadas de la muñeca para determinar si es derecha o izquierda
            wrist = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.WRIST]
            hand_x = wrist.x

            # Determinar si la mano es izquierda o derecha
            hand_label = "Derecha" if hand_x > 0.5 else "Izquierda"

            # Calcular la posición para mostrar la etiqueta en la imagen
            text_x = int(hand_x * width)
            text_y = int(wrist.y * height) - 10  # Coloca la etiqueta encima de la muñeca

            # Dibujar la etiqueta en la imagen
            cv2.putText(annotated_image, hand_label,
                        (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                        1, (255, 0, 0), 2, cv2.LINE_AA)

    return annotated_image

def visualize(image, detection_result) -> np.ndarray:
    """Dibuja cuadros delimitadores y puntos clave en la imagen de entrada."""
    height, width, _ = image.shape

    for detection in detection_result.detections:
        # Dibujar el cuadro delimitador del rostro
        bbox = detection.bounding_box
        start_point = (bbox.origin_x, bbox.origin_y)
        end_point = (bbox.origin_x + bbox.width, bbox.origin_y + bbox.height)
        cv2.rectangle(image, start_point, end_point, TEXT_COLOR, 3)

        # Dibujar puntos clave
        for keypoint in detection.keypoints:
            keypoint_px = _normalized_to_pixel_coordinates(keypoint.x, keypoint.y, width, height)
            cv2.circle(image, keypoint_px, 2, (0, 255, 0), -1)

    return image

def _normalized_to_pixel_coordinates(normalized_x, normalized_y, image_width, image_height):
    """Convierte las coordenadas normalizadas a píxeles."""
    x_px = min(math.floor(normalized_x * image_width), image_width - 1)
    y_px = min(math.floor(normalized_y * image_height), image_height - 1)
    return x_px, y_px

# Configuración de detección
model_file = open(r'C:\Users\charl\Desktop\VA_environment\2. Talleres\taller3\Taller3Files\blaze_face_short_range.tflite', "rb")
model_data = model_file.read()
model_file.close()
base_options = python.BaseOptions(model_asset_buffer=model_data)
options = vision.FaceDetectorOptions(base_options=base_options)
detector = vision.FaceDetector.create_from_options(options)

# Configuración de MediaPipe para manos
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

# Interfaz de usuario
root = customtkinter.CTk()

btnIniciar = customtkinter.CTkButton(root, text="Iniciar", width=45, command=iniciar)
btnIniciar.grid(column=0, row=0, padx=5, pady=5)

btnFinalizar = customtkinter.CTkButton(root, text="Finalizar", width=45, command=finalizar)
btnFinalizar.grid(column=1, row=0, padx=5, pady=5)

btnMediaPipe = customtkinter.CTkButton(root, text="Detectar", width=45, command=detectar)
btnMediaPipe.grid(column=2, row=0, padx=5, pady=5)

lblVideo = customtkinter.CTkLabel(root, text="")
lblVideo.grid(column=0, row=1, columnspan=3)

root.resizable(width=False, height=False)
root.mainloop()