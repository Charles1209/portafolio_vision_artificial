from tkinter import *
from PIL import Image, ImageTk
import cv2
import imutils
import customtkinter
import numpy as np
import math
from typing import Tuple, Union
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions

# Variables globales
cap = None
det = False

# Configuración MediaPipe para detección facial
from mediapipe.framework.formats import landmark_pb2
MARGIN = 10
ROW_SIZE = 10
FONT_SIZE = 1
FONT_THICKNESS = 1
TEXT_COLOR = (255, 0, 0)

def _normalized_to_pixel_coordinates(normalized_x: float, normalized_y: float, image_width: int, image_height: int) -> Union[None, Tuple[int, int]]:
    """Convierte coordenadas normalizadas a coordenadas en píxeles."""
    def is_valid_normalized_value(value: float) -> bool:
        return (value > 0 or math.isclose(0, value)) and (value < 1 or math.isclose(1, value))

    if not (is_valid_normalized_value(normalized_x) and is_valid_normalized_value(normalized_y)):
        return None
    x_px = min(math.floor(normalized_x * image_width), image_width - 1)
    y_px = min(math.floor(normalized_y * image_height), image_height - 1)
    return x_px, y_px

def visualize(image, detection_result) -> np.ndarray:
    """Dibuja cuadros delimitadores y puntos clave en la imagen."""
    annotated_image = image.copy()
    height, width, _ = image.shape

    for detection in detection_result.detections:
        bbox = detection.bounding_box
        start_point = bbox.origin_x, bbox.origin_y
        end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
        cv2.rectangle(annotated_image, start_point, end_point, TEXT_COLOR, 3)

        for keypoint in detection.keypoints:
            keypoint_px = _normalized_to_pixel_coordinates(keypoint.x, keypoint.y, width, height)
            if keypoint_px:
                cv2.circle(annotated_image, keypoint_px, 2, (0, 255, 0), -1)

    return annotated_image

def detectorFace(img):
    """Aplica detección facial a un cuadro."""
    global detector
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)
    detection_result = detector.detect(image)
    annotated_image = visualize(img, detection_result)
    return annotated_image

# Funciones principales
def iniciar():
    global cap
    cap = cv2.VideoCapture(0)
    visualizar()

def visualizar():
    global cap, det
    if cap is not None:
        ret, frame = cap.read()
        if ret:
            if det:
                frame = detectorFace(frame)
            frame = imutils.resize(frame, width=640)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            im = Image.fromarray(frame)
            img = customtkinter.CTkImage(light_image=im, size=(640, 480))
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
    if cap:
        cap.release()
    lblVideo.image = ""

# Configuración del modelo MediaPipe
model_file = open(r'C:\Users\charl\Desktop\VA_environment\2. Talleres\taller3\Taller3Files\blaze_face_short_range.tflite', "rb")
model_data = model_file.read()
model_file.close()

base_options = python.BaseOptions(model_asset_buffer=model_data)
options = vision.FaceDetectorOptions(base_options=base_options)
detector = vision.FaceDetector.create_from_options(options)

# Configuración de la interfaz gráfica
root = customtkinter.CTk()

btnIniciar = customtkinter.CTkButton(root, text="Iniciar", command=iniciar)
btnIniciar.grid(column=0, row=0, padx=5, pady=5)

btnDetectar = customtkinter.CTkButton(root, text="Detectar Rostros", command=detectar)
btnDetectar.grid(column=1, row=0, padx=5, pady=5)

btnFinalizar = customtkinter.CTkButton(root, text="Finalizar", command=finalizar)
btnFinalizar.grid(column=2, row=0, padx=5, pady=5)

lblVideo = customtkinter.CTkLabel(root, text="")
lblVideo.grid(column=0, row=1, columnspan=3)

root.resizable(width=False, height=False)
root.mainloop()
