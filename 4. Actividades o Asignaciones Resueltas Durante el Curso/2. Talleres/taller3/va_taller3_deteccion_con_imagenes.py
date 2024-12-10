import cv2
import customtkinter as ctk
from PIL import Image, ImageTk
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2

# Lista de imágenes y el índice actual
image_paths = [
    "becky.jpeg",   # Asegúrate de que las imágenes estén en el directorio correcto
    "descarga.jpeg",
    "sarocha.jpeg"
]
image_index = 0

# Variable para almacenar la imagen cargada
current_image = None

# Cargar el modelo de MediaPipe
with open(r"C:\Users\charl\Desktop\VA_environment\2. Talleres\taller3\Taller3Files\face_landmarker_v2_with_blendshapes.task", "rb") as model_file:
    model_data = model_file.read()

base_options = python.BaseOptions(model_asset_buffer=model_data)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=True,
    output_facial_transformation_matrixes=True,
    num_faces=1
)
detector = vision.FaceLandmarker.create_from_options(options)

# Función para dibujar los puntos faciales en la imagen
def draw_landmarks_on_image(rgb_image, detection_result):
    face_landmarks_list = detection_result.face_landmarks
    annotated_image = np.copy(rgb_image)

    for face_landmarks in face_landmarks_list:
        face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        face_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
        ])

        mp.solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_tesselation_style()
        )
        mp.solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_contours_style()
        )
        mp.solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_IRISES,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_iris_connections_style()
        )

    return annotated_image

# Función para mostrar la imagen sin detección
def mostrar_imagen():
    global current_image
    if 0 <= image_index < len(image_paths):
        image_path = image_paths[image_index]
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error al cargar la imagen: {image_path}")
            return

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        current_image = image  # Almacena la imagen en BGR para la detección
        img = Image.fromarray(image_rgb)
        imgtk = ImageTk.PhotoImage(image=img)
        video_frame.imgtk = imgtk
        video_frame.configure(image=imgtk)

# Función para avanzar a la siguiente imagen
def siguiente_imagen():
    global image_index
    image_index += 1
    if image_index >= len(image_paths):
        image_index = 0  # Reinicia al principio después de la última imagen
    mostrar_imagen()

# Función para realizar la detección facial
def detectar_rostro():
    if current_image is None:
        return  # No hacer nada si no hay una imagen cargada

    # Convertir la imagen a RGB para el detector
    image_rgb = cv2.cvtColor(current_image, cv2.COLOR_BGR2RGB)
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    
    # Realizar la detección
    detection_result = detector.detect(image)

    # Dibujar los puntos faciales en la imagen
    annotated_image = draw_landmarks_on_image(image_rgb, detection_result)

    # Mostrar la imagen anotada
    img = Image.fromarray(annotated_image)
    imgtk = ImageTk.PhotoImage(image=img)
    video_frame.imgtk = imgtk
    video_frame.configure(image=imgtk)

# Crear la ventana principal
ctk.set_appearance_mode("system")  # Modo de apariencia de la interfaz
window = ctk.CTk()
window.title("Detección Facial con OpenCV y MediaPipe")

# Crear un marco de imagen en la ventana
video_frame = ctk.CTkLabel(window)
video_frame.pack()

# Botones de la interfaz
button_frame = ctk.CTkFrame(window)
button_frame.pack(pady=10)

iniciar_button = ctk.CTkButton(button_frame, text="Iniciar", command=mostrar_imagen)
iniciar_button.grid(row=0, column=0, padx=5)

siguiente_button = ctk.CTkButton(button_frame, text="Siguiente", command=siguiente_imagen)
siguiente_button.grid(row=0, column=1, padx=5)

detectar_button = ctk.CTkButton(button_frame, text="Detectar", command=detectar_rostro)
detectar_button.grid(row=0, column=2, padx=5)

finalizar_button = ctk.CTkButton(button_frame, text="Finalizar", command=window.destroy)
finalizar_button.grid(row=0, column=3, padx=5)

# Ejecutar la ventana de Tkinter
window.mainloop()