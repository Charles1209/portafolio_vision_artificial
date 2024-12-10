# Taller 3 Vision Artificial
# UTP FISC
from PIL import Image
from PIL import ImageTk
import cv2
import imutils
import customtkinter
import numpy as np
from typing import Tuple, Union
import math
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions

def iniciar():
    global cap
    cap = cv2.VideoCapture(0)  
    visualizar()


def visualizar():
    global cap
    global det
    if cap is not None:
        ret, frame = cap.read()
        if ret == True:
            
            if det :
                frame = modificar(frame) 

            frame = imutils.resize(frame, width=640)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            im = Image.fromarray(frame)
            
            img = customtkinter.CTkImage(light_image=im, size=( frame.shape[1], frame.shape[0]))            

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
    cap.release()

def modificar (fram):
    proc_frame = detectorFace(fram)
    return proc_frame

# Media Pipe------------------------------------
from mediapipe.framework.formats import landmark_pb2
MARGIN = 10  # pixels
ROW_SIZE = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
TEXT_COLOR = (255, 0, 0) 


def _normalized_to_pixel_coordinates(
    normalized_x: float, normalized_y: float, image_width: int,
    image_height: int) -> Union[None, Tuple[int, int]]:
  """Converts normalized value pair to pixel coordinates."""

  # Checks if the float value is between 0 and 1.
  def is_valid_normalized_value(value: float) -> bool:
    return (value > 0 or math.isclose(0, value)) and (value < 1 or
                                                      math.isclose(1, value))

  if not (is_valid_normalized_value(normalized_x) and
          is_valid_normalized_value(normalized_y)):
    # TODO: Draw coordinates even if it's outside of the image bounds.
    return None
  x_px = min(math.floor(normalized_x * image_width), image_width - 1)
  y_px = min(math.floor(normalized_y * image_height), image_height - 1)
  return x_px, y_px


def visualize(image, detection_result ) -> np.ndarray:
  """Draws bounding boxes and keypoints on the input image and return it.
  Args:
    image: The input RGB image.
    detection_result: The list of all "Detection" entities to be visualize.
  Returns:
    Image with bounding boxes.
  """
  annotated_image = image.copy()
  height, width, _ = image.shape

  for detection in detection_result.detections:
    # Draw bounding_box
    bbox = detection.bounding_box
    start_point = bbox.origin_x, bbox.origin_y
    end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
    cv2.rectangle(annotated_image, start_point, end_point, TEXT_COLOR, 3)

    # Draw keypoints
    for keypoint in detection.keypoints:
      keypoint_px = _normalized_to_pixel_coordinates(keypoint.x, keypoint.y,
                                                     width, height)
      color, thickness, radius = (0, 255, 0), 2, 2
      cv2.circle(annotated_image, keypoint_px, thickness, color, radius)

    # Draw label and score
    category = detection.categories[0]
    category_name = category.category_name
    category_name = '' if category_name is None else category_name
    probability = round(category.score, 2)
    result_text = category_name + ' (' + str(probability) + ')'
    text_location = (MARGIN + bbox.origin_x,
                     MARGIN + ROW_SIZE + bbox.origin_y)
    cv2.putText(annotated_image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                FONT_SIZE, TEXT_COLOR, FONT_THICKNESS)

  return annotated_image

def detectorFace(img):
  global detector

  # STEP 3: Load the input image.
  image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)

  # STEP 4: Detect hand landmarks from the input image.
  detection_result = detector.detect(image)

  # STEP 5: Process the classification result. In this case, visualize it.
  image_copy = np.copy(image.numpy_view())
  annotated_image = visualize(image_copy, detection_result)
  
  return annotated_image


cap = None
det = False

model_file = open('blaze_face_short_range.tflite', "rb")
model_data = model_file.read()
model_file.close()

base_options = python.BaseOptions(model_asset_buffer=model_data)
options = vision.FaceDetectorOptions(base_options=base_options)
detector = vision.FaceDetector.create_from_options(options)

root = customtkinter.CTk()

btnIniciar = customtkinter.CTkButton(root, text="Iniciar", width=45, command=iniciar)
btnIniciar.grid(column=0, row=0, padx=5, pady=5)

btnFinalizar = customtkinter.CTkButton(root, text="Finalizar", width=45, command=finalizar)
btnFinalizar.grid(column=1, row=0, padx=5, pady=5)

btnMediaPipe = customtkinter.CTkButton(root, text="Detectar", width=45, command=detectar)
btnMediaPipe.grid(column=2, row=0, padx=5, pady=5)

lblVideo = customtkinter.CTkLabel(root, text="")
lblVideo.grid(column=0, row=1, columnspan=3)

root.resizable(width= False, height  =False)

root.mainloop()