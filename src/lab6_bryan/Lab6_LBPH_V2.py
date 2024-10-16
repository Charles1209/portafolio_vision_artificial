# Import OpenCV module
import cv2
import os
import numpy as np

subjects = ["", "Ruben Blades", "Elvis Presley", "Nicole Kidman", "Jennifer Aniston"]

# Function to detect face using OpenCV
def detect_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier('C:/Users/Brandon/vision_artificial/Vision_Artificial/src/Lab6Archivos Moodle 2024/Lab6AB/lbpcascade_frontalface.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
    if len(faces) == 0:
        return None, None
    (x, y, w, h) = faces[0]
    return gray[y:y+h, x:x+w], faces[0]

def resize_face(face, size=(200, 200)):
    return cv2.resize(face, size)

def prepare_training_data(data_folder_path):
    dirs = os.listdir(data_folder_path)
    faces = []
    labels = []
    for dir_name in dirs:
        if not dir_name.startswith("s"):
            continue
        label = int(dir_name.replace("s", ""))
        subject_dir_path = data_folder_path + "/" + dir_name
        subject_images_names = os.listdir(subject_dir_path)
        for image_name in subject_images_names:
            if image_name.startswith("."):
                continue
            image_path = subject_dir_path + "/" + image_name
            image = cv2.imread(image_path)
            face, rect = detect_face(image)
            if face is not None:
                resized_face = resize_face(face)
                faces.append(resized_face)
                labels.append(label)
    return faces, labels

# Train the face recognizer
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
faces, labels = prepare_training_data("C:/Users/Brandon/vision_artificial/Vision_Artificial/src/Lab6Archivos Moodle 2024/Lab6_C_Files/training-data")
face_recognizer.train(faces, np.array(labels))

def draw_rectangle(img, rect):
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

def draw_text(img, text, x, y):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)

# Modified predict function to return label
def predict(test_img):
    img = test_img.copy()
    face, rect = detect_face(img)
    if face is None:
        return None, None, img
    resized_face = resize_face(face)
    label, confidence = face_recognizer.predict(resized_face)
    label_text = subjects[label]
    draw_rectangle(img, rect)
    draw_text(img, label_text, rect[0], rect[1]-5)
    return label_text, confidence, img

# List of test image paths
test_images = [
    "C:/Users/Brandon/vision_artificial/Vision_Artificial/src/Lab6Archivos Moodle 2024/Lab6_C_Files/test-data/test7.jpg",
    "C:/Users/Brandon/vision_artificial/Vision_Artificial/src/Lab6Archivos Moodle 2024/Lab6_C_Files/test-data/test8.jpg",
    "C:/Users/Brandon/vision_artificial/Vision_Artificial/src/Lab6Archivos Moodle 2024/Lab6_C_Files/test-data/test9.jpg",
    "C:/Users/Brandon/vision_artificial/Vision_Artificial/src/Lab6Archivos Moodle 2024/Lab6_C_Files/test-data/test10.jpg",
    "C:/Users/Brandon/vision_artificial/Vision_Artificial/src/Lab6Archivos Moodle 2024/Lab6_C_Files/test-data/test11.jpg",
    "C:/Users/Brandon/vision_artificial/Vision_Artificial/src/Lab6Archivos Moodle 2024/Lab6_C_Files/test-data/test12.jpg",
    "C:/Users/Brandon/vision_artificial/Vision_Artificial/src/Lab6Archivos Moodle 2024/Lab6_C_Files/test-data/test13.jpg",
    "C:/Users/Brandon/vision_artificial/Vision_Artificial/src/Lab6Archivos Moodle 2024/Lab6_C_Files/test-data/test14.jpg",
    "C:/Users/Brandon/vision_artificial/Vision_Artificial/src/Lab6Archivos Moodle 2024/Lab6_C_Files/test-data/test15.jpg",
    "C:/Users/Brandon/vision_artificial/Vision_Artificial/src/Lab6Archivos Moodle 2024/Lab6_C_Files/test-data/test16.jpg",
    "C:/Users/Brandon/vision_artificial/Vision_Artificial/src/Lab6Archivos Moodle 2024/Lab6_C_Files/test-data/test17.jpg",
    "C:/Users/Brandon/vision_artificial/Vision_Artificial/src/Lab6Archivos Moodle 2024/Lab6_C_Files/test-data/test18.jpg"
]

# List of actual labels (names) for test images
actual_labels = ["Ruben Blades", "Elvis Presley"]  # Ajusta esta lista según el orden de las imágenes

# Initialize counters for correct and incorrect predictions
correct_predictions = 0
incorrect_predictions = 0

# Loop through test images and compare predictions
for idx, image_path in enumerate(test_images):
    test_img = cv2.imread(image_path)
    predicted_label, confidence, predicted_img = predict(test_img)
    
    # If no face is detected, skip this image
    if predicted_label is None:
        print(f"No face detected in image {image_path}")
        continue
    
    # Compare predicted label with actual label
    actual_label = actual_labels[idx]
    if predicted_label == actual_label:
        correct_predictions += 1
        print(f"Correct prediction for image {idx+1}: {predicted_label}")
    else:
        incorrect_predictions += 1
        print(f"Incorrect prediction for image {idx+1}: {predicted_label} (expected: {actual_label})")
    
    # Optionally, display the image with prediction
    cv2.imshow(f"Test Image {idx+1}", predicted_img)
    cv2.waitKey(500)

# Print the final results
total_images = correct_predictions + incorrect_predictions
print(f"\nTotal Images: {total_images}")
print(f"Correct Predictions: {correct_predictions}")
print(f"Incorrect Predictions: {incorrect_predictions}")
print(f"Accuracy: {(correct_predictions / total_images) * 100:.2f}%")

cv2.destroyAllWindows()
