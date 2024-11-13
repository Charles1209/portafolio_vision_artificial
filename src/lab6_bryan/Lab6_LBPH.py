#import OpenCV module
import cv2
#import os module for reading training data directories and paths
import os
#import numpy to convert python lists to numpy arrays as
#it is needed by OpenCV face recognizers
import numpy as np

# There is no label 0 in our training data so subject name for index/label 0 is empty
subjects = ["", "Ruben Blades", "Elvis Presley", "Nicole Kidman", "Jennifer Aniston"]

# Function to detect face using OpenCV
def detect_face(img):
    # Convert the test image to gray image as opencv face detector expects gray images
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Load OpenCV face detector, I am using LBP which is fast
    face_cascade = cv2.CascadeClassifier('C:/Users/Brandon/vision_artificial/Vision_Artificial/src/Lab6Archivos Moodle 2024/Lab6AB/lbpcascade_frontalface.xml')
    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
    # If no faces are detected, return None
    if (len(faces) == 0):
        return None, None
    # Extract the face area from the image
    (x, y, w, h) = faces[0]
    # Return the face and the rectangle for drawing later
    return gray[y:y+h, x:x+w], faces[0]

# Function to ensure all face crops have the same size
def resize_face(face, size=(200, 200)):
    return cv2.resize(face, size)

# This function reads all training images, detects faces, and labels each face
def prepare_training_data(data_folder_path):
    # Get the directories for each subject in the training data
    dirs = os.listdir(data_folder_path)
    faces = []
    labels = []
    # Iterate over all directories (subjects)
    for dir_name in dirs:
        if not dir_name.startswith("s"):
            continue
        label = int(dir_name.replace("s", ""))
        path_dir_subject = data_folder_path + "/" + dir_name
        subject_images_names = os.listdir(path_dir_subject)
        # Iterate over each image in the subject's directory
        for image_name in subject_images_names:
            if image_name.startswith("."):
                continue
            image_path = path_dir_subject + "/" + image_name
            image = cv2.imread(image_path)
            cv2.imshow("Training on image...", image)
            cv2.waitKey(100)
            # Detect face in the image
            face, rect = detect_face(image)
            # If a face is detected, add it to the list
            if face is not None:
                resized_face = resize_face(face)  # Resize the face
                faces.append(resized_face)
                labels.append(label)
    cv2.destroyAllWindows()
    return faces, labels

print("Preparing data...")
faces, labels = prepare_training_data("C:/Users/Brandon/vision_artificial/Vision_Artificial/src/Lab6Archivos Moodle 2024/Lab6_C_Files/training-data")
print("Data prepared")

# Train the LBPH face recognizer
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
# Train our face recognizer with our training faces
face_recognizer.train(faces, np.array(labels))

# Function to draw rectangle around the face
def draw_rectangle(img, rect):
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

# Function to draw the label (name) on the image
def draw_text(img, text, x, y):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)

# This function recognizes the person in the test image
def predict(test_img, save_path=None):
    img = test_img.copy()
    face, rect = detect_face(img)
    if face is None:
        print("No face detected")
        return img
    
    resized_face = resize_face(face)  # Resize the face before prediction
    label, confidence = face_recognizer.predict(resized_face)
    label_text = subjects[label]
    
    draw_rectangle(img, rect)
    draw_text(img, label_text, rect[0], rect[1]-5)
    
    print(f"Prediction: {label_text}, Confidence: {confidence}")

    # Save the image with the predicted label
    if save_path:
        cv2.imwrite(save_path, img)

    return img

print("Predicting images...")

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

# Loop over each test image
for idx, image_path in enumerate(test_images):
    # Read the test image
    test_img = cv2.imread(image_path)
    
    # Call the predict function and print the result
    print(f"Testing image {idx+1}: {image_path}")
    predicted_img = predict(test_img, f"C:/Users/Brandon/vision_artificial/Vision_Artificial/src/lab6/test_img{idx+1}_predicted.jpg")
    
    # Display the image with the prediction (optional)
    cv2.imshow(f"Test Image {idx+1}", predicted_img)
    cv2.waitKey(500)  # Display each image for 500 ms

cv2.destroyAllWindows()
