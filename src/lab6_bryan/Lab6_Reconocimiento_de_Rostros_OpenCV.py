#import OpenCV module
import cv2
#import os module for reading training data directories and paths
import os
#import numpy to convert python lists to numpy arrays as
#it is needed by OpenCV face recognizers
import numpy as np

#there is no label 0 in our training data so subject name for index/label 0 is empty
subjects = ["", "Ruben Blades", "Elvis Presley", "Nicole Kidman", "Jennifer Aniston"]

#function to detect face using OpenCV and resize it to a uniform size
def detect_face(img, target_size=(200, 200)):
    #convert the test image to gray image as opencv face detector expects gray images
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #load OpenCV face detector, I am using LBP which is fast
    face_cascade = cv2.CascadeClassifier('src/lab6_bryan/Lab6_C_Files/opencv-files/lbpcascade_frontalface.xml')
    #detect multiscale faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
    #if no faces are detected then return original img
    if len(faces) == 0:
        return None, None
    #extract the face area
    (x, y, w, h) = faces[0]
    #crop the face from the gray image
    face = gray[y:y+h, x:x+w]
    #resize the face to the target size (e.g., 200x200)
    face_resized = cv2.resize(face, target_size)
    return face_resized, faces[0]

#this function will read all persons' training images, detect face from each image
#and will return two lists of exactly same size, one list of faces and another list of labels
def prepare_training_data(data_folder_path, save_folder_path):
    dirs = os.listdir(data_folder_path)
    faces = []
    labels = []
    for dir_name in dirs:
        if not dir_name.startswith("s"):
            continue
        label = int(dir_name.replace("s", ""))
        subject_dir_path = os.path.join(data_folder_path, dir_name)
        subject_images_names = os.listdir(subject_dir_path)
        subject_save_path = os.path.join(save_folder_path, dir_name)
        if not os.path.exists(subject_save_path):
            os.makedirs(subject_save_path)
        
        for image_name in subject_images_names:
            if image_name.startswith("."):
                continue
            image_path = os.path.join(subject_dir_path, image_name)
            image = cv2.imread(image_path)
            cv2.imshow("Training on image...", image)
            cv2.waitKey(100)
            face, rect = detect_face(image)
            if face is not None:
                faces.append(face)
                labels.append(label)
                save_path = os.path.join(subject_save_path, image_name)
                cv2.imwrite(save_path, face)
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    cv2.destroyAllWindows()
    return faces, labels

# Paths

path_training_data = "src/lab6_bryan/Lab6_C_Files/training-data"
training_data = cv2.FileStorage(path_training_data, cv2.FILE_STORAGE_READ)
if not os.path.exists(path_training_data):
	raise FileNotFoundError(f"The file {path_training_data} does not exist.")
elif not training_data.isOpened():
	raise IOError(f"No se puede abrir el archivo {path_training_data} en modo lectura.")

path_cropped_faces = "src/lab6_bryan/Lab6_C_Files/cropped-faces"
cropped_faces = cv2.FileStorage(path_cropped_faces, cv2.FILE_STORAGE_READ)
if not os.path.exists(path_cropped_faces):
	raise FileNotFoundError(f"The file {path_cropped_faces} does not exist.")
elif not cropped_faces.isOpened():
	raise IOError(f"No se puede abrir el archivo {path_cropped_faces} en modo lectura.")

#prepare training data
print("Preparing data...")
faces, labels = prepare_training_data (
	path_training_data,
    path_cropped_faces
)
print("Data prepared")
print("Total faces: ", len(faces))
print("Total labels: ", len(labels))

#create the recognizer (use FisherFaceRecognizer)
face_recognizer = cv2.face.FisherFaceRecognizer_create()

#train our face recognizer of our training faces
face_recognizer.train(faces, np.array(labels))

#function to draw rectangle on image according to given coordinates
def draw_rectangle(img, rect):
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

#function to draw text on image starting from passed coordinates
def draw_text(img, text, x, y):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)

#this function recognizes the person in image passed and draws a rectangle around the detected face with name
# and prints the label and confidence score
def predict(test_img):
    img = test_img.copy()
    face, rect = detect_face(img)
    if face is not None:
        label, confidence = face_recognizer.predict(face)
        print(f"Prediccion: {subjects[label]}, Confidence: {confidence}")
        draw_rectangle(img, rect)
        draw_text(img, subjects[label], rect[0], rect[1]-5)
    else:
        print("No face detected in the image.")
    return img

print("Predicting images...")

# List of 12 test images (update the paths to match your 12 test images)
test_images_paths = [
    "src/lab6_bryan/Lab6_C_Files/test-data/222test2.jpg",
    "src/lab6_bryan/Lab6_C_Files/test-data/test0.jpg",
    "src/lab6_bryan/Lab6_C_Files/test-data/test2.jpg",
    "src/lab6_bryan/Lab6_C_Files/test-data/test3.jpg",
    "src/lab6_bryan/Lab6_C_Files/test-data/test4.jpg",
    "src/lab6_bryan/Lab6_C_Files/test-data/test4.jpg",
    "src/lab6_bryan/Lab6_C_Files/test-data/test6.jpg",
]

# Path where to save the predicted images
save_folder_path = "src/lab6_bryan/Lab6_C_Files/predicted-images"
if not os.path.exists(save_folder_path):
    os.makedirs(save_folder_path)

# Perform prediction on each test image and save the results
for idx, test_image_path in enumerate(test_images_paths):
    test_img = cv2.imread(test_image_path)
    predicted_img = predict(test_img)
    # Save the predicted image
    save_path = os.path.join(save_folder_path, f"predicted_{idx+1}.jpg")
    cv2.imwrite(save_path, predicted_img)
    # Optionally display the image
    cv2.imshow(f"Predicted Image {idx+1}", predicted_img)

print("Prediction and saving complete")

cv2.waitKey(0)
cv2.destroyAllWindows()
