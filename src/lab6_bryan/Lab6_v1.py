# Import required packages:
import cv2

# Load cascade classifiers for face and mouth detection:
face_cascade = cv2.CascadeClassifier("C:/Users/Brandon/vision_artificial/Vision_Artificial/src/Lab6Archivos Moodle 2024/Lab6AB/haarcascade_frontalface_default.xml")
mouth_cascade = cv2.CascadeClassifier("C:/Users/Brandon/vision_artificial/Vision_Artificial/src/Lab6Archivos Moodle 2024/Lab6AB/haarcascade_mcs_mouth.xml")

# Create VideoCapture object to get images from the webcam:
video_capture = cv2.VideoCapture(0)  # Use 0 for the default webcam

while video_capture.isOpened():

    # Capture frame from the VideoCapture object:
    ret, frame = video_capture.read()

    # Convert frame to grayscale:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces:
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Iterate over each detected face:
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

        # Detect mouth within the face region:
        mouths = mouth_cascade.detectMultiScale(roi_gray, 1.3, 11)

        # Iterate over detected mouths (only take the first detection):
        for (mx, my, mw, mh) in mouths:
            # Adjust mouth region based on its position relative to the face
            if my > h / 2:  # Ensure it's in the lower half of the face
                # Draw a black rectangle over the mouth region (as censorship):
                cv2.rectangle(roi_color, (mx, my), (mx + mw, my + mh), (0, 0, 0), -1)
                break

    # Display the resulting frame
    cv2.imshow('Mouth Censorship Filter', frame)

    # Press 's' to save the image
    if cv2.waitKey(1) & 0xFF == ord('s'):
        cv2.imwrite('censorship_output.png', frame)  # Save the image
        print("Imagen guardada como 'censorship_output.png'")

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release everything:
video_capture.release()
cv2.destroyAllWindows()
