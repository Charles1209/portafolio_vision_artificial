"""entregable 2 2"""
# Import required packages
import cv2
import os
import pickle
import numpy as np

OVERLAY_SIZE_PER = 1

# Check for camera calibration data
if not os.path.exists('./calibration.pckl'):
    print("You need to calibrate the camera before")
    exit()
else:
    f = open('calibration.pckl', 'rb')
    cameraMatrix, distCoeffs = pickle.load(f)
    f.close()
    if cameraMatrix is None or distCoeffs is None:
        print("Something went wrong. Recalibrate the camera")
        exit()

# Load the image overlay
overlay = cv2.imread(r"E:\git\vision_artificial_windows\src\lab5\tree_overlay.png")  # Replace with your overlay image

def draw_points(img, pts):
    """ Draw points in the image """
    pts = np.int32(pts).reshape(-1, 2)
    for p in pts:
        cv2.circle(img, (p[0], p[1]), 5, (255, 0, 255), -1)
    return img

def draw_augmented_overlay(pts_1, overlay_image, image):
    """ Overlay the image 'overlay_image' onto the image 'image' """
    pts_2 = np.float32([[0, 0], [overlay_image.shape[1], 0], [overlay_image.shape[1], overlay_image.shape[0]], [0, overlay_image.shape[0]]])
    M = cv2.getPerspectiveTransform(pts_2, pts_1)
    dst_image = cv2.warpPerspective(overlay_image, M, (image.shape[1], image.shape[0]))
    dst_image_gray = cv2.cvtColor(dst_image, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(dst_image_gray, 0, 255, cv2.THRESH_BINARY_INV)
    image_masked = cv2.bitwise_and(image, image, mask=mask)
    result = cv2.add(dst_image, image_masked)
    return result

# Create the dictionary object and parameters
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_7X7_250)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(dictionary, parameters)

# Create video capture object to capture frames from the first connected camera
capture = cv2.VideoCapture(0)

while True:
    # Capture frame by frame from the video capture object
    ret, frame = capture.read()
    if not ret:
        print("Error capturing video")
        break

    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect markers
    corners, ids, rejectedImgPoints = detector.detectMarkers(gray_frame)

    # Draw detected markers
    frame = cv2.aruco.drawDetectedMarkers(image=frame, corners=corners, ids=ids, borderColor=(0, 255, 0))

    if ids is not None:
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, 1, cameraMatrix, distCoeffs)

        # Process each marker independently
        for i, rvec in enumerate(rvecs):
            tvec = tvecs[i]
            marker_id = ids[i][0]

            # Apply overlay only if the marker ID is 73
            if marker_id == 73:
                # Define the points where you want the image to be overlaid (for each marker)
                desired_points = np.float32([[-1 / 2, 1 / 2, 0], [1 / 2, 1 / 2, 0], [1 / 2, -1 / 2, 0], [-1 / 2, -1 / 2, 0]]) * OVERLAY_SIZE_PER

                # Project the points for each marker
                projected_desired_points, jac = cv2.projectPoints(desired_points, rvec, tvec, cameraMatrix, distCoeffs)

                # Overlay the image for marker ID 73
                frame = draw_augmented_overlay(projected_desired_points, overlay, frame)

                # Draw the projected points (for debugging)
                draw_points(frame, projected_desired_points)

    # Display the resulting frame
    cv2.imshow('frame', frame)

    # Press 's' to save the current frame, 'q' to quit
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        cv2.imwrite("saved_frame2.png", frame)
        print("Image saved as 'saved_frame.png'")
    elif key == ord('q'):
        break

# Release everything
capture.release()
cv2.destroyAllWindows()