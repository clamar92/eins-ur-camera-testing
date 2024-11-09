import cv2
import numpy as np
import json
import time
import matplotlib.pyplot as plt
from skimage.color import rgb2hsv
from skimage.measure import find_contours
import rtde_control
import rtde_receive
import math

# Robot IP
ROBOT_HOST = '192.168.137.198'

# Initialize RTDE
rtde_c = rtde_control.RTDEControlInterface(ROBOT_HOST)
rtde_r = rtde_receive.RTDEReceiveInterface(ROBOT_HOST)

# Speed and acceleration parameters
acc = 0.4
vel = 0.4

# Initial robot position
robot_startposition = [math.radians(17.87),
                       math.radians(-78.87),
                       math.radians(-100.97),
                       math.radians(-90.22),
                       math.radians(90.03),
                       math.radians(15.62)]

speed = [0, 0, -0.1, 0, 0, 0]
direction = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0] 
acceleration = 1

rtde_c.moveJ(robot_startposition, vel, acc)

# Image resolution used during calibration
img_width = 640
img_height = 480

# Load calibration data
camera_calibration_file = '04_10_24/calibration_data.json'
with open(camera_calibration_file, 'r') as f:
    calib_data = json.load(f)
    camera_matrix = np.array(calib_data['mtx'])
    dist_coeffs = np.array(calib_data['dist'])

# Define Aruco marker dictionary and parameters
aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
aruco_params = cv2.aruco.DetectorParameters_create()

# Known real-world coordinates of the markers
marker_real_world_coords = {
    0: [0.0, 0.0],  # Marker 0 at (0,0)
    1: [500.0, 0.0],  # Marker 1 at (500,0)
    2: [0.0, 500.0],  # Marker 2 at (0,500)
    3: [500.0, 500.0] # Marker 3 at (500,500)
}

def detect_aruco_markers(image):
    """
    Detect Aruco markers in the image and return their IDs and corners.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)
    return corners, ids

def estimate_homography(corners, ids):
    """
    Estimate homography using the detected Aruco markers and their known real-world coordinates.
    """
    image_points = []
    real_world_points = []

    if ids is not None:
        for i, corner in enumerate(corners):
            marker_id = ids[i][0]
            if marker_id in marker_real_world_coords:
                image_points.append(np.mean(corner[0], axis=0))
                real_world_points.append(marker_real_world_coords[marker_id])

        if len(image_points) >= 4:
            image_points = np.array(image_points, dtype=np.float32)
            real_world_points = np.array(real_world_points, dtype=np.float32)
            homography_matrix, _ = cv2.findHomography(image_points, real_world_points)
            return homography_matrix
        else:
            print("Not enough markers detected to compute homography.")
            return None
    else:
        print("No markers detected.")
        return None

def map_image_to_real(homography_matrix, image_point):
    """
    Map image coordinates to real-world coordinates using the homography matrix.
    """
    image_point_homog = np.array([image_point[0], image_point[1], 1], dtype=np.float32)
    real_point_homog = np.dot(homography_matrix, image_point_homog)
    real_x = real_point_homog[0] / real_point_homog[2]
    real_y = real_point_homog[1] / real_point_homog[2]
    return real_x, real_y

def move_robot_to_real_point(real_x, real_y, first_mov, real_z=0.05):
    """
    Move the robot to a calculated real-world point.
    """
    center = rtde_r.getActualTCPPose()
    center[0] = real_x
    center[1] = real_y
    if first_mov == 1:
        rtde_c.moveUntilContact(speed, direction, acceleration)
    else:
        rtde_c.moveL(center, 0.2, 0.2)

# Start the webcam and capture an image
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, img_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, img_height)

if not cap.isOpened():
    print("Error opening the webcam")
    exit()

# Capture a single image
ret, frame = cap.read()

if not ret:
    print("Error capturing the frame")
else:
    # Detect Aruco markers in the image
    corners, ids = detect_aruco_markers(frame)

    # Estimate homography based on detected markers
    homography_matrix = estimate_homography(corners, ids)

    if homography_matrix is not None:
        # Correct the image distortion
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (img_width, img_height), 1)
        undistorted_frame = cv2.undistort(frame, camera_matrix, dist_coeffs, None, new_camera_matrix)

        # Convert the image to HSV and use the saturation channel to detect contours
        myimg_hsv = rgb2hsv(undistorted_frame)
        saturation = myimg_hsv[:, :, 1]

        # Create a binary image based on a threshold on saturation
        binary_image = np.where(saturation > 0.25, 1, 0).astype(np.uint8)

        # Find contours
        contours_gray = find_contours(binary_image, 0.8)

        if contours_gray:
            best_contour = max(contours_gray, key=len)

            # Visualize the images and coordinates
            fig, ax = plt.subplots(1, 2, figsize=(18, 6))
            ax[0].imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            ax[0].set_title('Original Image')

            # Draw the longest contour on the undistorted image
            best_contour_cv2 = np.array([[[int(p[1]), int(p[0])]] for p in best_contour])
            cv2.drawContours(undistorted_frame, [best_contour_cv2], -1, (0, 255, 0), 2)
            ax[1].imshow(cv2.cvtColor(undistorted_frame, cv2.COLOR_BGR2RGB))
            ax[1].set_title('Image with Longest Contour')
            plt.show()

            # Map the contour to real-world coordinates and move the robot
            real_coords = []
            for i, point in enumerate(best_contour):
                image_point = np.array([[point[1], point[0]]], dtype='float32')
                real_x, real_y = map_image_to_real(homography_matrix, image_point[0])
                if i == 0:
                    move_robot_to_real_point(real_x, real_y, 1)
                else:
                    move_robot_to_real_point(real_x, real_y, 0)
                real_coords.append([real_x, real_y])

            real_coords = np.array(real_coords)

            # Visualize the real-world coordinates on a 2D plot
            fig, ax = plt.subplots(1, 3, figsize=(18, 6))
            ax[0].imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            ax[0].set_title('Original Image')

            ax[1].imshow(cv2.cvtColor(undistorted_frame, cv2.COLOR_BGR2RGB))
            ax[1].set_title('Image with Longest Contour')

            ax[2].plot(real_coords[:, 0], real_coords[:, 1], 'ro-')
            ax[2].set_xlabel('X')
            ax[2].set_ylabel('Y')
            plt.tight_layout()
            plt.show()

    else:
        print("No valid homography matrix found.")

cap.release()
cv2.destroyAllWindows()

# Move the robot back to the start position
time.sleep(2)
print('Move robot to start position')
rtde_c.moveJ(robot_startposition, vel, acc)
