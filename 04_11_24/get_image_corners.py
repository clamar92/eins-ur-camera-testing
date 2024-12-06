import cv2
import cv2.aruco as aruco
import numpy as np
import time
import threading
import math
import json

# Uncomment these if you have access to the robot libraries
# import rtde_control
# import rtde_receive

# Robot IP address and port
ROBOT_HOST = '192.168.137.221'
# ROBOT_HOST = '192.168.186.135'

# Initialize RTDE (uncomment if you have the libraries available)
# rtde_c = rtde_control.RTDEControlInterface(ROBOT_HOST)
# rtde_r = rtde_receive.RTDEReceiveInterface(ROBOT_HOST)

# Global variables for coordinates
corner_points = []
robot_moving = False
free_drive_active = False

# Speed and acceleration parameters
acc = 0.4
vel = 0.4

# Robot start position (in radians)
robot_startposition = [math.radians(21.90),
                       math.radians(-82.13),
                       math.radians(-87.16),
                       math.radians(-101.36),
                       math.radians(90.43),
                       math.radians(18.88)]

# Define ArUco dictionary and parameters
dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)
parameters = aruco.DetectorParameters()
detector = aruco.ArucoDetector(dictionary)

def move_to_start_position():
    print('Move robot to start position')
    # rtde_c.moveJ(robot_startposition, vel, acc)
    time.sleep(2)  # Wait for the movement to complete

def free_drive_mode():
    global robot_moving, free_drive_active
    free_drive_active = True
    print("Free drive mode enabled. Move the robot to the desired position and press 'q' to save the coordinates.")
    # rtde_c.teachMode()
    while free_drive_active:
        time.sleep(0.1)
    # rtde_c.endTeachMode()
    robot_moving = False

def capture_real_coordinates():
    # Simulate getting the robot's position (uncomment if you have the libraries available)
    # pose = rtde_r.getActualTCPPose()
    pose = [0, 0, 0, 0, 0, 0]  # Replace this with actual coordinates
    real_x, real_y, real_z = pose[:3]
    return real_x, real_y, real_z

def save_corner_point(corner_point, real_point):
    corner_points.append({'corner': corner_point, 'real': real_point})

def main():
    global robot_moving, free_drive_active

    # Option 1: Capture image from a static file
    image_path = 'images_captured/image_1.jpg'
    frame = cv2.imread(image_path)

    # Option 2: Capture image from the camera (commented for now)
    # cap = cv2.VideoCapture(0)
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # if not cap.isOpened():
    #     print("Error opening the camera")
    #     return
    # ret, frame = cap.read()
    # if not ret:
    #     print("Error capturing frame from the camera")
    #     return

    if frame is None:
        print("Error loading the image")
        return

    # Move the robot to the start position
    move_to_start_position()

    # Detect ArUco markers in the image
    marker_centers = []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, rejected = detector.detectMarkers(gray)

    # If markers are detected
    if ids is not None:
        for corner, marker_id in zip(corners, ids.flatten()):
            # Calculate the center of each marker
            corner_points_2d = corner[0]
            center_x = int(np.mean(corner_points_2d[:, 0]))
            center_y = int(np.mean(corner_points_2d[:, 1]))
            marker_centers.append((marker_id, (center_x, center_y)))

            # Draw the marker center and ID on the original frame (once)
            cv2.circle(frame, (center_x, center_y), 10, (0, 255, 0), 2)
            cv2.putText(frame, f"ID: {marker_id}", (center_x - 10, center_y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            print(f"Detected marker ID {marker_id} at center: ({center_x}, {center_y})")

        # Draw bounding boxes around detected markers
        aruco.drawDetectedMarkers(frame, corners, ids)

    # Display the image with detected markers
    cv2.imshow("Rilevamento Marker ArUco", frame)
    cv2.waitKey(0)

    # Collect real coordinates for each detected marker center
    for i, (marker_id, center) in enumerate(marker_centers):
        print(f"Move the robot to the center of marker ID {marker_id}: {center}. Press 'q' to save this point.")
        threading.Thread(target=free_drive_mode).start()

        # Display the ID on a separate frame copy during the interactive session
        interactive_frame = frame.copy()
        cv2.circle(interactive_frame, center, 10, (0, 255, 0), 2)
        cv2.putText(interactive_frame, f"Move the robot to marker center ID {marker_id}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Wait for user input to confirm the position
        while True:
            cv2.imshow("Acquisizione Coordinate", interactive_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                free_drive_active = False
                robot_moving = True
                break

        real_point = capture_real_coordinates()
        save_corner_point(center, real_point)
        print(f"Point {i+1} saved: Marker ID {marker_id}, Image {center}, Real {real_point}")

        # Return the robot to the start position
        time.sleep(2)
        move_to_start_position()

    cv2.destroyAllWindows()

    # Save the coordinates of the marker centers
    with open('04_11_24/image_corners_real_coords.json', 'w') as f:
        json.dump(corner_points, f)

    print("Real coordinates of the marker centers have been saved.")

    # Uncomment the following line if using a camera
    # cap.release()

if __name__ == "__main__":
    main()
