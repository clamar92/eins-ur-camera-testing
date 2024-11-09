import cv2
import numpy as np
import json

def undistort_points(image_points, mtx, dist):
    image_points = np.array(image_points, dtype=np.float32)
    image_points = image_points.reshape(-1, 1, 2)
    undistorted_points = cv2.undistortPoints(image_points, mtx, dist, None, mtx)
    return undistorted_points.reshape(-1, 2)

def image_to_robot_coordinates(image_point, camera_to_robot_matrix):
    point = np.array([image_point[0], image_point[1], 1.0])
    robot_point = np.dot(camera_to_robot_matrix, np.append(point, [1]))
    robot_point /= robot_point[3]
    return robot_point[:3]

# Carica i dati di calibrazione dal file
with open('19_07_24/conf_v2.json', 'r') as f:
    calibration_data = json.load(f)

mtx = np.array(calibration_data['mtx'])
dist = np.array(calibration_data['dist'])
camera_to_robot_matrix = np.array(calibration_data['camera_to_robot_matrix'])

# Esempio di utilizzo
image_point = [320, 240]  # Sostituisci con le coordinate del punto dell'immagine che vuoi trasformare

# Correggi la distorsione del punto immagine
undistorted_image_point = undistort_points([image_point], mtx, dist)[0]

# Converti il punto immagine non distorto alle coordinate del robot
robot_coordinates = image_to_robot_coordinates(undistorted_image_point, camera_to_robot_matrix)

print(f"Coordinate reali nel sistema di riferimento del robot: X={robot_coordinates[0]}, Y={robot_coordinates[1]}, Z={robot_coordinates[2]}")
