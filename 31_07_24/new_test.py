import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.measure import find_contours
from skimage.color import rgb2hsv
import json

# Load the image
IMAGE_PATH = "ULTIMO/pizzacut_tmp.jpg"
myimg = imread(IMAGE_PATH)

# Vertices data
vertices = [
    {"corner": [0, 0], "real": [0.7421759340095093, 0.2960800053637704, -0.05290995235393037]},
    {"corner": [639, 0], "real": [0.7648839791957291, -0.3318842540668604, -0.052799199422714924]},
    {"corner": [0, 479], "real": [0.28929704643211324, 0.27759292099339944, -0.05805224717250261]},
    {"corner": [639, 479], "real": [0.3152508938772535, -0.3308237295620757, -0.05746304099665092]}
]

# Extract corner points and corresponding real coordinates
image_points = np.array([v["corner"] for v in vertices], dtype=np.float32)
real_points = np.array([v["real"][:2] for v in vertices], dtype=np.float32)  # Use only x and y coordinates

# Calculate the homography matrix
matrix, _ = cv2.findHomography(image_points, real_points)

# Convert the image to HSV and find contours based on saturation
myimg_hsv = rgb2hsv(myimg)
saturation = myimg_hsv[:,:,1]
binary_image = np.where(saturation > 0.25, 1, 0).astype(np.uint8)
contours_gray = find_contours(binary_image, 0.8)

# Find the longest contour to simulate the robot movement
best = max(contours_gray, key=len)

# Determine real-world plot limits based on real_points
real_x_min, real_y_min = np.min(real_points, axis=0)
real_x_max, real_y_max = np.max(real_points, axis=0)

# Plot the detected contour and the real-world coordinates
fig, ax = plt.subplots(2, 1, figsize=(15, 16))
ax[0].imshow(myimg)
ax[0].set_title("Detected Contour on Image")
ax[1].set_title("Simulated Robot Path")

# Simulate robot path based on the longest contour
for p in best:
    point = np.array([[p[1], p[0]]], dtype='float32').reshape(1, 1, 2)
    
    # Apply homography to convert to real-world coordinates
    real_coordinates = cv2.perspectiveTransform(point, matrix)
    
    # Extract real-world coordinates for plotting
    real_x, real_y = real_coordinates[0, 0]
    
    # Plot the image contour
    ax[0].plot(p[1], p[0], 'ro', markersize=1)
    ax[1].plot(real_x, real_y, 'bo', markersize=1)

# Set the limits for the real-world coordinate plot
ax[1].set_xlim(real_x_min, real_x_max)
ax[1].set_ylim(real_y_min, real_y_max)
ax[1].grid(True)
ax[1].set_xlabel("X (meters)")
ax[1].set_ylabel("Y (meters)")

# Overlay the original image on the real-world coordinates plot
ax[1].imshow(myimg, extent=[real_x_min, real_x_max, real_y_min, real_y_max], alpha=0.5)

plt.tight_layout()
plt.show()
