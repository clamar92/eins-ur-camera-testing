import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.measure import find_contours
from skimage.color import rgb2hsv
import time
import rtde_control
import rtde_receive
import json
import math

# Nome del file che contiene i parametri di trasformazione
PARAMS_FILE = '19_07_24/transformation_params.json'
CORNERS_FILE = '19_07_24/image_corners_real_coords.json'

# Funzioni dal file secondario
def load_transformation_parameters(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    transformation_matrix = np.array(data['transformation_matrix'])
    return transformation_matrix

def image_to_real_coords(image_x, image_y, transformation_matrix):
    point = np.array([[image_x, image_y]], dtype=np.float32)
    real_point = cv2.perspectiveTransform(point[None, :, :], transformation_matrix)
    real_x, real_y = real_point[0][0]
    return real_x, real_y

def load_corners(filename):
    with open(filename, 'r') as f:
        corners = json.load(f)
    return [(corner['corner'], corner['real']) for corner in corners]

def is_within_limits(x, y, limits):
    min_x = min(limit[0] for limit in limits)
    max_x = max(limit[0] for limit in limits)
    min_y = min(limit[1] for limit in limits)
    max_y = max(limit[1] for limit in limits)
    return min_x <= x <= max_x and min_y <= y <= max_y

# Indirizzo IP del robot e porta RTDE
ROBOT_HOST = '192.168.186.135'

# Inizializzazione RTDE
rtde_c = rtde_control.RTDEControlInterface(ROBOT_HOST)
rtde_r = rtde_receive.RTDEReceiveInterface(ROBOT_HOST)

# Accelerazione e velocità
acc = 0.4
vel = 0.4

# Posizione iniziale del robot
robot_startposition = [math.radians(0),
                       math.radians(-95),
                       math.radians(-100),
                       math.radians(-78),
                       math.radians(88),
                       math.radians(0)]

print('Move robot to start position')
rtde_c.moveJ(robot_startposition, vel, acc)
time.sleep(2)

speed = [0, 0, -0.1, 0, 0, 0]
direction = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
acceleration = 1  # Accelerazione (può essere omessa se si vuole usare il valore predefinito)

def scatta_foto(percorso_salvataggio):
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("Errore nell'apertura della webcam")
        return False

    # Leggi un frame dalla webcam
    ret, frame = cap.read()

    if ret:
        # Salva l'immagine catturata nel percorso specificato
        cv2.imwrite(percorso_salvataggio, frame)
        print(f"Foto scattata e salvata come '{percorso_salvataggio}'")
    else:
        print("Errore nella lettura del frame dalla webcam")
        return False

percorso_salvataggio = "19_07_24/pizzacut_tmp.jpg"
scatta_foto(percorso_salvataggio)
IMAGE_PATH = percorso_salvataggio

myimg = imread(IMAGE_PATH)
myimg_hsv = rgb2hsv(myimg)

saturation = myimg_hsv[:,:,1]
binary_image = np.where(saturation > 0.25, 1, 0).astype(np.uint8)

contours_gray = find_contours(binary_image, 0.8)

# Display the image and plot the contour
print("Spot ground truth based on grayscale")
fig, ax = plt.subplots(1, 2, figsize=(10, 10))

ax[0].imshow(binary_image, cmap='gray')

ax[1].imshow(myimg, interpolation='nearest')
X, Y = ax[1].get_xlim(), ax[1].get_ylim()

best = contours_gray[0]
for contour in contours_gray:
    if len(best) < len(contour):
        best = contour
ax[1].step(best.T[1], best.T[0], linewidth=1, c='r')

ax[1].set_xlim(X), ax[1].set_ylim(Y)
plt.show()

# Carica i parametri di trasformazione
transformation_matrix = load_transformation_parameters(PARAMS_FILE)

# Carica i limiti dei 4 angoli dell'immagine
corner_points = load_corners(CORNERS_FILE)
real_corners = [point[1][:2] for point in corner_points]

# Muovi il robot verso i punti rilevati
for i, p in enumerate(best):
    x_real, y_real = image_to_real_coords(p[1], p[0], transformation_matrix)

    if is_within_limits(x_real, y_real, real_corners):
        if i == 0:
            rtde_c.moveUntilContact(speed, direction, acceleration)
            center = rtde_r.getActualTCPPose()  # Centro del piano, ovvero posizione iniziale
        print(center)

        center[0] = x_real
        center[1] = y_real

        rtde_c.moveL(center, vel, acc)
    else:
        print(f"Coordinate ({x_real}, {y_real}) fuori dai limiti.")

print("Movimento completato.")

# Visualizza i contorni trovati e le coordinate reali sui grafici
plt.figure(figsize=(10, 10))
plt.imshow(myimg, interpolation='nearest')
X, Y = plt.xlim(), plt.ylim()

plt.step(best.T[1], best.T[0], linewidth=1, c='r')

# Sovrappone i punti reali trovati
real_xs = []
real_ys = []
for i, p in enumerate(best):
    x_real, y_real = image_to_real_coords(p[1], p[0], transformation_matrix)
    if is_within_limits(x_real, y_real, real_corners):
        real_xs.append(p[1])
        real_ys.append(p[0])
plt.scatter(real_xs, real_ys, c='b', marker='o')

plt.xlim(X)
plt.ylim(Y)
plt.title("Contorni trovati e coordinate reali")
plt.show()
