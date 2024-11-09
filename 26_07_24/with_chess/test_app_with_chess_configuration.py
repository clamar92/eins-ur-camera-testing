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




# Indirizzo IP del robot e porta RTDE
ROBOT_HOST = '192.168.137.198'
#ROBOT_HOST = '192.168.186.135'




# Inizializzazione RTDE
rtde_c = rtde_control.RTDEControlInterface(ROBOT_HOST)
rtde_r = rtde_receive.RTDEReceiveInterface(ROBOT_HOST)

# Accelerazione e velocità
acc = 0.4
vel = 0.4

# Posizione iniziale del robot
robot_startposition = [math.radians(17.87),
                       math.radians(-78.87),
                       math.radians(-100.97),
                       math.radians(-90.22),
                       math.radians(90.03),
                       math.radians(15.62)]

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

    mirrored_frame = cv2.flip(frame, -1)

    if ret:
        # Salva l'immagine catturata nel percorso specificato
        cv2.imwrite(percorso_salvataggio, mirrored_frame)
        print(f"Foto scattata e salvata come '{percorso_salvataggio}'")
    else:
        print("Errore nella lettura del frame dalla webcam")
        return False

percorso_salvataggio = "26_07_24/with_chess/pizzacut_tmp.jpg"
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




# Robot process
# Carica i dati di calibrazione
with open('26_07_24/with_chess/calibration_data.json', 'r') as f:
    calibration_data = json.load(f)

mtx = np.array(calibration_data['mtx'])
dist = np.array(calibration_data['dist'])

# Distanza nota dal piano in millimetri
known_distance = 440  # mm

# Coordinate fisiche corrispondenti all'angolo in basso a sinistra dell'immagine (in metri)
MINX = 0.28929704643211324
MINY = -0.3318842540668604



MAXX = 0.7648839791957291
MAXY = 0.27759292099339944


#physical_origin = np.array([MINX, MINY])  # metri
physical_origin = np.array([MAXX, MAXY])  # metri


# Dimensioni dell'immagine in pixel
image_width = 640
image_height = 480


# Muovi il robot verso i punti rilevati
for i, p in enumerate(best):


    # Rimuovi la distorsione dai punti selezionati
    #points = np.array([479 - p[0],639 - p[1]], dtype='float32').reshape(-1, 1, 2)
    points = np.array([p[0],p[1]], dtype='float32').reshape(-1, 1, 2)
    #points = np.array([p[0],p[1]], dtype='float32').reshape(-1, 1, 2)
    undistorted_points = cv2.undistortPoints(points, mtx, dist, P=mtx)

    # Appiattisci l'array undistorted_points
    undistorted_points_flat = undistorted_points.reshape(-1, 2)
    #print(undistorted_points_flat)

    # Calcola le coordinate reali in metri
    real_coordinates = physical_origin - (undistorted_points_flat / [image_width, image_height]) * known_distance / 1000
    real_coordinates = real_coordinates[0]
    #print(real_coordinates)

    if i == 0:
        center = rtde_r.getActualTCPPose()  # Centro del piano, ovvero posizione iniziale
        
        center[0] = real_coordinates[0] 
        center[1] = real_coordinates[1]

        if (real_coordinates[0] > MINX and real_coordinates[0] < MAXX) and (real_coordinates[1] > MINY or real_coordinates[1] < MAXY):
            rtde_c.moveL(center, 0.2, 0.2)
            rtde_c.moveUntilContact(speed, direction, acceleration)
            center = rtde_r.getActualTCPPose()  # Centro del piano, ovvero posizione iniziale
        else:
            print ("Errore, movimento bloccato")
    else:
        center[0] = real_coordinates[0] 
        center[1] = real_coordinates[1]
        if (real_coordinates[0] > MINX and real_coordinates[0] < MAXX) and (real_coordinates[1] > MINY or real_coordinates[1] < MAXY):
            rtde_c.moveL(center, 0.2, 0.2)
        else:
            print ("Errore, movimento bloccato")


    print("Image point")
    print(p[0])
    print(p[1])
    print("Real points:")
    print(real_coordinates[0])
    print(real_coordinates[1])


time.sleep(2)
print('Move robot to start position')
rtde_c.moveJ(robot_startposition, vel, acc)

