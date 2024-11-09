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

    if ret:
        # Salva l'immagine catturata nel percorso specificato
        cv2.imwrite(percorso_salvataggio, frame)
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
# Carica i dati dei vertici e delle coordinate reali
with open('26_07_24/without_chess/image_corners_real_coords.json', 'r') as f:
    corner_points = json.load(f)


# Estrai i punti immagine e i punti reali
image_corners = np.array([cp['corner'] for cp in corner_points], dtype=np.float32)
real_corners = np.array([cp['real'][:2] for cp in corner_points], dtype=np.float32)  # Usa solo x, y reali

# Calcola la matrice omografica
H, _ = cv2.findHomography(image_corners, real_corners)


# Dimensioni dell'immagine in pixel
image_width = 640
image_height = 480


min_x, max_x = np.min(real_corners[:, 0]), np.max(real_corners[:, 0])
min_y, max_y = np.min(real_corners[:, 1]), np.max(real_corners[:, 1])

# Muovi il robot verso i punti rilevati
for i, p in enumerate(best):

    pixelx = 640 - p[1]
    pixely = 480 - p[0]

    new_p = np.array([pixelx, pixely])


    real_point = cv2.perspectiveTransform(new_p, H)
    real_x, real_y = real_point[0][0]

    if i == 0:
        center = rtde_r.getActualTCPPose()  # Centro del piano, ovvero posizione iniziale
        
        center[0] = real_x
        center[1] = real_y

        if min_x <= real_x <= max_x and min_y <= real_y <= max_y:
            rtde_c.moveL(center, 0.2, 0.2)
            rtde_c.moveUntilContact(speed, direction, acceleration)
            center = rtde_r.getActualTCPPose()  # Centro del piano, ovvero posizione iniziale
        else:
            print ("Errore, movimento bloccato")
    else:
        center[0] = real_x
        center[1] = real_y
        if min_x <= real_x <= max_x and min_y <= real_y <= max_y:
            rtde_c.moveL(center, 0.2, 0.2)
        else:
            print ("Errore, movimento bloccato")


    print("Image point")
    print(p[0])
    print(p[1])
    print("Real points:")
    print(real_x)
    print(real_y)

