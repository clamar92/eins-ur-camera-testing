import cv2
import numpy as np
import json
import math
import torch
import matplotlib.pyplot as plt
from skimage.color import rgb2hsv
from skimage.measure import find_contours
#import rtde_control
#import rtde_receive
import time
from skimage.morphology import square, dilation
from skimage.draw import polygon
from fastsam import FastSAM, FastSAMPrompt
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import matplotlib.image
import skimage

FAST_SAM_CHECKPOINT_PATH = f"weights/FastSAM.pt"
SAM_SAM_CHECKPOINT_PATH = f"weights/sam_vit_h_4b8939.pth"

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
fast_sam = FastSAM(FAST_SAM_CHECKPOINT_PATH)

# IP del robot (commentato per ora)
ROBOT_HOST = '192.168.137.221'

# Inizializzazione RTDE (commentato per ora)
#rtde_c = rtde_control.RTDEControlInterface(ROBOT_HOST)
#rtde_r = rtde_receive.RTDEReceiveInterface(ROBOT_HOST)

# Parametri di velocità e accelerazione
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

#rtde_c.moveJ(robot_startposition, vel, acc)

# Risoluzione utilizzata durante la calibrazione
img_width = 640  # Larghezza dell'immagine
img_height = 480  # Altezza dell'immagine

# File contenente le coordinate dei punti immagine e dei punti reali
calibration_file = '04_10_24/image_corners_real_coords.json'

# File contenente i parametri di calibrazione della fotocamera
camera_calibration_file = '04_10_24/calibration_data.json'

def area_value(area_element):
  return(area_element[1])

# Carica i parametri di calibrazione della fotocamera (matrice intrinseca e coefficienti di distorsione)
with open(camera_calibration_file, 'r') as f:
    calib_data = json.load(f)
    camera_matrix = np.array(calib_data['mtx'])  # Matrice intrinseca
    dist_coeffs = np.array(calib_data['dist'])   # Coefficienti di distorsione

# Carica i punti di calibrazione (coordinate immagine -> coordinate reali robot)
with open(calibration_file, 'r') as f:
    calibration_data = json.load(f)

# Estrai i punti immagine e i punti reali dal file
image_points = np.array([point['corner'] for point in calibration_data], dtype=np.float32)
real_points = np.array([point['real'][:2] for point in calibration_data], dtype=np.float32)  # Usa solo X e Y dei punti reali

# Ottieni i limiti delle coordinate reali (minimo e massimo per X e Y)
real_x_min, real_y_min = np.min(real_points, axis=0)
real_x_max, real_y_max = np.max(real_points, axis=0)

# Calcola la matrice di omografia (trasformazione prospettica) immagine -> mondo reale
homography_matrix, _ = cv2.findHomography(image_points, real_points)

# Funzione per convertire i punti immagine (pixel) in coordinate reali (robot)
def map_image_to_real(image_point):
    # Aggiungi una terza coordinata per la trasformazione omografica
    image_point_homog = np.array([image_point[0], image_point[1], 1], dtype=np.float32)

    # Applica la matrice di omografia
    real_point_homog = np.dot(homography_matrix, image_point_homog)

    # Normalizza per ottenere le coordinate reali (X, Y)
    real_x = real_point_homog[0] / real_point_homog[2]
    real_y = real_point_homog[1] / real_point_homog[2]

    return real_x, real_y

# Funzione per verificare se le coordinate reali sono nei limiti definiti
def is_within_bounds(real_x, real_y):
    return real_x_min <= real_x <= real_x_max and real_y_min <= real_y <= real_y_max

# Funzione per stampare le coordinate reali (simulando lo spostamento del robot)
def move_robot_to_real_point(real_x, real_y, first_mov, real_z=0.05):
    if is_within_bounds(real_x, real_y):
       # print(f"Coordinate reali calcolate: X={real_x}, Y={real_y}, Z={real_z}")

        #center = rtde_r.getActualTCPPose()  # Centro del piano, ovvero posizione iniziale
        center = [0,0,0,0,0,0]

        center[0] = real_x
        center[1] = real_y

        #if first_mov == 1:
            #rtde_c.moveL(center, 0.8, 0.8)
            #rtde_c.moveUntilContact(speed, direction, acceleration)
        #else:
            #rtde_c.moveL(center, 0.8, 0.8)

    else:
        print(f"Coordinate fuori dai limiti: X={real_x}, Y={real_y}")

# Avvia la webcam e acquisisci una singola immagine
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, img_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, img_height)

if not cap.isOpened():
    print("Errore nell'apertura della webcam")
    exit()

# Cattura una singola immagine
ret, frame = cap.read()


#image_path = 'images_captured/image_1.jpg'
#frame = cv2.imread(image_path)
#ret = True


if not ret:
    print("Errore nella lettura del frame dalla webcam")
else:
    # Correggi la distorsione ottica utilizzando i parametri di calibrazione
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (img_width, img_height), 1)
    undistorted_frame = cv2.undistort(frame, camera_matrix, dist_coeffs, None, new_camera_matrix)

    IMAGE_PATH = 'images_captured/image_1.jpg'
    matplotlib.image.imsave(IMAGE_PATH, undistorted_frame)
    # perform fastSAM processing on image -> detection of package contours
    results = fast_sam(
      source=IMAGE_PATH,
      device=DEVICE,
      retina_masks=True,
      imgsz=1024, #fix image size?
      conf=0.4,
      iou=0.9)
    prompt_process = FastSAMPrompt(IMAGE_PATH, results, device=DEVICE)
    masks = prompt_process.everything_prompt()
    masks_fastsam = masks.cpu().numpy().astype(bool)
    # Mask obtained from FastSam -> look for package mask (bigger masks)
    final_mask = torch.zeros((masks_fastsam.shape[1],masks_fastsam.shape[2]))
    areas = []
    count_bigger = 0
    remove = 0
    print("Pre",masks_fastsam.shape)
    for j in range(masks_fastsam.shape[0]):
      areas.append([j,np.sum(masks_fastsam[j])])
      if (np.sum(masks_fastsam[j])>masks_fastsam.shape[1]*masks_fastsam.shape[2]/2*0.4 and np.sum(masks_fastsam[j])<masks_fastsam.shape[1]*masks_fastsam.shape[2]/2*0.98): #consider as pizza box detection masks bigger than 70% of the half of the image (if the image is cut around the pizza box)
       count_bigger += 1
       print("area mask",np.sum(masks_fastsam[j]))
       print("area image",masks_fastsam.shape[1]*masks_fastsam.shape[2])
      if (np.sum(masks_fastsam[j])>masks_fastsam.shape[1]*masks_fastsam.shape[2]*0.7):
        remove += 1
    #if remove_index is not None:
    #  masks_fastsam = np.delete(masks_fastsam,remove_index)
    print("Selected box",count_bigger, remove)
    areas.sort(key=area_value)
    areas = np.array(areas)
    areas = areas.transpose(1,0)
    areas_index = areas[0]
    masks_fastsam = masks_fastsam[areas_index]
    for i in masks_fastsam:
       plt.imshow(i)
       plt.show()
       plt.clf()
    print("Post",masks_fastsam.shape)
    if count_bigger > 0: #found big enough masks -> package mask
      package_mask = masks_fastsam[-(count_bigger+remove):-remove]
      print(package_mask.shape)
      for j in range(package_mask.shape[0]):
        final_mask = torch.logical_or(final_mask, torch.tensor(package_mask[j]))
    else: # did not find big enough masks to be package masks
      print("no box face detected, creating mask of ones")
      final_mask = torch.ones((masks_fastsam.shape[1],masks_fastsam.shape[2]))


    plt.imshow(final_mask)
    plt.show()
    plt.clf()
    footprint=np.ones((50, 50))
    final_mask = skimage.morphology.binary_erosion(final_mask, footprint=footprint, out=None)
    final_mask = torch.tensor(final_mask)
    plt.imshow(final_mask)
    plt.show()
    plt.clf()
    

    # Converti l'immagine in HSV e usa il canale di saturazione per il rilevamento dei contorni
    myimg_hsv = rgb2hsv(undistorted_frame)
    saturation = myimg_hsv[:, :, 1]  # Canale di saturazione

    # Crea un'immagine binaria basata su una soglia sulla saturazione
    binary_image = np.where(saturation > 0.25, 1, 0).astype(np.uint8)
    output_image = torch.logical_and(torch.tensor(binary_image),final_mask)

    # Trova i contorni basati sull'immagine binaria
    contours_gray = find_contours(output_image.detach().numpy(), 0.8)

    # Create masks based on finally selected contours (intersection of fastsam box detection + grayscale)
    detected_mask = np.zeros((output_image.shape[0],output_image.shape[1])) #ref è binary_image nel codice robot
    for c in contours_gray: #contours_gray nel codice robot
        rr, cc = polygon(c[:, 0], c[:, 1], detected_mask.shape)
        detected_mask[rr, cc] = 1

    dilated_gtmask = dilation(detected_mask, square(20)) # modificare il valore di square per ampliare la maschera

    contours_larger = find_contours(dilated_gtmask, 0.8) #nuovi contorni

    contours_gray = contours_larger


    # Sort the contours by length (number of points)
    sorted_contours = sorted(contours_gray, key=len, reverse=True)

    # Select the top two largest contours
    largest_contours = sorted_contours[:2]

    # Se ci sono contorni, trova il contorno più lungo
    for best_contour in largest_contours:

        # Trasforma il contorno in coordinate reali
        real_coords = []
        for i,point in enumerate(best_contour):
            image_point = np.array([[point[1], point[0]]], dtype='float32')  # Coordinate (x, y) nel piano immagine

            # Mappa i punti immagine alle coordinate reali del robot
            real_x, real_y = map_image_to_real(image_point[0])

            # Stampa le coordinate reali
            if i == 0:
                move_robot_to_real_point(real_x, real_y, 1)
            else:
                move_robot_to_real_point(real_x, real_y, 0)

            # Aggiungi le coordinate reali alla lista
            real_coords.append([real_x, real_y])

        real_coords = np.array(real_coords)

        # Visualizzazione delle immagini e delle coordinate
        fig, ax = plt.subplots(1, 3, figsize=(18, 6))

        # Mostra l'immagine originale
        ax[0].imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        ax[0].set_title('Immagine originale')

        # Converti il contorno in un formato compatibile con OpenCV e disegna il contorno sull'immagine
        best_contour_cv2 = np.array([[[int(p[1]), int(p[0])]] for p in best_contour])
        cv2.drawContours(undistorted_frame, [best_contour_cv2], -1, (0, 255, 0), 2)
        ax[1].imshow(cv2.cvtColor(undistorted_frame, cv2.COLOR_BGR2RGB))
        ax[1].set_title('Immagine con il contorno più lungo')

        # Visualizza le coordinate reali su un grafico 2D
        ax[2].plot(real_coords[:, 0], real_coords[:, 1],  linewidth=0.8) #'ro-',
        ax[2].set_xlabel('X (dal basso verso l\'alto)')
        ax[2].set_ylabel('Y (da destra verso sinistra)')
        ax[2].invert_xaxis()  # Inverti l'asse X per simulare il movimento dal basso verso l'alto
        ax[2].invert_yaxis()  # Inverti l'asse Y per simulare il movimento da destra verso sinistra

        # Imposta i limiti dell'asse basati sugli angoli reali
        ax[2].set_xlim(real_x_min, real_x_max)
        ax[2].set_ylim(real_y_min, real_y_max)

        plt.tight_layout()

    else:
        print("Nessun contorno trovato.")
    plt.show()

# Rilascia la webcam e chiudi tutte le finestre
#cap.release()
cv2.destroyAllWindows()

time.sleep(2)
print('Move robot to start position')
#rtde_c.moveJ(robot_startposition, vel, acc)