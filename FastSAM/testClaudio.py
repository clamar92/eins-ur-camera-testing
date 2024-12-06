import cv2
import numpy as np
import json
import torch
import matplotlib.pyplot as plt
from skimage.color import rgb2hsv
from skimage.measure import find_contours
from skimage.morphology import square, dilation
from skimage.draw import polygon
import matplotlib.image
import skimage
from fastsam import FastSAM, FastSAMPrompt

# Percorsi dei checkpoint e configurazioni
FAST_SAM_CHECKPOINT_PATH = 'weights/FastSAM.pt'
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
fast_sam = FastSAM(FAST_SAM_CHECKPOINT_PATH)

# File di calibrazione e parametri fotocamera
calibration_file = '04_10_24/image_corners_real_coords.json'
camera_calibration_file = '04_10_24/calibration_data.json'
img_width = 640
img_height = 480

def area_value(area_element):
    return area_element[1]

# Carica parametri di calibrazione fotocamera
with open(camera_calibration_file, 'r') as f:
    calib_data = json.load(f)
    camera_matrix = np.array(calib_data['mtx'])
    dist_coeffs = np.array(calib_data['dist'])

# Carica i punti di calibrazione (coordinate immagine -> coordinate reali robot)
with open(calibration_file, 'r') as f:
    calibration_data = json.load(f)
image_points = np.array([point['corner'] for point in calibration_data], dtype=np.float32)
real_points = np.array([point['real'][:2] for point in calibration_data], dtype=np.float32)

# Calcola la matrice di omografia immagine -> mondo reale
homography_matrix, _ = cv2.findHomography(image_points, real_points)

# Funzione per convertire i punti immagine in coordinate reali
def map_image_to_real(image_point):
    image_point_homog = np.array([image_point[0], image_point[1], 1], dtype=np.float32)
    real_point_homog = np.dot(homography_matrix, image_point_homog)
    real_x = real_point_homog[0] / real_point_homog[2]
    real_y = real_point_homog[1] / real_point_homog[2]
    return real_x, real_y

# Carica l'immagine specifica invece di acquisirla dalla webcam
IMAGE_PATH = 'images_captured/image_led_e_due_neon1.jpg'
frame = cv2.imread(IMAGE_PATH)
ret = frame is not None

if not ret:
    print("Errore nella lettura dell'immagine da file")
else:
    # Correggi la distorsione ottica
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (img_width, img_height), 1)
    undistorted_frame = cv2.undistort(frame, camera_matrix, dist_coeffs, None, new_camera_matrix)

    # Salva l'immagine corretta
    matplotlib.image.imsave(IMAGE_PATH, undistorted_frame)

    # Applica FastSAM per rilevare i contorni del cartone
    results = fast_sam(
        source=IMAGE_PATH,
        device=DEVICE,
        retina_masks=True,
        imgsz=1024,
        conf=0.4,
        iou=0.9
    )
    prompt_process = FastSAMPrompt(IMAGE_PATH, results, device=DEVICE)
    masks = prompt_process.everything_prompt()
    masks_fastsam = masks.cpu().numpy().astype(bool)

    # Seleziona la maschera più grande
    final_mask = torch.zeros((masks_fastsam.shape[1], masks_fastsam.shape[2]))
    areas = [(j, np.sum(masks_fastsam[j])) for j in range(masks_fastsam.shape[0])]
    areas = sorted(areas, key=lambda x: x[1], reverse=True)
    largest_mask_index = areas[0][0]
    largest_mask = masks_fastsam[largest_mask_index]

    # Visualizza la maschera più grande
    plt.imshow(largest_mask)
    plt.title("Maschera più grande (presumibilmente cartone)")
    plt.show()
    plt.clf()

    # Assegna la maschera più grande come final_mask
    final_mask = torch.tensor(largest_mask)

    # Applica la maschera a un'immagine in scala di grigi
    gray_image = cv2.cvtColor(undistorted_frame, cv2.COLOR_BGR2GRAY)
    masked_gray_image = gray_image * final_mask.numpy().astype(np.uint8)

    # Identifica le aree più scure (potenziali macchie di unto)
    _, thresholded_unto = cv2.threshold(masked_gray_image, 50, 255, cv2.THRESH_BINARY_INV)
    thresholded_unto = cv2.bitwise_and(thresholded_unto, thresholded_unto, mask=final_mask.numpy().astype(np.uint8))

    # Visualizza le macchie di unto
    plt.imshow(thresholded_unto, cmap='gray')
    plt.title("Macchie di unto rilevate")
    plt.show()
