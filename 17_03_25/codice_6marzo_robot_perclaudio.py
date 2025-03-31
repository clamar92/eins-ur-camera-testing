import cv2
import numpy as np
import json
import math
import torch
import matplotlib.pyplot as plt
from skimage.color import rgb2hsv
from skimage.measure import find_contours, approximate_polygon
import rtde_control
import rtde_receive
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
ROBOT_HOST = '192.168.137.74'

# Inizializzazione RTDE (commentato per ora)
rtde_c = rtde_control.RTDEControlInterface(ROBOT_HOST)
rtde_r = rtde_receive.RTDEReceiveInterface(ROBOT_HOST)

# Parametri di velocità e accelerazione
acc = 0.4
vel = 0.4



# Initial robot position
robot_startposition = [math.radians(21.90),
                       math.radians(-82.13),
                       math.radians(-87.16),
                       math.radians(-101.36),
                       math.radians(90.43),
                       math.radians(18.88)]

speed = [0, 0, -0.1, 0, 0, 0]
direction = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
acceleration = 1

#rtde_c.moveUntilContact(speed, direction, acceleration)

time.sleep(3)

rtde_c.moveJ(robot_startposition, vel, acc)
time.sleep(3)



center = rtde_r.getActualTCPPose()  # Centro del piano, ovvero posizione iniziale

# Risoluzione utilizzata durante la calibrazione
img_width = 640  # Larghezza dell'immagine
img_height = 480  # Altezza dell'immagine

# File contenente le coordinate dei punti immagine e dei punti reali
calibration_file = '17_03_25/image_corners_real_coords_aruco.json'

# File contenente i parametri di calibrazione della fotocamera
camera_calibration_file = '17_03_25/calibration_data.json'

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
    global center
    global rtde_r
    # print(f"Coordinate reali calcolate: X={real_x}, Y={real_y}, Z={real_z}")   
    #     

    if first_mov == 1:
        center[0] = real_x
        center[1] = real_y
        if is_within_bounds(real_x, real_y):
            time.sleep(2)
            rtde_c.moveL(center, 0.2, 0.2)
            time.sleep(2)
            rtde_c.moveUntilContact(speed, direction, acceleration)
            
            # while True:
            #     time.sleep(2)
            #     altezza = rtde_r.getActualTCPPose()
            #     if altezza[2] < 0.20:
            #         break
            #     rtde_c.moveL(center, 0.2, 0.2)
            #     rtde_c.moveUntilContact(speed, direction, acceleration)
            #     print(altezza[2])
                
            time.sleep(2)
            rtde_r = rtde_receive.RTDEReceiveInterface(ROBOT_HOST)
            center = rtde_r.getActualTCPPose()  # Centro del piano, ovvero posizione iniziale

        else:
            print(f"Coordinate fuori dai limiti: X={real_x}, Y={real_y}")
    else:
        center[0] = real_x
        center[1] = real_y
        if is_within_bounds(real_x, real_y):
            rtde_c.moveL(center, 0.4, 0.4)
        else:
            print(f"Coordinate fuori dai limiti: X={real_x}, Y={real_y}")

    print("Center:")
    print(center)


# Avvia la webcam e acquisisci una singola immagine
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, img_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, img_height)

if not cap.isOpened():
    print("Errore nell'apertura della webcam")
    exit()

# Cattura una singola immagine
ret, frame = cap.read()


#image_path = 'images_captured/image_led_e_neon_opposto1.jpg'
#frame = cv2.imread(image_path)
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
plt.imshow(frame)
plt.show()
ret = True


if not ret:
    print("Errore nella lettura del frame dalla webcam")
else:
    # Correggi la distorsione ottica utilizzando i parametri di calibrazione
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (img_width, img_height), 1)
    #undistorted_frame = cv2.undistort(frame, camera_matrix, dist_coeffs, None, new_camera_matrix)

    #IMAGE_PATH = 'image_led_e_due_neon2.jpg'
    IMAGE_PATH = 'image.jpg'
    fig, ax = plt.subplots(1, 5, figsize=(10,10))

    matplotlib.image.imsave(IMAGE_PATH, frame)
    # perform fastSAM processing on image -> detection of package contours
    results = fast_sam(
    source=IMAGE_PATH,
    device=DEVICE,
    retina_masks=True,
    imgsz=1024,
    conf=0.4,
    iou=0.9)
    prompt_process = FastSAMPrompt(IMAGE_PATH, results, device=DEVICE)
    masks = prompt_process.everything_prompt()
    prompt_process.plot(annotations=masks, output_path=f"output_fastsam.jpeg")

    masks_fastsam = masks.cpu().numpy().astype(bool)
    #annotated_image =annotate_image(image_path=IMAGE_PATH, masks=masks_fastsam)

    # create a new fastsam mask merging candidates for pizza box detection based on mask area (bigger ones, whole box or one of the faces)
    final_mask = torch.zeros((masks_fastsam.shape[1],masks_fastsam.shape[2]))
    areas = []
    count_bigger = 0
    for i in range(masks_fastsam.shape[0]):
        areas.append([i,np.sum(masks_fastsam[i])])
        if (np.sum(masks_fastsam[i])>masks_fastsam.shape[1]*masks_fastsam.shape[2]/2*0.4): #looking for pizza box shape: if a lot smaller than half of the image it is not the pizza box
        # with this distance, reference is smaller than 0.4*half the image
          count_bigger += 1
    areas.sort(key=area_value) #sort masks based on area values
    areas = np.array(areas)
    #print(areas)
    areas = areas.transpose(1,0)
    areas_index = areas[0]
    #print(areas_index)
    masks_fastsam = masks_fastsam[areas_index] #sort masks based on area values
    if count_bigger > 0:
        masks_fastsam = masks_fastsam[-count_bigger:] # only keep bigger masks
    areas = areas.transpose(1,0)
    max_area = areas[-1][1] # additional check: if a mask big as the whole image is created, remove it or it will break pizza box detection
    if (max_area == masks_fastsam.shape[1]*masks_fastsam.shape[2]): # additional check: if a mask big as the whole image is created, remove it or it will break pizza box detection
        masks_fastsam = masks_fastsam[:-1] # if it is there, it will be the biggest one, last position
    for i in range(masks_fastsam.shape[0]):
        fastsam_final_mask = torch.logical_or(final_mask, torch.tensor(masks_fastsam[i])) #create merged mask based on logical or of partial masks

    ax[0].title.set_text("FastSAM output")
    ax[0].imshow(fastsam_final_mask, cmap="gray")
    #print(fastsam_final_mask)

    # saturation for grayscale
    myimg_hsv = rgb2hsv(frame)
    a = myimg_hsv[:,:,0]
    b = myimg_hsv[:,:,1]
    c = myimg_hsv[:,:,2]
    saturation = myimg_hsv[:,:,2] # selected level
    saturation =saturation/max(saturation.flatten()) # normalize in 0-1
    #ax[0].title.set_text("a output")
    #ax[0].imshow(a, cmap='gray')
    #ax[1].title.set_text("b output")
    #ax[1].imshow(b, cmap='gray')
    #plt.show()
    #print(saturation)
    #print(max(saturation.flatten()),min(saturation.flatten()))

    saturation[fastsam_final_mask==False] = 1 # image outside the fastsam mask is forced WHITE
    ax[1].title.set_text("c output")
    ax[1].imshow(saturation, cmap='gray')

    # evaluate threshold for stain detection based on average color within the detected pizza box, reduced by a 0.9 factor
    avg_shade = np.sum(saturation[fastsam_final_mask==True]) 
    elements_shade = np.count_nonzero(fastsam_final_mask)
    avg_shade = avg_shade/elements_shade
    thr = avg_shade*0.9
    print("Selected threshold based on average color in the pizza box region",thr)

    #reference_binary_image = np.where(saturation > 0.65, 1, 0).astype(np.uint8) # se aumenta la soglia aumenta la sensibilità di rilevamento della macchia
    # apply selcted threshold: in the binary image, where saturation (white!!!) exceeds the thr, force to 1 (so make it WHITE)
    reference_binary_image = np.where(saturation > thr, 1, 0).astype(np.uint8) # se aumenta la soglia aumenta la sensibilità di rilevamento della macchia
    reference_binary_image = reference_binary_image.astype(bool) # here stain is black and background is white
    reference_binary_image = ~reference_binary_image # here stain is white and backround is black
    #filtered_img = torch.logical_or(final_mask, torch.tensor(reference_binary_image))
    #filtered_img = np.array(filtered_img).astype(bool)
    #filtered_img = ~filtered_img
    ax[2].title.set_text("binary output")
    ax[2].imshow(reference_binary_image, cmap='gray')
    #print(reference_binary_image)

    # combine fastsam and grayscal output: only look for stains inside the pizza box
    combined_mask = torch.logical_and(torch.tensor(reference_binary_image),fastsam_final_mask)
    ax[3].title.set_text("Grayscale output")
    ax[3].imshow(combined_mask, cmap='gray')

    # apply dilation to increase the contour area
    dilated_mask = dilation(combined_mask.detach().numpy(), square(10)) # modificare il valore di square per ampliare la maschera
    contours_gray_new = find_contours(dilated_mask, 0.8) #nuovi contorni

    #contours_gray_new = find_contours(combined_mask.detach().numpy(), 0.8)
    contours_gray_new = sorted(contours_gray_new,key=len)
    for c in contours_gray_new[-1:]:
        appr_contour = approximate_polygon(c,tolerance =2)
        ax[1].step(c.T[1],c.T[0],linewidth=1,c="r")
        ax[1].step(appr_contour.T[1],appr_contour.T[0],linewidth=1,c="b")

    plt.show()

    #fig, ax = plt.subplots()
    #myimg_hsv = rgb2hsv(frame)
    #c = myimg_hsv[:,:,2]
    #ax.imshow(c,cmap='gray') 
    #for c in contours_gray_new[-1:]:
    #    appr_contour = approximate_polygon(c,tolerance =2)
    #    ax.step(c.T[1],c.T[0],linewidth=1,c="r")
    #    ax.step(appr_contour.T[1],appr_contour.T[0],linewidth=1,c="b")
    #plt.show()  

    # Select the top two largest contours
    largest_contours = contours_gray_new[-1:]
    fig2, ax2 = plt.subplots(1, 2, figsize=(18, 6))

    for best_contour in largest_contours:
        # Visualizzazione delle immagini e delle coordinate
       # ax2[0].imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
       # ax2[0].set_title('Immagine originale')
        # Mostra l'immagine originale
        ax2[0].imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        ax2[0].set_title('Punti ridotti')
        # Converti il contorno in un formato compatibile con OpenCV e disegna il contorno sull'immagine
        best_contour_cv2 = np.array([[[int(p[1]), int(p[0])]] for p in best_contour])
        cv2.drawContours(frame, [best_contour_cv2], -1, (0, 255, 0), 2)
        ax2[1].imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        ax2[1].set_title('Immagine con il contorno più lungo')

        plt.tight_layout()
    plt.show()


    # Se ci sono contorni, trova il contorno più lungo
    for best_contour in largest_contours:

        # Visualizzazione delle immagini e delle coordinate
        fig2, ax2 = plt.subplots(1, 4, figsize=(18, 6))
        ax2[0].imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        ax2[0].set_title('Immagine originale')

        appr_contour = approximate_polygon(best_contour,tolerance =2)
        best_contour = appr_contour

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


        # Mostra l'immagine originale
        ax2[1].imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        ax2[1].set_title('Punti ridotti')

        # Converti il contorno in un formato compatibile con OpenCV e disegna il contorno sull'immagine
        best_contour_cv2 = np.array([[[int(p[1]), int(p[0])]] for p in best_contour])
        cv2.drawContours(frame, [best_contour_cv2], -1, (0, 255, 0), 2)
        ax2[2].imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        ax2[2].set_title('Immagine con il contorno più lungo')

        # Visualizza le coordinate reali su un grafico 2D
        ax2[3].plot(real_coords[:, 0], real_coords[:, 1],  linewidth=0.8) #'ro-',
        ax2[3].set_xlabel('X (dal basso verso l\'alto)')
        ax2[3].set_ylabel('Y (da destra verso sinistra)')
        ax2[3].invert_xaxis()  # Inverti l'asse X per simulare il movimento dal basso verso l'alto
        ax2[3].invert_yaxis()  # Inverti l'asse Y per simulare il movimento da destra verso sinistra

        # Imposta i limiti dell'asse basati sugli angoli reali
        ax2[3].set_xlim(real_x_min, real_x_max)
        ax2[3].set_ylim(real_y_min, real_y_max)

        plt.tight_layout()

        time.sleep(2)
        print('Move robot to start position')
        rtde_c.moveJ(robot_startposition, vel, acc)

    else:
        print("Nessun contorno trovato.")
    plt.show()

# Rilascia la webcam e chiudi tutte le finestre
#cap.release()
cv2.destroyAllWindows()

rtde_c.stopScript() 
#rtde_c.disconnect()



