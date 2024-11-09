import cv2
import numpy as np
import json
import os

# Carica i dati di calibrazione salvati
calibration_data_path = '13_09_2024/calibration_data.json'
if not os.path.exists(calibration_data_path):
    print(f"Errore: il file {calibration_data_path} non esiste. Esegui prima la calibrazione.")
    exit()

with open(calibration_data_path, 'r') as f:
    calibration_data = json.load(f)

# Carica i parametri della calibrazione
mtx = np.array(calibration_data['mtx'])
dist = np.array(calibration_data['dist'])

# Avvia la webcam per scattare una foto
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("Errore nell'apertura della webcam")
    exit()

ret, frame = cap.read()
if not ret:
    print("Errore nella cattura dell'immagine dalla webcam")
    cap.release()
    exit()

# Salva l'immagine catturata per il test
image_path = '13_09_2024/captured_image.jpg'
cv2.imwrite(image_path, frame)
print(f"Immagine catturata e salvata come {image_path}")

# Rilascia la webcam
cap.release()

# Leggi l'immagine appena salvata per il test
img = cv2.imread(image_path)

# Controlla se l'immagine è stata caricata correttamente
if img is None:
    print("Errore nel caricamento dell'immagine.")
    exit()

# Ottieni le dimensioni dell'immagine
h, w = img.shape[:2]

# Ottieni la nuova matrice della fotocamera per correggere la distorsione
new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

# Correggi la distorsione
undistorted_img = cv2.undistort(img, mtx, dist, None, new_camera_matrix)

# Ritaglia l'immagine usando l'area ROI (Region of Interest)
x, y, w, h = roi
undistorted_img = undistorted_img[y:y + h, x:x + w]

# Mostra l'immagine originale e quella corretta
cv2.imshow('Original Image', img)
cv2.imshow('Undistorted Image', undistorted_img)

# Aspetta che l'utente prema un tasto per chiudere le finestre
cv2.waitKey(0)
cv2.destroyAllWindows()

# Salva l'immagine non distorta
undistorted_image_path = '13_09_2024/undistorted_image.jpg'
cv2.imwrite(undistorted_image_path, undistorted_img)
print(f"Immagine non distorta salvata come {undistorted_image_path}")
