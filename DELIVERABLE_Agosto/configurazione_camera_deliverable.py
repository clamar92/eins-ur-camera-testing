import cv2
import numpy as np
import json
import os
import time

# Dimensioni della scacchiera (numero di intersezioni interne)
chessboard_size = (9, 6)

# Dimensione del lato di un quadrato della scacchiera in millimetri
square_size = 17  # mm

# Preparazione dei punti oggetto, con dimensioni reali in millimetri
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
objp *= square_size  # Converti in millimetri

# Array per memorizzare i punti oggetto e i punti immagine da tutte le immagini
objpoints = []  # Punti 3D nel mondo reale
imgpoints = []  # Punti 2D nello spazio dell'immagine

# Numero di immagini di calibrazione da catturare
num_calibration_images = 20

# Directory per salvare le immagini della scacchiera
image_dir = 'DELIVERABLE_AGOSTO/checkerboard_images'
os.makedirs(image_dir, exist_ok=True)

# Avvia la webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("Errore nell'apertura della webcam")
    exit()

captured_images = 0

while captured_images < num_calibration_images:
    # Leggi un frame dalla webcam
    ret, frame = cap.read()

    if not ret:
        print("Errore nella lettura del frame dalla webcam")
        break

    # Converti l'immagine in scala di grigi
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Trova gli angoli del pattern di scacchiera
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

    # Se trovato, aggiungi punti oggetto e punti immagine
    if ret:
        objpoints.append(objp)
        imgpoints.append(corners)

        # Migliora la precisione dei rilevamenti degli angoli
        corners_refined = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1), 
            (cv2.TermCriteria_EPS + cv2.TermCriteria_MAX_ITER, 30, 0.001)
        )

        # Disegna e visualizza gli angoli
        frame = cv2.drawChessboardCorners(frame, chessboard_size, corners_refined, ret)
        cv2.imshow('Calibrazione', frame)
        
        # Salva l'immagine della scacchiera
        image_path = os.path.join(image_dir, f'checkerboard_{captured_images + 1}.png')
        cv2.imwrite(image_path, frame)
        
        captured_images += 1
        print(f"Immagine di calibrazione catturata: {captured_images}/{num_calibration_images}")
        
        # Aspetta che l'utente prema un tasto per scattare la prossima foto
        cv2.waitKey(0)
    else:
        cv2.imshow('Webcam', frame)
        cv2.waitKey(1)  # Breve attesa per gestire gli eventi GUI

# Rilascia la webcam
cap.release()
cv2.destroyAllWindows()

if len(objpoints) == 0 or len(imgpoints) == 0:
    print("Nessun punto trovato per la calibrazione")
    exit()

# Calibrazione della camera
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

if not ret:
    print("Calibrazione fallita")
    exit()

print("Matrice di calibrazione:")
print(mtx)
print("Coefficienti di distorsione:")
print(dist)

# Calcolo dell'errore di reproiezione in millimetri
mean_error = 0
total_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
    mean_error += error
    total_error += np.sum((imgpoints[i] - imgpoints2) ** 2)

mean_error /= len(objpoints)
mse = total_error / len(imgpoints)
print(f"Errore medio di reproiezione (in mm): {mean_error}")
print(f"Mean Squared Error (MSE): {mse}")

# Geometric Distortion Analysis
for i, img_path in enumerate(os.listdir(image_dir)):
    img = cv2.imread(os.path.join(image_dir, img_path))
    h, w = img.shape[:2]
    new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    undistorted_img = cv2.undistort(img, mtx, dist, None, new_camera_mtx)
    x, y, w, h = roi
    undistorted_img = undistorted_img[y:y+h, x:x+w]
    cv2.imwrite(os.path.join(image_dir, f'undistorted_{i + 1}.png'), undistorted_img)

# Salva i dati di calibrazione e trasformazione in un file
output_dir = 'DELIVERABLE_AGOSTO/data'
os.makedirs(output_dir, exist_ok=True)
calibration_data_path = os.path.join(output_dir, 'calibration_data.json')

calibration_data = {
    'mtx': mtx.tolist(),
    'dist': dist.tolist()
}

with open(calibration_data_path, 'w') as f:
    json.dump(calibration_data, f)

print(f"Dati di calibrazione e trasformazione salvati in {calibration_data_path}")
