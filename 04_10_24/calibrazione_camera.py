import cv2
import numpy as np
import json
import os

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

# Calibrazione della fotocamera utilizzando i punti rilevati
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

if not ret:
    print("Calibrazione fallita")
    exit()

# Stampa i risultati
print("Matrice di calibrazione:")
print(mtx)
print("Coefficienti di distorsione:")
print(dist)

# Calcolo dell'errore di reproiezione in millimetri
mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
    mean_error += error

mean_error /= len(objpoints)
print(f"Errore medio di reproiezione (in mm): {mean_error}")

# Salva i dati di calibrazione e trasformazione in un file
output_dir = '13_09_24/'
os.makedirs(output_dir, exist_ok=True)
calibration_data_path = os.path.join(output_dir, 'calibration_data.json')

calibration_data = {
    'mtx': mtx.tolist(),
    'dist': dist.tolist(),
    'rvecs': [rvec.tolist() for rvec in rvecs],
    'tvecs': [tvec.tolist() for tvec in tvecs],
    'mean_error': mean_error
}

with open(calibration_data_path, 'w') as f:
    json.dump(calibration_data, f)

print(f"Dati di calibrazione e trasformazione salvati in {calibration_data_path}")

# Testare la calibrazione su una nuova immagine
# Se hai immagini da testare, puoi aggiungere questo codice per verificare
# img = cv2.imread('path_to_a_new_image.jpg')  # Usa una nuova immagine per testare la calibrazione
# h, w = img.shape[:2]
# new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
# undistorted_img = cv2.undistort(img, mtx, dist, None, new_camera_matrix)
# cv2.imshow('Undistorted Image', undistorted_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
