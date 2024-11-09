import cv2
import numpy as np
import json


# Preparazione dei punti oggetto, come (0,0,0), (1,0,0), (2,0,0) ....,(11,8,0)
objp = np.zeros((8*11,3), np.float32)
objp[:,:2] = np.mgrid[0:11,0:8].T.reshape(-1,2)

# Array per memorizzare i punti oggetto e i punti immagine da tutte le immagini.
objpoints = [] # punti 3d nel mondo reale
imgpoints = [] # punti 2d nello spazio dell'immagine.

# Numero di immagini di calibrazione da catturare
num_calibration_images = 10

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

    # Applicare un filtro blur per ridurre il rumore
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Trova gli angoli del pattern di scacchiera
    ret, corners = cv2.findChessboardCorners(gray, (11,8), None)

    # Se trovato, aggiungi punti oggetto, punti immagine
    if ret:
        objpoints.append(objp)
        imgpoints.append(corners)

        # Migliora la precisione dei rilevamenti degli angoli
        corners_refined = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), (cv2.TermCriteria_EPS + cv2.TermCriteria_MAX_ITER, 30, 0.001))

        # Disegna e visualizza gli angoli
        img = cv2.drawChessboardCorners(frame, (11,8), corners_refined, ret)
        cv2.imshow('img', img)
        
        captured_images += 1
        print(f"Immagine di calibrazione catturata: {captured_images}/{num_calibration_images}")
        
        # Aspetta che l'utente prema un tasto per scattare la prossima foto
        print("Premi un tasto per scattare la prossima foto...")
        cv2.waitKey(0)  # Wait for a key press to capture next image
    else:
        cv2.imshow('Webcam', frame)
        cv2.waitKey(1)  # Short wait to handle GUI events

# Rilascia la webcam
cap.release()
cv2.destroyAllWindows()

if len(objpoints) == 0 or len(imgpoints) == 0:
    print("Nessun punto trovato per la calibrazione")
    exit()

# Assicurati che objpoints e imgpoints abbiano la stessa lunghezza
if len(objpoints) != len(imgpoints):
    print("Errore: il numero di punti oggetto e punti immagine non corrisponde.")
    print(f"Punti oggetto: {len(objpoints)}, Punti immagine: {len(imgpoints)}")
    exit()

# Calibrazione della camera
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

print("Matrice di calibrazione:")
print(mtx)
print("Coefficienti di distorsione:")
print(dist)

# Salva i dati di calibrazione e trasformazione in un file
calibration_data = {
    'mtx': mtx.tolist(),
    'dist': dist.tolist()
}

with open('19_07_24/conf_v2.json', 'w') as f:
    json.dump(calibration_data, f)

print("Dati di calibrazione e trasformazione salvati in conf_v2.json")
