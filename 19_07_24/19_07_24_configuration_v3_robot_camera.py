import cv2
import numpy as np
import rtde_control
import rtde_receive
import json

def capture_real_coordinates(rtde_r):
    pose = rtde_r.getActualTCPPose()
    real_x, real_y, real_z = pose[:3]
    return real_x, real_y, real_z

# Indirizzo IP del robot e porta RTDE
ROBOT_HOST = '192.168.186.135'

# Inizializzazione RTDE
rtde_c = rtde_control.RTDEControlInterface(ROBOT_HOST)
rtde_r = rtde_receive.RTDEReceiveInterface(ROBOT_HOST)

# Preparazione dei punti oggetto, come (0,0,0), (1,0,0), (2,0,0) ....,(11,8,0)
objp = np.zeros((8*11,3), np.float32)
objp[:,:2] = np.mgrid[0:11,0:8].T.reshape(-1,2)

# Array per memorizzare i punti oggetto e i punti immagine da tutte le immagini.
objpoints = []  # punti 3d nel mondo reale
imgpoints = []  # punti 2d nello spazio dell'immagine.
robotpoints = []  # punti 3d nel sistema del robot

# Numero di immagini di calibrazione da catturare
num_calibration_images = 4

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
        imgpoints[-1] = corners_refined  # Usa i punti raffinati

        # Disegna e visualizza gli angoli
        img = cv2.drawChessboardCorners(frame, (11,8), corners_refined, ret)
        cv2.imshow('img', img)

        captured_images += 1
        print(f"Immagine di calibrazione catturata: {captured_images}/{num_calibration_images}")

        # Registra la posizione del robot
        robot_point = capture_real_coordinates(rtde_r)
        robotpoints.append(robot_point)

        # Aspetta che l'utente prema un tasto per scattare la prossima foto
        print("Premi un tasto per scattare la prossima foto...")
        cv2.waitKey(0)  # Wait for a key press to capture next image
    else:
        cv2.imshow('Webcam', frame)
        cv2.waitKey(1)  # Short wait to handle GUI events

# Rilascia la webcam
cap.release()
cv2.destroyAllWindows()

if len(objpoints) == 0 or len(imgpoints) == 0 or len(robotpoints) == 0:
    print("Nessun punto trovato per la calibrazione")
    exit()

# Assicurati che objpoints e imgpoints abbiano la stessa lunghezza
if len(objpoints) != len(imgpoints) or len(imgpoints) != len(robotpoints):
    print("Errore: il numero di punti oggetto, punti immagine e punti robot non corrisponde.")
    print(f"Punti oggetto: {len(objpoints)}, Punti immagine: {len(imgpoints)}, Punti robot: {len(robotpoints)}")
    exit()

# Calibrazione della camera
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

print("Matrice di calibrazione:")
print(mtx)
print("Coefficienti di distorsione:")
print(dist)

# Prepara i punti immagine e robot per cv2.solvePnP
# Utilizziamo solo i primi n punti corrispondenti per cv2.solvePnP
n = min(len(objp), len(imgpoints[0]))
object_points = objp[:n]  # Prendi i primi n punti oggetto
image_points = np.array([imgpoints[0][i][0] for i in range(n)], dtype=np.float32)  # Prendi i primi n punti immagine
robot_points = np.array([robotpoints[0]], dtype=np.float32)  # Prendi i primi n punti robot

# Risolvi per la posa della camera rispetto al robot usando cv2.solvePnP
success, rvec, tvec = cv2.solvePnP(object_points, image_points, mtx, dist)

if not success:
    print("Errore nella risoluzione di PnP")
    exit()

# Converti la rotazione vettore in matrice di rotazione
rotation_matrix, _ = cv2.Rodrigues(rvec)

# Crea la matrice di trasformazione camera-robot
camera_to_robot_matrix = np.hstack((rotation_matrix, tvec))
camera_to_robot_matrix = np.vstack((camera_to_robot_matrix, [0, 0, 0, 1]))

print("Matrice di trasformazione da camera a robot:")
print(camera_to_robot_matrix)

# Salva i dati di calibrazione e trasformazione in un file
calibration_data = {
    'mtx': mtx.tolist(),
    'dist': dist.tolist(),
    'camera_to_robot_matrix': camera_to_robot_matrix.tolist()
}

with open('19_07_24/conf_v2.json', 'w') as f:
    json.dump(calibration_data, f)

print("Dati di calibrazione e trasformazione salvati in conf_v2.json")
