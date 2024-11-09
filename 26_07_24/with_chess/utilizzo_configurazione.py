import cv2
import numpy as np
import json

# Funzione per selezionare i punti con il mouse
def select_points(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        param.append((x, y))
        if len(param) > 2:
            param.pop(0)  # Mantieni solo gli ultimi due punti selezionati

# Carica i dati di calibrazione
with open('26_07_24/with_chess/calibration_data.json', 'r') as f:
    calibration_data = json.load(f)

mtx = np.array(calibration_data['mtx'])
dist = np.array(calibration_data['dist'])

# Distanza nota dal piano in millimetri
known_distance = 440  # 1 metro = 1000 mm

# Avvia la webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
if not cap.isOpened():
    print("Errore nell'apertura della webcam")
    exit()

selected_points = []
cv2.namedWindow('Video Streaming')
cv2.setMouseCallback('Video Streaming', select_points, selected_points)

while True:
    # Leggi un frame dalla webcam
    ret, frame = cap.read()
    if not ret:
        print("Errore nella lettura del frame dalla webcam")
        break

    # Disegna i punti selezionati
    for point in selected_points:
        cv2.circle(frame, point, 5, (0, 0, 255), -1)  # Rosso, cerchio più grande

    # Se ci sono due punti, calcola e visualizza la distanza
    if len(selected_points) == 2:
        point1, point2 = selected_points

        # Rimuovi la distorsione dai punti selezionati
        points = np.array([point1, point2], dtype='float32').reshape(-1, 1, 2)
        undistorted_points = cv2.undistortPoints(points, mtx, dist, P=mtx)

        # Appiattisci l'array undistorted_points
        undistorted_points_flat = undistorted_points.reshape(-1, 2)

        # Calcola la scala in base alla distanza nota e alla focale
        # f_x e f_y sono i valori focali in pixel della matrice intrinseca
        f_x, f_y = mtx[0, 0], mtx[1, 1]

        # Converti le coordinate del piano immagine in coordinate reali
        # Utilizzando la distanza conosciuta (Z = 1000 mm) come profondità
        object_points_3d = np.zeros((2, 3))
        object_points_3d[:, 0] = (undistorted_points_flat[:, 0] * known_distance) / f_x
        object_points_3d[:, 1] = (undistorted_points_flat[:, 1] * known_distance) / f_y
        object_points_3d[:, 2] = known_distance  # Z = 1 metro

        # Calcola la distanza tra i due punti in 3D
        distance = np.linalg.norm(object_points_3d[0] - object_points_3d[1])
        cv2.putText(frame, f'Distanza: {distance:.2f} mm', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Mostra il frame
    cv2.imshow('Video Streaming', frame)

    # Esci con il tasto 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Rilascia la webcam e chiudi tutte le finestre
cap.release()
cv2.destroyAllWindows()
