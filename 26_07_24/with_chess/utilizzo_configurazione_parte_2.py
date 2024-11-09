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
known_distance = 750  # mm

# Coordinate fisiche corrispondenti all'angolo in basso a sinistra dell'immagine (in metri)
physical_origin = np.array([0.40, -0.38])  # metri

# Dimensioni dell'immagine in pixel
image_width = 640
image_height = 480

# Avvia la webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, image_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, image_height)
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

    # Se ci sono due punti, calcola e visualizza la distanza e le coordinate reali
    if len(selected_points) == 2:
        point1, point2 = selected_points

        # Rimuovi la distorsione dai punti selezionati
        points = np.array([point1, point2], dtype='float32').reshape(-1, 1, 2)
        undistorted_points = cv2.undistortPoints(points, mtx, dist, P=mtx)

        # Appiattisci l'array undistorted_points
        undistorted_points_flat = undistorted_points.reshape(-1, 2)

        # Calcola le coordinate reali in metri
        real_coordinates = physical_origin + (undistorted_points_flat / [image_width, image_height]) * known_distance / 1000

        # Calcola la distanza tra i due punti in 2D
        distance = np.linalg.norm(real_coordinates[0] - real_coordinates[1])
        
        # Mostra le coordinate reali e la distanza nell'immagine
        cv2.putText(frame, f'Distanza: {distance:.2f} m', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f'P1: {real_coordinates[0][0]:.2f}, {real_coordinates[0][1]:.2f} m', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f'P2: {real_coordinates[1][0]:.2f}, {real_coordinates[1][1]:.2f} m', (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


    # Mostra il frame
    cv2.imshow('Video Streaming', frame)

    # Esci con il tasto 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Rilascia la webcam e chiudi tutte le finestre
cap.release()
cv2.destroyAllWindows()
