import cv2
import numpy as np
import json

# Carica i dati dei vertici e delle coordinate reali
with open('26_07_24/without_chess/image_corners_real_coords.json', 'r') as f:
    corner_points = json.load(f)

# Estrai i punti immagine e i punti reali
image_corners = np.array([cp['corner'] for cp in corner_points], dtype=np.float32)
real_corners = np.array([cp['real'][:2] for cp in corner_points], dtype=np.float32)  # Usa solo x, y reali

# Calcola la matrice omografica
H, _ = cv2.findHomography(image_corners, real_corners)

selected_real_points = []  # Lista per memorizzare i punti reali calcolati
selected_image_points = []  # Lista per memorizzare i punti immagine cliccati

# Funzione per selezionare i punti con il mouse e convertire le coordinate
def select_point(event, x, y, flags, param):
    global selected_real_points, selected_image_points
    if event == cv2.EVENT_LBUTTONDOWN:
        # Converte le coordinate del punto cliccato
        image_point = np.array([[x, y]], dtype=np.float32).reshape(-1, 1, 2)
        real_point = cv2.perspectiveTransform(image_point, H)
        real_x, real_y = real_point[0][0]

        # Check se il punto è all'interno dei limiti definiti dai vertici
        min_x, max_x = np.min(real_corners[:, 0]), np.max(real_corners[:, 0])
        min_y, max_y = np.min(real_corners[:, 1]), np.max(real_corners[:, 1])
        if min_x <= real_x <= max_x and min_y <= real_y <= max_y:
            print(f"Coordinata reale: X = {real_x:.2f}, Y = {real_y:.2f}")
            selected_real_points.append((real_x, real_y))
            selected_image_points.append((x, y))
            
            # Mantieni solo gli ultimi due punti
            if len(selected_real_points) > 2:
                selected_real_points.pop(0)
                selected_image_points.pop(0)
        else:
            print("Il punto selezionato è fuori dai limiti definiti dai vertici.")

# Inizializza la webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("Errore nell'apertura della webcam")
    exit()

cv2.namedWindow('Video Streaming')
cv2.setMouseCallback('Video Streaming', select_point)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Errore nella lettura del frame dalla webcam")
        break

    # Disegna i vertici e mostra le istruzioni
    for corner in image_corners:
        corner_int = tuple(int(c) for c in corner)
        cv2.circle(frame, corner_int, 5, (0, 255, 0), -1)  # Cerchi verdi sui vertici
    
    # Disegna i cerchi rossi sui punti cliccati
    for point in selected_image_points:
        cv2.circle(frame, point, 5, (0, 0, 255), -1)  # Cerchi rossi sui punti cliccati
    
    # Se ci sono due punti selezionati, calcola e mostra la distanza
    if len(selected_real_points) == 2:
        x1, y1 = selected_real_points[0]
        x2, y2 = selected_real_points[1]
        distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        mid_point = (int((selected_image_points[0][0] + selected_image_points[1][0]) / 2),
                     int((selected_image_points[0][1] + selected_image_points[1][1]) / 2))
        cv2.putText(frame, f'Distanza: {distance:.2f}', mid_point,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.putText(frame, "Clicca su due punti per vedere la distanza reale.", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    cv2.imshow('Video Streaming', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
