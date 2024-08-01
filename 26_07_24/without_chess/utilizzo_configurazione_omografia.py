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

selected_real_point = None  # Variabile per memorizzare l'ultimo punto reale calcolato
selected_image_point = None  # Variabile per memorizzare l'ultimo punto immagine cliccato

# Funzione per selezionare i punti con il mouse e convertire le coordinate
def select_point(event, x, y, flags, param):
    global selected_real_point, selected_image_point
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
            selected_real_point = (real_x, real_y)
            selected_image_point = (x, y)  # Memorizza il punto immagine
        else:
            print("Il punto selezionato è fuori dai limiti definiti dai vertici.")
            selected_real_point = None
            selected_image_point = None

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
    
    # Disegna il cerchio rosso sull'ultimo punto cliccato
    if selected_image_point:
        cv2.circle(frame, selected_image_point, 5, (0, 0, 255), -1)  # Cerchio rosso sul punto cliccato
        cv2.putText(frame, f'X: {selected_real_point[0]:.2f}, Y: {selected_real_point[1]:.2f}', 
                    (selected_image_point[0] + 10, selected_image_point[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.putText(frame, "Clicca su un punto per vedere le coordinate reali.", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    cv2.imshow('Video Streaming', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
