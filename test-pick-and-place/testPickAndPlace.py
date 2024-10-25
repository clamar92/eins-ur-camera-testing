import cv2
import numpy as np
import time
import threading

# Parametri noti
real_center_x, real_center_y = 600, 300
image_center_x, image_center_y = 320, 240  # Centrali dell'immagine 640x480
z_height = 100

# Funzione per convertire le coordinate dell'immagine in coordinate reali
def image_to_real_coords(image_x, image_y):
    scale_x = real_center_x / image_center_x
    scale_y = real_center_y / image_center_y
    
    real_x = image_x * scale_x
    real_y = image_y * scale_y
    
    return real_x, real_y, z_height

# Funzione per muovere il robot
def move_robot(real_x, real_y, real_z):
    print(f"Muovo il robot alle coordinate: ({real_x:.2f}, {real_y:.2f}, {real_z})")
    time.sleep(20)
    global robot_moving
    robot_moving = False  # Robot ha finito di muoversi

def main():
    global robot_moving
    robot_moving = False  # Inizializza lo stato del robot come non in movimento

    # Inizializza la webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("Errore nell'apertura della webcam")
        return

    # Inizializza il frame precedente
    ret, prev_frame = cap.read()
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    # Inizializza il tempo di inattività
    last_motion_time = time.time()

    # Inizializza il flag di rilevamento di un oggetto
    object_detected = False

    # Inizializza una variabile per memorizzare il cerchio rilevato precedentemente
    prev_circle = None

    # Variabile per memorizzare l'ultima posizione stabile
    stable_position = None
    stable_time_start = None

    while True:
        # Cattura il fotogramma corrente
        ret, frame = cap.read()

        if not ret:
            print("Errore nella cattura del fotogramma")
            break

        # Converti il fotogramma in scala di grigi
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Trova i cerchi nell'immagine
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=50,
                                   param1=50, param2=30, minRadius=10, maxRadius=200)

        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")

            # Trova il cerchio più probabile basato sulla coerenza nel tempo
            if len(circles) == 1:
                (cX, cY, radius) = circles[0]

                # Filtra in base alle dimensioni del cerchio
                if radius < 10 or radius > 200:
                    continue

                # Filtra in base alla posizione del cerchio
                if cX < 100 or cX > 540 or cY < 100 or cY > 380:
                    continue

                # Verifica la coerenza nel tempo rispetto al cerchio precedente
                if prev_circle is not None:
                    prev_cX, prev_cY, prev_radius = prev_circle

                    # Calcola la distanza euclidea tra i centri dei cerchi
                    distance = np.sqrt((cX - prev_cX)**2 + (cY - prev_cY)**2)

                    # Se la distanza è troppo grande, considera il rilevamento non valido
                    if distance > 50:
                        continue

                prev_circle = (cX, cY, radius)

                # Converti le coordinate dell'immagine in coordinate reali
                real_x, real_y, real_z = image_to_real_coords(cX, cY)

                # Disegna il cerchio rilevato
                cv2.circle(frame, (cX, cY), radius, (0, 255, 0), 2)
                cv2.putText(frame, f"({real_x:.2f}, {real_y:.2f}, {real_z})", (cX + 10, cY - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Resetta il tempo di inattività se viene rilevato un oggetto
                last_motion_time = time.time()
                object_detected = True

                # Controlla la stabilità della posizione del cerchio
                if stable_position is None:
                    stable_position = (cX, cY)
                    stable_time_start = time.time()
                else:
                    stable_cX, stable_cY = stable_position
                    if abs(cX - stable_cX) <= 5 and abs(cY - stable_cY) <= 5:
                        # Se la posizione è rimasta stabile per 3 secondi, stampa il messaggio e muovi il robot
                        if time.time() - stable_time_start >= 3 and not robot_moving:
                            print(f"Nessun movimento rilevato. Ultima posizione stabile: ({real_x:.2f}, {real_y:.2f}, {real_z})")
                            robot_moving = True
                            threading.Thread(target=move_robot, args=(real_x, real_y, real_z)).start()
                    else:
                        stable_position = (cX, cY)
                        stable_time_start = time.time()
            else:
                prev_circle = None

        # Mostra il fotogramma con i cerchi rilevati e le coordinate reali
        cv2.imshow("Riconoscimento Oggetti Neri", frame)

        # Esci dal ciclo premendo 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Rilascia le risorse
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
