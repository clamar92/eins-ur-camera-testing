import cv2
import numpy as np
import time
import threading
import math
import rtde_control
import rtde_receive
import json

# Indirizzo IP del robot e porta RTDE
ROBOT_HOST = '192.168.137.198'
#ROBOT_HOST = '192.168.186.135'


# punti angolari salvati nel file image_corners_real_coords.json 
# definiti nell'ordine specificato: alto-sinistra, alto-destra, basso-sinistra, basso-destra

# Inizializzazione RTDE
rtde_c = rtde_control.RTDEControlInterface(ROBOT_HOST)
rtde_r = rtde_receive.RTDEReceiveInterface(ROBOT_HOST)

# Variabili globali per le coordinate
corner_points = []
robot_moving = False
free_drive_active = False

# Parametri di velocità e accelerazione
acc = 0.4
vel = 0.4

# Posizione iniziale del robot
robot_startposition = [math.radians(0),
                       math.radians(-95),
                       math.radians(-100),
                       math.radians(-78),
                       math.radians(88),
                       math.radians(0)]

def move_to_start_position():
    print('Move robot to start position')
    rtde_c.moveJ(robot_startposition, vel, acc)
    time.sleep(2)  # Attendi che il movimento sia completato

def free_drive_mode():
    global robot_moving, free_drive_active
    free_drive_active = True
    print("Free drive mode abilitato. Muovere il robot alla posizione desiderata e premere 'q' per salvare le coordinate.")
    rtde_c.teachMode()
    while free_drive_active:
        time.sleep(0.1)
    rtde_c.endTeachMode()
    robot_moving = False

def capture_real_coordinates():
    pose = rtde_r.getActualTCPPose()
    real_x, real_y, real_z = pose[:3]
    return real_x, real_y, real_z

def save_corner_point(corner_point, real_point):
    corner_points.append({'corner': corner_point, 'real': real_point})

def main():
    global robot_moving, free_drive_active

    # Inizializza la webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("Errore nell'apertura della webcam")
        return

    # Angoli dell'immagine
    image_corners = [(0, 0), (639, 0), (0, 479), (639, 479)]

    # Porta il robot alla posizione iniziale
    move_to_start_position()

    for i, corner in enumerate(image_corners):
        print(f"Spostare il robot al punto dell'angolo dell'immagine: {corner}. Premere 'q' per salvare questo punto.")
        threading.Thread(target=free_drive_mode).start()

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Errore nella cattura del fotogramma")
                break

            cv2.circle(frame, corner, 10, (0, 255, 0), 2)
            cv2.putText(frame, f"Muovere il robot all'angolo: {corner}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("Acquisizione Coordinate", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                free_drive_active = False
                robot_moving = True
                break

        real_point = capture_real_coordinates()
        save_corner_point(corner, real_point)
        print(f"Punto {i+1} salvato: Immagine {corner}, Reale {real_point}")

        # Porta il robot alla posizione iniziale
        time.sleep(2)
        move_to_start_position()

    cap.release()
    cv2.destroyAllWindows()

    with open('19_07_24/image_corners_real_coords.json', 'w') as f:
        json.dump(corner_points, f)

    print("Coordinate reali dei 4 angoli dell'immagine salvate.")

if __name__ == "__main__":
    main()
