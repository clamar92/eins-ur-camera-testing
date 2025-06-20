import cv2
import numpy as np
import time
import threading
import math
import rtde_control
import rtde_receive
import json

# Indirizzo IP del robot
ROBOT_HOST = '192.168.137.78'
#ROBOT_HOST = '192.168.186.136'


# Inizializzazione RTDE
rtde_c = rtde_control.RTDEControlInterface(ROBOT_HOST)
rtde_r = rtde_receive.RTDEReceiveInterface(ROBOT_HOST)

# Variabili globali
corner_points = []
free_drive_active = False

# Parametri di velocità e accelerazione
acc = 0.4
vel = 0.4

# Posizione iniziale del robot
robot_startposition = [math.radians(21.90),
                       math.radians(-82.13),
                       math.radians(-87.16),
                       math.radians(-101.36),
                       math.radians(90.43),
                       math.radians(18.88)]

def move_to_start_position():
    print('Move robot to start position')
    rtde_c.moveJ(robot_startposition, vel, acc)
    time.sleep(2)  # Attesa completamento movimento

def free_drive_mode():
    global free_drive_active
    free_drive_active = True
    print("Free drive mode abilitato. Muovere il robot alla posizione desiderata e premere 'q' per salvare.")
    rtde_c.teachMode()
    while free_drive_active:
        time.sleep(0.1)
    rtde_c.endTeachMode()

def capture_real_coordinates():
    pose = rtde_r.getActualTCPPose()
    return pose[:3]  # Restituisce solo x, y, z

def save_corner_point(corner_point, real_point):
    corner_points.append({'corner': corner_point, 'real': real_point})

def detect_aruco():
    global free_drive_active

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("Errore nell'apertura della webcam")
        return

    # Carica il dizionario ArUco
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    aruco_params = cv2.aruco.DetectorParameters()

    target_ids = [0, 1, 2, 3]  # Ordine specifico di ricerca

    for target_id in target_ids:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Errore nella cattura del fotogramma")
                continue

            # Rilevamento marker
            corners, ids, _ = cv2.aruco.detectMarkers(frame, aruco_dict, parameters=aruco_params)
            if ids is not None:
                ids = ids.flatten()
                if target_id in ids:
                    index = np.where(ids == target_id)[0][0]
                    center_x = int(np.mean(corners[index][0][:, 0]))
                    center_y = int(np.mean(corners[index][0][:, 1]))
                    print(f"Trovato ArUco {target_id} a ({center_x}, {center_y})")
                    
                    # Mostra l'ArUco identificato
                    cv2.circle(frame, (center_x, center_y), 10, (0, 255, 0), -1)
                    cv2.putText(frame, f"ArUco {target_id}", (center_x + 10, center_y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                    cv2.imshow("ArUco Detection", frame)
                    
                    # Entra in modalità freedrive per spostare il robot
                    print(f"Spostare il robot al centro di ArUco {target_id} e premere 'q' per salvare.")
                    threading.Thread(target=free_drive_mode).start()
                    while True:
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            free_drive_active = False
                            break
                    
                    real_point = capture_real_coordinates()
 
                    save_corner_point((center_x, center_y), real_point)
                    print(f"Salvato ArUco {target_id}: Immagine ({center_x}, {center_y}), Reale {real_point}")
                    
                    # Riporta il robot alla posizione iniziale
                    time.sleep(2)
                    move_to_start_position()
                    break

            cv2.imshow("ArUco Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('e'):
                cap.release()
                cv2.destroyAllWindows()
                return

    cap.release()
    cv2.destroyAllWindows()

    with open('20_06_25/image_corners_real_coords_aruco.json', 'w') as f:
        json.dump(corner_points, f)

    print("Coordinate reali dei marker ArUco salvate.")

if __name__ == "__main__":
    move_to_start_position()
    detect_aruco()
