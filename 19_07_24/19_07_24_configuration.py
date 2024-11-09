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

# Inizializzazione RTDE
rtde_c = rtde_control.RTDEControlInterface(ROBOT_HOST)
rtde_r = rtde_receive.RTDEReceiveInterface(ROBOT_HOST)

# Parametri noti
image_center_x, image_center_y = 320, 240  # Centrali dell'immagine 640x480
z_height = None

# Variabili globali per le coordinate
image_points = []
real_points = []
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
    global z_height
    pose = rtde_r.getActualTCPPose()
    real_x, real_y, real_z = pose[:3]
    if z_height is None:
        z_height = real_z  # Salva l'altezza Z
    return real_x, real_y, z_height

def save_point(image_point, real_point):
    image_points.append(image_point)
    real_points.append(real_point)

def calculate_transformation_parameters():
    image_pts = np.array(image_points, dtype=np.float32)
    real_pts = np.array(real_points, dtype=np.float32)

    if len(image_pts) < 4 or len(real_pts) < 4:
        print("Errore: servono almeno 4 punti per calcolare la trasformazione.")
        return None

    transformation_matrix, _ = cv2.findHomography(image_pts, real_pts)
    if transformation_matrix is None:
        print("Errore nel calcolo della matrice di trasformazione.")
    return transformation_matrix

def save_transformation_parameters(transformation_matrix, filename='19_07_24/transformation_params.json'):
    if transformation_matrix is None:
        print("Errore: la matrice di trasformazione è None e non può essere salvata.")
        return
    data = {
        'transformation_matrix': transformation_matrix.tolist()
    }
    with open(filename, 'w') as f:
        json.dump(data, f)

def image_to_real_coords(image_x, image_y, transformation_matrix):
    point = np.array([[image_x, image_y]], dtype=np.float32)
    real_point = cv2.perspectiveTransform(point[None, :, :], transformation_matrix)
    real_x, real_y = real_point[0][0]
    return real_x, real_y, z_height

def main():
    global robot_moving, free_drive_active
    # Inizializza la webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("Errore nell'apertura della webcam")
        return

    # Porta il robot alla posizione iniziale
    move_to_start_position()

    point_count = 0

    while point_count < 8:
        ret, frame = cap.read()
        if not ret:
            print("Errore nella cattura del fotogramma")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=50,
                                   param1=50, param2=30, minRadius=10, maxRadius=200)

        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")

            if len(circles) == 1:
                (cX, cY, radius) = circles[0]
                cv2.circle(frame, (cX, cY), radius, (0, 255, 0), 2)
                cv2.putText(frame, f"Centro: ({cX}, {cY})", (cX + 10, cY - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                print(f"Centro del cerchio rilevato: ({cX}, {cY}). Premere 'q' per salvare questo punto.")
                
                cv2.imshow("Riconoscimento Oggetti Neri", frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    image_point = (cX, cY)
                    print("Abilita il free drive per muovere il robot al centro del cerchio e premi 'q'.")
                    threading.Thread(target=free_drive_mode).start()
                    
                    while not robot_moving:
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            free_drive_active = False
                            robot_moving = True
                    
                    real_point = capture_real_coordinates()
                    save_point(image_point, real_point)
                    point_count += 1
                    print(f"Punto {point_count} salvato: Immagine {image_point}, Reale {real_point}")

                    # Porta il robot alla posizione iniziale
                    time.sleep(2)
                    move_to_start_position()

        cv2.imshow("Riconoscimento Oggetti Neri", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if len(image_points) == 4:
        transformation_matrix = calculate_transformation_parameters()
        save_transformation_parameters(transformation_matrix)
        print("Parametri di trasformazione calcolati e salvati.")
        
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Errore nella cattura del fotogramma")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=50,
                                       param1=50, param2=30, minRadius=10, maxRadius=200)

            if circles is not None:
                circles = np.round(circles[0, :]).astype("int")

                for (cX, cY, radius) in circles:
                    real_x, real_y, real_z = image_to_real_coords(cX, cY, transformation_matrix)
                    cv2.circle(frame, (cX, cY), radius, (0, 255, 0), 2)
                    cv2.putText(frame, f"({real_x:.2f}, {real_y:.2f}, {real_z:.2f})", (cX + 10, cY - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            cv2.imshow("Riconoscimento Oggetti Neri", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
