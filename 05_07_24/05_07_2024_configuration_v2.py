import cv2
import numpy as np
import time
import threading
import math
import rtde_control
import rtde_receive

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
    # Calcola i parametri di trasformazione lineare
    (x1_image, y1_image), (x2_image, y2_image) = image_points
    (x1_real, y1_real), (x2_real, y2_real) = real_points

    scale_x = (x2_real - x1_real) / (x2_image - x1_image)
    scale_y = (y2_real - y1_real) / (y2_image - y1_image)

    translation_x = x1_real - scale_x * x1_image
    translation_y = y1_real - scale_y * y1_image

    return scale_x, scale_y, translation_x, translation_y

def image_to_real_coords(image_x, image_y, scale_x, scale_y, translation_x, translation_y):
    real_x = scale_x * image_x + translation_x
    real_y = scale_y * image_y + translation_y
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

    while point_count < 2:
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

    if len(image_points) == 2:
        scale_x, scale_y, translation_x, translation_y = calculate_transformation_parameters()
        print("Parametri di trasformazione calcolati.")
        
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
                    real_x, real_y, real_z = image_to_real_coords(cX, cY, scale_x, scale_y, translation_x, translation_y)
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
