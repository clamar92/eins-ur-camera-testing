import math
import time
import rtde_control
import rtde_receive

# Impostazioni del robot
#ROBOT_HOST = '192.168.186.136'
ROBOT_HOST = '192.168.137.221'


ACC = 0.2  # Accelerazione
VEL = 0.2  # Velocità
RADIUS = 0.2  # Raggio del cerchio (30 cm)
NUM_POINTS = 50  # Numero di punti per la traiettoria circolare
BLEND_RADIUS = 0.05  # Blending per transizioni fluide

# Inizializzazione RTDE
rtde_c = rtde_control.RTDEControlInterface(ROBOT_HOST)
rtde_r = rtde_receive.RTDEReceiveInterface(ROBOT_HOST)

# Posizione iniziale del robot SIMULATORE
# robot_startposition = [
#     math.radians(91.90),
#     math.radians(-86.13),
#     math.radians(-113.16),
#     math.radians(-71.36),
#     math.radians(91.43),
#     math.radians(-1.88)
# ]

# Initial robot position REALE
robot_startposition = [math.radians(21.90),
                       math.radians(-82.13),
                       math.radians(-87.16),
                       math.radians(-101.36),
                       math.radians(90.43),
                       math.radians(18.88)]

# Sposta il robot nella posizione iniziale
print("Spostamento nella posizione iniziale...")
rtde_c.moveJ(robot_startposition, VEL, ACC)
time.sleep(2)

# Ottieni la posizione attuale del TCP (Tool Center Point)
center_pose = rtde_r.getActualTCPPose()
center_x, center_y, center_z = center_pose[0], center_pose[1], center_pose[2]  # Mantieni le unità originali
print(f"Posizione iniziale: X={center_x:.3f}, Y={center_y:.3f}, Z={center_z:.3f}")

# Genera punti della traiettoria circolare mantenendo la stessa altezza
circle_points = []
for i in range(NUM_POINTS):
    angle = (2 * math.pi * i) / NUM_POINTS  # Angolo corrente
    x = center_x + RADIUS * math.cos(angle)
    y = center_y + RADIUS * math.sin(angle)
    circle_points.append([x, y, center_z, center_pose[3], center_pose[4], center_pose[5]])

# Muovi il robot lungo la traiettoria circolare con blending per movimenti fluidi
print("Inizio movimento circolare fluido...")
for point in circle_points:
    rtde_c.moveL(point, VEL, ACC, True)  # Asynchronous move to avoid stopping
    time.sleep(0.8)

# Ritorna alla posizione iniziale
print("Ritorno alla posizione iniziale...")
rtde_c.moveJ(robot_startposition, VEL, ACC)
time.sleep(2)

print("Movimento circolare completato.")