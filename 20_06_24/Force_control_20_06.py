import math
import time
import rtde_control
import rtde_receive

# Indirizzo IP del robot e porta RTDE
ROBOT_HOST = '192.168.137.179'
RTDE_PORT = 30004

# Inizializzazione RTDE
rtde_c = rtde_control.RTDEControlInterface(ROBOT_HOST)
rtde_r = rtde_receive.RTDEReceiveInterface(ROBOT_HOST)

# Parametri di movimento
acc = 0.4
vel = 0.4

# Posizione iniziale del robot
robot_startposition = [math.radians(0),
                       math.radians(-95),
                       math.radians(-100),
                       math.radians(-78),
                       math.radians(88),
                       math.radians(0)]

print('Move robot to start position')
rtde_c.moveJ(robot_startposition, vel, acc)
time.sleep(2)

center = rtde_r.getActualTCPPose()  # Centro del piano, ovvero posizione iniziale
print(center)

def move_to_points_with_force(points):
    rtde_c.forceMode(center, [0, 0, 1, 0, 0, 0], [0., 0., 1., 0., 0., 0.], 2, [2, 2, 2, 1, 1, 1])

    for i in range(len(points)):
        rtde_c.moveL(points[i], vel, acc)
        time.sleep(1)
        # if i != 0:
        #     joints = rtde_r.getActualQ()
        #     joints[-1] += math.radians(90)  # Ruota l'ultimo giunto di 90 gradi
        #     rtde_c.moveJ(joints, vel, acc)
        force_z = rtde_r.getActualTCPForce()[2]
        print("TCP Force asse Z:", force_z)

    rtde_c.moveL(center, vel, acc)
    rtde_c.forceModeStop()



# 4 punti di prova
point1 = [center[0], center[1], center[2], center[3], center[4], center[5]]
point2 = [center[0] + 0.1, center[1], center[2], center[3], center[4], center[5]]
point3 = [center[0] + 0.1, center[1] + 0.1, center[2], center[3], center[4], center[5]  ]
point4 = [center[0], center[1] + 0.1, center[2], center[3], center[4], center[5]]
points = [point1, point2, point3, point4]

print('force_mode for entire path')
move_to_points_with_force(points)
