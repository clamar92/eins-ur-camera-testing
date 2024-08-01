import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.measure import find_contours
from skimage.color import rgb2hsv
import time
import rtde_control
import rtde_receive
import json
import math

# Indirizzo IP del robot e porta RTDE
ROBOT_HOST = '192.168.137.198'
#ROBOT_HOST = '192.168.186.135'

# Inizializzazione RTDE
rtde_c = rtde_control.RTDEControlInterface(ROBOT_HOST)
rtde_r = rtde_receive.RTDEReceiveInterface(ROBOT_HOST)

# Accelerazione e velocità
acc = 0.4
vel = 0.4

# Posizione iniziale del robot
robot_startposition = [math.radians(17.87),
                       math.radians(-78.87),
                       math.radians(-100.97),
                       math.radians(-90.22),
                       math.radians(90.03),
                       math.radians(15.62)]

#print('Move robot to start position')
rtde_c.moveJ(robot_startposition, vel, acc)
time.sleep(2)

speed = [0, 0, -0.1, 0, 0, 0]
direction = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0] 
acceleration = 1  # Accelerazione (può essere omessa se si vuole usare il valore predefinito)

#rtde_c.moveUntilContact(speed, direction, acceleration)



# Ottieni la posizione attuale del TCP (in metri)
center_meters = rtde_r.getActualTCPPose()
# Converti le prime tre misure (x, y, z) da metri a millimetri
center_millimeters = [value * 1000 if i < 3 else value for i, value in enumerate(center_meters)]
# Stampa la posizione del TCP con le prime tre misure in millimetri
print(center_millimeters)


# Ottieni le posizioni attuali delle giunture in radianti
joints_radians = rtde_r.getActualQ()
# Converti le posizioni delle giunture da radianti a gradi
joints_degrees = [math.degrees(joint) for joint in joints_radians]
# Stampa le posizioni delle giunture in gradi
print(joints_degrees)