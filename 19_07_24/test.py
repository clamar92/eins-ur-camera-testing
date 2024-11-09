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

center = rtde_r.getActualTCPPose()  # Centro del piano, ovvero posizione iniziale
print(center)


center[0] = center[0] + 0.1


rtde_c.moveL(center, 0.1, 0.1)