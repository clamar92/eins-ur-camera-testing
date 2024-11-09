import math
import time
import rtde_control

# Indirizzo IP del robot e porta RTDE
ROBOT_HOST = '192.168.137.179'

# Inizializzazione RTDE
rtde_c = rtde_control.RTDEControlInterface(ROBOT_HOST)

# Parametri di movimento
acc = 0.4
vel = 0.4

speed = [0,0,-0.50,0,0,0]
while 1:
    rtde_c.moveUntilContact(speed)
    
rtde_c.stopScript()