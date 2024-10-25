import math
import time
import rtde_control
import rtde_receive

# Indirizzo IP del robot e porta RTDE
ROBOT_HOST = '192.168.137.179'

# Inizializzazione RTDE
rtde_c = rtde_control.RTDEControlInterface(ROBOT_HOST)
rtde_r = rtde_receive.RTDEReceiveInterface(ROBOT_HOST)

#accellerazione e velocità
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

speed = [0, 0, -0.1, 0, 0, 0] 
rtde_c.moveUntilContact(speed)




rtde_r.stopScript()
rtde_c.stopScript()

