#############################################################

#Library importation
import socket
import time

#Socket setings
HOST="192.168.137.198" #replace by the IP address of the UR robot
PORT=63352 #PORT used by robotiq gripper

#Socket communication
# with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
#     #open the socket
#     s.connect((HOST, PORT))
#     s.sendall(b'GET POS\n')
#     data = s.recv(2**10)
    

# Comando per chiudere la pinza
# SET GTO 1
command = b'SET POS 255\n'
#command = b'SET ACT 1\n'


# Comunicazione tramite socket
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    # Apri il socket
    s.connect((HOST, PORT))
    # Invia il comando per chiudere la pinza
    #s.sendall(command1)
    s.sendall(command)
    # Ricevi la risposta (opzionale)
    data = s.recv(2**10)
    # Stampa la risposta per verificare che il comando sia stato ricevuto correttamente
    print('Risposta del gripper:', data)

    s.close()

#Print finger position
#Gripper finger position is between 0 (Full open) and 255 (Full close)
print('Gripper finger position is: ', data)
#############################################################


# The server can receive GET and SET requests.

# Example of GET request:
# GET POS

# Example of SET request:
# SET POS 100

# Here below are some possible commands. Some commands can be use with both GET and SET:

# ACT: Activation bit
# 0 - Gripper not activated
# 1 - Gripper activated
# GTO: 1 if the gripper is set to move to requested position 0 if gripper is set to stay at the same place
# PRE: Position request eco. Should be same a the requested position if
# the gripper successfully received the requested position.
# POS: Current position of the gripper
# SPE: Speed eco. Should be same as requested speed.
# FOR: Force parameter of the gripper
# OBJ: Object grippings status
# 0 - Fingers are inmotion towards requested position.No object detected.
# 1 - Fingers have stopped due to a contact while opening before requested position.Object detected opening.
# 2 - Fingers have stopped due to a contact while closing before requested position.Object detected closing.
# 3 - Fingers are at requested position.No object detected or object has been loss / dropped.
# STA: Gripper status, returns the current status & motion of theGripper fingers.
# 0 -Gripper is in reset ( or automatic release )state. See Fault Status if Gripper is activated.
# 1 - Activation in progress.
# 2 - Not used.
# 3 - Activation is completed.
# MOD: ...
# FLT: Fault status returns general errormessages that are useful for troubleshooting. Fault LED (red) is present on theGripper chassis,
# LED can be blue, red or both and be solid or blinking.
# 0 - No fault (LED is blue)
# Priority faults (LED is blue)
# 5 - Action delayed, activation (reactivation)must be completed prior to performing the action.
# 7 - The activation bit must be set prior to action.
# Minor faults (LED continuous red)
# 8 -Maximum operating temperature exceeded,wait for cool-down.
# 9 No communication during at least 1 second.
# Major faults (LED blinking red/blue) - Reset is required (rising edge on activation bit rACT needed).
# 10 - Underminimum operating voltage.
# 11- Automatic release in progress.
# 12- Internal fault; contact support@robotiq.com.
# 13 - Activation fault, verify that no interference or other error occurred.
# 14-Overcurrent triggered.
# 15- Automatic release completed.
# MSC: Gripper maximym current.
# COU: Gripper current.
# NCY: Number of cycles performed by the gripper
# DST: Gripper driver state
# 0 - Gripper Driver State : RQ_STATE_INIT
# 1 - Gripper Driver State : RQ_STATE_LISTEN
# 2 - Gripper Driver State : Q_STATE_READ_INFO
# 3 - Gripper Driver State : RQ_STATE_ACTIVATION
# Other - Gripper Driver State : RQ_STATE_RUN
# PCO: Gripper connection state
# 0 - Gripper Connection State : No connection problem detected
# Other - Gripper Connection State : Connection problem detected