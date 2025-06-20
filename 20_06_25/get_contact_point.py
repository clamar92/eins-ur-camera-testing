import time
import math
import json
import rtde_control
import rtde_receive

ROBOT_HOST = '192.168.137.78'

rtde_c = rtde_control.RTDEControlInterface(ROBOT_HOST)
rtde_r = rtde_receive.RTDEReceiveInterface(ROBOT_HOST)

vel = 0.4
acc = 0.4

robot_startposition = [math.radians(21.90),
                       math.radians(-82.13),
                       math.radians(-87.16),
                       math.radians(-101.36),
                       math.radians(90.43),
                       math.radians(18.88)]

saved_heights = []

def move_to_start_position():
    print("Spostamento alla posizione iniziale...")
    rtde_c.moveJ(robot_startposition, vel, acc)
    time.sleep(2)

def enter_freedrive():
    print("Freedrive attivato. Premi INVIO per uscire.")
    rtde_c.teachMode()
    input()
    rtde_c.endTeachMode()

def save_z_value(x=0.0, y=0.0):
    z = rtde_r.getActualTCPPose()[2]
    saved_heights.append({'real': [x, y, z]})
    print(f"Salvato: X={x}, Y={y}, Z={z:.4f}")

if __name__ == "__main__":
    move_to_start_position()

    while True:
        cmd = input("Comando [f=freedrive, s=salva Z, q=esci]: ").strip().lower()
        if cmd == 'f':
            enter_freedrive()
        elif cmd == 's':
            save_z_value()
        elif cmd == 'q':
            break

    with open('20_06_25/z_heights_contact_points.json', 'w') as f:
        json.dump(saved_heights, f, indent=2)

    print("Salvataggio completato.")
