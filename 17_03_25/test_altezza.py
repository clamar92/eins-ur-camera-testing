import rtde_control
import rtde_receive

ROBOT_HOST = '192.168.137.221'

rtde_c = rtde_control.RTDEControlInterface(ROBOT_HOST)
rtde_r = rtde_receive.RTDEReceiveInterface(ROBOT_HOST)


pose = rtde_r.getActualTCPPose()
real_x, real_y, real_z = pose[:3]
print(real_z)