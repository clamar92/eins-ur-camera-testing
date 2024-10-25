from pyModbusTCP.client import ModbusClient
import time

#Initialize variables
##################
executionTime=0
MODBUS_SERVER_IP="192.168.137.87"

#Process initialization
##################
#communication
# TCP auto connect on first modbus request
c = ModbusClient(host=MODBUS_SERVER_IP, port=502, auto_open=True, unit_id=9)

# managing TCP sessions with call to c.open()/c.close()
c.open()

#Wait for the connection to establish
time.sleep(5)

#Write output register to request and activation
response=c.write_multiple_registers(0,[0b0000000100000000,0,0])
print(response)

#Give some time for the gripper to activate
print("Gripper activate")
time.sleep(5)

#response=c.write_multiple_registers(0,[0b0000000100000000,0b0000000011111111,0b1111111111111111])
response=c.write_multiple_registers(0,[0b0000100100000000,0b0000000011111111,0b1111111111111111])

#Give some time for the gripper to reach the desired position
print("Close Gripper")
time.sleep(3)

#close connection
c.close()
exit()