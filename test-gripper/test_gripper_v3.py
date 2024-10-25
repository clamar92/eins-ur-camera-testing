import socket
import time

# Impostazioni del socket
HOST = "192.168.137.179"  # Sostituisci con l'indirizzo IP del robot UR
PORT = 63352  # Porta usata dal gripper Robotiq

# Comandi per attivare e disattivare il gripper
activate_gripper_command = b'SET ACT 1\n'
deactivate_gripper_command = b'SET ACT 0\n'

# Comandi per chiudere e aprire la pinza
close_gripper_command = b'SET POS 255\n'  # Comando per chiudere completamente la pinza
open_gripper_command = b'SET POS 0\n'  # Comando per aprire completamente la pinza

# Comando per ottenere la posizione del gripper
get_pos_command = b'GET POS\n'

# Funzione per inviare comandi al gripper
def send_command(command):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, PORT))
        s.sendall(command)
        data = s.recv(2**10)
        return data

# Attiva il gripper
print("Attivazione del gripper...")
response = send_command(activate_gripper_command)
print('Risposta del gripper (attivazione):', response)
time.sleep(1)  # Attendi 1 secondo per assicurarti che il gripper sia attivato

# Funzione per chiudere la pinza con ripetizione del comando se fallisce
def close_gripper():
    attempts = 5
    for attempt in range(attempts):
        print(f"Chiusura della pinza, tentativo {attempt + 1}...")
        response = send_command(close_gripper_command)
        print('Risposta del gripper (chiusura):', response)
        time.sleep(1)  # Attendi 1 secondo per permettere al comando di essere processato
        
        # Verifica la posizione della pinza per confermare la chiusura
        pos_response = send_command(get_pos_command)
        print('Risposta del gripper (posizione):', pos_response)
        
        # Controlla se la posizione indica che la pinza è chiusa
        if b'255' in pos_response:  # Adatta la condizione in base alla risposta effettiva
            print("La pinza è stata chiusa con successo.")
            return
        else:
            print("Tentativo di chiusura non riuscito, riprovo...")
    
    print("Impossibile chiudere la pinza dopo diversi tentativi.")

# Prova a chiudere la pinza
close_gripper()

# Verifica finale della posizione della pinza
print("Verifica finale della posizione della pinza...")
final_position_data = send_command(get_pos_command)
print('Gripper finger position is:', final_position_data)

# Se la pinza non si chiude, prova a inviare comandi di posizione intermedi
if b'255' not in final_position_data:
    print("Provo a chiudere la pinza con valori intermedi...")
    intermediate_positions = [64, 128, 192, 255]
    for pos in intermediate_positions:
        command = f'SET POS {pos}\n'.encode()
        print(f"Invio comando: {command}")
        response = send_command(command)
        print(f'Risposta del gripper (posizione {pos}):', response)
        time.sleep(1)  # Attendi 1 secondo tra i comandi
        pos_response = send_command(get_pos_command)
        print(f'Risposta del gripper (posizione dopo {pos}):', pos_response)
        if b'255' in pos_response:
            print("La pinza è stata chiusa con successo.")
            break
