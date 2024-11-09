import cv2
import os

# Numero di immagini da catturare
num_images_to_capture = 5

# Avvia la webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Controlla se la webcam si è aperta correttamente
if not cap.isOpened():
    print("Errore nell'apertura della webcam")
    exit()

# Crea una cartella per salvare le immagini
output_dir = 'images_captured'
os.makedirs(output_dir, exist_ok=True)

# Conta il numero di immagini catturate
captured_images = 0

while captured_images < num_images_to_capture:
    # Leggi un frame dalla webcam
    ret, frame = cap.read()

    if not ret:
        print("Errore nella lettura del frame dalla webcam")
        break

    # Mostra l'immagine corrente
    cv2.imshow('Webcam', frame)

    # Aspetta che l'utente prema un tasto per scattare la prossima foto
    key = cv2.waitKey(1)
    if key == ord('s'):  # Premere 's' per salvare l'immagine
        # Salva l'immagine con un nome incrementale
        image_path = os.path.join(output_dir, f'image_led_e_due_neon_con_marker{captured_images + 1}.jpg')
        cv2.imwrite(image_path, frame)
        print(f"Immagine salvata: {image_path}")
        
        captured_images += 1

# Rilascia la webcam e chiudi tutte le finestre
cap.release()
cv2.destroyAllWindows()

print("Cattura delle immagini completata.")
