import cv2
import matplotlib.pyplot as plt
import time 

img_width = 640  # Larghezza dell'immagine
img_height = 480  # Altezza dell'immagine


cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, img_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, img_height)

if not cap.isOpened():
    print("Errore nell'apertura della webcam")
    exit()

time.sleep(2)  # Attendi che la webcam si stabilizzi
for _ in range(10):  # Scarta i primi frame
    cap.read()

# Cattura una singola immagine
ret, frame = cap.read()

frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
plt.imshow(frame)
plt.show()